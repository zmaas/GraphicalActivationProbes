import torch
import torch.optim as optim
import warnings
from tqdm import tqdm
import time

USE_AMP = torch.cuda.is_available() # Set to False to disable AMP
if USE_AMP:
    from torch.cuda.amp import autocast, GradScaler
    print("CUDA available, enabling Automatic Mixed Precision (AMP).")
else:
    # Define dummy autocast and GradScaler for CPU compatibility
    from contextlib import contextmanager
    @contextmanager
    def autocast(enabled=False): # Dummy context manager
        yield
    class GradScaler: # Dummy scaler
        def __init__(self, enabled=False): pass
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass
        def unscale_(self, optimizer): pass # Dummy method
    print("CUDA not available or AMP disabled, using standard float32.")


def graphical_lasso_large_scale(
    data: torch.Tensor,
    lambda_reg: float,
    max_iter: int = 500, # Reduced default for large scale
    lr: float = 0.001,  # Potentially smaller LR for stability
    tol: float = 1e-5,  # Convergence tolerance
    verbose: bool = False,
    verbose_interval: int = 10, # Print more often for long runs
    eps_pd: float = 1e-6, # Slightly larger epsilon for stability
    reg_S: float = 1e-4   # Regularization for empirical covariance (MAY NEED TUNING)
) -> torch.Tensor:
    """
    Estimates a sparse precision matrix using Graphical Lasso in PyTorch,
    optimized for larger scale (n_features >> 100).

    Args:
        data (torch.Tensor): (n_samples, n_features). Assumed centered or handles internally.
        lambda_reg (float): L1 regularization parameter.
        max_iter (int): Max optimization iterations.
        lr (float): Learning rate.
        tol (float): Convergence tolerance (relative loss change).
        verbose (bool): Print progress.
        verbose_interval (int): How often to print progress.
        eps_pd (float): Epsilon for positive definiteness (Theta = A@A.T + eps*I).
        reg_S (float): Regularization for empirical covariance S. CRITICAL for large n.

    Returns:
        torch.Tensor: Estimated sparse precision matrix Theta (n_features, n_features).
    """
    n_samples, n_features = data.shape
    device = data.device
    print(f"Running on device: {device}. n_features={n_features}, n_samples={n_samples}")

    # 1. Center data and compute empirical covariance matrix S
    print("Calculating Covariance Matrix S...")
    start_cov = time.time()
    data_centered = data - data.mean(dim=0, keepdim=True)
    # Ensure float32 for calculation consistency if using AMP later
    S = torch.cov(data_centered.T).float()
    print(f"Covariance calculation took: {time.time() - start_cov:.2f}s")

    # CRITICAL: Add regularization to S for stability, potentially scale with n
    # Heuristic: increase reg_S if n_features is large relative to n_samples
    effective_reg_S = max(reg_S, 1e-6 * (n_features / max(1, n_samples)))
    # Clamp effective_reg_S to avoid excessively large values if needed
    effective_reg_S = min(effective_reg_S, 1e-1) # Example upper bound
    print(f"Using S regularization (reg_S): {effective_reg_S:.2e}")
    S = S + effective_reg_S * torch.eye(n_features, device=device)

    # 2. Initialize the parameter matrix A (CHEAPLY)
    print("Initializing parameter A...")
    try:
        diag_S = torch.diag(S)
        diag_S_safe = torch.clamp(diag_S, min=1e-6) # Avoid sqrt(0) or tiny values
        A = torch.diag(1.0 / torch.sqrt(diag_S_safe)).requires_grad_(True)
        print("Initialized A using diagonal(1/sqrt(diag(S)))")
    except Exception as e:
        warnings.warn(f"Diagonal initialization failed ({e}). Initializing A as scaled Identity.")
        A = torch.eye(n_features, device=device, requires_grad=True) * 0.1

    # Move S to float32 if not already, matching A and potential AMP usage
    S = S.float()

    # 3. Setup Optimizer
    optimizer = optim.Adam([A], lr=lr)
    scaler = GradScaler(enabled=(USE_AMP and data.is_cuda)) # Enable scaler only if using AMP

    # 4. Optimization loop
    print(f"Starting optimization loop (max_iter={max_iter}, tol={tol:.1e})...")
    prev_loss = torch.inf
    off_diag_mask = (1.0 - torch.eye(n_features, device=device)).bool()
    iter_start_time = time.time()

    for i in range(max_iter):
        optimizer.zero_grad(set_to_none=True) # Saves memory

        # Use autocast context for forward pass (matmul, slogdet)
        with autocast(enabled=(USE_AMP and data.is_cuda)):
            # Ensure A is float32 if AMP is used (should be by default from init)
            Theta = A @ A.T + eps_pd * torch.eye(n_features, device=device)

            # Ensure S and Theta are compatible types for matmul trace
            trace_term = torch.trace(S @ Theta.float()) # Ensure float inputs

            sign, logabsdet = torch.linalg.slogdet(Theta.float()) # slogdet might need float32

            if sign.item() <= 0:
                warnings.warn(f"\nIter {i}: Theta is not positive definite (sign={sign.item()}). Loss set to Inf.")
                # Try to recover? Maybe increase eps_pd slightly or just stop?
                # For now, let loss be Inf and break after check.
                current_loss = torch.tensor(float('inf'), device=device)
                log_det_term = torch.tensor(float('nan'), device=device)
                l1_term = torch.tensor(float('nan'), device=device)
            else:
                log_det_term = -logabsdet
                # Ensure Theta used for L1 norm is float for consistency
                l1_term = lambda_reg * torch.sum(torch.abs(Theta.float()[off_diag_mask]))
                # Ensure all terms are float for summation
                current_loss = trace_term + log_det_term + l1_term

        if not torch.isfinite(current_loss):
             warnings.warn(f"\nIter {i}: Loss is NaN or Inf ({current_loss.item()}). Stopping optimization.")
             # Maybe NaN gradient occurred in previous step?
             if i > 0: # Try to return previous A if current step failed
                 A = A_prev.detach().requires_grad_(False)
                 warnings.warn("Attempting to return parameters from previous stable iteration.")
             else: # Failed on first iter or cannot recover
                 A = A.detach() # Return current A detached
             break

        # --- Backward and Step ---
        scaler.scale(current_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # --- End Backward and Step ---

        current_loss_item = current_loss.item()

        # Store previous A in case next iteration fails
        A_prev = A.detach().clone()

        # --- Convergence Check ---
        if i > 0:
            loss_change = abs(current_loss_item - prev_loss) / (abs(prev_loss) + 1e-10)
            if loss_change < tol:
                print(f"\nConverged at iteration {i} with loss change {loss_change:.2e} < {tol:.1e}")
                break
        prev_loss = current_loss_item

        # --- Verbose Output ---
        if verbose and (i % verbose_interval == 0 or i == max_iter - 1):
            iter_end_time = time.time()
            time_per_iter = (iter_end_time - iter_start_time) / (i + 1e-9)
            eta = (max_iter - 1 - i) * time_per_iter if i > 0 else float('inf')
            print(f"Iter {i}/{max_iter}: Loss={current_loss_item:.4e}, "
                  f"Trace={trace_term.item():.3e}, "
                  f"LogDet={-log_det_term.item():.3e}, "
                  f"L1={l1_term.item():.3e}, "
                  f"ETA: {eta:.1f}s")
            # Optional: Check gradient norm if debugging
            # grad_norm = torch.linalg.norm(A.grad) if A.grad is not None else 0
            # print(f"  Grad Norm (A): {grad_norm:.2e}")


    # End of loop
    print(f"Optimization finished after {i+1} iterations.")

    # 5. Return the final estimated precision matrix
    # Ensure requires_grad=False and correct dtype (float32)
    Theta_final = (A @ A.T + eps_pd * torch.eye(n_features, device=device)).detach().float()

    return Theta_final

def graphical_lasso(
    data: torch.Tensor,
    lambda_reg: float,
    max_iter: int = 1000,
    lr: float = 0.01,
    tol: float = 1e-5,
    verbose: bool = False,
    eps_pd: float = 1e-8, # Small value to ensure positive definiteness
    reg_S: float = 1e-5   # Regularization for empirical covariance
) -> torch.Tensor:
    """
    Estimates a sparse precision matrix using Graphical Lasso in PyTorch.

    Minimizes: tr(S @ Theta) - log det(Theta) + lambda_reg * ||Theta||_1,off-diag
    using gradient descent on a parameterization Theta = A @ A.T + eps_pd * I.

    Args:
        data (torch.Tensor): Input data tensor of shape (n_samples, n_features).
                             Assumes data is already centered or handles centering internally.
        lambda_reg (float): Regularization parameter for the L1 penalty.
        max_iter (int): Maximum number of iterations for optimization.
        lr (float): Learning rate for the Adam optimizer.
        tol (float): Tolerance for convergence check (relative change in parameters).
        verbose (bool): If True, prints loss every 100 iterations.
        eps_pd (float): Small epsilon added to ensure Theta is positive definite.
        reg_S (float): Small epsilon added to diagonal of empirical covariance
                       for numerical stability, especially for inversion if needed
                       and potentially better conditioning.

    Returns:
        torch.Tensor: The estimated sparse precision matrix Theta of shape (n_features, n_features).
    """
    n_samples, n_features = data.shape
    device = data.device

    # 1. Center data and compute empirical covariance matrix S
    data_centered = data - data.mean(dim=0, keepdim=True)
    # S = (data_centered.T @ data_centered) / (n_samples - 1)
    # Using torch.cov for potentially better numerical stability
    S = torch.cov(data_centered.T)

    # Add small regularization to S for stability
    dynamic_reg_S = max(reg_S, 1e-4 * n_features) # Adjust multiplier as needed
    print(f"Using effective S regularization: {dynamic_reg_S:.2e}") # Add for debugging
    S = S + dynamic_reg_S * torch.eye(n_features, device=device)
    #S = S + reg_S * torch.eye(n_features, device=device)

    # 2. Initialize the parameter matrix A
    # Initialize A such that A @ A.T is close to S^{-1} initially
    # This often helps convergence.
    try:
        # Inverse of regularized covariance
        S_inv_init = torch.inverse(S)
        # Cholesky decomposition L where L @ L.T = S_inv_init
        L_init = torch.linalg.cholesky(S_inv_init)
        # Initialize A as this Cholesky factor
        A = L_init.clone().detach().requires_grad_(True)
    except torch.linalg.LinAlgError:
        warnings.warn(
            "Initial S was singular or non-positive definite even after regularization. "
            "Initializing A as scaled Identity."
        )
        # Fallback initialization if S is ill-conditioned
        A = torch.eye(n_features, device=device, requires_grad=True) * 0.1


    # 3. Setup Optimizer
    optimizer = optim.Adam([A], lr=lr)

    # 4. Optimization loop
    prev_A_norm = torch.inf
    off_diag_mask = (1.0 - torch.eye(n_features, device=device)).bool() # Mask for L1 penalty

    for i in tqdm(range(max_iter), desc="GLASSO Iter"):
        optimizer.zero_grad()

        # Ensure Theta is symmetric positive definite
        # Theta = A @ A.T + eps_pd * torch.eye(n_features, device=device)
        # Using matrix_power for potentially better gradient flow if needed,
        # but A @ A.T is standard. Let's stick to A @ A.T for clarity.
        Theta = A @ A.T + eps_pd * torch.eye(n_features, device=device)

        # Calculate terms of the objective function (minimization form)
        trace_term = torch.trace(S @ Theta)

        # Log-determinant term (use slogdet for stability)
        # logdet = torch.logdet(Theta) # logdet can be unstable if Theta near singular
        sign, logabsdet = torch.linalg.slogdet(Theta)
        if sign.item() <= 0:
            # This shouldn't happen with the eps_pd * I term if eps_pd > 0
            warnings.warn(f"Iteration {i}: Theta is not positive definite. Log-det is invalid.")
            # Handle error: could stop, adjust eps_pd, or clamp A
            # For now, let's try setting loss high to push away from this region
            loss = torch.tensor(float('inf'), device=device)
        else:
            log_det_term = -logabsdet # We minimize -logdet

            # L1 penalty term (only on off-diagonal elements)
            l1_term = lambda_reg * torch.sum(torch.abs(Theta[off_diag_mask]))

            # Total loss
            loss = trace_term + log_det_term + l1_term

        if not torch.isfinite(loss):
             warnings.warn(f"Iteration {i}: Loss is NaN or Inf. Stopping optimization.")
             # May need to decrease lr, increase eps_pd, or check data scaling
             break

        # Backpropagation
        loss.backward()

        # Optional: Gradient clipping if gradients explode
        # torch.nn.utils.clip_grad_norm_([A], max_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Convergence check
        with torch.no_grad():
            current_A_norm = torch.norm(A)
            relative_change = torch.norm(A - A.clone().detach()) / (prev_A_norm + 1e-10) # Use clone().detach() to get previous A before step? No, A is updated in place. Need A_prev.
            # Let's check change based on previous iteration's A norm saved before the step
            # A more robust check uses the loss, but parameter change is common
            # Re-calculate parameter norm *before* saving for next iter
            delta_A_norm = torch.norm(A - A.clone().detach()) # This seems wrong. Let's track A before the step.
            # Let's track loss change instead, usually more stable
            if i > 0:
                loss_change = abs(loss.item() - prev_loss) / (abs(prev_loss) + 1e-10)
                if loss_change < tol:
                    if verbose:
                        print(f"Converged at iteration {i} with loss change {loss_change:.2e}")
                    break

            prev_loss = loss.item()
            # prev_A_norm = current_A_norm # Save norm for next iteration's check (if using param check)


        if verbose and (i % 100 == 0 or i == max_iter - 1):
            print(f"Iter {i}: Loss = {loss.item():.4f}, Trace = {trace_term.item():.4f}, LogDet = {-log_det_term.item():.4f}, L1 = {l1_term.item():.4f}")


    # 5. Return the final estimated precision matrix
    # Ensure requires_grad=False for the returned tensor
    Theta_final = (A @ A.T + eps_pd * torch.eye(n_features, device=device)).detach()

    # Optional: Apply thresholding for exact zeros if desired (based on lambda_reg)
    # Note: Optimization naturally encourages small values, but might not be exactly zero.
    # threshold = lambda_reg * lr # Heuristic, not standard GLasso
    # Theta_final[torch.abs(Theta_final) < threshold] = 0.0

    return Theta_final

# --- Example Usage & Comparison ---
if __name__ == '__main__':
    # Set seed for reproducibility
    torch.manual_seed(42)
    import numpy as np
    np.random.seed(42) # Also seed numpy for sklearn consistency if needed

    # Check if sklearn is available
    try:
        from sklearn.covariance import GraphicalLasso
        sklearn_available = True
    except ImportError:
        warnings.warn("scikit-learn not found. Skipping comparison test.")
        sklearn_available = False

    # 1. Generate synthetic data
    n_samples = 200
    n_features = 8192
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") # Force CPU for easier debugging if needed
    print(f"Using device: {device}")

    true_theta = torch.zeros((n_features, n_features), device=device)
    for i in range(n_features):
        true_theta[i, i] = 1.0
        if i > 0:
            true_theta[i, i-1] = 0.6
            true_theta[i-1, i] = 0.6
        if i > 1:
             true_theta[i, i-2] = 0.3
             true_theta[i-2, i] = 0.3

    try:
        _ = torch.linalg.cholesky(true_theta)
        # print("True Theta is positive definite.")
    except torch.linalg.LinAlgError:
        print("Warning: Generated true Theta is not positive definite. Adding diagonal dominance.")
        true_theta += torch.eye(n_features, device=device) * (torch.sum(torch.abs(true_theta), dim=1) + 0.1)
        _ = torch.linalg.cholesky(true_theta)

    true_cov = torch.inverse(true_theta)
    mean = torch.zeros(n_features, device=device)
    dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=true_cov)
    data = dist.sample((n_samples,))
    print(f"Generated data shape: {data.shape}")

    # --- Parameters for Both Methods ---
    lambda_regularization = 0.1 # Use the same regularization strength
    pt_max_iter = 2000
    pt_lr = 0.005
    pt_tol = 1e-5
    pt_eps_pd = 1e-7
    pt_reg_S = 1e-4

    # --- Run PyTorch Implementation ---
    print("\n--- Running PyTorch Graphical Lasso ---")
    start_time_pt = time.time()
    estimated_theta_pt = graphical_lasso_large_scale(
        data,
        lambda_reg=lambda_regularization,
        max_iter=pt_max_iter,
        lr=pt_lr,
        tol=pt_tol,
        verbose=True,
        eps_pd=pt_eps_pd,
        reg_S=pt_reg_S
    )
    end_time_pt = time.time()
    print(f"PyTorch implementation took: {end_time_pt - start_time_pt:.4f} seconds")
    print("\nPyTorch Estimated Precision Matrix (Theta):")
    print(estimated_theta_pt.round(decimals=3))

    # --- Run Scikit-learn Implementation ---
    if sklearn_available:
        print("\n--- Running Scikit-learn Graphical Lasso ---")
        # Convert data to numpy array on CPU
        data_np = data.cpu().numpy()

        # Initialize and fit scikit-learn model
        # Note: sklearn's alpha corresponds to our lambda_reg
        # Sklearn uses coordinate descent, parameters like tol and max_iter behave differently
        sklearn_model = GraphicalLasso(
            alpha=lambda_regularization,
            mode='cd', # Coordinate Descent (default)
            tol=1e-5, # Sklearn's default tolerance
            max_iter=200, # Sklearn's default max_iter is lower
            verbose=False, # Set to True for sklearn verbosity
            assume_centered=False # Let sklearn handle centering if needed (or center data_np beforehand)
        )
        start_time_sk = time.time()
        sklearn_model.fit(data_np)
        end_time_sk = time.time()

        # Get the result and convert back to torch tensor
        estimated_theta_sk_np = sklearn_model.precision_
        estimated_theta_sk = torch.from_numpy(estimated_theta_sk_np).to(device).float()

        print(f"Scikit-learn implementation took: {end_time_sk - start_time_sk:.4f} seconds")
        print("\nScikit-learn Estimated Precision Matrix (Theta):")
        print(estimated_theta_sk.round(decimals=3))

        # --- Comparison ---
        print("\n--- Comparison ---")
        diff_norm = torch.linalg.norm(estimated_theta_pt - estimated_theta_sk)
        relative_diff_norm = diff_norm / torch.linalg.norm(estimated_theta_sk)

        print(f"Frobenius norm of difference: {diff_norm:.4f}")
        print(f"Relative Frobenius norm of difference: {relative_diff_norm:.4f}")

        # Compare sparsity patterns (thresholding might be needed for a fair comparison)
        thresh = 1e-3
        sparsity_pt = (torch.abs(estimated_theta_pt) > thresh).float()
        sparsity_sk = (torch.abs(estimated_theta_sk) > thresh).float()
        sparsity_diff = torch.sum(torch.abs(sparsity_pt - sparsity_sk)).item()

        print(f"\nSparsity pattern difference (threshold={thresh}): {sparsity_diff} elements")
        print("PyTorch Sparsity Pattern:")
        print(sparsity_pt)
        print("Scikit-learn Sparsity Pattern:")
        print(sparsity_sk)

    else:
        print("\nSkipping scikit-learn comparison.")

    # --- Original True Theta (for reference) ---
    print("\nTrue Precision Matrix (Theta):")
    print(true_theta.round(decimals=3))
