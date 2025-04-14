from typing import Dict, List, Optional, Tuple, Union
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from gglasso.problem import glasso_problem
from sklearn.preprocessing import StandardScaler
import graphviz

def prepare_activation_matrix(activations: torch.Tensor) -> np.ndarray:
    """Prepare activation matrix for GLasso.
    
    Args:
        activations: Tensor of activations (num_samples, num_neurons)
        
    Returns:
        Standardized numpy array of activations
    """
    # Make sure tensor is on CPU and convert to numpy
    if activations.device.type != 'cpu':
        activations = activations.cpu()
    act_np = activations.numpy()
    
    # Standardize features
    scaler = StandardScaler()
    act_standardized = scaler.fit_transform(act_np)
    
    return act_standardized

def run_glasso(activation_matrix: np.ndarray, alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Run Graphical Lasso on activation matrix using GGLasso for better performance.
    
    Args:
        activation_matrix: Matrix of standardized activations (samples x neurons)
        alpha: Regularization parameter for GLasso. If None, use default value.
        
    Returns:
        Tuple of (precision matrix, covariance matrix)
    """
    # Calculate empirical covariance matrix
    n_samples, n_features = activation_matrix.shape
    emp_cov = np.dot(activation_matrix.T, activation_matrix) / n_samples
    
    # Create a glasso_problem object with the empirical covariance
    problem = glasso_problem(S=emp_cov, N=n_samples, do_scaling=False)
    problem.set_reg_params(reg_params={"lambda1": 0.1, "lambda2": 0.01})
    
    # Solve the problem
    problem.solve(verbose=False, tol=1e-6, rtol=1e-5)
    
    # Get precision and covariance matrices
    precision_matrix = problem.solution.precision_
    covariance_matrix = emp_cov
    print(precision_matrix)
    
    return precision_matrix, covariance_matrix

def analyze_neuron_connections(precision_matrix: np.ndarray, 
                              threshold: float = 0.01) -> Dict[int, List[int]]:
    """Analyze connections between neurons based on precision matrix.
    
    Args:
        precision_matrix: Precision matrix from GLasso
        threshold: Threshold for considering a connection
        
    Returns:
        Dictionary mapping neurons to their connected neurons
    """
    # Get absolute values and zero out diagonal
    abs_precision = np.abs(precision_matrix)
    np.fill_diagonal(abs_precision, 0)
    
    # Find connections above threshold
    connections: Dict[int, List[int]] = {}
    for i in range(abs_precision.shape[0]):
        # Get indices of neurons connected to i
        indices = np.where(abs_precision[i] > threshold)[0]
        connected: List[int] = [int(idx) for idx in indices]
        connections[i] = connected
        
    return connections

def precision_to_partial_correlation(precision_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a precision matrix to a partial correlation matrix.

    Parameters:
    -----------
    precision_matrix : numpy.ndarray
        The precision matrix (inverse covariance)

    Returns:
    --------
    numpy.ndarray
        The partial correlation matrix with values between -1 and 1
    """
    # Get the diagonal elements for normalization
    diag = np.sqrt(np.diag(precision_matrix))

    # Outer product to create a matrix of sqrt(precision_ii * precision_jj)
    norm_factors = np.outer(diag, diag)

    # Calculate partial correlations
    # Note the negative sign as per the formula
    partial_corr = -precision_matrix / norm_factors

    # Set diagonal to 1 (self-correlation)
    np.fill_diagonal(partial_corr, 1.0)

    return partial_corr

def plot_precision_matrix(precision_matrix: np.ndarray, 
                         save_path: Optional[str] = None) -> None:
    """Plot precision matrix as a heatmap.
    
    Args:
        precision_matrix: Precision matrix from GLasso
        save_path: Path to save the figure (optional)
    """
    partial_corr = precision_to_partial_correlation(precision_matrix)
    plt.figure(figsize=(10, 8))
    plt.imshow(partial_corr, cmap='viridis')
    plt.colorbar(label='Partial Correlation')
    plt.title('Neuron Partial Correlation Structure')
    plt.xlabel('Neuron Index')
    plt.ylabel('Neuron Index')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
    
def create_graphviz_dot(connections: Dict[int, List[int]], 
                       layer_name: str,
                       save_path: Optional[str] = None,
                       render: bool = False) -> Union[str, graphviz.Digraph]:
    """Create a GraphViz DOT representation of neuron connections.
    
    Args:
        connections: Dictionary mapping neurons to their connected neurons
        layer_name: Name of the layer (used for graph title)
        save_path: Path to save the DOT file (optional)
        render: Whether to render the graph using the graphviz library (returns Digraph object)
        
    Returns:
        String containing the DOT representation or graphviz.Digraph object if render=True
    """
    if render:
        # Create Digraph object
        dot = graphviz.Digraph(name=layer_name, comment=f"Neuron connections for {layer_name}")
        
        # Set graph attributes
        dot.attr('graph', rankdir='LR', bgcolor='white')
        dot.attr('node', shape='circle', style='filled', fillcolor='#ADD8E6')
        dot.attr('edge', color='#666666', arrowsize='0.5')
        
        # Add nodes and edges
        for source, targets in connections.items():
            # Only include nodes that have at least one connection
            if targets:
                # Ensure source node is added
                dot.node(str(source))
                for target in targets:
                    # Add target node and edge
                    dot.node(str(target))
                    dot.edge(str(source), str(target))
        
        # Save to file if path provided
        if save_path:
            # Get directory and filename without extension
            dir_path = os.path.dirname(save_path)
            filename = os.path.splitext(os.path.basename(save_path))[0]
            
            # Save and render
            os.makedirs(dir_path, exist_ok=True)
            dot.render(filename=filename, directory=dir_path, format='png', cleanup=True)
            
        return dot
    else:
        # Start DOT file
        dot_content = [
            f'digraph "{layer_name}" {{',
            '  // Graph styling',
            '  graph [rankdir=LR, fontname="Arial", bgcolor="white"];',
            '  node [shape=circle, style=filled, fillcolor="#ADD8E6", fontname="Arial"];',
            '  edge [color="#666666", arrowsize=0.5];',
            '',
            '  // Nodes and edges'
        ]
        
        # Add nodes and edges
        for source, targets in connections.items():
            # Only include nodes that have at least one connection
            if targets:
                for target in targets:
                    dot_content.append(f'  {source} -> {target};')
        
        # Close the graph
        dot_content.append('}')
        
        # Join all lines
        dot_string = '\n'.join(dot_content)
        
        # Save to file if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(dot_string)
        
        return dot_string
