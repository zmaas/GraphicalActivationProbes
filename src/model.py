from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Gemma3ForCausalLM
from tqdm import tqdm

class ModelWithActivations(nn.Module):
    """Wrapper for Huggingface model that stores activations."""
    
    def __init__(self, model_name: str = "google/gemma-3-1b-it"):
        """Initialize model and hooks.
        
        Args:
            model_name: Name of the model to load
        """
        super().__init__()
        
        # Set device to MPS if available, otherwise CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS device for acceleration")
        else:
            self.device = torch.device("cpu")
            print("MPS not available, using CPU")
            
        self.model = Gemma3ForCausalLM.from_pretrained(
            model_name, 
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Storage for activations
        self.activations: Dict[str, List[torch.Tensor]] = {}
        self.hooks: List[Any] = []
        
        # Register hooks for MLP layers
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        # Clear any existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Print some debug info about the model's modules to help understand its structure
        module_types = {}
        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            if module_type not in module_types:
                module_types[module_type] = []
            if len(module_types[module_type]) < 2:  # Limit examples to 2 per type
                module_types[module_type].append(name)
                
        print("Model module types and examples:")
        for module_type, examples in module_types.items():
            print(f"  {module_type}: {examples}")
        
        # Register hooks for MLP layers in Gemma3 model
        for name, module in self.model.named_modules():
            # Use more comprehensive patterns to catch Gemma3 MLP components
            # if any(pattern in name for pattern in ["mlp", "gate", "feed_forward", "ffn", "fc"]):
            if name[-3:] == "mlp": # DEBUG - get only MLP
                print(f"Registering hook for: {name}")
                self.hooks.append(self._register_hook(module, f"{name}_output"))
        
        # If no hooks were registered, try a more general approach with linear layers
        if not self.hooks:
            print("No specific MLP modules found, trying more general approach...")
            hook_count = 0
            # Just register some Linear layers as fallback
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear) and "model.layers" in name:
                    hook_count += 1
                    if hook_count <= 10:  # Limit number of hooks to avoid too many
                        print(f"Registering hook for linear layer: {name}")
                        self.hooks.append(self._register_hook(module, f"{name}_output"))
        
        print(f"Registered {len(self.hooks)} hooks for activation collection")
                
    def _register_hook(self, module: nn.Module, name: str) -> Any:
        """Register a forward hook on a module.
        
        Args:
            module: PyTorch module to hook
            name: Name to use for storing activations
            
        Returns:
            Hook handle
        """
        def hook_fn(module, input, output):
            # Store the output activation
            if name not in self.activations:
                self.activations[name] = []
            self.activations[name].append(output.detach().cpu())
            
        return module.register_forward_hook(hook_fn)
    
    def forward(self, **inputs) -> Any:
        """Forward pass through the model.
        
        Args:
            inputs: Model inputs
            
        Returns:
            Model outputs
        """
        # For sentiment analysis, we don't need the labels for calculating activations
        # Remove 'labels' to avoid the mismatch error
        if 'labels' in inputs:
            inputs.pop('labels')
            
        return self.model(**inputs)
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}
        
    def collect_activations(self, dataloader: torch.utils.data.DataLoader, layer_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Collect activations for a dataloader.
        
        Args:
            dataloader: DataLoader with input samples
            layer_name: Optional name of specific layer to collect activations for.
                        If None, collects activations for all layers (memory intensive).
            
        Returns:
            Dictionary of activations for requested layer(s)
        """
        self.clear_activations()
        self.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting activations"):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self(**batch)
        
        # Concatenate all collected activations
        result = {}
        for name, acts in self.activations.items():
            # Skip layers we're not interested in if a specific layer was requested
            if layer_name is not None and name != layer_name:
                continue
                
            # Reshape to (num_samples, num_neurons)
            acts_reshaped = []
            for act in acts:
                # For sequence models, take the activation of the first token ([CLS])
                if len(act.shape) == 3:  # (batch_size, seq_len, hidden_dim)
                    act = act[:, 0, :]  # Take first token
                acts_reshaped.append(act)
                
            result[name] = torch.cat(acts_reshaped, dim=0)
            # Clear this layer's activations from memory immediately after processing
            self.activations[name] = []
            
        return result
        
    def get_layer_names(self) -> List[str]:
        """Get the names of all layers with registered hooks.
        
        Returns:
            List of layer names
        """
        # Run a forward pass first to populate self.activations
        # Then return the keys
        return list(self.activations.keys())
