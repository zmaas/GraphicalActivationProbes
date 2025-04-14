import argparse
import os
import numpy as np
import sys
import torch
import gc
from tqdm import tqdm

from data import load_sentiment_dataset, SentimentDataset, get_dataloader
from model import ModelWithActivations
from glasso import (
    prepare_activation_matrix, 
    run_glasso, 
    analyze_neuron_connections, 
    plot_precision_matrix,
    create_graphviz_dot
)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GLasso interpretation on transformer model.")
    parser.add_argument("--dataset", type=str, default="imdb", help="Dataset to use")
    parser.add_argument(
        "--model", 
        type=str, 
        default="google/gemma-3-1b-it", 
        help="Model to analyze"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=None, 
        help="GLasso regularization parameter (None for CV)"
    )
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument(
        "--skip_viz", 
        action="store_true",
        help="Skip generating visualizations (useful for headless environments)"
    )
    
    return parser.parse_args()

def main() -> None:
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model {args.model}...")
    model = ModelWithActivations(model_name=args.model)
    tokenizer = model.tokenizer
    
    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    df = load_sentiment_dataset(dataset_name=args.dataset)
    
    # Use a smaller subset for faster processing (configurable with env var)
    subset_size = int(os.environ.get("SUBSET_SIZE", "32"))  # Default to 100 samples
    print(f"Using {subset_size} samples for analysis")
    df = df.head(subset_size)
    
    # Create dataset and dataloader
    dataset = SentimentDataset(
        texts=df["text"].tolist(), 
        labels=df["label"].tolist(),
        tokenizer=tokenizer
    )
    dataloader = get_dataloader(dataset, batch_size=args.batch_size)
    
    # Detect layers with registered hooks
    print("Detecting layers with registered hooks...")
    # Run a sample batch to detect which layers have hooks
    sample_batch = next(iter(dataloader))
    sample_batch = {k: v.to(model.device) for k, v in sample_batch.items()}
    with torch.no_grad():
        model(**sample_batch)
    
    # Get layer names from activations dict
    layer_names = list(model.activations.keys())
    print(f"Detected layers: {layer_names}")
    model.clear_activations()
    
    print(f"Found {len(layer_names)} layers with activations")
    if not layer_names:
        print("WARNING: No activations were collected!")
        print("This could be because the hook registration didn't match any layers.")
        print("Check the model architecture and hook registration logic.")
        sys.exit(1)

    print("Collecting activations for all layers...")
    activations_dict = model.collect_activations(dataloader)
    
    # Process each layer one at a time to save memory
    for layer_name in tqdm(layer_names):
        print(f"Processing layer {layer_name}...")
        
        # Collect activations only for this layer
        # print(f"Collecting activations for {layer_name}...")
        # activations_dict = model.collect_activations(dataloader, layer_name=layer_name)
        
        if layer_name not in activations_dict:
            print(f"Warning: No activations collected for {layer_name}, skipping")
            continue
            
        activations = activations_dict[layer_name]
        print(f"Layer: {layer_name}, Shape: {activations.shape}")
        
        # Prepare activation matrix
        activation_matrix = prepare_activation_matrix(activations)
        
        # Run GLasso
        precision_matrix, covariance_matrix = run_glasso(activation_matrix, alpha=args.alpha)
        
        # Analyze neuron connections
        connections = analyze_neuron_connections(precision_matrix)
        
        # Plot and save results
        plot_path = os.path.join(args.output_dir, f"{layer_name}_precision.png")
        plot_precision_matrix(precision_matrix, save_path=plot_path)
        
        # Save connection information
        connection_path = os.path.join(args.output_dir, f"{layer_name}_connections.txt")
        with open(connection_path, "w") as f:
            for neuron, connected in connections.items():
                f.write(f"Neuron {neuron} is connected to: {connected}\n")
        
        # Create and save GraphViz DOT file
        dot_path = os.path.join(args.output_dir, f"{layer_name}_graph.dot")
        create_graphviz_dot(connections, layer_name, save_path=dot_path)
        print(f"GraphViz DOT file saved to {dot_path}")
        
        # Create rendered graph image if graphviz is installed on the system and not skipped
        if not args.skip_viz:
            try:
                # Create a directory for rendered graphs
                viz_dir = os.path.join(args.output_dir, "visualizations")
                os.makedirs(viz_dir, exist_ok=True)
                
                # Create and render graph - we'll check if graphviz CLI tools are available first
                from shutil import which
                if which("dot"):
                    png_path = os.path.join(viz_dir, f"{layer_name}_graph")
                    create_graphviz_dot(connections, layer_name, save_path=png_path, render=True)
                    print(f"Rendered graph saved to {png_path}.png")
                else:
                    print("Note: Graphviz command-line tools not found. Skipping rendering to PNG.")
                    print("Install Graphviz (https://graphviz.org/) for PNG rendering.")
                    print("You can still use the generated DOT files with external tools.")
            except Exception as e:
                print(f"Warning: Could not render graph. Ensure graphviz is installed on your system: {e}")
        
        # Save matrices
        np.save(os.path.join(args.output_dir, f"{layer_name}_precision.npy"), precision_matrix)
        np.save(os.path.join(args.output_dir, f"{layer_name}_covariance.npy"), covariance_matrix)
        
        # Explicitly delete large objects to free memory
        # del activations_dict, activations, activation_matrix, precision_matrix, covariance_matrix
        # gc.collect()
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
