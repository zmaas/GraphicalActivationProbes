import argparse
import gc
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data import SentimentDataset, get_dataloader, load_sentiment_dataset
from glasso import (analyze_neuron_connections, create_graphviz_dot,
                   plot_precision_matrix, prepare_activation_matrix, run_glasso)
from model import ModelWithActivations
from synthetic_data import (create_sentiment_dataset,
                           generate_multiclass_dataset)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GLasso interpretation on transformer model.")
    
    # Dataset options
    dataset_group = parser.add_argument_group("Dataset Options")
    dataset_group.add_argument(
        "--dataset", 
        type=str, 
        default="imdb", 
        help="Dataset to use. Use 'synthetic' for synthetic data."
    )
    dataset_group.add_argument(
        "--synthetic_topic", 
        type=str, 
        default=None,
        help="Topic for synthetic data generation (only used if dataset='synthetic')"
    )
    dataset_group.add_argument(
        "--num_examples", 
        type=int, 
        default=60,
        help="Number of examples to generate for synthetic data (will be split between positive/negative)"
    )
    dataset_group.add_argument(
        "--force_regenerate", 
        action="store_true",
        help="Force regeneration of synthetic data (ignore cache)"
    )
    
    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--model", 
        type=str, 
        default="google/gemma-3-1b-it", 
        help="Model to analyze"
    )
    model_group.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Batch size"
    )
    
    # GLasso options
    glasso_group = parser.add_argument_group("GLasso Options")
    glasso_group.add_argument(
        "--alpha", 
        type=float, 
        default=None, 
        help="GLasso regularization parameter (None for CV)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output_dir", 
        type=str, 
        default="results", 
        help="Output directory"
    )
    output_group.add_argument(
        "--skip_viz", 
        action="store_true",
        help="Skip generating visualizations (useful for headless environments)"
    )
    
    # Comparison options
    comparison_group = parser.add_argument_group("Comparison Options")
    comparison_group.add_argument(
        "--compare_topics", 
        nargs="+", 
        type=str,
        help="List of topics to compare (requires dataset='synthetic')"
    )
    
    return parser.parse_args()

def load_or_generate_dataset(args: argparse.Namespace) -> pd.DataFrame:
    """Load dataset or generate synthetic data based on args."""
    if args.dataset.lower() == "synthetic":
        if args.synthetic_topic:
            print(f"Generating synthetic sentiment data for topic: {args.synthetic_topic}")
            num_per_class = args.num_examples // 2
            return create_sentiment_dataset(
                topic=args.synthetic_topic,
                num_positive=num_per_class,
                num_negative=num_per_class,
                force_regenerate=args.force_regenerate
            )
        elif args.compare_topics:
            print(f"Generating synthetic multi-topic data for: {', '.join(args.compare_topics)}")
            num_per_topic = args.num_examples // len(args.compare_topics)
            return generate_multiclass_dataset(
                topics=args.compare_topics,
                num_examples_per_topic=num_per_topic,
                force_regenerate=args.force_regenerate
            )
        else:
            print("Error: synthetic_topic or compare_topics must be provided when using synthetic data")
            sys.exit(1)
    else:
        print(f"Loading dataset {args.dataset}...")
        return load_sentiment_dataset(dataset_name=args.dataset)

def process_layer(
    layer_name: str,
    activations_dict: Dict[str, torch.Tensor],
    output_dir: str,
    alpha: Optional[float] = None,
    skip_viz: bool = False,
    topic: Optional[str] = None
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[int, List[int]]]]:
    """Process a single layer to extract GLasso results."""
    # Add topic to output directory path if provided
    if topic:
        layer_output_dir = os.path.join(output_dir, topic)
        os.makedirs(layer_output_dir, exist_ok=True)
    else:
        layer_output_dir = output_dir
    
    print(f"Processing layer {layer_name}...")
    
    if layer_name not in activations_dict:
        print(f"Warning: No activations collected for {layer_name}, skipping")
        return None
        
    activations = activations_dict[layer_name]
    print(f"Layer: {layer_name}, Shape: {activations.shape}")
    
    # Prepare activation matrix
    activation_matrix = prepare_activation_matrix(activations)
    
    # Run GLasso
    precision_matrix, covariance_matrix = run_glasso(activation_matrix, alpha=alpha)
    
    # Analyze neuron connections
    connections = analyze_neuron_connections(precision_matrix)
    
    # Plot and save results
    plot_path = os.path.join(layer_output_dir, f"{layer_name}_precision.png")
    plot_precision_matrix(precision_matrix, save_path=plot_path)
    
    # Save connection information
    connection_path = os.path.join(layer_output_dir, f"{layer_name}_connections.txt")
    with open(connection_path, "w") as f:
        for neuron, connected in connections.items():
            f.write(f"Neuron {neuron} is connected to: {connected}\n")
    
    # Create and save GraphViz DOT file
    dot_path = os.path.join(layer_output_dir, f"{layer_name}_graph.dot")
    create_graphviz_dot(connections, layer_name, save_path=dot_path)
    print(f"GraphViz DOT file saved to {dot_path}")
    
    # Create rendered graph image if graphviz is installed on the system and not skipped
    if not skip_viz:
        try:
            # Create a directory for rendered graphs
            viz_dir = os.path.join(layer_output_dir, "visualizations")
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
    np.save(os.path.join(layer_output_dir, f"{layer_name}_precision.npy"), precision_matrix)
    np.save(os.path.join(layer_output_dir, f"{layer_name}_covariance.npy"), covariance_matrix)
    
    return precision_matrix, covariance_matrix, connections

def compare_precision_matrices(
    precision_matrices: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
    layer_name: str
) -> None:
    """Compare precision matrices across different topics for a specific layer."""
    # Create a directory for comparisons
    comparison_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    
    topics = list(precision_matrices.keys())
    
    if len(topics) < 2:
        print("Cannot compare precision matrices: need at least 2 topics")
        return
    
    # Create comparison visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot heatmaps for each topic
    num_topics = len(topics)
    for topic_idx, topic in enumerate(topics):
        if layer_name not in precision_matrices[topic]:
            print(f"Warning: Layer {layer_name} not found for topic {topic}, skipping comparison")
            return
            
        plt.subplot(2, num_topics, topic_idx + 1)
        precmat = precision_matrices[topic][layer_name]
        partial_corr = np.zeros_like(precmat)
        for i in range(precmat.shape[0]):
            for j in range(precmat.shape[1]):
                if i == j:
                    partial_corr[i, j] = 1.0
                else:
                    partial_corr[i, j] = -precmat[i, j] / np.sqrt(precmat[i, i] * precmat[j, j])
        
        plt.imshow(partial_corr, cmap='viridis', vmin=-1, vmax=1)
        plt.colorbar(label='Partial Correlation')
        plt.title(f'{topic} - {layer_name}')
        plt.xlabel('Neuron Index')
        plt.ylabel('Neuron Index')
    
    # Plot differences between pairs
    for i in range(len(topics) - 1):
        for j in range(i + 1, len(topics)):
            topic1, topic2 = topics[i], topics[j]
            precmat1 = precision_matrices[topic1][layer_name]
            precmat2 = precision_matrices[topic2][layer_name]
            
            # Calculate difference in partial correlations
            partial_corr1 = np.zeros_like(precmat1)
            partial_corr2 = np.zeros_like(precmat2)
            
            for x in range(precmat1.shape[0]):
                for y in range(precmat1.shape[1]):
                    if x == y:
                        partial_corr1[x, y] = 1.0
                        partial_corr2[x, y] = 1.0
                    else:
                        partial_corr1[x, y] = -precmat1[x, y] / np.sqrt(precmat1[x, x] * precmat1[y, y])
                        partial_corr2[x, y] = -precmat2[x, y] / np.sqrt(precmat2[x, x] * precmat2[y, y])
            
            diff = partial_corr1 - partial_corr2
            
            # Plot the difference
            idx = (i * (len(topics) - 1)) + j
            plt.subplot(2, num_topics, num_topics + idx)
            plt.imshow(diff, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(label='Difference')
            plt.title(f'Diff: {topic1} vs {topic2}')
            plt.xlabel('Neuron Index')
            plt.ylabel('Neuron Index')
    
    # Save comparison figure
    comparison_path = os.path.join(comparison_dir, f"{layer_name}_comparison.png")
    plt.tight_layout()
    plt.savefig(comparison_path)
    plt.close()
    print(f"Comparison visualization saved to {comparison_path}")

def analyze_topic(
    topic: str,
    model: ModelWithActivations,
    df: pd.DataFrame,
    args: argparse.Namespace
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict[int, List[int]]]]:
    """Analyze a single topic with the model."""
    # Create topic output directory
    topic_output_dir = os.path.join(args.output_dir, topic)
    os.makedirs(topic_output_dir, exist_ok=True)
    
    # Make sure we only have text and label columns
    if 'topic' in df.columns:
        # Filter to just this topic and create a binary label
        topic_df = df[df['topic'] == topic].copy()
        topic_df['label'] = 1  # Arbitrary label since we're analyzing a single topic
    else:
        topic_df = df
    
    # Create dataset and dataloader
    dataset = SentimentDataset(
        texts=topic_df["text"].tolist(), 
        labels=topic_df["label"].tolist(),
        tokenizer=model.tokenizer
    )
    dataloader = get_dataloader(dataset, batch_size=args.batch_size)
    
    # Collect activations for all layers
    print(f"Collecting activations for topic: {topic}...")
    activations_dict = model.collect_activations(dataloader)
    
    # Get layer names
    layer_names = list(activations_dict.keys())
    
    # Process each layer
    results = {}
    for layer_name in tqdm(layer_names):
        layer_results = process_layer(
            layer_name=layer_name,
            activations_dict=activations_dict,
            output_dir=topic_output_dir,
            alpha=args.alpha,
            skip_viz=args.skip_viz,
            topic=topic
        )
        if layer_results is not None:
            results[layer_name] = layer_results
    
    return results

def main() -> None:
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model {args.model}...")
    model = ModelWithActivations(model_name=args.model)
    tokenizer = model.tokenizer
    
    # Load or generate dataset
    df = load_or_generate_dataset(args)
    
    # Special case for comparing multiple topics
    if args.compare_topics:
        # Store precision matrices for each topic
        precision_matrices = {}
        
        # Process each topic
        for topic in args.compare_topics:
            print(f"\n===== Processing topic: {topic} =====")
            
            # Filter to just this topic if using a multi-topic dataset
            if 'topic' in df.columns:
                topic_df = df[df['topic'] == topic]
            else:
                # Generate synthetic data for each topic individually if needed
                topic_df = create_sentiment_dataset(
                    topic=topic,
                    num_positive=args.num_examples // 2,
                    num_negative=args.num_examples // 2,
                    force_regenerate=args.force_regenerate
                )
            
            # Analyze this topic
            topic_results = analyze_topic(topic, model, topic_df, args)
            
            # Store precision matrices for comparison
            topic_precision = {}
            for layer_name, (prec_mat, _, _) in topic_results.items():
                topic_precision[layer_name] = prec_mat
            
            precision_matrices[topic] = topic_precision
        
        # Compare precision matrices for each layer
        print("\n===== Comparing topics =====")
        layer_names = list(next(iter(precision_matrices.values())).keys())
        for layer_name in layer_names:
            compare_precision_matrices(precision_matrices, args.output_dir, layer_name)
        
    else:
        # Detect layers with registered hooks
        print("Detecting layers with registered hooks...")
        # Run a sample batch to detect which layers have hooks
        # Create a small dataset for this
        small_df = df.head(2)
        sample_dataset = SentimentDataset(
            texts=small_df["text"].tolist(), 
            labels=small_df["label"].tolist(),
            tokenizer=tokenizer
        )
        sample_dataloader = get_dataloader(sample_dataset, batch_size=1)
        sample_batch = next(iter(sample_dataloader))
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
        
        # Use a smaller subset for faster processing if needed
        subset_size = int(os.environ.get("SUBSET_SIZE", str(min(len(df), 100))))
        print(f"Using {subset_size} samples for analysis")
        df = df.head(subset_size)
        
        # Process the dataset as a single topic (using synthetic_topic if provided)
        topic = args.synthetic_topic if args.synthetic_topic else args.dataset
        analyze_topic(topic, model, df, args)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()