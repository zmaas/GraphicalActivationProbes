import argparse
import gc
import os
import pickle
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
    # Use the provided output directory directly (topic handling is done by the caller)
    layer_output_dir = output_dir
    os.makedirs(layer_output_dir, exist_ok=True)
    
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
    
    # Create NetworkX graph and save it
    nx_path = os.path.join(layer_output_dir, f"{layer_name}_graph")
    create_graphviz_dot(connections, layer_name, save_path=nx_path, precision_matrix=precision_matrix)
    
    # Generate GraphViz DOT file as well for backward compatibility
    dot_path = os.path.join(layer_output_dir, f"{layer_name}_graph.dot")
    with open(dot_path, "w") as f:
        # Generate DOT file content
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
        
        # Write to file
        f.write('\n'.join(dot_content))
    
    print(f"NetworkX graph saved to {nx_path}.nx")
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

def compare_sentiment_results(
    topic: str,
    layer_name: str,
    pos_precision: np.ndarray,
    neg_precision: np.ndarray,
    pos_connections: Dict[int, List[int]],
    neg_connections: Dict[int, List[int]],
    output_dir: str
) -> None:
    """Compare positive and negative sentiment results for a topic.
    
    Args:
        topic: The topic being analyzed
        layer_name: Name of the layer
        pos_precision: Precision matrix for positive examples
        neg_precision: Precision matrix for negative examples
        pos_connections: Neuron connections for positive examples
        neg_connections: Neuron connections for negative examples
        output_dir: Output directory
    """
    from src.glasso import precision_to_partial_correlation, connections_to_networkx, merge_networkx_graphs
    import networkx as nx
    
    # Create a comparisons directory
    comparison_dir = os.path.join(output_dir, "sentiment_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Create NetworkX graphs for positive and negative
    pos_graph = connections_to_networkx(pos_connections, f"{layer_name}_positive", pos_precision)
    neg_graph = connections_to_networkx(neg_connections, f"{layer_name}_negative", neg_precision)
    
    # Save individual graphs
    pos_graph_path = os.path.join(comparison_dir, f"{layer_name}_positive.nx")
    neg_graph_path = os.path.join(comparison_dir, f"{layer_name}_negative.nx")
    
    with open(pos_graph_path, 'wb') as f:
        pickle.dump(pos_graph, f)
    
    with open(neg_graph_path, 'wb') as f:
        pickle.dump(neg_graph, f)
    
    # Merge the graphs
    graphs = [pos_graph, neg_graph]
    merged_graph = merge_networkx_graphs(graphs, layer_name)
    
    # Save the merged graph
    merged_path = os.path.join(comparison_dir, f"{layer_name}_sentiment_merged.nx")
    with open(merged_path, 'wb') as f:
        pickle.dump(merged_graph, f)
    
    # Create visual comparison of precision matrices
    plt.figure(figsize=(15, 5))
    
    # Convert to partial correlations
    pos_partial_corr = precision_to_partial_correlation(pos_precision)
    neg_partial_corr = precision_to_partial_correlation(neg_precision)
    
    # Plot positive partial correlation
    plt.subplot(1, 3, 1)
    plt.imshow(pos_partial_corr, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(label='Partial Correlation')
    plt.title(f'{topic} - {layer_name} (Positive)')
    plt.xlabel('Neuron Index')
    plt.ylabel('Neuron Index')
    
    # Plot negative partial correlation
    plt.subplot(1, 3, 2)
    plt.imshow(neg_partial_corr, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(label='Partial Correlation')
    plt.title(f'{topic} - {layer_name} (Negative)')
    plt.xlabel('Neuron Index')
    plt.ylabel('Neuron Index')
    
    # Plot difference
    diff = pos_partial_corr - neg_partial_corr
    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Difference (Pos - Neg)')
    plt.title(f'{topic} - {layer_name} (Difference)')
    plt.xlabel('Neuron Index')
    plt.ylabel('Neuron Index')
    
    # Save comparison figure
    plt.tight_layout()
    comparison_path = os.path.join(comparison_dir, f"{layer_name}_sentiment_comparison.png")
    plt.savefig(comparison_path)
    plt.close()
    
    # Create a detailed report of the differences
    report_path = os.path.join(comparison_dir, f"{layer_name}_sentiment_analysis.txt")
    with open(report_path, 'w') as f:
        f.write(f"Sentiment Analysis for {topic} - {layer_name}\n")
        f.write("=" * 50 + "\n\n")
        
        # Write graph statistics
        f.write("Graph Statistics:\n")
        f.write(f"  Positive: {len(pos_graph.nodes())} nodes, {len(pos_graph.edges())} edges\n")
        f.write(f"  Negative: {len(neg_graph.nodes())} nodes, {len(neg_graph.edges())} edges\n")
        
        # Calculate shared connections
        shared_connections = set()
        for neuron, connections in pos_connections.items():
            if neuron in neg_connections:
                shared = set(connections) & set(neg_connections[neuron])
                if shared:
                    shared_connections.add(neuron)
                    for conn in shared:
                        shared_connections.add(conn)
        
        f.write(f"\nShared active neurons: {len(shared_connections)}\n")
        
        # Calculate unique connections (only in positive)
        pos_only_connections = set()
        for neuron, connections in pos_connections.items():
            if neuron not in neg_connections or not set(connections) & set(neg_connections.get(neuron, [])):
                pos_only_connections.add(neuron)
                for conn in connections:
                    pos_only_connections.add(conn)
        
        f.write(f"Neurons active only in positive: {len(pos_only_connections)}\n")
        
        # Calculate unique connections (only in negative)
        neg_only_connections = set()
        for neuron, connections in neg_connections.items():
            if neuron not in pos_connections or not set(connections) & set(pos_connections.get(neuron, [])):
                neg_only_connections.add(neuron)
                for conn in connections:
                    neg_only_connections.add(conn)
        
        f.write(f"Neurons active only in negative: {len(neg_only_connections)}\n")
        
        # Find strongest unique positive connections
        f.write("\nTop 10 strongest positive-only connections:\n")
        pos_only_edges = []
        for src, targets in pos_connections.items():
            for tgt in targets:
                if src not in neg_connections or tgt not in neg_connections.get(src, []):
                    strength = abs(pos_precision[src, tgt])
                    pos_only_edges.append((src, tgt, strength))
        
        # Sort by strength
        pos_only_edges.sort(key=lambda x: x[2], reverse=True)
        for src, tgt, strength in pos_only_edges[:10]:
            f.write(f"  Neuron {src} -> {tgt} (strength: {strength:.6f})\n")
        
        # Find strongest unique negative connections
        f.write("\nTop 10 strongest negative-only connections:\n")
        neg_only_edges = []
        for src, targets in neg_connections.items():
            for tgt in targets:
                if src not in pos_connections or tgt not in pos_connections.get(src, []):
                    strength = abs(neg_precision[src, tgt])
                    neg_only_edges.append((src, tgt, strength))
        
        # Sort by strength
        neg_only_edges.sort(key=lambda x: x[2], reverse=True)
        for src, tgt, strength in neg_only_edges[:10]:
            f.write(f"  Neuron {src} -> {tgt} (strength: {strength:.6f})\n")
    
    print(f"Sentiment comparison for {layer_name} saved to {comparison_dir}")

def compare_and_merge_topic_graphs(
    topics: List[str],
    output_dir: str,
    layer_name: str
) -> None:
    """Compare and merge NetworkX graphs across different topics.
    
    Args:
        topics: List of topics to compare
        output_dir: Base output directory
        layer_name: Name of the layer to compare
    """
    from src.glasso import load_networkx_graph, merge_networkx_graphs
    import networkx as nx
    
    # Create a directory for comparisons
    comparison_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    
    if len(topics) < 2:
        print("Cannot compare topic graphs: need at least 2 topics")
        return
    
    # Load NetworkX graphs for each topic
    graphs = []
    topic_graphs = {}
    
    for topic in topics:
        topic_dir = os.path.join(output_dir, topic)
        nx_path = os.path.join(topic_dir, f"{layer_name}_graph.nx")
        
        if os.path.exists(nx_path):
            try:
                G = load_networkx_graph(nx_path)
                graphs.append(G)
                topic_graphs[topic] = G
                print(f"Loaded graph for {topic}: {len(G.nodes())} nodes, {len(G.edges())} edges")
            except Exception as e:
                print(f"Error loading graph for topic {topic}: {e}")
        else:
            print(f"NetworkX graph not found for topic {topic} at {nx_path}")
    
    if len(graphs) < 2:
        print("Not enough graphs found for comparison.")
        return
    
    # Merge the graphs
    merged_graph = merge_networkx_graphs(graphs, layer_name)
    print(f"Created merged graph with {len(merged_graph.nodes())} nodes and {len(merged_graph.edges())} edges")
    
    # Save the merged graph
    merged_path = os.path.join(comparison_dir, f"{layer_name}_merged_graph.nx")
    with open(merged_path, 'wb') as f:
        pickle.dump(merged_graph, f)
    print(f"Merged graph saved to {merged_path}")
    
    # Create a summary of the comparison
    summary_path = os.path.join(comparison_dir, f"{layer_name}_graph_comparison.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Graph comparison for layer {layer_name}:\n\n")
        
        # Write information about each individual graph
        f.write("Individual graphs:\n")
        for topic, G in topic_graphs.items():
            f.write(f"  {topic}:\n")
            f.write(f"    Nodes: {len(G.nodes())}\n")
            f.write(f"    Edges: {len(G.edges())}\n")
            # Calculate graph metrics
            if len(G.nodes()) > 0:
                f.write(f"    Density: {nx.density(G):.6f}\n")
                if nx.is_connected(G.to_undirected()):
                    f.write(f"    Diameter: {nx.diameter(G.to_undirected())}\n")
                else:
                    f.write("    Diameter: Graph is not connected\n")
                
                # Find most central nodes
                betweenness = nx.betweenness_centrality(G)
                sorted_centrality = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
                f.write("    Most central nodes:\n")
                for node, score in sorted_centrality[:5]:  # Top 5 nodes
                    f.write(f"      Node {node}: {score:.6f}\n")
            f.write("\n")
        
        # Write information about the merged graph
        f.write("\nMerged graph:\n")
        f.write(f"  Nodes: {len(merged_graph.nodes())}\n")
        f.write(f"  Edges: {len(merged_graph.edges())}\n")
        f.write(f"  Density: {nx.density(merged_graph):.6f}\n")
        
        # Check if there are connections between topics
        cross_topic_edges = []
        for u, v, data in merged_graph.edges(data=True):
            u_topic = int(u.split("_")[0][1:])  # Extract graph_id from node name "g0_123"
            v_topic = int(v.split("_")[0][1:])
            if u_topic != v_topic:
                cross_topic_edges.append((u, v))
        
        f.write(f"\nCross-topic connections: {len(cross_topic_edges)}\n")
        if cross_topic_edges:
            f.write("  Sample connections:\n")
            for u, v in cross_topic_edges[:10]:  # Show first 10
                u_parts = u.split("_")
                v_parts = v.split("_")
                u_topic = topics[int(u_parts[0][1:])]
                v_topic = topics[int(v_parts[0][1:])]
                u_node = u_parts[1]
                v_node = v_parts[1]
                f.write(f"    {u_topic} node {u_node} -> {v_topic} node {v_node}\n")
    
    print(f"Comparison summary saved to {summary_path}")

def analyze_topic(
    topic: str,
    model: ModelWithActivations,
    df: pd.DataFrame,
    args: argparse.Namespace
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict[int, List[int]]]]:
    """Analyze a single topic with the model."""
    # Create topic output directory - avoid nesting by using a flat structure with topic in the name
    topic_output_dir = os.path.join(args.output_dir, f"{topic}")
    os.makedirs(topic_output_dir, exist_ok=True)
    
    # Check if we need to analyze based on sentiment
    has_sentiment = 'label' in df.columns and not 'topic' in df.columns
    
    if has_sentiment:
        # We have positive vs negative sentiment for the same topic
        # Analyze positive and negative examples separately
        return analyze_topic_with_sentiment(topic, model, df, args, topic_output_dir)
    else:
        # Just analyzing a single topic or comparing different topics
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
                skip_viz=args.skip_viz
            )
            if layer_results is not None:
                results[layer_name] = layer_results
        
        return results

def analyze_topic_with_sentiment(
    topic: str,
    model: ModelWithActivations,
    df: pd.DataFrame,
    args: argparse.Namespace,
    topic_output_dir: str
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict[int, List[int]]]]:
    """Analyze a topic with positive and negative sentiment examples.
    
    This separates positive and negative examples, processes them separately,
    and then compares the results.
    """
    print(f"Analyzing topic '{topic}' with sentiment comparison (positive vs negative)")
    
    # Create directories for positive and negative sentiment
    pos_dir = os.path.join(topic_output_dir, "positive")
    neg_dir = os.path.join(topic_output_dir, "negative")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)
    
    # Split dataframe into positive and negative examples
    pos_df = df[df['label'] == 1].copy()
    neg_df = df[df['label'] == 0].copy()
    
    print(f"Positive examples: {len(pos_df)}")
    print(f"Negative examples: {len(neg_df)}")
    
    # Check that we have examples of both
    if len(pos_df) == 0 or len(neg_df) == 0:
        print("Error: Need both positive and negative examples for sentiment analysis")
        return {}
    
    # Process positive examples
    print(f"\n===== Processing POSITIVE examples for topic: {topic} =====")
    pos_dataset = SentimentDataset(
        texts=pos_df["text"].tolist(), 
        labels=pos_df["label"].tolist(),
        tokenizer=model.tokenizer
    )
    pos_dataloader = get_dataloader(pos_dataset, batch_size=args.batch_size)
    
    # Collect activations for positive examples
    print(f"Collecting activations for POSITIVE {topic}...")
    pos_activations = model.collect_activations(pos_dataloader)
    
    # Process negative examples
    print(f"\n===== Processing NEGATIVE examples for topic: {topic} =====")
    neg_dataset = SentimentDataset(
        texts=neg_df["text"].tolist(), 
        labels=neg_df["label"].tolist(),
        tokenizer=model.tokenizer
    )
    neg_dataloader = get_dataloader(neg_dataset, batch_size=args.batch_size)
    
    # Collect activations for negative examples
    print(f"Collecting activations for NEGATIVE {topic}...")
    neg_activations = model.collect_activations(neg_dataloader)
    
    # Get layer names (should be the same for both)
    layer_names = list(set(pos_activations.keys()) & set(neg_activations.keys()))
    
    # Process each layer for both positive and negative
    results = {}
    sentiment_graphs = {}
    
    for layer_name in tqdm(layer_names):
        # Process positive examples
        pos_results = process_layer(
            layer_name=layer_name,
            activations_dict=pos_activations,
            output_dir=pos_dir,
            alpha=args.alpha,
            skip_viz=args.skip_viz
        )
        
        # Process negative examples
        neg_results = process_layer(
            layer_name=layer_name,
            activations_dict=neg_activations,
            output_dir=neg_dir,
            alpha=args.alpha,
            skip_viz=args.skip_viz
        )
        
        if pos_results is not None and neg_results is not None:
            # Extract results
            pos_precision, pos_covariance, pos_connections = pos_results
            neg_precision, neg_covariance, neg_connections = neg_results
            
            # Store results for combined analysis
            results[layer_name] = (pos_precision, pos_covariance, pos_connections)
            
            # Compare positive and negative results
            compare_sentiment_results(
                topic=topic,
                layer_name=layer_name,
                pos_precision=pos_precision,
                neg_precision=neg_precision,
                pos_connections=pos_connections,
                neg_connections=neg_connections,
                output_dir=topic_output_dir
            )
    
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
        
        # Process each topic - use a flat directory structure
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
            
            # Analyze this topic - create a directory specifically for this topic
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
            # Also compare and merge NetworkX graphs
            compare_and_merge_topic_graphs(args.compare_topics, args.output_dir, layer_name)
        
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