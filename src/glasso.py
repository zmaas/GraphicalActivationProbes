from typing import Dict, List, Optional, Tuple, Union
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from gglasso.problem import glasso_problem
from sklearn.preprocessing import StandardScaler
import graphviz
import networkx as nx

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
    """Run Graphical Lasso on activation matrix.
    
    Args:
        activation_matrix: Matrix of standardized activations (samples x neurons)
        alpha: Regularization parameter for GLasso.
        
    Returns:
        Tuple of (precision matrix, covariance matrix)
    """
    # Calculate empirical covariance matrix
    n_samples, n_features = activation_matrix.shape
    emp_cov = np.dot(activation_matrix.T, activation_matrix) / n_samples
    
    # Use GGLasso implementation
    problem = glasso_problem(S=emp_cov, N=n_samples, do_scaling=False)
    problem.set_reg_params(reg_params={"lambda1": 0.1, "lambda2": 0.01})
    
    # Solve the problem
    problem.solve(verbose=False, tol=1e-6, rtol=1e-5)
    
    # Get precision and covariance matrices
    precision_matrix = problem.solution.precision_
    covariance_matrix = emp_cov
    
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
    
def connections_to_networkx(connections: Dict[int, List[int]], 
                            layer_name: str,
                            weights: Optional[np.ndarray] = None) -> nx.DiGraph:
    """Convert neuron connections to a NetworkX graph.
    
    Args:
        connections: Dictionary mapping neurons to their connected neurons
        layer_name: Name of the layer
        weights: Precision matrix to use for edge weights (optional)
        
    Returns:
        NetworkX directed graph
    """
    # Create a directed graph
    G = nx.DiGraph(name=layer_name)
    
    # Add all neurons as nodes first
    all_neurons = set(connections.keys())
    for neuron_list in connections.values():
        all_neurons.update(neuron_list)
    
    for neuron in all_neurons:
        G.add_node(neuron, label=f"Neuron {neuron}")
    
    # Add edges with weights if provided
    for source, targets in connections.items():
        for target in targets:
            if weights is not None:
                # Use the absolute value of the precision matrix entry as the weight
                weight = abs(weights[source, target])
                G.add_edge(source, target, weight=weight)
            else:
                G.add_edge(source, target)
    
    return G

def save_networkx_graph(G: nx.DiGraph, save_path: str) -> None:
    """Save a NetworkX graph to a file.
    
    Args:
        G: NetworkX graph
        save_path: Path to save the graph
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save as pickle
    with open(save_path, 'wb') as f:
        pickle.dump(G, f)
        
def load_networkx_graph(load_path: str) -> nx.DiGraph:
    """Load a NetworkX graph from a file.
    
    Args:
        load_path: Path to load the graph from
        
    Returns:
        NetworkX graph
    """
    with open(load_path, 'rb') as f:
        G = pickle.load(f)
    return G

def merge_networkx_graphs(graphs: List[nx.DiGraph], layer_name: str) -> nx.DiGraph:
    """Merge multiple NetworkX graphs into one.
    
    Args:
        graphs: List of NetworkX graphs to merge
        layer_name: Name of the merged layer
        
    Returns:
        Merged NetworkX graph
    """
    merged_graph = nx.DiGraph(name=f"{layer_name}_merged")
    
    # Add all nodes and edges from all graphs
    for i, G in enumerate(graphs):
        # Add a prefix to each node to distinguish nodes from different graphs
        for node in G.nodes():
            merged_graph.add_node(f"g{i}_{node}", label=f"G{i} Neuron {node}", graph_id=i)
        
        # Add edges with the same prefix
        for u, v, data in G.edges(data=True):
            merged_graph.add_edge(f"g{i}_{u}", f"g{i}_{v}", **data, graph_id=i)
    
    return merged_graph

def create_graphviz_dot(connections: Dict[int, List[int]], 
                       layer_name: str,
                       save_path: Optional[str] = None,
                       render: bool = False,
                       precision_matrix: Optional[np.ndarray] = None) -> Union[str, graphviz.Digraph, nx.DiGraph]:
    """Create a GraphViz DOT representation of neuron connections.
    
    Args:
        connections: Dictionary mapping neurons to their connected neurons
        layer_name: Name of the layer (used for graph title)
        save_path: Path to save the DOT file (optional)
        render: Whether to render the graph using the graphviz library (returns Digraph object)
        precision_matrix: Precision matrix to use for edge weights (optional)
        
    Returns:
        String containing the DOT representation, graphviz.Digraph object if render=True,
        or NetworkX graph
    """
    # Create NetworkX graph
    nx_graph = connections_to_networkx(connections, layer_name, precision_matrix)
    
    # Save NetworkX graph if path is provided
    if save_path:
        nx_path = f"{os.path.splitext(save_path)[0]}.nx"
        save_networkx_graph(nx_graph, nx_path)
        print(f"NetworkX graph saved to {nx_path}")
    
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
        
        return nx_graph
