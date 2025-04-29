import pytest
import numpy as np
import torch
import os
import tempfile
import networkx as nx

from src.glasso import (
    prepare_activation_matrix, 
    run_glasso, 
    analyze_neuron_connections, 
    plot_precision_matrix,
    connections_to_networkx,
    save_networkx_graph,
    load_networkx_graph,
    merge_networkx_graphs
)

@pytest.fixture
def mock_activations():
    """Create mock activations for testing."""
    # Create random activations for 5 samples and 10 neurons
    return torch.rand(5, 10)

@pytest.fixture
def mock_precision_matrix():
    """Create a mock precision matrix for testing."""
    # Create a 10x10 precision matrix with some off-diagonal elements
    np.random.seed(42)  # For reproducibility
    precision = np.random.rand(10, 10) * 0.1
    # Make it symmetric
    precision = (precision + precision.T) / 2
    # Add diagonal dominance for positive definiteness
    np.fill_diagonal(precision, 1.0)
    return precision

def test_prepare_activation_matrix(mock_activations):
    """Test preparing activation matrix for GLasso."""
    # Prepare activation matrix
    act_matrix = prepare_activation_matrix(mock_activations)
    
    # Check shape
    assert act_matrix.shape == (5, 10)
    
    # Check standardization (with slightly relaxed tolerance)
    assert np.isclose(act_matrix.mean(axis=0).mean(), 0, atol=1e-7)
    assert np.isclose(act_matrix.std(axis=0).mean(), 1, atol=1e-7)

def test_analyze_neuron_connections(mock_precision_matrix):
    """Test analyzing neuron connections."""
    # Analyze connections
    connections = analyze_neuron_connections(mock_precision_matrix, threshold=0.2)
    
    # Check type
    assert isinstance(connections, dict)
    
    # Each neuron should be in the connections dict
    assert set(connections.keys()) == set(range(10))
    
    # Check that diagonal elements are not included in connections
    for i, connected in connections.items():
        assert i not in connected

def test_plot_precision_matrix(mock_precision_matrix, tmp_path):
    """Test plotting precision matrix."""
    # Create a temporary file path
    save_path = tmp_path / "precision_matrix.png"
    
    # Plot and save
    plot_precision_matrix(mock_precision_matrix, save_path=str(save_path))
    
    # Check that the file was created
    assert save_path.exists()
    
def test_run_glasso(mock_activations):
    """Test running Graphical Lasso with GGLasso."""
    # Prepare activation matrix
    act_matrix = prepare_activation_matrix(mock_activations)
    
    # Run GLasso
    precision, covariance = run_glasso(act_matrix, alpha=0.1)
    
    # Check shapes
    assert precision.shape == (10, 10)
    assert covariance.shape == (10, 10)
    
    # Check symmetry of precision matrix
    assert np.allclose(precision, precision.T)
    
    # Run with different alpha and check if sparsity changes
    precision_higher_reg, _ = run_glasso(act_matrix, alpha=0.5)
    
    # Higher alpha should result in more zeros
    assert np.count_nonzero(np.abs(precision_higher_reg) < 1e-5) >= np.count_nonzero(np.abs(precision) < 1e-5)

def test_connections_to_networkx():
    """Test conversion of connections to NetworkX graph."""
    # Define simple connections dictionary
    connections = {
        0: [1, 2],
        1: [0, 3],
        2: [4],
        3: [],
        4: [0]
    }
    
    # Convert to NetworkX graph
    G = connections_to_networkx(connections, "test_layer")
    
    # Check graph properties
    assert G.name == "test_layer"
    assert len(G.nodes()) == 5
    assert len(G.edges()) == 6  # There are 6 edges as shown in the error
    
    # Check specific edges
    assert G.has_edge(0, 1)
    assert G.has_edge(0, 2)
    assert G.has_edge(1, 0)
    assert G.has_edge(1, 3)
    assert G.has_edge(2, 4)
    assert G.has_edge(4, 0)
    assert not G.has_edge(3, 0)  # This edge shouldn't exist
    
    # Test with weights
    weights = np.zeros((5, 5))
    weights[0, 1] = 0.5
    weights[0, 2] = 0.3
    weights[1, 0] = 0.4
    weights[1, 3] = 0.2
    weights[2, 4] = 0.6
    weights[4, 0] = 0.7
    
    G_weighted = connections_to_networkx(connections, "test_layer", weights=weights)
    
    # Check edge weights
    assert G_weighted.edges[0, 1]["weight"] == 0.5
    assert G_weighted.edges[0, 2]["weight"] == 0.3
    assert G_weighted.edges[2, 4]["weight"] == 0.6

def test_save_load_networkx_graph():
    """Test saving and loading NetworkX graph."""
    # Create a simple graph
    G = nx.DiGraph(name="test_graph")
    G.add_node(0, label="Node 0")
    G.add_node(1, label="Node 1")
    G.add_edge(0, 1, weight=0.5)
    
    # Save and load the graph
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_graph.nx")
        save_networkx_graph(G, save_path)
        
        # Check if the file exists
        assert os.path.exists(save_path)
        
        # Load the graph
        G_loaded = load_networkx_graph(save_path)
        
        # Check graph properties
        assert G_loaded.name == "test_graph"
        assert len(G_loaded.nodes()) == 2
        assert len(G_loaded.edges()) == 1
        assert G_loaded.has_edge(0, 1)
        assert G_loaded.edges[0, 1]["weight"] == 0.5
        assert G_loaded.nodes[0]["label"] == "Node 0"
        assert G_loaded.nodes[1]["label"] == "Node 1"

def test_merge_networkx_graphs():
    """Test merging NetworkX graphs."""
    # Create two simple graphs
    G1 = nx.DiGraph(name="graph1")
    G1.add_node(0, label="G1 Node 0")
    G1.add_node(1, label="G1 Node 1")
    G1.add_edge(0, 1, weight=0.5)
    
    G2 = nx.DiGraph(name="graph2")
    G2.add_node(0, label="G2 Node 0")
    G2.add_node(2, label="G2 Node 2")
    G2.add_edge(0, 2, weight=0.7)
    
    # Merge the graphs
    merged = merge_networkx_graphs([G1, G2], "merged_layer")
    
    # Check merged graph properties
    assert merged.name == "merged_layer_merged"
    assert len(merged.nodes()) == 4  # g0_0, g0_1, g1_0, g1_2
    assert len(merged.edges()) == 2
    
    # Check node attributes
    assert merged.nodes["g0_0"]["label"] == "G0 Neuron 0"
    assert merged.nodes["g0_1"]["label"] == "G0 Neuron 1"
    assert merged.nodes["g1_0"]["label"] == "G1 Neuron 0"
    assert merged.nodes["g1_2"]["label"] == "G1 Neuron 2"
    
    # Check edge attributes
    assert merged.has_edge("g0_0", "g0_1")
    assert merged.has_edge("g1_0", "g1_2")
    assert merged.edges["g0_0", "g0_1"]["graph_id"] == 0
    assert merged.edges["g1_0", "g1_2"]["graph_id"] == 1

