import pytest
import numpy as np
import torch
from src.glasso import prepare_activation_matrix, run_glasso, analyze_neuron_connections, plot_precision_matrix

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
    
    # Run GLasso with default alpha
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
