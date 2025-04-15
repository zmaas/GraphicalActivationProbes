#!/usr/bin/env python
"""
Example script to analyze a single synthetic topic.

This script demonstrates how to use the synthetic data generation
functionality in GLasso Interp to analyze a specific topic.

Example usage:
    python examples/synthetic_topic.py --synthetic_topic "quantum computing" --num_examples 60
"""

import os
import sys
import argparse

# Add parent directory to path to import from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import main

if __name__ == "__main__":
    # Re-use the main function, but with a better help message
    parser = argparse.ArgumentParser(description="Analyze neural network activations on synthetic topic data.")
    
    # Dataset options
    parser.add_argument(
        "--synthetic_topic", 
        type=str,
        required=True,
        help="Topic for synthetic data generation (required)"
    )
    parser.add_argument(
        "--num_examples", 
        type=int, 
        default=60,
        help="Number of examples to generate (will be split between positive/negative) (default: 60)"
    )
    parser.add_argument(
        "--force_regenerate", 
        action="store_true",
        help="Force regeneration of synthetic data (ignore cache)"
    )
    
    # Model options
    parser.add_argument(
        "--model", 
        type=str, 
        default="google/gemma-3-1b-it", 
        help="Model to analyze (default: google/gemma-3-1b-it)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Batch size for processing (default: 8)"
    )
    
    # GLasso options
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.1, 
        help="GLasso regularization parameter (default: 0.1)"
    )
    
    # Output options
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/synthetic", 
        help="Output directory (default: results/synthetic)"
    )
    parser.add_argument(
        "--skip_viz", 
        action="store_true",
        help="Skip generating visualizations"
    )
    
    args = parser.parse_args()
    
    # Set dataset to synthetic to use synthetic data generation
    setattr(args, 'dataset', 'synthetic')
    
    # Call the main function with the parsed arguments
    main()