#!/usr/bin/env python
"""
Test script to verify sentiment comparison functionality
"""

import os
import sys
import shutil
import tempfile
import unittest
import argparse
import numpy as np
import pandas as pd
import pickle

import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.main import compare_sentiment_results


class TestSentimentComparison(unittest.TestCase):
    """Test that sentiment comparison works correctly."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock precision matrices
        self.pos_precision = np.eye(10)
        self.neg_precision = np.eye(10)
        
        # Add some off-diagonal elements
        self.pos_precision[0, 1] = 0.5
        self.pos_precision[1, 0] = 0.5
        self.pos_precision[2, 3] = 0.3
        self.pos_precision[3, 2] = 0.3
        
        self.neg_precision[4, 5] = 0.4
        self.neg_precision[5, 4] = 0.4
        self.neg_precision[6, 7] = 0.2
        self.neg_precision[7, 6] = 0.2
        
        # Create mock connections
        self.pos_connections = {
            0: [1],
            1: [0],
            2: [3],
            3: [2]
        }
        
        self.neg_connections = {
            4: [5],
            5: [4],
            6: [7],
            7: [6]
        }
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_compare_sentiment_results(self):
        """Test sentiment comparison functionality."""
        # Call the function
        compare_sentiment_results(
            topic="test_topic",
            layer_name="test_layer",
            pos_precision=self.pos_precision,
            neg_precision=self.neg_precision,
            pos_connections=self.pos_connections,
            neg_connections=self.neg_connections,
            output_dir=self.test_dir
        )
        
        # Check that expected files were created
        sentiment_dir = os.path.join(self.test_dir, "sentiment_comparison")
        self.assertTrue(os.path.exists(sentiment_dir))
        
        # Check that individual graphs were saved
        pos_graph_path = os.path.join(sentiment_dir, "test_layer_positive.nx")
        neg_graph_path = os.path.join(sentiment_dir, "test_layer_negative.nx")
        self.assertTrue(os.path.exists(pos_graph_path))
        self.assertTrue(os.path.exists(neg_graph_path))
        
        # Check that merged graph was saved
        merged_path = os.path.join(sentiment_dir, "test_layer_sentiment_merged.nx")
        self.assertTrue(os.path.exists(merged_path))
        
        # Check that comparison image was created
        comparison_path = os.path.join(sentiment_dir, "test_layer_sentiment_comparison.png")
        self.assertTrue(os.path.exists(comparison_path))
        
        # Check that analysis report was created
        report_path = os.path.join(sentiment_dir, "test_layer_sentiment_analysis.txt")
        self.assertTrue(os.path.exists(report_path))
        
        # Load the merged graph and check properties
        with open(merged_path, 'rb') as f:
            merged_graph = pickle.load(f)
        
        # The merged graph should have 8 nodes from both graphs (nodes 0-7)
        # With the prefix "g0_" for positive nodes and "g1_" for negative nodes
        self.assertEqual(len(merged_graph.nodes()), 8)
        
        # Check that each group's nodes are present with the correct prefix
        for i in range(4):
            self.assertTrue(f"g0_{i}" in merged_graph.nodes())
        
        for i in range(4, 8):
            self.assertTrue(f"g1_{i}" in merged_graph.nodes())
        
        # Check the edges in the merged graph
        expected_edges = [
            ("g0_0", "g0_1"),
            ("g0_1", "g0_0"),
            ("g0_2", "g0_3"),
            ("g0_3", "g0_2"),
            ("g1_4", "g1_5"),
            ("g1_5", "g1_4"),
            ("g1_6", "g1_7"),
            ("g1_7", "g1_6")
        ]
        
        for edge in expected_edges:
            self.assertTrue(merged_graph.has_edge(*edge))

if __name__ == "__main__":
    unittest.main()