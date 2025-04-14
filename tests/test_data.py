import pytest
import pandas as pd
from src.data import load_sentiment_dataset, SentimentDataset, get_dataloader

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    class MockTokenizer:
        def __call__(self, text, padding=None, truncation=None, max_length=None):
            import torch
            return {"input_ids": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    
    return MockTokenizer()

def test_load_sentiment_dataset(monkeypatch):
    """Test loading a sentiment dataset."""
    # Mock the load_dataset function
    def mock_load_dataset(dataset_name, split):
        return {"text": ["This is positive", "This is negative"], "label": [1, 0]}
    
    # Apply the monkeypatch
    monkeypatch.setattr("src.data.load_dataset", mock_load_dataset)
    
    # Test the function
    df = load_sentiment_dataset("mock_dataset")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "text" in df.columns
    assert "label" in df.columns

def test_sentiment_dataset(mock_tokenizer):
    """Test the SentimentDataset class."""
    texts = ["This is positive", "This is negative"]
    labels = [1, 0]
    
    dataset = SentimentDataset(texts, labels, mock_tokenizer)
    
    # Test length
    assert len(dataset) == 2
    
    # Test getting an item
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    
    # Check label
    assert item["labels"].item() == 1

def test_get_dataloader(mock_tokenizer):
    """Test creating a dataloader."""
    texts = ["This is positive", "This is negative"]
    labels = [1, 0]
    
    dataset = SentimentDataset(texts, labels, mock_tokenizer)
    dataloader = get_dataloader(dataset, batch_size=2)
    
    # Check dataloader properties
    assert dataloader.batch_size == 2
    
    # Get a batch
    batch = next(iter(dataloader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
