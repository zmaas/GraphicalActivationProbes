from typing import Dict, Any, List
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch

def load_sentiment_dataset(dataset_name: str = "imdb", split: str = "train") -> pd.DataFrame:
    """Load a standard sentiment analysis dataset.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split to load
        
    Returns:
        DataFrame with text and labels
    """
    dataset = load_dataset(dataset_name, split=split)
    df = pd.DataFrame(dataset)
    return df

class SentimentDataset(Dataset):
    """PyTorch dataset for sentiment analysis tasks."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: Any):
        """Initialize dataset with texts and labels.
        
        Args:
            texts: List of text samples
            labels: List of labels (0 or 1 for binary classification)
            tokenizer: Tokenizer for the model
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(text, padding="max_length", 
                               truncation=True, max_length=512)
        
        # Convert to tensors without the batch dimension
        item = {k: torch.tensor(v) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label)
        
        return item

def get_dataloader(dataset: Dataset, batch_size: int = 8, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader for the dataset.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the dataset
        
    Returns:
        PyTorch DataLoader
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
