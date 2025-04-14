import sys
sys.path.append('/Users/zach/Dropbox/codes/glasso_interp/src')
from data import load_sentiment_dataset, SentimentDataset, get_dataloader
from model import ModelWithActivations
import torch

# Load model with reduced size
print("Loading model...")
model = ModelWithActivations(model_name="google/gemma-3-1b-it")
tokenizer = model.tokenizer

# Load a small subset of the dataset
print("Loading dataset...")
df = load_sentiment_dataset(dataset_name="imdb")
df_small = df.head(16)  # Just use 16 samples

# Create dataset and dataloader
print("Creating dataset...")
dataset = SentimentDataset(
    texts=df_small["text"].tolist(), 
    labels=df_small["label"].tolist(),
    tokenizer=tokenizer
)
print(f"Dataset size: {len(dataset)}")

# Check a single item
print("Checking single item shape:")
item = dataset[0]
print({k: v.shape for k, v in item.items()})

# Create dataloader with a smaller batch size
print("Creating dataloader...")
batch_size = 2
dataloader = get_dataloader(dataset, batch_size=batch_size)

# Check a batch
print("Checking batch shape:")
batch = next(iter(dataloader))
print({k: v.shape for k, v in batch.items()})

# Collect activations with try/except to see errors
print("Collecting activations...")
try:
    activations_dict = model.collect_activations(dataloader)
    print("Success! Collected activations for these layers:")
    for layer_name, activations in activations_dict.items():
        print(f"  - {layer_name}: {activations.shape}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()