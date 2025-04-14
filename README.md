# Graphical Lasso Neural Network Interpretability

This project uses Graphical Lasso to analyze the conditional independence structure between neurons in transformer models. By analyzing activations across many samples, we can determine which neurons are conditionally independent, providing insights into how the model processes information.

> Note: This is an ongoing independent ML interp research project, and a heavy draft

## Features

- Collection of activations from transformer models (using Gemma3-1.1b)
- Processing activations to extract neuron interactions
- Applying Graphical Lasso to determine conditional independence structure
- Visualization of neuron connections:
  - Heatmap visualizations of precision matrices
  - GraphViz DOT files for graph visualization
  - Rendered PNG graph visualizations (requires GraphViz installation)

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- PyGlasso
- GraphViz Python package (installed with requirements.txt)
- GraphViz command-line tools (optional, for PNG rendering)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/glasso_interp.git
cd glasso_interp

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Usage

```bash
# Run with default parameters
python -m src.main

# Run with custom parameters
python -m src.main --dataset imdb --model google/gemma3-1.1b --batch_size 8 --alpha 0.1 --output_dir results

# Skip graph visualizations (helpful in environments without GraphViz CLI tools)
python -m src.main --skip_viz
```

## How It Works

1. We load a sentiment analysis dataset and a pre-trained transformer model
2. For each class of data, we run samples through the model and collect activations
3. We create a matrix where each row is a sample and each column is a neuron
4. We apply Graphical Lasso to this matrix to get the conditional independence structure
5. We analyze and visualize the connections between neurons:
   - Generate heatmap visualizations of precision matrices
   - Create GraphViz DOT files to represent neuron connection graphs
   - Optionally render PNG visualizations of the connection graphs

## License

MIT
