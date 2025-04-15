# Graphical Lasso Neural Network Interpretability

This project uses Graphical Lasso (GLasso) to interpret neural networks by analyzing the dependency structure between neurons.

## Overview

The main idea is to use Graphical Lasso to infer a sparse precision matrix from activation patterns, revealing the conditional independence structure between neurons. This structure can be visualized as a graph, offering insights into how information flows through the neural network.

## Features

- Collect activations from transformer models (specifically MLP layers)
- Apply Graphical Lasso to determine conditional independence structure
- Visualize neuron connections using heatmaps and graph visualizations
- Synthetic data generation for specific topics
- Compare neuron activation patterns across different topics

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/glasso_interp.git
   cd glasso_interp
   ```

2. Create and activate a virtual environment:
   ```
   uv venv
   source .venv/bin/activate
   ```

3. Install the required packages:
   ```
   uv pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```
   cp .env.example .env
   ```
   Then edit the `.env` file to add your OpenAI API key.

## Usage

### Basic Usage

Run the analysis on the IMDB dataset:

```bash
python src/main.py --dataset imdb --model google/gemma-3-1b-it --output_dir results/imdb
```

### Using Synthetic Data

Generate synthetic data for a specific topic:

```bash
python examples/synthetic_topic.py --synthetic_topic "quantum computing" --num_examples 60
```

### Compare Topics

Compare neuron activation patterns across different topics:

```bash
python examples/compare_topics.py --compare_topics "climate change" "artificial intelligence" --num_examples 60
```

## Example Output

The analysis outputs:
- Precision matrix heatmaps
- GraphViz DOT files representing the neuron connection structure
- Rendered graph visualizations (if GraphViz is installed)
- For topic comparisons, side-by-side visualizations and difference maps

## Implementation Details

- Uses a modified transformer model with hooks to capture activations
- Applies Graphical Lasso with a tunable regularization parameter (alpha)
- Implements both single-topic analysis and multi-topic comparison workflows
- Caches synthetic data to avoid regenerating examples during development
- Optimized to handle large activation matrices efficiently

## Roadmap

- [x] Initial implementation with IMDB dataset
- [x] Add synthetic data generation for specific topics
- [x] Implement topic comparison visualization
- [ ] Extend to other model architectures beyond transformers
- [ ] Add interactive visualization options
- [ ] Develop neuron/pathway importance scoring
- [ ] Add causal intervention experiments

## License

MIT License