# Graphical Lasso Neural Network Interpretability

This project uses Graphical Lasso to analyze the conditional independence structure between neurons in transformer models. By analyzing activations across many samples, we can determine which neurons are conditionally independent, providing insights into how the model processes information.

> Note: This is an ongoing independent ML interp research project, and a heavy draft.

## The Idea

My research background is in deconvolution problems and MCMC, and graphical LASSO gets used a lot in biological systems. 
If we assume that a key interpretability issue in LLMs is superposition (e.g. linear features that are mixed), then one alternative hypothesis is that linearity can be embedded in mutually dependent features, which we could dissect at the MLP layer level of an LLM. 
It's not an exact match, but toes up nicely with the idea of circuits pioneered by other interp work. 
Here I have an initial proof of concept of our ability to generate graphs showing co-activated features across a set of inputs.
The best proof of concept that I'm shooting for would be identifying graphs at the MLP level that we could fix activations to and steer model outputs.
So far, I've been using the IMDB dataset as input but I don't love it, it's pretty broad in terms of content vs specificity of something like sentiment.

Here's the things that need to get done to get us there:
- [x] Build initial proof of concept that we can generate graphs from MLPs by co-activation
- [ ] Synthetically generate topically-oriented data (e.g. refusals, locations, language) for testing using a big model
    - [ ] BONUS: Generate golden-gate bridge based content ala golden gate claude
- [ ] Implement steering based off of individual layer graphs, and develop the underlying math
    - [ ] Test the simplest thing - rescale each GLASSO module's MLP activations by its weight matrix
    - [ ] See if feature sensitivity varies over the depth of all layers?
- [ ] Develop a faster GLASSO implementation in Torch/JAX since current implementations are slow and don't scale well
    - [ ] Consider the dual-primal method proposed by Dallakyan (2403.12357)
- [ ] See if this generalizes into circuits over layers and study the underlying math a bit more

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
