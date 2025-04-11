spec for this project
- grab some standard sentiment analysis dataset as the original inputs
- pull gemma3:1b as the model to test since it's small for my minimal hardware
- run each class of the dataset through the model, 
general goal
- Collect activations for each neuron across many samples (tokens/sequences)
- Build a matrix where each row is a sample and each column is a neuron
- Run GLasso on that matrix to get the conditional independence structure between neurons
- intuitively this seems like it might work best on the MLP layer? 
- would also probably want to handle attention heads combined at first then separately
implementation details
- python, let's use torch + transformers library
- want to use some fast implementation of GLASSO since our activations may be large
