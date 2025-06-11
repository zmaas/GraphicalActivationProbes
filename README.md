# Graphical Activation Probes of Transformer Models

This repository contains research on developing graphical methods to interpret LLMs to understand the dependency structure between neurons.

The goal of this work is to develop a probe for LLMs that lies between linear probes and SAEs in function, building on the Linear Representation Hypothesis.

Here's a cool early result - when testing an initial design of this probe on GPT-2 + a 10k subset of The Pile, this probing approach yields semantically meaningful graphs connecting SAE neurons. For example, we discover a link in coactivation between the following SAE neurons: `mentions of catastrophic events or overwhelming situations <-> phrases related to fox news channel`

For an overview of experiments and results, please see [/log/README.md](/log/README.md).
The current state of the repository is as follows:

- /log/ - Contains ongoing research logs and experiments.
- /notebooks/ - Contains Jupyter notebooks on active research, ongoing
- /src/ - Contains early source code for GLasso-based activation probes, not actively used.
- /tests/ - Documentation and examples of how to use the GLasso-based activation probes.
- /test_results/ - Preliminary results on activation probing

## License

MIT License
