# Running Research Log

---
Bottom line: I'm trying to build a systematic method for discovering non-obvious structural relationships in concept representations, ideally with evidence that SAEs reveal cleaner hierarchical structure than raw MLP activations.
---

## Project design

The purpose of this project is to explore methods for probing large language models (LLMs) and their internal state using graph-based methods. The idea here, after some initial iteration and refinement, is to use graph-based correlation methods to understand the structure of internal linear dependence within LLMs as well as SAEs.
This boils down to two distinct hypotheses:
1. *(Hierarchical Structure in SAEs)* Graphical probes of SAE activations can reveal hierarchical structure in concept representations, providing us with better understanding of how these linearly separable concept representations are organized and used.
  - If it works: We will have a new tool for understanding the hierarchical structure in SAE activations which should allow for new applications in both model steering (using the learned graphs to steer in some weighted manner) as well as in potentially understanding deceptive behavior and alignment.
  - If it doesn't work: This would raise interesting questions about SAEs and how they capture LRH predicted structure.
2. *(Supervised Probes of Linear Dependence)* Graphical probes of MLP activations can reveal linear dependence in the learned activation space of each separate layer, acting as a supplement to standard linear probes to understand the structure of linear vs dependent activation of MLP neurons.
  - If it works: We have a new tool for understanding the structure of MLP activations in LLMs, and could use the learned graphs to examine "concept subspaces" (the connected components of the linear dependence graph)
  - If it doesn't work: This would be surprising! MLPs are understood to coordinate their activations [Kornblith2019, maybe? SVCCA stuff is in the same ballpark] to represent things in lower dimensional linear manifolds (check math...), finding a lack of linear dependence in MLP activations while probing, while unlikely, would be highly unexpected within our current understanding.

## Recent Work

- The Linear Representation Hypothesis posits that LLMs encode semantic concepts as linearly separable vectors within some representation space [Park2023].
- Ways to probe the LRH include Linear Probes (testing in a supervised way) and SAEs (unsupervised).
- The GDM Mech Interp team has also deprioritized SAEs recently [Smith2025] with results suggesting that linear probes outperform SAEs.
- Work at Anthropic has also recently used models augmented with cross-layer transcoders to probe causal flow during LLM inference [Lindsey2025].

This work is inspired by all of the above, particularly [Lindsey2025], trying to take a different approach while still using graphical methods. Rather than the graphical approach used there to probe causal flow across graphs in local prompts, I am suggesting using graphical methods to instead probe how concepts are linked within the representation space of a model / SAE on average across a probe or datset.

## Why this might not work

- Computational Complexity
  - Graphical LASSO (the obvious method) is `O(p^3)` which won't scale to foundation model scale, and other methods like QUIC may not be fast enough. This is a very reasonable concern, but can either be tackled by (a) throwing more compute at the problem, (b) generating subgraphs for subsets of neurons separately on the same set and then combining them, or (c) using approximate methods for linear dependence instead, such as element-wise LASSO, partial correlation metrics, mutual information, or the Nystrom approximation (if assumptions about rank hold)
- SAEs have limited expressivity compared to linear probes and so are not a great basis to build a graphical method on top of.
  - SAEs still have descriptive power and present a great testing ground for developing graphical methods like the ones proposed here. However, because the graphical methods proposed here are based on analyzing activations, there's no reason we couldn't also apply them to a set of linear probes to examine how the representation space of those linear probes is graphically connected in terms of linear dependence.
- Generating graphs on top of SAEs or linear probes is risky from an interpretability perspective because it's another step away from the base model and doesn't factor in the error term of the SAE/probe.
  - This is certainly true, but can be mitigated by (a) using SAEs/probes that are high quality and have a small error term to begin with, and (b) factoring in reconstruction error bounds when generating precision matrices, to see if uncertainty affects the generated graphs.

## Experiments

1. Probing SAE activations on GPT-2 + subsets of The Pile

- Contained at [/notebooks/02_SAE_Probes.ipynb]
- This is a first proof of feasibility that has helped clarify constraints in the proposed methodology. Uses random samples rather than a conceptual probing approach.
- Due to VRAM limits (8Gb on what I have available), I currently only use a 512 SAE neuron subset for generating graphs, and calculate a partial correlation matrix that I verify as matching the full GLASSO graph on a subset of data.
- Within these limits, I'm getting interesting preliminary results on the structure of SAE dependence.
  - Positive example: `mentions of catastrophic events or overwhelming situations <-> phrases related to fox news channel`
  - Negative example: ``
- I'm working on restructing the example generation to gather the 30ish most activating examples from each SAE neuron to use for a complete SAE graph. Might use reservoir sampling, or a specialized heap based data structure... The partial correlation method is more reliable since the super sparse matrix generated by activations is sometimes poorly conditioned to the extend that SKLearn's GLASSO chokes.

2. Probing MLP activations on Gemma2:2b + subsets of The Pile

- Contained at [/notebooks/01_Linear_Probes.ipynb FIXME]
- Preliminary proof of concept on MLP activations. I need to extend this to use topical subsets rather than random samples, and solve the engineering challenge of storing enough activations efficiently.

## Harebrained ideas

- Using a heapq based data structure isn't working efficiently enough, the number of SAE neurons can be too large. To be dumber, I'm going to offload the storage to a SQLite DB on disk, and just grab everything since disk is cheap. Then, getting top-k will be dirt cheap with proper indexing.
