- Need to select smaller subsets of the activations, obviously, otherwise we'll rapidly OOM...

# Bug Fix Plan

1. **Nested Directory Issue:**
   - Modify the `process_layer` function in `src/main.py` to avoid creating nested directories
   - Fix how paths are constructed between `analyze_topic` and `process_layer`

2. **Switch from PNG to NetworkX Graphs:**
   - Add networkx to requirements.txt
   - Create a new function in `src/glasso.py` to generate networkx graphs from connections
   - Modify the `create_graphviz_dot` function to also return a networkx graph
   - Add functions to save/load networkx graphs

3. **Fix Pos/Neg Example Generation:**
   - Revise the comparison logic in `main.py` to properly handle positive and negative examples
   - Update the visualization to show relationships between concepts
