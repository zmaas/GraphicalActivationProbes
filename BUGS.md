Current bugs to fix:
- ✅ FIXED: generating synthetic examples creates a nested directory, /results/topic/topic/...
  - Fixed by modifying process_layer to use the provided output directory directly and ensuring analyze_topic creates a flat directory structure.
  - Added a test to verify the fix in test_directory_fix.py.

- ✅ FIXED: generating pngs isn't useful for this... i'd rather generate networkx outputs that we can visualize
  - Added NetworkX graph generation and saving to pickle files (.nx extension)
  - Created functions to:
    - Convert connections to NetworkX graphs
    - Save and load NetworkX graphs
    - Merge NetworkX graphs across topics
    - Compare and analyze graphs with metrics and cross-topic connections
  - Still generate GraphViz DOT files for backward compatibility
    
- ✅ FIXED: the pos and neg example generation for a topic doesn't make sense as written, this isn't just generating graphs for each and comparing
  - Implemented a proper sentiment analysis approach that:
    - Separates positive and negative examples and analyzes them separately
    - Generates graphs for both positive and negative sentiment
    - Creates a merged graph that shows connections specific to each sentiment
    - Generates detailed comparison reports highlighting the differences
    - Produces visualizations of the differences between positive and negative sentiment
