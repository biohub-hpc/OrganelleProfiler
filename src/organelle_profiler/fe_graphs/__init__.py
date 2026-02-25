"""
Feature Graph Generation Package (v3.0).

This package provides a modular, extensible framework for generating
UMAP embeddings, clustering, and statistical visualizations from
feature extraction AnnData files.

Architecture:
-------------
- **Level-based**: Analysis flows cell → guide → gene with result chaining
- **Stage-based**: Each level runs QC → Embedding → Clustering → Differential → Summary
- **Modular**: Core utilities (embedding, clustering, cache) are reusable

Output Organization:
--------------------
    graphs/
    ├── 1_cell_level/
    │   ├── 1_qc/
    │   ├── 2_embedding/
    │   ├── 3_clustering/
    │   ├── 4_differential/
    │   └── 5_summary/
    ├── 2_guide_level/
    │   └── ...
    ├── 3_gene_level/
    │   └── ...
    └── 4_integrated/

Usage:
------
    # Full pipeline (cell → guide → gene)
    from organelle_profiler.feature_extraction.fe_graphs import AnalysisOrchestrator
    orchestrator = AnalysisOrchestrator(experiment="ops0094_20251217")
    result = orchestrator.run()
    
    # Specific levels only
    result = orchestrator.run(levels=["gene"])
    
    # Skip stages
    result = orchestrator.run(skip_stages=["qc"])
    
    # Individual level pipeline
    from organelle_profiler.feature_extraction.fe_graphs import LevelPipeline
    pipeline = LevelPipeline(data_context, config, level="cell")
    level_result = pipeline.run()
"""

from .fe_graphs_config import GraphConfig, PlotConfig, AnalysisConfig
from .fe_graphs_orchestrator import AnalysisOrchestrator, OrchestratorResult
from .fe_graphs_level_pipeline import LevelPipeline, LevelResult

# Legacy alias for backwards compatibility
GraphGenerator = AnalysisOrchestrator

__all__ = [
    # Configuration
    "GraphConfig",
    "PlotConfig",
    "AnalysisConfig",
    # Main orchestrator
    "AnalysisOrchestrator",
    "OrchestratorResult",
    # Level pipeline
    "LevelPipeline",
    "LevelResult",
    # Legacy
    "GraphGenerator",
]
