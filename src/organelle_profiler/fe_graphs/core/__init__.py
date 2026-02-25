"""
Core utilities for feature graph generation.

Provides:
- EmbeddingEngine: UMAP/PCA computation with GPU fallback
- ClusteringEngine: Multiple clustering algorithms
- EmbeddingCache: Caching for embeddings and cluster results
- DataLoader: Data loading and enrichment utilities
- discover_organelle_groups: Single source of truth for organelle grouping
"""

from .fe_graphs_embedding import EmbeddingEngine
from .fe_graphs_clustering import ClusteringEngine
from .fe_graphs_cache import EmbeddingCache
from .fe_graphs_data_loader import (
    DataLoader, 
    DataContext, 
    discover_organelle_groups,
    discover_organelle_groups_from_adata,
)

__all__ = [
    "EmbeddingEngine",
    "ClusteringEngine",
    "EmbeddingCache",
    "DataLoader",
    "DataContext",
    "discover_organelle_groups",
    "discover_organelle_groups_from_adata",
]
