"""
Stage-based analysis modules.

Each stage runs a specific phase of analysis across any level (cell/guide/gene):
- QCStage: Basic quality control (batch effects, feature quality)
- SpatialDriftStage: Comprehensive radial drift analysis (cell level)
- EmbeddingStage: Dimensionality reduction (PCA, UMAP)
- EmbeddingVisualizationStage: Comprehensive annotated UMAP/PCA plots
- ClusteringStage: Unsupervised clustering and enrichment
- DifferentialStage: NTC comparison and statistical testing
- OrganelleDiscriminationStage: Compare segmentation groups' discrimination power
- PositiveControlsStage: Validate using known gene clusters
- SummaryStage: Aggregation and reporting

This allows the same analytical suite to run consistently across levels.
"""

from .fe_graphs_stage_base import BaseStage, StageResult
from .fe_graphs_qc_stage import QCStage
from .fe_graphs_spatial_drift_stage import SpatialDriftStage
from .fe_graphs_embedding_stage import EmbeddingStage
from .fe_graphs_embedding_visualization_stage import EmbeddingVisualizationStage
from .fe_graphs_clustering_stage import ClusteringStage
from .fe_graphs_differential_stage import DifferentialStage
from .fe_graphs_summary_stage import SummaryStage
from .fe_graphs_organelle_discrimination_stage import OrganelleDiscriminationStage
from .fe_graphs_positive_controls_stage import PositiveControlsStage
from .fe_graphs_cp_comparison_stage import CPComparisonStage
from .fe_graphs_gene_relationships_stage import GeneRelationshipsStage

__all__ = [
    "BaseStage",
    "StageResult",
    "QCStage",
    "SpatialDriftStage",
    "EmbeddingStage",
    "EmbeddingVisualizationStage",
    "ClusteringStage",
    "DifferentialStage",
    "SummaryStage",
    "OrganelleDiscriminationStage",
    "PositiveControlsStage",
    "CPComparisonStage",
    "GeneRelationshipsStage",
]
