"""
Configuration dataclasses for feature graph generation.

Provides structured configuration for the analysis pipeline,
plotting parameters, and output settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple


@dataclass
class GraphConfig:
    """
    Main configuration for feature graph generation.
    
    Parameters
    ----------
    experiment : str
        Name of the experiment (e.g., 'ops0094_20251217').
    debug_cell_fraction : float, optional
        Fraction of cells to sample for debug mode (e.g., 0.1 for 10%).
    skip_object_features : bool
        Whether to skip loading object-level feature CSVs.
    just_cell_painting : bool
        If True, only include Cell Painting organelles (cp1_*, cp2_*).
        Excludes: nuclear_seg, cell_seg, phase2d_*, focus3d_*, nucleoli_phase2d, nucleoli_focus3d.
    use_cuml : bool
        Whether to use cuML GPU acceleration where available.
    cluster_algo : str
        Clustering algorithm(s): "all", "hdbscan", "kmeans", or "leiden".
    skip_interactive : bool
        Whether to skip generating interactive dashboard data.
    drift_method : str
        Method for drift analysis: "three_segment_regression", 
        "segmented_regression", or "second_derivative".
    analysis_mode : str
        Which analyses to run: "all", "guide", "gene", or "guide_and_gene".
    random_state : int
        Random seed for reproducibility.
    use_cache : bool
        Whether to use cached embeddings/clustering results. Defaults to True.
    """
    experiment: str
    debug_cell_fraction: Optional[float] = None
    skip_object_features: bool = False
    just_cell_painting: bool = False
    use_cuml: bool = True
    cluster_algo: str = "all"
    skip_interactive: bool = True
    drift_method: str = "three_segment_regression"
    analysis_mode: str = "all"
    random_state: int = 42
    use_cache: bool = True
    skip_complete: bool = False  # Skip generating outputs that already exist

    def get_clustering_methods(self) -> List[str]:
        """Get list of clustering methods to run based on cluster_algo setting."""
        if self.cluster_algo == "all":
            return ["hdbscan", "kmeans", "leiden"]
        return [self.cluster_algo]


@dataclass
class PlotConfig:
    """
    Configuration for plot styling and output.
    
    Parameters
    ----------
    dpi : int
        Resolution for saved figures.
    figsize_umap : tuple
        Default figure size for UMAP plots.
    figsize_heatmap : tuple
        Default figure size for heatmaps.
    figsize_bar : tuple
        Default figure size for bar charts.
    point_size : int
        Default scatter plot point size.
    alpha : float
        Default transparency for scatter plots.
    cmap_continuous : str
        Default colormap for continuous data.
    cmap_diverging : str
        Default colormap for diverging data (e.g., fold change).
    cmap_categorical : str
        Default colormap for categorical data.
    """
    dpi: int = 150
    figsize_umap: Tuple[int, int] = (14, 12)
    figsize_heatmap: Tuple[int, int] = (12, 10)
    figsize_bar: Tuple[int, int] = (12, 8)
    figsize_volcano: Tuple[int, int] = (12, 9)
    point_size: int = 15  # Increased from 5 for better visibility
    alpha: float = 0.5
    cmap_continuous: str = "viridis"
    cmap_diverging: str = "RdBu_r"
    cmap_categorical: str = "turbo"
    
    # Clustering visualization
    cluster_noise_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    highlight_color: str = "red"
    background_color: str = "lightgray"


@dataclass
class AnalysisConfig:
    """
    Configuration specific to analysis algorithms.
    
    Parameters
    ----------
    min_cluster_size : int
        Minimum cluster size for HDBSCAN.
    min_samples : int
        Minimum samples for HDBSCAN core points.
    kmeans_k_range : tuple
        Range of k values to search for KMeans.
    leiden_resolution : float
        Resolution parameter for Leiden clustering.
    umap_n_neighbors : int
        Number of neighbors for UMAP.
    umap_min_dist : float
        Minimum distance for UMAP.
    low_variance_threshold : float
        Threshold for removing low-variance features.
    min_enrichment_cluster_size : int
        Minimum cluster size for enrichment analysis.
    """
    # Clustering parameters
    min_cluster_size: int = 25
    min_samples: int = 25
    kmeans_k_range: Tuple[int, int] = (2, 31)
    leiden_resolution: float = 1.0
    
    # UMAP parameters
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 2
    
    # Feature filtering
    low_variance_threshold: float = 1e-4
    
    # Enrichment analysis
    min_enrichment_cluster_size: int = 10
    min_gene_count_for_enrichment: int = 2
    enrichment_p_threshold: float = 0.05
    
    # Volcano plot thresholds
    volcano_p_threshold: float = 0.05
    volcano_log2fc_threshold: float = 1.0
    
    # Differential analysis thresholds
    zscore_threshold: float = 2.0
    pvalue_threshold: float = 0.05
    
    # Guide/gene QC thresholds
    min_cells_per_guide: int = 10
    min_guides_per_gene: int = 2


@dataclass 
class OutputConfig:
    """
    Configuration for output paths and file naming.
    
    Parameters
    ----------
    base_output_dir : Path, optional
        Base directory for outputs. If None, uses default from OpsDataset.
    create_subdirs : bool
        Whether to create subdirectories for each analysis type.
    save_embeddings_to_anndata : bool
        Whether to save embeddings back to AnnData files.
    """
    base_output_dir: Optional[Path] = None
    create_subdirs: bool = True
    save_embeddings_to_anndata: bool = True
    save_intermediate_csvs: bool = True


# Common exclude patterns for identifying feature columns vs metadata
METADATA_EXCLUDE_PATTERNS = [
    "_id",  # More specific: _id suffix or prefix (not just "id" which matches "phalloidin")
    "cell_id",
    "gene_name",
    "gene_symbol",
    "barcode",
    "sgRNA",
    "guide_",
    "perturbation",
    "_effect",
    "NCBI",
    "_index",
    "well_",
    "_well",
    "pos_",
    "y_global_pheno",  # Specific pheno metadata columns
    "x_global_pheno",
    "y_local_pheno",
    "x_local_pheno",
    "tile_pheno",
    "area_pheno",
    "cp1_pheno_distance",
    "y_global_bc",
    "x_global_bc",
    "Unnamed",
    "bbox",
    "umap_",  # UMAP coordinates
    "cluster_",  # Cluster assignments
    "radial_",  # Radial positions
    "label",  # Segmentation labels
]

# Patterns for identifying NTC (non-targeting control) genes
NTC_PATTERNS = ["ntc", "non-targeting", "^0$"]
