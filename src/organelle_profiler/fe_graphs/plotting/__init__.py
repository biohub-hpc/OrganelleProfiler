"""
Plotting utilities for feature graph generation.

Provides reusable plotting functions to eliminate duplication:
- UMAP scatter plots
- Heatmaps and clustermaps
- Statistical plots (volcano, bar charts)
- Drift visualizations
"""

from .fe_graphs_utils import (
    save_figure,
    create_scatter_plot,
    add_gene_labels,
    create_cluster_palette,
    plot_umap_with_colorbar,
    plot_umap_categorical,
    create_broken_axis_bar,
)
from .fe_graphs_umap_plots import (
    plot_umap_cluster,
    plot_umap_gene_highlight,
    plot_umap_continuous,
    plot_umap_ntc_vs_perturbed,
)
from .fe_graphs_heatmaps import (
    plot_heatmap,
    plot_clustermap,
    plot_feature_drift_heatmap,
)
from .fe_graphs_statistical_plots import (
    plot_volcano,
    plot_bar_chart,
    plot_enrichment_bar,
)

__all__ = [
    # Utils
    "save_figure",
    "create_scatter_plot",
    "add_gene_labels",
    "create_cluster_palette",
    "plot_umap_with_colorbar",
    "plot_umap_categorical",
    "create_broken_axis_bar",
    # UMAP plots
    "plot_umap_cluster",
    "plot_umap_gene_highlight",
    "plot_umap_continuous",
    "plot_umap_ntc_vs_perturbed",
    # Heatmaps
    "plot_heatmap",
    "plot_clustermap",
    "plot_feature_drift_heatmap",
    # Statistical plots
    "plot_volcano",
    "plot_bar_chart",
    "plot_enrichment_bar",
]
