"""
UMAP visualization functions.

Provides specialized UMAP plotting functions for:
- Cluster visualizations
- Gene highlighting
- Continuous value coloring
- NTC vs perturbed comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

from .fe_graphs_utils import save_figure, create_cluster_palette, add_gene_labels

logger = logging.getLogger(__name__)


def plot_umap_cluster(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    x_col: str = "umap_1",
    y_col: str = "umap_2",
    annotations: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (16, 12),
    dpi: int = 150,
    palette: Optional[Dict] = None,
) -> plt.Figure:
    """
    Create a UMAP plot colored by cluster with optional annotations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with UMAP coordinates and cluster labels.
    cluster_col : str
        Column name for cluster labels.
    x_col, y_col : str
        Column names for UMAP coordinates.
    annotations : dict, optional
        Dictionary mapping cluster ID to annotation text.
    title : str, optional
        Plot title.
    save_path : Path, optional
        Path to save figure.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution.
    palette : dict, optional
        Color palette. If None, auto-generated.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create palette if not provided
    if palette is None:
        palette = create_cluster_palette(df[cluster_col])
    
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=cluster_col,
        s=5,
        alpha=0.5,
        palette=palette,
        legend=False,
        ax=ax,
        rasterized=True,
    )
    
    # Add annotations
    if annotations:
        _add_cluster_annotations(ax, df, cluster_col, x_col, y_col, annotations)
    
    ax.set_title(title or f"UMAP by {cluster_col}", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    
    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)
    
    return fig


def _add_cluster_annotations(
    ax: plt.Axes,
    df: pd.DataFrame,
    cluster_col: str,
    x_col: str,
    y_col: str,
    annotations: Dict[str, str],
) -> None:
    """Add text annotations at cluster centers."""
    cluster_centers = df.groupby(cluster_col)[[x_col, y_col]].median()
    plot_center_x = df[x_col].mean()
    plot_center_y = df[y_col].mean()
    
    texts = []
    for cluster_id, center in cluster_centers.iterrows():
        if cluster_id in annotations and cluster_id != "c-1":
            # Determine alignment based on quadrant
            ha = "left" if center[x_col] > plot_center_x else "right"
            va = "bottom" if center[y_col] > plot_center_y else "top"
            
            text = ax.text(
                center[x_col], center[y_col],
                annotations[cluster_id],
                fontdict={"size": 9},
                color="black",
                ha=ha, va=va,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="black", alpha=0.5),
            )
            texts.append(text)
    
    # Try to adjust text positions
    if texts:
        try:
            from adjustText import adjust_text
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.7),
            )
        except ImportError:
            pass


def plot_umap_gene_highlight(
    df: pd.DataFrame,
    gene: str,
    gene_col: str = "gene_name",
    x_col: str = "umap_1",
    y_col: str = "umap_2",
    cluster_col: str = "cluster",
    enrichment_df: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 10),
    dpi: int = 150,
    highlight_color: str = "red",
    background_color: str = "lightgray",
) -> plt.Figure:
    """
    Create a UMAP plot highlighting cells for a specific gene.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with UMAP coordinates and gene names.
    gene : str
        Gene to highlight.
    gene_col : str
        Column name for gene names.
    x_col, y_col : str
        Column names for UMAP coordinates.
    cluster_col : str
        Column name for clusters (for enrichment annotations).
    enrichment_df : pd.DataFrame, optional
        Enrichment results for annotating significant clusters.
    title : str, optional
        Plot title.
    save_path : Path, optional
        Path to save figure.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution.
    highlight_color : str
        Color for highlighted gene.
    background_color : str
        Color for background cells.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot background
    ax.scatter(
        df[x_col], df[y_col],
        color=background_color,
        s=5, alpha=0.3,
        rasterized=True,
    )
    
    # Highlight gene
    highlight_mask = df[gene_col] == gene
    if highlight_mask.any():
        ax.scatter(
            df.loc[highlight_mask, x_col],
            df.loc[highlight_mask, y_col],
            color=highlight_color,
            s=15,
            label=f"{gene} ({highlight_mask.sum()} cells)",
        )
    
    # Add enrichment annotations
    if enrichment_df is not None and not enrichment_df.empty:
        _add_enrichment_annotations(
            ax, df, gene, enrichment_df, cluster_col, x_col, y_col
        )
    
    ax.set_title(title or f"UMAP Highlighting Gene: {gene}", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()
    ax.set_aspect("equal")
    
    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)
    
    return fig


def _add_enrichment_annotations(
    ax: plt.Axes,
    df: pd.DataFrame,
    gene: str,
    enrichment_df: pd.DataFrame,
    cluster_col: str,
    x_col: str,
    y_col: str,
) -> None:
    """Add enrichment annotations for a specific gene."""
    gene_enrichment = enrichment_df[enrichment_df["gene_name"] == gene]
    significant = gene_enrichment[gene_enrichment["p_adj"] < 0.05]
    
    if significant.empty:
        return
    
    cluster_centers = df.groupby(cluster_col)[[x_col, y_col]].median()
    cluster_counts = df[cluster_col].value_counts()
    plot_center_x = df[x_col].mean()
    plot_center_y = df[y_col].mean()
    
    texts = []
    for _, row in significant.iterrows():
        cluster_id = row["cluster"]
        odds_ratio = row["odds_ratio"]
        
        if cluster_id in cluster_centers.index:
            center = cluster_centers.loc[cluster_id]
            cell_count = cluster_counts.get(cluster_id, 0)
            
            ha = "left" if center[x_col] > plot_center_x else "right"
            va = "bottom" if center[y_col] > plot_center_y else "top"
            
            text = ax.text(
                center[x_col], center[y_col],
                f"{odds_ratio:.1f}x\nn={cell_count}",
                fontdict={"size": 10, "weight": "bold"},
                color="black",
                ha=ha, va=va,
                bbox=dict(boxstyle="round,pad=0.2", fc="gray", ec="black", alpha=0.2),
            )
            texts.append(text)
    
    if texts:
        try:
            from adjustText import adjust_text
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
            )
        except ImportError:
            pass


def plot_umap_continuous(
    df: pd.DataFrame,
    color_col: str,
    x_col: str = "umap_1",
    y_col: str = "umap_2",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    cbar_label: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 10),
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a UMAP plot colored by continuous variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with UMAP coordinates.
    color_col : str
        Column name for color values.
    x_col, y_col : str
        Column names for UMAP coordinates.
    cmap : str
        Colormap name.
    vmin, vmax : float, optional
        Color scale limits.
    title : str, optional
        Plot title.
    cbar_label : str, optional
        Colorbar label.
    save_path : Path, optional
        Path to save figure.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    from matplotlib.colors import Normalize
    
    fig, ax = plt.subplots(figsize=figsize)
    
    values = pd.to_numeric(df[color_col], errors="coerce")
    
    if vmax is None:
        vmax = values.quantile(0.99)
    if vmin is None:
        vmin = values.min()
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    scatter = ax.scatter(
        df[x_col], df[y_col],
        c=values,
        cmap=cmap,
        norm=norm,
        s=5, alpha=0.5,
        rasterized=True,
    )
    
    cbar = fig.colorbar(scatter, ax=ax, orientation="vertical")
    if cbar_label:
        cbar.set_label(cbar_label)
    
    ax.set_title(title or f"UMAP by {color_col}", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    
    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)
    
    return fig


def plot_umap_ntc_vs_perturbed(
    df: pd.DataFrame,
    ntc_mask: pd.Series,
    x_col: str = "umap_1",
    y_col: str = "umap_2",
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (16, 12),
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a UMAP plot comparing NTC and perturbed cells.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with UMAP coordinates.
    ntc_mask : pd.Series
        Boolean mask where True indicates NTC.
    x_col, y_col : str
        Column names for UMAP coordinates.
    title : str, optional
        Plot title.
    save_path : Path, optional
        Path to save figure.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ntc_data = df[ntc_mask]
    pert_data = df[~ntc_mask]
    
    # Plot perturbed as background
    ax.scatter(
        pert_data[x_col], pert_data[y_col],
        color="lightgray",
        s=5, alpha=0.5,
        label=f"Perturbed ({len(pert_data)} cells)",
        rasterized=True,
    )
    
    # Plot NTC on top
    ax.scatter(
        ntc_data[x_col], ntc_data[y_col],
        color="red",
        s=8, alpha=0.7,
        label=f"NTC ({len(ntc_data)} cells)",
    )
    
    ax.set_title(title or "UMAP: NTC vs Perturbed", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()
    
    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)
    
    return fig
