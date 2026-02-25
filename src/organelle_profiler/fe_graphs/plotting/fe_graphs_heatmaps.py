"""
Heatmap visualization functions.

Provides functions for:
- Standard heatmaps
- Clustered heatmaps
- Feature drift heatmaps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

from .fe_graphs_utils import save_figure

logger = logging.getLogger(__name__)


def plot_heatmap(
    data: pd.DataFrame,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cbar_label: Optional[str] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 10),
    dpi: int = 150,
    annot: bool = False,
    fmt: str = ".2f",
) -> plt.Figure:
    """
    Create a simple heatmap.
    
    Parameters
    ----------
    data : pd.DataFrame
        2D data to plot as heatmap.
    title : str, optional
        Plot title.
    xlabel, ylabel : str, optional
        Axis labels.
    cbar_label : str, optional
        Colorbar label.
    cmap : str
        Colormap name.
    vmin, vmax : float, optional
        Color scale limits.
    save_path : Path, optional
        Path to save figure.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution.
    annot : bool
        Whether to annotate cells with values.
    fmt : str
        Format string for annotations.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=fmt,
        ax=ax,
        cbar_kws={"label": cbar_label} if cbar_label else {},
    )
    
    if title:
        ax.set_title(title, fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)
    
    return fig


def plot_clustermap(
    data: pd.DataFrame,
    title: Optional[str] = None,
    cbar_label: Optional[str] = None,
    cmap: str = "vlag",
    metric: str = "correlation",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    save_path: Optional[Path] = None,
    figsize: Optional[tuple] = None,
    dpi: int = 150,
    z_score: Optional[int] = None,
    row_cluster: bool = True,
    col_cluster: bool = True,
) -> sns.matrix.ClusterGrid:
    """
    Create a clustered heatmap.
    
    Parameters
    ----------
    data : pd.DataFrame
        2D data to cluster and plot.
    title : str, optional
        Plot title.
    cbar_label : str, optional
        Colorbar label.
    cmap : str
        Colormap name.
    metric : str
        Distance metric for clustering.
    vmin, vmax : float, optional
        Color scale limits.
    save_path : Path, optional
        Path to save figure.
    figsize : tuple, optional
        Figure size. Auto-calculated if None.
    dpi : int
        Resolution.
    z_score : int, optional
        0 for row normalization, 1 for column normalization.
    row_cluster, col_cluster : bool
        Whether to cluster rows/columns.
        
    Returns
    -------
    sns.matrix.ClusterGrid
        The clustermap object.
    """
    # Auto-calculate figure size
    if figsize is None:
        n_rows, n_cols = data.shape
        figsize = (max(10, n_cols * 0.3), max(10, n_rows * 0.3))
    
    # Standardize if requested
    plot_data = data.copy()
    if z_score is not None:
        plot_data = (plot_data - plot_data.mean(axis=z_score)) / plot_data.std(axis=z_score)
        plot_data = plot_data.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Remove zero-variance rows/columns for clustering
    if row_cluster:
        row_stds = plot_data.std(axis=1)
        plot_data = plot_data[row_stds > 0]
    if col_cluster:
        col_stds = plot_data.std(axis=0)
        plot_data = plot_data.loc[:, col_stds > 0]
    
    if len(plot_data) < 2 or len(plot_data.columns) < 2:
        logger.warning("Not enough data for clustermap after filtering")
        return None
    
    g = sns.clustermap(
        plot_data,
        figsize=figsize,
        cmap=cmap,
        metric=metric,
        vmin=vmin,
        vmax=vmax,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        cbar_kws={"label": cbar_label} if cbar_label else {},
    )
    
    if title:
        g.fig.suptitle(title, y=1.02)
    
    if save_path:
        g.fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    
    return g


def plot_feature_drift_heatmap(
    drift_scores: Dict[str, list],
    well_indices: Dict[str, list],
    wells: list,
    grid_size: int = 30,
    normalization_factor: float = 1.0,
    title: Optional[str] = None,
    cbar_label: Optional[str] = None,
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a feature drift heatmap across wells.
    
    Parameters
    ----------
    drift_scores : dict
        Dictionary mapping well to list of drift scores.
    well_indices : dict
        Dictionary mapping well to list of (i, j) tile indices.
    wells : list
        List of wells to plot.
    grid_size : int
        Size of the grid per well.
    normalization_factor : float
        Factor to normalize scores by.
    title : str, optional
        Plot title.
    cbar_label : str, optional
        Colorbar label.
    save_path : Path, optional
        Path to save figure.
    dpi : int
        Resolution.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    num_wells = len(wells)
    fig, axes = plt.subplots(1, num_wells, figsize=(5 * num_wells, 5), squeeze=False)
    ax_flat = axes.flatten()
    
    # Get color limits from all data
    all_scores = [
        s / normalization_factor
        for well_scores in drift_scores.values()
        for s in well_scores
        if not np.isnan(s)
    ]
    
    if not all_scores:
        logger.warning("No data to plot for feature drift heatmap")
        plt.close(fig)
        return None
    
    vmin = np.percentile(all_scores, 5)
    vmax = np.percentile(all_scores, 95)
    
    im = None
    for i, well in enumerate(wells):
        scores = drift_scores.get(well, [])
        indx = well_indices.get(well, [])
        
        if not scores or not indx:
            continue
        
        indx_i = [a[0] for a in indx]
        indx_j = [a[1] for a in indx]
        normalized_scores = [s / normalization_factor for s in scores]
        
        out = np.full((grid_size, grid_size), np.nan)
        out[indx_i, indx_j] = normalized_scores
        
        ax = ax_flat[i]
        im = ax.imshow(out, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(well)
    
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.045, pad=0.04)
        cbar.set_label(cbar_label or "Feature Drift Score")
    
    fig.suptitle(title or "Feature Drift Heatmap", fontsize=16)
    
    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)
    
    return fig
