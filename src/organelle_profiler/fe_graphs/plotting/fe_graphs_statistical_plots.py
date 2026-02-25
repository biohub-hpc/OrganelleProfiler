"""
Statistical visualization functions.

Provides functions for:
- Volcano plots
- Bar charts
- Enrichment visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List
import logging

from .fe_graphs_utils import save_figure, add_gene_labels

logger = logging.getLogger(__name__)


def plot_volcano(
    df: pd.DataFrame,
    x_col: str = "log2_fold_change",
    y_col: str = "-log10p",
    gene_col: str = "gene_name",
    p_threshold: float = 0.05,
    fc_threshold: float = 1.0,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    n_labels: int = 10,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 9),
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a volcano plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with fold change and p-values.
    x_col : str
        Column for x-axis (log2 fold change).
    y_col : str
        Column for y-axis (-log10 p-value).
    gene_col : str
        Column for gene labels.
    p_threshold : float
        P-value threshold for significance.
    fc_threshold : float
        Fold change threshold (absolute log2).
    title : str, optional
        Plot title.
    xlabel, ylabel : str, optional
        Axis labels.
    n_labels : int
        Number of top genes to label.
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
    
    # Define significance conditions
    cond_down = (df["p_value"] < p_threshold) & (df[x_col] < -fc_threshold)
    cond_up = (df["p_value"] < p_threshold) & (df[x_col] > fc_threshold)
    
    # Plot points
    ax.scatter(
        df[x_col], df[y_col],
        c="grey", alpha=0.6, label="Not Significant",
    )
    ax.scatter(
        df.loc[cond_down, x_col], df.loc[cond_down, y_col],
        c="cornflowerblue", alpha=0.8, label="Downregulated",
    )
    ax.scatter(
        df.loc[cond_up, x_col], df.loc[cond_up, y_col],
        c="red", alpha=0.8, label="Upregulated",
    )
    
    # Label top genes
    genes_to_label = pd.concat([
        df[cond_down].nsmallest(n_labels, "p_value"),
        df[cond_up].nsmallest(n_labels, "p_value"),
    ])
    
    for _, row in genes_to_label.iterrows():
        ax.text(row[x_col], row[y_col], str(row[gene_col]), fontsize=9)
    
    # Add threshold lines
    ax.axhline(-np.log10(p_threshold), color="black", linestyle="--", lw=1)
    ax.axvline(fc_threshold, color="black", linestyle="--", lw=1)
    ax.axvline(-fc_threshold, color="black", linestyle="--", lw=1)
    
    ax.set_title(title or "Volcano Plot", fontsize=16)
    ax.set_xlabel(xlabel or "log2(Fold Change)")
    ax.set_ylabel(ylabel or "-log10(p-value)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)
    
    return fig


def plot_bar_chart(
    data: pd.Series,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    horizontal: bool = True,
    palette: str = "viridis",
    save_path: Optional[Path] = None,
    figsize: Optional[tuple] = None,
    dpi: int = 150,
    add_labels: bool = True,
    label_fmt: str = ".2f",
) -> plt.Figure:
    """
    Create a bar chart.
    
    Parameters
    ----------
    data : pd.Series
        Data to plot (index = labels, values = bar lengths).
    title : str, optional
        Plot title.
    xlabel, ylabel : str, optional
        Axis labels.
    horizontal : bool
        Whether bars are horizontal.
    palette : str
        Color palette name.
    save_path : Path, optional
        Path to save figure.
    figsize : tuple, optional
        Figure size. Auto-calculated if None.
    dpi : int
        Resolution.
    add_labels : bool
        Whether to add value labels on bars.
    label_fmt : str
        Format string for labels.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    if figsize is None:
        if horizontal:
            figsize = (10, max(6, len(data) * 0.4))
        else:
            figsize = (max(10, len(data) * 0.4), 6)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = sns.color_palette(palette, len(data))
    
    if horizontal:
        bars = ax.barh(range(len(data)), data.values, color=colors)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data.index)
        ax.invert_yaxis()
        if xlabel:
            ax.set_xlabel(xlabel)
    else:
        bars = ax.bar(range(len(data)), data.values, color=colors)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data.index, rotation=45, ha="right")
        if ylabel:
            ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title, fontsize=14)
    
    # Add value labels
    if add_labels:
        for bar, val in zip(bars, data.values):
            if horizontal:
                ax.text(
                    bar.get_width() + 0.01 * data.max(),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:{label_fmt}}",
                    va="center", fontsize=9,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * data.max(),
                    f"{val:{label_fmt}}",
                    ha="center", va="bottom", fontsize=9,
                )
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)
    
    return fig


def plot_enrichment_bar(
    enrichment_df: pd.DataFrame,
    cluster_id: str,
    n_genes: int = 10,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 8),
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a bar chart showing top enriched genes for a cluster.
    
    Parameters
    ----------
    enrichment_df : pd.DataFrame
        Enrichment results with columns: gene_name, odds_ratio, p_adj, cluster.
    cluster_id : str
        Cluster to show enrichment for.
    n_genes : int
        Number of top genes to show.
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
    cluster_data = enrichment_df[enrichment_df["cluster"] == cluster_id].copy()
    
    if cluster_data.empty:
        logger.warning(f"No enrichment data for cluster {cluster_id}")
        return None
    
    # Get top genes by odds ratio
    top_genes = cluster_data.nlargest(n_genes, "odds_ratio")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ["salmon" if p < 0.05 else "lightblue" for p in top_genes["p_adj"]]
    
    bars = ax.barh(range(len(top_genes)), top_genes["odds_ratio"], color=colors)
    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(top_genes["gene_name"])
    ax.invert_yaxis()
    
    ax.axvline(1, color="gray", linestyle="--", alpha=0.5)
    
    ax.set_xlabel("Odds Ratio")
    ax.set_title(title or f"Top Enriched Genes in {cluster_id}", fontsize=14)
    
    # Legend for significance
    ax.scatter([], [], c="salmon", label="p_adj < 0.05")
    ax.scatter([], [], c="lightblue", label="p_adj >= 0.05")
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path, dpi=dpi, close=False)
    
    return fig
