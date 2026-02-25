"""
Common plotting utilities.

Provides reusable functions for scatter plots, colorbars, labels,
and figure saving to eliminate code duplication.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def should_skip_output(path: Path, skip_complete: bool = False) -> bool:
    """
    Check if output generation should be skipped.

    Parameters
    ----------
    path : Path
        Output path to check.
    skip_complete : bool
        Whether --skip-complete flag was passed.

    Returns
    -------
    bool
        True if output exists AND skip_complete is True.
    """
    if not skip_complete:
        return False
    path = Path(path)
    if path.exists():
        logger.debug(f"Skipping (exists): {path}")
        return True
    return False


def save_figure(
    fig: plt.Figure,
    path: Path,
    dpi: int = 150,
    bbox_inches: str = "tight",
    close: bool = True,
) -> Path:
    """
    Save figure with consistent settings.
    
    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    path : Path
        Output path.
    dpi : int
        Resolution.
    bbox_inches : str
        Bounding box setting.
    close : bool
        Whether to close figure after saving.
        
    Returns
    -------
    Path
        Path where figure was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    # logger.info(f"Saved: {path}")  # Commented out - too verbose with tqdm progress bars
    
    if close:
        plt.close(fig)
    
    return path


def create_cluster_palette(
    cluster_labels: pd.Series,
    cmap_name: str = "turbo",
    noise_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
) -> Dict[str, Any]:
    """
    Create a color palette for cluster labels.
    
    Parameters
    ----------
    cluster_labels : pd.Series
        Series of cluster labels (e.g., "c0", "c1", "c-1").
    cmap_name : str
        Colormap name for non-noise clusters.
    noise_color : tuple
        RGB color for noise cluster (c-1).
        
    Returns
    -------
    dict
        Mapping from cluster label to color.
    """
    unique_clusters = sorted(cluster_labels.unique())
    non_noise = [c for c in unique_clusters if c != "c-1"]
    n_clusters = len(non_noise)
    
    if n_clusters > 0:
        colors = sns.color_palette(cmap_name, n_colors=n_clusters)
        palette = {c: colors[i] for i, c in enumerate(non_noise)}
    else:
        palette = {}
    
    palette["c-1"] = noise_color
    return palette


def create_scatter_plot(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    c: Optional[np.ndarray] = None,
    s: int = 5,
    alpha: float = 0.5,
    cmap: str = "viridis",
    label: Optional[str] = None,
    rasterized: bool = True,
    **kwargs,
) -> PathCollection:
    """
    Create a scatter plot with consistent defaults.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes.
    x, y : np.ndarray
        Coordinates.
    c : np.ndarray, optional
        Colors or values for colormap.
    s : int
        Point size.
    alpha : float
        Transparency.
    cmap : str
        Colormap name.
    label : str, optional
        Label for legend.
    rasterized : bool
        Whether to rasterize (faster rendering for many points).
        
    Returns
    -------
    plt.PathCollection
        The scatter plot collection.
    """
    scatter = ax.scatter(
        x, y,
        c=c,
        s=s,
        alpha=alpha,
        cmap=cmap if c is not None else None,
        label=label,
        rasterized=rasterized,
        **kwargs,
    )
    return scatter


def add_gene_labels(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    mask: Optional[pd.Series] = None,
    fontsize: int = 8,
    use_adjust_text: bool = True,
    max_labels: int = 50,
    **text_kwargs,
) -> List[plt.Text]:
    """
    Add text labels to a scatter plot.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes.
    df : pd.DataFrame
        Data with coordinates and labels.
    x_col, y_col : str
        Column names for coordinates.
    label_col : str
        Column name for labels.
    mask : pd.Series, optional
        Boolean mask for which points to label.
    fontsize : int
        Font size for labels.
    use_adjust_text : bool
        Whether to use adjustText library for non-overlapping labels.
    max_labels : int
        Maximum number of labels to add.
        
    Returns
    -------
    list
        List of Text objects.
    """
    if mask is not None:
        df = df[mask]
    
    if len(df) > max_labels:
        df = df.head(max_labels)
    
    texts = []
    for _, row in df.iterrows():
        label_text = str(row[label_col])
        if label_text and label_text != "nan":
            text = ax.text(
                row[x_col], row[y_col], label_text,
                fontsize=fontsize,
                **text_kwargs,
            )
            texts.append(text)
    
    if use_adjust_text and texts:
        try:
            from adjustText import adjust_text
            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.7),
            )
        except ImportError:
            logger.warning("adjustText not available. Labels may overlap.")
    
    return texts


def plot_umap_with_colorbar(
    ax: plt.Axes,
    df: pd.DataFrame,
    color_col: str,
    x_col: str = "umap_1",
    y_col: str = "umap_2",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    s: int = 5,
    alpha: float = 0.5,
    cbar_label: Optional[str] = None,
    fig: Optional[plt.Figure] = None,
) -> PathCollection:
    """
    Create a UMAP scatter plot with colorbar for continuous values.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes.
    df : pd.DataFrame
        Data with UMAP coordinates and color values.
    color_col : str
        Column name for color values.
    x_col, y_col : str
        Column names for UMAP coordinates.
    cmap : str
        Colormap name.
    vmin, vmax : float, optional
        Color scale limits.
    s : int
        Point size.
    alpha : float
        Transparency.
    cbar_label : str, optional
        Label for colorbar.
    fig : plt.Figure, optional
        Figure for colorbar (uses ax.figure if None).
        
    Returns
    -------
    plt.PathCollection
        The scatter plot collection.
    """
    values = pd.to_numeric(df[color_col], errors="coerce")
    
    if vmax is None:
        vmax = values.quantile(0.99)
    if vmin is None:
        vmin = values.min()
    
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    scatter = ax.scatter(
        df[x_col], df[y_col],
        c=values,
        cmap=cmap,
        norm=norm,
        s=s,
        alpha=alpha,
        rasterized=True,
    )
    
    fig = fig or ax.figure
    cbar = fig.colorbar(scatter, ax=ax, orientation="vertical")
    if cbar_label:
        cbar.set_label(cbar_label)
    
    return scatter


def plot_umap_categorical(
    ax: plt.Axes,
    df: pd.DataFrame,
    hue_col: str,
    x_col: str = "umap_1",
    y_col: str = "umap_2",
    palette: Optional[Dict] = None,
    s: int = 5,
    alpha: float = 0.5,
    legend: bool = True,
) -> None:
    """
    Create a UMAP scatter plot colored by categorical variable.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes.
    df : pd.DataFrame
        Data with UMAP coordinates.
    hue_col : str
        Column name for categorical coloring.
    x_col, y_col : str
        Column names for UMAP coordinates.
    palette : dict, optional
        Color palette mapping values to colors.
    s : int
        Point size.
    alpha : float
        Transparency.
    legend : bool
        Whether to show legend.
    """
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        s=s,
        alpha=alpha,
        palette=palette,
        legend=legend,
        ax=ax,
        rasterized=True,
    )


def create_broken_axis_bar(
    data: pd.Series,
    title: str,
    xlabel: str,
    save_path: Path,
    dpi: int = 150,
) -> Path:
    """
    Create a horizontal bar plot with broken axis for outliers.
    
    If the max value is not a significant outlier, produces a standard plot.
    
    Parameters
    ----------
    data : pd.Series
        Data to plot (index = labels, values = bar lengths).
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    save_path : Path
        Output path.
    dpi : int
        Resolution.
        
    Returns
    -------
    Path
        Path where figure was saved.
    """
    if data.empty:
        logger.warning(f"No data provided for plotting '{title}'. Skipping.")
        return None
    
    max_val = data.max()
    p95 = data.quantile(0.95)
    
    # Use standard plot if max is not much larger than 95th percentile
    if max_val < p95 * 2 or len(data) <= 1:
        plt.figure(figsize=(10, 12))
        sns.barplot(x=data.values, y=data.index, orient="h", palette="rocket")
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.close()
        logger.info(f"Saved: {save_path}")
        return save_path
    
    # Create broken axis plot
    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharey=True, figsize=(14, 12),
        gridspec_kw={"width_ratios": [4, 1]}
    )
    fig.subplots_adjust(wspace=0.05)
    
    # Plot same data on both axes
    sns.barplot(x=data.values, y=data.index, orient="h", palette="rocket", ax=ax1)
    sns.barplot(x=data.values, y=data.index, orient="h", palette="rocket", ax=ax2)
    
    # Set axis limits
    cutoff = p95 * 1.1
    ax1.set_xlim(0, cutoff)
    ax2.set_xlim(max_val * 0.99, max_val * 1.01)
    
    # Hide connecting spines
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.yaxis.set_ticks_position("none")
    ax2.tick_params(labelleft=False)
    ax2.set_ylabel("")
    
    # Add break lines
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    
    fig.suptitle(title, fontsize=18)
    ax1.set_xlabel(xlabel)
    ax2.set_xlabel("")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    
    logger.info(f"Saved broken-axis plot: {save_path}")
    return save_path
