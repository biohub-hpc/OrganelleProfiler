#!/usr/bin/env python3
"""Gene x Reporter distinctiveness heatmap.

Computes per-geneKO mean mAP distinctiveness scores for each reporter and a
global baseline (all reporter features pooled), then produces clustered
heatmaps.  Uses the same mAP distinctiveness call as the reporter_radar stage:
all geneKOs included (no activity filter), copairs mean_average_precision with
pos_sameby=perturbation, pos_diffby=sgRNA, neg_diffby=perturbation.

Produces 4 heatmaps per subset (genes as rows x reporters as columns):

  1. Raw mAP values -- includes an "all_combined" column showing the true
     global mAP (all reporter features pooled into one feature vector).
  2. Normalized -- each reporter's per-gene mAP divided by the global baseline.
     Values >1 mean that reporter captures the gene's phenotype better than
     the full feature set.

Both all-cells and downsampled subsets are processed.

Rows (genes) and columns (reporters) are hierarchically clustered (Ward
linkage, Euclidean distance).  Two annotation colour bars are drawn on the
left of each heatmap:

  - CHAD-Boosted supercategory (8 categories)
  - CHAD protein complex / cluster name

Genes not assigned to a category are shown in gray ("Uncategorized").

Inputs
------
Reads PCA-optimized h5ad files to compute fresh mAP scores::

    <pca-dir>/
        all/guide_pca_optimized.h5ad, gene_pca_optimized.h5ad
        downsampled/guide_pca_optimized.h5ad, gene_pca_optimized.h5ad

Outputs
-------
Results are saved into <radar-dir>/heatmaps/<subset>/::

    gene_reporter_distinctiveness_raw.csv
    gene_reporter_distinctiveness_normalized.csv
    global_baseline_distinctiveness.csv
    heatmap_gene_reporter_raw.png
    heatmap_gene_reporter_normalized.png

Usage
-----
Submit as SLURM jobs (recommended -- one job per subset)::

    python gene_reporter_distinctiveness_heatmap.py --slurm -y

Submit with custom params::

    python gene_reporter_distinctiveness_heatmap.py --slurm -y \\
        --pca-dir /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v2/dino \\
        --radar-dir /home/gav.sturm/linked_folders/icd.fast.ops/reporter_radar/14_reporter_radar \\
        --slurm-memory 64GB --slurm-time 10 --slurm-cpus 16

Run only one subset::

    python gene_reporter_distinctiveness_heatmap.py --slurm -y --subset all

Run locally (interactive node)::

    python gene_reporter_distinctiveness_heatmap.py

Override output directory::

    python gene_reporter_distinctiveness_heatmap.py --output-dir ./my_heatmaps

Options::

    --pca-dir PATH            Root PCA-optimized dir (contains all/ and downsampled/).
                              Default: /hpc/projects/icd.fast.ops/organelle_attribution/
                              pca_optimized_v2/dino
    --radar-dir PATH          Root reporter_radar output dir. Results go into
                              <radar-dir>/heatmaps/<subset>/. Default:
                              /home/gav.sturm/linked_folders/icd.fast.ops/
                              reporter_radar/14_reporter_radar
    --output-dir PATH         Override output directory (ignores --radar-dir).
    --subset {all,downsampled}  Process only one subset instead of both.
    --null-size INT           Null distribution size for p-value estimation
                              (default: 100000). Lower than radar stage since
                              we only use mean mAP, not p-values.
    --supercategory-config PATH  Path to gene_supercategory_mapping.yaml.

SLURM options::

    --slurm                   Submit as SLURM job(s) instead of running locally.
    --no-wait                 Don't wait for SLURM jobs to complete.
    --yes / -y                Skip confirmation prompt.
    --slurm-memory STR        Memory per job (default: 64GB).
    --slurm-time INT          Time limit in minutes (default: 10).
    --slurm-cpus INT          CPUs per job (default: 16).
"""
import argparse
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ops_utils.analysis.map_scores import phenotypic_distinctivness

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("copairs").setLevel(logging.WARNING)

DEFAULT_PCA_DIR = Path(
    "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v2/dino"
)
DEFAULT_RADAR_DIR = Path(
    "/home/gav.sturm/linked_folders/icd.fast.ops/reporter_radar/14_reporter_radar"
)
DEFAULT_SUPERCATEGORY_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "gene_supercategory_mapping.yaml"
)


# ---------------------------------------------------------------------------
# Gene annotation helpers
# ---------------------------------------------------------------------------

def load_supercategory_config(config_path: Path) -> dict:
    """Load the gene_supercategory_mapping.yaml config."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def build_chad_cluster_gene_map(supercategory_config: dict) -> Dict[str, str]:
    """Build gene -> CHAD protein complex/cluster name mapping.

    Walks the ENTIRE CHAD v5 hierarchy (all clusters) rather than just the
    subset referenced in the supercategory config.

    Returns a dict: gene_name -> cluster_name (first cluster found).
    """
    from ops_utils.analysis.gene_supercategories import (
        _load_chad_hierarchy,
        DEFAULT_CHAD_PATH,
    )

    chad_path = Path(
        supercategory_config.get("chad_hierarchy_path", str(DEFAULT_CHAD_PATH))
    )
    chad = _load_chad_hierarchy(chad_path)

    gene_to_cluster: Dict[str, str] = {}
    for _id, cluster in chad.items():
        if not isinstance(cluster, dict) or "name" not in cluster:
            continue
        cluster_name = cluster["name"]
        for gene in cluster.get("genes", []):
            if gene not in gene_to_cluster:
                gene_to_cluster[gene] = cluster_name

    return gene_to_cluster


# ---------------------------------------------------------------------------
# Reporter stats from pca_report.csv
# ---------------------------------------------------------------------------

def load_reporter_stats(pca_dir: Path) -> Dict[str, str]:
    """Load reporter -> stats string from pca_report.csv.

    Returns dict: signal_name -> "N cells | M exps: exp1, exp2"
    """
    pca_report = pca_dir / "pca_report.csv"
    if not pca_report.exists():
        return {}

    df = pd.read_csv(pca_report)
    stats: Dict[str, str] = {}
    for _, row in df.iterrows():
        signal = row["signal"]
        n_cells = int(row["n_cells"]) if "n_cells" in df.columns else 0
        exps = []
        if "experiment" in df.columns:
            exps = sorted(set(e.strip() for e in str(row["experiment"]).split(",") if e.strip()))
        n_exps = len(exps)
        parts = [f"{n_cells:,} cells"]
        if n_exps <= 3 and n_exps > 0:
            parts.append(f"{n_exps} exps: {', '.join(exps)}")
        elif n_exps > 3:
            parts.append(f"{n_exps} exps")
        stats[signal] = " | ".join(parts)
    return stats


def build_annotated_labels(reporters: List[str], stats: Dict[str, str]) -> Dict[str, str]:
    """Build reporter -> 'reporter\n(stats)' mapping for plot labels."""
    labels = {}
    for r in reporters:
        if r == "all_combined":
            labels[r] = "all_combined"
        elif r in stats:
            labels[r] = f"{r}\n({stats[r]})"
        else:
            labels[r] = r
    return labels


# ---------------------------------------------------------------------------
# mAP computation
# ---------------------------------------------------------------------------

def _run_distinctiveness(
    adata_guide: ad.AnnData,
    null_size: int,
) -> pd.DataFrame:
    """Run mAP distinctiveness on all geneKOs (no activity filter).

    Matches the radar stage approach exactly:
      - all perturbations marked as active (below_corrected_p=True)
      - phenotypic_distinctivness with plot_results=False
    """
    _all_active = pd.DataFrame({
        "perturbation": adata_guide.obs["perturbation"].unique(),
        "below_corrected_p": True,
    })
    dmap, ratio = phenotypic_distinctivness(
        adata_guide, _all_active, plot_results=False, null_size=null_size,
    )
    logger.info(f"    {ratio:.2%} distinctive ({len(dmap)} genes)")
    return dmap


def compute_all_scores(
    pca_dir: Path,
    null_size: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute per-reporter + global distinctiveness mAP from h5ad files.

    Returns
    -------
    raw_df : genes x (reporters + all_combined) DataFrame
    global_series : per-gene mAP for global baseline (all features pooled)
    """
    guide_path = pca_dir / "guide_pca_optimized.h5ad"
    gene_path = pca_dir / "gene_pca_optimized.h5ad"
    if not guide_path.exists():
        raise FileNotFoundError(f"Guide file not found: {guide_path}")
    if not gene_path.exists():
        raise FileNotFoundError(f"Gene file not found: {gene_path}")

    adata_guide = ad.read_h5ad(guide_path)
    adata_gene = ad.read_h5ad(gene_path)
    logger.info(
        f"  Loaded guide: {adata_guide.n_obs} obs x {adata_guide.n_vars} features | "
        f"gene: {adata_gene.n_obs} obs x {adata_gene.n_vars} features"
    )

    # Build label -> feature-columns map from var_name prefixes (label_PCN)
    pc_re = re.compile(r'^(.+)_PC\d+$')
    label_to_cols: Dict[str, List[str]] = {}
    for v in adata_guide.var_names:
        m = pc_re.match(v)
        prefix = m.group(1) if m else v
        label_to_cols.setdefault(prefix, []).append(v)

    reporter_labels = sorted(label_to_cols.keys())
    logger.info(f"  {len(reporter_labels)} reporters to process")

    # --- Global baseline: ALL reporter features pooled ---
    logger.info("  Computing global baseline (all features pooled)...")
    global_map = _run_distinctiveness(adata_guide, null_size)

    # --- Per-reporter ---
    reporter_maps: Dict[str, pd.DataFrame] = {}
    for i, label in enumerate(reporter_labels, 1):
        cols = label_to_cols[label]
        logger.info(f"  [{i}/{len(reporter_labels)}] {label} ({len(cols)} features)")
        col_mask = np.array([v in set(cols) for v in adata_guide.var_names])
        adata_sub = adata_guide[:, col_mask].copy()
        reporter_maps[label] = _run_distinctiveness(adata_sub, null_size)

    # --- Pivot to genes x reporters ---
    all_genes = sorted(set().union(
        *(dmap["perturbation"].tolist() for dmap in reporter_maps.values()),
        global_map["perturbation"].tolist(),
    ))

    data = {}
    for label, dmap in sorted(reporter_maps.items()):
        g2m = dict(zip(dmap["perturbation"], dmap["mean_average_precision"]))
        data[label] = [g2m.get(g, np.nan) for g in all_genes]

    g2m_global = dict(zip(global_map["perturbation"], global_map["mean_average_precision"]))
    data["all_combined"] = [g2m_global.get(g, np.nan) for g in all_genes]
    global_series = pd.Series(data["all_combined"], index=all_genes, name="all_combined")

    raw_df = pd.DataFrame(data, index=all_genes)
    raw_df.index.name = "gene"
    return raw_df, global_series


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_to_baseline(raw_df: pd.DataFrame, global_series: pd.Series) -> pd.DataFrame:
    """Normalize each reporter column by dividing by the true global baseline."""
    reporter_cols = [c for c in raw_df.columns if c != "all_combined"]
    norm_df = raw_df[reporter_cols].copy()
    baseline = global_series.reindex(norm_df.index).replace(0, np.nan)
    for col in reporter_cols:
        norm_df[col] = norm_df[col] / baseline
    norm_df = norm_df.replace([np.inf, -np.inf], np.nan)
    return norm_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _build_color_maps(categories: list, palette_name: str = "tab10") -> Tuple[Dict[str, str], list]:
    """Build category -> hex colour mapping, with gray for Uncategorized."""
    unique = sorted(set(c for c in categories if c != "Uncategorized"))
    palette = sns.color_palette(palette_name, n_colors=max(len(unique), 1))
    cmap = {cat: matplotlib.colors.to_hex(palette[i]) for i, cat in enumerate(unique)}
    cmap["Uncategorized"] = "#d3d3d3"
    colors = [cmap.get(c, "#d3d3d3") for c in categories]
    return cmap, colors


def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    gene_supercats: Optional[Dict[str, str]] = None,
    gene_clusters: Optional[Dict[str, str]] = None,
    cmap: str = "viridis",
    metric: str = "correlation",
    method: str = "average",
    vmin: float = None,
    vmax: float = None,
    center: float = None,
    power_scale: float = None,
) -> Tuple[List[int], List[int]]:
    """Plot a clustered genes x reporters heatmap and return the leaf orderings.

    Seaborn's clustermap handles all clustering, dendrograms, and leaf
    ordering.  The computed row/column orders are extracted and returned
    so the interactive HTML heatmap can reuse them exactly.

    Parameters
    ----------
    metric : str
        Distance metric for clustering (default ``"correlation"``).
    method : str
        Linkage method (default ``"average"``).
    power_scale : float, optional
        Power transform for the colour scale (e.g. 0.5 = sqrt).

    Returns
    -------
    ordered_genes, ordered_reporters : list[str]
        Gene and reporter labels in clustered order (after any zero-variance
        filtering).  Passed directly to ``plot_interactive_heatmap``.
    """
    plot_df = df.fillna(0)

    # Drop zero-variance rows/columns — correlation and cosine distance are
    # undefined for constant/zero vectors
    if metric in ("correlation", "cosine"):
        row_std = plot_df.std(axis=1)
        zero_var_rows = row_std[row_std == 0].index.tolist()
        if zero_var_rows:
            logger.info(f"Dropping {len(zero_var_rows)} zero-variance genes for "
                        f"{metric} clustering: {zero_var_rows[:5]}...")
            plot_df = plot_df.loc[row_std > 0]
        col_std = plot_df.std(axis=0)
        zero_var_cols = col_std[col_std == 0].index.tolist()
        if zero_var_cols:
            logger.info(f"Dropping {len(zero_var_cols)} zero-variance reporters for "
                        f"{metric} clustering: {zero_var_cols[:5]}...")
            plot_df = plot_df.loc[:, col_std > 0]

    if power_scale is not None:
        plot_df = plot_df.clip(lower=0).pow(power_scale)

    # Build row colour annotation DataFrames
    row_colors_df = None
    supercat_cmap = {}
    cluster_cmap = {}
    if gene_supercats is not None or gene_clusters is not None:
        annot = {}
        if gene_supercats is not None:
            cats = [gene_supercats.get(g, "Uncategorized") for g in plot_df.index]
            supercat_cmap, colors = _build_color_maps(cats, "tab10")
            annot["Supercategory"] = colors
        if gene_clusters is not None:
            clusters = [gene_clusters.get(g, "Uncategorized") for g in plot_df.index]
            cluster_cmap, colors = _build_color_maps(clusters, "tab20")
            annot["CHAD Cluster"] = colors
        if annot:
            row_colors_df = pd.DataFrame(annot, index=plot_df.index)

    # Figure sizing — tall enough for gene labels, narrow columns
    n_genes = len(plot_df)
    n_reporters = len(plot_df.columns)
    row_height = 0.14  # inches per gene row
    col_width = 0.45   # inches per reporter column
    fig_w = max(18, n_reporters * col_width + 10)
    fig_h = max(12, n_genes * row_height + 5)

    g = sns.clustermap(
        plot_df,
        metric=metric,
        method=method,
        row_colors=row_colors_df,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        xticklabels=True,
        yticklabels=True,
        figsize=(fig_w, fig_h),
        cbar_kws={"label": "mAP distinctiveness", "shrink": 0.4},
        dendrogram_ratio=(0.05, 0.04),
        colors_ratio=0.03 if row_colors_df is not None else 0,
    )

    # Extract leaf orderings as label lists for the HTML heatmap
    if g.dendrogram_row:
        ordered_genes = [plot_df.index[i] for i in g.dendrogram_row.reordered_ind]
    else:
        ordered_genes = list(plot_df.index)
    if g.dendrogram_col:
        ordered_reporters = [plot_df.columns[i] for i in g.dendrogram_col.reordered_ind]
    else:
        ordered_reporters = list(plot_df.columns)

    g.ax_heatmap.set_xlabel("Reporter", fontsize=20)
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.tick_params(axis="y", labelsize=10)
    g.ax_heatmap.tick_params(axis="x", labelsize=16)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    g.fig.suptitle(title, fontsize=24, y=1.01)

    # Relabel colorbar with real mAP values at their power-scaled positions
    if power_scale is not None:
        real_ticks = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        data_max = float(df.fillna(0).clip(lower=0).values.max())
        real_ticks = real_ticks[real_ticks <= data_max * 1.05]
        scaled_ticks = real_ticks ** power_scale
        g.cax.set_yticks(scaled_ticks)
        g.cax.set_yticklabels([f"{t:.1f}" for t in real_ticks])
    g.cax.tick_params(labelsize=14)
    g.cax.set_ylabel("mAP distinctiveness", fontsize=16)

    # Add reporter names on top of columns as well
    ax_hm = g.ax_heatmap
    ordered_labels = ordered_reporters
    ax_top = ax_hm.secondary_xaxis("top")
    ax_top.set_xticks(np.arange(len(ordered_labels)) + 0.5)
    ax_top.set_xticklabels(ordered_labels, rotation=45, ha="left", fontsize=16)
    ax_top.tick_params(length=0)

    # Label the colour-bar columns with category type names on top
    if row_colors_df is not None and hasattr(g, "ax_row_colors"):
        ax_rc = g.ax_row_colors
        col_names = list(row_colors_df.columns)
        for i, name in enumerate(col_names):
            ax_rc.text(
                i + 0.5, -0.005, name,
                transform=ax_rc.get_xaxis_transform(),
                ha="center", va="top", fontsize=11, rotation=45,
            )

    # Move the colorbar to far right
    g.cax.set_position([1.03, 0.4, 0.008, 0.2])

    # Legends for colour bars
    legend_handles = []
    if supercat_cmap:
        for cat, color in sorted(supercat_cmap.items()):
            legend_handles.append(plt.Line2D([0], [0], color=color, marker="s",
                                             linestyle="", markersize=14, label=cat))
    if cluster_cmap and len(cluster_cmap) <= 150:
        if legend_handles:
            legend_handles.append(plt.Line2D([0], [0], color="white", marker="",
                                             linestyle="", label=""))  # spacer
        for cat, color in sorted(cluster_cmap.items()):
            legend_handles.append(plt.Line2D([0], [0], color=color, marker="s",
                                             linestyle="", markersize=12, label=cat))

    if legend_handles:
        g.fig.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 0.95),
            fontsize=14,
            frameon=False,
            title="Annotations",
            title_fontsize=16,
        )

    g.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    logger.info(f"Saved: {out_path}")

    return ordered_genes, ordered_reporters


def plot_interactive_heatmap(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    ordered_genes: Optional[List[str]] = None,
    ordered_reporters: Optional[List[str]] = None,
    gene_supercats: Optional[Dict[str, str]] = None,
    gene_clusters: Optional[Dict[str, str]] = None,
    reporter_stats: Optional[Dict[str, str]] = None,
):
    """Save an interactive HTML heatmap with hover showing gene, reporter, mAP, cells, exps."""
    import plotly.graph_objects as go

    plot_df = df.fillna(0)
    # Reorder and subset to match the PNG (which may have dropped zero-variance entries)
    if ordered_genes is not None:
        plot_df = plot_df.loc[[g for g in ordered_genes if g in plot_df.index]]
    if ordered_reporters is not None:
        plot_df = plot_df[[r for r in ordered_reporters if r in plot_df.columns]]

    genes = list(plot_df.index)
    reporters = list(plot_df.columns)
    z = plot_df.values

    # Build hover text matrix
    hover = []
    for i, gene in enumerate(genes):
        row = []
        for j, reporter in enumerate(reporters):
            val = z[i, j]
            parts = [
                f"<b>Gene:</b> {gene}",
                f"<b>Reporter:</b> {reporter}",
                f"<b>mAP:</b> {val:.4f}",
            ]
            if gene_supercats and gene in gene_supercats:
                parts.append(f"<b>Supercategory:</b> {gene_supercats[gene]}")
            if gene_clusters and gene in gene_clusters:
                parts.append(f"<b>CHAD Cluster:</b> {gene_clusters[gene]}")
            if reporter_stats and reporter in reporter_stats:
                parts.append(f"<b>Stats:</b> {reporter_stats[reporter]}")
            row.append("<br>".join(parts))
        hover.append(row)

    # Apply power scale (x^0.5) for colour mapping; colorbar shows real mAP values
    z_display = np.clip(z, 0, None) ** 0.5

    # Evenly-spaced real mAP ticks, positioned at their power-scaled locations
    real_ticks = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    z_max = float(z.max()) if z.size else 1.0
    real_ticks = real_ticks[real_ticks <= z_max * 1.05]
    tick_vals = real_ticks ** 0.5
    tick_text = [f"{t:.1f}" for t in real_ticks]

    fig = go.Figure(data=go.Heatmap(
        z=z_display,
        x=reporters,
        y=genes,
        hovertext=hover,
        hoverinfo="text",
        colorscale="Inferno",
        colorbar=dict(title="mAP", tickvals=tick_vals, ticktext=tick_text, len=0.3, y=0.5),
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Reporter", tickfont=dict(size=9), tickangle=45),
        yaxis=dict(title="", tickfont=dict(size=5), autorange="reversed"),
        width=max(750, len(reporters) * 18 + 175),
        height=max(900, len(genes) * 7 + 175),
    )

    fig.write_html(str(out_path), include_plotlyjs="cdn")
    logger.info(f"Saved interactive: {out_path}")


# ---------------------------------------------------------------------------
# Per-category strip plots
# ---------------------------------------------------------------------------

def plot_category_strip(
    raw_df: pd.DataFrame,
    category_name: str,
    gene_list: List[str],
    out_path: Path,
    n_label: int = 10,
    label_all: bool = False,
    dot_size: int = 25,
    reporter_stats: Optional[Dict[str, str]] = None,
):
    """Strip plot of per-gene mAP scores across reporters for one category.

    X-axis = reporters, Y-axis = raw mAP, each dot = one gene.
    Top and bottom n_label genes are labelled per reporter, or all if label_all=True.
    """
    # Subset to genes in this category that exist in the matrix
    reporter_cols = list(raw_df.columns)
    genes = sorted(set(gene_list) & set(raw_df.index))
    if len(genes) < 2:
        return

    sub = raw_df.loc[genes, reporter_cols]

    # Melt to long form
    long = sub.reset_index().melt(id_vars="gene", var_name="reporter", value_name="mAP")
    long = long.dropna(subset=["mAP"])

    n_reporters = len(reporter_cols)
    fig_w = max(14, n_reporters * 0.5 + 4)
    fig_h = max(6, 8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Use inferno colormap for dot colors based on mAP^0.3
    cmap_obj = plt.cm.inferno
    norm_vals = np.clip(long["mAP"].values, 0, 1)
    colors = cmap_obj(norm_vals)

    # Jittered strip plot
    reporter_order = sub.mean(axis=0).sort_values(ascending=True).index.tolist()
    reporter_to_x = {r: i for i, r in enumerate(reporter_order)}
    x_vals = long["reporter"].map(reporter_to_x).values
    jitter = np.random.default_rng(42).uniform(-0.3, 0.3, size=len(x_vals))

    ax.scatter(x_vals + jitter, long["mAP"].values, c=colors, s=dot_size, alpha=0.7,
               edgecolors="none", zorder=3)

    # Category mean per reporter
    means = sub.mean(axis=0).reindex(reporter_order)
    ax.plot(range(len(reporter_order)), means.values, color="white", linewidth=5,
            zorder=4, alpha=0.9)
    ax.plot(range(len(reporter_order)), means.values, color="red", linewidth=2.5,
            zorder=5, alpha=0.9, linestyle="--", label="category mean")

    # Label top genes only for the top 15 reporters (by mean mAP)
    y_range = sub.values.max() - sub.values.min()
    min_gap = y_range * 0.035  # vertical gap between stacked labels
    top_reporters = set(reporter_order[-15:])

    for reporter in reporter_order:
        col_vals = sub[reporter].dropna().sort_values()
        rx = reporter_to_x[reporter]
        if reporter not in top_reporters and not label_all:
            continue
        if label_all:
            to_label = col_vals
        else:
            to_label = col_vals.tail(n_label).sort_values()

        # Place labels above dots, pushing up when they overlap
        prev_y = -np.inf
        for gene in to_label.index:
            y = max(to_label[gene] + min_gap, prev_y + min_gap)
            ax.annotate(
                gene, xy=(rx, to_label[gene]), xytext=(rx, y),
                fontsize=7.5, ha="center", va="bottom", alpha=0.8,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.3),
            )
            prev_y = y

    ax.set_xticks(range(len(reporter_order)))
    ax.set_xticks(range(len(reporter_order)))
    ax.set_xticklabels([""] * len(reporter_order))  # clear default labels
    for idx, r in enumerate(reporter_order):
        # Bold reporter name
        ax.text(idx, -0.02, r, transform=ax.get_xaxis_transform(),
                ha="right", va="top", fontsize=9, fontweight="bold", rotation=45)
        # Gray stats below
        if reporter_stats and r in reporter_stats:
            ax.text(idx, -0.06, f"({reporter_stats[r]})", transform=ax.get_xaxis_transform(),
                    ha="right", va="top", fontsize=6, color="gray", rotation=45)
    ax.set_ylabel("mAP distinctiveness", fontsize=14)
    ax.set_title(f"{category_name}  (n={len(genes)} genes)", fontsize=16)
    ax.set_xlim(-0.5, len(reporter_order) - 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("mAP", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_category_ridge(
    raw_df: pd.DataFrame,
    category_name: str,
    gene_list: List[str],
    out_path: Path,
    reporter_stats: Optional[Dict[str, str]] = None,
    n_label: int = 10,
):
    """Ridge plot of per-gene mAP distributions across reporters for one category.

    Y-axis = reporters stacked vertically, X-axis = mAP score.
    Each ridge is a KDE with rug ticks for individual genes.
    Top n_label outlier genes are labelled.
    """
    from scipy.stats import gaussian_kde

    reporter_cols = list(raw_df.columns)
    genes = sorted(set(gene_list) & set(raw_df.index))
    if len(genes) < 3:
        return

    sub = raw_df.loc[genes, reporter_cols]

    reporter_means = sub.mean(axis=0).sort_values(ascending=True)
    reporter_order = reporter_means.index.tolist()

    n_reporters = len(reporter_order)
    fig_h = max(12, n_reporters * 0.8 + 4)
    fig_w = 20
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap_obj = plt.cm.inferno
    overlap = 0.7  # how much ridges overlap
    x_grid = np.linspace(0, sub.values.max() * 1.1, 300)

    for i, reporter in enumerate(reporter_order):
        vals = sub[reporter].dropna().values
        if len(vals) < 3:
            continue

        # KDE
        try:
            kde = gaussian_kde(vals, bw_method=0.3)
            density = kde(x_grid)
        except Exception:
            continue

        # Normalize density to a fixed visual height
        density = density / density.max() * overlap

        baseline = i
        # Fill color based on reporter mean
        mean_val = reporter_means[reporter]
        fill_color = cmap_obj(np.clip(mean_val, 0, 1) ** 0.3)

        ax.fill_between(x_grid, baseline, baseline + density,
                         color=fill_color, alpha=0.7, zorder=n_reporters - i)
        ax.plot(x_grid, baseline + density, color="black", linewidth=0.5,
                zorder=n_reporters - i + 1)

        # Rug ticks
        ax.scatter(vals, np.full_like(vals, baseline + 0.02),
                   marker="|", s=15, color="black", alpha=0.4,
                   zorder=n_reporters - i + 2, linewidths=0.5)

        # Label top outlier genes — stack horizontally to avoid overlap
        sorted_vals = pd.Series(vals, index=sub[reporter].dropna().index).sort_values()
        top_genes = sorted_vals.tail(n_label).sort_values()
        x_max = sub.values.max() * 1.1
        min_x_gap = x_max * 0.08
        prev_x = -np.inf
        for gene in top_genes.index:
            gval = top_genes[gene]
            lx = max(gval, prev_x + min_x_gap)
            ax.annotate(
                gene, xy=(gval, baseline + 0.02),
                xytext=(lx, baseline + overlap * 0.5),
                fontsize=8, ha="center", va="bottom", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                zorder=n_reporters + 10,
            )
            prev_x = lx

    ax.set_yticks(range(n_reporters))
    ax.set_yticklabels([""] * n_reporters)  # clear default labels
    for idx, r in enumerate(reporter_order):
        # Bold reporter name
        ax.text(-0.01, idx, r, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9, fontweight="bold")
        # Gray stats to the right of name
        if reporter_stats and r in reporter_stats:
            ax.text(-0.01, idx - 0.3, f"({reporter_stats[r]})",
                    transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=6, color="gray")
    ax.set_xlabel("mAP distinctiveness", fontsize=14)
    ax.set_title(f"{category_name}  (n={len(genes)} genes)", fontsize=16)
    ax.set_ylim(-0.3, n_reporters + 0.5)
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.3)

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap_obj,
        norm=matplotlib.colors.PowerNorm(gamma=0.3, vmin=0, vmax=1),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_label("mAP", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_one_category(args):
    """Worker for parallel category plotting. Unpacks args tuple."""
    raw_df, cat, genes, strip_path, ridge_path, strip_kwargs, reporter_stats = args
    plot_category_strip(raw_df, cat, genes, strip_path, reporter_stats=reporter_stats, **strip_kwargs)
    plot_category_ridge(raw_df, cat, genes, ridge_path, reporter_stats=reporter_stats)
    return cat


def plot_all_category_plots(
    raw_df: pd.DataFrame,
    gene_to_cat: Dict[str, str],
    out_dir: Path,
    label: str,
    n_workers: int = 8,
    label_all: bool = False,
    dot_size: int = 25,
    reporter_stats: Optional[Dict[str, str]] = None,
):
    """Generate strip and ridge plots for all categories in a mapping (parallel)."""
    # Invert: category -> [genes]
    cat_to_genes: Dict[str, List[str]] = {}
    for gene, cat in gene_to_cat.items():
        if cat == "Uncategorized" or cat == "Other":
            continue
        cat_to_genes.setdefault(cat, []).append(gene)

    strip_dir = out_dir / "strip"
    ridge_dir = out_dir / "ridge"
    strip_dir.mkdir(parents=True, exist_ok=True)
    ridge_dir.mkdir(parents=True, exist_ok=True)

    strip_kwargs = {"label_all": label_all, "dot_size": dot_size}
    tasks = []
    for cat, genes in sorted(cat_to_genes.items()):
        safe_name = cat.replace(" ", "_").replace("/", "_").replace("&", "and")
        tasks.append((raw_df, cat, genes, strip_dir / f"{safe_name}.png", ridge_dir / f"{safe_name}.png", strip_kwargs, reporter_stats))

    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n = len(tasks)
    logger.info(f"  Plotting {n} categories with {n_workers} threads...")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_plot_one_category, t): t[1] for t in tasks}
        with tqdm(total=n, desc=f"  {label}", unit="cat") as pbar:
            for future in as_completed(futures):
                cat = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"  {label}: {cat} FAILED: {e}")
                pbar.update(1)


# ---------------------------------------------------------------------------
# Main pipeline per subset
# ---------------------------------------------------------------------------

def run_for_subset(
    subset_name: str,
    pca_dir: Path,
    out_dir: Path,
    gene_supercats: Dict[str, str],
    gene_clusters: Dict[str, str],
    null_size: int,
):
    """Compute mAP scores and generate heatmaps for one subset.

    If CSVs from a previous run already exist in the output dir, skips mAP
    computation and just regenerates heatmaps from the cached data.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {subset_name}")
    logger.info(f"{'='*60}")

    subset_out = out_dir / subset_name
    subset_out.mkdir(parents=True, exist_ok=True)

    raw_csv = subset_out / "gene_reporter_distinctiveness_raw.csv"
    global_csv = subset_out / "global_baseline_distinctiveness.csv"

    # Load reporter stats (n_cells, experiments) from pca_report.csv
    pca_subdir = pca_dir / subset_name
    reporter_stats = load_reporter_stats(pca_subdir) if pca_subdir.exists() else {}

    if raw_csv.exists() and global_csv.exists():
        # Reuse cached CSVs — skip mAP computation
        logger.info(f"  Found existing CSVs, skipping mAP computation")
        logger.info(f"    {raw_csv}")
        logger.info(f"    {global_csv}")
        raw_df = pd.read_csv(raw_csv, index_col=0)
        raw_df.index.name = "gene"
        global_df = pd.read_csv(global_csv, index_col=0)
        global_series = global_df["mean_average_precision"]
        global_series.index.name = "gene"
    else:
        # Compute fresh mAP scores
        pca_subdir = pca_dir / subset_name
        if not pca_subdir.exists():
            logger.warning(f"Skipping {subset_name}: {pca_subdir} not found")
            return

        raw_df, global_series = compute_all_scores(pca_subdir, null_size)

        # Save CSVs for future reuse
        raw_df.to_csv(raw_csv)
        logger.info(f"Saved raw matrix: {raw_csv} ({raw_df.shape})")

        global_series.to_frame("mean_average_precision").to_csv(global_csv)
        logger.info(f"Saved global baseline: {global_csv}")

    # Return data for two-phase plotting (heatmaps first, then strip/ridge)
    norm_df = normalize_to_baseline(raw_df, global_series)
    norm_csv = subset_out / "gene_reporter_distinctiveness_normalized.csv"
    norm_df.to_csv(norm_csv)
    logger.info(f"Saved normalized matrix: {norm_csv} ({norm_df.shape})")

    return raw_df, norm_df, subset_out, reporter_stats


def generate_heatmaps(
    raw_df: pd.DataFrame,
    norm_df: pd.DataFrame,
    subset_name: str,
    subset_out: Path,
    gene_supercats: Dict[str, str],
    gene_clusters: Dict[str, str],
    reporter_stats: Dict[str, str],
):
    """Generate all heatmaps (PNG + HTML) for one subset.

    Two clustering variants are produced:
      - correlation/ : correlation distance + average linkage (pattern-based)
      - euclidean/   : euclidean distance + ward linkage (magnitude-sensitive)
    """
    norm_clipped = norm_df.clip(upper=1.0)

    clustering_variants = [
        ("correlation", "correlation", "average"),
        ("cosine", "cosine", "average"),
        ("euclidean", "euclidean", "ward"),
    ]

    for variant_name, metric, method in clustering_variants:
        variant_dir = subset_out / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        suffix = f" [{variant_name}]"

        # 1. Raw heatmap PNG — seaborn owns clustering; returns leaf order
        raw_ro, raw_co = plot_heatmap(
            raw_df,
            title=f"Gene x Reporter mAP Distinctiveness -- {subset_name}{suffix}",
            out_path=variant_dir / "heatmap_gene_reporter_raw.png",
            gene_supercats=gene_supercats,
            gene_clusters=gene_clusters,
            cmap="inferno",
            metric=metric,
            method=method,
            power_scale=0.5,
        )

        # 2. Normalized heatmap PNG
        norm_ro, norm_co = plot_heatmap(
            norm_clipped,
            title=f"Gene x Reporter mAP Distinctiveness (normalized) -- {subset_name}{suffix}",
            out_path=variant_dir / "heatmap_gene_reporter_normalized.png",
            gene_supercats=gene_supercats,
            gene_clusters=gene_clusters,
            cmap="inferno",
            metric=metric,
            method=method,
            vmin=0.0,
            vmax=1.0,
            power_scale=0.5,
        )

        # 3. Raw interactive HTML — reuses PNG leaf order
        plot_interactive_heatmap(
            raw_df,
            title=f"Gene x Reporter mAP Distinctiveness -- {subset_name}{suffix}",
            out_path=variant_dir / "heatmap_gene_reporter_raw.html",
            ordered_genes=raw_ro,
            ordered_reporters=raw_co,
            gene_supercats=gene_supercats,
            gene_clusters=gene_clusters,
            reporter_stats=reporter_stats,
        )

        # 4. Normalized interactive HTML — reuses PNG leaf order
        plot_interactive_heatmap(
            norm_clipped,
            title=f"Gene x Reporter mAP Distinctiveness (normalized) -- {subset_name}{suffix}",
            out_path=variant_dir / "heatmap_gene_reporter_normalized.html",
            ordered_genes=norm_ro,
            ordered_reporters=norm_co,
            gene_supercats=gene_supercats,
            gene_clusters=gene_clusters,
            reporter_stats=reporter_stats,
        )


def generate_strip_ridge(
    raw_df: pd.DataFrame,
    subset_out: Path,
    gene_supercats: Dict[str, str],
    gene_clusters: Dict[str, str],
    reporter_stats: Dict[str, str],
):
    """Generate per-category strip + ridge plots for one subset."""
    logger.info("Generating per-supercategory plots...")
    plot_all_category_plots(
        raw_df, gene_supercats, subset_out / "per_supercategory", "Supercategory",
        reporter_stats=reporter_stats,
    )
    logger.info("Generating per-CHAD-cluster plots...")
    plot_all_category_plots(
        raw_df, gene_clusters, subset_out / "per_chad_cluster", "CHAD Cluster",
        label_all=True, reporter_stats=reporter_stats,
    )


# ---------------------------------------------------------------------------
# SLURM worker (top-level, picklable)
# ---------------------------------------------------------------------------

def slurm_worker(
    subset_name: str,
    pca_dir: str,
    out_dir: str,
    supercategory_config_path: str,
    null_size: int,
) -> str:
    """Top-level worker function for submit_parallel_jobs.

    Must be importable and picklable — all heavy imports happen inside.
    """
    import traceback
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("copairs").setLevel(logging.WARNING)

    pca_dir = Path(pca_dir)
    out_dir = Path(out_dir)

    try:
        supercat_config = load_supercategory_config(Path(supercategory_config_path))
        gene_clusters = build_chad_cluster_gene_map(supercat_config)

        from ops_utils.analysis.gene_supercategories import build_gene_supercategory_map
        gene_supercats = build_gene_supercategory_map(supercat_config, boosted=True)

        result = run_for_subset(
            subset_name=subset_name,
            pca_dir=pca_dir,
            out_dir=out_dir,
            gene_supercats=gene_supercats,
            gene_clusters=gene_clusters,
            null_size=null_size,
        )
        if result is None:
            return f"SKIPPED: {subset_name}"
        raw_df, norm_df, subset_out, reporter_stats = result
        generate_heatmaps(raw_df, norm_df, subset_name, subset_out,
                          gene_supercats, gene_clusters, reporter_stats)
        generate_strip_ridge(raw_df, subset_out, gene_supercats, gene_clusters, reporter_stats)
        return f"OK: {subset_name}"
    except Exception as e:
        traceback.print_exc()
        return f"FAILED: {subset_name}: {e}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gene x Reporter distinctiveness heatmap"
    )
    parser.add_argument(
        "--pca-dir", type=Path, default=DEFAULT_PCA_DIR,
        help="Root PCA-optimized dir (contains all/ and downsampled/ subdirs)",
    )
    parser.add_argument(
        "--radar-dir", type=Path, default=DEFAULT_RADAR_DIR,
        help="Root reporter_radar output dir. Results go into <radar-dir>/heatmaps/<subset>/",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory (ignores --radar-dir)",
    )
    parser.add_argument(
        "--subset", type=str, default=None, choices=["all", "downsampled"],
        help="Run only one subset (default: both)",
    )
    parser.add_argument(
        "--null-size", type=int, default=100_000,
        help="Null distribution size for p-value estimation",
    )
    parser.add_argument(
        "--supercategory-config", type=Path, default=DEFAULT_SUPERCATEGORY_CONFIG,
        help="Path to gene_supercategory_mapping.yaml",
    )

    slurm_group = parser.add_argument_group("SLURM options")
    slurm_group.add_argument("--slurm", action="store_true",
                             help="Submit as SLURM job(s)")
    slurm_group.add_argument("--no-wait", action="store_true",
                             help="Don't wait for SLURM job to complete")
    slurm_group.add_argument("--yes", "-y", action="store_true",
                             help="Skip confirmation prompt")
    slurm_group.add_argument("--slurm-memory", type=str, default="64GB",
                             help="Memory per job (default: 64GB)")
    slurm_group.add_argument("--slurm-time", type=int, default=10,
                             help="Time limit in minutes (default: 10)")
    slurm_group.add_argument("--slurm-cpus", type=int, default=16,
                             help="CPUs per job (default: 16)")

    args = parser.parse_args()

    out_dir = args.output_dir or (args.radar_dir / "heatmaps")
    subsets = [args.subset] if args.subset else ["all", "downsampled"]

    # ------------------------------------------------------------------
    # SLURM mode
    # ------------------------------------------------------------------
    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        slurm_params = {
            "timeout_min": args.slurm_time,
            "mem": args.slurm_memory,
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": "cpu,gpu",
        }

        jobs = []
        for subset in subsets:
            jobs.append({
                "name": f"distinctiveness_heatmap_{subset}",
                "func": slurm_worker,
                "kwargs": {
                    "subset_name": subset,
                    "pca_dir": str(args.pca_dir),
                    "out_dir": str(out_dir),
                    "supercategory_config_path": str(args.supercategory_config),
                    "null_size": args.null_size,
                },
            })

        if not args.yes:
            print(f"\nDistinctiveness Heatmap SLURM Job(s):")
            print(f"  PCA dir:  {args.pca_dir}")
            print(f"  Output:   {out_dir}")
            print(f"  Subsets:  {', '.join(subsets)}")
            print(f"  Jobs:     {len(jobs)}")
            print(f"  Memory:   {args.slurm_memory}")
            print(f"  Time:     {args.slurm_time} min")
            print(f"  CPUs:     {args.slurm_cpus}")
            confirm = input("\nSubmit? [y/N] ").strip().lower()
            if confirm != "y":
                print("Cancelled.")
                return

        submit_result = submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment="distinctiveness_heatmap",
            slurm_params=slurm_params,
            log_dir="distinctiveness_heatmap",
            manifest_prefix="distinctiveness_heatmap",
            wait_for_completion=not args.no_wait,
        )

        if submit_result.get("success"):
            print(f"\nJob(s) submitted: {submit_result.get('base_job_id')}")
            print(f"  Jobs: {len(jobs)}")
        else:
            print("\nJob submission failed!")
        return

    # ------------------------------------------------------------------
    # Local mode
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    logger.info("Loading gene annotations...")
    supercat_config = load_supercategory_config(args.supercategory_config)
    gene_clusters = build_chad_cluster_gene_map(supercat_config)
    logger.info(f"  CHAD cluster map: {len(gene_clusters)} genes")

    from ops_utils.analysis.gene_supercategories import build_gene_supercategory_map
    gene_supercats = build_gene_supercategory_map(supercat_config, boosted=True)
    logger.info(f"  Supercategory map: {len(gene_supercats)} genes")

    # Phase 1: Load/compute data + generate all heatmaps (PNG + HTML) for all subsets
    subset_data = {}
    for subset in subsets:
        result = run_for_subset(
            subset_name=subset,
            pca_dir=args.pca_dir,
            out_dir=out_dir,
            gene_supercats=gene_supercats,
            gene_clusters=gene_clusters,
            null_size=args.null_size,
        )
        if result is not None:
            subset_data[subset] = result

    for subset, (raw_df, norm_df, subset_out, reporter_stats) in subset_data.items():
        logger.info(f"\nGenerating heatmaps for {subset}...")
        generate_heatmaps(
            raw_df, norm_df, subset, subset_out,
            gene_supercats, gene_clusters, reporter_stats,
        )

    # Phase 2: Strip + ridge plots (slower, after all heatmaps are done)
    for subset, (raw_df, norm_df, subset_out, reporter_stats) in subset_data.items():
        logger.info(f"\nGenerating strip/ridge plots for {subset}...")
        generate_strip_ridge(
            raw_df, subset_out, gene_supercats, gene_clusters, reporter_stats,
        )

    logger.info(f"\nDone in {time.time()-t0:.0f}s. Output: {out_dir}")


if __name__ == "__main__":
    main()
