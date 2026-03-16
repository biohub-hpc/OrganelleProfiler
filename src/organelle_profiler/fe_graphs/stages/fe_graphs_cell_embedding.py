"""
Cell-Level Cross-Signal Embedding Visualization.

Loads subsampled PCA-reduced cells from each signal group, concatenates them,
and runs UMAP + PHATE colored by signal type. This shows whether cells from
different biological signals occupy distinct regions of feature space.

Usage:
  python -m organelle_profiler.fe_graphs.stages.fe_graphs_cell_embedding \
    --input /hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized/downsampled \
    --slurm

The script looks for *_cells_sub.h5ad files in per_signal/ (downsampled) or
per_channel/ (non-downsampled) subdirectories.
"""

import time
import logging
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import anndata as ad

logger = logging.getLogger(__name__)


def run_cell_embedding(
    input_dir: str,
    max_cells: int = 1_000_000,
    random_seed: int = 42,
) -> str:
    """
    Load cell subsamples, concatenate, run UMAP + PHATE, save plots.

    Parameters
    ----------
    input_dir : str
        Path to the PCA optimization output dir (contains per_signal/ or per_channel/).
    max_cells : int
        Maximum total cells to embed. If total exceeds this, subsample proportionally.
    random_seed : int
        Random seed for reproducibility.
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _logger = logging.getLogger(__name__)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    t_start = time.time()
    input_dir = Path(input_dir)
    rng = np.random.RandomState(random_seed)

    # Find cell subsample h5ads
    cell_files = sorted(input_dir.glob("per_signal/*_cells_sub.h5ad"))
    if not cell_files:
        cell_files = sorted(input_dir.glob("per_channel/*_cells_sub.h5ad"))
    if not cell_files:
        return f"FAILED: no *_cells_sub.h5ad files found in {input_dir}"

    _logger.info(f"Found {len(cell_files)} cell subsample files")

    # Load all cell subsamples
    blocks = []
    signal_counts = {}
    for f in cell_files:
        adata = ad.read_h5ad(f)
        signal = adata.obs["signal"].iloc[0] if "signal" in adata.obs.columns else f.stem.replace("_cells_sub", "")
        signal_counts[signal] = adata.n_obs
        blocks.append(adata)
        _logger.info(f"  {signal}: {adata.n_obs:,} cells, {adata.n_vars} PCs")

    # Concatenate — each signal group has different PCs, so use outer join (fills missing with 0)
    adata_all = ad.concat(blocks, join="outer")
    adata_all.X = np.nan_to_num(np.asarray(adata_all.X, dtype=np.float32), nan=0.0)
    del blocks

    n_total = adata_all.n_obs
    _logger.info(f"Total: {n_total:,} cells, {adata_all.n_vars} features")

    # Subsample if exceeds max
    if n_total > max_cells:
        _logger.info(f"Subsampling {n_total:,} → {max_cells:,} cells (proportional per signal)")
        signals = adata_all.obs["signal"].values
        unique_signals = np.unique(signals)
        keep_idx = []
        for sig in unique_signals:
            sig_idx = np.where(signals == sig)[0]
            fraction = len(sig_idx) / n_total
            n_take = max(100, int(round(fraction * max_cells)))
            n_take = min(n_take, len(sig_idx))
            chosen = rng.choice(sig_idx, n_take, replace=False)
            keep_idx.extend(chosen)
        keep_idx = np.sort(keep_idx)
        adata_all = adata_all[keep_idx].copy()
        _logger.info(f"  After subsampling: {adata_all.n_obs:,} cells")

    X = np.asarray(adata_all.X, dtype=np.float32)
    signals = adata_all.obs["signal"].values
    unique_signals = sorted(set(signals))
    n_signals = len(unique_signals)

    # Print summary table
    _logger.info(f"\nSignal group summary:")
    _logger.info(f"  {'Signal':<35} {'Cells':>8}")
    _logger.info(f"  {'-'*35} {'-'*8}")
    for sig in unique_signals:
        n = (signals == sig).sum()
        _logger.info(f"  {sig:<35} {n:>8,}")
    _logger.info(f"  {'TOTAL':<35} {len(signals):>8,}")

    # Color map
    palette = sns.color_palette("husl", n_signals)
    signal_to_color = {s: palette[i] for i, s in enumerate(unique_signals)}
    colors = [signal_to_color[s] for s in signals]

    plots_dir = input_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- UMAP ---
    try:
        from umap import UMAP
        _logger.info(f"Computing UMAP on {len(X):,} cells...")
        t_umap = time.time()
        n_neighbors = min(30, len(X) - 1)
        coords = UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
            random_state=random_seed, verbose=True,
        ).fit_transform(X)
        _logger.info(f"  UMAP done in {time.time() - t_umap:.0f}s")

        _plot_cell_embedding(
            coords, signals, unique_signals, signal_to_color, palette,
            "UMAP", plots_dir, plt, _logger,
        )
    except ImportError:
        _logger.warning("  UMAP skipped: install umap-learn")
    except Exception as e:
        _logger.error(f"  UMAP failed: {e}")
        import traceback
        traceback.print_exc()

    # --- PHATE ---
    try:
        import phate
        _logger.info(f"Computing PHATE on {len(X):,} cells...")
        t_phate = time.time()
        knn = min(30, len(X) - 1)
        phate_op = phate.PHATE(
            n_components=2, knn=knn, decay=15, t="auto",
            n_jobs=-1, random_state=random_seed, verbose=1,
        )
        coords = phate_op.fit_transform(X)
        _logger.info(f"  PHATE done in {time.time() - t_phate:.0f}s")

        _plot_cell_embedding(
            coords, signals, unique_signals, signal_to_color, palette,
            "PHATE", plots_dir, plt, _logger,
        )
    except ImportError:
        _logger.warning("  PHATE skipped: install phate")
    except Exception as e:
        _logger.error(f"  PHATE failed: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - t_start
    return f"SUCCESS: cell embedding plots saved in {elapsed:.0f}s"


def _plot_cell_embedding(
    coords, signals, unique_signals, signal_to_color, palette,
    embed_name, plots_dir, plt, _logger,
):
    """Generate cell-level embedding plot colored by signal type."""
    import seaborn as sns
    from matplotlib.lines import Line2D

    n_signals = len(unique_signals)

    # --- Main plot: all signals colored ---
    fig, ax = plt.subplots(figsize=(16, 14))

    # Shuffle plot order so no signal is consistently on top
    rng = np.random.RandomState(0)
    order = rng.permutation(len(signals))

    for sig in unique_signals:
        mask = np.array(signals) == sig
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[signal_to_color[sig]], s=1.5, alpha=0.3, rasterized=True,
        )

    # Legend
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=signal_to_color[s],
               markersize=8, label=f"{s} ({(np.array(signals) == s).sum():,})")
        for s in unique_signals
    ]
    ax.legend(
        handles=handles, fontsize=7, loc="upper left",
        bbox_to_anchor=(1.01, 1.0), ncol=1 if n_signals <= 20 else 2,
        framealpha=0.9,
    )

    ax.set_xlabel(f"{embed_name} 1", fontsize=12)
    ax.set_ylabel(f"{embed_name} 2", fontsize=12)
    ax.set_title(
        f"Cell-Level {embed_name} — {len(signals):,} cells, {n_signals} signal groups",
        fontsize=14, fontweight="bold",
    )

    fig.tight_layout()
    fname = f"cell_{embed_name.lower()}_by_signal.png"
    fig.savefig(plots_dir / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _logger.info(f"  Saved plots/{fname}")

    # --- Grid: one panel per signal ---
    n_cols = min(6, n_signals)
    n_rows = (n_signals + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_signals == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    flat_axes = axes.flatten()
    for i, sig in enumerate(unique_signals):
        if i >= len(flat_axes):
            break
        ax = flat_axes[i]
        mask = np.array(signals) == sig
        n_sig = mask.sum()

        # Background: all cells gray
        ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=0.5, alpha=0.2, rasterized=True)
        # This signal highlighted
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[signal_to_color[sig]], s=2, alpha=0.5, rasterized=True,
        )
        ax.set_title(f"{sig} ({n_sig:,})", fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=5)
        ax.set_xlabel(f"{embed_name} 1", fontsize=7)
        ax.set_ylabel(f"{embed_name} 2", fontsize=7)

    for idx in range(n_signals, len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.suptitle(
        f"Cell-Level {embed_name} — Per Signal Group",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fname = f"cell_{embed_name.lower()}_per_signal_grid.png"
    fig.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _logger.info(f"  Saved plots/{fname}")


def main():
    parser = argparse.ArgumentParser(
        description="Cell-level cross-signal UMAP/PHATE embedding visualization"
    )
    parser.add_argument(
        "--input", "-i", type=str,
        default="/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized/downsampled",
        help="PCA optimization output dir (contains per_signal/ or per_channel/ with *_cells_sub.h5ad)",
    )
    parser.add_argument(
        "--max-cells", type=int, default=1_000_000,
        help="Maximum cells to embed (default: 1M). Proportionally subsampled if exceeded.",
    )
    parser.add_argument("--seed", type=int, default=42)

    # SLURM options
    parser.add_argument("--slurm", action="store_true", help="Submit as SLURM job")
    parser.add_argument("--slurm-memory", type=str, default="200GB")
    parser.add_argument("--slurm-time", type=int, default=120, help="Time in minutes")
    parser.add_argument("--slurm-cpus", type=int, default=32)
    parser.add_argument("--slurm-partition", type=str, default="cpu,gpu")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        jobs = [{
            "name": "cell_embedding",
            "func": run_cell_embedding,
            "kwargs": {
                "input_dir": args.input,
                "max_cells": args.max_cells,
                "random_seed": args.seed,
            },
        }]

        slurm_params = {
            "timeout_min": args.slurm_time,
            "mem": args.slurm_memory,
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": args.slurm_partition,
        }

        if not args.yes:
            print(f"Cell Embedding SLURM Job:")
            print(f"  Input:     {args.input}")
            print(f"  Max cells: {args.max_cells:,}")
            print(f"  Memory:    {args.slurm_memory}")
            print(f"  Time:      {args.slurm_time} min")
            print(f"  CPUs:      {args.slurm_cpus}")
            confirm = input("\nSubmit? [y/N] ").strip().lower()
            if confirm != "y":
                print("Cancelled.")
                return

        result = submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment="cell_embedding",
            slurm_params=slurm_params,
            log_dir="organelle_attribution",
            manifest_prefix="cell_embedding",
            wait_for_completion=True,
        )
        print(result)
    else:
        result = run_cell_embedding(
            input_dir=args.input,
            max_cells=args.max_cells,
            random_seed=args.seed,
        )
        print(result)


if __name__ == "__main__":
    main()
