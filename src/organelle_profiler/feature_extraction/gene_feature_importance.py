"""
Identify distinguishing features for a given geneKO using Cohen's d effect size.

Loads a gene-level AnnData (.h5ad) file and computes per-feature Cohen's d
between the target gene and all other genes. Outputs a ranked table of features.

Usage
-----
# Basic usage
python -m organelle_profiler.feature_extraction.gene_feature_importance --h5ad /path/to/gene_features.h5ad --gene ARGLU1

# Show top 30 features, save CSV
python -m organelle_profiler.feature_extraction.gene_feature_importance \
    --h5ad /path/to/gene_features.h5ad --gene ARGLU1 --top 30 --output ARGLU1_features.csv

# Generate bar plot
python -m organelle_profiler.feature_extraction.gene_feature_importance \
    --h5ad /path/to/gene_features.h5ad --gene ARGLU1 --plot
"""

import argparse
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path


def compute_gene_feature_importance(
    adata: ad.AnnData,
    gene: str,
    gene_col: str = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Compute Cohen's d effect size for each feature, comparing one gene vs all others.

    Parameters
    ----------
    adata : AnnData
        Gene-level AnnData with features in .X and gene identifiers in .obs
    gene : str
        Target gene name (e.g. "ARGLU1")
    gene_col : str, optional
        Column in obs containing gene names. Auto-detected if None.
    top_n : int
        Number of top features to return (0 = all)

    Returns
    -------
    pd.DataFrame
        Ranked features with columns: feature, cohens_d, abs_cohens_d,
        mean_gene, mean_other, fold_change
    """
    # Auto-detect gene column
    if gene_col is None:
        for candidate in ("gene_name", "gene", "geneKO"):
            if candidate in adata.obs.columns:
                gene_col = candidate
                break
        if gene_col is None:
            # Assume gene is in the index
            gene_col = "_index"
            adata.obs["_index"] = adata.obs.index

    if gene not in adata.obs[gene_col].values:
        available = sorted(adata.obs[gene_col].unique())
        raise ValueError(
            f"Gene '{gene}' not found in column '{gene_col}'. "
            f"Available ({len(available)}): {available[:20]}..."
        )

    # Extract feature matrix
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    features = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)

    # Split into target vs rest
    mask = adata.obs[gene_col].values == gene
    gene_features = features.loc[mask]
    other_features = features.loc[~mask]

    # Compute Cohen's d for each feature (same pattern as fe_graphs_positive_controls_stage.py)
    scores = []
    for feat in features.columns:
        gene_vals = gene_features[feat].dropna()
        other_vals = other_features[feat].dropna()

        if len(gene_vals) < 1 or len(other_vals) < 2:
            continue

        mean_gene = gene_vals.mean()
        mean_other = other_vals.mean()

        if len(gene_vals) == 1:
            pooled_std = other_vals.std()
        else:
            n1, n2 = len(gene_vals), len(other_vals)
            pooled_std = np.sqrt(
                ((n1 - 1) * gene_vals.std() ** 2 + (n2 - 1) * other_vals.std() ** 2)
                / (n1 + n2 - 2)
            )

        if pooled_std == 0:
            continue

        d = (mean_gene - mean_other) / pooled_std

        scores.append({
            "feature": feat,
            "cohens_d": d,
            "abs_cohens_d": abs(d),
            "mean_gene": mean_gene,
            "mean_other": mean_other,
            "fold_change": mean_gene / mean_other if mean_other != 0 else np.nan,
        })

    if not scores:
        raise RuntimeError(f"No valid features computed for gene '{gene}'")

    df = pd.DataFrame(scores).sort_values("abs_cohens_d", ascending=False)

    if top_n > 0:
        df = df.head(top_n)

    return df.reset_index(drop=True)


def plot_feature_importance(df: pd.DataFrame, gene: str, output_path: Path = None):
    """Generate a horizontal bar plot of top distinguishing features."""
    import matplotlib.pyplot as plt

    top = df.head(30)
    fig, ax = plt.subplots(figsize=(12, max(6, len(top) * 0.3)))

    colors = ["#d62728" if d > 0 else "#1f77b4" for d in top["cohens_d"]]
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top["cohens_d"], color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's d (effect size)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top Distinguishing Features for {gene}", fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Identify distinguishing features for a geneKO using Cohen's d"
    )
    parser.add_argument("--h5ad", required=True, help="Path to gene-level .h5ad file")
    parser.add_argument("--gene", required=True, help="Target gene name (e.g. ARGLU1)")
    parser.add_argument("--gene-col", default=None, help="Column name for gene IDs (auto-detected)")
    parser.add_argument("--top", type=int, default=20, help="Number of top features (0=all)")
    parser.add_argument("--output", default=None, help="Save results CSV to this path")
    parser.add_argument("--plot", action="store_true", help="Generate bar plot")
    parser.add_argument("--plot-output", default=None, help="Save plot to this path (default: show)")
    args = parser.parse_args()

    print(f"Loading {args.h5ad}...")
    adata = ad.read_h5ad(args.h5ad)
    print(f"  Shape: {adata.shape[0]} genes × {adata.shape[1]} features")

    df = compute_gene_feature_importance(
        adata, gene=args.gene, gene_col=args.gene_col, top_n=args.top
    )

    print(f"\nTop {len(df)} distinguishing features for {args.gene}:")
    print(df.to_string(index=False))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")

    if args.plot:
        plot_path = Path(args.plot_output) if args.plot_output else None
        plot_feature_importance(df, args.gene, plot_path)


if __name__ == "__main__":
    main()
