"""
Volcano Enrichment Stage: Per-gene volcano plots + Enrichr GO enrichment.

For each top-variance feature:
1. Computes per-gene log2FC and p-value vs NTC
2. Generates volcano plots (up=red, down=blue)
3. Extracts significantly up/down-regulated gene lists
4. Runs Enrichr GO enrichment via speedrichr (maayanlab-bioinformatics)
5. Produces summary heatmaps of GO terms × features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from scipy.stats import ttest_ind, norm
from tqdm import tqdm
import logging
import time

from .fe_graphs_stage_base import BaseStage, StageResult
from ..plotting.fe_graphs_utils import save_figure

logger = logging.getLogger(__name__)


class VolcanoEnrichmentStage(BaseStage):
    """Per-gene volcano plots with Enrichr GO enrichment analysis."""

    STAGE_NUMBER = 6.5
    STAGE_NAME = "volcano_enrichment"

    def run(self) -> StageResult:
        """Run volcano enrichment analysis."""
        self.log_start("Volcano Enrichment Analysis")
        result = StageResult()

        # Get data from upstream or compute fresh
        df, features = self._get_data()
        if df is None or features is None:
            result.add_error("No features/df available")
            return result

        ntc_mask = self.get_ntc_mask(df)
        if ntc_mask.sum() == 0:
            result.add_error("No NTC items found")
            return result
        if (~ntc_mask).sum() == 0:
            result.add_error("No perturbed items found")
            return result

        # Step 1: Identify top-variance features
        n_top = self.analysis_config.volcano_enrichment_n_top_features
        top_features = self._identify_top_features(features, ntc_mask, n_top)
        result.add_metric("n_top_features", len(top_features))
        logger.info(f"Selected {len(top_features)} top features for analysis")

        # Step 2: Compute per-gene stats for each top feature
        per_gene_stats = self._compute_per_gene_stats(
            features, df, ntc_mask, top_features
        )
        if per_gene_stats.empty:
            result.add_error("No per-gene stats computed")
            return result

        # Save per-gene stats
        stats_dir = self.output_dir / "per_gene_stats"
        stats_dir.mkdir(exist_ok=True)
        per_gene_stats.to_csv(stats_dir / "per_gene_differential_stats.csv", index=False)
        result.add_file(stats_dir / "per_gene_differential_stats.csv")
        result.data["per_gene_stats"] = per_gene_stats

        # Step 3: Generate volcano plots
        volcano_dir = self.output_dir / "volcano_plots"
        volcano_dir.mkdir(exist_ok=True)
        self._generate_volcano_plots(per_gene_stats, top_features, volcano_dir, result)

        # Step 4: Run Enrichr GO enrichment
        # Background = all unique gene names in the dataset
        all_genes = df.loc[~ntc_mask, "gene_name"].dropna().unique().tolist()
        enrichment_dir = self.output_dir / "enrichment"
        enrichment_dir.mkdir(exist_ok=True)
        all_enrichment = self._run_enrichment_analysis(
            per_gene_stats, top_features, all_genes, enrichment_dir, result
        )

        # Step 5: Summary heatmap
        if all_enrichment is not None and not all_enrichment.empty:
            summary_dir = self.output_dir / "summary"
            summary_dir.mkdir(exist_ok=True)
            self._generate_summary_heatmap(all_enrichment, summary_dir, result)
            self._generate_enrichment_summary_table(all_enrichment, summary_dir, result)
            result.data["enrichment_results"] = all_enrichment

        self.log_complete(result)
        return result

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    def _get_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Get features and df from upstream embedding or compute fresh."""
        if "embedding" in self.upstream:
            features = self.upstream["embedding"].data.get("features")
            df = self.upstream["embedding"].data.get("df")
            if features is not None and df is not None:
                return df, features

        df = self.df.copy()
        features = self.get_features(df)
        return df, features

    # ------------------------------------------------------------------
    # Step 1: Identify top-variance features
    # ------------------------------------------------------------------

    def _identify_top_features(
        self,
        features: pd.DataFrame,
        ntc_mask: pd.Series,
        n_top: int,
    ) -> List[str]:
        """Rank features by differential signal (upstream) or raw variance (fallback)."""
        # Try upstream differential stats first
        if "differential" in self.upstream:
            diff_stats = self.upstream["differential"].data.get("differential_stats")
            if diff_stats is not None and not diff_stats.empty:
                ranked = diff_stats.sort_values("abs_zscore", ascending=False)
                top = [f for f in ranked["feature"].tolist() if f in features.columns]
                if len(top) >= 5:
                    logger.info(f"Using top features from upstream differential stats (abs_zscore)")
                    return top[:n_top]

        # Fallback: raw feature variance
        logger.info("Computing feature variance for ranking (no upstream differential)")
        variances = features.var().sort_values(ascending=False)
        return variances.head(n_top).index.tolist()

    # ------------------------------------------------------------------
    # Step 2: Per-gene differential stats
    # ------------------------------------------------------------------

    def _compute_per_gene_stats(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        ntc_mask: pd.Series,
        top_features: List[str],
    ) -> pd.DataFrame:
        """Compute per-gene log2FC and p-value for each top feature vs NTC."""
        from statsmodels.stats.multitest import fdrcorrection

        ntc_features = features.loc[ntc_mask]
        # At gene level there may be only 1 row per gene
        min_samples = {"cell": 5, "guide": 2, "gene": 1}.get(self.level, 2)

        all_stats = []

        for gene, group in tqdm(
            df[~ntc_mask].groupby("gene_name"),
            desc=f"Per-gene stats ({self.level} level)",
        ):
            if len(group) < min_samples:
                continue

            gene_features = features.loc[group.index]

            for feat in top_features:
                ntc_vals = ntc_features[feat].dropna()
                gene_vals = gene_features[feat].dropna()

                if len(gene_vals) < 1 or len(ntc_vals) < 1:
                    continue

                gene_mean = gene_vals.mean()
                ntc_mean = ntc_vals.mean()

                # Log2 fold change
                eps = 1e-10
                log2fc = np.log2((gene_mean + eps) / (ntc_mean + eps))

                # P-value: t-test when possible, z-score fallback for single samples
                if len(gene_vals) >= 2 and len(ntc_vals) >= 2:
                    _, pval = ttest_ind(gene_vals, ntc_vals, equal_var=False)
                elif len(ntc_vals) >= 2:
                    ntc_std = ntc_vals.std()
                    z = (gene_mean - ntc_mean) / (ntc_std + eps)
                    pval = 2 * norm.sf(abs(z))
                else:
                    pval = np.nan

                all_stats.append({
                    "gene_name": gene,
                    "feature": feat,
                    "gene_mean": gene_mean,
                    "ntc_mean": ntc_mean,
                    "log2_fold_change": log2fc,
                    "pvalue": pval,
                    "n_samples": len(gene_vals),
                })

        stats_df = pd.DataFrame(all_stats)
        if stats_df.empty:
            return stats_df

        # FDR correction per feature
        for feat in top_features:
            mask = stats_df["feature"] == feat
            pvals = stats_df.loc[mask, "pvalue"].fillna(1.0)
            if len(pvals) > 0:
                _, padj = fdrcorrection(pvals)
                stats_df.loc[mask, "pvalue_adj"] = padj

        stats_df["neg_log10_padj"] = -np.log10(stats_df["pvalue_adj"].clip(lower=1e-300))

        n_genes = stats_df["gene_name"].nunique()
        logger.info(f"Computed stats for {n_genes} genes × {len(top_features)} features")
        return stats_df

    # ------------------------------------------------------------------
    # Step 3: Volcano plots
    # ------------------------------------------------------------------

    def _generate_volcano_plots(
        self,
        stats_df: pd.DataFrame,
        top_features: List[str],
        output_dir: Path,
        result: StageResult,
    ) -> None:
        """Generate one volcano plot per top feature."""
        p_thresh = self.analysis_config.volcano_p_threshold
        fc_thresh = self.analysis_config.volcano_log2fc_threshold

        for feat in tqdm(top_features, desc="Generating volcano plots"):
            feat_data = stats_df[stats_df["feature"] == feat].copy()
            if feat_data.empty:
                continue

            fig, ax = plt.subplots(figsize=self.plot_config.figsize_volcano)

            # Classify significance
            sig_up = (feat_data["pvalue_adj"] < p_thresh) & (
                feat_data["log2_fold_change"] > fc_thresh
            )
            sig_down = (feat_data["pvalue_adj"] < p_thresh) & (
                feat_data["log2_fold_change"] < -fc_thresh
            )
            nonsig = ~(sig_up | sig_down)

            # Non-significant
            ax.scatter(
                feat_data.loc[nonsig, "log2_fold_change"],
                feat_data.loc[nonsig, "neg_log10_padj"],
                c="gray", alpha=0.4, s=30, label="NS",
            )
            # Significant up (red)
            if sig_up.any():
                ax.scatter(
                    feat_data.loc[sig_up, "log2_fold_change"],
                    feat_data.loc[sig_up, "neg_log10_padj"],
                    c="#d73027", alpha=0.8, s=50, label=f"Up ({sig_up.sum()})",
                )
            # Significant down (blue)
            if sig_down.any():
                ax.scatter(
                    feat_data.loc[sig_down, "log2_fold_change"],
                    feat_data.loc[sig_down, "neg_log10_padj"],
                    c="#4575b4", alpha=0.8, s=50, label=f"Down ({sig_down.sum()})",
                )

            # Label top significant genes
            sig_genes = feat_data[sig_up | sig_down].nlargest(15, "neg_log10_padj")
            for _, row in sig_genes.iterrows():
                ax.annotate(
                    row["gene_name"],
                    (row["log2_fold_change"], row["neg_log10_padj"]),
                    fontsize=7, alpha=0.8,
                    xytext=(5, 5), textcoords="offset points",
                )

            # Threshold lines
            ax.axhline(-np.log10(p_thresh), color="gray", linestyle="--", alpha=0.5)
            ax.axvline(-fc_thresh, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(fc_thresh, color="gray", linestyle="--", alpha=0.5)

            ax.set_xlabel("Log2 Fold Change (vs NTC)")
            ax.set_ylabel("-Log10 Adjusted P-value")
            ax.set_title(f"Volcano: {feat}")
            ax.legend(loc="upper right", fontsize=8)

            plt.tight_layout()
            safe_name = feat[:60].replace("/", "_").replace(" ", "_")
            path = save_figure(fig, output_dir / f"volcano_{safe_name}.png")
            result.add_file(path)

    # ------------------------------------------------------------------
    # Step 4: Enrichr GO enrichment
    # ------------------------------------------------------------------

    def _run_enrichment_analysis(
        self,
        stats_df: pd.DataFrame,
        top_features: List[str],
        background_genes: List[str],
        output_dir: Path,
        result: StageResult,
    ) -> Optional[pd.DataFrame]:
        """Run Enrichr GO enrichment for up/down gene lists per feature."""
        try:
            from maayanlab_bioinformatics.api.speedrichr import speedenrich
        except ImportError:
            logger.warning(
                "maayanlab-bioinformatics not installed — skipping GO enrichment. "
                "Install with: pip install 'maayanlab-bioinformatics@git+https://github.com/MaayanLab/maayanlab-bioinformatics.git'"
            )
            result.add_metric("enrichment_skipped", True)
            return None

        p_thresh = self.analysis_config.volcano_p_threshold
        fc_thresh = self.analysis_config.volcano_log2fc_threshold
        min_genes = self.analysis_config.volcano_enrichment_min_genes
        libraries = self.analysis_config.enrichr_libraries

        all_enrichment = []

        for feat in tqdm(top_features, desc="Running GO enrichment"):
            feat_data = stats_df[stats_df["feature"] == feat]

            # Extract up/down gene lists
            up_genes = feat_data[
                (feat_data["pvalue_adj"] < p_thresh)
                & (feat_data["log2_fold_change"] > fc_thresh)
            ]["gene_name"].tolist()

            down_genes = feat_data[
                (feat_data["pvalue_adj"] < p_thresh)
                & (feat_data["log2_fold_change"] < -fc_thresh)
            ]["gene_name"].tolist()

            for direction, gene_list in [("up", up_genes), ("down", down_genes)]:
                if len(gene_list) < min_genes:
                    continue

                try:
                    enr_df = speedenrich(
                        userlist=gene_list,
                        libraries=libraries,
                        background=background_genes,
                    )

                    if enr_df is not None and not enr_df.empty:
                        enr_df["feature"] = feat
                        enr_df["direction"] = direction
                        enr_df["n_genes_in_list"] = len(gene_list)
                        all_enrichment.append(enr_df)

                        # Save per-feature-direction CSV
                        safe_name = feat[:50].replace("/", "_").replace(" ", "_")
                        csv_path = output_dir / f"enrichment_{safe_name}_{direction}.csv"
                        enr_df.to_csv(csv_path, index=False)
                        result.add_file(csv_path)

                except Exception as e:
                    logger.warning(f"Enrichr failed for {feat} ({direction}): {e}")

                # Brief pause to avoid API rate limiting
                time.sleep(0.3)

        if all_enrichment:
            combined = pd.concat(all_enrichment, ignore_index=True)
            combined.to_csv(output_dir / "all_enrichment_results.csv", index=False)
            result.add_file(output_dir / "all_enrichment_results.csv")
            result.add_metric("n_enrichment_results", len(combined))
            n_sig = (combined["adj pvalue"] < 0.05).sum() if "adj pvalue" in combined.columns else 0
            result.add_metric("n_significant_enrichments", n_sig)
            return combined

        logger.info("No gene lists met the minimum size threshold for enrichment")
        return None

    # ------------------------------------------------------------------
    # Step 5: Summary heatmaps and tables
    # ------------------------------------------------------------------

    def _generate_summary_heatmap(
        self,
        enrichment_df: pd.DataFrame,
        output_dir: Path,
        result: StageResult,
    ) -> None:
        """Generate GO term × feature summary heatmaps (one per direction)."""
        sig = enrichment_df[enrichment_df["adj pvalue"] < 0.05].copy()
        if sig.empty or len(sig) < 2:
            return

        for direction in ["up", "down"]:
            dir_data = sig[sig["direction"] == direction]
            if dir_data.empty:
                continue

            # Top GO terms by frequency across features
            term_counts = dir_data["term"].value_counts()
            top_terms = term_counts.head(30).index.tolist()

            # Build matrix: terms × features
            features_with_data = dir_data["feature"].unique()
            if len(features_with_data) < 2 or len(top_terms) < 2:
                continue

            matrix = pd.DataFrame(0.0, index=top_terms, columns=features_with_data)
            for _, row in dir_data[dir_data["term"].isin(top_terms)].iterrows():
                neg_log10_p = -np.log10(max(row["adj pvalue"], 1e-300))
                matrix.loc[row["term"], row["feature"]] = max(
                    matrix.loc[row["term"], row["feature"]], neg_log10_p
                )

            # Truncate long labels for readability
            matrix.index = [t[:60] + "..." if len(t) > 60 else t for t in matrix.index]
            matrix.columns = [c[:30] + "..." if len(c) > 30 else c for c in matrix.columns]

            fig_h = max(10, len(top_terms) * 0.4)
            fig_w = max(12, len(features_with_data) * 0.8)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            sns.heatmap(
                matrix, cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "-log10(adj. p-value)"},
            )

            direction_label = "Up-regulated" if direction == "up" else "Down-regulated"
            ax.set_title(
                f"GO Enrichment: {direction_label} Genes\n"
                f"({self.level.capitalize()} level — top-variance features)"
            )
            ax.set_xlabel("Feature")
            ax.set_ylabel("GO Term")

            plt.tight_layout()
            path = save_figure(fig, output_dir / f"enrichment_heatmap_{direction}.png")
            result.add_file(path)

    def _generate_enrichment_summary_table(
        self,
        enrichment_df: pd.DataFrame,
        output_dir: Path,
        result: StageResult,
    ) -> None:
        """Generate a compact summary: top 3 GO terms per feature per direction."""
        summary_rows = []

        for feat in enrichment_df["feature"].unique():
            for direction in ["up", "down"]:
                feat_dir = enrichment_df[
                    (enrichment_df["feature"] == feat)
                    & (enrichment_df["direction"] == direction)
                    & (enrichment_df["adj pvalue"] < 0.05)
                ]
                if feat_dir.empty:
                    continue

                top = feat_dir.nsmallest(3, "adj pvalue")
                for _, row in top.iterrows():
                    summary_rows.append({
                        "feature": feat,
                        "direction": direction,
                        "GO_term": row["term"],
                        "library": row.get("library", ""),
                        "adj_pvalue": row["adj pvalue"],
                        "overlap": row.get("overlap", ""),
                        "combined_score": row.get("combined score", ""),
                    })

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            csv_path = output_dir / "enrichment_summary.csv"
            summary_df.to_csv(csv_path, index=False)
            result.add_file(csv_path)
            logger.info(f"Enrichment summary: {len(summary_rows)} entries across {enrichment_df['feature'].nunique()} features")
