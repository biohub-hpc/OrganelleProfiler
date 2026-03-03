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

        # Step 6: Positive control cluster analysis
        # For each cluster, find top-3 discriminating features, make volcanos + enrichment
        pc_dir = self.output_dir / "positive_controls"
        pc_dir.mkdir(exist_ok=True)
        self._run_positive_controls_analysis(
            features, df, ntc_mask, all_genes, pc_dir, result
        )

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

        # Fallback: raw feature variance (float64 to avoid overflow)
        logger.info("Computing feature variance for ranking (no upstream differential)")
        variances = features.astype(np.float64).var().sort_values(ascending=False)
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

        # Cast to float64 to avoid overflow in variance/std computations
        features = features[top_features].astype(np.float64)
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

            # Label top-20 genes by |fold-change| (regardless of significance)
            feat_data["abs_log2fc"] = feat_data["log2_fold_change"].abs()
            top_genes = feat_data.nlargest(20, "abs_log2fc")
            from matplotlib import patheffects
            for _, row in top_genes.iterrows():
                ax.annotate(
                    row["gene_name"],
                    (row["log2_fold_change"], row["neg_log10_padj"]),
                    fontsize=7, alpha=0.9, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points",
                    path_effects=[
                        patheffects.withStroke(linewidth=2, foreground="white"),
                    ],
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
        num_workers: int = 8,
    ) -> Optional[pd.DataFrame]:
        """Run Enrichr GO enrichment for up/down gene lists per feature."""
        try:
            # Load speedrichr.py directly to bypass maayanlab_bioinformatics'
            # top-level __init__.py which eagerly imports dge (requires pydeseq2).
            import importlib.util, sys
            for search_path in sys.path:
                _candidate = Path(search_path) / "maayanlab_bioinformatics" / "api" / "speedrichr.py"
                if _candidate.exists():
                    _spec = importlib.util.spec_from_file_location("_speedrichr", _candidate)
                    _mod = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    speedenrich = _mod.speedenrich
                    break
            else:
                raise ImportError("speedrichr.py not found")
        except (ImportError, ModuleNotFoundError):
            logger.warning(
                "maayanlab-bioinformatics not installed — skipping GO enrichment. "
                "Install with: pip install 'maayanlab-bioinformatics@git+https://github.com/MaayanLab/maayanlab-bioinformatics.git'"
            )
            result.add_metric("enrichment_skipped", True)
            return None

        min_genes = self.analysis_config.volcano_enrichment_min_genes
        n_top_enrich = 20  # Top-N genes per direction for enrichment
        libraries = self.analysis_config.enrichr_libraries

        # Build all enrichment jobs: (feature, direction, gene_list)
        jobs = []
        for feat in top_features:
            feat_data = stats_df[stats_df["feature"] == feat].copy()
            positive_fc = feat_data[feat_data["log2_fold_change"] > 0].nlargest(
                n_top_enrich, "log2_fold_change"
            )
            negative_fc = feat_data[feat_data["log2_fold_change"] < 0].nsmallest(
                n_top_enrich, "log2_fold_change"
            )
            for direction, gene_list in [
                ("up", positive_fc["gene_name"].tolist()),
                ("down", negative_fc["gene_name"].tolist()),
            ]:
                if len(gene_list) >= min_genes:
                    jobs.append((feat, direction, gene_list))

        # Save the gene lists used for enrichment
        gene_list_rows = []
        for feat, direction, gene_list in jobs:
            for rank, gene in enumerate(gene_list, 1):
                row = {"feature": feat, "direction": direction, "rank": rank, "gene_name": gene}
                # Add fold-change from stats
                match = stats_df[(stats_df["feature"] == feat) & (stats_df["gene_name"] == gene)]
                if not match.empty:
                    row["log2_fold_change"] = match.iloc[0]["log2_fold_change"]
                    row["pvalue_adj"] = match.iloc[0].get("pvalue_adj", np.nan)
                gene_list_rows.append(row)
        if gene_list_rows:
            gene_list_df = pd.DataFrame(gene_list_rows)
            csv_path = output_dir / "enrichment_gene_lists.csv"
            gene_list_df.to_csv(csv_path, index=False)
            result.add_file(csv_path)
            logger.info(f"Saved enrichment gene lists: {len(gene_list_rows)} entries")

        logger.info(f"Running {len(jobs)} enrichment queries with {num_workers} workers")

        def _run_single_enrichment(args):
            feat, direction, gene_list = args
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
                    return enr_df
            except Exception as e:
                logger.warning(f"Enrichr failed for {feat} ({direction}): {e}")
            return None

        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_enrichment = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_run_single_enrichment, job): job for job in jobs}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Running GO enrichment"
            ):
                enr_df = future.result()
                if enr_df is not None:
                    feat, direction, _ = futures[future]
                    all_enrichment.append(enr_df)
                    safe_name = feat[:50].replace("/", "_").replace(" ", "_")
                    csv_path = output_dir / f"enrichment_{safe_name}_{direction}.csv"
                    enr_df.to_csv(csv_path, index=False)
                    result.add_file(csv_path)

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

    # ------------------------------------------------------------------
    # Step 6: Positive control cluster volcano + enrichment
    # ------------------------------------------------------------------

    def _run_positive_controls_analysis(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        ntc_mask: pd.Series,
        background_genes: List[str],
        output_dir: Path,
        result: StageResult,
        top_features_per_cluster: int = 3,
    ) -> None:
        """
        For each positive control cluster, find top-3 discriminating features
        (by Cohen's d), then generate volcano plots showing ALL perturbations
        with cluster members highlighted, plus enrichment on the top-20 up/down.
        """
        import yaml

        # Load positive controls YAML
        pc_path = Path("/hpc/projects/icd.ops/configs/gene_clusters/chad_positive_controls_v3.yml")
        if not pc_path.exists():
            logger.warning(f"Positive controls file not found: {pc_path}")
            return

        try:
            with open(pc_path, "r") as f:
                raw_clusters = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load positive controls: {e}")
            return

        # Parse clusters (skip NTCs, require >= 2 genes)
        clusters = {}
        for cluster_id, cluster_data in raw_clusters.items():
            name = cluster_data.get("name", f"cluster_{cluster_id}")
            genes = cluster_data.get("genes", [])
            if name == "NTCs" or len(genes) < 2:
                continue
            clusters[name] = genes

        if not clusters:
            logger.warning("No valid positive control clusters found")
            return

        gene_col = "gene_name"
        logger.info(f"Running volcano+enrichment for {len(clusters)} positive control clusters "
                     f"(top {top_features_per_cluster} features each)")

        # Cast to float64 to avoid overflow in variance/std computations
        features_f64 = features.astype(np.float64)

        # Compute Cohen's d per cluster to find top discriminating features
        cluster_top_features = {}
        for cluster_name, cluster_genes in clusters.items():
            cluster_mask = df[gene_col].isin(cluster_genes)
            if cluster_mask.sum() < 1:
                continue

            cluster_feat = features_f64.loc[cluster_mask]
            other_feat = features_f64.loc[~cluster_mask]

            scores = []
            for feat in features_f64.columns:
                c_vals = cluster_feat[feat].dropna()
                o_vals = other_feat[feat].dropna()
                if len(c_vals) < 1 or len(o_vals) < 2:
                    continue
                mean_c, mean_o = c_vals.mean(), o_vals.mean()
                if len(c_vals) == 1:
                    pooled_std = o_vals.std()
                else:
                    n1, n2 = len(c_vals), len(o_vals)
                    pooled_std = np.sqrt(
                        ((n1 - 1) * c_vals.std() ** 2 + (n2 - 1) * o_vals.std() ** 2)
                        / (n1 + n2 - 2)
                    )
                if pooled_std == 0 or not np.isfinite(pooled_std):
                    continue
                d = (mean_c - mean_o) / pooled_std
                if not np.isfinite(d):
                    continue
                scores.append({"feature": feat, "cohens_d": d})

            if not scores:
                continue
            scores_df = pd.DataFrame(scores)
            scores_df["abs_d"] = scores_df["cohens_d"].abs()
            cluster_top_features[cluster_name] = (
                scores_df.nlargest(top_features_per_cluster, "abs_d")["feature"].tolist()
            )

        if not cluster_top_features:
            logger.warning("No clusters had enough data to compute discriminating features")
            return

        # Collect ALL unique features needed across all clusters, compute stats ONCE
        all_pc_features = list({f for feats in cluster_top_features.values() for f in feats})
        logger.info(f"Computing per-gene stats for {len(all_pc_features)} cluster features "
                     f"across ALL perturbations")
        all_pc_stats = self._compute_per_gene_stats(
            features, df, ntc_mask, all_pc_features
        )
        if all_pc_stats.empty:
            logger.warning("No per-gene stats computed for positive control features")
            return
        n_pc_genes = all_pc_stats["gene_name"].nunique()
        n_pc_feats = all_pc_stats["feature"].nunique()
        logger.info(f"Positive control stats: {n_pc_genes} genes × {n_pc_feats} features = {len(all_pc_stats)} rows")

        # Try to load speedenrich
        speedenrich = None
        try:
            import importlib.util, sys as _sys
            for search_path in _sys.path:
                _candidate = Path(search_path) / "maayanlab_bioinformatics" / "api" / "speedrichr.py"
                if _candidate.exists():
                    _spec = importlib.util.spec_from_file_location("_speedrichr_pc", _candidate)
                    _mod = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    speedenrich = _mod.speedenrich
                    break
        except (ImportError, ModuleNotFoundError):
            pass

        libraries = self.analysis_config.enrichr_libraries
        n_top_enrich = 20
        min_genes = self.analysis_config.volcano_enrichment_min_genes
        p_thresh = self.analysis_config.volcano_p_threshold
        fc_thresh = self.analysis_config.volcano_log2fc_threshold
        from matplotlib import patheffects

        for cluster_name, top_feats in tqdm(
            cluster_top_features.items(), desc="Positive control volcanos"
        ):
            safe_cluster = cluster_name[:40].replace("/", "_").replace(" ", "_")
            cluster_dir = output_dir / safe_cluster
            cluster_dir.mkdir(exist_ok=True)
            cluster_genes_set = set(clusters[cluster_name])

            # Generate volcano plots: ALL perturbations, cluster highlighted
            for feat in top_feats:
                feat_data = all_pc_stats[all_pc_stats["feature"] == feat].copy()
                if feat_data.empty:
                    continue

                fig, ax = plt.subplots(figsize=self.plot_config.figsize_volcano)

                is_cluster = feat_data["gene_name"].isin(cluster_genes_set)
                n_total = len(feat_data)
                n_cluster = is_cluster.sum()
                logger.info(f"  {cluster_name} / {feat}: {n_total} total genes, {n_cluster} in cluster")

                # All perturbations (gray background)
                ax.scatter(
                    feat_data.loc[~is_cluster, "log2_fold_change"],
                    feat_data.loc[~is_cluster, "neg_log10_padj"],
                    c="#aaaaaa", alpha=0.5, s=30,
                    label=f"Other genes ({n_total - n_cluster})", zorder=1,
                )
                # Cluster genes highlighted (red, large)
                if is_cluster.any():
                    ax.scatter(
                        feat_data.loc[is_cluster, "log2_fold_change"],
                        feat_data.loc[is_cluster, "neg_log10_padj"],
                        c="#d62728", alpha=0.95, s=120, edgecolors="black",
                        linewidths=0.8, label=f"{cluster_name} ({n_cluster})", zorder=3,
                    )
                    for _, row in feat_data.loc[is_cluster].iterrows():
                        ax.annotate(
                            row["gene_name"],
                            (row["log2_fold_change"], row["neg_log10_padj"]),
                            fontsize=8, fontweight="bold", color="#d62728",
                            xytext=(6, 6), textcoords="offset points",
                            path_effects=[
                                patheffects.withStroke(linewidth=2.5, foreground="white"),
                            ],
                        )

                # Label top-20 non-cluster genes by |FC|
                non_cluster = feat_data.loc[~is_cluster].copy()
                non_cluster["abs_log2fc"] = non_cluster["log2_fold_change"].abs()
                for _, row in non_cluster.nlargest(20, "abs_log2fc").iterrows():
                    ax.annotate(
                        row["gene_name"],
                        (row["log2_fold_change"], row["neg_log10_padj"]),
                        fontsize=6, alpha=0.7,
                        xytext=(4, 4), textcoords="offset points",
                        path_effects=[
                            patheffects.withStroke(linewidth=1.5, foreground="white"),
                        ],
                    )

                # Threshold lines
                ax.axhline(-np.log10(p_thresh), color="gray", linestyle="--", alpha=0.5)
                ax.axvline(-fc_thresh, color="gray", linestyle="--", alpha=0.5)
                ax.axvline(fc_thresh, color="gray", linestyle="--", alpha=0.5)

                ax.set_xlabel("Log2 Fold Change (vs NTC)")
                ax.set_ylabel("-Log10 Adjusted P-value")
                ax.set_title(
                    f"Volcano: {feat}\n"
                    f"Positive Control: {cluster_name}",
                    fontsize=11,
                )
                ax.legend(loc="upper right", fontsize=8)
                plt.tight_layout()
                safe_feat = feat[:50].replace("/", "_").replace(" ", "_")
                path = save_figure(fig, cluster_dir / f"volcano_{safe_feat}.png")
                result.add_file(path)

            # Enrichment on top-20 up/down for each feature
            if speedenrich is not None:
                jobs = []
                for feat in top_feats:
                    feat_data = all_pc_stats[all_pc_stats["feature"] == feat]
                    if feat_data.empty:
                        continue
                    pos_fc = feat_data[feat_data["log2_fold_change"] > 0].nlargest(
                        n_top_enrich, "log2_fold_change"
                    )
                    neg_fc = feat_data[feat_data["log2_fold_change"] < 0].nsmallest(
                        n_top_enrich, "log2_fold_change"
                    )
                    for direction, gene_list in [
                        ("up", pos_fc["gene_name"].tolist()),
                        ("down", neg_fc["gene_name"].tolist()),
                    ]:
                        if len(gene_list) >= min_genes:
                            jobs.append((feat, direction, gene_list))

                # Save gene lists for this cluster
                if jobs:
                    gl_rows = []
                    for feat, direction, gene_list in jobs:
                        for rank, gene in enumerate(gene_list, 1):
                            row = {"feature": feat, "direction": direction, "rank": rank, "gene_name": gene}
                            match = all_pc_stats[
                                (all_pc_stats["feature"] == feat) & (all_pc_stats["gene_name"] == gene)
                            ]
                            if not match.empty:
                                row["log2_fold_change"] = match.iloc[0]["log2_fold_change"]
                                row["pvalue_adj"] = match.iloc[0].get("pvalue_adj", np.nan)
                            row["in_cluster"] = gene in cluster_genes_set
                            gl_rows.append(row)
                    if gl_rows:
                        pd.DataFrame(gl_rows).to_csv(
                            cluster_dir / "enrichment_gene_lists.csv", index=False
                        )
                        result.add_file(cluster_dir / "enrichment_gene_lists.csv")

                if jobs:
                    from concurrent.futures import ThreadPoolExecutor, as_completed

                    def _enrich_single(args):
                        feat, direction, gene_list = args
                        try:
                            enr_df = speedenrich(
                                userlist=gene_list, libraries=libraries,
                                background=background_genes,
                            )
                            if enr_df is not None and not enr_df.empty:
                                enr_df["feature"] = feat
                                enr_df["direction"] = direction
                                enr_df["cluster"] = cluster_name
                                enr_df["n_genes_in_list"] = len(gene_list)
                                return enr_df
                        except Exception as e:
                            logger.warning(f"Enrichr failed for {cluster_name}/{feat} ({direction}): {e}")
                        return None

                    enrich_results = []
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        futures = {executor.submit(_enrich_single, j): j for j in jobs}
                        for future in as_completed(futures):
                            enr_df = future.result()
                            if enr_df is not None:
                                enrich_results.append(enr_df)
                                feat, direction, _ = futures[future]
                                safe_feat = feat[:50].replace("/", "_").replace(" ", "_")
                                csv_path = cluster_dir / f"enrichment_{safe_feat}_{direction}.csv"
                                enr_df.to_csv(csv_path, index=False)
                                result.add_file(csv_path)

                    if enrich_results:
                        combined = pd.concat(enrich_results, ignore_index=True)
                        combined.to_csv(cluster_dir / "all_enrichment.csv", index=False)
                        result.add_file(cluster_dir / "all_enrichment.csv")

        logger.info(f"Positive control analysis complete: {len(cluster_top_features)} clusters")
