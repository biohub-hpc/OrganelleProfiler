"""
Differential Stage: NTC comparison and statistical testing.

Performs:
- Z-score comparison vs NTC
- Fold change analysis
- Statistical tests (t-test)
- Volcano plots
- Feature importance
- Organelle contribution analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List
from scipy.stats import ttest_ind
from tqdm import tqdm
import logging

from .fe_graphs_stage_base import BaseStage, StageResult
from ..plotting.fe_graphs_utils import save_figure, create_broken_axis_bar

logger = logging.getLogger(__name__)


class DifferentialStage(BaseStage):
    """Differential analysis stage."""
    
    STAGE_NUMBER = 6
    STAGE_NAME = "differential"
    
    def run(self) -> StageResult:
        """Run differential analysis."""
        self.log_start()
        result = StageResult()
        
        # Get data from upstream or use current data
        features = None
        df = None
        
        if "embedding" in self.upstream:
            # Get from upstream
            features = self.upstream["embedding"].data.get("features")
            df = self.upstream["embedding"].data.get("df")
        else:
            # Embedding stage was skipped - use current level data
            logger.info("Embedding stage not in upstream, using current level data...")
            df = self.df.copy()
            features = self.get_features(df)
        
        if features is None or df is None:
            result.add_error("No features/df available")
            return result
        
        # Get NTC mask
        ntc_mask = self.get_ntc_mask(df)
        n_ntc = ntc_mask.sum()
        n_pert = (~ntc_mask).sum()
        
        result.add_metric("n_ntc", n_ntc)
        result.add_metric("n_perturbed", n_pert)
        
        # Debug NTC detection
        if n_ntc == 0 and "gene_name" in df.columns:
            unique_genes = df["gene_name"].value_counts().head(10)
            logger.warning(f"No NTC items found. Sample of gene names: {unique_genes.index.tolist()}")
            logger.warning(f"NTC patterns used: {['ntc', 'non-targeting', '^0$']}")
        
        if n_ntc == 0:
            result.add_error("No NTC items found")
            return result
        
        if n_pert == 0:
            result.add_error("No perturbed items found")
            return result
        
        # NTC comparison
        ntc_dir = self.output_dir / "ntc_comparison"
        ntc_dir.mkdir(exist_ok=True)
        diff_stats = self._run_ntc_comparison(features, df, ntc_mask, ntc_dir, result)
        result.data["differential_stats"] = diff_stats
        
        # Volcano plots
        volcano_dir = self.output_dir / "volcano_plots"
        volcano_dir.mkdir(exist_ok=True)
        self._generate_volcano_plots(features, df, ntc_mask, volcano_dir, result)
        
        # Organelle contribution
        organelle_dir = self.output_dir / "organelle_contribution"
        organelle_dir.mkdir(exist_ok=True)
        self._analyze_organelle_contribution(features, df, ntc_mask, organelle_dir, result)
        
        # Gene highlights (for cell level with clustering)
        if self.level == "cell" and "clustering" in self.upstream:
            highlights_dir = self.output_dir / "gene_highlights"
            highlights_dir.mkdir(exist_ok=True)
            self._generate_gene_highlights(df, highlights_dir, result)
        
        self.log_complete(result)
        return result
    
    def _run_ntc_comparison(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        ntc_mask: pd.Series,
        output_dir: Path,
        result: StageResult,
    ) -> pd.DataFrame:
        """Run NTC comparison analysis."""
        logger.info("Running NTC comparison...")
        
        ntc_features = features.loc[ntc_mask]
        pert_features = features.loc[~ntc_mask]
        
        # Calculate NTC statistics
        ntc_means = ntc_features.mean()
        ntc_stds = ntc_features.std()
        
        # Filter zero-variance features
        valid_mask = ntc_stds > 1e-6
        feature_cols = ntc_stds[valid_mask].index.tolist()
        
        logger.info(f"Using {len(feature_cols)} features (removed {(~valid_mask).sum()} zero-variance)")
        
        # Calculate differential statistics
        diff_data = []
        skipped_reasons = {"too_few_ntc": 0, "too_few_pert": 0, "both_too_few": 0}
        
        # At gene/guide level, we may have only 1 NTC (all NTC cells aggregated to 1 gene)
        # In this case, we can still compute effect sizes but not statistical tests
        allow_single_sample = (len(ntc_features) == 1)
        if allow_single_sample:
            logger.info(f"Only 1 NTC sample - will compute effect sizes without p-values")
        
        for col in tqdm(feature_cols, desc="Computing differential stats"):
            ntc_vals = ntc_features[col].dropna()
            pert_vals = pert_features[col].dropna()
            
            # Skip if both have too few samples
            if len(ntc_vals) < 1 and len(pert_vals) < 1:
                skipped_reasons["both_too_few"] += 1
                continue
            elif len(ntc_vals) < 1:
                skipped_reasons["too_few_ntc"] += 1
                continue
            elif len(pert_vals) < 1:
                skipped_reasons["too_few_pert"] += 1
                continue
            
            # Z-score (can compute with single NTC sample)
            ntc_mean = ntc_vals.mean()
            ntc_std = ntc_vals.std() if len(ntc_vals) > 1 else 1.0  # Use 1.0 as fallback for single sample
            zscore = (pert_vals.mean() - ntc_mean) / (ntc_std + 1e-10)
            
            # Fold change
            fold_change = pert_vals.mean() / (ntc_mean if abs(ntc_mean) > 1e-10 else 1e-10)
            
            # T-test (only if we have enough samples)
            if len(ntc_vals) >= 2 and len(pert_vals) >= 2:
                _, pvalue = ttest_ind(pert_vals, ntc_vals, equal_var=False)
            else:
                pvalue = np.nan  # Can't compute p-value with single sample
            
            diff_data.append({
                "feature": col,
                "ntc_mean": ntc_mean,
                "ntc_std": ntc_std,
                "pert_mean": pert_vals.mean(),
                "pert_std": pert_vals.std(),
                "zscore": zscore,
                "fold_change": fold_change,
                "log2_fold_change": np.log2(fold_change) if fold_change > 0 else np.nan,
                "pvalue": pvalue,
            })
        
        # Log skip reasons
        total_skipped = sum(skipped_reasons.values())
        if total_skipped > 0:
            logger.warning(f"Skipped {total_skipped} features:")
            for reason, count in skipped_reasons.items():
                if count > 0:
                    logger.warning(f"  {reason}: {count}")
        
        diff_df = pd.DataFrame(diff_data)
        
        if diff_df.empty:
            logger.error(f"No differential statistics computed. Total features attempted: {len(feature_cols)}")
            logger.error(f"NTC samples: {len(ntc_features)}, Perturbed samples: {len(pert_features)}")
            result.add_error("No differential statistics computed - all features skipped")
            return diff_df
        
        # FDR correction
        from statsmodels.stats.multitest import fdrcorrection
        diff_df["pvalue_adj"] = fdrcorrection(diff_df["pvalue"].fillna(1))[1]
        diff_df["neg_log10_pvalue"] = -np.log10(diff_df["pvalue"].clip(lower=1e-300))
        
        # Sort by absolute z-score
        diff_df["abs_zscore"] = diff_df["zscore"].abs()
        diff_df = diff_df.sort_values("abs_zscore", ascending=False)
        
        # Save full results
        diff_df.to_csv(output_dir / "differential_stats.csv", index=False)
        result.add_file(output_dir / "differential_stats.csv")
        
        # Top features by z-score
        self._plot_top_features(diff_df, "zscore", output_dir, result)
        self._plot_top_features(diff_df, "log2_fold_change", output_dir, result)
        
        # Summary metrics
        sig_features = (diff_df["pvalue_adj"] < 0.05).sum()
        result.add_metric("n_significant_features", sig_features)
        result.add_metric("mean_abs_zscore", diff_df["abs_zscore"].mean())
        
        return diff_df
    
    def _plot_top_features(
        self,
        diff_df: pd.DataFrame,
        metric: str,
        output_dir: Path,
        result: StageResult,
        n_top: int = 20,
    ) -> None:
        """Plot top features by given metric."""
        # Get top positive and negative
        sorted_df = diff_df.sort_values(metric)
        
        top_neg = sorted_df.head(n_top // 2)
        top_pos = sorted_df.tail(n_top // 2)
        top_df = pd.concat([top_neg, top_pos]).sort_values(metric)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ["#d73027" if x < 0 else "#1a9850" for x in top_df[metric]]
        
        ax.barh(top_df["feature"], top_df[metric], color=colors)
        ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_ylabel("Feature")
        ax.set_title(f"Top {n_top} Features by {metric.replace('_', ' ').title()}")
        
        plt.tight_layout()
        path = save_figure(fig, output_dir / f"top_features_{metric}.png")
        result.add_file(path)
    
    def _generate_volcano_plots(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        ntc_mask: pd.Series,
        output_dir: Path,
        result: StageResult,
    ) -> None:
        """Generate volcano plots per gene."""
        if "gene_name" not in df.columns:
            return
        
        logger.info("Generating volcano plots...")
        
        ntc_features = features.loc[ntc_mask]
        ntc_means = ntc_features.mean()
        ntc_stds = ntc_features.std()
        
        # Filter low variance
        valid_mask = ntc_stds > 1e-6
        feature_cols = ntc_stds[valid_mask].index.tolist()[:50]  # Top 50 for visualization
        
        # Per-gene statistics
        gene_stats = []
        
        for gene, group in df[~ntc_mask].groupby("gene_name"):
            if len(group) < 5:
                continue
            
            gene_features = features.loc[group.index, feature_cols]
            
            gene_data = {"gene_name": gene}
            for col in feature_cols:
                ntc_vals = ntc_features[col].dropna()
                gene_vals = gene_features[col].dropna()
                
                if len(ntc_vals) < 2 or len(gene_vals) < 2:
                    continue
                
                # Mean comparison
                gene_data[f"{col}_log2fc"] = np.log2(
                    (gene_vals.mean() + 1e-10) / (ntc_vals.mean() + 1e-10)
                )
                
                # T-test
                _, pval = ttest_ind(gene_vals, ntc_vals, equal_var=False)
                gene_data[f"{col}_pval"] = pval
            
            gene_stats.append(gene_data)
        
        if not gene_stats:
            return
        
        # Plot volcano for top features
        top_features = feature_cols[:5]
        
        for feat in top_features:
            log2fc_col = f"{feat}_log2fc"
            pval_col = f"{feat}_pval"
            
            plot_data = []
            for g in gene_stats:
                if log2fc_col in g and pval_col in g:
                    plot_data.append({
                        "gene": g["gene_name"],
                        "log2_fold_change": g[log2fc_col],
                        "neg_log10_pvalue": -np.log10(g[pval_col]) if g[pval_col] > 0 else 10,
                    })
            
            if not plot_data:
                continue
            
            plot_df = pd.DataFrame(plot_data)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Color by significance
            sig_mask = (plot_df["neg_log10_pvalue"] > -np.log10(0.05)) & (plot_df["log2_fold_change"].abs() > 0.5)
            
            ax.scatter(
                plot_df.loc[~sig_mask, "log2_fold_change"],
                plot_df.loc[~sig_mask, "neg_log10_pvalue"],
                c="gray", alpha=0.5, s=30,
            )
            ax.scatter(
                plot_df.loc[sig_mask, "log2_fold_change"],
                plot_df.loc[sig_mask, "neg_log10_pvalue"],
                c="red", alpha=0.8, s=50,
            )
            
            # Add labels for significant genes
            for _, row in plot_df[sig_mask].iterrows():
                ax.annotate(
                    row["gene"],
                    (row["log2_fold_change"], row["neg_log10_pvalue"]),
                    fontsize=8, alpha=0.8,
                )
            
            ax.axhline(-np.log10(0.05), color="gray", linestyle="--", alpha=0.5)
            ax.axvline(-0.5, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
            
            ax.set_xlabel("Log2 Fold Change")
            ax.set_ylabel("-Log10 P-value")
            ax.set_title(f"Volcano Plot: {feat}")
            
            path = save_figure(fig, output_dir / f"volcano_{feat[:50]}.png")
            result.add_file(path)
    
    def _analyze_organelle_contribution(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        ntc_mask: pd.Series,
        output_dir: Path,
        result: StageResult,
    ) -> None:
        """Analyze organelle contribution to phenotypes."""
        logger.info("Analyzing organelle contribution...")
        
        organelle_features = self.group_features_by_organelle(features.columns.tolist())
        
        if len(organelle_features) < 2:
            return
        
        ntc_features = features.loc[ntc_mask]
        
        # Calculate contribution per organelle per gene
        if "gene_name" not in df.columns:
            return
        
        contribution_data = []
        
        for gene, group in df[~ntc_mask].groupby("gene_name"):
            if len(group) < 5:
                continue
            
            gene_features = features.loc[group.index]
            
            row = {"gene_name": gene, "n_cells": len(group)}
            
            for organelle, cols in organelle_features.items():
                # Calculate z-scores for this organelle
                org_ntc = ntc_features[cols]
                org_gene = gene_features[cols]
                
                ntc_mean = org_ntc.mean()
                ntc_std = org_ntc.std()
                
                # Average absolute z-score
                zscores = ((org_gene.mean() - ntc_mean) / (ntc_std + 1e-10)).abs()
                row[f"{organelle}_zscore"] = zscores.mean()
            
            contribution_data.append(row)
        
        if not contribution_data:
            return
        
        contrib_df = pd.DataFrame(contribution_data)
        contrib_df.to_csv(output_dir / "contribution_scores.csv", index=False)
        result.add_file(output_dir / "contribution_scores.csv")
        
        # Heatmap
        organelle_cols = [c for c in contrib_df.columns if c.endswith("_zscore")]
        if organelle_cols:
            heatmap_df = contrib_df.set_index("gene_name")[organelle_cols]
            heatmap_df.columns = [c.replace("_zscore", "") for c in heatmap_df.columns]
            
            # Top genes by total zscore
            heatmap_df["total"] = heatmap_df.sum(axis=1)
            heatmap_df = heatmap_df.nlargest(30, "total").drop(columns=["total"])
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(heatmap_df, cmap="viridis", ax=ax)
            ax.set_title("Organelle Contribution by Gene (Top 30)")
            ax.set_xlabel("Organelle")
            ax.set_ylabel("Gene")
            
            plt.tight_layout()
            path = save_figure(fig, output_dir / "contribution_heatmap.png")
            result.add_file(path)
    
    def _generate_gene_highlights(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        result: StageResult,
    ) -> None:
        """Generate UMAP highlights for top genes."""
        if "clustering" not in self.upstream:
            return
        
        plot_df = self.upstream["clustering"].data.get("plot_df")
        if plot_df is None:
            return
        
        enrichment = self.upstream["clustering"].data.get("enrichment")
        if enrichment is None or enrichment.empty:
            return
        
        # Top enriched genes
        top_genes = enrichment.groupby("gene_name")["p_adj"].min().nsmallest(10).index
        
        for gene in top_genes:
            fig, ax = plt.subplots(figsize=self.plot_config.figsize_umap)
            
            gene_mask = plot_df["gene_name"] == gene
            
            ax.scatter(
                plot_df.loc[~gene_mask, "umap_1"],
                plot_df.loc[~gene_mask, "umap_2"],
                c="lightgray", s=self.plot_config.point_size * 0.5,
                alpha=0.3, rasterized=True,
            )
            ax.scatter(
                plot_df.loc[gene_mask, "umap_1"],
                plot_df.loc[gene_mask, "umap_2"],
                c="red", s=self.plot_config.point_size,
                alpha=0.8, label=gene, rasterized=True,
            )
            
            ax.set_title(f"Gene Highlight: {gene}")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.legend()
            
            path = save_figure(fig, output_dir / f"umap_highlight_{gene}.png")
            result.add_file(path)