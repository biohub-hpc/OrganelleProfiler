"""
CP vs Non-CP Comparison Stage

This stage compares the discriminative power and information content of 
Cell Painting organelles vs non-Cell Painting organelles (phase2d, focus3d, etc.).

Only runs when --just-cp mode is enabled to provide context on what information
is retained vs lost when excluding non-CP features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_ind
from scipy.spatial.distance import pdist

from .fe_graphs_stage_base import BaseStage, StageResult
from ..plotting.fe_graphs_utils import save_figure

logger = logging.getLogger(__name__)


class CPComparisonStage(BaseStage):
    """
    Compare Cell Painting vs Non-Cell Painting organelle features.
    
    Analyzes:
    - Variance and information content
    - KO discrimination power  
    - Positive control separation
    - Feature redundancy
    """
    
    STAGE_NUMBER = 8
    STAGE_NAME = "cp_comparison"
    
    def run(self) -> StageResult:
        """Run CP vs non-CP comparison analysis."""
        self.log_start("Comparing Cell Painting vs Non-CP organelles")
        result = StageResult()
        
        # This stage only runs at cell level and only makes sense if we have both CP and non-CP data
        # We need to load the FULL dataset (not filtered) to compare
        if self.level != "cell":
            logger.info(f"  Skipping: CP comparison only runs at cell level")
            result.success = True
            return result
        
        # Load FULL unfiltered data
        try:
            full_df, full_features, full_organelle_groups = self._load_full_unfiltered_data()
        except Exception as e:
            result.add_error(f"Could not load full unfiltered data: {e}")
            return result
        
        # Separate CP vs non-CP organelles
        cp_groups, noncp_groups = self._separate_cp_and_noncp(full_organelle_groups)
        
        if not noncp_groups:
            logger.warning("  No non-CP organelle groups found, skipping comparison")
            result.success = True
            return result
        
        logger.info(f"  CP organelles: {len(cp_groups)} groups")
        logger.info(f"  Non-CP organelles: {len(noncp_groups)} groups")
        
        # Extract feature subsets
        cp_features = self._extract_organelle_features(full_features, cp_groups)
        noncp_features = self._extract_organelle_features(full_features, noncp_groups)
        
        logger.info(f"  CP features: {cp_features.shape[1]}")
        logger.info(f"  Non-CP features: {noncp_features.shape[1]}")
        
        # Run comparisons
        self._compare_variance(cp_features, noncp_features, cp_groups, noncp_groups, result)
        self._compare_dimensionality(cp_features, noncp_features, result)
        self._compare_ko_discrimination(cp_features, noncp_features, full_df, result)
        
        # Positive control comparison (if available)
        if "positive_controls" in self.upstream and self.upstream["positive_controls"].data.get("clusters"):
            clusters = self.upstream["positive_controls"].data["clusters"]
            gene_col = self._get_gene_column()
            self._compare_positive_control_separation(
                cp_features, noncp_features, full_df, gene_col, clusters, result
            )
        
        # Feature correlation analysis
        self._analyze_feature_redundancy(cp_features, noncp_features, result)
        
        # Per-organelle discrimination power
        self._rank_organelles_by_discrimination(
            full_features, full_df, full_organelle_groups, cp_groups, noncp_groups, result
        )
        
        self.log_complete(result)
        return result
    
    def _load_full_unfiltered_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
        """
        Load the full unfiltered dataset (including non-CP organelles).
        This is needed because when --just-cp is used, data_context already has filtered data.
        """
        # Need to reload from the AnnData without filtering
        cell_adata = self.data.adata["cell"]
        
        # Reconstruct full organelle groups from adata.var
        from ..core.fe_graphs_data_loader import discover_organelle_groups_from_adata
        full_organelle_groups = discover_organelle_groups_from_adata(cell_adata)
        
        # Get full feature matrix from current df
        # The df should still have all features, just organelle_groups was filtered
        full_features = self.get_features(self.df)
        
        return self.df, full_features, full_organelle_groups
    
    def _separate_cp_and_noncp(
        self, organelle_groups: Dict[str, List[str]]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Separate organelle groups into CP and non-CP."""
        cp_patterns = ["cp1_", "cp2_", "CP1_", "CP2_"]
        noncp_patterns = ["phase2d_", "focus3d_", "nucleoli_phase2d", "nucleoli_focus3d"]
        exclude_exact = ["nuclei", "nuclear_seg", "cell_seg"]
        
        cp_groups = {}
        noncp_groups = {}
        
        for org_name, features in organelle_groups.items():
            is_cp = any(org_name.startswith(p) for p in cp_patterns)
            is_noncp = any(p in org_name for p in noncp_patterns) or org_name in exclude_exact
            
            # Special case: cell outlines are CP
            if org_name in ["cell", "cp_cell"]:
                is_cp = True
                is_noncp = False
            
            if is_cp and not is_noncp:
                cp_groups[org_name] = features
            elif is_noncp and not is_cp:
                noncp_groups[org_name] = features
        
        return cp_groups, noncp_groups
    
    def _extract_organelle_features(
        self, features: pd.DataFrame, organelle_groups: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Extract features for a set of organelle groups."""
        all_cols = []
        for group_features in organelle_groups.values():
            all_cols.extend([c for c in group_features if c in features.columns])
        
        return features[all_cols].copy()
    
    def _compare_variance(
        self,
        cp_features: pd.DataFrame,
        noncp_features: pd.DataFrame,
        cp_groups: Dict[str, List[str]],
        noncp_groups: Dict[str, List[str]],
        result: StageResult,
    ) -> None:
        """Compare variance and information content."""
        logger.info("  Comparing variance and information content...")
        
        variance_dir = self.output_dir / "variance_comparison"
        variance_dir.mkdir(exist_ok=True)
        
        # Overall variance
        cp_var = cp_features.var()
        noncp_var = noncp_features.var()
        
        metrics = {
            "group": ["Cell Painting", "Non-CP"],
            "n_features": [len(cp_var), len(noncp_var)],
            "total_variance": [cp_var.sum(), noncp_var.sum()],
            "mean_variance": [cp_var.mean(), noncp_var.mean()],
            "median_variance": [cp_var.median(), noncp_var.median()],
            "pct_high_var": [
                (cp_var > cp_var.quantile(0.75)).sum() / len(cp_var) * 100,
                (noncp_var > noncp_var.quantile(0.75)).sum() / len(noncp_var) * 100,
            ],
        }
        
        metrics_df = pd.DataFrame(metrics)
        csv_path = variance_dir / "variance_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        result.add_file(csv_path)
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Variance distributions
        ax = axes[0, 0]
        ax.hist(np.log10(cp_var + 1e-10), bins=50, alpha=0.6, label="CP", color="steelblue")
        ax.hist(np.log10(noncp_var + 1e-10), bins=50, alpha=0.6, label="Non-CP", color="coral")
        ax.set_xlabel("log10(Variance)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Feature Variance Distribution", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: Total variance per group
        ax = axes[0, 1]
        ax.bar(metrics_df["group"], metrics_df["total_variance"], color=["steelblue", "coral"])
        ax.set_ylabel("Total Variance", fontsize=11)
        ax.set_title("Total Information Content", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        
        # Plot 3: Mean variance comparison
        ax = axes[1, 0]
        x = np.arange(len(metrics_df))
        width = 0.35
        ax.bar(x - width/2, metrics_df["mean_variance"], width, label="Mean", color="steelblue", alpha=0.8)
        ax.bar(x + width/2, metrics_df["median_variance"], width, label="Median", color="coral", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df["group"])
        ax.set_ylabel("Variance", fontsize=11)
        ax.set_title("Mean & Median Variance", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        
        # Plot 4: Per-organelle variance
        ax = axes[1, 1]
        org_variances = []
        org_names = []
        org_types = []
        
        for org_name, feats in cp_groups.items():
            available = [f for f in feats if f in cp_features.columns]
            if available:
                org_variances.append(cp_features[available].var().mean())
                org_names.append(org_name)
                org_types.append("CP")
        
        for org_name, feats in noncp_groups.items():
            available = [f for f in feats if f in noncp_features.columns]
            if available:
                org_variances.append(noncp_features[available].var().mean())
                org_names.append(org_name)
                org_types.append("Non-CP")
        
        org_df = pd.DataFrame({"organelle": org_names, "mean_var": org_variances, "type": org_types})
        org_df = org_df.sort_values("mean_var", ascending=False)
        
        colors = ["steelblue" if t == "CP" else "coral" for t in org_df["type"]]
        ax.barh(range(len(org_df)), org_df["mean_var"], color=colors)
        ax.set_yticks(range(len(org_df)))
        ax.set_yticklabels(org_df["organelle"], fontsize=8)
        ax.set_xlabel("Mean Variance", fontsize=11)
        ax.set_title("Per-Organelle Variance", fontsize=12, fontweight="bold")
        ax.invert_yaxis()
        
        # Add legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="steelblue", label="CP"),
            Patch(facecolor="coral", label="Non-CP"),
        ], loc="lower right")
        
        plt.tight_layout()
        plot_path = variance_dir / "variance_comparison.png"
        save_figure(fig, plot_path, dpi=150)
        result.add_file(plot_path)
        
        logger.info(f"    CP total variance: {metrics_df.loc[0, 'total_variance']:.1f}")
        logger.info(f"    Non-CP total variance: {metrics_df.loc[1, 'total_variance']:.1f}")
    
    def _compare_dimensionality(
        self, cp_features: pd.DataFrame, noncp_features: pd.DataFrame, result: StageResult
    ) -> None:
        """Compare effective dimensionality using PCA."""
        logger.info("  Comparing effective dimensionality...")
        
        dim_dir = self.output_dir / "dimensionality"
        dim_dir.mkdir(exist_ok=True)
        
        # Standardize
        scaler = StandardScaler()
        cp_scaled = scaler.fit_transform(cp_features)
        noncp_scaled = scaler.fit_transform(noncp_features)
        
        # PCA
        n_components = min(50, cp_scaled.shape[1], noncp_scaled.shape[1])
        
        pca_cp = PCA(n_components=n_components)
        pca_cp.fit(cp_scaled)
        
        pca_noncp = PCA(n_components=n_components)
        pca_noncp.fit(noncp_scaled)
        
        # Effective dimensionality (90% variance threshold)
        cp_cum_var = np.cumsum(pca_cp.explained_variance_ratio_)
        noncp_cum_var = np.cumsum(pca_noncp.explained_variance_ratio_)
        
        cp_eff_dim = np.argmax(cp_cum_var >= 0.90) + 1
        noncp_eff_dim = np.argmax(noncp_cum_var >= 0.90) + 1
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Cumulative variance explained
        ax = axes[0]
        ax.plot(range(1, len(cp_cum_var) + 1), cp_cum_var * 100, marker="o", label="CP", color="steelblue", linewidth=2)
        ax.plot(range(1, len(noncp_cum_var) + 1), noncp_cum_var * 100, marker="s", label="Non-CP", color="coral", linewidth=2)
        ax.axhline(90, color="black", linestyle="--", alpha=0.5, label="90% threshold")
        ax.axvline(cp_eff_dim, color="steelblue", linestyle=":", alpha=0.7)
        ax.axvline(noncp_eff_dim, color="coral", linestyle=":", alpha=0.7)
        ax.set_xlabel("Number of Components", fontsize=11)
        ax.set_ylabel("Cumulative Variance Explained (%)", fontsize=11)
        ax.set_title("Effective Dimensionality", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: Per-component variance
        ax = axes[1]
        x = np.arange(1, n_components + 1)
        width = 0.35
        ax.bar(x - width/2, pca_cp.explained_variance_ratio_ * 100, width, label="CP", color="steelblue", alpha=0.8)
        ax.bar(x + width/2, pca_noncp.explained_variance_ratio_ * 100, width, label="Non-CP", color="coral", alpha=0.8)
        ax.set_xlabel("Principal Component", fontsize=11)
        ax.set_ylabel("Variance Explained (%)", fontsize=11)
        ax.set_title("Per-Component Variance", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 21)  # Show first 20
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        plot_path = dim_dir / "dimensionality_comparison.png"
        save_figure(fig, plot_path, dpi=150)
        result.add_file(plot_path)
        
        logger.info(f"    CP effective dimensionality (90% var): {cp_eff_dim} PCs")
        logger.info(f"    Non-CP effective dimensionality (90% var): {noncp_eff_dim} PCs")
    
    def _compare_ko_discrimination(
        self, cp_features: pd.DataFrame, noncp_features: pd.DataFrame, df: pd.DataFrame, result: StageResult
    ) -> None:
        """Compare ability to discriminate KOs from NTCs."""
        logger.info("  Comparing KO discrimination power...")
        
        ko_dir = self.output_dir / "ko_discrimination"
        ko_dir.mkdir(exist_ok=True)
        
        # Get NTC vs perturbed labels
        ntc_mask = self.get_ntc_mask(df)
        
        if ntc_mask.sum() == 0 or (~ntc_mask).sum() == 0:
            logger.warning("    No NTCs or no perturbations found, skipping KO discrimination")
            return
        
        # Subsample if needed
        max_cells = 50000
        if len(df) > max_cells:
            sample_idx = np.random.choice(len(df), max_cells, replace=False)
            cp_features = cp_features.iloc[sample_idx]
            noncp_features = noncp_features.iloc[sample_idx]
            ntc_mask = ntc_mask.iloc[sample_idx]
        
        # Standardize
        scaler = StandardScaler()
        cp_scaled = scaler.fit_transform(cp_features)
        noncp_scaled = scaler.fit_transform(noncp_features)
        
        # Compute mean separation (effect size)
        cp_ntc_mean = cp_scaled[ntc_mask].mean(axis=0)
        cp_pert_mean = cp_scaled[~ntc_mask].mean(axis=0)
        cp_effect = np.abs(cp_ntc_mean - cp_pert_mean).mean()
        
        noncp_ntc_mean = noncp_scaled[ntc_mask].mean(axis=0)
        noncp_pert_mean = noncp_scaled[~ntc_mask].mean(axis=0)
        noncp_effect = np.abs(noncp_ntc_mean - noncp_pert_mean).mean()
        
        # Feature-wise t-tests
        cp_pvals = []
        for i in range(cp_scaled.shape[1]):
            _, p = ttest_ind(cp_scaled[ntc_mask, i], cp_scaled[~ntc_mask, i])
            cp_pvals.append(p)
        
        noncp_pvals = []
        for i in range(noncp_scaled.shape[1]):
            _, p = ttest_ind(noncp_scaled[ntc_mask, i], noncp_scaled[~ntc_mask, i])
            noncp_pvals.append(p)
        
        cp_sig_frac = (np.array(cp_pvals) < 0.05).sum() / len(cp_pvals)
        noncp_sig_frac = (np.array(noncp_pvals) < 0.05).sum() / len(noncp_pvals)
        
        # Save metrics
        metrics = {
            "group": ["Cell Painting", "Non-CP"],
            "mean_effect_size": [cp_effect, noncp_effect],
            "pct_significant_features": [cp_sig_frac * 100, noncp_sig_frac * 100],
            "n_ntc": [ntc_mask.sum(), ntc_mask.sum()],
            "n_perturbed": [(~ntc_mask).sum(), (~ntc_mask).sum()],
        }
        
        metrics_df = pd.DataFrame(metrics)
        csv_path = ko_dir / "ko_discrimination_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        result.add_file(csv_path)
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Effect size
        ax = axes[0]
        ax.bar(metrics_df["group"], metrics_df["mean_effect_size"], color=["steelblue", "coral"])
        ax.set_ylabel("Mean Effect Size (standardized)", fontsize=11)
        ax.set_title("NTC vs Perturbed Separation", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        
        # Plot 2: % significant features
        ax = axes[1]
        ax.bar(metrics_df["group"], metrics_df["pct_significant_features"], color=["steelblue", "coral"])
        ax.set_ylabel("% Features with p < 0.05", fontsize=11)
        ax.set_title("Discriminative Features", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        plot_path = ko_dir / "ko_discrimination.png"
        save_figure(fig, plot_path, dpi=150)
        result.add_file(plot_path)
        
        logger.info(f"    CP effect size: {cp_effect:.4f}")
        logger.info(f"    Non-CP effect size: {noncp_effect:.4f}")
        logger.info(f"    CP % significant: {cp_sig_frac*100:.1f}%")
        logger.info(f"    Non-CP % significant: {noncp_sig_frac*100:.1f}%")
    
    def _compare_positive_control_separation(
        self,
        cp_features: pd.DataFrame,
        noncp_features: pd.DataFrame,
        df: pd.DataFrame,
        gene_col: str,
        clusters: Dict[str, Dict],
        result: StageResult,
    ) -> None:
        """Compare positive control cluster separation."""
        logger.info("  Comparing positive control separation...")
        
        pc_dir = self.output_dir / "positive_control_separation"
        pc_dir.mkdir(exist_ok=True)
        
        # For each cluster, compute silhouette score
        cp_silhouettes = []
        noncp_silhouettes = []
        cluster_names = []
        
        for cluster_name, cluster_data in clusters.items():
            genes = cluster_data["genes"]
            cluster_mask = df[gene_col].isin(genes)
            
            if cluster_mask.sum() < 10:  # Too few cells
                continue
            
            # Create labels: 0 = not in cluster, 1 = in cluster
            labels = cluster_mask.astype(int).values
            
            # Subsample if needed
            if len(labels) > 50000:
                sample_idx = np.random.choice(len(labels), 50000, replace=False)
                cp_sample = cp_features.iloc[sample_idx]
                noncp_sample = noncp_features.iloc[sample_idx]
                labels_sample = labels[sample_idx]
            else:
                cp_sample = cp_features
                noncp_sample = noncp_features
                labels_sample = labels
            
            # Standardize
            scaler = StandardScaler()
            cp_scaled = scaler.fit_transform(cp_sample)
            noncp_scaled = scaler.fit_transform(noncp_sample)
            
            # Compute silhouette scores
            if len(np.unique(labels_sample)) == 2:  # Need at least 2 classes
                cp_sil = silhouette_score(cp_scaled, labels_sample)
                noncp_sil = silhouette_score(noncp_scaled, labels_sample)
                
                cp_silhouettes.append(cp_sil)
                noncp_silhouettes.append(noncp_sil)
                cluster_names.append(cluster_name)
        
        if not cluster_names:
            logger.warning("    No valid positive control clusters for comparison")
            return
        
        # Save metrics
        pc_df = pd.DataFrame({
            "cluster": cluster_names,
            "cp_silhouette": cp_silhouettes,
            "noncp_silhouette": noncp_silhouettes,
            "difference": np.array(cp_silhouettes) - np.array(noncp_silhouettes),
        })
        pc_df = pc_df.sort_values("difference", ascending=False)
        
        csv_path = pc_dir / "positive_control_silhouettes.csv"
        pc_df.to_csv(csv_path, index=False)
        result.add_file(csv_path)
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, max(6, len(pc_df) * 0.4)))
        
        x = np.arange(len(pc_df))
        width = 0.35
        
        ax.barh(x - width/2, pc_df["cp_silhouette"], width, label="CP", color="steelblue", alpha=0.8)
        ax.barh(x + width/2, pc_df["noncp_silhouette"], width, label="Non-CP", color="coral", alpha=0.8)
        
        ax.set_yticks(x)
        ax.set_yticklabels(pc_df["cluster"], fontsize=9)
        ax.set_xlabel("Silhouette Score (cluster separation)", fontsize=11)
        ax.set_title("Positive Control Separation: CP vs Non-CP", fontsize=12, fontweight="bold")
        ax.legend()
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        ax.axvline(0, color="black", linewidth=0.8)
        
        plt.tight_layout()
        plot_path = pc_dir / "positive_control_silhouettes.png"
        save_figure(fig, plot_path, dpi=150)
        result.add_file(plot_path)
        
        logger.info(f"    Mean CP silhouette: {np.mean(cp_silhouettes):.3f}")
        logger.info(f"    Mean Non-CP silhouette: {np.mean(noncp_silhouettes):.3f}")
    
    def _analyze_feature_redundancy(
        self, cp_features: pd.DataFrame, noncp_features: pd.DataFrame, result: StageResult
    ) -> None:
        """Analyze correlation between CP and non-CP feature spaces."""
        logger.info("  Analyzing feature redundancy...")
        
        redund_dir = self.output_dir / "redundancy"
        redund_dir.mkdir(exist_ok=True)
        
        # Subsample for speed
        if len(cp_features) > 10000:
            sample_idx = np.random.choice(len(cp_features), 10000, replace=False)
            cp_sample = cp_features.iloc[sample_idx]
            noncp_sample = noncp_features.iloc[sample_idx]
        else:
            cp_sample = cp_features
            noncp_sample = noncp_features
        
        # Standardize
        scaler = StandardScaler()
        cp_scaled = scaler.fit_transform(cp_sample)
        noncp_scaled = scaler.fit_transform(noncp_sample)
        
        # PCA to reduce dimensionality for cross-correlation
        pca_cp = PCA(n_components=min(20, cp_scaled.shape[1]))
        pca_noncp = PCA(n_components=min(20, noncp_scaled.shape[1]))
        
        cp_pcs = pca_cp.fit_transform(cp_scaled)
        noncp_pcs = pca_noncp.fit_transform(noncp_scaled)
        
        # Cross-correlation between PC spaces
        cross_corr = np.corrcoef(cp_pcs.T, noncp_pcs.T)
        n_cp = cp_pcs.shape[1]
        n_noncp = noncp_pcs.shape[1]
        cross_corr_block = cross_corr[:n_cp, n_cp:]
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(np.abs(cross_corr_block), cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
        ax.set_xlabel("Non-CP PCs", fontsize=11)
        ax.set_ylabel("CP PCs", fontsize=11)
        ax.set_title("Feature Space Correlation (|r|)", fontsize=12, fontweight="bold")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("|Pearson r|", fontsize=10)
        
        # Annotations
        max_corr = np.max(np.abs(cross_corr_block))
        mean_corr = np.mean(np.abs(cross_corr_block))
        
        ax.text(
            0.02, 0.98,
            f"Max |r|: {max_corr:.3f}\nMean |r|: {mean_corr:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        
        plt.tight_layout()
        plot_path = redund_dir / "feature_space_correlation.png"
        save_figure(fig, plot_path, dpi=150)
        result.add_file(plot_path)
        
        logger.info(f"    Max cross-correlation: {max_corr:.3f}")
        logger.info(f"    Mean cross-correlation: {mean_corr:.3f}")
    
    def _rank_organelles_by_discrimination(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        organelle_groups: Dict[str, List[str]],
        cp_groups: Dict[str, List[str]],
        noncp_groups: Dict[str, List[str]],
        result: StageResult,
    ) -> None:
        """Rank individual organelles by their KO discrimination power."""
        logger.info("  Ranking organelles by discrimination power...")
        
        rank_dir = self.output_dir / "organelle_ranking"
        rank_dir.mkdir(exist_ok=True)
        
        ntc_mask = self.get_ntc_mask(df)
        
        if ntc_mask.sum() == 0 or (~ntc_mask).sum() == 0:
            logger.warning("    No NTCs or no perturbations, skipping organelle ranking")
            return
        
        # Subsample
        max_cells = 50000
        if len(df) > max_cells:
            sample_idx = np.random.choice(len(df), max_cells, replace=False)
            features = features.iloc[sample_idx]
            ntc_mask = ntc_mask.iloc[sample_idx]
        
        organelle_scores = []
        organelle_names = []
        organelle_types = []
        
        for org_name, org_features in organelle_groups.items():
            available = [f for f in org_features if f in features.columns]
            
            if len(available) < 3:
                continue
            
            org_df = features[available]
            
            # Standardize
            scaler = StandardScaler()
            org_scaled = scaler.fit_transform(org_df)
            
            # Effect size
            ntc_mean = org_scaled[ntc_mask].mean(axis=0)
            pert_mean = org_scaled[~ntc_mask].mean(axis=0)
            effect = np.abs(ntc_mean - pert_mean).mean()
            
            organelle_scores.append(effect)
            organelle_names.append(org_name)
            
            # Determine type
            if org_name in cp_groups:
                organelle_types.append("CP")
            elif org_name in noncp_groups:
                organelle_types.append("Non-CP")
            else:
                organelle_types.append("Other")
        
        # Create DataFrame and sort
        rank_df = pd.DataFrame({
            "organelle": organelle_names,
            "discrimination_score": organelle_scores,
            "type": organelle_types,
        })
        rank_df = rank_df.sort_values("discrimination_score", ascending=False)
        
        csv_path = rank_dir / "organelle_discrimination_ranking.csv"
        rank_df.to_csv(csv_path, index=False)
        result.add_file(csv_path)
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, max(6, len(rank_df) * 0.35)))
        
        colors = ["steelblue" if t == "CP" else "coral" if t == "Non-CP" else "gray" for t in rank_df["type"]]
        
        ax.barh(range(len(rank_df)), rank_df["discrimination_score"], color=colors)
        ax.set_yticks(range(len(rank_df)))
        ax.set_yticklabels(rank_df["organelle"], fontsize=9)
        ax.set_xlabel("Discrimination Score (NTC vs Perturbed)", fontsize=11)
        ax.set_title("Organelle Ranking by KO Discrimination Power", fontsize=12, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="steelblue", label="Cell Painting"),
            Patch(facecolor="coral", label="Non-CP"),
        ], loc="lower right")
        
        plt.tight_layout()
        plot_path = rank_dir / "organelle_discrimination_ranking.png"
        save_figure(fig, plot_path, dpi=150)
        result.add_file(plot_path)
        
        # Summary stats
        cp_scores = rank_df[rank_df["type"] == "CP"]["discrimination_score"]
        noncp_scores = rank_df[rank_df["type"] == "Non-CP"]["discrimination_score"]
        
        logger.info(f"    Best CP organelle: {rank_df[rank_df['type'] == 'CP'].iloc[0]['organelle']} ({cp_scores.iloc[0]:.4f})")
        logger.info(f"    Best Non-CP organelle: {rank_df[rank_df['type'] == 'Non-CP'].iloc[0]['organelle']} ({noncp_scores.iloc[0]:.4f})")
        logger.info(f"    Mean CP score: {cp_scores.mean():.4f}")
        logger.info(f"    Mean Non-CP score: {noncp_scores.mean():.4f}")


