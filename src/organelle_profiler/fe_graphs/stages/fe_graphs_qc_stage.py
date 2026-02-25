"""
QC Stage: Quality control and artifact detection.

Performs:
- Spatial drift analysis (edge effects)
- Batch effects (well distribution)
- Feature quality checks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import logging

from .fe_graphs_stage_base import BaseStage, StageResult
from ..plotting.fe_graphs_utils import save_figure

logger = logging.getLogger(__name__)


class QCStage(BaseStage):
    """Quality control stage."""
    
    STAGE_NUMBER = 1
    STAGE_NAME = "qc"
    
    def run(self) -> StageResult:
        """Run QC analysis."""
        self.log_start()
        result = StageResult()
        
        df = self.df
        if df.empty:
            result.add_error(f"No data at {self.level} level")
            return result
        
        result.add_metric("n_items", len(df))
        
        # Level-specific QC
        if self.level == "cell":
            self._run_cell_qc(result)
        elif self.level == "guide":
            self._run_guide_qc(result)
        elif self.level == "gene":
            self._run_gene_qc(result)
        
        # Common: Feature quality
        self._check_feature_quality(result)
        
        # Save QC summary
        self._save_qc_summary(result)
        
        self.log_complete(result)
        return result
    
    def _run_cell_qc(self, result: StageResult) -> None:
        """Cell-level specific QC."""
        # Spatial drift
        drift_dir = self.output_dir / "spatial_drift"
        drift_dir.mkdir(exist_ok=True)
        self._analyze_spatial_drift(drift_dir, result)
        
        # Batch effects (well distribution)
        batch_dir = self.output_dir / "batch_effects"
        batch_dir.mkdir(exist_ok=True)
        self._analyze_batch_effects(batch_dir, result)
    
    def _run_guide_qc(self, result: StageResult) -> None:
        """Guide-level specific QC."""
        # Guide cell counts
        if "n_cells" in self.df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.df["n_cells"].hist(bins=50, ax=ax)
            ax.set_xlabel("Number of cells per guide")
            ax.set_ylabel("Count")
            ax.set_title(f"Guide Cell Count Distribution (n={len(self.df)})")
            ax.axvline(self.analysis_config.min_enrichment_cluster_size, 
                      color='red', linestyle='--', label='Min threshold')
            ax.legend()
            path = save_figure(fig, self.output_dir / "guide_cell_counts.png")
            result.add_file(path)
            
            # Flag low-count guides
            low_count = (self.df["n_cells"] < self.analysis_config.min_enrichment_cluster_size).sum()
            result.add_metric("guides_low_cell_count", low_count)
        
        # Guide consistency (within-gene agreement)
        self._analyze_guide_consistency(result)
    
    def _run_gene_qc(self, result: StageResult) -> None:
        """Gene-level specific QC."""
        # Gene cell counts
        if "n_cells" in self.df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.df["n_cells"].hist(bins=50, ax=ax)
            ax.set_xlabel("Number of cells per gene")
            ax.set_ylabel("Count")
            ax.set_title(f"Gene Cell Count Distribution (n={len(self.df)})")
            path = save_figure(fig, self.output_dir / "gene_cell_counts.png")
            result.add_file(path)
        
        # Gene guide counts
        if "n_guides" in self.df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.df["n_guides"].hist(bins=20, ax=ax)
            ax.set_xlabel("Number of guides per gene")
            ax.set_ylabel("Count")
            ax.set_title(f"Guides per Gene Distribution (n={len(self.df)})")
            path = save_figure(fig, self.output_dir / "gene_guide_counts.png")
            result.add_file(path)
    
    def _analyze_spatial_drift(self, output_dir: Path, result: StageResult) -> None:
        """Analyze spatial drift (edge effects)."""
        df = self.df.copy()
        
        # Calculate radial positions
        for pos_type in ["well", "tile"]:
            df = self._calculate_radial_position(df, pos_type)
        
        # Calculate drift score
        features = self.get_features(df)
        if features.empty:
            return
        
        scaled = StandardScaler().fit_transform(features)
        df["drift_score"] = np.mean(np.abs(scaled), axis=1)
        
        # Plot drift vs radial position
        for pos_type in ["well_radial_pos", "tile_radial_pos"]:
            if pos_type not in df.columns or df[pos_type].isna().all():
                continue
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Bin and plot
            df["radial_bin"] = pd.cut(df[pos_type], bins=30)
            binned = df.groupby("radial_bin", observed=False)["drift_score"].mean()
            
            ax.plot(range(len(binned)), binned.values, "o-")
            ax.set_xlabel(f"Radial Distance Bin ({pos_type.replace('_', ' ')})")
            ax.set_ylabel("Mean Feature Drift Score")
            ax.set_title(f"Spatial Drift Analysis: {pos_type.replace('_', ' ').title()}")
            
            path = save_figure(fig, output_dir / f"{pos_type.replace('_pos', '')}_radial_drift.png")
            result.add_file(path)
        
        # Save drift summary
        drift_summary = df.groupby("well", observed=True)["drift_score"].agg(["mean", "std"]).reset_index()
        drift_summary.to_csv(output_dir / "drift_summary.csv", index=False)
        result.add_file(output_dir / "drift_summary.csv")
    
    def _calculate_radial_position(self, df: pd.DataFrame, pos_type: str) -> pd.DataFrame:
        """Calculate radial position for well or tile."""
        if pos_type == "well":
            x_col, y_col, group_col = "x_global_pheno", "y_global_pheno", "well"
        else:
            x_col, y_col, group_col = "x_local_pheno", "y_local_pheno", "tile_pheno"
        
        if not all(c in df.columns for c in [x_col, y_col, group_col]):
            return df
        
        # Ensure numeric
        for col in [x_col, y_col]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        result_col = f"{pos_type}_radial_pos"
        df[result_col] = np.nan
        
        valid_idx = df[[x_col, y_col, group_col]].dropna().index
        if valid_idx.empty:
            return df
        
        pos_df = df.loc[valid_idx].copy()
        
        # Calculate center per group
        centers = pos_df.groupby(group_col, observed=True).agg(
            cx=(x_col, lambda x: (x.min() + x.max()) / 2),
            cy=(y_col, lambda x: (x.min() + x.max()) / 2),
        )
        
        cx = pos_df[group_col].map(centers["cx"]).astype(float).values
        cy = pos_df[group_col].map(centers["cy"]).astype(float).values
        
        radial = np.sqrt((pos_df[x_col].values - cx)**2 + (pos_df[y_col].values - cy)**2)
        df.loc[valid_idx, result_col] = radial
        
        return df
    
    def _analyze_batch_effects(self, output_dir: Path, result: StageResult) -> None:
        """Analyze batch effects across wells."""
        if "well" not in self.df.columns:
            return
        
        # Well distribution
        well_counts = self.df["well"].value_counts()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        well_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Well")
        ax.set_ylabel("Cell Count")
        ax.set_title("Cell Distribution Across Wells")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        path = save_figure(fig, output_dir / "well_distribution.png")
        result.add_file(path)
        
        result.add_metric("n_wells", len(well_counts))
        result.add_metric("cells_per_well_mean", well_counts.mean())
        result.add_metric("cells_per_well_std", well_counts.std())
    
    def _analyze_guide_consistency(self, result: StageResult) -> None:
        """Analyze guide consistency within genes."""
        if "gene_name" not in self.df.columns:
            return
        
        # Group guides by gene
        gene_guide_counts = self.df.groupby("gene_name").size()
        
        # For genes with multiple guides, check correlation
        features = self.get_features()
        if features.empty:
            return
        
        consistency_dir = self.output_dir / "guide_consistency"
        consistency_dir.mkdir(exist_ok=True)
        
        # Simple metric: coefficient of variation within genes
        consistency_data = []
        for gene, group in self.df.groupby("gene_name"):
            if len(group) < 2:
                continue
            gene_features = features.loc[group.index]
            cv = gene_features.std() / (gene_features.mean() + 1e-10)
            consistency_data.append({
                "gene": gene,
                "n_guides": len(group),
                "mean_cv": cv.mean(),
            })
        
        if consistency_data:
            consistency_df = pd.DataFrame(consistency_data)
            consistency_df.to_csv(consistency_dir / "guide_consistency.csv", index=False)
            result.add_file(consistency_dir / "guide_consistency.csv")
            
            # Flag inconsistent genes
            inconsistent = consistency_df[consistency_df["mean_cv"] > 1.0]
            if not inconsistent.empty:
                inconsistent.to_csv(consistency_dir / "inconsistent_guides.csv", index=False)
                result.add_file(consistency_dir / "inconsistent_guides.csv")
                result.add_metric("inconsistent_genes", len(inconsistent))
    
    def _check_feature_quality(self, result: StageResult) -> None:
        """Check feature quality (variance, correlations)."""
        features = self.get_features()
        if features.empty:
            return
        
        quality_dir = self.output_dir / "feature_quality"
        quality_dir.mkdir(exist_ok=True)
        
        # Feature variance
        variances = features.var().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        variances.head(30).plot(kind="bar", ax=ax)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Variance")
        ax.set_title("Top 30 Features by Variance")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        path = save_figure(fig, quality_dir / "feature_variance.png")
        result.add_file(path)
        
        # Low variance features
        low_var = (variances < self.analysis_config.low_variance_threshold).sum()
        result.add_metric("low_variance_features", low_var)
        result.add_metric("total_features", len(variances))
        
        # Organelle-specific QC
        self._check_organelle_feature_quality(features, quality_dir, result)
    
    def _check_organelle_feature_quality(
        self,
        features: pd.DataFrame,
        output_dir: Path,
        result: StageResult,
    ) -> None:
        """
        Check feature quality per organelle group.
        
        Detects organelle groups with:
        - Uniformly low variance (poor segmentation/features)
        - High percentage of zero/NaN values
        - Extreme correlations (redundant features)
        
        This helps identify problematic organelle channels or segmentations.
        
        NOTE: Only runs at cell level - organelle features don't exist at guide/gene levels.
        """
        # Only meaningful at cell level
        if self.level != "cell":
            logger.debug(f"Skipping organelle QC at {self.level} level (cell level only)")
            return
        
        if not hasattr(self.data, 'organelle_groups') or not self.data.organelle_groups:
            logger.info("No organelle groups defined, skipping organelle QC")
            return
        
        logger.info("Checking feature quality per organelle group...")
        
        organelle_qc_results = []
        
        for organelle_name, feature_cols in self.data.organelle_groups.items():
            # Get features for this organelle
            available_cols = [col for col in feature_cols if col in features.columns]
            
            # Log if features are missing
            if len(available_cols) < len(feature_cols):
                missing_count = len(feature_cols) - len(available_cols)
                logger.warning(f"  {organelle_name}: {missing_count}/{len(feature_cols)} features missing from DataFrame")
            
            if not available_cols:
                logger.warning(f"  {organelle_name}: NO FEATURES AVAILABLE (expected {len(feature_cols)})")
                organelle_qc_results.append({
                    "organelle": organelle_name,
                    "n_features": 0,
                    "mean_variance": 0,
                    "median_variance": 0,
                    "pct_low_variance": 100.0,
                    "pct_zero_or_nan": 100.0,
                    "status": "NO_FEATURES",
                })
                continue
            
            org_features = features[available_cols]
            
            # Compute metrics
            variances = org_features.var()
            
            # Percentage of low variance features (use percentile-based threshold)
            if len(variances) > 0:
                low_var_threshold = max(np.percentile(variances, 5.0), 1e-8)
            else:
                low_var_threshold = 1e-8
            pct_low_var = (variances < low_var_threshold).sum() / len(variances) * 100
            
            # Percentage of features that are mostly zero or NaN
            zero_or_nan_counts = []
            for col in available_cols:
                col_data = org_features[col]
                pct_zero_or_nan = (col_data.isna() | (col_data == 0)).sum() / len(col_data) * 100
                zero_or_nan_counts.append(pct_zero_or_nan)
            
            mean_zero_or_nan = np.mean(zero_or_nan_counts)
            
            # Determine status
            if pct_low_var > 80:
                status = "POOR"  # Most features have low variance
            elif pct_low_var > 50:
                status = "MARGINAL"
            elif mean_zero_or_nan > 80:
                status = "SPARSE"  # Mostly zeros/NaNs
            else:
                status = "GOOD"
            
            organelle_qc_results.append({
                "organelle": organelle_name,
                "n_features": len(available_cols),
                "mean_variance": variances.mean(),
                "median_variance": variances.median(),
                "pct_low_variance": pct_low_var,
                "pct_zero_or_nan": mean_zero_or_nan,
                "status": status,
            })
        
        # Convert to DataFrame
        organelle_qc_df = pd.DataFrame(organelle_qc_results)
        organelle_qc_df = organelle_qc_df.sort_values("mean_variance", ascending=False)
        
        # Save CSV
        csv_path = output_dir / "organelle_feature_quality.csv"
        organelle_qc_df.to_csv(csv_path, index=False)
        result.add_file(csv_path)
        
        # Generate visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Mean variance by organelle
        ax = axes[0, 0]
        colors = organelle_qc_df["status"].map({
            "GOOD": "green",
            "MARGINAL": "orange",
            "POOR": "red",
            "SPARSE": "purple",
            "NO_FEATURES": "gray",
        })
        y_pos = np.arange(len(organelle_qc_df))
        ax.barh(y_pos, organelle_qc_df["mean_variance"], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(organelle_qc_df["organelle"], fontsize=8)
        ax.set_xlabel("Mean Variance")
        ax.set_title("Mean Feature Variance by Organelle")
        ax.invert_yaxis()
        
        # 2. % Low variance features
        ax = axes[0, 1]
        ax.barh(y_pos, organelle_qc_df["pct_low_variance"], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(organelle_qc_df["organelle"], fontsize=8)
        ax.set_xlabel("% Low Variance Features")
        ax.set_title("Percentage of Low Variance Features")
        ax.axvline(50, color="orange", linestyle="--", alpha=0.5, label="50% threshold")
        ax.axvline(80, color="red", linestyle="--", alpha=0.5, label="80% threshold")
        ax.legend()
        ax.invert_yaxis()
        
        # 3. % Zero/NaN values
        ax = axes[1, 0]
        ax.barh(y_pos, organelle_qc_df["pct_zero_or_nan"], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(organelle_qc_df["organelle"], fontsize=8)
        ax.set_xlabel("% Zero/NaN Values")
        ax.set_title("Percentage of Zero/NaN Values (avg across features)")
        ax.axvline(80, color="red", linestyle="--", alpha=0.5, label="80% threshold")
        ax.legend()
        ax.invert_yaxis()
        
        # 4. Number of features per organelle
        ax = axes[1, 1]
        ax.barh(y_pos, organelle_qc_df["n_features"], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(organelle_qc_df["organelle"], fontsize=8)
        ax.set_xlabel("Number of Features")
        ax.set_title("Feature Count by Organelle")
        ax.invert_yaxis()
        
        # Add legend for status colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="green", label="GOOD: High variance, low zeros"),
            Patch(facecolor="orange", label="MARGINAL: Some low variance"),
            Patch(facecolor="red", label="POOR: Mostly low variance"),
            Patch(facecolor="purple", label="SPARSE: Mostly zeros/NaNs"),
            Patch(facecolor="gray", label="NO_FEATURES: Missing"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9)
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for legend
        path = save_figure(fig, output_dir / "organelle_feature_quality.png")
        result.add_file(path)
        
        # Log warnings for poor organelles
        poor_organelles = organelle_qc_df[organelle_qc_df["status"].isin(["POOR", "SPARSE", "NO_FEATURES"])]
        if not poor_organelles.empty:
            logger.warning(f"Found {len(poor_organelles)} organelle groups with quality issues:")
            for _, row in poor_organelles.iterrows():
                logger.warning(f"  - {row['organelle']}: {row['status']} "
                             f"(mean_var={row['mean_variance']:.3f}, "
                             f"{row['pct_low_variance']:.1f}% low_var, "
                             f"{row['pct_zero_or_nan']:.1f}% zero/NaN)")
        
        # Add metrics to result
        result.add_metric("organelles_good", (organelle_qc_df["status"] == "GOOD").sum())
        result.add_metric("organelles_poor", (organelle_qc_df["status"] == "POOR").sum())
        result.add_metric("organelles_sparse", (organelle_qc_df["status"] == "SPARSE").sum())
    
    def _save_qc_summary(self, result: StageResult) -> None:
        """Save QC summary."""
        summary = pd.DataFrame([result.metrics])
        summary.to_csv(self.output_dir / "qc_summary.csv", index=False)
        result.add_file(self.output_dir / "qc_summary.csv")
