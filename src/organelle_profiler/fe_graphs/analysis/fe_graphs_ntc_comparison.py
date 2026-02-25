"""
NTC comparison analysis.

Compares perturbed cells against non-targeting controls (NTC) to identify
features with the strongest deviation from baseline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

from .fe_graphs_base import BaseAnalyzer, AnalysisResult
from ..plotting.fe_graphs_utils import create_broken_axis_bar

logger = logging.getLogger(__name__)


class NTCComparisonAnalyzer(BaseAnalyzer):
    """
    NTC comparison analysis.
    
    Generates:
    - Z-score comparison (features with strongest deviation from NTC)
    - Fold change comparison
    
    Parameters
    ----------
    data_context : DataContext
        Container with loaded data and paths.
    config : GraphConfig
        Main configuration settings.
    features : pd.DataFrame, optional
        Feature matrix. If None, extracted from cell_df.
    top_n : int
        Number of top features to display.
    """
    
    def __init__(
        self,
        *args,
        features: Optional[pd.DataFrame] = None,
        top_n: int = 20,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._features = features
        self.top_n = top_n
    
    @property
    def analysis_name(self) -> str:
        return "ntc_comparison"
    
    def run(self) -> AnalysisResult:
        """Execute NTC comparison analysis."""
        self.log_start("Generating NTC Comparison Plots")
        result = AnalysisResult()
        
        # Get features
        if self._features is not None:
            features = self._features
            feature_cols = features.columns.tolist()
        else:
            feature_cols = self.get_feature_columns(self.cell_df)
            features = self.cell_df[feature_cols].copy()
        
        # Split NTC and perturbed
        ntc_mask = self.get_ntc_mask(self.cell_df)
        ntc_df = self.cell_df[ntc_mask]
        pert_df = self.cell_df[~ntc_mask]
        
        if ntc_df.empty:
            result.add_error("No Non-Targeting Control (NTC) genes found")
            return result
        
        if pert_df.empty:
            result.add_error("No perturbed genes found to compare against NTCs")
            return result
        
        # Generate Z-score comparison
        self._generate_zscore_comparison(features, feature_cols, ntc_df, pert_df, result)
        
        # Generate fold change comparison
        self._generate_fold_change_comparison(features, feature_cols, ntc_df, pert_df, result)
        
        self.log_complete()
        return result
    
    def _generate_zscore_comparison(
        self,
        features: pd.DataFrame,
        feature_cols: list,
        ntc_df: pd.DataFrame,
        pert_df: pd.DataFrame,
        result: AnalysisResult,
    ) -> None:
        """Generate Z-score based comparison."""
        logger.info("Generating NTC comparison plot (robust Z-score method)...")
        
        ntc_features = features.loc[ntc_df.index]
        pert_features = features.loc[pert_df.index]
        
        # Calculate NTC statistics
        ntc_means = ntc_features[feature_cols].mean()
        ntc_stds = ntc_features[feature_cols].std()
        
        logger.info(f"NTC statistics derived from {len(ntc_df)} cells")
        
        # Filter features with no variance
        valid_mask = ntc_stds > 1e-6
        if not valid_mask.all():
            n_removed = (~valid_mask).sum()
            logger.info(f"Removing {n_removed} features with zero/low variance in NTC population")
            feature_cols = ntc_stds[valid_mask].index.tolist()
            
            if not feature_cols:
                result.add_error("No features with variance remaining after filtering")
                return
            
            ntc_means = ntc_means[feature_cols]
            ntc_stds = ntc_stds[feature_cols]
        
        # Calculate Z-scores for perturbed genes
        pert_gene_means = pert_df.join(pert_features).groupby("gene_name")[feature_cols].mean()
        z_scores = (pert_gene_means - ntc_means) / ntc_stds
        
        # Find features with largest effects
        mean_abs_z = z_scores.abs().mean().sort_values(ascending=False)
        top_features = mean_abs_z.head(self.top_n)
        
        if top_features.empty or top_features.iloc[0] == 0:
            result.add_error("No features showed significant deviation from NTCs")
            return
        
        result.data["top_zscore_features"] = top_features
        
        # Plot
        path = create_broken_axis_bar(
            top_features,
            title=f"Top {self.top_n} Features with Strongest Deviation from NTC (Z-score)",
            xlabel="Mean Absolute Z-score vs. NTC Population",
            save_path=self.output_dir / "ntc_feature_comparison_zscore.png",
        )
        if path:
            result.add_file(path)
    
    def _generate_fold_change_comparison(
        self,
        features: pd.DataFrame,
        feature_cols: list,
        ntc_df: pd.DataFrame,
        pert_df: pd.DataFrame,
        result: AnalysisResult,
    ) -> None:
        """Generate fold change based comparison."""
        logger.info("Generating NTC comparison plot (log2 fold change)...")
        
        ntc_features = features.loc[ntc_df.index]
        pert_features = features.loc[pert_df.index]
        
        # Calculate fold change
        ntc_means = ntc_features[feature_cols].mean()
        ntc_means_nozero = ntc_means.replace(0, 1e-9)
        
        pert_gene_means = pert_df.join(pert_features).groupby("gene_name")[feature_cols].mean()
        fold_change = pert_gene_means.div(ntc_means_nozero, axis=1)
        
        # Find features with largest effects
        mean_abs_log_fc = np.log2(fold_change).abs().mean()
        finite_fc = mean_abs_log_fc[np.isfinite(mean_abs_log_fc)]
        top_fc = finite_fc.sort_values(ascending=False).head(self.top_n)
        
        if top_fc.empty or top_fc.iloc[0] == 0:
            result.add_error("No features showed significant fold change vs NTCs")
            return
        
        result.data["top_fc_features"] = top_fc
        
        # Plot
        path = create_broken_axis_bar(
            top_fc,
            title=f"Top {self.top_n} Features with Strongest Deviation from NTC (Fold Change)",
            xlabel="Mean Absolute log2(Fold Change vs NTC)",
            save_path=self.output_dir / "ntc_feature_comparison_fold_change.png",
        )
        if path:
            result.add_file(path)
