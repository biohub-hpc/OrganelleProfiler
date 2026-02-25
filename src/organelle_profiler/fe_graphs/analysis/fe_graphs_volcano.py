"""
Volcano plot analysis.

Generates volcano plots showing statistical significance vs fold change
for features comparing perturbed genes against NTC controls.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
from scipy.stats import ttest_ind
from tqdm import tqdm
import logging

from .fe_graphs_base import BaseAnalyzer, AnalysisResult
from ..plotting.fe_graphs_utils import save_figure

logger = logging.getLogger(__name__)


class VolcanoAnalyzer(BaseAnalyzer):
    """
    Volcano plot analysis for gene-feature combinations.
    
    Generates volcano plots for top differentially expressed features,
    comparing each gene's feature values against NTC controls.
    
    Parameters
    ----------
    data_context : DataContext
        Container with loaded data and paths.
    config : GraphConfig
        Main configuration settings.
    features : pd.DataFrame, optional
        Feature matrix. If None, extracted from cell_df.
    output_subdir : str
        Subdirectory for output files.
    top_n_features : int
        Number of top features to generate plots for.
    """
    
    def __init__(
        self,
        *args,
        features: Optional[pd.DataFrame] = None,
        output_subdir: str = "volcano",
        top_n_features: int = 30,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._features = features
        self.output_subdir = output_subdir
        self.top_n_features = top_n_features
        self._analysis_name = output_subdir
    
    @property
    def analysis_name(self) -> str:
        return self._analysis_name
    
    def run(self) -> AnalysisResult:
        """Execute volcano plot analysis."""
        self.log_start(f"Generating Volcano Plots (top {self.top_n_features} features)")
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
        
        if ntc_df.empty or pert_df.empty:
            result.add_error("Cannot generate volcano plots: NTC or perturbed data missing")
            return result
        
        ntc_features = features.loc[ntc_mask]
        
        # Find top features by Z-score deviation
        logger.info("Identifying top features by Z-score deviation...")
        ntc_means = ntc_features.mean()
        ntc_stds = ntc_features.std()
        
        valid_mask = ntc_stds > 1e-6
        feature_cols = ntc_stds[valid_mask].index.tolist()
        ntc_means = ntc_means[feature_cols]
        ntc_stds = ntc_stds[feature_cols]
        
        pert_gene_means = pert_df.groupby("gene_name")[feature_cols].mean()
        z_scores = (pert_gene_means - ntc_means) / ntc_stds
        mean_abs_z = z_scores.abs().mean().sort_values(ascending=False)
        
        top_features = mean_abs_z.head(self.top_n_features).index.tolist()
        logger.info(f"Selected {len(top_features)} features based on mean absolute Z-score")
        
        # Calculate statistics for volcano plots
        logger.info("Calculating p-values and fold changes...")
        ntc_means_nozero = ntc_means.replace(0, 1e-9)
        fold_change = pert_gene_means[feature_cols].div(ntc_means_nozero[feature_cols], axis=1)
        log2_fc = np.log2(fold_change)
        
        volcano_data = []
        for feature in tqdm(top_features, desc="Computing statistics"):
            ntc_vals = ntc_features[feature]
            
            for gene_name, group in pert_df.groupby("gene_name"):
                pert_vals = group[feature] if feature in group.columns else pd.Series()
                
                if len(pert_vals.dropna()) > 1 and len(ntc_vals.dropna()) > 1:
                    _, p_value = ttest_ind(pert_vals, ntc_vals, equal_var=False, nan_policy="omit")
                else:
                    p_value = 1.0
                
                log2fc_val = log2_fc.loc[gene_name, feature] if gene_name in log2_fc.index else np.nan
                
                volcano_data.append({
                    "gene_name": gene_name,
                    "feature": feature,
                    "p_value": p_value,
                    "log2_fold_change": log2fc_val,
                })
        
        if not volcano_data:
            result.add_error("No data generated for volcano plots")
            return result
        
        volcano_df = pd.DataFrame(volcano_data)
        volcano_df["-log10p"] = -np.log10(volcano_df["p_value"])
        
        # Handle infinite values
        max_logp = volcano_df.loc[np.isfinite(volcano_df["-log10p"]), "-log10p"].max()
        if pd.notna(max_logp):
            volcano_df.replace([np.inf], max_logp * 1.1, inplace=True)
        volcano_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        volcano_df.dropna(subset=["log2_fold_change", "-log10p"], inplace=True)
        
        result.data["volcano_df"] = volcano_df
        
        # Generate plots
        self._generate_volcano_plots(volcano_df, top_features, result)
        
        self.log_complete()
        return result
    
    def _generate_volcano_plots(
        self,
        volcano_df: pd.DataFrame,
        top_features: List[str],
        result: AnalysisResult,
    ) -> None:
        """Generate individual volcano plots for each feature."""
        logger.info("Generating individual volcano plots...")
        
        p_threshold = self.analysis_config.volcano_p_threshold
        fc_threshold = self.analysis_config.volcano_log2fc_threshold
        
        for feature in top_features:
            plot_df = volcano_df[volcano_df["feature"] == feature].copy()
            
            if plot_df.empty:
                continue
            
            fig, ax = plt.subplots(figsize=self.plot_config.figsize_volcano)
            
            # Define significance conditions
            cond_down = (plot_df["p_value"] < p_threshold) & (plot_df["log2_fold_change"] < -fc_threshold)
            cond_up = (plot_df["p_value"] < p_threshold) & (plot_df["log2_fold_change"] > fc_threshold)
            
            # Plot points
            ax.scatter(
                plot_df["log2_fold_change"], plot_df["-log10p"],
                c="grey", alpha=0.6, label="Not Significant"
            )
            ax.scatter(
                plot_df.loc[cond_down, "log2_fold_change"],
                plot_df.loc[cond_down, "-log10p"],
                c="cornflowerblue", alpha=0.8, label="Downregulated"
            )
            ax.scatter(
                plot_df.loc[cond_up, "log2_fold_change"],
                plot_df.loc[cond_up, "-log10p"],
                c="red", alpha=0.8, label="Upregulated"
            )
            
            # Label top genes
            genes_to_label = pd.concat([
                plot_df[cond_down].nsmallest(10, "p_value"),
                plot_df[cond_up].nsmallest(10, "p_value"),
            ])
            
            for _, row in genes_to_label.iterrows():
                ax.text(row["log2_fold_change"], row["-log10p"], str(row["gene_name"]), fontsize=9)
            
            # Threshold lines
            ax.axhline(-np.log10(p_threshold), color="black", linestyle="--", lw=1)
            ax.axvline(fc_threshold, color="black", linestyle="--", lw=1)
            ax.axvline(-fc_threshold, color="black", linestyle="--", lw=1)
            
            ax.set_title(f"Volcano Plot for: {feature}", fontsize=16)
            ax.set_xlabel("log2(Fold Change) vs NTC Population")
            ax.set_ylabel("-log10(p-value)")
            ax.legend()
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            
            path = save_figure(fig, self.output_dir / f"volcano_{feature}.png")
            result.add_file(path)
        
        logger.info(f"Saved {len(top_features)} volcano plots to: {self.output_dir}")
