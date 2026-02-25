"""
Organelle contribution analysis.

Analyzes how much each organelle/segmentation group contributes to
the phenotypic signature of each gene perturbation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict
import logging

from .fe_graphs_base import BaseAnalyzer, AnalysisResult
from ..plotting.fe_graphs_utils import save_figure
from ..plotting.fe_graphs_heatmaps import plot_heatmap, plot_clustermap

logger = logging.getLogger(__name__)


class OrganelleContributionAnalyzer(BaseAnalyzer):
    """
    Organelle contribution analysis.
    
    For each gene, computes how much each organelle group contributes
    to the overall phenotypic signature by calculating mean absolute
    z-scores per organelle.
    
    Generates:
    - Normalized contribution heatmap
    - Raw z-score heatmap
    - Clustered heatmap
    - Average contribution bar plot
    
    Parameters
    ----------
    data_context : DataContext
        Container with loaded data and paths.
    config : GraphConfig
        Main configuration settings.
    features : pd.DataFrame, optional
        Feature matrix. If None, extracted from cell_df.
    """
    
    def __init__(self, *args, features: Optional[pd.DataFrame] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._features = features
    
    @property
    def analysis_name(self) -> str:
        return "organelle_contribution"
    
    def run(self) -> AnalysisResult:
        """Execute organelle contribution analysis."""
        self.log_start("Generating Organelle Contribution Heatmap")
        result = AnalysisResult()
        
        if "gene_name" not in self.cell_df.columns:
            result.add_error("gene_name column not found in cell metadata")
            return result
        
        # Get features
        if self._features is not None:
            features = self._features
        else:
            feature_cols = self.get_feature_columns(self.cell_df)
            features = self.cell_df[feature_cols].copy()
        
        # Group features by organelle
        organelle_features = self.group_features_by_organelle(features.columns.tolist())
        organelles = sorted(organelle_features.keys())
        
        logger.info(f"Found {len(organelles)} organelle groups: {organelles}")
        
        if len(organelles) < 2:
            result.add_error("Need at least 2 organelle groups for comparison")
            return result
        
        # Identify NTC cells
        ntc_mask = self.get_ntc_mask(self.cell_df)
        ntc_features = features.loc[ntc_mask]
        
        if len(ntc_features) < 10:
            result.add_error(f"Not enough NTC cells ({len(ntc_features)})")
            return result
        
        # Compute NTC statistics
        ntc_mean = ntc_features.mean()
        ntc_std = ntc_features.std().replace(0, 1)
        
        # Get unique genes (excluding NTC)
        genes = self.cell_df.loc[~ntc_mask, "gene_name"].unique()
        genes = [g for g in genes if pd.notna(g) and str(g) != "nan"]
        logger.info(f"Computing contribution scores for {len(genes)} genes...")
        
        # Compute contribution scores
        contribution_data = self._compute_contribution_scores(
            features, organelle_features, organelles, ntc_mean, ntc_std, genes
        )
        
        if not contribution_data:
            result.add_error("No genes with enough cells for analysis")
            return result
        
        contrib_df = pd.DataFrame(contribution_data)
        contrib_df = contrib_df.sort_values("n_cells", ascending=False)
        
        # Save raw data
        contrib_df.to_csv(self.output_dir / "organelle_contribution_scores.csv", index=False)
        result.add_file(self.output_dir / "organelle_contribution_scores.csv")
        
        result.data["contribution_df"] = contrib_df
        
        # Generate plots
        self._generate_plots(contrib_df, organelles, result)
        
        self.log_complete()
        return result
    
    def _compute_contribution_scores(
        self,
        features: pd.DataFrame,
        organelle_features: Dict[str, List[str]],
        organelles: List[str],
        ntc_mean: pd.Series,
        ntc_std: pd.Series,
        genes: List,
    ) -> List[Dict]:
        """Compute contribution scores for each gene."""
        contribution_data = []
        
        for gene in genes:
            gene_mask = self.cell_df["gene_name"] == gene
            gene_features = features.loc[gene_mask]
            
            if len(gene_features) < 3:
                continue
            
            gene_scores = {}
            for organelle, cols in organelle_features.items():
                cols_present = [c for c in cols if c in features.columns]
                if not cols_present:
                    gene_scores[organelle] = 0
                    continue
                
                # Z-score for this gene vs NTC
                gene_mean = gene_features[cols_present].mean()
                z_scores = (gene_mean - ntc_mean[cols_present]) / ntc_std[cols_present]
                
                # Mean absolute z-score as contribution metric
                mean_abs_z = np.abs(z_scores).mean()
                gene_scores[organelle] = mean_abs_z
            
            # Normalize to sum to 1
            total = sum(gene_scores.values())
            if total > 0:
                gene_scores_norm = {k: v / total for k, v in gene_scores.items()}
            else:
                gene_scores_norm = {k: 0 for k in gene_scores}
            
            contribution_data.append({
                "gene": gene,
                "n_cells": len(gene_features),
                **gene_scores,
                **{f"{k}_norm": v for k, v in gene_scores_norm.items()},
            })
        
        return contribution_data
    
    def _generate_plots(
        self,
        contrib_df: pd.DataFrame,
        organelles: List[str],
        result: AnalysisResult,
    ) -> None:
        """Generate visualization plots."""
        
        # 1. Normalized contribution heatmap (top genes)
        self._plot_normalized_heatmap(contrib_df, organelles, result)
        
        # 2. Raw z-score heatmap
        self._plot_raw_heatmap(contrib_df, organelles, result)
        
        # 3. Clustered heatmap
        self._plot_clustered_heatmap(contrib_df, organelles, result)
        
        # 4. Average contribution bar plot
        self._plot_average_contribution(contrib_df, organelles, result)
    
    def _plot_normalized_heatmap(
        self,
        contrib_df: pd.DataFrame,
        organelles: List[str],
        result: AnalysisResult,
    ) -> None:
        """Generate normalized contribution heatmap."""
        top_n = min(50, len(contrib_df))
        top_genes = contrib_df.head(top_n)
        
        norm_cols = [f"{org}_norm" for org in organelles]
        heatmap_data = top_genes.set_index("gene")[norm_cols]
        heatmap_data.columns = organelles
        
        fig, ax = plt.subplots(figsize=(max(12, len(organelles) * 0.8), max(10, top_n * 0.3)))
        
        sns.heatmap(
            heatmap_data,
            cmap="YlOrRd",
            annot=False,
            ax=ax,
            cbar_kws={"label": "Relative Contribution (normalized)"},
            vmin=0,
            vmax=0.5,
        )
        
        ax.set_title(f"Organelle Contribution to Gene Phenotypes (Top {top_n} genes by cell count)")
        ax.set_xlabel("Organelle / Segmentation Group")
        ax.set_ylabel("Gene")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        path = save_figure(fig, self.output_dir / "organelle_contribution_heatmap_normalized.png")
        result.add_file(path)
    
    def _plot_raw_heatmap(
        self,
        contrib_df: pd.DataFrame,
        organelles: List[str],
        result: AnalysisResult,
    ) -> None:
        """Generate raw z-score heatmap."""
        top_n = min(50, len(contrib_df))
        top_genes = contrib_df.head(top_n)
        
        heatmap_data = top_genes.set_index("gene")[organelles]
        
        fig, ax = plt.subplots(figsize=(max(12, len(organelles) * 0.8), max(10, top_n * 0.3)))
        
        sns.heatmap(
            heatmap_data,
            cmap="viridis",
            annot=False,
            ax=ax,
            cbar_kws={"label": "Mean Absolute Z-score vs NTC"},
        )
        
        ax.set_title(f"Organelle Effect Magnitude by Gene (Top {top_n} genes by cell count)")
        ax.set_xlabel("Organelle / Segmentation Group")
        ax.set_ylabel("Gene")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        path = save_figure(fig, self.output_dir / "organelle_contribution_heatmap_raw.png")
        result.add_file(path)
    
    def _plot_clustered_heatmap(
        self,
        contrib_df: pd.DataFrame,
        organelles: List[str],
        result: AnalysisResult,
    ) -> None:
        """Generate clustered heatmap."""
        try:
            norm_cols = [f"{org}_norm" for org in organelles]
            cluster_data = contrib_df.set_index("gene")[norm_cols]
            cluster_data.columns = organelles
            cluster_data = cluster_data.dropna()
            
            if len(cluster_data) < 10:
                return
            
            g = sns.clustermap(
                cluster_data,
                cmap="YlOrRd",
                figsize=(max(12, len(organelles) * 0.8), max(14, len(cluster_data) * 0.15)),
                dendrogram_ratio=(0.1, 0.15),
                cbar_pos=(0.02, 0.8, 0.03, 0.15),
                vmin=0,
                vmax=0.5,
            )
            g.ax_heatmap.set_xlabel("Organelle / Segmentation Group")
            g.ax_heatmap.set_ylabel("Gene")
            g.fig.suptitle("Clustered Organelle Contribution Heatmap (all genes)", y=1.02)
            
            path = self.output_dir / "organelle_contribution_heatmap_clustered.png"
            g.fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(g.fig)
            result.add_file(path)
            logger.info(f"Saved clustered heatmap: {path}")
            
        except Exception as e:
            logger.warning(f"Could not generate clustered heatmap: {e}")
    
    def _plot_average_contribution(
        self,
        contrib_df: pd.DataFrame,
        organelles: List[str],
        result: AnalysisResult,
    ) -> None:
        """Generate average contribution bar plot."""
        norm_cols = [f"{org}_norm" for org in organelles]
        avg_contrib = contrib_df[norm_cols].mean()
        avg_contrib.index = organelles
        
        fig, ax = plt.subplots(figsize=(max(10, len(organelles) * 0.6), 6))
        
        bars = ax.bar(
            avg_contrib.index, avg_contrib.values,
            color=sns.color_palette("husl", len(organelles))
        )
        
        ax.set_xlabel("Organelle / Segmentation Group")
        ax.set_ylabel("Average Relative Contribution")
        ax.set_title("Average Organelle Contribution Across All Genes")
        plt.xticks(rotation=45, ha="right")
        
        # Add value labels
        for bar, val in zip(bars, avg_contrib.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9
            )
        
        plt.tight_layout()
        
        path = save_figure(fig, self.output_dir / "organelle_average_contribution.png")
        result.add_file(path)
        
        logger.info("Organelle contribution analysis complete!")
