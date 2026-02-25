"""
Aggregated-level (guide/gene) UMAP analysis.

Generates UMAP embeddings and visualizations for guide-level and gene-level
aggregated feature data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from scipy.stats import spearmanr, ttest_ind
import logging

from .fe_graphs_base import BaseAnalyzer, AnalysisResult
from ..core.fe_graphs_embedding import EmbeddingEngine
from ..core.fe_graphs_clustering import ClusteringEngine
from ..core.fe_graphs_cache import EmbeddingCache
from ..plotting.fe_graphs_utils import save_figure, create_cluster_palette, add_gene_labels
from ..plotting.fe_graphs_umap_plots import plot_umap_continuous

logger = logging.getLogger(__name__)


class AggregatedLevelAnalyzer(BaseAnalyzer):
    """
    Aggregated-level (guide/gene) UMAP analysis.
    
    Generates UMAPs for guide-level or gene-level aggregated features,
    including clustering, gene effect coloring, and feature importance analysis.
    
    Parameters
    ----------
    data_context : DataContext
        Container with loaded data and paths.
    config : GraphConfig
        Main configuration settings.
    level : str
        Analysis level: "guide" or "gene".
    """
    
    def __init__(self, *args, level: str = "guide", **kwargs):
        super().__init__(*args, **kwargs)
        self.level = level
        self._analysis_name = f"{level}_umap"
    
    @property
    def analysis_name(self) -> str:
        return self._analysis_name
    
    @property
    def df(self) -> pd.DataFrame:
        """Get the appropriate DataFrame for this level."""
        if self.level == "guide":
            return self.guide_df
        elif self.level == "gene":
            return self.gene_df
        else:
            raise ValueError(f"Unknown level: {self.level}")
    
    @property
    def id_col(self) -> str:
        """Column for item identification."""
        return "barcode" if self.level == "guide" else "gene_name"
    
    @property
    def label_col(self) -> str:
        """Column for labeling."""
        return "gene_name"
    
    def run(self) -> AnalysisResult:
        """Execute aggregated-level analysis."""
        self.log_start(f"Generating {self.level.capitalize()}-Level UMAPs")
        result = AnalysisResult()
        
        if self.df.empty:
            result.add_error(f"No {self.level}-level data available")
            return result
        
        # Get feature columns
        feature_cols = self._get_aggregated_feature_columns()
        
        if len(feature_cols) < 2:
            result.add_error(f"Not enough numeric features for {self.level}-level UMAP")
            return result
        
        logger.info(f"Using {len(feature_cols)} features for {self.level}-level UMAP")
        
        # Prepare features
        features = self.df[feature_cols].copy()
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Generate UMAP
        embedding, clusters_dict, plot_df = self._generate_umap(features)
        
        if embedding is None:
            result.add_error("Failed to generate UMAP embedding")
            return result
        
        # Store results
        result.data["embedding"] = embedding
        result.data["clusters"] = clusters_dict
        result.data["plot_df"] = plot_df
        
        # Generate plots
        self._generate_plots(plot_df, features, result)
        
        # Save embeddings to AnnData
        self._save_to_anndata(embedding, clusters_dict, plot_df)
        
        # Save embedding data CSV
        plot_df.to_csv(self.output_dir / f"{self.level}_umap_data.csv", index=False)
        result.add_file(self.output_dir / f"{self.level}_umap_data.csv")
        
        self.log_complete()
        return result
    
    def _get_aggregated_feature_columns(self) -> List[str]:
        """Get feature columns appropriate for aggregated analysis."""
        metadata_cols = {
            "barcode", "sgRNA", "gene_name", "gene_effect", "NCBI_ID",
            "n_cells", "n_guides", "well"
        }
        
        feature_cols = [
            c for c in self.df.columns
            if pd.api.types.is_numeric_dtype(self.df[c])
            and c not in metadata_cols
            and not c.endswith("_count")
        ]
        return feature_cols
    
    def _generate_umap(
        self, features: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray], pd.DataFrame]:
        """Generate UMAP embedding and clustering."""
        # Initialize engines
        embedding_engine = EmbeddingEngine(use_cuml=self.config.use_cuml)
        clustering_engine = ClusteringEngine(use_cuml=self.config.use_cuml)
        cache = EmbeddingCache(self.data.cache_dir)
        
        n_items = len(self.df)
        cache_key = f"{self.data.experiment}_{self.level}_level"
        
        # Check cache
        cached = cache.load(cache_key, n_items)
        clustering_methods = self.config.get_clustering_methods()
        
        if cached is not None:
            embedding = cached['embedding']
            clusters_dict = {}
            
            for method in clustering_methods:
                cache_cluster_key = f'clusters_{method}'
                if cache_cluster_key in cached:
                    clusters_dict[method] = cached[cache_cluster_key]
        else:
            # Compute embedding
            logger.info(f"Running UMAP on {len(self.df)} {self.level}s...")
            
            try:
                embedding = embedding_engine.compute_umap(features.values, scale=True)
            except Exception as e:
                logger.error(f"UMAP failed: {e}")
                return None, {}, pd.DataFrame()
            
            # Run clustering
            clusters_dict = clustering_engine.cluster_all(embedding, clustering_methods)
            
            # Save to cache
            cache.save(cache_key, embedding, clusters_dict, n_items)
        
        # Build plot DataFrame
        plot_df = self.df.copy()
        plot_df["umap_1"] = embedding[:, 0]
        plot_df["umap_2"] = embedding[:, 1]
        
        for method, clusters in clusters_dict.items():
            plot_df[f"cluster_{method}"] = "c" + clusters.astype(str)
        
        # Use first method as primary
        primary_method = clustering_methods[0]
        plot_df["cluster"] = plot_df[f"cluster_{primary_method}"]
        
        return embedding, clusters_dict, plot_df
    
    def _generate_plots(
        self,
        plot_df: pd.DataFrame,
        features: pd.DataFrame,
        result: AnalysisResult,
    ) -> None:
        """Generate visualization plots."""
        logger.info(f"Generating {self.level}-level plot suite...")
        
        clustering_methods = self.config.get_clustering_methods()
        
        # Try to use adjustText
        try:
            from adjustText import adjust_text
            use_adjust_text = True
        except ImportError:
            use_adjust_text = False
        
        # 1. Cluster plots for each method
        for method in clustering_methods:
            self._plot_cluster_umap(plot_df, f"cluster_{method}", method, use_adjust_text, result)
        
        # 2. Gene effect colored UMAP
        self._plot_gene_effect_umap(plot_df, use_adjust_text, result)
        
        # 3. Cell count colored UMAP
        self._plot_n_cells_umap(plot_df, result)
        
        # 4. NTC vs Perturbed
        self._plot_ntc_vs_perturbed(plot_df, result)
        
        # 5. Feature importance
        self._analyze_feature_importance(plot_df, features, result)
        
        # 6. Feature volcano (vs NTC)
        self._plot_feature_volcano(features, plot_df, use_adjust_text, result)
        
        # 7. Per-organelle UMAPs
        self._generate_organelle_umaps(plot_df, features, use_adjust_text, result)
    
    def _plot_cluster_umap(
        self,
        plot_df: pd.DataFrame,
        cluster_col: str,
        method: str,
        use_adjust_text: bool,
        result: AnalysisResult,
    ) -> None:
        """Generate cluster-colored UMAP."""
        fig, ax = plt.subplots(figsize=(14, 12))
        
        palette = create_cluster_palette(plot_df[cluster_col])
        unique_clusters = sorted(plot_df[cluster_col].unique())
        n_clusters = len([c for c in unique_clusters if c != "c-1"])
        
        # Plot noise first
        noise_df = plot_df[plot_df[cluster_col] == "c-1"]
        if len(noise_df) > 0:
            ax.scatter(
                noise_df["umap_1"], noise_df["umap_2"],
                c="lightgray", s=40, alpha=0.3, label="Noise"
            )
        
        # Plot clustered points
        clustered_df = plot_df[plot_df[cluster_col] != "c-1"]
        ax.scatter(
            clustered_df["umap_1"], clustered_df["umap_2"],
            c=[palette.get(c, "gray") for c in clustered_df[cluster_col]],
            s=60, alpha=0.8,
        )
        
        # Add annotations
        self._add_outlier_labels(ax, plot_df, cluster_col, use_adjust_text)
        
        ax.set_title(
            f"{self.level.capitalize()}-Level UMAP ({method.upper()} Clustering, {n_clusters} clusters)",
            fontsize=14
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        
        path = save_figure(fig, self.output_dir / f"{self.level}_umap_{method}.png")
        result.add_file(path)
    
    def _add_outlier_labels(
        self,
        ax: plt.Axes,
        plot_df: pd.DataFrame,
        cluster_col: str,
        use_adjust_text: bool,
    ) -> None:
        """Add labels to outlier points."""
        if self.label_col not in plot_df.columns:
            return
        
        # Find outliers
        center_x, center_y = plot_df["umap_1"].mean(), plot_df["umap_2"].mean()
        distances = np.sqrt(
            (plot_df["umap_1"] - center_x)**2 + 
            (plot_df["umap_2"] - center_y)**2
        )
        outlier_threshold = np.percentile(distances, 92)
        outliers_mask = distances > outlier_threshold
        
        # Also label cluster centers
        unique_clusters = sorted(plot_df[cluster_col].unique())
        for cluster in unique_clusters:
            if cluster == "c-1":
                continue
            cluster_points = plot_df[plot_df[cluster_col] == cluster]
            if len(cluster_points) > 0:
                cx = cluster_points["umap_1"].mean()
                cy = cluster_points["umap_2"].mean()
                dists = np.sqrt((cluster_points["umap_1"] - cx)**2 + (cluster_points["umap_2"] - cy)**2)
                center_idx = dists.idxmin()
                outliers_mask.loc[center_idx] = True
        
        texts = []
        for idx, row in plot_df[outliers_mask].iterrows():
            label_text = str(row[self.label_col])
            if label_text and label_text != "nan":
                texts.append(ax.text(row["umap_1"], row["umap_2"], label_text, fontsize=8, alpha=0.9))
        
        if use_adjust_text and texts:
            try:
                from adjustText import adjust_text
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))
            except Exception:
                pass
    
    def _plot_gene_effect_umap(
        self,
        plot_df: pd.DataFrame,
        use_adjust_text: bool,
        result: AnalysisResult,
    ) -> None:
        """Generate gene effect colored UMAP."""
        if "gene_effect" not in plot_df.columns:
            return
        
        gene_effect = pd.to_numeric(plot_df["gene_effect"], errors="coerce")
        if gene_effect.notna().sum() < 10:
            return
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Plot NTC as gray
        ntc_mask = self.get_ntc_mask(plot_df)
        ntc_df = plot_df[ntc_mask]
        pert_df = plot_df[~ntc_mask]
        
        if len(ntc_df) > 0:
            ax.scatter(ntc_df["umap_1"], ntc_df["umap_2"], c="lightgray", 
                      s=self.plot_config.point_size * 3, alpha=0.5, label="NTC")
        
        # Plot perturbed colored by gene effect
        pert_effect = gene_effect[~ntc_mask]
        scatter = ax.scatter(
            pert_df["umap_1"], pert_df["umap_2"],
            c=pert_effect, cmap="RdBu_r", s=self.plot_config.point_size * 4, alpha=0.8,
            vmin=-1, vmax=0.5,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Gene Effect (CERES)")
        
        # Label extreme effects
        if self.level == "gene" and self.label_col in pert_df.columns:
            pert_sorted = pert_df.copy()
            pert_sorted["_effect"] = pert_effect
            pert_sorted = pert_sorted.dropna(subset=["_effect"])
            
            top_essential = pert_sorted.nsmallest(10, "_effect")
            top_nonessential = pert_sorted.nlargest(5, "_effect")
            
            texts = []
            for idx, row in pd.concat([top_essential, top_nonessential]).iterrows():
                label_text = str(row[self.label_col])
                if label_text and label_text != "nan":
                    texts.append(ax.text(row["umap_1"], row["umap_2"], label_text, fontsize=8))
            
            if use_adjust_text and texts:
                try:
                    from adjustText import adjust_text
                    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))
                except Exception:
                    pass
        
        ax.set_title(f"{self.level.capitalize()}-Level UMAP Colored by Gene Effect", fontsize=14)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        plt.tight_layout()
        
        path = save_figure(fig, self.output_dir / f"{self.level}_umap_gene_effect.png")
        result.add_file(path)
    
    def _plot_n_cells_umap(self, plot_df: pd.DataFrame, result: AnalysisResult) -> None:
        """Generate cell count colored UMAP."""
        if "n_cells" not in plot_df.columns:
            return
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        scatter = ax.scatter(
            plot_df["umap_1"], plot_df["umap_2"],
            c=np.log10(plot_df["n_cells"] + 1),
            cmap="viridis", s=60, alpha=0.8,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("log10(n_cells + 1)")
        
        ax.set_title(f"{self.level.capitalize()}-Level UMAP Colored by Cell Count", fontsize=14)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        plt.tight_layout()
        
        path = save_figure(fig, self.output_dir / f"{self.level}_umap_n_cells.png")
        result.add_file(path)
    
    def _plot_ntc_vs_perturbed(self, plot_df: pd.DataFrame, result: AnalysisResult) -> None:
        """Generate NTC vs perturbed comparison plot."""
        ntc_mask = self.get_ntc_mask(plot_df)
        
        if ntc_mask.sum() == 0 or (~ntc_mask).sum() == 0:
            return
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        pert_df = plot_df[~ntc_mask]
        ntc_df = plot_df[ntc_mask]
        
        ax.scatter(
            pert_df["umap_1"], pert_df["umap_2"],
            c="steelblue", s=50, alpha=0.6,
            label=f"Perturbed ({len(pert_df)})"
        )
        ax.scatter(
            ntc_df["umap_1"], ntc_df["umap_2"],
            c="orange", s=80, alpha=0.9,
            label=f"NTC ({len(ntc_df)})",
            edgecolors="black", linewidths=0.5,
        )
        
        ax.set_title(f"{self.level.capitalize()}-Level UMAP: NTC vs Perturbed", fontsize=14)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(markerscale=1.5)
        plt.tight_layout()
        
        path = save_figure(fig, self.output_dir / f"{self.level}_umap_ntc_vs_perturbed.png")
        result.add_file(path)
    
    def _analyze_feature_importance(
        self,
        plot_df: pd.DataFrame,
        features: pd.DataFrame,
        result: AnalysisResult,
    ) -> None:
        """Analyze which features drive UMAP structure."""
        logger.info("Computing feature correlations with UMAP coordinates...")
        
        correlations = []
        for col in features.columns:
            corr_umap1, _ = spearmanr(features[col], plot_df["umap_1"])
            corr_umap2, _ = spearmanr(features[col], plot_df["umap_2"])
            max_corr = max(abs(corr_umap1), abs(corr_umap2))
            correlations.append({
                "feature": col,
                "corr_umap1": corr_umap1,
                "corr_umap2": corr_umap2,
                "max_abs_corr": max_corr,
            })
        
        corr_df = pd.DataFrame(correlations).sort_values("max_abs_corr", ascending=False)
        corr_df.to_csv(self.output_dir / f"{self.level}_feature_umap_correlations.csv", index=False)
        result.add_file(self.output_dir / f"{self.level}_feature_umap_correlations.csv")
        
        # Plot top features
        top_features = corr_df.head(20)
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ["steelblue" if c > 0 else "coral" for c in top_features["corr_umap1"]]
        ax.barh(range(len(top_features)), top_features["max_abs_corr"], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"])
        ax.invert_yaxis()
        ax.set_xlabel("Max Absolute Spearman Correlation with UMAP")
        ax.set_title(f"Top Features Driving {self.level.capitalize()}-Level UMAP Structure")
        plt.tight_layout()
        
        path = save_figure(fig, self.output_dir / f"{self.level}_top_features_umap.png")
        result.add_file(path)
    
    def _plot_feature_volcano(
        self,
        features: pd.DataFrame,
        plot_df: pd.DataFrame,
        use_adjust_text: bool,
        result: AnalysisResult,
    ) -> None:
        """Generate volcano plot for features vs NTC."""
        ntc_mask = self.get_ntc_mask(plot_df)
        
        if ntc_mask.sum() < 3 or (~ntc_mask).sum() < 3:
            return
        
        logger.info(f"Computing {self.level}-level volcano plot...")
        
        ntc_features = features.loc[ntc_mask]
        pert_features = features.loc[~ntc_mask]
        
        volcano_data = []
        for col in features.columns:
            ntc_vals = ntc_features[col].dropna()
            pert_vals = pert_features[col].dropna()
            
            if len(ntc_vals) >= 2 and len(pert_vals) >= 2:
                try:
                    stat, pval = ttest_ind(pert_vals, ntc_vals)
                    fold_change = pert_vals.mean() - ntc_vals.mean()
                    volcano_data.append({
                        "feature": col,
                        "fold_change": fold_change,
                        "pvalue": pval,
                        "-log10_pvalue": -np.log10(pval + 1e-300),
                    })
                except Exception:
                    pass
        
        if not volcano_data:
            return
        
        volcano_df = pd.DataFrame(volcano_data)
        volcano_df.to_csv(self.output_dir / f"{self.level}_feature_volcano.csv", index=False)
        result.add_file(self.output_dir / f"{self.level}_feature_volcano.csv")
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sig_threshold = 0.05 / len(volcano_df)  # Bonferroni
        fc_threshold = 0.5
        
        sig_up = (volcano_df["pvalue"] < sig_threshold) & (volcano_df["fold_change"] > fc_threshold)
        sig_down = (volcano_df["pvalue"] < sig_threshold) & (volcano_df["fold_change"] < -fc_threshold)
        
        ax.scatter(
            volcano_df.loc[~(sig_up | sig_down), "fold_change"],
            volcano_df.loc[~(sig_up | sig_down), "-log10_pvalue"],
            c="gray", alpha=0.5, s=30, label="Not significant"
        )
        ax.scatter(
            volcano_df.loc[sig_up, "fold_change"],
            volcano_df.loc[sig_up, "-log10_pvalue"],
            c="red", alpha=0.7, s=50, label=f"Up ({sig_up.sum()})"
        )
        ax.scatter(
            volcano_df.loc[sig_down, "fold_change"],
            volcano_df.loc[sig_down, "-log10_pvalue"],
            c="blue", alpha=0.7, s=50, label=f"Down ({sig_down.sum()})"
        )
        
        # Label top significant
        top_sig = volcano_df.nlargest(10, "-log10_pvalue")
        texts = []
        for _, row in top_sig.iterrows():
            texts.append(ax.text(row["fold_change"], row["-log10_pvalue"], row["feature"], fontsize=7))
        
        if use_adjust_text and texts:
            try:
                from adjustText import adjust_text
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))
            except Exception:
                pass
        
        ax.axhline(-np.log10(sig_threshold), color="gray", linestyle="--", alpha=0.5)
        ax.axvline(fc_threshold, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(-fc_threshold, color="gray", linestyle="--", alpha=0.3)
        
        ax.set_xlabel("Mean Difference (Perturbed - NTC)")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title(f"{self.level.capitalize()}-Level Feature Volcano: Perturbed vs NTC")
        ax.legend()
        plt.tight_layout()
        
        path = save_figure(fig, self.output_dir / f"{self.level}_feature_volcano.png")
        result.add_file(path)
    
    def _generate_organelle_umaps(
        self,
        plot_df: pd.DataFrame,
        features: pd.DataFrame,
        use_adjust_text: bool,
        result: AnalysisResult,
    ) -> None:
        """Generate per-organelle UMAPs."""
        logger.info(f"Generating per-organelle {self.level}-level UMAPs...")
        
        organelle_features = self.group_features_by_organelle(features.columns.tolist())
        
        if len(organelle_features) < 2:
            logger.info("Skipping per-organelle UMAPs: need at least 2 organelle groups")
            return
        
        organelle_dir = self.output_dir / "per_organelle"
        organelle_dir.mkdir(exist_ok=True)
        
        embedding_engine = EmbeddingEngine(use_cuml=self.config.use_cuml)
        
        for organelle, cols in sorted(organelle_features.items()):
            if len(cols) < 3:
                continue
            
            org_features = features[cols].copy()
            org_features = org_features.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Check variance
            variances = org_features.var()
            if (variances == 0).all():
                continue
            
            org_features = org_features.loc[:, variances > 0]
            if org_features.shape[1] < 2:
                continue
            
            # Run UMAP
            logger.info(f"Running UMAP for {organelle} ({len(cols)} features)...")
            try:
                n_neighbors = min(15, len(org_features) - 1)
                org_embedding = embedding_engine.compute_umap(
                    org_features.values, n_neighbors=n_neighbors, scale=True
                )
            except Exception as e:
                logger.warning(f"Failed for {organelle}: {e}")
                continue
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 10))
            
            if "gene_effect" in plot_df.columns:
                gene_effect = pd.to_numeric(plot_df["gene_effect"], errors="coerce")
                scatter = ax.scatter(
                    org_embedding[:, 0], org_embedding[:, 1],
                    c=gene_effect, cmap="RdBu_r", s=50, alpha=0.7,
                    vmin=-1, vmax=0.5,
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label("Gene Effect")
            else:
                ax.scatter(org_embedding[:, 0], org_embedding[:, 1], 
                          s=self.plot_config.point_size * 3, alpha=0.7, c="steelblue")
            
            ax.set_title(f"{self.level.capitalize()}-Level UMAP: {organelle.upper()} Features ({len(cols)} features)")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            plt.tight_layout()
            
            path = save_figure(fig, organelle_dir / f"{self.level}_umap_{organelle}.png")
            result.add_file(path)
    
    def _save_to_anndata(
        self,
        embedding: np.ndarray,
        clusters_dict: Dict[str, np.ndarray],
        plot_df: pd.DataFrame,
    ) -> None:
        """Save embeddings to AnnData."""
        if self.level not in self.data.adata:
            return
        
        adata = self.data.adata[self.level]
        adata.obsm["X_umap"] = embedding
        
        for method in self.config.get_clustering_methods():
            cluster_col = f"cluster_{method}"
            if cluster_col in plot_df.columns:
                adata.obs[cluster_col] = plot_df[cluster_col].values
        
        # Save
        h5ad_path = self.data.analysis_path / f"{self.data.experiment}_{self.level}_features.h5ad"
        adata.write_h5ad(h5ad_path)
        logger.info(f"Saved {self.level}-level embeddings to AnnData: {h5ad_path}")
