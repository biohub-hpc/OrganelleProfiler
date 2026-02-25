"""
Embedding Visualization Stage: Comprehensive annotated UMAP/PCA plots.

Generates systematic visualizations of embeddings with various annotations:
- Density (KDE overlays)
- Clustering (all methods)
- Spatial (radial position, well, tile)
- Morphological (cell size, shape metrics)
- Biological (gene effects, NTC vs control)
- Feature-driven (top variable features, per-organelle features)

This provides a complete library of embedding visualizations for exploration.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from scipy.stats import gaussian_kde
import logging

from .fe_graphs_stage_base import BaseStage, StageResult
from ..plotting.fe_graphs_utils import save_figure

logger = logging.getLogger(__name__)


class EmbeddingVisualizationStage(BaseStage):
    """
    Generate comprehensive annotated visualizations of embeddings.
    
    Runs after embedding stage to create a systematic library of plots
    showing embeddings colored/annotated by various metadata and features.
    
    Output organization:
    - pca/ - PCA projections with annotations
    - umap_all_features/ - Overall UMAP with annotations
    - umap_per_organelle/{organelle}/ - Per-organelle UMAPs with annotations
    """
    
    STAGE_NUMBER = 3.5  # Between embedding (3) and positive controls (4)
    STAGE_NAME = "embedding_visualization"
    
    def run(self) -> StageResult:
        """Generate annotated embedding visualizations."""
        self.log_start("Generating annotated embedding visualizations")
        result = StageResult()

        # Check if visualizations already exist (only skip if --skip-complete was passed)
        if self.config.skip_complete:
            umap_dir = self.output_dir / "umap_all_features_annotated"
            sentinel_files = [
                umap_dir / "density" / "density.png",
                umap_dir / "biological" / "gene_effect.png",
            ]

            if all(f.exists() for f in sentinel_files):
                logger.info("Visualizations already exist - skipping (--skip-complete)")
                logger.info(f"  Output directory: {self.output_dir}")
                # Still return success and list existing files
                for root, dirs, files in os.walk(self.output_dir):
                    for file in files:
                        if file.endswith('.png'):
                            result.add_file(Path(root) / file)
                return result
        
        # Get embeddings from upstream or cache
        umap_embedding = None
        pca_embedding = None
        organelle_embeddings = {}
        df = None
        features = None
        
        if "embedding" in self.upstream:
            # Get from upstream
            embedding_data = self.upstream["embedding"].data
            umap_embedding = embedding_data.get("umap_embedding")
            pca_embedding = embedding_data.get("pca_result")
            organelle_embeddings = embedding_data.get("organelle_embeddings", {})
            df = embedding_data.get("df")
            features = embedding_data.get("features")
        else:
            # Embedding stage was skipped - try to load from cache
            logger.info("Embedding stage not in upstream, attempting to load from cache...")
            from ..core.fe_graphs_cache import EmbeddingCache
            
            df = self.df.copy()  # Use current df from base stage
            features = self.get_features(df)
            
            cache = EmbeddingCache(self.data.cache_dir)
            cache_key = f"{self.data.experiment}_{self.level}_all_features"
            cached_result = cache.load(cache_key, n_cells=len(df))
            
            if cached_result is None:
                result.add_error("No embedding stage results and no cached embedding found. Run embedding stage first.")
                return result
            
            umap_embedding = cached_result["embedding"]
            pca_embedding = cached_result.get("pca")
            
            # Try to load per-organelle embeddings from cache too
            organelle_groups = self.data.organelle_groups_all_levels.get(self.level, {})
            for org_name in organelle_groups.keys():
                org_cache_key = f"{self.data.experiment}_{self.level}_organelle_{org_name}"
                # Don't validate n_cells for organelle embeddings - they have variable length due to NaN dropping
                # Just try to load and we'll validate the shape later
                cache_path = cache._get_cache_path(org_cache_key)
                if cache_path.exists():
                    try:
                        org_cached = np.load(cache_path, allow_pickle=True)
                        # Cache only stores the embedding, not indices
                        # This will be handled by the old format path in processing
                        organelle_embeddings[org_name] = org_cached["embedding"]
                        logger.info(f"  Loaded {org_name} embedding from cache (shape: {org_cached['embedding'].shape})")
                    except Exception as e:
                        logger.warning(f"  Failed to load {org_name} from cache: {e}")
            
            logger.info(f"Loaded cached UMAP embedding: {umap_embedding.shape}")
            if pca_embedding is not None:
                logger.info(f"Loaded cached PCA embedding: {pca_embedding.shape}")
            if organelle_embeddings:
                logger.info(f"Loaded {len(organelle_embeddings)} per-organelle embeddings from cache")
        
        if umap_embedding is None or df is None:
            result.add_error("Missing embedding data")
            return result
        
        # Build base DataFrame with coordinates
        plot_df = df.copy()
        plot_df["umap_1"] = umap_embedding[:, 0]
        plot_df["umap_2"] = umap_embedding[:, 1]
        
        if pca_embedding is not None and len(pca_embedding) == len(df):
            plot_df["pca_1"] = pca_embedding[:, 0]
            plot_df["pca_2"] = pca_embedding[:, 1]
        
        # Get clustering data if available
        clusters_dict = {}
        if "clustering" in self.upstream:
            clusters_dict = self.upstream["clustering"].data.get("clusters_dict", {})
        
        # 1. Overall UMAP Visualizations (skip PCA - user requested UMAP only)
        umap_dir = self.output_dir / "umap_all_features_annotated"
        umap_dir.mkdir(parents=True, exist_ok=True)
        self._generate_annotated_plots(
            plot_df, features, clusters_dict,
            x_col="umap_1", y_col="umap_2",
            output_dir=umap_dir,
            result=result,
            title_prefix="UMAP (All Features)"
        )
        
        # 2. Per-Organelle UMAP Visualizations
        logger.info(f"Organelle embeddings dict keys: {list(organelle_embeddings.keys())}")
        logger.info(f"Generating annotated plots for {len(organelle_embeddings)} organelle embeddings")
        
        if len(organelle_embeddings) == 0:
            logger.warning("No organelle embeddings found - skipping per-organelle visualizations")
            logger.warning("This means the embedding stage did not generate per-organelle UMAPs")
        
        for org_name, org_data in organelle_embeddings.items():
            # Handle both old format (just array) and new format (dict with 'embedding' and 'indices')
            if isinstance(org_data, dict):
                org_embedding = org_data['embedding']
                org_indices = org_data.get('indices', None)
            else:
                # Old format - just the embedding array
                org_embedding = org_data
                org_indices = None
            
            logger.info(f"  Processing {org_name} embedding (shape: {org_embedding.shape})...")
            
            # Get features for this organelle
            org_features = self._get_organelle_features(features, org_name)
            
            if org_features.empty:
                logger.warning(f"  Skipping {org_name}: no features found")
                continue
            
            # Subset df and features to match embedding
            if org_indices is not None:
                # New format with explicit indices
                org_plot_df = df.loc[org_indices].copy()
                org_features_final = org_features.loc[org_indices]
                logger.info(f"  Using stored indices: {len(org_indices)} samples")
            elif len(org_embedding) != len(df):
                # Old format - try to reconstruct (may not work perfectly)
                logger.warning(f"  No indices stored, attempting to reconstruct subset (embedding: {len(org_embedding)}, df: {len(df)})")
                org_features_subset = org_features.dropna(how='all', axis=1).fillna(0)
                org_features_subset = org_features_subset.drop_duplicates()
                
                if len(org_features_subset) != len(org_embedding):
                    logger.warning(f"  Skipping {org_name}: cannot determine valid indices (subset {len(org_features_subset)} != embedding {len(org_embedding)})")
                    continue
                
                org_plot_df = df.loc[org_features_subset.index].copy()
                org_features_final = org_features.loc[org_features_subset.index]
            else:
                # Perfect match - use full df
                org_plot_df = df.copy()
                org_features_final = org_features
            
            # Add UMAP coordinates
            org_plot_df["umap_1"] = org_embedding[:, 0]
            org_plot_df["umap_2"] = org_embedding[:, 1]
            
            org_dir = self.output_dir / f"umap_{org_name}_annotated"
            org_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"  Generating annotated plots for {org_name} ({len(org_features_final.columns)} features, {len(org_plot_df)} samples)...")
            
            self._generate_annotated_plots(
                org_plot_df, org_features_final, clusters_dict,
                x_col="umap_1", y_col="umap_2",
                output_dir=org_dir,
                result=result,
                title_prefix=f"UMAP ({org_name})"
            )
            
            # NEW: Per-organelle cluster discovery and analysis
            logger.info(f"  Running cluster analysis for {org_name}...")
            self._analyze_organelle_clusters(
                org_plot_df, org_features_final, org_embedding, org_name, org_dir, result
            )
        
        self.log_complete(result)
        return result
    
    def _generate_annotated_plots(
        self,
        plot_df: pd.DataFrame,
        features: Optional[pd.DataFrame],
        clusters_dict: Dict[str, np.ndarray],
        x_col: str,
        y_col: str,
        output_dir: Path,
        result: StageResult,
        title_prefix: str = "Embedding",
    ) -> None:
        """
        Generate all annotation types for a given embedding.
        
        Creates subdirectories:
        - density/ - Density heatmaps
        - clustering/ - Cluster assignments
        - spatial/ - Well, tile, radial position
        - morphological/ - Cell size, shape metrics
        - biological/ - Gene effects, NTC/control
        - features/ - Top variable features
        """
        # 1. Density
        density_dir = output_dir / "density"
        density_dir.mkdir(exist_ok=True)
        self._plot_density(plot_df, x_col, y_col, density_dir, result, title_prefix)
        
        # 2. Clustering
        if clusters_dict:
            clustering_dir = output_dir / "clustering"
            clustering_dir.mkdir(exist_ok=True)
            self._plot_clustering(plot_df, clusters_dict, x_col, y_col, clustering_dir, result, title_prefix)
        
        # 3. Spatial
        spatial_dir = output_dir / "spatial"
        spatial_dir.mkdir(exist_ok=True)
        self._plot_spatial(plot_df, x_col, y_col, spatial_dir, result, title_prefix)
        
        # 4. Morphological
        morphological_dir = output_dir / "morphological"
        morphological_dir.mkdir(exist_ok=True)
        self._plot_morphological(plot_df, x_col, y_col, morphological_dir, result, title_prefix)
        
        # 5. Biological
        biological_dir = output_dir / "biological"
        biological_dir.mkdir(exist_ok=True)
        self._plot_biological(plot_df, x_col, y_col, biological_dir, result, title_prefix)
        
        # 6. Top Variable Features
        if features is not None:
            features_dir = output_dir / "top_features"
            features_dir.mkdir(exist_ok=True)
            self._plot_top_features(plot_df, features, x_col, y_col, features_dir, result, title_prefix)
    
    def _plot_density(
        self,
        plot_df: pd.DataFrame,
        x_col: str,
        y_col: str,
        output_dir: Path,
        result: StageResult,
        title_prefix: str,
    ) -> None:
        """Generate density heatmap overlay."""
        fig, ax = plt.subplots(figsize=self.plot_config.figsize_umap)
        
        # Compute KDE
        x = plot_df[x_col].values
        y = plot_df[y_col].values
        
        # Subsample if too many points (KDE is slow)
        if len(x) > 10000:
            indices = np.random.choice(len(x), 10000, replace=False)
            x_kde, y_kde = x[indices], y[indices]
        else:
            x_kde, y_kde = x, y
        
        try:
            kde = gaussian_kde(np.vstack([x_kde, y_kde]))
            
            # Evaluate on grid
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            density = kde(positions).reshape(xx.shape)
            
            # Plot density
            ax.contourf(xx, yy, density, levels=20, cmap="viridis", alpha=0.6)
            ax.scatter(x, y, s=self.plot_config.point_size * 0.5, c="white", alpha=0.3, rasterized=True)
            
            ax.set_xlabel(x_col.upper().replace("_", " "), fontsize=20, fontweight='bold')
            ax.set_ylabel(y_col.upper().replace("_", " "), fontsize=20, fontweight='bold')
            ax.set_title(f"{title_prefix} - Density", fontsize=22, fontweight='bold', pad=20)
            ax.tick_params(labelsize=16)
            
            path = save_figure(fig, output_dir / "density.png")
            result.add_file(path)
        except Exception as e:
            logger.warning(f"Could not compute density: {e}")
            plt.close(fig)
    
    def _plot_clustering(
        self,
        plot_df: pd.DataFrame,
        clusters_dict: Dict[str, np.ndarray],
        x_col: str,
        y_col: str,
        output_dir: Path,
        result: StageResult,
        title_prefix: str,
    ) -> None:
        """Plot each clustering method."""
        for method, clusters in clusters_dict.items():
            if len(clusters) != len(plot_df):
                continue
            
            fig, ax = plt.subplots(figsize=self.plot_config.figsize_umap)
            
            unique_clusters = np.unique(clusters)
            n_clusters = len(unique_clusters)
            colors = sns.color_palette("husl", n_clusters)
            
            for cluster_id, color in zip(unique_clusters, colors):
                mask = clusters == cluster_id
                ax.scatter(
                    plot_df.loc[mask, x_col],
                    plot_df.loc[mask, y_col],
                    c=[color], s=self.plot_config.point_size,
                    alpha=0.7, label=f"C{cluster_id}",
                    rasterized=True
                )
            
            ax.set_xlabel(x_col.upper().replace("_", " "), fontsize=20, fontweight='bold')
            ax.set_ylabel(y_col.upper().replace("_", " "), fontsize=20, fontweight='bold')
            ax.set_title(f"{title_prefix} - Clustering ({method})", fontsize=22, fontweight='bold', pad=20)
            ax.legend(fontsize=14, markerscale=2.0, framealpha=0.9)
            ax.tick_params(labelsize=16)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=2 if n_clusters > 15 else 1)
            
            plt.tight_layout()
            path = save_figure(fig, output_dir / f"clustering_{method}.png")
            result.add_file(path)
    
    def _plot_spatial(
        self,
        plot_df: pd.DataFrame,
        x_col: str,
        y_col: str,
        output_dir: Path,
        result: StageResult,
        title_prefix: str,
    ) -> None:
        """Plot spatial annotations (radial position, well, tile)."""
        # Radial position (if available)
        if "radial_position_norm" in plot_df.columns:
            self._plot_continuous(
                plot_df, x_col, y_col, "radial_position_norm",
                output_dir, result, title_prefix,
                cmap="coolwarm", title_suffix="Radial Position"
            )
        
        # Well
        if "well" in plot_df.columns:
            self._plot_categorical(
                plot_df, x_col, y_col, "well",
                output_dir, result, title_prefix,
                title_suffix="Well"
            )
        
        # Tile
        if "tile" in plot_df.columns:
            self._plot_categorical(
                plot_df, x_col, y_col, "tile",
                output_dir, result, title_prefix,
                title_suffix="Tile"
            )
    
    def _plot_morphological(
        self,
        plot_df: pd.DataFrame,
        x_col: str,
        y_col: str,
        output_dir: Path,
        result: StageResult,
        title_prefix: str,
    ) -> None:
        """Plot morphological annotations (cell size, shape)."""
        # Cell area (if available)
        size_cols = ["cell_seg_area", "nuclei_area", "cell_area"]
        for col in size_cols:
            if col in plot_df.columns:
                self._plot_continuous(
                    plot_df, x_col, y_col, col,
                    output_dir, result, title_prefix,
                    cmap="viridis", title_suffix=col.replace("_", " ").title()
                )
                break  # Only plot first available size metric
        
        # Nuclear/Cytoplasmic ratio (if available)
        if "nuclei_area" in plot_df.columns and "cell_seg_area" in plot_df.columns:
            plot_df["nc_ratio"] = plot_df["nuclei_area"] / plot_df["cell_seg_area"]
            self._plot_continuous(
                plot_df, x_col, y_col, "nc_ratio",
                output_dir, result, title_prefix,
                cmap="RdYlBu_r", title_suffix="Nuclear/Cytoplasmic Ratio"
            )
    
    def _plot_biological(
        self,
        plot_df: pd.DataFrame,
        x_col: str,
        y_col: str,
        output_dir: Path,
        result: StageResult,
        title_prefix: str,
    ) -> None:
        """Plot biological annotations (gene effects, NTC/control)."""
        # Gene effects (if available from differential stage)
        if "gene_effect" in plot_df.columns:
            self._plot_continuous(
                plot_df, x_col, y_col, "gene_effect",
                output_dir, result, title_prefix,
                cmap="RdBu_r", title_suffix="Gene Effect"
            )
        
        # NTC vs Control
        if "is_ntc" in plot_df.columns:
            fig, ax = plt.subplots(figsize=self.plot_config.figsize_umap)
            
            ntc_mask = plot_df["is_ntc"].astype(bool)
            
            ax.scatter(
                plot_df.loc[~ntc_mask, x_col],
                plot_df.loc[~ntc_mask, y_col],
                c="steelblue", s=self.plot_config.point_size,
                alpha=0.5, label="Perturbed", rasterized=True
            )
            ax.scatter(
                plot_df.loc[ntc_mask, x_col],
                plot_df.loc[ntc_mask, y_col],
                c="orange", s=self.plot_config.point_size,
                alpha=0.7, label="NTC", rasterized=True
            )
            
            ax.set_xlabel(x_col.upper().replace("_", " "), fontsize=20, fontweight='bold')
            ax.set_ylabel(y_col.upper().replace("_", " "), fontsize=20, fontweight='bold')
            ax.set_title(f"{title_prefix} - NTC vs Perturbed", fontsize=22, fontweight='bold', pad=20)
            ax.legend(fontsize=14, markerscale=2.0, framealpha=0.9)
            ax.tick_params(labelsize=16)
            
            path = save_figure(fig, output_dir / "ntc_vs_perturbed.png")
            result.add_file(path)
    
    def _plot_top_features(
        self,
        plot_df: pd.DataFrame,
        features: pd.DataFrame,
        x_col: str,
        y_col: str,
        output_dir: Path,
        result: StageResult,
        title_prefix: str,
        top_n: int = 10,
    ) -> None:
        """
        Plot embeddings colored by top informative features.
        
        Selects features based on:
        1. High variance (captures biological signal)
        2. Spatial structure in the embedding (correlation with UMAP neighbors)
        
        This ensures we show features that actually have gradients visible in the embedding.
        """
        from scipy.stats import spearmanr
        from sklearn.neighbors import NearestNeighbors
        
        # Compute variance for each feature (captures signal-to-noise)
        feature_vars = features.var()
        
        # Filter out zero-variance features
        valid_features = feature_vars[feature_vars > 0]
        
        if len(valid_features) == 0:
            logger.warning("No features with non-zero variance")
            return
        
        # Normalize variance scores (0-1)
        var_scores = (valid_features - valid_features.min()) / (valid_features.max() - valid_features.min())
        
        # Compute spatial structure score: how well does each feature correlate with embedding structure?
        # Use k-nearest neighbors in embedding space and check if feature values are smooth
        embedding_coords = plot_df[[x_col, y_col]].values
        
        # Sample if too large (for speed)
        if len(embedding_coords) > 10000:
            sample_idx = np.random.choice(len(embedding_coords), 10000, replace=False)
            embedding_sample = embedding_coords[sample_idx]
            features_sample = features.iloc[sample_idx]
        else:
            embedding_sample = embedding_coords
            features_sample = features
        
        # Find k-nearest neighbors in embedding space
        knn = NearestNeighbors(n_neighbors=min(30, len(embedding_sample) // 10))
        knn.fit(embedding_sample)
        distances, indices = knn.kneighbors(embedding_sample)
        
        # For each feature, compute smoothness: correlation with neighbor average
        smoothness_scores = {}
        for feat in valid_features.index:
            if feat not in features_sample.columns:
                continue
            
            feat_values = features_sample[feat].values
            
            # Skip if all NaN
            if np.all(np.isnan(feat_values)):
                continue
            
            # Compute neighbor average for each point
            neighbor_avgs = []
            point_values = []
            
            for i, neighbor_indices in enumerate(indices):
                neighbor_vals = feat_values[neighbor_indices[1:]]  # Exclude self (index 0)
                neighbor_vals_clean = neighbor_vals[~np.isnan(neighbor_vals)]
                
                if len(neighbor_vals_clean) > 0 and not np.isnan(feat_values[i]):
                    neighbor_avgs.append(np.mean(neighbor_vals_clean))
                    point_values.append(feat_values[i])
            
            if len(point_values) > 10:
                # Correlation between point value and neighbor average
                # High correlation = smooth gradient = visible structure
                corr, _ = spearmanr(point_values, neighbor_avgs)
                smoothness_scores[feat] = abs(corr) if not np.isnan(corr) else 0
            else:
                smoothness_scores[feat] = 0
        
        if not smoothness_scores:
            logger.warning("Could not compute smoothness scores for any features")
            # Fall back to variance only
            top_features = feature_vars.nlargest(top_n).index
        else:
            # Normalize smoothness scores
            smoothness_series = pd.Series(smoothness_scores)
            if smoothness_series.max() > 0:
                smoothness_norm = (smoothness_series - smoothness_series.min()) / (smoothness_series.max() - smoothness_series.min())
            else:
                smoothness_norm = smoothness_series
            
            # Combined score: 50% variance, 50% smoothness
            combined_scores = {}
            for feat in smoothness_norm.index:
                if feat in var_scores.index:
                    combined_scores[feat] = 0.5 * var_scores[feat] + 0.5 * smoothness_norm[feat]
            
            # Select top features
            combined_series = pd.Series(combined_scores).sort_values(ascending=False)
            top_features = combined_series.head(top_n).index
            
            logger.info(f"  Top {top_n} informative features selected (var + smoothness)")
        
        # Plot top features
        for feat in top_features:
            if feat in plot_df.columns or feat in features.columns:
                # Add feature to plot_df if not already there
                if feat not in plot_df.columns:
                    plot_df[feat] = features[feat]
                
                self._plot_continuous(
                    plot_df, x_col, y_col, feat,
                    output_dir, result, title_prefix,
                    cmap="viridis", title_suffix=feat,
                    annotate_extremes=True  # Add gene/guide/cell annotations for extreme values
                )
    
    def _plot_continuous(
        self,
        plot_df: pd.DataFrame,
        x_col: str,
        y_col: str,
        value_col: str,
        output_dir: Path,
        result: StageResult,
        title_prefix: str,
        cmap: str = "viridis",
        title_suffix: str = None,
        annotate_extremes: bool = False,
        n_extremes: int = 20,  # 20 top + 20 bottom = 40 total annotations
    ) -> None:
        """
        Plot embedding colored by continuous variable.
        
        Parameters
        ----------
        annotate_extremes : bool
            If True, identify and annotate genes/guides/cells with most extreme values
        n_extremes : int
            Number of extreme values to annotate (top and bottom)
        """
        if value_col not in plot_df.columns:
            return
        
        fig, ax = plt.subplots(figsize=self.plot_config.figsize_umap)
        
        values = plot_df[value_col].copy()
        
        # Convert categorical to numeric if needed (matplotlib can't handle categorical colors)
        if pd.api.types.is_categorical_dtype(values) or values.dtype == 'object':
            try:
                values = pd.to_numeric(values, errors='coerce')
                logger.debug(f"Converted {value_col} from categorical to numeric")
            except Exception as e:
                logger.warning(f"Could not convert {value_col} to numeric: {e}")
                plt.close(fig)
                return
        
        # Filter to valid (non-NaN) values
        valid_mask = pd.notna(values)
        
        if valid_mask.sum() == 0:
            plt.close(fig)
            return
        
        # Extract numeric array for matplotlib
        color_values = values[valid_mask].values
        
        scatter = ax.scatter(
            plot_df.loc[valid_mask, x_col],
            plot_df.loc[valid_mask, y_col],
            c=color_values,
            s=self.plot_config.point_size,
            alpha=0.7,
            cmap=cmap,
            rasterized=True
        )
        
        cbar = plt.colorbar(scatter, ax=ax, label=value_col.replace("_", " ").title())
        cbar.ax.tick_params(labelsize=14)
        
        ax.set_xlabel(x_col.upper().replace("_", " "), fontsize=20, fontweight='bold')
        ax.set_ylabel(y_col.upper().replace("_", " "), fontsize=20, fontweight='bold')
        ax.set_title(f"{title_prefix} - {title_suffix or value_col}", fontsize=22, fontweight='bold', pad=20)
        ax.tick_params(labelsize=16)
        
        # Save the clean version WITHOUT annotations first
        safe_name = value_col.replace("/", "_").replace(" ", "_")
        path_clean = save_figure(fig, output_dir / f"{safe_name}.png")
        result.add_file(path_clean)
        
        # Now add annotations and save annotated version
        if annotate_extremes and "gene_name" in plot_df.columns:
            valid_data = plot_df[valid_mask].copy()
            valid_data['value'] = values[valid_mask]
            
            # Sort by value to get extremes
            sorted_data = valid_data.sort_values('value')
            
            # Get top and bottom n_extremes
            bottom_extremes = sorted_data.head(n_extremes)
            top_extremes = sorted_data.tail(n_extremes)
            extremes = pd.concat([bottom_extremes, top_extremes])
            
            # Collect annotation data
            annotation_data = []
            
            # Aggregate by gene if at cell/guide level
            if self.level in ["cell", "guide"]:
                # Group by gene and show the gene with most extreme average value
                gene_stats = valid_data.groupby('gene_name')['value'].agg(['mean', 'count'])
                gene_stats = gene_stats[gene_stats['count'] >= 3]  # At least 3 samples
                
                if len(gene_stats) > 0:
                    sorted_genes = gene_stats.sort_values('mean')
                    top_genes = sorted_genes.tail(min(n_extremes, len(sorted_genes)))
                    bottom_genes = sorted_genes.head(min(n_extremes, len(sorted_genes)))
                    extreme_genes = list(top_genes.index) + list(bottom_genes.index)
                    
                    # Collect annotation data
                    for gene in extreme_genes:
                        gene_data = valid_data[valid_data['gene_name'] == gene]
                        centroid_x = gene_data[x_col].mean()
                        centroid_y = gene_data[y_col].mean()
                        mean_val = gene_data['value'].mean()
                        n_items = len(gene_data)
                        
                        annotation_data.append({
                            'x': centroid_x,
                            'y': centroid_y,
                            'label': f"{gene}\n({mean_val:.2f}, n={n_items})",
                            'value': mean_val
                        })
            else:
                # Gene level - annotate individual gene extremes
                for _, row in extremes.iterrows():
                    annotation_data.append({
                        'x': row[x_col],
                        'y': row[y_col],
                        'label': f"{row['gene_name']}\n({row['value']:.2f})",
                        'value': row['value']
                    })
            
            # Add visual markers and annotations
            if annotation_data:
                # Get colormap for matching annotation colors to data
                import matplotlib.cm as cm
                from matplotlib.colors import Normalize
                from sklearn.cluster import DBSCAN
                
                # Normalize values to match the scatter plot's color range
                # Use the FULL data range, not just the extreme values
                norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
                colormap = cm.get_cmap(cmap)
                
                # Debug: Log color mapping info
                logger.info(f"  Color normalization: vmin={color_values.min():.3f}, vmax={color_values.max():.3f}")
                logger.info(f"  Annotation value range: {min(a['value'] for a in annotation_data):.3f} to {max(a['value'] for a in annotation_data):.3f}")
                logger.info(f"  Number of annotations: {len(annotation_data)}")
                
                # Group nearby annotations using spatial clustering
                # Extract positions
                positions = np.array([[a['x'], a['y']] for a in annotation_data])
                
                # Normalize positions to 0-1 range for clustering
                x_range = positions[:, 0].max() - positions[:, 0].min()
                y_range = positions[:, 1].max() - positions[:, 1].min()
                positions_norm = positions.copy()
                if x_range > 0:
                    positions_norm[:, 0] = (positions[:, 0] - positions[:, 0].min()) / x_range
                if y_range > 0:
                    positions_norm[:, 1] = (positions[:, 1] - positions[:, 1].min()) / y_range
                
                # Cluster annotations that are close together
                # eps=0.15 means ~15% of the plot width/height
                clustering = DBSCAN(eps=0.15, min_samples=1).fit(positions_norm)
                cluster_labels = clustering.labels_
                
                logger.info(f"  Grouped {len(annotation_data)} annotations into {len(set(cluster_labels))} clusters")
                
                # Group annotations by cluster
                clustered_annotations = {}
                for i, label in enumerate(cluster_labels):
                    if label not in clustered_annotations:
                        clustered_annotations[label] = []
                    clustered_annotations[label].append(annotation_data[i])
                
                # Create grouped annotations
                texts = []
                for cluster_id, cluster_annots in clustered_annotations.items():
                    # Sort by value (high to low) within cluster
                    cluster_annots = sorted(cluster_annots, key=lambda a: a['value'], reverse=True)
                    
                    # Compute cluster centroid for arrow placement
                    centroid_x = np.mean([a['x'] for a in cluster_annots])
                    centroid_y = np.mean([a['y'] for a in cluster_annots])
                    
                    # Get average color for the cluster
                    avg_value = np.mean([a['value'] for a in cluster_annots])
                    normalized_val = norm(avg_value)
                    color_rgba = colormap(normalized_val)
                    
                    # Determine text color based on background brightness
                    r, g, b = color_rgba[:3]
                    luminance = 0.299 * r + 0.587 * g + 0.114 * b
                    text_color = 'white' if luminance < 0.5 else 'black'
                    
                    # Create multi-line label with all genes in cluster
                    if len(cluster_annots) == 1:
                        # Single annotation - use original label
                        label_text = cluster_annots[0]['label']
                    else:
                        # Multiple annotations - create vertical list
                        # Extract just the gene names (first part before \n)
                        gene_lines = []
                        for a in cluster_annots:
                            gene_name = a['label'].split('\n')[0]
                            value = a['value']
                            gene_lines.append(f"{gene_name} ({value:.2f})")
                        label_text = '\n'.join(gene_lines)
                    
                    # Log first few clusters
                    if cluster_id < 2:
                        logger.info(f"  Cluster {cluster_id}: {len(cluster_annots)} genes, avg_value={avg_value:.3f}, rgba={color_rgba}")
                    
                    # Add text annotation with color-matched background
                    text = ax.annotate(
                        label_text,
                        xy=(centroid_x, centroid_y),
                        xytext=(60, 60),  # Initial offset - adjustText will optimize
                        textcoords='offset points',
                        fontsize=9 if len(cluster_annots) > 1 else 10,  # Slightly smaller for multi-line
                        fontweight='bold',
                        color=text_color,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=color_rgba, alpha=0.9, 
                                edgecolor='black', linewidth=2),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', 
                                      lw=2, color='black', alpha=0.8),
                        zorder=11,
                        ha='left',
                        va='bottom'
                    )
                    texts.append(text)
                
                # Use adjustText to prevent overlaps between clusters
                try:
                    from adjustText import adjust_text
                    adjust_text(
                        texts,
                        ax=ax,
                        expand_points=(1.5, 1.5),  # Expand repulsion from points
                        expand_text=(1.3, 1.3),    # More repulsion between texts (they're bigger now)
                        expand_objects=(1.3, 1.3), # More repulsion from other objects
                        arrowprops=None,  # Don't override individual arrow properties
                        force_points=(0.5, 0.5),   # Force away from points
                        force_text=(0.7, 0.7),     # Stronger force away from other text
                        only_move={'points': 'xy', 'text': 'xy'},  # Allow full movement
                        lim=1000,  # Max iterations
                    )
                    logger.info("  Applied adjustText to prevent label overlaps")
                except ImportError:
                    logger.info("  adjustText not available, annotations may overlap")
                
                # Save annotated version
                path_annotated = save_figure(fig, output_dir / f"{safe_name}_annotated.png")
                result.add_file(path_annotated)
                logger.info(f"  Saved annotated version with {len(texts)} annotation groups")
                
                # Save a CSV with extreme genes/items
                if self.level in ["cell", "guide"]:
                    extreme_summary = gene_stats.loc[extreme_genes].copy()
                    extreme_summary['gene'] = extreme_summary.index
                    extreme_summary = extreme_summary.sort_values('mean')
                else:
                    extreme_summary = extremes[['gene_name', 'value']].copy()
                    extreme_summary.columns = ['gene', 'value']
                
                csv_path = output_dir / f"{safe_name}_extreme_genes.csv"
                extreme_summary.to_csv(csv_path, index=False)
                result.add_file(csv_path)
                logger.info(f"  Saved extreme genes for {value_col} to {csv_path.name}")
        
        plt.close(fig)
    
    def _plot_categorical(
        self,
        plot_df: pd.DataFrame,
        x_col: str,
        y_col: str,
        value_col: str,
        output_dir: Path,
        result: StageResult,
        title_prefix: str,
        title_suffix: str = None,
    ) -> None:
        """Plot embedding colored by categorical variable."""
        if value_col not in plot_df.columns:
            return
        
        fig, ax = plt.subplots(figsize=self.plot_config.figsize_umap)
        
        unique_values = plot_df[value_col].unique()
        n_categories = len(unique_values)
        
        # Limit number of categories to avoid legend overload
        if n_categories > 50:
            logger.warning(f"Too many categories ({n_categories}) for {value_col}, skipping")
            plt.close(fig)
            return
        
        colors = sns.color_palette("husl", n_categories)
        
        for val, color in zip(unique_values, colors):
            mask = plot_df[value_col] == val
            ax.scatter(
                plot_df.loc[mask, x_col],
                plot_df.loc[mask, y_col],
                c=[color], s=self.plot_config.point_size * 0.7,
                alpha=0.6, label=str(val),
                rasterized=True
            )
        
        ax.set_xlabel(x_col.upper().replace("_", " "), fontsize=20, fontweight='bold')
        ax.set_ylabel(y_col.upper().replace("_", " "), fontsize=20, fontweight='bold')
        ax.set_title(f"{title_prefix} - {title_suffix or value_col}", fontsize=22, fontweight='bold', pad=20)
        ax.tick_params(labelsize=16)
        
        if n_categories <= 20:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12, markerscale=1.5, framealpha=0.9)
        
        plt.tight_layout()
        safe_name = value_col.replace("/", "_").replace(" ", "_")
        path = save_figure(fig, output_dir / f"{safe_name}.png")
        result.add_file(path)
    
    def _get_organelle_features(self, features: pd.DataFrame, organelle_name: str) -> pd.DataFrame:
        """
        Extract features for a specific organelle using the global organelle groups map.
        
        This ensures consistency across all levels (cell/guide/gene) since the map
        handles aggregated feature names (_mean, _std, etc.) correctly.
        """
        # Use the level-specific organelle groups from data context
        if hasattr(self.data, 'organelle_groups_all_levels') and self.data.organelle_groups_all_levels:
            if self.level in self.data.organelle_groups_all_levels:
                organelle_groups = self.data.organelle_groups_all_levels[self.level]
            else:
                organelle_groups = self.data.organelle_groups
        else:
            organelle_groups = self.data.organelle_groups
        
        # Get feature list for this organelle
        if organelle_name in organelle_groups:
            org_feature_list = organelle_groups[organelle_name]
            # Filter to only features that exist in the current feature set
            matching_cols = [col for col in org_feature_list if col in features.columns]
            if matching_cols:
                return features[matching_cols]
        
        return pd.DataFrame()
    
    def _analyze_organelle_clusters(
        self,
        plot_df: pd.DataFrame,
        features: pd.DataFrame,
        embedding: np.ndarray,
        organelle_name: str,
        output_dir: Path,
        result: StageResult,
    ) -> None:
        """
        Discover clusters in organelle embedding using Leiden, compute distinguishing features,
        and visualize representative cells.
        
        This mirrors the positive control analysis but for unsupervised clusters.
        """
        from ..analysis.fe_graphs_cluster_analysis import (
            compute_distinguishing_features,
            compute_organelle_importance_from_features,
            select_representative_items_by_features,
        )
        
        # 1. Run Leiden clustering on the embedding
        try:
            import leidenalg
            import igraph as ig
        except ImportError:
            logger.warning(f"leidenalg or igraph not available - skipping cluster analysis for {organelle_name}")
            return
        
        logger.info(f"  Running Leiden clustering on {organelle_name} embedding...")
        
        # Build k-NN graph
        from sklearn.neighbors import NearestNeighbors
        k = min(15, len(embedding) // 10)
        if k < 2:
            logger.warning(f"  Not enough samples ({len(embedding)}) for clustering")
            return
        
        knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(embedding)
        distances, indices = knn.kneighbors(embedding)
        
        # Create igraph
        edges = []
        weights = []
        for i in range(len(indices)):
            for j, neighbor_idx in enumerate(indices[i][1:]):  # Skip self
                edges.append((i, neighbor_idx))
                weights.append(1.0 / (distances[i][j + 1] + 1e-10))
        
        g = ig.Graph(n=len(embedding), edges=edges, directed=False)
        g.es['weight'] = weights
        
        # Leiden clustering
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=0.5,  # Lower resolution for broader clusters
            seed=42
        )
        
        cluster_labels = np.array(partition.membership)
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        logger.info(f"  Found {n_clusters} clusters in {organelle_name}")
        
        if n_clusters < 2:
            logger.info(f"  Only 1 cluster found - skipping analysis")
            return
        
        # 2. For each cluster, compute distinguishing features
        cluster_dir = output_dir / "cluster_analysis"
        cluster_dir.mkdir(exist_ok=True)
        
        # Plot clusters on embedding
        fig, ax = plt.subplots(figsize=self.plot_config.figsize_umap)
        
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=[colors[cluster_id]], s=self.plot_config.point_size,
                alpha=0.7, label=f"Cluster {cluster_id} (n={mask.sum()})",
                rasterized=True
            )
        
        ax.set_xlabel("UMAP 1", fontsize=20, fontweight='bold')
        ax.set_ylabel("UMAP 2", fontsize=20, fontweight='bold')
        ax.set_title(f"{organelle_name} Leiden Clusters", fontsize=22, fontweight='bold', pad=20)
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=10, markerscale=1.5, loc='best')
        
        plt.tight_layout()
        cluster_plot_path = cluster_dir / f"{organelle_name}_leiden_clusters.png"
        save_figure(fig, cluster_plot_path, dpi=150)
        result.add_file(cluster_plot_path)
        
        # 3. For each cluster, find distinguishing features and representative cells
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_size = cluster_mask.sum()
            
            if cluster_size < 10:  # Skip very small clusters
                logger.debug(f"    Skipping cluster {cluster_id} (only {cluster_size} items)")
                continue
            
            logger.info(f"    Analyzing cluster {cluster_id} ({cluster_size} items)...")
            
            # Get items in this cluster
            cluster_items = plot_df.index[cluster_mask]
            
            # Compute distinguishing features
            feature_df = compute_distinguishing_features(
                cluster_items=cluster_items,
                all_items=plot_df.index,
                features=features,
                top_n=10,  # Top 10 features per cluster
            )
            
            if feature_df.empty:
                logger.warning(f"    No distinguishing features for cluster {cluster_id}")
                continue
            
            # Save feature importance CSV
            csv_path = cluster_dir / f"{organelle_name}_cluster_{cluster_id}_features.csv"
            feature_df.to_csv(csv_path, index=False)
            result.add_file(csv_path)
            
            # Select representative items
            representative_items = select_representative_items_by_features(
                items=cluster_items,
                features=features,
                top_features=feature_df,
                n_items=6,  # 6 representative cells
            )
            
            # Save representative item list
            rep_df = plot_df.loc[representative_items].copy()
            rep_csv_path = cluster_dir / f"{organelle_name}_cluster_{cluster_id}_representative_items.csv"
            rep_df.to_csv(rep_csv_path)
            result.add_file(rep_csv_path)
            
            logger.info(f"      Top feature: {feature_df.iloc[0]['feature']} (d={feature_df.iloc[0]['cohens_d']:.2f})")
            logger.info(f"      Selected {len(representative_items)} representative items")
            
            # 4. Visualize representative cells with 3-panel layout
            if self.data.morphology_path:
                # Check for tensorstore
                try:
                    import tensorstore as ts
                    
                    logger.info(f"      Generating cell image visualization...")
                    try:
                        self._visualize_cluster_representative_cells(
                            representative_items=representative_items,
                            cluster_id=cluster_id,
                            organelle_name=organelle_name,
                            top_features=feature_df,
                            plot_df=plot_df,
                            features=features,
                            output_dir=cluster_dir,
                            result=result,
                        )
                    except Exception as viz_error:
                        logger.warning(f"      Cell visualization failed for cluster {cluster_id}: {viz_error}")
                        import traceback
                        traceback.print_exc()
                except ImportError:
                    logger.debug(f"      Skipping cell visualization (tensorstore not available)")
        
        logger.info(f"  Completed cluster analysis for {organelle_name}")
    
    def _visualize_cluster_representative_cells(
        self,
        representative_items: pd.Index,
        cluster_id: int,
        organelle_name: str,
        top_features: pd.DataFrame,
        plot_df: pd.DataFrame,
        features: pd.DataFrame,
        output_dir: Path,
        result: StageResult,
    ) -> None:
        """
        Visualize representative cells with 3-panel layout using the working code
        from positive controls representative cells.
        
        Handles both cell-level and gene-level analysis:
        - Cell level: visualize the representative cells directly
        - Gene level: look up cells for each representative gene and select top cells
        """
        # Import the working visualization functions
        from .fe_graphs_positive_controls_representative_cells import (
            _collect_viz_data_fast,
            _generate_representative_cell_canvas,
            select_top_cells_by_features,
        )
        
        # Get cell indices to visualize
        # Over-select to compensate for potential duplicates
        target_cells_per_item = 2
        overselect_factor = 2

        if self.level == "cell":
            # Cell level: representative_items are cell indices
            cell_indices_to_viz = representative_items
            target_cell_count = len(representative_items)
        else:
            # Gene/Guide level: representative_items are gene/guide identifiers
            # Need to look up cells for each gene/guide and select top cells
            # Over-select to compensate for duplicates
            cell_indices_to_viz = self._select_cells_for_gene_level_viz(
                representative_items,
                plot_df,
                features,
                top_features,
                n_cells_per_item=target_cells_per_item * overselect_factor,
            )
            target_cell_count = len(representative_items) * target_cells_per_item

            if len(cell_indices_to_viz) == 0:
                logger.warning(f"      No cells found for representative {self.level}s")
                return

            logger.info(f"      Found {len(cell_indices_to_viz)} cells from {len(representative_items)} {self.level}s (over-selected)")
        
        # Load cell-level data
        try:
            cell_df = self._load_cell_level_data()
            if cell_df is None or len(cell_df) == 0:
                logger.warning(f"      Could not load cell-level data for visualization")
                return
        except Exception as e:
            logger.warning(f"      Failed to load cell data: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Get cells to visualize
        cells_to_viz = cell_df.loc[cell_indices_to_viz].copy()

        # Diagnostic: check cell data before visualization
        required_cols = ['well', 'bbox', 'gene_name', 'segmentation_id', 'total_index']
        missing_cols = [c for c in required_cols if c not in cells_to_viz.columns]
        if missing_cols:
            logger.warning(f"      Missing required columns: {missing_cols}")
            logger.warning(f"      Available columns: {list(cells_to_viz.columns)[:10]}...")

        if len(cells_to_viz) > 0:
            first = cells_to_viz.iloc[0]
            logger.info(f"      First cell: well={first.get('well')}, bbox type={type(first.get('bbox'))}")

        # DEDUPLICATION: Remove cells that map to the same physical location (well + bbox)
        # Different cell IDs can refer to the same cell if data has duplicates
        n_before_physical_dedup = len(cells_to_viz)
        if 'well' in cells_to_viz.columns and 'bbox' in cells_to_viz.columns:
            # Create a hashable key from well + bbox
            def _make_cell_key(row):
                well = row.get('well', '')
                bbox = row.get('bbox')
                if bbox is None:
                    return None
                if isinstance(bbox, (list, tuple, np.ndarray)):
                    bbox_tuple = tuple(bbox)
                elif isinstance(bbox, str):
                    # Parse string bbox for comparison
                    bbox = bbox.strip()
                    if bbox.startswith('[') and bbox.endswith(']'):
                        try:
                            values = bbox[1:-1].split()
                            bbox_tuple = tuple(int(v) for v in values if v)
                        except ValueError:
                            bbox_tuple = bbox
                    else:
                        bbox_tuple = bbox
                else:
                    bbox_tuple = str(bbox)
                return (well, bbox_tuple)

            cells_to_viz['_cell_key'] = cells_to_viz.apply(_make_cell_key, axis=1)

            # Keep first occurrence of each unique cell
            cells_to_viz = cells_to_viz.drop_duplicates(subset='_cell_key', keep='first')
            cells_to_viz = cells_to_viz.drop(columns=['_cell_key'])

            n_physical_dups = n_before_physical_dedup - len(cells_to_viz)
            if n_physical_dups > 0:
                logger.info(f"      Removed {n_physical_dups} physically duplicate cells (same well+bbox)")

        # Trim to target count (we over-selected to compensate for duplicates)
        if len(cells_to_viz) > target_cell_count:
            cells_to_viz = cells_to_viz.iloc[:target_cell_count]
            logger.info(f"      Trimmed to {target_cell_count} cells (target count)")

        logger.info(f"      Collecting visualization data for {len(cells_to_viz)} cells...")

        # Use pre-built mappings from DataContext (single source of truth)
        channel_names = self.data.channel_names
        organelle_channel_indices = self.data.label_to_channel_index

        if not channel_names:
            logger.warning(f"      No channel names in DataContext - visualization may fail")
        if not organelle_channel_indices:
            logger.warning(f"      No label_to_channel_index in DataContext - visualization may fail")

        logger.debug(f"      Using {len(channel_names)} channels and {len(organelle_channel_indices)} label mappings from DataContext")

        # Collect visualization data using the fast, working method
        viz_data_list = _collect_viz_data_fast(
            morphology_path=self.data.morphology_path,
            cells_df=cells_to_viz,
            channel_names=channel_names,
            organelle_to_load=organelle_name,  # Load this specific organelle
            organelle_channel_indices=organelle_channel_indices,  # Pass mapping
        )
        
        if not viz_data_list:
            logger.warning(f"      No visualization data collected")
            return
        
        logger.info(f"      Successfully collected {len(viz_data_list)} cells")
        
        # Prepare output path
        img_path = output_dir / f"{organelle_name}_cluster_{cluster_id}_cells.png"
        
        # Generate the canvas using the working visualization function
        cluster_name = f"{organelle_name} - Cluster {cluster_id}"
        if self.level != "cell":
            cluster_name += f" (Representative Cells from {len(representative_items)} {self.level.capitalize()}s)"
        
        _generate_representative_cell_canvas(
            viz_data_list=viz_data_list,
            cluster_name=cluster_name,
            top_features_list=top_features,
            output_path=img_path,
            result=result,
            morphology_path=self.data.morphology_path,
        )
        
        logger.info(f"      Saved cell visualization: {img_path.name}")
    def _select_cells_for_gene_level_viz(
        self,
        representative_items: pd.Index,
        plot_df: pd.DataFrame,
        features: pd.DataFrame,
        top_features: pd.DataFrame,
        n_cells_per_item: int = 3,
    ) -> pd.Index:
        """
        Select representative cells for gene/guide-level clusters.

        For each representative item (gene/guide), finds cells belonging to that item
        and selects the top N cells based on the top distinguishing features using
        feature-based scoring (not random sampling).

        Parameters
        ----------
        representative_items : pd.Index
            Gene/guide identifiers to visualize (these are AnnData indices, need to map to names)
        plot_df : pd.DataFrame
            Gene/guide-level dataframe with metadata
        features : pd.DataFrame
            Gene/guide-level feature matrix
        top_features : pd.DataFrame
            Top distinguishing features for the cluster
        n_cells_per_item : int
            Number of cells to select per gene/guide

        Returns
        -------
        pd.Index
            Cell indices to visualize
        """
        # Import the shared helper for feature-based selection
        from .fe_graphs_positive_controls_representative_cells import select_top_cells_by_features

        # Load cell-level data to look up cells
        try:
            cell_df = self._load_cell_level_data()
            if cell_df is None or len(cell_df) == 0:
                logger.warning("      Could not load cell-level data")
                return pd.Index([])
        except Exception as e:
            logger.warning(f"      Failed to load cell data: {e}")
            return pd.Index([])

        # Load cell-level features for feature-based selection
        cell_features = self._load_cell_level_features()

        # Get the correct column name based on level
        gene_col = self._get_gene_column()

        if gene_col not in cell_df.columns:
            logger.warning(f"      {gene_col} column not found in cell data. Available columns: {list(cell_df.columns[:20])}")
            return pd.Index([])

        # CRITICAL FIX: Map representative_items (integer indices) to actual gene/guide names
        # representative_items are indices into plot_df, so we look up the gene names
        if gene_col not in plot_df.columns:
            logger.warning(f"      {gene_col} column not found in plot_df. Available columns: {list(plot_df.columns[:20])}")
            return pd.Index([])

        # Get the actual gene/guide names from the indices
        representative_names = plot_df.loc[representative_items, gene_col].values

        selected_cells = []

        # Debug: Log what we're looking for
        logger.info(f"      Looking for cells from {len(representative_names)} {self.level}s")
        logger.info(f"      Representative {self.level}s (first 3): {list(representative_names[:3])}")
        logger.info(f"      Cell df has {len(cell_df)} cells with {cell_df[gene_col].nunique()} unique {self.level}s")
        if cell_features is not None:
            logger.info(f"      Using feature-based cell selection ({len(cell_features.columns)} features)")
        else:
            logger.warning(f"      No cell features available - falling back to random selection")

        # Sample check: do any of the first 3 items match?
        for test_item in list(representative_names[:3]):
            test_matches = len(cell_df[cell_df[gene_col] == test_item])
            logger.info(f"      Test: {self.level} '{test_item}' has {test_matches} cells")

        for item_name in representative_names:
            # Find cells for this item
            item_cells_mask = cell_df[gene_col] == item_name
            item_cell_indices = cell_df.index[item_cells_mask].values

            if len(item_cell_indices) == 0:
                logger.info(f"        ❌ No cells found for {self.level} '{item_name}' (type: {type(item_name)})")
                continue

            n_to_select = min(n_cells_per_item, len(item_cell_indices))

            # Use feature-based scoring to select cells that best exemplify the phenotype
            selected = select_top_cells_by_features(
                cell_indices=item_cell_indices,
                cell_features=cell_features,
                top_features_list=top_features,
                n_cells=n_to_select,
                map_to_cell_level=True,  # Map aggregated feature names to cell-level
            )
            selected_cells.extend(selected)

            selection_method = "feature scoring" if cell_features is not None else "random"
            logger.info(f"        ✅ {self.level.capitalize()} '{item_name}': selected {len(selected)}/{len(item_cell_indices)} cells ({selection_method})")

        if len(selected_cells) == 0:
            logger.warning(f"      ⚠️ No cells selected from any of the {len(representative_names)} {self.level}s!")
            logger.warning(f"      Item names: {list(representative_names[:5])}")
            logger.warning(f"      Cell df gene_col ('{gene_col}') sample: {cell_df[gene_col].head(10).tolist()}")

        return pd.Index(selected_cells)
    
    def _get_gene_column(self) -> str:
        """Get the correct gene/guide column name for the current level."""
        if self.level == "gene":
            return "gene_name"
        elif self.level == "guide":
            return "barcode"
        else:
            return "gene_name"  # Default fallback
    
    def _load_cell_level_data(self) -> Optional[pd.DataFrame]:
        """
        Load cell-level dataframe for gene/guide-level visualization.

        Returns
        -------
        pd.DataFrame or None
            Cell-level dataframe, or None if not available
        """
        # Check if we already have it cached
        if hasattr(self, '_cell_df_cache') and self._cell_df_cache is not None:
            return self._cell_df_cache

        # Access cell-level AnnData from data context
        try:
            if "cell" in self.data.adata:
                cell_df = self.data.adata["cell"].obs.copy()

                # Ensure bbox and cp_bbox are not categorical
                if 'bbox' in cell_df.columns and pd.api.types.is_categorical_dtype(cell_df['bbox']):
                    cell_df['bbox'] = cell_df['bbox'].astype(object)
                if 'cp_bbox' in cell_df.columns and pd.api.types.is_categorical_dtype(cell_df['cp_bbox']):
                    cell_df['cp_bbox'] = cell_df['cp_bbox'].astype(object)

                # Cache it
                self._cell_df_cache = cell_df
                logger.debug(f"      Loaded cell-level data: {len(cell_df)} cells")
                return cell_df

            return None

        except Exception as e:
            logger.warning(f"      Failed to load cell-level data: {e}")
            return None

    def _load_cell_level_features(self) -> Optional[pd.DataFrame]:
        """
        Load cell-level feature matrix for feature-based cell selection.

        Returns
        -------
        pd.DataFrame or None
            Cell-level feature matrix (cells x features), or None if not available
        """
        # Check if we already have it cached
        if hasattr(self, '_cell_features_cache') and self._cell_features_cache is not None:
            return self._cell_features_cache

        # Access cell-level AnnData from data context
        try:
            if "cell" in self.data.adata:
                cell_adata = self.data.adata["cell"]
                cell_features = pd.DataFrame(
                    cell_adata.X,
                    index=cell_adata.obs_names,
                    columns=cell_adata.var_names,
                )

                # Cache it
                self._cell_features_cache = cell_features
                logger.debug(f"      Loaded cell-level features: {cell_features.shape}")
                return cell_features

            return None

        except Exception as e:
            logger.warning(f"      Failed to load cell-level features: {e}")
            return None


