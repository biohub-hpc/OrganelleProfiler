"""
Embedding Stage: Dimensionality reduction.

Performs:
- PCA for variance explained
- UMAP for visualization
- Per-organelle embeddings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List
import logging

from .fe_graphs_stage_base import BaseStage, StageResult
from ..core.fe_graphs_embedding import EmbeddingEngine
from ..core.fe_graphs_cache import EmbeddingCache
from ..plotting.fe_graphs_utils import save_figure

logger = logging.getLogger(__name__)


class EmbeddingStage(BaseStage):
    """Embedding stage for dimensionality reduction."""
    
    STAGE_NUMBER = 3
    STAGE_NAME = "embedding"
    
    def run(self) -> StageResult:
        """Run embedding analysis."""
        self.log_start()
        result = StageResult()
        
        df = self.df
        features = self.get_features()
        
        if features.empty:
            result.add_error("No features available for embedding")
            return result
        
        # Remove low variance features
        features = self._filter_low_variance(features)
        
        # Remove duplicates
        n_orig = len(features)
        features = features.drop_duplicates()
        if len(features) < n_orig:
            logger.info(f"Removed {n_orig - len(features)} duplicate rows")
        
        # Filter df to match
        df = df.loc[features.index].copy()
        
        result.add_metric("n_items", len(df))
        result.add_metric("n_features", features.shape[1])
        
        # Initialize engine
        engine = EmbeddingEngine(use_cuml=self.config.use_cuml)
        cache = EmbeddingCache(self.data.cache_dir)
        
        # PCA
        pca_dir = self.output_dir / "pca"
        pca_dir.mkdir(exist_ok=True)
        pca_result = self._run_pca(features, pca_dir, result)
        
        # UMAP - all features
        umap_dir = self.output_dir / "umap_all_features"
        umap_dir.mkdir(exist_ok=True)
        umap_embedding = self._run_umap(features, df, engine, cache, umap_dir, "all_features", result, n_total=n_orig)
        
        # Store embedding in result for downstream stages
        result.data["features"] = features
        result.data["df"] = df
        result.data["umap_embedding"] = umap_embedding
        result.data["pca_result"] = pca_result
        
        # Per-organelle UMAPs
        organelle_dir = self.output_dir / "umap_by_organelle"
        organelle_dir.mkdir(exist_ok=True)
        organelle_embeddings = self._run_organelle_umaps(features, df, engine, cache, organelle_dir, result)
        logger.info(f"Finished per-organelle UMAPs: generated {len(organelle_embeddings)} embeddings")
        logger.info(f"  Organelle keys: {list(organelle_embeddings.keys())}")
        result.data["organelle_embeddings"] = organelle_embeddings
        
        # Save embedding coordinates
        if umap_embedding is not None:
            coords_df = pd.DataFrame({
                "umap_1": umap_embedding[:, 0],
                "umap_2": umap_embedding[:, 1],
            }, index=df.index)
            coords_df.to_csv(self.output_dir / "umap_coordinates.csv")
            result.add_file(self.output_dir / "umap_coordinates.csv")
        
        self.log_complete(result)
        return result
    
    def _filter_low_variance(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Remove low variance features.
        
        Uses a relative threshold (percentile) instead of absolute threshold
        to handle different scales across aggregation levels (cell/guide/gene).
        
        At cell level: raw feature scales
        At gene level: mean-aggregated features (lower absolute variance)
        
        Strategy: Remove features with variance < 5th percentile
        """
        variances = features.var()
        
        # Use percentile-based threshold (more robust across levels)
        variance_threshold_percentile = 5.0  # Remove bottom 5%
        threshold = np.percentile(variances, variance_threshold_percentile)
        
        # But don't remove anything with variance > absolute minimum (true constants)
        absolute_min_threshold = 1e-8  # Only remove true near-constants
        threshold = max(threshold, absolute_min_threshold)
        
        low_var_mask = variances < threshold
        low_var_cols = features.columns[low_var_mask]
        
        if len(low_var_cols) > 0:
            pct_removed = len(low_var_cols) / len(features.columns) * 100
            logger.info(f"Removing {len(low_var_cols)} low-variance features ({pct_removed:.1f}%, threshold={threshold:.2e})")
        
        return features.drop(columns=low_var_cols)
    
    def _run_pca(self, features: pd.DataFrame, output_dir: Path, result: StageResult) -> Optional[np.ndarray]:
        """Run PCA and plot variance explained."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        logger.info("Running PCA...")
        
        # Scale
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        
        # PCA - use up to 250 components for comprehensive variance analysis
        n_components = min(250, features.shape[1], features.shape[0])
        pca = PCA(n_components=n_components, random_state=self.config.random_state)
        pca_result = pca.fit_transform(scaled)
        
        # Variance explained plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual variance
        axes[0].bar(range(1, n_components + 1), pca.explained_variance_ratio_)
        axes[0].set_xlabel("Principal Component")
        axes[0].set_ylabel("Variance Explained Ratio")
        axes[0].set_title("Variance Explained by PC")
        
        # Cumulative variance
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        axes[1].plot(range(1, n_components + 1), cumulative, "o-")
        axes[1].axhline(0.9, color="red", linestyle="--", label="90%")
        axes[1].axhline(0.95, color="orange", linestyle="--", label="95%")
        axes[1].set_xlabel("Number of Components")
        axes[1].set_ylabel("Cumulative Variance Explained")
        axes[1].set_title("Cumulative Variance Explained")
        axes[1].legend()
        
        plt.tight_layout()
        path = save_figure(fig, output_dir / "variance_explained.png")
        result.add_file(path)
        
        # Save loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            index=features.columns,
            columns=[f"PC{i+1}" for i in range(n_components)]
        )
        loadings.to_csv(output_dir / "pca_loadings.csv")
        result.add_file(output_dir / "pca_loadings.csv")
        
        # Metrics
        result.add_metric("pca_var_explained_pc1", pca.explained_variance_ratio_[0])
        result.add_metric("pca_var_explained_10pcs", cumulative[9] if len(cumulative) > 9 else cumulative[-1])
        
        return pca_result
    
    def _run_umap(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        engine: EmbeddingEngine,
        cache: EmbeddingCache,
        output_dir: Path,
        name: str,
        result: StageResult,
        n_total: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Run UMAP and generate basic visualization.

        Parameters
        ----------
        n_total : int, optional
            Total number of cells before any filtering. If provided, shown in subtitle
            for clarity on how many cells were excluded.
        """
        cache_key = f"{self.data.experiment}_{self.level}_{name}"
        n_items = len(features)

        # Check cache (if enabled)
        cached = None
        if self.config.use_cache:
            cached = cache.load(cache_key, n_items)

        if cached is not None:
            embedding = cached["embedding"]
            logger.info(f"Loaded {name} UMAP from cache")
        else:
            logger.info(f"Computing {name} UMAP for {n_items:,} {self.level}s...")
            embedding = engine.compute_umap(
                features.values,
                n_neighbors=self.analysis_config.umap_n_neighbors,
                min_dist=self.analysis_config.umap_min_dist,
                scale=True,
            )
            # Save to cache (always save, even if use_cache=False, for future runs)
            cache.save(cache_key, embedding, {}, n_items)

        # Basic UMAP plot
        fig, ax = plt.subplots(figsize=self.plot_config.figsize_umap)
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            s=self.plot_config.point_size,
            alpha=self.plot_config.alpha,
            rasterized=True,
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        # Title with cell counts
        title = f"{self.level_label} Level UMAP ({name})"
        if n_total is not None and n_total != n_items:
            subtitle = f"n={n_items:,} cells in embedding (of {n_total:,} total)"
        else:
            subtitle = f"n={n_items:,} cells"
        ax.set_title(f"{title}\n{subtitle}", fontsize=12)

        path = save_figure(fig, output_dir / f"umap_{name}.png")
        result.add_file(path)

        return embedding
    
    def _run_organelle_umaps(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        engine: EmbeddingEngine,
        cache: EmbeddingCache,
        output_dir: Path,
        result: StageResult,
    ) -> Dict[str, np.ndarray]:
        """Run UMAP for each organelle feature subset.
        
        Returns
        -------
        dict
            Mapping of organelle name to dict with 'embedding' and 'indices' keys.
            indices are the DataFrame indices that correspond to the embedding rows.
        """
        organelle_features = self.group_features_by_organelle(features.columns.tolist())
        organelle_embeddings = {}
        
        n_groups = len(organelle_features)
        logger.info(f"Running per-organelle UMAPs for {n_groups} organelle groups")
        
        # Validate organelle groups
        if n_groups == 0:
            logger.warning("No organelle groups found - skipping per-organelle UMAPs")
            return organelle_embeddings
        elif n_groups > 50:
            logger.warning(f"Found {n_groups} groups - expected ~20. Check adata.var['organelle']")
            logger.warning(f"First 5 groups: {list(organelle_features.keys())[:5]}")
        
        for organelle, cols in sorted(organelle_features.items()):
            if len(cols) < 3:
                logger.debug(f"Skipping {organelle}: only {len(cols)} features (need >= 3)")
                continue
            
            org_features = features[cols].copy()
            
            # Filter low variance using percentile-based threshold
            variances = org_features.var()
            if len(variances) > 0:
                # Use 5th percentile, minimum 1e-8 for true constants
                threshold = max(np.percentile(variances, 5.0), 1e-8)
                org_features = org_features.loc[:, variances > threshold]
            
            if org_features.shape[1] < 2:
                continue
            
            # Remove duplicate rows to prevent cuML RAFT errors
            n_before = len(org_features)
            org_features = org_features.drop_duplicates()
            n_duplicates = n_before - len(org_features)
            if n_duplicates > 0:
                logger.info(f"  {organelle}: Removed {n_duplicates:,} duplicate rows ({n_duplicates/n_before*100:.1f}%)")
            
            # Update df to match filtered features
            org_df = df.loc[org_features.index].copy()
            
            # Run UMAP (pass n_before as total for subtitle clarity)
            embedding = self._run_umap(
                org_features, org_df, engine, cache,
                output_dir, f"organelle_{organelle}", result,
                n_total=n_before
            )
            
            if embedding is not None:
                # Store both embedding and the indices it corresponds to
                organelle_embeddings[organelle] = {
                    'embedding': embedding,
                    'indices': org_features.index.tolist()
                }
                logger.debug(f"  {organelle}: saved embedding with shape {embedding.shape} and {len(org_features.index)} indices")
            else:
                logger.debug(f"  {organelle}: UMAP returned None")
        
        logger.info(f"Completed per-organelle UMAPs: {len(organelle_embeddings)}/{len(organelle_features)} successful")
        return organelle_embeddings
