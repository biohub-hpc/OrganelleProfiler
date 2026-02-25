"""
Caching module for embeddings and clustering results.

Provides persistent caching to avoid recomputing expensive UMAP/clustering
operations when data hasn't changed.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Cache for embedding and clustering results.
    
    Stores results as compressed NumPy files (.npz) with validation
    based on cell count to detect when recomputation is needed.
    
    Parameters
    ----------
    cache_dir : Path
        Directory to store cache files.
    
    Examples
    --------
    >>> cache = EmbeddingCache(Path("./cache"))
    >>> cached = cache.load("cell_all_features", n_cells=10000)
    >>> if cached is None:
    ...     # Compute embedding and clustering
    ...     cache.save("cell_all_features", embedding, clusters_dict, n_cells)
    """
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given cache key."""
        # Sanitize cache key for filename
        safe_key = cache_key.replace("/", "_").replace(" ", "_")
        return self.cache_dir / f"{safe_key}.npz"
    
    def load(
        self, cache_key: str, n_cells: int
    ) -> Optional[Dict[str, Any]]:
        """
        Load cached embedding and clustering results if they exist and match current data.
        
        Parameters
        ----------
        cache_key : str
            Unique identifier for this embedding (e.g., "cell_all_features").
        n_cells : int
            Expected number of cells (used to validate cache).
            
        Returns
        -------
        dict or None
            Dictionary with 'embedding' and 'clusters_*' keys, or None if cache miss.
        """
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            cached = np.load(cache_path, allow_pickle=True)
            cached_n_cells = cached.get('n_cells', np.array(0)).item()
            
            if cached_n_cells != n_cells:
                logger.info(
                    f"Cache invalidated: cell count changed ({cached_n_cells} -> {n_cells})"
                )
                return None
            
            result = {'embedding': cached['embedding']}
            
            # Load all cluster arrays
            for key in cached.files:
                if key.startswith('clusters_'):
                    result[key] = cached[key]
            
            logger.info(f"Loaded from cache: {cache_path.name}")
            return result
            
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return None
    
    def save(
        self,
        cache_key: str,
        embedding: np.ndarray,
        clusters_dict: Dict[str, np.ndarray],
        n_cells: int,
    ) -> None:
        """
        Save embedding and clustering results to cache.
        
        Parameters
        ----------
        cache_key : str
            Unique identifier for this embedding.
        embedding : np.ndarray
            UMAP embedding array (n_cells, 2).
        clusters_dict : dict
            Dictionary mapping method name to cluster labels.
        n_cells : int
            Number of cells (for validation on reload).
        """
        cache_path = self._get_cache_path(cache_key)
        
        try:
            save_dict = {
                'embedding': embedding,
                'n_cells': np.array(n_cells),
            }
            for method, clusters in clusters_dict.items():
                save_dict[f'clusters_{method}'] = clusters
            
            np.savez_compressed(cache_path, **save_dict)
            logger.info(f"Saved to cache: {cache_path.name}")
            
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def invalidate(self, cache_key: str) -> bool:
        """
        Remove a cached result.
        
        Parameters
        ----------
        cache_key : str
            Cache key to invalidate.
            
        Returns
        -------
        bool
            True if cache was removed, False if it didn't exist.
        """
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Invalidated cache: {cache_path.name}")
            return True
        
        return False
    
    def clear_all(self) -> int:
        """
        Remove all cached results.
        
        Returns
        -------
        int
            Number of cache files removed.
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.npz"):
            cache_file.unlink()
            count += 1
        
        logger.info(f"Cleared {count} cache files")
        return count
    
    def load_cohens_d(self, cache_key: str, n_items: int) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Load cached Cohen's d statistics if they exist and match current data.
        
        Parameters
        ----------
        cache_key : str
            Unique identifier for this dataset (e.g., "gene_positive_controls").
        n_items : int
            Expected number of items (genes/guides) - used to validate cache.
            
        Returns
        -------
        dict or None
            Dictionary mapping cluster name to DataFrame of Cohen's d stats, or None if cache miss.
        """
        cache_path = self.cache_dir / f"{cache_key}_cohens_d.parquet"
        
        if not cache_path.exists():
            return None
        
        try:
            # Load the parquet file
            cached_df = pd.read_parquet(cache_path)
            
            # Validate using stored n_items
            cached_n_items = cached_df.attrs.get('n_items', 0)
            if cached_n_items != n_items:
                logger.info(
                    f"Cohen's d cache invalidated: item count changed ({cached_n_items} -> {n_items})"
                )
                return None
            
            # Reconstruct dictionary keyed by cluster
            result = {}
            for cluster_name in cached_df['cluster'].unique():
                cluster_df = cached_df[cached_df['cluster'] == cluster_name].copy()
                cluster_df = cluster_df.drop(columns=['cluster'])
                result[cluster_name] = cluster_df
            
            logger.info(f"Loaded Cohen's d from cache: {cache_path.name} ({len(result)} clusters)")
            return result
            
        except Exception as e:
            logger.warning(f"Cohen's d cache load failed: {e}")
            return None
    
    def save_cohens_d(
        self,
        cache_key: str,
        cohens_d_dict: Dict[str, pd.DataFrame],
        n_items: int,
    ) -> None:
        """
        Save Cohen's d statistics to cache.
        
        Parameters
        ----------
        cache_key : str
            Unique identifier for this dataset.
        cohens_d_dict : dict
            Dictionary mapping cluster name to DataFrame of Cohen's d statistics.
        n_items : int
            Number of items (for validation on reload).
        """
        cache_path = self.cache_dir / f"{cache_key}_cohens_d.parquet"
        
        try:
            # Combine all cluster DataFrames into one with cluster column
            all_dfs = []
            for cluster_name, cluster_df in cohens_d_dict.items():
                df_copy = cluster_df.copy()
                df_copy.insert(0, 'cluster', cluster_name)
                all_dfs.append(df_copy)
            
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Store n_items as attribute for validation
            combined_df.attrs['n_items'] = n_items
            
            # Save as parquet (efficient for DataFrames)
            combined_df.to_parquet(cache_path, index=False)
            logger.info(f"Saved Cohen's d to cache: {cache_path.name} ({len(cohens_d_dict)} clusters)")
            
        except Exception as e:
            logger.warning(f"Cohen's d cache save failed: {e}")
