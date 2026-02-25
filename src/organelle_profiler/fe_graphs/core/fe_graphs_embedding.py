"""
Embedding computation module.

Provides unified UMAP and PCA computation with automatic GPU fallback.
Consolidates 5+ duplicate UMAP implementations from the original fe_graphs.py.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Unified UMAP/PCA computation with GPU fallback.
    
    Handles cuML GPU acceleration when available, with automatic
    fallback to CPU implementations (umap-learn, sklearn).
    
    Parameters
    ----------
    use_cuml : bool
        Whether to attempt GPU acceleration via cuML. Defaults to True.
    random_state : int
        Random seed for reproducibility. Defaults to 42.
    
    Examples
    --------
    >>> engine = EmbeddingEngine(use_cuml=True)
    >>> embedding = engine.compute_umap(scaled_features)
    >>> pca_result = engine.compute_pca(scaled_features, n_components=250)
    """
    
    def __init__(self, use_cuml: bool = True, random_state: int = 42):
        self.use_cuml = use_cuml
        self.random_state = random_state
        self._cuml_available = None
        
    @property
    def cuml_available(self) -> bool:
        """Check if cuML is available for GPU acceleration."""
        if self._cuml_available is None:
            try:
                import cuml
                self._cuml_available = True
            except ImportError:
                self._cuml_available = False
        return self._cuml_available
    
    def compute_umap(
        self,
        features: np.ndarray,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 2,
        scale: bool = True,
    ) -> np.ndarray:
        """
        Compute UMAP embedding with automatic GPU/CPU selection.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        n_neighbors : int
            Number of neighbors for UMAP. Defaults to 15.
        min_dist : float
            Minimum distance parameter for UMAP. Defaults to 0.1.
        n_components : int
            Number of output dimensions. Defaults to 2.
        scale : bool
            Whether to standardize features before UMAP. Defaults to True.
            
        Returns
        -------
        np.ndarray
            UMAP embedding of shape (n_samples, n_components).
        """
        # Optionally scale features
        if scale:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        
        embedding = None
        
        # Try cuML GPU implementation first
        if self.use_cuml and self.cuml_available:
            embedding = self._compute_umap_cuml(
                features, n_neighbors, min_dist, n_components
            )
        
        # Fallback to CPU implementation
        if embedding is None:
            embedding = self._compute_umap_cpu(
                features, n_neighbors, min_dist, n_components
            )
        
        return embedding
    
    def _compute_umap_cuml(
        self,
        features: np.ndarray,
        n_neighbors: int,
        min_dist: float,
        n_components: int,
    ) -> Optional[np.ndarray]:
        """Attempt UMAP computation using cuML (GPU)."""
        try:
            from cuml.manifold import UMAP as cuMLUMAP
            
            logger.info("Using cuML (GPU) for UMAP computation...")
            
            # Data must be float32 and contiguous for cuML
            features_32 = np.ascontiguousarray(features, dtype=np.float32)
            
            reducer = cuMLUMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                random_state=self.random_state,
                init="random",  # Using random init for stability
            )
            embedding = reducer.fit_transform(features_32)
            
            logger.info("cuML UMAP successful.")
            return np.asarray(embedding)
            
        except RuntimeError as e:
            # Handle RAFT failures (often due to duplicate data points)
            error_str = str(e).lower()
            if "raft" in error_str or "duplicate" in error_str:
                logger.warning(
                    "cuML UMAP failed with RAFT error (likely duplicate data points). "
                    "Falling back to CPU."
                )
                return None
            raise
            
        except (ImportError, TypeError) as e:
            logger.warning(f"cuML UMAP not available: {e}. Falling back to CPU.")
            return None
    
    def _compute_umap_cpu(
        self,
        features: np.ndarray,
        n_neighbors: int,
        min_dist: float,
        n_components: int,
    ) -> np.ndarray:
        """Compute UMAP using umap-learn (CPU)."""
        import umap
        
        logger.info("Using umap-learn (CPU) for UMAP computation...")
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=self.random_state,
        )
        embedding = reducer.fit_transform(features)
        
        logger.info("umap-learn UMAP successful.")
        return embedding
    
    def compute_pca(
        self,
        features: np.ndarray,
        n_components: int = 250,
        scale: bool = True,
    ) -> np.ndarray:
        """
        Compute PCA embedding.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        n_components : int
            Number of principal components. Defaults to 50.
        scale : bool
            Whether to standardize features before PCA. Defaults to True.
            
        Returns
        -------
        np.ndarray
            PCA embedding of shape (n_samples, n_components).
        """
        from sklearn.decomposition import PCA
        
        if scale:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        
        # Adjust n_components if larger than feature count
        n_components = min(n_components, features.shape[1], features.shape[0])
        
        pca = PCA(n_components=n_components, random_state=self.random_state)
        return pca.fit_transform(features)
