"""
Clustering module.

Provides unified clustering with multiple algorithms:
- HDBSCAN (density-based)
- KMeans (with elbow method for optimal k)
- Leiden (graph-based community detection)

Consolidates 3+ duplicate clustering implementations from the original fe_graphs.py.
"""

import numpy as np
from typing import Optional, Dict, List
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseClusterer(ABC):
    """Abstract base class for clustering algorithms."""
    
    @abstractmethod
    def fit_predict(self, embedding: np.ndarray) -> np.ndarray:
        """
        Fit the clustering model and return cluster labels.
        
        Parameters
        ----------
        embedding : np.ndarray
            2D embedding of shape (n_samples, n_dims).
            
        Returns
        -------
        np.ndarray
            Cluster labels. -1 indicates noise/unclustered.
        """
        pass


class HDBSCANClusterer(BaseClusterer):
    """HDBSCAN density-based clustering."""
    
    def __init__(
        self,
        min_cluster_size: int = 25,
        min_samples: int = 25,
        use_cuml: bool = True,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.use_cuml = use_cuml
        
    def fit_predict(self, embedding: np.ndarray) -> np.ndarray:
        """Run HDBSCAN clustering."""
        clusters = None
        
        if self.use_cuml:
            clusters = self._fit_predict_cuml(embedding)
        
        if clusters is None:
            clusters = self._fit_predict_cpu(embedding)
        
        return clusters
    
    def _fit_predict_cuml(self, embedding: np.ndarray) -> Optional[np.ndarray]:
        """Attempt HDBSCAN using cuML (GPU)."""
        try:
            from cuml.cluster import HDBSCAN as cuMLHDBSCAN
            
            logger.info("Using cuML (GPU) for HDBSCAN...")
            clusterer = cuMLHDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                gen_min_span_tree=True,
            )
            clusters = clusterer.fit_predict(embedding)
            return np.asarray(clusters)
            
        except (ImportError, TypeError) as e:
            logger.warning(f"cuML HDBSCAN not available: {e}. Falling back to CPU.")
            return None
    
    def _fit_predict_cpu(self, embedding: np.ndarray) -> np.ndarray:
        """Run HDBSCAN using hdbscan library (CPU)."""
        import hdbscan
        
        logger.info("Using hdbscan (CPU) for clustering...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            gen_min_span_tree=True,
            core_dist_n_jobs=-1,
        )
        return clusterer.fit_predict(embedding)


class KMeansClusterer(BaseClusterer):
    """KMeans clustering with elbow method for optimal k."""
    
    def __init__(
        self,
        k_range: range = range(2, 31),
        random_state: int = 42,
        max_sample_for_k_search: int = 10000,
    ):
        self.k_range = k_range
        self.random_state = random_state
        self.max_sample_for_k_search = max_sample_for_k_search
        self.optimal_k_ = None
        self.inertias_ = None
        
    def fit_predict(self, embedding: np.ndarray) -> np.ndarray:
        """Run KMeans with automatic k selection via elbow method."""
        from sklearn.cluster import KMeans
        
        # Find optimal k
        self.optimal_k_ = self._find_optimal_k(embedding)
        
        if self.optimal_k_ > 1:
            kmeans = KMeans(
                n_clusters=self.optimal_k_,
                random_state=self.random_state,
                n_init=10,
            )
            return kmeans.fit_predict(embedding)
        else:
            logger.warning("Could not determine optimal k, assigning all to cluster -1")
            return np.full(len(embedding), -1)
    
    def _find_optimal_k(self, data: np.ndarray) -> int:
        """Find optimal k using the elbow method."""
        from sklearn.cluster import KMeans
        
        # Subsample for efficiency on large datasets
        if len(data) > self.max_sample_for_k_search:
            logger.info(
                f"Subsampling from {len(data)} to {self.max_sample_for_k_search} "
                "for optimal k search."
            )
            sample_indices = np.random.choice(
                data.shape[0], self.max_sample_for_k_search, replace=False
            )
            data = data[sample_indices]
        
        logger.info(
            f"Searching for optimal k in range {self.k_range.start}-{self.k_range.stop-1} "
            "using elbow method..."
        )
        
        inertias = []
        for k in self.k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        self.inertias_ = inertias
        
        # Find elbow point (maximum distance from line connecting first and last points)
        p1 = np.array([self.k_range.start, inertias[0]])
        p2 = np.array([self.k_range.stop - 1, inertias[-1]])
        
        distances = []
        for i, k in enumerate(self.k_range):
            p3 = np.array([k, inertias[i]])
            distance = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
            distances.append(distance)
        
        elbow_index = np.argmax(distances)
        optimal_k = self.k_range[elbow_index]
        
        logger.info(f"Optimal k found at elbow point: {optimal_k}")
        return optimal_k


class LeidenClusterer(BaseClusterer):
    """Leiden graph-based community detection clustering."""
    
    def __init__(
        self,
        resolution: float = 1.0,
        n_neighbors: int = 15,
        random_state: int = 42,
        max_cells_full_leiden: int = 100000,
    ):
        self.resolution = resolution
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.max_cells_full_leiden = max_cells_full_leiden
        
    def fit_predict(self, embedding: np.ndarray) -> np.ndarray:
        """Run Leiden clustering."""
        import anndata as ad
        import scanpy as sc
        
        n_cells = len(embedding)
        
        if n_cells > self.max_cells_full_leiden:
            return self._fit_predict_subsampled(embedding, n_cells)
        else:
            return self._fit_predict_full(embedding)
    
    def _fit_predict_full(self, embedding: np.ndarray) -> np.ndarray:
        """Run full Leiden clustering."""
        import anndata as ad
        import scanpy as sc
        
        adata_temp = ad.AnnData(X=embedding)
        
        logger.info("Computing neighbor graph on UMAP embedding...")
        sc.pp.neighbors(adata_temp, n_neighbors=self.n_neighbors, use_rep='X')
        
        logger.info("Running Leiden algorithm (igraph backend)...")
        sc.tl.leiden(
            adata_temp,
            resolution=self.resolution,
            random_state=self.random_state,
            flavor='igraph',
            n_iterations=2,
            directed=False,
        )
        
        return adata_temp.obs['leiden'].astype(int).values
    
    def _fit_predict_subsampled(
        self, embedding: np.ndarray, n_cells: int
    ) -> np.ndarray:
        """Run Leiden on subsample and propagate labels via nearest neighbor."""
        import anndata as ad
        import scanpy as sc
        from sklearn.neighbors import NearestNeighbors
        
        logger.info(
            f"Large dataset ({n_cells:,} cells), using subsampled Leiden + "
            "nearest-neighbor propagation..."
        )
        
        # Subsample for clustering
        np.random.seed(self.random_state)
        sample_idx = np.random.choice(
            n_cells, self.max_cells_full_leiden, replace=False
        )
        embedding_sample = embedding[sample_idx]
        
        # Run Leiden on subsample
        adata_temp = ad.AnnData(X=embedding_sample)
        sc.pp.neighbors(adata_temp, n_neighbors=self.n_neighbors, use_rep='X')
        sc.tl.leiden(
            adata_temp,
            resolution=self.resolution,
            random_state=self.random_state,
            flavor='igraph',
            n_iterations=2,
            directed=False,
        )
        sample_clusters = adata_temp.obs['leiden'].astype(int).values
        
        # Propagate labels to all cells
        logger.info(f"Propagating cluster labels to all {n_cells:,} cells...")
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
        nn.fit(embedding_sample)
        _, indices = nn.kneighbors(embedding)
        
        return sample_clusters[indices.flatten()]


class ClusteringEngine:
    """
    Unified clustering interface with multiple algorithms.
    
    Parameters
    ----------
    use_cuml : bool
        Whether to use cuML GPU acceleration where available. Defaults to True.
    random_state : int
        Random seed for reproducibility. Defaults to 42.
    
    Examples
    --------
    >>> engine = ClusteringEngine(use_cuml=True)
    >>> labels = engine.cluster(embedding, method="hdbscan")
    >>> all_labels = engine.cluster_all(embedding)
    """
    
    METHODS = {
        "hdbscan": HDBSCANClusterer,
        "kmeans": KMeansClusterer,
        "leiden": LeidenClusterer,
    }
    
    def __init__(self, use_cuml: bool = True, random_state: int = 42):
        self.use_cuml = use_cuml
        self.random_state = random_state
        self._clusterers: Dict[str, BaseClusterer] = {}
    
    def _get_clusterer(self, method: str) -> BaseClusterer:
        """Get or create a clusterer instance."""
        if method not in self._clusterers:
            if method not in self.METHODS:
                raise ValueError(
                    f"Unknown clustering method: {method}. "
                    f"Available: {list(self.METHODS.keys())}"
                )
            
            ClustererClass = self.METHODS[method]
            
            if method == "hdbscan":
                self._clusterers[method] = ClustererClass(use_cuml=self.use_cuml)
            elif method == "kmeans":
                self._clusterers[method] = ClustererClass(random_state=self.random_state)
            elif method == "leiden":
                self._clusterers[method] = ClustererClass(random_state=self.random_state)
        
        return self._clusterers[method]
    
    def cluster(
        self, embedding: np.ndarray, method: str = "hdbscan"
    ) -> np.ndarray:
        """
        Run clustering with specified algorithm.
        
        Parameters
        ----------
        embedding : np.ndarray
            2D embedding of shape (n_samples, 2).
        method : str
            Clustering method: "hdbscan", "kmeans", or "leiden".
            
        Returns
        -------
        np.ndarray
            Cluster labels. -1 indicates noise.
        """
        logger.info(f"Running {method.upper()} clustering...")
        clusterer = self._get_clusterer(method)
        labels = clusterer.fit_predict(embedding)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"{method.upper()}: {n_clusters} clusters found")
        
        return labels
    
    def cluster_all(
        self, embedding: np.ndarray, methods: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Run all clustering methods.
        
        Parameters
        ----------
        embedding : np.ndarray
            2D embedding of shape (n_samples, 2).
        methods : list[str], optional
            List of methods to run. Defaults to all available.
            
        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping method name to cluster labels.
        """
        methods = methods or list(self.METHODS.keys())
        
        results = {}
        for method in methods:
            results[method] = self.cluster(embedding, method)
        
        return results
