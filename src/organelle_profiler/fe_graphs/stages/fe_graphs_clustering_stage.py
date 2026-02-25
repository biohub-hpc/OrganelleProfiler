"""
Clustering Stage: Unsupervised clustering and enrichment.

Performs:
- Multiple clustering algorithms (HDBSCAN, Leiden, KMeans)
- Cluster quality assessment
- Gene enrichment per cluster
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import fdrcorrection
from joblib import Parallel, delayed
from tqdm import tqdm
import logging

from .fe_graphs_stage_base import BaseStage, StageResult
from ..core.fe_graphs_clustering import ClusteringEngine
from ..core.fe_graphs_cache import EmbeddingCache
from ..plotting.fe_graphs_utils import save_figure, create_cluster_palette

logger = logging.getLogger(__name__)


class ClusteringStage(BaseStage):
    """Clustering stage for unsupervised analysis."""
    
    STAGE_NUMBER = 5
    STAGE_NAME = "clustering"
    
    def run(self) -> StageResult:
        """Run clustering analysis."""
        self.log_start()
        result = StageResult()
        
        # Get embedding from upstream or cache
        embedding = None
        df = None
        features = None
        
        if "embedding" in self.upstream and "umap_embedding" in self.upstream["embedding"].data:
            # Get from upstream
            embedding = self.upstream["embedding"].data["umap_embedding"]
            df = self.upstream["embedding"].data["df"]
            features = self.upstream["embedding"].data["features"]
        else:
            # Embedding stage was skipped - try to load from cache
            logger.info("Embedding stage not in upstream, attempting to load from cache...")
            cache = EmbeddingCache(self.data.cache_dir)
            
            df = self.df.copy()
            features = self.get_features(df)
            
            cache_key = f"{self.data.experiment}_{self.level}_all_features"
            cached_result = cache.load(cache_key, n_cells=len(df))
            
            if cached_result is None:
                result.add_error("No embedding available from upstream and no cached embedding found. Run embedding stage first.")
                return result
            
            embedding = cached_result["embedding"]
            logger.info(f"Loaded cached embedding: {embedding.shape}")
        
        if embedding is None:
            result.add_error("Embedding is None")
            return result
        
        result.add_metric("n_items", len(df))
        
        # Initialize clustering engine
        cluster_engine = ClusteringEngine(use_cuml=self.config.use_cuml)
        cache = EmbeddingCache(self.data.cache_dir)
        
        # Run clustering
        clustering_methods = self.config.get_clustering_methods()
        clusters_dict = self._run_clustering(embedding, cluster_engine, cache, clustering_methods, result)
        
        # Build plot DataFrame with cluster assignments
        plot_df = df.copy()
        plot_df["umap_1"] = embedding[:, 0]
        plot_df["umap_2"] = embedding[:, 1]
        
        for method, clusters in clusters_dict.items():
            plot_df[f"cluster_{method}"] = "c" + clusters.astype(str)
        
        # Use primary method as default
        primary_method = clustering_methods[0]
        plot_df["cluster"] = plot_df[f"cluster_{primary_method}"]
        
        # Save cluster assignments
        assignments_dir = self.output_dir / "cluster_assignments"
        assignments_dir.mkdir(exist_ok=True)
        self._save_cluster_assignments(plot_df, clusters_dict, assignments_dir, result)
        
        # Cluster quality
        quality_dir = self.output_dir / "cluster_quality"
        quality_dir.mkdir(exist_ok=True)
        self._assess_cluster_quality(embedding, clusters_dict, quality_dir, result)
        
        # Cluster visualization
        viz_dir = self.output_dir / "cluster_visualization"
        viz_dir.mkdir(exist_ok=True)
        self._visualize_clusters(plot_df, clusters_dict, viz_dir, result)
        
        # Cluster enrichment (for cell level)
        if self.level == "cell" and "gene_name" in plot_df.columns:
            enrichment_dir = self.output_dir / "cluster_enrichment"
            enrichment_dir.mkdir(exist_ok=True)
            enrichment_df = self._compute_enrichment(plot_df, enrichment_dir, result)
            result.data["enrichment"] = enrichment_df
        
        # Store for downstream
        result.data["plot_df"] = plot_df
        result.data["clusters"] = clusters_dict
        
        self.log_complete(result)
        return result
    
    def _run_clustering(
        self,
        embedding: np.ndarray,
        engine: ClusteringEngine,
        cache: EmbeddingCache,
        methods: List[str],
        result: StageResult,
    ) -> Dict[str, np.ndarray]:
        """Run clustering with caching."""
        cache_key = f"{self.data.experiment}_{self.level}_clustering"
        n_items = len(embedding)
        
        # Check cache (if enabled)
        cached = None
        if self.config.use_cache:
            cached = cache.load(cache_key, n_items)
        
        clusters_dict = {}
        
        if cached is not None:
            for method in methods:
                key = f"clusters_{method}"
                if key in cached:
                    clusters_dict[method] = cached[key]
                    n_clusters = len(set(clusters_dict[method])) - (1 if -1 in clusters_dict[method] else 0)
                    logger.info(f"  {method.upper()} (cached): {n_clusters} clusters")
                    result.add_metric(f"n_clusters_{method}", n_clusters)
        
        # Run any missing methods
        for method in methods:
            if method not in clusters_dict:
                clusters = engine.cluster(embedding, method)
                clusters_dict[method] = clusters
                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                result.add_metric(f"n_clusters_{method}", n_clusters)
        
        # Update cache (always save for future runs)
        cache.save(cache_key, embedding, clusters_dict, n_items)
        
        return clusters_dict
    
    def _save_cluster_assignments(
        self,
        plot_df: pd.DataFrame,
        clusters_dict: Dict[str, np.ndarray],
        output_dir: Path,
        result: StageResult,
    ) -> None:
        """Save cluster assignments."""
        for method, clusters in clusters_dict.items():
            assign_df = pd.DataFrame({
                "index": plot_df.index,
                f"cluster_{method}": "c" + clusters.astype(str),
            })
            path = output_dir / f"{method}_clusters.csv"
            assign_df.to_csv(path, index=False)
            result.add_file(path)
    
    def _assess_cluster_quality(
        self,
        embedding: np.ndarray,
        clusters_dict: Dict[str, np.ndarray],
        output_dir: Path,
        result: StageResult,
    ) -> None:
        """Assess clustering quality."""
        from sklearn.metrics import silhouette_score
        
        # Silhouette scores
        silhouette_data = []
        for method, clusters in clusters_dict.items():
            # Filter out noise
            mask = clusters != -1
            if mask.sum() < 2 or len(set(clusters[mask])) < 2:
                continue
            
            try:
                score = silhouette_score(embedding[mask], clusters[mask])
                silhouette_data.append({"method": method, "silhouette": score})
                result.add_metric(f"silhouette_{method}", score)
            except Exception as e:
                logger.warning(f"Could not compute silhouette for {method}: {e}")
        
        if silhouette_data:
            fig, ax = plt.subplots(figsize=(8, 5))
            sil_df = pd.DataFrame(silhouette_data)
            sns.barplot(data=sil_df, x="method", y="silhouette", ax=ax)
            ax.set_xlabel("Clustering Method")
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Clustering Quality Comparison")
            path = save_figure(fig, output_dir / "silhouette_scores.png")
            result.add_file(path)
        
        # Cluster sizes
        for method, clusters in clusters_dict.items():
            sizes = pd.Series(clusters).value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sizes.plot(kind="bar", ax=ax)
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Size")
            ax.set_title(f"{method.upper()} Cluster Sizes")
            plt.tight_layout()
            path = save_figure(fig, output_dir / f"{method}_cluster_sizes.png")
            result.add_file(path)
    
    def _visualize_clusters(
        self,
        plot_df: pd.DataFrame,
        clusters_dict: Dict[str, np.ndarray],
        output_dir: Path,
        result: StageResult,
    ) -> None:
        """Generate cluster visualizations."""
        for method in clusters_dict.keys():
            cluster_col = f"cluster_{method}"
            
            fig, ax = plt.subplots(figsize=self.plot_config.figsize_umap)
            
            palette = create_cluster_palette(plot_df[cluster_col])
            
            sns.scatterplot(
                data=plot_df,
                x="umap_1", y="umap_2",
                hue=cluster_col,
                palette=palette,
                s=self.plot_config.point_size,
                alpha=self.plot_config.alpha,
                legend=False,
                ax=ax,
                rasterized=True,
            )
            
            n_clusters = len([c for c in plot_df[cluster_col].unique() if c != "c-1"])
            ax.set_title(f"{self.level_label} Level UMAP - {method.upper()} ({n_clusters} clusters)")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            
            path = save_figure(fig, output_dir / f"umap_{method}.png")
            result.add_file(path)
    
    def _compute_enrichment(
        self,
        plot_df: pd.DataFrame,
        output_dir: Path,
        result: StageResult,
    ) -> pd.DataFrame:
        """Compute gene enrichment per cluster."""
        logger.info("Computing gene enrichment per cluster...")
        
        # Get clusters to analyze
        cluster_counts = plot_df["cluster"].value_counts()
        min_size = self.analysis_config.min_enrichment_cluster_size
        clusters_to_analyze = [c for c in cluster_counts[cluster_counts >= min_size].index if c != "c-1"]
        
        logger.info(f"Analyzing {len(clusters_to_analyze)} clusters with >={min_size} cells")
        
        if not clusters_to_analyze:
            return pd.DataFrame()
        
        # Parallel enrichment computation
        from ops_utils.hpc.resource_manager import get_optimal_workers
        n_workers = get_optimal_workers(use_gpu=False)
        
        results = Parallel(n_jobs=n_workers)(
            delayed(self._enrichment_for_cluster)(cluster_id, plot_df)
            for cluster_id in tqdm(clusters_to_analyze, desc="Gene enrichment")
        )
        
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            return pd.DataFrame()
        
        enrichment_df = pd.concat(valid_results, ignore_index=True)
        
        # Save full results
        enrichment_df.to_csv(output_dir / "gene_enrichment_per_cluster.csv", index=False)
        result.add_file(output_dir / "gene_enrichment_per_cluster.csv")
        
        # Save top genes per cluster
        top_genes = enrichment_df.groupby("cluster").head(10)
        top_genes.to_csv(output_dir / "top_enriched_genes.csv", index=False)
        result.add_file(output_dir / "top_enriched_genes.csv")
        
        return enrichment_df
    
    def _enrichment_for_cluster(self, cluster_id: str, plot_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute enrichment for a single cluster."""
        in_cluster = plot_df["cluster"] == cluster_id
        genes_in = plot_df[in_cluster]["gene_name"]
        genes_out = plot_df[~in_cluster]["gene_name"]
        
        in_counts = genes_in.value_counts()
        out_counts = genes_out.value_counts()
        
        results = []
        for gene, a in in_counts.items():
            if a < 2:
                continue
            
            b = out_counts.get(gene, 0)
            c = len(genes_in) - a
            d = len(genes_out) - b
            
            odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
            results.append({
                "gene_name": gene,
                "cluster": cluster_id,
                "count_in_cluster": a,
                "count_outside": b,
                "p_value": p_value,
                "odds_ratio": odds_ratio,
            })
        
        if results:
            df = pd.DataFrame(results)
            df["p_adj"] = fdrcorrection(df["p_value"])[1]
            return df.sort_values("p_adj")
        
        return None
