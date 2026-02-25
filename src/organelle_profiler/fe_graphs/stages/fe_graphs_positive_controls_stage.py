"""
Positive Controls Validation Stage: Measure clustering of known gene groups.

Uses predefined gene clusters (e.g., proteasome subunits, ribosome subunits) 
to validate that morphological features capture known biology.

Key Questions:
- Do genes known to function together cluster together in feature space?
- Which organelle's features best recapitulate known functional relationships?

Metrics:
- Intra-cluster cohesion: How tight are known clusters?
- Inter-cluster separation: How distinct are different functional groups?
- Cluster recovery score: Can we recover known clusters from embeddings?
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import spearmanr
import logging

from .fe_graphs_stage_base import BaseStage, StageResult
from ..core.fe_graphs_embedding import EmbeddingEngine
from ..plotting.fe_graphs_utils import save_figure

logger = logging.getLogger(__name__)

# Default path to positive controls
DEFAULT_POSITIVE_CONTROLS_PATH = Path("/hpc/projects/icd.ops/configs/gene_clusters/chad_positive_controls_v3.yml")


class PositiveControlsStage(BaseStage):
    """
    Validate embeddings using known positive control gene clusters.
    
    Runs immediately after embedding stage for early biological validation.
    
    Loads predefined gene clusters and measures:
    1. Whether genes in the same cluster stay together in embeddings
    2. Which organelle best captures known functional relationships
    3. Visualizations highlighting known clusters on UMAPs
    
    Output includes a single-canvas figure showing all positive control
    clusters highlighted on each embedding type (main + per-organelle).
    """
    
    STAGE_NUMBER = 4  # Now runs right after embedding (stage 3)
    STAGE_NAME = "positive_controls"
    
    # Clusters to skip (e.g., NTCs aren't functionally related)
    SKIP_CLUSTERS = {"NTCs"}
    
    # Minimum genes per cluster to analyze
    MIN_GENES_PER_CLUSTER = 2
    
    def __init__(self, *args, positive_controls_path: Optional[Path] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_controls_path = positive_controls_path or DEFAULT_POSITIVE_CONTROLS_PATH
    
    def run(self) -> StageResult:
        """Run positive controls validation."""
        self.log_start("Validating embeddings with positive control clusters")
        result = StageResult()
        
        # Load positive control clusters
        clusters = self._load_positive_controls()
        if not clusters:
            result.add_error("Could not load positive control clusters")
            return result
        
        logger.info(f"Loaded {len(clusters)} positive control clusters")
        result.add_metric("n_clusters_defined", len(clusters))
        
        # Get embedding data from upstream OR load from cache
        organelle_embeddings = {}  # Initialize here so it's available later
        
        if "embedding" in self.upstream:
            # Embedding stage was run - use its results
            embedding = self.upstream["embedding"].data.get("umap_embedding")
            df = self.upstream["embedding"].data.get("df")
            features = self.upstream["embedding"].data.get("features")
            organelle_embeddings = self.upstream["embedding"].data.get("organelle_embeddings", {})
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
            
            embedding = cached_result["embedding"]
            
            # Try to load per-organelle embeddings from cache too
            organelle_groups = self.data.organelle_groups_all_levels.get(self.level, {})
            for org_name in organelle_groups.keys():
                org_cache_key = f"{self.data.experiment}_{self.level}_organelle_{org_name}"
                org_cached = cache.load(org_cache_key, n_cells=len(df))
                if org_cached is not None:
                    organelle_embeddings[org_name] = org_cached["embedding"]
            
            logger.info(f"Loaded cached embedding: {embedding.shape}")
            if organelle_embeddings:
                logger.info(f"Loaded {len(organelle_embeddings)} per-organelle embeddings from cache")
        
        if embedding is None or df is None:
            result.add_error("No embedding data available")
            return result
        
        # Get gene column based on level
        gene_col = self._get_gene_column()
        if gene_col not in df.columns:
            result.add_error(f"Gene column '{gene_col}' not found")
            return result
        
        # Filter clusters to genes present in data
        filtered_clusters = self._filter_clusters_to_data(clusters, df, gene_col)
        n_valid = len(filtered_clusters)
        result.add_metric("n_clusters_with_data", n_valid)
        
        if n_valid < 2:
            result.add_error(f"Only {n_valid} clusters found in data")
            return result
        
        logger.info(f"{n_valid} clusters have genes in this dataset")
        
        # Compute cluster cohesion metrics
        cohesion_results = self._compute_cluster_cohesion(
            embedding, df, gene_col, filtered_clusters
        )
        
        if cohesion_results:
            cohesion_df = pd.DataFrame(cohesion_results)
            cohesion_df = cohesion_df.sort_values("cohesion_score", ascending=False)
            cohesion_df.to_csv(self.output_dir / "cluster_cohesion_scores.csv", index=False)
            result.add_file(self.output_dir / "cluster_cohesion_scores.csv")
            result.data["cohesion_df"] = cohesion_df
            
            # Overall metrics
            result.add_metric("mean_cohesion", cohesion_df["cohesion_score"].mean())
            result.add_metric("best_cluster", cohesion_df.iloc[0]["cluster_name"])
        
        # Generate single-embedding canvas with each cluster highlighted separately
        # This is the PRIMARY output: one embedding, multiple panels, each panel = one cluster
        self._generate_single_embedding_multi_cluster_canvas(
            embedding, df, gene_col, filtered_clusters, result, "All Features"
        )
        
        # Also generate the overview plot with all clusters overlaid (for reference)
        self._generate_umap_highlights(embedding, df, gene_col, filtered_clusters, result)
        self._plot_cohesion_summary(cohesion_df, result)
        
        # Feature importance analysis: which features distinguish each cluster?
        if features is not None:
            logger.info("Computing distinguishing features for each cluster")
            self._analyze_distinguishing_features(
                features, df, gene_col, filtered_clusters, result
            )

            # Organelle-level summary: which organelles are most important?
            logger.info("Computing organelle-level importance for each cluster")
            self._analyze_distinguishing_organelles(
                features, df, gene_col, filtered_clusters, result
            )

            # Radar plots: compare gene group phenotype vs NTCs
            logger.info("Generating radar plots (gene group vs NTCs)")
            self._generate_radar_plots(
                features, df, gene_col, filtered_clusters, result,
                top_n_features=10,  # Show top 10 distinguishing features on radar
            )

        # Generate representative cell visualizations
        if self.data.morphology_path:
            logger.info("Generating representative cell visualizations for clusters...")
            try:
                # Check if tensorstore is available (required for loading zarr v3 images)
                try:
                    import tensorstore as ts
                except ImportError:
                    logger.warning(
                        "tensorstore not available - skipping representative cell visualization. "
                        "Install tensorstore or use zarr_v3_env to enable this feature."
                    )
                    # Don't return - just skip this part and continue
                else:
                    from .fe_graphs_positive_controls_representative_cells import visualize_representative_cells

                    # At guide/gene level, need cell_df and cell_features for:
                    # 1. Looking up actual cells belonging to each guide/gene
                    # 2. Feature-based cell selection (cells that best exemplify the phenotype)
                    cell_df_for_viz = None
                    cell_features_for_viz = None
                    if self.level in ["guide", "gene"]:
                        if "cell" in self.data.adata:
                            # Convert cell-level adata to DataFrame
                            cell_adata = self.data.adata["cell"]
                            cell_df_for_viz = cell_adata.obs.copy()
                            # Extract cell-level features for feature-based cell selection
                            try:
                                cell_features_for_viz = pd.DataFrame(
                                    cell_adata.X,
                                    index=cell_adata.obs_names,
                                    columns=cell_adata.var_names,
                                )
                                logger.info(f"  Loaded cell-level features: {cell_features_for_viz.shape}")
                            except Exception as e:
                                logger.warning(f"Could not extract cell-level features: {e}")
                                cell_features_for_viz = None
                        else:
                            logger.warning(f"Cell-level data not available for {self.level}-level representative cells")
                            cell_df_for_viz = None

                    visualize_representative_cells(
                        features=features,
                        df=df,
                        gene_col=gene_col,
                        clusters=filtered_clusters,
                        result=result,
                        morphology_path=self.data.morphology_path,
                        output_dir=self.output_dir,
                        sanitize_filename_func=self._sanitize_filename,
                        level=self.level,
                        cell_df=cell_df_for_viz if self.level in ["guide", "gene"] else None,
                        cell_features=cell_features_for_viz,  # For feature-based cell selection
                        n_items_per_cluster=10 if self.level == "cell" else 6,  # More items to ensure 10+ cells
                        n_cells_per_item=4,  # Cells per guide/gene (6×4=24 max at gene level)
                        top_n_features=3,
                        skip_complete=self.config.skip_complete,
                        # Pass pre-built mappings from DataContext (single source of truth)
                        channel_names=self.data.channel_names,
                        available_labels=self.data.available_labels,
                        label_to_channel_index=self.data.label_to_channel_index,
                    )
            except Exception as e:
                logger.warning(f"Representative cell visualization failed: {e}")
                import traceback
                traceback.print_exc()

        
        # Per-organelle comparison
        if features is not None:
            self._compare_organelle_cohesion(
                features, df, gene_col, filtered_clusters, result
            )
            
            # Generate per-organelle canvases too (if not too many)
            if organelle_embeddings and len(organelle_embeddings) <= 5:
                for org_name, org_embedding in organelle_embeddings.items():
                    if len(org_embedding) == len(df):
                        self._generate_single_embedding_multi_cluster_canvas(
                            org_embedding, df, gene_col, filtered_clusters, result, org_name
                        )
        
        self.log_complete(result)
        return result
    
    def _load_positive_controls(self) -> Dict[str, Dict]:
        """Load positive control clusters from YAML."""
        if not self.positive_controls_path.exists():
            logger.error(f"Positive controls file not found: {self.positive_controls_path}")
            return {}
        
        try:
            with open(self.positive_controls_path, "r") as f:
                raw_clusters = yaml.safe_load(f)
            
            # Convert to dict with name as key
            clusters = {}
            for cluster_id, cluster_data in raw_clusters.items():
                name = cluster_data.get("name", f"cluster_{cluster_id}")
                genes = cluster_data.get("genes", [])
                
                # Skip unwanted clusters
                if name in self.SKIP_CLUSTERS:
                    continue
                
                if len(genes) >= self.MIN_GENES_PER_CLUSTER:
                    clusters[name] = {
                        "id": cluster_id,
                        "genes": genes,
                    }
            
            return clusters
            
        except Exception as e:
            logger.error(f"Failed to load positive controls: {e}")
            return {}
    
    def _get_gene_column(self) -> str:
        """Get the gene column name based on level."""
        if self.level == "gene":
            return "gene_name" if "gene_name" in self.df.columns else "index"
        else:
            return "gene_name"
    
    def _filter_clusters_to_data(
        self,
        clusters: Dict[str, Dict],
        df: pd.DataFrame,
        gene_col: str,
    ) -> Dict[str, Dict]:
        """Filter clusters to only include genes present in data."""
        genes_in_data = set(df[gene_col].unique())
        
        filtered = {}
        for name, cluster_data in clusters.items():
            genes_present = [g for g in cluster_data["genes"] if g in genes_in_data]
            
            if len(genes_present) >= self.MIN_GENES_PER_CLUSTER:
                filtered[name] = {
                    "id": cluster_data["id"],
                    "genes": genes_present,
                    "genes_missing": [g for g in cluster_data["genes"] if g not in genes_in_data],
                }
        
        return filtered
    
    def _compute_cluster_cohesion(
        self,
        embedding: np.ndarray,
        df: pd.DataFrame,
        gene_col: str,
        clusters: Dict[str, Dict],
    ) -> List[Dict]:
        """Compute cohesion metrics for each known cluster."""
        results = []
        
        # For comparison: compute random baseline
        # At cell level with millions of points, subsample to avoid memory issues
        if len(embedding) > 50000:
            logger.info(f"Subsampling {len(embedding):,} points to 50,000 for baseline distance calculation")
            sample_indices = np.random.choice(len(embedding), 50000, replace=False)
            sample_embedding = embedding[sample_indices]
            all_distances = pdist(sample_embedding)
        else:
            all_distances = pdist(embedding)
        
        random_mean_dist = np.mean(all_distances)
        random_std_dist = np.std(all_distances)
        
        for cluster_name, cluster_data in clusters.items():
            genes = cluster_data["genes"]
            
            # Get positions of cluster genes in embedding
            if self.level == "gene":
                # Gene-level: each row is a gene
                gene_indices = df[df[gene_col].isin(genes)].index.tolist()
                # Convert to integer positions
                gene_positions = [df.index.get_loc(idx) for idx in gene_indices]
            else:
                # Cell/guide level: aggregate by gene
                # Get centroid for each gene
                gene_centroids = []
                for gene in genes:
                    gene_mask = df[gene_col] == gene
                    if gene_mask.sum() > 0:
                        positions = np.where(gene_mask)[0]
                        centroid = embedding[positions].mean(axis=0)
                        gene_centroids.append(centroid)
                
                if len(gene_centroids) < 2:
                    continue
                
                gene_centroids = np.array(gene_centroids)
                
                # Compute intra-cluster distances between gene centroids
                intra_dists = pdist(gene_centroids)
                mean_intra = np.mean(intra_dists) if len(intra_dists) > 0 else np.nan
                
                # Cohesion score: how much tighter than random?
                # Lower ratio = tighter cluster
                cohesion = 1 - (mean_intra / random_mean_dist) if random_mean_dist > 0 else 0
                cohesion = max(0, min(1, cohesion))  # Clip to [0, 1]
                
                # Silhouette-like score for this cluster
                # Compare to random genes
                n_random = len(gene_centroids)
                random_indices = np.random.choice(len(embedding), min(n_random * 10, len(embedding)), replace=False)
                random_points = embedding[random_indices]
                
                # Distance from cluster centroids to random points
                inter_dists = cdist(gene_centroids, random_points).flatten()
                mean_inter = np.mean(inter_dists)
                
                # Silhouette-like: (inter - intra) / max(inter, intra)
                sil_score = (mean_inter - mean_intra) / max(mean_inter, mean_intra) if max(mean_inter, mean_intra) > 0 else 0
                
                results.append({
                    "cluster_name": cluster_name,
                    "cluster_id": cluster_data["id"],
                    "n_genes_defined": len(cluster_data["genes"]),
                    "n_genes_found": len(genes),
                    "mean_intra_distance": mean_intra,
                    "random_baseline_distance": random_mean_dist,
                    "cohesion_score": cohesion,
                    "silhouette_like": sil_score,
                    "genes": ", ".join(genes[:5]) + ("..." if len(genes) > 5 else ""),
                })
                continue
            
            # For gene-level with integer positions
            if len(gene_positions) < 2:
                continue
            
            cluster_embedding = embedding[gene_positions]
            intra_dists = pdist(cluster_embedding)
            mean_intra = np.mean(intra_dists) if len(intra_dists) > 0 else np.nan
            
            cohesion = 1 - (mean_intra / random_mean_dist) if random_mean_dist > 0 else 0
            cohesion = max(0, min(1, cohesion))
            
            results.append({
                "cluster_name": cluster_name,
                "cluster_id": cluster_data["id"],
                "n_genes_defined": len(cluster_data["genes"]),
                "n_genes_found": len(genes),
                "mean_intra_distance": mean_intra,
                "random_baseline_distance": random_mean_dist,
                "cohesion_score": cohesion,
                "genes": ", ".join(genes[:5]) + ("..." if len(genes) > 5 else ""),
            })
        
        return results
    
    def _generate_umap_highlights(
        self,
        embedding: np.ndarray,
        df: pd.DataFrame,
        gene_col: str,
        clusters: Dict[str, Dict],
        result: StageResult,
    ) -> None:
        """Generate UMAP plots highlighting known clusters."""
        highlight_dir = self.output_dir / "cluster_highlights"
        highlight_dir.mkdir(exist_ok=True)
        
        # Color palette for clusters
        n_clusters = len(clusters)
        colors = sns.color_palette("husl", n_clusters)
        
        # 1. Overview plot with all clusters
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Background: all points (larger and more visible)
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c="lightgray", s=self.plot_config.point_size * 1.2, alpha=0.5, rasterized=True, label="_nolegend_"
        )
        
        # Highlight each cluster
        for idx, (cluster_name, cluster_data) in enumerate(clusters.items()):
            genes = cluster_data["genes"]
            
            if self.level == "gene":
                mask = df[gene_col].isin(genes)
                if mask.sum() == 0:
                    continue
                positions = np.where(mask)[0]
                cluster_points = embedding[positions]
            else:
                # Get all cells/guides for these genes
                mask = df[gene_col].isin(genes)
                if mask.sum() == 0:
                    continue
                positions = np.where(mask)[0]
                cluster_points = embedding[positions]
            
            ax.scatter(
                cluster_points[:, 0], cluster_points[:, 1],
                c=[colors[idx]], s=self.plot_config.point_size * 5, alpha=0.95,
                label=f"{cluster_name} ({len(genes)}g)",
                rasterized=True,
                edgecolors="black", linewidths=0.8,
            )
        
        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_title(f"Positive Control Clusters - {self.level.title()} Level", fontsize=14)
        
        # Legend outside plot
        ax.legend(
            bbox_to_anchor=(1.02, 1), loc="upper left",
            fontsize=8, ncol=2 if n_clusters > 20 else 1
        )
        
        plt.tight_layout()
        path = save_figure(fig, highlight_dir / "all_clusters_overview.png", dpi=150)
        result.add_file(path)
        
        # 2. Individual cluster plots (top 10 by cohesion if available)
        cohesion_df = result.data.get("cohesion_df")
        if cohesion_df is not None:
            top_clusters = cohesion_df.head(10)["cluster_name"].tolist()
        else:
            top_clusters = list(clusters.keys())[:10]
        
        for cluster_name in top_clusters:
            if cluster_name not in clusters:
                continue
            
            genes = clusters[cluster_name]["genes"]
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Background (larger and more visible)
            ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c="lightgray", s=self.plot_config.point_size * 1.0, alpha=0.4, rasterized=True,
            )
            
            # Highlight cluster (much larger and more prominent)
            mask = df[gene_col].isin(genes)
            if mask.sum() == 0:
                plt.close(fig)
                continue
            
            positions = np.where(mask)[0]
            cluster_points = embedding[positions]
            
            ax.scatter(
                cluster_points[:, 0], cluster_points[:, 1],
                c="red", s=self.plot_config.point_size * 10, alpha=0.95, label=cluster_name,
                edgecolors="black", linewidths=1.0,
                rasterized=True,
            )
            
            # Add gene labels at centroids (for gene level) or just count
            if self.level == "gene":
                for gene in genes:
                    gene_mask = df[gene_col] == gene
                    if gene_mask.sum() > 0:
                        pos = np.where(gene_mask)[0][0]
                        ax.annotate(
                            gene, embedding[pos],
                            fontsize=8, alpha=0.8,
                            xytext=(5, 5), textcoords="offset points",
                        )
            else:
                # Show gene counts
                gene_counts = df.loc[mask, gene_col].value_counts()
                gene_text = "\n".join([f"{g}: {c}" for g, c in gene_counts.head(5).items()])
                ax.text(
                    0.02, 0.98, gene_text,
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
            
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title(f"{cluster_name}\n({len(genes)} genes)")
            ax.legend()
            
            plt.tight_layout()
            safe_name = self._sanitize_filename(cluster_name)
            cluster_dir = self.output_dir / safe_name
            cluster_dir.mkdir(parents=True, exist_ok=True)
            path = save_figure(fig, cluster_dir / "cluster_highlight.png")
            result.add_file(path)
    
    def _plot_cohesion_summary(
        self,
        cohesion_df: pd.DataFrame,
        result: StageResult,
    ) -> None:
        """Plot summary of cluster cohesion scores."""
        if cohesion_df is None or cohesion_df.empty:
            return
        
        # Bar chart of cohesion scores
        fig, ax = plt.subplots(figsize=(14, max(8, len(cohesion_df) * 0.3)))
        
        # Sort and plot
        sorted_df = cohesion_df.sort_values("cohesion_score", ascending=True)
        
        colors = sns.color_palette("RdYlGn", len(sorted_df))
        
        bars = ax.barh(
            sorted_df["cluster_name"],
            sorted_df["cohesion_score"],
            color=colors,
        )
        
        ax.set_xlabel("Cohesion Score (higher = tighter cluster)", fontsize=12)
        ax.set_ylabel("Positive Control Cluster", fontsize=12)
        ax.set_title(
            f"Cluster Cohesion Validation - {self.level.title()} Level\n"
            "(Do known functional groups cluster together?)",
            fontsize=14,
        )
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
        
        # Add value labels
        for bar, val in zip(bars, sorted_df["cohesion_score"]):
            ax.text(
                val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center", fontsize=9,
            )
        
        plt.tight_layout()
        path = save_figure(fig, self.output_dir / "cluster_cohesion_summary.png")
        result.add_file(path)
        
        # Distribution plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(cohesion_df["cohesion_score"], bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(
            cohesion_df["cohesion_score"].mean(),
            color="red", linestyle="--",
            label=f"Mean: {cohesion_df['cohesion_score'].mean():.2f}",
        )
        ax.axvline(0.5, color="gray", linestyle=":", label="Random baseline")
        
        ax.set_xlabel("Cohesion Score")
        ax.set_ylabel("Number of Clusters")
        ax.set_title("Distribution of Cluster Cohesion Scores")
        ax.legend()
        
        plt.tight_layout()
        path = save_figure(fig, self.output_dir / "cluster_cohesion_distribution.png")
        result.add_file(path)
    
    def _compare_organelle_cohesion(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        gene_col: str,
        clusters: Dict[str, Dict],
        result: StageResult,
    ) -> None:
        """Compare cohesion across different organelle feature sets."""
        logger.info("Comparing cluster cohesion across organelles...")
        
        organelle_features = self.group_features_by_organelle(features.columns.tolist())
        
        if len(organelle_features) < 2:
            return
        
        organelle_cohesion = []
        
        for organelle, cols in organelle_features.items():
            if len(cols) < 5:
                continue
            
            org_features = features[cols].copy().fillna(0)
            
            # Scale and compute UMAP
            scaler = StandardScaler()
            scaled = scaler.fit_transform(org_features)
            
            # Use PCA-reduced space for speed
            from sklearn.decomposition import PCA
            n_components = min(10, scaled.shape[1])
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(scaled)
            
            # Compute mean cohesion for this organelle
            cohesion_scores = []
            
            for cluster_name, cluster_data in clusters.items():
                genes = cluster_data["genes"]
                
                if self.level == "gene":
                    mask = df[gene_col].isin(genes)
                    if mask.sum() < 2:
                        continue
                    positions = np.where(mask)[0]
                    cluster_points = reduced[positions]
                else:
                    # Compute gene centroids
                    centroids = []
                    for gene in genes:
                        gene_mask = df[gene_col] == gene
                        if gene_mask.sum() > 0:
                            positions = np.where(gene_mask)[0]
                            centroid = reduced[positions].mean(axis=0)
                            centroids.append(centroid)
                    
                    if len(centroids) < 2:
                        continue
                    
                    cluster_points = np.array(centroids)
                
                # Compute cohesion
                # Subsample if needed to avoid memory issues
                if len(reduced) > 50000:
                    sample_indices = np.random.choice(len(reduced), 50000, replace=False)
                    all_dists = pdist(reduced[sample_indices])
                else:
                    all_dists = pdist(reduced)
                random_mean = np.mean(all_dists)
                
                intra_dists = pdist(cluster_points)
                if len(intra_dists) > 0:
                    cohesion = 1 - (np.mean(intra_dists) / random_mean)
                    cohesion = max(0, min(1, cohesion))
                    cohesion_scores.append(cohesion)
            
            if cohesion_scores:
                organelle_cohesion.append({
                    "organelle": organelle,
                    "mean_cohesion": np.mean(cohesion_scores),
                    "std_cohesion": np.std(cohesion_scores),
                    "n_clusters": len(cohesion_scores),
                    "n_features": len(cols),
                })
        
        if not organelle_cohesion:
            return
        
        org_df = pd.DataFrame(organelle_cohesion)
        org_df = org_df.sort_values("mean_cohesion", ascending=False)
        org_df.to_csv(self.output_dir / "organelle_cohesion_comparison.csv", index=False)
        result.add_file(self.output_dir / "organelle_cohesion_comparison.csv")
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = sns.color_palette("viridis", len(org_df))
        
        bars = ax.bar(org_df["organelle"], org_df["mean_cohesion"], color=colors)
        ax.errorbar(
            range(len(org_df)), org_df["mean_cohesion"],
            yerr=org_df["std_cohesion"],
            fmt="none", c="black", capsize=3,
        )
        
        ax.set_xlabel("Organelle / Segmentation Group")
        ax.set_ylabel("Mean Cluster Cohesion Score")
        ax.set_title(
            "Which Organelle Best Captures Known Functional Relationships?\n"
            "(Higher = known clusters stay together better)"
        )
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        
        # Add value labels
        for bar, val in zip(bars, org_df["mean_cohesion"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center", fontsize=9,
            )
        
        plt.tight_layout()
        path = save_figure(fig, self.output_dir / "organelle_cohesion_comparison.png")
        result.add_file(path)
        
        # Best organelle
        best = org_df.iloc[0]
        result.add_metric("best_organelle_for_biology", best["organelle"])
        result.add_metric("best_organelle_cohesion", best["mean_cohesion"])
        
        logger.info(f"Best organelle for known biology: {best['organelle']} ({best['mean_cohesion']:.2f})")
    
    def _generate_single_embedding_multi_cluster_canvas(
        self,
        embedding: np.ndarray,
        df: pd.DataFrame,
        gene_col: str,
        clusters: Dict[str, Dict],
        result: StageResult,
        embedding_name: str = "All Features",
    ) -> None:
        """
        Generate a single canvas with one embedding, each subplot highlights a different cluster.
        
        Creates a multi-panel figure where:
        - Same embedding shown in each panel
        - Each panel highlights ONE positive control group
        - Easy visual comparison of where each functional group lands
        """
        logger.info(f"Generating single-embedding multi-cluster canvas for {embedding_name}...")
        
        n_clusters = len(clusters)
        
        if n_clusters == 0:
            return
        
        # Determine grid layout (aim for roughly square)
        n_cols = min(4, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols
        
        # Create figure
        fig_width = 5 * n_cols
        fig_height = 4.5 * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        if n_clusters == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Use distinct colors for each cluster
        cluster_colors = sns.color_palette("husl", n_clusters)
        
        # Plot each cluster in its own subplot
        for idx, (cluster_name, cluster_data) in enumerate(clusters.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            genes = cluster_data["genes"]
            mask = df[gene_col].isin(genes)
            positions = np.where(mask)[0]
            
            # Background: all points in light gray (larger and more visible)
            ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c="lightgray", s=self.plot_config.point_size * 1.0, alpha=0.4, rasterized=True,
            )
            
            # Highlight this cluster (much larger and more prominent)
            if mask.sum() > 0 and positions.max() < len(embedding):
                cluster_points = embedding[positions]
                
                ax.scatter(
                    cluster_points[:, 0], cluster_points[:, 1],
                    c=[cluster_colors[idx]], s=self.plot_config.point_size * 10, alpha=0.95,
                    edgecolors="black", linewidths=1.0,
                    rasterized=True,
                )
                
                # Add gene labels for small clusters at gene level
                if self.level == "gene" and len(genes) <= 15:
                    for gene in genes:
                        gene_mask = df[gene_col] == gene
                        if gene_mask.sum() > 0:
                            pos_idx = np.where(gene_mask)[0][0]
                            if pos_idx < len(embedding):
                                ax.annotate(
                                    gene, embedding[pos_idx],
                                    fontsize=7, alpha=0.9,
                                    xytext=(3, 3), textcoords="offset points",
                                )
            
            # Title with cluster name and gene count
            n_genes_found = mask.sum() if self.level == "gene" else len(genes)
            ax.set_title(f"{cluster_name}\n({n_genes_found} genes)", fontsize=10, fontweight="bold")
            ax.set_xlabel("UMAP 1", fontsize=8)
            ax.set_ylabel("UMAP 2", fontsize=8)
            ax.tick_params(labelsize=7)
            
            # Remove spines for cleaner look
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
        
        # Hide empty subplots
        for idx in range(n_clusters, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")
        
        fig.suptitle(
            f"Positive Control Clusters - {self.level.title()} Level ({embedding_name})\n"
            f"Each panel highlights one functional gene group",
            fontsize=13, fontweight="bold", y=1.02,
        )
        
        plt.tight_layout()
        
        # Save
        safe_name = embedding_name.replace(" ", "_").lower()
        path = save_figure(fig, self.output_dir / f"positive_controls_canvas_{safe_name}.png", dpi=150)
        result.add_file(path)
        
        logger.info(f"Saved canvas: {path.name}")
    
    def _analyze_distinguishing_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        gene_col: str,
        clusters: Dict[str, Dict],
        result: StageResult,
        top_n: int = 40,
    ) -> None:
        """
        Identify features that most distinguish each positive control cluster from all other genes.
        
        Uses Cohen's d effect size to rank features by their ability to discriminate
        the cluster from background.
        
        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix (items x features)
        df : pd.DataFrame
            Full dataframe with gene annotations
        gene_col : str
            Column name containing gene identifiers
        clusters : dict
            Positive control clusters
        result : StageResult
            Results container
        top_n : int
            Number of top features to display per cluster (default: 40)
        """
        logger.info(f"Computing distinguishing features (top {top_n}) for each cluster...")

        # Outputs will be organized by cluster subdirectory (not by analysis type)
        
        # Try to load Cohen's d statistics from cache
        from ..core.fe_graphs_cache import EmbeddingCache
        cache = EmbeddingCache(self.data.cache_dir)
        cache_key = f"{self.data.experiment}_{self.level}_positive_controls"
        n_items = len(df)
        
        cached_cohens_d = cache.load_cohens_d(cache_key, n_items)
        
        if cached_cohens_d is not None:
            # Use cached results!
            logger.info(f"  Using cached Cohen's d statistics ({len(cached_cohens_d)} clusters)")
            cohens_d_dict = cached_cohens_d
        else:
            # Compute Cohen's d for all clusters
            from tqdm import tqdm
            logger.info(f"  Computing Cohen's d for {len(clusters)} clusters (this will be cached)...")
            cohens_d_dict = {}
            
            for cluster_name, cluster_data in tqdm(clusters.items(), desc="Computing Cohen's d"):
                genes = cluster_data["genes"]
                
                # Get mask for this cluster
                cluster_mask = df[gene_col].isin(genes)
                n_cluster = cluster_mask.sum()
                
                if n_cluster < 1:
                    logger.warning(f"  {cluster_name}: No items found, skipping")
                    continue
                
                # Align features with df
                if len(features) != len(df):
                    logger.warning(f"  Features length mismatch for {cluster_name}")
                    continue
                
                cluster_features = features.loc[cluster_mask]
                other_features = features.loc[~cluster_mask]
                
                # Compute Cohen's d for each feature
                feature_scores = []
                for feat in features.columns:
                    cluster_vals = cluster_features[feat].dropna()
                    other_vals = other_features[feat].dropna()
                    
                    # Need at least 1 value in cluster, 2 in other for comparison
                    if len(cluster_vals) < 1 or len(other_vals) < 2:
                        continue
                    
                    # Cohen's d = (mean1 - mean2) / pooled_std
                    mean_cluster = cluster_vals.mean()
                    mean_other = other_vals.mean()
                    
                    # For single-item clusters, use other_std as pooled_std
                    if len(cluster_vals) == 1:
                        std_cluster = 0
                        std_other = other_vals.std()
                        pooled_std = std_other
                    else:
                        std_cluster = cluster_vals.std()
                        std_other = other_vals.std()
                        
                        n1 = len(cluster_vals)
                        n2 = len(other_vals)
                        
                        # Pooled standard deviation
                        pooled_std = np.sqrt(((n1 - 1) * std_cluster**2 + (n2 - 1) * std_other**2) / (n1 + n2 - 2))
                    
                    if pooled_std == 0:
                        continue
                    
                    cohens_d = (mean_cluster - mean_other) / pooled_std
                    
                    feature_scores.append({
                        "feature": feat,
                        "cohens_d": cohens_d,
                        "abs_cohens_d": abs(cohens_d),
                        "mean_cluster": mean_cluster,
                        "mean_other": mean_other,
                        "fold_change": mean_cluster / mean_other if mean_other != 0 else np.nan,
                    })
                
                if not feature_scores:
                    logger.warning(f"  {cluster_name}: No valid features")
                    continue
                
                # Sort by absolute Cohen's d (most distinguishing regardless of direction)
                feature_df = pd.DataFrame(feature_scores)
                feature_df = feature_df.sort_values("abs_cohens_d", ascending=False)
                
                cohens_d_dict[cluster_name] = feature_df
            
            # Save to cache for future runs
            cache.save_cohens_d(cache_key, cohens_d_dict, n_items)
        
        # Now generate plots and outputs using the (cached or freshly computed) Cohen's d
        from tqdm import tqdm
        all_feature_importance = []
        
        for cluster_name, feature_df in tqdm(cohens_d_dict.items(), desc="Generating feature plots"):
            # Get n_cluster for title
            genes = clusters[cluster_name]["genes"]
            cluster_mask = df[gene_col].isin(genes)
            n_cluster = cluster_mask.sum()

            # Create cluster-specific output directory
            # IMPORTANT: Use _sanitize_filename for consistency with reading code
            safe_cluster_name = self._sanitize_filename(cluster_name)
            cluster_dir = self.output_dir / safe_cluster_name
            cluster_dir.mkdir(parents=True, exist_ok=True)

            # Save full feature importance table
            csv_path = cluster_dir / "feature_importance.csv"
            feature_df.to_csv(csv_path, index=False)
            result.add_file(csv_path)
            
            # Store for cross-cluster analysis
            for _, row in feature_df.head(top_n).iterrows():
                all_feature_importance.append({
                    "cluster": cluster_name,
                    "feature": row["feature"],
                    "cohens_d": row["cohens_d"],
                    "abs_cohens_d": row["abs_cohens_d"],
                })
            
            # Generate bar plot for top N features
            top_features = feature_df.head(top_n)
            
            fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.25)))
            
            # Color bars by direction (positive = enriched in cluster, negative = depleted)
            colors = ["#d62728" if d > 0 else "#1f77b4" for d in top_features["cohens_d"]]
            
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_features["cohens_d"], color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
            
            # Add Cohen's d value annotations INSIDE bars (tight to the bar edge)
            for i, (d_val, feat_name) in enumerate(zip(top_features["cohens_d"], top_features["feature"])):
                # Position text based on bar direction
                if d_val > 0:
                    # Positive bars: text on the right inside the bar
                    x_pos = d_val - abs(d_val) * 0.05  # 5% inset from edge
                    ha = 'right'
                    color = 'white'
                else:
                    # Negative bars: text on the left inside the bar
                    x_pos = d_val + abs(d_val) * 0.05  # 5% inset from edge
                    ha = 'left'
                    color = 'white'
                
                # Only show if bar is wide enough (abs value > 0.3)
                if abs(d_val) > 0.3:
                    ax.text(x_pos, i, f'{d_val:.2f}', 
                           va='center', ha=ha, fontsize=8, color=color, fontweight='bold')
            
            # Feature names on y-axis
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features["feature"], fontsize=9)
            ax.invert_yaxis()  # Highest at top
            
            # Labels and title
            ax.set_xlabel("Cohen's d (effect size)", fontsize=11, fontweight="bold")
            ax.set_title(
                f"Top {top_n} Distinguishing Features\n{cluster_name} ({n_cluster} items)",
                fontsize=13,
                fontweight="bold",
                pad=15,
            )
            
            # Add zero line
            ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#d62728", edgecolor="black", label="Enriched in cluster"),
                Patch(facecolor="#1f77b4", edgecolor="black", label="Depleted in cluster"),
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
            
            # Add grid
            ax.grid(axis="x", alpha=0.3, linestyle=":")
            
            plt.tight_layout()

            # Save to cluster-specific directory
            plot_path = cluster_dir / "top_features.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            result.add_file(plot_path)
            
            # Don't log each cluster individually - use tqdm progress bar instead
        
        # Create summary heatmap of top features across all clusters (in root output_dir)
        if all_feature_importance:
            self._plot_feature_importance_heatmap(all_feature_importance, self.output_dir, result, top_n)
    
    def _plot_feature_importance_heatmap(
        self,
        all_importance: List[Dict],
        output_dir: Path,
        result: StageResult,
        top_n: int,
    ) -> None:
        """
        Create a heatmap showing which features distinguish which clusters.
        
        Helps identify common distinguishing features across functional groups.
        """
        logger.info("Creating cross-cluster feature importance heatmap...")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_importance)
        
        # Pivot to matrix: clusters x features
        pivot = df.pivot_table(
            index="feature",
            columns="cluster",
            values="cohens_d",
            aggfunc="first",
        )
        
        # Sort features by max absolute importance
        pivot["max_abs"] = pivot.abs().max(axis=1)
        pivot = pivot.sort_values("max_abs", ascending=False).drop(columns="max_abs")
        
        # Limit to top features overall
        pivot = pivot.head(min(50, len(pivot)))
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.8), max(8, len(pivot) * 0.3)))
        
        # Use diverging colormap centered at zero
        sns.heatmap(
            pivot,
            cmap="RdBu_r",
            center=0,
            cbar_kws={"label": "Cohen's d"},
            linewidths=0.5,
            linecolor="lightgray",
            ax=ax,
            vmin=-3,
            vmax=3,
            fmt=".2f",
        )
        
        ax.set_xlabel("Positive Control Cluster", fontsize=11, fontweight="bold")
        ax.set_ylabel("Feature", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Feature Importance Across Positive Control Clusters\n"
            f"(Top {top_n} features per cluster, Cohen's d effect size)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), fontsize=8)
        
        plt.tight_layout()
        
        # Save
        path = output_dir / "feature_importance_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        result.add_file(path)
        
        logger.info(f"Saved feature importance heatmap: {path.name}")
    
    def _analyze_distinguishing_organelles(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        gene_col: str,
        clusters: Dict[str, Dict],
        result: StageResult,
    ) -> None:
        """
        Aggregate distinguishing features by organelle to identify which organelles
        best explain each positive control cluster.
        
        This is a higher-level summary than per-feature analysis: instead of showing
        individual features, we group by organelle and aggregate the effect sizes.
        
        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix (items x features)
        df : pd.DataFrame
            Full dataframe with gene annotations
        gene_col : str
            Column name containing gene identifiers
        clusters : dict
            Positive control clusters
        result : StageResult
            Results container
        """
        from tqdm import tqdm
        
        logger.info("Computing organelle-level importance for each cluster...")

        # Organelle outputs go in cluster-specific subdirectories

        all_organelle_importance = []  # For cross-cluster heatmap

        for cluster_name, cluster_data in tqdm(clusters.items(), desc="Analyzing organelle importance"):
            genes = cluster_data["genes"]

            # Get mask for this cluster
            cluster_mask = df[gene_col].isin(genes)
            n_cluster = cluster_mask.sum()

            if n_cluster < 1:
                continue

            # Load feature importance from cluster-specific directory
            safe_cluster_name = self._sanitize_filename(cluster_name)
            cluster_dir = self.output_dir / safe_cluster_name
            csv_path = cluster_dir / "feature_importance.csv"
            if not csv_path.exists():
                continue
            
            feature_df = pd.read_csv(csv_path)
            
            if feature_df.empty:
                continue
            
            # Map features to organelles using the level-specific organelle_groups map
            # This ensures consistency with the rest of the pipeline
            feature_to_organelle = {}
            
            # Use level-specific organelle groups if available
            if hasattr(self.data, 'organelle_groups_all_levels') and self.data.organelle_groups_all_levels:
                if self.level in self.data.organelle_groups_all_levels:
                    organelle_groups_to_use = self.data.organelle_groups_all_levels[self.level]
                else:
                    organelle_groups_to_use = self.data.organelle_groups
            else:
                organelle_groups_to_use = self.data.organelle_groups
            
            for org_name, org_features in organelle_groups_to_use.items():
                for feat in org_features:
                    feature_to_organelle[feat] = org_name
            
            # Assign organelle based on global map
            feature_df["organelle"] = feature_df["feature"].map(feature_to_organelle)
            
            # Filter out features not in organelle groups (metadata/unmapped)
            feature_df = feature_df[feature_df["organelle"].notna()]
            
            if feature_df.empty:
                continue
            
            # Aggregate by organelle using MAX Cohen's d (simplest, most interpretable)
            # If one feature in an organelle really distinguishes the group, that organelle shines
            organelle_agg = feature_df.groupby("organelle").agg({
                "abs_cohens_d": ["max", "count"],
                "cohens_d": "mean"  # For direction (enriched vs depleted)
            }).reset_index()
            
            # Flatten column names
            organelle_agg.columns = ["organelle", "max_abs_d", "n_features", "mean_cohens_d"]
            
            # For each organelle, find the feature with max |Cohen's d|
            max_feature_per_organelle = {}
            for org_name in organelle_agg["organelle"]:
                org_features = feature_df[feature_df["organelle"] == org_name]
                max_idx = org_features["abs_cohens_d"].idxmax()
                max_feature_per_organelle[org_name] = org_features.loc[max_idx, "feature"]
            
            organelle_agg["max_feature"] = organelle_agg["organelle"].map(max_feature_per_organelle)
            
            # Sort by max_abs_d (descending = top to bottom, largest to smallest)
            organelle_agg = organelle_agg.sort_values("max_abs_d", ascending=False)


            
            # Save CSV with all metrics to cluster directory
            csv_out = cluster_dir / "organelle_importance.csv"
            organelle_agg.to_csv(csv_out, index=False)
            result.add_file(csv_out)
            
            # Store for cross-cluster analysis
            for _, row in organelle_agg.iterrows():
                all_organelle_importance.append({
                    "cluster_name": cluster_name,
                    "organelle": row["organelle"],
                    "max_abs_d": row["max_abs_d"],
                    "max_feature": row["max_feature"],
                    "n_features": row["n_features"],
                })

            
            # Generate bar chart for this cluster
            # Show max |Cohen's d| per organelle, labeled with the specific feature name
            # Add extra width for gene list on the right
            fig, ax = plt.subplots(figsize=(18, max(8, len(organelle_agg) * 0.6)))
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(organelle_agg)))
            
            y_pos = np.arange(len(organelle_agg))
            
            # Main bars: max |Cohen's d|
            bars = ax.barh(y_pos, organelle_agg["max_abs_d"], color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
            
            # Annotate with max value and feature name
            for i, (idx, row) in enumerate(organelle_agg.iterrows()):
                bar_val = row["max_abs_d"]
                feature_name = row["max_feature"]
                n_feat = int(row["n_features"])
                
                # Position text at 95% of bar width (inside bar)
                text_x = bar_val * 0.95
                
                # Show max value inside bar
                ax.text(text_x, i, f"{bar_val:.2f}", va='center', ha='right', 
                       fontsize=10, fontweight='bold', color='white')
                
                # Add full feature name outside bar
                ax.text(bar_val + max(organelle_agg["max_abs_d"]) * 0.02, i, 
                       f"{feature_name} (N={n_feat})", 
                       va='center', ha='left', fontsize=8, color='black', style='italic')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(organelle_agg["organelle"], fontsize=11, fontweight='bold')
            ax.set_xlabel("Max |Cohen's d| (Best Single Feature)", fontsize=13, fontweight='bold')
            ax.set_ylabel("Organelle Group", fontsize=13, fontweight='bold')
            ax.set_title(
                f"Organelle Importance for {cluster_name}\n({n_cluster} items)\n"
                f"Ranked by best discriminating feature",
                fontsize=14,
                fontweight="bold",
                pad=15,
            )
            ax.grid(axis='x', alpha=0.3, linestyle=':')
            ax.invert_yaxis()  # Largest at top
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='gray', alpha=0.3, label='Bar = Max |Cohen\'s d| in organelle'),
                Patch(facecolor='white', alpha=0, label='Label = Feature name (N features)'),
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)
            
            # Add gene list on the right side
            gene_list = genes
            gene_text = f"Genes in {cluster_name}:\n" + "\n".join([f"  • {gene}" for gene in sorted(gene_list)])
            
            # Position text box on the right side of the plot
            ax.text(
                1.02, 0.5, gene_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=0.8),
                family='monospace'
            )

            
            plt.tight_layout(rect=[0, 0, 0.75, 1])  # Leave space on right for gene list
            plot_path = cluster_dir / "organelle_importance.png"
            save_figure(fig, plot_path, dpi=150)
            result.add_file(plot_path)

            # Don't log each cluster individually - use tqdm progress bar instead

        # Generate cross-cluster heatmap (in root output_dir)
        if all_organelle_importance:
            self._plot_organelle_importance_heatmap(all_organelle_importance, self.output_dir, result)
    
    def _plot_organelle_importance_heatmap(
        self,
        all_organelle_importance: List[Dict],
        output_dir: Path,
        result,
    ) -> None:
        """
        Generate a heatmap showing organelle importance across all clusters.
        
        Parameters
        ----------
        all_organelle_importance : list
            List of dicts with cluster_name, organelle, and importance scores
        output_dir : Path
            Output directory
        result : StageResult
            Results container
        """
        logger.info("Generating organelle importance heatmap across clusters...")
        
        if not all_organelle_importance:
            logger.warning("No organelle importance data to plot.")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(all_organelle_importance)
        
        # Pivot to get organelles as index, clusters as columns
        # Use max_abs_d as the metric (best single feature per organelle)
        pivot_table = df.pivot_table(
            index="organelle",
            columns="cluster_name",
            values="max_abs_d",
            fill_value=0
        )
        
        # Sort organelles by total importance across all clusters (sum of max values)
        organelle_totals = pivot_table.sum(axis=1).sort_values(ascending=False)
        pivot_table = pivot_table.loc[organelle_totals.index]
        
        # Sort clusters by total organelle importance (most discriminative clusters first)
        cluster_totals = pivot_table.sum(axis=0).sort_values(ascending=False)
        pivot_table = pivot_table[cluster_totals.index]
        
        if pivot_table.empty:
            logger.warning("Pivot table is empty. Skipping heatmap.")
            return
        
        # Plotting
        fig, ax = plt.subplots(figsize=(max(10, len(pivot_table.columns) * 0.6), max(8, len(pivot_table.index) * 0.5)))
        
        # Use a sequential colormap (higher = more important)
        sns.heatmap(
            pivot_table,
            cmap="YlOrRd",  # Yellow to Orange to Red
            annot=True,
            fmt=".2f",  # Show 2 decimal places for max |d| values
            linewidths=0.5,
            linecolor="lightgray",
            cbar_kws={"label": "Max |Cohen's d| (Best Feature)"},
            ax=ax,
        )
        
        ax.set_title(
            "Organelle Importance Across Positive Control Clusters\n(Max |Cohen's d| per organelle)",
            fontsize=14,
            fontweight="bold",
            pad=15
        )
        ax.set_xlabel("Positive Control Cluster", fontsize=12)
        ax.set_ylabel("Organelle", fontsize=12)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.tick_params(axis='y', rotation=0, labelsize=10)
        
        plt.tight_layout()
        heatmap_path = output_dir / "organelle_importance_heatmap.png"
        save_figure(fig, heatmap_path, dpi=150)
        result.add_file(heatmap_path)
        logger.info(f"Saved organelle importance heatmap: {heatmap_path.name}")
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize string for use as a filename."""
        # Replace spaces and special characters with underscores
        s = re.sub(r'[^\w\s-]', '', name) # Remove non-alphanumeric except space and hyphen
        s = re.sub(r'[\s-]+', '_', s).strip('_') # Replace spaces/hyphens with single underscore
        return s

    def _load_ntc_genes(self) -> List[str]:
        """Load NTC genes from positive controls YAML (even though they're in SKIP_CLUSTERS)."""
        if not self.positive_controls_path.exists():
            logger.warning(f"Positive controls file not found: {self.positive_controls_path}")
            return []

        try:
            with open(self.positive_controls_path, "r") as f:
                raw_clusters = yaml.safe_load(f)

            for cluster_id, cluster_data in raw_clusters.items():
                name = cluster_data.get("name", "")
                if name == "NTCs":
                    ntc_genes = cluster_data.get("genes", [])
                    logger.info(f"Loaded {len(ntc_genes)} NTC genes from YAML (e.g., {ntc_genes[:3]}...)")
                    return ntc_genes
            logger.warning("No 'NTCs' cluster found in positive controls YAML")
            return []
        except Exception as e:
            logger.warning(f"Could not load NTC genes: {e}")
            return []

    def _identify_ntc_mask(self, df: pd.DataFrame, gene_col: str) -> pd.Series:
        """
        Identify NTC (non-targeting control) rows using multiple methods.

        Uses the same logic as validate_feature_anndata.py:
        1. gene_id == -1 or NCBI_ID == -1 (if columns exist)
        2. gene_name is None or empty string
        3. barcode column starts with "NTC_" or contains NTC patterns

        Returns a boolean mask for NTC rows.
        """
        ntc_mask = pd.Series(False, index=df.index)
        detection_methods = []

        # Method 1: Check for gene_id == -1 or NCBI_ID == -1 (most reliable)
        if "gene_id" in df.columns:
            gene_id_ntc = df["gene_id"] == -1
            n_gene_id_ntc = gene_id_ntc.sum()
            if n_gene_id_ntc > 0:
                ntc_mask |= gene_id_ntc
                detection_methods.append(f"gene_id==-1: {n_gene_id_ntc}")

        if "NCBI_ID" in df.columns:
            ncbi_id_ntc = df["NCBI_ID"] == -1
            n_ncbi_id_ntc = ncbi_id_ntc.sum()
            if n_ncbi_id_ntc > 0:
                ntc_mask |= ncbi_id_ntc
                detection_methods.append(f"NCBI_ID==-1: {n_ncbi_id_ntc}")

        # Method 2: Check for None or empty gene_name (common in OPS data for NTCs)
        if gene_col in df.columns:
            none_or_empty = (
                df[gene_col].isna() |
                (df[gene_col].astype(str).str.strip() == "") |
                (df[gene_col].astype(str) == "None")
            )
            n_none = none_or_empty.sum()
            if n_none > 0:
                ntc_mask |= none_or_empty
                detection_methods.append(f"gene_name is None/empty: {n_none}")

        # Method 3: Check barcode column for NTC patterns (NTC guides have NTC_ prefix)
        if "barcode" in df.columns:
            barcode_ntc = df["barcode"].astype(str).str.upper().str.startswith("NTC")
            n_barcode_ntc = barcode_ntc.sum()
            if n_barcode_ntc > 0:
                ntc_mask |= barcode_ntc
                detection_methods.append(f"barcode starts with NTC: {n_barcode_ntc}")

        # Method 4: String pattern matching on gene_name (fallback)
        if gene_col in df.columns:
            NTC_PATTERNS = ["ntc", "non-targeting", "^0$"]
            pattern = "|".join(NTC_PATTERNS)
            pattern_match = df[gene_col].astype(str).str.contains(pattern, case=False, regex=True, na=False)
            n_pattern = pattern_match.sum()
            if n_pattern > 0:
                ntc_mask |= pattern_match
                detection_methods.append(f"pattern match: {n_pattern}")

        n_ntc = ntc_mask.sum()
        if n_ntc > 0:
            logger.info(f"Found {n_ntc} NTC rows using: {', '.join(detection_methods)}")
        else:
            logger.warning("No NTC rows found using any detection method")
            # Log sample data to help debug
            if "barcode" in df.columns:
                sample_barcodes = df["barcode"].head(10).tolist()
                logger.warning(f"  Sample barcodes: {sample_barcodes}")
            if gene_col in df.columns:
                sample_genes = df[gene_col].head(10).tolist()
                logger.warning(f"  Sample {gene_col}: {sample_genes}")

        return ntc_mask

    def _group_features_by_base_metric(
        self,
        feature_df: pd.DataFrame,
        top_n_groups: int = 10,
    ) -> List[Dict]:
        """
        Group features by their base metric when aggregation suffixes are present.

        Features like 'cp1_mito_area_mean_std' and 'cp1_mito_area_mean_max' share
        the base metric 'cp1_mito_area' with different aggregation suffixes.

        Only groups features where all Cohen's d values have the same sign
        (all positive or all negative), indicating they change in the same direction.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Feature importance with columns: feature, cohens_d, abs_cohens_d
        top_n_groups : int
            Number of top feature groups to return

        Returns
        -------
        List[Dict]
            List of feature groups with keys:
            - 'base_metric': base feature name without suffix
            - 'display_name': formatted name with suffixes in parentheses
            - 'features': list of original feature names
            - 'suffixes': list of aggregation suffixes
            - 'cohens_d': mean Cohen's d across grouped features
            - 'abs_cohens_d': max absolute Cohen's d (for ranking)
        """
        # Single-level aggregation suffixes (applied iteratively to strip all levels)
        # At cell level: organelle stats aggregated with _mean, _std, etc.
        # At guide/gene level: cell stats further aggregated, so _mean_std means
        # the std (across guides) of the mean (across organelles)
        SINGLE_SUFFIXES = ['_mean', '_std', '_median', '_max', '_min']

        def extract_base_and_suffix(feature_name: str) -> Tuple[str, str]:
            """
            Extract base metric and full aggregation suffix from feature name.

            Recursively strips all aggregation suffixes to get the true base metric.
            Example: 'cell_area_mean_std' -> ('cell_area', 'mean_std')
            Example: 'cp1_mito_area_mean' -> ('cp1_mito_area', 'mean')
            """
            suffixes_found = []
            current = feature_name

            # Iteratively strip suffixes from the end
            while True:
                found_suffix = False
                for suffix in SINGLE_SUFFIXES:
                    if current.endswith(suffix):
                        suffixes_found.insert(0, suffix.lstrip('_'))  # Insert at front to maintain order
                        current = current[:-len(suffix)]
                        found_suffix = True
                        break
                if not found_suffix:
                    break

            # Combine suffixes in order (e.g., ['mean', 'std'] -> 'mean_std')
            combined_suffix = '_'.join(suffixes_found) if suffixes_found else ''
            return current, combined_suffix

        # Parse all features
        feature_df = feature_df.copy()
        parsed = feature_df['feature'].apply(lambda f: extract_base_and_suffix(f))
        feature_df['base_metric'] = parsed.apply(lambda x: x[0])
        feature_df['suffix'] = parsed.apply(lambda x: x[1])

        # Group by base metric
        grouped_features = []
        for base_metric, group in feature_df.groupby('base_metric'):
            # Check if all Cohen's d values have the same sign
            cohens_d_values = group['cohens_d'].values
            all_positive = all(d > 0 for d in cohens_d_values)
            all_negative = all(d < 0 for d in cohens_d_values)

            if not (all_positive or all_negative):
                # Mixed signs - don't group, add each feature individually
                for _, row in group.iterrows():
                    suffix = row['suffix']
                    display = f"{base_metric} ({suffix})" if suffix else base_metric
                    grouped_features.append({
                        'base_metric': base_metric,
                        'display_name': display,
                        'features': [row['feature']],
                        'suffixes': [suffix] if suffix else [],
                        'cohens_d': row['cohens_d'],
                        'abs_cohens_d': row['abs_cohens_d'],
                    })
            else:
                # Same sign - group together
                suffixes = [s for s in group['suffix'].tolist() if s]
                features = group['feature'].tolist()
                mean_cohens_d = group['cohens_d'].mean()
                max_abs_d = group['abs_cohens_d'].max()

                # Format display name
                if suffixes:
                    suffix_str = ', '.join(sorted(set(suffixes)))
                    display_name = f"{base_metric} ({suffix_str})"
                else:
                    display_name = base_metric

                grouped_features.append({
                    'base_metric': base_metric,
                    'display_name': display_name,
                    'features': features,
                    'suffixes': suffixes,
                    'cohens_d': mean_cohens_d,
                    'abs_cohens_d': max_abs_d,
                })

        # Sort by absolute Cohen's d and take top N
        grouped_features.sort(key=lambda x: x['abs_cohens_d'], reverse=True)
        return grouped_features[:top_n_groups]

    def _generate_radar_plots(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        gene_col: str,
        clusters: Dict[str, Dict],
        result: StageResult,
        top_n_features: int = 10,
    ) -> None:
        """
        Generate radar plots comparing each gene group vs NTCs for top 3 organelles.

        Both gene group and NTCs are z-scored relative to all cells (mean=0, std=1).
        This shows:
        - How each gene group differs from the population baseline (all cells)
        - How NTCs differ from baseline (should be near 0)
        - Direct visual comparison between gene group and NTCs

        Generates separate radar plots for top 3 organelles per cluster,
        organized in organelle-specific subdirectories.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix (items x features), already z-scored or will be z-scored
        df : pd.DataFrame
            Full dataframe with gene annotations
        gene_col : str
            Column name containing gene identifiers
        clusters : dict
            Positive control clusters (gene groups to analyze)
        result : StageResult
            Results container
        top_n_features : int
            Number of top distinguishing features to show on radar (default: 10)
        """
        logger.info(f"Generating radar plots (top {top_n_features} features per organelle, top 3 organelles per cluster)...")

        # Identify NTCs using multiple methods (same as validate_feature_anndata.py)
        ntc_mask = self._identify_ntc_mask(df, gene_col)
        n_ntc = ntc_mask.sum()

        if n_ntc == 0:
            logger.warning("No NTC rows found in data - radar plots will show gene group only")
        else:
            logger.info(f"NTC mask: {n_ntc} rows identified as NTCs for radar comparison")

        # Z-score features relative to all cells (population baseline)
        # Mean = 0, Std = 1 for each feature across all cells
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = pd.DataFrame(
            scaler.fit_transform(features),
            index=features.index,
            columns=features.columns
        )

        from tqdm import tqdm

        for cluster_name, cluster_data in tqdm(clusters.items(), desc="Generating radar plots"):
            genes = cluster_data["genes"]
            cluster_mask = df[gene_col].isin(genes)
            n_cluster = cluster_mask.sum()

            if n_cluster < 1:
                continue

            # Load feature importance to get top features
            safe_cluster_name = self._sanitize_filename(cluster_name)
            cluster_dir = self.output_dir / safe_cluster_name
            feature_csv_path = cluster_dir / "feature_importance.csv"

            if not feature_csv_path.exists():
                logger.warning(f"No feature importance file for {cluster_name}, skipping radar")
                continue

            feature_df = pd.read_csv(feature_csv_path)

            # Load organelle importance to get top 3 organelles
            organelle_importance_path = cluster_dir / "organelle_importance.csv"
            if organelle_importance_path.exists():
                organelle_importance_df = pd.read_csv(organelle_importance_path)
                # Get top 3 organelles by max_abs_d (already sorted descending)
                top_organelles = organelle_importance_df.head(3)[['organelle', 'max_feature', 'max_abs_d']].to_dict('records')
                logger.debug(f"  {cluster_name}: Generating radar for top 3 organelles: {[o['organelle'] for o in top_organelles]}")
            else:
                # Fallback: use all features (old behavior)
                logger.warning(f"  {cluster_name}: No organelle_importance.csv, using top features directly")
                top_organelles = [{'organelle': 'all_features', 'max_feature': None, 'max_abs_d': None}]

            # Generate radar plot for each of the top 3 organelles
            for org_rank, org_info in enumerate(top_organelles, 1):
                org_name = org_info.get('organelle', 'all_features')
                org_safe_name = self._sanitize_filename(org_name)

                # Create organelle subdirectory under radar/
                radar_dir = cluster_dir / "radar"
                organelle_dir = radar_dir / org_safe_name
                organelle_dir.mkdir(parents=True, exist_ok=True)

                # Get top features for THIS organelle with grouping by base metric
                if org_name and org_name != 'all_features':
                    # Filter to features from this organelle
                    org_features = feature_df[
                        feature_df['feature'].str.lower().str.startswith(org_name.lower() + '_')
                    ]

                    if org_features.empty:
                        # Fallback: try without underscore requirement
                        org_features = feature_df[
                            feature_df['feature'].str.lower().str.contains(org_name.lower())
                        ]

                    if org_features.empty:
                        logger.warning(f"    {cluster_name}/{org_safe_name}: No features found for organelle, skipping radar")
                        continue
                else:
                    # Use all features
                    org_features = feature_df

                # Group features by base metric (with same-direction Cohen's d)
                grouped_features = self._group_features_by_base_metric(
                    org_features, top_n_groups=top_n_features
                )

                if len(grouped_features) < 3:
                    logger.warning(f"    {cluster_name}/{org_safe_name}: Only {len(grouped_features)} feature groups, need at least 3 for radar")
                    continue

                # Filter to groups where all features exist in scaled data
                valid_groups = []
                for group in grouped_features:
                    valid_features = [f for f in group['features'] if f in features_scaled.columns]
                    if valid_features:
                        group['features'] = valid_features
                        valid_groups.append(group)

                if len(valid_groups) < 3:
                    logger.warning(f"    {cluster_name}/{org_safe_name}: Only {len(valid_groups)} valid feature groups in data, need at least 3 for radar")
                    continue

                # Calculate averaged z-scores for each feature group
                display_names = []
                gene_group_means = []
                ntc_means_list = []

                for group in valid_groups:
                    display_names.append(group['display_name'])
                    # Average z-scores across all features in the group
                    group_features = group['features']
                    gene_mean = features_scaled.loc[cluster_mask, group_features].mean().mean()
                    gene_group_means.append(gene_mean)

                    if ntc_mask.sum() > 0:
                        ntc_mean = features_scaled.loc[ntc_mask, group_features].mean().mean()
                        ntc_means_list.append(ntc_mean)

                gene_group_means = np.array(gene_group_means)

                if ntc_mask.sum() > 0:
                    ntc_means = np.array(ntc_means_list)
                    logger.debug(f"    {cluster_name}/{org_safe_name}: NTC means (grouped), range: [{ntc_means.min():.2f}, {ntc_means.max():.2f}]")
                else:
                    ntc_means = None
                    logger.debug(f"    {cluster_name}/{org_safe_name}: No NTC data for radar")

                # Log feature grouping info
                n_individual = sum(1 for g in valid_groups if len(g['features']) == 1)
                n_grouped = len(valid_groups) - n_individual
                logger.debug(f"    {cluster_name}/{org_safe_name}: {n_grouped} grouped + {n_individual} individual = {len(valid_groups)} radar axes")

                # Generate radar plot with grouped feature names
                self._plot_radar(
                    feature_names=display_names,
                    gene_group_values=gene_group_means,
                    ntc_values=ntc_means,
                    cluster_name=f"{cluster_name} ({org_name})",
                    n_items=n_cluster,
                    output_path=organelle_dir / "radar_vs_ntc.png",
                    result=result,
                )

    def _plot_radar(
        self,
        feature_names: List[str],
        gene_group_values: np.ndarray,
        ntc_values: Optional[np.ndarray],
        cluster_name: str,
        n_items: int,
        output_path: Path,
        result: StageResult,
    ) -> None:
        """
        Create a radar plot comparing gene group phenotype to NTCs.

        Parameters
        ----------
        feature_names : list
            Names of features (radar axes)
        gene_group_values : np.ndarray
            Mean z-scores for gene group
        ntc_values : np.ndarray or None
            Mean z-scores for NTCs (None if no NTCs in data)
        cluster_name : str
            Name of the gene group
        n_items : int
            Number of items in gene group
        output_path : Path
            Where to save the figure
        result : StageResult
            Results container
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        n_features = len(feature_names)

        # Compute angles for each feature (evenly spaced around circle)
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()

        # Close the polygon by repeating the first value
        gene_group_values = np.concatenate([gene_group_values, [gene_group_values[0]]])
        angles_closed = angles + [angles[0]]

        if ntc_values is not None:
            ntc_values = np.concatenate([ntc_values, [ntc_values[0]]])

        # Create larger figure to accommodate full feature names
        fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))

        # Plot gene group
        ax.plot(angles_closed, gene_group_values, 'o-', linewidth=2.5,
                label=f'{cluster_name} (n={n_items})', color='#d62728', markersize=8)
        ax.fill(angles_closed, gene_group_values, alpha=0.25, color='#d62728')

        # Plot NTCs if available
        if ntc_values is not None:
            ax.plot(angles_closed, ntc_values, 's--', linewidth=2,
                    label='NTCs (baseline)', color='#1f77b4', markersize=6, alpha=0.8)
            ax.fill(angles_closed, ntc_values, alpha=0.15, color='#1f77b4')

        # Plot the zero reference circle
        zero_circle = np.zeros(len(angles_closed))
        ax.plot(angles_closed, zero_circle, '--', color='gray', linewidth=1, alpha=0.5, label='All cells (μ=0)')

        # Set feature labels - FULL names (no truncation)
        # Adjust font size based on longest label length
        max_label_len = max(len(name) for name in feature_names)
        label_fontsize = 8 if max_label_len < 40 else 7 if max_label_len < 55 else 6
        ax.set_xticks(angles)
        ax.set_xticklabels(feature_names, fontsize=label_fontsize, wrap=True)

        # Adjust label padding to prevent overlap with plot
        ax.tick_params(axis='x', pad=18)

        # Set y-axis limits symmetrically around 0
        max_abs = max(
            np.max(np.abs(gene_group_values)),
            np.max(np.abs(ntc_values)) if ntc_values is not None else 0
        )
        y_limit = max(2.0, max_abs * 1.2)  # At least +/- 2 std, or 20% beyond max
        ax.set_ylim(-y_limit, y_limit)

        # Add gridlines at key z-score values
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_yticklabels(['-2σ', '-1σ', '0', '+1σ', '+2σ'], fontsize=8, color='gray')

        # Title and legend
        ax.set_title(
            f"Phenotypic Profile: {cluster_name}\n"
            f"(Top {len(feature_names)} distinguishing feature groups, z-scored vs all cells)",
            fontsize=12, fontweight='bold', pad=20
        )

        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

        plt.tight_layout()

        # Save
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        result.add_file(output_path)

        logger.debug(f"Saved radar plot: {output_path.name}")
    
