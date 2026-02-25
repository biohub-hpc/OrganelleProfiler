"""
Gene Relationships Stage: PHATE-based clustering and hierarchical analysis.

Mirrors the workflow:
1. PHATE low-dimensional embedding of gene-level features
2. Leiden clustering to group phenotypes by similarity
3. Hierarchical sub-clustering for detailed gene relationships
4. Feature enrichment heatmap showing distinguishing features per cluster
5. Gene lists for each cluster

Runs at gene level only, for:
- All features combined
- Each organelle group individually
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import ttest_ind
from scipy.spatial.distance import pdist, squareform
import logging

from .fe_graphs_stage_base import BaseStage, StageResult
from ..plotting.fe_graphs_utils import save_figure
from ..fe_graphs_config import NTC_PATTERNS

logger = logging.getLogger(__name__)


class GeneRelationshipsStage(BaseStage):
    """
    Discover gene relationships through PHATE embedding and hierarchical clustering.
    
    This stage performs unsupervised discovery of gene phenotype clusters and their
    distinguishing morphological features.
    """
    
    STAGE_NUMBER = 9  # After Organelle Discrimination
    STAGE_NAME = "gene_relationships"
    
    # PHATE parameters for cluster discrimination
    # Adjust these for different separation characteristics:
    # - knn: Lower (2) = tightest clusters, Higher (10) = smoother transitions
    # - decay: Lower (10-15) = sharpest boundaries, Higher (40-60) = gradual transitions
    # - gamma: Lower (0.2) = local structure, Higher (1-2) = global structure
    PHATE_KNN = 2  # Minimum value - VERY tight, discrete clusters
    PHATE_DECAY = 12  # Very sharp boundaries
    PHATE_GAMMA = 0.2  # Strong local emphasis
    PHATE_T = 'auto'  # Diffusion time (auto-selection usually optimal)
    
    # Leiden clustering resolution
    # Higher = more clusters (smaller groups)
    # Default is 1.0, increase to 1.5-3.0 for finer granularity
    LEIDEN_RESOLUTION = 2.0  # Increased from 1.0 for more, smaller clusters
    
    def run(self) -> StageResult:
        self.log_start("Discovering Gene Relationships via PHATE Clustering")
        result = StageResult()
        
        # Only run at gene level
        if self.level != "gene":
            logger.info(f"Skipping gene relationships at {self.level} level (gene level only)")
            result.data["skipped"] = True
            return result
        
        df = self.df.copy()
        all_features = self.get_features(df)
        
        if all_features.empty or len(df) < 10:
            result.add_error("Insufficient data for gene relationships analysis")
            return result
        
        # Remove NTCs
        ntc_mask = self.get_ntc_mask(df)
        df = df.loc[~ntc_mask].copy()
        all_features = all_features.loc[~ntc_mask].copy()
        
        logger.info(f"Analyzing {len(df)} genes (excluding NTCs)")
        
        # 1. All features analysis
        all_features_dir = self.output_dir / "all_features"
        all_features_dir.mkdir(parents=True, exist_ok=True)
        self._run_phate_clustering_analysis(
            all_features, df, all_features_dir, result, "All Features"
        )
        
        # 2. Per-organelle analysis
        organelle_features = self.group_features_by_organelle(all_features.columns.tolist())
        logger.info(f"Running per-organelle analysis for {len(organelle_features)} organelle groups")
        
        for org_name, org_cols in sorted(organelle_features.items()):
            if len(org_cols) < 10:
                logger.debug(f"Skipping {org_name}: only {len(org_cols)} features (need >= 10)")
                continue
            
            org_features = all_features[org_cols].copy()
            
            # Check variance
            if org_features.var().sum() == 0:
                logger.debug(f"Skipping {org_name}: zero variance")
                continue
            
            org_dir = self.output_dir / f"organelle_{org_name}"
            org_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"  {org_name}: {len(org_cols)} features...")
            self._run_phate_clustering_analysis(
                org_features, df, org_dir, result, org_name
            )
        
        self.log_complete(result)
        return result
    
    def _run_phate_clustering_analysis(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        output_dir: Path,
        result: StageResult,
        name: str,
    ) -> None:
        """
        Run complete PHATE clustering workflow for a feature set.
        
        Steps:
        1. PHATE embedding
        2. Leiden clustering
        3. Hierarchical sub-clustering
        4. Feature enrichment heatmap
        5. Gene list export
        """
        logger.info(f"  Running PHATE clustering for {name}...")
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = pd.DataFrame(
            scaler.fit_transform(features),
            index=features.index,
            columns=features.columns
        )
        
        # 1. PHATE embedding
        phate_coords = self._compute_phate_embedding(features_scaled, output_dir, result, name)
        
        if phate_coords is None:
            logger.warning(f"  PHATE failed for {name}, skipping")
            return
        
        # 2. Leiden clustering on PHATE embedding
        cluster_labels = self._leiden_clustering(phate_coords, output_dir, result, name)
        
        if cluster_labels is None:
            logger.warning(f"  Clustering failed for {name}, skipping")
            return
        
        # Add cluster labels to df
        df_clustered = df.copy()
        df_clustered["phate_cluster"] = cluster_labels
        
        n_clusters = len(np.unique(cluster_labels))
        logger.info(f"  Found {n_clusters} clusters via Leiden")
        
        # 3. Hierarchical sub-clustering within each cluster
        hierarchical_labels = self._hierarchical_subclustering(
            features_scaled, cluster_labels, output_dir, result, name
        )
        df_clustered["hierarchical_subcluster"] = hierarchical_labels
        
        # 4. Feature enrichment heatmap
        self._create_enrichment_heatmap(
            features_scaled, df_clustered, cluster_labels, output_dir, result, name
        )
        
        # 5. Export gene lists
        self._export_gene_lists(df_clustered, cluster_labels, output_dir, result, name)
        
        # 6. Create comprehensive multi-panel summary figure
        self._create_comprehensive_summary(
            phate_coords, cluster_labels, features_scaled, df_clustered, 
            output_dir, result, name
        )
    
    def _compute_phate_embedding(
        self,
        features: pd.DataFrame,
        output_dir: Path,
        result: StageResult,
        name: str,
    ) -> Optional[np.ndarray]:
        """
        Compute PHATE embedding.
        
        PHATE (Potential of Heat-diffusion for Affinity-based Trajectory Embedding)
        is better than UMAP for preserving continuous transitions and trajectories.
        """
        try:
            import phate
        except ImportError:
            logger.error("PHATE library not installed. Install with: pip install phate")
            return None
        
        try:
            # PHATE parameters - tuned for cluster discrimination
            # Lower knn and decay = sharper cluster boundaries
            # Lower gamma = emphasize local structure over global
            phate_op = phate.PHATE(
                n_components=2,
                knn=self.PHATE_KNN,  # Nearest neighbors (3 = tight clusters, 10 = smooth)
                decay=self.PHATE_DECAY,  # Kernel decay (20 = sharp, 60 = gradual)
                t=self.PHATE_T,  # Diffusion time (auto usually best)
                gamma=self.PHATE_GAMMA,  # Distance constant (0.5 = local, 2 = global)
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            
            logger.info(f"  PHATE params: knn={self.PHATE_KNN}, decay={self.PHATE_DECAY}, gamma={self.PHATE_GAMMA}")
            
            phate_coords = phate_op.fit_transform(features.values)
            
            # Plot PHATE embedding
            fig, ax = plt.subplots(figsize=(14, 12))
            ax.scatter(
                phate_coords[:, 0],
                phate_coords[:, 1],
                s=self.plot_config.point_size * 3,
                alpha=0.7,
                c='steelblue',
                edgecolors='black',
                linewidths=0.5,
                rasterized=True
            )
            ax.set_xlabel("PHATE 1", fontsize=20, fontweight='bold')
            ax.set_ylabel("PHATE 2", fontsize=20, fontweight='bold')
            ax.set_title(f"PHATE Embedding - {name}\n{len(features)} genes", fontsize=22, fontweight='bold', pad=20)
            ax.tick_params(labelsize=16)
            ax.grid(alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            path = save_figure(fig, output_dir / "phate_embedding.png", dpi=150)
            result.add_file(path)
            
            logger.info(f"  PHATE embedding computed: {phate_coords.shape}")
            return phate_coords
            
        except Exception as e:
            logger.error(f"PHATE computation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _leiden_clustering(
        self,
        phate_coords: np.ndarray,
        output_dir: Path,
        result: StageResult,
        name: str,
    ) -> Optional[np.ndarray]:
        """
        Perform Leiden clustering on PHATE embedding.
        
        Leiden is an improvement over Louvain clustering that guarantees
        well-connected communities.
        """
        try:
            import leidenalg
            import igraph as ig
        except ImportError:
            logger.warning("Leiden clustering requires: pip install leidenalg python-igraph")
            logger.info("Falling back to KMeans clustering")
            return self._kmeans_clustering(phate_coords, output_dir, result, name)
        
        try:
            # Build k-nearest neighbor graph
            from sklearn.neighbors import kneighbors_graph
            
            k = min(15, len(phate_coords) // 2)
            knn_graph = kneighbors_graph(
                phate_coords, n_neighbors=k, mode='distance', include_self=False
            )
            
            # Convert to igraph
            sources, targets = knn_graph.nonzero()
            weights = knn_graph.data
            
            # Invert distances to get similarities
            weights = 1.0 / (weights + 1e-10)
            
            edges = list(zip(sources.tolist(), targets.tolist()))
            g = ig.Graph(n=len(phate_coords), edges=edges, directed=False)
            g.es['weight'] = weights
            
            # Leiden clustering with adjustable resolution
            # Higher resolution = more, smaller clusters
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights='weight',
                resolution_parameter=self.LEIDEN_RESOLUTION,  # Use class parameter
                seed=42
            )
            
            cluster_labels = np.array(partition.membership)
            n_clusters = len(np.unique(cluster_labels))
            
            logger.info(f"  Leiden clustering: {n_clusters} clusters (resolution={self.LEIDEN_RESOLUTION}), modularity={partition.modularity:.3f}")
            
            # Plot clustering
            self._plot_clustering(phate_coords, cluster_labels, output_dir, result, name, "Leiden")
            
            return cluster_labels
            
        except Exception as e:
            logger.error(f"Leiden clustering failed: {e}")
            import traceback
            traceback.print_exc()
            return self._kmeans_clustering(phate_coords, output_dir, result, name)
    
    def _kmeans_clustering(
        self,
        phate_coords: np.ndarray,
        output_dir: Path,
        result: StageResult,
        name: str,
    ) -> np.ndarray:
        """Fallback clustering using KMeans."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Find optimal k using silhouette score
        # Increased range to find more clusters
        best_k = 8  # Higher default
        best_score = -1
        
        k_range = range(max(3, int(len(phate_coords) * 0.02)), min(50, len(phate_coords) // 10))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(phate_coords)
            score = silhouette_score(phate_coords, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(phate_coords)
        
        logger.info(f"  KMeans clustering: {best_k} clusters, silhouette={best_score:.3f}")
        
        # Plot clustering
        self._plot_clustering(phate_coords, cluster_labels, output_dir, result, name, "KMeans")
        
        return cluster_labels
    
    def _plot_clustering(
        self,
        phate_coords: np.ndarray,
        cluster_labels: np.ndarray,
        output_dir: Path,
        result: StageResult,
        name: str,
        method: str,
    ) -> None:
        """Plot PHATE embedding colored by cluster."""
        fig, ax = plt.subplots(figsize=(14, 12))
        
        n_clusters = len(np.unique(cluster_labels))
        colors = sns.color_palette("husl", n_clusters)
        
        for cluster_id, color in zip(np.unique(cluster_labels), colors):
            mask = cluster_labels == cluster_id
            n_genes = mask.sum()
            ax.scatter(
                phate_coords[mask, 0],
                phate_coords[mask, 1],
                c=[color],
                s=self.plot_config.point_size * 3,
                alpha=0.8,
                label=f"C{cluster_id} (n={n_genes})",
                edgecolors='black',
                linewidths=0.5,
                rasterized=True
            )
        
        ax.set_xlabel("PHATE 1", fontsize=20, fontweight='bold')
        ax.set_ylabel("PHATE 2", fontsize=20, fontweight='bold')
        ax.set_title(
            f"{method} Clustering on PHATE - {name}\n{n_clusters} clusters",
            fontsize=22, fontweight='bold', pad=20
        )
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=12, markerscale=1.5, framealpha=0.9, loc='best')
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        path = save_figure(fig, output_dir / f"phate_clustering_{method.lower()}.png", dpi=150)
        result.add_file(path)
    
    def _hierarchical_subclustering(
        self,
        features: pd.DataFrame,
        cluster_labels: np.ndarray,
        output_dir: Path,
        result: StageResult,
        name: str,
    ) -> np.ndarray:
        """
        Perform hierarchical sub-clustering within each Leiden cluster.
        
        This provides finer-grained relationships between genes.
        """
        hierarchical_labels = np.zeros_like(cluster_labels, dtype=int)
        subcluster_counter = 0
        
        hier_dir = output_dir / "hierarchical_subclusters"
        hier_dir.mkdir(exist_ok=True)
        
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            cluster_features = features.loc[mask]
            
            if len(cluster_features) < 3:
                # Too small for sub-clustering
                hierarchical_labels[mask] = subcluster_counter
                subcluster_counter += 1
                continue
            
            # Compute linkage
            try:
                # Use correlation distance for morphological features
                linkage_matrix = linkage(cluster_features.values, method='ward', metric='euclidean')
                
                # Determine number of subclusters (e.g., cut tree at 50% of max height)
                max_height = linkage_matrix[:, 2].max()
                cut_height = max_height * 0.5
                subclusters = fcluster(linkage_matrix, cut_height, criterion='distance')
                
                # Plot dendrogram
                fig, ax = plt.subplots(figsize=(16, 8))
                dendrogram(
                    linkage_matrix,
                    ax=ax,
                    color_threshold=cut_height,
                    above_threshold_color='gray',
                    leaf_font_size=8,
                )
                ax.axhline(y=cut_height, color='red', linestyle='--', linewidth=2, label=f'Cut height: {cut_height:.1f}')
                ax.set_xlabel("Gene Index", fontsize=16, fontweight='bold')
                ax.set_ylabel("Distance", fontsize=16, fontweight='bold')
                ax.set_title(
                    f"Hierarchical Sub-clustering - {name} - Cluster {cluster_id}\n"
                    f"{len(subclusters)} genes, {len(np.unique(subclusters))} subclusters",
                    fontsize=18, fontweight='bold', pad=15
                )
                ax.tick_params(labelsize=14)
                ax.legend(fontsize=14)
                
                plt.tight_layout()
                path = save_figure(fig, hier_dir / f"cluster_{cluster_id}_dendrogram.png", dpi=120)
                result.add_file(path)
                
                # Assign hierarchical labels
                hierarchical_labels[mask] = subclusters + subcluster_counter
                subcluster_counter += len(np.unique(subclusters))
                
            except Exception as e:
                logger.warning(f"Hierarchical clustering failed for cluster {cluster_id}: {e}")
                hierarchical_labels[mask] = subcluster_counter
                subcluster_counter += 1
        
        logger.info(f"  Hierarchical sub-clustering: {subcluster_counter} total subclusters")
        return hierarchical_labels
    
    def _create_enrichment_heatmap(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        cluster_labels: np.ndarray,
        output_dir: Path,
        result: StageResult,
        name: str,
    ) -> None:
        """
        Create heatmap showing enriched/depleted features for each cluster.
        
        Similar to the figure's heatmap showing which features distinguish each phenotype cluster.
        """
        logger.info(f"  Creating feature enrichment heatmap...")
        
        # For each cluster, compute mean feature values
        cluster_profiles = []
        cluster_names = []
        
        for cluster_id in sorted(np.unique(cluster_labels)):
            mask = cluster_labels == cluster_id
            n_genes = mask.sum()
            
            if n_genes < 2:
                continue
            
            # Mean feature values for this cluster
            cluster_mean = features.loc[mask].mean()
            cluster_profiles.append(cluster_mean.values)
            cluster_names.append(f"C{cluster_id} (n={n_genes})")
        
        if len(cluster_profiles) == 0:
            logger.warning("No valid clusters for heatmap")
            return
        
        # Create DataFrame
        profile_df = pd.DataFrame(
            cluster_profiles,
            index=cluster_names,
            columns=features.columns
        )
        
        # Select top distinguishing features (high variance across clusters)
        feature_variance = profile_df.var(axis=0)
        top_n = min(100, len(feature_variance))
        top_features = feature_variance.nlargest(top_n).index
        
        profile_subset = profile_df[top_features]
        
        # Z-score normalize for visualization
        from scipy.stats import zscore
        profile_subset_z = profile_subset.apply(zscore, axis=0)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(20, max(8, len(cluster_names) * 0.8)))
        
        sns.heatmap(
            profile_subset_z.T,  # Features as rows, clusters as columns
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-2, vmax=2,
            cbar_kws={"label": "Z-score", "shrink": 0.8},
            xticklabels=True,
            yticklabels=True,
            linewidths=0.5,
            linecolor='gray',
        )
        
        ax.set_xlabel("Cluster", fontsize=18, fontweight='bold')
        ax.set_ylabel("Feature", fontsize=18, fontweight='bold')
        ax.set_title(
            f"Feature Enrichment Heatmap - {name}\n"
            f"Top {top_n} distinguishing features across {len(cluster_names)} clusters",
            fontsize=20, fontweight='bold', pad=20
        )
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=8)
        
        # Adjust colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        
        plt.tight_layout()
        path = save_figure(fig, output_dir / "feature_enrichment_heatmap.png", dpi=150)
        result.add_file(path)
        
        # Also save the profile data
        profile_df.to_csv(output_dir / "cluster_feature_profiles.csv")
        result.add_file(output_dir / "cluster_feature_profiles.csv")
        
        # Save top features per cluster
        top_features_per_cluster = []
        for cluster_name in cluster_names:
            cluster_profile = profile_df.loc[cluster_name]
            top_enriched = cluster_profile.nlargest(20)
            top_depleted = cluster_profile.nsmallest(20)
            
            for feat, val in top_enriched.items():
                top_features_per_cluster.append({
                    'cluster': cluster_name,
                    'feature': feat,
                    'value': val,
                    'type': 'enriched'
                })
            for feat, val in top_depleted.items():
                top_features_per_cluster.append({
                    'cluster': cluster_name,
                    'feature': feat,
                    'value': val,
                    'type': 'depleted'
                })
        
        top_features_df = pd.DataFrame(top_features_per_cluster)
        top_features_df.to_csv(output_dir / "top_features_per_cluster.csv", index=False)
        result.add_file(output_dir / "top_features_per_cluster.csv")
    
    def _export_gene_lists(
        self,
        df: pd.DataFrame,
        cluster_labels: np.ndarray,
        output_dir: Path,
        result: StageResult,
        name: str,
    ) -> None:
        """Export gene lists for each cluster."""
        gene_col = "gene_name"
        
        # Main cluster assignments
        cluster_assignments = []
        for cluster_id in sorted(np.unique(cluster_labels)):
            mask = cluster_labels == cluster_id
            genes = df.loc[mask, gene_col].tolist()
            
            for gene in genes:
                cluster_assignments.append({
                    'gene_name': gene,
                    'leiden_cluster': cluster_id,
                    'hierarchical_subcluster': df.loc[df[gene_col] == gene, 'hierarchical_subcluster'].iloc[0]
                })
        
        assignments_df = pd.DataFrame(cluster_assignments)
        assignments_df.to_csv(output_dir / "gene_cluster_assignments.csv", index=False)
        result.add_file(output_dir / "gene_cluster_assignments.csv")
        
        # Per-cluster gene lists
        genes_dir = output_dir / "gene_lists"
        genes_dir.mkdir(exist_ok=True)
        
        for cluster_id in sorted(np.unique(cluster_labels)):
            mask = cluster_labels == cluster_id
            genes = df.loc[mask, gene_col].tolist()
            
            with open(genes_dir / f"cluster_{cluster_id}_genes.txt", 'w') as f:
                f.write(f"# Cluster {cluster_id} - {len(genes)} genes\n")
                for gene in sorted(genes):
                    f.write(f"{gene}\n")
            
            result.add_file(genes_dir / f"cluster_{cluster_id}_genes.txt")
        
        logger.info(f"  Exported gene lists for {len(np.unique(cluster_labels))} clusters")
    
    def _create_comprehensive_summary(
        self,
        phate_coords: np.ndarray,
        cluster_labels: np.ndarray,
        features: pd.DataFrame,
        df: pd.DataFrame,
        output_dir: Path,
        result: StageResult,
        name: str,
    ) -> None:
        """
        Create comprehensive multi-panel summary figure.
        
        Panel A: PHATE embedding with Leiden clustering and biological process annotations
        Panel B: Top 10 distinguishing features per cluster (heatmap)
        Panel C: Gene lists per cluster (aligned vertically with heatmap)
        """
        logger.info(f"  Creating comprehensive summary figure for {name}...")
        
        gene_col = "gene_name"
        n_clusters = len(np.unique(cluster_labels))
        
        # Infer biological processes for each cluster
        cluster_annotations = self._infer_cluster_biology(df, cluster_labels, features)
        
        # Calculate figure dimensions - much taller for gene lists
        fig_width = 28  # Wider for annotations
        fig_height = 10 + max(4, n_clusters * 0.4)  # PHATE + features + gene lists
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Use GridSpec for flexible layout
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(
            3, 1, 
            figure=fig,
            height_ratios=[2.5, 1.5, max(2, n_clusters * 0.15)],  # PHATE, Features, Gene lists
            hspace=0.35
        )
        
        # ============================================================
        # PANEL A: PHATE Embedding with Biological Process Annotations
        # ============================================================
        ax_phate = fig.add_subplot(gs[0, 0])
        
        colors = sns.color_palette("husl", n_clusters)
        
        # Plot each cluster with annotations
        for cluster_id, color in zip(sorted(np.unique(cluster_labels)), colors):
            mask = cluster_labels == cluster_id
            n_genes = mask.sum()
            
            # Plot points
            ax_phate.scatter(
                phate_coords[mask, 0],
                phate_coords[mask, 1],
                c=[color],
                s=self.plot_config.point_size * 5,  # Larger points
                alpha=0.85,
                edgecolors='black',
                linewidths=1.0,
                rasterized=True,
                zorder=2
            )
            
            # Add cluster label with biological annotation at centroid
            centroid_x = phate_coords[mask, 0].mean()
            centroid_y = phate_coords[mask, 1].mean()
            
            bio_label = cluster_annotations.get(cluster_id, "Unknown")
            label_text = f"C{cluster_id}\n{bio_label}\n(n={n_genes})"
            
            ax_phate.text(
                centroid_x, centroid_y, label_text,
                fontsize=14, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color, linewidth=2.5, alpha=0.95),
                zorder=3
            )
        
        ax_phate.set_xlabel("PHATE 1", fontsize=24, fontweight='bold')
        ax_phate.set_ylabel("PHATE 2", fontsize=24, fontweight='bold')
        ax_phate.set_title(
            f"A. PHATE Embedding with Biological Process Annotations - {name}\n"
            f"{len(phate_coords)} genes, {n_clusters} Leiden clusters",
            fontsize=26, fontweight='bold', pad=25, loc='left'
        )
        ax_phate.tick_params(labelsize=18)
        ax_phate.grid(alpha=0.3, linestyle='--', linewidth=1.5)
        
        # ============================================================
        # PANEL B: Top 10 Distinguishing Features (Heatmap)
        # ============================================================
        ax_features = fig.add_subplot(gs[1, 0])
        
        # Compute mean feature values per cluster
        cluster_profiles = []
        cluster_names = []
        
        for cluster_id in sorted(np.unique(cluster_labels)):
            mask = cluster_labels == cluster_id
            cluster_mean = features.loc[mask].mean()
            cluster_profiles.append(cluster_mean.values)
            cluster_names.append(f"C{cluster_id}")
        
        profile_df = pd.DataFrame(
            cluster_profiles,
            index=cluster_names,
            columns=features.columns
        )
        
        # Select top 10 most variable features
        feature_variance = profile_df.var(axis=0)
        top_10_features = feature_variance.nlargest(10).index
        
        # Z-score normalize
        from scipy.stats import zscore
        profile_subset_z = profile_df[top_10_features].apply(zscore, axis=0)
        
        # Create heatmap
        sns.heatmap(
            profile_subset_z.T,  # Features as rows, clusters as columns
            ax=ax_features,
            cmap="RdBu_r",
            center=0,
            vmin=-2, vmax=2,
            cbar_kws={"label": "Z-score", "shrink": 0.5},
            xticklabels=True,
            yticklabels=True,
            linewidths=1.5,
            linecolor='white',
        )
        
        ax_features.set_xlabel("Cluster", fontsize=22, fontweight='bold')
        ax_features.set_ylabel("Feature", fontsize=22, fontweight='bold')
        ax_features.set_title(
            f"B. Top 10 Distinguishing Morphological Features",
            fontsize=24, fontweight='bold', pad=20, loc='left'
        )
        
        plt.setp(ax_features.get_xticklabels(), rotation=0, ha='center', fontsize=18, fontweight='bold')
        plt.setp(ax_features.get_yticklabels(), fontsize=14)
        
        # Adjust colorbar
        cbar = ax_features.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label("Z-score", fontsize=20, fontweight='bold')
        
        # ============================================================
        # PANEL C: Gene Lists Aligned with Heatmap Columns
        # ============================================================
        ax_genes = fig.add_subplot(gs[2, 0])
        ax_genes.axis('off')
        
        # Create gene list table aligned with heatmap columns
        gene_lists_text = "C. Gene Cluster Assignments (top 8 genes shown, full lists in CSV files)\n\n"
        
        # Build table with columns aligned
        col_width = 1.0 / n_clusters
        y_start = 0.95
        
        for i, cluster_id in enumerate(sorted(np.unique(cluster_labels))):
            mask = cluster_labels == cluster_id
            genes = df.loc[mask, gene_col].tolist()
            n_genes = len(genes)
            
            # Position aligned with heatmap column
            x_pos = (i + 0.5) / n_clusters
            
            # Show top 8 genes
            genes_display = genes[:8]
            if n_genes > 8:
                genes_str = "\n".join(genes_display) + f"\n... +{n_genes-8} more"
            else:
                genes_str = "\n".join(genes_display)
            
            # Get biological annotation
            bio_label = cluster_annotations.get(cluster_id, "")
            
            # Create text block
            text_content = f"Cluster {cluster_id}\n{bio_label}\n({n_genes} genes)\n\n{genes_str}"
            
            ax_genes.text(
                x_pos, y_start, text_content,
                transform=ax_genes.transAxes,
                fontsize=12,  # Much larger
                verticalalignment='top',
                horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.2, edgecolor='black', linewidth=1.5),
                linespacing=1.5
            )
        
        # ============================================================
        # Save Figure
        # ============================================================
        plt.tight_layout()
        path = save_figure(fig, output_dir / "comprehensive_summary.png", dpi=200)  # Higher DPI
        result.add_file(path)
        
        logger.info(f"  Saved comprehensive summary figure")
    
    def _infer_cluster_biology(
        self,
        df: pd.DataFrame,
        cluster_labels: np.ndarray,
        features: pd.DataFrame,
    ) -> Dict[int, str]:
        """
        Infer biological process/function for each cluster based on gene names.
        
        Uses simple keyword matching to identify common biological themes.
        For more sophisticated analysis, integrate with GO/KEGG enrichment.
        """
        gene_col = "gene_name"
        cluster_annotations = {}
        
        # Define biological keywords
        bio_keywords = {
            'Ribosome': ['RPL', 'RPS', 'MRPL', 'MRPS'],
            'Translation': ['EIF', 'ETF', 'TUFM'],
            'Mitochondria': ['MT-', 'MTND', 'MTCO', 'MTCY', 'NDUFA', 'NDUFB', 'COX', 'ATP5'],
            'DNA Repair': ['RAD', 'BRCA', 'ATM', 'PARP', 'XRCC'],
            'Cell Cycle': ['CDK', 'CCND', 'CCNE', 'CDC', 'BUB'],
            'Transcription': ['TAF', 'TBP', 'POL', 'MED', 'GTF'],
            'Splicing': ['SNRP', 'SF3', 'PRPF', 'LSM'],
            'Proteasome': ['PSM', 'PSMA', 'PSMB', 'PSMC', 'PSMD'],
            'Chromatin': ['H2A', 'H2B', 'H3', 'H4', 'HIST'],
            'ER/Golgi': ['SEC', 'COPA', 'COPB', 'COPG', 'AP1', 'AP2'],
            'Cytoskeleton': ['TUB', 'ACT', 'MYO', 'VIM', 'KRT'],
            'Signaling': ['MAP', 'AKT', 'MTOR', 'RAS', 'RAF'],
            'Metabolism': ['ALDH', 'ACSS', 'ACLY', 'IDH', 'MDH'],
            'Lipid': ['FASN', 'ACAC', 'SCD', 'ELOV', 'FADS'],
        }
        
        for cluster_id in sorted(np.unique(cluster_labels)):
            mask = cluster_labels == cluster_id
            genes = df.loc[mask, gene_col].tolist()
            
            # Count matches for each biological category
            category_counts = {}
            for category, keywords in bio_keywords.items():
                count = sum(1 for gene in genes if any(kw in str(gene).upper() for kw in keywords))
                if count > 0:
                    category_counts[category] = count
            
            # Determine dominant category
            if category_counts:
                dominant_category = max(category_counts, key=category_counts.get)
                dominant_count = category_counts[dominant_category]
                pct = (dominant_count / len(genes)) * 100
                
                if pct >= 30:  # Strong signal
                    cluster_annotations[cluster_id] = dominant_category
                elif pct >= 15:  # Moderate signal
                    cluster_annotations[cluster_id] = f"{dominant_category}?"
                else:
                    # Show top 2 if mixed
                    sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
                    if len(sorted_cats) >= 2:
                        cluster_annotations[cluster_id] = f"{sorted_cats[0][0]}/{sorted_cats[1][0]}"
                    else:
                        cluster_annotations[cluster_id] = "Mixed"
            else:
                cluster_annotations[cluster_id] = "Uncharacterized"
        
        return cluster_annotations

