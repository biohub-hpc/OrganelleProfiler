"""
Cell-level UMAP analysis.

Generates UMAP embeddings, clustering, and visualizations for cell-level
feature data. This is the primary analysis that processes individual cells.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import fdrcorrection
from joblib import Parallel, delayed
from tqdm import tqdm
import logging

from .fe_graphs_base import BaseAnalyzer, AnalysisResult
from ..core.fe_graphs_embedding import EmbeddingEngine
from ..core.fe_graphs_clustering import ClusteringEngine
from ..core.fe_graphs_cache import EmbeddingCache
from ..plotting.fe_graphs_utils import save_figure, create_cluster_palette
from ..plotting.fe_graphs_umap_plots import (
    plot_umap_cluster,
    plot_umap_gene_highlight,
    plot_umap_continuous,
    plot_umap_ntc_vs_perturbed,
)

logger = logging.getLogger(__name__)


class CellLevelAnalyzer(BaseAnalyzer):
    """
    Cell-level UMAP and clustering analysis.
    
    Generates:
    - UMAP embeddings for all features
    - UMAP embeddings per organelle
    - Clustering with HDBSCAN/KMeans/Leiden
    - Gene enrichment analysis
    - Various visualization plots
    
    Parameters
    ----------
    data_context : DataContext
        Container with loaded data and paths.
    config : GraphConfig
        Main configuration settings.
    """
    
    @property
    def analysis_name(self) -> str:
        return "cell_all_features"
    
    def run(self) -> AnalysisResult:
        """Execute cell-level analysis."""
        self.log_start("Generating Cell-Level UMAPs")
        result = AnalysisResult()
        
        # Prepare features
        features, cell_df = self.prepare_features(self.cell_df)
        
        if features.empty:
            result.add_error("No numeric features found for UMAP generation")
            return result
        
        # Save feature summary
        self._save_feature_summary(features, result)
        
        # Select random highlight genes
        highlight_genes = self._select_highlight_genes(cell_df)
        
        # Generate main UMAP suite
        embedding, clusters_dict, plot_df, enrichment_df = self._generate_umap_analysis(
            features, cell_df, self.analysis_name
        )
        
        if embedding is None:
            result.add_error("Failed to generate UMAP embedding")
            return result
        
        # Store results
        result.data["embedding"] = embedding
        result.data["clusters"] = clusters_dict
        result.data["enrichment"] = enrichment_df
        result.data["plot_df"] = plot_df
        result.data["highlight_genes"] = highlight_genes
        
        # Generate plots
        self._generate_plots(plot_df, features, enrichment_df, highlight_genes, result)
        
        # Generate per-organelle UMAPs
        self._generate_organelle_umaps(features, cell_df, result)
        
        # Save embeddings to AnnData
        self._save_to_anndata(embedding, clusters_dict)
        
        self.log_complete()
        return result
    
    def _save_feature_summary(self, features: pd.DataFrame, result: AnalysisResult) -> None:
        """Save summary statistics of input features."""
        logger.info("Calculating feature summary...")
        
        summary = features.describe().transpose()
        summary = summary[["min", "max", "mean", "50%", "std"]].copy()
        summary.rename(columns={"50%": "median"}, inplace=True)
        
        save_path = self.output_dir / "umap_input_feature_summary.csv"
        summary.to_csv(save_path)
        result.add_file(save_path)
        logger.info(f"Saved feature summary to: {save_path}")
    
    def _select_highlight_genes(self, cell_df: pd.DataFrame, n_genes: int = 5) -> List[str]:
        """Select random perturbation genes for highlight plots."""
        if "gene_name" not in cell_df.columns:
            return []
        
        # Exclude NTC from selection
        ntc_mask = self.get_ntc_mask(cell_df)
        pert_genes = cell_df.loc[~ntc_mask, "gene_name"].unique()
        
        if len(pert_genes) == 0:
            return []
        
        n_to_sample = min(n_genes, len(pert_genes))
        highlight_genes = np.random.choice(pert_genes, n_to_sample, replace=False).tolist()
        logger.info(f"Selected {n_to_sample} random genes to highlight: {highlight_genes}")
        
        return highlight_genes
    
    def _generate_umap_analysis(
        self,
        features: pd.DataFrame,
        cell_df: pd.DataFrame,
        cache_key: str,
    ) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
        """
        Generate UMAP embedding and clustering.
        
        Returns
        -------
        tuple
            (embedding, clusters_dict, plot_df, enrichment_df)
        """
        # Initialize engines
        embedding_engine = EmbeddingEngine(use_cuml=self.config.use_cuml)
        clustering_engine = ClusteringEngine(use_cuml=self.config.use_cuml)
        cache = EmbeddingCache(self.data.cache_dir)
        
        n_cells = len(features)
        full_cache_key = f"{self.data.experiment}_{cache_key}"
        
        # Check cache
        cached = cache.load(full_cache_key, n_cells)
        clustering_methods = self.config.get_clustering_methods()
        
        if cached is not None:
            # Use cached results
            embedding = cached['embedding']
            clusters_dict = {}
            
            for method in clustering_methods:
                cache_cluster_key = f'clusters_{method}'
                if cache_cluster_key in cached:
                    clusters_dict[method] = cached[cache_cluster_key]
                else:
                    # Run missing clustering
                    clusters_dict[method] = clustering_engine.cluster(embedding, method)
            
            # Update cache with any new clusters
            cache.save(full_cache_key, embedding, clusters_dict, n_cells)
        else:
            # Compute everything
            logger.info(f"Computing UMAP embedding for {n_cells:,} cells...")
            
            embedding = embedding_engine.compute_umap(
                features.values,
                n_neighbors=self.analysis_config.umap_n_neighbors,
                min_dist=self.analysis_config.umap_min_dist,
                scale=True,
            )
            
            # Run clustering
            clusters_dict = clustering_engine.cluster_all(embedding, clustering_methods)
            
            # Save to cache
            cache.save(full_cache_key, embedding, clusters_dict, n_cells)
        
        # Build plot DataFrame
        plot_df = cell_df.copy()
        plot_df["umap_1"] = embedding[:, 0]
        plot_df["umap_2"] = embedding[:, 1]
        
        for method, clusters in clusters_dict.items():
            col_name = f"cluster_{method}"
            plot_df[col_name] = "c" + clusters.astype(str)
        
        # Use first method as primary cluster column
        primary_method = clustering_methods[0]
        plot_df["cluster"] = plot_df[f"cluster_{primary_method}"]
        
        # Gene enrichment analysis
        enrichment_df = self._calculate_enrichment(plot_df)
        
        return embedding, clusters_dict, plot_df, enrichment_df
    
    def _calculate_enrichment(self, plot_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate gene enrichment for each cluster."""
        if "gene_name" not in plot_df.columns or "cluster" not in plot_df.columns:
            return pd.DataFrame()
        
        logger.info("Performing gene enrichment analysis for clusters...")
        
        # Filter to clusters large enough for meaningful stats
        min_size = self.analysis_config.min_enrichment_cluster_size
        cluster_counts = plot_df["cluster"].value_counts()
        clusters_to_analyze = cluster_counts[cluster_counts >= min_size].index.tolist()
        
        # Remove noise cluster
        if "c-1" in clusters_to_analyze:
            clusters_to_analyze.remove("c-1")
        
        logger.info(
            f"Found {len(cluster_counts)} total clusters. "
            f"Analyzing {len(clusters_to_analyze)} clusters with >={min_size} cells."
        )
        
        if not clusters_to_analyze:
            return pd.DataFrame()
        
        # Run enrichment in parallel
        from ops_utils.hpc.resource_manager import get_optimal_workers
        num_workers = get_optimal_workers(use_gpu=False)
        
        cluster_enrichment_dfs = Parallel(n_jobs=num_workers)(
            delayed(self._enrichment_for_cluster)(cluster_id, plot_df)
            for cluster_id in tqdm(clusters_to_analyze, desc="Calculating gene enrichment")
        )
        
        # Combine results
        valid_results = [df for df in cluster_enrichment_dfs if df is not None]
        if valid_results:
            return pd.concat(valid_results, ignore_index=True)
        
        return pd.DataFrame()
    
    def _enrichment_for_cluster(
        self, cluster_id: str, plot_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Calculate enrichment for a single cluster (parallelizable)."""
        in_cluster = plot_df["cluster"] == cluster_id
        genes_in = plot_df[in_cluster]["gene_name"]
        genes_out = plot_df[~in_cluster]["gene_name"]
        
        in_counts = genes_in.value_counts()
        out_counts = genes_out.value_counts()
        
        results = []
        min_count = self.analysis_config.min_gene_count_for_enrichment
        
        for gene, a in in_counts.items():
            if a < min_count:
                continue
            
            b = out_counts.get(gene, 0)
            c = len(genes_in) - a
            d = len(genes_out) - b
            
            odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
            results.append({
                "gene_name": gene,
                "p_value": p_value,
                "odds_ratio": odds_ratio,
            })
        
        if results:
            df = pd.DataFrame(results)
            df["p_adj"] = fdrcorrection(df["p_value"])[1]
            df["cluster"] = cluster_id
            return df.sort_values("p_adj")
        
        return None
    
    def _generate_plots(
        self,
        plot_df: pd.DataFrame,
        features: pd.DataFrame,
        enrichment_df: pd.DataFrame,
        highlight_genes: List[str],
        result: AnalysisResult,
    ) -> None:
        """Generate all visualization plots."""
        logger.info("Generating static UMAP plots...")
        
        # Add gene group column
        plot_df = self.add_gene_group_column(plot_df)
        
        # Create cluster palette
        palette = create_cluster_palette(plot_df["cluster"])
        
        # 1. Cluster plot (annotated)
        annotations = self._create_cluster_annotations(enrichment_df)
        fig = plot_umap_cluster(
            plot_df, 
            annotations=annotations,
            title=f"UMAP by Cluster ({self.analysis_name})",
            palette=palette,
        )
        path = save_figure(fig, self.output_dir / f"umap_{self.analysis_name}_cluster_annotated.png")
        result.add_file(path)
        
        # 2. Cluster plot (no annotations)
        fig = plot_umap_cluster(
            plot_df,
            title=f"UMAP by Cluster ({self.analysis_name})",
            palette=palette,
        )
        path = save_figure(fig, self.output_dir / f"umap_{self.analysis_name}_cluster.png")
        result.add_file(path)
        
        # 3. NTC vs Perturbed
        ntc_mask = self.get_ntc_mask(plot_df)
        fig = plot_umap_ntc_vs_perturbed(
            plot_df, ntc_mask,
            title=f"UMAP: NTC vs Perturbed ({self.analysis_name})",
        )
        path = save_figure(fig, self.output_dir / f"umap_{self.analysis_name}_perturbation_status.png")
        result.add_file(path)
        
        # 4. Gene effect (if available)
        if "gene_effect" in plot_df.columns:
            fig = plot_umap_continuous(
                plot_df, "gene_effect",
                cmap="RdBu_r",
                title=f"UMAP by Gene Effect ({self.analysis_name})",
                cbar_label="Gene Effect (CERES)",
            )
            path = save_figure(fig, self.output_dir / f"umap_{self.analysis_name}_gene_effect.png")
            result.add_file(path)
        
        # 5. Well colored
        if "well" in plot_df.columns:
            self._plot_well_umap(plot_df, result)
        
        # 6. Radial position plots
        self._plot_radial_umaps(plot_df, result)
        
        # 7. Gene highlight plots
        self._generate_gene_highlights(plot_df, enrichment_df, highlight_genes, result)
    
    def _create_cluster_annotations(
        self, enrichment_df: pd.DataFrame, top_n: int = 5
    ) -> Dict[str, str]:
        """Create cluster annotation text from enrichment results."""
        if enrichment_df.empty:
            return {}
        
        annotations = {}
        top_genes = enrichment_df.groupby("cluster").head(top_n)
        
        for cluster_id, group in top_genes.groupby("cluster"):
            sorted_group = group.sort_values("odds_ratio", ascending=False)
            gene_list = [
                f"{row.gene_name} ({row.odds_ratio:.1f}x)"
                for _, row in sorted_group.iterrows()
            ]
            annotations[cluster_id] = "\n".join(gene_list)
        
        return annotations
    
    def _plot_well_umap(self, plot_df: pd.DataFrame, result: AnalysisResult) -> None:
        """Generate UMAP colored by well."""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        well_values = plot_df["well"].astype(str)
        well_ids = sorted(well_values.unique())
        
        if len(well_ids) <= 20:
            cmap = plt.get_cmap("tab20", len(well_ids))
        else:
            cmap = plt.get_cmap("turbo", len(well_ids))
        
        well_colors = {well: cmap(i) for i, well in enumerate(well_ids)}
        colors = well_values.map(well_colors)
        
        ax.scatter(
            plot_df["umap_1"], plot_df["umap_2"],
            c=colors, s=8, alpha=0.7, rasterized=True,
        )
        
        ax.set_title(f"UMAP by Well ({self.analysis_name})")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        
        path = save_figure(fig, self.output_dir / f"umap_{self.analysis_name}_well.png")
        result.add_file(path)
    
    def _plot_radial_umaps(self, plot_df: pd.DataFrame, result: AnalysisResult) -> None:
        """Generate UMAPs colored by radial position."""
        plot_df = self.calculate_radial_positions(plot_df)
        
        for pos_type, cmap in [("well_radial_pos", "viridis"), ("tile_radial_pos", "magma")]:
            if pos_type not in plot_df.columns or plot_df[pos_type].isna().all():
                continue
            
            fig = plot_umap_continuous(
                plot_df, pos_type,
                cmap=cmap,
                title=f'UMAP by {pos_type.replace("_", " ").title()} ({self.analysis_name})',
                cbar_label=f'Radial Distance from {pos_type.split("_")[0].capitalize()} Center',
            )
            path = save_figure(fig, self.output_dir / f"umap_{self.analysis_name}_{pos_type}.png")
            result.add_file(path)
    
    def _generate_gene_highlights(
        self,
        plot_df: pd.DataFrame,
        enrichment_df: pd.DataFrame,
        highlight_genes: List[str],
        result: AnalysisResult,
    ) -> None:
        """Generate gene highlight plots."""
        if not highlight_genes:
            return
        
        highlight_dir = self.output_dir / "random_gene_highlights"
        highlight_dir.mkdir(exist_ok=True)
        
        logger.info(f"Generating {len(highlight_genes)} gene highlight plots...")
        
        for gene in highlight_genes:
            fig = plot_umap_gene_highlight(
                plot_df, gene,
                enrichment_df=enrichment_df,
                title=f"UMAP Highlighting Gene: {gene}",
            )
            path = save_figure(
                fig, highlight_dir / f"umap_{self.analysis_name}_highlight_{gene}.png"
            )
            result.add_file(path)
        
        # Also generate plots for enriched genes
        if not enrichment_df.empty:
            enriched_dir = self.output_dir / "enriched_gene_highlights"
            enriched_dir.mkdir(exist_ok=True)
            
            top_enriched = (
                enrichment_df
                .loc[enrichment_df.groupby("gene_name")["odds_ratio"].idxmax()]
                .nlargest(5, "odds_ratio")
            )
            
            for _, row in top_enriched.iterrows():
                gene = row["gene_name"]
                fig = plot_umap_gene_highlight(
                    plot_df, gene,
                    enrichment_df=enrichment_df,
                    title=f"UMAP Highlighting Gene: {gene}",
                )
                path = save_figure(
                    fig, enriched_dir / f"umap_{self.analysis_name}_highlight_{gene}.png"
                )
                result.add_file(path)
    
    def _generate_organelle_umaps(
        self,
        features: pd.DataFrame,
        cell_df: pd.DataFrame,
        result: AnalysisResult,
    ) -> None:
        """Generate per-organelle UMAP embeddings."""
        logger.info("Generating per-organelle UMAPs...")
        
        organelle_features = self.group_features_by_organelle(features.columns.tolist())
        
        logger.info(f"Found {len(organelle_features)} organelle groups: {sorted(organelle_features.keys())}")
        
        for organelle, cols in sorted(organelle_features.items()):
            if len(cols) < 2:
                logger.info(f"Skipping {organelle}: not enough features ({len(cols)})")
                continue
            
            # Create organelle output directory
            org_output_dir = self.data.graph_output_path / f"organelle_{organelle}"
            org_output_dir.mkdir(exist_ok=True)
            
            # Filter features
            org_features = features[cols].copy()
            
            # Remove low variance features using percentile-based threshold
            variances = org_features.var()
            if len(variances) > 0:
                threshold = max(np.percentile(variances, 5.0), 1e-8)
                org_features = org_features.loc[:, variances > threshold]
            
            if org_features.shape[1] < 2:
                logger.info(f"Skipping {organelle}: not enough features after filtering")
                continue
            
            # Remove duplicates
            org_features.drop_duplicates(inplace=True)
            org_cell_df = cell_df.loc[org_features.index].copy()
            
            # Generate embedding
            cache_key = f"original_organelle_{organelle}"
            embedding, clusters_dict, plot_df, enrichment_df = self._generate_umap_analysis(
                org_features, org_cell_df, cache_key
            )
            
            if embedding is None:
                continue
            
            # Generate basic plots
            plot_df = self.add_gene_group_column(plot_df)
            palette = create_cluster_palette(plot_df["cluster"])
            
            fig = plot_umap_cluster(
                plot_df,
                title=f"UMAP: {organelle.upper()} Features",
                palette=palette,
            )
            path = save_figure(fig, org_output_dir / f"umap_{cache_key}_cluster.png")
            result.add_file(path)
    
    def _save_to_anndata(
        self,
        embedding: np.ndarray,
        clusters_dict: Dict[str, np.ndarray],
    ) -> None:
        """Save embeddings and clusters back to AnnData."""
        if "cell" not in self.data.adata:
            return
        
        adata = self.data.adata["cell"]
        adata.obsm["X_umap"] = embedding.astype(np.float32)
        
        for method, clusters in clusters_dict.items():
            adata.obs[f"cluster_{method}"] = clusters
        
        # Save to disk
        h5ad_path = self.data.analysis_path / f"{self.data.experiment}_cell_features.h5ad"
        logger.info(f"Saving embeddings to {h5ad_path.name}...")
        adata.write_h5ad(h5ad_path)
        logger.info("Successfully saved UMAP embedding and clusters to AnnData.")
