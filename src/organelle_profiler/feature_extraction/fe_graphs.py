"""
Feature Graph Generation Pipeline for OPS Phenotyping Data.

Generates UMAP embeddings, clustering, and statistical visualizations from
feature extraction AnnData files. Saves embeddings back to the AnnData files
for downstream analysis.

This pipeline reads from:
  - {experiment}_cell_features.h5ad
  - {experiment}_guide_features.h5ad
  - {experiment}_gene_features.h5ad

And generates:
  - UMAP embeddings (saved to obsm['X_umap'] in each AnnData)
  - PCA embeddings (saved to obsm['X_pca'] in each AnnData)
  - Clustering assignments
  - Statistical plots and visualizations

Usage:
------
# Generate graphs for an experiment (shorthand resolves to full name)
python -m organelle_profiler.feature_extraction.fe_graphs -e 94

# Run with full experiment name
python -m organelle_profiler.feature_extraction.fe_graphs -e ops0094_20251217

# Debug mode with 10% of cells
python -m organelle_profiler.feature_extraction.fe_graphs -e 94 --debug 0.1

# Use KMeans clustering instead of HDBSCAN
python -m organelle_profiler.feature_extraction.fe_graphs -e 94 --cluster-algo kmeans

# Force CPU usage (no cuML GPU acceleration)
python -m organelle_profiler.feature_extraction.fe_graphs -e 94 --no-cuml

# Generate interactive plots (skipped by default)
python -m organelle_profiler.feature_extraction.fe_graphs -e 94 --interactive

Output:
-------
Graphs are saved to: {experiment}/4-features/graphs/
Interactive plots: {experiment}/4-features/graphs/interactive_umaps/
Embeddings are saved back to the source AnnData files in obsm.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.getcwd())

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import logging
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
import umap
import hdbscan
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
import anndata as ad
import scanpy as sc
import leidenalg


from scipy.stats import fisher_exact, ttest_ind, spearmanr
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm


from ops_utils.profiling.decorators import notify_step, versioned_function
from ops_utils.data.experiment import OpsDataset
from ops_utils.data.cell_data_loader import CellDataLoader
from ops_utils.hpc.resource_manager import get_optimal_workers


# Set up logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GraphGenerator:
    """
    Generates a suite of plots from AnnData feature extraction files.
    """

    def __init__(
        self,
        experiment: str,
        debug_cell_fraction: float = None,
        skip_object_features: bool = False,
        use_cuml: bool = True,
        cluster_algo: str = "all",
        skip_interactive_plots: bool = True,
        drift_method: str = "three_segment_regression",
        analysis_mode: str = "all",
        skip_complete: bool = False,
    ):
        print("Initializing GraphGenerator...")
        self.experiment = experiment
        self.debug_cell_fraction = debug_cell_fraction
        self.skip_object_features = skip_object_features
        self.use_cuml = use_cuml
        self.cluster_algo = cluster_algo  # "all", "hdbscan", "kmeans", or "leiden"
        self.skip_interactive_plots = skip_interactive_plots
        self.drift_method = drift_method
        self.analysis_mode = analysis_mode  # "all", "guide", "gene", or "guide_and_gene"
        self.skip_complete = skip_complete  # Skip generating plots that already exist
        self.dataset = OpsDataset(experiment)
        self.random_highlight_genes = None
        self.well_drift_summary_data = []
        self.tile_drift_summary_data = []

        # Cache directory for embeddings and clustering results
        self.cache_dir = None  # Will be set after paths are defined

        # Define paths
        self.analysis_path = self.dataset.analysis_path
        print(f"Feature data will be read from: {self.analysis_path}")
        self.graph_output_path = self.analysis_path / "graphs"
        self.graph_output_path.mkdir(parents=True, exist_ok=True)
        self.interactive_output_path = self.graph_output_path / "interactive_umaps"
        self.interactive_output_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.graph_output_path / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Graphs will be saved to: {self.graph_output_path}")
        print(f"Interactive plots will be saved to: {self.interactive_output_path}")
        print(f"Cache directory: {self.cache_dir}")

        # Store AnnData objects for later embedding updates
        self.adata = {}

        self.data = self._load_data()

    def _load_data(self) -> dict:
        """
        Loads feature data from AnnData files into a dictionary of DataFrames.

        Expects AnnData files at:
        - {experiment}_cell_features.h5ad
        - {experiment}_guide_features.h5ad
        - {experiment}_gene_features.h5ad
        """
        print("Loading data for graph generation...")
        data = {}

        # Define paths for AnnData files
        cell_h5ad_path = self.analysis_path / f"{self.experiment}_cell_features.h5ad"
        guide_h5ad_path = self.analysis_path / f"{self.experiment}_guide_features.h5ad"
        gene_h5ad_path = self.analysis_path / f"{self.experiment}_gene_features.h5ad"

        # --- Load cell-level data (required) ---
        if not cell_h5ad_path.exists():
            raise FileNotFoundError(
                f"Cell-level AnnData not found at {cell_h5ad_path}"
            )

        print(f"  - Loading cell features from AnnData: {cell_h5ad_path.name}...")
        cell_adata = ad.read_h5ad(cell_h5ad_path)

        # Enrich AnnData with any missing metadata from linked_results CSV
        cell_adata = self._enrich_metadata_from_linked_csv(cell_adata)
        self.adata["cell"] = cell_adata

        # Convert AnnData to DataFrame: combine obs (metadata) with X (features)
        cell_df = cell_adata.obs.copy()
        feature_df = pd.DataFrame(
            cell_adata.X,
            index=cell_adata.obs_names,
            columns=cell_adata.var_names
        )
        cell_df = pd.concat([cell_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

        # Ensure coordinate columns are numeric (not categorical) for downstream calculations
        numeric_coord_cols = [
            "x_global_pheno", "y_global_pheno", "x_local_pheno", "y_local_pheno",
            "tile_pheno", "x_global_bc", "y_global_bc", "segmentation_id",
        ]
        for col in numeric_coord_cols:
            if col in cell_df.columns:
                cell_df[col] = pd.to_numeric(cell_df[col], errors="coerce")

        # Handle debug mode - subsample if requested
        if self.debug_cell_fraction and 0 < self.debug_cell_fraction <= 1:
            n_sample = int(len(cell_df) * self.debug_cell_fraction)
            print(f"--- DEBUG MODE: Sampling {n_sample} cells ({self.debug_cell_fraction:.0%}) ---")
            cell_df = cell_df.sample(n=n_sample, random_state=42).reset_index(drop=True)

        print(f"  - Loaded {len(cell_df)} cells with {len(cell_adata.var_names)} features")
        print(f"  - obs columns: {list(cell_adata.obs.columns)}")

        # Standardize NTC gene name
        if "gene_name" in cell_df.columns:
            print("  - Standardizing NTC gene name '0' to 'NTC'...")
            cell_df["gene_name"] = cell_df["gene_name"].astype(str).replace({"0": "NTC"})

        data["cell"] = cell_df

        # --- Load guide-level data ---
        if not guide_h5ad_path.exists():
            raise FileNotFoundError(
                f"Guide-level AnnData not found at {guide_h5ad_path}"
            )

        print(f"  - Loading guide features from AnnData: {guide_h5ad_path.name}...")
        guide_adata = ad.read_h5ad(guide_h5ad_path)
        self.adata["guide"] = guide_adata

        guide_df = guide_adata.obs.copy()
        feature_df = pd.DataFrame(
            guide_adata.X,
            index=guide_adata.obs_names,
            columns=guide_adata.var_names
        )
        guide_df = pd.concat([guide_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
        data["guideRNA"] = guide_df
        print(f"  - Loaded {len(guide_df)} guides")

        # --- Load gene-level data ---
        if not gene_h5ad_path.exists():
            raise FileNotFoundError(
                f"Gene-level AnnData not found at {gene_h5ad_path}"
            )

        print(f"  - Loading gene features from AnnData: {gene_h5ad_path.name}...")
        gene_adata = ad.read_h5ad(gene_h5ad_path)
        self.adata["gene"] = gene_adata

        gene_df = gene_adata.obs.copy()
        feature_df = pd.DataFrame(
            gene_adata.X,
            index=gene_adata.obs_names,
            columns=gene_adata.var_names
        )
        gene_df = pd.concat([gene_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
        data["gene"] = gene_df
        print(f"  - Loaded {len(gene_df)} genes")

        # --- Object and network features (still CSV-based for now) ---
        data["object"] = {}
        data["network"] = {}

        if not self.skip_object_features:
            print("--- Loading object-level features (CSV) ---")
            for f in self.analysis_path.glob("*_features.csv"):
                if "object_features" in f.name:
                    organelle_name = f.name.replace("_object_features.csv", "")
                    print(f"  - Loading {f.name}...")
                    data["object"][organelle_name] = pd.read_csv(f)
                elif "network_features" in f.name:
                    organelle_name = f.name.replace("_network_features.csv", "")
                    print(f"  - Loading {f.name}...")
                    data["network"][organelle_name] = pd.read_csv(f)

        print("Finished loading data.")
        return data

    def _enrich_metadata_from_linked_csv(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Enrich AnnData obs with missing metadata columns from linked_results CSV.

        If the AnnData is missing columns like tile_pheno, x_local_pheno, y_local_pheno
        (needed for radial drift analysis), this method loads them from the source
        linked_results CSV and adds them to obs by matching on segmentation_id + well.

        Note: In most cases, the tile column already exists as 'tile' and just needs
        to be renamed to 'tile_pheno' for consistency with the naming convention.

        Parameters
        ----------
        adata : ad.AnnData
            The cell-level AnnData object to enrich.

        Returns
        -------
        ad.AnnData
            The enriched AnnData object (modified in place and returned).
        """
        # First, check if we just need to rename existing columns
        if "tile" in adata.obs.columns and "tile_pheno" not in adata.obs.columns:
            print("  - Renaming 'tile' column to 'tile_pheno' for consistency")
            adata.obs["tile_pheno"] = adata.obs["tile"]
        
        # Columns needed for radial drift analysis
        required_cols = ["tile_pheno", "x_local_pheno", "y_local_pheno"]
        missing_cols = [c for c in required_cols if c not in adata.obs.columns]

        # Also check if coordinate columns are all zeros (need to be populated from CSV)
        coord_cols_to_fix = []
        if "x_global_pheno" in adata.obs.columns and (adata.obs["x_global_pheno"] == 0).all():
            coord_cols_to_fix.append(("x_global_pheno", "x_pheno"))
        if "y_global_pheno" in adata.obs.columns and (adata.obs["y_global_pheno"] == 0).all():
            coord_cols_to_fix.append(("y_global_pheno", "y_pheno"))

        if not missing_cols and not coord_cols_to_fix:
            return adata

        if missing_cols:
            print(f"  - AnnData missing columns: {missing_cols}")
        if coord_cols_to_fix:
            print(f"  - AnnData has zero-valued coordinate columns that need fixing: {[c[0] for c in coord_cols_to_fix]}")
        print(f"  - Attempting to enrich from linked_results CSV...")

        # Load linked_results CSVs for all wells
        linked_dfs = []
        wells = adata.obs["well"].unique() if "well" in adata.obs.columns else []

        for well in wells:
            # well format in obs is "A/1/0", need to convert to get the CSV path
            well_short = well.rsplit("/", 1)[0] if "/" in str(well) else str(well)
            results_path = self.dataset.append_well("linked_results", well_short)
            if results_path.exists():
                df = pd.read_csv(results_path)
                df["well"] = well  # Ensure well format matches obs
                linked_dfs.append(df)

        if not linked_dfs:
            print(f"  - Warning: No linked_results CSVs found. Cannot enrich metadata.")
            return adata

        linked_df = pd.concat(linked_dfs, ignore_index=True)
        print(f"  - Loaded {len(linked_df)} rows from linked_results CSVs")

        # Match by cell_id (format: "well_segmentation_id", e.g., "A/1/0_17926899")
        # The AnnData index is cell_id, and we construct the same key from the linked CSV
        if "segmentation_id" in linked_df.columns:
            # Report NaN counts in linked CSV
            n_nan_linked = linked_df["segmentation_id"].isna().sum()
            if n_nan_linked > 0:
                print(f"  - Warning: {n_nan_linked} rows in linked_results have NaN segmentation_id (skipped)")

            # Create cell_id in linked_df to match AnnData index (format: well_segmentation_id)
            valid_mask = linked_df["segmentation_id"].notna()
            linked_df = linked_df[valid_mask].copy()
            linked_df["cell_id"] = (
                linked_df["well"].astype(str) + "_" +
                linked_df["segmentation_id"].astype(float).astype(int).astype(str)
            )

            # Debug: check cell_id matching
            adata_ids = set(adata.obs.index[:100].tolist())
            linked_ids = set(linked_df["cell_id"].head(100).tolist())
            
            # Build lookup dict for each missing column
            # Handle CSV->AnnData column name mapping (e.g., 'x_pheno' -> 'x_global_pheno')
            cols_to_add = []
            for missing_col in missing_cols:
                # Check if we have a mapping for this column
                csv_col = missing_col
                for obs_col, csv_name in coord_cols_to_fix:
                    if obs_col == missing_col:
                        csv_col = csv_name
                        break
                if csv_col in linked_df.columns:
                    cols_to_add.append((missing_col, csv_col))  # (target_col, source_col)

            # Also fix zero-valued coordinate columns by mapping from CSV column names
            for obs_col, csv_col in coord_cols_to_fix:
                if obs_col not in [c[0] for c in cols_to_add] and csv_col in linked_df.columns:
                    cols_to_add.append((obs_col, csv_col))

            if cols_to_add:
                # Use cell_id as index for lookup (matches AnnData obs index)
                lookup = {}
                for target_col, source_col in cols_to_add:
                    lookup[target_col] = linked_df.set_index("cell_id")[source_col].to_dict()

                enriched_count = 0
                for target_col, source_col in cols_to_add:
                    # Map using obs index (which is cell_id)
                    adata.obs[target_col] = adata.obs.index.map(lookup[target_col])
                    enriched_count = max(enriched_count, adata.obs[target_col].notna().sum())

                # Ensure coordinate columns are stored as numeric, not categorical
                coord_columns = ["x_global_pheno", "y_global_pheno", "x_local_pheno", "y_local_pheno",
                                 "x_pheno", "y_pheno", "tile_pheno"]
                for coord_col in coord_columns:
                    if coord_col in adata.obs.columns:
                        adata.obs[coord_col] = pd.to_numeric(adata.obs[coord_col], errors="coerce")

                print(f"  - Enriched {enriched_count}/{len(adata.obs)} cells by cell_id match")

                # Save the enriched AnnData back to disk so we don't have to recompute
                h5ad_path = self.dataset.analysis_path / f"{self.dataset.experiment}_cell_features.h5ad"
                adata.write_h5ad(h5ad_path)
                print(f"  - Saved enriched AnnData to {h5ad_path}")

        else:
            print(f"  - Warning: Cannot match cells - segmentation_id not found in linked_results CSV")

        return adata

    def save_embeddings_to_anndata(
        self,
        embedding: np.ndarray,
        embedding_name: str,
        level: str = "cell",
        cell_indices: np.ndarray = None,
        clusters: np.ndarray = None,
    ):
        """
        Save embeddings (UMAP, PCA, etc.) back to the AnnData object in obsm.

        Parameters
        ----------
        embedding : np.ndarray
            The embedding array of shape (n_cells, n_dims), e.g., UMAP coordinates.
        embedding_name : str
            Name for the embedding key in obsm, e.g., "X_umap", "X_pca".
        level : str
            Which AnnData to update: "cell", "guide", or "gene".
        cell_indices : np.ndarray, optional
            If provided, the indices of cells in the original AnnData that correspond
            to the embedding rows (used when subset was analyzed).
        clusters : np.ndarray, optional
            If provided, cluster labels to store in obs.
        """
        if level not in self.adata:
            print(f"  - Warning: No AnnData loaded for level '{level}'. Cannot save embedding.")
            return

        adata = self.adata[level]

        # If indices provided, we need to handle partial embeddings
        if cell_indices is not None:
            # Create full-size embedding array filled with NaN
            full_embedding = np.full((adata.n_obs, embedding.shape[1]), np.nan, dtype=np.float32)
            full_embedding[cell_indices] = embedding
            adata.obsm[embedding_name] = full_embedding

            if clusters is not None:
                full_clusters = np.full(adata.n_obs, -1, dtype=int)
                full_clusters[cell_indices] = clusters
                adata.obs[f"{embedding_name.replace('X_', '')}_cluster"] = full_clusters
        else:
            # Full embedding for all cells
            if embedding.shape[0] != adata.n_obs:
                print(f"  - Warning: Embedding size ({embedding.shape[0]}) doesn't match AnnData ({adata.n_obs}). Skipping save.")
                return
            adata.obsm[embedding_name] = embedding.astype(np.float32)

            if clusters is not None:
                adata.obs[f"{embedding_name.replace('X_', '')}_cluster"] = clusters

        # Save the updated AnnData back to disk
        h5ad_path = self.analysis_path / f"{self.experiment}_{level}_features.h5ad"
        print(f"  - Saving {embedding_name} embedding to {h5ad_path.name}...")
        adata.write_h5ad(h5ad_path)
        print(f"  - Successfully saved embedding '{embedding_name}' to {level}-level AnnData.")

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given cache key."""
        # Sanitize cache key for filename
        safe_key = cache_key.replace("/", "_").replace(" ", "_")
        return self.cache_dir / f"{safe_key}.npz"

    def _load_from_cache(self, cache_key: str, n_cells: int) -> dict | None:
        """
        Load cached embedding and clustering results if they exist and match current data.

        Parameters
        ----------
        cache_key : str
            Unique identifier for this embedding (e.g., "cell_all_features")
        n_cells : int
            Expected number of cells (used to validate cache)

        Returns
        -------
        dict or None
            Dictionary with 'embedding' and 'clusters_*' keys, or None if cache miss
        """
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None

        try:
            cached = np.load(cache_path, allow_pickle=True)
            cached_n_cells = cached.get('n_cells', np.array(0)).item()

            if cached_n_cells != n_cells:
                print(f"    - Cache invalidated: cell count changed ({cached_n_cells} -> {n_cells})")
                return None

            result = {'embedding': cached['embedding']}

            # Load all cluster arrays
            for key in cached.files:
                if key.startswith('clusters_'):
                    result[key] = cached[key]

            print(f"    - Loaded from cache: {cache_path.name}")
            return result

        except Exception as e:
            print(f"    - Cache load failed: {e}")
            return None

    def _save_to_cache(self, cache_key: str, embedding: np.ndarray, clusters_dict: dict, n_cells: int):
        """
        Save embedding and clustering results to cache.

        Parameters
        ----------
        cache_key : str
            Unique identifier for this embedding
        embedding : np.ndarray
            UMAP embedding array (n_cells, 2)
        clusters_dict : dict
            Dictionary mapping method name to cluster labels, e.g., {'hdbscan': array, 'kmeans': array}
        n_cells : int
            Number of cells (for validation on reload)
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
            print(f"    - Saved to cache: {cache_path.name}")
        except Exception as e:
            print(f"    - Cache save failed: {e}")

    @notify_step(
        step_message="Started graph generation",
        success_message="Finished graph generation",
    )
    @versioned_function("v1.0")
    def run(self):
        """Generate and save all plots."""
        print("--- Starting Graph Generation ---")
        print(f"--- Analysis mode: {self.analysis_mode} ---")

        # Handle guide-only or gene-only analysis modes
        if self.analysis_mode in ("guide", "gene", "guide_and_gene"):
            if self.analysis_mode in ("guide", "guide_and_gene"):
                print("\n\n--- Generating Guide-Level UMAPs ---")
                self._generate_aggregated_level_umap("guide")

            if self.analysis_mode in ("gene", "guide_and_gene"):
                print("\n\n--- Generating Gene-Level UMAPs ---")
                self._generate_aggregated_level_umap("gene")

            print("\n--- Analysis complete (guide/gene mode) ---")
            return

        # Full analysis mode
        if self.data.get("cell") is not None:
            # self.plot_feature_correlation_heatmap() # Re-commented as per edit hint

            # --- Refactored UMAP Generation ---
            # 1. Prepare the primary feature set and corresponding cell dataframe
            features, cell_df_for_umap = self._prepare_features_for_umap()

            if features is not None:
                # --- NEW: Select 5 random genes to highlight across all plots ---
                if "gene_name" in cell_df_for_umap.columns:
                    # Exclude non-targeting controls from the selection pool
                    pert_genes = cell_df_for_umap[
                        ~cell_df_for_umap["gene_name"]
                        .astype(str)
                        .str.contains("ntc|non-targeting", case=False, regex=True)
                    ]["gene_name"].unique()
                    if len(pert_genes) > 0:
                        num_to_sample = min(5, len(pert_genes))
                        self.random_highlight_genes = np.random.choice(
                            pert_genes, num_to_sample, replace=False
                        ).tolist()
                        print(
                            f"--- Selected {num_to_sample} random genes to highlight: {self.random_highlight_genes} ---"
                        )

                # 2. Generate and save the standard suite of UMAPs (cell level)
                print("\n\n--- Generating Cell-Level UMAPs ---")
                self._generate_umap_suite(features, cell_df_for_umap, "cell_all_features", self.graph_output_path, self.interactive_output_path)

                # 2b. Generate guide-level UMAPs
                print("\n\n--- Generating Guide-Level UMAPs ---")
                self._generate_aggregated_level_umap("guide")

                # 2c. Generate gene-level UMAPs
                print("\n\n--- Generating Gene-Level UMAPs ---")
                self._generate_aggregated_level_umap("gene")

                # 3. Run regression analysis and generate UMAPs on corrected features
                print("\n\n--- Generating UMAPs after Spatial Covariate Regression ---")
                features_regressed = self.run_regression_analysis(features, cell_df_for_umap)

                # --- Volcano Plots ---
                print("\n\n--- Generating Volcano Plots ---")
                # Generate plots for original data
                self.plot_volcano_plots(features, cell_df_for_umap, top_n_features=10, output_subdir="volcano")
                # Generate plots for regressed data
                if features_regressed is not None:
                    self.plot_volcano_plots(features_regressed, cell_df_for_umap, top_n_features=30, output_subdir="volcano_regressed")

                # --- NTC Comparison plots using the filtered feature set ---
                print("\n\n--- Generating NTC Comparison Plots ---")
                self.plot_ntc_comparison(features, cell_df_for_umap)
                self.plot_ntc_comparison_fold_change(features, cell_df_for_umap)

                # --- Organelle Contribution Heatmap ---
                self.plot_organelle_contribution_heatmap(features, cell_df_for_umap)

                # --- NEW: Generate Radial Drift Plot Suites ---
                print("\n\n--- Generating Radial Drift Plots ---")
                radial_drift_output_path = self.graph_output_path / "radial_drift"
                radial_drift_output_path.mkdir(exist_ok=True)

                # 1. Run for all features
                self._generate_radial_drift_suite(
                    features, cell_df_for_umap, "all_features", radial_drift_output_path
                )

                # 2. Run for organelle-specific feature subsets
                print(f"\n--- Generating separate organelle Radial Drift plots ---")
                organelles = set()
                for col in features.columns:
                    parts = col.split("_")
                    if parts[0] == "network":
                        if len(parts) > 2:
                            organelles.add(parts[1])
                    else:
                        if len(parts) > 1:
                            organelles.add(parts[0])

                print(
                    f"  - Found {len(organelles)} organelle groups to analyze for radial drift: {sorted(list(organelles))}"
                )

                for organelle in sorted(list(organelles)):
                    organelle_cols = [
                        col
                        for col in features.columns
                        if col.startswith(f"{organelle}_")
                        or col.startswith(f"network_{organelle}_")
                    ]

                    if len(organelle_cols) < 2:
                        print(
                            f"  - Skipping {organelle} for radial drift, not enough features found ({len(organelle_cols)})."
                        )
                        continue

                    organelle_features = features[organelle_cols].copy()

                    # Call the suite generator
                    self._generate_radial_drift_suite(
                        organelle_features,
                        cell_df_for_umap,  # The full cell_df is needed for metadata
                        f"organelle_{organelle}",
                        radial_drift_output_path,
                    )

                # --- NEW: Generate summary plot of inflection points ---
                self.plot_inflection_summary()

        if self.data.get("gene") is not None:
            print("Skipping gene clustermap")
            # self.plot_gene_clustermap()
        if self.data.get("guideRNA") is not None:
            print("Skipping guideRNA plots")
            # self.plot_guide_consistency()
        if self.data.get("object") or self.data.get("network"):
            print("Skipping organelle plots")
            # self.plot_organelle_distributions()
            # self.plot_organelle_comparison()
        print("--- Graph Generation Complete ---")

    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Returns a list of feature columns from a dataframe by excluding metadata columns.
        """
        exclude_patterns = [
            "id",
            "gene",
            "barcode",
            "sgRNA",
            "effect",
            "NCBI",
            "index",
            "well",
            "pos",
            "_pheno",
            "_bc",
            "Unnamed",
            "bbox",
            "umap",
            "cluster",
            "radial",
        ]
        feature_cols = [
            col
            for col in df.select_dtypes(include=np.number).columns
            if not any(pattern in col for pattern in exclude_patterns)
        ]
        return feature_cols

    def _prepare_features_for_umap(self):
        """
        Prepares data for UMAP by selecting, cleaning, and returning the primary feature matrix
        and the corresponding cell metadata dataframe.
        """
        print("--- Preparing features for UMAP analysis ---")
        cell_df = self.data["cell"].copy()

        # --- Robustly select features for UMAP using the utility function ---
        feature_cols = self._get_feature_columns(cell_df)
        features = cell_df[feature_cols].copy()

        if features.empty:
            print("Warning: No numeric features found for UMAP generation.")
            return None, None

        features.fillna(0, inplace=True)

        # --- FIX: Remove duplicate rows based on feature values ---
        n_original = len(features)
        # Keep track of original index to filter cell_df accordingly
        original_index = features.index
        features.drop_duplicates(inplace=True)

        if len(features) < n_original:
            print(
                f"  - INFO: Removed {n_original - len(features)} duplicate rows from feature set."
            )
            # Filter cell_df to match the deduplicated features
            cell_df = cell_df.loc[features.index].copy()
        else:
            # If no duplicates, ensure cell_df is a fresh copy
            cell_df = cell_df.copy()

        # --- NEW: Remove features with zero variance ---
        print("  - Checking for features with very low variance...")
        low_variance_threshold = 1e-4
        variances = features.var()
        low_variance_mask = variances < low_variance_threshold
        low_variance_cols = features.columns[low_variance_mask]

        if len(low_variance_cols) > 0:
            print(
                f"  - INFO: Removing {len(low_variance_cols)} features with variance below {low_variance_threshold}."
            )
            print(f"    - Removed columns: {low_variance_cols.tolist()}")
            features = features.drop(columns=low_variance_cols)

        # --- NEW: Save summary of UMAP input features ---
        print("  - Calculating and saving summary of UMAP input features...")
        feature_summary = features.describe().transpose()
        feature_summary_to_save = feature_summary[
            ["min", "max", "mean", "50%", "std"]
        ].copy()
        feature_summary_to_save.rename(columns={"50%": "median"}, inplace=True)

        save_path = self.graph_output_path / "umap_input_feature_summary.csv"
        feature_summary_to_save.to_csv(save_path)
        print(f"  - Saved feature summary to: {save_path}")

        return features, cell_df

    def _perform_umap_and_generate_plots(
        self,
        features: pd.DataFrame,
        cell_df: pd.DataFrame,
        plot_prefix: str,
        output_dir: Path,
    ):
        """
        Performs UMAP embedding, clustering, and generates a suite of standard plots.
        Tries to use cuML for GPU acceleration and falls back to CPU if not available.

        Returns
        -------
        pd.DataFrame
            A dataframe with UMAP coordinates and cluster labels for each cell.
        pd.DataFrame
            A dataframe with top enriched genes for each cluster.
        """
        output_dir.mkdir(exist_ok=True)

        # --- Check cache first ---
        cache_key = f"{self.experiment}_{plot_prefix}"
        n_cells = len(features)
        cached = self._load_from_cache(cache_key, n_cells)

        clustering_methods = []
        if self.cluster_algo == "all":
            clustering_methods = ["hdbscan", "kmeans", "leiden"]
        else:
            clustering_methods = [self.cluster_algo]

        if cached is not None:
            # Use cached embedding and clusters
            embedding = cached['embedding']
            plot_df = cell_df.copy()
            plot_df["umap_1"] = embedding[:, 0]
            plot_df["umap_2"] = embedding[:, 1]

            # Load cached clusters
            clusters_dict = {}
            for method in clustering_methods:
                cache_cluster_key = f'clusters_{method}'
                if cache_cluster_key in cached:
                    clusters = cached[cache_cluster_key]
                    clusters_dict[method] = clusters
                    cluster_col = f"cluster_{method}"
                    plot_df[cluster_col] = "c" + clusters.astype(str)
                    n_clusters = len([c for c in plot_df[cluster_col].unique() if c != "c-1"])
                    print(f"    - {method.upper()} (cached): {n_clusters} clusters")
                else:
                    # Run clustering if not in cache
                    print(f"  - Running {method.upper()} clustering (not in cache)...")
                    clusters = self._run_clustering(embedding, method, features)
                    clusters_dict[method] = clusters
                    cluster_col = f"cluster_{method}"
                    plot_df[cluster_col] = "c" + clusters.astype(str)
                    n_clusters = len([c for c in plot_df[cluster_col].unique() if c != "c-1"])
                    print(f"    - {method.upper()}: {n_clusters} clusters found")

            # Update cache with any new clusters
            self._save_to_cache(cache_key, embedding, clusters_dict, n_cells)
        else:
            # No cache - compute everything
            # 1. Scale features and run UMAP
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            embedding = None
            if self.use_cuml:
                try:
                    from cuml.manifold import UMAP as cuMLUMAP

                    print("    - Attempting to use cuML (GPU) for UMAP...")
                    # Data must be float32 and contiguous for cuML
                    scaled_features_32 = np.ascontiguousarray(
                        scaled_features, dtype=np.float32
                    )

                    reducer = cuMLUMAP(
                        n_neighbors=15,
                        min_dist=0.1,
                        n_components=2,
                        random_state=42,
                        init="random",  # Using random init to hopefully improve on 'spectral'
                    )
                    embedding = reducer.fit_transform(scaled_features_32)
                    print("    - cuML UMAP successful.")
                except RuntimeError as e:  # Specifically for the RAFT failure
                    if "RAFT failure" in str(e):
                        print(
                            f"    - WARNING: cuML UMAP (GPU) failed with a RAFT error, likely due to duplicate data points in this feature subset."
                        )
                        print(
                            "    - Falling back to umap-learn (CPU) for this specific UMAP."
                        )
                        embedding = None  # Ensure embedding is None so CPU path is taken
                    else:
                        raise  # Re-raise other runtime errors
                except (ImportError, TypeError) as e:
                    print(
                        f"    - cuML for UMAP not available or failed: {e}. Falling back to CPU."
                    )
                    embedding = None  # Also ensure fallback

            if embedding is None:
                import umap

                print("    - Using umap-learn (CPU) for UMAP...")
                # --- UMAP REVERT: Restore original, known-good umap-learn parameters ---
                reducer = umap.UMAP(
                    n_neighbors=15, min_dist=0.1, n_components=2, random_state=42
                )
                embedding = reducer.fit_transform(scaled_features)
                print("    - umap-learn UMAP successful.")

            plot_df = cell_df.copy()
            plot_df["umap_1"] = embedding[:, 0]
            plot_df["umap_2"] = embedding[:, 1]

            # 2. Run clustering - collect results for caching
            clusters_dict = {}

            # Run each clustering method and store results
            for method in clustering_methods:
                cluster_col = f"cluster_{method}"
                clusters = self._run_clustering(embedding, method, features)
                clusters_dict[method] = clusters
                plot_df[cluster_col] = "c" + clusters.astype(str)
                n_clusters = len([c for c in plot_df[cluster_col].unique() if c != "c-1"])
                print(f"    - {method.upper()}: {n_clusters} clusters found")

            # Save to cache
            self._save_to_cache(cache_key, embedding, clusters_dict, n_cells)

        # Use the first method's clusters as the default "cluster" column for downstream analysis
        primary_method = clustering_methods[0]
        plot_df["cluster"] = plot_df[f"cluster_{primary_method}"]

        # 3. Gene Enrichment Analysis
        print("  - Performing gene enrichment analysis for clusters...")

        # --- OPTIMIZATION: Filter clusters before running enrichment ---
        # Only analyze clusters that are large enough to yield meaningful stats.
        min_enrichment_cluster_size = 10
        cluster_counts = plot_df["cluster"].value_counts()
        clusters_to_analyze = cluster_counts[
            cluster_counts >= min_enrichment_cluster_size
        ].index.tolist()

        # Remove the noise cluster if it's present
        if "c-1" in clusters_to_analyze:
            clusters_to_analyze.remove("c-1")

        print(
            f"  - Found {len(cluster_counts)} total clusters. Analyzing {len(clusters_to_analyze)} clusters with at least {min_enrichment_cluster_size} cells."
        )

        full_enrichment_results_df = (
            pd.DataFrame()
        )  # Store all results for interactive plot
        top_genes_for_annotation = pd.DataFrame()  # Store top 5 for static plot
        if len(clusters_to_analyze) > 0 and "gene_name" in plot_df.columns:
            num_workers = get_optimal_workers(use_gpu=False)
            print(f"  - Running enrichment in parallel with {num_workers} workers...")

            # This returns a list of dataframes, one for each cluster
            cluster_enrichment_dfs = Parallel(n_jobs=num_workers)(
                delayed(self._calculate_enrichment_for_cluster)(cluster_id, plot_df)
                for cluster_id in tqdm(
                    clusters_to_analyze, desc="Calculating gene enrichment"
                )
            )

            # Collect and format results from parallel execution
            valid_results = [df for df in cluster_enrichment_dfs if df is not None]
            if valid_results:
                full_enrichment_results_df = pd.concat(valid_results, ignore_index=True)
                # For the static plot annotation, we only need the top few genes
                top_genes_for_annotation = full_enrichment_results_df.groupby(
                    "cluster"
                ).head(5)

        # --- NEW: Select top 5 genes for highlight plots from this specific embedding's enrichment results ---
        highlight_genes_for_this_embedding = []
        if not full_enrichment_results_df.empty:
            print(
                "  - Selecting top 5 genes for highlight plots based on max odds ratio for this embedding..."
            )
            # Find the row with the maximum odds_ratio for each gene
            top_enrichment_per_gene = full_enrichment_results_df.loc[
                full_enrichment_results_df.groupby("gene_name")["odds_ratio"].idxmax()
            ]
            # Sort these genes by that maximum odds_ratio
            top_5_genes = top_enrichment_per_gene.sort_values(
                "odds_ratio", ascending=False
            ).head(5)
            highlight_genes_for_this_embedding = top_5_genes["gene_name"].tolist()
            print(
                f"  - Genes selected for highlighting for this embedding: {highlight_genes_for_this_embedding}"
            )

        # Create annotation text similar to the old script, with odds ratio
        cluster_annotations = {}
        if "top_genes_for_annotation" in locals():
            for cluster_id, group in top_genes_for_annotation.groupby("cluster"):
                # NEW: Sort by enrichment (odds_ratio) to ensure the list is ordered correctly
                sorted_group = group.sort_values("odds_ratio", ascending=False)
                gene_list = [
                    f"{row.gene_name} ({row.odds_ratio:.1f}x)"
                    for _, row in sorted_group.iterrows()
                ]
                cluster_annotations[cluster_id] = "\n".join(gene_list)

        # 4. Generate plots
        print("  - Generating static UMAP plots...")

        # --- NEW: Define plot configs with improved settings ---
        plot_configs = {
            "cluster_annotated": {
                "hue_col": "cluster",
                "title_suffix": "HDBSCAN Clustering (Annotated)",
                "legend": False,
                "annotate": True,
            },
            "cluster": {
                "hue_col": "cluster",
                "title_suffix": "HDBSCAN Clustering",
                "legend": False,
                "annotate": False,
            },
            "perturbation_status": {
                "hue_col": "gene_group",
                "title_suffix": "Perturbation Status",
                "legend": True,
                "annotate": False,
            },
            "gene_effect": {
                "hue_col": "gene_effect",
                "title_suffix": "Gene Effect",
                "annotate": False,
                "is_continuous": True,
            },
            "well": {
                "hue_col": "well",
                "title_suffix": "Well",
                "legend": False,
                "annotate": False,
            },
        }

        # Add a gene_group column for the perturbation status plot
        if "gene_name" in plot_df.columns:
            plot_df["gene_group"] = np.where(
                plot_df["gene_name"]
                .astype(str)
                .str.contains("ntc|non-targeting", case=False, regex=True),
                "NTC",
                "Perturbed",
            )

        # --- PRE-GENERATE PALETTES for consistent colors ---
        # Palette for clusters
        cluster_palette = None
        if "cluster" in plot_df.columns:
            unique_clusters = sorted(plot_df["cluster"].unique())
            n_clusters = len([c for c in unique_clusters if c != "c-1"])
            if n_clusters > 0:
                colors = sns.color_palette("turbo", n_colors=n_clusters)
                cluster_palette = {
                    c: colors[i]
                    for i, c in enumerate(u for u in unique_clusters if u != "c-1")
                }
            else:
                cluster_palette = {}
            cluster_palette["c-1"] = (0.8, 0.8, 0.8)  # Faded grey for noise

        for name, config in plot_configs.items():
            hue_col = config["hue_col"]
            if hue_col not in plot_df.columns:
                print(
                    f"  - Skipping UMAP plot for '{name}': column '{hue_col}' not found."
                )
                continue

            fig, ax = plt.subplots(figsize=(16, 12))

            # --- Custom plotting logic based on plot type ---

            # 1. NTC vs Perturbed: Use styling from gene highlight plots
            if name == "perturbation_status":
                ntc_data = plot_df[plot_df[hue_col] == "NTC"]
                pert_data = plot_df[plot_df[hue_col] == "Perturbed"]

                # Plot perturbed cells as faint gray background
                ax.scatter(
                    x=pert_data["umap_1"],
                    y=pert_data["umap_2"],
                    color="lightgray",
                    s=5,
                    alpha=0.5,
                    label=f"Perturbed ({len(pert_data)} cells)",
                    rasterized=True,
                )

                # Plot NTC cells in bright red on top
                ax.scatter(
                    x=ntc_data["umap_1"],
                    y=ntc_data["umap_2"],
                    color="red",
                    s=8,
                    alpha=0.7,
                    label=f"NTC ({len(ntc_data)} cells)",
                )

                if config.get("legend", False):
                    ax.legend()

            # 2. Gene Effect: Continuous color map with a colorbar
            elif config.get("is_continuous", False):
                if pd.api.types.is_numeric_dtype(plot_df[hue_col]):
                    from matplotlib.colors import Normalize

                    cmap = plt.get_cmap("viridis")
                    values = plot_df[hue_col].fillna(0)
                    vmax = values.quantile(0.99)
                    norm = Normalize(vmin=values.min(), vmax=vmax)

                    scatter = ax.scatter(
                        plot_df["umap_1"],
                        plot_df["umap_2"],
                        c=values,
                        cmap=cmap,
                        norm=norm,
                        s=5,
                        alpha=0.5,
                    )

                    cbar = fig.colorbar(scatter, ax=ax, orientation="vertical")
                    cbar.set_label(config["title_suffix"])
                else:
                    print(
                        f"  - Warning: '{hue_col}' is not numeric, skipping continuous plot for '{name}'."
                    )
                    plt.close(fig)
                    continue

            # 3. Well Plot: High-contrast colors and low alpha
            elif name == "well":
                # Convert to string to avoid categorical/MultiIndex issues
                well_values = plot_df[hue_col].astype(str)
                well_ids = sorted(well_values.unique())
                # Use a high-contrast colormap
                if len(well_ids) <= 20:
                    cmap = plt.get_cmap("tab20", len(well_ids))
                else:
                    cmap = plt.get_cmap("turbo", len(well_ids))

                well_color_map = {well: cmap(i) for i, well in enumerate(well_ids)}
                colors = well_values.map(well_color_map)

                # No legend for well plot to avoid clutter
                # ax.legend(fontsize=22)

                ax.scatter(
                    plot_df["umap_1"],
                    plot_df["umap_2"],
                    c=colors,
                    s=8,
                    alpha=0.7,
                    rasterized=True,
                )

            # 4. Cluster plots: Use the pre-generated consistent palette
            elif "cluster" in name:
                sns.scatterplot(
                    data=plot_df,
                    x="umap_1",
                    y="umap_2",
                    hue=hue_col,
                    s=5,
                    alpha=0.5,
                    palette=cluster_palette,
                    legend=config.get("legend", False),
                    ax=ax,
                )

            # --- Common plot settings ---
            plt.title(f'UMAP by {config["title_suffix"]} ({plot_prefix})')

            if config.get("annotate", False):
                try:
                    from adjustText import adjust_text

                    print("  - Using 'adjustText' to improve annotation layout.")
                    if "cluster" in plot_df.columns:
                        cluster_centers = plot_df.groupby("cluster")[
                            ["umap_1", "umap_2"]
                        ].median()

                        # --- NEW: Offset annotation starting position by changing alignment ---
                        plot_center_x = plot_df["umap_1"].mean()
                        plot_center_y = plot_df["umap_2"].mean()

                        texts_to_adjust = []

                        for cluster_id, center in cluster_centers.iterrows():
                            if (
                                cluster_id in cluster_annotations
                                and cluster_id != "c-1"
                            ):

                                # Determine alignment based on quadrant relative to plot center
                                ha = (
                                    "left"
                                    if center["umap_1"] > plot_center_x
                                    else "right"
                                )
                                va = (
                                    "bottom"
                                    if center["umap_2"] > plot_center_y
                                    else "top"
                                )

                                texts_to_adjust.append(
                                    ax.text(
                                        x=center["umap_1"],
                                        y=center["umap_2"],
                                        s=cluster_annotations[cluster_id],
                                        fontdict={"size": 9},
                                        color="black",
                                        ha=ha,  # Dynamically set horizontal alignment
                                        va=va,  # Dynamically set vertical alignment
                                        bbox=dict(
                                            boxstyle="round,pad=0.1",
                                            fc="white",
                                            ec="black",
                                            alpha=0.5,
                                        ),
                                    )
                                )

                        if texts_to_adjust:
                            adjust_text(
                                texts_to_adjust,
                                ax=ax,
                                arrowprops=dict(
                                    arrowstyle="-", color="gray", lw=0.5, alpha=0.7
                                ),
                            )

                except ImportError:
                    print(
                        "  - WARNING: 'adjustText' library not found. Annotations may overlap."
                    )
                    print(
                        "    For improved annotation layout, run: pip install adjustText"
                    )
                    if "cluster" in plot_df.columns:
                        cluster_centers = plot_df.groupby("cluster")[
                            ["umap_1", "umap_2"]
                        ].median()
                        for cluster_id, center in cluster_centers.iterrows():
                            if (
                                cluster_id in cluster_annotations
                                and cluster_id != "c-1"
                            ):
                                ax.text(
                                    center["umap_1"],
                                    center["umap_2"],
                                    cluster_annotations[cluster_id],
                                    fontdict={"size": 9},
                                    color="black",
                                    bbox=dict(
                                        boxstyle="round,pad=0.1",
                                        fc="white",
                                        ec="black",
                                        alpha=0.5,
                                    ),
                                )

            save_path = output_dir / f"umap_{plot_prefix}_{name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        # --- Generate the two sets of single-gene highlight plots ---
        # 1. Enriched Genes for this embedding
        if highlight_genes_for_this_embedding:
            self._generate_gene_highlight_umaps(
                plot_df,
                plot_prefix,
                output_dir,
                genes_to_plot=highlight_genes_for_this_embedding,
                subfolder_name="enriched_gene_highlights",
                enrichment_df=full_enrichment_results_df,
            )

        # 2. Random Genes
        if self.random_highlight_genes:
            self._generate_gene_highlight_umaps(
                plot_df,
                plot_prefix,
                output_dir,
                genes_to_plot=self.random_highlight_genes,
                subfolder_name="random_gene_highlights",
                enrichment_df=full_enrichment_results_df,
            )

        # --- RE-INTRODUCE RADIAL POSITION PLOTS ---
        plot_df_with_pos = self._calculate_radial_positions(plot_df)
        for pos_type in ["well_radial_pos", "tile_radial_pos"]:
            if (
                pos_type in plot_df_with_pos.columns
                and plot_df_with_pos[pos_type].notna().any()
            ):
                plt.figure(figsize=(14, 10))
                cmap = "viridis" if "well" in pos_type else "magma"
                scatter = plt.scatter(
                    plot_df_with_pos["umap_1"],
                    plot_df_with_pos["umap_2"],
                    c=plot_df_with_pos[pos_type],
                    cmap=cmap,
                    s=5,
                    alpha=0.4,
                )
                plt.colorbar(
                    scatter,
                    label=f'Radial Distance from {pos_type.split("_")[0].capitalize()} Center',
                )
                plt.title(
                    f'UMAP colored by {pos_type.replace("_", " ").title()} ({plot_prefix})'
                )
                save_path = output_dir / f"umap_{plot_prefix}_{pos_type}.png"
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()

        return plot_df, full_enrichment_results_df

    def _run_clustering(self, embedding: np.ndarray, method: str, features: pd.DataFrame = None) -> np.ndarray:
        """
        Run a specific clustering algorithm on the UMAP embedding.

        Parameters
        ----------
        embedding : np.ndarray
            2D UMAP embedding (n_samples, 2)
        method : str
            Clustering method: "hdbscan", "kmeans", or "leiden"
        features : pd.DataFrame, optional
            Original feature matrix (used for Leiden which builds a neighbor graph)

        Returns
        -------
        np.ndarray
            Cluster labels (integers, -1 for noise in HDBSCAN)
        """
        print(f"  - Running {method.upper()} clustering...")

        if method == "hdbscan":
            clusters = None
            if self.use_cuml:
                try:
                    from cuml.cluster import HDBSCAN as cuMLHDBSCAN
                    clusterer = cuMLHDBSCAN(
                        min_cluster_size=25, min_samples=25, gen_min_span_tree=True
                    )
                    clusters = clusterer.fit_predict(embedding)
                except (ImportError, TypeError) as e:
                    print(f"    - cuML HDBSCAN not available: {e}. Falling back to CPU.")

            if clusters is None:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=25,
                    min_samples=25,
                    gen_min_span_tree=True,
                    core_dist_n_jobs=-1,
                )
                clusters = clusterer.fit_predict(embedding)

            return clusters

        elif method == "kmeans":
            optimal_k = self._find_optimal_k(embedding)
            if optimal_k > 1:
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(embedding)
                return clusters
            else:
                print("    - Could not determine optimal k, assigning all to cluster -1")
                return np.full(len(embedding), -1)

        elif method == "leiden":
            # Leiden clustering using scanpy with fast igraph backend
            # For large datasets, use subsampling + label propagation
            n_cells = len(embedding)
            max_cells_for_full_leiden = 100000  # 100k cells max for full Leiden

            if n_cells > max_cells_for_full_leiden:
                print(f"    - Large dataset ({n_cells:,} cells), using subsampled Leiden + nearest-neighbor propagation...")

                # Subsample for clustering
                np.random.seed(42)
                sample_idx = np.random.choice(n_cells, max_cells_for_full_leiden, replace=False)
                embedding_sample = embedding[sample_idx]

                # Run Leiden on subsample
                adata_temp = ad.AnnData(X=embedding_sample)
                sc.pp.neighbors(adata_temp, n_neighbors=15, use_rep='X')
                sc.tl.leiden(
                    adata_temp,
                    resolution=1.0,
                    random_state=42,
                    flavor='igraph',
                    n_iterations=2,
                    directed=False,
                )
                sample_clusters = adata_temp.obs['leiden'].astype(int).values

                # Propagate labels to all cells using nearest neighbor in embedding space
                print(f"    - Propagating cluster labels to all {n_cells:,} cells...")
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
                nn.fit(embedding_sample)
                _, indices = nn.kneighbors(embedding)
                clusters = sample_clusters[indices.flatten()]
                return clusters
            else:
                # Full Leiden for smaller datasets
                adata_temp = ad.AnnData(X=embedding)

                # Compute neighbors on the UMAP embedding
                print("    - Computing neighbor graph on UMAP embedding...")
                sc.pp.neighbors(adata_temp, n_neighbors=15, use_rep='X')

                # Run Leiden clustering with fast igraph backend
                print("    - Running Leiden algorithm (igraph backend)...")
                sc.tl.leiden(
                    adata_temp,
                    resolution=1.0,
                    random_state=42,
                    flavor='igraph',
                    n_iterations=2,
                    directed=False,
                )

                clusters = adata_temp.obs['leiden'].astype(int).values
                return clusters

        else:
            raise ValueError(f"Unknown clustering method: {method}")

    def _generate_aggregated_level_umap(self, level: str):
        """
        Generate UMAP embedding for guide or gene level aggregated features.

        Parameters
        ----------
        level : str
            Either "guide" or "gene"
        """
        if level == "guide":
            df = self.data.get("guideRNA")
            id_col = "barcode"
            label_col = "gene_name"
        elif level == "gene":
            df = self.data.get("gene")
            id_col = "gene_name"
            label_col = "gene_name"
        else:
            raise ValueError(f"Unknown level: {level}. Use 'guide' or 'gene'.")

        if df is None or df.empty:
            print(f"  - No {level}-level data available, skipping UMAP.")
            return

        # Get feature columns (numeric, excluding metadata)
        metadata_cols = {
            "barcode", "sgRNA", "gene_name", "gene_effect", "NCBI_ID",
            "n_cells", "n_guides", "well"
        }
        feature_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and c not in metadata_cols
            and not c.endswith("_count")  # Exclude simple count columns
        ]

        if len(feature_cols) < 2:
            print(f"  - Not enough numeric features for {level}-level UMAP, skipping.")
            return

        print(f"  - Using {len(feature_cols)} features for {level}-level UMAP")

        # Prepare feature matrix
        features = df[feature_cols].copy()

        # Handle NaN/Inf values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        # --- Check cache first ---
        cache_key = f"{self.experiment}_{level}_level"
        n_items = len(df)
        cached = self._load_from_cache(cache_key, n_items)

        clustering_methods = ["hdbscan", "kmeans", "leiden"] if self.cluster_algo == "all" else [self.cluster_algo]

        if cached is not None:
            # Use cached embedding and clusters
            embedding = cached['embedding']
            plot_df = df.copy()
            plot_df["umap_1"] = embedding[:, 0]
            plot_df["umap_2"] = embedding[:, 1]

            # Load cached clusters
            clusters_dict = {}
            for method in clustering_methods:
                cache_cluster_key = f'clusters_{method}'
                if cache_cluster_key in cached:
                    clusters = cached[cache_cluster_key]
                    clusters_dict[method] = clusters
                    cluster_col = f"cluster_{method}"
                    plot_df[cluster_col] = "c" + clusters.astype(str)
                    n_clusters = len([c for c in plot_df[cluster_col].unique() if c != "c-1"])
                    print(f"    - {method.upper()} (cached): {n_clusters} clusters")
        else:
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Run UMAP
            print(f"  - Running UMAP on {len(df)} {level}s...")
            try:
                if self.use_cuml:
                    try:
                        from cuml.manifold import UMAP as cuMLUMAP
                        reducer = cuMLUMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
                        embedding = reducer.fit_transform(features_scaled)
                        print(f"    - cuML UMAP successful.")
                    except (ImportError, Exception) as e:
                        print(f"    - cuML UMAP failed: {e}. Falling back to CPU.")
                        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
                        embedding = reducer.fit_transform(features_scaled)
                else:
                    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
                    embedding = reducer.fit_transform(features_scaled)
            except Exception as e:
                print(f"  - UMAP failed for {level} level: {e}")
                return

            # Create plot dataframe
            plot_df = df.copy()
            plot_df["umap_1"] = embedding[:, 0]
            plot_df["umap_2"] = embedding[:, 1]

            # Run clustering (all methods)
            clusters_dict = {}
            for method in clustering_methods:
                cluster_col = f"cluster_{method}"
                clusters = self._run_clustering(embedding, method, features)
                clusters_dict[method] = clusters
                plot_df[cluster_col] = "c" + clusters.astype(str)
                n_clusters = len([c for c in plot_df[cluster_col].unique() if c != "c-1"])
                print(f"    - {method.upper()}: {n_clusters} clusters found")

            # Save to cache
            self._save_to_cache(cache_key, embedding, clusters_dict, n_items)

        # Use first method as primary
        primary_method = clustering_methods[0]
        plot_df["cluster"] = plot_df[f"cluster_{primary_method}"]

        # Generate plots
        output_dir = self.graph_output_path / f"{level}_umap"
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save embedding data
        plot_df.to_csv(output_dir / f"{level}_umap_data.csv", index=False)

        # --- Generate full suite of plots like cell-level ---
        self._generate_aggregated_level_plots(
            plot_df, features, embedding, level, output_dir, clustering_methods, label_col
        )

        # Save embeddings to AnnData
        if level in self.adata:
            adata = self.adata[level]
            adata.obsm["X_umap"] = embedding
            for method in clustering_methods:
                cluster_col = f"cluster_{method}"
                adata.obs[cluster_col] = plot_df[cluster_col].values

            # Save updated AnnData
            h5ad_path = self.dataset.analysis_path / f"{self.experiment}_{level}_features.h5ad"
            adata.write_h5ad(h5ad_path)
            print(f"  - Saved {level}-level embeddings to AnnData: {h5ad_path}")

    def _generate_aggregated_level_plots(
        self,
        plot_df: pd.DataFrame,
        features: pd.DataFrame,
        embedding: np.ndarray,
        level: str,
        output_dir: Path,
        clustering_methods: list,
        label_col: str,
    ):
        """
        Generate full suite of plots for guide/gene level analysis.

        Mirrors the cell-level analysis with:
        - Cluster-colored UMAPs with proper annotations
        - Gene effect colored UMAPs
        - n_cells colored UMAPs
        - NTC vs perturbed comparison
        - Feature importance / top features analysis
        """
        print(f"\n  --- Generating {level}-level plot suite ---")

        # Try to use adjustText for better annotations
        try:
            from adjustText import adjust_text
            use_adjust_text = True
            print("  - Using 'adjustText' for improved label layout.")
        except ImportError:
            use_adjust_text = False
            print("  - 'adjustText' not available, using basic annotations.")

        # Identify NTC items
        if "gene_name" in plot_df.columns:
            ntc_mask = plot_df["gene_name"].astype(str).str.contains(
                "ntc|non-targeting|^0$", case=False, regex=True
            )
        else:
            ntc_mask = pd.Series([False] * len(plot_df), index=plot_df.index)

        # --- 1. Cluster-colored UMAPs for each method ---
        for method in clustering_methods:
            cluster_col = f"cluster_{method}"
            fig, ax = plt.subplots(figsize=(14, 12))

            # Get unique clusters and create color palette
            unique_clusters = sorted(plot_df[cluster_col].unique())
            n_clusters = len([c for c in unique_clusters if c != "c-1"])
            if n_clusters > 0:
                colors = sns.color_palette("husl", n_colors=max(n_clusters, 1))
                palette = {c: colors[i % len(colors)] for i, c in enumerate(c for c in unique_clusters if c != "c-1")}
            else:
                palette = {}
            palette["c-1"] = (0.8, 0.8, 0.8)

            # Plot noise points first (background)
            noise_df = plot_df[plot_df[cluster_col] == "c-1"]
            if len(noise_df) > 0:
                ax.scatter(noise_df["umap_1"], noise_df["umap_2"], c="lightgray", s=40, alpha=0.3, label="Noise")

            # Plot clustered points
            clustered_df = plot_df[plot_df[cluster_col] != "c-1"]
            scatter = ax.scatter(
                clustered_df["umap_1"],
                clustered_df["umap_2"],
                c=[palette.get(c, "gray") for c in clustered_df[cluster_col]],
                s=60,
                alpha=0.8,
            )

            # Add annotations with adjustText
            texts = []
            # Label all points for gene level, or outliers for guide level
            if level == "gene":
                # Label outliers (top 5% by distance from center) plus cluster centers
                center_x, center_y = plot_df["umap_1"].mean(), plot_df["umap_2"].mean()
                distances = np.sqrt((plot_df["umap_1"] - center_x)**2 + (plot_df["umap_2"] - center_y)**2)
                outlier_threshold = np.percentile(distances, 92)
                outliers_mask = distances > outlier_threshold

                # Also label cluster centers
                for cluster in unique_clusters:
                    if cluster == "c-1":
                        continue
                    cluster_points = plot_df[plot_df[cluster_col] == cluster]
                    if len(cluster_points) > 0:
                        # Find point closest to cluster center
                        cx = cluster_points["umap_1"].mean()
                        cy = cluster_points["umap_2"].mean()
                        dists = np.sqrt((cluster_points["umap_1"] - cx)**2 + (cluster_points["umap_2"] - cy)**2)
                        center_idx = dists.idxmin()
                        outliers_mask.loc[center_idx] = True

                for idx, row in plot_df[outliers_mask].iterrows():
                    label_text = str(row[label_col]) if pd.notna(row[label_col]) else ""
                    if label_text and label_text != "nan":
                        texts.append(ax.text(row["umap_1"], row["umap_2"], label_text, fontsize=8, alpha=0.9))

            elif level == "guide":
                # For guides, label top guides by n_cells within each cluster
                for cluster in unique_clusters:
                    if cluster == "c-1":
                        continue
                    cluster_points = plot_df[plot_df[cluster_col] == cluster]
                    if len(cluster_points) > 0 and "n_cells" in cluster_points.columns:
                        top_guides = cluster_points.nlargest(2, "n_cells")
                        for idx, row in top_guides.iterrows():
                            label_text = str(row.get("barcode", row.get(label_col, "")))[:15]
                            if label_text:
                                texts.append(ax.text(row["umap_1"], row["umap_2"], label_text, fontsize=7, alpha=0.8))

            if use_adjust_text and texts:
                try:
                    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))
                except Exception as e:
                    print(f"    - Warning: adjustText failed: {e}")

            ax.set_title(f"{level.capitalize()}-Level UMAP ({method.upper()} Clustering, {n_clusters} clusters)", fontsize=14)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")

            # Add legend
            handles = [plt.scatter([], [], color=palette[c], s=60, label=c) for c in sorted(palette.keys()) if c != "c-1"]
            if handles:
                ax.legend(handles=handles, title="Cluster", bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=1.5)

            plt.tight_layout()
            save_path = output_dir / f"{level}_umap_{method}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  - Saved {level}-level UMAP ({method}): {save_path}")

        # --- 2. Gene effect colored UMAP (if available) ---
        if "gene_effect" in plot_df.columns:
            gene_effect = pd.to_numeric(plot_df["gene_effect"], errors="coerce")
            if gene_effect.notna().sum() > 10:
                fig, ax = plt.subplots(figsize=(14, 12))

                # Plot NTC as gray
                ntc_df = plot_df[ntc_mask]
                pert_df = plot_df[~ntc_mask]

                if len(ntc_df) > 0:
                    ax.scatter(ntc_df["umap_1"], ntc_df["umap_2"], c="lightgray", s=40, alpha=0.5, label="NTC")

                # Plot perturbed colored by gene effect
                pert_effect = gene_effect[~ntc_mask]
                scatter = ax.scatter(
                    pert_df["umap_1"],
                    pert_df["umap_2"],
                    c=pert_effect,
                    cmap="RdBu_r",
                    s=60,
                    alpha=0.8,
                    vmin=-1,
                    vmax=0.5,
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label("Gene Effect (CERES)")

                # Label extreme gene effects
                texts = []
                if level == "gene":
                    # Label genes with strongest effects
                    pert_sorted = pert_df.copy()
                    pert_sorted["_effect"] = pert_effect
                    pert_sorted = pert_sorted.dropna(subset=["_effect"])
                    top_essential = pert_sorted.nsmallest(10, "_effect")
                    top_nonessential = pert_sorted.nlargest(5, "_effect")

                    for idx, row in pd.concat([top_essential, top_nonessential]).iterrows():
                        label_text = str(row[label_col])
                        if label_text and label_text != "nan":
                            texts.append(ax.text(row["umap_1"], row["umap_2"], label_text, fontsize=8))

                if use_adjust_text and texts:
                    try:
                        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))
                    except Exception:
                        pass

                ax.set_title(f"{level.capitalize()}-Level UMAP Colored by Gene Effect", fontsize=14)
                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
                plt.tight_layout()

                save_path = output_dir / f"{level}_umap_gene_effect.png"
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  - Saved {level}-level UMAP (gene_effect): {save_path}")

        # --- 3. n_cells colored UMAP ---
        if "n_cells" in plot_df.columns:
            fig, ax = plt.subplots(figsize=(14, 12))

            scatter = ax.scatter(
                plot_df["umap_1"],
                plot_df["umap_2"],
                c=np.log10(plot_df["n_cells"] + 1),
                cmap="viridis",
                s=60,
                alpha=0.8,
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("log10(n_cells + 1)")

            ax.set_title(f"{level.capitalize()}-Level UMAP Colored by Cell Count", fontsize=14)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            plt.tight_layout()

            save_path = output_dir / f"{level}_umap_n_cells.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  - Saved {level}-level UMAP (n_cells): {save_path}")

        # --- 4. NTC vs Perturbed comparison ---
        if ntc_mask.sum() > 0 and (~ntc_mask).sum() > 0:
            fig, ax = plt.subplots(figsize=(14, 12))

            pert_df = plot_df[~ntc_mask]
            ntc_df = plot_df[ntc_mask]

            ax.scatter(pert_df["umap_1"], pert_df["umap_2"], c="steelblue", s=50, alpha=0.6, label=f"Perturbed ({len(pert_df)})")
            ax.scatter(ntc_df["umap_1"], ntc_df["umap_2"], c="orange", s=80, alpha=0.9, label=f"NTC ({len(ntc_df)})", edgecolors="black", linewidths=0.5)

            ax.set_title(f"{level.capitalize()}-Level UMAP: NTC vs Perturbed", fontsize=14)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.legend(markerscale=1.5)
            plt.tight_layout()

            save_path = output_dir / f"{level}_umap_ntc_vs_perturbed.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  - Saved {level}-level UMAP (NTC vs perturbed): {save_path}")

        # --- 5. Feature importance: top features driving UMAP structure ---
        print(f"  - Computing feature correlations with UMAP coordinates...")
        feature_correlations = []
        for col in features.columns:
            corr_umap1, _ = spearmanr(features[col], plot_df["umap_1"])
            corr_umap2, _ = spearmanr(features[col], plot_df["umap_2"])
            max_corr = max(abs(corr_umap1), abs(corr_umap2))
            feature_correlations.append({
                "feature": col,
                "corr_umap1": corr_umap1,
                "corr_umap2": corr_umap2,
                "max_abs_corr": max_corr,
            })

        corr_df = pd.DataFrame(feature_correlations).sort_values("max_abs_corr", ascending=False)
        corr_df.to_csv(output_dir / f"{level}_feature_umap_correlations.csv", index=False)

        # Plot top features
        top_features = corr_df.head(20)
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ["steelblue" if c > 0 else "coral" for c in top_features["corr_umap1"]]
        bars = ax.barh(range(len(top_features)), top_features["max_abs_corr"], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"])
        ax.invert_yaxis()
        ax.set_xlabel("Max Absolute Spearman Correlation with UMAP")
        ax.set_title(f"Top Features Driving {level.capitalize()}-Level UMAP Structure")
        plt.tight_layout()

        save_path = output_dir / f"{level}_top_features_umap.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  - Saved top features plot: {save_path}")

        # --- 6. Volcano plot: feature z-scores vs NTC ---
        if ntc_mask.sum() >= 3 and (~ntc_mask).sum() >= 3:
            print(f"  - Computing {level}-level volcano plot (features vs NTC)...")
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

            if volcano_data:
                volcano_df = pd.DataFrame(volcano_data)
                volcano_df.to_csv(output_dir / f"{level}_feature_volcano.csv", index=False)

                fig, ax = plt.subplots(figsize=(12, 10))

                # Color by significance
                sig_threshold = 0.05 / len(volcano_df)  # Bonferroni
                fc_threshold = 0.5

                sig_up = (volcano_df["pvalue"] < sig_threshold) & (volcano_df["fold_change"] > fc_threshold)
                sig_down = (volcano_df["pvalue"] < sig_threshold) & (volcano_df["fold_change"] < -fc_threshold)

                ax.scatter(volcano_df.loc[~(sig_up | sig_down), "fold_change"],
                          volcano_df.loc[~(sig_up | sig_down), "-log10_pvalue"],
                          c="gray", alpha=0.5, s=30, label="Not significant")
                ax.scatter(volcano_df.loc[sig_up, "fold_change"],
                          volcano_df.loc[sig_up, "-log10_pvalue"],
                          c="red", alpha=0.7, s=50, label=f"Up ({sig_up.sum()})")
                ax.scatter(volcano_df.loc[sig_down, "fold_change"],
                          volcano_df.loc[sig_down, "-log10_pvalue"],
                          c="blue", alpha=0.7, s=50, label=f"Down ({sig_down.sum()})")

                # Label top significant features
                top_sig = volcano_df.nlargest(10, "-log10_pvalue")
                texts = []
                for _, row in top_sig.iterrows():
                    texts.append(ax.text(row["fold_change"], row["-log10_pvalue"], row["feature"], fontsize=7))

                if use_adjust_text and texts:
                    try:
                        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))
                    except Exception:
                        pass

                ax.axhline(-np.log10(sig_threshold), color="gray", linestyle="--", alpha=0.5)
                ax.axvline(fc_threshold, color="gray", linestyle="--", alpha=0.3)
                ax.axvline(-fc_threshold, color="gray", linestyle="--", alpha=0.3)

                ax.set_xlabel("Mean Difference (Perturbed - NTC)")
                ax.set_ylabel("-log10(p-value)")
                ax.set_title(f"{level.capitalize()}-Level Feature Volcano: Perturbed vs NTC")
                ax.legend()
                plt.tight_layout()

                save_path = output_dir / f"{level}_feature_volcano.png"
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  - Saved feature volcano plot: {save_path}")

        # --- 7. Per-organelle UMAPs ---
        self._generate_organelle_level_umaps(plot_df, features, level, output_dir, use_adjust_text, label_col)

        # --- 8. Metadata heatmaps on UMAP ---
        self._generate_metadata_heatmaps(plot_df, features, level, output_dir)

        print(f"  --- {level.capitalize()}-level plot suite complete ---")

    def _generate_organelle_level_umaps(
        self,
        plot_df: pd.DataFrame,
        features: pd.DataFrame,
        level: str,
        output_dir: Path,
        use_adjust_text: bool,
        label_col: str,
    ):
        """
        Generate per-organelle UMAPs for guide/gene level analysis.

        Groups features by organelle prefix and generates a separate UMAP for each.
        """
        print(f"\n  --- Generating per-organelle {level}-level UMAPs ---")

        # Group features by organelle prefix
        organelle_features = {}
        for col in features.columns:
            parts = col.split("_")
            if parts[0] == "network":
                if len(parts) > 2:
                    organelle = parts[1]
                else:
                    continue
            else:
                if len(parts) > 1:
                    organelle = parts[0]
                else:
                    continue

            if organelle not in organelle_features:
                organelle_features[organelle] = []
            organelle_features[organelle].append(col)

        organelles = sorted(organelle_features.keys())
        print(f"  - Found {len(organelles)} organelle groups: {organelles}")

        if len(organelles) < 2:
            print("  - Skipping per-organelle UMAPs: need at least 2 organelle groups")
            return

        organelle_dir = output_dir / "per_organelle"
        organelle_dir.mkdir(exist_ok=True, parents=True)

        for organelle in organelles:
            cols = organelle_features[organelle]
            if len(cols) < 3:
                print(f"  - Skipping {organelle}: only {len(cols)} features")
                continue

            org_features = features[cols].copy()
            org_features = org_features.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Check for zero variance
            variances = org_features.var()
            if (variances == 0).all():
                print(f"  - Skipping {organelle}: all features have zero variance")
                continue

            # Remove zero-variance columns
            org_features = org_features.loc[:, variances > 0]
            if org_features.shape[1] < 2:
                print(f"  - Skipping {organelle}: not enough features after filtering")
                continue

            # Run UMAP on organelle-specific features
            print(f"  - Running UMAP for {organelle} ({len(cols)} features)...")
            try:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(org_features)

                if self.use_cuml:
                    try:
                        from cuml.manifold import UMAP as cuMLUMAP
                        reducer = cuMLUMAP(n_neighbors=min(15, len(org_features)-1), min_dist=0.1, n_components=2, random_state=42)
                        org_embedding = reducer.fit_transform(scaled.astype(np.float32))
                    except Exception:
                        reducer = umap.UMAP(n_neighbors=min(15, len(org_features)-1), min_dist=0.1, n_components=2, random_state=42)
                        org_embedding = reducer.fit_transform(scaled)
                else:
                    reducer = umap.UMAP(n_neighbors=min(15, len(org_features)-1), min_dist=0.1, n_components=2, random_state=42)
                    org_embedding = reducer.fit_transform(scaled)

                # Create plot
                fig, ax = plt.subplots(figsize=(12, 10))

                # Color by gene effect if available
                if "gene_effect" in plot_df.columns:
                    gene_effect = pd.to_numeric(plot_df["gene_effect"], errors="coerce")
                    scatter = ax.scatter(
                        org_embedding[:, 0],
                        org_embedding[:, 1],
                        c=gene_effect,
                        cmap="RdBu_r",
                        s=50,
                        alpha=0.7,
                        vmin=-1,
                        vmax=0.5,
                    )
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label("Gene Effect")
                else:
                    ax.scatter(org_embedding[:, 0], org_embedding[:, 1], s=50, alpha=0.7, c="steelblue")

                # Add annotations for outliers
                if level == "gene" and label_col in plot_df.columns:
                    center_x, center_y = org_embedding[:, 0].mean(), org_embedding[:, 1].mean()
                    distances = np.sqrt((org_embedding[:, 0] - center_x)**2 + (org_embedding[:, 1] - center_y)**2)
                    outlier_threshold = np.percentile(distances, 93)
                    outliers_mask = distances > outlier_threshold

                    texts = []
                    for i, (idx, row) in enumerate(plot_df.iterrows()):
                        if outliers_mask[i]:
                            label_text = str(row[label_col])
                            if label_text and label_text != "nan":
                                texts.append(ax.text(org_embedding[i, 0], org_embedding[i, 1], label_text, fontsize=7))

                    if use_adjust_text and texts:
                        try:
                            from adjustText import adjust_text
                            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))
                        except Exception:
                            pass

                ax.set_title(f"{level.capitalize()}-Level UMAP: {organelle.upper()} Features ({len(cols)} features)")
                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
                plt.tight_layout()

                save_path = organelle_dir / f"{level}_umap_{organelle}.png"
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"    - Saved: {save_path.name}")

            except Exception as e:
                print(f"    - Failed for {organelle}: {e}")
                continue

    def _generate_metadata_heatmaps(
        self,
        plot_df: pd.DataFrame,
        features: pd.DataFrame,
        level: str,
        output_dir: Path,
    ):
        """
        Generate heatmaps showing metadata variables on the UMAP embedding.

        Includes: well distribution, radial position, cell size, n_cells, etc.
        """
        print(f"\n  --- Generating {level}-level metadata heatmaps ---")

        heatmap_dir = output_dir / "metadata_heatmaps"
        heatmap_dir.mkdir(exist_ok=True, parents=True)

        umap_coords = plot_df[["umap_1", "umap_2"]].values

        # Define metadata columns to visualize
        # These are aggregated statistics at the guide/gene level
        metadata_configs = [
            # Column name, colormap, title suffix, log_scale
            ("n_cells", "viridis", "Cell Count", True),
            ("gene_effect", "RdBu_r", "Gene Effect (CERES)", False),
        ]

        # Add any mean/median aggregated metadata columns that exist
        potential_metadata = [
            "cell_area_mean", "cell_area_median",
            "cell_perimeter_mean", "cell_perimeter_median",
            "cell_axis_major_length_mean", "cell_axis_minor_length_mean",
            "nuclei_area_mean", "nuclei_area_median",
            "x_global_pheno_mean", "y_global_pheno_mean",
            "x_local_pheno_mean", "y_local_pheno_mean",
            "tile_pheno_mean",
        ]

        for col in potential_metadata:
            if col in plot_df.columns:
                # Determine colormap and scale
                if "area" in col or "length" in col or "perimeter" in col:
                    metadata_configs.append((col, "plasma", col.replace("_", " ").title(), True))
                elif "x_" in col or "y_" in col:
                    metadata_configs.append((col, "coolwarm", col.replace("_", " ").title(), False))
                else:
                    metadata_configs.append((col, "viridis", col.replace("_", " ").title(), False))

        # Also look for organelle-specific size metrics
        for col in features.columns:
            if col.endswith("_area_mean") or col.endswith("_count_mean"):
                if col not in [c[0] for c in metadata_configs]:
                    metadata_configs.append((col, "plasma", col.replace("_", " ").title(), True))

        # Generate heatmap for each metadata variable
        for col, cmap, title_suffix, log_scale in metadata_configs:
            if col not in plot_df.columns and col not in features.columns:
                continue

            # Get values from plot_df or features
            if col in plot_df.columns:
                values = pd.to_numeric(plot_df[col], errors="coerce")
            else:
                values = pd.to_numeric(features[col], errors="coerce")

            if values.isna().all():
                continue

            # Apply log scale if needed
            plot_values = values.copy()
            if log_scale:
                plot_values = np.log10(plot_values.clip(lower=1e-10) + 1)
                title_suffix = f"log10({title_suffix})"

            fig, ax = plt.subplots(figsize=(12, 10))

            # Handle NaN values
            valid_mask = plot_values.notna()
            if valid_mask.sum() < 10:
                plt.close()
                continue

            scatter = ax.scatter(
                plot_df.loc[valid_mask, "umap_1"],
                plot_df.loc[valid_mask, "umap_2"],
                c=plot_values[valid_mask],
                cmap=cmap,
                s=50,
                alpha=0.7,
            )

            # Plot NaN values as gray
            if (~valid_mask).sum() > 0:
                ax.scatter(
                    plot_df.loc[~valid_mask, "umap_1"],
                    plot_df.loc[~valid_mask, "umap_2"],
                    c="lightgray",
                    s=30,
                    alpha=0.3,
                    label="N/A",
                )

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(title_suffix)

            ax.set_title(f"{level.capitalize()}-Level UMAP: {title_suffix}")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            plt.tight_layout()

            # Clean filename
            safe_col = col.replace("/", "_").replace(" ", "_")
            save_path = heatmap_dir / f"{level}_umap_{safe_col}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    - Saved: {save_path.name}")

        # --- Generate well distribution heatmap if well info exists ---
        if "well" in plot_df.columns or any("well" in c.lower() for c in plot_df.columns):
            well_col = "well" if "well" in plot_df.columns else None
            if well_col is None:
                for c in plot_df.columns:
                    if "well" in c.lower():
                        well_col = c
                        break

            if well_col:
                fig, ax = plt.subplots(figsize=(14, 10))

                # Get unique wells and assign colors
                wells = plot_df[well_col].astype(str).unique()
                n_wells = len(wells)
                if n_wells > 1:
                    colors = sns.color_palette("tab20", n_colors=min(20, n_wells))
                    well_palette = {w: colors[i % len(colors)] for i, w in enumerate(sorted(wells))}

                    for well in sorted(wells):
                        mask = plot_df[well_col].astype(str) == well
                        ax.scatter(
                            plot_df.loc[mask, "umap_1"],
                            plot_df.loc[mask, "umap_2"],
                            color=well_palette[well],
                            s=50,
                            alpha=0.6,
                            label=well if n_wells <= 20 else None,
                        )

                    ax.set_title(f"{level.capitalize()}-Level UMAP: Well Distribution ({n_wells} wells)")
                    ax.set_xlabel("UMAP 1")
                    ax.set_ylabel("UMAP 2")
                    if n_wells <= 20:
                        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=1.5, title="Well")
                    plt.tight_layout()

                    save_path = heatmap_dir / f"{level}_umap_well_distribution.png"
                    plt.savefig(save_path, dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"    - Saved: {save_path.name}")

    def _find_optimal_k(self, data: np.ndarray, k_range: range = range(2, 31)) -> int:
        """
        Finds the optimal number of clusters (k) using the elbow method.
        This method is more lenient than silhouette score and tends to find a
        higher number of clusters by identifying the point of maximum curvature
        in the inertia plot.
        """
        # To speed up, if data is too large, use a sample
        if len(data) > 10000:
            print(
                f"  - Subsampling data from {len(data)} to 10000 for optimal k search."
            )
            sample_indices = np.random.choice(data.shape[0], 10000, replace=False)
            data = data[sample_indices]

        print(
            f"  - Searching for optimal k in range {k_range.start}-{k_range.stop-1} using the Elbow method..."
        )
        inertias = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
            print(f"    - k={k}, inertia={kmeans.inertia_:.2f}")

        # --- Find the elbow point programmatically ---
        # We're looking for the point with the maximum distance from a line
        # drawn between the first and last points of the inertia plot.
        p1 = np.array([k_range.start, inertias[0]])
        p2 = np.array([k_range.stop - 1, inertias[-1]])

        distances = []
        for i, k in enumerate(k_range):
            p3 = np.array([k, inertias[i]])
            distance = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(
                p2 - p1
            )
            distances.append(distance)

        elbow_index = np.argmax(distances)
        optimal_k = k_range[elbow_index]

        print(f"  - Optimal k found at elbow point: {optimal_k}")

        # For debugging, let's also save the elbow plot
        plt.figure(figsize=(8, 6))
        plt.plot(k_range, inertias, "bx-")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Inertia")
        plt.title("Elbow Method For Optimal k")
        plt.vlines(
            optimal_k,
            plt.ylim()[0],
            plt.ylim()[1],
            linestyles="--",
            colors="r",
            label=f"Optimal k = {optimal_k}",
        )
        plt.legend()
        save_path = self.graph_output_path / "umap_kmeans_elbow_plot.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  - Saved elbow plot to: {save_path}")

        return optimal_k

    def _calculate_enrichment_for_cluster(self, cluster_id: str, plot_df: pd.DataFrame):
        """
        Calculates gene enrichment for a single cluster. Designed to be called in parallel.
        Returns a dataframe with ALL enriched genes for the cluster, not just the top N.
        """
        in_cluster_mask = plot_df["cluster"] == cluster_id

        genes_in_cluster = plot_df[in_cluster_mask]["gene_name"]
        genes_out_cluster = plot_df[~in_cluster_mask]["gene_name"]

        in_cluster_counts = genes_in_cluster.value_counts()
        out_cluster_counts = genes_out_cluster.value_counts()

        enrichment_results = []

        # Create contingency table for Fisher's exact test
        #        | in_cluster | out_cluster
        # -----------------------------------
        # gene   | a          | b
        # !gene  | c          | d

        for gene, a in in_cluster_counts.items():
            # --- OPTIMIZATION: Skip test if gene count in cluster is very low ---
            # A gene appearing only once is highly unlikely to be statistically significant
            # after multiple testing correction, so we skip the expensive calculation.
            if a < 2:
                continue

            # a = count of gene in cluster
            b = out_cluster_counts.get(gene, 0)

            # c = number of other cells in cluster
            c = len(genes_in_cluster) - a
            # d = number of other cells outside cluster
            d = len(genes_out_cluster) - b

            odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
            enrichment_results.append(
                {"gene_name": gene, "p_value": p_value, "odds_ratio": odds_ratio}
            )

        if enrichment_results:
            enrichment_df = pd.DataFrame(enrichment_results)
            enrichment_df["p_adj"] = fdrcorrection(enrichment_df["p_value"])[1]
            # No longer pre-filter to top 10, return all results
            enrichment_df["cluster"] = cluster_id
            return enrichment_df.sort_values("p_adj")

        return None

    def _generate_umap_suite(
        self,
        features: pd.DataFrame,
        cell_df: pd.DataFrame,
        plot_prefix_base: str,
        output_dir: Path,
        interactive_output_dir: Path,
    ):
        """
        Generates a full suite of UMAPs: one for all features, and one for each organelle.
        Also triggers the generation of corresponding interactive plots.

        Parameters
        ----------
        features : pd.DataFrame
            The feature matrix (e.g., original or regressed) to use for UMAPs.
        cell_df : pd.DataFrame
            The corresponding cell metadata dataframe, indexed correctly to match features.
        plot_prefix_base : str
            A base name for the plots (e.g., 'all_features' or 'regressed_features').
        output_dir : Path
            The directory where the static UMAP plots will be saved.
        interactive_output_dir : Path
            The directory where the interactive UMAP plots will be saved.
        """
        # --- 1. UMAP on all features ---
        print(f"\n--- Generating UMAP for '{plot_prefix_base}' ---")

        # NEW: Create a dedicated subdirectory for this suite of plots
        suite_output_dir = output_dir / plot_prefix_base
        suite_output_dir.mkdir(exist_ok=True, parents=True)

        plot_df, full_enrichment_df = self._perform_umap_and_generate_plots(
            features, cell_df, plot_prefix_base, suite_output_dir
        )

        if plot_df is None:
            print(
                "  - Halting further UMAP plots as main embedding could not be generated."
            )
            return

        # --- Save UMAP embedding to AnnData (only for main 'all_features' UMAP) ---
        if plot_prefix_base == "all_features" and "cell" in self.adata:
            # Extract embedding from plot_df
            embedding = plot_df[["umap_1", "umap_2"]].values

            # Save embedding to AnnData
            adata = self.adata["cell"]
            adata.obsm["X_umap"] = embedding.astype(np.float32)

            # Save ALL clustering methods to AnnData obs
            clustering_methods = ["hdbscan", "kmeans", "leiden"] if self.cluster_algo == "all" else [self.cluster_algo]
            for method in clustering_methods:
                cluster_col = f"cluster_{method}"
                if cluster_col in plot_df.columns:
                    # Convert from string 'c0', 'c1', etc. to int
                    clusters = plot_df[cluster_col].str.replace("c", "").astype(int).values
                    adata.obs[cluster_col] = clusters
                    n_clusters = len([c for c in np.unique(clusters) if c != -1])
                    print(f"  - Saved {cluster_col} to AnnData ({n_clusters} clusters)")

            # Save the updated AnnData back to disk
            h5ad_path = self.analysis_path / f"{self.experiment}_cell_features.h5ad"
            print(f"  - Saving embeddings and clusters to {h5ad_path.name}...")
            adata.write_h5ad(h5ad_path)
            print(f"  - Successfully saved UMAP embedding and all cluster assignments to cell-level AnnData.")

        # --- Save data for the interactive Dash dashboard, if requested ---
        if not self.skip_interactive_plots:
            print(
                f"  - Saving data for Interactive Dashboard for '{plot_prefix_base}'..."
            )
            interactive_plot_df = self._prepare_interactive_plot_data(plot_df, features)

            # Add enrichment info to the hover data for relevant clusters
            if full_enrichment_df is not None and not full_enrichment_df.empty:
                # Create a summarized string of top genes for each cluster
                top_genes = full_enrichment_df.groupby("cluster").head(5)
                enrichment_text = (
                    top_genes.groupby("cluster")["gene_name"]
                    .apply(lambda x: ", ".join(x))
                    .rename("top_genes")
                )
                # Merge this back into the main dataframe for the dashboard
                interactive_plot_df = interactive_plot_df.merge(
                    enrichment_text, on="cluster", how="left"
                )
                interactive_plot_df["top_genes"] = interactive_plot_df[
                    "top_genes"
                ].fillna("N/A")

            save_path = (
                interactive_output_dir / f"interactive_data_{plot_prefix_base}.parquet"
            )
            interactive_plot_df.to_parquet(save_path)
            print(f"  - Interactive plot data saved to: {save_path}")
            print(
                f"  - To view, run: python -m organelle_profiler.extra.interactive_dashboard {save_path}"
            )
        else:
            print("  - Skipping interactive plot generation as requested.")

        # --- 2. UMAPs on organelle-specific feature subsets ---
        print(f"\n--- Generating separate organelle UMAPs for '{plot_prefix_base}' ---")

        # Identify Organelles from the columns of the provided features dataframe
        organelles = set()
        for col in features.columns:
            parts = col.split("_")
            if parts[0] == "network":
                if len(parts) > 2:
                    organelles.add(parts[1])
            else:
                if len(parts) > 1:
                    organelles.add(parts[0])

        print(
            f"  - Found {len(organelles)} organelle groups to analyze: {sorted(list(organelles))}"
        )

        for organelle in sorted(list(organelles)):

            base_name = "regressed" if "regressed" in plot_prefix_base else "original"
            plot_prefix = f"{base_name}_organelle_{organelle}"

            # NEW: Create a dedicated subdirectory for this organelle's plots
            organelle_output_dir = output_dir / f"organelle_{organelle}"
            organelle_output_dir.mkdir(exist_ok=True, parents=True)

            organelle_cols = [
                col
                for col in features.columns
                if col.startswith(f"{organelle}_")
                or col.startswith(f"network_{organelle}_")
            ]

            if len(organelle_cols) < 2:
                print(
                    f"  - Skipping {organelle}, not enough features found ({len(organelle_cols)})."
                )
                continue

            organelle_features = features[organelle_cols].copy()

            # --- FIX: Re-run low-variance feature removal on the specific organelle subset ---
            min_variance_threshold = 1e-4
            variances = organelle_features.var(numeric_only=True)
            low_variance_cols = variances[variances < min_variance_threshold].index

            if not low_variance_cols.empty:
                print(
                    f"  - INFO [{organelle}]: Removing {len(low_variance_cols)} low-variance features from subset."
                )
                organelle_features.drop(columns=low_variance_cols, inplace=True)

            if organelle_features.shape[1] < 2:
                print(
                    f"  - Skipping {organelle}, not enough features remaining after removing low-variance columns."
                )
                continue

            # --- NEW FIX for RAFT error: Deduplicate within the feature subset ---
            # This is crucial because subsetting columns can create new duplicate rows.
            n_original_org = len(organelle_features)
            # We must drop duplicates from the organelle-specific feature set
            organelle_features.drop_duplicates(inplace=True)
            # Then, we filter the main cell dataframe to only include the cells that remain
            cell_df_org = cell_df.loc[organelle_features.index].copy()

            if len(organelle_features) < n_original_org:
                print(
                    f"  - INFO [{organelle}]: Removed {n_original_org - len(organelle_features)} duplicate rows from feature subset."
                )

            if organelle_features.shape[1] < 2:
                print(
                    f"  - Skipping {organelle}, not enough features remaining after removing zero-variance columns."
                )
                continue

            # Call the helper to generate static plots
            plot_df_org, full_enrichment_df_org = self._perform_umap_and_generate_plots(
                organelle_features,
                cell_df_org,  # Pass the correctly filtered cell_df
                plot_prefix,
                organelle_output_dir,  # Pass the new dedicated directory
            )

            # --- Generate the interactive plot for this organelle UMAP view, if requested ---
            if not self.skip_interactive_plots:
                if (
                    plot_df_org is not None
                ):  # Check plot_df_org, enrichment might be empty
                    print(
                        f"  - Saving data for Interactive Dashboard for '{plot_prefix}'..."
                    )

                    interactive_plot_df_org = self._prepare_interactive_plot_data(
                        plot_df_org, organelle_features
                    )

                    # Add enrichment info if it exists
                    if (
                        full_enrichment_df_org is not None
                        and not full_enrichment_df_org.empty
                    ):
                        top_genes_org = full_enrichment_df_org.groupby("cluster").head(
                            5
                        )
                        enrichment_text_org = (
                            top_genes_org.groupby("cluster")["gene_name"]
                            .apply(lambda x: ", ".join(x))
                            .rename("top_genes")
                        )
                        interactive_plot_df_org = interactive_plot_df_org.merge(
                            enrichment_text_org, on="cluster", how="left"
                        )
                        interactive_plot_df_org["top_genes"] = interactive_plot_df_org[
                            "top_genes"
                        ].fillna("N/A")

                    save_path_org = (
                        interactive_output_dir
                        / f"interactive_data_{plot_prefix}.parquet"
                    )
                    interactive_plot_df_org.to_parquet(save_path_org)
                    print(f"  - Interactive plot data saved to: {save_path_org}")
                    print(
                        f"  - To view, run: python -m organelle_profiler.extra.interactive_dashboard {save_path_org}"
                    )

    def _generate_gene_highlight_umaps(
        self,
        plot_df,
        plot_prefix,
        output_dir,
        genes_to_plot: list,
        subfolder_name: str,
        enrichment_df: pd.DataFrame = None,
    ):
        """
        Generates and saves UMAP plots highlighting one specific gene at a time.
        NEW: Also annotates clusters where the gene is significantly enriched.
        """
        if not genes_to_plot:
            print(
                f"  - Skipping single-gene highlight plots for '{subfolder_name}' (no genes selected)."
            )
            return

        highlight_output_dir = output_dir / subfolder_name
        highlight_output_dir.mkdir(exist_ok=True)

        # --- NEW: Get cluster centers once for all plots ---
        cluster_centers = plot_df.groupby("cluster")[["umap_1", "umap_2"]].median()

        # --- NEW: Get plot center for text alignment ---
        plot_center_x = plot_df["umap_1"].mean()
        plot_center_y = plot_df["umap_2"].mean()

        # --- NEW: Get cell counts for each cluster ---
        cluster_counts = plot_df["cluster"].value_counts()

        print(
            f"  - Generating {len(genes_to_plot)} single-gene highlight plots in '{subfolder_name}'..."
        )
        for gene in genes_to_plot:
            fig, ax = plt.subplots(figsize=(12, 10))

            # Plot all points in a faint gray as the background
            ax.scatter(
                plot_df["umap_1"],
                plot_df["umap_2"],
                color="lightgray",
                s=5,
                alpha=0.3,
                rasterized=True,
            )

            # Highlight the cells for the specific gene in bright red
            highlight_mask = plot_df["gene_name"] == gene
            if highlight_mask.any():
                ax.scatter(
                    plot_df.loc[highlight_mask, "umap_1"],
                    plot_df.loc[highlight_mask, "umap_2"],
                    color="red",
                    s=15,  # Make highlighted points slightly larger
                    label=f"{gene} ({highlight_mask.sum()} cells)",
                )

            # --- NEW: Add enrichment annotations ---
            if enrichment_df is not None and not enrichment_df.empty:
                gene_enrichment = enrichment_df[enrichment_df["gene_name"] == gene]
                # Filter for clusters with significant enrichment
                significant_clusters = gene_enrichment[gene_enrichment["p_adj"] < 0.05]

                print(
                    f"    - [{subfolder_name}] For gene '{gene}', found {len(significant_clusters)} significantly enriched clusters to annotate."
                )

                texts_to_adjust = []
                for _, row in significant_clusters.iterrows():
                    cluster_id = row["cluster"]
                    odds_ratio = row["odds_ratio"]
                    if cluster_id in cluster_centers.index:
                        center = cluster_centers.loc[cluster_id]
                        cell_count = cluster_counts.get(cluster_id, 0)

                        # --- NEW: Determine alignment for better initial positioning ---
                        ha = "left" if center["umap_1"] > plot_center_x else "right"
                        va = "bottom" if center["umap_2"] > plot_center_y else "top"

                        texts_to_adjust.append(
                            ax.text(
                                x=center["umap_1"],
                                y=center["umap_2"],
                                s=f"{odds_ratio:.1f}x\nn={cell_count}",  # The annotation text
                                fontdict={"size": 10, "weight": "bold"},
                                color="black",
                                ha=ha,  # Set horizontal alignment
                                va=va,  # Set vertical alignment
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    fc="gray",
                                    ec="black",
                                    alpha=0.2,
                                ),
                            )
                        )

                if texts_to_adjust:
                    try:
                        from adjustText import adjust_text

                        adjust_text(
                            texts_to_adjust,
                            ax=ax,
                            arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
                        )
                    except ImportError:
                        print(
                            f"  - WARNING: Cannot adjust text for gene '{gene}'. Run 'pip install adjustText' for better labels."
                        )

            ax.set_title(f"UMAP Highlighting Gene: {gene}", fontsize=18)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.legend()
            ax.set_aspect("equal")

            save_path = (
                highlight_output_dir / f"umap_{plot_prefix}_highlight_{gene}.png"
            )
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    def _prepare_interactive_plot_data(
        self, plot_df: pd.DataFrame, features: pd.DataFrame, n_top_features: int = 25
    ) -> pd.DataFrame:
        """
        Prepares a dataframe for the interactive plot by selecting key metadata
        and the most variable features. This version does NOT downsample.
        """
        print(f"  - Preparing data for interactive plot...")
        # 1. Define and select essential metadata columns that exist in the dataframe
        essential_cols = [
            col
            for col in [
                "cell_id",
                "gene_name",
                "umap_1",
                "umap_2",
                "cluster",
                "gene_group",
                "gene_effect",
                "well",
                "barcode",
                "sgRNA",
                "effect",
                "NCBI",
                "index",
            ]
            if col in plot_df.columns
        ]

        # 2. Select the top N most variable feature columns
        top_feature_cols = []
        if features.shape[1] > n_top_features:
            print(
                f"  - Selecting top {n_top_features} (out of {features.shape[1]}) most variable features for interactive plot."
            )
            top_feature_cols = features.var().nlargest(n_top_features).index.tolist()
        else:
            top_feature_cols = features.columns.tolist()

        # 3. Combine the essential metadata with the raw feature values for hover info
        # We need to join the original features back onto the plot_df
        cols_to_keep = essential_cols + top_feature_cols

        # Ensure we don't have duplicate columns if a feature was somehow in essential_cols
        final_cols = plot_df[essential_cols].copy()

        # Join the selected raw features. plot_df and features share the same index.
        final_cols = final_cols.join(features[top_feature_cols])

        print(
            f"  - Final dataframe for interactive plot has {len(final_cols.columns)} columns and {len(final_cols)} rows."
        )
        return final_cols

    def _calculate_radial_positions(self, cell_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates and returns a dataframe with well and tile radial positions.
        This version calculates the geometric center of the grid of cells, rather
        than the centroid of cell positions.
        """
        df = cell_df.copy()

        # --- Calculate Well Radial Position ---
        pos_cols_well = ["x_global_pheno", "y_global_pheno", "well"]
        if all(col in df.columns for col in pos_cols_well):
            print(
                "  - Calculating well radial positions based on geometric center of cell grid..."
            )
            # Ensure coordinate columns are numeric (may be categorical from AnnData)
            for coord_col in ["x_global_pheno", "y_global_pheno"]:
                df[coord_col] = pd.to_numeric(df[coord_col], errors="coerce")

            radial_pos = pd.Series(index=df.index, dtype=float, name="well_radial_pos")
            valid_pos_idx = df[pos_cols_well].dropna().index
            if not valid_pos_idx.empty:
                pos_df = df.loc[valid_pos_idx].copy()

                # For each well, find the bounding box of all cells
                well_bounds = pos_df.groupby("well", observed=True).agg(
                    x_min=("x_global_pheno", "min"),
                    x_max=("x_global_pheno", "max"),
                    y_min=("y_global_pheno", "min"),
                    y_max=("y_global_pheno", "max"),
                )

                # Calculate the geometric center from the bounding box
                well_bounds["center_x"] = (
                    well_bounds["x_min"] + well_bounds["x_max"]
                ) / 2
                well_bounds["center_y"] = (
                    well_bounds["y_min"] + well_bounds["y_max"]
                ) / 2

                # Map the calculated centers back to each cell (convert to float arrays)
                centers_x = pos_df["well"].map(well_bounds["center_x"]).astype(float).values
                centers_y = pos_df["well"].map(well_bounds["center_y"]).astype(float).values

                # Calculate Euclidean distance from each cell to its well's geometric center
                radial_values = np.sqrt(
                    (pos_df["x_global_pheno"].values - centers_x) ** 2
                    + (pos_df["y_global_pheno"].values - centers_y) ** 2
                )
                radial_pos.loc[valid_pos_idx] = radial_values

            df["well_radial_pos"] = radial_pos
        else:
            print(
                "  - Warning: Skipping well radial position calculation. Required columns not found."
            )

        # --- Calculate Tile Radial Position ---
        pos_cols_tile = ["x_local_pheno", "y_local_pheno", "tile_pheno"]
        if all(col in df.columns for col in pos_cols_tile):
            print(
                "  - Calculating tile radial positions based on geometric center of cell grid..."
            )
            # Ensure coordinate columns are numeric (may be categorical from AnnData)
            for coord_col in ["x_local_pheno", "y_local_pheno", "tile_pheno"]:
                if coord_col in df.columns:
                    df[coord_col] = pd.to_numeric(df[coord_col], errors="coerce")

            radial_pos = pd.Series(index=df.index, dtype=float, name="tile_radial_pos")
            valid_pos_idx = df[pos_cols_tile].dropna().index
            if not valid_pos_idx.empty:
                pos_df = df.loc[valid_pos_idx].copy()

                # For each tile, find the bounding box of all cells
                tile_bounds = pos_df.groupby("tile_pheno", observed=True).agg(
                    x_min=("x_local_pheno", "min"),
                    x_max=("x_local_pheno", "max"),
                    y_min=("y_local_pheno", "min"),
                    y_max=("y_local_pheno", "max"),
                )

                # Calculate the geometric center from the bounding box
                tile_bounds["center_x"] = (
                    tile_bounds["x_min"] + tile_bounds["x_max"]
                ) / 2
                tile_bounds["center_y"] = (
                    tile_bounds["y_min"] + tile_bounds["y_max"]
                ) / 2

                # Map the calculated centers back to each cell (convert to float arrays)
                centers_x = pos_df["tile_pheno"].map(tile_bounds["center_x"]).astype(float).values
                centers_y = pos_df["tile_pheno"].map(tile_bounds["center_y"]).astype(float).values

                # Calculate Euclidean distance from each cell to its tile's geometric center
                radial_values = np.sqrt(
                    (pos_df["x_local_pheno"].values - centers_x) ** 2
                    + (pos_df["y_local_pheno"].values - centers_y) ** 2
                )
                radial_pos.loc[valid_pos_idx] = radial_values

            df["tile_radial_pos"] = radial_pos
        else:
            print(
                "  - Warning: Skipping tile radial position calculation. Required columns not found."
            )

        return df

    def run_regression_analysis(self, features: pd.DataFrame, cell_df: pd.DataFrame):
        """
        Regresses out spatial covariates (well and tile radial distance) and
        reruns the full UMAP suite on the corrected features.
        """
        regression_name = "regressed_spatial_covariates"
        output_dir = self.graph_output_path / regression_name
        output_dir.mkdir(exist_ok=True)
        print(f"Saving regressed analysis plots to: {output_dir}")

        # 1. Calculate radial positions and add them to the cell_df
        cell_df_with_pos = self._calculate_radial_positions(cell_df)

        # 2. Prepare covariates for regression
        covariates = ["well_radial_pos", "tile_radial_pos"]
        if not all(c in cell_df_with_pos.columns for c in covariates):
            print(
                "  - ERROR: Cannot perform regression, spatial covariates could not be calculated."
            )
            return

        covariate_df = cell_df_with_pos[covariates]

        # 3. Perform regression
        print("  - Regressing out spatial covariates from features...")
        regressed_features = features.copy()

        # Only use cells that have valid covariate data for fitting the model
        valid_covariate_mask = covariate_df.notna().all(axis=1)

        if valid_covariate_mask.sum() < 2:
            print(
                "  - ERROR: Not enough cells with valid spatial data to perform regression."
            )
            return

        X_fit = covariate_df[valid_covariate_mask]

        model = LinearRegression()

        # Regress each feature column against the covariates
        for feature_col in tqdm(features.columns, desc="Regressing features"):
            y_fit = features.loc[valid_covariate_mask, feature_col]

            # Ensure there are no NaNs in the target feature for the fitting data
            if y_fit.isnull().any():
                # If there are NaNs, fit only on the non-NaN part
                valid_y_mask = y_fit.notna()
                if valid_y_mask.sum() > 2:
                    model.fit(X_fit[valid_y_mask], y_fit[valid_y_mask])
                else:
                    continue  # Not enough data
            else:
                model.fit(X_fit, y_fit)

            # The "artifact" is the part of the feature value predicted by the spatial position
            predicted_artifact = model.predict(X_fit)

            # The corrected feature is the residual (original value minus the predicted artifact)
            regressed_features.loc[valid_covariate_mask, feature_col] = (
                y_fit - predicted_artifact
            )

        print("  - Regression complete.")

        # 4. Generate UMAPs with the new, corrected features
        self._generate_umap_suite(
            regressed_features,
            cell_df,  # Pass the original cell_df for annotations
            "regressed_spatial",
            output_dir,
            self.interactive_output_path,  # Pass the central interactive plot directory
        )

        return regressed_features

    def plot_cell_umap(self):
        """
        DEPRECATED: This method is now a wrapper. The logic has been moved to
        _prepare_features_for_umap and _generate_umap_suite. It is kept for
        potential direct calls but should not be used in the main `run` flow.
        """
        print(
            "--- plot_cell_umap is deprecated. Using new UMAP suite generation flow. ---"
        )
        features, cell_df_for_umap = self._prepare_features_for_umap()
        if features is not None:
            self._generate_umap_suite(
                features,
                cell_df_for_umap,
                "all_features",
                self.graph_output_path,
                self.interactive_output_path,
            )

    def plot_organelle_umaps(self):
        """
        DEPRECATED: This logic is now part of _generate_umap_suite. Kept for
        compatibility but should not be used in the main `run` flow.
        """
        print(
            "--- plot_organelle_umaps is deprecated. UMAPs are generated via _generate_umap_suite. ---"
        )
        # The new flow consolidates this, so this function is now a no-op to avoid duplication.
        pass

    def plot_gene_clustermap(self):
        """Generates a clustermap of mean features per gene."""
        print("Generating gene-level feature clustermap...")
        df = self.data["gene"]

        # Use only mean values for clarity
        mean_cols = [col for col in df.columns if col.endswith("_mean")]
        plot_df = df[["gene_name"] + mean_cols].set_index("gene_name")

        # Standardize data (z-score by column) for better color mapping.
        # We explicitly check for a standard deviation of zero to prevent division
        # by zero, which would create non-finite values and crash the clustering.
        # Columns with no variance are mapped to zero.
        scaled_df = plot_df.apply(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0, axis=0
        )
        # Final check to ensure no NaN/inf values are passed to the plotter
        scaled_df.replace([np.inf, -np.inf], 0, inplace=True)
        scaled_df.fillna(0, inplace=True)

        # Before clustering, remove rows AND columns with zero variance, as they
        # cannot be clustered by correlation and will cause an error.

        # Filter rows (genes)
        row_stds = scaled_df.std(axis=1)
        original_gene_count = len(scaled_df)
        scaled_df = scaled_df[row_stds > 0]

        if len(scaled_df) < original_gene_count:
            print(
                f"  - Warning: Removed {original_gene_count - len(scaled_df)} genes with zero variance after scaling. These cannot be clustered by correlation."
            )

        # Filter columns (features)
        col_stds = scaled_df.std(axis=0)
        original_feature_count = len(scaled_df.columns)
        scaled_df = scaled_df.loc[:, col_stds > 0]

        if len(scaled_df.columns) < original_feature_count:
            print(
                f"  - Warning: Removed {original_feature_count - len(scaled_df.columns)} features with zero variance after scaling. These cannot be clustered by correlation."
            )

        if len(scaled_df) < 2 or len(scaled_df.columns) < 2:
            print(
                "  - Warning: Not enough data left to generate clustermap after removing zero-variance data."
            )
            return

        sns.clustermap(
            scaled_df,
            figsize=(15, max(10, len(plot_df.index) * 0.5)),
            cmap="vlag",
            metric="correlation",
        )
        save_path = self.graph_output_path / "gene_feature_clustermap.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_guide_consistency(self):
        """Plots boxplots to show guide consistency for top variance genes."""
        print("Generating guide consistency plots...")
        cell_df = self.data["cell"]

        # Find top 5 genes with highest variance in cell_area
        top_variance_genes = (
            cell_df.groupby("gene_name")["cell_area"].var().nlargest(5).index.tolist()
        )

        plot_df = cell_df[cell_df["gene_name"].isin(top_variance_genes)]

        if plot_df.empty:
            print(
                "Warning: Could not generate guide consistency plot, no data for top variance genes."
            )
            return

        g = sns.catplot(
            data=plot_df,
            x="barcode",
            y="cell_area",
            col="gene_name",
            kind="box",
            col_wrap=3,
            sharex=False,
            height=5,
        )
        g.fig.suptitle(
            "Guide RNA Consistency for Cell Area (Top 5 Variant Genes)", y=1.03
        )
        g.set_xticklabels(rotation=90)
        save_path = self.graph_output_path / "guide_consistency_boxplot.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_organelle_distributions(self):
        """
        Plots histograms of key features for each organelle type, aggregated at the cell level.
        This shows the distribution of the mean property value per cell, split by
        Non-Targeting Controls (NTC) and perturbed cells.
        """
        print("Generating cell-level organelle property distributions...")
        for organelle, df in self.data["object"].items():
            print(f"\n-> Processing organelle: {organelle}")
            print(f"  - Original object-level dataframe has {len(df)} rows.")

            if "cell_id" not in df.columns:
                print(
                    f"Warning: Skipping {organelle}: 'cell_id' column not found in object features."
                )
                continue

            # Aggregate at the cell level by taking the mean of key properties
            cell_level_df = (
                df.groupby("cell_id")[["area", "aspect_ratio"]].mean().reset_index()
            )
            print(f"  - Aggregated to {len(cell_level_df)} cells.")

            # Get gene names to distinguish NTC from perturbed
            cell_metadata = self.data["cell"][["cell_id", "gene_name"]].copy()

            # --- FIX: Ensure dtypes match before merging ---
            cell_level_df["cell_id"] = cell_level_df["cell_id"].astype(str)
            cell_metadata["cell_id"] = cell_metadata["cell_id"].astype(str)

            plot_df = pd.merge(cell_level_df, cell_metadata, on="cell_id", how="left")
            plot_df["gene_group"] = np.where(
                plot_df["gene_name"].astype(str) == "0", "NTC", "Perturbed"
            )
            print(
                f"  - Split into NTC ({len(plot_df[plot_df.gene_group == 'NTC'])}) and Perturbed ({len(plot_df[plot_df.gene_group == 'Perturbed'])}) groups."
            )

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(
                f"Distribution of Mean {organelle.capitalize()} Object Properties per Cell (NTC vs Perturbed)",
                fontsize=16,
            )

            sns.histplot(
                data=plot_df,
                x="area",
                hue="gene_group",
                bins=50,
                ax=axes[0],
                element="step",
                stat="density",
                common_norm=False,
            )
            axes[0].set_title("Distribution of Mean Area per Cell")

            sns.histplot(
                data=plot_df,
                x="aspect_ratio",
                hue="gene_group",
                bins=50,
                ax=axes[1],
                element="step",
                stat="density",
                common_norm=False,
            )
            axes[1].set_title("Distribution of Mean Aspect Ratio per Cell")

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            save_path = (
                self.graph_output_path
                / f"{organelle}_cell_level_object_distribution.png"
            )
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"  - Finished processing {organelle} object features.")

        # --- NEW: Generate plots for network features ---
        print("\nGenerating cell-level network property distributions...")
        for organelle, df in self.data["network"].items():
            print(f"\n-> Processing network: {organelle}")
            print(f"  - Original branch-level dataframe has {len(df)} rows.")

            if "cell_id" not in df.columns:
                print(
                    f"Warning: Skipping {organelle}: 'cell_id' column not found in network features."
                )
                continue

            network_features_to_agg = [
                col
                for col in ["branch_length", "branch_thickness", "tortuosity"]
                if col in df.columns
            ]
            if not network_features_to_agg:
                print(
                    f"Warning: Skipping {organelle}: No standard network feature columns found."
                )
                continue

            cell_level_df = (
                df.groupby("cell_id")[network_features_to_agg].mean().reset_index()
            )
            print(f"  - Aggregated to {len(cell_level_df)} cells.")

            cell_metadata = self.data["cell"][["cell_id", "gene_name"]].copy()
            cell_level_df["cell_id"] = cell_level_df["cell_id"].astype(str)
            cell_metadata["cell_id"] = cell_metadata["cell_id"].astype(str)
            plot_df = pd.merge(cell_level_df, cell_metadata, on="cell_id", how="left")
            plot_df["gene_group"] = np.where(
                plot_df["gene_name"].astype(str) == "0", "NTC", "Perturbed"
            )

            num_features = len(network_features_to_agg)
            fig, axes = plt.subplots(1, num_features, figsize=(7 * num_features, 6))
            if num_features == 1:
                axes = [axes]
            fig.suptitle(
                f"Distribution of Mean {organelle.capitalize()} Network Properties per Cell",
                fontsize=16,
            )

            for i, feature in enumerate(network_features_to_agg):
                sns.histplot(
                    data=plot_df,
                    x=feature,
                    hue="gene_group",
                    bins=50,
                    ax=axes[i],
                    element="step",
                    stat="density",
                    common_norm=False,
                )
                axes[i].set_title(f"Distribution of Mean {feature}")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            save_path = (
                self.graph_output_path
                / f"{organelle}_cell_level_network_distribution.png"
            )
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"  - Finished processing network features for {organelle}.")

    def _plot_broken_barh(
        self, data: pd.Series, title: str, xlabel: str, save_filename: str
    ):
        """
        Generates a horizontal bar plot with a broken axis for handling outliers.
        If the max value is not a significant outlier, it produces a standard plot.
        """
        if data.empty:
            print(f"  - Warning: No data provided for plotting '{title}'. Skipping.")
            return

        max_val = data.max()
        p95 = data.quantile(0.95)

        # Heuristic: Use a standard plot if max value is not much larger than 95th percentile
        if max_val < p95 * 2 or len(data) <= 1:
            plt.figure(figsize=(10, 12))
            sns.barplot(x=data.values, y=data.index, orient="h", palette="rocket")
            plt.title(title, fontsize=16)
            plt.xlabel(xlabel)
            plt.tight_layout()
            save_path = self.graph_output_path / save_filename
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Saved: {save_path}")
            return

        # --- Create a broken axis plot ---
        # The axis is split into two subplots with an 80/20 width ratio
        fig, (ax1, ax2) = plt.subplots(
            1, 2, sharey=True, figsize=(14, 12), gridspec_kw={"width_ratios": [4, 1]}
        )
        fig.subplots_adjust(wspace=0.05)

        # Plot the same data on both axes
        sns.barplot(x=data.values, y=data.index, orient="h", palette="rocket", ax=ax1)
        sns.barplot(x=data.values, y=data.index, orient="h", palette="rocket", ax=ax2)

        # Set x-axis limits for the two parts of the plot
        cutoff = p95 * 1.1
        ax1.set_xlim(0, cutoff)
        ax2.set_xlim(max_val * 0.99, max_val * 1.01)

        # Hide the spines and ticks that connect the two plots
        ax1.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.yaxis.set_ticks_position("none")
        ax2.tick_params(labelleft=False)
        ax2.set_ylabel("")

        # Add diagonal "break" lines
        d = 0.015
        kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (-d, +d), **kwargs)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

        fig.suptitle(title, fontsize=18)
        ax1.set_xlabel(xlabel)
        ax2.set_xlabel("")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = self.graph_output_path / save_filename
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved broken-axis plot: {save_path}")

    def plot_ntc_comparison(self, features: pd.DataFrame, cell_df: pd.DataFrame):
        """
        Plots features with the most robust difference from NTCs using a robust Z-score.
        """
        print("Generating NTC comparison plot (using robust Z-score method)...")

        feature_cols = features.columns.tolist()

        # --- 1. Combine features and metadata, then split data ---
        # The 'cell_df' already contains the feature columns. The join is redundant and causes an error.
        full_df = cell_df

        ntc_identifier_patterns = ["ntc", "non-targeting"]
        ntc_mask = (
            full_df["gene_name"]
            .astype(str)
            .str.contains("|".join(ntc_identifier_patterns), case=False, regex=True)
        )
        ntc_df = full_df[ntc_mask]

        if ntc_df.empty:
            print(
                "\n---\nWarning: Could not generate NTC comparison plot: No Non-Targeting Control (NTC) genes found.\n---"
            )
            return

        pert_df = full_df[~ntc_mask]
        if pert_df.empty:
            print("Warning: No perturbed genes found to compare against NTCs.")
            return

        # --- 2. Calculate NTC mean/std and filter features with no variance ---
        print(
            f"  - INFO: NTC statistics are derived from {len(ntc_df)} cells from genes: {ntc_df['gene_name'].unique().tolist()}"
        )
        ntc_means = ntc_df[feature_cols].mean()
        ntc_stds = ntc_df[feature_cols].std()

        # This check is still useful for the NTC sub-population, even after global filtering
        valid_stds_mask = ntc_stds > 1e-6
        if not valid_stds_mask.all():
            removed_cols = ntc_stds[~valid_stds_mask].index.tolist()
            print(
                f"  - INFO: Removing {len(removed_cols)} features with zero/low variance in NTC population."
            )

            feature_cols = ntc_stds[valid_stds_mask].index.tolist()
            if not feature_cols:
                print(
                    "  - Warning: No features with variance remaining after filtering. Cannot generate plot."
                )
                return
            ntc_means = ntc_means[feature_cols]
            ntc_stds = ntc_stds[feature_cols]

        # --- 3. Calculate Z-scores for all perturbed genes ---
        pert_gene_means = pert_df.groupby("gene_name")[feature_cols].mean()
        z_scores = (pert_gene_means - ntc_means) / ntc_stds

        # --- 4. Find features with the largest effects ---
        mean_abs_z = z_scores.abs().mean().sort_values(ascending=False)
        top_20_features = mean_abs_z.head(20)

        if top_20_features.empty or top_20_features.iloc[0] == 0:
            print(
                "\n---\nWarning: Could not generate NTC comparison plot: No features showed significant deviation from NTCs.\n---"
            )
            return

        # --- 5. Plotting ---
        self._plot_broken_barh(
            data=top_20_features,
            title="Top 20 Features with Strongest Deviation from NTC (Z-score)",
            xlabel="Mean Absolute Z-score vs. NTC Population",
            save_filename="ntc_feature_comparison_zscore.png",
        )

    def plot_ntc_comparison_fold_change(
        self, features: pd.DataFrame, cell_df: pd.DataFrame, top_n=20
    ):
        """
        Calculates and plots the log2 fold change for features relative to NTCs.
        """
        print("Generating NTC comparison plot (log2 fold change)...")

        feature_cols = features.columns.tolist()

        # --- 1. Combine features and metadata, then split data ---
        # The 'cell_df' already contains the feature columns. The join is redundant and causes an error.
        full_df = cell_df

        ntc_identifier_patterns = ["ntc", "non-targeting"]
        ntc_mask = (
            full_df["gene_name"]
            .astype(str)
            .str.contains("|".join(ntc_identifier_patterns), case=False, regex=True)
        )
        ntc_df = full_df[ntc_mask]

        if ntc_df.empty:
            print(
                "\n---\nWarning: Could not generate NTC fold change plot: No NTC genes found.\n---"
            )
            return

        pert_df = full_df[~ntc_mask]
        if pert_df.empty:
            print("Warning: No perturbed genes found for fold change comparison.")
            return

        # --- 2. Calculate fold change ---
        ntc_means = ntc_df[feature_cols].mean()
        # Add a small epsilon to avoid division by zero for features where NTC mean is 0
        ntc_means_nozero = ntc_means.replace(0, 1e-9)

        pert_gene_means = pert_df.groupby("gene_name")[feature_cols].mean()
        fold_change = pert_gene_means.div(ntc_means_nozero, axis=1)

        # --- 3. Find features with the largest effects ---
        mean_abs_log_fold_change = np.log2(fold_change).abs().mean()
        # Filter out infinite values that can result from log2(0) before sorting
        finite_fc = mean_abs_log_fold_change[np.isfinite(mean_abs_log_fold_change)]
        top_n_fc = finite_fc.sort_values(ascending=False).head(top_n)

        if top_n_fc.empty or top_n_fc.iloc[0] == 0:
            print(
                "\n---\nWarning: Could not generate NTC fold change plot: No features showed significant change.\n---"
            )
            return

        # --- 4. Plotting ---
        self._plot_broken_barh(
            data=top_n_fc,
            title=f"Top {top_n} Features with Strongest Deviation from NTC (Fold Change)",
            xlabel="Mean Absolute log2(Fold Change vs NTC)",
            save_filename="ntc_feature_comparison_fold_change.png",
        )

    def plot_volcano_plots(
        self,
        features: pd.DataFrame,
        cell_df: pd.DataFrame,
        top_n_features: int = 30,
        output_subdir: str = "volcano",
    ):
        """
        Generates volcano plots for the most differentially expressed features.
        The x-axis represents log2 fold change, and the y-axis represents
        statistical significance, mirroring common conventions.

        Parameters
        ----------
        features : pd.DataFrame
            The feature matrix (original or regressed) to use.
        cell_df : pd.DataFrame
            The corresponding cell metadata dataframe.
        top_n_features : int
            The number of top features to generate plots for.
        output_subdir : str
            The name of the subdirectory to save the plots in.
        """
        print(
            f"\n--- Generating Volcano Plots for top {top_n_features} features (saving to '{output_subdir}') ---"
        )

        # --- Create subdirectory for volcano plots ---
        volcano_output_path = self.graph_output_path / output_subdir
        volcano_output_path.mkdir(exist_ok=True)

        # --- 1. Use the pre-filtered features passed into the function ---
        feature_cols = features.columns.tolist()

        # --- 2. Identify NTC and Perturbed populations from the cell_df ---
        # The 'cell_df' already contains the feature columns. The join is redundant and causes an error.
        full_df = cell_df

        ntc_identifier_patterns = ["ntc", "non-targeting"]
        ntc_mask = (
            full_df["gene_name"]
            .astype(str)
            .str.contains("|".join(ntc_identifier_patterns), case=False, regex=True)
        )
        ntc_df = full_df[ntc_mask]
        pert_df = full_df[~ntc_mask]

        if ntc_df.empty or pert_df.empty:
            print(
                "  - Warning: Cannot generate volcano plots. NTC or perturbed gene data is missing."
            )
            return

        # Use the provided features for NTC and perturbed calculations
        ntc_feature_df = ntc_df[feature_cols]
        pert_df_with_features = pert_df

        # --- 3. Find top features to plot based on Z-score deviation ---
        ntc_means = ntc_feature_df[feature_cols].mean()
        ntc_stds = ntc_feature_df[feature_cols].std()

        valid_stds_mask = ntc_stds > 1e-6
        if not valid_stds_mask.all():
            feature_cols = ntc_stds[valid_stds_mask].index.tolist()
            if not feature_cols:
                print(
                    "  - Warning: No features with variance remaining after filtering. Cannot generate plots."
                )
                return
            ntc_means = ntc_means[feature_cols]
            ntc_stds = ntc_stds[feature_cols]

        pert_gene_means = pert_df_with_features.groupby("gene_name")[
            feature_cols
        ].mean()
        z_scores = (pert_gene_means - ntc_means) / ntc_stds
        mean_abs_z = z_scores.abs().mean().sort_values(ascending=False)

        top_features_to_plot = mean_abs_z.head(top_n_features).index.tolist()
        print(
            f"  - Selected top {len(top_features_to_plot)} features based on mean absolute Z-score."
        )

        # --- 4. Calculate p-values and log2 fold change for volcano plot data ---
        print(
            "  - Calculating p-values (t-test) and log2 fold change for each gene against NTCs..."
        )

        ntc_means_nozero = ntc_means.replace(0, 1e-9)
        fold_change = pert_gene_means[feature_cols].div(
            ntc_means_nozero[feature_cols], axis=1
        )
        log2_fold_change = np.log2(fold_change)

        volcano_data = []
        for feature in top_features_to_plot:
            ntc_values = ntc_feature_df[feature]
            for gene_name, group in pert_df_with_features.groupby("gene_name"):
                pert_values = group[feature]

                # Welch's t-test for significance
                if len(pert_values.dropna()) > 1 and len(ntc_values.dropna()) > 1:
                    _, p_value = ttest_ind(
                        pert_values, ntc_values, equal_var=False, nan_policy="omit"
                    )
                else:
                    p_value = 1.0  # Not enough data for a test

                # Retrieve the pre-calculated log2fc for this gene-feature combo
                log2fc_value = (
                    log2_fold_change.loc[gene_name, feature]
                    if gene_name in log2_fold_change.index
                    else np.nan
                )

                volcano_data.append(
                    {
                        "gene_name": gene_name,
                        "feature": feature,
                        "p_value": p_value,
                        "log2_fold_change": log2fc_value,
                    }
                )

        if not volcano_data:
            print(
                "  - Warning: No data generated for volcano plots after statistical tests."
            )
            return

        volcano_df = pd.DataFrame(volcano_data)
        volcano_df["-log10p"] = -np.log10(volcano_df["p_value"])

        # Replace infinite values from p_value=0 or log2(0)
        max_logp = volcano_df.loc[np.isfinite(volcano_df["-log10p"]), "-log10p"].max()
        if pd.notna(max_logp):
            volcano_df.replace([np.inf], max_logp * 1.1, inplace=True)
        volcano_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        volcano_df.dropna(subset=["log2_fold_change", "-log10p"], inplace=True)

        # --- 5. Generate a plot for each feature ---
        print("  - Generating individual volcano plots...")
        p_val_threshold = 0.05
        log2fc_threshold = 1.0  # Standard threshold for 2-fold change

        for feature in top_features_to_plot:
            plot_df = volcano_df[volcano_df["feature"] == feature].copy()

            plt.figure(figsize=(12, 9))

            # Define conditions for coloring
            cond_down = (plot_df["p_value"] < p_val_threshold) & (
                plot_df["log2_fold_change"] < -log2fc_threshold
            )
            cond_up = (plot_df["p_value"] < p_val_threshold) & (
                plot_df["log2_fold_change"] > log2fc_threshold
            )

            # Plot data points
            plt.scatter(
                plot_df["log2_fold_change"],
                plot_df["-log10p"],
                c="grey",
                alpha=0.6,
                label="Not Significant",
            )
            plt.scatter(
                plot_df.loc[cond_down, "log2_fold_change"],
                plot_df.loc[cond_down, "-log10p"],
                c="cornflowerblue",
                alpha=0.8,
                label=f"Downregulated",
            )
            plt.scatter(
                plot_df.loc[cond_up, "log2_fold_change"],
                plot_df.loc[cond_up, "-log10p"],
                c="red",
                alpha=0.8,
                label=f"Upregulated",
            )

            # --- Label top genes ---
            # Sort by significance (p-value) to label the most significant hits
            plot_df["abs_log2fc"] = plot_df["log2_fold_change"].abs()

            genes_to_label = pd.concat(
                [
                    plot_df[cond_down]
                    .sort_values(by="p_value", ascending=True)
                    .head(10),
                    plot_df[cond_up].sort_values(by="p_value", ascending=True).head(10),
                ]
            )

            for _, row in genes_to_label.iterrows():
                plt.text(
                    row["log2_fold_change"],
                    row["-log10p"],
                    str(row["gene_name"]),
                    fontsize=9,
                )

            # Add threshold lines
            plt.axhline(-np.log10(p_val_threshold), color="black", linestyle="--", lw=1)
            plt.axvline(log2fc_threshold, color="black", linestyle="--", lw=1)
            plt.axvline(-log2fc_threshold, color="black", linestyle="--", lw=1)

            plt.title(f"Volcano Plot for: {feature}", fontsize=16)
            plt.xlabel("log2(Fold Change) vs NTC Population")
            plt.ylabel("-log10(p-value)")
            plt.legend()
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)

            save_path = volcano_output_path / f"volcano_{feature}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

        print(
            f"  - Saved {len(top_features_to_plot)} volcano plots to: {volcano_output_path}"
        )

    def plot_radial_distance_correlations(
        self, features: pd.DataFrame, cell_df: pd.DataFrame
    ):
        """
        Calculates and plots the features most correlated with cell radial
        position within the well and the tile.
        """
        print("\n--- Generating Radial Distance Correlation Plots ---")
        # Calculate positions using the passed cell_df
        cell_df_with_pos = self._calculate_radial_positions(cell_df)

        # Use the pre-filtered features passed to the function
        feature_cols = features.columns.tolist()

        # Helper function for analysis and plotting
        def _analyze_and_plot_corr(df, target_feature, title, save_filename):
            if target_feature not in df.columns or df[target_feature].notna().sum() < 2:
                print(
                    f"  - Skipping correlation analysis for '{target_feature}' as it's not available or has insufficient data."
                )
                return

            # Use only the features and the target, dropping rows where the target is NaN
            # The 'df' (cell_df) already contains the feature columns. The join is redundant.
            analysis_df = df[feature_cols + [target_feature]].copy()
            analysis_df.dropna(subset=[target_feature], inplace=True)

            print(f"\n--- Analyzing correlations with {target_feature} ---")

            # --- OPTIMIZATION: Calculate correlation feature by feature ---
            # The original df.corr(method='spearman') is slow because it computes a
            # full (M x M) matrix. We only need the correlation of each of the M
            # features against a single target vector. This is much faster.
            print(
                f"  - Calculating Spearman correlation for {len(feature_cols)} features..."
            )

            correlations = {}
            target_values = analysis_df[target_feature]
            for col in feature_cols:
                # spearmanr returns correlation and p-value; we only need the correlation.
                # It handles NaNs gracefully by default.
                corr, _ = spearmanr(analysis_df[col], target_values)
                if not np.isnan(corr):
                    correlations[col] = corr

            correlations = pd.Series(correlations)
            top_corr = correlations.abs().sort_values(ascending=False).head(20)

            if top_corr.empty:
                print("  - No significant correlations found.")
                return

            print(f"Top 20 features most correlated with {target_feature}:")
            print(top_corr)

            # Get the actual correlation values (with sign) for plotting
            top_corr_values = correlations.loc[top_corr.index]

            plt.figure(figsize=(12, 8))
            sns.barplot(
                x=top_corr_values.values,
                y=top_corr_values.index,
                orient="h",
                palette="coolwarm",
            )
            plt.title(title, fontsize=16)
            plt.xlabel(f"Spearman Correlation with {target_feature}")
            plt.xlim(-1, 1)  # Ensure consistent x-axis for comparison
            plt.tight_layout()
            save_path = self.graph_output_path / save_filename
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Saved correlation plot: {save_path}")

        # --- Run Analysis for Both Distances ---
        _analyze_and_plot_corr(
            cell_df_with_pos,
            "well_radial_pos",
            "Top Features Correlated with Well Radial Position",
            "well_radial_pos_correlation.png",
        )
        _analyze_and_plot_corr(
            cell_df_with_pos,
            "tile_radial_pos",
            "Top Features Correlated with Tile Radial Position",
            "tile_radial_pos_correlation.png",
        )

    def plot_radial_feature_drift(
        self,
        features: pd.DataFrame,
        cell_df: pd.DataFrame,
        output_dir: Path,
        file_prefix: str,
    ) -> tuple[float, float]:
        """
        Analyzes and plots the drift of feature magnitudes against radial distance
        to visualize systematic spatial effects on the overall cell phenotype.
        Returns the percentage of cells after the well and tile inflection points.
        """
        print("\n--- Generating Radial Feature Drift Plots ---")

        # 1. Scale features and calculate a "feature deviation score" for each cell
        scaled_features = StandardScaler().fit_transform(features)
        feature_drift_score = np.mean(np.abs(scaled_features), axis=1)

        # 2. Prepare dataframe for analysis
        analysis_df = cell_df.copy()
        analysis_df["feature_drift_score"] = feature_drift_score
        analysis_df = self._calculate_radial_positions(analysis_df)

        # 3. Helper function to generate a plot for a given position type
        def _create_drift_plot(df, pos_type, title_base, save_filename_base) -> float:
            if pos_type not in df.columns or df[pos_type].isna().all():
                print(
                    f"  - Skipping drift plot for '{pos_type}': column not found or all NaN."
                )
                return None

            plot_data = df[[pos_type, "feature_drift_score"]].dropna()
            if len(plot_data) < 100:
                print(
                    f"  - Skipping drift plot for '{pos_type}': insufficient data points ({len(plot_data)})."
                )
                return None

            # Bin data by radial distance and calculate stats
            num_bins = 50
            plot_data["radial_bin"] = pd.cut(plot_data[pos_type], bins=num_bins)
            binned_analysis = (
                plot_data.groupby("radial_bin", observed=False)
                .agg(
                    mean_drift=("feature_drift_score", "mean"),
                    cell_count=("feature_drift_score", "count"),
                )
                .reset_index()
            )

            # Calculate cell density
            binned_analysis["r_inner"] = [b.left for b in binned_analysis["radial_bin"]]
            binned_analysis["r_outer"] = [
                b.right for b in binned_analysis["radial_bin"]
            ]
            bin_area = np.pi * (
                binned_analysis["r_outer"] ** 2 - binned_analysis["r_inner"] ** 2
            )
            bin_area = bin_area.replace(0, np.nan)
            binned_analysis["cell_density"] = binned_analysis["cell_count"] / bin_area

            binned_analysis["radial_midpoint"] = [
                b.mid for b in binned_analysis["radial_bin"]
            ]
            binned_analysis.dropna(
                subset=["mean_drift", "cell_density", "radial_midpoint"], inplace=True
            )

            if len(binned_analysis) < 5:  # Need enough points for segmented regression
                print(
                    f"  - Skipping drift plot for '{pos_type}': not enough valid bins after processing."
                )
                return None

            # --- Normalize drift score relative to the centermost bin ---
            y_axis_label = (
                "Mean Absolute Scaled Feature Value (Drift Score)"  # Default label
            )
            center_bin_drift = binned_analysis.sort_values("radial_midpoint").iloc[0][
                "mean_drift"
            ]
            if center_bin_drift > 1e-9:  # Avoid division by zero
                print(
                    f"  - Normalizing drift plot for '{pos_type}' relative to center drift value of {center_bin_drift:.3f}."
                )
                normalized_drift = binned_analysis["mean_drift"] / center_bin_drift
                y_axis_label = "Normalized Feature Drift (Relative to Center)"
            else:
                print(
                    f"  - Skipping normalization for '{pos_type}' as center drift is near zero."
                )
                normalized_drift = binned_analysis["mean_drift"]

            # --- Find breakpoint using the selected method ---
            x_data = binned_analysis["radial_midpoint"].values
            y_data = normalized_drift.values

            breakpoint_idx = None  # Initialize to handle fallback cases

            if self.drift_method == "three_segment_regression":

                def _find_two_breakpoints(x_vals, y_vals):
                    min_error = np.inf
                    best_breakpoints = (-1, -1)
                    n = len(x_vals)
                    # Iterate through all possible pairs of split points
                    # Leave at least 2 points for each regression
                    for i in range(2, n - 4):
                        for j in range(i + 2, n - 2):
                            x1, y1 = x_vals[:i], y_vals[:i]
                            x2, y2 = x_vals[i:j], y_vals[i:j]
                            x3, y3 = x_vals[j:], y_vals[j:]
                            model1 = LinearRegression().fit(x1.reshape(-1, 1), y1)
                            error1 = np.sum(
                                (y1 - model1.predict(x1.reshape(-1, 1))) ** 2
                            )
                            model2 = LinearRegression().fit(x2.reshape(-1, 1), y2)
                            error2 = np.sum(
                                (y2 - model2.predict(x2.reshape(-1, 1))) ** 2
                            )
                            model3 = LinearRegression().fit(x3.reshape(-1, 1), y3)
                            error3 = np.sum(
                                (y3 - model3.predict(x3.reshape(-1, 1))) ** 2
                            )
                            total_error = error1 + error2 + error3
                            if total_error < min_error:
                                min_error = total_error
                                best_breakpoints = (i, j)
                    return best_breakpoints

                breakpoint1_idx, breakpoint2_idx = _find_two_breakpoints(x_data, y_data)

                if breakpoint1_idx == -1:
                    print(
                        f"  - Could not find suitable breakpoints for '{pos_type}'. Skipping regression lines."
                    )
                    breakpoint_x = x_data.mean()  # Fallback
                else:
                    breakpoint_idx = breakpoint1_idx  # Flag to indicate success

                    # Fit models for all three segments regardless of which breakpoint is chosen for the v-line
                    x1_fit, y1_fit = x_data[:breakpoint1_idx], y_data[:breakpoint1_idx]
                    model1_fit = LinearRegression().fit(x1_fit.reshape(-1, 1), y1_fit)
                    x2_fit, y2_fit = (
                        x_data[breakpoint1_idx:breakpoint2_idx],
                        y_data[breakpoint1_idx:breakpoint2_idx],
                    )
                    model2_fit = LinearRegression().fit(x2_fit.reshape(-1, 1), y2_fit)
                    x3_fit, y3_fit = x_data[breakpoint2_idx:], y_data[breakpoint2_idx:]
                    model3_fit = LinearRegression().fit(x3_fit.reshape(-1, 1), y3_fit)

                    # --- NEW: Conditionally choose the inflection point based on position ---
                    x_midpoint = (x_data[0] + x_data[-1]) / 2
                    if x_data[breakpoint1_idx] < x_midpoint:
                        print(
                            f"  - First breakpoint is before 50% mark. Using second breakpoint as inflection."
                        )
                        breakpoint_x = x_data[breakpoint2_idx]
                    else:
                        print(
                            f"  - First breakpoint is after 50% mark. Using first breakpoint as inflection."
                        )
                        breakpoint_x = x_data[breakpoint1_idx]

                v_line_label = f"Inflection Point at {breakpoint_x:.2f}"
                info_text_pre = "Cells Before Inflection"
                info_text_post = "Cells After Inflection"

            elif self.drift_method == "segmented_regression":

                def _find_breakpoint(x_vals, y_vals):
                    min_error = np.inf
                    best_breakpoint_idx = -1
                    # Iterate through all possible split points (leave at least 2 points for each regression)
                    for i in range(2, len(x_vals) - 2):
                        x1, y1 = x_vals[:i], y_vals[:i]
                        x2, y2 = x_vals[i:], y_vals[i:]
                        model1 = LinearRegression().fit(x1.reshape(-1, 1), y1)
                        error1 = np.sum((y1 - model1.predict(x1.reshape(-1, 1))) ** 2)
                        model2 = LinearRegression().fit(x2.reshape(-1, 1), y2)
                        error2 = np.sum((y2 - model2.predict(x2.reshape(-1, 1))) ** 2)
                        if error1 + error2 < min_error:
                            min_error = error1 + error2
                            best_breakpoint_idx = i
                    return best_breakpoint_idx

                breakpoint_idx = _find_breakpoint(x_data, y_data)

                if breakpoint_idx is None:
                    print(
                        f"  - Could not find a suitable breakpoint for '{pos_type}'. Skipping regression lines."
                    )
                    breakpoint_x = x_data.mean()
                else:
                    breakpoint_x = x_data[breakpoint_idx]
                    x1_fit, y1_fit = x_data[:breakpoint_idx], y_data[:breakpoint_idx]
                    model1_fit = LinearRegression().fit(x1_fit.reshape(-1, 1), y1_fit)
                    x2_fit, y2_fit = x_data[breakpoint_idx:], y_data[breakpoint_idx:]
                    model2_fit = LinearRegression().fit(x2_fit.reshape(-1, 1), y2_fit)

                v_line_label = f"Breakpoint at {breakpoint_x:.2f}"
                info_text_pre = "Cells Before Breakpoint"
                info_text_post = "Cells After Breakpoint"

            elif self.drift_method == "second_derivative":
                smoothed = lowess(y_data, x_data, frac=0.5)
                second_derivative = np.gradient(
                    np.gradient(smoothed[:, 1], smoothed[:, 0]), smoothed[:, 0]
                )
                breakpoint_idx = np.argmax(
                    second_derivative
                )  # Using this name for consistency
                breakpoint_x, _ = (
                    smoothed[breakpoint_idx, 0],
                    smoothed[breakpoint_idx, 1],
                )
                v_line_label = f"Point of Max Acceleration at {breakpoint_x:.2f}"
                info_text_pre = "Cells Before Inflection"
                info_text_post = "Cells After Inflection"

            else:
                raise ValueError(f"Unknown drift_method: {self.drift_method}")

            # --- Calculate cell counts for annotation ---
            cells_before = plot_data[plot_data[pos_type] <= breakpoint_x].shape[0]
            cells_after = plot_data[plot_data[pos_type] > breakpoint_x].shape[0]

            # --- Loop to create both plot types (Density and Count) ---
            for plot_type in ["density", "count"]:

                fig, ax1 = plt.subplots(figsize=(14, 8))

                ax1.plot(
                    x_data,
                    y_data,
                    "o-",
                    alpha=0.6,
                    label="Mean Feature Drift per Bin",
                    zorder=10,
                )

                # --- Plot based on method ---
                if (
                    self.drift_method == "three_segment_regression"
                    and breakpoint_idx is not None
                ):
                    ax1.plot(
                        x1_fit,
                        model1_fit.predict(x1_fit.reshape(-1, 1)),
                        color="red",
                        linestyle="--",
                        linewidth=2.5,
                        label="Segment 1 Fit",
                        zorder=20,
                    )
                    ax1.plot(
                        x2_fit,
                        model2_fit.predict(x2_fit.reshape(-1, 1)),
                        color="cyan",
                        linestyle="--",
                        linewidth=2.5,
                        label="Segment 2 Fit",
                        zorder=20,
                    )
                    ax1.plot(
                        x3_fit,
                        model3_fit.predict(x3_fit.reshape(-1, 1)),
                        color="magenta",
                        linestyle="--",
                        linewidth=2.5,
                        label="Segment 3 Fit",
                        zorder=20,
                    )
                elif (
                    self.drift_method == "segmented_regression"
                    and breakpoint_idx is not None
                ):
                    ax1.plot(
                        x1_fit,
                        model1_fit.predict(x1_fit.reshape(-1, 1)),
                        color="red",
                        linestyle="--",
                        linewidth=2.5,
                        label="Segment 1 Fit",
                        zorder=20,
                    )
                    ax1.plot(
                        x2_fit,
                        model2_fit.predict(x2_fit.reshape(-1, 1)),
                        color="cyan",
                        linestyle="--",
                        linewidth=2.5,
                        label="Segment 2 Fit",
                        zorder=20,
                    )
                elif self.drift_method == "second_derivative":
                    ax1.plot(
                        smoothed[:, 0],
                        smoothed[:, 1],
                        color="red",
                        linewidth=2.5,
                        label="LOWESS Smoothed Trend",
                        zorder=20,
                    )
                    ax1.scatter(
                        breakpoint_x,
                        y_data[breakpoint_idx],
                        s=150,
                        c="black",
                        marker="X",
                        zorder=30,
                        label="Inflection Point",
                    )

                ax1.axvline(
                    breakpoint_x,
                    color="black",
                    linestyle="--",
                    label=v_line_label,
                    zorder=15,
                )
                ax1.set_xlabel(
                    f'Radial Distance from {pos_type.split("_")[0].capitalize()} Center'
                )
                ax1.set_ylabel(y_axis_label)
                ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

                ax2 = ax1.twinx()
                if plot_type == "density":
                    ax2.bar(
                        binned_analysis["radial_midpoint"],
                        binned_analysis["cell_density"],
                        width=(binned_analysis["radial_midpoint"].max() / num_bins)
                        * 0.9,
                        alpha=0.2,
                        color="gray",
                        label="Cell Density",
                    )
                    ax2.set_ylabel("Cell Density (Cells / Pixel²)", color="gray")
                    title = f"{title_base} ({file_prefix}, vs. Density)"
                    save_filename = f"{file_prefix}_{save_filename_base}_density.png"
                else:  # plot_type == 'count'
                    ax2.bar(
                        binned_analysis["radial_midpoint"],
                        binned_analysis["cell_count"],
                        width=(binned_analysis["radial_midpoint"].max() / num_bins)
                        * 0.9,
                        alpha=0.2,
                        color="gray",
                        label="Cell Count",
                    )
                    ax2.set_ylabel("Cell Count per Bin", color="gray")
                    title = f"{title_base} ({file_prefix}, vs. Count)"
                    save_filename = f"{file_prefix}_{save_filename_base}_count.png"

                ax2.tick_params(axis="y", labelcolor="gray")

                # --- NEW: Ensure ax1 and its annotations are drawn on top of ax2's bars ---
                ax1.set_zorder(ax2.get_zorder() + 1)  # Bring ax1 to the front
                ax1.patch.set_visible(
                    False
                )  # Make ax1's background transparent so ax2 is visible

                fig.suptitle(title, fontsize=16)
                handles, labels = ax1.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()

                # --- NEW: Add cell count stats as a title to the legend ---
                total_cells = cells_before + cells_after
                percent_before = (
                    100 * cells_before / total_cells if total_cells > 0 else 0
                )
                percent_after = (
                    100 * cells_after / total_cells if total_cells > 0 else 0
                )
                info_text = (
                    f"{info_text_pre}: {cells_before:,} ({percent_before:.1f}%)\n"
                    f"{info_text_post}:  {cells_after:,} ({percent_after:.1f}%)"
                )

                leg = fig.legend(
                    handles=handles + handles2,
                    labels=labels + labels2,
                    loc="upper right",
                    bbox_to_anchor=(0.9, 0.9),
                )
                leg.set_title(info_text, prop={"size": "small", "weight": "bold"})

                plt.tight_layout(rect=[0, 0, 1, 0.96])
                save_path = output_dir / save_filename
                plt.savefig(save_path, dpi=300)
                plt.close(fig)
                print(f"Saved drift plot: {save_path}")

            return percent_after

        # 4. Run for both well and tile position
        well_percent_cutoff = _create_drift_plot(
            analysis_df,
            "well_radial_pos",
            "Feature Drift vs. Well Radial Position",
            "well_radial_feature_drift",
        )
        tile_percent_cutoff = _create_drift_plot(
            analysis_df,
            "tile_radial_pos",
            "Feature Drift vs. Tile Radial Position",
            "tile_radial_feature_drift",
        )

        return well_percent_cutoff, tile_percent_cutoff

    def plot_feature_drift_heatmap(
        self,
        features: pd.DataFrame,
        cell_df: pd.DataFrame,
        output_dir: Path,
        file_prefix: str,
    ):
        """
        Generates a heatmap of the mean absolute feature value (drift score)
        for each tile across each well, similar to a confluency map.
        """
        print("\n--- Generating Feature Drift Heatmap ---")

        # 1. Scale features and calculate "feature deviation score"
        scaled_features = StandardScaler().fit_transform(features)
        feature_drift_score = np.mean(np.abs(scaled_features), axis=1)

        # 2. Prepare dataframe for analysis
        analysis_df = cell_df.copy()
        analysis_df["feature_drift_score"] = feature_drift_score

        # Check for required columns
        required_cols = ["well", "x_global_pheno", "y_global_pheno"]
        if not all(col in analysis_df.columns for col in required_cols):
            print(
                f"  - Warning: Skipping feature drift heatmap. Required columns ({', '.join(required_cols)}) not found."
            )
            return

        # Need to import split_into_tiles
        try:
            from ops_utils.io.tiling import split_into_tiles
        except ImportError:
            print(
                "  - ERROR: Cannot generate heatmap. `split_into_tiles` function not found. Please ensure `ops_utils` is available."
            )
            return

        wells = sorted(analysis_df["well"].unique())
        num_wells = len(wells)

        plate_drift_scores = {}
        well_indices = {}
        grid_size = 30  # As in confluency plot

        for well in wells:
            well_df = analysis_df[analysis_df["well"] == well].copy()
            if well_df.empty:
                continue

            # Shape is (rows, cols), which corresponds to (y, x)
            shape = (
                int(well_df["y_global_pheno"].max()) + 1,
                int(well_df["x_global_pheno"].max()) + 1,
            )
            tile_list, indx = split_into_tiles(shape, grid_size, 0)

            well_drift_scores = []
            for tile in tile_list:
                # Per `metrics.py`, split_into_tiles returns (row_min, row_max, col_min, col_max)
                row_min, row_max, col_min, col_max = tile

                # y_global_pheno corresponds to rows, x_global_pheno to columns
                tile_cells = well_df[
                    (well_df["y_global_pheno"] >= row_min)
                    & (well_df["y_global_pheno"] < row_max)
                    & (well_df["x_global_pheno"] >= col_min)
                    & (well_df["x_global_pheno"] < col_max)
                ]

                mean_drift = (
                    tile_cells["feature_drift_score"].mean()
                    if not tile_cells.empty
                    else np.nan
                )
                well_drift_scores.append(mean_drift)

            plate_drift_scores[well] = well_drift_scores
            well_indices[well] = indx

        # --- NEW: Calculate a global normalization factor from the center tiles of all wells ---
        center_tile_scores = []
        grid_center_coord = (grid_size // 2, grid_size // 2)

        for well in wells:
            indx = well_indices.get(well)
            scores = plate_drift_scores.get(well)
            if not indx or not scores:
                continue

            # Find the index of the tile closest to the geometric center of the grid
            distances = [
                np.sqrt(
                    (i - grid_center_coord[0]) ** 2 + (j - grid_center_coord[1]) ** 2
                )
                for i, j in indx
            ]
            center_tile_k = np.argmin(distances)

            center_score = scores[center_tile_k]
            if not np.isnan(center_score):
                center_tile_scores.append(center_score)

        normalization_factor = (
            np.mean(center_tile_scores) if center_tile_scores else 1.0
        )
        cbar_label = "Mean Absolute Feature Drift Score"
        if normalization_factor > 1e-9:
            print(
                f"  - Normalizing well heatmaps by average center tile drift: {normalization_factor:.3f}"
            )
            cbar_label = "Normalized Feature Drift (Relative to Avg. Center)"
        else:
            print(
                "  - Skipping normalization for well heatmaps, normalization factor is invalid or zero."
            )
            normalization_factor = 1.0

        # 4. Plotting
        fig, axes = plt.subplots(
            1, num_wells, figsize=(5 * num_wells, 5), squeeze=False
        )
        ax_flat = axes.flatten()

        all_scores_raw = [
            score
            for well_scores in plate_drift_scores.values()
            for score in well_scores
            if not np.isnan(score)
        ]
        all_scores_normalized = [s / normalization_factor for s in all_scores_raw]

        if not all_scores_normalized:
            print("  - No data to plot for feature drift heatmap.")
            plt.close(fig)
            return

        vmin = np.percentile(all_scores_normalized, 5)
        vmax = np.percentile(all_scores_normalized, 95)

        for i, well in enumerate(wells):
            scores = plate_drift_scores.get(well)
            indx = well_indices.get(well)
            if not scores or not indx:
                continue

            indx_i = [a[0] for a in indx]
            indx_j = [a[1] for a in indx]

            normalized_scores = [s / normalization_factor for s in scores]

            out = np.full((grid_size, grid_size), np.nan)
            out[indx_i, indx_j] = normalized_scores

            ax = ax_flat[i]
            im = ax.imshow(out, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(well)

        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.045, pad=0.04)
        cbar.set_label(cbar_label)
        fig.suptitle("Feature Drift Heatmap Across Plate", fontsize=16)

        save_path = output_dir / f"{file_prefix}_feature_drift_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved feature drift heatmap: {save_path}")

    def plot_organelle_comparison(self):
        """
        Generates violin plots to compare key metrics across different organelles,
        using the cell-level aggregated data. Each data point in the plot represents
        the mean value for a specific feature within a single cell.
        """
        print("Generating organelle comparison plots from cell-level summary data...")

        cell_df = self.data.get("cell")
        if cell_df is None:
            print(
                "Warning: Skipping organelle comparison: cell_summary_features.csv not found."
            )
            return

        # Identify all aggregated feature columns
        feature_cols = [
            col for col in cell_df.columns if "_mean" in col and ("cell_" not in col)
        ]

        if not feature_cols:
            print("Warning: No aggregated organelle feature columns found to compare.")
            return

        # Melt the dataframe from wide to long format for easier plotting
        id_vars = ["cell_id", "gene_name"]  # Keep these columns as identifiers
        melted_df = cell_df.melt(
            id_vars=id_vars,
            value_vars=feature_cols,
            var_name="feature",
            value_name="value",
        )

        # Create a new column to group genes into NTC and Perturbed
        melted_df["gene_group"] = np.where(
            melted_df["gene_name"] == "0", "NTC", "Perturbed"
        )

        # Extract organelle and metric name from the feature column name
        def parse_feature(feature_name):
            parts = feature_name.split("_")
            if parts[0] == "network":
                organelle = parts[1]
                metric = "_".join(parts[2:])
            else:
                organelle = parts[0]
                metric = "_".join(parts[1:])
            return organelle, metric

        melted_df[["organelle", "metric"]] = melted_df["feature"].apply(
            lambda x: pd.Series(parse_feature(x))
        )

        # --- Generate Plots ---
        unique_metrics = melted_df["metric"].unique()

        fig, axes = plt.subplots(
            len(unique_metrics), 1, figsize=(12, 6 * len(unique_metrics)), sharex=True
        )
        if len(unique_metrics) == 1:
            axes = [axes]

        fig.suptitle(
            "Comparison of Mean Organelle Features per Cell (NTC vs. Perturbed)",
            fontsize=18,
            y=1.02,
        )

        for ax, metric in zip(axes, unique_metrics):
            plot_data = melted_df[melted_df["metric"] == metric]
            sns.violinplot(
                data=plot_data,
                x="organelle",
                y="value",
                hue="gene_group",
                ax=ax,
                inner="box",
                palette="viridis",
                split=True,
            )
            ax.set_title(f"Distribution of '{metric}' per Cell", fontsize=14)
            ax.set_xlabel("")
            ax.set_ylabel("Mean Value per Cell")
            if "area" in metric or "length" in metric:
                ax.set_yscale("log")
                ax.set_ylabel(f"Mean Value per Cell (log scale)")
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        save_path = self.graph_output_path / "organelle_cell_level_comparison.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_average_tile_drift_heatmap(
        self,
        features: pd.DataFrame,
        cell_df: pd.DataFrame,
        output_dir: Path,
        file_prefix: str,
    ):
        """
        Generates a heatmap of the average feature drift score across a composite
        "average tile" to visualize consistent intra-tile spatial patterns.
        """
        print("\n--- Generating Average Tile Feature Drift Heatmap ---")

        # 1. Scale features and calculate "feature deviation score"
        scaled_features = StandardScaler().fit_transform(features)
        feature_drift_score = np.mean(np.abs(scaled_features), axis=1)

        # 2. Prepare dataframe for analysis
        analysis_df = cell_df.copy()
        analysis_df["feature_drift_score"] = feature_drift_score

        # Check for required columns for local tile position
        required_cols = ["x_local_pheno", "y_local_pheno"]
        if not all(col in analysis_df.columns for col in required_cols):
            print(
                f"  - Warning: Skipping average tile heatmap. Required columns ({', '.join(required_cols)}) not found."
            )
            return

        # Drop cells with no local position data
        analysis_df.dropna(subset=required_cols, inplace=True)
        if analysis_df.empty:
            print(
                "  - No cells with local position data found. Skipping average tile heatmap."
            )
            return

        # 3. Define the grid for the average tile
        bin_size = 25  # As requested, 25 pixel bins
        max_x = int(analysis_df["x_local_pheno"].max())
        max_y = int(analysis_df["y_local_pheno"].max())

        x_bins = np.arange(0, max_x + bin_size, bin_size)
        y_bins = np.arange(0, max_y + bin_size, bin_size)

        # 4. Bin cells into the grid and calculate mean drift score per bin
        analysis_df["x_bin"] = pd.cut(
            analysis_df["x_local_pheno"], bins=x_bins, labels=False, right=False
        )
        analysis_df["y_bin"] = pd.cut(
            analysis_df["y_local_pheno"], bins=y_bins, labels=False, right=False
        )

        # Group by the bins and calculate the mean score
        binned_drift = (
            analysis_df.groupby(["y_bin", "x_bin"])["feature_drift_score"]
            .mean()
            .unstack()
        )

        # --- NEW: Normalize by the value in the center of the tile ---
        cbar_label = "Mean Absolute Feature Drift Score"
        if not binned_drift.empty:
            center_y_idx = binned_drift.shape[0] // 2
            center_x_idx = binned_drift.shape[1] // 2
            center_drift_value = binned_drift.iloc[center_y_idx, center_x_idx]

            if pd.notna(center_drift_value) and center_drift_value > 1e-9:
                print(
                    f"  - Normalizing average tile heatmap by center value: {center_drift_value:.3f}"
                )
                binned_drift = binned_drift / center_drift_value
                cbar_label = "Normalized Feature Drift (Relative to Center)"
            else:
                print(
                    "  - Skipping normalization for average tile heatmap, center value is invalid or zero."
                )

        # 5. Plotting
        plt.figure(figsize=(10, 10))

        # Use percentile for color limits to handle outliers
        all_scores = binned_drift.values.flatten()
        all_scores = all_scores[~np.isnan(all_scores)]
        if len(all_scores) == 0:
            print("  - No data to plot for average tile heatmap.")
            plt.close()
            return

        vmin = np.percentile(all_scores, 5)
        vmax = np.percentile(all_scores, 95)

        im = plt.imshow(
            binned_drift, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax
        )

        plt.title("Average Feature Drift Across All Tiles", fontsize=16)
        plt.xlabel(f"Tile X-position (in {bin_size}px bins)")
        plt.ylabel(f"Tile Y-position (in {bin_size}px bins)")

        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

        save_path = output_dir / f"{file_prefix}_average_tile_drift_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved average tile drift heatmap: {save_path}")

    def _generate_radial_drift_suite(
        self,
        features: pd.DataFrame,
        cell_df: pd.DataFrame,
        suite_name: str,
        base_output_dir: Path,
    ):
        """
        Generates the full suite of radial drift plots for a given feature set.
        """
        print(f"\n--- Generating Radial Drift plot suite for '{suite_name}' ---")

        # Create a dedicated subdirectory for this suite of plots
        suite_output_dir = base_output_dir / suite_name
        suite_output_dir.mkdir(exist_ok=True, parents=True)

        # The feature set name (e.g., 'all_features') will be the prefix for all files
        file_prefix = suite_name

        well_cutoff, tile_cutoff = self.plot_radial_feature_drift(
            features, cell_df, suite_output_dir, file_prefix
        )

        # Store the result for the summary plots
        if well_cutoff is not None:
            self.well_drift_summary_data.append(
                {"group": suite_name, "percent_cutoff": well_cutoff}
            )
        if tile_cutoff is not None:
            self.tile_drift_summary_data.append(
                {"group": suite_name, "percent_cutoff": tile_cutoff}
            )

        self.plot_average_tile_drift_heatmap(
            features, cell_df, suite_output_dir, file_prefix
        )
        self.plot_feature_drift_heatmap(
            features, cell_df, suite_output_dir, file_prefix
        )

    def plot_inflection_summary(self):
        """
        Generates summary bar charts for well and tile inflection point cutoffs.
        """

        def _plot_summary(summary_data: list, title: str, save_path: Path):
            """Helper function to generate a single summary plot."""
            if not summary_data:
                print(
                    f"\n--- No data available to generate summary plot for: {title} ---"
                )
                return

            print(f"\n--- Generating {title} Summary Plot ---")
            df = pd.DataFrame(summary_data)

            # Clean up names for plotting
            df["group"] = (
                df["group"]
                .str.replace("organelle_", "")
                .str.replace("_", " ")
                .str.title()
            )
            df["group"] = df["group"].str.replace(
                "All Features", "All Features (Combined)"
            )

            # --- NEW: Set a fixed order for the plot ---
            # Define the desired order, adding the cleaned name for 'All Features'
            category_order = [
                "All Features (Combined)",
                "Mitochondria",
                "Phase",
                "Nuclei",
                "Cell",
            ]

            # Get any other groups present in the data but not in the fixed order list
            other_groups = sorted(
                [g for g in df["group"].unique() if g not in category_order]
            )

            # The final order is the fixed list followed by any others
            final_order = category_order + other_groups

            # Convert the 'group' column to a categorical type with the specified order
            df["group"] = pd.Categorical(
                df["group"], categories=final_order, ordered=True
            )

            # Sort the DataFrame by the new categorical order
            df.sort_values("group", ascending=True, inplace=True)

            plt.figure(figsize=(10, max(6, len(df) * 0.5)))
            ax = sns.barplot(
                x="percent_cutoff", y="group", data=df, palette="viridis", orient="h"
            )

            plt.title(title, fontsize=16)
            plt.xlabel("Percent of Cells After Inflection Point (%)")
            plt.ylabel("Feature Group")
            plt.xlim(0, 100)

            # Add data labels to the bars
            for container in ax.containers:
                ax.bar_label(container, fmt="%.1f%%", padding=3)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Saved inflection point summary plot: {save_path}")

        # Define the main output directory for these summaries
        summary_output_dir = self.graph_output_path / "radial_drift"
        summary_output_dir.mkdir(exist_ok=True)

        # Generate Well Summary Plot
        _plot_summary(
            summary_data=self.well_drift_summary_data,
            title="Well Inflection Point Analysis Summary",
            save_path=summary_output_dir / "well_inflection_point_summary.png",
        )

        # Generate Tile Summary Plot
        _plot_summary(
            summary_data=self.tile_drift_summary_data,
            title="Tile Inflection Point Analysis Summary",
            save_path=summary_output_dir / "tile_inflection_point_summary.png",
        )

    def plot_organelle_contribution_heatmap(self, features: pd.DataFrame, cell_df: pd.DataFrame):
        """
        Generate a heatmap showing how much each organelle/segmentation group contributes
        to the phenotypic signature of each gene.

        For each gene, we compute a "contribution score" per organelle group by:
        1. Grouping features by organelle prefix
        2. Computing mean absolute z-score (vs NTC) for each organelle group
        3. Normalizing to show relative contribution (sum to 1 per gene)

        This reveals which organelles are most affected by each gene perturbation.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix (cells x features)
        cell_df : pd.DataFrame
            Cell metadata with gene_name column
        """
        print("\n--- Generating Organelle Contribution Heatmap ---")

        if "gene_name" not in cell_df.columns:
            print("  - Skipping: gene_name column not found in cell metadata")
            return

        # Group features by organelle prefix
        organelle_features = {}
        for col in features.columns:
            parts = col.split("_")
            if parts[0] == "network":
                if len(parts) > 2:
                    organelle = parts[1]
                else:
                    continue
            else:
                if len(parts) > 1:
                    organelle = parts[0]
                else:
                    continue

            if organelle not in organelle_features:
                organelle_features[organelle] = []
            organelle_features[organelle].append(col)

        organelles = sorted(organelle_features.keys())
        print(f"  - Found {len(organelles)} organelle groups: {organelles}")

        if len(organelles) < 2:
            print("  - Skipping: Need at least 2 organelle groups for comparison")
            return

        # Identify NTC cells
        ntc_mask = cell_df["gene_name"].astype(str).str.contains(
            "ntc|non-targeting|^0$", case=False, regex=True
        )
        ntc_features = features.loc[ntc_mask]

        if len(ntc_features) < 10:
            print(f"  - Skipping: Not enough NTC cells ({len(ntc_features)})")
            return

        # Compute NTC mean and std for each feature
        ntc_mean = ntc_features.mean()
        ntc_std = ntc_features.std()
        ntc_std = ntc_std.replace(0, 1)  # Avoid division by zero

        # Get unique genes (excluding NTC)
        genes = cell_df.loc[~ntc_mask, "gene_name"].unique()
        genes = [g for g in genes if pd.notna(g) and str(g) != "nan"]
        print(f"  - Computing contribution scores for {len(genes)} genes...")

        # Compute contribution scores for each gene
        contribution_data = []
        for gene in genes:
            gene_mask = cell_df["gene_name"] == gene
            gene_features = features.loc[gene_mask]

            if len(gene_features) < 3:
                continue

            # Compute mean z-score per organelle group
            gene_scores = {}
            for organelle, cols in organelle_features.items():
                cols_present = [c for c in cols if c in features.columns]
                if not cols_present:
                    gene_scores[organelle] = 0
                    continue

                # Z-score for this gene's cells vs NTC
                gene_mean = gene_features[cols_present].mean()
                z_scores = (gene_mean - ntc_mean[cols_present]) / ntc_std[cols_present]

                # Mean absolute z-score as contribution metric
                mean_abs_z = np.abs(z_scores).mean()
                gene_scores[organelle] = mean_abs_z

            # Normalize to sum to 1 (relative contribution)
            total = sum(gene_scores.values())
            if total > 0:
                gene_scores_norm = {k: v / total for k, v in gene_scores.items()}
            else:
                gene_scores_norm = {k: 0 for k in gene_scores}

            contribution_data.append({
                "gene": gene,
                "n_cells": len(gene_features),
                **gene_scores,  # Raw scores
                **{f"{k}_norm": v for k, v in gene_scores_norm.items()}  # Normalized
            })

        if not contribution_data:
            print("  - Skipping: No genes with enough cells for analysis")
            return

        contrib_df = pd.DataFrame(contribution_data)
        contrib_df = contrib_df.sort_values("n_cells", ascending=False)

        # Save raw data
        output_dir = self.graph_output_path / "organelle_contribution"
        output_dir.mkdir(exist_ok=True, parents=True)
        contrib_df.to_csv(output_dir / "organelle_contribution_scores.csv", index=False)
        print(f"  - Saved contribution scores: {output_dir / 'organelle_contribution_scores.csv'}")

        # --- Heatmap 1: Top genes by cell count (normalized contributions) ---
        top_n = min(50, len(contrib_df))
        top_genes = contrib_df.head(top_n)

        norm_cols = [f"{org}_norm" for org in organelles]
        heatmap_data = top_genes.set_index("gene")[norm_cols]
        heatmap_data.columns = organelles  # Clean column names

        fig, ax = plt.subplots(figsize=(max(12, len(organelles) * 0.8), max(10, top_n * 0.3)))
        sns.heatmap(
            heatmap_data,
            cmap="YlOrRd",
            annot=False,
            fmt=".2f",
            ax=ax,
            cbar_kws={"label": "Relative Contribution (normalized)"},
            vmin=0,
            vmax=0.5,  # Cap at 50% for better color contrast
        )
        ax.set_title(f"Organelle Contribution to Gene Phenotypes (Top {top_n} genes by cell count)")
        ax.set_xlabel("Organelle / Segmentation Group")
        ax.set_ylabel("Gene")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        save_path = output_dir / "organelle_contribution_heatmap_normalized.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  - Saved normalized heatmap: {save_path}")

        # --- Heatmap 2: Raw z-scores (not normalized) ---
        raw_cols = [org for org in organelles]
        heatmap_raw = top_genes.set_index("gene")[raw_cols]

        fig, ax = plt.subplots(figsize=(max(12, len(organelles) * 0.8), max(10, top_n * 0.3)))
        sns.heatmap(
            heatmap_raw,
            cmap="viridis",
            annot=False,
            fmt=".2f",
            ax=ax,
            cbar_kws={"label": "Mean Absolute Z-score vs NTC"},
        )
        ax.set_title(f"Organelle Effect Magnitude by Gene (Top {top_n} genes by cell count)")
        ax.set_xlabel("Organelle / Segmentation Group")
        ax.set_ylabel("Gene")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        save_path = output_dir / "organelle_contribution_heatmap_raw.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  - Saved raw z-score heatmap: {save_path}")

        # --- Heatmap 3: Clustered heatmap to find gene-organelle patterns ---
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import pdist

            # Use normalized data for clustering
            cluster_data = contrib_df.set_index("gene")[norm_cols]
            cluster_data.columns = organelles
            cluster_data = cluster_data.dropna()

            if len(cluster_data) >= 10:
                g = sns.clustermap(
                    cluster_data,
                    cmap="YlOrRd",
                    figsize=(max(12, len(organelles) * 0.8), max(14, len(cluster_data) * 0.15)),
                    dendrogram_ratio=(0.1, 0.15),
                    cbar_pos=(0.02, 0.8, 0.03, 0.15),
                    vmin=0,
                    vmax=0.5,
                )
                g.ax_heatmap.set_xlabel("Organelle / Segmentation Group")
                g.ax_heatmap.set_ylabel("Gene")
                g.fig.suptitle("Clustered Organelle Contribution Heatmap (all genes)", y=1.02)

                save_path = output_dir / "organelle_contribution_heatmap_clustered.png"
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  - Saved clustered heatmap: {save_path}")
        except Exception as e:
            print(f"  - Warning: Could not generate clustered heatmap: {e}")

        # --- Summary bar plot: Average contribution per organelle ---
        avg_contrib = contrib_df[norm_cols].mean()
        avg_contrib.index = organelles

        fig, ax = plt.subplots(figsize=(max(10, len(organelles) * 0.6), 6))
        bars = ax.bar(avg_contrib.index, avg_contrib.values, color=sns.color_palette("husl", len(organelles)))
        ax.set_xlabel("Organelle / Segmentation Group")
        ax.set_ylabel("Average Relative Contribution")
        ax.set_title("Average Organelle Contribution Across All Genes")
        plt.xticks(rotation=45, ha="right")

        # Add value labels
        for bar, val in zip(bars, avg_contrib.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        save_path = output_dir / "organelle_average_contribution.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  - Saved average contribution bar plot: {save_path}")

        print(f"  - Organelle contribution analysis complete!")


def run_graphs(
    experiment: str,
    debug_cell_fraction: float | None = None,
    skip_object_features: bool = False,
    use_cuml: bool = True,
    cluster_algo: str = "hdbscan",
    skip_interactive_plots: bool = True,
    drift_method: str = "three_segment_regression",
):
    """Module-level entrypoint to generate graphs without constructing the class upfront.

    This defers side effects until the runner actually executes the step, so in rerun mode
    non-target steps won't instantiate the generator.
    """
    generator = GraphGenerator(
        experiment,
        debug_cell_fraction=debug_cell_fraction,
        skip_object_features=skip_object_features,
        use_cuml=use_cuml,
        cluster_algo=cluster_algo,
        skip_interactive_plots=skip_interactive_plots,
        drift_method=drift_method,
    )
    generator.run()


def main():
    """Main execution function."""
    from ops_utils.data.filesystem import resolve_experiment_name

    parser = argparse.ArgumentParser(
        description="Generate graphs from morphological feature data."
    )
    parser.add_argument(
        "-e", "--experiment",
        type=str,
        required=True,
        help="Name or shorthand for the experiment (e.g., '94', 'ops94', 'ops0094_20251217').",
    )
    parser.add_argument(
        "--debug",
        type=float,
        default=None,
        help="Fraction of cells to sample for debug mode (e.g., 0.1 for 10%).",
    )
    parser.add_argument(
        "--skip_object_features",
        action="store_true",
        help="If set, do not load object-level feature CSVs to speed up execution.",
    )
    parser.add_argument(
        "--no-cuml",
        action="store_true",
        help="If set, disable GPU acceleration with cuML and force CPU usage.",
    )
    parser.add_argument(
        "--cluster-algo",
        type=str,
        default="all",
        choices=["all", "hdbscan", "kmeans", "leiden"],
        help="Clustering algorithm(s) to use after UMAP. 'all' (default) runs all three methods. 'hdbscan' is density-based. 'kmeans' uses elbow method. 'leiden' uses graph-based community detection.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="If set, generate data for interactive dashboards (skipped by default).",
    )
    parser.add_argument(
        "--drift-method",
        type=str,
        default="three_segment_regression",
        choices=[
            "three_segment_regression",
            "segmented_regression",
            "second_derivative",
        ],
        help="Method to determine the breakpoint in radial drift plots.",
    )
    parser.add_argument(
        "--guide-analysis",
        action="store_true",
        help="Only run guide-level UMAP embedding (skip cell-level analysis).",
    )
    parser.add_argument(
        "--gene-analysis",
        action="store_true",
        help="Only run gene-level UMAP embedding (skip cell-level analysis).",
    )
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="Skip generating plots/images that already exist on disk.",
    )
    args = parser.parse_args()

    # Resolve experiment name (e.g., "94" -> "ops0094_20251217")
    experiment = resolve_experiment_name(args.experiment, allow_interactive=True)

    # Determine analysis mode
    analysis_mode = "all"  # Default: run everything
    if args.guide_analysis and args.gene_analysis:
        analysis_mode = "guide_and_gene"
    elif args.guide_analysis:
        analysis_mode = "guide"
    elif args.gene_analysis:
        analysis_mode = "gene"

    try:
        generator = GraphGenerator(
            experiment,
            debug_cell_fraction=args.debug,
            skip_object_features=args.skip_object_features,
            use_cuml=not args.no_cuml,
            cluster_algo=args.cluster_algo,
            skip_interactive_plots=not args.interactive,
            drift_method=args.drift_method,
            analysis_mode=analysis_mode,
            skip_complete=args.skip_complete,
        )
        generator.run()
    except FileNotFoundError as e:
        print(f"Execution failed: {e}")
    except Exception as e:
        import traceback

        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
