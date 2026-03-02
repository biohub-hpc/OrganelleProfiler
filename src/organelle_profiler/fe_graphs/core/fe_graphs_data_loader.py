"""
Data loading and enrichment module.

Handles loading AnnData files and CSV data, enriching metadata,
and providing a unified data context for analysis.
"""

import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import logging

from ops_utils.data.experiment import OpsDataset

logger = logging.getLogger(__name__)


# =============================================================================
# Organelle Group Discovery - Single Source of Truth
# =============================================================================

def discover_organelle_groups_from_adata(adata: ad.AnnData) -> Dict[str, List[str]]:
    """
    Discover organelle groups from AnnData var metadata.
    
    This is the SINGLE SOURCE OF TRUTH for organelle grouping throughout the analysis.
    Reads the 'organelle' column from adata.var which is populated during feature extraction.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with var["organelle"] column.
        
    Returns
    -------
    dict
        Mapping of organelle name -> list of feature columns.
    """
    organelle_groups = {}
    
    if "organelle" not in adata.var.columns:
        logger.warning("No 'organelle' column in adata.var - cannot discover organelle groups")
        return organelle_groups
    
    # Group features by their organelle assignment
    # Uses same pattern as validate_feature_anndata.py
    for feature_name, row in adata.var.iterrows():
        organelle = row.get("organelle")
        
        # Handle missing/NaN organelle assignments
        if organelle is None or pd.isna(organelle):
            organelle = "_unassigned"
        
        organelle = str(organelle)
        
        if organelle not in organelle_groups:
            organelle_groups[organelle] = []
        organelle_groups[organelle].append(feature_name)
    
    # Validate: organelle groups should be much fewer than features
    n_groups = len(organelle_groups)
    n_features = len(adata.var)
    
    if n_groups > 50:
        # Something is wrong - probably parsing each feature as its own "organelle"
        logger.warning(
            f"WARNING: Found {n_groups} organelle groups for {n_features} features. "
            f"This seems too high! Check that adata.var['organelle'] contains organelle names, "
            f"not feature names."
        )
        # Show first few "organelle" values for debugging
        sample_orgs = list(organelle_groups.keys())[:5]
        logger.warning(f"  Sample organelle values: {sample_orgs}")
        
    # Log discovered groups
    logger.info(f"Discovered {n_groups} organelle groups from adata.var['organelle']")
    for org, cols in sorted(organelle_groups.items(), key=lambda x: -len(x[1]))[:20]:  # Limit logging
        logger.info(f"  {org}: {len(cols)} features")
    if n_groups > 20:
        logger.info(f"  ... and {n_groups - 20} more groups")
    
    return organelle_groups


def discover_organelle_groups(feature_columns: List[str], adata: Optional[ad.AnnData] = None) -> Dict[str, List[str]]:
    """
    Discover organelle groups - uses adata.var if available, otherwise raises error.
    
    Parameters
    ----------
    feature_columns : list
        List of feature column names (used for validation).
    adata : AnnData, optional
        AnnData object with var["organelle"] column.
        
    Returns
    -------
    dict
        Mapping of organelle name -> list of feature columns.
    """
    if adata is not None and "organelle" in adata.var.columns:
        return discover_organelle_groups_from_adata(adata)
    
    raise ValueError(
        "Cannot discover organelle groups: adata.var['organelle'] not available. "
        "Feature extraction should populate this column. "
        "Run validate_feature_anndata.py to check your data."
    )


@dataclass
class DataContext:
    """
    Container for loaded data and paths used by analyzers.
    
    Attributes
    ----------
    experiment : str
        Experiment name.
    dataset : OpsDataset
        OpsDataset instance for path resolution.
    cell_df : pd.DataFrame
        Cell-level data with features and metadata.
    guide_df : pd.DataFrame
        Guide-level aggregated data.
    gene_df : pd.DataFrame
        Gene-level aggregated data.
    adata : dict
        Dictionary of AnnData objects by level (cell, guide, gene).
    feature_columns : list
        List of feature column names.
    organelle_groups : dict
        Single source of truth mapping organelle names to their feature columns.
        E.g., {"cp1_mitochondria_tomm20": ["cp1_mitochondria_tomm20_area", ...]}
        At cell level: feature names without aggregation suffixes.
        Use `organelle_groups_all_levels` for guide/gene level features.
    organelle_groups_all_levels : dict
        Level-specific organelle group mappings.
        E.g., {"cell": {...}, "guide": {...}, "gene": {...}}
        Guide/gene features include aggregation suffixes (_mean, _std, etc.)
    graph_output_path : Path
        Output directory for graphs.
    cache_dir : Path
        Cache directory for embeddings.
    """
    experiment: str
    dataset: OpsDataset
    cell_df: pd.DataFrame
    guide_df: pd.DataFrame
    gene_df: pd.DataFrame
    adata: Dict[str, ad.AnnData] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    organelle_groups: Dict[str, List[str]] = field(default_factory=dict)  # Cell level (backward compatibility)
    organelle_groups_all_levels: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)  # NEW: Per-level organelle groups
    object_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    network_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    morphology_path: Optional[Path] = None  # Path to zarr store for representative cell visualization
    just_cell_painting: bool = False  # Whether we're in CP-only mode
    # Single source of truth for channel/label mapping (built once at initialization)
    channel_names: List[str] = field(default_factory=list)  # Channel names from zarr store
    available_labels: Dict[str, str] = field(default_factory=dict)  # internal_name -> zarr_label_name
    label_to_channel_index: Dict[str, int] = field(default_factory=dict)  # label_internal_name -> channel index
    
    @property
    def analysis_path(self) -> Path:
        return self.dataset.results_fast / "feature_extraction"
    
    @property
    def graph_output_path(self) -> Path:
        folder_name = "graphs_cell_painting" if self.just_cell_painting else "graphs"
        path = self.analysis_path / folder_name
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def cache_dir(self) -> Path:
        # Use separate cache for CP-only mode since it's a different feature set
        cache_name = ".cache_cp" if self.just_cell_painting else ".cache"
        path = self.graph_output_path / cache_name
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def interactive_output_path(self) -> Path:
        path = self.graph_output_path / "interactive_umaps"
        path.mkdir(parents=True, exist_ok=True)
        return path


class DataLoader:
    """
    Loads and prepares data for feature graph generation.
    
    Parameters
    ----------
    experiment : str
        Experiment name or shorthand.
    debug_cell_fraction : float, optional
        Fraction of cells to sample for debug mode.
    skip_object_features : bool
        Whether to skip loading object-level CSVs.
    
    Examples
    --------
    >>> loader = DataLoader("ops0094_20251217")
    >>> data_context = loader.load()
    >>> cell_df = data_context.cell_df
    """
    
    # Columns that should be numeric (not categorical)
    # Note: tile_pheno is categorical (e.g., "A/1/053025"), not numeric
    NUMERIC_COORD_COLS = [
        "x_global_pheno", "y_global_pheno", 
        "x_local_pheno", "y_local_pheno",
        "x_global_bc", "y_global_bc", 
        "segmentation_id",
    ]
    
    # Patterns to exclude when identifying feature columns
    METADATA_EXCLUDE_PATTERNS = [
        "id", "gene", "barcode", "sgRNA", "effect", "NCBI",
        "index", "well", "pos", "_pheno", "_bc", "Unnamed",
        "bbox", "umap", "cluster", "radial",
    ]
    
    def __init__(
        self,
        experiment: str,
        debug_cell_fraction: Optional[float] = None,
        skip_object_features: bool = False,
        just_cell_painting: bool = False,
    ):
        self.experiment = experiment
        self.debug_cell_fraction = debug_cell_fraction
        self.skip_object_features = skip_object_features
        self.just_cell_painting = just_cell_painting
        self.dataset = OpsDataset(experiment)
        
    def load(self) -> DataContext:
        """
        Load all data and return a DataContext.
        
        Returns
        -------
        DataContext
            Container with all loaded data and paths.
        """
        logger.info("Loading data for graph generation...")
        
        adata_dict = {}
        
        # Load cell-level data (required)
        cell_df, cell_adata = self._load_cell_data()
        adata_dict["cell"] = cell_adata
        
        # Load guide-level data
        guide_df, guide_adata = self._load_guide_data()
        adata_dict["guide"] = guide_adata
        
        # Load gene-level data
        gene_df, gene_adata = self._load_gene_data()
        adata_dict["gene"] = gene_adata
        
        # Load object and network data (optional)
        object_data, network_data = self._load_object_network_data()
        
        # Identify feature columns directly from adata.var (SINGLE SOURCE OF TRUTH)
        # This avoids fragile pattern matching that can exclude valid features
        feature_columns = cell_adata.var_names.tolist()
        
        # Discover organelle groups from ALL levels - guide/gene have same organelle assignments
        # Cell level: feature = "cp1_nuclei_area"
        # Guide/Gene level: feature = "cp1_nuclei_area_mean" (still maps to "cp1_nuclei" organelle)
        organelle_groups_all_levels = {
            "cell": discover_organelle_groups(feature_columns, adata=cell_adata)
        }
        
        # For guide/gene levels, create organelle groups from their aggregated features
        if guide_adata is not None:
            organelle_groups_all_levels["guide"] = discover_organelle_groups_from_adata(guide_adata)
        
        if gene_adata is not None:
            organelle_groups_all_levels["gene"] = discover_organelle_groups_from_adata(gene_adata)
        
        # Primary organelle groups (cell level) for backward compatibility
        organelle_groups = organelle_groups_all_levels["cell"]
        
        # Filter to Cell Painting organelles only if requested
        if self.just_cell_painting:
            organelle_groups = self._filter_to_cell_painting_only(organelle_groups)
            # Update feature_columns to only include features from CP organelles
            feature_columns = [f for group_features in organelle_groups.values() for f in group_features]
            logger.info(f"Filtered to Cell Painting organelles only: {len(organelle_groups)} groups, {len(feature_columns)} features")
            
            # Also filter guide/gene level organelle groups
            for level in ["guide", "gene"]:
                if level in organelle_groups_all_levels:
                    organelle_groups_all_levels[level] = self._filter_to_cell_painting_only(organelle_groups_all_levels[level])
        
        logger.info("Finished loading data.")

        # Get morphology path for representative cell visualization
        try:
            morphology_path = self.dataset.morphology_path_v3
        except AttributeError:
            morphology_path = None
            logger.warning("Could not resolve morphology path from dataset. Representative cell visualization may not work.")

        # Build channel/label mappings ONCE (single source of truth for visualization)
        channel_names = []
        available_labels = {}
        label_to_channel_index = {}

        if morphology_path and morphology_path.exists():
            from iohub import open_ome_zarr
            from ...feature_extraction.fe_metadata import _discover_available_labels

            # Get channel names from zarr store
            try:
                with open_ome_zarr(morphology_path, mode="r") as store:
                    channel_names = list(store.channel_names)
                logger.info(f"Loaded {len(channel_names)} channel names from zarr store")

                # Discover available labels
                available_labels = _discover_available_labels(morphology_path)
                logger.info(f"Discovered {len(available_labels)} label mappings")

                # Build label_to_channel_index mapping
                # Maps label internal names (used in features) to channel indices (for intensity visualization)
                label_to_channel_index = self._build_label_to_channel_mapping(
                    available_labels, channel_names
                )
                logger.info(f"Built {len(label_to_channel_index)} label-to-channel mappings")

            except Exception as e:
                logger.warning(f"Could not build channel/label mappings: {e}")

        return DataContext(
            experiment=self.experiment,
            dataset=self.dataset,
            cell_df=cell_df,
            guide_df=guide_df,
            gene_df=gene_df,
            adata=adata_dict,
            feature_columns=feature_columns,
            organelle_groups=organelle_groups,
            organelle_groups_all_levels=organelle_groups_all_levels,  # NEW: level-specific mappings
            object_data=object_data,
            network_data=network_data,
            morphology_path=morphology_path,
            just_cell_painting=self.just_cell_painting,
            channel_names=channel_names,
            available_labels=available_labels,
            label_to_channel_index=label_to_channel_index,
        )
    
    def _load_cell_data(self) -> tuple[pd.DataFrame, ad.AnnData]:
        """Load cell-level AnnData and convert to DataFrame."""
        h5ad_path = self.dataset.results_fast / "feature_extraction" / f"{self.experiment}_cell_features.h5ad"

        if not h5ad_path.exists():
            raise FileNotFoundError(f"Cell-level AnnData not found at {h5ad_path}")
        
        logger.info(f"Loading cell features from: {h5ad_path.name}")
        cell_adata = ad.read_h5ad(h5ad_path)
        
        # Enrich with missing metadata from linked_results CSV
        cell_adata = self._enrich_metadata_from_linked_csv(cell_adata)
        
        # Convert to DataFrame
        cell_df = cell_adata.obs.copy()
        
        # Convert bbox columns from string format to lists if needed
        # AnnData serialization can convert arrays to strings like "[1 2 3 4]" (no commas)
        for bbox_col in ['bbox', 'cp_bbox']:
            if bbox_col in cell_df.columns:
                logger.info(f"Processing {bbox_col} column (dtype: {cell_df[bbox_col].dtype})")
                # Convert categorical to object first if needed (lists can't be hashed in categorical)
                if pd.api.types.is_categorical_dtype(cell_df[bbox_col]):
                    logger.info(f"  Converting {bbox_col} from categorical to object")
                    cell_df[bbox_col] = cell_df[bbox_col].astype(object)
                
                def parse_bbox_string(x):
                    if pd.isna(x) or x is None:
                        return x
                    # Already a list - keep as is
                    if isinstance(x, (list, np.ndarray)):
                        return list(x) if isinstance(x, np.ndarray) else x
                    if isinstance(x, str):
                        # Handle numpy array string format: "[1 2 3 4]" -> [1, 2, 3, 4]
                        x = x.strip()
                        if x.startswith('[') and x.endswith(']'):
                            values = x[1:-1].split()
                            try:
                                return [int(v) for v in values if v]
                            except ValueError:
                                return x
                    return x
                cell_df[bbox_col] = cell_df[bbox_col].apply(parse_bbox_string)
                
                # CRITICAL: Also update the adata.obs to persist the conversion
                # Otherwise, downstream code that accesses adata.obs will get unconverted data
                if bbox_col in cell_adata.obs.columns:
                    if pd.api.types.is_categorical_dtype(cell_adata.obs[bbox_col]):
                        cell_adata.obs[bbox_col] = cell_adata.obs[bbox_col].astype(object)
                    cell_adata.obs[bbox_col] = cell_adata.obs[bbox_col].apply(parse_bbox_string)
                
                sample_val = cell_df[bbox_col].iloc[0] if len(cell_df) > 0 else 'N/A'
                logger.info(f"  After conversion: {bbox_col} sample={sample_val}, type={type(sample_val).__name__}")
        
        feature_df = pd.DataFrame(
            cell_adata.X,
            index=cell_adata.obs_names,
            columns=cell_adata.var_names
        )
        cell_df = pd.concat([cell_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
        
        # Ensure coordinate columns are numeric
        for col in self.NUMERIC_COORD_COLS:
            if col in cell_df.columns:
                cell_df[col] = pd.to_numeric(cell_df[col], errors="coerce")
        
        # Debug mode subsampling
        if self.debug_cell_fraction and 0 < self.debug_cell_fraction <= 1:
            n_sample = int(len(cell_df) * self.debug_cell_fraction)
            logger.info(f"DEBUG MODE: Sampling {n_sample} cells ({self.debug_cell_fraction:.0%})")
            cell_df = cell_df.sample(n=n_sample, random_state=42).reset_index(drop=True)
        
        # Standardize NTC gene name
        if "gene_name" in cell_df.columns:
            cell_df["gene_name"] = cell_df["gene_name"].astype(str).replace({"0": "NTC"})
        
        logger.info(f"Loaded {len(cell_df)} cells with {len(cell_adata.var_names)} features")
        
        return cell_df, cell_adata
    
    def _load_guide_data(self) -> tuple[pd.DataFrame, Optional[ad.AnnData]]:
        """Load guide-level AnnData and convert to DataFrame."""
        h5ad_path = self.dataset.results_fast / "feature_extraction" / f"{self.experiment}_guide_features.h5ad"
        
        if not h5ad_path.exists():
            logger.warning(f"Guide-level AnnData not found at {h5ad_path}")
            return pd.DataFrame(), None
        
        logger.info(f"Loading guide features from: {h5ad_path.name}")
        guide_adata = ad.read_h5ad(h5ad_path)
        
        guide_df = guide_adata.obs.copy()
        feature_df = pd.DataFrame(
            guide_adata.X,
            index=guide_adata.obs_names,
            columns=guide_adata.var_names
        )
        guide_df = pd.concat([guide_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
        
        logger.info(f"Loaded {len(guide_df)} guides")
        
        return guide_df, guide_adata
    
    def _load_gene_data(self) -> tuple[pd.DataFrame, Optional[ad.AnnData]]:
        """Load gene-level AnnData and convert to DataFrame."""
        h5ad_path = self.dataset.results_fast / "feature_extraction" / f"{self.experiment}_gene_features.h5ad"
        
        if not h5ad_path.exists():
            logger.warning(f"Gene-level AnnData not found at {h5ad_path}")
            return pd.DataFrame(), None
        
        logger.info(f"Loading gene features from: {h5ad_path.name}")
        gene_adata = ad.read_h5ad(h5ad_path)
        
        gene_df = gene_adata.obs.copy()
        feature_df = pd.DataFrame(
            gene_adata.X,
            index=gene_adata.obs_names,
            columns=gene_adata.var_names
        )
        gene_df = pd.concat([gene_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
        
        logger.info(f"Loaded {len(gene_df)} genes")
        
        return gene_df, gene_adata
    
    def _load_object_network_data(self) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Load object and network feature CSVs. Deprecated - all data is in AnnData."""
        return {}, {}
    
    def _enrich_metadata_from_linked_csv(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Enrich AnnData obs with missing metadata columns from linked_results CSV.
        
        If the AnnData is missing columns like tile_pheno, x_local_pheno, y_local_pheno
        (needed for radial drift analysis), this method loads them from the source
        linked_results CSV and adds them to obs by matching on segmentation_id + well.
        
        Note: In most cases, the tile column already exists as 'tile' and just needs
        to be renamed to 'tile_pheno' for consistency with the naming convention.
        """
        # First, check if we just need to rename existing columns
        if "tile" in adata.obs.columns and "tile_pheno" not in adata.obs.columns:
            logger.info("Renaming 'tile' column to 'tile_pheno' for consistency")
            adata.obs["tile_pheno"] = adata.obs["tile"]
        
        # Columns needed for radial drift analysis
        required_cols = ["tile_pheno", "x_local_pheno", "y_local_pheno"]
        missing_cols = [c for c in required_cols if c not in adata.obs.columns]
        
        # Check if coordinate columns are all zeros (need to be populated from CSV)
        coord_cols_to_fix = []
        if "x_global_pheno" in adata.obs.columns and (adata.obs["x_global_pheno"] == 0).all():
            coord_cols_to_fix.append(("x_global_pheno", "x_pheno"))
        if "y_global_pheno" in adata.obs.columns and (adata.obs["y_global_pheno"] == 0).all():
            coord_cols_to_fix.append(("y_global_pheno", "y_pheno"))
        
        if not missing_cols and not coord_cols_to_fix:
            return adata
        
        if missing_cols:
            logger.info(f"AnnData missing columns: {missing_cols}")
        if coord_cols_to_fix:
            logger.info(f"AnnData has zero-valued coordinate columns: {[c[0] for c in coord_cols_to_fix]}")
        logger.info("Attempting to enrich from linked_results CSV...")
        
        # Load linked_results CSVs for all wells
        linked_dfs = []
        wells = adata.obs["well"].unique() if "well" in adata.obs.columns else []
        
        for well in wells:
            well_short = well.rsplit("/", 1)[0] if "/" in str(well) else str(well)
            results_path = self.dataset.append_well("linked_results", well_short)
            if results_path.exists():
                df = pd.read_csv(results_path)
                df["well"] = well
                linked_dfs.append(df)
        
        if not linked_dfs:
            logger.warning("No linked_results CSVs found. Cannot enrich metadata.")
            return adata
        
        linked_df = pd.concat(linked_dfs, ignore_index=True)
        logger.info(f"Loaded {len(linked_df)} rows from linked_results CSVs")
        
        if "segmentation_id" not in linked_df.columns:
            logger.warning("Cannot match cells - segmentation_id not found in linked_results CSV")
            return adata
        
        # Create cell_id in linked_df to match AnnData index
        n_nan_linked = linked_df["segmentation_id"].isna().sum()
        if n_nan_linked > 0:
            logger.warning(f"{n_nan_linked} rows in linked_results have NaN segmentation_id (skipped)")
        
        valid_mask = linked_df["segmentation_id"].notna()
        linked_df = linked_df[valid_mask].copy()
        linked_df["cell_id"] = (
            linked_df["well"].astype(str) + "_" +
            linked_df["segmentation_id"].astype(float).astype(int).astype(str)
        )
        
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
        
        # Also add columns that need fixing from coord_cols_to_fix
        for obs_col, csv_col in coord_cols_to_fix:
            if obs_col not in [c[0] for c in cols_to_add] and csv_col in linked_df.columns:
                cols_to_add.append((obs_col, csv_col))
        
        if cols_to_add:
            lookup = {}
            for target_col, source_col in cols_to_add:
                lookup[target_col] = linked_df.set_index("cell_id")[source_col].to_dict()
            
            enriched_count = 0
            for target_col, source_col in cols_to_add:
                adata.obs[target_col] = adata.obs.index.map(lookup[target_col])
                enriched_count = max(enriched_count, adata.obs[target_col].notna().sum())
            
            # Ensure coordinate columns are numeric
            coord_columns = [
                "x_global_pheno", "y_global_pheno", "x_local_pheno", "y_local_pheno",
                "x_pheno", "y_pheno", "tile_pheno"
            ]
            for coord_col in coord_columns:
                if coord_col in adata.obs.columns:
                    adata.obs[coord_col] = pd.to_numeric(adata.obs[coord_col], errors="coerce")
            
            logger.info(f"Enriched {enriched_count}/{len(adata.obs)} cells by cell_id match")
            
            # Save the enriched AnnData back to disk
            h5ad_path = self.dataset.results_fast / "feature_extraction" / f"{self.dataset.experiment}_cell_features.h5ad"
            adata.write_h5ad(h5ad_path)
            logger.info(f"Saved enriched AnnData to {h5ad_path}")
        
        return adata

    def _build_label_to_channel_mapping(
        self, available_labels: Dict[str, str], channel_names: List[str]
    ) -> Dict[str, int]:
        """
        Build mapping from label internal names (used in features) to channel indices.

        This is the single source of truth for visualization - maps organelle/label names
        from features to the corresponding intensity channel index.

        Parameters
        ----------
        available_labels : dict
            Mapping from internal name to zarr label name (e.g., "cp1_nuclei_hoechst" -> "cp1_nuclei_hoechst_seg")
        channel_names : list
            List of channel names from zarr store (e.g., ["Phase2D", "Focus3D", "CP1_nuclei_Hoechst", ...])

        Returns
        -------
        dict
            Mapping from label internal name to channel index
        """
        mapping = {}
        channel_names_lower = [ch.lower() for ch in channel_names]

        # Direct mapping: index channels by position
        for i, ch_name in enumerate(channel_names):
            ch_lower = ch_name.lower().replace(" ", "_").replace(",", "").replace("-", "_")
            mapping[ch_lower] = i
            # Also add without cp prefix variations
            if ch_lower.startswith("cp1_") or ch_lower.startswith("cp2_"):
                mapping[ch_lower[4:]] = i  # Without prefix

        # Map label names to channels based on naming patterns
        for label_name in available_labels.keys():
            label_lower = label_name.lower()

            # Skip if already mapped
            if label_lower in mapping:
                continue

            # Try to find matching channel
            # Pattern 1: Label name matches channel name directly
            for i, ch_lower in enumerate(channel_names_lower):
                ch_normalized = ch_lower.replace(" ", "_").replace(",", "").replace("-", "_")
                if label_lower == ch_normalized or label_lower in ch_normalized or ch_normalized in label_lower:
                    mapping[label_lower] = i
                    break

            # Pattern 2: CP labels map to CP channels (e.g., "CP1_nuclear" -> "CP1_nuclei_Hoechst")
            if label_lower not in mapping:
                if label_lower.startswith("cp1_"):
                    # Map CP1 labels to CP1 channels
                    base = label_lower[4:]  # Remove "cp1_"
                    for i, ch_lower in enumerate(channel_names_lower):
                        if ch_lower.startswith("cp1_") and (base in ch_lower or ch_lower[4:].startswith(base[:4])):
                            mapping[label_lower] = i
                            break
                elif label_lower.startswith("cp2_"):
                    # Map CP2 labels to CP2 channels
                    base = label_lower[4:]  # Remove "cp2_"
                    for i, ch_lower in enumerate(channel_names_lower):
                        if ch_lower.startswith("cp2_") and (base in ch_lower or ch_lower[4:].startswith(base[:4])):
                            mapping[label_lower] = i
                            break

            # Pattern 3: Phase/Focus labels map to Phase2D/Focus3D channels
            if label_lower not in mapping:
                if "phase2d" in label_lower or label_lower.startswith("phase"):
                    for i, ch_lower in enumerate(channel_names_lower):
                        if "phase" in ch_lower:
                            mapping[label_lower] = i
                            break
                elif "focus3d" in label_lower or label_lower.startswith("focus"):
                    for i, ch_lower in enumerate(channel_names_lower):
                        if "focus" in ch_lower:
                            mapping[label_lower] = i
                            break

            # Pattern 4: Nucleoli labels map to phase or focus channels
            if label_lower not in mapping:
                if "nucleoli" in label_lower:
                    if "phase" in label_lower:
                        for i, ch_lower in enumerate(channel_names_lower):
                            if "phase" in ch_lower:
                                mapping[label_lower] = i
                                break
                    elif "focus" in label_lower:
                        for i, ch_lower in enumerate(channel_names_lower):
                            if "focus" in ch_lower:
                                mapping[label_lower] = i
                                break

        return mapping

    def _filter_to_cell_painting_only(self, organelle_groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Filter organelle groups to only include Cell Painting organelles.
        
        Keeps: cp1_*, cp2_*, CP1_*, CP2_*, cp_cell, cell (if it's truly a cell outline)
        Removes: nuclear_seg, cell_seg, nuclei, phase2d_*, focus3d_*, nucleoli_phase2d, nucleoli_focus3d
        
        Parameters
        ----------
        organelle_groups : dict
            Original organelle groups
            
        Returns
        -------
        dict
            Filtered organelle groups with only Cell Painting channels
        """
        # Patterns to KEEP (Cell Painting organelles)
        cp_patterns = ["cp1_", "cp2_", "CP1_", "CP2_"]
        
        # Patterns to EXCLUDE (non-Cell Painting segmentations)
        exclude_patterns = [
            "phase2d_",
            "focus3d_", 
            "nucleoli_phase2d",
            "nucleoli_focus3d",
        ]
        
        # Special cases to exclude
        exclude_exact = ["nuclei", "nuclear_seg", "cell_seg"]
        
        filtered = {}
        excluded_groups = []
        
        for organelle_name, features in organelle_groups.items():
            # Check if it's a CP organelle
            is_cp = any(organelle_name.startswith(pattern) for pattern in cp_patterns)
            
            # Check if it should be excluded
            is_excluded = (
                any(pattern in organelle_name for pattern in exclude_patterns) or
                organelle_name in exclude_exact
            )
            
            # Special case: keep "cell" and "cp_cell" as they're cell outlines
            if organelle_name in ["cell", "cp_cell"]:
                is_cp = True
                is_excluded = False
            
            if is_cp and not is_excluded:
                filtered[organelle_name] = features
            else:
                excluded_groups.append(organelle_name)
        
        if excluded_groups:
            logger.info(f"  Excluded {len(excluded_groups)} non-CP organelle groups: {excluded_groups}")
        
        return filtered
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns by excluding metadata columns."""
        feature_cols = [
            col for col in df.select_dtypes(include=np.number).columns
            if not any(pattern in col for pattern in self.METADATA_EXCLUDE_PATTERNS)
        ]
        return feature_cols
