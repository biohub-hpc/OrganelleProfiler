"""
Representative Cell Visualization for Positive Control Clusters.

For each positive control cluster, finds cells that exemplify the distinguishing
morphological features and visualizes them with feature overlays.

Supports all levels:
- Cell level: Select representative cells directly
- Guide/Gene level: Select representative guides/genes, then show their top cells
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Feature type sets (mirrored from fe_visualization.py)
PER_OBJECT_NETWORK_FEATURES = {
    "num_branches",
    "num_nodes",
    "num_endpoints",
    "average_degree",
    "branch_length",
    "branch_thickness",
    "tortuosity",
}

LOCALIZATION_FEATURES = {
    "distance_from_nucleus_centroid",
    "distance_from_nuclear_boundary",
    "distance_from_cell_edge",
    "normalized_radial_position",
    "angular_position",
}

# Basic morphological features from regionprops
BASIC_MORPHOLOGICAL_FEATURES = {
    "area",
    "perimeter",
    "axis_major_length",
    "eccentricity",
}


def _extract_metric_from_feature_name(feature_name: str, organelle_name: str) -> str:
    """
    Extract the metric (e.g., 'area', 'num_branches') from a full feature name.

    Feature names follow patterns like:
    - "cp1_mitochondria_tomm20_area_mean" -> "area"
    - "network_cp1_mitochondria_tomm20_num_branches_mean" -> "num_branches"
    - "cp1_plasma_membrane_wga_eccentricity_std" -> "eccentricity"

    Parameters
    ----------
    feature_name : str
        Full feature name from the distinguishing features analysis
    organelle_name : str
        Organelle name (used to strip prefix)

    Returns
    -------
    str
        The metric name (e.g., "area", "num_branches", "eccentricity")
    """
    # Strip network prefix if present
    feat = feature_name
    if feat.startswith("network_"):
        feat = feat[8:]

    # Strip organelle prefix
    if organelle_name and feat.startswith(organelle_name + "_"):
        feat = feat[len(organelle_name) + 1:]

    # Strip ALL aggregation suffixes (mean, std, median, etc.)
    # Gene-level features have double suffixes like "_median_median"
    suffixes = ["_mean", "_std", "_median", "_min", "_max", "_sum", "_count"]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if feat.endswith(suffix):
                feat = feat[: -len(suffix)]
                changed = True
                break

    # Check against known feature sets
    # Network features (may have multi-word names like "num_branches")
    for net_feat in PER_OBJECT_NETWORK_FEATURES:
        if feat == net_feat or feat.endswith("_" + net_feat):
            return net_feat

    # Localization features
    for loc_feat in LOCALIZATION_FEATURES:
        if feat == loc_feat or feat.endswith("_" + loc_feat):
            return loc_feat

    # Basic morphological features
    for morph_feat in BASIC_MORPHOLOGICAL_FEATURES:
        if feat == morph_feat or feat.endswith("_" + morph_feat):
            return morph_feat

    # Fallback: return the parsed feature or default to "area"
    if feat in (PER_OBJECT_NETWORK_FEATURES | LOCALIZATION_FEATURES | BASIC_MORPHOLOGICAL_FEATURES):
        return feat

    logger.debug(f"Could not parse metric from '{feature_name}', defaulting to 'area'")
    return "area"


def _compute_per_object_features_for_metric(
    label_array: np.ndarray,
    metric: str,
    cell_mask: np.ndarray,
    organelle_labels_dict: Dict,
) -> pd.DataFrame:
    """
    Compute per-object features on-demand based on the metric type.

    Mirrors the logic in fe_visualization.py for computing features dynamically.

    Parameters
    ----------
    label_array : np.ndarray
        2D label array for the organelle (already masked to cell)
    metric : str
        The metric to compute (e.g., "area", "num_branches", "normalized_radial_position")
    cell_mask : np.ndarray
        2D binary cell mask
    organelle_labels_dict : dict
        Dict of all organelle labels (for finding nuclear mask if needed)

    Returns
    -------
    pd.DataFrame
        DataFrame with 'label' column and the computed metric column
    """
    from ...fe_visualization import (
        _compute_basic_features,
        _compute_per_object_network_features,
        _compute_localization_features_for_viz,
        NETWORK_ANALYSIS_AVAILABLE,
        LOCALIZATION_AVAILABLE,
    )

    if not np.any(label_array > 0):
        return pd.DataFrame()

    # Determine feature type and compute appropriately
    if metric in PER_OBJECT_NETWORK_FEATURES:
        if NETWORK_ANALYSIS_AVAILABLE:
            return _compute_per_object_network_features(label_array)
        else:
            logger.warning(f"Network analysis not available, cannot compute '{metric}'")
            return pd.DataFrame()

    elif metric in LOCALIZATION_FEATURES:
        if LOCALIZATION_AVAILABLE:
            # Find nuclear mask from organelle_labels_dict
            nuclear_mask = None
            for nuc_key in ("nuclei", "nuclear_seg"):
                if nuc_key in organelle_labels_dict:
                    nuc_arr = organelle_labels_dict[nuc_key]
                    nuclear_mask = (nuc_arr > 0).astype(np.uint8)
                    break
            return _compute_localization_features_for_viz(label_array, cell_mask, nuclear_mask)
        else:
            logger.warning(f"Localization features not available, cannot compute '{metric}'")
            return pd.DataFrame()

    else:
        # Default: compute basic morphological features
        return _compute_basic_features(label_array)


def _collect_viz_data_fast(
    morphology_path: Path,
    cells_df: pd.DataFrame,
    channel_names: list,
    organelle_to_load: Optional[str] = None,
    organelle_channel_indices: Optional[Dict[str, int]] = None,
) -> list:
    """
    Fast collection of visualization data using BaseDataset (like view_by_gene.py).

    Loads intensity and cell mask using BaseDataset. Organelle labels are NOT loaded
    here because BaseDataset doesn't support them - they must be loaded directly from
    zarr using the cell's bbox coordinates.

    Parameters
    ----------
    morphology_path : Path
        Path to the zarr v3 store
    cells_df : pd.DataFrame
        DataFrame with cell metadata (must have 'well', 'store_key' columns)
    channel_names : list
        List of channel names from zarr
    organelle_to_load : str, optional
        Name of the organelle to visualize (used to select which channel to load)

    Returns
    -------
    list
        List of viz_data dicts for each cell
    """
    from iohub import open_ome_zarr
    from ops_utils.data.bbox_utils import BaseDataset

    viz_data_list = []

    # Open zarr store
    pheno_store = open_ome_zarr(morphology_path, mode="r")
    stores = {"pheno_assembled_v3": pheno_store}

    # Prepare cells_df for BaseDataset
    # CRITICAL: Reset index so BaseDataset can access by sequential integer index
    cells_for_dataset = cells_df.copy().reset_index(drop=True)
    if 'store_key' not in cells_for_dataset.columns:
        cells_for_dataset['store_key'] = 'pheno_assembled_v3'

    # Diagnostic logging - check if required columns exist
    required_cols = ['well', 'bbox', 'gene_name', 'segmentation_id', 'total_index']
    missing_cols = [c for c in required_cols if c not in cells_for_dataset.columns]
    if missing_cols:
        logger.warning(f"      Missing required columns for BaseDataset: {missing_cols}")
        logger.warning(f"      Available columns: {list(cells_for_dataset.columns)}")

    # Check first cell's data
    if len(cells_for_dataset) > 0:
        first_cell = cells_for_dataset.iloc[0]
        logger.info(f"      First cell data: well={first_cell.get('well')}, bbox={first_cell.get('bbox')}, store_key={first_cell.get('store_key')}")

    logger.debug(f"      Prepared {len(cells_for_dataset)} cells for BaseDataset")

    # Determine which channels to load (OPTIMIZATION: only load what we need)
    # We need: Phase2D (channel 0) for context + the organelle's channel
    channels_to_load_indices = [0]  # Always include Phase2D index
    if organelle_to_load and organelle_channel_indices:
        # Case-insensitive lookup (organelle names may have different case than channel mapping keys)
        org_ch = organelle_channel_indices.get(organelle_to_load)
        if org_ch is None:
            # Try lowercase
            org_lower = organelle_to_load.lower()
            org_ch = organelle_channel_indices.get(org_lower)
        if org_ch is None:
            # Try case-insensitive key comparison
            for key, value in organelle_channel_indices.items():
                if key.lower() == organelle_to_load.lower():
                    org_ch = value
                    logger.debug(f"      Case-insensitive match: '{organelle_to_load}' -> '{key}' -> ch {value}")
                    break
        if org_ch is not None and org_ch != 0:
            channels_to_load_indices.append(org_ch)
        elif org_ch is None:
            logger.warning(f"      Could not find channel for organelle '{organelle_to_load}' in mapping. "
                          f"Available keys: {list(organelle_channel_indices.keys())[:10]}...")

    # Convert channel indices to channel NAMES (BaseDataset expects names, not indices!)
    channels_to_load_names = []
    for idx in channels_to_load_indices:
        if 0 <= idx < len(channel_names):
            channels_to_load_names.append(channel_names[idx])
        else:
            logger.warning(f"      Channel index {idx} out of range (have {len(channel_names)} channels)")

    # Fallback to "all" if we couldn't resolve channel names
    if not channels_to_load_names:
        logger.warning(f"      Could not resolve channel names, falling back to 'all'")
        channels_to_load_names = "all"

    logger.debug(f"      Loading channels {channels_to_load_names} (indices {channels_to_load_indices})")

    import time
    t0 = time.time()

    # Create BaseDataset - only load required channels for speed
    base_dataset = BaseDataset(
        stores=stores,
        labels_df=cells_for_dataset,
        initial_yx_patch_size=(300, 300),
        final_yx_patch_size=(300, 300),
        out_channels=channels_to_load_names,  # Must be names, not indices!
        mask_cell=False,
        use_original_crop_size=False,  # Match fe_visualization.py
    )

    logger.debug(f"      BaseDataset created in {time.time() - t0:.2f}s")

    # Map from loaded channel index to original channel index
    # E.g., if channels_to_load_indices=[0, 7], then loaded idx 0 -> original 0, loaded idx 1 -> original 7
    loaded_to_original = {i: ch for i, ch in enumerate(channels_to_load_indices)}

    skipped = 0
    n_cells = len(cells_for_dataset)
    t_loop = time.time()
    for i in range(n_cells):
        try:
            batch = base_dataset[i]
            data = batch["data"].numpy() if hasattr(batch["data"], "numpy") else np.array(batch["data"])
            mask = batch["mask"].numpy() if hasattr(batch["mask"], "numpy") else np.array(batch["mask"])
            crop_info = batch["crop_info"]
            bbox = batch.get("bbox")
            well = crop_info.get("well")
            
            # Load cp_cell_mask using BaseDataset's robust loading method (same pattern for all zarr reads)
            cp_cell_mask = None
            if bbox is not None and len(bbox) == 4:
                cp_labels_full = base_dataset.load_label_array(
                    store_key='pheno_assembled_v3',
                    well=well,
                    label_name='cp_cell_seg',
                    bbox=bbox
                )
                if cp_labels_full is not None:
                    # Find the main cell (center or most common)
                    center_y, center_x = cp_labels_full.shape[0] // 2, cp_labels_full.shape[1] // 2
                    cell_id = cp_labels_full[center_y, center_x]
                    if cell_id == 0:
                        unique, counts = np.unique(cp_labels_full[cp_labels_full > 0], return_counts=True)
                        if len(unique) > 0:
                            cell_id = unique[np.argmax(counts)]

                    # Make binary mask for this cell
                    cp_cell_mask_binary = (cp_labels_full == cell_id).astype(np.uint8)

                    # Crop/pad to match data size
                    target_h, target_w = data.shape[1], data.shape[2]
                    if cp_cell_mask_binary.shape != (target_h, target_w):
                        h, w = cp_cell_mask_binary.shape
                        if h > target_h and w > target_w:
                            start_y = (h - target_h) // 2
                            start_x = (w - target_w) // 2
                            cp_cell_mask_binary = cp_cell_mask_binary[start_y:start_y+target_h, start_x:start_x+target_w]
                        else:
                            padded = np.zeros((target_h, target_w), dtype=cp_cell_mask_binary.dtype)
                            padded[:min(h, target_h), :min(w, target_w)] = cp_cell_mask_binary[:min(h, target_h), :min(w, target_w)]
                            cp_cell_mask_binary = padded

                    cp_cell_mask = cp_cell_mask_binary[np.newaxis, ...]  # Add channel dim
            
            # VALIDATE: Check if actual channels loaded matches what we requested
            # BaseDataset may return fewer channels if some don't exist in zarr
            actual_n_channels = data.shape[0] if data.ndim == 3 else 1
            actual_loaded_channels = channels_to_load_indices[:actual_n_channels]  # Truncate if fewer loaded
            actual_loaded_to_original = {i: ch for i, ch in enumerate(actual_loaded_channels)}

            if actual_n_channels < len(channels_to_load_indices):
                logger.debug(f"      Cell {i}: Requested {len(channels_to_load_indices)} channels, "
                            f"got {actual_n_channels}. Adjusting loaded_channels.")

            # Pre-load organelle labels using BaseDataset's robust loading method
            organelle_labels = None
            organelle_load_failure_reason = None  # Track why loading failed

            if not organelle_to_load:
                organelle_load_failure_reason = "no organelle specified"
            elif bbox is None or len(bbox) != 4:
                organelle_load_failure_reason = f"invalid bbox: {bbox}"
            else:
                # Discover available labels if not yet cached
                if not hasattr(_collect_viz_data_fast, '_available_labels'):
                    from ...feature_extraction.fe_metadata import _discover_available_labels
                    _collect_viz_data_fast._available_labels = _discover_available_labels(morphology_path)
                available_labels = _collect_viz_data_fast._available_labels

                # Look up the zarr label name
                zarr_label_name = available_labels.get(organelle_to_load)
                if zarr_label_name is None:
                    # Case-insensitive fallback
                    for key, value in available_labels.items():
                        if key.lower() == organelle_to_load.lower():
                            zarr_label_name = value
                            break

                if zarr_label_name is None:
                    organelle_load_failure_reason = f"'{organelle_to_load}' not in available labels: {list(available_labels.keys())[:5]}..."
                else:
                    try:
                        organelle_labels = base_dataset.load_label_array(
                            store_key='pheno_assembled_v3',
                            well=well,
                            label_name=zarr_label_name,
                            bbox=bbox
                        )
                        if organelle_labels is None:
                            organelle_load_failure_reason = f"load_label_array returned None for '{zarr_label_name}' at bbox={bbox}"
                        elif not np.any(organelle_labels > 0):
                            organelle_load_failure_reason = f"loaded '{zarr_label_name}' but all zeros (no objects in bbox={bbox})"
                            organelle_labels = None  # Treat as failure
                    except Exception as load_err:
                        organelle_load_failure_reason = f"load_label_array error: {load_err}"

            # Log organelle loading failure with specific reason (first 3 only to avoid spam)
            if organelle_load_failure_reason and i < 3:
                logger.warning(f"      Cell {i} organelle load failed: {organelle_load_failure_reason}")

            # Store visualization data with pre-loaded organelle labels
            viz_data_list.append({
                "intensity": data,  # (C, H, W) - only loaded channels
                "cell_mask": mask,  # (1, H, W) - standard cell_seg
                "cp_cell_mask": cp_cell_mask,  # (1, H, W) or None - CP cell_seg
                "organelle_labels": organelle_labels,  # (H, W) or None - pre-loaded
                "organelle_load_failure": organelle_load_failure_reason,  # Why loading failed (or None if success)
                "bbox": bbox,  # For reference
                "well": well,
                "crop_info": crop_info,
                "channel_names": channel_names,
                "organelle_to_load": organelle_to_load,
                "organelle_channel_indices": organelle_channel_indices or {},  # Pass through
                "loaded_channels": actual_loaded_channels,  # Actual channels loaded (may be fewer than requested)
                "loaded_to_original": actual_loaded_to_original,  # Map loaded idx -> original idx
            })
            
        except Exception as e:
            # Log first few errors at INFO level for debugging
            if skipped < 3:
                logger.info(f"      Cell {i} failed: {e}")
            else:
                logger.debug(f"      Skipped cell {i}: {e}")
            skipped += 1
            continue
    
    pheno_store.close()

    logger.debug(f"      Cell loop completed in {time.time() - t_loop:.2f}s")

    if skipped > 0:
        logger.debug(f"      Skipped {skipped}/{len(cells_for_dataset)} cells")

    logger.info(f"Collected visualization data for {len(viz_data_list)} cells in {time.time() - t0:.1f}s.")
    return viz_data_list


def _load_organelle_label_for_cell_fast(
    pheno_store,  # Already-open zarr store (NOT path!)
    well: str,
    bbox: tuple,
    organelle_name: str,
    available_labels: Dict,  # Required - must be pre-discovered
) -> Optional[np.ndarray]:
    """
    Load organelle label for a single cell from an ALREADY-OPEN zarr store.

    PERFORMANCE: This function expects an already-open store to avoid the
    massive overhead of opening/closing the store for each cell.

    Parameters
    ----------
    pheno_store : open zarr store
        Already-open iohub zarr store (from open_ome_zarr)
    well : str
        Well identifier (e.g., "A/1/0")
    bbox : tuple
        Cell bounding box (y_min, y_max, x_min, x_max)
    organelle_name : str
        Internal organelle name to load (or "cell" for cell segmentation)
    available_labels : dict
        Pre-discovered labels mapping (required for performance)

    Returns
    -------
    np.ndarray or None
        Organelle label array, or None if not found
    """
    # Parse bbox - scikit-image regionprops format: (y_min, x_min, y_max, x_max)
    if bbox is None or len(bbox) != 4:
        return None
    y_min, x_min, y_max, x_max = bbox

    # Special case: "cell" means load cell_seg
    if organelle_name == "cell":
        zarr_label_name = "cell_seg"
    else:
        # Try direct lookup first
        zarr_label_name = available_labels.get(organelle_name)

        # Case-insensitive fallback if direct lookup fails
        if zarr_label_name is None:
            organelle_lower = organelle_name.lower()
            for key, value in available_labels.items():
                if key.lower() == organelle_lower:
                    zarr_label_name = value
                    logger.debug(f"Case-insensitive match: '{organelle_name}' -> '{key}' -> '{value}'")
                    break

        if zarr_label_name is None:
            logger.warning(f"Organelle '{organelle_name}' not found in available_labels. "
                          f"Available: {list(available_labels.keys())[:10]}...")
            return None

    try:
        position = pheno_store[well]
        if "labels" not in position.zgroup:
            return None

        labels_group = position.zgroup["labels"]
        if zarr_label_name not in labels_group:
            return None

        label_array = labels_group[zarr_label_name]["0"]

        # Handle different array dimensions using BaseDataset pattern:
        # np.asarray(...).copy() with explicit slice() objects
        # This is critical for zarr v3 sharded arrays to ensure complete chunk retrieval
        if label_array.ndim == 5:
            label_crop = np.asarray(
                label_array[0:1, 0:1, 0:1, slice(y_min, y_max), slice(x_min, x_max)]
            ).copy()
            label_crop = np.squeeze(label_crop)
        elif label_array.ndim == 4:
            label_crop = np.asarray(
                label_array[0:1, 0:1, slice(y_min, y_max), slice(x_min, x_max)]
            ).copy()
            label_crop = np.squeeze(label_crop)
        elif label_array.ndim == 3:
            label_crop = np.asarray(
                label_array[0:1, slice(y_min, y_max), slice(x_min, x_max)]
            ).copy()
            label_crop = np.squeeze(label_crop)
        else:
            label_crop = np.asarray(
                label_array[slice(y_min, y_max), slice(x_min, x_max)]
            ).copy()

        return label_crop
    except Exception as e:
        logger.debug(f"Failed to load {organelle_name} for {well}: {e}")
        return None


def visualize_representative_cells(
    features: pd.DataFrame,
    df: pd.DataFrame,
    gene_col: str,
    clusters: Dict[str, Dict],
    result,
    morphology_path: Path,
    output_dir: Path,
    sanitize_filename_func,
    level: str,  # NEW: "cell", "guide", or "gene"
    cell_df: Optional[pd.DataFrame] = None,  # Cell-level df for guide/gene levels
    cell_features: Optional[pd.DataFrame] = None,  # Cell-level features for guide/gene levels
    n_items_per_cluster: int = 6,  # Number of guides/genes (or cells at cell level)
    n_cells_per_item: int = 3,  # Number of cells per guide/gene (for guide/gene level)
    top_n_features: int = 3,
    skip_complete: bool = False,  # Skip if output exists
    # Pre-built mappings from DataContext (single source of truth)
    channel_names: Optional[List[str]] = None,
    available_labels: Optional[Dict[str, str]] = None,
    label_to_channel_index: Optional[Dict[str, int]] = None,
) -> None:
    """
    Visualize representative cells for each positive control cluster.

    At cell level: Shows top N cells that best exemplify distinguishing features.
    At guide/gene level: Shows top N guides/genes, then selects the M cells from
    each that best exemplify the distinguishing phenotype (feature-based scoring).

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix (items x features) at the current level
    df : pd.DataFrame
        Full dataframe with metadata at the current level
    gene_col : str
        Column name containing gene identifiers
    clusters : dict
        Positive control clusters
    result : StageResult
        Results container
    morphology_path : Path
        Path to zarr store for loading cell images
    output_dir : Path
        Output directory for the stage
    sanitize_filename_func : callable
        Function to sanitize cluster names for filenames
    level : str
        Analysis level: "cell", "guide", or "gene"
    cell_df : pd.DataFrame, optional
        Cell-level dataframe (required for guide/gene levels to look up actual cells)
    cell_features : pd.DataFrame, optional
        Cell-level feature matrix. Required for guide/gene levels to select cells
        that best exemplify the distinguishing features. If not provided, cells
        are randomly sampled from each guide/gene.
    n_items_per_cluster : int
        Number of items to show per cluster (cells for cell level, guides/genes for other levels)
    n_cells_per_item : int
        Number of cells to show per guide/gene (only used at guide/gene level)
    top_n_features : int
        Number of top features to visualize per cluster
    """
    if not morphology_path.exists():
        logger.warning(f"Morphology path {morphology_path} not found, skipping representative cell visualization.")
        return
    
    # Ensure required columns exist (they should already be present in adata.obs)
    # Required by collect_validation_viz_data: well, bbox, cp1_label, store_key
    def validate_required_columns(df_input, df_name="df"):
        """Validate that required columns for cell visualization exist."""
        required_cols = ['well', 'bbox', 'cp1_label', 'store_key']
        missing = [col for col in required_cols if col not in df_input.columns]
        if missing:
            logger.warning(f"{df_name} missing required columns: {missing}. Cannot load cell images.")
            return False
        return True
    
    # At cell level: validate df directly
    # At guide/gene level: validate cell_df (which has the actual cell-level metadata)
    if level == "cell":
        if not validate_required_columns(df, f"{level}-level df"):
            logger.warning(f"Cannot generate representative cells - missing required columns in {level}-level data")
            return
    else:
        # Guide/gene level - need cell_df
        if cell_df is None:
            logger.warning(f"Cell-level dataframe required for {level}-level visualization, skipping.")
            return
        if not validate_required_columns(cell_df, "cell_df"):
            logger.warning("Cannot generate representative cells - missing required columns in cell-level data")
            return
    
    logger.info(f"Generating representative cell visualizations at {level} level...")
    logger.info(f"  Top {top_n_features} features, {n_items_per_cluster} {level}s per cluster")
    if level in ["guide", "gene"]:
        logger.info(f"  {n_cells_per_item} cells per {level}")
    
    # Import visualization utilities
    try:
        from ...fe_visualization import (
            collect_validation_viz_data,
            create_heatmap_rgba,
            blend_intensity_heatmap,
            _expand_mask_to_intensity,
            _compute_basic_features,
        )
        from skimage.segmentation import find_boundaries
        from iohub import open_ome_zarr
    except ImportError as e:
        logger.warning(f"Could not import visualization utilities: {e}")
        return
    
    # Representative cells output to cluster-specific directories
    for cluster_name, cluster_data in clusters.items():
        genes = cluster_data["genes"]

        # Get mask for this cluster at current level
        cluster_mask = df[gene_col].isin(genes)
        n_cluster = cluster_mask.sum()

        if n_cluster < 1:
            logger.info(f"  {cluster_name}: No {level}s found, skipping.")
            continue

        # Create cluster-specific directory
        safe_cluster_name = sanitize_filename_func(cluster_name)
        cluster_dir = output_dir / safe_cluster_name
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # Load feature importance from cluster-specific directory
        csv_path = cluster_dir / "feature_importance.csv"
        if not csv_path.exists():
            logger.warning(f"  {cluster_name}: No feature importance file found at {csv_path}, skipping.")
            continue

        feature_importance_df = pd.read_csv(csv_path)
        top_features_list = feature_importance_df.head(top_n_features)

        if top_features_list.empty:
            logger.info(f"  {cluster_name}: No distinguishing features, skipping.")
            continue

        logger.info(f"  {cluster_name}: Top features: {list(top_features_list['feature'].values[:3])}")

        # Build available_labels for organelle detection (do this once)
        if available_labels is None:
            from ...feature_extraction.fe_metadata import _discover_available_labels
            available_labels = _discover_available_labels(morphology_path)

        # ============================================================
        # LOAD TOP 3 ORGANELLES from organelle_importance.csv
        # Generate representative cells for EACH of the top 3 organelles
        # ============================================================
        organelle_importance_path = cluster_dir / "organelle_importance.csv"
        if organelle_importance_path.exists():
            organelle_importance_df = pd.read_csv(organelle_importance_path)
            # Get top 3 organelles by max_abs_d (already sorted descending)
            top_organelles = organelle_importance_df.head(3)[['organelle', 'max_feature', 'max_abs_d']].to_dict('records')
            logger.info(f"  {cluster_name}: Generating visualizations for top 3 organelles: {[o['organelle'] for o in top_organelles]}")
        else:
            # Fallback: detect single organelle from top features (old behavior)
            logger.warning(f"  {cluster_name}: No organelle_importance.csv, using single organelle detection")
            top_organelles = [{'organelle': None, 'max_feature': None, 'max_abs_d': None}]

        # Loop through top organelles (up to 3)
        for org_rank, org_info in enumerate(top_organelles, 1):
            organelle_to_load_early = None
            is_cp_organelle_early = False
            organelle_feature_name = org_info.get('max_feature')

            # If organelle specified from organelle_importance.csv, map to zarr label name
            if org_info.get('organelle'):
                org_name = org_info['organelle']
                # Try to find matching label in available_labels
                for label_key in available_labels.keys():
                    # Match by prefix (e.g., "mito" matches "mito_tomm20")
                    if label_key.lower().startswith(org_name.lower()) or org_name.lower().startswith(label_key.lower()):
                        organelle_to_load_early = label_key
                        is_cp_organelle_early = label_key.lower().startswith("cp")
                        break

                # If not found in available_labels, try direct match with the feature name
                if not organelle_to_load_early and organelle_feature_name:
                    feat_for_org = organelle_feature_name
                    if feat_for_org.startswith('network_'):
                        feat_for_org = feat_for_org[8:]

                    for org_key in available_labels.keys():
                        if feat_for_org.lower().startswith(org_key.lower() + '_'):
                            organelle_to_load_early = org_key
                            is_cp_organelle_early = org_key.lower().startswith("cp")
                            break

                if organelle_to_load_early:
                    logger.info(f"    [{org_rank}/3] Organelle '{org_info['organelle']}' -> zarr label '{organelle_to_load_early}' (CP: {is_cp_organelle_early})")
                else:
                    logger.warning(f"    [{org_rank}/3] Could not map organelle '{org_info['organelle']}' to zarr label, skipping")
                    continue
            else:
                # Fallback: detect organelle from ALL top features (old behavior)
                for feat_idx in range(len(top_features_list)):
                    if organelle_to_load_early:
                        break

                    feat_name = top_features_list.iloc[feat_idx]['feature']
                    feat_for_org = feat_name
                    if feat_for_org.startswith('network_'):
                        feat_for_org = feat_for_org[8:]

                    for org_key in available_labels.keys():
                        if feat_for_org.lower().startswith(org_key.lower() + '_'):
                            organelle_to_load_early = org_key
                            is_cp_organelle_early = org_key.lower().startswith("cp")
                            organelle_feature_name = feat_name
                            break

                    # FALLBACK: regex for CP organelles
                    if not organelle_to_load_early and feat_for_org.lower().startswith('cp'):
                        import re
                        metric_suffixes = r'_(area|perimeter|axis_major_length|axis_minor_length|eccentricity|solidity|extent|euler_number|equivalent_diameter|num_branches|num_nodes|num_endpoints|average_degree|branch_length|branch_thickness|tortuosity|distance_from|normalized_radial|angular_position)'
                        cp_match = re.match(rf'^(cp\d+_[a-z0-9_]+?){metric_suffixes}', feat_for_org, re.IGNORECASE)
                        if cp_match:
                            organelle_to_load_early = cp_match.group(1)
                            is_cp_organelle_early = True
                            organelle_feature_name = feat_name

                if organelle_to_load_early:
                    logger.info(f"    Detected organelle from features: '{organelle_to_load_early}' (CP: {is_cp_organelle_early})")

            # Create organelle-specific top_features_list if we have max_feature
            if organelle_feature_name:
                # Filter feature_importance to features from THIS organelle
                org_features = feature_importance_df[
                    feature_importance_df['feature'].str.lower().str.startswith(
                        organelle_to_load_early.lower() + '_' if organelle_to_load_early else ''
                    )
                ].head(top_n_features)
                if org_features.empty:
                    # Fallback to full list
                    org_features = top_features_list
                organelle_top_features = org_features
            else:
                organelle_top_features = top_features_list

            # Create organelle-specific subdirectory under representative_cells/
            raw_org_name = org_info.get('organelle', 'unknown') if org_info.get('organelle') else 'top_organelle'
            organelle_safe_name = sanitize_filename_func(raw_org_name)
            rep_cells_dir = cluster_dir / "representative_cells"
            organelle_dir = rep_cells_dir / organelle_safe_name
            organelle_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"      Output directory: {organelle_dir}")

            # Prepare filtered versions of dataframes for selection
            df_for_selection = df
            features_for_selection = features
            cluster_mask_for_selection = cluster_mask
            cell_df_for_selection = cell_df
            cell_features_for_selection = cell_features

            if is_cp_organelle_early:
                # Determine which df has cp_bbox (cell-level data)
                # CRITICAL: First filter to cells belonging to genes in THIS cluster, THEN check cp_bbox
                if level in ["guide", "gene"] and cell_df is not None:
                    # Filter cell_df to only cells from genes in this cluster
                    cluster_cell_mask = cell_df[gene_col].isin(genes) if gene_col in cell_df.columns else pd.Series(True, index=cell_df.index)
                    lookup_df_for_filter = cell_df[cluster_cell_mask].copy()
                else:
                    lookup_df_for_filter = df[cluster_mask].copy() if level == "cell" else None

                if lookup_df_for_filter is not None and len(lookup_df_for_filter) > 0 and 'cp_bbox' in lookup_df_for_filter.columns:
                    def _is_valid_bbox_early(bbox_val):
                        """Check if bbox value is valid (non-empty list/tuple with 4 elements)."""
                        if bbox_val is None:
                            return False
                        if isinstance(bbox_val, (list, tuple, np.ndarray)):
                            return len(bbox_val) == 4 and all(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in bbox_val)
                        if isinstance(bbox_val, str):
                            return bbox_val.strip() != '' and bbox_val.strip() != '[]'
                        return False

                    valid_cp_bbox_mask = lookup_df_for_filter['cp_bbox'].apply(_is_valid_bbox_early)
                    n_valid = valid_cp_bbox_mask.sum()
                    n_total = len(lookup_df_for_filter)

                    logger.info(f"      CP organelle '{organelle_to_load_early}': {n_valid}/{n_total} cluster cells have valid cp_bbox")

                    if n_valid == 0:
                        logger.warning(f"      {cluster_name}/{organelle_safe_name}: No cells with valid cp_bbox, skipping.")
                        continue

                    # Always filter to valid cp_bbox cells for CP organelles
                    valid_indices = lookup_df_for_filter.index[valid_cp_bbox_mask]

                    if level in ["guide", "gene"]:
                        # Filter cell_df and cell_features to ONLY valid cp_bbox cells from this cluster
                        cell_df_for_selection = cell_df.loc[cell_df.index.intersection(valid_indices)].copy()
                        if cell_features is not None:
                            cell_features_for_selection = cell_features.loc[cell_features.index.intersection(valid_indices)].copy()
                        else:
                            cell_features_for_selection = None
                        logger.info(f"      Selection pool: {len(cell_df_for_selection)} cells with valid cp_bbox")
                    else:
                        # Cell level: filter df and features
                        df_for_selection = df.loc[df.index.intersection(valid_indices)].copy()
                        features_for_selection = features.loc[features.index.intersection(valid_indices)].copy()
                        # Update cluster mask for filtered df
                        cluster_mask_for_selection = df_for_selection[gene_col].isin(genes)
                        logger.info(f"      Selection pool: {len(df_for_selection)} cells with valid cp_bbox")

            # Selection logic depends on level
            # NOTE: Use filtered versions (*_for_selection) to ensure cells with valid cp_bbox for CP organelles
            # Calculate target cell count and over-select to compensate for duplicates
            if level == "cell":
                target_cell_count = n_items_per_cluster
            else:
                target_cell_count = n_items_per_cluster * n_cells_per_item

            # Over-select by 2x to ensure we have enough after deduplication
            overselect_factor = 2

            if level == "cell":
                # Cell level: Select representative cells directly (over-select)
                selected_cell_indices = _select_representative_cells_at_cell_level(
                    df_for_selection, features_for_selection, cluster_mask_for_selection,
                    organelle_top_features, n_items_per_cluster * overselect_factor
                )
                items_to_visualize = [(None, selected_cell_indices)]  # No item grouping at cell level

            else:
                # Guide/Gene level: Select representative guides/genes, then their best cells (over-select)
                selected_items, selected_cells_per_item = _select_representative_items_at_aggregated_level(
                    df, features, cluster_mask, organelle_top_features,
                    cell_df_for_selection,  # Use filtered cell_df!
                    level,
                    n_items_per_cluster, n_cells_per_item * overselect_factor, gene_col,
                    cell_features=cell_features_for_selection,  # Use filtered cell_features!
                )

                if not selected_items:
                    logger.info(f"      {cluster_name}/{organelle_safe_name}: No valid {level}s found, skipping.")
                    continue

                items_to_visualize = list(zip(selected_items, selected_cells_per_item))

            # Flatten all selected cell indices for visualization
            # Use dict.fromkeys() to deduplicate while preserving order
            all_selected_cells = []
            seen_cells = set()
            for item_name, cell_indices in items_to_visualize:
                for cell_idx in cell_indices:
                    if cell_idx not in seen_cells:
                        all_selected_cells.append(cell_idx)
                        seen_cells.add(cell_idx)

            n_before_dedup = sum(len(ci) for _, ci in items_to_visualize)
            if len(all_selected_cells) < n_before_dedup:
                logger.info(f"      Removed {n_before_dedup - len(all_selected_cells)} duplicate cell indices")

            if not all_selected_cells:
                logger.info(f"      {cluster_name}/{organelle_safe_name}: No cells to visualize, skipping.")
                continue

            # EARLY PHYSICAL DEDUPLICATION: Remove cells with same (well, bbox) before final selection
            # This ensures we don't count the same physical cell multiple times
            # For CP organelles, use cp_bbox; for others, use regular bbox
            lookup_df_for_dedup = cell_df if level in ["guide", "gene"] else df
            is_cp_organelle_early = organelle_to_load_early and organelle_to_load_early.lower().startswith("cp")
            bbox_col_for_dedup = 'cp_bbox' if (is_cp_organelle_early and 'cp_bbox' in lookup_df_for_dedup.columns) else 'bbox'

            if 'well' in lookup_df_for_dedup.columns and bbox_col_for_dedup in lookup_df_for_dedup.columns:
                # Get cell data for deduplication
                cells_for_dedup = lookup_df_for_dedup.loc[all_selected_cells].copy()

                def _make_cell_key(row):
                    well = row.get('well', '')
                    bbox = row.get(bbox_col_for_dedup)
                    if bbox is None:
                        return None
                    if isinstance(bbox, (list, tuple, np.ndarray)):
                        bbox_tuple = tuple(bbox)
                    elif isinstance(bbox, str):
                        # Parse string bbox for comparison
                        bbox = bbox.strip()
                        if bbox.startswith('[') and bbox.endswith(']'):
                            try:
                                values = bbox[1:-1].split()
                                bbox_tuple = tuple(int(v) for v in values if v)
                            except ValueError:
                                bbox_tuple = bbox
                        else:
                            bbox_tuple = bbox
                    else:
                        bbox_tuple = str(bbox)
                    return (well, bbox_tuple)

                cells_for_dedup['_cell_key'] = cells_for_dedup.apply(_make_cell_key, axis=1)

                # Keep first occurrence of each unique physical cell (preserves ranking order)
                unique_mask = ~cells_for_dedup['_cell_key'].duplicated(keep='first')
                unique_indices = cells_for_dedup.index[unique_mask].tolist()

                n_physical_dups = len(all_selected_cells) - len(unique_indices)
                if n_physical_dups > 0:
                    logger.info(f"      Removed {n_physical_dups} physically duplicate cells (same well+{bbox_col_for_dedup})")

                all_selected_cells = unique_indices

            # Trim to target count (we over-selected to compensate for duplicates)
            if len(all_selected_cells) > target_cell_count:
                all_selected_cells = all_selected_cells[:target_cell_count]
                logger.info(f"      Trimmed to {target_cell_count} cells (target count)")

            # Check minimum cell count (target: at least 10 cells per gene group)
            MIN_CELLS_FOR_VIZ = 10
            if len(all_selected_cells) < MIN_CELLS_FOR_VIZ:
                logger.warning(f"      {cluster_name}/{organelle_safe_name}: Only {len(all_selected_cells)} unique cells available "
                              f"(target: {MIN_CELLS_FOR_VIZ}). Consider increasing n_items or n_cells_per_item.")

            # Collect visualization data for these cells
            try:
                # Determine which df to use for cell lookup
                lookup_df = cell_df if level in ["guide", "gene"] else df

                # Use pre-built mappings if provided (from DataContext), otherwise discover them
                if channel_names is None or available_labels is None or label_to_channel_index is None:
                    logger.warning("    Mappings not provided - building locally (slower)")
                    # Fallback: discover from zarr store
                    from iohub import open_ome_zarr
                    from ...feature_extraction.fe_metadata import _discover_available_labels

                    with open_ome_zarr(morphology_path, mode="r") as pheno_store:
                        channel_names = list(pheno_store.channel_names) if hasattr(pheno_store, 'channel_names') else []

                    available_labels = _discover_available_labels(morphology_path)

                    # Build basic channel index mapping
                    label_to_channel_index = {}
                    for i, ch_name in enumerate(channel_names):
                        ch_lower = ch_name.lower().replace(' ', '_').replace('-', '_')
                        label_to_channel_index[ch_lower] = i

                # Use label_to_channel_index as organelle_channel_indices
                organelle_channel_indices = label_to_channel_index
                logger.debug(f"        Discovered {len(available_labels)} organelle labels from zarr store")

                # Use the organelle detected for this iteration
                organelle_to_load = organelle_to_load_early

                if organelle_to_load:
                    logger.info(f"      Loading organelle '{organelle_to_load}' for visualization")
                else:
                    logger.warning(f"      No organelle specified for {cluster_name}/{organelle_safe_name}")

                # Use BaseDataset for FAST loading (like view_by_gene.py does)
                selected_cells_df = lookup_df.loc[list(all_selected_cells)].copy()

                # CRITICAL: For CP organelles, use cp_bbox instead of bbox
                # CP organelles are segmented in a different coordinate system (CellProfiling)
                # NOTE: cp_bbox validation was already done upstream (before cell selection),
                # so we just need to swap bbox with cp_bbox here
                is_cp_organelle = organelle_to_load and organelle_to_load.lower().startswith("cp")
                if is_cp_organelle and 'cp_bbox' in selected_cells_df.columns:
                    logger.info(f"      Using cp_bbox for CP organelle '{organelle_to_load}' ({len(selected_cells_df)} cells)")

                    # Parse cp_bbox if it's a string (e.g., "[44741 33736 44819 33901]")
                    def _parse_bbox_value(x):
                        if x is None or (isinstance(x, float) and np.isnan(x)):
                            return None
                        if isinstance(x, (list, tuple, np.ndarray)):
                            return list(x) if isinstance(x, np.ndarray) else x
                        if isinstance(x, str):
                            x = x.strip()
                            if x.startswith('[') and x.endswith(']'):
                                try:
                                    values = x[1:-1].split()
                                    return [int(v) for v in values if v]
                                except ValueError:
                                    return None
                            if not x or x == '[]':
                                return None
                        return x

                    # Swap bbox with cp_bbox for BaseDataset loading
                    selected_cells_df['original_bbox'] = selected_cells_df['bbox']
                    selected_cells_df['bbox'] = selected_cells_df['cp_bbox'].apply(_parse_bbox_value)

                    # Log how many have valid bboxes after parsing
                    n_valid_after_swap = selected_cells_df['bbox'].apply(
                        lambda x: x is not None and isinstance(x, (list, tuple)) and len(x) == 4
                    ).sum()
                    logger.info(f"      After cp_bbox swap: {n_valid_after_swap}/{len(selected_cells_df)} cells have valid bbox")

                # SAFETY NET DEDUPLICATION: Remove any remaining duplicates after cp_bbox swap
                # This catches edge cases where bbox values differ after parsing
                n_before_physical_dedup = len(selected_cells_df)
                if 'well' in selected_cells_df.columns and 'bbox' in selected_cells_df.columns:
                    # Create a hashable key from well + bbox
                    def _make_cell_key(row):
                        well = row.get('well', '')
                        bbox = row.get('bbox')
                        if bbox is None:
                            return None
                        if isinstance(bbox, (list, tuple, np.ndarray)):
                            bbox_tuple = tuple(bbox)
                        else:
                            bbox_tuple = str(bbox)
                        return (well, bbox_tuple)

                    selected_cells_df['_cell_key'] = selected_cells_df.apply(_make_cell_key, axis=1)

                    # Keep first occurrence of each unique cell
                    selected_cells_df = selected_cells_df.drop_duplicates(subset='_cell_key', keep='first')
                    selected_cells_df = selected_cells_df.drop(columns=['_cell_key'])

                    n_physical_dups = n_before_physical_dedup - len(selected_cells_df)
                    if n_physical_dups > 0:
                        logger.info(f"      Removed {n_physical_dups} physically duplicate cells (same well+bbox)")

                logger.info(f"      Collecting visualization data for {len(selected_cells_df)} selected cells...")

                # Collect viz data using BaseDataset (FAST)
                viz_data_list = _collect_viz_data_fast(
                    morphology_path=morphology_path,
                    cells_df=selected_cells_df,
                    channel_names=channel_names,
                    organelle_to_load=organelle_to_load,
                    organelle_channel_indices=organelle_channel_indices,  # Pass mapping!
                )

                if not viz_data_list:
                    logger.warning(f"      {cluster_name}/{organelle_safe_name}: No visualization data collected.")
                    continue

                # Check if output already exists (skip if --skip-complete)
                output_path = organelle_dir / "representative_cells.png"
                if skip_complete and output_path.exists():
                    logger.info(f"      {cluster_name}/{organelle_safe_name}: Skipping (output exists, --skip-complete)")
                    result.add_file(output_path)
                    continue

                # Generate canvas showing these cells with feature overlays
                _generate_representative_cell_canvas(
                    viz_data_list=viz_data_list,
                    cluster_name=f"{cluster_name} - {org_info.get('organelle', 'Top Organelle')}",
                    top_features_list=organelle_top_features,
                    output_path=output_path,
                    result=result,
                    level=level,
                    items_to_visualize=items_to_visualize if level in ["guide", "gene"] else None,
                    morphology_path=morphology_path,  # Pass morphology_path for loading organelles
                )

            except Exception as e:
                logger.warning(f"      {cluster_name}/{organelle_safe_name}: Error generating representative cells: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    logger.info(f"Representative cell visualizations saved to cluster subdirectories in {output_dir}")


def _generate_representative_cell_canvas(
    viz_data_list: list,
    cluster_name: str,
    top_features_list: pd.DataFrame,
    output_path: Path,
    result,
    level: str = "cell",
    items_to_visualize: Optional[List[tuple]] = None,
    morphology_path: Optional[Path] = None,  # NEW: Required for loading organelles
) -> None:
    """
    Generate a canvas showing representative cells with 3-panel layout.
    
    Layout for each cell (3 columns):
    1. Raw Phase2D image
    2. Organelle segmentation overlay (for the organelle with the top distinguishing feature)
    3. Heatmap overlay of the top feature values
    
    Parameters
    ----------
    viz_data_list : list
        List of viz_data dicts for selected cells
    cluster_name : str
        Name of the cluster
    top_features_list : pd.DataFrame
        Top distinguishing features (with 'feature', 'cohens_d' columns)
    output_path : Path
        Output path for the canvas PNG
    result : StageResult
        Results container
    level : str
        Analysis level ("cell", "guide", or "gene")
    items_to_visualize : list of tuples, optional
        For guide/gene level: [(item_name, cell_indices), ...]
    morphology_path : Path, optional
        Path to morphology zarr store (required for loading organelle labels)
    """
    from ...fe_visualization import (
        create_heatmap_rgba,
        blend_intensity_heatmap,
        _compute_basic_features,
    )
    from ..plotting.fe_graphs_utils import save_figure
    from skimage.segmentation import find_boundaries
    
    if not viz_data_list or len(top_features_list) == 0:
        logger.warning(f"No data to visualize for {cluster_name}")
        return
    
    # Get the top feature to determine which organelle to show
    top_feature = top_features_list.iloc[0]
    top_feat_name = top_feature['feature']
    top_cohens_d = top_feature['cohens_d']
    
    # Parse organelle from feature name (e.g., "cp1_plasma_membrane_wga_area" -> "cp1_plasma_membrane_wga")
    # Handle network features: "network_cp1_plasma_membrane_wga_..." -> "cp1_plasma_membrane_wga"
    organelle_name = viz_data_list[0].get("organelle_to_load")

    if not organelle_name:
        logger.warning(f"No organelle specified for {cluster_name}")
        return

    # Parse the per-object metric from the top feature name ONCE (same for all cells)
    feature_metric = _extract_metric_from_feature_name(top_feat_name, organelle_name)
    logger.info(f"    Per-object metric for heatmap: '{feature_metric}' (from '{top_feat_name}')")

    # Discover available labels ONCE for all cells (not per-cell!)
    available_labels = None
    if morphology_path is not None:
        from ...feature_extraction.fe_metadata import _discover_available_labels
        available_labels = _discover_available_labels(morphology_path)
    
    n_cells = len(viz_data_list)
    n_cols = 3  # Raw channel | Segmentation overlay | Feature heatmap

    # Use GridSpec to add colorbar column PER ROW for better legibility
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14, 4 * n_cells))
    # Reduced wspace (0.03) for tighter columns, hspace (0.15) for tighter rows
    gs = GridSpec(n_cells, 4, figure=fig, width_ratios=[1, 1, 1, 0.08], wspace=0.03, hspace=0.15)

    # Create axes for the 3-column grid + per-row colorbar
    axes = []
    cbar_axes = []
    for row in range(n_cells):
        axes.append([
            fig.add_subplot(gs[row, 0]),
            fig.add_subplot(gs[row, 1]),
            fig.add_subplot(gs[row, 2]),
        ])
        cbar_axes.append(fig.add_subplot(gs[row, 3]))

    # Track per-row feature values for individual colorbars
    row_feature_values = [[] for _ in range(n_cells)]

    # PERFORMANCE: Open zarr store ONCE before the loop (not per-cell!)
    from iohub import open_ome_zarr
    pheno_store = None
    if morphology_path is not None:
        pheno_store = open_ome_zarr(morphology_path, mode="r")

    for cell_idx, viz_data in enumerate(viz_data_list):
        intensity = viz_data["intensity"]
        cell_mask = viz_data.get("cell_mask")
        cp_cell_mask = viz_data.get("cp_cell_mask")
        bbox = viz_data.get("bbox")
        well = viz_data.get("well")
        crop_info = viz_data.get("crop_info", {})
        channel_names = viz_data.get("channel_names", [])
        organelle_channel_indices = viz_data.get("organelle_channel_indices", {})
        
        # Determine if this is a CP organelle and use appropriate cell mask
        is_cp_organelle = organelle_name and organelle_name.lower().startswith("cp")
        if is_cp_organelle and cp_cell_mask is not None:
            current_cell_mask = cp_cell_mask
            logger.debug(f"      Using CP cell mask for cell {cell_idx}")
        else:
            current_cell_mask = cell_mask
            logger.debug(f"      Using standard cell mask for cell {cell_idx}")
        
        # Use pre-loaded organelle labels from viz_data (loaded via BaseDataset's robust method)
        organelle_labels = viz_data.get("organelle_labels")

        # Crop/pad organelle labels to match intensity size
        if organelle_labels is not None:
            target_h, target_w = intensity.shape[1], intensity.shape[2]  # intensity is (C, H, W)
            if organelle_labels.shape != (target_h, target_w):
                h, w = organelle_labels.shape
                if h > target_h and w > target_w:
                    # Crop from center
                    start_y = (h - target_h) // 2
                    start_x = (w - target_w) // 2
                    organelle_labels = organelle_labels[start_y:start_y+target_h, start_x:start_x+target_w]
                else:
                    # Pad if smaller
                    padded = np.zeros((target_h, target_w), dtype=organelle_labels.dtype)
                    padded[:min(h, target_h), :min(w, target_w)] = organelle_labels[:min(h, target_h), :min(w, target_w)]
                    organelle_labels = padded

            # CRITICAL: Mask organelle labels to the current cell's mask
            # Without this, we show organelles from neighboring cells!
            cell_mask_2d = current_cell_mask[0] if current_cell_mask.ndim == 3 else current_cell_mask
            organelle_labels = organelle_labels * (cell_mask_2d > 0)
        
        # Get the failure reason if organelle loading failed
        organelle_failure_reason = viz_data.get("organelle_load_failure")

        if organelle_labels is None or not np.any(organelle_labels > 0):
            # Build a more detailed warning message
            if organelle_failure_reason:
                logger.warning(f"      Cell {cell_idx} organelle failed: {organelle_failure_reason}")
            else:
                logger.warning(f"      Cell {cell_idx} organelle labels empty after masking to cell boundary")
            organelle_labels = None
        else:
            # Log unique objects, not max ID!
            n_unique = len(np.unique(organelle_labels[organelle_labels > 0]))
            logger.info(f"      ✓ Loaded organelle for cell {cell_idx}: shape={organelle_labels.shape}, n_objects={n_unique}")
        
        # Get the correct channel for this organelle using the mapping
        # Special case: "cell" (cell_seg) doesn't have a fluorescence channel, use Phase2D
        loaded_channels = viz_data.get("loaded_channels", list(range(intensity.shape[0])))

        if organelle_name == "cell":
            original_channel_idx = 0  # Phase2D is always channel 0
            channel_name = "Phase2D"
        else:
            # Use the mapping for real organelles
            from ...fe_visualization import _get_channel_index_for_organelle
            original_channel_idx = _get_channel_index_for_organelle(organelle_name, channel_names, organelle_channel_indices)
            channel_name = channel_names[original_channel_idx] if original_channel_idx < len(channel_names) else f"ch{original_channel_idx}"

        # Convert original channel index to loaded channel index
        # (we only loaded a subset of channels for speed)
        if original_channel_idx in loaded_channels:
            loaded_idx = loaded_channels.index(original_channel_idx)
        else:
            # Fallback to Phase2D (channel 0) if requested channel wasn't loaded
            loaded_idx = 0
            logger.debug(f"      Channel {original_channel_idx} not loaded, using Phase2D")

        # SAFETY: Verify loaded_idx is within bounds of actual intensity data
        # This can fail if BaseDataset returns fewer channels than requested (e.g., channel not in zarr)
        actual_n_channels = intensity.shape[0] if intensity.ndim == 3 else 1
        if loaded_idx >= actual_n_channels:
            logger.warning(f"      Channel index {loaded_idx} out of bounds (intensity has {actual_n_channels} channels). "
                          f"Expected channels {loaded_channels}, falling back to Phase2D (ch 0)")
            loaded_idx = 0
            channel_name = "Phase2D"  # Update channel name for display

        # Get intensity for the selected channel
        channel_img = intensity[loaded_idx] if intensity.ndim == 3 else intensity
        vmin = np.percentile(channel_img, 1)
        vmax = np.percentile(channel_img, 99)

        # Handle blank images (all zeros or constant) to avoid division by zero
        if vmax <= vmin:
            logger.warning(f"      Cell {cell_idx}: Blank/constant intensity (vmin={vmin}, vmax={vmax})")
            channel_norm = np.zeros_like(channel_img, dtype=np.float32)
        else:
            channel_norm = np.clip((channel_img - vmin) / (vmax - vmin), 0, 1)

        # Convert to RGB for consistent display
        channel_rgb = np.stack([channel_norm] * 3, axis=-1)
        
        # Get cell mask (use the correct one based on organelle type)
        cell_mask_2d = current_cell_mask[0] if current_cell_mask.ndim == 3 else current_cell_mask
        
        boundary_overlay = np.zeros((*channel_norm.shape, 4))
        if np.any(cell_mask_2d):
            boundary = find_boundaries(cell_mask_2d > 0, mode="inner")
            boundary_overlay[boundary] = [0, 1, 1, 1]  # Cyan
        
        # Get metadata
        gene = crop_info.get("gene_name", "?")
        
        # Panel 1: Raw channel image for the organelle
        ax = axes[cell_idx][0]
        ax.imshow(channel_rgb)
        ax.imshow(boundary_overlay)
        ax.axis("off")
        ax.set_title(f"{channel_name}\n{well} | {gene}", fontsize=10)
        
        # Panel 2: Organelle segmentation overlay (FILLED, masked to cell)
        ax = axes[cell_idx][1]
        
        if organelle_labels is not None and np.any(organelle_labels > 0):
            # Mask organelle labels to cell boundary
            org_labels_masked = organelle_labels * (cell_mask_2d > 0)
            
            if np.any(org_labels_masked > 0):
                # Create filled colored overlay for segmentation (like in organelle_seg/visualizations.py)
                # Use random colors for each UNIQUE object (not max ID!)
                np.random.seed(42)
                unique_labels = np.unique(org_labels_masked[org_labels_masked > 0])
                n_objects = len(unique_labels)
                seg_overlay = np.zeros((*channel_norm.shape, 3))
                
                # Assign random color to each unique label
                for obj_id in unique_labels:
                    color = np.random.rand(3)
                    seg_overlay[org_labels_masked == obj_id] = color
                
                # Blend with base image
                alpha = 0.5
                label_mask = org_labels_masked > 0
                blended_seg = channel_rgb.copy()
                blended_seg[label_mask] = (1 - alpha) * channel_rgb[label_mask] + alpha * seg_overlay[label_mask]
                ax.imshow(blended_seg)
                ax.imshow(boundary_overlay)  # Add boundary on top
                ax.set_title(f"Segmentation\n{organelle_name} ({n_objects} objects)", fontsize=10)
            else:
                # No objects in cell
                ax.imshow(channel_rgb)
                ax.imshow(boundary_overlay)
                ax.set_title(f"Segmentation\n{organelle_name} (no objs in cell)", fontsize=10)
        else:
            # No organelle data - show specific reason
            ax.imshow(channel_rgb)
            ax.imshow(boundary_overlay)
            # Truncate long failure reasons for display
            fail_reason = organelle_failure_reason or "no data"
            if len(fail_reason) > 30:
                fail_reason = fail_reason[:27] + "..."
            ax.set_title(f"Segmentation\n{organelle_name or 'N/A'} ({fail_reason})", fontsize=9)
        
        ax.axis("off")
        
        # Panel 3: Feature heatmap overlay (using fe_visualization.py methods)
        ax = axes[cell_idx][2]

        if organelle_labels is not None and np.any(organelle_labels > 0):
            # Mask to cell boundary
            org_labels_masked = organelle_labels * (cell_mask_2d > 0)

            if np.any(org_labels_masked > 0):
                # Compute per-object features on-demand for the metric
                # (feature_metric was parsed once before the loop)
                features_df = _compute_per_object_features_for_metric(
                    org_labels_masked,
                    feature_metric,
                    cell_mask_2d,
                    viz_data.get("organelle_labels", {}),
                )

                # Collect feature values for this row's colorbar
                if feature_metric in features_df.columns:
                    row_feature_values[cell_idx].extend(features_df[feature_metric].values)
                
                # Create heatmap using fe_visualization methods
                heatmap_rgba = create_heatmap_rgba(
                    org_labels_masked,
                    features_df,
                    feature_metric,
                    cmap_name="viridis",
                )
                
                # Blend with organelle channel (use the 2D intensity, not normalized RGB)
                blended = blend_intensity_heatmap(channel_img, heatmap_rgba, alpha=0.7)
                
                # Add boundary overlay ON TOP of the blended image
                # Convert to RGBA if needed
                if blended.shape[-1] == 3:
                    blended_rgba = np.dstack([blended, np.ones(blended.shape[:2])])
                else:
                    blended_rgba = blended.copy()
                
                # Overlay boundary
                boundary_mask = boundary_overlay[..., 3] > 0
                blended_rgba[boundary_mask] = boundary_overlay[boundary_mask]
                
                ax.imshow(blended_rgba)
                ax.set_title(f"Feature: {top_feat_name}\n(Cohen's d={top_cohens_d:.2f})", fontsize=10)
            else:
                # No objects in cell
                ax.imshow(channel_rgb)
                ax.set_title(f"Feature: {top_feat_name}\n(no objs in cell)", fontsize=10)
        else:
            # Organelle not available, just show channel with failure reason
            ax.imshow(channel_rgb)
            fail_reason = organelle_failure_reason or "no data"
            if len(fail_reason) > 30:
                fail_reason = fail_reason[:27] + "..."
            ax.set_title(f"Feature: {top_feat_name}\n({fail_reason})", fontsize=9)
        
        ax.axis("off")
    
    # Add per-row colorbars for feature heatmaps (more legible than global)
    from matplotlib import cm
    for row_idx, feat_values in enumerate(row_feature_values):
        cbar_ax = cbar_axes[row_idx]
        if feat_values:
            vmin = min(feat_values)
            vmax = max(feat_values)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.ax.tick_params(labelsize=8)
            # Show label on ALL rows for clarity
            cbar.set_label(feature_metric, fontsize=8, weight='bold')
        else:
            # Hide empty colorbar axes
            cbar_ax.axis('off')

    # Close the zarr store after the loop
    if pheno_store is not None:
        pheno_store.close()

    # Adjust layout: reduce space from title to first row
    fig.suptitle(f"Representative Cells: {cluster_name}", fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.95)  # Bring content closer to title
    save_figure(fig, output_path, dpi=150)
    result.add_file(output_path)
    # Log relative path from parent's parent (shows cluster/representative_cells/organelle/file.png)
    try:
        rel_path = output_path.relative_to(output_path.parent.parent.parent.parent)
    except ValueError:
        rel_path = output_path
    logger.info(f"    Saved: {rel_path}")



# ============================================================================
# Helper Functions for Item/Cell Selection
# ============================================================================

def _select_representative_cells_at_cell_level(
    df: pd.DataFrame,
    features: pd.DataFrame,
    cluster_mask: np.ndarray,
    top_features_list: pd.DataFrame,
    n_cells: int,
) -> np.ndarray:
    """
    Select representative cells at cell level based on feature profile.
    
    Returns array of cell indices.
    """
    cluster_df = df.loc[cluster_mask].copy()
    cluster_features = features.loc[cluster_mask].copy()
    
    # Score each cell by how well it matches the distinguishing feature profile
    cell_scores = np.zeros(len(cluster_df))
    for _, feat_row in top_features_list.iterrows():
        feat_name = feat_row['feature']
        cohens_d = feat_row['cohens_d']
        
        if feat_name not in cluster_features.columns:
            continue
        
        feat_values = cluster_features[feat_name].values
        feat_mean = np.nanmean(feat_values)
        feat_std = np.nanstd(feat_values)
        
        if feat_std == 0:
            continue
        
        # Z-score: how extreme is this cell for this feature?
        z_scores = (feat_values - feat_mean) / feat_std
        
        # Weight by Cohen's d (sign matters: positive = enriched, negative = depleted)
        if cohens_d > 0:
            cell_scores += z_scores * abs(cohens_d)
        else:
            cell_scores += (-z_scores) * abs(cohens_d)
    
    # Get top cells by composite score
    valid_scores = ~np.isnan(cell_scores)
    if valid_scores.sum() == 0:
        return np.array([])
    
    n_to_select = min(n_cells, valid_scores.sum())
    top_indices = np.argsort(cell_scores[valid_scores])[-n_to_select:][::-1]
    selected_cell_indices = cluster_df.index[valid_scores].values[top_indices]
    
    return selected_cell_indices


def _map_aggregated_feature_to_cell_level(feature_name: str) -> str:
    """
    Map an aggregated feature name (guide/gene level) to its cell-level equivalent.

    Aggregated features have suffixes like '_median', '_mean', '_std' added during
    aggregation. Cell-level features don't have these outer aggregation suffixes.

    Examples:
    - 'cp1_mito_area_mean_median' -> 'cp1_mito_area_mean' (guide level)
    - 'cp1_mito_area_mean_median_median' -> 'cp1_mito_area_mean' (gene level, double suffix)
    """
    # Aggregation suffixes added during guide/gene level aggregation
    agg_suffixes = ['_median', '_mean', '_std', '_min', '_max', '_sum', '_count']

    result = feature_name
    # Strip outer aggregation suffixes (may have multiple for gene level)
    changed = True
    while changed:
        changed = False
        for suffix in agg_suffixes:
            if result.endswith(suffix):
                result = result[:-len(suffix)]
                changed = True
                break

    return result


def select_top_cells_by_features(
    cell_indices: np.ndarray,
    cell_features: pd.DataFrame,
    top_features_list: pd.DataFrame,
    n_cells: int,
    map_to_cell_level: bool = True,
) -> np.ndarray:
    """
    Select top N cells from a set of cell indices based on distinguishing features.

    Scores each cell by how well it exemplifies the distinguishing feature profile
    using z-score weighting by Cohen's d. Cells with the highest composite scores
    are selected.

    This is a reusable utility for selecting representative cells that can be used
    by both positive control visualization and organelle cluster visualization.

    Parameters
    ----------
    cell_indices : np.ndarray
        Array of cell indices to select from
    cell_features : pd.DataFrame
        Cell-level feature matrix (cells x features)
    top_features_list : pd.DataFrame
        Top distinguishing features with 'feature' and 'cohens_d' columns
    n_cells : int
        Number of cells to select
    map_to_cell_level : bool
        If True, map aggregated feature names to cell-level names (strip _median etc.)
        Set to False if features are already cell-level names.

    Returns
    -------
    np.ndarray
        Array of selected cell indices (top N by feature score)
    """
    if len(cell_indices) == 0:
        return np.array([])

    if cell_features is None or len(cell_features) == 0:
        # No features - return random sample
        n_to_select = min(n_cells, len(cell_indices))
        return np.random.choice(cell_indices, n_to_select, replace=False)

    # Build list of cell-level feature names with their Cohen's d
    cell_level_features = []
    for _, feat_row in top_features_list.iterrows():
        if map_to_cell_level:
            cell_feat = _map_aggregated_feature_to_cell_level(feat_row['feature'])
        else:
            cell_feat = feat_row['feature']
        cell_level_features.append({
            'feature': cell_feat,
            'cohens_d': feat_row['cohens_d'],
        })

    # Score cells by how well they match the distinguishing feature profile
    cell_scores = np.zeros(len(cell_indices))

    # Track which cells have valid (non-NaN) values for the TOP feature
    # This helps ensure we select cells that actually have the organelle visible
    top_feature_valid_mask = np.ones(len(cell_indices), dtype=bool)
    if len(cell_level_features) > 0:
        top_feat_name = cell_level_features[0]['feature']
        if top_feat_name in cell_features.columns:
            try:
                top_feat_values = cell_features.loc[cell_indices, top_feat_name].values
                top_feature_valid_mask = ~np.isnan(top_feat_values)
                n_valid = top_feature_valid_mask.sum()
                if n_valid < len(cell_indices):
                    logger.debug(f"      {n_valid}/{len(cell_indices)} cells have valid values for top feature '{top_feat_name}'")
            except KeyError:
                pass

    for feat_info in cell_level_features:
        cell_feat_name = feat_info['feature']
        cohens_d = feat_info['cohens_d']

        if cell_feat_name not in cell_features.columns:
            continue

        # Get feature values for the cells we're selecting from
        try:
            feat_values = cell_features.loc[cell_indices, cell_feat_name].values
        except KeyError:
            # Some cell indices might not be in cell_features
            continue

        # Use GLOBAL statistics (all cells) to find cells most different from population
        global_mean = cell_features[cell_feat_name].mean()
        global_std = cell_features[cell_feat_name].std()

        if global_std == 0 or np.isnan(global_std):
            continue

        # Z-score relative to global population
        z_scores = (feat_values - global_mean) / global_std

        # Weight by Cohen's d (sign matters: positive = enriched, negative = depleted)
        if cohens_d > 0:
            cell_scores += np.nan_to_num(z_scores * abs(cohens_d), 0)
        else:
            cell_scores += np.nan_to_num((-z_scores) * abs(cohens_d), 0)

    # Select top cells by composite score
    # PREFER cells that have valid (non-NaN) values for the top feature
    # This ensures we visualize cells that actually have the organelle segmented
    valid_scores = ~np.isnan(cell_scores)
    n_to_select = min(n_cells, len(cell_indices))

    # First, try to select from cells with valid top feature values
    valid_with_top_feat = valid_scores & top_feature_valid_mask
    if valid_with_top_feat.sum() >= n_to_select:
        # Enough cells with valid top feature - select from those
        top_cell_indices = np.argsort(cell_scores[valid_with_top_feat])[-n_to_select:][::-1]
        return cell_indices[valid_with_top_feat][top_cell_indices]
    elif valid_with_top_feat.sum() > 0:
        # Some valid cells but not enough - take all valid ones plus top-scored others
        n_from_valid = valid_with_top_feat.sum()
        valid_cells = cell_indices[valid_with_top_feat][np.argsort(cell_scores[valid_with_top_feat])[::-1]]
        n_remaining = n_to_select - n_from_valid
        # Get top-scored cells from those without valid top feature
        other_mask = valid_scores & ~top_feature_valid_mask
        if other_mask.sum() > 0:
            other_cells = cell_indices[other_mask][np.argsort(cell_scores[other_mask])[-n_remaining:][::-1]]
            return np.concatenate([valid_cells, other_cells])
        return valid_cells
    elif valid_scores.sum() > 0:
        # No cells with valid top feature - fall back to any valid scores
        top_cell_indices = np.argsort(cell_scores[valid_scores])[-n_to_select:][::-1]
        return cell_indices[valid_scores][top_cell_indices]
    else:
        # Fallback to random if scoring failed
        return np.random.choice(cell_indices, n_to_select, replace=False)


def _select_representative_items_at_aggregated_level(
    df: pd.DataFrame,
    features: pd.DataFrame,
    cluster_mask: np.ndarray,
    top_features_list: pd.DataFrame,
    cell_df: pd.DataFrame,
    level: str,
    n_items: int,
    n_cells_per_item: int,
    gene_col: str,
    cell_features: Optional[pd.DataFrame] = None,
) -> tuple:
    """
    Select representative guides/genes at aggregated level, then their best cells.

    Two-step selection:
    1. Score guides/genes by their aggregated feature profile
    2. For each selected guide/gene, score its cells by cell-level features
       to find cells that best exemplify the distinguishing phenotype

    Parameters
    ----------
    df : pd.DataFrame
        Guide/gene level dataframe with metadata
    features : pd.DataFrame
        Guide/gene level feature matrix
    cluster_mask : np.ndarray
        Boolean mask for items in this cluster
    top_features_list : pd.DataFrame
        Top distinguishing features (with 'feature', 'cohens_d' columns)
    cell_df : pd.DataFrame
        Cell-level dataframe with metadata
    level : str
        "guide" or "gene"
    n_items : int
        Number of guides/genes to select
    n_cells_per_item : int
        Number of cells to select per guide/gene
    gene_col : str
        Column name for gene identifier
    cell_features : pd.DataFrame, optional
        Cell-level feature matrix. If provided, cells are selected by feature
        scoring. If None, cells are randomly sampled.

    Returns
    -------
    tuple
        (selected_items, selected_cells_per_item) - list of item names and
        list of cell index arrays
    """
    # Column to group by (barcode for guide, gene_name for gene)
    if level == "guide":
        item_col = "barcode"  # Assuming guides have barcode column
    else:  # gene
        item_col = gene_col

    if item_col not in df.columns:
        logger.warning(f"Column {item_col} not found in {level}-level df")
        return [], []

    cluster_df = df.loc[cluster_mask].copy()
    cluster_features = features.loc[cluster_mask].copy()

    # Score each guide/gene by aggregated feature profile
    item_scores = {}
    for item_name, item_group in cluster_df.groupby(item_col):
        item_indices = item_group.index
        item_features = cluster_features.loc[item_indices]

        # Compute composite score for this item (mean of its feature values)
        item_score = 0
        for _, feat_row in top_features_list.iterrows():
            feat_name = feat_row['feature']
            cohens_d = feat_row['cohens_d']

            if feat_name not in item_features.columns:
                continue

            feat_values = item_features[feat_name].values
            feat_mean = np.nanmean(feat_values)

            # For aggregated level, features are already means, so just use them directly
            # Weight by Cohen's d direction
            if cohens_d > 0:
                item_score += feat_mean * abs(cohens_d)
            else:
                item_score += (-feat_mean) * abs(cohens_d)

        item_scores[item_name] = item_score

    if not item_scores:
        return [], []

    # Select top items
    n_to_select = min(n_items, len(item_scores))
    top_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:n_to_select]
    selected_items = [item_name for item_name, _ in top_items]

    # For each selected item, find its top cells using feature-based scoring
    selected_cells_per_item = []
    # Track already-selected cells to avoid duplicates across items
    already_selected = set()

    for item_name in selected_items:
        # Find cells belonging to this item in cell_df
        if item_col == "barcode":
            # Guide level: match by barcode
            if "barcode" not in cell_df.columns:
                logger.warning("barcode column not found in cell_df")
                selected_cells_per_item.append(np.array([]))
                continue
            item_cell_mask = cell_df["barcode"] == item_name
        else:
            # Gene level: match by gene_name
            if gene_col not in cell_df.columns:
                logger.warning(f"{gene_col} column not found in cell_df")
                selected_cells_per_item.append(np.array([]))
                continue
            item_cell_mask = cell_df[gene_col] == item_name

        item_cell_indices = cell_df.index[item_cell_mask].values

        # Exclude cells already selected for other items
        available_cells = np.array([c for c in item_cell_indices if c not in already_selected])

        if len(available_cells) == 0:
            logger.debug(f"    {item_name}: no available cells (all {len(item_cell_indices)} already selected)")
            selected_cells_per_item.append(np.array([]))
            continue

        n_cells_to_select = min(n_cells_per_item, len(available_cells))

        # Use the shared helper for feature-based cell selection
        selected_cell_indices = select_top_cells_by_features(
            cell_indices=available_cells,
            cell_features=cell_features,
            top_features_list=top_features_list,
            n_cells=n_cells_to_select,
            map_to_cell_level=True,  # Map aggregated feature names to cell-level
        )

        # Add to already_selected set
        already_selected.update(selected_cell_indices)

        if cell_features is not None and len(cell_features) > 0:
            logger.debug(f"    {item_name}: selected {len(selected_cell_indices)} cells by feature scoring")
        else:
            logger.debug(f"    {item_name}: random selection (no cell_features provided)")

        selected_cells_per_item.append(selected_cell_indices)

    return selected_items, selected_cells_per_item
