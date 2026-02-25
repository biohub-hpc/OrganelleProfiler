"""
Tiled Processing for Organelle Segmentation
============================================

This module provides functions for tiled processing of large microscopy images
using Frangi vesselness filter or LoG blob detection.

The tiled approach:
1. Divides large images into overlapping tiles
2. Processes tiles in parallel using joblib workers
3. Stitches results with Union-Find label merging

Key functions:
- _process_single_frangi_tile: Process a single tile (worker function)
- _stitch_tiled_labels_pass2: Stitch tiles with Union-Find merging
- segment_position_frangi_tiled: Full tiled segmentation pipeline
- segment_position_frangi: Entry point (wraps tiled pipeline)
"""

import time
from pathlib import Path

import numpy as np
import scipy.ndimage as scipy_ndi
from iohub import open_ome_zarr
from skimage.filters import frangi as skimage_frangi
from skimage.exposure import equalize_adapthist
from tqdm import tqdm

from ops_utils.hpc.resource_manager import get_optimal_workers
from organelle_profiler.organelle_seg.visualizations import (
    _save_tiled_debug_images,
)
from organelle_profiler.organelle_seg.frangi import (
    compute_frangi_threshold,
)
from organelle_profiler.organelle_seg.postprocessing import (
    postprocess_vesicular_mask,
    postprocess_nucleoli_mask,
    postprocess_tubular_mask,
    watershed_label,
)

from .configs import (
    um_to_sigmas,
)
from .naming import (
    get_output_label_name,
)
from .metadata import (
    _build_vesselness_metadata,
    _build_segmentation_metadata,
)
from ops_utils.io.zarr_labels import (
    _update_labels_metadata,
)
from .blob_detection import (
    _segment_blob_log,
)


def _process_single_frangi_tile(
    tile_info: dict,
    source_zarr_path: str,
    pos_path: str,
    channel_index: int,
    frangi_params: dict,
    pixel_resolution: dict,
    use_clahe: bool,
    clahe_params: dict,
    post_clahe_smoothing_sigma: float,
    frangi_postprocess: bool,
    input_mask_name: str = None,
    nucleoli_method: str = None,
    vesicular_method: str = None,
    output_label_name: str = None,
    save_vesselness: bool = False,
    write_to_zarr: bool = False,
    tile_overlap: int = 256,
    n_tiles_y: int = 1,
    n_tiles_x: int = 1,
    output_zarr_path: str = None,
) -> dict:
    """
    Process a single tile with Frangi filter or blob detection.

    This is a standalone worker function designed to be called by joblib workers
    in parallel. Each worker loads the tile, runs Frangi filter (or LoG blob
    detection), and writes ONLY the core (non-overlapping) region to zarr.

    Core Region Strategy:
    - Each tile processes the full tile_size (with overlap for context)
    - But writes ONLY its core region = center part excluding overlap margins
    - First/last tiles in each dimension include the edge margin
    - This ensures each pixel is written by exactly ONE tile (no race condition)

    Args:
        tile_info: Dict with tile coordinates (tile_idx, ty, tx, y_start_tile, x_start_tile,
                   y_end_tile, x_end_tile, src_y_start, src_y_end, src_x_start, src_x_end)
        source_zarr_path: Path to source zarr
        pos_path: Position path like "A/1/0"
        channel_index: Index of channel to segment
        frangi_params: Parameters for Frangi filter (or blob params for LoG)
        pixel_resolution: Dict with Z, Y, X resolution in um
        use_clahe: Whether to apply CLAHE
        clahe_params: CLAHE parameters
        post_clahe_smoothing_sigma: Post-CLAHE smoothing sigma
        frangi_postprocess: Whether to apply postprocessing
        input_mask_name: Optional mask name to constrain segmentation
        nucleoli_method: For nucleoli: "blob" for LoG, "frangi" for Frangi
        vesicular_method: For vesicles: "blob" for LoG, "frangi" for Frangi
        output_label_name: Name for output labels (required if write_to_zarr=True)
        save_vesselness: Whether to also save vesselness map
        write_to_zarr: If True, write results directly to zarr with locking
        tile_overlap: Overlap between adjacent tiles in pixels
        n_tiles_y: Total number of tiles in Y dimension
        n_tiles_x: Total number of tiles in X dimension
        output_zarr_path: Path to write output (defaults to source_zarr_path if None).
            Used in preview mode to write to temp zarr instead of original.

    Returns:
        Dict with tile_info, vesselness, binary_mask, and labels arrays
        (or minimal dict if write_to_zarr=True to save memory)
    """
    try:
        tile_idx = tile_info["tile_idx"]
        ty, tx = tile_info["ty"], tile_info["tx"]
        src_y_start = tile_info["src_y_start"]
        src_y_end = tile_info["src_y_end"]
        src_x_start = tile_info["src_x_start"]
        src_x_end = tile_info["src_x_end"]

        # Load tile from zarr
        input_mask_tile = None
        with open_ome_zarr(source_zarr_path, mode="r") as ds:
            source_pos = ds[pos_path]
            image_array = source_pos["0"]
            # Load tile: [T, C, Z, Y, X] -> squeeze to (Y, X) for 2D
            tile_data = np.squeeze(
                np.asarray(image_array[0, channel_index, :, src_y_start:src_y_end, src_x_start:src_x_end])
            )

            # Load input mask tile if specified (for nucleoli segmentation)
            if input_mask_name:
                labels_group = source_pos.zgroup.get("labels", None)
                if labels_group is not None and input_mask_name in labels_group:
                    mask_array = labels_group[input_mask_name]["0"]
                    input_mask_tile = np.squeeze(
                        np.asarray(mask_array[0, 0, :, src_y_start:src_y_end, src_x_start:src_x_end])
                    )

        # Apply input mask BEFORE other processing (for nucleoli segmentation)
        # Note: We do NOT erode the input mask here - erosion should be applied to
        # the final labels instead (Frangi detects edges, so eroding the mask doesn't help)
        if input_mask_tile is not None:
            # Ensure mask matches tile_data shape (edge tiles may have 1px difference)
            if input_mask_tile.shape != tile_data.shape:
                # Crop or pad mask to match tile_data
                mask_h, mask_w = input_mask_tile.shape
                tile_h, tile_w = tile_data.shape
                if mask_h != tile_h or mask_w != tile_w:
                    # Create new mask matching tile_data shape
                    new_mask = np.zeros(tile_data.shape, dtype=input_mask_tile.dtype)
                    copy_h = min(mask_h, tile_h)
                    copy_w = min(mask_w, tile_w)
                    new_mask[:copy_h, :copy_w] = input_mask_tile[:copy_h, :copy_w]
                    input_mask_tile = new_mask
            binary_mask_input = input_mask_tile > 0
            tile_data = tile_data.astype(np.float32)
            tile_data[~binary_mask_input] = 0.0

        # Apply CLAHE if needed (no tiling - tile is already small enough)
        # IMPORTANT: Match org_seg_old2.py EXACTLY - normalize each tile to [0,1], apply CLAHE, scale to uint16
        if use_clahe:
            if clahe_params is None:
                clahe_params = {"clip_limit": 0.03}

            clip_limit = clahe_params.get("clip_limit", 0.03)
            kernel_size = clahe_params.get("kernel_size", None)

            original_dtype = tile_data.dtype
            original_min, original_max = None, None

            if not np.issubdtype(original_dtype, np.integer):
                original_min, original_max = np.min(tile_data), np.max(tile_data)

            # Normalize to [0, 1] for CLAHE (matches sweep script exactly)
            tile_min, tile_max = tile_data.min(), tile_data.max()
            if tile_max > tile_min:
                tile_normalized = (tile_data - tile_min) / (tile_max - tile_min)
            else:
                tile_normalized = np.zeros_like(tile_data, dtype=np.float64)

            # Apply CLAHE directly (tile is small enough)
            # Keep as [0,1] float for Frangi (matching sweep script exactly)
            tile_data = equalize_adapthist(
                tile_normalized,
                kernel_size=kernel_size,
                clip_limit=clip_limit,
            )

            # Optional Gaussian smoothing (on [0,1] data)
            if post_clahe_smoothing_sigma > 0:
                tile_data = scipy_ndi.gaussian_filter(tile_data, sigma=post_clahe_smoothing_sigma)

        # Choose segmentation method
        pixel_res_um = pixel_resolution.get("y", pixel_resolution.get("x", 0.108))

        if nucleoli_method == "blob" and input_mask_tile is not None:
            # Use LoG blob detection for nucleoli (with nuclear mask)
            labeled_mask = _segment_blob_log(
                tile_data=tile_data,
                pixel_resolution_um=pixel_res_um,
                blob_params=frangi_params,  # frangi_params contains blob params when nucleoli_method="blob"
                mask=input_mask_tile,
            )
            # Create placeholder vesselness and binary for consistency
            vesselness_map = np.zeros_like(tile_data, dtype=np.float32)
            binary_mask = (labeled_mask > 0).astype(np.uint8)
        elif vesicular_method == "blob":
            # Use LoG blob detection for vesicles (no mask - detect everywhere)
            # Check if this is vesicular_dark (invert image for dark blobs)
            invert_image = frangi_params.get("black_ridges", False)
            labeled_mask = _segment_blob_log(
                tile_data=tile_data,
                pixel_resolution_um=pixel_res_um,
                blob_params=frangi_params,  # frangi_params contains blob params when vesicular_method="blob"
                mask=None,
                invert=invert_image,
            )
            # Create placeholder vesselness and binary for consistency
            vesselness_map = np.zeros_like(tile_data, dtype=np.float32)
            binary_mask = (labeled_mask > 0).astype(np.uint8)
        else:
            # Run Frangi filter using skimage.frangi (matching sweep script exactly)
            # tile_data is already [0,1] from CLAHE - use directly like sweep script
            # Get Frangi params - use um_to_sigmas matching sweep script exactly
            min_r = frangi_params.get("min_radius_um", 0.2)
            max_r = frangi_params.get("max_radius_um", 1.5)
            num_sigmas = frangi_params.get("num_sigma", 5)
            black_ridges = frangi_params.get("black_ridges", False)
            pixel_size_um = pixel_resolution.get("X", 0.1625)

            # Convert radius to sigmas using the sweep script formula (radius/pixel_size directly)
            sigmas = um_to_sigmas(min_r, max_r, pixel_size_um, num_sigmas=num_sigmas)

            # Use skimage.frangi directly (matching sweep script behavior exactly)
            # tile_data is already [0,1] from CLAHE output
            vesselness_map = skimage_frangi(
                tile_data,
                sigmas=sigmas,
                black_ridges=black_ridges
            )

            # Threshold and label using fixed or dynamic thresholding
            if np.any(vesselness_map > 0):
                # Check for fixed threshold first, otherwise use dynamic
                fixed_threshold = frangi_params.get("threshold", 0.01)
                if fixed_threshold is not None:
                    # Use fixed threshold directly
                    threshold = fixed_threshold
                else:
                    # Use dynamic thresholding with threshold_mult
                    threshold_mult = frangi_params.get("threshold_mult", 0.01)
                    threshold = compute_frangi_threshold(vesselness_map, threshold_mult=threshold_mult, xp=np)
                binary_mask = vesselness_map > threshold

                # Use frangi_postprocess parameter (or fall back to config if not explicitly set)
                # Priority: frangi_postprocess param > frangi_params["postprocess"] > False
                do_postprocess = frangi_postprocess or frangi_params.get("postprocess", False)
                if do_postprocess:
                    is_3d = binary_mask.ndim == 3

                    # Check if this is nucleoli segmentation (needs aggressive rounding)
                    is_nucleoli = nucleoli_method is not None

                    # Get structure type from params (set by get_frangi_params)
                    structure_type = frangi_params.get("structure_type", None)
                    is_vesicular = structure_type in ("vesicular", "vesicular_dark")
                    is_tubular = structure_type == "tubular"

                    if is_nucleoli and not is_3d:
                        # Use aggressive nucleoli post-processing for large round structures
                        # Extract postprocess params from config
                        pp_min_size = frangi_params.get("min_object_size", 20)
                        pp_do_opening = frangi_params.get("postprocess_opening", True)
                        pp_opening_radius = frangi_params.get("postprocess_opening_radius", 1)
                        pp_do_closing = frangi_params.get("postprocess_closing", True)
                        pp_closing_radius = frangi_params.get("postprocess_closing_radius", 3)
                        print(f"    [POSTPROCESS] Applying nucleoli postprocess (opening={pp_do_opening}/r={pp_opening_radius}, closing={pp_do_closing}/r={pp_closing_radius})")
                        binary_mask = postprocess_nucleoli_mask(
                            binary_mask,
                            min_size=pp_min_size,
                            do_opening=pp_do_opening,
                            opening_radius=pp_opening_radius,
                            do_closing=pp_do_closing,
                            closing_radius=pp_closing_radius,
                        )
                    elif is_tubular:
                        # Use helper function for tubular post-processing
                        # Extract postprocess params from config
                        pp_min_size = frangi_params.get("min_object_size", 5)
                        pp_do_opening = frangi_params.get("postprocess_opening", True)
                        pp_opening_size = frangi_params.get("postprocess_opening_size", 2)
                        pp_fill_holes = frangi_params.get("postprocess_fill_holes", False)
                        print(f"    [POSTPROCESS] Applying tubular postprocess (opening={pp_do_opening}/size={pp_opening_size}, fill_holes={pp_fill_holes}, min_size={pp_min_size})")
                        binary_mask = postprocess_tubular_mask(
                            binary_mask,
                            min_size=pp_min_size,
                            do_opening=pp_do_opening,
                            opening_size=pp_opening_size,
                            do_fill_holes=pp_fill_holes,
                        )
                    elif is_vesicular and not is_3d:
                        # Use helper function for vesicular post-processing
                        print(f"    [POSTPROCESS] Applying vesicular postprocess (gentle smoothing)")
                        binary_mask = postprocess_vesicular_mask(binary_mask)
                    else:
                        # Fallback: no specific postprocess, just skip
                        print(f"    [POSTPROCESS] No structure-specific postprocess (structure_type={structure_type})")

                # Use watershed labeling if watershed=True (for discrete round objects)
                # Otherwise fall back to connected components
                use_watershed = frangi_params.get("watershed", False)
                watershed_min_dist = frangi_params.get("watershed_min_distance", 1)
                min_object_size = frangi_params.get("min_object_size", 0)
                watershed_compactness = frangi_params.get("watershed_compactness", 0.0)
                watershed_erosion = frangi_params.get("watershed_erosion", 0)
                watershed_min_peak = frangi_params.get("watershed_min_peak", 1.0)
                watershed_h_maxima = frangi_params.get("watershed_h_maxima", 0.0)
                if use_watershed:
                    labeled_mask = watershed_label(
                        binary_mask,
                        min_distance=watershed_min_dist,
                        min_object_size=min_object_size,
                        compactness=watershed_compactness,
                        erosion_iterations=watershed_erosion,
                        min_peak_distance=watershed_min_peak,
                        h_maxima=watershed_h_maxima,
                    )
                    num_labels = int(labeled_mask.max())
                else:
                    footprint = scipy_ndi.generate_binary_structure(binary_mask.ndim, 1)
                    labeled_mask, num_labels = scipy_ndi.label(binary_mask, structure=footprint)

                    # Apply min_object_size filtering (independent of watershed)
                    if min_object_size > 0 and labeled_mask.max() > 0:
                        label_ids, counts = np.unique(labeled_mask, return_counts=True)
                        small_labels = label_ids[(label_ids > 0) & (counts < min_object_size)]
                        if len(small_labels) > 0:
                            labeled_mask[np.isin(labeled_mask, small_labels)] = 0
                            # Relabel to ensure continuous IDs
                            labeled_mask, num_labels = scipy_ndi.label(labeled_mask > 0, structure=footprint)
                            labeled_mask = labeled_mask.astype(np.int32)
            else:
                binary_mask = np.zeros_like(vesselness_map, dtype=bool)
                labeled_mask = np.zeros_like(vesselness_map, dtype=np.int32)

        # Prepare tile coordinates and dimensions
        y_start = tile_info["y_start_tile"]
        x_start = tile_info["x_start_tile"]
        y_end = tile_info["y_end_tile"]
        x_end = tile_info["x_end_tile"]

        # Get actual tile dimensions (may be smaller at edges)
        actual_height = y_end - y_start
        actual_width = x_end - x_start
        tile_labels = labeled_mask[:actual_height, :actual_width].astype(np.int32)
        tile_vesselness = vesselness_map[:actual_height, :actual_width].astype(np.float32)

        # Calculate CORE region (non-overlapping) to avoid race conditions
        # Each tile writes EXACTLY its step x step region, which maps to one zarr chunk
        # The tile data includes overlap for processing context, but only core is written
        #
        # Tile layout (1D example with tile_size=4096, overlap=256, step=3840):
        #   Tile 0: global [0:4096], writes core [0:3840]
        #   Tile 1: global [3840:7936], writes core [3840:7680]
        #   Tile 2: global [7680:11776], writes core [7680:11520]
        #
        # Within each tile's local coordinates:
        #   - Tile 0: local [0:3840] -> global [0:3840]
        #   - Tile 1: local [0:3840] -> global [3840:7680] (tile starts at global 3840)
        #   - Edge tiles may write less than step to stay within image bounds
        tile_size = tile_info["tile_size"]
        step = tile_size - tile_overlap

        # Global coordinates: each tile writes [ty*step : (ty+1)*step, tx*step : (tx+1)*step]
        # Clamped to image dimensions to avoid writing beyond bounds
        core_y_start_global = ty * step
        core_y_end_global = min((ty + 1) * step, actual_height + y_start)  # Clamp to actual image
        core_x_start_global = tx * step
        core_x_end_global = min((tx + 1) * step, actual_width + x_start)

        # Local coordinates: offset from tile's starting position
        # Tile starts at global [y_start, x_start], so local = global - start
        core_y_start_local = core_y_start_global - y_start
        core_y_end_local = core_y_end_global - y_start
        core_x_start_local = core_x_start_global - x_start
        core_x_end_local = core_x_end_global - x_start

        # Clamp local coordinates to tile bounds
        core_y_start_local = max(0, core_y_start_local)
        core_y_end_local = min(actual_height, core_y_end_local)
        core_x_start_local = max(0, core_x_start_local)
        core_x_end_local = min(actual_width, core_x_end_local)

        # Extract core region from tile data
        core_labels = tile_labels[core_y_start_local:core_y_end_local, core_x_start_local:core_x_end_local]
        core_vesselness = tile_vesselness[core_y_start_local:core_y_end_local, core_x_start_local:core_x_end_local]

        # Write ONLY core region to zarr - each tile writes to unique chunk (parallel-safe)
        # Use output_zarr_path if provided (for preview mode), otherwise source_zarr_path
        write_path = output_zarr_path if output_zarr_path else source_zarr_path

        if output_zarr_path:
            # Preview mode: use raw zarr (not iohub) since temp zarr doesn't have OME-Zarr metadata
            import zarr
            store = zarr.open(write_path, mode="r+")
            if output_label_name:
                temp_name = f"{output_label_name}_unstitched"
                labels_arr = store[pos_path]["labels"][temp_name]["0"]
                labels_arr[0, 0, 0, core_y_start_global:core_y_end_global, core_x_start_global:core_x_end_global] = core_labels

            if save_vesselness and output_label_name:
                vesselness_label_name = output_label_name.replace("_seg", "_vesselness")
                temp_vesselness_name = f"{vesselness_label_name}_unstitched"
                vesselness_arr = store[pos_path]["labels"][temp_vesselness_name]["0"]
                vesselness_arr[0, 0, 0, core_y_start_global:core_y_end_global, core_x_start_global:core_x_end_global] = core_vesselness
        else:
            # Normal mode: use iohub's open_ome_zarr
            with open_ome_zarr(write_path, mode="r+") as ds:
                # Write labels
                if output_label_name:
                    temp_name = f"{output_label_name}_unstitched"
                    labels_arr = ds[pos_path].zgroup["labels"][temp_name]["0"]
                    labels_arr[0, 0, 0, core_y_start_global:core_y_end_global, core_x_start_global:core_x_end_global] = core_labels

                # Write vesselness if requested
                if save_vesselness and output_label_name:
                    vesselness_label_name = output_label_name.replace("_seg", "_vesselness")
                    temp_vesselness_name = f"{vesselness_label_name}_unstitched"
                    vesselness_arr = ds[pos_path].zgroup["labels"][temp_vesselness_name]["0"]
                    vesselness_arr[0, 0, 0, core_y_start_global:core_y_end_global, core_x_start_global:core_x_end_global] = core_vesselness

        # Return minimal dict - only store center tile for debug
        is_center = tile_info.get("is_center", False)
        return {
            "tile_info": tile_info,
            "success": True,
            "vesselness": tile_vesselness if is_center else None,
            "labels": tile_labels.copy() if is_center else None,
        }

    except Exception as e:
        print(f"Error processing Frangi tile {tile_info.get('tile_idx', '?')}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "tile_info": tile_info,
            "success": False,
        }


def _stitch_tiled_labels_pass2(
    source_zarr_path: str,
    pos_path: str,
    organelle_name: str,
    n_tiles_y: int,
    n_tiles_x: int,
    tile_size: int,
    tile_overlap: int,
    height: int,
    width: int,
    input_mask_name: str = None,
    mask_erosion_pixels: int = 0,
    crop_bbox: tuple = None,
    target_chunks: tuple = (1, 1, 1, 512, 512),
    target_shards_ratio: tuple = (1, 1, 1, 32, 32),
):
    """
    Pass 2: Two-phase stitching with Union-Find for proper label merging.

    Phase A: Offset all tile labels to be globally unique, collect ALL merge pairs
    Phase B: Use Union-Find to build connected components, apply global relabeling
    Phase C: Rechunk from parallel-write-safe 1:1 sharding to efficient storage sharding

    This properly handles:
    - Multiple labels mapping to the same neighbor
    - Transitive merges (A→B, B→C implies A→C)
    - Labels that span multiple tile boundaries

    Args:
        source_zarr_path: Path to the v3 zarr store
        pos_path: Position path like "A/1/0"
        organelle_name: Name of the organelle being segmented
        n_tiles_y: Number of tiles in Y direction
        n_tiles_x: Number of tiles in X direction
        tile_size: Size of each tile
        tile_overlap: Overlap between tiles
        height: Total image height
        width: Total image width
        input_mask_name: If provided (e.g., "nuclear_seg"), erode this mask and remove
            labels outside the eroded zone (removes boundary artifacts).
        mask_erosion_pixels: Pixels to erode the input mask by. Labels outside the
            eroded mask are removed (e.g., nucleoli near nuclear boundary).
        crop_bbox: If provided (y_start, y_end, x_start, x_end), the labels were generated
        target_chunks: Target chunk size for final array (default: 512x512)
        target_shards_ratio: Target sharding ratio for final array (default: 32x32 for ~1GB shards)
            from a cropped region of the source image. Used to correctly slice the input_mask.
    """
    temp_name = f"{organelle_name}_unstitched"
    step = tile_size - tile_overlap

    print(f"  Pass 2: Stitching {n_tiles_y * n_tiles_x} tiles with Union-Find merging...")
    print(f"    tile_size={tile_size}, tile_overlap={tile_overlap}, step={step}")

    # =========================================================================
    # PHASE A: Offset all tiles and collect merge pairs
    # =========================================================================
    all_merge_pairs = []  # List of (label_a, label_b) pairs to merge
    max_label_seen = 0

    with open_ome_zarr(source_zarr_path, mode="r+") as ds:
        source_pos = ds[pos_path]
        labels_group = source_pos.zgroup["labels"]
        labels_arr = labels_group[temp_name]["0"]

        running_offset = 0

        # Phase A: Offset labels and collect merge pairs
        print(f"    Phase A: Offsetting labels and collecting merge pairs...")
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                y_start = ty * step
                x_start = tx * step
                y_end = min((ty + 1) * step, height)
                x_end = min((tx + 1) * step, width)

                tile_labels = np.asarray(labels_arr[0, 0, 0, y_start:y_end, x_start:x_end]).copy()

                if tile_labels.max() == 0:
                    continue

                # Offset tile labels to be globally unique
                tile_mask = tile_labels > 0
                tile_labels[tile_mask] += running_offset
                tile_max = tile_labels.max()

                # Write offset labels back immediately (so neighbors can read them)
                labels_arr[0, 0, 0, y_start:y_end, x_start:x_end] = tile_labels

                # Check LEFT neighbor for merge pairs
                if tx > 0:
                    left_boundary_x = x_start - 1
                    if left_boundary_x >= 0:
                        left_boundary = np.asarray(
                            labels_arr[0, 0, 0, y_start:y_end, left_boundary_x:left_boundary_x+1]
                        ).flatten()
                        tile_left_edge = tile_labels[:, 0]

                        if left_boundary.max() > 0 and tile_left_edge.max() > 0:
                            for row_idx in range(len(tile_left_edge)):
                                tile_label = tile_left_edge[row_idx]
                                if tile_label == 0:
                                    continue
                                for neighbor_row in [row_idx - 1, row_idx, row_idx + 1]:
                                    if 0 <= neighbor_row < len(left_boundary):
                                        neighbor_label = left_boundary[neighbor_row]
                                        if neighbor_label > 0 and tile_label != neighbor_label:
                                            all_merge_pairs.append((int(tile_label), int(neighbor_label)))

                # Check TOP neighbor for merge pairs
                if ty > 0:
                    top_boundary_y = y_start - 1
                    if top_boundary_y >= 0:
                        top_boundary = np.asarray(
                            labels_arr[0, 0, 0, top_boundary_y:top_boundary_y+1, x_start:x_end]
                        ).flatten()
                        tile_top_edge = tile_labels[0, :]

                        if top_boundary.max() > 0 and tile_top_edge.max() > 0:
                            for col_idx in range(len(tile_top_edge)):
                                tile_label = tile_top_edge[col_idx]
                                if tile_label == 0:
                                    continue
                                for neighbor_col in [col_idx - 1, col_idx, col_idx + 1]:
                                    if 0 <= neighbor_col < len(top_boundary):
                                        neighbor_label = top_boundary[neighbor_col]
                                        if neighbor_label > 0 and tile_label != neighbor_label:
                                            all_merge_pairs.append((int(tile_label), int(neighbor_label)))

                running_offset = max(running_offset, tile_max)
                max_label_seen = max(max_label_seen, tile_max)

        print(f"    Phase A complete: {len(all_merge_pairs)} merge pairs, max_label={max_label_seen}")

        # =========================================================================
        # PHASE B: Build Union-Find and apply global relabeling
        # =========================================================================
        if all_merge_pairs:
            print(f"    Phase B: Building Union-Find structure...")

            # Union-Find with path compression
            parent = list(range(int(max_label_seen) + 1))

            def find(x):
                root = x
                while parent[root] != root:
                    root = parent[root]
                # Path compression
                while parent[x] != root:
                    next_x = parent[x]
                    parent[x] = root
                    x = next_x
                return root

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    # Always merge to the smaller root (ensures consistency)
                    if ra < rb:
                        parent[rb] = ra
                    else:
                        parent[ra] = rb

            # Process all merge pairs
            for a, b in all_merge_pairs:
                if a <= max_label_seen and b <= max_label_seen:
                    union(a, b)

            # Build final LUT: each label maps to its root
            lut = np.arange(int(max_label_seen) + 1, dtype=np.int32)
            for i in range(1, int(max_label_seen) + 1):
                lut[i] = find(i)

            # Count unique merged labels
            unique_roots = len(set(lut[1:]))
            print(f"    Phase B: {int(max_label_seen)} labels merged to {unique_roots} unique labels")

            # Apply LUT to all tiles
            print(f"    Phase B: Applying global relabeling...")
            for ty in tqdm(range(n_tiles_y), desc="    Relabeling tiles"):
                for tx in range(n_tiles_x):
                    y_start = ty * step
                    x_start = tx * step
                    y_end = min((ty + 1) * step, height)
                    x_end = min((tx + 1) * step, width)

                    tile_labels = np.asarray(labels_arr[0, 0, 0, y_start:y_end, x_start:x_end]).copy()

                    if tile_labels.max() == 0:
                        continue

                    # Apply LUT (handle labels larger than LUT size)
                    mask = tile_labels <= max_label_seen
                    tile_labels[mask] = lut[tile_labels[mask]]

                    labels_arr[0, 0, 0, y_start:y_end, x_start:x_end] = tile_labels
        else:
            print(f"    Phase B: No merge pairs found, skipping relabeling")

    # Rename from temp to final using filesystem rename (zarr v3 doesn't support copy)
    print(f"  Renaming {temp_name} -> {organelle_name}")

    # Get the filesystem paths for the zarr groups
    zarr_store_path = Path(source_zarr_path)
    labels_path = zarr_store_path / pos_path / "labels"
    temp_path = labels_path / temp_name
    final_path = labels_path / organelle_name

    # Remove existing final path if it exists
    if final_path.exists():
        import shutil
        shutil.rmtree(final_path)

    # Rename temp to final
    temp_path.rename(final_path)

    print(f"  Pass 2 complete: {running_offset} total objects after stitching")

    # Optional Pass 3: Remove labels near mask boundary (e.g., nucleoli near nuclear edge)
    # We erode the INPUT MASK (e.g., nuclear_seg) and remove any labels outside the eroded zone
    if input_mask_name and mask_erosion_pixels > 0:
        from scipy.ndimage import binary_erosion

        print(f"  Pass 3: Removing labels within {mask_erosion_pixels}px of {input_mask_name} boundary...")
        erosion_structure = np.ones((mask_erosion_pixels * 2 + 1, mask_erosion_pixels * 2 + 1), dtype=bool)

        with open_ome_zarr(source_zarr_path, mode="r+") as ds:
            labels_arr = ds[pos_path].zgroup["labels"][organelle_name]["0"]

            # Check if the input mask exists
            labels_group = ds[pos_path].zgroup.get("labels", None)
            if labels_group is None or input_mask_name not in labels_group:
                print(f"  Warning: Input mask '{input_mask_name}' not found, skipping Pass 3")
            else:
                mask_arr = labels_group[input_mask_name]["0"]

                # IMPORTANT: Load and erode the FULL mask first, then apply tile by tile
                # Eroding tile-by-tile causes edge artifacts (tile boundaries get eroded)
                print(f"    Loading full mask for erosion...")
                # If crop_bbox is provided, slice the mask to match the cropped region
                if crop_bbox:
                    crop_y_start, crop_y_end, crop_x_start, crop_x_end = crop_bbox
                    full_mask = np.asarray(mask_arr[0, 0, 0, crop_y_start:crop_y_end, crop_x_start:crop_x_end])
                    print(f"    Using crop_bbox: y=[{crop_y_start}:{crop_y_end}], x=[{crop_x_start}:{crop_x_end}]")
                else:
                    full_mask = np.asarray(mask_arr[0, 0, 0, :height, :width])
                binary_full_mask = full_mask > 0
                mask_pixels_before = binary_full_mask.sum()
                eroded_full_mask = binary_erosion(binary_full_mask, structure=erosion_structure)
                mask_pixels_after = eroded_full_mask.sum()
                print(f"    Mask pixels: {mask_pixels_before:,} -> {mask_pixels_after:,} after erosion ({100*mask_pixels_after/max(1,mask_pixels_before):.1f}% retained)")
                print(f"    Mask eroded, applying to labels...")

                # Process tile by tile to avoid loading full labels into memory
                labels_before_total = 0
                labels_after_total = 0
                for ty in tqdm(range(n_tiles_y), desc="  Removing boundary labels"):
                    for tx in range(n_tiles_x):
                        # Use CORE region coordinates (matching Pass 1 and Pass 2)
                        y_start = ty * step
                        x_start = tx * step
                        y_end = min((ty + 1) * step, height)
                        x_end = min((tx + 1) * step, width)

                        tile_labels = np.asarray(labels_arr[0, 0, 0, y_start:y_end, x_start:x_end]).copy()
                        if tile_labels.max() == 0:
                            continue

                        labels_before_total += (tile_labels > 0).sum()

                        # Get the corresponding eroded mask tile
                        # IMPORTANT: Account for crop_bbox offset when slicing the mask
                        if crop_bbox:
                            crop_y_start_bbox, crop_y_end_bbox, crop_x_start_bbox, crop_x_end_bbox = crop_bbox
                            mask_y_start = y_start
                            mask_y_end = y_end
                            mask_x_start = x_start
                            mask_x_end = x_end
                            eroded_mask_tile = eroded_full_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
                        else:
                            eroded_mask_tile = eroded_full_mask[y_start:y_end, x_start:x_end]

                        # Verify dimensions match before applying boolean index
                        if eroded_mask_tile.shape != tile_labels.shape:
                            print(f"    Warning: Dimension mismatch at tile ({ty}, {tx}): "
                                  f"mask {eroded_mask_tile.shape} vs labels {tile_labels.shape}, skipping")
                            continue

                        # Remove labels outside the eroded mask (i.e., near the boundary)
                        tile_labels[~eroded_mask_tile] = 0

                        labels_after_total += (tile_labels > 0).sum()

                        # Write back
                        labels_arr[0, 0, 0, y_start:y_end, x_start:x_end] = tile_labels

                print(f"  Pass 3 complete: Label pixels {labels_before_total:,} -> {labels_after_total:,} ({100*labels_after_total/max(1,labels_before_total):.1f}% retained)")

    # Final phase: Reshard from parallel-write-safe 1:1 sharding to efficient storage sharding
    # This is the second phase of the two-phase write strategy:
    # - Pass 1+2 used chunks=shards (1:1 mapping) for safe parallel writes
    # - Now we reshard to target_shards_ratio for efficient storage (~1GB shard files)
    from ops_utils.io.zarr_utils import reshard_zarr_array
    label_array_path = Path(source_zarr_path) / pos_path / "labels" / organelle_name / "0"
    reshard_zarr_array(
        source_path=label_array_path,
        dest_path=None,  # In-place resharding
        chunks=target_chunks,
        shards_ratio=target_shards_ratio,
        tile_size=4096,
        show_progress=True,
    )

    return running_offset


def segment_position_frangi_tiled(
    pos_path,
    source_zarr_path,
    channel_to_segment,
    organelle_name,
    frangi_params,
    frangi_postprocess: bool,
    use_clahe: bool,
    post_clahe_smoothing_sigma: float,
    clahe_params: dict = None,
    crop_bbox: tuple = None,
    tile_size: int = 4096,
    tile_overlap: int = 512,
    save_vesselness: bool = False,
    input_mask_name: str = None,
    structure_type: str = None,
    debug_output_path: str = None,
    nucleoli_method: str = None,
    vesicular_method: str = None,
    preview_mode: bool = False,
    shards_ratio: tuple = (1, 1, 1, 32, 32),
):
    """
    Two-pass tiled Frangi segmentation with Dask parallelization for large images.

    Pass 1: Process all tiles in parallel, write raw (unstitched) labels directly to zarr.
            Each tile's labels start from 1 - NOT globally unique.
    Pass 2: Sequential overlap correction - read tile boundaries, match labels, update in-place.

    Memory requirements:
    - Pass 1: ~300-500 MB per worker (tile + Frangi overhead), writes directly to disk
    - Pass 2: ~200 MB peak (only loads overlap regions)
    - No full-size in-memory canvas needed!

    Args:
        pos_path: Position path like "A/1/0"
        source_zarr_path: Path to the v3 zarr store
        channel_to_segment: Channel name to segment
        organelle_name: Name of the organelle being segmented
        frangi_params: Parameters for Frangi filter
        frangi_postprocess: Whether to apply postprocessing
        use_clahe: Whether to apply CLAHE preprocessing
        post_clahe_smoothing_sigma: Sigma for Gaussian smoothing after CLAHE
        clahe_params: Parameters for CLAHE
        crop_bbox: Optional tuple (y_start, y_end, x_start, x_end) for debug center crop
        tile_size: Size of tiles to process (default: 4096)
        tile_overlap: Overlap between tiles for stitching (default: 256)
        save_vesselness: If True, also save the continuous Frangi vesselness map
            as a separate array in labels/ (default: False). Note: This is not
            strictly NGFF-compliant as labels/ should only contain integer masks,
            but is useful for visualization/debugging.
        input_mask_name: Optional mask name (e.g., "nuclear_seg") to constrain segmentation.
                         If provided, Frangi will only detect structures within the mask.
        nucleoli_method: For nucleoli segmentation: "blob" for LoG, "frangi" for Frangi
        vesicular_method: For vesicular segmentation: "blob" for LoG, "frangi" for Frangi
        preview_mode: If True, write results to a temp zarr instead of the original.
            Used for preview/debug mode to avoid modifying production data.
        shards_ratio: Sharding ratio for zarr v3 storage (default: (1, 1, 1, 32, 32)).
            This determines the shard file size. With 512x512 base chunks and 32x32 ratio,
            each shard covers 16384x16384 pixels (~1GB for int32 single-channel labels).

    Returns:
        Tuple of (pos_path, vesselness_5d, binary_5d, objects_5d, source_scale, crop_bbox)
    """
    from PIL import Image
    from joblib import Parallel, delayed
    import tempfile
    import shutil

    try:
        start_time = time.time()
        if crop_bbox:
            y_start, y_end, x_start, x_end = crop_bbox
            print(f"[{pos_path}] Loading center crop for tiled Frangi segmentation (two-pass)...")
            print(f"  Crop region: Y[{y_start}:{y_end}], X[{x_start}:{x_end}]")
        else:
            print(f"[{pos_path}] Starting two-pass tiled Frangi segmentation...")

        with open_ome_zarr(source_zarr_path, mode="r") as ds:
            channel_names = list(ds.channel_names)  # Capture for metadata
            channel_index = channel_names.index(channel_to_segment)
            source_pos = ds[pos_path]
            image_array = source_pos["0"]
            source_scale = source_pos.scale
            full_shape = image_array.shape  # (T, C, Z, Y, X)

            # Determine image dimensions
            if crop_bbox:
                y_start_crop, y_end_crop, x_start_crop, x_end_crop = crop_bbox
                height = y_end_crop - y_start_crop
                width = x_end_crop - x_start_crop
            else:
                height, width = full_shape[3], full_shape[4]
                y_start_crop, x_start_crop = 0, 0

        print(f"  Full position size: {height} x {width}")
        print(f"  Processing in {tile_size}x{tile_size} tiles with {tile_overlap}px overlap")

        # Calculate tile grid
        step = tile_size - tile_overlap
        n_tiles_y = max(1, int(np.ceil((height - tile_overlap) / step)))
        n_tiles_x = max(1, int(np.ceil((width - tile_overlap) / step)))
        total_tiles = n_tiles_y * n_tiles_x

        print(f"  Tile grid: {n_tiles_y} x {n_tiles_x} = {total_tiles} tiles")

        # Build list of tile info dicts for parallel processing
        # Store center tile indices for debug output
        center_ty = (n_tiles_y * 3) // 4  # 75% from top
        center_tx = (n_tiles_x * 3) // 4  # 75% from left

        tile_infos = []
        tile_idx = 0
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                tile_idx += 1

                # Calculate tile boundaries in the output canvas
                y_start_tile = ty * step
                x_start_tile = tx * step
                y_end_tile = min(y_start_tile + tile_size, height)
                x_end_tile = min(x_start_tile + tile_size, width)

                # Calculate boundaries in the source image (accounting for crop)
                src_y_start = y_start_crop + y_start_tile
                src_y_end = y_start_crop + y_end_tile
                src_x_start = x_start_crop + x_start_tile
                src_x_end = x_start_crop + x_end_tile

                tile_infos.append({
                    "tile_idx": tile_idx,
                    "ty": ty,
                    "tx": tx,
                    "y_start_tile": y_start_tile,
                    "x_start_tile": x_start_tile,
                    "y_end_tile": y_end_tile,
                    "x_end_tile": x_end_tile,
                    "src_y_start": src_y_start,
                    "src_y_end": src_y_end,
                    "src_x_start": src_x_start,
                    "src_x_end": src_x_end,
                    "tile_size": tile_size,  # Full tile size (for core region calculation)
                    "is_center": (ty == center_ty and tx == center_tx),
                })

        # Set up pixel resolution from config (unified: 0.1625um)
        pixel_size_um = frangi_params.get("pixel_size_um", 0.1625)
        pixel_resolution = {"Z": 1.0, "Y": pixel_size_um, "X": pixel_size_um}

        # Determine number of workers based on available CPU resources
        # Frangi filter uses significant RAM per tile due to Hessian calculations:
        # - Tile data: 4096x4096 x 4 bytes = 64 MB
        # - Hessian elements (3 for 2D): 3 x 64 MB = 192 MB
        # - Eigenvalues/vectors: ~256 MB
        # - scipy intermediate arrays: ~500 MB
        # - Total peak per worker: ~2-3 GB
        num_workers = get_optimal_workers(
            use_gpu=False,  # CPU only for Frangi tiled
            model_ram_gb=1.5,  # Frangi/scipy overhead per worker
            data_ram_gb=1.5,  # Tile data + Hessian arrays
            verbose=True,
        )

        # Cap workers at total tiles and ensure reasonable minimum
        num_workers = min(num_workers, total_tiles)
        num_workers = max(1, num_workers)  # At least 1 worker

        # Calculate memory per worker (for LocalCluster)
        import psutil
        total_mem_gb = psutil.virtual_memory().total / (1024**3)
        mem_per_worker_gb = max(3.0, total_mem_gb * 0.8 / max(num_workers, 1))  # 80% of RAM divided by workers, min 3GB

        print(f"  Using {num_workers} parallel CPU workers for Frangi tile processing")
        print(f"  Memory per worker: {mem_per_worker_gb:.1f} GB")

        # --- PASS 1: Create output zarr array and process tiles in parallel ---
        # Compute the standardized output label name (same as caller uses)
        # Include structure_type for dual Frangi segmentation to avoid race conditions
        output_label_name = get_output_label_name(organelle_name, channel_to_segment, structure_type)
        temp_name = f"{output_label_name}_unstitched"
        print(f"  Pass 1: Processing {total_tiles} tiles in parallel, writing to zarr...")
        print(f"  Output label: {output_label_name}")

        # Create the output zarr arrays with chunking matching tile size
        # Optionally also create vesselness (float32) array if save_vesselness=True
        vesselness_label_name = output_label_name.replace("_seg", "_vesselness") if save_vesselness else None
        temp_vesselness_name = f"{vesselness_label_name}_unstitched" if save_vesselness else None

        # In preview_mode, create a temporary zarr to avoid modifying the original
        # Use raw zarr (not iohub) for preview since we don't need full OME-Zarr compliance
        temp_zarr_dir = None
        output_zarr_path = source_zarr_path  # Default to source

        # Calculate chunking for parallel write safety:
        # - Each tile writes exactly one chunk (step x step) to avoid race conditions
        # - CRITICAL: With zarr v3, shards must NOT be used with parallel writes unless synchronized
        # - Setting shards=chunks means each chunk gets its own file (safe for parallel writes)
        label_chunks = (1, 1, 1, step, step)
        label_shards = (1, 1, 1, step, step)  # 1:1 mapping, each chunk = one shard file (parallel-safe)

        if preview_mode:
            import zarr
            temp_zarr_dir = tempfile.mkdtemp(prefix="organelle_seg_preview_")
            output_zarr_path = str(Path(temp_zarr_dir) / "preview.zarr")
            print(f"  PREVIEW MODE: Using temp zarr at {output_zarr_path}")

            # Create zarr structure using raw zarr (not iohub)
            zarr_store = zarr.open(output_zarr_path, mode="w")
            pos_group = zarr_store.require_group(pos_path)
            labels_group = pos_group.require_group("labels")

            # Create labels array with sharding
            temp_subgroup = labels_group.create_group(temp_name)
            temp_subgroup.create_array(
                "0",
                shape=(1, 1, 1, height, width),
                dtype=np.int32,
                chunks=label_chunks,
                shards=label_shards,
                fill_value=0,
            )

            # Create vesselness array if needed (also with sharding)
            if save_vesselness:
                temp_vesselness_subgroup = labels_group.create_group(temp_vesselness_name)
                temp_vesselness_subgroup.create_array(
                    "0",
                    shape=(1, 1, 1, height, width),
                    dtype=np.float32,
                    chunks=label_chunks,
                    shards=label_shards,
                    fill_value=0.0,
                )
                print(f"  Also storing vesselness map as: {vesselness_label_name}")

        else:
            # Normal mode: use iohub's open_ome_zarr for proper OME-Zarr handling
            with open_ome_zarr(source_zarr_path, mode="r+") as ds:
                source_pos = ds[pos_path]
                # Use existing labels group (pre-created by SLURM submission script)
                # to avoid race conditions when multiple jobs run in parallel
                labels_group = source_pos.zgroup["labels"]

                # Delete if exists (fresh start)
                if temp_name in labels_group:
                    del labels_group[temp_name]
                if save_vesselness and temp_vesselness_name in labels_group:
                    del labels_group[temp_vesselness_name]

                # Create labels array with STEP-aligned chunks to avoid race conditions
                # AND sharding for efficient storage (matching convert_v3.py behavior)
                temp_subgroup = labels_group.create_group(temp_name)
                temp_subgroup.create_array(
                    "0",
                    shape=(1, 1, 1, height, width),
                    dtype=np.int32,
                    chunks=label_chunks,
                    shards=label_shards,
                    fill_value=0,
                )

                # Optionally create vesselness array (also with sharding)
                if save_vesselness:
                    temp_vesselness_subgroup = labels_group.create_group(temp_vesselness_name)
                    temp_vesselness_subgroup.create_array(
                        "0",
                        shape=(1, 1, 1, height, width),
                        dtype=np.float32,
                        chunks=label_chunks,
                        shards=label_shards,
                        fill_value=0.0,
                    )
                    print(f"  Also storing vesselness map as: {vesselness_label_name}")

        # Process tiles - use joblib for parallel processing, or direct loop for single tile/worker
        # Each worker processes a tile and writes directly to zarr
        if num_workers == 1 or total_tiles == 1:
            # Single worker or single tile: skip joblib overhead, process directly
            print(f"  Pass 1: Processing {total_tiles} tile(s) sequentially (num_workers=1)...")
            all_results = []
            for tile_info in tqdm(tile_infos, desc="  Pass 1: Segmenting tiles"):
                result = _process_single_frangi_tile(
                    tile_info=tile_info,
                    source_zarr_path=source_zarr_path,
                    pos_path=pos_path,
                    channel_index=channel_index,
                    frangi_params=frangi_params,
                    pixel_resolution=pixel_resolution,
                    use_clahe=use_clahe,
                    clahe_params=clahe_params,
                    post_clahe_smoothing_sigma=post_clahe_smoothing_sigma,
                    frangi_postprocess=frangi_postprocess,
                    input_mask_name=input_mask_name,
                    nucleoli_method=nucleoli_method,
                    vesicular_method=vesicular_method,
                    output_label_name=output_label_name,
                    save_vesselness=save_vesselness,
                    tile_overlap=tile_overlap,
                    n_tiles_y=n_tiles_y,
                    n_tiles_x=n_tiles_x,
                    output_zarr_path=output_zarr_path if preview_mode else None,
                )
                all_results.append(result)
        else:
            # Multiple tiles and workers: use joblib for parallel processing
            print(f"  Pass 1: Processing {total_tiles} tiles in parallel with {num_workers} workers...")

            # Process all tiles in parallel - each worker writes directly to zarr with locking
            # Pass tile grid info so workers can compute core (non-overlapping) regions
            all_results = Parallel(n_jobs=num_workers)(
                delayed(_process_single_frangi_tile)(
                    tile_info=tile_info,
                    source_zarr_path=source_zarr_path,
                    pos_path=pos_path,
                    channel_index=channel_index,
                    frangi_params=frangi_params,
                    pixel_resolution=pixel_resolution,
                    use_clahe=use_clahe,
                    clahe_params=clahe_params,
                    post_clahe_smoothing_sigma=post_clahe_smoothing_sigma,
                    frangi_postprocess=frangi_postprocess,
                    input_mask_name=input_mask_name,
                    nucleoli_method=nucleoli_method,
                    vesicular_method=vesicular_method,
                    output_label_name=output_label_name,
                    save_vesselness=save_vesselness,
                    tile_overlap=tile_overlap,
                    n_tiles_y=n_tiles_y,
                    n_tiles_x=n_tiles_x,
                    output_zarr_path=output_zarr_path if preview_mode else None,
                )
                for tile_info in tqdm(tile_infos, desc="  Pass 1: Segmenting tiles")
            )

        print(f"  Pass 1 complete: {total_tiles} tiles written to zarr")

        # Extract center tile result for debug output
        center_tile_result = None
        for result in all_results:
            if result.get("vesselness") is not None and result.get("labels") is not None:
                center_tile_result = {
                    "vesselness": result["vesselness"],
                    "labels_before_stitch": result["labels"],
                }
                break

        # In preview_mode, do in-memory stitching, then return data for canvas
        if preview_mode:
            # Load unstitched labels from temp zarr for preview
            import zarr
            preview_store = zarr.open(output_zarr_path, mode="r+")
            labels_data = np.asarray(preview_store[pos_path]["labels"][temp_name]["0"][...])

            # Load vesselness if saved
            vesselness_5d = None
            if save_vesselness and temp_vesselness_name:
                vesselness_data = np.asarray(preview_store[pos_path]["labels"][temp_vesselness_name]["0"][...])
                vesselness_5d = vesselness_data

            # Perform in-memory stitching (same algorithm as _stitch_tiled_labels_pass2)
            print(f"  Pass 2 (PREVIEW): In-memory stitching of {n_tiles_y * n_tiles_x} tiles...")
            labels_2d = np.squeeze(labels_data)  # Work with 2D array

            # Phase A: Offset all tiles and collect merge pairs
            all_merge_pairs = []
            step = tile_size - tile_overlap
            running_offset = 0
            max_label_seen = 0

            for ty in range(n_tiles_y):
                for tx in range(n_tiles_x):
                    y_start = ty * step
                    x_start = tx * step
                    y_end = min((ty + 1) * step, height)
                    x_end = min((tx + 1) * step, width)

                    tile_labels = labels_2d[y_start:y_end, x_start:x_end].copy()

                    if tile_labels.max() == 0:
                        continue

                    # Offset tile labels to be globally unique
                    tile_mask = tile_labels > 0
                    tile_labels[tile_mask] += running_offset
                    tile_max = tile_labels.max()

                    # Write offset labels back
                    labels_2d[y_start:y_end, x_start:x_end] = tile_labels

                    # Check LEFT neighbor for merge pairs
                    if tx > 0:
                        left_boundary_x = x_start - 1
                        if left_boundary_x >= 0:
                            left_boundary = labels_2d[y_start:y_end, left_boundary_x].flatten()
                            tile_left_edge = tile_labels[:, 0]

                            if left_boundary.max() > 0 and tile_left_edge.max() > 0:
                                for row_idx in range(len(tile_left_edge)):
                                    tile_label = tile_left_edge[row_idx]
                                    if tile_label == 0:
                                        continue
                                    for neighbor_row in [row_idx - 1, row_idx, row_idx + 1]:
                                        if 0 <= neighbor_row < len(left_boundary):
                                            neighbor_label = left_boundary[neighbor_row]
                                            if neighbor_label > 0 and tile_label != neighbor_label:
                                                all_merge_pairs.append((int(tile_label), int(neighbor_label)))

                    # Check TOP neighbor for merge pairs
                    if ty > 0:
                        top_boundary_y = y_start - 1
                        if top_boundary_y >= 0:
                            top_boundary = labels_2d[top_boundary_y, x_start:x_end].flatten()
                            tile_top_edge = tile_labels[0, :]

                            if top_boundary.max() > 0 and tile_top_edge.max() > 0:
                                for col_idx in range(len(tile_top_edge)):
                                    tile_label = tile_top_edge[col_idx]
                                    if tile_label == 0:
                                        continue
                                    for neighbor_col in [col_idx - 1, col_idx, col_idx + 1]:
                                        if 0 <= neighbor_col < len(top_boundary):
                                            neighbor_label = top_boundary[neighbor_col]
                                            if neighbor_label > 0 and tile_label != neighbor_label:
                                                all_merge_pairs.append((int(tile_label), int(neighbor_label)))

                    running_offset = max(running_offset, tile_max)
                    max_label_seen = max(max_label_seen, tile_max)

            print(f"    Phase A complete: {len(all_merge_pairs)} merge pairs, max_label={max_label_seen}")

            # Phase B: Build Union-Find and apply global relabeling
            if all_merge_pairs and max_label_seen > 0:
                print(f"    Phase B: Building Union-Find structure...")

                # Union-Find with path compression
                parent = list(range(int(max_label_seen) + 1))

                def find(x):
                    if parent[x] != x:
                        parent[x] = find(parent[x])
                    return parent[x]

                def union(a, b):
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)

                for a, b in all_merge_pairs:
                    union(a, b)

                # Build lookup for final labels
                unique_labels = np.unique(labels_2d)
                unique_labels = unique_labels[unique_labels > 0]

                remap = {0: 0}
                next_label = 1
                root_to_new = {}

                for old_label in unique_labels:
                    root = find(int(old_label))
                    if root not in root_to_new:
                        root_to_new[root] = next_label
                        next_label += 1
                    remap[int(old_label)] = root_to_new[root]

                # Apply remapping
                labels_2d_flat = labels_2d.flatten()
                remapped = np.array([remap.get(int(x), 0) for x in labels_2d_flat], dtype=labels_2d.dtype)
                labels_2d = remapped.reshape(labels_2d.shape)

                n_merged = len(unique_labels) - (next_label - 1)
                print(f"    Phase B complete: merged {n_merged} labels, final count: {next_label - 1}")

            # Reshape back to 5D
            objects_5d = labels_2d.reshape(1, 1, 1, labels_2d.shape[0], labels_2d.shape[1])

            # Cleanup temp zarr
            if temp_zarr_dir and Path(temp_zarr_dir).exists():
                shutil.rmtree(temp_zarr_dir)
                print(f"  PREVIEW MODE: Cleaned up temp zarr at {temp_zarr_dir}")

            elapsed_time = time.time() - start_time
            print(f"[{pos_path}] PREVIEW complete, took {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
            print(f"  No data written to original zarr (preview_mode=True)")

            binary_5d = None
            return pos_path, vesselness_5d, binary_5d, objects_5d, source_scale, crop_bbox

        # --- Normal mode: Continue with full processing ---

        # Optionally rename vesselness from temp to final (no stitching needed - continuous map)
        # Use filesystem rename since zarr v3 doesn't support zarr.copy()
        if save_vesselness and temp_vesselness_name and vesselness_label_name:
            print(f"  Renaming vesselness {temp_vesselness_name} -> {vesselness_label_name}")
            zarr_store_path = Path(source_zarr_path)
            labels_path = zarr_store_path / pos_path / "labels"
            temp_vesselness_path = labels_path / temp_vesselness_name
            final_vesselness_path = labels_path / vesselness_label_name

            if final_vesselness_path.exists():
                shutil.rmtree(final_vesselness_path)
            temp_vesselness_path.rename(final_vesselness_path)
            print(f"  Vesselness map saved: {vesselness_label_name}")

            # Reshard vesselness from parallel-write-safe 1:1 sharding to efficient storage
            from ops_utils.io.zarr_utils import reshard_zarr_array
            vesselness_array_path = Path(source_zarr_path) / pos_path / "labels" / vesselness_label_name / "0"
            reshard_zarr_array(
                source_path=vesselness_array_path,
                dest_path=None,  # In-place resharding
                chunks=(1, 1, 1, 512, 512),
                shards_ratio=shards_ratio,
                tile_size=4096,
                show_progress=True,
            )

            # Build and update metadata for the vesselness map
            # Use organelle_name as channel_label (it often contains "organelle, marker" info)
            vesselness_metadata = _build_vesselness_metadata(
                label_name=vesselness_label_name,
                organelle_name=organelle_name,
                channel_name=channel_to_segment,
                channel_label=organelle_name,  # organelle_name often is "mitochondria, TOMM20" etc.
                channel_index=channel_index,
                channel_names=channel_names,
            )
            _update_labels_metadata(
                zarr_path=Path(source_zarr_path),
                pos_path=pos_path,
                new_label_name=vesselness_label_name,
                metadata=vesselness_metadata,
            )

        # --- PASS 2: Sequential overlap correction (for labels only) ---
        # If input_mask_name is provided (e.g., nucleoli with nuclear mask), also run Pass 3
        # to remove labels near the mask boundary (erode mask by 6px, remove labels outside)
        _stitch_tiled_labels_pass2(
            source_zarr_path=source_zarr_path,
            pos_path=pos_path,
            organelle_name=output_label_name,  # Use the standardized output name
            n_tiles_y=n_tiles_y,
            n_tiles_x=n_tiles_x,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            height=height,
            width=width,
            input_mask_name=input_mask_name,
            mask_erosion_pixels=3 if input_mask_name else 0,
            crop_bbox=crop_bbox,
            target_chunks=(1, 1, 1, 512, 512),
            target_shards_ratio=shards_ratio,  # Use shards_ratio passed to this function
        )

        # Build and update metadata for the segmentation labels
        # Use organelle_name as channel_label (it often contains "organelle, marker" info)
        seg_metadata = _build_segmentation_metadata(
            label_name=output_label_name,
            organelle_name=organelle_name,
            channel_name=channel_to_segment,
            channel_label=organelle_name,  # organelle_name often is "mitochondria, TOMM20" etc.
            channel_index=channel_index,
            segmenter_type="frangi",
            channel_names=channel_names,
            structure_type=structure_type,
        )
        _update_labels_metadata(
            zarr_path=Path(source_zarr_path),
            pos_path=pos_path,
            new_label_name=output_label_name,
            metadata=seg_metadata,
        )

        # --- Build pyramids for the segmentation label ---
        # Skip in preview_mode since the data is in a temp location
        if not preview_mode:
            try:
                from ops_analysis.napari.dask.build_dask import build_organelle_seg_pyramids
            except ImportError:
                build_organelle_seg_pyramids = None
                print("  Warning: ops_analysis not available — skipping pyramid generation.")
                print("  Pyramids improve napari visualization but are not required.")

            if build_organelle_seg_pyramids is not None:
                print(f"\n  Building pyramids for {output_label_name}...")
                build_organelle_seg_pyramids(
                    source_store=source_zarr_path,
                    levels=5,
                    positions=[pos_path],
                    resume=True,
                    label_names=[output_label_name],
                )
                print(f"  Pyramids built for {output_label_name}")

                # Also build pyramids for vesselness if saved
                if save_vesselness and vesselness_label_name:
                    print(f"\n  Building pyramids for {vesselness_label_name}...")
                    build_organelle_seg_pyramids(
                        source_store=source_zarr_path,
                        levels=5,
                        positions=[pos_path],
                        resume=True,
                        label_names=[vesselness_label_name],
                    )
                    print(f"  Pyramids built for {vesselness_label_name}")

        # --- Save debug images using shared helper ---
        _save_tiled_debug_images(
            source_zarr_path=source_zarr_path,
            pos_path=pos_path,
            output_label_name=output_label_name,
            method_name="frangi",
            center_tile_result=center_tile_result,
            center_ty=center_ty,
            center_tx=center_tx,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            height=height,
            width=width,
            extra_arrays={"vesselness": center_tile_result.get("vesselness") if center_tile_result else None},
            channel_index=channel_index,
        )

        # Free the center tile result
        if center_tile_result is not None:
            del center_tile_result

        # Return values - load final count from zarr
        # Note: We don't load the full array into memory - just return metadata
        with open_ome_zarr(source_zarr_path, mode="r") as ds:
            final_labels_arr = ds[pos_path].zgroup["labels"][output_label_name]
            # Get max label by sampling (avoid loading full array)
            # For a proper count, we'd need to load the full array, but that defeats the purpose
            # The caller doesn't actually use objects_5d for Frangi - it writes directly from zarr
            pass

        # Return None for all arrays - the data is already in zarr
        # The caller (segment_organelles) will read from zarr directly
        elapsed_time = time.time() - start_time
        print(f"[{pos_path}] Two-pass tiled Frangi complete, took {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
        print(f"  Labels written to zarr: {source_zarr_path}/{pos_path}/labels/{output_label_name}")
        if save_vesselness and vesselness_label_name:
            print(f"  Vesselness written to zarr: {source_zarr_path}/{pos_path}/labels/{vesselness_label_name}")

        # Return None for all arrays - the data is already in zarr
        # The caller (segment_organelles) should detect this and skip writing
        vesselness_5d = None
        binary_5d = None
        objects_5d = None  # Data is in zarr, not in memory

        return pos_path, vesselness_5d, binary_5d, objects_5d, source_scale, crop_bbox

    except Exception as e:
        print(f"Error processing position {pos_path} for {organelle_name} with tiled Frangi: {e}")
        import traceback
        traceback.print_exc()
        raise


def segment_position_frangi(
    pos_path,
    source_zarr_path,
    channel_to_segment,
    organelle_name,
    frangi_params,
    use_gpu: bool,
    frangi_postprocess: bool,
    use_clahe: bool,
    post_clahe_smoothing_sigma: float,
    debug_output_path: str = None,
    clahe_params: dict = None,
    crop_bbox: tuple = None,
    save_vesselness: bool = False,
    input_mask_name: str = None,
    structure_type: str = None,
    force_tiled: bool = False,
    tile_size: int = 4096,
    tile_overlap: int = 256,
    nucleoli_method: str = None,
    vesicular_method: str = None,
    preview_mode: bool = False,
    shards_ratio: tuple = (1, 1, 1, 32, 32),
):
    """
    Worker function to segment an entire position using the Frangi filter.

    This function always uses tiled processing via segment_position_frangi_tiled(),
    which handles both small and large images efficiently. For small images, tiling
    still works correctly (may use just 1-2 tiles) with minimal overhead.

    Args:
        pos_path: Position path like "A/1/0"
        source_zarr_path: Path to the v3 zarr store
        channel_to_segment: Channel name to segment
        organelle_name: Name of the organelle being segmented
        frangi_params: Parameters for Frangi filter
        use_gpu: Whether to use GPU (note: tiled processing uses CPU workers)
        frangi_postprocess: Whether to apply postprocessing
        use_clahe: Whether to apply CLAHE preprocessing
        post_clahe_smoothing_sigma: Sigma for Gaussian smoothing after CLAHE
        debug_output_path: Optional path to save debug output
        clahe_params: Parameters for CLAHE
        crop_bbox: Optional tuple (y_start, y_end, x_start, x_end) for debug center crop
        save_vesselness: If True, also save the continuous Frangi vesselness map (default: False)
        input_mask_name: Optional mask name (e.g., "nuclear_seg") to constrain segmentation.
                         If provided, Frangi will only detect structures within the mask.
        force_tiled: Ignored (always uses tiled processing now)
        tile_size: Size of each tile for tiled processing (default: 4096)
        tile_overlap: Overlap between tiles (default: 256)
        nucleoli_method: For nucleoli segmentation: "blob" for LoG, "frangi" for Frangi
        vesicular_method: For vesicular segmentation: "blob" for LoG, "frangi" for Frangi
        preview_mode: If True, write results to temp zarr instead of original.
            Used for preview/debug mode to avoid modifying production data.
        shards_ratio: Sharding ratio for zarr v3 storage (default: (1, 1, 1, 32, 32)).
            Determines shard file size. With 512x512 base chunks and 32x32 ratio,
            each shard covers 16384x16384 pixels (~1GB for int32 labels).

    Returns:
        Tuple of (pos_path, vesselness_5d, binary_5d, objects_5d, source_scale, crop_bbox)
    """
    # Always use tiled processing - it handles both small and large images efficiently
    # For small images, it will use fewer tiles but the overhead is minimal
    return segment_position_frangi_tiled(
        pos_path=pos_path,
        source_zarr_path=source_zarr_path,
        channel_to_segment=channel_to_segment,
        organelle_name=organelle_name,
        frangi_params=frangi_params,
        frangi_postprocess=frangi_postprocess,
        use_clahe=use_clahe,
        post_clahe_smoothing_sigma=post_clahe_smoothing_sigma,
        clahe_params=clahe_params,
        crop_bbox=crop_bbox,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        save_vesselness=save_vesselness,
        input_mask_name=input_mask_name,
        structure_type=structure_type,
        debug_output_path=debug_output_path,
        nucleoli_method=nucleoli_method,
        vesicular_method=vesicular_method,
        preview_mode=preview_mode,
        shards_ratio=shards_ratio,
    )
