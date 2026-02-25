"""
Organelle feature extraction and comparison between fluorescent labels and brightfield.

This script compares organelle segmentation and feature extraction across 5 image types:
1. Fluorescent label (ground truth - e.g., mCherry mitochondria)
2. Raw brightfield (mid-slice)
3. Phase3D reconstruction (mid-slice)
4. Focus3D map (from autofocus - best Z-slice selection)
5. Phase2D reconstruction (from autofocus)

The goal is to validate how well phase reconstruction captures organelle information
compared to fluorescent markers, and compare different reconstruction approaches.

Optional nuclear masking (--mask-nuclei):
- Restricts segmentation to cytoplasm only (excludes nucleus)
- Loads cell and nuclear segmentation masks from stitched pheno_assembled store
- Cytoplasm mask = (cell_mask > 0) & (nuclear_mask == 0)
- Particularly useful for mitochondria, which are cytoplasmic organelles
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import tempfile

import click
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from iohub.ngff import open_ome_zarr
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import scipy.ndimage as ndi
import pandas as pd

from ops_utils.data.experiment import OpsDataset
from ops_utils.data.shifts import read_shifts
from organelle_profiler.feature_extraction.organelle_segmentation import (
    FrangiFilter,
    otsu_threshold,
    triangle_threshold,
)
from organelle_profiler.feature_extraction.network_analysis import calculate_network_features


def get_offset_cache_path(experiment: str, tile: str, output_dir: Path) -> Path:
    """
    Get path to the offset cache YAML file.

    Args:
        experiment: Experiment name
        tile: Tile position (e.g., "A/1/029020")
        output_dir: Output directory for caching

    Returns:
        Path to the offset cache file
    """
    tile_safe = tile.replace('/', '_')
    return output_dir / f"fluor_offset_cache_{experiment}_{tile_safe}.yaml"


def load_cached_offset(cache_path: Path) -> Optional[Tuple[float, float]]:
    """
    Load cached fluorescent offset from YAML file.

    Args:
        cache_path: Path to the cache file

    Returns:
        (y_offset, x_offset) tuple if cache exists, None otherwise
    """
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r') as f:
            data = yaml.safe_load(f)

        if data and 'offset_y' in data and 'offset_x' in data:
            y_offset = float(data['offset_y'])
            x_offset = float(data['offset_x'])
            print(f"[OffsetCache] Loaded cached offset: Y={y_offset}, X={x_offset}")
            return (y_offset, x_offset)
    except Exception as e:
        print(f"[OffsetCache] Warning: Failed to load cache from {cache_path}: {e}")

    return None


def save_offset_to_cache(cache_path: Path, y_offset: float, x_offset: float, overlap_area: float):
    """
    Save optimal fluorescent offset to YAML file.

    Args:
        cache_path: Path to the cache file
        y_offset: Optimal Y offset in pixels
        x_offset: Optimal X offset in pixels
        overlap_area: Overlap area at optimal offset
    """
    data = {
        'offset_y': float(y_offset),
        'offset_x': float(x_offset),
        'overlap_area': float(overlap_area),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"[OffsetCache] Saved offset to cache: {cache_path}")


def compute_overlap_area(fluor_seg: np.ndarray, recon_seg: np.ndarray) -> float:
    """
    Compute overlap area between two binary segmentations.

    Args:
        fluor_seg: Binary fluorescent segmentation
        recon_seg: Binary reconstruction segmentation

    Returns:
        Overlap area (number of overlapping pixels)
    """
    return float(np.sum((fluor_seg > 0) & (recon_seg > 0)))


def sweep_offset_for_optimal_overlap(
    fluor_img: np.ndarray,
    recon_img: np.ndarray,
    pixel_size: float = 0.325,
    sweep_range: int = 5,
    step_size: int = 1,
    use_gpu: bool = False,
    verbose: bool = False,
) -> Tuple[float, float, float]:
    """
    Sweep pixel offsets to find optimal overlap between fluorescent and reconstruction segmentations.

    Args:
        fluor_img: Fluorescent image
        recon_img: Reconstruction image (e.g., phase2d)
        pixel_size: Pixel size in microns
        sweep_range: Sweep range in pixels (+/- this value)
        step_size: Step size in pixels
        use_gpu: Use GPU for Frangi filter
        verbose: Print debug information

    Returns:
        (optimal_y_offset, optimal_x_offset, max_overlap_area) tuple
    """
    print(f"\n[OffsetSweep] Starting offset sweep...")
    print(f"[OffsetSweep] Range: [{-sweep_range}, {sweep_range}] pixels in Y and X")
    print(f"[OffsetSweep] Step size: {step_size} pixel(s)")

    # Segment the reference reconstruction image (no shift)
    print(f"[OffsetSweep] Segmenting reference reconstruction image...")
    _, recon_binary, _ = segment_organelles_frangi(
        recon_img,
        pixel_size=pixel_size,
        min_radius_um=0.1,
        max_radius_um=1.5,
        alpha=4.0,
        beta=0.5,
        threshold_multiplier=0.001,
        use_gpu=use_gpu,
        use_clahe=True,
        clahe_clip_limit=0.01,
        clahe_kernel_size=(64, 64),
        post_clahe_smoothing_sigma=1.0,
        verbose=verbose,
    )

    # Sweep through offsets
    best_offset_y = 0
    best_offset_x = 0
    max_overlap = 0

    y_offsets = range(-sweep_range, sweep_range + 1, step_size)
    x_offsets = range(-sweep_range, sweep_range + 1, step_size)
    total_iterations = len(y_offsets) * len(x_offsets)

    print(f"[OffsetSweep] Testing {total_iterations} offset combinations...")

    iteration = 0
    for dy in y_offsets:
        for dx in x_offsets:
            iteration += 1

            # Shift fluorescent image
            fluor_shifted = ndi.shift(fluor_img, shift=(dy, dx), order=1, mode='constant', cval=0.0)

            # Segment shifted fluorescent
            _, fluor_binary, _ = segment_organelles_frangi(
                fluor_shifted,
                pixel_size=pixel_size,
                min_radius_um=0.1,
                max_radius_um=1.5,
                alpha=4.0,
                beta=0.5,
                threshold_multiplier=0.001,
                use_gpu=use_gpu,
                use_clahe=False,
                verbose=False,
            )

            # Compute overlap
            overlap = compute_overlap_area(fluor_binary, recon_binary)

            if verbose and iteration % 10 == 0:
                print(f"[OffsetSweep] Progress: {iteration}/{total_iterations} | Current: dy={dy}, dx={dx}, overlap={overlap:.0f}")

            if overlap > max_overlap:
                max_overlap = overlap
                best_offset_y = dy
                best_offset_x = dx

    print(f"[OffsetSweep] ✓ Optimal offset found: Y={best_offset_y}, X={best_offset_x}")
    print(f"[OffsetSweep] ✓ Maximum overlap area: {max_overlap:.0f} pixels")

    return (float(best_offset_y), float(best_offset_x), max_overlap)


def load_crop_from_stores(
    dataset: OpsDataset,
    tile: str,
    crop_size: int = 512,
    crop_center: Optional[Tuple[int, int]] = None,
    t_index: int = 0,
    fluorescent_channel: str = "mCherry",
    apply_fluor_registration: bool = True,
    fluor_shift: Tuple[float, float] = (2.0, 0.0),
) -> Dict[str, np.ndarray]:
    """
    Load center crop from all available stores.
    
    Args:
        dataset: OpsDataset instance
        tile: Tile position (e.g., "A/1/029020")
        crop_size: Size of square crop in pixels
        crop_center: Optional (Y, X) pixel coordinates for crop center. 
                     If None, uses image center.
        t_index: Time index
        fluorescent_channel: Name of fluorescent channel to load
        apply_fluor_registration: If True, load from lc_20x_fluor_2d_registered store (pre-registered)
        fluor_shift: (Y, X) pixel shift to apply to fluorescent for fine alignment (default: (2, 0))
    
    Returns dict with keys:
        'fluorescent': Fluorescent channel (e.g., mCherry), optionally from registered store
        'raw_midslice': Raw BF mid-slice
        'phase3d_midslice': Phase3D reconstruction mid-slice
        'phase2d': Phase2D reconstruction (Channel 0 from autofocus)
        'crop_offset': (y, x) offset of crop in original image
    """
    result = {
        'fluorescent': None,
        'raw_midslice': None,
        'phase3d_midslice': None,
        'phase2d': None,
        'crop_offset': None,
    }
    
    # Get store paths for 20x pheno data
    raw_store_path = dataset.store_paths.get("lc_20x")
    fluor_registered_store_path = dataset.store_paths.get("lc_20x_fluor_2d_registered")
    phase3d_store_path = dataset.store_paths.get("lc_20x_phase")
    phase2d_store_path = dataset.store_paths.get("lc_20x_phase_2d")
    
    print(f"[LoadCrop] Loading tile: {tile}")
    print(f"[LoadCrop] Crop size: {crop_size}x{crop_size}")
    if crop_center:
        print(f"[LoadCrop] Custom crop center: Y={crop_center[0]}, X={crop_center[1]}")
    
    # Load from raw store (for BF channel) or registered store (for fluorescent)
    if raw_store_path and raw_store_path.exists():
        print(f"[LoadCrop] Loading from raw: {raw_store_path}")
        with open_ome_zarr(raw_store_path, mode="r") as store:
            channel_names = store.channel_names
            print(f"[LoadCrop] Available channels: {channel_names}")
            
            arr = store[tile]["0"]
            T, C, Z, Y_full, X_full = arr.shape
            
            # Calculate crop boundaries
            if crop_center is not None:
                cy, cx = crop_center
            else:
                # Default to image center
                cy, cx = Y_full // 2, X_full // 2
            
            half_crop = crop_size // 2
            y_start = cy - half_crop
            x_start = cx - half_crop
            y_end = y_start + crop_size
            x_end = x_start + crop_size
            
            # Validate bounds
            if y_start < 0 or x_start < 0 or y_end > Y_full or x_end > X_full:
                raise ValueError(
                    f"Crop region is out of bounds!\n"
                    f"  Image size: Y={Y_full}, X={X_full}\n"
                    f"  Crop center: Y={cy}, X={cx}\n"
                    f"  Crop size: {crop_size}x{crop_size}\n"
                    f"  Requested region: Y=[{y_start}, {y_end}), X=[{x_start}, {x_end})\n"
                    f"  Valid Y range for center: [{half_crop}, {Y_full - half_crop})\n"
                    f"  Valid X range for center: [{half_crop}, {X_full - half_crop})"
                )
            
            result['crop_offset'] = (y_start, x_start)
            
            # Load BF channel (mid-slice)
            if "BF" in channel_names:
                bf_idx = channel_names.index("BF")
                bf_vol = np.asarray(arr[t_index, bf_idx, :, y_start:y_end, x_start:x_end])
                mid_z = Z // 2
                result['raw_midslice'] = bf_vol[mid_z, :, :].astype(np.float32)
                print(f"[LoadCrop] ✓ Loaded BF mid-slice (Z={mid_z}): shape={result['raw_midslice'].shape}")
    
    # Load fluorescent channel from registered store (or raw if registration disabled)
    if apply_fluor_registration and fluor_registered_store_path and fluor_registered_store_path.exists():
        print(f"[LoadCrop] Loading registered fluorescent from: {fluor_registered_store_path}")
        with open_ome_zarr(fluor_registered_store_path, mode="r") as store:
            channel_names = store.channel_names
            print(f"[LoadCrop] Available channels: {channel_names}")
            
            if fluorescent_channel in channel_names:
                arr = store[tile]["0"]
                fluor_idx = channel_names.index(fluorescent_channel)
                # Registered store is already 2D (no Z dimension)
                T_fluor, C_fluor, Z_fluor, Y_fluor, X_fluor = arr.shape
                y_start, x_start = result['crop_offset']
                y_end = y_start + crop_size
                x_end = x_start + crop_size
                
                fluor_img = np.asarray(arr[t_index, fluor_idx, 0, y_start:y_end, x_start:x_end])
                fluor_img = fluor_img.astype(np.float32)
                
                # Apply fine alignment adjustment
                if fluor_shift and (fluor_shift[0] != 0 or fluor_shift[1] != 0):
                    fluor_img = ndi.shift(fluor_img, shift=fluor_shift, order=1, mode='constant', cval=0.0)
                    print(f"[LoadCrop]   Applied fine alignment shift: Y={fluor_shift[0]}, X={fluor_shift[1]}")
                
                result['fluorescent'] = fluor_img
                print(f"[LoadCrop] ✓ Loaded registered {fluorescent_channel}: shape={result['fluorescent'].shape}")
            else:
                print(f"[LoadCrop] ✗ Channel '{fluorescent_channel}' not found in registered store")
    elif raw_store_path and raw_store_path.exists():
        # Fall back to raw store (unregistered)
        print(f"[LoadCrop] Loading unregistered fluorescent from raw store")
        with open_ome_zarr(raw_store_path, mode="r") as store:
            channel_names = store.channel_names
            
            if fluorescent_channel in channel_names:
                arr = store[tile]["0"]
                fluor_idx = channel_names.index(fluorescent_channel)
                y_start, x_start = result['crop_offset']
                y_end = y_start + crop_size
                x_end = x_start + crop_size
                
                fluor_vol = np.asarray(arr[t_index, fluor_idx, :, y_start:y_end, x_start:x_end])
                result['fluorescent'] = fluor_vol.max(axis=0).astype(np.float32)
                print(f"[LoadCrop] ✓ Loaded unregistered {fluorescent_channel}: shape={result['fluorescent'].shape}")
            else:
                print(f"[LoadCrop] ✗ Channel '{fluorescent_channel}' not found")
    
    # Load from phase3d store (3D phase reconstruction)
    if phase3d_store_path and phase3d_store_path.exists():
        print(f"[LoadCrop] Loading from phase3d: {phase3d_store_path}")
        with open_ome_zarr(phase3d_store_path, mode="r") as store:
            arr = store[tile]["0"]
            T, C, Z, Y_full, X_full = arr.shape
            
            y_start, x_start = result['crop_offset']
            y_end = y_start + crop_size
            x_end = x_start + crop_size
            
            phase3d_vol = np.asarray(arr[t_index, 0, :, y_start:y_end, x_start:x_end])
            mid_z = Z // 2
            result['phase3d_midslice'] = phase3d_vol[mid_z, :, :].astype(np.float32)
            print(f"[LoadCrop] ✓ Loaded Phase3D mid-slice (Z={mid_z}): shape={result['phase3d_midslice'].shape}")
    
    # Load from phase2d store (contains both Phase2D and Focus3D channels)
    if phase2d_store_path and phase2d_store_path.exists():
        print(f"[LoadCrop] Loading from phase2d: {phase2d_store_path}")
        with open_ome_zarr(phase2d_store_path, mode="r") as store:
            arr = store[tile]["0"]
            T, C, Z, Y_full, X_full = arr.shape
            
            y_start, x_start = result['crop_offset']
            y_end = y_start + crop_size
            x_end = x_start + crop_size
            
            # Channel 0 is Phase2D (the actual 2D reconstruction)
            phase2d_vol = np.asarray(arr[t_index, 0, :, y_start:y_end, x_start:x_end])
            result['phase2d'] = phase2d_vol.max(axis=0).astype(np.float32)
            print(f"[LoadCrop] ✓ Loaded Phase2D (channel 0): shape={result['phase2d'].shape}")
            
    
    return result


def load_cytoplasm_mask_from_stitched(
    dataset: OpsDataset,
    tile: str,
    crop_size: int = 512,
    crop_offset: Tuple[int, int] = (0, 0),
    t_index: int = 0,
    verbose: bool = False,
) -> Optional[np.ndarray]:
    """
    Load cytoplasm mask from stitched pheno_assembled store.
    
    Cytoplasm mask = pixels inside cell BUT outside nucleus, with buffer zones.
    This is computed as: (eroded_cell_mask) & ~(eroded_nuclear_mask)
    - Cell mask is eroded by 2 pixels to exclude plasma membrane
    - Nuclear mask is eroded by 2 pixels to create safety buffer around nucleus
    
    If multiple cells are present in the crop, only the LARGEST connected component
    is kept to ensure analysis focuses on a single cell.
    
    Args:
        dataset: OpsDataset instance
        tile: Tile position (e.g., "A/1/029020")
        crop_size: Size of square crop in pixels
        crop_offset: (y, x) offset of crop in tile space
        t_index: Time index
        verbose: Print debug information
    
    Returns:
        Binary mask array (crop_size x crop_size) where 1 = cytoplasm, 0 = background/nucleus/membrane
        Only the largest cell's cytoplasm region is included.
        Returns None if masks cannot be loaded.
    """
    # Use pheno_assembled store (organized by well, e.g., "A/1/0")
    pheno_assembled_path = dataset.store_paths.get("pheno_assembled")
    
    if not pheno_assembled_path or not pheno_assembled_path.exists():
        print(f"[CytoplasmMask] ✗ pheno_assembled store not found: {pheno_assembled_path}")
        return None
    
    try:
        # Parse tile to get well (e.g., "A/1/029020" -> "A/1/0")
        tile_parts = Path(tile).parts
        if len(tile_parts) < 2:
            print(f"[CytoplasmMask] ✗ Invalid tile format: {tile}")
            return None
        well = f"{tile_parts[0]}/{tile_parts[1]}/0"  # Well format for pheno_assembled
        
        print(f"[CytoplasmMask] Loading masks from {well} in stitched space...")
        print(f"[CytoplasmMask] Store path: {pheno_assembled_path}")
        
        # Open pheno_assembled store
        with open_ome_zarr(pheno_assembled_path, mode="r") as store:
            if verbose:
                print(f"[CytoplasmMask] Store opened successfully")
            
            position = store[well]
            if verbose:
                print(f"[CytoplasmMask] Position {well} accessed")
                print(f"[CytoplasmMask] Available keys in position: {list(position.zgroup.keys())}")
            
            # Access 'seg' (cell mask) and 'nuclear_seg' (nuclear mask) as raw zarr arrays
            # These are stored at: phenotyping.zarr/A/1/0/seg and phenotyping.zarr/A/1/0/nuclear_seg
            # Use .zgroup to access the underlying zarr group directly
            cell_seg_zarr = position.zgroup["seg"]["0"]  # Access the '0' array inside seg
            nuclear_seg_zarr = position.zgroup["nuclear_seg"]["0"]  # Access the '0' array inside nuclear_seg
            if verbose:
                print(f"[CytoplasmMask] Accessed seg and nuclear_seg successfully")
                print(f"[CytoplasmMask] Full seg shape: {cell_seg_zarr.shape}")
            
            # Get stitch shifts to map tile coords to stitched coords
            shifts = read_shifts(dataset, "lc_20x_stitch", well)
            
            # Find our tile's shift
            if tile not in shifts:
                print(f"[CytoplasmMask] ✗ Tile {tile} not found in stitch shifts")
                print(f"[CytoplasmMask] Available tiles: {list(shifts.keys())[:5]}...")
                return None
            
            tile_shift_y, tile_shift_x = shifts[tile]
            print(f"[CytoplasmMask] Tile shift: Y={tile_shift_y}, X={tile_shift_x}")
            
            # Convert tile crop coords to stitched coords
            crop_y, crop_x = crop_offset
            stitched_y = tile_shift_y + crop_y
            stitched_x = tile_shift_x + crop_x
            
            print(f"[CytoplasmMask] Stitched crop region: Y=[{stitched_y}, {stitched_y + crop_size}), X=[{stitched_x}, {stitched_x + crop_size})")
            
            # Determine dimensionality to slice correctly
            # Arrays could be (T, C, Z, Y, X) or (C, Z, Y, X) or (Z, Y, X) or (Y, X)
            ndim = cell_seg_zarr.ndim
            
            # Bounds check for the spatial dimensions (last 2 dims are always Y, X)
            full_shape = cell_seg_zarr.shape
            spatial_y_max = full_shape[-2]
            spatial_x_max = full_shape[-1]
            
            if (stitched_y < 0 or stitched_x < 0 or 
                stitched_y + crop_size > spatial_y_max or 
                stitched_x + crop_size > spatial_x_max):
                print(f"[CytoplasmMask] ✗ Crop region out of bounds (stitched image shape: {full_shape})")
                print(f"[CytoplasmMask]   Y range: [0, {spatial_y_max}), X range: [0, {spatial_x_max})")
                return None
            
            # Load ONLY the crop region directly from zarr (efficient!)
            # Slice based on dimensionality
            if ndim == 5:
                # (T, C, Z, Y, X)
                cell_crop = np.asarray(cell_seg_zarr[t_index, 0, 0, stitched_y:stitched_y + crop_size, stitched_x:stitched_x + crop_size])
                nuclear_crop = np.asarray(nuclear_seg_zarr[t_index, 0, 0, stitched_y:stitched_y + crop_size, stitched_x:stitched_x + crop_size])
            elif ndim == 4:
                # (C, Z, Y, X)
                cell_crop = np.asarray(cell_seg_zarr[0, 0, stitched_y:stitched_y + crop_size, stitched_x:stitched_x + crop_size])
                nuclear_crop = np.asarray(nuclear_seg_zarr[0, 0, stitched_y:stitched_y + crop_size, stitched_x:stitched_x + crop_size])
            elif ndim == 3:
                # (Z, Y, X)
                cell_crop = np.asarray(cell_seg_zarr[0, stitched_y:stitched_y + crop_size, stitched_x:stitched_x + crop_size])
                nuclear_crop = np.asarray(nuclear_seg_zarr[0, stitched_y:stitched_y + crop_size, stitched_x:stitched_x + crop_size])
            else:
                # (Y, X) - already 2D
                cell_crop = np.asarray(cell_seg_zarr[stitched_y:stitched_y + crop_size, stitched_x:stitched_x + crop_size])
                nuclear_crop = np.asarray(nuclear_seg_zarr[stitched_y:stitched_y + crop_size, stitched_x:stitched_x + crop_size])
            
            # Find the largest cell label in the cell segmentation
            # cell_crop contains integer labels for each cell (0 = background)
            unique_cells = np.unique(cell_crop[cell_crop > 0])
            
            if len(unique_cells) == 0:
                print(f"[CytoplasmMask] ✗ No cells found in crop")
                return None
            
            # Find which cell label has the most pixels
            cell_sizes = {cell_id: np.sum(cell_crop == cell_id) for cell_id in unique_cells}
            largest_cell_id = max(cell_sizes, key=cell_sizes.get)
            
            if len(unique_cells) > 1:
                print(f"[CytoplasmMask] Found {len(unique_cells)} cells in crop, using largest (ID={largest_cell_id}, {cell_sizes[largest_cell_id]} pixels)")
            
            # Create cytoplasm mask for ONLY the largest cell: cell mask minus nuclear mask
            cell_mask_original = (cell_crop == largest_cell_id)
            
            # Also find the corresponding nuclear mask for this cell
            # Nuclear segmentation typically has matching IDs to cell segmentation
            nuclear_mask = (nuclear_crop == largest_cell_id) if largest_cell_id in np.unique(nuclear_crop) else (nuclear_crop > 0) & cell_mask_original
            
            # Apply morphological operations to exclude membrane and create buffer zones
            from scipy.ndimage import binary_erosion, generate_binary_structure
            erosion_structure = generate_binary_structure(2, 1)  # 2D cross-shaped kernel

            # Use less erosion for vesicular organelles to preserve small round structures
            is_vesicular = (dataset.experiment == "ops0065_20250812")
            cell_erosion_iterations = 2 if is_vesicular else 10

            # 1) Erode cell mask to exclude plasma membrane segmentations
            cell_mask = cell_mask_original.copy()
            for _ in range(cell_erosion_iterations):
                cell_mask = binary_erosion(cell_mask, structure=erosion_structure)
            
            # 2) Shrink nuclear mask by 2 pixels to create buffer zone around nucleus
            nuclear_mask_original = nuclear_mask.copy()
            for _ in range(2):
                nuclear_mask = binary_erosion(nuclear_mask, structure=erosion_structure)
            
            # Cytoplasm = eroded cell - eroded nucleus (excludes membrane + nuclear buffer)
            cytoplasm_mask = cell_mask & ~nuclear_mask
            
            print(f"[CytoplasmMask] ✓ Cytoplasm mask shape: {cytoplasm_mask.shape}")
            print(f"[CytoplasmMask] ✓ Cell pixels (original): {cell_mask_original.sum()}, Cell pixels (eroded 2px): {cell_mask.sum()}")
            print(f"[CytoplasmMask] ✓ Nuclear pixels (original): {nuclear_mask_original.sum()}, Nuclear pixels (eroded 20px): {nuclear_mask.sum()}")
            print(f"[CytoplasmMask] ✓ Cytoplasm pixels: {cytoplasm_mask.sum()} ({100*cytoplasm_mask.sum()/cytoplasm_mask.size:.1f}% of crop)")
            print(f"[CytoplasmMask] ✓ Exclusions: 2px cell erosion (membrane) + 20px nuclear buffer")
            
            return cytoplasm_mask.astype(np.uint8)
            
    except KeyError as e:
        print(f"[CytoplasmMask] ✗ KeyError accessing masks: {e}")
        print(f"[CytoplasmMask]   Tried to access: {pheno_assembled_path}/{well}/seg and {pheno_assembled_path}/{well}/nuclear_seg")
        return None
    except Exception as e:
        import traceback
        print(f"[CytoplasmMask] ✗ Error loading cytoplasm mask: {type(e).__name__}: {e}")
        if verbose:
            print(f"[CytoplasmMask] Full traceback:")
            traceback.print_exc()
        return None


def segment_organelles_frangi(
    image: np.ndarray,
    pixel_size: float = 0.325,
    min_radius_um: float = 0.2,
    max_radius_um: float = 1.5,
    alpha: float = 0.5,
    beta: float = 0.5,
    threshold_multiplier: float = 0.1,
    use_gpu: bool = False,
    use_clahe: bool = True,
    clahe_clip_limit: float = 0.03,
    clahe_kernel_size: Tuple[int, int] = (256, 256),
    post_clahe_smoothing_sigma: float = 1.0,
    cytoplasm_mask: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment organelles using Frangi vesselness filter.
    
    Args:
        image: 2D image array
        pixel_size: Pixel size in microns (for Y and X)
        min_radius_um: Minimum organelle radius in microns
        max_radius_um: Maximum organelle radius in microns
        alpha: Frangi plate-like sensitivity (default: 0.5, higher=more sensitive to tubes)
        beta: Frangi blob-like sensitivity (default: 0.5)
        threshold_multiplier: Multiplier for vesselness threshold (default: 0.1, higher=more conservative)
        use_gpu: Whether to use GPU acceleration
        use_clahe: Whether to apply CLAHE preprocessing (default: True)
        clahe_clip_limit: CLAHE clip limit (default: 0.03)
        clahe_kernel_size: CLAHE kernel size in pixels (default: (256, 256))
        post_clahe_smoothing_sigma: Gaussian smoothing sigma after CLAHE (default: 1.0)
        cytoplasm_mask: Optional binary mask to restrict segmentation (1=cytoplasm, 0=exclude)
        verbose: Print debug information
    
    Returns:
        vesselness_map: Filtered vesselness response
        binary_mask: Thresholded binary mask
        labeled_mask: Connected component labels
    """
    if verbose:
        print(f"[Segment] Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"[Segment] Image range: [{image.min():.2f}, {image.max():.2f}]")
    
    # --- Pre-processing with CLAHE (matching organelle_segmentation.py) ---
    if use_clahe:
        from skimage.exposure import equalize_adapthist
        
        original_dtype = image.dtype
        if verbose:
            print(f"[Segment] Applying CLAHE (clip_limit={clahe_clip_limit}, kernel_size={clahe_kernel_size})...")
        
        # CLAHE on integer images
        if np.issubdtype(original_dtype, np.integer):
            image = equalize_adapthist(
                image.astype(np.uint16), 
                clip_limit=clahe_clip_limit,
                kernel_size=clahe_kernel_size
            )
            image = image.astype(original_dtype)
        else:  # Float images
            original_min, original_max = np.min(image), np.max(image)
            if original_max > original_min:
                image_float = (image - original_min) / (original_max - original_min)
                image_clahe = equalize_adapthist(image_float, clip_limit=clahe_clip_limit, kernel_size=clahe_kernel_size)
                image = (image_clahe * (original_max - original_min) + original_min).astype(original_dtype)
        
        # Optional Gaussian smoothing to reduce CLAHE-induced noise
        if post_clahe_smoothing_sigma > 0:
            import scipy.ndimage as scipy_ndi_smooth
            image = scipy_ndi_smooth.gaussian_filter(image, sigma=post_clahe_smoothing_sigma)
            if verbose:
                print(f"[Segment] Applied post-CLAHE smoothing (sigma={post_clahe_smoothing_sigma})")
    
    # Setup for CPU processing
    import scipy.ndimage as scipy_ndi
    xp = np
    ndi = scipy_ndi
    
    # Convert to GPU if requested
    if use_gpu:
        import cupy as cp
        import cupyx.scipy.ndimage as cupy_ndi
        xp = cp
        ndi = cupy_ndi
        image = xp.asarray(image)
    
    # Create pixel resolution dict
    # Match organelle_segmentation.py exactly - they override with fixed values
    pixel_resolution = {'Z': 1.0, 'Y': 0.325, 'X': 0.325}
    
    if verbose:
        print(f"[Frangi] Pixel resolution: Y={pixel_resolution['Y']}, X={pixel_resolution['X']} μm")
        print(f"[Frangi] Radius range: {min_radius_um} - {max_radius_um} μm")
    
    # Run Frangi filter
    frangi = FrangiFilter(
        image_data=image,
        pixel_resolution=pixel_resolution,
        min_radius_um=min_radius_um,
        max_radius_um=max_radius_um,
        alpha=alpha,
        beta=beta,
        verbose=verbose,
        use_gpu=use_gpu,
        remove_edges=True,
    )
    vesselness_map = frangi.run()
    
    # Threshold using log-based method
    if xp.any(vesselness_map > 0):
        positive_vesselness = vesselness_map[vesselness_map > 0]
        log_vesselness = xp.log10(positive_vesselness)
        
        tri_thresh_log = triangle_threshold(log_vesselness, xp=xp)
        otsu_thresh_log, _ = otsu_threshold(log_vesselness, xp=xp)
        
        triangle_thresh_linear = 10**tri_thresh_log
        otsu_thresh_linear = 10**otsu_thresh_log
        
        threshold = threshold_multiplier * min(triangle_thresh_linear, otsu_thresh_linear)
        binary_mask = vesselness_map > threshold
        
        if verbose:
            print(f"[Segment] Threshold: {threshold:.6f}")
            print(f"[Segment] Pixels above threshold: {xp.sum(binary_mask)}")
        
        # Apply cytoplasm mask if provided (restrict to cytoplasm only)
        if cytoplasm_mask is not None:
            if use_gpu:
                cytoplasm_mask_gpu = xp.asarray(cytoplasm_mask)
                binary_mask = binary_mask & cytoplasm_mask_gpu
            else:
                binary_mask = binary_mask & cytoplasm_mask
            if verbose:
                print(f"[Segment] Applied cytoplasm mask, pixels remaining: {xp.sum(binary_mask)}")
        
        # Post-processing (skip morphological operations for maximum sensitivity)
        # Binary opening can remove thin tubular structures, so skip it
        
        # Label connected components directly
        footprint = ndi.generate_binary_structure(2, 1)
        labeled_mask, num_features = ndi.label(binary_mask, structure=footprint)
        
        if verbose:
            print(f"[Segment] Found {num_features} objects")
        
        # Remove small objects (< 4 pixels)
        if num_features > 0:
            areas = xp.bincount(labeled_mask.ravel())[1:]
            small_objects = xp.where(areas < 2)[0] + 1
            binary_mask[xp.isin(labeled_mask, small_objects)] = False
            labeled_mask, num_features = ndi.label(binary_mask, structure=footprint)
            
            if verbose:
                print(f"[Segment] After filtering: {num_features} objects")
    else:
        binary_mask = xp.zeros_like(vesselness_map, dtype=bool)
        labeled_mask = xp.zeros_like(vesselness_map, dtype=xp.int32)
    
    # Convert back to CPU
    if use_gpu:
        vesselness_map = xp.asnumpy(vesselness_map)
        binary_mask = xp.asnumpy(binary_mask)
        labeled_mask = xp.asnumpy(labeled_mask)
    
    return vesselness_map.astype(np.float32), binary_mask.astype(np.uint8), labeled_mask.astype(np.int32)


def extract_organelle_features(
    labeled_mask: np.ndarray,
    spacing: tuple = (0.325, 0.325),
    intensity_image: np.ndarray = None,
    frangi_image: np.ndarray = None,
    full_features: bool = True,
) -> pd.DataFrame:
    """
    Extract comprehensive morphological features from labeled organelles.
    
    This uses the same feature extraction approach as feature_extraction.py
    to provide consistent measurements across the analysis pipeline.
    
    Args:
        labeled_mask: Labeled mask with integer IDs for each organelle
        spacing: Pixel spacing (Y, X) in microns
        intensity_image: Optional intensity image for texture/intensity features
        frangi_image: Optional Frangi vesselness map for additional features
        full_features: If True, compute expensive features (Haralick, Hu moments, etc.)
    
    Returns DataFrame with comprehensive feature set including:
        - Basic morphology: area, perimeter, axes, eccentricity, solidity, extent
        - Shape descriptors: aspect_ratio, circularity, convex_area, equivalent_diameter
        - Hu moments (shape invariants)
        - Intensity statistics (if intensity_image provided)
        - Frangi statistics (if frangi_image provided)
        - Haralick texture features (if full_features=True)
        - Centroid coordinates
    """
    from skimage.measure import regionprops_table
    
    if not np.any(labeled_mask > 0):
        return pd.DataFrame()
    
    # Define base properties
    base_properties = [
        'label',
        'area',
        'perimeter',
        'axis_major_length',
        'axis_minor_length',
        'solidity',
        'extent',
        'orientation',
        'equivalent_diameter_area',
        'convex_area',
        'eccentricity',
        'centroid',
    ]
    
    # Add intensity properties if available
    if intensity_image is not None:
        base_properties.extend(['mean_intensity', 'min_intensity', 'max_intensity'])
    
    # Extract base features with spacing
    props_df = pd.DataFrame(
        regionprops_table(
            labeled_mask,
            intensity_image=intensity_image,
            properties=tuple(base_properties),
            spacing=spacing,
        )
    )
    
    if props_df.empty:
        return pd.DataFrame()
    
    # Rename centroid columns
    props_df.rename(columns={'centroid-0': 'centroid_y', 'centroid-1': 'centroid_x'}, inplace=True)
    
    # Extract Hu moments (without spacing)
    hu_props = pd.DataFrame(
        regionprops_table(
            labeled_mask,
            properties=('label', 'moments_hu'),
            spacing=(1, 1),
        )
    )
    props_df = pd.merge(props_df, hu_props, on='label', how='left')
    
    # Calculate derived features
    minor_axis = props_df['axis_minor_length'].replace(0, 1)
    props_df['aspect_ratio'] = props_df['axis_major_length'] / minor_axis
    
    perimeter_sq = props_df['perimeter'] ** 2
    props_df['circularity'] = np.divide(
        4 * np.pi * props_df['area'],
        perimeter_sq,
        out=np.ones_like(perimeter_sq),
        where=perimeter_sq != 0,
    )
    
    # Add intensity sum and std if intensity image provided
    if intensity_image is not None:
        sum_std_features = []
        for prop in regionprops(labeled_mask, intensity_image=intensity_image):
            min_r, min_c, max_r, max_c = prop.bbox
            region_pixels = intensity_image[min_r:max_r, min_c:max_c][prop.image]
            if region_pixels.size > 0:
                sum_std_features.append({
                    'label': prop.label,
                    'sum_intensity': np.sum(region_pixels),
                    'std_intensity': np.std(region_pixels),
                })
        if sum_std_features:
            sum_std_df = pd.DataFrame(sum_std_features)
            props_df = pd.merge(props_df, sum_std_df, on='label', how='left')
    
    # Add Frangi features if available
    if frangi_image is not None:
        frangi_props = pd.DataFrame(
            regionprops_table(
                labeled_mask,
                intensity_image=frangi_image,
                properties=('label', 'mean_intensity', 'min_intensity', 'max_intensity'),
            )
        )
        frangi_props.rename(
            columns={
                'mean_intensity': 'frangi_mean_intensity',
                'min_intensity': 'frangi_min_intensity',
                'max_intensity': 'frangi_max_intensity',
            },
            inplace=True,
        )
        props_df = pd.merge(props_df, frangi_props, on='label', how='left')
        
        # Frangi sum and std
        frangi_sum_std = []
        for prop in regionprops(labeled_mask):
            min_r, min_c, max_r, max_c = prop.bbox
            region_pixels = frangi_image[min_r:max_r, min_c:max_c][prop.image]
            if region_pixels.size > 0:
                frangi_sum_std.append({
                    'label': prop.label,
                    'frangi_sum_intensity': np.sum(region_pixels),
                    'frangi_std_intensity': np.std(region_pixels),
                })
        if frangi_sum_std:
            frangi_sum_std_df = pd.DataFrame(frangi_sum_std)
            props_df = pd.merge(props_df, frangi_sum_std_df, on='label', how='left')
    
    # Expensive features if requested
    if full_features:
        from skimage.feature import graycomatrix, graycoprops
        
        haralick_features = []
        convexity_features = []
        
        for prop in regionprops(labeled_mask, intensity_image=intensity_image):
            # Haralick texture features (requires intensity image)
            if intensity_image is not None:
                min_r, min_c, max_r, max_c = prop.bbox
                region_intensity = intensity_image[min_r:max_r, min_c:max_c] * prop.image
                
                min_val, max_val = region_intensity.min(), region_intensity.max()
                if max_val > min_val:
                    region_intensity_uint = ((region_intensity - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    region_intensity_uint = np.zeros_like(region_intensity, dtype=np.uint8)
                
                glcm = graycomatrix(
                    region_intensity_uint,
                    distances=[1],
                    angles=[0],
                    levels=256,
                    symmetric=True,
                    normed=True,
                )
                haralick_features.append({
                    'label': prop.label,
                    'haralick_contrast': graycoprops(glcm, 'contrast')[0, 0],
                    'haralick_correlation': graycoprops(glcm, 'correlation')[0, 0],
                    'haralick_energy': graycoprops(glcm, 'energy')[0, 0],
                    'haralick_homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
                })
            
            # Convexity
            if prop.perimeter > 0 and hasattr(prop, 'perimeter_crofton') and prop.perimeter_crofton > 0:
                convexity = prop.perimeter / prop.perimeter_crofton
            else:
                convexity = 1.0
            convexity_features.append({'label': prop.label, 'convexity': convexity})
        
        if haralick_features:
            haralick_df = pd.DataFrame(haralick_features)
            props_df = pd.merge(props_df, haralick_df, on='label', how='left')
        
        if convexity_features:
            convexity_df = pd.DataFrame(convexity_features)
            props_df = pd.merge(props_df, convexity_df, on='label', how='left')
    
    return props_df


def compute_feature_overlap(
    ref_features: pd.DataFrame,
    test_features: pd.DataFrame,
    distance_threshold: float = 10.0,
) -> Dict[str, float]:
    """
    Compute feature overlap between reference (fluorescent) and test (BF) organelles.
    
    Strategy:
    1. Match organelles by centroid distance
    2. For matched pairs, compute feature similarity
    3. Return per-feature overlap percentages
    
    Returns dict with keys:
        'area', 'axis_major_length', 'axis_minor_length', 'eccentricity', 
        'solidity', 'extent', 'n_matched', 'n_ref', 'n_test'
    """
    if len(ref_features) == 0 or len(test_features) == 0:
        return {
            'area': 0.0,
            'axis_major_length': 0.0,
            'axis_minor_length': 0.0,
            'eccentricity': 0.0,
            'solidity': 0.0,
            'extent': 0.0,
            'n_matched': 0,
            'n_ref': len(ref_features),
            'n_test': len(test_features),
        }
    
    # Match by centroid distance using Hungarian algorithm
    ref_centroids = ref_features[['centroid_y', 'centroid_x']].values
    test_centroids = test_features[['centroid_y', 'centroid_x']].values
    
    # Compute pairwise distances
    distances = cdist(ref_centroids, test_centroids)
    
    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(distances)
    
    # Filter matches by distance threshold
    valid_matches = distances[row_ind, col_ind] < distance_threshold
    row_ind = row_ind[valid_matches]
    col_ind = col_ind[valid_matches]
    
    n_matched = len(row_ind)
    
    if n_matched == 0:
        return {
            'area': 0.0,
            'axis_major_length': 0.0,
            'axis_minor_length': 0.0,
            'eccentricity': 0.0,
            'solidity': 0.0,
            'extent': 0.0,
            'n_matched': 0,
            'n_ref': len(ref_features),
            'n_test': len(test_features),
        }
    
    # Compute feature deviations for matched pairs
    # Use column names as they appear in regionprops_table output
    feature_cols = ['area', 'axis_major_length', 'axis_minor_length', 'eccentricity', 'solidity', 'extent']
    
    overlaps = {}
    for feat in feature_cols:
        ref_vals = ref_features.iloc[row_ind][feat].values
        test_vals = test_features.iloc[col_ind][feat].values
        
        # Compute similarity as percentage: 100% = perfect match, 0% = completely off
        # similarity = 100 * (1 - |test - ref| / ref)
        # Result: 100% = perfect match, 50% = 50% off, 0% = 100% off or worse
        with np.errstate(divide='ignore', invalid='ignore'):
            abs_deviation = np.abs(test_vals - ref_vals) / ref_vals
            # Cap at 1.0 (100% deviation) for values that differ by more than 100%
            abs_deviation = np.minimum(abs_deviation, 1.0)
            similarity = (1.0 - abs_deviation) * 100.0
            # Handle division by zero (when ref_val = 0)
            similarity = np.where(np.isfinite(similarity), similarity, 0.0)
        
        overlaps[feat] = np.mean(similarity)
    
    overlaps['n_matched'] = n_matched
    overlaps['n_ref'] = len(ref_features)
    overlaps['n_test'] = len(test_features)
    
    return overlaps


def plot_raw_images(
    images: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "Raw Image Comparison",
    fluorophore: str = "mCherry",
):
    """
    Create 1x4 grid showing raw images without segmentation overlays.
    """
    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, 4, figure=fig, hspace=0.3, wspace=0.05)

    order = ['fluorescent', 'raw_midslice', 'phase3d_midslice', 'phase2d']
    titles = {
        'fluorescent': f'Fluorescent ({fluorophore})\nGround Truth',
        'raw_midslice': 'Raw BF\n(mid-slice)',
        'phase3d_midslice': 'Phase3D Recon\n(mid-slice)',
        'phase2d': 'Phase2D Recon\n(autofocus)',
    }

    for idx, key in enumerate(order):
        if key not in images or images[key] is None:
            continue

        ax = fig.add_subplot(gs[0, idx])
        
        # Show image
        img = images[key]
        if key == 'fluorescent':
            # Use inferno for fluorescent
            ax.imshow(img, cmap='inferno', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        else:
            # Use gray for BF/phase
            ax.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        
        ax.set_title(titles[key], fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                             edgecolor='black', alpha=0.8, linewidth=1.5))
        ax.axis('off')

    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.05)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved raw images to {output_path}")
    plt.close()


def plot_labeled_masks(
    masks: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "Labeled Object Segmentations",
    fluorophore: str = "mCherry",
):
    """
    Create 1x4 grid showing labeled segmentation masks to visualize connectivity.
    Each object gets a unique color to show individual organelle structures.
    """
    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, 4, figure=fig, hspace=0.3, wspace=0.05)

    order = ['fluorescent', 'raw_midslice', 'phase3d_midslice', 'phase2d']
    titles = {
        'fluorescent': f'Fluorescent ({fluorophore})\nGround Truth Labels',
        'raw_midslice': 'Raw BF\n(mid-slice) Labels',
        'phase3d_midslice': 'Phase3D Recon\n(mid-slice) Labels',
        'phase2d': 'Phase2D Recon\n(autofocus) Labels',
    }

    for idx, key in enumerate(order):
        if key not in masks or masks[key] is None:
            continue

        ax = fig.add_subplot(gs[0, idx])
        
        # Show labeled mask with unique random colors per object
        mask = masks[key]
        n_objects = int(mask.max())
        
        # Generate unique random colors for each object
        # Use a deterministic seed based on object count for reproducibility
        np.random.seed(42)
        
        # Create RGB image with unique color per label
        colored_labels = np.zeros((*mask.shape, 3), dtype=np.float32)
        
        # Generate distinct colors using HSV space for better distribution
        if n_objects > 0:
            for label_id in range(1, n_objects + 1):
                # Generate unique hue for each object (evenly spaced in HSV space)
                hue = (label_id * 0.618033988749895) % 1.0  # Golden ratio for good distribution
                saturation = 0.7 + 0.3 * np.random.rand()  # High saturation
                value = 0.7 + 0.3 * np.random.rand()  # High brightness
                
                # Convert HSV to RGB
                from matplotlib.colors import hsv_to_rgb
                rgb = hsv_to_rgb([hue, saturation, value])
                
                # Apply color to all pixels with this label
                colored_labels[mask == label_id] = rgb
        
        ax.imshow(colored_labels, interpolation='nearest')
        
        # Add object count
        ax.text(10, 30, f'N={n_objects}', color='white', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        ax.set_title(titles[key], fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                             edgecolor='black', alpha=0.8, linewidth=1.5))
        ax.axis('off')
    
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.05)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved labeled masks to {output_path}")
    plt.close()


def plot_segmentation_overlays(
    images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "Organelle Segmentation Comparison",
    fluorophore: str = "mCherry",
):
    """
    Create 1x4 grid showing segmentations overlaid on images.
    """
    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, 4, figure=fig, hspace=0.3, wspace=0.05)

    order = ['fluorescent', 'raw_midslice', 'phase3d_midslice', 'phase2d']
    titles = {
        'fluorescent': f'Fluorescent ({fluorophore})\nGround Truth',
        'raw_midslice': 'Raw BF\n(mid-slice)',
        'phase3d_midslice': 'Phase3D Recon\n(mid-slice)',
        'phase2d': 'Phase2D Recon\n(autofocus)',
    }

    for idx, key in enumerate(order):
        if key not in images or images[key] is None:
            continue

        ax = fig.add_subplot(gs[0, idx])
        
        # Show image
        img = images[key]
        if key == 'fluorescent':
            # Use inferno for fluorescent
            ax.imshow(img, cmap='inferno', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        else:
            # Use gray for BF/phase
            ax.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        
        # Overlay segmentation contours
        if key in masks and masks[key] is not None:
            from skimage.segmentation import find_boundaries
            boundaries = find_boundaries(masks[key], mode='outer')
            # Create RGBA overlay (cyan contours)
            overlay = np.zeros((*boundaries.shape, 4))
            overlay[boundaries, :] = [0, 1, 1, 1]  # Cyan
            ax.imshow(overlay)
            
            # Count objects
            n_objects = masks[key].max()
            ax.text(10, 30, f'N={n_objects}', color='cyan', fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        ax.set_title(titles[key], fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                             edgecolor='black', alpha=0.8, linewidth=1.5))
        ax.axis('off')
    
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.05)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved segmentation overlays to {output_path}")
    plt.close()


def plot_feature_overlap_bars(
    overlap_results: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = "Feature Overlap with Fluorescent Label",
):
    """
    Create bar plot showing feature overlap percentages.
    
    X-axis: Image type (Fluorescent, Raw BF, Phase3D, Phase2D)
    Y-axis: % Feature overlap
    Each dot: Different feature metric
    Bar height: Mean across features
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    image_types = ['fluorescent', 'raw_midslice', 'phase3d_midslice', 'phase2d']
    image_labels = {
        'fluorescent': 'Fluorescent\n(mCherry)',
        'raw_midslice': 'Raw BF\n(mid-slice)',
        'phase3d_midslice': 'Phase3D\n(mid-slice)',
        'phase2d': 'Phase2D\n(autofocus)',
    }
    
    feature_names = ['area', 'axis_major_length', 'axis_minor_length', 'eccentricity', 'solidity', 'extent']
    feature_labels = {
        'area': 'Area',
        'axis_major_length': 'Major Axis',
        'axis_minor_length': 'Minor Axis',
        'eccentricity': 'Eccentricity',
        'solidity': 'Solidity',
        'extent': 'Extent',
    }
    
    x_positions = np.arange(len(image_types))
    bar_width = 0.6
    
    # Collect data
    means = []
    feature_values = {feat: [] for feat in feature_names}
    
    for img_type in image_types:
        if img_type in overlap_results:
            overlap = overlap_results[img_type]
            values = [overlap.get(feat, 0.0) for feat in feature_names]
            means.append(np.mean(values))
            
            for feat, val in zip(feature_names, values):
                feature_values[feat].append(val)
        else:
            means.append(0.0)
            for feat in feature_names:
                feature_values[feat].append(0.0)
    
    # Plot bars
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']  # Red, Blue, Green, Purple, Orange
    bars = ax.bar(x_positions, means, bar_width, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Plot individual feature dots
    feature_colors = plt.cm.Set3(np.linspace(0, 1, len(feature_names)))
    
    for feat_idx, feat in enumerate(feature_names):
        y_values = feature_values[feat]
        # Add jitter to x positions for visibility
        jitter = np.random.normal(0, 0.05, len(x_positions))
        ax.scatter(x_positions + jitter, y_values, 
                  s=120, alpha=0.8, edgecolors='black', linewidth=1.5,
                  color=feature_colors[feat_idx], label=feature_labels[feat], zorder=3)
    
    # Styling
    ax.set_ylabel('Feature Similarity (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Image Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([image_labels[t] for t in image_types], fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', frameon=True, fontsize=10, ncol=2)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{mean_val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.text(0.5, -0.12, 'Note: 100% = perfect match with fluorescent, 0% = completely different. Similarity = 100 × (1 - |deviation|/baseline).',
           ha='center', transform=ax.transAxes, fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved feature overlap bars to {output_path}")
    plt.close()


def compute_per_object_spatial_overlap(
    ref_mask: np.ndarray,
    test_mask: np.ndarray,
) -> pd.DataFrame:
    """
    For each object in reference mask, compute spatial overlap with test mask.
    
    Args:
        ref_mask: Reference labeled mask (fluorescent ground truth)
        test_mask: Test labeled mask (BF/phase reconstruction)
    
    Returns:
        DataFrame with columns:
        - ref_label: Reference object ID
        - overlap_percent: % of ref object pixels that overlap with ANY test object
        - num_matches: Number of distinct test objects that overlap with this ref object
        - best_match_label: Label of test object with most overlap (0 if no matches)
        - best_match_iou: IoU with best matching test object
    """
    results = []
    
    # Get all reference object IDs
    ref_labels = np.unique(ref_mask)
    ref_labels = ref_labels[ref_labels > 0]  # Exclude background
    
    for ref_id in ref_labels:
        # Get pixels for this reference object
        ref_pixels = (ref_mask == ref_id)
        ref_size = ref_pixels.sum()
        
        if ref_size == 0:
            continue
        
        # Find which test objects overlap with this ref object
        test_labels_at_ref = test_mask[ref_pixels]
        overlapping_test_labels = np.unique(test_labels_at_ref)
        overlapping_test_labels = overlapping_test_labels[overlapping_test_labels > 0]  # Exclude background
        
        num_matches = len(overlapping_test_labels)
        
        # Compute total overlap (union of all matching test objects)
        overlap_pixels = test_mask[ref_pixels] > 0
        overlap_count = overlap_pixels.sum()
        overlap_percent = (overlap_count / ref_size) * 100.0
        
        # Find best matching test object (highest IoU)
        best_match_label = 0
        best_match_iou = 0.0
        
        if num_matches > 0:
            for test_id in overlapping_test_labels:
                test_pixels = (test_mask == test_id)
                intersection = (ref_pixels & test_pixels).sum()
                union = (ref_pixels | test_pixels).sum()
                iou = intersection / union if union > 0 else 0.0
                
                if iou > best_match_iou:
                    best_match_iou = iou
                    best_match_label = test_id
        
        results.append({
            'ref_label': ref_id,
            'overlap_percent': overlap_percent,
            'num_matches': num_matches,
            'best_match_label': best_match_label,
            'best_match_iou': best_match_iou * 100.0,  # Convert to percentage
        })
    
    return pd.DataFrame(results)


def plot_per_object_overlap_violins(
    ref_mask: np.ndarray,
    test_masks: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "Per-Object Spatial Overlap Analysis",
    filter_top_percent: Optional[float] = None,
):
    """
    Create violin plots showing per-object overlap statistics.
    
    For each fluorescent-labeled object, shows:
    - Top row: % overlap with detected objects in each image type
    - Bottom row: Number of objects matched in each image type
    
    Args:
        ref_mask: Reference labeled mask (fluorescent)
        test_masks: Dict of test labeled masks
        output_path: Where to save the plot
        title: Plot title
        filter_top_percent: If set, only analyze the top X% largest reference objects (e.g., 10.0 for top 10%)
    """
    # Filter reference mask to top X% largest objects if requested
    filtered_ref_mask = ref_mask.copy()
    if filter_top_percent is not None:
        from skimage.measure import regionprops
        props = regionprops(ref_mask)
        if len(props) > 0:
            areas = np.array([p.area for p in props])
            labels = np.array([p.label for p in props])
            
            # Find threshold for top X%
            area_threshold = np.percentile(areas, 100 - filter_top_percent)
            large_labels = labels[areas >= area_threshold]
            
            # Create filtered mask with only large objects
            filtered_ref_mask = np.zeros_like(ref_mask)
            for label in large_labels:
                filtered_ref_mask[ref_mask == label] = label
            
            print(f"[PerObjectOverlap] Filtering to top {filter_top_percent}% largest: {len(large_labels)}/{len(props)} objects (area >= {area_threshold:.1f} pixels)")
    
    image_types = ['raw_midslice', 'phase3d_midslice', 'phase2d']
    image_labels = {
        'raw_midslice': 'Raw BF',
        'phase3d_midslice': 'Phase3D',
        'phase2d': 'Phase2D',
    }
    
    # Compute overlap for all image types using filtered mask
    overlap_data = {}
    for img_type in image_types:
        if img_type in test_masks and test_masks[img_type] is not None:
            overlap_df = compute_per_object_spatial_overlap(filtered_ref_mask, test_masks[img_type])
            overlap_data[img_type] = overlap_df
    
    if not overlap_data:
        print("[PerObjectOverlap] No test masks available for comparison")
        return

    # Compute total area overlap for each image type
    total_area_overlap = {}
    ref_total_area = (filtered_ref_mask > 0).sum()
    for img_type in image_types:
        if img_type in test_masks and test_masks[img_type] is not None:
            test_mask = test_masks[img_type]
            # Total overlap: pixels that are non-zero in both masks
            overlap_pixels = ((filtered_ref_mask > 0) & (test_mask > 0)).sum()
            total_area_overlap[img_type] = (overlap_pixels / ref_total_area * 100.0) if ref_total_area > 0 else 0.0

    # Create 3x1 subplot (three separate plots)
    fig, axes = plt.subplots(3, 1, figsize=(16, 16))
    
    # Use viridis colormap
    viridis = plt.cm.viridis
    colors = [viridis(i / (len(image_types) - 1)) for i in range(len(image_types))]
    
    # ===== Top Plot: Overlap Percentage =====
    ax = axes[0]
    data_for_violin = []
    labels_for_violin = []
    positions = []
    stats_list = []
    
    for img_idx, img_type in enumerate(image_types):
        if img_type in overlap_data:
            values = overlap_data[img_type]['overlap_percent'].values
            if len(values) > 0:
                data_for_violin.append(values)
                labels_for_violin.append(image_labels[img_type])
                positions.append(img_idx)
                stats_list.append({
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'pos': img_idx
                })
    
    if data_for_violin:
        # Create violin plot
        parts = ax.violinplot(data_for_violin, positions=positions, widths=0.7, 
                             showmeans=True, showmedians=True)
        
        # Color violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[positions[i]])
            pc.set_alpha(0.3)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1.5)
        
        # Scatter points
        for i, (pos, values) in enumerate(zip(positions, data_for_violin)):
            if len(values) > 50:
                sample_indices = np.random.choice(len(values), 50, replace=False)
                values_to_plot = values[sample_indices]
            else:
                values_to_plot = values
            
            jitter = np.random.normal(0, 0.04, len(values_to_plot))
            ax.scatter(pos + jitter, values_to_plot, alpha=0.4, s=20, color='black', zorder=3)
        
        # Set y-axis with padding for annotations
        ax.set_ylim(-5, 115)
        
        # Add annotations (positioned inside plot area)
        for stats in stats_list:
            text = f'{stats["mean"]:.1f}%\n±{stats["std"]:.1f}'
            ax.text(stats['pos'], 105, text, ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', 
                            edgecolor='black', alpha=0.8, linewidth=1))
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_for_violin, fontsize=12)
        ax.set_ylabel('% of Fluorescent Object\nOverlapping with Detected Objects', 
                     fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_title('Spatial Overlap: % of Each Fluorescent Object Covered', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=2, label='100% overlap')
        ax.legend(loc='upper right')
    
    # ===== Bottom Plot: Number of Matches =====
    ax = axes[1]
    data_for_violin = []
    labels_for_violin = []
    positions = []
    stats_list = []
    
    for img_idx, img_type in enumerate(image_types):
        if img_type in overlap_data:
            values = overlap_data[img_type]['num_matches'].values
            if len(values) > 0:
                data_for_violin.append(values)
                labels_for_violin.append(image_labels[img_type])
                positions.append(img_idx)
                stats_list.append({
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'pos': img_idx
                })
    
    if data_for_violin:
        # Create violin plot
        parts = ax.violinplot(data_for_violin, positions=positions, widths=0.7,
                             showmeans=True, showmedians=True)
        
        # Color violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[positions[i]])
            pc.set_alpha(0.3)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1.5)
        
        # Scatter points
        for i, (pos, values) in enumerate(zip(positions, data_for_violin)):
            if len(values) > 50:
                sample_indices = np.random.choice(len(values), 50, replace=False)
                values_to_plot = values[sample_indices]
            else:
                values_to_plot = values
            
            jitter = np.random.normal(0, 0.04, len(values_to_plot))
            ax.scatter(pos + jitter, values_to_plot, alpha=0.4, s=20, color='black', zorder=3)
        
        # Set y-axis with padding for annotations
        all_values = np.concatenate(data_for_violin)
        y_max = max(3, np.percentile(all_values, 98))
        ax.set_ylim(-0.2, y_max + 1.2)
        
        # Add annotations (positioned inside plot area)
        for stats in stats_list:
            text = f'{stats["mean"]:.2f}\n±{stats["std"]:.2f}'
            ax.text(stats['pos'], y_max + 0.1, text, ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray',
                            edgecolor='black', alpha=0.8, linewidth=1))
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_for_violin, fontsize=12)
        ax.set_ylabel('Number of Detected Objects\nOverlapping with Each Fluorescent Object',
                     fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_title('Object Matching: How Many Detected Objects per Fluorescent Object',
                    fontsize=14, fontweight='bold', pad=15)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=2, label='1:1 matching')
        ax.legend(loc='upper right')

    # ===== Third Plot: Total Area Overlap =====
    ax = axes[2]
    if total_area_overlap:
        area_positions = []
        area_values = []
        area_labels = []

        for img_idx, img_type in enumerate(image_types):
            if img_type in total_area_overlap:
                area_positions.append(img_idx)
                area_values.append(total_area_overlap[img_type])
                area_labels.append(image_labels[img_type])

        if area_values:
            # Create bar plot
            bars = ax.bar(area_positions, area_values, width=0.7,
                         color=[colors[p] for p in area_positions],
                         alpha=0.7, edgecolor='black', linewidth=2)

            # Add value labels on bars
            for bar, val in zip(bars, area_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                       f'{val:.1f}%', ha='center', va='bottom',
                       fontweight='bold', fontsize=12)

            ax.set_xticks(area_positions)
            ax.set_xticklabels(area_labels, fontsize=12)
            ax.set_ylabel('% of Total Fluorescent Area\nCovered by Detected Organelles',
                         fontsize=13, fontweight='bold')
            ax.set_ylim(0, 110)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_title('Total Area Coverage: % of Fluorescent Pixels Detected',
                        fontsize=14, fontweight='bold', pad=15)
            ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=2, label='100% coverage')
            ax.legend(loc='upper right')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.997)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved per-object overlap violins to {output_path}")
    plt.close()


def plot_combined_canvas(
    images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    ref_mask: np.ndarray,
    test_masks: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "Organelle Comparison",
    fluorophore: str = "mCherry",
):
    """
    Create a single 4x4 canvas with all visualizations:
    Row 1: Raw images
    Row 2: Segmentation overlays
    Row 3: Labeled masks
    Row 4: Mask overlays

    Columns: Fluorescent, Raw BF, Phase3D, Phase2D
    """
    from skimage.segmentation import find_boundaries

    fig = plt.figure(figsize=(24, 24))
    gs = GridSpec(4, 4, figure=fig, hspace=0.02, wspace=0.02)

    order = ['fluorescent', 'raw_midslice', 'phase3d_midslice', 'phase2d']
    col_titles = {
        'fluorescent': f'Fluorescent ({fluorophore})\nGround Truth',
        'raw_midslice': 'Raw BF\n(mid-slice)',
        'phase3d_midslice': 'Phase3D Recon\n(mid-slice)',
        'phase2d': 'Phase2D Recon\n(autofocus)',
    }

    # Row 1: Raw images
    for col_idx, key in enumerate(order):
        if key not in images or images[key] is None:
            continue

        ax = fig.add_subplot(gs[0, col_idx])
        img = images[key]

        if key == 'fluorescent':
            ax.imshow(img, cmap='inferno', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        else:
            ax.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))

        # Only add title to top row
        ax.set_title(col_titles[key], fontsize=14, fontweight='bold', pad=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray',
                             edgecolor='black', alpha=0.8, linewidth=1.5))
        ax.axis('off')

    # Row 2: Segmentation overlays
    for col_idx, key in enumerate(order):
        if key not in images or images[key] is None:
            continue

        ax = fig.add_subplot(gs[1, col_idx])
        img = images[key]

        if key == 'fluorescent':
            ax.imshow(img, cmap='inferno', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        else:
            ax.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))

        # Overlay segmentation contours
        if key in masks and masks[key] is not None:
            boundaries = find_boundaries(masks[key], mode='outer')
            overlay = np.zeros((*boundaries.shape, 4))
            overlay[boundaries, :] = [0, 1, 1, 1]  # Cyan
            ax.imshow(overlay)

            n_objects = masks[key].max()
            ax.text(10, 30, f'N={n_objects}', color='cyan', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        ax.axis('off')

    # Row 3: Labeled masks
    for col_idx, key in enumerate(order):
        if key not in masks or masks[key] is None:
            continue

        ax = fig.add_subplot(gs[2, col_idx])
        mask = masks[key]
        n_objects = int(mask.max())

        np.random.seed(42)
        colored_mask = np.zeros((*mask.shape, 3))
        for obj_id in range(1, n_objects + 1):
            color = np.random.rand(3)
            colored_mask[mask == obj_id] = color

        ax.imshow(colored_mask)
        ax.text(10, 30, f'N={n_objects}', color='white', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

    # Row 4: Mask overlays (fluorescent vs each modality)
    overlay_types = ['raw_midslice', 'phase3d_midslice', 'phase2d', 'fluorescent']
    for col_idx, img_type in enumerate(overlay_types):
        ax = fig.add_subplot(gs[3, col_idx])

        if img_type == 'fluorescent':
            # Fluorescent reference with Phase2D overlay
            fluor_img = images.get('fluorescent')
            if fluor_img is not None and 'phase2d' in test_masks and test_masks['phase2d'] is not None:
                img_min, img_max = np.percentile(fluor_img, [1, 99])
                fluor_display = np.clip((fluor_img - img_min) / (img_max - img_min), 0, 1)
                ax.imshow(fluor_display, cmap='gray', interpolation='nearest')

                ref_boundaries = find_boundaries(ref_mask, mode='thick')
                phase2d_boundaries = find_boundaries(test_masks['phase2d'], mode='thick')

                overlay = np.zeros((*ref_mask.shape, 4))
                overlay[ref_boundaries, :] = [1, 0, 0, 0.8]  # Red
                overlay[phase2d_boundaries, :] = [0, 1, 0, 0.8]  # Green
                overlap = ref_boundaries & phase2d_boundaries
                overlay[overlap, :] = [1, 1, 0, 1.0]  # Yellow

                ax.imshow(overlay, interpolation='nearest')
            ax.axis('off')
        else:
            # Other modalities with fluorescent overlay
            if img_type not in test_masks or test_masks[img_type] is None:
                continue

            base_img = images.get(img_type)
            if base_img is not None:
                img_min, img_max = np.percentile(base_img, [1, 99])
                base_display = np.clip((base_img - img_min) / (img_max - img_min), 0, 1)
                ax.imshow(base_display, cmap='gray', interpolation='nearest')
            else:
                ax.imshow(np.zeros_like(ref_mask), cmap='gray')

            ref_boundaries = find_boundaries(ref_mask, mode='thick')
            test_boundaries = find_boundaries(test_masks[img_type], mode='thick')

            overlay = np.zeros((*ref_mask.shape, 4))
            overlay[ref_boundaries, :] = [1, 0, 0, 0.8]  # Red
            overlay[test_boundaries, :] = [0, 1, 0, 0.8]  # Green
            overlap = ref_boundaries & test_boundaries
            overlay[overlap, :] = [1, 1, 0, 1.0]  # Yellow

            ax.imshow(overlay, interpolation='nearest')
            ax.axis('off')

    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved combined canvas to {output_path}")
    plt.close()


def plot_mask_overlays(
    ref_mask: np.ndarray,
    test_masks: Dict[str, np.ndarray],
    images: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "Mask Overlay Comparison",
):
    """
    Plot fluorescent mask overlaid with each test mask for visual alignment verification.
    
    Shows fluorescent objects (red contours) overlaid with detected objects from each
    image type (green contours) to verify spatial alignment. Also includes fluorescent
    background image.
    """
    from skimage.segmentation import find_boundaries

    image_types = ['raw_midslice', 'phase3d_midslice', 'phase2d', 'fluorescent']
    image_labels = {
        'raw_midslice': 'Raw BF (mid-slice)',
        'phase3d_midslice': 'Phase3D (mid-slice)',
        'phase2d': 'Phase2D (autofocus)',
        'fluorescent': 'Fluorescent (reference)',
    }

    # Create 1x4 grid
    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, 4, figure=fig, hspace=0.3, wspace=0.08)
    
    for idx, img_type in enumerate(image_types):
        # For fluorescent, show Phase2D reconstruction overlay on fluorescent background
        if img_type == 'fluorescent':
            ax = fig.add_subplot(gs[0, idx])
            
            # Get fluorescent image for display
            fluor_img = images.get('fluorescent')
            if fluor_img is not None and 'phase2d' in test_masks and test_masks['phase2d'] is not None:
                # Normalize fluorescent to 0-1 for display
                img_min, img_max = np.percentile(fluor_img, [1, 99])
                fluor_display = np.clip((fluor_img - img_min) / (img_max - img_min), 0, 1)
                ax.imshow(fluor_display, cmap='gray', interpolation='nearest')
                
                # Find boundaries for fluorescent mask (red)
                ref_boundaries = find_boundaries(ref_mask, mode='thick')
                
                # Find boundaries for Phase2D mask (green)
                phase2d_boundaries = find_boundaries(test_masks['phase2d'], mode='thick')
                
                # Create RGB overlay
                overlay = np.zeros((*ref_mask.shape, 4))
                overlay[ref_boundaries, :] = [1, 0, 0, 0.8]  # Red for fluorescent
                overlay[phase2d_boundaries, :] = [0, 1, 0, 0.8]  # Green for Phase2D
                
                # Where they overlap, make it yellow
                overlap = ref_boundaries & phase2d_boundaries
                overlay[overlap, :] = [1, 1, 0, 1.0]  # Yellow for overlap
                
                ax.imshow(overlay, interpolation='nearest')
                
                # Count objects
                ref_count = len(np.unique(ref_mask)) - 1
                phase2d_count = len(np.unique(test_masks['phase2d'])) - 1
                title_text = f"{image_labels[img_type]}\nRed: Fluorescent ({ref_count} obj) | Green: Phase2D ({phase2d_count} obj) | Yellow: Overlap"
            else:
                ax.imshow(np.zeros_like(ref_mask), cmap='gray')
                title_text = f"{image_labels[img_type]}\n(No image)"
            
            ax.text(0.5, 1.02, title_text, transform=ax.transAxes, 
                   fontsize=12, fontweight='bold', ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                            edgecolor='black', alpha=0.9, linewidth=1.5))
            ax.axis('off')
            continue
        
        # For other images, show overlay comparison
        if img_type not in test_masks or test_masks[img_type] is None:
            continue

        ax = fig.add_subplot(gs[0, idx])
        
        # Get base image for display
        base_img = images.get(img_type)
        if base_img is not None:
            # Normalize to 0-1 for display
            img_min, img_max = np.percentile(base_img, [1, 99])
            base_display = np.clip((base_img - img_min) / (img_max - img_min), 0, 1)
            ax.imshow(base_display, cmap='gray', interpolation='nearest')
        else:
            # Black background if no image
            ax.imshow(np.zeros_like(ref_mask), cmap='gray')
        
        # Find boundaries for fluorescent mask (red)
        ref_boundaries = find_boundaries(ref_mask, mode='thick')
        
        # Find boundaries for test mask (green)
        test_boundaries = find_boundaries(test_masks[img_type], mode='thick')
        
        # Create RGB overlay
        overlay = np.zeros((*ref_mask.shape, 4))
        overlay[ref_boundaries, :] = [1, 0, 0, 0.8]  # Red for fluorescent
        overlay[test_boundaries, :] = [0, 1, 0, 0.8]  # Green for detected
        
        # Where they overlap, make it yellow
        overlap = ref_boundaries & test_boundaries
        overlay[overlap, :] = [1, 1, 0, 1.0]  # Yellow for overlap
        
        ax.imshow(overlay, interpolation='nearest')
        
        # Count objects
        ref_count = len(np.unique(ref_mask)) - 1
        test_count = len(np.unique(test_masks[img_type])) - 1
        
        # Add title with annotation
        title_text = f"{image_labels[img_type]}\nRed: Fluorescent ({ref_count} obj) | Green: Detected ({test_count} obj) | Yellow: Overlap"
        ax.text(0.5, 1.02, title_text, transform=ax.transAxes, 
               fontsize=12, fontweight='bold', ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                        edgecolor='black', alpha=0.9, linewidth=1.5))
        
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved mask overlays to {output_path}")
    plt.close()


def plot_cytoplasm_mask_reference(
    cytoplasm_mask: np.ndarray,
    output_path: Path,
    title: str = "Cytoplasm Mask Reference",
):
    """
    Plot nuclear and cell segmentation masks side by side for reference.
    
    Shows:
    - Left: Cytoplasm mask (largest cell component)
    - Right: Excluded regions (nucleus + background)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Cytoplasm mask
    ax = axes[0]
    im = ax.imshow(cytoplasm_mask, cmap='Greens', interpolation='nearest')
    ax.set_title('Cytoplasm Mask\n(Cell -2px, Nucleus -20px)\nWhite = Analyzed', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Cytoplasm')
    
    # Add text showing coverage
    coverage = 100 * cytoplasm_mask.sum() / cytoplasm_mask.size
    ax.text(0.02, 0.98, f'{coverage:.1f}% of crop', 
            transform=ax.transAxes, fontsize=11, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Right: Inverted (excluded regions)
    ax = axes[1]
    nucleus_region = 1 - cytoplasm_mask  # Everything that's NOT cytoplasm
    im = ax.imshow(nucleus_region, cmap='Reds', interpolation='nearest')
    ax.set_title('Excluded Regions\n(Membrane + Nucleus + Background)\nWhite = Excluded', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Excluded')
    
    # Add text showing excluded percentage
    excluded = 100 * nucleus_region.sum() / nucleus_region.size
    ax.text(0.02, 0.98, f'{excluded:.1f}% of crop', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f"{title}\n(Largest cell: 2px cell erosion + 20px nuclear buffer)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved cytoplasm mask reference to {output_path}")
    plt.close()


def plot_feature_distributions(
    features: Dict[str, pd.DataFrame],
    output_path: Path,
    title: str = "Organelle Feature Distributions",
):
    """
    Create violin plots showing per-organelle feature distributions across image types.
    
    Each subplot shows one morphological feature, with violin plots representing
    the distribution of that feature across all detected organelles in each image type.
    Individual dots show each organelle's value.
    """
    image_types = ['fluorescent', 'raw_midslice', 'phase3d_midslice', 'phase2d']
    image_labels = {
        'fluorescent': 'Fluor',
        'raw_midslice': 'Raw BF',
        'phase3d_midslice': 'Phase3D',
        'phase2d': 'Phase2D',
    }
    
    # Select comprehensive set of features to plot (3 rows x 4 columns = 12 features)
    feature_names = [
        'area', 'perimeter', 'axis_major_length', 'axis_minor_length',
        'eccentricity', 'solidity', 'extent', 'convex_area',
        'equivalent_diameter', 'aspect_ratio', 'circularity', 'orientation'
    ]
    feature_display = {
        'area': 'Area (pixels²)',
        'perimeter': 'Perimeter (pixels)',
        'axis_major_length': 'Major Axis (pixels)',
        'axis_minor_length': 'Minor Axis (pixels)',
        'eccentricity': 'Eccentricity',
        'solidity': 'Solidity',
        'extent': 'Extent',
        'convex_area': 'Convex Area (pixels²)',
        'equivalent_diameter': 'Equiv. Diameter (pixels)',
        'aspect_ratio': 'Aspect Ratio',
        'circularity': 'Circularity',
        'orientation': 'Orientation (radians)',
    }
    
    # Create subplot grid (3 rows x 4 columns)
    fig, axes = plt.subplots(3, 4, figsize=(24, 16))
    axes = axes.flatten()
    
    # Use viridis colormap
    viridis = plt.cm.viridis
    colors = [viridis(i / (len(image_types) - 1)) for i in range(len(image_types))]
    
    for feat_idx, feat in enumerate(feature_names):
        if feat_idx >= len(axes):
            break  # Safety check
        
        ax = axes[feat_idx]
        
        # Collect data for this feature across all image types
        data_for_violin = []
        labels_for_violin = []
        positions = []
        stats_list = []  # Store mean and std for annotations
        
        for img_idx, img_type in enumerate(image_types):
            if img_type in features and not features[img_type].empty:
                if feat in features[img_type].columns:
                    values = features[img_type][feat].values
                    # Filter out any NaN or infinite values
                    values = values[np.isfinite(values)]
                    if len(values) > 0:
                        data_for_violin.append(values)
                        labels_for_violin.append(image_labels[img_type])
                        positions.append(img_idx)
                        # Calculate statistics
                        stats_list.append({
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'pos': img_idx
                        })
        
        if data_for_violin:
            # Create violin plot
            parts = ax.violinplot(
                data_for_violin,
                positions=positions,
                widths=0.7,
                showmeans=True,
                showmedians=True,
            )
            
            # Color the violins with lower alpha
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[positions[i]])
                pc.set_alpha(0.3)  # Lower alpha for background
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
            
            # Style the lines
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                if partname in parts:
                    vp = parts[partname]
                    vp.set_edgecolor('black')
                    vp.set_linewidth(1.5)
            
            # Overlay scatter points (sample if too many)
            for i, (pos, values) in enumerate(zip(positions, data_for_violin)):
                # Sample if more than 50 points per violin
                if len(values) > 50:
                    sample_indices = np.random.choice(len(values), 50, replace=False)
                    values_to_plot = values[sample_indices]
                else:
                    values_to_plot = values
                
                # Add jitter for visibility
                jitter = np.random.normal(0, 0.04, len(values_to_plot))
                ax.scatter(
                    pos + jitter,
                    values_to_plot,
                    alpha=0.4,
                    s=20,
                    color='black',
                    zorder=3
                )
            
            # Set y-axis limits to exclude extreme outliers (use 2nd to 98th percentile)
            all_values = np.concatenate(data_for_violin)
            if len(all_values) > 0:
                y_min = np.percentile(all_values, 2)
                y_max = np.percentile(all_values, 98)
                # Add padding for annotations at top
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.20 * y_range)  # More padding at top for annotations
            
            # Add mean ± std annotations above each violin with gray background
            for stats in stats_list:
                mean_val = stats['mean']
                std_val = stats['std']
                pos = stats['pos']
                
                # Position text above the violin
                y_pos = y_max + 0.08 * y_range
                
                # Format text based on magnitude
                if abs(mean_val) < 10:
                    text = f'{mean_val:.2f}\n±{std_val:.2f}'
                else:
                    text = f'{mean_val:.1f}\n±{std_val:.1f}'
                
                ax.text(pos, y_pos, text,
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', 
                                facecolor='lightgray', 
                                edgecolor='black',
                                alpha=0.8,
                                linewidth=1))
            
            # Styling
            ax.set_xticks(positions)
            ax.set_xticklabels([labels_for_violin[i] for i in range(len(positions))], rotation=45, ha='right')
            ax.set_ylabel(feature_display[feat], fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_title(feature_display[feat], fontsize=12, fontweight='bold', pad=10)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title(feature_display.get(feat, feat), fontsize=12, fontweight='bold', pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide any unused subplots
    for idx in range(len(feature_names), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved feature distributions to {output_path}")
    plt.close()


def plot_network_features(
    network_results: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = "Network/Connectivity Features",
):
    """
    Create 3-panel bar plot showing key network metrics as % of fluorescent.

    Shows 3 key metrics normalized to fluorescent baseline:
    - Network length density (total skeleton length / area)
    - Branching density (number of branches / area)
    - Average degree (connectivity of junctions)
    """
    test_image_types = ['raw_midslice', 'phase3d_midslice', 'phase2d']
    image_labels = {
        'raw_midslice': 'Raw BF',
        'phase3d_midslice': 'Phase3D',
        'phase2d': 'Phase2D',
    }

    # Define 3 key network features
    network_feature_names = [
        'network_length_density',
        'branching_density',
        'average_degree',
    ]
    feature_labels = {
        'network_length_density': 'Length Density (μm/μm²)',
        'branching_density': 'Branch Density (branches/μm²)',
        'average_degree': 'Avg Degree (connectivity)',
    }

    # Get fluorescent reference values
    fluor_values = {}
    if 'fluorescent' in network_results:
        for feat in network_feature_names:
            fluor_values[feat] = network_results['fluorescent'].get(feat, 0.0)

    # Create 3 subplots (one per metric)
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    # Use viridis colormap
    viridis = plt.cm.viridis
    colors = [viridis(i / (len(test_image_types) - 1)) for i in range(len(test_image_types))]

    for feat_idx, feat in enumerate(network_feature_names):
        ax = axes[feat_idx]

        # Get values for this feature
        values = []
        positions = []
        labels = []

        for img_idx, img_type in enumerate(test_image_types):
            if img_type in network_results and network_results[img_type]:
                raw_value = network_results[img_type].get(feat, 0.0)
                ref_value = fluor_values.get(feat, 1.0)

                # Normalize to % of fluorescent
                if ref_value > 0:
                    percent_value = (raw_value / ref_value) * 100
                else:
                    percent_value = 0.0

                values.append(percent_value)
                positions.append(img_idx)
                labels.append(image_labels[img_type])

        if values:
            # Create bar plot
            bars = ax.bar(positions, values, width=0.7,
                         color=[colors[p] for p in positions],
                         alpha=0.7, edgecolor='black', linewidth=2)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 3,
                       f'{val:.1f}%', ha='center', va='bottom',
                       fontweight='bold', fontsize=12)

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=12)
            ax.set_ylabel('% of Fluorescent Ground Truth',
                         fontsize=13, fontweight='bold')
            ax.set_ylim(0, max(110, max(values) + 15))
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_title(f'{feature_labels[feat]}',
                        fontsize=14, fontweight='bold', pad=15)
            ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=2,
                      label='100% = Fluorescent baseline')
            ax.legend(loc='upper right')

            # Add fluorescent reference value as text
            if feat in fluor_values:
                ax.text(0.02, 0.98, f'Fluorescent: {fluor_values[feat]:.3f}',
                       transform=ax.transAxes, fontsize=11, fontweight='bold',
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray',
                                edgecolor='black', alpha=0.8))

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.997)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved network feature comparison to {output_path}")
    plt.close()


@click.command()
@click.argument("experiment")
@click.argument("tile")
@click.option("--crop-size", type=int, default=512, help="Size of crop (default: 512)")
@click.option(
    "--crop-center",
    type=(int, int),
    default=None,
    help="Crop center as (Y, X) pixel coordinates. If not specified, uses image center.",
)
@click.option(
    "--fluorescent-channel",
    type=str,
    default="mCherry",
    help="Fluorescent channel name (default: mCherry)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory (default: ./organelle_comparison_results/EXPERIMENT_TILE/)",
)
@click.option("--use-gpu", is_flag=True, help="Use GPU for Frangi filter")
@click.option("--mask-nuclei", is_flag=True, help="Mask out nuclei using nuclear_seg from pheno_assembled (cytoplasm only)")
@click.option("--register-to-fluor", is_flag=True, default=True, help="Use pre-registered fluorescent from lc_20x_fluor_2d_registered store (default: True)")
@click.option("--fluor-shift-y", type=float, default=None, help="Fine alignment: Y shift in pixels for fluorescent (if not provided, will use cached or sweep)")
@click.option("--fluor-shift-x", type=float, default=None, help="Fine alignment: X shift in pixels for fluorescent (if not provided, will use cached or sweep)")
@click.option("--sweep-offset/--no-sweep-offset", default=True, help="Sweep to find optimal offset if not cached (default: True)")
@click.option("--sweep-range", type=int, default=5, help="Sweep range in pixels (+/- value, default: 5)")
@click.option("--sweep-step", type=int, default=1, help="Sweep step size in pixels (default: 1)")
@click.option("--verbose", is_flag=True, help="Verbose output")
def main_cli(experiment, tile, crop_size, crop_center, fluorescent_channel, output_dir, use_gpu, mask_nuclei, register_to_fluor, fluor_shift_y, fluor_shift_x, sweep_offset, sweep_range, sweep_step, verbose):
    """
    Compare organelle features between fluorescent label and brightfield reconstructions.

    By default, uses the pre-registered fluorescent image from lc_20x_fluor_2d_registered store
    (created by register_fluor_2d_tiles pipeline step). This ensures proper spatial alignment
    for accurate feature comparison.

    FLUORESCENT OFFSET OPTIMIZATION:
    - If --fluor-shift-y/x are not provided, the script will automatically find the optimal offset
    - First checks for cached offset in YAML file (saved from previous runs)
    - If no cache exists, performs a pixel sweep to maximize segmentation overlap (default: +/-5 pixels)
    - The optimal offset is saved to cache for future runs
    - Use --fluor-shift-y/x to manually override the automatic optimization

    EXPERIMENT: experiment name (e.g., ops0033_20250429)
    TILE: tile position (e.g., A/1/029020)

    Examples:
        # Basic usage (automatic offset optimization with caching)
        python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/029020

        # With nuclear masking (cytoplasm only)
        python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/029020 --mask-nuclei

        # Custom crop center and nuclear masking
        python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/029020 --crop-center 1024 1024 --mask-nuclei

        # Manual fine alignment shift (overrides automatic optimization)
        python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/029020 --fluor-shift-y 5 --fluor-shift-x 3

        # Disable automatic sweep (use default or cached offset only)
        python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/029020 --no-sweep-offset

        # Custom sweep parameters
        python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/029020 --sweep-range 10 --sweep-step 2

        # Disable registration (use raw, unaligned images)
        python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/029020 --no-register-to-fluor
    """
    # Initialize dataset
    dataset = OpsDataset(experiment)

    # Special case: ops0065_20250812 uses GFP channel instead of mCherry
    if experiment == "ops0065_20250812" and fluorescent_channel == "mCherry":
        fluorescent_channel = "GFP"
        print(f"Note: Using GFP channel for experiment {experiment}")

    # Create output directory
    if output_dir:
        output_dir = Path(output_dir)
    else:
        # Default: save in current working directory under results/
        output_dir = Path.cwd() / "organelle_comparison_results" / f"{experiment}_{tile.replace('/', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine fluorophore based on experiment name
    fluorophore = "GFP" if "ops0065" in experiment else "mCherry"

    print(f"\n{'='*80}")
    print(f"Organelle Feature Comparison")
    print(f"{'='*80}")
    print(f"Experiment: {experiment}")
    print(f"Tile: {tile}")
    print(f"Crop size: {crop_size}x{crop_size}")
    if crop_center:
        print(f"Crop center: Y={crop_center[0]}, X={crop_center[1]} (custom)")
    else:
        print(f"Crop center: Image center (default)")
    print(f"Fluorescent channel: {fluorescent_channel}")
    print(f"Output: {output_dir}")
    print(f"Nuclear masking: {'Enabled (cytoplasm only)' if mask_nuclei else 'Disabled'}")
    print(f"Fluorescent registration: {'Enabled (using pre-registered store)' if register_to_fluor else 'Disabled (using raw)'}")
    print(f"{'='*80}\n")

    # Determine fluorescent offset (use cached, sweep, or manual)
    final_fluor_shift_y = fluor_shift_y
    final_fluor_shift_x = fluor_shift_x

    if register_to_fluor and (fluor_shift_y is None or fluor_shift_x is None):
        # Try to load cached offset
        cache_path = get_offset_cache_path(experiment, tile, output_dir)
        cached_offset = load_cached_offset(cache_path)

        if cached_offset is not None:
            # Use cached offset
            cached_y, cached_x = cached_offset
            if fluor_shift_y is None:
                final_fluor_shift_y = cached_y
            if fluor_shift_x is None:
                final_fluor_shift_x = cached_x
            print(f"Using cached fluorescent offset: Y={final_fluor_shift_y}, X={final_fluor_shift_x}")
        elif sweep_offset:
            # Perform offset sweep to find optimal offset
            print(f"\n{'='*80}")
            print("Performing Offset Sweep")
            print(f"{'='*80}")

            # Load images without offset first
            images_no_offset = load_crop_from_stores(
                dataset=dataset,
                tile=tile,
                crop_size=crop_size,
                crop_center=crop_center,
                fluorescent_channel=fluorescent_channel,
                apply_fluor_registration=register_to_fluor,
                fluor_shift=(0.0, 0.0),
            )

            # Check if we have both fluorescent and phase2d for sweep
            if images_no_offset['fluorescent'] is not None and images_no_offset['phase2d'] is not None:
                optimal_y, optimal_x, overlap = sweep_offset_for_optimal_overlap(
                    fluor_img=images_no_offset['fluorescent'],
                    recon_img=images_no_offset['phase2d'],
                    pixel_size=0.325,
                    sweep_range=sweep_range,
                    step_size=sweep_step,
                    use_gpu=use_gpu,
                    verbose=verbose,
                )

                # Save to cache
                save_offset_to_cache(cache_path, optimal_y, optimal_x, overlap)

                # Use the optimal offset
                if fluor_shift_y is None:
                    final_fluor_shift_y = optimal_y
                if fluor_shift_x is None:
                    final_fluor_shift_x = optimal_x
            else:
                print("[WARNING] Cannot perform offset sweep - missing fluorescent or phase2d image")
                final_fluor_shift_y = 0.0 if fluor_shift_y is None else fluor_shift_y
                final_fluor_shift_x = 0.0 if fluor_shift_x is None else fluor_shift_x
        else:
            # No cached offset and sweep disabled - use defaults
            final_fluor_shift_y = 0.0 if fluor_shift_y is None else fluor_shift_y
            final_fluor_shift_x = 0.0 if fluor_shift_x is None else fluor_shift_x
            print(f"Using default offset: Y={final_fluor_shift_y}, X={final_fluor_shift_x}")
    elif not register_to_fluor:
        # No registration, so no offset
        final_fluor_shift_y = 0.0
        final_fluor_shift_x = 0.0
    else:
        # Manual offset provided
        print(f"Using manual offset: Y={final_fluor_shift_y}, X={final_fluor_shift_x}")

    # Load crops from all stores with final offset
    print(f"\n{'='*80}")
    print("Loading Images")
    print(f"{'='*80}")
    if register_to_fluor and (final_fluor_shift_y != 0 or final_fluor_shift_x != 0):
        print(f"Fluorescent offset: Y={final_fluor_shift_y} pixels, X={final_fluor_shift_x} pixels")

    images = load_crop_from_stores(
        dataset=dataset,
        tile=tile,
        crop_size=crop_size,
        crop_center=crop_center,
        fluorescent_channel=fluorescent_channel,
        apply_fluor_registration=register_to_fluor,
        fluor_shift=(final_fluor_shift_y, final_fluor_shift_x),
    )
    
    # Load cytoplasm mask if requested
    cytoplasm_mask = None
    if mask_nuclei:
        print(f"\n{'='*80}")
        print("Loading Cytoplasm Mask")
        print(f"{'='*80}")
        cytoplasm_mask = load_cytoplasm_mask_from_stitched(
            dataset=dataset,
            tile=tile,
            crop_size=crop_size,
            crop_offset=images['crop_offset'],
            verbose=verbose,
        )
        if cytoplasm_mask is None:
            print("[WARNING] Failed to load cytoplasm mask. Proceeding without nuclear masking.")
        else:
            # Save reference plot showing the cytoplasm mask
            mask_ref_path = output_dir / f"cytoplasm_mask_reference_{tile.replace('/', '_')}.png"
            plot_cytoplasm_mask_reference(cytoplasm_mask, mask_ref_path,
                                         title=f"Cytoplasm Mask: {experiment} - {tile}")
        print()
    
    # Segment organelles in each image type
    print(f"\n{'='*80}")
    print("Segmenting Organelles")
    print(f"{'='*80}")
    
    masks = {}
    features = {}
    vesselness_maps = {}
    
    for key in ['fluorescent', 'raw_midslice', 'phase3d_midslice', 'phase2d']:
        if images[key] is None:
            print(f"\n[{key}] Skipping (not available)")
            continue
        
        # Adjust CLAHE and Frangi parameters based on image type
        # Phase images benefit from stronger CLAHE (matching organelle_segmentation.py)
        # Mitochondria are tubular like ER, so use higher alpha for better tube detection
        # Vesicular organelles (like in ops0065_20250812) need lower alpha for round structures

        # Special parameters for vesicular organelles (round, not tubular)
        is_vesicular = (experiment == "ops0065_20250812")

        if key in ['raw_midslice','phase3d_midslice', 'phase2d']:
            use_clahe = True
            clahe_clip = 0.01  # Lower clip for phase images (more contrast)
            clahe_kernel = (64, 64)  # Smaller kernel for phase images
            smoothing = 1.0
            # Adjust alpha and min radius based on organelle type
            if is_vesicular:
                # Lower alpha for vesicular/blob-like structures (more isotropic)
                # Following Nellie's approach for blob-like organelles
                frangi_alpha = 0.5
                frangi_min_radius = 0.2  # Match diffraction limit, filter out sub-diffraction noise
                frangi_threshold_mult = 0.1
            else:
                # Higher alpha for tubular structure detection (like ER/mitochondria)
                frangi_alpha = 4.0
                frangi_min_radius = 0.1
                # Higher threshold for phase to reduce false positives from CLAHE artifacts
                frangi_threshold_mult = 0.001
        else:  # fluorescent
            use_clahe = False  # No CLAHE for fluorescent labels
            clahe_clip = 0.01
            clahe_kernel = (256, 256)  # Larger kernel (not used since use_clahe=False)
            smoothing = 0.0
            # Adjust alpha and min radius based on organelle type
            if is_vesicular:
                # Lower alpha for vesicular/blob-like structures
                # Following Nellie's approach for blob-like organelles
                frangi_alpha = 0.5
                frangi_min_radius = 0.2  # Match diffraction limit, filter out sub-diffraction noise
                frangi_threshold_mult = 0.1
            else:
                # Higher alpha for tubular mitochondria detection
                frangi_alpha = 4.0
                frangi_min_radius = 0.1
                frangi_threshold_mult = 0.001  # Keep standard threshold for clean fluorescent data
        
        print(f"\n[{key}] Segmenting...")
        if verbose and use_clahe:
            print(f"  Using CLAHE preprocessing (clip={clahe_clip}, kernel={clahe_kernel}, smooth={smoothing})")
        if verbose:
            print(f"  Using Frangi filter (alpha={frangi_alpha}, min_radius={frangi_min_radius}μm, threshold_mult={frangi_threshold_mult})")
        if verbose and cytoplasm_mask is not None:
            print(f"  Using cytoplasm mask (excluding nuclei)")
        
        vesselness, binary, labeled = segment_organelles_frangi(
            images[key],
            pixel_size=0.325,
            min_radius_um=frangi_min_radius,
            max_radius_um=1.5,
            alpha=frangi_alpha,
            beta=0.5,
            threshold_multiplier=frangi_threshold_mult,
            use_gpu=use_gpu,
            use_clahe=use_clahe,
            clahe_clip_limit=clahe_clip,
            clahe_kernel_size=clahe_kernel,
            post_clahe_smoothing_sigma=smoothing,
            cytoplasm_mask=cytoplasm_mask,
            verbose=verbose,
        )
        
        masks[key] = labeled
        vesselness_maps[key] = vesselness
        
        # Extract comprehensive features (including intensity and Frangi features)
        features[key] = extract_organelle_features(
            labeled,
            spacing=(0.325, 0.325),
            intensity_image=images[key],
            frangi_image=vesselness,
            full_features=True,
        )
        
        print(f"[{key}] Found {labeled.max()} organelles")
        if verbose and not features[key].empty:
            print(f"  Extracted {len(features[key].columns)} feature types per organelle")
    
    # Extract network/connectivity features
    print(f"\n{'='*80}")
    print("Extracting Network Features")
    print(f"{'='*80}")
    
    network_features = {}
    for key in ['fluorescent', 'raw_midslice', 'phase3d_midslice', 'phase2d']:
        if key not in masks or masks[key] is None:
            continue
        
        print(f"\n[{key}] Computing network features...")
        binary_mask = masks[key] > 0
        
        # Use Frangi vesselness for robust skeleton extraction
        intensity_for_skeleton = vesselness_maps.get(key)
        
        network_result = calculate_network_features(
            binary_mask,
            spacing=(0.325, 0.325),
            intensity_image=intensity_for_skeleton,
            full_features=False,  # Skip expensive features for speed
        )
        
        if network_result:
            branch_df, network_summary = network_result
            network_features[key] = network_summary
            
            if verbose and network_summary:
                print(f"  Network: {network_summary.get('num_branches', 0)} branches, "
                      f"{network_summary.get('num_endpoints', 0)} endpoints, "
                      f"{network_summary.get('num_nodes', 0)} junctions")
        else:
            network_features[key] = {}
            print(f"  No network detected")
    
    # Compute feature overlaps (using fluorescent as reference)
    print(f"\n{'='*80}")
    print("Computing Feature Overlaps")
    print(f"{'='*80}")
    
    overlap_results = {}
    
    if 'fluorescent' in features:
        ref_features = features['fluorescent']
        
        # Fluorescent vs itself = 100%
        overlap_results['fluorescent'] = {
            'area': 100.0,
            'axis_major_length': 100.0,
            'axis_minor_length': 100.0,
            'eccentricity': 100.0,
            'solidity': 100.0,
            'extent': 100.0,
            'n_matched': len(ref_features),
            'n_ref': len(ref_features),
            'n_test': len(ref_features),
        }
        
        # Compare other modalities to fluorescent
        for key in ['raw_midslice', 'phase3d_midslice', 'phase2d']:
            if key in features:
                print(f"\n[{key}] Computing overlap with fluorescent...")
                overlap = compute_feature_overlap(ref_features, features[key])
                overlap_results[key] = overlap
                
                print(f"  Matched: {overlap['n_matched']}/{overlap['n_ref']} organelles")
                print(f"  Area overlap: {overlap['area']:.1f}%")
                print(f"  Major axis overlap: {overlap['axis_major_length']:.1f}%")
    
    # Generate visualizations
    print(f"\n{'='*80}")
    print("Generating Visualizations")
    print(f"{'='*80}")

    # Combined canvas with all visualizations
    if 'fluorescent' in masks and masks['fluorescent'] is not None:
        combined_path = output_dir / f"combined_canvas_{tile.replace('/', '_')}.png"
        test_masks_for_overlay = {
            'raw_midslice': masks.get('raw_midslice'),
            'phase3d_midslice': masks.get('phase3d_midslice'),
            'phase2d': masks.get('phase2d'),
        }
        plot_combined_canvas(
            images=images,
            masks=masks,
            ref_mask=masks['fluorescent'],
            test_masks=test_masks_for_overlay,
            output_path=combined_path,
            title=f"{experiment} - {tile}",
            fluorophore=fluorophore
        )
    
    # 4. Feature overlap bars
    if overlap_results:
        # Save numerical results to CSV
        csv_path = output_dir / f"feature_overlap_{tile.replace('/', '_')}.csv"
        overlap_df = pd.DataFrame(overlap_results).T
        overlap_df.to_csv(csv_path)
        print(f"[Results] Saved overlap data to {csv_path}")
    
    # 5. Feature distributions (violin plots)
    if features:
        distributions_path = output_dir / f"feature_distributions_{tile.replace('/', '_')}.png"
        plot_feature_distributions(features, distributions_path,
                                   title=f"Organelle Feature Distributions: {experiment} - {tile}")
    
    # 6. Network features comparison
    if network_features:
        network_path = output_dir / f"network_features_{tile.replace('/', '_')}.png"
        plot_network_features(network_features, network_path,
                             title=f"Network/Connectivity Features: {experiment} - {tile}")
        
        # Save network features to CSV
        network_csv_path = output_dir / f"network_features_{tile.replace('/', '_')}.csv"
        network_df = pd.DataFrame(network_features).T
        network_df.to_csv(network_csv_path)
        print(f"[Results] Saved network features to {network_csv_path}")
    
    # 7. Per-object spatial overlap analysis
    if 'fluorescent' in masks and masks['fluorescent'] is not None:
        print(f"\n{'='*80}")
        print("Computing Per-Object Spatial Overlap")
        print(f"{'='*80}")
        
        # Collect test masks for comparison
        test_masks = {
            'raw_midslice': masks.get('raw_midslice'),
            'phase3d_midslice': masks.get('phase3d_midslice'),
            'phase2d': masks.get('phase2d'),
        }
        
        # Create violin plot for ALL objects
        overlap_violins_path = output_dir / f"per_object_overlap_{tile.replace('/', '_')}.png"
        plot_per_object_overlap_violins(
            ref_mask=masks['fluorescent'],
            test_masks=test_masks,
            output_path=overlap_violins_path,
            title=f"Per-Object Spatial Overlap (All Objects): {experiment} - {tile}"
        )
        
        # Create violin plot for TOP 10% LARGEST objects
        overlap_violins_large_path = output_dir / f"per_object_overlap_top10pct_{tile.replace('/', '_')}.png"
        plot_per_object_overlap_violins(
            ref_mask=masks['fluorescent'],
            test_masks=test_masks,
            output_path=overlap_violins_large_path,
            title=f"Per-Object Spatial Overlap (Top 10% Largest Objects): {experiment} - {tile}",
            filter_top_percent=10.0
        )
        
        # Save detailed overlap data to CSV for each image type
        total_area_metrics = []
        ref_total_area = (masks['fluorescent'] > 0).sum()

        for img_type in ['raw_midslice', 'phase3d_midslice', 'phase2d']:
            if test_masks.get(img_type) is not None:
                overlap_df = compute_per_object_spatial_overlap(
                    ref_mask=masks['fluorescent'],
                    test_mask=test_masks[img_type]
                )
                csv_path = output_dir / f"per_object_overlap_{img_type}_{tile.replace('/', '_')}.csv"
                overlap_df.to_csv(csv_path, index=False)
                print(f"[Results] Saved {img_type} per-object overlap to {csv_path}")

                # Compute total area overlap
                test_mask = test_masks[img_type]
                overlap_pixels = ((masks['fluorescent'] > 0) & (test_mask > 0)).sum()
                area_overlap_pct = (overlap_pixels / ref_total_area * 100.0) if ref_total_area > 0 else 0.0

                total_area_metrics.append({
                    'image_type': img_type,
                    'total_area_overlap_percent': area_overlap_pct
                })

        # Save total area overlap to CSV
        if total_area_metrics:
            area_csv_path = output_dir / f"total_area_overlap_{tile.replace('/', '_')}.csv"
            area_df = pd.DataFrame(total_area_metrics)
            area_df.to_csv(area_csv_path, index=False)
            print(f"[Results] Saved total area overlap to {area_csv_path}")
    
    print(f"\n{'='*80}")
    print(f"Complete! Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    sys.exit(main_cli())


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
"""
USAGE EXAMPLES
==============

Basic usage (saves to ./organelle_comparison_results/EXPERIMENT_TILE/):
    python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/029020
    
    # Results saved to:
    # ./organelle_comparison_results/ops0033_20250429_A_1_029020/
    #   ├── raw_images_A_1_029020.png                      (Raw images without overlays)
    #   ├── labeled_masks_A_1_029020.png                   (Colored labels to show connectivity)
    #   ├── segmentation_overlays_A_1_029020.png           (Segmentation contours on images)
    #   ├── mask_overlays_A_1_029020.png                   (Fluorescent vs detected mask overlays)
    #   ├── cytoplasm_mask_reference_A_1_029020.png        (Cytoplasm mask reference - if --mask-nuclei)
    #   ├── feature_overlap_A_1_029020.png                 (Morphology feature comparison - bars)
    #   ├── feature_overlap_A_1_029020.csv                 (Morphology feature data)
    #   ├── feature_distributions_A_1_029020.png           (Per-organelle distributions - violins)
    #   ├── network_features_A_1_029020.png                (Network/connectivity comparison)
    #   ├── network_features_A_1_029020.csv                (Network feature data)
    #   ├── per_object_overlap_A_1_029020.png              (Per-object spatial overlap - all objects)
    #   ├── per_object_overlap_top10pct_A_1_029020.png     (Per-object spatial overlap - top 10% largest)
    #   ├── per_object_overlap_raw_midslice_A_1_029020.csv (Detailed overlap data for raw BF)
    #   ├── per_object_overlap_phase3d_midslice_A_1_029020.csv (Detailed overlap data for Phase3D)
    #   ├── per_object_overlap_focus3d_A_1_029020.csv      (Detailed overlap data for Focus3D)
    #   └── per_object_overlap_phase2d_A_1_029020.csv      (Detailed overlap data for Phase2D)

With custom crop size:
    python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/029020 --crop-size 256

With custom crop center (specify Y and X pixel coordinates):
    python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/029020 \\
        --crop-center 1024 1536
    
    # This will center the 512x512 crop at pixel (Y=1024, X=1536)
    # Crop region will be Y=[768, 1280), X=[1280, 1792)

With GPU acceleration:
    python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/029020 --use-gpu

With custom output directory:
    python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/029020 \\
        --output-dir /path/to/my/results

Full example with all options:
    python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/029020 \\
        --crop-size 512 \\
        --crop-center 1024 1536 \\
        --fluorescent-channel mCherry \\
        --output-dir ./my_organelle_results \\
        --use-gpu \\
        --verbose

OUTPUT
======

Default save location:
    ./organelle_comparison_results/EXPERIMENT_TILE/
    
    Example: ops0033_20250429 tile A/1/029020 saves to:
    ./organelle_comparison_results/ops0033_20250429_A_1_029020/

The script generates seven output files:

1. raw_images_*.png
   - 2x3 grid showing raw images WITHOUT segmentation overlays
   - Clean visualization of all 5 image types
   - Useful for quality control and visual comparison

2. labeled_masks_*.png
   - 2x3 grid showing LABELED segmentations (each object has unique color)
   - Visualizes individual organelles and their connectivity
   - Especially useful for seeing mitochondrial networks

3. segmentation_overlays_*.png
   - 2x3 grid showing segmentation contours overlaid on raw images
   - Cyan contours mark detected organelles
   - Object counts displayed per image type

4. feature_overlap_*.png
   - Bar plot comparing MORPHOLOGICAL features
   - X-axis: 5 image types
   - Y-axis: % feature similarity with fluorescent label (0-100%)
   - Bars: Mean across all features
   - Dots: Individual features (area, axes, eccentricity, solidity, extent)
   - Includes comprehensive features: Hu moments, Haralick texture, Frangi intensity

5. feature_overlap_*.csv
   - Numerical morphological feature overlap data
   - Key columns: area, axis_major_length, axis_minor_length, eccentricity, 
     solidity, extent, n_matched, n_ref, n_test
   - Comprehensive feature set (30+ measurements per organelle)
   - Includes intensity statistics, texture features, shape descriptors

6. network_features_*.png
   - Bar plot comparing NETWORK/CONNECTIVITY features
   - Shows skeleton-based measurements: branch count, endpoints, junctions,
     network density, branching density, average degree
   - Y-axis: % relative to fluorescent (ground truth = 100%)
   - Values >100% indicate better connectivity than fluorescent label
   - Bars: Mean across all network metrics
   - Dots: Individual network metrics
   - Red dashed line: Fluorescent baseline (100%)

7. network_features_*.csv
   - Numerical network feature data
   - Skeleton analysis results: branches, nodes, topology, connectivity metrics
   - Useful for quantifying mitochondrial network complexity

INTERPRETATION
==============
This comprehensive comparison validates:

1. **Morphological Preservation**: How well phase reconstruction captures organelle
   shape and texture compared to fluorescent labeling (file #4, #5).

2. **Network Connectivity**: How well phase reconstruction preserves the filamentous
   structure and connectivity of organelles like mitochondria (file #6, #7).

3. **Method Comparison**: Compare different reconstruction approaches:
   - Raw BF mid-slice: Baseline (low signal, unfocused)
   - Phase3D mid-slice: 3D reconstruction at arbitrary Z
   - Focus3D: Best Z-slice per region (autofocus map)
   - Phase2D: Optimized 2D autofocus reconstruction

Higher scores indicate better preservation of organelle structure. Network features
are particularly important for filamentous organelles (mitochondria, ER) where
connectivity matters.
"""


# usage: python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/047039 --crop-size 412 --crop-center 974 1750 --mask-nuclei --verbose
# usage: python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0065_20250812 A/1/025022 --crop-size 412 --crop-center 974 1750 --mask-nuclei --verbose


# mitochondria list
# 1: python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/047039 --crop-size 412 --crop-center 974 1750 --mask-nuclei --verbose --fluor-shift-y 2.0
# 2: python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/049031 --crop-size 352 --crop-center 1474 1500 --mask-nuclei --verbose
# 3: python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/048032 --crop-size 352 --crop-center 900 1024 --mask-nuclei --verbose
# 4: python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/049033 --crop-size 352 --crop-center 824 1524 --mask-nuclei --verbose
# 5: python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/050033 --crop-size 352 --crop-center 1124 1124 --mask-nuclei --verbose --fluor-shift-y 0.0 
# 6: python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/050034 --crop-size 252 --crop-center 1024 1024 --mask-nuclei --verbose --fluor-shift-y 0.0 
# 7: python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/051035 --crop-size 252 --crop-center 1324 1324 --mask-nuclei --verbose --fluor-shift-y 0.0 
# 8: python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/047037 --crop-size 252 --crop-center 1324 1424 --mask-nuclei --verbose --fluor-shift-y 0.0 
# 9: python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/047038 --crop-size 352 --crop-center 954 1444 --mask-nuclei --verbose --fluor-shift-y 0.0 
# 10: python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/046038 --crop-size 352 --crop-center 500 700 --mask-nuclei --verbose --fluor-shift-y 2.0 --fluor-shift-x 2.0 
# 11: python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0033_20250429 A/1/044038 --crop-size 352 --crop-center 800 500 --mask-nuclei --verbose --fluor-shift-y 0.0 
# late endosome list
# 1: python -m ops_process.ops_analysis.napari.organelle_feature_comparison ops0065_20250812 A/1/025022 --crop-size 412 --crop-center 974 1750 --mask-nuclei --verbose