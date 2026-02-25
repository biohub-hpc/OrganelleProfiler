"""
Blob Detection for Organelle Segmentation
==========================================

This module provides LoG (Laplacian of Gaussian) blob detection methods
for detecting round, blob-like structures such as nucleoli, vesicles,
and other punctate organelles.

LoG is specifically designed for detecting blob-like structures and is
better suited for round objects than Frangi (which targets ridges/tubes).

Key functions:
- _segment_blob_log: General LoG blob detection with optional masking
- _segment_nucleoli_blob: Wrapper for nucleoli detection
- _segment_nucleoli_frangi: Alternative Frangi-based nucleoli detection
- _segment_nucleoli_in_tile: Dispatcher for nucleoli segmentation methods
"""

import numpy as np
from skimage.feature import blob_log
from skimage.draw import disk
from skimage.measure import label
from skimage.filters import frangi

from .configs import (
    DEFAULT_METHODS,
    SEGMENTATION_CONFIGS,
)
from .frangi import compute_frangi_threshold
from .postprocessing import postprocess_nucleoli_mask


def _segment_nucleoli_frangi(
    tile_data: np.ndarray,
    tile_mask: np.ndarray,
    pixel_resolution_um: float,
    frangi_params: dict = None,
) -> np.ndarray:
    """
    Segment nucleoli within a tile using Vesicular Frangi filter with nuclear masking.

    This approach is effective for nucleoli because:
    1. Frangi is gradient-based and handles masked images naturally
    2. Vesicular params (alpha=0.5) detect round blob-like structures
    3. Nuclear mask constrains detection to within nuclei only

    Args:
        tile_data: Image tile data (2D array, typically Phase2D channel)
        tile_mask: Nuclear mask for the tile (labels, not binary) - from nuclear_seg
        pixel_resolution_um: Pixel size in micrometers
        frangi_params: Optional custom Frangi params (default: nucleoli frangi config from SEGMENTATION_CONFIGS)

    Returns:
        Labeled mask for nucleoli in the tile (int32)
    """
    # Skip tiles with no nuclei
    if tile_mask.max() == 0:
        return np.zeros_like(tile_mask, dtype=np.int32)

    # Use default nucleoli Frangi params if not specified
    if frangi_params is None:
        frangi_params = SEGMENTATION_CONFIGS[("nucleoli", "phase2d", "frangi")].copy()

    # Create binary mask for nuclei regions
    binary_mask = tile_mask > 0

    # Apply nuclear mask BEFORE Frangi - zeros outside nuclei
    # Frangi's gradient calculation handles masked images naturally
    masked_tile = tile_data.astype(np.float32)
    masked_tile[~binary_mask] = 0.0

    # Calculate sigma range in pixels
    min_sigma = frangi_params["min_radius_um"] / pixel_resolution_um
    max_sigma = frangi_params["max_radius_um"] / pixel_resolution_um
    sigmas = np.geomspace(min_sigma, max_sigma, num=5)

    # Run vesicular Frangi (matching sweep script - only sigmas and black_ridges)
    # Since nucleoli are bright inside nuclei, use black_ridges=False
    vesselness = frangi(
        masked_tile,
        sigmas=sigmas,
        black_ridges=False,  # Bright nucleoli on darker nuclear interior
    )

    if vesselness.max() == 0:
        return np.zeros_like(tile_mask, dtype=np.int32)

    # Use fixed or dynamic thresholding
    fixed_threshold = frangi_params.get("threshold", 0.01)
    if fixed_threshold is not None:
        # Use fixed threshold directly
        threshold = fixed_threshold
    else:
        # Use dynamic thresholding with threshold_mult
        threshold_mult = frangi_params.get("threshold_mult", 0.01)
        threshold = compute_frangi_threshold(vesselness, threshold_mult=threshold_mult, xp=np)

    # Apply threshold
    binary_result = vesselness > threshold

    # Ensure nucleoli are only within nuclear boundaries
    binary_result = binary_result & binary_mask

    # Apply aggressive post-processing for large round structures
    # Extract postprocess params from config
    pp_min_size = frangi_params.get("min_object_size", 20)
    pp_do_opening = frangi_params.get("postprocess_opening", True)
    pp_opening_radius = frangi_params.get("postprocess_opening_radius", 1)
    pp_do_closing = frangi_params.get("postprocess_closing", True)
    pp_closing_radius = frangi_params.get("postprocess_closing_radius", 3)
    binary_result = postprocess_nucleoli_mask(
        binary_result,
        min_size=pp_min_size,
        do_opening=pp_do_opening,
        opening_radius=pp_opening_radius,
        do_closing=pp_do_closing,
        closing_radius=pp_closing_radius,
    )

    # Label connected components
    nucleoli_labels = label(binary_result).astype(np.int32)

    return nucleoli_labels


def _segment_blob_log(
    tile_data: np.ndarray,
    pixel_resolution_um: float,
    blob_params: dict,
    mask: np.ndarray = None,
    invert: bool = False,
) -> np.ndarray:
    """
    Unified LoG blob detection for nucleoli, vesicles, etc.

    LoG is specifically designed for detecting blob-like structures and is better
    suited for round objects than Frangi (which targets ridges/tubes).

    The algorithm:
    1. Applies LoG at multiple scales (sigma values)
    2. Finds local maxima in scale-space
    3. Converts blob centers to circular masks
    4. Labels connected components

    Args:
        tile_data: Image tile data (2D array)
        pixel_resolution_um: Pixel size in micrometers
        blob_params: Blob detection params (min_radius_um, max_radius_um, threshold, etc.)
        mask: Optional mask (labels or binary). If provided, only detect within mask.
        invert: If True, invert image to detect dark blobs on bright background.

    Returns:
        Labeled mask (int32)
    """
    # Handle optional mask
    if mask is not None:
        binary_mask = mask > 0
        if binary_mask.sum() == 0:
            return np.zeros(tile_data.shape, dtype=np.int32)
    else:
        binary_mask = None

    # Normalize image for blob detection (0-1 range works best for LoG)
    tile_float = tile_data.astype(np.float32)
    if binary_mask is not None:
        tile_float[~binary_mask] = 0.0
        mask_values = tile_float[binary_mask]
    else:
        mask_values = tile_float.ravel()

    if mask_values.size > 0 and mask_values.max() > mask_values.min():
        vmin, vmax = np.percentile(mask_values, [1, 99])
        tile_norm = np.clip((tile_float - vmin) / (vmax - vmin + 1e-8), 0, 1)
    else:
        return np.zeros(tile_data.shape, dtype=np.int32)

    # Invert for dark blob detection (vesicular_dark)
    if invert:
        tile_norm = 1.0 - tile_norm

    # Calculate sigma range in pixels (sigma H radius / sqrt(2) for LoG)
    min_sigma = blob_params["min_radius_um"] / pixel_resolution_um / np.sqrt(2)
    max_sigma = blob_params["max_radius_um"] / pixel_resolution_um / np.sqrt(2)

    # Ensure reasonable sigma values
    min_sigma = max(1.0, min_sigma)
    max_sigma = max(min_sigma + 1, max_sigma)

    # Run LoG blob detection
    # Returns array of (y, x, sigma) for each detected blob
    blobs = blob_log(
        tile_norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=blob_params.get("num_sigma", 10),
        threshold=blob_params.get("threshold", 0.02),
        overlap=blob_params.get("overlap", 0.5),
        exclude_border=blob_params.get("exclude_border", False),
    )

    if len(blobs) == 0:
        return np.zeros(tile_data.shape, dtype=np.int32)

    # Create labeled mask from blob detections
    labels = np.zeros(tile_data.shape, dtype=np.int32)

    for i, (y, x, sigma) in enumerate(blobs):
        # Radius is approximately sigma * sqrt(2) for LoG
        radius = max(2, int(sigma * np.sqrt(2)))

        # Create circular mask for this blob
        rr, cc = disk((int(y), int(x)), radius, shape=tile_data.shape)

        if binary_mask is not None:
            # Only include pixels within the mask
            valid = binary_mask[rr, cc]
            rr, cc = rr[valid], cc[valid]

        # Assign label (don't overwrite existing labels - first come first served)
        unlabeled = labels[rr, cc] == 0
        labels[rr[unlabeled], cc[unlabeled]] = i + 1

    return labels


def _segment_nucleoli_blob(
    tile_data: np.ndarray,
    tile_mask: np.ndarray,
    pixel_resolution_um: float,
    blob_params: dict = None,
) -> np.ndarray:
    """Segment nucleoli using LoG blob detection (wrapper for _segment_blob_log)."""
    if blob_params is None:
        blob_params = SEGMENTATION_CONFIGS[("nucleoli", "phase2d", "blob")].copy()
    return _segment_blob_log(tile_data, pixel_resolution_um, blob_params, mask=tile_mask)


def _segment_nucleoli_in_tile(
    tile_data,
    tile_mask,
    pixel_resolution_um=0.108,
    method: str = None,
    frangi_params: dict = None,
    blob_params: dict = None,
):
    """
    Segment nucleoli within a tile using the specified method.

    Args:
        tile_data: Image tile data
        tile_mask: Nuclear mask for the tile (labels, not binary)
        pixel_resolution_um: Pixel size in micrometers (used for Frangi/blob)
        method: "blob" (LoG, default) or "frangi"
        frangi_params: Optional custom Frangi params (only used if method="frangi")
        blob_params: Optional custom blob params (only used if method="blob")

    Returns:
        Labeled mask for nucleoli in the tile
    """
    # Use default method if not specified
    if method is None:
        method = DEFAULT_METHODS.get(("nucleoli", "phase2d"), "frangi")

    if method == "blob":
        return _segment_nucleoli_blob(
            tile_data=tile_data,
            tile_mask=tile_mask,
            pixel_resolution_um=pixel_resolution_um,
            blob_params=blob_params,
        )
    else:
        # Frangi method (default)
        return _segment_nucleoli_frangi(
            tile_data=tile_data,
            tile_mask=tile_mask,
            pixel_resolution_um=pixel_resolution_um,
            frangi_params=frangi_params,
        )
