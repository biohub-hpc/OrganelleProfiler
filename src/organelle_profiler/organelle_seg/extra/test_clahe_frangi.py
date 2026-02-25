"""
Test script for comparing CLAHE + Frangi pipeline parameters.

Loads small regions from fluorescent and label-free channels and tests
different CLAHE clip limits, kernel sizes, and Frangi parameters.

Also includes LoG blob detection parameter sweeps for vesicular structures.

Usage:
    python -m organelle_profiler.feature_extraction.test_clahe_frangi --experiment 33 --position A/1/0
    python -m organelle_profiler.feature_extraction.test_clahe_frangi -e ops0049_20250626 -p A/2/0 --crop-size 1024
    python -m organelle_profiler.feature_extraction.test_clahe_frangi -e 33 -p A/1/0 --single-test -c mCherry --pixel-size 0.1625
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import exposure, filters, restoration
from skimage.filters import frangi
from skimage.measure import label
from skimage.color import label2rgb
from scipy import ndimage as ndi
from iohub import open_ome_zarr

import sys
import os
sys.path.insert(0, os.getcwd())

from ops_utils.data.experiment import OpsDataset
from ops_utils.data.filesystem import resolve_experiment_name

# Import postprocessing and thresholding from org_seg_utils
from organelle_profiler.feature_extraction.org_seg_utils import (
    postprocess_nucleoli_mask,
    compute_frangi_threshold,
    watershed_label,
)

# Import blob detection sweep functions
from organelle_profiler.feature_extraction.test_blob_detection import (
    compare_blob_thresholds,
    compare_blob_radius,
    compare_blob_num_sigma,
    compare_blob_overlap,
    VESICLE_BLOB_CONFIG,
)

# Global verbose flag - set by --verbose/-v argument
VERBOSE = False


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Crop the center of an image to the specified size."""
    h, w = img.shape[:2]
    if size >= h and size >= w:
        return img
    cy, cx = h // 2, w // 2
    half = size // 2
    return img[cy - half:cy + half, cx - half:cx + half]


def um_to_sigmas(min_radius_um: float, max_radius_um: float, pixel_size_um: float, num_sigmas: int = 5):
    """
    Convert radius range in microns to Frangi sigma array in pixels.

    Args:
        min_radius_um: Minimum structure radius in microns
        max_radius_um: Maximum structure radius in microns
        pixel_size_um: Pixel size in microns (e.g., 0.108)
        num_sigmas: Number of sigma values to generate

    Returns:
        numpy array of sigma values in pixels
    """
    min_sigma_px = min_radius_um / pixel_size_um
    max_sigma_px = max_radius_um / pixel_size_um
    return np.geomspace(min_sigma_px, max_sigma_px, num=num_sigmas)


def load_crop(zarr_path: str, position: str, channel: str, crop_size: int = 512, offset: tuple = None, bbox: tuple = None):
    """
    Load a small crop from a zarr store.

    Args:
        zarr_path: Path to zarr store
        position: Position path (e.g., "A/1/0")
        channel: Channel name
        crop_size: Size of square crop (ignored if bbox is provided)
        offset: (y_offset, x_offset) from center. If None, uses center. (ignored if bbox is provided)
        bbox: (y_start, y_end, x_start, x_end) exact crop coordinates. If provided, overrides crop_size and offset.

    Returns:
        2D numpy array
    """
    with open_ome_zarr(zarr_path, mode="r") as ds:
        channel_names = list(ds.channel_names)
        if channel not in channel_names:
            raise ValueError(f"Channel '{channel}' not found. Available: {channel_names}")

        ch_idx = channel_names.index(channel)
        pos = ds[position]
        arr = pos["0"]

        # Get image dimensions
        _, _, _, height, width = arr.shape

        # Calculate crop region
        if bbox is not None:
            # Use exact bbox coordinates
            y_start, y_end, x_start, x_end = bbox
            # Clamp to image bounds
            y_start = max(0, y_start)
            y_end = min(height, y_end)
            x_start = max(0, x_start)
            x_end = min(width, x_end)
        elif offset:
            cy = height // 2 + offset[0]
            cx = width // 2 + offset[1]
            half = crop_size // 2
            y_start = max(0, cy - half)
            y_end = min(height, cy + half)
            x_start = max(0, cx - half)
            x_end = min(width, cx + half)
        else:
            cy, cx = height // 2, width // 2
            half = crop_size // 2
            y_start = max(0, cy - half)
            y_end = min(height, cy + half)
            x_start = max(0, cx - half)
            x_end = min(width, cx + half)

        # Load crop
        crop = np.squeeze(np.asarray(arr[0, ch_idx, :, y_start:y_end, x_start:x_end]))

        if VERBOSE:
            print(f"  Loaded {channel}: shape {crop.shape}, dtype {crop.dtype}")
            print(f"    Crop region: Y[{y_start}:{y_end}], X[{x_start}:{x_end}]")
            print(f"    Value range: [{crop.min():.2f}, {crop.max():.2f}]")

        return crop


def load_mask_crop(zarr_path: str, position: str, mask_name: str, crop_size: int = 512, offset: tuple = None):
    """
    Load a mask crop from the labels group of a zarr store.

    Args:
        zarr_path: Path to zarr store
        position: Position path (e.g., "A/1/0")
        mask_name: Name of the mask in labels group (e.g., "nucle_vs_seg")
        crop_size: Size of square crop
        offset: (y_offset, x_offset) from center. If None, uses center.

    Returns:
        2D numpy array (int32 labels or binary mask)
    """
    with open_ome_zarr(zarr_path, mode="r") as ds:
        pos = ds[position]

        # Access labels group
        if not hasattr(pos, 'zgroup') or 'labels' not in pos.zgroup:
            raise ValueError(f"No labels group found in position {position}")

        labels_group = pos.zgroup['labels']
        if mask_name not in labels_group:
            available = list(labels_group.keys())
            raise ValueError(f"Mask '{mask_name}' not found. Available: {available}")

        mask_arr = labels_group[mask_name]['0']

        # Get mask dimensions
        shape = mask_arr.shape
        if len(shape) == 5:
            _, _, _, height, width = shape
        elif len(shape) == 4:
            _, _, height, width = shape
        else:
            height, width = shape[-2], shape[-1]

        # Calculate crop region (center by default)
        if offset:
            cy = height // 2 + offset[0]
            cx = width // 2 + offset[1]
        else:
            cy, cx = height // 2, width // 2

        half = crop_size // 2
        y_start = max(0, cy - half)
        y_end = min(height, cy + half)
        x_start = max(0, cx - half)
        x_end = min(width, cx + half)

        # Load crop
        if len(shape) == 5:
            crop = np.squeeze(np.asarray(mask_arr[0, 0, :, y_start:y_end, x_start:x_end]))
        elif len(shape) == 4:
            crop = np.squeeze(np.asarray(mask_arr[0, :, y_start:y_end, x_start:x_end]))
        else:
            crop = np.squeeze(np.asarray(mask_arr[..., y_start:y_end, x_start:x_end]))

        if VERBOSE:
            print(f"  Loaded mask {mask_name}: shape {crop.shape}, dtype {crop.dtype}")
            print(f"    Crop region: Y[{y_start}:{y_end}], X[{x_start}:{x_end}]")
            print(f"    Unique labels: {len(np.unique(crop))}, max={crop.max()}")

        return crop


def compare_clahe_frangi(
    image: np.ndarray,
    title: str = "Test",
    clip_limits: list = [0.01, 0.03, 0.05, 0.1],
    kernel_sizes: list = [32, 64, 128, 256],
    min_radius_um: float = 0.1,
    max_radius_um: float = 1.5,
    pixel_size_um: float = 0.325,
    threshold: float = None,
    threshold_mult: float = 0.01,
    denoise: bool = False,
    denoise_weight: float = 0.1,
    gamma: float = 1.0,
    frangi_gamma: float = 0.3,
    output_path: str = None,
    display_size: int = None,
):
    """
    Compare CLAHE + Frangi pipeline with varying parameters.

    Creates a grid showing for each kernel size:
    - Row 0: Raw/CLAHE
    - Row 1: Frangi response
    - Row 2: Labeled objects (colored)
    - Row 3: Overlay on raw

    Args:
        image: 2D numpy array (grayscale)
        title: Title for the figure
        clip_limits: List of clip limits to test
        kernel_sizes: List of kernel sizes to test
        min_radius_um: Minimum structure radius in microns
        max_radius_um: Maximum structure radius in microns
        pixel_size_um: Pixel size in microns
        threshold: Fixed threshold for Frangi. If None, uses dynamic thresholding.
        threshold_mult: Multiplier for dynamic threshold (only used if threshold is None)
        denoise: If True, apply TV denoising before CLAHE
        denoise_weight: Weight for TV denoising
        gamma: Gamma correction for display (< 1 brightens dim structures)
        frangi_gamma: Gamma for Frangi response (< 1 brightens dim structures)
        output_path: Path to save figure (optional)
        display_size: Size of center crop for display (default: full image)
    """
    n_clips = len(clip_limits)
    n_kernels = len(kernel_sizes)

    # DEBUG: Print input state
    if VERBOSE:
        print(f"  [SWEEP DEBUG] Input: dtype={image.dtype}, shape={image.shape}, range=[{image.min():.4f}, {image.max():.4f}]")

    # Normalize image to [0, 1]
    img_norm = image.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

    # DEBUG: Print after normalization
    if VERBOSE:
        print(f"  [SWEEP DEBUG] After normalize: dtype={img_norm.dtype}, range=[{img_norm.min():.4f}, {img_norm.max():.4f}]")

    # Optional denoising
    if denoise:
        if VERBOSE:
            print(f"  Applying TV denoising (weight={denoise_weight})...")
        img_input = restoration.denoise_tv_chambolle(img_norm, weight=denoise_weight)
    else:
        img_input = img_norm

    # Create figure: 4 rows per kernel size (raw, frangi, labeled, overlay), cols = clip limits
    n_rows = n_kernels * 4
    fig, axes = plt.subplots(n_rows, n_clips + 1, figsize=(3.5 * (n_clips + 1), 3 * n_rows))

    # Display image with gamma
    display_img = np.power(img_norm, gamma) if gamma != 1.0 else img_norm
    # Apply display crop for first column
    disp_orig = center_crop(display_img, display_size) if display_size else display_img

    # Process each kernel size
    for k_idx, kernel_size in enumerate(kernel_sizes):
        base_row = k_idx * 4

        # First column: labels for each row type
        row_labels = ["Raw", "Frangi", "Labeled", "Overlay"]
        for r, lbl in enumerate(row_labels):
            ax = axes[base_row + r, 0]
            ax.imshow(disp_orig, cmap='gray', vmin=0, vmax=1)
            if r == 0:
                ax.set_title(f"Original\n(γ={gamma})", fontsize=9)
            # Add row label as text on the left side (ylabel doesn't work with axis off)
            ax.text(-0.1, 0.5, f"K={kernel_size}\n{lbl}", fontsize=9, fontweight='bold',
                   transform=ax.transAxes, ha='right', va='center')
            ax.axis('off')

        # Process each clip limit
        for col, clip_limit in enumerate(clip_limits):
            # Apply CLAHE
            try:
                if VERBOSE:
                    print(f"  [SWEEP DEBUG] Before CLAHE (K={kernel_size}, clip={clip_limit}): dtype={img_input.dtype}, range=[{img_input.min():.4f}, {img_input.max():.4f}]")
                img_clahe = exposure.equalize_adapthist(
                    img_input,
                    clip_limit=clip_limit,
                    kernel_size=kernel_size
                )
                if VERBOSE:
                    print(f"  [SWEEP DEBUG] After CLAHE: dtype={img_clahe.dtype}, range=[{img_clahe.min():.4f}, {img_clahe.max():.4f}]")
            except Exception as e:
                for r in range(4):
                    ax = axes[base_row + r, col + 1]
                    ax.text(0.5, 0.5, f"CLAHE Error:\n{str(e)[:20]}",
                           ha='center', va='center', transform=ax.transAxes, fontsize=8)
                    ax.axis('off')
                continue

            # Apply Frangi with adaptive sigmas
            try:
                sigmas = um_to_sigmas(min_radius_um, max_radius_um, pixel_size_um)
                if VERBOSE:
                    print(f"  [SWEEP DEBUG] Before Frangi: dtype={img_clahe.dtype}, range=[{img_clahe.min():.4f}, {img_clahe.max():.4f}]")
                    print(f"  [SWEEP DEBUG] Frangi sigmas (px): {sigmas}")
                img_frangi = frangi(
                    img_clahe,
                    sigmas=sigmas,
                    black_ridges=False
                )
                if VERBOSE:
                    print(f"  [SWEEP DEBUG] After Frangi: dtype={img_frangi.dtype}, range=[{img_frangi.min():.6f}, {img_frangi.max():.6f}]")
            except Exception as e:
                for r in range(4):
                    ax = axes[base_row + r, col + 1]
                    ax.text(0.5, 0.5, f"Frangi Error:\n{str(e)[:20]}",
                           ha='center', va='center', transform=ax.transAxes, fontsize=8)
                    ax.axis('off')
                continue

            # Create labeled mask - use fixed or dynamic threshold
            if threshold is not None:
                thresh_val = threshold
                if VERBOSE:
                    print(f"  [SWEEP DEBUG] Using FIXED threshold={thresh_val:.6f}")
            else:
                thresh_val = compute_frangi_threshold(img_frangi, threshold_mult=threshold_mult, xp=np)
                if VERBOSE:
                    print(f"  [SWEEP DEBUG] Using DYNAMIC threshold (mult={threshold_mult}): {thresh_val:.6f}")
            binary = img_frangi > thresh_val
            labeled = label(binary)
            n_objects = labeled.max()
            if VERBOSE:
                print(f"  [SWEEP DEBUG] Binary pixels: {binary.sum()}, Num objects: {n_objects}")

            # Apply display crop if specified (process full, display center)
            disp_clahe = center_crop(img_clahe, display_size) if display_size else img_clahe
            disp_frangi = center_crop(img_frangi, display_size) if display_size else img_frangi
            disp_labeled = center_crop(labeled, display_size) if display_size else labeled

            # Row 0: Raw/CLAHE
            ax = axes[base_row + 0, col + 1]
            ax.imshow(disp_clahe, cmap='gray')
            if k_idx == 0:
                ax.set_title(f"Clip: {clip_limit}", fontsize=9)
            ax.axis('off')

            # Row 1: Frangi
            ax = axes[base_row + 1, col + 1]
            frangi_enhanced = np.power(disp_frangi / (disp_frangi.max() + 1e-8), frangi_gamma)
            ax.imshow(frangi_enhanced, cmap='inferno')
            ax.text(0.02, 0.98, f"max={img_frangi.max():.3f}",
                   transform=ax.transAxes, fontsize=7, color='white',
                   va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.axis('off')

            # Row 2: Labeled objects
            ax = axes[base_row + 2, col + 1]
            labeled_rgb = label2rgb(disp_labeled, bg_label=0, bg_color=(0, 0, 0))
            ax.imshow(labeled_rgb)
            ax.text(0.02, 0.98, f"n={n_objects}",
                   transform=ax.transAxes, fontsize=7, color='white',
                   va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.axis('off')

            # Row 3: Overlay (always apply gamma=0.5 for visibility)
            ax = axes[base_row + 3, col + 1]
            clahe_display = np.power(disp_clahe, 0.5)  # Always brighten for overlay visibility
            overlay_rgb = label2rgb(disp_labeled, image=clahe_display, bg_label=0, alpha=0.5)
            ax.imshow(overlay_rgb)
            ax.axis('off')

    thresh_str = f"fixed={threshold}" if threshold is not None else f"dynamic(mult={threshold_mult})"
    plt.suptitle(f"{title}\nFrangi: r={min_radius_um}-{max_radius_um}μm (px={pixel_size_um}), α=0.5, β=0.5 | Thresh: {thresh_str} | Frangi γ={frangi_gamma}", fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        if VERBOSE:
            print(f"  Saved: {output_path}")

    plt.close(fig)


def compare_frangi_params(
    image: np.ndarray,
    title: str = "Frangi Parameter Comparison",
    clahe_clip: float = 0.01,
    clahe_kernel: int = 256,
    beta_values: list = [0.1, 0.3, 0.5, 1.0],
    radius_ranges_um: list = [(0.1, 0.5), (0.1, 1.0), (0.2, 1.5), (0.1, 2.0)],
    pixel_size_um: float = 0.325,
    threshold: float = None,
    threshold_mult: float = 0.01,
    gamma: float = 1.0,
    frangi_gamma: float = 0.3,
    output_path: str = None,
):
    """
    Compare Frangi filter parameters after fixed CLAHE preprocessing.

    Creates a grid showing for each beta value:
    - Row 0: Raw/CLAHE
    - Row 1: Frangi response
    - Row 2: Labeled objects (colored)
    - Row 3: Overlay on raw

    Args:
        image: 2D numpy array
        title: Figure title
        clahe_clip: Fixed CLAHE clip limit
        clahe_kernel: Fixed CLAHE kernel size
        beta_values: List of beta values (sensitivity to blob-like structures, relevant for 2D)
        radius_ranges_um: List of (min_radius_um, max_radius_um) tuples in microns
        pixel_size_um: Pixel size in microns
        threshold: Fixed threshold for Frangi. If None, uses dynamic thresholding.
        threshold_mult: Multiplier for dynamic threshold (only used if threshold is None)
        gamma: Gamma for original image display
        frangi_gamma: Gamma for Frangi response (< 1 brightens dim structures)
        output_path: Path to save figure
    """
    n_betas = len(beta_values)
    n_ranges = len(radius_ranges_um)

    # Normalize and apply CLAHE
    img_norm = image.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

    if VERBOSE:
        print(f"  Applying CLAHE (clip={clahe_clip}, kernel={clahe_kernel})...")
    img_clahe = exposure.equalize_adapthist(img_norm, clip_limit=clahe_clip, kernel_size=clahe_kernel)

    # Display image with gamma
    display_img = np.power(img_norm, gamma) if gamma != 1.0 else img_norm

    # Create figure: 4 rows per beta (raw, frangi, labeled, overlay), cols = radius ranges
    n_rows = n_betas * 4
    fig, axes = plt.subplots(n_rows, n_ranges + 1, figsize=(3.5 * (n_ranges + 1), 3 * n_rows))

    # Process each beta value
    for b_idx, beta in enumerate(beta_values):
        base_row = b_idx * 4

        # First column: labels for each row type
        row_labels = ["CLAHE", "Frangi", "Labeled", "Overlay"]
        for r, lbl in enumerate(row_labels):
            ax = axes[base_row + r, 0]
            if r == 0:
                ax.imshow(img_clahe, cmap='gray')
                if b_idx == 0:
                    ax.set_title(f"CLAHE\nclip={clahe_clip}", fontsize=9)
            else:
                ax.imshow(display_img, cmap='gray', vmin=0, vmax=1)
            # Add row label as text on the left side (ylabel doesn't work with axis off)
            ax.text(-0.1, 0.5, f"β={beta}\n{lbl}", fontsize=9, fontweight='bold',
                   transform=ax.transAxes, ha='right', va='center')
            ax.axis('off')

        # Process each radius range
        for col, (r_min, r_max) in enumerate(radius_ranges_um):
            # Apply Frangi with adaptive sigmas
            try:
                sigmas = um_to_sigmas(r_min, r_max, pixel_size_um)
                img_frangi = frangi(
                    img_clahe,
                    sigmas=sigmas,
                    alpha=0.5,
                    beta=beta,
                    black_ridges=False
                )
            except Exception as e:
                for r in range(4):
                    ax = axes[base_row + r, col + 1]
                    ax.text(0.5, 0.5, f"Error:\n{str(e)[:20]}",
                           ha='center', va='center', transform=ax.transAxes, fontsize=8)
                    ax.axis('off')
                continue

            # Create labeled mask - use fixed or dynamic threshold
            if threshold is not None:
                thresh_val = threshold
            else:
                thresh_val = compute_frangi_threshold(img_frangi, threshold_mult=threshold_mult, xp=np)
            binary = img_frangi > thresh_val
            labeled = label(binary)
            n_objects = labeled.max()

            # Row 0: CLAHE
            ax = axes[base_row + 0, col + 1]
            ax.imshow(img_clahe, cmap='gray')
            if b_idx == 0:
                ax.set_title(f"r: {r_min}-{r_max}μm", fontsize=9)
            ax.axis('off')

            # Row 1: Frangi
            ax = axes[base_row + 1, col + 1]
            frangi_enhanced = np.power(img_frangi / (img_frangi.max() + 1e-8), frangi_gamma)
            ax.imshow(frangi_enhanced, cmap='inferno')
            ax.text(0.02, 0.98, f"max={img_frangi.max():.3f}",
                   transform=ax.transAxes, fontsize=7, color='white',
                   va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.axis('off')

            # Row 2: Labeled objects
            ax = axes[base_row + 2, col + 1]
            labeled_rgb = label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))
            ax.imshow(labeled_rgb)
            ax.text(0.02, 0.98, f"n={n_objects}",
                   transform=ax.transAxes, fontsize=7, color='white',
                   va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.axis('off')

            # Row 3: Overlay (use CLAHE image with gamma for better contrast)
            ax = axes[base_row + 3, col + 1]
            clahe_display = np.power(img_clahe, 0.5)  # Always brighten for overlay visibility
            overlay_rgb = label2rgb(labeled, image=clahe_display, bg_label=0, alpha=0.5)
            ax.imshow(overlay_rgb)
            ax.axis('off')

    thresh_str = f"fixed={threshold}" if threshold is not None else f"dynamic(mult={threshold_mult})"
    plt.suptitle(f"{title}\nCLAHE: clip={clahe_clip}, k={clahe_kernel} | Frangi: α=0.5, px={pixel_size_um}μm | Thresh: {thresh_str} | Frangi γ={frangi_gamma}", fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        if VERBOSE:
            print(f"  Saved: {output_path}")

    plt.close(fig)


def compare_frangi_gamma(
    image: np.ndarray,
    title: str = "Frangi Gamma Comparison",
    clahe_clip: float = 0.01,
    clahe_kernel: int = 256,
    frangi_gamma_values: list = [None, 5, 15, 50],
    radius_ranges_um: list = [(0.1, 0.5), (0.1, 1.0), (0.2, 1.5), (0.1, 2.0)],
    pixel_size_um: float = 0.325,
    threshold: float = None,
    threshold_mult: float = 0.01,
    gamma: float = 1.0,
    display_frangi_gamma: float = 0.3,
    output_path: str = None,
):
    """
    Compare Frangi filter gamma parameter (sensitivity to high variance/texture).

    Creates a grid showing for each Frangi gamma value:
    - Row 0: Raw/CLAHE
    - Row 1: Frangi response
    - Row 2: Labeled objects (colored)
    - Row 3: Overlay on raw

    Args:
        image: 2D numpy array
        title: Figure title
        clahe_clip: Fixed CLAHE clip limit
        clahe_kernel: Fixed CLAHE kernel size
        frangi_gamma_values: List of Frangi gamma values (None = adaptive based on max Hessian norm)
        radius_ranges_um: List of (min_radius_um, max_radius_um) tuples in microns
        pixel_size_um: Pixel size in microns
        threshold: Fixed threshold for Frangi. If None, uses dynamic thresholding.
        threshold_mult: Multiplier for dynamic threshold (only used if threshold is None)
        gamma: Gamma for original image display
        display_frangi_gamma: Gamma for Frangi response visualization (< 1 brightens dim structures)
        output_path: Path to save figure
    """
    n_gammas = len(frangi_gamma_values)
    n_ranges = len(radius_ranges_um)

    # Normalize and apply CLAHE
    img_norm = image.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

    if VERBOSE:
        print(f"  Applying CLAHE (clip={clahe_clip}, kernel={clahe_kernel})...")
    img_clahe = exposure.equalize_adapthist(img_norm, clip_limit=clahe_clip, kernel_size=clahe_kernel)

    # Display image with gamma
    display_img = np.power(img_norm, gamma) if gamma != 1.0 else img_norm

    # Create figure: 4 rows per gamma (raw, frangi, labeled, overlay), cols = radius ranges
    n_rows = n_gammas * 4
    fig, axes = plt.subplots(n_rows, n_ranges + 1, figsize=(3.5 * (n_ranges + 1), 3 * n_rows))

    # Process each gamma value
    for g_idx, fg in enumerate(frangi_gamma_values):
        base_row = g_idx * 4
        fg_label = "auto" if fg is None else str(fg)

        # First column: labels for each row type
        row_labels = ["CLAHE", "Frangi", "Labeled", "Overlay"]
        for r, lbl in enumerate(row_labels):
            ax = axes[base_row + r, 0]
            if r == 0:
                ax.imshow(img_clahe, cmap='gray')
                if g_idx == 0:
                    ax.set_title(f"CLAHE\nclip={clahe_clip}", fontsize=9)
            else:
                ax.imshow(display_img, cmap='gray', vmin=0, vmax=1)
            # Add row label as text on the left side (ylabel doesn't work with axis off)
            ax.text(-0.1, 0.5, f"γ={fg_label}\n{lbl}", fontsize=9, fontweight='bold',
                   transform=ax.transAxes, ha='right', va='center')
            ax.axis('off')

        # Process each radius range
        for col, (r_min, r_max) in enumerate(radius_ranges_um):
            # Apply Frangi with adaptive sigmas
            try:
                sigmas = um_to_sigmas(r_min, r_max, pixel_size_um)
                img_frangi = frangi(
                    img_clahe,
                    sigmas=sigmas,
                    alpha=0.5,
                    beta=0.5,
                    gamma=fg,
                    black_ridges=False
                )
            except Exception as e:
                for r in range(4):
                    ax = axes[base_row + r, col + 1]
                    ax.text(0.5, 0.5, f"Error:\n{str(e)[:20]}",
                           ha='center', va='center', transform=ax.transAxes, fontsize=8)
                    ax.axis('off')
                continue

            # Create labeled mask - use fixed or dynamic threshold
            if threshold is not None:
                thresh_val = threshold
            else:
                thresh_val = compute_frangi_threshold(img_frangi, threshold_mult=threshold_mult, xp=np)
            binary = img_frangi > thresh_val
            labeled = label(binary)
            n_objects = labeled.max()

            # Row 0: CLAHE
            ax = axes[base_row + 0, col + 1]
            ax.imshow(img_clahe, cmap='gray')
            if g_idx == 0:
                ax.set_title(f"r: {r_min}-{r_max}μm", fontsize=9)
            ax.axis('off')

            # Row 1: Frangi
            ax = axes[base_row + 1, col + 1]
            frangi_enhanced = np.power(img_frangi / (img_frangi.max() + 1e-8), display_frangi_gamma)
            ax.imshow(frangi_enhanced, cmap='inferno')
            ax.text(0.02, 0.98, f"max={img_frangi.max():.3f}",
                   transform=ax.transAxes, fontsize=7, color='white',
                   va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.axis('off')

            # Row 2: Labeled objects
            ax = axes[base_row + 2, col + 1]
            labeled_rgb = label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))
            ax.imshow(labeled_rgb)
            ax.text(0.02, 0.98, f"n={n_objects}",
                   transform=ax.transAxes, fontsize=7, color='white',
                   va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.axis('off')

            # Row 3: Overlay (use CLAHE image with gamma for better contrast)
            ax = axes[base_row + 3, col + 1]
            clahe_display = np.power(img_clahe, 0.5)  # Always brighten for overlay visibility
            overlay_rgb = label2rgb(labeled, image=clahe_display, bg_label=0, alpha=0.5)
            ax.imshow(overlay_rgb)
            ax.axis('off')

    thresh_str = f"fixed={threshold}" if threshold is not None else f"dynamic(mult={threshold_mult})"
    plt.suptitle(f"{title}\nCLAHE: clip={clahe_clip}, k={clahe_kernel} | Frangi: α=0.5, β=0.5, px={pixel_size_um}μm | Thresh: {thresh_str} | Display γ={display_frangi_gamma}", fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        if VERBOSE:
            print(f"  Saved: {output_path}")

    plt.close(fig)


def compare_pixel_sizes(
    image: np.ndarray,
    title: str = "Pixel Size Comparison",
    clahe_clip: float = 0.01,
    clahe_kernel: int = 256,
    pixel_sizes_um: list = [0.1625, 0.2, 0.225, 0.25, 0.275, 0.325],
    min_radius_um: float = 0.1,
    max_radius_um: float = 1.5,
    threshold: float = None,
    threshold_mult: float = 0.01,
    gamma: float = 1.0,
    frangi_gamma: float = 0.3,
    output_path: str = None,
):
    """
    Compare effect of different pixel size assumptions on Frangi sigma calculation.

    This helps understand how pixel size affects the sigma values computed from
    a fixed micron-based radius range.

    Creates a single set of 4 rows:
    - Row 0: Raw/CLAHE
    - Row 1: Frangi response
    - Row 2: Labeled objects (colored)
    - Row 3: Overlay on raw

    Columns are different pixel sizes.

    Args:
        image: 2D numpy array
        title: Figure title
        clahe_clip: Fixed CLAHE clip limit
        clahe_kernel: Fixed CLAHE kernel size
        pixel_sizes_um: List of pixel sizes in microns to test (columns)
        min_radius_um: Minimum structure radius in microns
        max_radius_um: Maximum structure radius in microns
        threshold: Fixed threshold for Frangi. If None, uses dynamic thresholding.
        threshold_mult: Multiplier for dynamic threshold (only used if threshold is None)
        gamma: Gamma for original image display
        frangi_gamma: Gamma for Frangi response visualization (< 1 brightens dim structures)
        output_path: Path to save figure
    """
    n_px_sizes = len(pixel_sizes_um)

    # Normalize and apply CLAHE
    img_norm = image.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

    if VERBOSE:
        print(f"  Applying CLAHE (clip={clahe_clip}, kernel={clahe_kernel})...")
    img_clahe = exposure.equalize_adapthist(img_norm, clip_limit=clahe_clip, kernel_size=clahe_kernel)

    # Display image with gamma
    display_img = np.power(img_norm, gamma) if gamma != 1.0 else img_norm

    # Create figure: 4 rows (raw, frangi, labeled, overlay), cols = pixel sizes + 1 for labels
    fig, axes = plt.subplots(4, n_px_sizes + 1, figsize=(3.5 * (n_px_sizes + 1), 12))

    # First column: labels for each row type
    row_labels = ["CLAHE", "Frangi", "Labeled", "Overlay"]
    for r, lbl in enumerate(row_labels):
        ax = axes[r, 0]
        if r == 0:
            ax.imshow(img_clahe, cmap='gray')
            ax.set_title(f"CLAHE\nclip={clahe_clip}", fontsize=9)
        else:
            ax.imshow(display_img, cmap='gray', vmin=0, vmax=1)
        # Add row label as text on the left side
        ax.text(-0.1, 0.5, lbl, fontsize=9, fontweight='bold',
               transform=ax.transAxes, ha='right', va='center')
        ax.axis('off')

    # Process each pixel size
    for col, px_size in enumerate(pixel_sizes_um):
        # Apply Frangi with adaptive sigmas based on this pixel size
        try:
            sigmas = um_to_sigmas(min_radius_um, max_radius_um, px_size)
            img_frangi = frangi(
                img_clahe,
                sigmas=sigmas,
                alpha=0.5,
                beta=0.5,
                black_ridges=False
            )
        except Exception as e:
            for r in range(4):
                ax = axes[r, col + 1]
                ax.text(0.5, 0.5, f"Error:\n{str(e)[:20]}",
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.axis('off')
            continue

        # Create labeled mask - use fixed or dynamic threshold
        if threshold is not None:
            thresh_val = threshold
        else:
            thresh_val = compute_frangi_threshold(img_frangi, threshold_mult=threshold_mult, xp=np)
        binary = img_frangi > thresh_val
        labeled = label(binary)
        n_objects = labeled.max()

        # Show sigma range
        sigma_min = min_radius_um / px_size
        sigma_max = max_radius_um / px_size

        # Row 0: CLAHE
        ax = axes[0, col + 1]
        ax.imshow(img_clahe, cmap='gray')
        ax.set_title(f"px={px_size}μm", fontsize=9)
        ax.axis('off')

        # Row 1: Frangi
        ax = axes[1, col + 1]
        frangi_enhanced = np.power(img_frangi / (img_frangi.max() + 1e-8), frangi_gamma)
        ax.imshow(frangi_enhanced, cmap='inferno')
        ax.text(0.02, 0.98, f"σ={sigma_min:.1f}-{sigma_max:.1f}px\nmax={img_frangi.max():.3f}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 2: Labeled objects
        ax = axes[2, col + 1]
        labeled_rgb = label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))
        ax.imshow(labeled_rgb)
        ax.text(0.02, 0.98, f"n={n_objects}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 3: Overlay (use CLAHE image with gamma for better contrast)
        ax = axes[3, col + 1]
        clahe_display = np.power(img_clahe, 0.5)  # Always brighten for overlay visibility
        overlay_rgb = label2rgb(labeled, image=clahe_display, bg_label=0, alpha=0.5)
        ax.imshow(overlay_rgb)
        ax.axis('off')

    thresh_str = f"fixed={threshold}" if threshold is not None else f"dynamic(mult={threshold_mult})"
    plt.suptitle(f"{title}\nCLAHE: clip={clahe_clip}, k={clahe_kernel} | Frangi: r={min_radius_um}-{max_radius_um}μm, α=0.5, β=0.5 | Thresh: {thresh_str} | Display γ={frangi_gamma}", fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        if VERBOSE:
            print(f"  Saved: {output_path}")

    plt.close(fig)


def compare_nucleoli_clahe_frangi(
    image: np.ndarray,
    nuclear_mask: np.ndarray,
    title: str = "Nucleoli - CLAHE Parameter Grid",
    clip_limits: list = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
    kernel_sizes: list = [64, 128, 256, 512],
    min_radius_um: float = 0.5,
    max_radius_um: float = 3.0,
    pixel_size_um: float = 0.325,
    alpha: float = 0.1,
    beta: float = 0.5,
    threshold_mult: float = 0.01,
    gamma: float = 1.0,
    frangi_gamma: float = 0.3,
    output_path: str = None,
):
    """
    Compare CLAHE + Frangi for nucleoli detection with nuclear masking.

    Uses DYNAMIC thresholding like the main script (compute_frangi_threshold with threshold_mult).

    Similar to compare_clahe_frangi but:
    - Applies nuclear mask BEFORE Frangi (masks outside nuclei)
    - Uses low alpha (0.1) to favor round/isotropic structures
    - Larger default radius range (0.5-3.0μm) for nucleoli
    - Uses dynamic Otsu+Triangle thresholding with threshold_mult

    Args:
        image: 2D numpy array (Phase2D channel)
        nuclear_mask: 2D numpy array (nuclear segmentation labels)
        title: Title for the figure
        clip_limits: List of clip limits to test
        kernel_sizes: List of kernel sizes to test
        min_radius_um: Minimum nucleoli radius in microns (default 0.5)
        max_radius_um: Maximum nucleoli radius in microns (default 3.0)
        pixel_size_um: Pixel size in microns
        alpha: Frangi alpha (low = favor round, default 0.1)
        beta: Frangi beta (blob suppression, default 0.5)
        threshold_mult: Multiplier for dynamic threshold (default 0.01 for nucleoli)
        gamma: Gamma correction for display
        frangi_gamma: Gamma for Frangi response display
        output_path: Path to save figure
    """
    n_clips = len(clip_limits)
    n_kernels = len(kernel_sizes)

    # Normalize image to [0, 1]
    img_norm = image.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

    # Create binary mask from nuclear labels
    binary_mask = nuclear_mask > 0

    # Display image with gamma
    display_img = np.power(img_norm, gamma) if gamma != 1.0 else img_norm

    # Calculate sigmas from microns
    sigmas = um_to_sigmas(min_radius_um, max_radius_um, pixel_size_um)

    # DEBUG: Print exact parameters being used
    if VERBOSE:
        print(f"  [SWEEP DEBUG] Nucleoli CLAHE+Frangi comparison:")
        print(f"  [SWEEP DEBUG] pixel_size_um={pixel_size_um}")
        print(f"  [SWEEP DEBUG] radius range: {min_radius_um}-{max_radius_um}um")
        print(f"  [SWEEP DEBUG] sigmas (px): {sigmas}")
        print(f"  [SWEEP DEBUG] alpha={alpha}, beta={beta}, threshold_mult={threshold_mult}")

    # Create figure: 5 rows per kernel (CLAHE, Frangi, Binary, Postproc, Overlay), columns = clip limits + 1 for labels
    n_rows = 5 * n_kernels
    fig, axes = plt.subplots(n_rows, n_clips + 1, figsize=(3.5 * (n_clips + 1), 3.5 * n_rows))

    # First column: raw image with row labels
    for k_idx, kernel in enumerate(kernel_sizes):
        base_row = k_idx * 5
        for r_offset, lbl in enumerate(["CLAHE", "Frangi", "Binary", "Postproc", "Overlay"]):
            ax = axes[base_row + r_offset, 0]
            if r_offset == 0:
                ax.imshow(display_img, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"k={kernel}", fontsize=9)
            elif r_offset == 1:
                # Show masked region
                ax.imshow(display_img * binary_mask, cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(display_img, cmap='gray', vmin=0, vmax=1)
            ax.text(-0.1, 0.5, lbl, fontsize=9, fontweight='bold',
                   transform=ax.transAxes, ha='right', va='center')
            ax.axis('off')

    # Process each combination
    for k_idx, kernel in enumerate(kernel_sizes):
        base_row = k_idx * 5

        for col, clip in enumerate(clip_limits):
            # Apply nuclear mask BEFORE CLAHE (like main script)
            masked_input = img_norm.copy()
            masked_input[~binary_mask] = 0.0

            # Apply CLAHE on masked data
            img_clahe = exposure.equalize_adapthist(masked_input, clip_limit=clip, kernel_size=kernel)

            # Apply Frangi with low alpha for round structures
            img_frangi = frangi(
                img_clahe,
                sigmas=sigmas,
                alpha=alpha,
                beta=beta,
                black_ridges=False  # Nucleoli are bright
            )

            # Use DYNAMIC thresholding like main script (Otsu+Triangle on log scale)
            threshold = compute_frangi_threshold(img_frangi, threshold_mult=threshold_mult, xp=np)

            # Create binary mask
            binary = img_frangi > threshold
            # Ensure results are within nuclei
            binary = binary & binary_mask
            n_binary_pixels = binary.sum()

            # Apply postprocessing then watershed labeling (like main script)
            postprocessed = postprocess_nucleoli_mask(binary.copy(), min_size=20)
            # Use watershed_label with nucleoli-specific params from NUCLEOLI_FRANGI_CONFIG
            labeled = watershed_label(
                postprocessed,
                min_distance=3,
                min_object_size=20,
                compactness=1.0,
                erosion_iterations=2,
                min_peak_distance=2.0,
            )
            n_objects = labeled.max()

            # Row 0: CLAHE
            ax = axes[base_row, col + 1]
            ax.imshow(img_clahe, cmap='gray')
            ax.set_title(f"clip={clip}", fontsize=9)
            ax.axis('off')

            # Row 1: Frangi (with gamma enhancement)
            ax = axes[base_row + 1, col + 1]
            frangi_enhanced = np.power(img_frangi / (img_frangi.max() + 1e-8), frangi_gamma)
            ax.imshow(frangi_enhanced, cmap='inferno')
            ax.text(0.02, 0.98, f"max={img_frangi.max():.3f}\nth={threshold:.4f}",
                   transform=ax.transAxes, fontsize=7, color='white',
                   va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.axis('off')

            # Row 2: Binary (before postprocessing)
            ax = axes[base_row + 2, col + 1]
            ax.imshow(binary, cmap='gray')
            ax.text(0.02, 0.98, f"px={n_binary_pixels}",
                   transform=ax.transAxes, fontsize=7, color='white',
                   va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.axis('off')

            # Row 3: Postprocessed + watershed labeled objects
            ax = axes[base_row + 3, col + 1]
            labeled_rgb = label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))
            ax.imshow(labeled_rgb)
            ax.text(0.02, 0.98, f"n={n_objects}",
                   transform=ax.transAxes, fontsize=7, color='white',
                   va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.axis('off')

            # Row 4: Overlay
            ax = axes[base_row + 4, col + 1]
            clahe_display = np.power(img_clahe, 0.5)  # Always brighten for overlay visibility
            overlay_rgb = label2rgb(labeled, image=clahe_display, bg_label=0, alpha=0.5)
            ax.imshow(overlay_rgb)
            ax.axis('off')

    plt.suptitle(f"{title}\nFrangi: r={min_radius_um}-{max_radius_um}μm (px={pixel_size_um}), α={alpha}, β={beta} | Thresh mult={threshold_mult} | Nuclear masked + Watershed",
                fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        if VERBOSE:
            print(f"  Saved: {output_path}")

    plt.close(fig)


def compare_nucleoli_frangi_params(
    image: np.ndarray,
    nuclear_mask: np.ndarray,
    title: str = "Nucleoli - Frangi Parameter Grid",
    clahe_clip: float = 0.01,
    clahe_kernel: int = 256,
    alpha_values: list = [0.05, 0.1, 0.2, 0.5],
    radius_ranges_um: list = [(0.3, 3.0), (0.5, 3.0), (0.7, 3.0), (1.0, 3.0), (1.5, 3.0)],
    pixel_size_um: float = 0.325,
    beta: float = 0.5,
    threshold_mult: float = 0.01,
    gamma: float = 1.0,
    frangi_gamma: float = 0.3,
    output_path: str = None,
):
    """
    Compare Frangi parameters for nucleoli detection with nuclear masking.

    Uses DYNAMIC thresholding like the main script.
    Sweeps alpha values (roundness preference) and min radius (max fixed).

    Args:
        image: 2D numpy array (Phase2D channel)
        nuclear_mask: 2D numpy array (nuclear segmentation labels)
        title: Title for the figure
        clahe_clip: CLAHE clip limit
        clahe_kernel: CLAHE kernel size
        alpha_values: List of alpha values to test (lower = more round)
        radius_ranges_um: List of (min, max) radius ranges in microns
        pixel_size_um: Pixel size in microns
        beta: Fixed beta value
        threshold_mult: Multiplier for dynamic threshold (default 0.01)
        gamma: Gamma correction for display
        frangi_gamma: Gamma for Frangi response display
        output_path: Path to save figure
    """
    n_alphas = len(alpha_values)
    n_radii = len(radius_ranges_um)

    # Normalize image
    img_norm = image.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

    # Create binary mask from nuclear labels
    binary_mask = nuclear_mask > 0

    # Apply nuclear mask BEFORE CLAHE (like main script)
    masked_input = img_norm.copy()
    masked_input[~binary_mask] = 0.0

    # Apply CLAHE on masked data
    if VERBOSE:
        print(f"  Applying CLAHE (clip={clahe_clip}, kernel={clahe_kernel})...")
    img_clahe = exposure.equalize_adapthist(masked_input, clip_limit=clahe_clip, kernel_size=clahe_kernel)

    # Display image with gamma
    display_img = np.power(img_norm, gamma) if gamma != 1.0 else img_norm

    # Create figure: 5 rows per alpha (CLAHE, Frangi, Binary, Postproc, Overlay), columns = radii + 1 for labels
    n_rows = 5 * n_alphas
    fig, axes = plt.subplots(n_rows, n_radii + 1, figsize=(3.5 * (n_radii + 1), 3.5 * n_rows))

    # First column: CLAHE/masked with row labels
    for a_idx, alpha in enumerate(alpha_values):
        base_row = a_idx * 5
        for r_offset, lbl in enumerate(["CLAHE", "Frangi", "Binary", "Postproc", "Overlay"]):
            ax = axes[base_row + r_offset, 0]
            if r_offset == 0:
                ax.imshow(img_clahe, cmap='gray')
                ax.set_title(f"α={alpha}", fontsize=9)
            elif r_offset == 1:
                ax.imshow(img_clahe, cmap='gray')
            else:
                ax.imshow(display_img, cmap='gray', vmin=0, vmax=1)
            ax.text(-0.1, 0.5, lbl, fontsize=9, fontweight='bold',
                   transform=ax.transAxes, ha='right', va='center')
            ax.axis('off')

    # Process each combination
    for a_idx, alpha in enumerate(alpha_values):
        base_row = a_idx * 5

        for col, (min_r, max_r) in enumerate(radius_ranges_um):
            # Calculate sigmas
            sigmas = um_to_sigmas(min_r, max_r, pixel_size_um)

            # Apply Frangi
            img_frangi = frangi(
                img_clahe,
                sigmas=sigmas,
                alpha=alpha,
                beta=beta,
                black_ridges=False
            )

            # Use DYNAMIC thresholding like main script
            threshold = compute_frangi_threshold(img_frangi, threshold_mult=threshold_mult, xp=np)

            # Create binary mask
            binary = img_frangi > threshold
            binary = binary & binary_mask
            n_binary_pixels = binary.sum()

            # Apply postprocessing then watershed labeling
            postprocessed = postprocess_nucleoli_mask(binary.copy(), min_size=20)
            labeled = watershed_label(
                postprocessed,
                min_distance=3,
                min_object_size=20,
                compactness=1.0,
                erosion_iterations=2,
                min_peak_distance=2.0,
            )
            n_objects = labeled.max()

            # Row 0: CLAHE (show radius range in title)
            ax = axes[base_row, col + 1]
            ax.imshow(img_clahe, cmap='gray')
            ax.set_title(f"r={min_r}-{max_r}μm", fontsize=9)
            ax.axis('off')

            # Row 1: Frangi
            ax = axes[base_row + 1, col + 1]
            frangi_enhanced = np.power(img_frangi / (img_frangi.max() + 1e-8), frangi_gamma)
            ax.imshow(frangi_enhanced, cmap='inferno')
            sigma_min = min_r / pixel_size_um
            sigma_max = max_r / pixel_size_um
            ax.text(0.02, 0.98, f"σ={sigma_min:.1f}-{sigma_max:.1f}px\nth={threshold:.4f}",
                   transform=ax.transAxes, fontsize=7, color='white',
                   va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.axis('off')

            # Row 2: Binary (before postprocessing)
            ax = axes[base_row + 2, col + 1]
            ax.imshow(binary, cmap='gray')
            ax.text(0.02, 0.98, f"px={n_binary_pixels}",
                   transform=ax.transAxes, fontsize=7, color='white',
                   va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.axis('off')

            # Row 3: Postprocessed + watershed labeled objects
            ax = axes[base_row + 3, col + 1]
            labeled_rgb = label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))
            ax.imshow(labeled_rgb)
            ax.text(0.02, 0.98, f"n={n_objects}",
                   transform=ax.transAxes, fontsize=7, color='white',
                   va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.axis('off')

            # Row 4: Overlay
            ax = axes[base_row + 4, col + 1]
            clahe_display = np.power(img_clahe, 0.5)  # Always brighten for overlay visibility
            overlay_rgb = label2rgb(labeled, image=clahe_display, bg_label=0, alpha=0.5)
            ax.imshow(overlay_rgb)
            ax.axis('off')

    plt.suptitle(f"{title}\nCLAHE: clip={clahe_clip}, k={clahe_kernel} | Frangi: β={beta}, px={pixel_size_um}μm | Thresh mult={threshold_mult} | Nuclear masked + Watershed",
                fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        if VERBOSE:
            print(f"  Saved: {output_path}")

    plt.close(fig)


def compare_thresholds(
    image: np.ndarray,
    title: str = "Threshold Comparison",
    thresholds: list = [0.0001, 0.001, 0.01, 0.1],
    clahe_clip: float = 0.01,
    clahe_kernel: int = 256,
    min_radius_um: float = 0.1,
    max_radius_um: float = 1.5,
    pixel_size_um: float = 0.325,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 1.0,
    frangi_gamma: float = 0.3,
    output_path: str = None,
    nuclear_mask: np.ndarray = None,
):
    """
    Compare different threshold values for converting Frangi response to binary mask.

    Creates a single set of 4 rows:
    - Row 0: CLAHE
    - Row 1: Frangi response
    - Row 2: Labeled objects (colored)
    - Row 3: Overlay on raw

    Columns are different threshold values.

    Args:
        image: 2D numpy array
        title: Figure title
        thresholds: List of threshold values to test
        clahe_clip: CLAHE clip limit
        clahe_kernel: CLAHE kernel size
        min_radius_um: Minimum structure radius in microns
        max_radius_um: Maximum structure radius in microns
        pixel_size_um: Pixel size in microns
        alpha: Frangi alpha parameter
        beta: Frangi beta parameter
        gamma: Gamma for original image display
        frangi_gamma: Gamma for Frangi response visualization
        output_path: Path to save figure
        nuclear_mask: Optional nuclear mask for nucleoli detection
    """
    n_thresholds = len(thresholds)

    # Normalize
    img_norm = image.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

    # Handle nuclear masking for nucleoli - mask BEFORE CLAHE (like main script)
    binary_nuc_mask = None
    if nuclear_mask is not None:
        binary_nuc_mask = nuclear_mask > 0
        masked_input = img_norm.copy()
        masked_input[~binary_nuc_mask] = 0.0
        clahe_input = masked_input
    else:
        clahe_input = img_norm

    if VERBOSE:
        print(f"  Applying CLAHE (clip={clahe_clip}, kernel={clahe_kernel})...")
    img_clahe = exposure.equalize_adapthist(clahe_input, clip_limit=clahe_clip, kernel_size=clahe_kernel)

    # Display image with gamma
    display_img = np.power(img_norm, gamma) if gamma != 1.0 else img_norm

    # Calculate sigmas from microns
    sigmas = um_to_sigmas(min_radius_um, max_radius_um, pixel_size_um)

    # Apply Frangi once (same for all thresholds)
    if VERBOSE:
        print(f"  Applying Frangi (r={min_radius_um}-{max_radius_um}μm, α={alpha}, β={beta})...")
    img_frangi = frangi(
        img_clahe,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        black_ridges=False
    )

    # Determine if we need postprocessing (for nucleoli with nuclear mask)
    use_postprocess = nuclear_mask is not None

    # Create figure: 5 rows if postprocessing (CLAHE, Frangi, Binary, Postproc, Overlay), else 4 rows
    n_rows = 5 if use_postprocess else 4
    fig, axes = plt.subplots(n_rows, n_thresholds + 1, figsize=(3.5 * (n_thresholds + 1), 3 * n_rows))

    # First column: labels for each row type
    row_labels = ["CLAHE", "Frangi", "Binary", "Postproc", "Overlay"] if use_postprocess else ["CLAHE", "Frangi", "Labeled", "Overlay"]
    for r, lbl in enumerate(row_labels):
        ax = axes[r, 0]
        if r == 0:
            ax.imshow(img_clahe, cmap='gray')
            ax.set_title(f"CLAHE\nclip={clahe_clip}", fontsize=9)
        elif r == 1:
            # Show Frangi response
            frangi_enhanced = np.power(img_frangi / (img_frangi.max() + 1e-8), frangi_gamma)
            ax.imshow(frangi_enhanced, cmap='inferno')
            ax.set_title(f"max={img_frangi.max():.4f}", fontsize=8)
        else:
            ax.imshow(display_img, cmap='gray', vmin=0, vmax=1)
        ax.text(-0.1, 0.5, lbl, fontsize=9, fontweight='bold',
               transform=ax.transAxes, ha='right', va='center')
        ax.axis('off')

    # Process each threshold
    for col, thresh in enumerate(thresholds):
        # Create binary mask
        binary = img_frangi > thresh
        if binary_nuc_mask is not None:
            binary = binary & binary_nuc_mask
        n_binary_pixels = binary.sum()

        # Apply postprocessing for nucleoli (with watershed labeling like main script)
        if use_postprocess:
            postprocessed = postprocess_nucleoli_mask(binary.copy(), min_size=20)
            labeled = watershed_label(
                postprocessed,
                min_distance=3,
                min_object_size=20,
                compactness=1.0,
                erosion_iterations=2,
                min_peak_distance=2.0,
            )
        else:
            labeled = label(binary)
        n_objects = labeled.max()

        # Row 0: CLAHE
        ax = axes[0, col + 1]
        ax.imshow(img_clahe, cmap='gray')
        ax.set_title(f"thresh={thresh}", fontsize=9)
        ax.axis('off')

        # Row 1: Frangi with threshold line indicator
        ax = axes[1, col + 1]
        frangi_enhanced = np.power(img_frangi / (img_frangi.max() + 1e-8), frangi_gamma)
        ax.imshow(frangi_enhanced, cmap='inferno')
        # Show what fraction of pixels are above threshold
        frac_above = (img_frangi > thresh).sum() / img_frangi.size * 100
        ax.text(0.02, 0.98, f">{thresh:.4f}\n{frac_above:.1f}% pixels",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        if use_postprocess:
            # Row 2: Binary (before postprocessing)
            ax = axes[2, col + 1]
            ax.imshow(binary, cmap='gray')
            ax.text(0.02, 0.98, f"px={n_binary_pixels}",
                   transform=ax.transAxes, fontsize=7, color='white',
                   va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.axis('off')

            # Row 3: Postprocessed labeled objects
            ax = axes[3, col + 1]
            labeled_rgb = label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))
            ax.imshow(labeled_rgb)
            ax.text(0.02, 0.98, f"n={n_objects}",
                   transform=ax.transAxes, fontsize=7, color='white',
                   va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.axis('off')

            # Row 4: Overlay
            ax = axes[4, col + 1]
            clahe_display = np.power(img_clahe, 0.5)  # Always brighten for overlay visibility
            overlay_rgb = label2rgb(labeled, image=clahe_display, bg_label=0, alpha=0.5)
            ax.imshow(overlay_rgb)
            ax.axis('off')
        else:
            # Row 2: Labeled objects
            ax = axes[2, col + 1]
            labeled_rgb = label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))
            ax.imshow(labeled_rgb)
            ax.text(0.02, 0.98, f"n={n_objects}",
                   transform=ax.transAxes, fontsize=7, color='white',
                   va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.axis('off')

            # Row 3: Overlay
            ax = axes[3, col + 1]
            clahe_display = np.power(img_clahe, 0.5)  # Always brighten for overlay visibility
            overlay_rgb = label2rgb(labeled, image=clahe_display, bg_label=0, alpha=0.5)
            ax.imshow(overlay_rgb)
            ax.axis('off')

    mask_str = " | Nuclear masked + Watershed" if nuclear_mask is not None else ""
    plt.suptitle(f"{title}\nCLAHE: clip={clahe_clip}, k={clahe_kernel} | Frangi: r={min_radius_um}-{max_radius_um}μm, α={alpha}, β={beta}{mask_str}",
                fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        if VERBOSE:
            print(f"  Saved: {output_path}")

    plt.close(fig)


def compare_nucleoli_denoising(
    image: np.ndarray,
    nuclear_mask: np.ndarray,
    title: str = "Nucleoli - Denoising Comparison",
    methods: list = ['none', 'gaussian', 'tv', 'bilateral'],
    clahe_clip: float = 0.01,
    clahe_kernel: int = 256,
    min_radius_um: float = 0.5,
    max_radius_um: float = 3.0,
    pixel_size_um: float = 0.325,
    alpha: float = 0.1,
    beta: float = 0.5,
    threshold_mult: float = 0.01,
    gamma: float = 1.0,
    frangi_gamma: float = 0.3,
    output_path: str = None,
):
    """
    Compare different denoising methods for nucleoli detection with nuclear masking.

    Uses DYNAMIC thresholding like the main script (compute_frangi_threshold with threshold_mult).
    Shows 5 rows for each method: Raw, Frangi, Binary, Postprocessed, Overlay

    Args:
        image: 2D numpy array (Phase2D channel)
        nuclear_mask: 2D numpy array (nuclear segmentation labels)
        title: Figure title
        methods: List of denoising methods to compare
        clahe_clip: CLAHE clip limit
        clahe_kernel: CLAHE kernel size
        min_radius_um: Minimum nucleoli radius in microns
        max_radius_um: Maximum nucleoli radius in microns
        pixel_size_um: Pixel size in microns
        alpha: Frangi alpha (low = favor round structures)
        beta: Frangi beta
        threshold_mult: Multiplier for dynamic threshold (default 0.01)
        gamma: Gamma for display
        frangi_gamma: Gamma for Frangi response
        output_path: Path to save figure
    """
    from scipy.ndimage import gaussian_filter
    from skimage.restoration import denoise_tv_chambolle, denoise_bilateral

    n_methods = len(methods)

    # Normalize
    img_norm = image.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

    # Create binary mask from nuclear labels
    binary_nuc_mask = nuclear_mask > 0

    # Display image with gamma
    display_img = np.power(img_norm, gamma) if gamma != 1.0 else img_norm

    # Calculate sigmas from microns
    sigmas = um_to_sigmas(min_radius_um, max_radius_um, pixel_size_um)

    # 5 rows: CLAHE, Frangi, Binary, Postprocessed, Overlay
    fig, axes = plt.subplots(5, n_methods + 1, figsize=(3.5 * (n_methods + 1), 15))

    # Row labels in first column
    row_labels = ["CLAHE", "Frangi", "Binary", "Postproc", "Overlay"]
    for r, lbl in enumerate(row_labels):
        ax = axes[r, 0]
        if r == 1:
            # Show masked region for Frangi row
            ax.imshow(display_img * binary_nuc_mask, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(display_img, cmap='gray', vmin=0, vmax=1)
        if r == 0:
            ax.set_title(f"Original\n(γ={gamma})", fontsize=9)
        ax.text(-0.1, 0.5, lbl, fontsize=9, fontweight='bold',
               transform=ax.transAxes, ha='right', va='center')
        ax.axis('off')

    for col, method in enumerate(methods):
        # Apply denoising
        if method == 'none':
            img_denoised = img_norm
        elif method == 'gaussian':
            img_denoised = gaussian_filter(img_norm, sigma=1.0)
        elif method == 'tv':
            img_denoised = denoise_tv_chambolle(img_norm, weight=0.1)
        elif method == 'bilateral':
            img_denoised = denoise_bilateral(img_norm, sigma_color=0.05, sigma_spatial=5)
        else:
            img_denoised = img_norm

        # Apply nuclear mask BEFORE CLAHE (like main script)
        masked_input = img_denoised.copy()
        masked_input[~binary_nuc_mask] = 0.0

        # Apply CLAHE on masked data
        img_clahe = exposure.equalize_adapthist(masked_input, clip_limit=clahe_clip, kernel_size=clahe_kernel)

        # Apply Frangi with nucleoli-specific params
        img_frangi = frangi(
            img_clahe,
            sigmas=sigmas,
            alpha=alpha,
            beta=beta,
            black_ridges=False
        )

        # Use DYNAMIC thresholding like main script (Otsu+Triangle on log scale)
        threshold = compute_frangi_threshold(img_frangi, threshold_mult=threshold_mult, xp=np)

        # Create binary mask
        binary = img_frangi > threshold
        binary = binary & binary_nuc_mask
        n_binary_pixels = binary.sum()

        # Apply postprocessing then watershed labeling (like main script)
        postprocessed = postprocess_nucleoli_mask(binary.copy(), min_size=20)
        labeled = watershed_label(
            postprocessed,
            min_distance=3,
            min_object_size=20,
            compactness=1.0,
            erosion_iterations=2,
            min_peak_distance=2.0,
        )
        n_objects = labeled.max()

        # Row 0: CLAHE
        ax = axes[0, col + 1]
        ax.imshow(img_clahe, cmap='gray')
        ax.set_title(f"Denoise: {method}", fontsize=9, fontweight='bold')
        ax.axis('off')

        # Row 1: Frangi
        ax = axes[1, col + 1]
        frangi_enhanced = np.power(img_frangi / (img_frangi.max() + 1e-8), frangi_gamma)
        ax.imshow(frangi_enhanced, cmap='inferno')
        ax.text(0.02, 0.98, f"max={img_frangi.max():.3f}\nth={threshold:.4f}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 2: Binary (before postprocessing)
        ax = axes[2, col + 1]
        ax.imshow(binary, cmap='gray')
        ax.text(0.02, 0.98, f"px={n_binary_pixels}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 3: Postprocessed labeled objects
        ax = axes[3, col + 1]
        labeled_rgb = label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))
        ax.imshow(labeled_rgb)
        ax.text(0.02, 0.98, f"n={n_objects}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 4: Overlay
        ax = axes[4, col + 1]
        clahe_display = np.power(img_clahe, 0.5)  # Always brighten for overlay visibility
        overlay_rgb = label2rgb(labeled, image=clahe_display, bg_label=0, alpha=0.5)
        ax.imshow(overlay_rgb)
        ax.axis('off')

    plt.suptitle(f"{title}\nCLAHE: clip={clahe_clip}, k={clahe_kernel} | Frangi: r={min_radius_um}-{max_radius_um}μm, α={alpha} | Thresh mult={threshold_mult} | Nuclear masked + Watershed",
                fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        if VERBOSE:
            print(f"  Saved: {output_path}")

    plt.close(fig)


def compare_nucleoli_pixel_sizes(
    image: np.ndarray,
    nuclear_mask: np.ndarray,
    title: str = "Nucleoli - Pixel Size Comparison",
    clahe_clip: float = 0.01,
    clahe_kernel: int = 256,
    pixel_sizes_um: list = [0.08125, 0.1625, 0.325, 0.65],
    min_radius_um: float = 0.5,
    max_radius_um: float = 3.0,
    alpha: float = 0.1,
    beta: float = 0.5,
    threshold_mult: float = 0.01,
    gamma: float = 1.0,
    frangi_gamma: float = 0.3,
    output_path: str = None,
):
    """
    Compare effect of different pixel size assumptions on nucleoli detection.

    Uses DYNAMIC thresholding like the main script (compute_frangi_threshold with threshold_mult).
    Creates 5 rows: CLAHE, Frangi, Binary, Postprocessed, Overlay

    Args:
        image: 2D numpy array (Phase2D channel)
        nuclear_mask: 2D numpy array (nuclear segmentation labels)
        title: Figure title
        clahe_clip: Fixed CLAHE clip limit
        clahe_kernel: Fixed CLAHE kernel size
        pixel_sizes_um: List of pixel sizes in microns to test
        min_radius_um: Minimum nucleoli radius in microns
        max_radius_um: Maximum nucleoli radius in microns
        alpha: Frangi alpha (low = favor round structures)
        beta: Frangi beta
        threshold_mult: Multiplier for dynamic threshold (default 0.01)
        gamma: Gamma for display
        frangi_gamma: Gamma for Frangi response
        output_path: Path to save figure
    """
    n_px_sizes = len(pixel_sizes_um)

    # Normalize
    img_norm = image.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

    # Create binary mask from nuclear labels
    binary_nuc_mask = nuclear_mask > 0

    # Apply nuclear mask BEFORE CLAHE (like main script)
    masked_input = img_norm.copy()
    masked_input[~binary_nuc_mask] = 0.0

    if VERBOSE:
        print(f"  Applying CLAHE (clip={clahe_clip}, kernel={clahe_kernel})...")
    img_clahe = exposure.equalize_adapthist(masked_input, clip_limit=clahe_clip, kernel_size=clahe_kernel)

    # Display image with gamma
    display_img = np.power(img_norm, gamma) if gamma != 1.0 else img_norm

    # 5 rows: CLAHE, Frangi, Binary, Postprocessed, Overlay
    fig, axes = plt.subplots(5, n_px_sizes + 1, figsize=(3.5 * (n_px_sizes + 1), 15))

    # First column: labels for each row type
    row_labels = ["CLAHE", "Frangi", "Binary", "Postproc", "Overlay"]
    for r, lbl in enumerate(row_labels):
        ax = axes[r, 0]
        if r == 0:
            ax.imshow(img_clahe, cmap='gray')
            ax.set_title(f"CLAHE\nclip={clahe_clip}", fontsize=9)
        elif r == 1:
            ax.imshow(img_clahe, cmap='gray')
        else:
            ax.imshow(display_img, cmap='gray', vmin=0, vmax=1)
        ax.text(-0.1, 0.5, lbl, fontsize=9, fontweight='bold',
               transform=ax.transAxes, ha='right', va='center')
        ax.axis('off')

    # Process each pixel size
    for col, px_size in enumerate(pixel_sizes_um):
        # Apply Frangi with adaptive sigmas based on this pixel size
        sigmas = um_to_sigmas(min_radius_um, max_radius_um, px_size)
        img_frangi = frangi(
            img_clahe,
            sigmas=sigmas,
            alpha=alpha,
            beta=beta,
            black_ridges=False
        )

        # Use DYNAMIC thresholding like main script (Otsu+Triangle on log scale)
        threshold = compute_frangi_threshold(img_frangi, threshold_mult=threshold_mult, xp=np)

        # Create binary mask
        binary = img_frangi > threshold
        binary = binary & binary_nuc_mask
        n_binary_pixels = binary.sum()

        # Apply postprocessing then watershed labeling (like main script)
        postprocessed = postprocess_nucleoli_mask(binary.copy(), min_size=20)
        labeled = watershed_label(
            postprocessed,
            min_distance=3,
            min_object_size=20,
            compactness=1.0,
            erosion_iterations=2,
            min_peak_distance=2.0,
        )
        n_objects = labeled.max()

        # Show sigma range
        sigma_min = min_radius_um / px_size
        sigma_max = max_radius_um / px_size

        # Row 0: CLAHE
        ax = axes[0, col + 1]
        ax.imshow(img_clahe, cmap='gray')
        ax.set_title(f"px={px_size}μm", fontsize=9)
        ax.axis('off')

        # Row 1: Frangi
        ax = axes[1, col + 1]
        frangi_enhanced = np.power(img_frangi / (img_frangi.max() + 1e-8), frangi_gamma)
        ax.imshow(frangi_enhanced, cmap='inferno')
        ax.text(0.02, 0.98, f"σ={sigma_min:.1f}-{sigma_max:.1f}px\nth={threshold:.4f}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 2: Binary (before postprocessing)
        ax = axes[2, col + 1]
        ax.imshow(binary, cmap='gray')
        ax.text(0.02, 0.98, f"px={n_binary_pixels}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 3: Postprocessed labeled objects
        ax = axes[3, col + 1]
        labeled_rgb = label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))
        ax.imshow(labeled_rgb)
        ax.text(0.02, 0.98, f"n={n_objects}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 4: Overlay
        ax = axes[4, col + 1]
        clahe_display = np.power(img_clahe, 0.5)  # Always brighten for overlay visibility
        overlay_rgb = label2rgb(labeled, image=clahe_display, bg_label=0, alpha=0.5)
        ax.imshow(overlay_rgb)
        ax.axis('off')

    plt.suptitle(f"{title}\nCLAHE: clip={clahe_clip}, k={clahe_kernel} | Frangi: r={min_radius_um}-{max_radius_um}μm, α={alpha} | Thresh mult={threshold_mult} | Nuclear masked + Watershed",
                fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        if VERBOSE:
            print(f"  Saved: {output_path}")

    plt.close(fig)


def compare_denoising(
    image: np.ndarray,
    title: str = "Denoising Comparison",
    methods: list = ['none', 'gaussian', 'tv', 'bilateral'],
    clahe_clip: float = 0.01,
    clahe_kernel: int = 256,
    min_radius_um: float = 0.1,
    max_radius_um: float = 1.5,
    pixel_size_um: float = 0.325,
    threshold: float = None,
    threshold_mult: float = 0.01,
    gamma: float = 1.0,
    frangi_gamma: float = 0.3,
    output_path: str = None,
):
    """
    Compare different denoising methods before CLAHE + Frangi.

    Shows 4 rows for each method: Raw, Frangi, Labeled, Overlay

    Args:
        image: 2D numpy array
        title: Figure title
        methods: List of denoising methods to compare
        clahe_clip: CLAHE clip limit
        clahe_kernel: CLAHE kernel size
        min_radius_um: Minimum structure radius in microns
        max_radius_um: Maximum structure radius in microns
        pixel_size_um: Pixel size in microns
        threshold: Fixed threshold for Frangi. If None, uses dynamic thresholding.
        threshold_mult: Multiplier for dynamic threshold (only used if threshold is None)
        gamma: Gamma for display
        frangi_gamma: Gamma for Frangi response (< 1 brightens dim structures)
        output_path: Path to save figure
    """
    from scipy.ndimage import gaussian_filter
    from skimage.restoration import denoise_tv_chambolle, denoise_bilateral

    n_methods = len(methods)

    # Normalize
    img_norm = image.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

    # Display image with gamma
    display_img = np.power(img_norm, gamma) if gamma != 1.0 else img_norm

    # Calculate sigmas from microns
    sigmas = um_to_sigmas(min_radius_um, max_radius_um, pixel_size_um)

    # 4 rows: Raw/CLAHE, Frangi, Labeled, Overlay - add extra column for row labels
    fig, axes = plt.subplots(4, n_methods + 1, figsize=(3.5 * (n_methods + 1), 12))

    # Row labels in first column
    row_labels = ["CLAHE", "Frangi", "Labeled", "Overlay"]
    for r, lbl in enumerate(row_labels):
        ax = axes[r, 0]
        ax.imshow(display_img, cmap='gray', vmin=0, vmax=1)
        if r == 0:
            ax.set_title(f"Original\n(γ={gamma})", fontsize=9)
        # Add row label as text on the left side (ylabel doesn't work with axis off)
        ax.text(-0.1, 0.5, lbl, fontsize=9, fontweight='bold',
               transform=ax.transAxes, ha='right', va='center')
        ax.axis('off')

    for col, method in enumerate(methods):
        # Apply denoising
        if method == 'none':
            img_denoised = img_norm
        elif method == 'gaussian':
            img_denoised = gaussian_filter(img_norm, sigma=1.0)
        elif method == 'tv':
            img_denoised = denoise_tv_chambolle(img_norm, weight=0.1)
        elif method == 'bilateral':
            img_denoised = denoise_bilateral(img_norm, sigma_color=0.05, sigma_spatial=5)
        else:
            img_denoised = img_norm

        # Apply CLAHE
        img_clahe = exposure.equalize_adapthist(img_denoised, clip_limit=clahe_clip, kernel_size=clahe_kernel)

        # Apply Frangi with adaptive sigmas
        img_frangi = frangi(img_clahe, sigmas=sigmas, black_ridges=False)

        # Create labeled mask - use fixed or dynamic threshold
        if threshold is not None:
            thresh_val = threshold
        else:
            thresh_val = compute_frangi_threshold(img_frangi, threshold_mult=threshold_mult, xp=np)
        binary = img_frangi > thresh_val
        labeled = label(binary)
        n_objects = labeled.max()

        # Row 0: Denoised/CLAHE
        ax = axes[0, col + 1]
        ax.imshow(img_clahe, cmap='gray')
        ax.set_title(f"Denoise: {method}", fontsize=9, fontweight='bold')
        ax.axis('off')

        # Row 1: Frangi
        ax = axes[1, col + 1]
        frangi_enhanced = np.power(img_frangi / (img_frangi.max() + 1e-8), frangi_gamma)
        ax.imshow(frangi_enhanced, cmap='inferno')
        ax.text(0.02, 0.98, f"max={img_frangi.max():.3f}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 2: Labeled objects
        ax = axes[2, col + 1]
        labeled_rgb = label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))
        ax.imshow(labeled_rgb)
        ax.text(0.02, 0.98, f"n={n_objects}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 3: Overlay (use CLAHE image with gamma for better contrast)
        ax = axes[3, col + 1]
        clahe_display = np.power(img_clahe, 0.5)  # Always brighten for overlay visibility
        overlay_rgb = label2rgb(labeled, image=clahe_display, bg_label=0, alpha=0.5)
        ax.imshow(overlay_rgb)
        ax.axis('off')

    thresh_str = f"fixed={threshold}" if threshold is not None else f"dynamic(mult={threshold_mult})"
    plt.suptitle(f"{title}\nCLAHE: clip={clahe_clip}, k={clahe_kernel} | Frangi: r={min_radius_um}-{max_radius_um}μm (px={pixel_size_um}), α=0.5, β=0.5 | Thresh: {thresh_str} | Frangi γ={frangi_gamma}",
                fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        if VERBOSE:
            print(f"  Saved: {output_path}")

    plt.close(fig)


def test_single_setting(
    image: np.ndarray,
    title: str = "Single Setting Test",
    clip_limit: float = 0.01,
    kernel_size: int = 256,
    min_radius_um: float = 0.1,
    max_radius_um: float = 1.5,
    pixel_size_um: float = 0.325,
    threshold: float = 0.01,
    output_path: str = None,
):
    """
    Test a single CLAHE + Frangi setting with detailed debug output.

    This function tests EXACTLY the settings used in the main organelle_segmentation.py
    script for fluorescent channels, allowing direct comparison of pipeline behavior.

    Args:
        image: 2D numpy array (grayscale)
        title: Title for the figure
        clip_limit: CLAHE clip limit (main script default: 0.01)
        kernel_size: CLAHE kernel size (main script default: 256)
        min_radius_um: Min structure radius in microns (main script default: 0.1)
        max_radius_um: Max structure radius in microns (main script default: 1.5)
        pixel_size_um: Pixel size in microns (main script default: 0.325)
        threshold: Fixed threshold for Frangi (main script default: 0.01)
        output_path: Path to save figure (optional)
    """
    print(f"\n{'='*60}")
    print(f"SINGLE SETTING DEBUG (matching main script)")
    print(f"{'='*60}")

    # Step 1: Input
    print(f"\n[STEP 1] INPUT:")
    print(f"  dtype={image.dtype}, shape={image.shape}")
    print(f"  range=[{image.min():.4f}, {image.max():.4f}]")

    # Step 2: Normalize to [0, 1]
    img_norm = image.astype(np.float32)
    img_min, img_max = img_norm.min(), img_norm.max()
    if img_max > img_min:
        img_norm = (img_norm - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img_norm)

    print(f"\n[STEP 2] NORMALIZE TO [0,1]:")
    print(f"  dtype={img_norm.dtype}")
    print(f"  range=[{img_norm.min():.4f}, {img_norm.max():.4f}]")

    # Step 3: CLAHE
    print(f"\n[STEP 3] CLAHE (clip={clip_limit}, kernel={kernel_size}):")
    img_clahe = exposure.equalize_adapthist(
        img_norm,
        clip_limit=clip_limit,
        kernel_size=kernel_size
    )
    print(f"  dtype={img_clahe.dtype}")
    print(f"  range=[{img_clahe.min():.4f}, {img_clahe.max():.4f}]")

    # Step 4: No smoothing (main script default is now 0.0)
    print(f"\n[STEP 4] SMOOTHING: NONE (sigma=0)")

    # Step 5: Frangi
    sigmas = um_to_sigmas(min_radius_um, max_radius_um, pixel_size_um)
    print(f"\n[STEP 5] FRANGI:")
    print(f"  radius: {min_radius_um}-{max_radius_um}um")
    print(f"  pixel_size: {pixel_size_um}um")
    print(f"  sigmas (px): {sigmas}")
    print(f"  input dtype={img_clahe.dtype}, range=[{img_clahe.min():.4f}, {img_clahe.max():.4f}]")

    img_frangi = frangi(
        img_clahe,
        sigmas=sigmas,
        black_ridges=False
    )
    print(f"  output dtype={img_frangi.dtype}")
    print(f"  output range=[{img_frangi.min():.6f}, {img_frangi.max():.6f}]")

    # Step 6: Threshold
    print(f"\n[STEP 6] THRESHOLD:")
    print(f"  fixed threshold={threshold}")
    binary = img_frangi > threshold
    labeled = label(binary)
    n_objects = labeled.max()
    print(f"  binary pixels: {binary.sum()}")
    print(f"  num objects: {n_objects}")

    print(f"\n{'='*60}")
    print(f"DONE - Found {n_objects} objects")
    print(f"{'='*60}\n")

    # Create 2x3 figure with high contrast overlay
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Raw normalized, CLAHE, CLAHE high contrast
    axes[0, 0].imshow(img_norm, cmap='gray')
    axes[0, 0].set_title(f"Normalized [0,1]")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_clahe, cmap='gray')
    axes[0, 1].set_title(f"CLAHE (clip={clip_limit}, k={kernel_size})")
    axes[0, 1].axis('off')

    # High contrast CLAHE (gamma=0.3 for very bright)
    clahe_high_contrast = np.power(img_clahe, 0.3)
    axes[0, 2].imshow(clahe_high_contrast, cmap='gray')
    axes[0, 2].set_title(f"CLAHE High Contrast (γ=0.3)")
    axes[0, 2].axis('off')

    # Row 2: Frangi, Labeled overlay, High contrast overlay
    frangi_display = np.power(img_frangi / (img_frangi.max() + 1e-8), 0.3)
    axes[1, 0].imshow(frangi_display, cmap='inferno')
    axes[1, 0].set_title(f"Frangi (max={img_frangi.max():.4f})")
    axes[1, 0].axis('off')

    # Labeled overlay (normal contrast)
    clahe_display = np.power(img_clahe, 0.5)
    overlay_rgb = label2rgb(labeled, image=clahe_display, bg_label=0, alpha=0.5)
    axes[1, 1].imshow(overlay_rgb)
    axes[1, 1].set_title(f"Overlay (n={n_objects}, thresh={threshold})")
    axes[1, 1].axis('off')

    # High contrast overlay (gamma=0.3)
    overlay_high_contrast = label2rgb(labeled, image=clahe_high_contrast, bg_label=0, alpha=0.5)
    axes[1, 2].imshow(overlay_high_contrast)
    axes[1, 2].set_title(f"High Contrast Overlay (γ=0.3)")
    axes[1, 2].axis('off')

    plt.suptitle(f"{title}\nSettings: clip={clip_limit}, k={kernel_size}, r={min_radius_um}-{max_radius_um}um, px={pixel_size_um}, thresh={threshold}",
                fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        if VERBOSE:
            print(f"  Saved: {output_path}")

    plt.close(fig)

    return {
        'frangi_max': img_frangi.max(),
        'frangi_min': img_frangi.min(),
        'binary_pixels': binary.sum(),
        'n_objects': n_objects,
    }


def main():
    parser = argparse.ArgumentParser(description="Test CLAHE + Frangi pipeline parameters")

    parser.add_argument("--experiment", "-e", type=str, required=True,
                       help="Experiment name or number")
    parser.add_argument("--position", "-p", type=str, default="A/1/0",
                       help="Position path (default: A/1/0)")
    parser.add_argument("--crop-size", type=int, default=512,
                       help="Size of crop to test (default: 512)")
    parser.add_argument("--offset-y", type=int, default=28000,
                       help="Y offset from center (default: 28000 to match preview mode)")
    parser.add_argument("--offset-x", type=int, default=28000,
                       help="X offset from center (default: 28000 to match preview mode)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                       help="Output directory for figures (default: experiment/3-assembly/clahe_frangi_test)")
    parser.add_argument("--gamma", type=float, default=0.25,
                       help="Gamma for fluorescence display (default: 0.25)")
    parser.add_argument("--frangi-gamma", type=float, default=0.3,
                       help="Gamma for Frangi response display - lower values brighten dim structures (default: 0.3)")
    parser.add_argument("--threshold", "-t", type=float, default=0.01,
                       help="Threshold for converting Frangi to binary mask (default: 0.01)")
    parser.add_argument("--pixel-size", type=float, default=0.325,
                       help="Pixel size in microns (default: 0.325)")
    parser.add_argument("--min-radius", type=float, default=0.1,
                       help="Minimum structure radius in microns (default: 0.1)")
    parser.add_argument("--max-radius", type=float, default=1.5,
                       help="Maximum structure radius in microns (default: 1.5)")
    parser.add_argument("--nucleoli-only", action="store_true",
                       help="Only run nucleoli tests (skip labelfree and fluorescent)")
    parser.add_argument("--skip-nucleoli", action="store_true",
                       help="Skip nucleoli tests (run only labelfree and fluorescent)")
    parser.add_argument("--single-test", action="store_true",
                       help="Run single-setting test with debug output (matches main script settings)")
    parser.add_argument("--channel", "-c", type=str, default=None,
                       help="Specific channel to test (for --single-test mode)")
    parser.add_argument("--bbox", type=str, default=None,
                       help="Exact crop bbox as 'y_start,y_end,x_start,x_end' (e.g., '79394,81314,79657,81577')")
    parser.add_argument("--kernel-size", "-k", type=int, default=256,
                       help="CLAHE kernel size (default: 256). Try larger values for larger images.")
    parser.add_argument("--clip-limit", type=float, default=0.01,
                       help="CLAHE clip limit (default: 0.01)")
    parser.add_argument("--display-size", "-d", type=int, default=None,
                       help="Size of center crop for display (default: same as crop-size). "
                            "Use 512 to display 512x512 center while processing on full crop-size.")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose debug output (default: False)")

    args = parser.parse_args()

    # Set global verbose flag
    global VERBOSE
    VERBOSE = args.verbose

    # Default display-size to crop-size if not specified
    if args.display_size is None:
        args.display_size = args.crop_size

    # Resolve experiment
    experiment = resolve_experiment_name(args.experiment, allow_interactive=True)
    if experiment is None:
        print("No experiment selected. Exiting.")
        return

    print(f"\n{'='*60}")
    print(f"CLAHE + Frangi Parameter Testing")
    print(f"{'='*60}")
    print(f"Experiment: {experiment}")
    print(f"Position: {args.position}")
    print(f"Process size: {args.crop_size}x{args.crop_size}")
    print(f"Display size: {args.display_size}x{args.display_size}")
    print(f"Offset: ({args.offset_y}, {args.offset_x})")
    print(f"{'='*60}\n")

    # Get zarr path
    dataset = OpsDataset(experiment)
    zarr_path = str(dataset.store_paths["pheno_assembled_v3"])

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(zarr_path).parent / "clahe_frangi_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Get available channels
    with open_ome_zarr(zarr_path, mode="r") as ds:
        channel_names = list(ds.channel_names)

    print(f"Available channels: {channel_names}\n")

    # Define channels to test
    labelfree_channels = [ch for ch in channel_names if ch in ['Phase2D', 'Focus3D']]
    fluorescent_channels = [ch for ch in channel_names if ch in ['GFP', 'mCherry', 'Cy5', 'FarRed']]

    offset = (args.offset_y, args.offset_x) if args.offset_y or args.offset_x else None

    # Parse bbox if provided
    bbox = None
    if args.bbox:
        try:
            parts = [int(x.strip()) for x in args.bbox.split(',')]
            if len(parts) == 4:
                bbox = tuple(parts)  # (y_start, y_end, x_start, x_end)
                print(f"Using exact bbox: Y[{bbox[0]}:{bbox[1]}], X[{bbox[2]}:{bbox[3]}]")
            else:
                print(f"ERROR: bbox must have 4 values, got {len(parts)}")
                return
        except ValueError as e:
            print(f"ERROR: Invalid bbox format: {e}")
            return

    # Handle --single-test mode: run just one setting with full debug output
    if args.single_test:
        # Determine which channels to test
        if args.channel:
            channels_to_test = [args.channel] if args.channel in channel_names else []
            if not channels_to_test:
                print(f"ERROR: Channel '{args.channel}' not found. Available: {channel_names}")
                return
        else:
            # Default: test first fluorescent channel if available, else first channel
            channels_to_test = fluorescent_channels[:1] if fluorescent_channels else channel_names[:1]

        for ch in channels_to_test:
            print(f"\n--- Single Setting Test: {ch} ---")
            img = load_crop(zarr_path, args.position, ch, args.crop_size, offset, bbox=bbox)
            test_single_setting(
                img,
                title=f"{ch} - Single Setting Test",
                clip_limit=args.clip_limit,
                kernel_size=args.kernel_size,
                min_radius_um=args.min_radius,
                max_radius_um=args.max_radius,
                pixel_size_um=args.pixel_size,
                threshold=args.threshold,
                output_path=output_dir / f"{ch}_single_test_k{args.kernel_size}.png"
            )
        print("\n--- Single test complete ---")
        return

    # If --channel is specified, only run that channel (works for all modes)
    if args.channel:
        if args.channel in labelfree_channels:
            labelfree_channels = [args.channel]
            fluorescent_channels = []
        elif args.channel in fluorescent_channels:
            labelfree_channels = []
            fluorescent_channels = [args.channel]
        elif args.channel in channel_names:
            # Channel exists but not in our predefined lists - treat as fluorescent
            labelfree_channels = []
            fluorescent_channels = [args.channel]
        else:
            print(f"ERROR: Channel '{args.channel}' not found. Available: {channel_names}")
            return

    # Test label-free channels (skip if --nucleoli-only)
    for ch in labelfree_channels:
        if args.nucleoli_only:
            print(f"\n--- Skipping {ch} (--nucleoli-only) ---")
            continue
        print(f"\n--- Testing {ch} (label-free) ---")
        try:
            img = load_crop(zarr_path, args.position, ch, args.crop_size, offset)

            # Test 1: CLAHE clip limits and kernel sizes
            if VERBOSE:
                print(f"  Test 1: CLAHE parameters grid...")
            compare_clahe_frangi(
                img,
                title=f"{ch} - CLAHE Parameter Grid",
                clip_limits=[0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
                kernel_sizes=[32, 64, 128, 256],
                min_radius_um=0.2,
                max_radius_um=args.max_radius,
                pixel_size_um=args.pixel_size,
                threshold=args.threshold,
                gamma=1.0,
                frangi_gamma=args.frangi_gamma,
                output_path=output_dir / f"{ch}_clahe_grid.png"
            )

            # Test 2: Frangi parameters (sweep beta and min radius, max fixed at 1.5)
            if VERBOSE:
                print(f"  Test 2: Frangi parameters grid...")
            compare_frangi_params(
                img,
                title=f"{ch} - Frangi Parameter Grid",
                clahe_clip=0.01,
                clahe_kernel=256,
                beta_values=[0.1, 0.3, 0.5, 1.0],
                radius_ranges_um=[(0.05, 1.5), (0.1, 1.5), (0.2, 1.5), (0.3, 1.5), (0.4, 1.5)],
                pixel_size_um=args.pixel_size,
                threshold=args.threshold,
                gamma=1.0,
                frangi_gamma=args.frangi_gamma,
                output_path=output_dir / f"{ch}_frangi_grid.png"
            )

            # Test 3: Denoising methods
            if VERBOSE:
                print(f"  Test 3: Denoising comparison...")
            compare_denoising(
                img,
                title=f"{ch} - Denoising Comparison",
                clahe_clip=0.01,
                clahe_kernel=256,
                min_radius_um=0.2,
                max_radius_um=args.max_radius,
                pixel_size_um=args.pixel_size,
                threshold=args.threshold,
                gamma=1.0,
                frangi_gamma=args.frangi_gamma,
                output_path=output_dir / f"{ch}_denoise_comparison.png"
            )

            # Test 4: Pixel size sweep (effect on sigma calculation)
            if VERBOSE:
                print(f"  Test 4: Pixel size comparison...")
            compare_pixel_sizes(
                img,
                title=f"{ch} - Pixel Size Grid",
                clahe_clip=0.01,
                clahe_kernel=256,
                pixel_sizes_um=[0.1625, 0.2, 0.225, 0.25, 0.275, 0.325],
                min_radius_um=0.2,
                max_radius_um=args.max_radius,
                threshold=args.threshold,
                gamma=1.0,
                frangi_gamma=args.frangi_gamma,
                output_path=output_dir / f"{ch}_pixel_size_grid.png"
            )

            # Test 5: Threshold sweep
            if VERBOSE:
                print(f"  Test 5: Threshold comparison...")
            compare_thresholds(
                img,
                title=f"{ch} - Threshold Comparison",
                thresholds=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
                clahe_clip=0.01,
                clahe_kernel=256,
                min_radius_um=0.2,
                max_radius_um=args.max_radius,
                pixel_size_um=args.pixel_size,
                alpha=0.5,
                beta=0.5,
                gamma=1.0,
                frangi_gamma=args.frangi_gamma,
                output_path=output_dir / f"{ch}_threshold_comparison.png"
            )

            # ===== VESICLE BLOB DETECTION TESTS (COMMENTED OUT) =====
            # Uncomment to test blob detection for vesicular structures
            # # Test bright vesicles using LoG blob detection (default params from organelle_segmentation.py)
            # print(f"\n  --- Vesicle Blob Detection (Bright) ---")
            #
            # # Test 6: Blob threshold sweep (bright vesicles) - 3 values around default 0.05
            # print(f"  Test 6: Blob threshold comparison (bright)...")
            # compare_blob_thresholds(
            #     img,
            #     title=f"{ch} - Bright Vesicle Blob Threshold",
            #     thresholds=[0.05, 0.06, 0.07],
            #     min_radius_um=VESICLE_BLOB_CONFIG["min_radius_um"],
            #     max_radius_um=VESICLE_BLOB_CONFIG["max_radius_um"],
            #     pixel_size_um=args.pixel_size,
            #     num_sigma=VESICLE_BLOB_CONFIG["num_sigma"],
            #     overlap=VESICLE_BLOB_CONFIG["overlap"],
            #     invert=False,  # Bright blobs
            #     clahe_clip=0.01,
            #     clahe_kernel=32,  # Small kernel for vesicular
            #     gamma=1.0,
            #     output_path=output_dir / f"{ch}_blob_bright_threshold.png"
            # )
            #
            # # Test 7: Blob radius sweep (bright vesicles)
            # print(f"  Test 7: Blob radius comparison (bright)...")
            # compare_blob_radius(
            #     img,
            #     title=f"{ch} - Bright Vesicle Blob Radius",
            #     radius_ranges_um=[(0.1, 0.4), (0.2, 0.4), (0.2, 0.5), (0.2, 0.3)],
            #     threshold=VESICLE_BLOB_CONFIG["threshold"],
            #     pixel_size_um=args.pixel_size,
            #     num_sigma=VESICLE_BLOB_CONFIG["num_sigma"],
            #     overlap=VESICLE_BLOB_CONFIG["overlap"],
            #     invert=False,
            #     clahe_clip=0.01,
            #     clahe_kernel=32,
            #     gamma=1.0,
            #     output_path=output_dir / f"{ch}_blob_bright_radius.png"
            # )
            #
            # # Test dark vesicles (inverted - dark blobs on bright background)
            # print(f"\n  --- Vesicle Blob Detection (Dark) ---")
            #
            # # Test 8: Blob threshold sweep (dark vesicles) - 3 values around default 0.05
            # print(f"  Test 8: Blob threshold comparison (dark)...")
            # compare_blob_thresholds(
            #     img,
            #     title=f"{ch} - Dark Vesicle Blob Threshold",
            #     thresholds=[0.05, 0.06, 0.07],
            #     min_radius_um=VESICLE_BLOB_CONFIG["min_radius_um"],
            #     max_radius_um=VESICLE_BLOB_CONFIG["max_radius_um"],
            #     pixel_size_um=args.pixel_size,
            #     num_sigma=VESICLE_BLOB_CONFIG["num_sigma"],
            #     overlap=VESICLE_BLOB_CONFIG["overlap"],
            #     invert=True,  # Dark blobs
            #     clahe_clip=0.01,
            #     clahe_kernel=32,
            #     gamma=1.0,
            #     output_path=output_dir / f"{ch}_blob_dark_threshold.png"
            # )
            #
            # # Test 9: Blob radius sweep (dark vesicles)
            # print(f"  Test 9: Blob radius comparison (dark)...")
            # compare_blob_radius(
            #     img,
            #     title=f"{ch} - Dark Vesicle Blob Radius",
            #     radius_ranges_um=[(0.1, 0.4), (0.2, 0.4), (0.2, 0.5), (0.2, 0.3)],
            #     threshold=VESICLE_BLOB_CONFIG["threshold"],
            #     pixel_size_um=args.pixel_size,
            #     num_sigma=VESICLE_BLOB_CONFIG["num_sigma"],
            #     overlap=VESICLE_BLOB_CONFIG["overlap"],
            #     invert=True,
            #     clahe_clip=0.01,
            #     clahe_kernel=32,
            #     gamma=1.0,
            #     output_path=output_dir / f"{ch}_blob_dark_radius.png"
            # )

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Test fluorescent channels (skip if --nucleoli-only)
    for ch in fluorescent_channels:
        if args.nucleoli_only:
            print(f"\n--- Skipping {ch} (--nucleoli-only) ---")
            continue
        print(f"\n--- Testing {ch} (fluorescent) ---")
        try:
            img = load_crop(zarr_path, args.position, ch, args.crop_size, offset)

            # Test 1: CLAHE parameters (use gamma for display)
            # Use larger kernel sizes for bigger crops (4096 tiles need 512-2048 kernels)
            if VERBOSE:
                print(f"  Test 1: CLAHE parameters grid...")
            compare_clahe_frangi(
                img,
                title=f"{ch} - CLAHE Parameter Grid (proc={args.crop_size}, disp={args.display_size})",
                clip_limits=[0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
                kernel_sizes=[256, 512, 1024, 2048],
                min_radius_um=args.min_radius,
                max_radius_um=args.max_radius,
                pixel_size_um=args.pixel_size,
                threshold=args.threshold,
                gamma=args.gamma,
                frangi_gamma=args.frangi_gamma,
                output_path=output_dir / f"{ch}_clahe_grid.png",
                display_size=args.display_size,
            )

            # Test 2: Frangi parameters (sweep beta and min radius, max fixed at 1.5)
            if VERBOSE:
                print(f"  Test 2: Frangi parameters grid...")
            compare_frangi_params(
                img,
                title=f"{ch} - Frangi Parameter Grid",
                clahe_clip=0.01,
                clahe_kernel=256,
                beta_values=[0.1, 0.3, 0.5, 1.0],
                radius_ranges_um=[(0.05, 1.5), (0.1, 1.5), (0.2, 1.5), (0.3, 1.5), (0.4, 1.5)],
                pixel_size_um=args.pixel_size,
                threshold=args.threshold,
                gamma=args.gamma,
                frangi_gamma=args.frangi_gamma,
                output_path=output_dir / f"{ch}_frangi_grid.png"
            )

            # Test 3: Denoising methods
            if VERBOSE:
                print(f"  Test 3: Denoising comparison...")
            compare_denoising(
                img,
                title=f"{ch} - Denoising Comparison",
                clahe_clip=0.01,
                clahe_kernel=256,
                min_radius_um=args.min_radius,
                max_radius_um=args.max_radius,
                pixel_size_um=args.pixel_size,
                threshold=args.threshold,
                gamma=args.gamma,
                frangi_gamma=args.frangi_gamma,
                output_path=output_dir / f"{ch}_denoise_comparison.png"
            )

            # Test 4: Pixel size sweep (effect on sigma calculation)
            if VERBOSE:
                print(f"  Test 4: Pixel size comparison...")
            compare_pixel_sizes(
                img,
                title=f"{ch} - Pixel Size Grid",
                clahe_clip=0.01,
                clahe_kernel=256,
                pixel_sizes_um=[0.1625, 0.2, 0.225, 0.25, 0.275, 0.325],
                min_radius_um=args.min_radius,
                max_radius_um=args.max_radius,
                threshold=args.threshold,
                gamma=args.gamma,
                frangi_gamma=args.frangi_gamma,
                output_path=output_dir / f"{ch}_pixel_size_grid.png"
            )

            # Test 5: Threshold sweep
            if VERBOSE:
                print(f"  Test 5: Threshold comparison...")
            compare_thresholds(
                img,
                title=f"{ch} - Threshold Comparison",
                thresholds=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
                clahe_clip=0.01,
                clahe_kernel=256,
                min_radius_um=args.min_radius,
                max_radius_um=args.max_radius,
                pixel_size_um=args.pixel_size,
                alpha=0.5,
                beta=0.5,
                gamma=args.gamma,
                frangi_gamma=args.frangi_gamma,
                output_path=output_dir / f"{ch}_threshold_comparison.png"
            )

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Test nucleoli (Phase2D with nuclear mask) - skip if --skip-nucleoli
    # Nucleoli detection requires masking to the nuclear interior
    if 'Phase2D' in channel_names and not args.skip_nucleoli:
        print(f"\n--- Testing nucleoli (Phase2D + nuclear mask) ---")
        try:
            # Load Phase2D image
            img = load_crop(zarr_path, args.position, 'Phase2D', args.crop_size, offset)

            # Try to load nuclear mask
            nuclear_mask = None
            for mask_name in ['nucle_vs_seg', 'nuclear_seg', 'nuclei_seg']:
                try:
                    nuclear_mask = load_mask_crop(zarr_path, args.position, mask_name, args.crop_size, offset)
                    print(f"  Using nuclear mask: {mask_name}")
                    break
                except ValueError:
                    continue

            if nuclear_mask is None:
                print(f"  WARNING: No nuclear mask found, skipping nucleoli tests")
            else:
                # Nucleoli-specific parameters (larger structures, lower alpha for roundness)
                nucleoli_min_radius = 0.5  # μm
                nucleoli_max_radius = 3.0  # μm
                nucleoli_alpha = 0.1       # Low alpha favors round structures

                # Test 1: CLAHE parameters with nuclear masking
                if VERBOSE:
                print(f"  Test 1: CLAHE parameters grid (nuclear masked)...")
                compare_nucleoli_clahe_frangi(
                    img,
                    nuclear_mask,
                    title=f"Nucleoli - CLAHE Parameter Grid",
                    clip_limits=[0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
                    kernel_sizes=[64, 128, 256, 512],
                    min_radius_um=nucleoli_min_radius,
                    max_radius_um=nucleoli_max_radius,
                    pixel_size_um=args.pixel_size,
                    alpha=nucleoli_alpha,
                    beta=0.5,
                    threshold_mult=args.threshold,  # Uses dynamic thresholding
                    gamma=1.0,
                    frangi_gamma=args.frangi_gamma,
                    output_path=output_dir / "nucleoli_clahe_grid.png"
                )

                # Test 2: Frangi alpha and radius sweep (nucleoli-specific)
                if VERBOSE:
                print(f"  Test 2: Frangi parameters grid (alpha + radius sweep)...")
                compare_nucleoli_frangi_params(
                    img,
                    nuclear_mask,
                    title=f"Nucleoli - Frangi Parameter Grid",
                    clahe_clip=0.01,
                    clahe_kernel=256,
                    alpha_values=[0.05, 0.1, 0.2, 0.5],
                    radius_ranges_um=[(0.3, 3.0), (0.5, 3.0), (0.7, 3.0), (1.0, 3.0), (1.5, 3.0)],
                    pixel_size_um=args.pixel_size,
                    beta=0.5,
                    threshold_mult=args.threshold,  # Uses dynamic thresholding
                    gamma=1.0,
                    frangi_gamma=args.frangi_gamma,
                    output_path=output_dir / "nucleoli_frangi_grid.png"
                )

                # Test 3: Threshold sweep (with nuclear masking)
                if VERBOSE:
                print(f"  Test 3: Threshold comparison (nuclear masked)...")
                compare_thresholds(
                    img,
                    title=f"Nucleoli - Threshold Comparison",
                    thresholds=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
                    clahe_clip=0.01,
                    clahe_kernel=256,
                    min_radius_um=nucleoli_min_radius,
                    max_radius_um=nucleoli_max_radius,
                    pixel_size_um=args.pixel_size,
                    alpha=nucleoli_alpha,
                    beta=0.5,
                    gamma=1.0,
                    frangi_gamma=args.frangi_gamma,
                    output_path=output_dir / "nucleoli_threshold_comparison.png",
                    nuclear_mask=nuclear_mask,
                )

                # Test 4: Denoising comparison (with nuclear masking + postprocessing)
                if VERBOSE:
                print(f"  Test 4: Denoising comparison (nuclear masked + watershed)...")
                compare_nucleoli_denoising(
                    img,
                    nuclear_mask,
                    title=f"Nucleoli - Denoising Comparison",
                    clahe_clip=0.01,
                    clahe_kernel=256,
                    min_radius_um=nucleoli_min_radius,
                    max_radius_um=nucleoli_max_radius,
                    pixel_size_um=args.pixel_size,
                    alpha=nucleoli_alpha,
                    beta=0.5,
                    threshold_mult=args.threshold,  # Uses dynamic thresholding
                    gamma=1.0,
                    frangi_gamma=args.frangi_gamma,
                    output_path=output_dir / "nucleoli_denoise_comparison.png"
                )

                # Test 5: Pixel size comparison (with nuclear masking + postprocessing)
                if VERBOSE:
                print(f"  Test 5: Pixel size comparison (nuclear masked + watershed)...")
                compare_nucleoli_pixel_sizes(
                    img,
                    nuclear_mask,
                    title=f"Nucleoli - Pixel Size Comparison",
                    clahe_clip=0.01,
                    clahe_kernel=256,
                    pixel_sizes_um=[0.1625, 0.2, 0.225, 0.25, 0.275, 0.325],
                    min_radius_um=nucleoli_min_radius,
                    max_radius_um=nucleoli_max_radius,
                    alpha=nucleoli_alpha,
                    beta=0.5,
                    threshold_mult=args.threshold,  # Uses dynamic thresholding
                    gamma=1.0,
                    frangi_gamma=args.frangi_gamma,
                    output_path=output_dir / "nucleoli_pixel_size_grid.png"
                )

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Testing complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
