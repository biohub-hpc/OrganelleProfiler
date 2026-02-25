"""
Test script for comparing LoG blob detection parameters for vesicular structures.

This module provides functions to sweep blob detection parameters for:
- Vesicles (bright blobs on dark background)
- Dark vesicles (dark blobs on bright background)

These are used for vesicular structure detection in label-free images.

Usage:
    # Import and use in test_clahe_frangi.py
    from organelle_profiler.feature_extraction.test_blob_detection import (
        compare_blob_thresholds,
        compare_blob_radius,
    )
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.feature import blob_log
from skimage.draw import disk
from skimage.measure import label
from skimage.color import label2rgb


# Default blob detection configs from organelle_segmentation.py
VESICLE_BLOB_CONFIG = {
    "min_radius_um": 0.2,    # Keep small vesicles
    "max_radius_um": 0.4,    # Tighter max to exclude large circles
    "num_sigma": 4,          # Narrow scale range for small vesicles only
    "threshold": 0.05,       # Higher threshold for cleaner detection (was 0.03)
    "overlap": 0.3,          # Lower overlap = fewer merged blobs
    "exclude_border": False, # Include border blobs
}


def detect_blobs(
    image: np.ndarray,
    min_radius_um: float,
    max_radius_um: float,
    pixel_size_um: float,
    threshold: float = 0.03,
    num_sigma: int = 4,
    overlap: float = 0.3,
    invert: bool = False,
) -> tuple:
    """
    Run LoG blob detection on an image.

    Args:
        image: 2D numpy array (normalized to 0-1)
        min_radius_um: Minimum blob radius in microns
        max_radius_um: Maximum blob radius in microns
        pixel_size_um: Pixel size in microns
        threshold: Detection threshold (lower = more sensitive)
        num_sigma: Number of sigma values to use
        overlap: Maximum overlap between blobs (0-1)
        invert: If True, detect dark blobs on bright background

    Returns:
        Tuple of (labeled_mask, n_blobs, blobs_array)
    """
    # Normalize image
    img_float = image.astype(np.float32)
    vmin, vmax = np.percentile(img_float, [1, 99])
    img_norm = np.clip((img_float - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # Invert for dark blob detection
    if invert:
        img_norm = 1.0 - img_norm

    # Calculate sigma range in pixels (sigma ≈ radius / sqrt(2) for LoG)
    min_sigma = min_radius_um / pixel_size_um / np.sqrt(2)
    max_sigma = max_radius_um / pixel_size_um / np.sqrt(2)

    # Ensure reasonable sigma values
    min_sigma = max(1.0, min_sigma)
    max_sigma = max(min_sigma + 1, max_sigma)

    # Run LoG blob detection
    try:
        blobs = blob_log(
            img_norm,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=overlap,
        )
    except Exception as e:
        print(f"    Blob detection error: {e}")
        return np.zeros(image.shape, dtype=np.int32), 0, np.array([])

    if len(blobs) == 0:
        return np.zeros(image.shape, dtype=np.int32), 0, blobs

    # Convert blobs to labeled mask
    # blobs is Nx3 array: (y, x, sigma)
    labels = np.zeros(image.shape, dtype=np.int32)
    for i, (y, x, sigma) in enumerate(blobs):
        # Radius is sigma * sqrt(2) for LoG
        radius = int(sigma * np.sqrt(2))
        rr, cc = disk((int(y), int(x)), radius, shape=image.shape)
        labels[rr, cc] = i + 1

    # Re-label to handle overlapping disks
    labels = label(labels > 0).astype(np.int32)
    n_blobs = labels.max()

    return labels, n_blobs, blobs


def compare_blob_thresholds(
    image: np.ndarray,
    title: str = "Blob Detection - Threshold Comparison",
    thresholds: list = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1],
    min_radius_um: float = 0.2,
    max_radius_um: float = 0.4,
    pixel_size_um: float = 0.325,
    num_sigma: int = 4,
    overlap: float = 0.3,
    invert: bool = False,
    clahe_clip: float = 0.01,
    clahe_kernel: int = 32,
    gamma: float = 1.0,
    output_path: str = None,
):
    """
    Compare different threshold values for LoG blob detection.

    Creates a grid showing for each threshold:
    - Row 0: CLAHE image
    - Row 1: Detected blobs (circles)
    - Row 2: Labeled mask
    - Row 3: Overlay

    Args:
        image: 2D numpy array
        title: Figure title
        thresholds: List of threshold values to test
        min_radius_um: Minimum blob radius in microns
        max_radius_um: Maximum blob radius in microns
        pixel_size_um: Pixel size in microns
        num_sigma: Number of sigma values
        overlap: Maximum blob overlap
        invert: If True, detect dark blobs
        clahe_clip: CLAHE clip limit
        clahe_kernel: CLAHE kernel size
        gamma: Display gamma
        output_path: Path to save figure
    """
    n_thresholds = len(thresholds)

    # Normalize and apply CLAHE
    img_norm = image.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

    print(f"  Applying CLAHE (clip={clahe_clip}, kernel={clahe_kernel})...")
    img_clahe = exposure.equalize_adapthist(img_norm, clip_limit=clahe_clip, kernel_size=clahe_kernel)

    # Display image with gamma
    display_img = np.power(img_clahe, 0.5)  # Always brighten for visibility

    # Create figure
    fig, axes = plt.subplots(4, n_thresholds + 1, figsize=(3.5 * (n_thresholds + 1), 12))

    # First column: labels
    row_labels = ["CLAHE", "Blobs", "Labeled", "Overlay"]
    for r, lbl in enumerate(row_labels):
        ax = axes[r, 0]
        ax.imshow(display_img, cmap='gray')
        if r == 0:
            ax.set_title(f"CLAHE\nclip={clahe_clip}", fontsize=9)
        ax.text(-0.1, 0.5, lbl, fontsize=9, fontweight='bold',
               transform=ax.transAxes, ha='right', va='center')
        ax.axis('off')

    # Process each threshold
    for col, thresh in enumerate(thresholds):
        print(f"    Testing threshold={thresh}...")

        # Detect blobs
        labeled, n_blobs, blobs = detect_blobs(
            img_clahe,
            min_radius_um=min_radius_um,
            max_radius_um=max_radius_um,
            pixel_size_um=pixel_size_um,
            threshold=thresh,
            num_sigma=num_sigma,
            overlap=overlap,
            invert=invert,
        )

        # Row 0: CLAHE
        ax = axes[0, col + 1]
        ax.imshow(display_img, cmap='gray')
        ax.set_title(f"thresh={thresh}", fontsize=9)
        ax.axis('off')

        # Row 1: Show detected blob circles on image
        ax = axes[1, col + 1]
        ax.imshow(display_img, cmap='gray')
        if len(blobs) > 0:
            for y, x, sigma in blobs:
                radius = sigma * np.sqrt(2)
                circle = plt.Circle((x, y), radius, color='red', fill=False, linewidth=0.5)
                ax.add_patch(circle)
        ax.text(0.02, 0.98, f"n={n_blobs}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 2: Labeled mask
        ax = axes[2, col + 1]
        labeled_rgb = label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))
        ax.imshow(labeled_rgb)
        ax.text(0.02, 0.98, f"n={n_blobs}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 3: Overlay
        ax = axes[3, col + 1]
        overlay_rgb = label2rgb(labeled, image=display_img, bg_label=0, alpha=0.5)
        ax.imshow(overlay_rgb)
        ax.axis('off')

    blob_type = "dark" if invert else "bright"
    plt.suptitle(f"{title}\nLoG Blob ({blob_type}): r={min_radius_um}-{max_radius_um}μm, num_σ={num_sigma}, overlap={overlap} | CLAHE: clip={clahe_clip}",
                fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")

    plt.close(fig)


def compare_blob_radius(
    image: np.ndarray,
    title: str = "Blob Detection - Radius Comparison",
    radius_ranges_um: list = [(0.1, 0.3), (0.2, 0.4), (0.3, 0.5), (0.4, 0.6), (0.5, 1.0)],
    threshold: float = 0.03,
    pixel_size_um: float = 0.325,
    num_sigma: int = 4,
    overlap: float = 0.3,
    invert: bool = False,
    clahe_clip: float = 0.01,
    clahe_kernel: int = 32,
    gamma: float = 1.0,
    output_path: str = None,
):
    """
    Compare different radius ranges for LoG blob detection.

    Args:
        image: 2D numpy array
        title: Figure title
        radius_ranges_um: List of (min, max) radius ranges in microns
        threshold: Detection threshold
        pixel_size_um: Pixel size in microns
        num_sigma: Number of sigma values
        overlap: Maximum blob overlap
        invert: If True, detect dark blobs
        clahe_clip: CLAHE clip limit
        clahe_kernel: CLAHE kernel size
        gamma: Display gamma
        output_path: Path to save figure
    """
    n_ranges = len(radius_ranges_um)

    # Normalize and apply CLAHE
    img_norm = image.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

    print(f"  Applying CLAHE (clip={clahe_clip}, kernel={clahe_kernel})...")
    img_clahe = exposure.equalize_adapthist(img_norm, clip_limit=clahe_clip, kernel_size=clahe_kernel)

    # Display image with gamma
    display_img = np.power(img_clahe, 0.5)  # Always brighten for visibility

    # Create figure
    fig, axes = plt.subplots(4, n_ranges + 1, figsize=(3.5 * (n_ranges + 1), 12))

    # First column: labels
    row_labels = ["CLAHE", "Blobs", "Labeled", "Overlay"]
    for r, lbl in enumerate(row_labels):
        ax = axes[r, 0]
        ax.imshow(display_img, cmap='gray')
        if r == 0:
            ax.set_title(f"CLAHE\nclip={clahe_clip}", fontsize=9)
        ax.text(-0.1, 0.5, lbl, fontsize=9, fontweight='bold',
               transform=ax.transAxes, ha='right', va='center')
        ax.axis('off')

    # Process each radius range
    for col, (min_r, max_r) in enumerate(radius_ranges_um):
        print(f"    Testing radius={min_r}-{max_r}μm...")

        # Detect blobs
        labeled, n_blobs, blobs = detect_blobs(
            img_clahe,
            min_radius_um=min_r,
            max_radius_um=max_r,
            pixel_size_um=pixel_size_um,
            threshold=threshold,
            num_sigma=num_sigma,
            overlap=overlap,
            invert=invert,
        )

        # Calculate sigma range for display
        min_sigma = min_r / pixel_size_um / np.sqrt(2)
        max_sigma = max_r / pixel_size_um / np.sqrt(2)

        # Row 0: CLAHE
        ax = axes[0, col + 1]
        ax.imshow(display_img, cmap='gray')
        ax.set_title(f"r={min_r}-{max_r}μm", fontsize=9)
        ax.axis('off')

        # Row 1: Show detected blob circles on image
        ax = axes[1, col + 1]
        ax.imshow(display_img, cmap='gray')
        if len(blobs) > 0:
            for y, x, sigma in blobs:
                radius = sigma * np.sqrt(2)
                circle = plt.Circle((x, y), radius, color='red', fill=False, linewidth=0.5)
                ax.add_patch(circle)
        ax.text(0.02, 0.98, f"σ={min_sigma:.1f}-{max_sigma:.1f}px\nn={n_blobs}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 2: Labeled mask
        ax = axes[2, col + 1]
        labeled_rgb = label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))
        ax.imshow(labeled_rgb)
        ax.text(0.02, 0.98, f"n={n_blobs}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 3: Overlay
        ax = axes[3, col + 1]
        overlay_rgb = label2rgb(labeled, image=display_img, bg_label=0, alpha=0.5)
        ax.imshow(overlay_rgb)
        ax.axis('off')

    blob_type = "dark" if invert else "bright"
    plt.suptitle(f"{title}\nLoG Blob ({blob_type}): thresh={threshold}, num_σ={num_sigma}, overlap={overlap} | CLAHE: clip={clahe_clip}",
                fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")

    plt.close(fig)


def compare_blob_num_sigma(
    image: np.ndarray,
    title: str = "Blob Detection - Num Sigma Comparison",
    num_sigmas: list = [2, 4, 6, 8, 10, 12],
    min_radius_um: float = 0.2,
    max_radius_um: float = 0.4,
    threshold: float = 0.03,
    pixel_size_um: float = 0.325,
    overlap: float = 0.3,
    invert: bool = False,
    clahe_clip: float = 0.01,
    clahe_kernel: int = 32,
    gamma: float = 1.0,
    output_path: str = None,
):
    """
    Compare different num_sigma values for LoG blob detection.

    More sigma values = finer scale sampling = potentially more accurate detection
    but slower computation.

    Args:
        image: 2D numpy array
        title: Figure title
        num_sigmas: List of num_sigma values to test
        min_radius_um: Minimum blob radius in microns
        max_radius_um: Maximum blob radius in microns
        threshold: Detection threshold
        pixel_size_um: Pixel size in microns
        overlap: Maximum blob overlap
        invert: If True, detect dark blobs
        clahe_clip: CLAHE clip limit
        clahe_kernel: CLAHE kernel size
        gamma: Display gamma
        output_path: Path to save figure
    """
    n_vals = len(num_sigmas)

    # Normalize and apply CLAHE
    img_norm = image.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

    print(f"  Applying CLAHE (clip={clahe_clip}, kernel={clahe_kernel})...")
    img_clahe = exposure.equalize_adapthist(img_norm, clip_limit=clahe_clip, kernel_size=clahe_kernel)

    # Display image with gamma
    display_img = np.power(img_clahe, 0.5)  # Always brighten for visibility

    # Create figure
    fig, axes = plt.subplots(4, n_vals + 1, figsize=(3.5 * (n_vals + 1), 12))

    # First column: labels
    row_labels = ["CLAHE", "Blobs", "Labeled", "Overlay"]
    for r, lbl in enumerate(row_labels):
        ax = axes[r, 0]
        ax.imshow(display_img, cmap='gray')
        if r == 0:
            ax.set_title(f"CLAHE\nclip={clahe_clip}", fontsize=9)
        ax.text(-0.1, 0.5, lbl, fontsize=9, fontweight='bold',
               transform=ax.transAxes, ha='right', va='center')
        ax.axis('off')

    # Process each num_sigma value
    for col, num_sigma in enumerate(num_sigmas):
        print(f"    Testing num_sigma={num_sigma}...")

        # Detect blobs
        labeled, n_blobs, blobs = detect_blobs(
            img_clahe,
            min_radius_um=min_radius_um,
            max_radius_um=max_radius_um,
            pixel_size_um=pixel_size_um,
            threshold=threshold,
            num_sigma=num_sigma,
            overlap=overlap,
            invert=invert,
        )

        # Row 0: CLAHE
        ax = axes[0, col + 1]
        ax.imshow(display_img, cmap='gray')
        ax.set_title(f"num_σ={num_sigma}", fontsize=9)
        ax.axis('off')

        # Row 1: Show detected blob circles on image
        ax = axes[1, col + 1]
        ax.imshow(display_img, cmap='gray')
        if len(blobs) > 0:
            for y, x, sigma in blobs:
                radius = sigma * np.sqrt(2)
                circle = plt.Circle((x, y), radius, color='red', fill=False, linewidth=0.5)
                ax.add_patch(circle)
        ax.text(0.02, 0.98, f"n={n_blobs}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 2: Labeled mask
        ax = axes[2, col + 1]
        labeled_rgb = label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))
        ax.imshow(labeled_rgb)
        ax.text(0.02, 0.98, f"n={n_blobs}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 3: Overlay
        ax = axes[3, col + 1]
        overlay_rgb = label2rgb(labeled, image=display_img, bg_label=0, alpha=0.5)
        ax.imshow(overlay_rgb)
        ax.axis('off')

    blob_type = "dark" if invert else "bright"
    plt.suptitle(f"{title}\nLoG Blob ({blob_type}): r={min_radius_um}-{max_radius_um}μm, thresh={threshold}, overlap={overlap} | CLAHE: clip={clahe_clip}",
                fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")

    plt.close(fig)


def compare_blob_overlap(
    image: np.ndarray,
    title: str = "Blob Detection - Overlap Comparison",
    overlaps: list = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    min_radius_um: float = 0.2,
    max_radius_um: float = 0.4,
    threshold: float = 0.03,
    pixel_size_um: float = 0.325,
    num_sigma: int = 4,
    invert: bool = False,
    clahe_clip: float = 0.01,
    clahe_kernel: int = 32,
    gamma: float = 1.0,
    output_path: str = None,
):
    """
    Compare different overlap values for LoG blob detection.

    Lower overlap = stricter non-maximum suppression = fewer overlapping blobs.
    Higher overlap = allows more overlapping detections.

    Args:
        image: 2D numpy array
        title: Figure title
        overlaps: List of overlap values to test (0-1)
        min_radius_um: Minimum blob radius in microns
        max_radius_um: Maximum blob radius in microns
        threshold: Detection threshold
        pixel_size_um: Pixel size in microns
        num_sigma: Number of sigma values
        invert: If True, detect dark blobs
        clahe_clip: CLAHE clip limit
        clahe_kernel: CLAHE kernel size
        gamma: Display gamma
        output_path: Path to save figure
    """
    n_vals = len(overlaps)

    # Normalize and apply CLAHE
    img_norm = image.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)

    print(f"  Applying CLAHE (clip={clahe_clip}, kernel={clahe_kernel})...")
    img_clahe = exposure.equalize_adapthist(img_norm, clip_limit=clahe_clip, kernel_size=clahe_kernel)

    # Display image with gamma
    display_img = np.power(img_clahe, 0.5)  # Always brighten for visibility

    # Create figure
    fig, axes = plt.subplots(4, n_vals + 1, figsize=(3.5 * (n_vals + 1), 12))

    # First column: labels
    row_labels = ["CLAHE", "Blobs", "Labeled", "Overlay"]
    for r, lbl in enumerate(row_labels):
        ax = axes[r, 0]
        ax.imshow(display_img, cmap='gray')
        if r == 0:
            ax.set_title(f"CLAHE\nclip={clahe_clip}", fontsize=9)
        ax.text(-0.1, 0.5, lbl, fontsize=9, fontweight='bold',
               transform=ax.transAxes, ha='right', va='center')
        ax.axis('off')

    # Process each overlap value
    for col, overlap in enumerate(overlaps):
        print(f"    Testing overlap={overlap}...")

        # Detect blobs
        labeled, n_blobs, blobs = detect_blobs(
            img_clahe,
            min_radius_um=min_radius_um,
            max_radius_um=max_radius_um,
            pixel_size_um=pixel_size_um,
            threshold=threshold,
            num_sigma=num_sigma,
            overlap=overlap,
            invert=invert,
        )

        # Row 0: CLAHE
        ax = axes[0, col + 1]
        ax.imshow(display_img, cmap='gray')
        ax.set_title(f"overlap={overlap}", fontsize=9)
        ax.axis('off')

        # Row 1: Show detected blob circles on image
        ax = axes[1, col + 1]
        ax.imshow(display_img, cmap='gray')
        if len(blobs) > 0:
            for y, x, sigma in blobs:
                radius = sigma * np.sqrt(2)
                circle = plt.Circle((x, y), radius, color='red', fill=False, linewidth=0.5)
                ax.add_patch(circle)
        ax.text(0.02, 0.98, f"n={n_blobs}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 2: Labeled mask
        ax = axes[2, col + 1]
        labeled_rgb = label2rgb(labeled, bg_label=0, bg_color=(0, 0, 0))
        ax.imshow(labeled_rgb)
        ax.text(0.02, 0.98, f"n={n_blobs}",
               transform=ax.transAxes, fontsize=7, color='white',
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')

        # Row 3: Overlay
        ax = axes[3, col + 1]
        overlay_rgb = label2rgb(labeled, image=display_img, bg_label=0, alpha=0.5)
        ax.imshow(overlay_rgb)
        ax.axis('off')

    blob_type = "dark" if invert else "bright"
    plt.suptitle(f"{title}\nLoG Blob ({blob_type}): r={min_radius_um}-{max_radius_um}μm, thresh={threshold}, num_σ={num_sigma} | CLAHE: clip={clahe_clip}",
                fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")

    plt.close(fig)
