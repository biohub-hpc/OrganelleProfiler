

# --- nellie imports ---
import numpy as np

# --- end nellie imports ---


def otsu_effectiveness(image, inter_variance, xp):
    # flatten image and create histogram
    flattened_image = image.flatten()
    sigma_total_squared = xp.var(flattened_image)
    normalized_sigma_B_squared = inter_variance / sigma_total_squared
    return normalized_sigma_B_squared


def otsu_threshold(matrix, nbins=256, xp=np):
    # gpu version of skimage.filters.threshold_otsu
    counts, bin_edges = xp.histogram(
        matrix.reshape(-1), bins=nbins, range=(matrix.min(), matrix.max())
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Ensure counts is not empty and sum is not zero
    if xp.sum(counts) == 0:
        return bin_centers[0], 0  # Or handle as an error

    counts = counts / xp.sum(counts)

    weight1 = xp.cumsum(counts)
    mean1 = xp.cumsum(counts * bin_centers) / xp.where(
        weight1 > 0, weight1, 1
    )  # Avoid division by zero

    weight2 = xp.cumsum(counts[::-1])[::-1]
    mean2 = (
        xp.cumsum((counts * bin_centers)[::-1])
        / xp.where(weight2 > 0, weight2, 1)[::-1]
    )[::-1]

    # Prevent issues with empty slices
    if len(weight1) < 2 or len(weight2) < 2:
        return bin_centers[0], 0

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Handle case where variance is all zero
    if xp.max(variance12) == 0:
        return bin_centers[0], 0

    idx = xp.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold, variance12[idx]


def triangle_threshold(matrix, nbins=256, xp=np):
    # gpu version of skimage.filters.threshold_triangle
    hist, bin_edges = xp.histogram(
        matrix.reshape(-1), bins=nbins, range=(xp.min(matrix), xp.max(matrix))
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    if xp.sum(hist) == 0:
        return bin_centers[0]

    hist = hist / xp.sum(hist)

    arg_peak_height = xp.argmax(hist)
    peak_height = hist[arg_peak_height]

    try:
        non_zero_indices = xp.flatnonzero(hist)
        if len(non_zero_indices) < 2:
            return bin_centers[0]
        arg_low_level, arg_high_level = non_zero_indices[[0, -1]]
    except IndexError:
        return bin_centers[0]

    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        hist = xp.flip(hist, axis=0)
        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1
    del arg_high_level

    width = arg_peak_height - arg_low_level
    if width <= 0:
        return bin_centers[arg_low_level]

    x1 = xp.arange(width)
    y1 = hist[x1 + arg_low_level]

    norm = xp.sqrt(peak_height**2 + width**2)
    peak_height = peak_height / norm
    width = width / norm

    length = peak_height * x1 - width * y1
    arg_level = xp.argmax(length) + arg_low_level

    if flip:
        arg_level = nbins - arg_level - 1

    return bin_centers[arg_level]
