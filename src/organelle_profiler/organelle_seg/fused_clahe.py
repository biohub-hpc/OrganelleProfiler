"""Fused CLAHE implementation — drop-in for cucim's equalize_adapthist.

Two-kernel pipeline matching the hand-off doc's design
(memory/project_organelle_seg_clahe_kernel.md):

  Kernel 1: per-block CDF tensor (n_ty, n_tx, nbins) built from input (H, W).
            One CUDA block per image tile; shared-memory histogram →
            clip/redistribute → parallel prefix sum → global write.

  Kernel 2: output pixel ← bilinear blend of the 4 neighboring tile CDFs
            looked up at the pixel's intensity.

Target: drop-in replacement for cucim.skimage.exposure.equalize_adapthist
inside `_compute_tile_batch_on_gpu_streams` in organelle_seg/tiled_processing.py,
gated by ORG_SEG_FUSED_CLAHE=1.

Correctness bar (Tier 2, per CLAUDE.md):
  Pearson correlation ≥ 0.975 vs cucim on real 4096² Phase2D tiles.
  max-abs-diff < 0.05.

Speedup target: 1.3-1.5x over cucim in isolation. Bigger under 2-worker
contention (we're HBM-bandwidth-bound at steady state, 57% of Pass 1 time).
"""
from __future__ import annotations

import cupy as cp
import numpy as np


# ============================================================================
# Kernel 1 — per-block CDF tensor
# ============================================================================
# One CUDA block per 256×256 image tile. 256 threads per block.
# Shared memory: histogram[256] int32 (1 KB) + prefix-sum scratch.
#
# Algorithm per block (ty, tx):
#   1. Zero shared histogram (threads 0..255 each zero 1 bin)
#   2. Iterate pixels in this tile (256*256 / 256 = 256 pixels per thread).
#      For each pixel: bin = int(clamp(img * (nbins-1))). atomicAdd(hist[bin], 1).
#   3. __syncthreads(). Clip excess:
#      total_excess = sum over bins of max(hist[bin] - clip_count, 0)
#      hist[bin] = min(hist[bin], clip_count)  + (total_excess / nbins)
#      Any leftover from non-integer division can be ignored (skimage does this)
#   4. Parallel prefix sum on the clipped histogram to build CDF.
#   5. Normalize CDF by tile_size (make it [0, 1]).
#   6. Write (n_ty, n_tx, nbins) int/fp32 tensor.

_KERNEL1_SRC = r"""
extern "C" __global__ void
build_tile_cdfs(
    const float * __restrict__ img,
    float       * __restrict__ cdfs,     // out: (n_ty, n_tx, nbins)
    const int H, const int W,
    const int kh, const int kw,
    const int nbins,
    const float clip_count,              // clip_limit * tile_size, in pixel-count units
    const int n_ty, const int n_tx)
{
    // One CUDA block per tile. blockIdx.y = ty, blockIdx.x = tx.
    // Thread layout: blockDim.x == nbins == 256.
    const int ty = blockIdx.y;
    const int tx = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ int smem[];
    int * hist = smem;                   // nbins ints
    int * scan_tmp = smem + nbins;       // nbins ints for prefix sum

    // 1. zero histogram (one thread per bin)
    if (tid < nbins) hist[tid] = 0;
    __syncthreads();

    // 2. per-tile histogram
    const int y0 = ty * kh;
    const int x0 = tx * kw;
    const int tile_pixels = kh * kw;
    // Each thread handles `tile_pixels / nbins` pixels (256 for 256² tiles)
    for (int p = tid; p < tile_pixels; p += blockDim.x) {
        const int ly = p / kw;
        const int lx = p - ly * kw;
        const int y = y0 + ly;
        const int x = x0 + lx;
        // Clamp to image bounds (edge tiles may be partial, but we call
        // only on H/W divisible by kh/kw in v1; add bounds check for safety).
        if (y < H && x < W) {
            float v = img[y * W + x];
            // clamp to [0, 1] and bin
            v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
            int bin = (int)(v * (float)(nbins - 1) + 0.5f);
            if (bin < 0) bin = 0;
            if (bin >= nbins) bin = nbins - 1;
            atomicAdd(&hist[bin], 1);
        }
    }
    __syncthreads();

    // 3. Clip + redistribute (single pass, matching skimage default)
    //    Compute per-thread excess, then a block-wide reduction.
    int clip_int = (int)(clip_count + 0.5f);
    int my_excess = 0;
    if (tid < nbins) {
        int h = hist[tid];
        if (h > clip_int) {
            my_excess = h - clip_int;
            hist[tid] = clip_int;
        }
    }
    __syncthreads();
    // Reduce my_excess across threads using scan_tmp
    if (tid < nbins) scan_tmp[tid] = my_excess;
    __syncthreads();
    // Block-wide sum via tree reduction (assumes nbins is power-of-2)
    for (int stride = nbins / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scan_tmp[tid] += scan_tmp[tid + stride];
        __syncthreads();
    }
    int total_excess = scan_tmp[0];
    int redistribute = total_excess / nbins;   // integer floor
    if (tid < nbins) hist[tid] += redistribute;
    __syncthreads();

    // 4. Parallel prefix sum (inclusive) to build CDF counts
    //    Blelloch-style scan on nbins=256 with one thread per element.
    if (tid < nbins) scan_tmp[tid] = hist[tid];
    __syncthreads();
    for (int offset = 1; offset < nbins; offset *= 2) {
        int v = 0;
        if (tid >= offset && tid < nbins) v = scan_tmp[tid - offset];
        __syncthreads();
        if (tid < nbins) scan_tmp[tid] += v;
        __syncthreads();
    }
    // scan_tmp[tid] now holds inclusive prefix sum (cumulative count)

    // 5. Normalize to [0, 1] and write out
    const float norm = 1.0f / (float)tile_pixels;
    if (tid < nbins) {
        int out_idx = (ty * n_tx + tx) * nbins + tid;
        cdfs[out_idx] = (float)scan_tmp[tid] * norm;
    }
}
"""


# ============================================================================
# Kernel 2 — bilinear CDF interpolation
# ============================================================================
# For each output pixel (y, x), identify the 4 neighboring tile CDFs and
# bilinear-blend their CDF values at the pixel's binned intensity.
#
# Tile CENTERS are at (ty + 0.5) * kh, (tx + 0.5) * kw. A pixel at (y, x)
# maps to fractional tile coord (ty_f = y/kh - 0.5, tx_f = x/kw - 0.5).
# The 4 neighbors are at (floor(ty_f), floor(tx_f)) ..
# (floor(ty_f)+1, floor(tx_f)+1). At boundaries we clamp to valid tile range.

_KERNEL2_SRC = r"""
extern "C" __global__ void
interpolate_cdfs(
    const float * __restrict__ img,     // (H, W)
    const float * __restrict__ cdfs,    // (n_ty, n_tx, nbins)
    float       * __restrict__ out,     // (H, W)
    const int H, const int W,
    const int kh, const int kw,
    const int nbins, const int n_ty, const int n_tx)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    // Fractional tile coordinate (center of pixel → tile-center distance)
    float ty_f = (float)y / (float)kh - 0.5f;
    float tx_f = (float)x / (float)kw - 0.5f;

    int ty0 = (int)floorf(ty_f);
    int tx0 = (int)floorf(tx_f);
    float fy = ty_f - (float)ty0;
    float fx = tx_f - (float)tx0;

    int ty1 = ty0 + 1;
    int tx1 = tx0 + 1;

    // Clamp to valid tile range (replicate at boundaries)
    if (ty0 < 0) ty0 = 0;
    if (tx0 < 0) tx0 = 0;
    if (ty1 < 0) ty1 = 0;
    if (tx1 < 0) tx1 = 0;
    if (ty0 >= n_ty) ty0 = n_ty - 1;
    if (tx0 >= n_tx) tx0 = n_tx - 1;
    if (ty1 >= n_ty) ty1 = n_ty - 1;
    if (tx1 >= n_tx) tx1 = n_tx - 1;

    // Look up CDF bin for this pixel's intensity
    float v = img[y * W + x];
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    int bin = (int)(v * (float)(nbins - 1) + 0.5f);
    if (bin < 0) bin = 0;
    if (bin >= nbins) bin = nbins - 1;

    const float c00 = cdfs[(ty0 * n_tx + tx0) * nbins + bin];
    const float c01 = cdfs[(ty0 * n_tx + tx1) * nbins + bin];
    const float c10 = cdfs[(ty1 * n_tx + tx0) * nbins + bin];
    const float c11 = cdfs[(ty1 * n_tx + tx1) * nbins + bin];

    const float c0 = c00 * (1.0f - fx) + c01 * fx;
    const float c1 = c10 * (1.0f - fx) + c11 * fx;
    const float c  = c0  * (1.0f - fy) + c1  * fy;

    out[y * W + x] = c;
}
"""


_kernel1 = None
_kernel2 = None


def _get_kernels():
    global _kernel1, _kernel2
    if _kernel1 is None:
        _kernel1 = cp.RawKernel(_KERNEL1_SRC, "build_tile_cdfs")
    if _kernel2 is None:
        _kernel2 = cp.RawKernel(_KERNEL2_SRC, "interpolate_cdfs")
    return _kernel1, _kernel2


def fused_clahe(
    image: cp.ndarray,
    kernel_size: tuple = (256, 256),
    clip_limit: float = 0.01,
    nbins: int = 256,
) -> cp.ndarray:
    """Fused CLAHE. Drop-in replacement for
    ``cucim.skimage.exposure.equalize_adapthist`` on fp32 2D inputs in [0, 1].

    Requires H % kh == 0 and W % kw == 0 in v1. Falls back by raising on
    bad inputs so the caller can route to cucim.
    """
    if image.dtype != cp.float32:
        image = image.astype(cp.float32)
    if image.ndim != 2:
        raise ValueError(f"fused_clahe: expected 2D input, got shape {image.shape}")

    H_orig, W_orig = image.shape
    kh, kw = kernel_size
    if nbins != 256:
        raise ValueError(f"fused_clahe: nbins must be 256 in v1, got {nbins}")

    # Pad to a multiple of kernel_size via edge replication (matches the
    # cucim/skimage equalize_adapthist semantics for non-divisible inputs).
    pad_h = (kh - H_orig % kh) % kh
    pad_w = (kw - W_orig % kw) % kw
    if pad_h or pad_w:
        image = cp.pad(image, ((0, pad_h), (0, pad_w)), mode="edge")
    H, W = image.shape

    n_ty = H // kh
    n_tx = W // kw
    tile_pixels = kh * kw
    clip_count = float(clip_limit) * float(tile_pixels)

    cdfs = cp.empty((n_ty, n_tx, nbins), dtype=cp.float32)
    out = cp.empty_like(image)

    k1, k2 = _get_kernels()

    # Kernel 1 — nbins threads/block, one block per tile, shared mem = 2*nbins*4 B
    shared_bytes = 2 * nbins * 4
    k1(
        (n_tx, n_ty),                 # grid (x=n_tx, y=n_ty)
        (nbins,),                     # block size = nbins (256)
        (image, cdfs,
         cp.int32(H), cp.int32(W),
         cp.int32(kh), cp.int32(kw),
         cp.int32(nbins),
         cp.float32(clip_count),
         cp.int32(n_ty), cp.int32(n_tx)),
        shared_mem=shared_bytes,
    )

    # Kernel 2 — 32x32 threads/block, grid covers full image
    bx, by = 32, 32
    gx = (W + bx - 1) // bx
    gy = (H + by - 1) // by
    k2(
        (gx, gy),
        (bx, by),
        (image, cdfs, out,
         cp.int32(H), cp.int32(W),
         cp.int32(kh), cp.int32(kw),
         cp.int32(nbins),
         cp.int32(n_ty), cp.int32(n_tx)),
    )

    # Crop the padded region back off before returning.
    if pad_h or pad_w:
        out = out[:H_orig, :W_orig].copy()

    return out
