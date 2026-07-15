from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .rim_exact import _to_bchw_torch, _resize_cuda, downsample_binomial_aa_2d_cuda


ArrayLike = Union[np.ndarray, torch.Tensor]


def _q_eff(alpha: np.ndarray | float) -> np.ndarray | float:
    """Collision survival factor q(alpha) = (1 - exp(-alpha)) / alpha with q(0)=1."""
    if isinstance(alpha, np.ndarray):
        out = np.ones_like(alpha, dtype=np.float64)
        mask = alpha > 1e-12
        out[mask] = (1.0 - np.exp(-alpha[mask])) / alpha[mask]
        return out
    if alpha <= 1e-12:
        return 1.0
    return float((1.0 - math.exp(-alpha)) / alpha)


@dataclass
class CandidateBands:
    candidate_resolutions: np.ndarray          # (K,)
    block_band_mass: np.ndarray                # (B, K)
    num_blocks_y: int
    num_blocks_x: int
    image_hw: Tuple[int, int]
    block_size: int


@dataclass
class MaskedRIMResult:
    selected_resolutions: np.ndarray           # (L,)
    selected_growth: np.ndarray                # (L-1,)
    level_block_mask: np.ndarray               # (L, B) bool
    block_size: int
    num_blocks_y: int
    num_blocks_x: int
    schedule_ranges: List[Tuple[int, int]]     # length L, inclusive candidate ranges
    objective_value: float
    candidate_resolutions: np.ndarray          # (K,)
    block_band_mass: np.ndarray                # (B, K)
    details: Dict[str, Any]


def _map_level_to_u8_bins(level_bchw: torch.Tensor) -> torch.Tensor:
    """
    Map a floating-point pyramid level to uint8 bins in [0, 255].

    Important memory detail: keep the full quantized image as uint8, not int64.
    We only cast one channel/block view to int64 immediately before scatter_add_.
    This avoids an 8x memory blow-up during RIM preprocessing.
    """
    if level_bchw.ndim != 4:
        raise ValueError(f"Expected BCHW, got {tuple(level_bchw.shape)}")

    x = level_bchw.to(torch.float32)
    ch_min = x.amin(dim=(0, 2, 3), keepdim=True)
    ch_max = x.amax(dim=(0, 2, 3), keepdim=True)
    return _map_level_to_u8_bins_with_minmax(x, ch_min, ch_max)


def _map_level_to_u8_bins_with_minmax(
    level_bchw: torch.Tensor,
    ch_min: torch.Tensor,
    ch_max: torch.Tensor,
) -> torch.Tensor:
    """Quantize BCHW tensor using externally supplied per-channel min/max."""
    if level_bchw.ndim != 4:
        raise ValueError(f"Expected BCHW, got {tuple(level_bchw.shape)}")
    x = level_bchw.to(torch.float32)
    ch_min = ch_min.to(device=x.device, dtype=torch.float32)
    ch_max = ch_max.to(device=x.device, dtype=torch.float32)
    ch_range = ch_max - ch_min
    y = torch.zeros_like(x, dtype=torch.float32)
    valid = ch_range > 1e-12
    y = torch.where(valid, (x - ch_min) * (255.0 / ch_range.clamp_min(1e-12)), y)
    return y.round().clamp_(0.0, 255.0).to(torch.uint8)


def _resize_bchw_tile_align_corners_false(
    y: torch.Tensor,
    out_hw: Tuple[int, int],
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    *,
    mode: Literal["bilinear", "nearest"] = "bilinear",
) -> torch.Tensor:
    """
    Return y resized to out_hw, restricted to output tile [row0:row1, col0:col1].

    This avoids materializing the full resized tensor. For bilinear/nearest it matches
    F.interpolate(..., size=out_hw, align_corners=False) up to numerical precision,
    using grid_sample with border padding.
    """
    if y.ndim != 4 or y.shape[0] != 1:
        raise ValueError(f"Expected (1,C,H,W), got {tuple(y.shape)}")
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    row0, row1, col0, col1 = int(row0), int(row1), int(col0), int(col1)
    if row0 < 0 or col0 < 0 or row1 > out_h or col1 > out_w or row1 <= row0 or col1 <= col0:
        raise ValueError(f"Invalid tile rows=({row0},{row1}) cols=({col0},{col1}) for out_hw={out_hw}")

    _, _, in_h, in_w = y.shape
    if in_h == out_h and in_w == out_w:
        return y[:, :, row0:row1, col0:col1]

    # Normalized grid coordinates corresponding to global output pixel centers
    # under PyTorch's align_corners=False convention.
    yy = (torch.arange(row0, row1, device=y.device, dtype=torch.float32) + 0.5) * (2.0 / float(out_h)) - 1.0
    xx = (torch.arange(col0, col1, device=y.device, dtype=torch.float32) + 0.5) * (2.0 / float(out_w)) - 1.0
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)

    if mode == "nearest":
        return F.grid_sample(y, grid, mode="nearest", padding_mode="border", align_corners=False)
    return F.grid_sample(y, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _quantized_block_entropy(
    q_bchw: torch.Tensor,
    block_size: int,
    *,
    pad_to_hw: Optional[Tuple[int, int]] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute per-block entropy from a uint8 quantized BCHW tensor."""
    if q_bchw.ndim != 4 or q_bchw.shape[0] != 1:
        raise ValueError(f"Expected (1,C,H,W), got {tuple(q_bchw.shape)}")
    if q_bchw.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 quantized tensor, got {q_bchw.dtype}")

    _, C, H, W = q_bchw.shape
    bs = int(block_size)
    if bs <= 0:
        raise ValueError("block_size must be positive")

    if pad_to_hw is None:
        target_h = H + ((bs - (H % bs)) % bs)
        target_w = W + ((bs - (W % bs)) % bs)
    else:
        target_h, target_w = int(pad_to_hw[0]), int(pad_to_hw[1])
        if target_h < H or target_w < W:
            raise ValueError(f"pad_to_hw={pad_to_hw} smaller than current HW={(H, W)}")
        if target_h % bs != 0 or target_w % bs != 0:
            raise ValueError(f"pad_to_hw must be divisible by block_size={bs}, got {pad_to_hw}")

    pad_h = target_h - H
    pad_w = target_w - W
    q = q_bchw
    if pad_h or pad_w:
        q = F.pad(q, (0, pad_w, 0, pad_h), mode="constant", value=0)

    _, _, Hp, Wp = q.shape
    by = Hp // bs
    bx = Wp // bs
    B = by * bx

    q = q.squeeze(0).contiguous().view(C, by, bs, bx, bs).permute(0, 1, 3, 2, 4).reshape(C, B, bs * bs)
    ent = torch.zeros(B, device=q.device, dtype=torch.float32)
    ones = torch.ones((B, bs * bs), device=q.device, dtype=torch.float32)

    for c in range(C):
        vals = q[c].to(torch.int64)
        counts = torch.zeros((B, 256), device=q.device, dtype=torch.float32)
        counts.scatter_add_(1, vals, ones)
        probs = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        ent = ent + (-(probs * torch.log2(probs.clamp_min(eps))).sum(dim=1))
    return ent


def _aggregate_block_entropy(level_bchw: torch.Tensor, block_size: int, eps: float = 1e-12) -> torch.Tensor:
    """Aggregate empirical Shannon entropy per block from a full BCHW pyramid level."""
    q = _map_level_to_u8_bins(level_bchw)
    return _quantized_block_entropy(q, block_size=block_size, eps=eps)


def _channel_minmax_resized_tiled(
    level_bchw: torch.Tensor,
    *,
    out_hw: Tuple[int, int],
    tile_blocks: int,
    block_size: int,
    resize_mode: Literal["bilinear", "nearest"] = "bilinear",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """First streaming pass: compute per-channel min/max over a resized level."""
    _, C, _, _ = level_bchw.shape
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    bs = int(block_size)
    tile_px = max(bs, int(tile_blocks) * bs)
    ch_min = torch.full((1, C, 1, 1), float("inf"), device=level_bchw.device, dtype=torch.float32)
    ch_max = torch.full((1, C, 1, 1), float("-inf"), device=level_bchw.device, dtype=torch.float32)

    for r0 in range(0, out_h, tile_px):
        r1 = min(out_h, r0 + tile_px)
        for c0 in range(0, out_w, tile_px):
            c1 = min(out_w, c0 + tile_px)
            tile = _resize_bchw_tile_align_corners_false(
                level_bchw, out_hw, r0, r1, c0, c1, mode=resize_mode
            ).to(torch.float32)
            ch_min = torch.minimum(ch_min, tile.amin(dim=(0, 2, 3), keepdim=True))
            ch_max = torch.maximum(ch_max, tile.amax(dim=(0, 2, 3), keepdim=True))
            del tile
    return ch_min, ch_max


def _difference_resized_tile(
    high_bchw: torch.Tensor,
    low_bchw: torch.Tensor,
    *,
    out_hw: Tuple[int, int],
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    resize_mode: Literal["bilinear", "nearest"] = "bilinear",
) -> torch.Tensor:
    """
    Return the full-grid tile for the band

        resize(high_bchw, out_hw) - resize(low_bchw, out_hw).

    This is the memory-minimal RIM statistic path: it never forms the full
    low-pass image at the output grid, the native upsampled low-pass image, or
    the residual image.  It only materializes the requested tile.
    """
    high_tile = _resize_bchw_tile_align_corners_false(
        high_bchw, out_hw, row0, row1, col0, col1, mode=resize_mode
    ).to(torch.float32)
    low_tile = _resize_bchw_tile_align_corners_false(
        low_bchw, out_hw, row0, row1, col0, col1, mode=resize_mode
    ).to(torch.float32)
    return high_tile - low_tile


def _channel_minmax_difference_resized_tiled(
    high_bchw: torch.Tensor,
    low_bchw: torch.Tensor,
    *,
    out_hw: Tuple[int, int],
    tile_blocks: int,
    block_size: int,
    resize_mode: Literal["bilinear", "nearest"] = "bilinear",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """First streaming pass for a low-pass difference band: per-channel min/max."""
    if high_bchw.ndim != 4 or low_bchw.ndim != 4 or high_bchw.shape[0] != 1 or low_bchw.shape[0] != 1:
        raise ValueError(f"Expected high/low as (1,C,H,W), got {tuple(high_bchw.shape)} and {tuple(low_bchw.shape)}")
    if high_bchw.shape[1] != low_bchw.shape[1]:
        raise ValueError(f"Channel mismatch: high has {high_bchw.shape[1]}, low has {low_bchw.shape[1]}")

    _, C, _, _ = high_bchw.shape
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    bs = int(block_size)
    tile_px = max(bs, int(tile_blocks) * bs)
    ch_min = torch.full((1, C, 1, 1), float("inf"), device=high_bchw.device, dtype=torch.float32)
    ch_max = torch.full((1, C, 1, 1), float("-inf"), device=high_bchw.device, dtype=torch.float32)

    for r0 in range(0, out_h, tile_px):
        r1 = min(out_h, r0 + tile_px)
        for c0 in range(0, out_w, tile_px):
            c1 = min(out_w, c0 + tile_px)
            tile = _difference_resized_tile(
                high_bchw,
                low_bchw,
                out_hw=(out_h, out_w),
                row0=r0,
                row1=r1,
                col0=c0,
                col1=c1,
                resize_mode=resize_mode,
            )
            ch_min = torch.minimum(ch_min, tile.amin(dim=(0, 2, 3), keepdim=True))
            ch_max = torch.maximum(ch_max, tile.amax(dim=(0, 2, 3), keepdim=True))
            del tile
    return ch_min, ch_max


def _aggregate_difference_block_entropy_resized_tiled(
    high_bchw: torch.Tensor,
    low_bchw: torch.Tensor,
    *,
    out_hw: Tuple[int, int],
    block_size: int,
    tile_blocks: int = 8,
    resize_mode: Literal["bilinear", "nearest"] = "bilinear",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Aggregate block entropy for a low-pass difference band using only tile statistics.

    The band is defined directly on the training grid as

        resize(high_bchw, out_hw) - resize(low_bchw, out_hw).

    This is intentionally different from the old memory-heavy implementation,
    which formed resize(high - resize(low, high.shape), out_hw).  The direct
    full-grid difference is a valid Laplacian-style band for RIM and is much
    safer for large images because only per-tile values and per-block entropies
    are materialized.
    """
    if high_bchw.ndim != 4 or low_bchw.ndim != 4 or high_bchw.shape[0] != 1 or low_bchw.shape[0] != 1:
        raise ValueError(f"Expected high/low as (1,C,H,W), got {tuple(high_bchw.shape)} and {tuple(low_bchw.shape)}")

    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    bs = int(block_size)
    tb = max(1, int(tile_blocks))
    num_blocks_y = int(math.ceil(out_h / float(bs)))
    num_blocks_x = int(math.ceil(out_w / float(bs)))

    ch_min, ch_max = _channel_minmax_difference_resized_tiled(
        high_bchw,
        low_bchw,
        out_hw=(out_h, out_w),
        tile_blocks=tb,
        block_size=bs,
        resize_mode=resize_mode,
    )

    ent_grid_cpu = torch.empty((num_blocks_y, num_blocks_x), dtype=torch.float32, device="cpu")

    for by0 in range(0, num_blocks_y, tb):
        by1 = min(num_blocks_y, by0 + tb)
        r0 = by0 * bs
        r1 = min(out_h, by1 * bs)
        target_h = (by1 - by0) * bs
        for bx0 in range(0, num_blocks_x, tb):
            bx1 = min(num_blocks_x, bx0 + tb)
            c0 = bx0 * bs
            c1 = min(out_w, bx1 * bs)
            target_w = (bx1 - bx0) * bs

            tile = _difference_resized_tile(
                high_bchw,
                low_bchw,
                out_hw=(out_h, out_w),
                row0=r0,
                row1=r1,
                col0=c0,
                col1=c1,
                resize_mode=resize_mode,
            )
            q = _map_level_to_u8_bins_with_minmax(tile, ch_min, ch_max)
            ent_patch = _quantized_block_entropy(
                q,
                block_size=bs,
                pad_to_hw=(target_h, target_w),
                eps=eps,
            ).reshape(by1 - by0, bx1 - bx0)
            ent_grid_cpu[by0:by1, bx0:bx1] = ent_patch.detach().cpu()
            del tile, q, ent_patch

    return ent_grid_cpu.reshape(-1)


def _aggregate_block_entropy_resized_tiled(
    level_bchw: torch.Tensor,
    *,
    out_hw: Tuple[int, int],
    block_size: int,
    tile_blocks: int = 8,
    resize_mode: Literal["bilinear", "nearest"] = "bilinear",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Aggregate empirical block entropy after resizing level_bchw to out_hw, tile by tile.

    This is the memory-safe RIM preprocessing path. It never materializes the full
    resized residual/base band and never materializes a full int64 quantized image.
    Tiling is aligned to the RIM block grid, so each block histogram is computed
    exactly on its own pixels; only floating-point interpolation may differ at the
    last bit from a full F.interpolate call.
    """
    if level_bchw.ndim != 4 or level_bchw.shape[0] != 1:
        raise ValueError(f"Expected (1,C,H,W), got {tuple(level_bchw.shape)}")

    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    bs = int(block_size)
    tb = max(1, int(tile_blocks))
    num_blocks_y = int(math.ceil(out_h / float(bs)))
    num_blocks_x = int(math.ceil(out_w / float(bs)))

    ch_min, ch_max = _channel_minmax_resized_tiled(
        level_bchw,
        out_hw=(out_h, out_w),
        tile_blocks=tb,
        block_size=bs,
        resize_mode=resize_mode,
    )

    ent_grid_cpu = torch.empty((num_blocks_y, num_blocks_x), dtype=torch.float32, device="cpu")

    for by0 in range(0, num_blocks_y, tb):
        by1 = min(num_blocks_y, by0 + tb)
        r0 = by0 * bs
        r1 = min(out_h, by1 * bs)
        target_h = (by1 - by0) * bs
        for bx0 in range(0, num_blocks_x, tb):
            bx1 = min(num_blocks_x, bx0 + tb)
            c0 = bx0 * bs
            c1 = min(out_w, bx1 * bs)
            target_w = (bx1 - bx0) * bs

            tile = _resize_bchw_tile_align_corners_false(
                level_bchw, (out_h, out_w), r0, r1, c0, c1, mode=resize_mode
            )
            q = _map_level_to_u8_bins_with_minmax(tile, ch_min, ch_max)
            ent_patch = _quantized_block_entropy(
                q,
                block_size=bs,
                pad_to_hw=(target_h, target_w),
                eps=eps,
            ).reshape(by1 - by0, bx1 - bx0)
            ent_grid_cpu[by0:by1, bx0:bx1] = ent_patch.detach().cpu()
            del tile, q, ent_patch

    return ent_grid_cpu.reshape(-1)



@torch.no_grad()
def build_candidate_band_block_masses(
    img: ArrayLike,
    *,
    block_size: int,
    res_min: int = 16,
    scale_step: float = 1.2,
    max_candidates: Optional[int] = None,
    pyramid_dtype: Optional[torch.dtype] = None,
    upsample_mode: Literal["bilinear", "nearest"] = "bilinear",
    preprocess_tiled: bool = False,
    preprocess_tile_blocks: int = 8,
    preprocess_device: Literal["auto", "cuda", "cpu"] = "auto",
    preprocess_stream_stats: bool = False,
) -> CandidateBands:
    """
    Build candidate residual bands and per-block band masses on the *full* training grid.

    If preprocess_tiled=False, this follows the original full-grid path. If
    preprocess_tiled=True, each residual/base band is resized and quantized patch by
    patch, avoiding the large residual_up and int64 q tensors that cause OOM.
    If preprocess_stream_stats=True, RIM uses only streamed block statistics for
    each low-pass difference band and never materializes g_low_up or residual at
    the native pyramid scale.
    """
    if scale_step <= 1.0:
        raise ValueError(f"scale_step must be > 1, got {scale_step}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    if preprocess_device == "cpu":
        device = torch.device("cpu")
    elif preprocess_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--rim_preprocess_device cuda requested, but CUDA is not available")
        device = torch.device("cuda")
    elif preprocess_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise ValueError(f"Unknown preprocess_device={preprocess_device!r}; expected auto/cuda/cpu")

    x = _to_bchw_torch(img, device=device)
    if x.shape[0] > 1:
        x = x[:1]
    if x.dtype == torch.uint8:
        x = x.to(torch.float16 if (device.type == "cuda") else torch.float32)
    else:
        x = x.to(torch.float16 if (device.type == "cuda") else torch.float32)
    if pyramid_dtype is not None:
        x = x.to(pyramid_dtype)

    _, _, H0, W0 = x.shape
    if max_candidates is None:
        max_candidates = 64

    candidate_res: List[float] = []
    block_masses: List[np.ndarray] = []

    g_high = x
    Hh, Wh = int(H0), int(W0)
    res_high = float(min(Hh, Wh))

    while (res_high > float(res_min)) and (len(candidate_res) < int(max_candidates)):
        Hn = max(1, int(round(Hh / float(scale_step))))
        Wn = max(1, int(round(Wh / float(scale_step))))
        if Hn == Hh and Wn == Wh:
            break

        g_low = downsample_binomial_aa_2d_cuda(g_high, out_hw=(Hn, Wn), k=5, passes=1)

        if preprocess_tiled and preprocess_stream_stats:
            # Strict sufficient-statistics path: do not materialize g_low_up, residual,
            # residual_up, or a full quantized image.  For RIM we only need the
            # per-block entropy of the low-pass difference band on the training grid.
            block_mass = _aggregate_difference_block_entropy_resized_tiled(
                g_high,
                g_low,
                out_hw=(H0, W0),
                block_size=block_size,
                tile_blocks=preprocess_tile_blocks,
                resize_mode=upsample_mode,
            )
        else:
            # Legacy/exact-native residual path.  This can be memory-heavy because it
            # materializes g_low_up and residual at the current pyramid scale.
            g_low_up = _resize_cuda(g_low, (Hh, Wh), mode=upsample_mode, antialias=False)
            residual = g_high - g_low_up

            if preprocess_tiled:
                block_mass = _aggregate_block_entropy_resized_tiled(
                    residual,
                    out_hw=(H0, W0),
                    block_size=block_size,
                    tile_blocks=preprocess_tile_blocks,
                    resize_mode=upsample_mode,
                )
            else:
                if (Hh, Wh) != (H0, W0):
                    residual_up = _resize_cuda(residual, (H0, W0), mode=upsample_mode, antialias=False)
                else:
                    residual_up = residual
                block_mass = _aggregate_block_entropy(residual_up, block_size=block_size)
            del g_low_up, residual
        candidate_res.append(float(res_high))
        block_masses.append(block_mass.detach().cpu().to(torch.float64).numpy())

        g_high = g_low
        Hh, Wh = Hn, Wn
        res_high = float(min(Hh, Wh))
        if res_high <= 2.0:
            break

    # Base band.
    if preprocess_tiled:
        base_block_mass = _aggregate_block_entropy_resized_tiled(
            g_high,
            out_hw=(H0, W0),
            block_size=block_size,
            tile_blocks=preprocess_tile_blocks,
            resize_mode=upsample_mode,
        )
    else:
        if (Hh, Wh) != (H0, W0):
            base_up = _resize_cuda(g_high, (H0, W0), mode=upsample_mode, antialias=False)
        else:
            base_up = g_high
        base_block_mass = _aggregate_block_entropy(base_up, block_size=block_size)
    candidate_res.append(float(res_high))
    block_masses.append(base_block_mass.detach().cpu().to(torch.float64).numpy())

    candidate_res_np = np.asarray(candidate_res, dtype=np.float64)
    block_band_mass_np = np.stack(block_masses, axis=1)  # (B, K_desc)

    # Sort ascending resolution so DP operates on contiguous low->high candidate ranges.
    perm = np.argsort(candidate_res_np)
    candidate_res_np = candidate_res_np[perm]
    block_band_mass_np = block_band_mass_np[:, perm]

    block_band_mass_np = np.maximum(block_band_mass_np, 0.0)

    num_blocks_y = int(math.ceil(H0 / float(block_size)))
    num_blocks_x = int(math.ceil(W0 / float(block_size)))
    return CandidateBands(
        candidate_resolutions=candidate_res_np,
        block_band_mass=block_band_mass_np,
        num_blocks_y=num_blocks_y,
        num_blocks_x=num_blocks_x,
        image_hw=(int(H0), int(W0)),
        block_size=int(block_size),
    )


def _build_segment_scores_exact(
    candidate_resolutions: np.ndarray,
    block_band_mass: np.ndarray,
    candidate_mask: np.ndarray,
    hash_size: int,
) -> np.ndarray:
    """Exact segment scores for the additive retrievable-information objective."""
    B, K = block_band_mass.shape
    cand_mask_bool = candidate_mask.astype(bool, copy=False)
    masked_mass = block_band_mass * cand_mask_bool.astype(np.float64)
    col_mass = masked_mass.sum(axis=0)

    scores = np.full((K, K), -np.inf, dtype=np.float64)
    for j in range(K):
        union = np.zeros(B, dtype=bool)
        mass_sum = 0.0
        for k in range(j, K):
            union |= cand_mask_bool[:, k]
            mass_sum += float(col_mass[k])
            active_blocks = int(union.sum())
            alpha = (active_blocks / max(1, B)) * ((candidate_resolutions[k] ** 2) / float(hash_size))
            scores[j, k] = float(_q_eff(alpha) * mass_sum)
    return scores



def _solve_additive_dp_exact(segment_scores: np.ndarray, num_levels: int) -> List[Tuple[int, int]]:
    """Exact O(L K^2) DP for additive contiguous partitioning."""
    K = int(segment_scores.shape[0])
    L = int(num_levels)
    if K < L:
        raise ValueError(f"Need at least L={L} candidate bands, got K={K}.")

    dp = np.full((L + 1, K), -np.inf, dtype=np.float64)
    parent = np.full((L + 1, K), -1, dtype=np.int64)

    for k in range(K):
        dp[1, k] = segment_scores[0, k]
        parent[1, k] = -1

    for ell in range(2, L + 1):
        for k in range(ell - 1, K):
            best_val = -np.inf
            best_j = -1
            for j in range(ell - 2, k):
                cand = dp[ell - 1, j] + segment_scores[j + 1, k]
                if cand > best_val:
                    best_val = cand
                    best_j = j
            dp[ell, k] = best_val
            parent[ell, k] = best_j

    ranges: List[Tuple[int, int]] = []
    k = K - 1
    for ell in range(L, 0, -1):
        j = int(parent[ell, k])
        start = 0 if j < 0 else (j + 1)
        ranges.append((start, k))
        k = j
    ranges.reverse()
    return ranges



def _expand_level_mask_to_candidates(
    level_block_mask: np.ndarray,
    schedule_ranges: Sequence[Tuple[int, int]],
    num_candidates: int,
) -> np.ndarray:
    B = int(level_block_mask.shape[1])
    out = np.zeros((B, num_candidates), dtype=bool)
    for ell, (j, k) in enumerate(schedule_ranges):
        out[:, j:k + 1] = level_block_mask[ell][:, None]
    return out



def _selector_update_exact_per_segment(
    block_band_mass: np.ndarray,
    schedule_ranges: Sequence[Tuple[int, int]],
    candidate_resolutions: np.ndarray,
    hash_size: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Exact batch selector update for fixed schedule.

    For one segment, all blocks contribute equal load, so the exact optimum is obtained by
    dropping the blocks with the smallest segment masses.
    """
    B, _ = block_band_mass.shape
    L = len(schedule_ranges)
    level_mask = np.zeros((L, B), dtype=bool)
    per_level_details: List[Dict[str, Any]] = []

    for ell, (j, k) in enumerate(schedule_ranges):
        seg_mass = block_band_mass[:, j:k + 1].sum(axis=1)
        order = np.argsort(seg_mass, kind="stable")
        sorted_mass = seg_mass[order]
        prefix = np.zeros(B + 1, dtype=np.float64)
        prefix[1:] = np.cumsum(sorted_mass, dtype=np.float64)
        total_mass = float(prefix[-1])
        res_end = float(candidate_resolutions[k])

        best_score = -np.inf
        best_drop = 0
        for s in range(B + 1):
            active = B - s
            kept_mass = total_mass - float(prefix[s])
            alpha = (active / max(1, B)) * ((res_end ** 2) / float(hash_size))
            score = float(_q_eff(alpha) * kept_mass)
            if score > best_score:
                best_score = score
                best_drop = s

        keep_mask = np.ones(B, dtype=bool)
        if best_drop > 0:
            keep_mask[order[:best_drop]] = False
        level_mask[ell] = keep_mask
        per_level_details.append(
            {
                "range": (int(j), int(k)),
                "resolution": int(round(res_end)),
                "active_blocks": int(keep_mask.sum()),
                "dropped_blocks": int(best_drop),
                "segment_total_mass": total_mass,
                "segment_objective": float(best_score),
            }
        )
    return level_mask, per_level_details



def _objective_from_schedule_and_mask(
    block_band_mass: np.ndarray,
    schedule_ranges: Sequence[Tuple[int, int]],
    level_block_mask: np.ndarray,
    candidate_resolutions: np.ndarray,
    hash_size: int,
) -> float:
    B, _ = block_band_mass.shape
    total = 0.0
    for ell, (j, k) in enumerate(schedule_ranges):
        seg_mass = block_band_mass[:, j:k + 1].sum(axis=1)
        active = level_block_mask[ell].astype(np.float64)
        kept_mass = float((seg_mass * active).sum())
        active_blocks = int(active.sum())
        alpha = (active_blocks / max(1, B)) * ((candidate_resolutions[k] ** 2) / float(hash_size))
        total += float(_q_eff(alpha) * kept_mass)
    return total


@torch.no_grad()
def solve_blockwise_masked_rim_exact(
    img: ArrayLike,
    *,
    num_levels: int,
    block_size: int,
    hash_log2: int,
    res_min: int = 16,
    scale_step: float = 1.2,
    max_candidates: Optional[int] = None,
    max_iters: int = 4,
    accept_tol: float = 1e-9,
    upsample_mode: Literal["bilinear", "nearest"] = "bilinear",
    preprocess_tiled: bool = False,
    preprocess_tile_blocks: int = 8,
    preprocess_device: Literal["auto", "cuda", "cpu"] = "auto",
    preprocess_stream_stats: bool = False,
    verbose: bool = False,
) -> MaskedRIMResult:
    """
    Alternating exact masked-RIM solver.

    Schedule step: exact additive DP over contiguous candidate bands.
    Selector step: exact per-level batch optimization over arbitrary block masks.
    """
    cand = build_candidate_band_block_masses(
        img,
        block_size=block_size,
        res_min=res_min,
        scale_step=scale_step,
        max_candidates=max_candidates,
        upsample_mode=upsample_mode,
        preprocess_tiled=bool(preprocess_tiled),
        preprocess_tile_blocks=int(preprocess_tile_blocks),
        preprocess_device=preprocess_device,
        preprocess_stream_stats=bool(preprocess_stream_stats),
    )
    B, K = cand.block_band_mass.shape
    if K < int(num_levels):
        raise ValueError(
            f"Need at least num_levels={num_levels} candidate bands, but only K={K} were generated. "
            f"Try reducing --levels or increasing --rim_max_candidates / decreasing --scale_step."
        )

    hash_size = int(1 << int(hash_log2))
    all_on = np.ones((B, K), dtype=bool)
    seg_scores = _build_segment_scores_exact(cand.candidate_resolutions, cand.block_band_mass, all_on, hash_size)
    schedule_ranges = _solve_additive_dp_exact(seg_scores, num_levels=num_levels)
    level_block_mask, init_level_details = _selector_update_exact_per_segment(
        cand.block_band_mass,
        schedule_ranges,
        cand.candidate_resolutions,
        hash_size,
    )
    best_obj = _objective_from_schedule_and_mask(
        cand.block_band_mass,
        schedule_ranges,
        level_block_mask,
        cand.candidate_resolutions,
        hash_size,
    )
    best_schedule = list(schedule_ranges)
    best_mask = level_block_mask.copy()
    iter_log: List[Dict[str, Any]] = [
        {
            "iter": 0,
            "objective": float(best_obj),
            "schedule_ranges": [(int(a), int(b)) for a, b in schedule_ranges],
            "levels": init_level_details,
        }
    ]

    for it in range(1, int(max_iters) + 1):
        candidate_mask = _expand_level_mask_to_candidates(best_mask, best_schedule, num_candidates=K)
        seg_scores = _build_segment_scores_exact(cand.candidate_resolutions, cand.block_band_mass, candidate_mask, hash_size)
        new_schedule = _solve_additive_dp_exact(seg_scores, num_levels=num_levels)
        new_mask, level_details = _selector_update_exact_per_segment(
            cand.block_band_mass,
            new_schedule,
            cand.candidate_resolutions,
            hash_size,
        )
        new_obj = _objective_from_schedule_and_mask(
            cand.block_band_mass,
            new_schedule,
            new_mask,
            cand.candidate_resolutions,
            hash_size,
        )

        improved = new_obj > (best_obj + float(accept_tol))
        iter_log.append(
            {
                "iter": int(it),
                "objective": float(new_obj),
                "accepted": bool(improved),
                "schedule_ranges": [(int(a), int(b)) for a, b in new_schedule],
                "levels": level_details,
            }
        )
        if verbose:
            print(
                f"[masked-RIM] iter={it} objective={new_obj:.6e} "
                f"accepted={improved} schedule={new_schedule}",
                flush=True,
            )
        if improved:
            best_obj = float(new_obj)
            best_schedule = list(new_schedule)
            best_mask = new_mask.copy()
        else:
            break

    selected_res = np.asarray([cand.candidate_resolutions[k] for _, k in best_schedule], dtype=np.int64)
    selected_res = np.maximum.accumulate(selected_res)
    growth = selected_res[1:] / np.maximum(selected_res[:-1], 1)

    details: Dict[str, Any] = {
        "method": "masked_rim_exact_additive_dp",
        "candidate_resolutions": cand.candidate_resolutions.copy(),
        "schedule_ranges": [(int(a), int(b)) for a, b in best_schedule],
        "iterations": iter_log,
        "hash_size": int(hash_size),
        "block_size": int(block_size),
        "num_blocks": int(B),
        "image_hw": tuple(int(v) for v in cand.image_hw),
        "preprocess_tiled": bool(preprocess_tiled),
        "preprocess_tile_blocks": int(preprocess_tile_blocks),
        "preprocess_device": str(preprocess_device),
        "preprocess_stream_stats": bool(preprocess_stream_stats),
    }
    return MaskedRIMResult(
        selected_resolutions=selected_res,
        selected_growth=growth,
        level_block_mask=best_mask.astype(bool, copy=False),
        block_size=int(block_size),
        num_blocks_y=int(cand.num_blocks_y),
        num_blocks_x=int(cand.num_blocks_x),
        schedule_ranges=list(best_schedule),
        objective_value=float(best_obj),
        candidate_resolutions=cand.candidate_resolutions.copy(),
        block_band_mass=cand.block_band_mass.copy(),
        details=details,
    )


__all__ = [
    "CandidateBands",
    "MaskedRIMResult",
    "build_candidate_band_block_masses",
    "solve_blockwise_masked_rim_exact",
]
