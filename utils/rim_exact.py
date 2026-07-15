"""
zip_rim_exact.py

Paper-faithful RIM schedule construction for multiresolution hash encoding.

This file implements the RIM scheduler described in the uploaded ECCV 2026 paper:
  - residual-band information density
  - anti-aliased low-pass pyramid construction
  - collision-aware end-of-band weighting
  - exact O(L K^2) min-max dynamic programming solver

Main entry point:
    get_entropy_grid(...)

Returns:
    selected integer resolutions and per-step growth factors.

Notes
-----
1) The paper defines the band information density as
       I(f) = H(Y_{<=f+df} - Y_{<=f}) * N_f^2
   for 2D images, or more generally * N_f^d.
   Here we discretize that quantity by building a low-pass pyramid. For each pair of
   adjacent candidate resolutions, we subtract the upsampled coarser low-pass image
   from the finer low-pass image and measure the residual entropy.

2) The paper uses anti-aliased resizing as the default low-pass operator in the
   experiments. This implementation therefore uses binomial blur + antialiased
   bilinear resize for downsampling.

3) The supplementary gives the exact DP recursion:
       D[l, k] = min_j max(D[l-1, j], w[k] * sum_{t=j+1}^k psi[t])
   with base case
       D[1, k] = w[k] * sum_{t=1}^k psi[t].
   This file implements that recurrence directly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


# ----------------------------
# Hash collision weighting
# ----------------------------
def _hash_weight(
    resolutions_t: torch.Tensor,
    hash_size: int,
    grid_dim: int,
    mode: Literal["efficiency", "marginal"] = "efficiency",
    min_weight: float = 1e-4,
) -> torch.Tensor:
    """
    Collision efficiency for a hashed grid.

    Paper-faithful mode is "efficiency":
        w = (1 - exp(-alpha)) / alpha,
        alpha = N^d / T.
    """
    T = float(hash_size)
    N = resolutions_t.clamp_min(1.0)
    n = N.pow(float(grid_dim))
    alpha = n / T

    if mode == "efficiency":
        w = (1.0 - torch.exp(-alpha)) / (alpha + 1e-12)
    elif mode == "marginal":
        # kept only for experimentation; not the paper's default
        w = grid_dim * n * torch.exp(-alpha)
        w = w / (w.max() + 1e-12)
    else:
        raise ValueError(f"Unknown mode={mode}")

    return w.clamp_min(min_weight)


# ----------------------------
# Helpers: normalize shapes -> BCHW
# ----------------------------
def _to_bchw_torch(img: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Accept HW/HWC/CHW/BCHW/BHWC and return BCHW on `device`."""
    if isinstance(img, np.ndarray):
        x = torch.from_numpy(img)
    elif isinstance(img, torch.Tensor):
        x = img
    else:
        raise TypeError(f"img must be numpy array or torch tensor, got {type(img)}")

    if x.ndim == 2:
        x = x[None, None, ...]
    elif x.ndim == 3:
        if (x.shape[-1] >= 1 and x.shape[-1] <= 4096) and (x.shape[0] > 8 and x.shape[1] > 8):
            x = x.permute(2, 0, 1)[None, ...]
        else:
            x = x[None, ...]
    elif x.ndim == 4:
        if (x.shape[-1] >= 1 and x.shape[-1] <= 4096) and (x.shape[1] > 8 and x.shape[2] > 8):
            x = x.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unsupported img ndim={x.ndim}")

    return x.to(device, non_blocking=True)


def _resize_cuda(
    y: torch.Tensor,
    out_hw: Tuple[int, int],
    mode: Literal["nearest", "bilinear"] = "bilinear",
    antialias: bool = True,
    chunk_c: int = 8,
) -> torch.Tensor:
    """Resize BCHW tensor, chunking channels on CUDA when helpful."""
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    _, C, H, W = y.shape
    if (out_h == H) and (out_w == W):
        return y

    def _interp(t: torch.Tensor) -> torch.Tensor:
        t = t.contiguous()
        if mode == "nearest":
            return F.interpolate(t, size=(out_h, out_w), mode="nearest")
        if antialias:
            try:
                return F.interpolate(
                    t,
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            except (TypeError, RuntimeError):
                return F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)

    if (not y.is_cuda) or (C <= chunk_c):
        return _interp(y)

    ys = []
    for s in range(0, C, chunk_c):
        ys.append(_interp(y[:, s:s + chunk_c, :, :]))
    return torch.cat(ys, dim=1)


# ----------------------------
# Cached binomial weights
# ----------------------------
_BINOM_CACHE: Dict[Tuple[int, int, str, Optional[int], torch.dtype], Tuple[torch.Tensor, torch.Tensor, int]] = {}


def _get_binom_weights(C: int, k: int, device: torch.device, dtype: torch.dtype):
    key = (C, k, device.type, device.index, dtype)
    if key in _BINOM_CACHE:
        return _BINOM_CACHE[key]

    if k == 3:
        coeffs = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
    elif k == 5:
        coeffs = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported k={k} (use 3 or 5)")

    coeffs = coeffs / coeffs.sum()
    pad = k // 2
    wv = coeffs.view(1, 1, k, 1).repeat(C, 1, 1, 1)
    wh = coeffs.view(1, 1, 1, k).repeat(C, 1, 1, 1)
    _BINOM_CACHE[key] = (wv, wh, pad)
    return wv, wh, pad


def binomial_blur_2d_cuda(x: torch.Tensor, k: int = 5, passes: int = 1) -> torch.Tensor:
    """Depthwise separable binomial blur for BCHW tensors."""
    if passes <= 0:
        return x
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW, got {tuple(x.shape)}")

    _, C, _, _ = x.shape
    wv, wh, pad = _get_binom_weights(C, k, x.device, x.dtype)
    y = x
    for _ in range(int(passes)):
        y = F.conv2d(y, wv, padding=(pad, 0), groups=C)
        y = F.conv2d(y, wh, padding=(0, pad), groups=C)
    return y


def downsample_binomial_aa_2d_cuda(
    x: torch.Tensor,
    out_hw: Tuple[int, int],
    k: int = 5,
    passes: int = 1,
) -> torch.Tensor:
    """
    Paper-aligned low-pass step: blur + antialiased bilinear resize.
    """
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    in_h, in_w = int(x.shape[-2]), int(x.shape[-1])
    if out_h >= in_h and out_w >= in_w:
        return x
    y = binomial_blur_2d_cuda(x, k=k, passes=passes)
    return _resize_cuda(y, (out_h, out_w), mode="bilinear", antialias=True)


# ----------------------------
# Entropy: sample-only quantization + bincount
# ----------------------------
@torch.no_grad()
def compute_entropy_cuda_u8like(
    img_bchw: torch.Tensor,
    clip: int = 255,
    max_samples_per_channel: Optional[int] = 5_000_000, 
    eps: float = 1e-12,
) -> float:
    """
    Empirical per-channel entropy for values expected in [0, clip] after rounding.
    Quantize only sampled values for efficiency.
    """
    if img_bchw.ndim != 4:
        raise ValueError(f"Expected BCHW, got {tuple(img_bchw.shape)}")

    x = img_bchw
    B, C, H, W = x.shape
    N = int(B * H * W)

    if not x.is_floating_point():
        x = x.to(torch.float16 if x.is_cuda else torch.float32)

    entropy = 0.0
    for c in range(C):
        bins = x[:, c].reshape(-1)
        if (max_samples_per_channel is not None) and (N > int(max_samples_per_channel)):
            stride = int(math.ceil(N / float(max_samples_per_channel)))
            bins = bins[::stride]

        q = bins.round().clamp(0, float(clip)).to(torch.int64)
        counts = torch.bincount(q, minlength=int(clip) + 1).to(torch.float32)
        counts = counts[counts > 0]
        if counts.numel() == 0:
            continue
        probs = counts / counts.sum()
        entropy += float((-(probs * torch.log2(probs + eps)).sum()).item())
    return entropy


@torch.no_grad()
def compute_entropy_cuda_signed(
    img_bchw: torch.Tensor,
    clip: int = 255,
    max_samples_per_channel: Optional[int] = 500_000,
    eps: float = 1e-12,
) -> float:
    """
    Empirical per-channel entropy for signed residual values in [-clip, clip].
    Matches the paper's residual-signal entropy idea using a fixed signed alphabet.
    """
    if img_bchw.ndim != 4:
        raise ValueError(f"Expected BCHW, got {tuple(img_bchw.shape)}")

    x = img_bchw
    B, C, H, W = x.shape
    N = int(B * H * W)

    if not x.is_floating_point():
        x = x.to(torch.float16 if x.is_cuda else torch.float32)

    entropy = 0.0
    nbins = int(2 * clip + 1)
    offset = int(clip)

    for c in range(C):
        bins = x[:, c].reshape(-1)
        if (max_samples_per_channel is not None) and (N > int(max_samples_per_channel)):
            stride = int(math.ceil(N / float(max_samples_per_channel)))
            bins = bins[::stride]

        q = bins.round().clamp(-float(clip), float(clip)).to(torch.int64) + offset
        counts = torch.bincount(q, minlength=nbins).to(torch.float32)
        counts = counts[counts > 0]
        if counts.numel() == 0:
            continue
        probs = counts / counts.sum()
        entropy += float((-(probs * torch.log2(probs + eps)).sum()).item())
    return entropy


# ----------------------------
# Optional CPU pre-downscale for ultra-large inputs
# ----------------------------
def _maybe_cpu_pre_downscale_uint8(
    img: Union[np.ndarray, torch.Tensor],
    target_min_dim: int,
) -> Union[np.ndarray, torch.Tensor]:
    """
    For very large CPU inputs, downscale on CPU using INTER_AREA before moving to CUDA.
    This is an engineering optimization; it does not change the scheduler logic.
    """
    if cv2 is None:
        return img

    if isinstance(img, torch.Tensor):
        if img.device.type != "cpu":
            return img
        arr = img.detach()
        try:
            np_img = arr.numpy()
        except Exception:
            return img
        out_type = "torch"
    else:
        np_img = img
        out_type = "numpy"

    if np_img.ndim < 2:
        return img

    if np_img.ndim == 2:
        H, W = np_img.shape
    elif np_img.ndim == 3:
        if np_img.shape[0] in (1, 3, 4) and np_img.shape[-1] not in (1, 3, 4):
            H, W = np_img.shape[1], np_img.shape[2]
        else:
            H, W = np_img.shape[0], np_img.shape[1]
    elif np_img.ndim == 4:
        np_img0 = np_img[0]
        resized0 = _maybe_cpu_pre_downscale_uint8(np_img0, target_min_dim)
        if out_type == "torch":
            return torch.from_numpy(resized0).to(img.dtype)
        return resized0
    else:
        return img

    if min(H, W) <= int(target_min_dim):
        return img

    scale = float(min(H, W)) / float(target_min_dim)
    new_h = max(1, int(round(H / scale)))
    new_w = max(1, int(round(W / scale)))

    if np_img.ndim == 2:
        resized = cv2.resize(np_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        if np_img.shape[0] in (1, 3, 4) and np_img.shape[-1] not in (1, 3, 4):
            chw = np_img
            hwc = np.transpose(chw, (1, 2, 0))
            hwc2 = cv2.resize(hwc, (new_w, new_h), interpolation=cv2.INTER_AREA)
            resized = np.transpose(hwc2, (2, 0, 1))
        else:
            resized = cv2.resize(np_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if out_type == "torch":
        return torch.from_numpy(resized).to(img.dtype)
    return resized


# ----------------------------
# Exact DP from the paper
# ----------------------------
def _solve_rim_dp_exact(psi: np.ndarray, w_end: np.ndarray, num_levels: int) -> np.ndarray:
    """
    Exact O(L K^2) DP for the paper's min-max objective.

    psi   : length-K array of discretized information masses psi_1..psi_K
    w_end : length-K array of collision efficiencies w_1..w_K
    num_levels : L

    Returns
    -------
    idx0 : length-L array of 0-based selected cutoff indices [k1-1, ..., kL-1],
           with the final index fixed to K-1.
    """
    psi = np.asarray(psi, dtype=np.float64)
    w_end = np.asarray(w_end, dtype=np.float64)
    K = int(psi.shape[0])
    L = int(num_levels)

    if K < L:
        raise ValueError(f"Need at least L={L} candidate intervals, but got K={K}.")
    if w_end.shape[0] != K:
        raise ValueError("w_end and psi must have the same length.")

    # Prefix sums P[0]=0, P[k]=sum_{t=1}^k psi_t.
    P = np.zeros(K + 1, dtype=np.float64)
    P[1:] = np.cumsum(psi)

    D = np.full((L + 1, K + 1), np.inf, dtype=np.float64)
    parent = np.full((L + 1, K + 1), -1, dtype=np.int64)

    # Base case: one band covering intervals 1..k.
    for k in range(1, K + 1):
        D[1, k] = w_end[k - 1] * P[k]
        parent[1, k] = 0

    # Recurrence from the paper.
    for ell in range(2, L + 1):
        # Need at least `ell` intervals to build `ell` non-empty bands.
        for k in range(ell, K + 1):
            wk = w_end[k - 1]
            best_val = np.inf
            best_j = -1
            for j in range(ell - 1, k):
                last_band = wk * (P[k] - P[j])
                val = max(D[ell - 1, j], last_band)
                if val < best_val:
                    best_val = val
                    best_j = j
            D[ell, k] = best_val
            parent[ell, k] = best_j

    # Backtrack with k_L = K.
    k_idx = np.zeros(L + 1, dtype=np.int64)
    k_idx[L] = K
    for ell in range(L, 0, -1):
        k_idx[ell - 1] = parent[ell, k_idx[ell]]
        if ell > 1 and k_idx[ell - 1] < 1:
            raise RuntimeError("DP backtracking failed; predecessor became invalid.")

    # Convert 1-based k_1..k_L to 0-based indices into candidate resolutions.
    idx0 = k_idx[1:] - 1
    return idx0


# ----------------------------
# Main: exact RIM schedule
# ----------------------------
@torch.no_grad()
def get_entropy_grid(
    img: Union[np.ndarray, torch.Tensor],
    num_levels: int = 16,
    res_min: int = 16,
    res_max: int = 25_000,
    scale_step: float = 1.2,
    clip: int = 255,
    residual_clip: int = 255,
    bw_order: int = 2,
    grid_dim: int = 2,
    hash_size: Optional[int] = None,
    hash_log2: Optional[int] = None,
    hash_weight_mode: Literal["efficiency", "marginal"] = "efficiency",
    max_samples_per_channel: Optional[int] = 500_000,
    pyramid_dtype: torch.dtype = torch.float16,
    return_details: bool = False,
    max_candidates: Optional[int] = None,
    upsample_mode: Literal["bilinear", "nearest"] = "bilinear",
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, Dict[str, Any]],
]:
    """
    Build a paper-faithful RIM schedule.

    Parameters
    ----------
    num_levels : L, the number of hash-grid levels to select.
    scale_step : geometric factor used to generate candidate resolutions.
    hash_size / hash_log2 : hash table size T. If omitted, collision weighting is disabled.
    upsample_mode : discrete implementation choice for comparing adjacent low-pass images.
                    "bilinear" is the default because it is the smoothest choice consistent
                    with the paper's low-pass interpretation.

    Returns
    -------
    res_sel : int64 array of length L containing the selected resolutions.
    growth  : float array of length L-1 with successive growth ratios.
    details : optional diagnostic dictionary.
    """
    if scale_step <= 1.0:
        raise ValueError(f"scale_step must be > 1, got {scale_step}")
    if num_levels < 2:
        raise ValueError("num_levels must be >= 2")
    if grid_dim < 1:
        raise ValueError("grid_dim must be >= 1")

    # Optional CPU-side pre-downscale for huge CPU arrays.
    img = _maybe_cpu_pre_downscale_uint8(img, target_min_dim=int(res_max))

    # Choose device.
    if isinstance(img, torch.Tensor):
        device = img.device
        if device.type == "cpu" and torch.cuda.is_available():
            device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = _to_bchw_torch(img, device=device)
    if x.shape[0] > 1:
        x = x[:1]
    if x.dtype != torch.uint8:
        x = x.clamp(0, 255).to(torch.uint8)

    _, _, H, W = x.shape
    min_dim = min(H, W)
    effective_res_max = int(min(res_max, min_dim))

    # Keep the working pyramid bounded.
    if min_dim > effective_res_max:
        s0 = float(min_dim) / float(effective_res_max)
        new_h = max(1, int(round(H / s0)))
        new_w = max(1, int(round(W / s0)))
        xf = x.to(pyramid_dtype)
        xf = _resize_cuda(xf, (new_h, new_w), mode="bilinear", antialias=True)
        x = torch.round(xf).clamp_(0, 255).to(torch.uint8)

    g_high = x.to(pyramid_dtype)
    blur_passes = 0 if int(bw_order) <= 0 else (1 if int(bw_order) <= 2 else 2)
    blur_ksize = 5 if blur_passes > 0 else 3

    candidate_res: list[float] = []
    candidate_band_info: list[float] = []

    if max_candidates is None:
        max_candidates = max(int(8 * num_levels), int(num_levels + 16))

    Hh, Wh = int(g_high.shape[-2]), int(g_high.shape[-1])
    res_high = float(min(Hh, Wh))

    # Walk from fine -> coarse and build discretized band masses.
    while (res_high > float(res_min)) and (len(candidate_res) < int(max_candidates)):
        Hn = max(1, int(round(Hh / float(scale_step))))
        Wn = max(1, int(round(Wh / float(scale_step))))
        if Hn == Hh and Wn == Wh:
            break

        g_low = downsample_binomial_aa_2d_cuda(
            g_high,
            out_hw=(Hn, Wn),
            k=blur_ksize,
            passes=blur_passes,
        )
        g_low_up = _resize_cuda(g_low, (Hh, Wh), mode=upsample_mode, antialias=False)
        residual = g_high - g_low_up
        h_res = compute_entropy_cuda_signed(
            residual,
            clip=int(residual_clip),
            max_samples_per_channel=max_samples_per_channel,
        )
        band_info = float(h_res) * (float(res_high) ** float(grid_dim))
        candidate_res.append(float(res_high))
        candidate_band_info.append(float(band_info))

        g_high = g_low
        Hh, Wh = Hn, Wn
        res_high = float(min(Hh, Wh))
        if res_high <= 2.0:
            break

    # Add the coarsest base band.
    h_base = compute_entropy_cuda_u8like(
        g_high,
        clip=clip,
        max_samples_per_channel=max_samples_per_channel,
    )
    base_info = float(h_base) * (float(res_high) ** float(grid_dim))
    candidate_res.append(float(res_high))
    candidate_band_info.append(float(base_info))

    resolutions_t = torch.tensor(candidate_res, device=device, dtype=torch.float32)
    band_info_t = torch.tensor(candidate_band_info, device=device, dtype=torch.float32)
    perm = torch.argsort(resolutions_t)
    resolutions_t = resolutions_t[perm]
    band_info_t = band_info_t[perm].clamp_min(0.0)

    if resolutions_t.numel() == 0:
        raise RuntimeError("No candidate resolutions were generated.")

    if resolutions_t.numel() < int(num_levels):
        raise ValueError(
            f"Exact RIM DP needs at least num_levels={num_levels} candidates, "
            f"but only {int(resolutions_t.numel())} were generated. "
            f"Try decreasing num_levels or increasing max_candidates / res_max."
        )

    if hash_log2 is not None:
        hash_size = 1 << int(hash_log2)

    if hash_size is not None:
        w_hash = _hash_weight(
            resolutions_t=resolutions_t,
            hash_size=int(hash_size),
            grid_dim=int(grid_dim),
            mode=hash_weight_mode,
        )
    else:
        w_hash = torch.ones_like(resolutions_t)

    psi = band_info_t.detach().cpu().numpy().astype(np.float64)
    w_np = w_hash.detach().cpu().numpy().astype(np.float64)

    idx_np = _solve_rim_dp_exact(psi=psi, w_end=w_np, num_levels=int(num_levels))

    res_sel = (
        resolutions_t[torch.from_numpy(idx_np).to(resolutions_t.device)]
        .round()
        .to(torch.int64)
        .detach()
        .cpu()
        .numpy()
    )
    res_sel = np.clip(res_sel, int(res_min), int(resolutions_t[-1].item()))
    res_sel = np.maximum.accumulate(res_sel)
    growth = res_sel[1:] / np.maximum(res_sel[:-1], 1)

    if return_details:
        P = np.zeros(len(psi) + 1, dtype=np.float64)
        P[1:] = np.cumsum(psi)
        k_idx_1based = idx_np + 1
        prev = 0
        band_loads = []
        for k in k_idx_1based:
            band_loads.append(float(w_np[k - 1] * (P[k] - P[prev])))
            prev = int(k)

        details: Dict[str, Any] = {
            "reason": "ok",
            "method": "paper_exact_residual_dp",
            "upsample_mode": upsample_mode,
            "candidate_resolutions": resolutions_t.detach().cpu().numpy(),
            "candidate_band_info": band_info_t.detach().cpu().numpy(),
            "candidate_hash_weight": w_hash.detach().cpu().numpy(),
            "selected_indices": idx_np,
            "selected_resolutions": res_sel,
            "selected_band_effective_loads": np.asarray(band_loads, dtype=np.float64),
            "objective_value": float(np.max(band_loads)),
        }
        return res_sel, growth, details

    return res_sel, growth


__all__ = [
    "get_entropy_grid",
    "compute_entropy_cuda_u8like",
    "compute_entropy_cuda_signed",
    "downsample_binomial_aa_2d_cuda",
    "binomial_blur_2d_cuda",
    "_hash_weight",
]
