from __future__ import annotations

import math
import sys
from typing import Sequence

import torch
import torch.nn.functional as F

from .common import assert_strictly_increasing, parse_pair

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional for library use
    tqdm = None


def _downsample_antialiased_staged(
    x: torch.Tensor,
    size: tuple[int, int],
    *,
    max_downsample_factor: float = 8.0,
) -> torch.Tensor:
    """Downsample with antialiasing in stages to avoid CUDA shared-memory limits."""
    target_h, target_w = int(size[0]), int(size[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target size {size}")
    y = x
    factor = float(max_downsample_factor)
    if factor <= 1.0:
        raise ValueError("max_downsample_factor must be > 1")
    while int(y.shape[-2]) > target_h * factor or int(y.shape[-1]) > target_w * factor:
        cur_h, cur_w = int(y.shape[-2]), int(y.shape[-1])
        next_h = max(target_h, int(math.ceil(cur_h / factor)))
        next_w = max(target_w, int(math.ceil(cur_w / factor)))
        y = F.interpolate(
            y,
            size=(next_h, next_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    if (int(y.shape[-2]), int(y.shape[-1])) != (target_h, target_w):
        y = F.interpolate(
            y,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return y


def lowpass_resize(
    x: torch.Tensor,
    resolution: int | Sequence[int],
    out_hw: tuple[int, int],
    *,
    max_downsample_factor: float = 8.0,
) -> torch.Tensor:
    if isinstance(resolution, int):
        size = (int(resolution), int(resolution))
    else:
        vals = [int(v) for v in resolution]
        if len(vals) != 2:
            raise ValueError(f"resolution must be int or pair, got {resolution!r}")
        size = (vals[0], vals[1])
    y = _downsample_antialiased_staged(x, size, max_downsample_factor=max_downsample_factor)
    return F.interpolate(y, size=out_hw, mode="bilinear", align_corners=False)


def _quantize_band(band: torch.Tensor, *, first_band: bool) -> tuple[torch.Tensor, int]:
    """Map band values to nonnegative histogram bin ids.

    Base image band:
      value range [0, 255] maps directly to bins [0, 255].

    Residual bands:
      signed values [-255, 255] are shifted to bins [0, 510], so residual
      value 0 intentionally maps to bin 255. This is not integer overflow.
    """
    if first_band:
        return band.round().clamp(0.0, 255.0).to(torch.int16), 256
    signed = band.round().clamp(-255.0, 255.0).to(torch.int16)
    return (signed + 255).to(torch.int16), 511


def _entropy_from_quantized_blocks(q: torch.Tensor, bins: int, eps: float = 1e-12) -> torch.Tensor:
    # q: [N, B, C, P]
    # breakpoint()
    N, B, C, P = q.shape
    out = torch.zeros((N, B), dtype=torch.float32, device=q.device)
    ones = torch.ones((N * B, P), dtype=torch.float32, device=q.device)
    for c in range(C):
        vals = q[:, :, c, :].reshape(N * B, P).to(torch.int64)
        counts = torch.zeros((N * B, int(bins)), dtype=torch.float32, device=q.device)
        counts.scatter_add_(1, vals, ones)
        probs = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        ent = -(probs * torch.log2(probs.clamp_min(eps))).sum(dim=1)
        out += ent.reshape(N, B)
    return out


def _entropy_from_quantized_tiled(
    q: torch.Tensor,
    bins: int,
    *,
    block_size: tuple[int, int],
    block_stride: tuple[int, int],
    num_blocks_h: int,
    num_blocks_w: int,
    chunk_size: int,
) -> torch.Tensor:
    """Compute block entropy without materializing the full unfolded image."""
    if chunk_size <= 0:
        raise ValueError(f"entropy block chunk size must be positive, got {chunk_size}")
    N, C = int(q.shape[0]), int(q.shape[1])
    bh, bw = int(block_size[0]), int(block_size[1])
    sh, sw = int(block_stride[0]), int(block_stride[1])
    num_blocks = int(num_blocks_h * num_blocks_w)
    out = torch.empty((N, num_blocks), dtype=torch.float32, device=q.device)
    for start in range(0, num_blocks, int(chunk_size)):
        end = min(num_blocks, start + int(chunk_size))
        blocks = []
        for b in range(start, end):
            row = b // num_blocks_w
            col = b - row * num_blocks_w
            top = row * sh
            left = col * sw
            blocks.append(q[:, :, top:top + bh, left:left + bw])
        chunk = torch.stack(blocks, dim=1).reshape(N, end - start, C, bh * bw)
        out[:, start:end] = _entropy_from_quantized_blocks(chunk, bins)
    return out


def block_entropy(
    band: torch.Tensor,
    *,
    block_size: int | Sequence[int],
    block_stride: int | Sequence[int] | None = None,
    first_band: bool = False,
    unfold_dtype: torch.dtype | None = None,
    entropy_block_chunk_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, int | list[int]]]:
    """Return entropy mass [N,B] for a BCHW band tensor."""
    if band.ndim != 4:
        raise ValueError(f"band must have shape [N,C,H,W], got {tuple(band.shape)}")
    bh, bw = parse_pair(block_size, name="block_size")
    sh, sw = (bh, bw) if block_stride is None else parse_pair(block_stride, name="block_stride")
    N, C, H, W = [int(v) for v in band.shape]
    num_blocks_h = 1 + max(0, math.ceil((H - bh) / sh)) if H > bh else 1
    num_blocks_w = 1 + max(0, math.ceil((W - bw) / sw)) if W > bw else 1
    cover_h = (num_blocks_h - 1) * sh + bh
    cover_w = (num_blocks_w - 1) * sw + bw
    pad_h = max(0, cover_h - H)
    pad_w = max(0, cover_w - W)
    x = F.pad(band, (0, pad_w, 0, pad_h), mode="constant", value=0.0) if (pad_h or pad_w) else band
    q, bins = _quantize_band(x, first_band=first_band)
    meta = {
        "block_size": [bh, bw],
        "block_stride": [sh, sw],
        "num_blocks_h": int(num_blocks_h),
        "num_blocks_w": int(num_blocks_w),
        "num_blocks": int(num_blocks_h * num_blocks_w),
    }
    if entropy_block_chunk_size is not None:
        ent = _entropy_from_quantized_tiled(
            q,
            bins,
            block_size=(bh, bw),
            block_stride=(sh, sw),
            num_blocks_h=int(num_blocks_h),
            num_blocks_w=int(num_blocks_w),
            chunk_size=int(entropy_block_chunk_size),
        )
        return ent, meta
    if unfold_dtype is None:
        unfold_dtype = torch.float16 if q.device.type == "cuda" else torch.float32
    # F.unfold needs floating input. Float16 exactly represents all bin ids up to 510.
    q_float = q.reshape(N * C, 1, x.shape[-2], x.shape[-1]).to(unfold_dtype)
    patches = F.unfold(q_float, kernel_size=(bh, bw), stride=(sh, sw))
    patches = patches.round().to(torch.int16).reshape(N, C, bh * bw, num_blocks_h * num_blocks_w)
    patches = patches.permute(0, 3, 1, 2).contiguous()
    return _entropy_from_quantized_blocks(patches, bins), meta


@torch.no_grad()
def compute_block_info_batch(
    images_bchw: torch.Tensor,
    candidate_resolutions: Sequence[int],
    *,
    block_size: int | Sequence[int],
    block_stride: int | Sequence[int] | None = None,
    include_highpass_tail: bool = False,
    max_downsample_factor: float = 8.0,
    lowpass_channelwise: bool = False,
    entropy_block_chunk_size: int | None = None,
    show_progress: bool = False,
    progress_desc: str = "B-RIM bands",
) -> tuple[torch.Tensor, dict[str, int | list[int]]]:
    """Compute A_batch with shape [N, B, M] on the input tensor device."""
    if images_bchw.ndim != 4:
        raise ValueError(f"images_bchw must have shape [N,C,H,W], got {tuple(images_bchw.shape)}")
    cand = [int(r) for r in candidate_resolutions]
    assert_strictly_increasing(cand, name="candidate_resolutions")
    if torch.is_floating_point(images_bchw):
        x = images_bchw
        if x.dtype == torch.float64:
            x = x.to(torch.float32)
        if x.device.type != "cuda" and x.dtype == torch.float16:
            x = x.to(torch.float32)
    else:
        x = images_bchw.to(torch.float32)
    if x.max() <= 1.5:
        x = x * 255.0
    out_hw = (int(x.shape[-2]), int(x.shape[-1]))

    if lowpass_channelwise:
        C = int(x.shape[1])
        num_bands = len(cand) + (1 if include_highpass_tail else 0)
        masses: list[torch.Tensor | None] = [None] * num_bands
        layout_meta = None
        total_steps = C * num_bands
        progress_bar = None
        if show_progress and tqdm is not None:
            progress_bar = tqdm(total=total_steps, desc=progress_desc, unit="chan-band", leave=True, file=sys.stdout)
        for c in range(C):
            x_ch = x[:, c:c + 1, :, :]
            prev_low = None
            for j, r in enumerate(cand):
                if progress_bar is not None:
                    progress_bar.set_postfix_str(f"c={c + 1}/{C}, r={r}")
                elif show_progress:
                    print(
                        f"{progress_desc}: channel {c + 1}/{C} band {j + 1}/{num_bands} resolution={r}",
                        flush=True,
                    )
                curr_low = lowpass_resize(
                    x_ch,
                    r,
                    out_hw,
                    max_downsample_factor=float(max_downsample_factor),
                )
                band = curr_low if prev_low is None else curr_low - prev_low
                ent, meta = block_entropy(
                    band,
                    block_size=block_size,
                    block_stride=block_stride,
                    first_band=(j == 0),
                    unfold_dtype=(torch.float16 if x.device.type == "cuda" else torch.float32),
                    entropy_block_chunk_size=entropy_block_chunk_size,
                )
                masses[j] = ent if masses[j] is None else masses[j] + ent
                layout_meta = meta
                prev_low = curr_low
                if progress_bar is not None:
                    progress_bar.update(1)
            if include_highpass_tail:
                if prev_low is None:
                    raise ValueError("include_highpass_tail requires at least one candidate resolution")
                tail_idx = len(cand)
                ent, meta = block_entropy(
                    x_ch - prev_low,
                    block_size=block_size,
                    block_stride=block_stride,
                    first_band=False,
                    unfold_dtype=(torch.float16 if x.device.type == "cuda" else torch.float32),
                    entropy_block_chunk_size=entropy_block_chunk_size,
                )
                masses[tail_idx] = ent if masses[tail_idx] is None else masses[tail_idx] + ent
                layout_meta = meta
                if progress_bar is not None:
                    progress_bar.set_postfix_str(f"c={c + 1}/{C}, tail")
                    progress_bar.update(1)
                elif show_progress:
                    print(f"{progress_desc}: channel {c + 1}/{C} highpass tail", flush=True)
        if progress_bar is not None:
            progress_bar.close()
        if any(m is None for m in masses):
            raise RuntimeError("Channelwise low-pass precompute did not produce all bands")
        A = torch.stack([m for m in masses if m is not None], dim=-1).clamp_min(0.0)
        return A, dict(layout_meta or {})

    masses = []
    layout_meta = None
    prev_low = None
    iterator = list(enumerate(cand))
    progress_bar = None
    if show_progress and tqdm is not None:
        progress_bar = tqdm(iterator, desc=progress_desc, unit="band", leave=True, file=sys.stdout)
        iterator = progress_bar
    for j, r in iterator:
        if progress_bar is not None:
            progress_bar.set_postfix_str(f"r={r}")
        if show_progress and tqdm is None:
            print(f"{progress_desc}: band {j + 1}/{len(cand)} resolution={r}", flush=True)
        curr_low = lowpass_resize(
            x,
            r,
            out_hw,
            max_downsample_factor=float(max_downsample_factor),
        )
        band = curr_low if prev_low is None else curr_low - prev_low
        ent, meta = block_entropy(
            band,
            block_size=block_size,
            block_stride=block_stride,
            first_band=(j == 0),
            unfold_dtype=(torch.float16 if x.device.type == "cuda" else torch.float32),
            entropy_block_chunk_size=entropy_block_chunk_size,
        )
        masses.append(ent)
        layout_meta = meta
        prev_low = curr_low
    if include_highpass_tail:
        if prev_low is None:
            raise ValueError("include_highpass_tail requires at least one candidate resolution")
        ent, meta = block_entropy(
            x - prev_low,
            block_size=block_size,
            block_stride=block_stride,
            first_band=False,
            unfold_dtype=(torch.float16 if x.device.type == "cuda" else torch.float32),
            entropy_block_chunk_size=entropy_block_chunk_size,
        )
        masses.append(ent)
        layout_meta = meta
    A = torch.stack(masses, dim=-1).clamp_min(0.0)
    return A, dict(layout_meta or {})
