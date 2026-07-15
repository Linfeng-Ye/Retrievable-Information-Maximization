from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch


def parse_pair(value: int | Sequence[int], *, name: str) -> Tuple[int, int]:
    if isinstance(value, int):
        out = (int(value), int(value))
    else:
        vals = [int(v) for v in value]
        if len(vals) != 2:
            raise ValueError(f"{name} must be an int or a pair, got {value!r}")
        out = (vals[0], vals[1])
    if out[0] <= 0 or out[1] <= 0:
        raise ValueError(f"{name} entries must be positive, got {out}")
    return out


def q_survival(alpha: torch.Tensor) -> torch.Tensor:
    """q(alpha)=(1-exp(-alpha))/alpha with a Taylor branch near zero."""
    alpha = alpha.to(torch.float64)
    small = alpha.abs() < 1e-6
    taylor = 1.0 - alpha / 2.0 + alpha.square() / 6.0
    exact = -torch.expm1(-alpha) / alpha.clamp_min(1e-300)
    return torch.where(small, taylor, exact)


def assert_strictly_increasing(values: Sequence[int], *, name: str = "values") -> None:
    vals = [int(v) for v in values]
    if any(v <= 0 for v in vals):
        raise ValueError(f"{name} must be positive, got {vals}")
    if any(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
        raise ValueError(f"{name} must be strictly increasing, got {vals}")


def init_partition(num_bands: int, num_levels: int, mode: str = "geometric") -> list[tuple[int, int]]:
    M = int(num_bands)
    L = int(num_levels)
    if M < L:
        raise ValueError(f"Need num_bands >= num_levels, got M={M}, L={L}")
    if mode not in {"geometric", "linear"}:
        raise ValueError(f"Unknown init_scheduler={mode!r}")
    if mode == "linear":
        cuts = [round(i * M / L) for i in range(L + 1)]
    else:
        if L == 1:
            cuts = [0, M]
        else:
            raw = torch.logspace(0.0, 1.0, steps=L + 1, base=2.0)
            raw = (raw - raw[0]) / (raw[-1] - raw[0])
            cuts = [int(round(float(x) * M)) for x in raw]
            cuts[0], cuts[-1] = 0, M
    for i in range(1, len(cuts)):
        min_allowed = cuts[i - 1] + 1
        remaining = L - i
        max_allowed = M - remaining
        cuts[i] = max(min_allowed, min(int(cuts[i]), max_allowed))
    return [(int(cuts[i]), int(cuts[i + 1] - 1)) for i in range(L)]


def validate_partition(partition: Sequence[tuple[int, int]], num_bands: int, num_levels: int) -> None:
    if len(partition) != int(num_levels):
        raise ValueError(f"Partition length mismatch: expected {num_levels}, got {len(partition)}")
    expected_s = 0
    for s, e in partition:
        if int(s) != expected_s:
            raise ValueError(f"Partition is not contiguous at segment {(s, e)}; expected start {expected_s}")
        if int(e) < int(s):
            raise ValueError(f"Empty partition segment {(s, e)}")
        expected_s = int(e) + 1
    if expected_s != int(num_bands):
        raise ValueError(f"Partition does not cover all bands: ended at {expected_s}, M={num_bands}")


def block_kappa(
    resolution: int,
    *,
    image_hw: tuple[int, int],
    block_size: tuple[int, int],
    num_blocks_h: int,
    num_blocks_w: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Approximate touched grid vertices for each non-overlapping image block."""
    R = int(resolution)
    H, W = int(image_hw[0]), int(image_hw[1])
    bh, bw = int(block_size[0]), int(block_size[1])
    vals: list[int] = []
    for by in range(int(num_blocks_h)):
        h = min(bh, max(0, H - by * bh))
        for bx in range(int(num_blocks_w)):
            w = min(bw, max(0, W - bx * bw))
            kh = math.ceil(R * h / max(1, H)) + 1
            kw = math.ceil(R * w / max(1, W)) + 1
            vals.append(max(1, kh * kw))
    return torch.tensor(vals, dtype=torch.float64, device=device)
