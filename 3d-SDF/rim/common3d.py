from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch


def q_survival(alpha: torch.Tensor) -> torch.Tensor:
    """q(alpha)=(1-exp(-alpha))/alpha with q(0)=1."""
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


def parse_candidate_resolutions(
    candidate_resolutions: str | None,
    num_candidates: int,
    base_resolution: int,
    desired_resolution: int,
    num_levels: int | None = None,
) -> list[int]:
    if candidate_resolutions:
        vals = [int(v.strip()) for v in candidate_resolutions.split(",") if v.strip()]
    else:
        if int(num_candidates) <= 1:
            vals = [int(desired_resolution)]
        else:
            vals = np.rint(
                np.linspace(int(base_resolution), int(desired_resolution), int(num_candidates), dtype=np.float64)
            ).astype(np.int64).tolist()
        if num_levels is not None:
            # Keep the INGP geometric ladder in the candidate set. Without these
            # low-resolution candidates the scheduler cannot choose the baseline
            # coarse-to-fine allocation, so "RIM resolution only" is biased toward
            # overloaded high-resolution hash levels from the start.
            vals.extend(geometric_resolutions(int(num_levels), int(base_resolution), int(desired_resolution)))
    vals = [max(2, int(v)) for v in vals]
    vals = sorted(set(vals))
    assert_strictly_increasing(vals, name="candidate_resolutions")
    return vals


def geometric_resolutions(num_levels: int, base_resolution: int, desired_resolution: int) -> list[int]:
    if int(num_levels) <= 1:
        return [int(desired_resolution)]
    vals = np.geomspace(int(base_resolution), int(desired_resolution), int(num_levels))
    out = [max(2, int(round(v))) for v in vals]
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1
    return out


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


def cube_kappa(
    resolution: int,
    *,
    cube_dims: Sequence[int],
    device: torch.device | str,
) -> torch.Tensor:
    """Approximate touched trilinear grid vertices for each non-overlapping cube."""
    R = int(resolution)
    dx, dy, dz = [max(1, int(v)) for v in cube_dims]
    kx = math.ceil(R / dx) + 1
    ky = math.ceil(R / dy) + 1
    kz = math.ceil(R / dz) + 1
    val = max(1, int(kx * ky * kz))
    return torch.full((dx * dy * dz,), float(val), dtype=torch.float64, device=device)
