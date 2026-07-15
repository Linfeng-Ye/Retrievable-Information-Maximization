from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pysdf
import torch
import torch.nn.functional as F


@dataclass
class SDF3DInfoMetadata:
    analysis_resolution: int
    cube_size: int
    cube_dims: tuple[int, int, int]
    candidate_resolutions: list[int]
    analysis_band_resolutions: list[int]
    num_cubes: int
    sdf_min: float
    sdf_max: float
    sdf_mean_abs: float
    information_metric: str


def _candidate_to_analysis_resolutions(candidates: list[int], analysis_resolution: int) -> list[int]:
    if len(candidates) == 1:
        return [analysis_resolution]
    if len(candidates) > analysis_resolution - 1:
        raise ValueError(
            "Need analysis_resolution >= number of candidate resolutions + 1 "
            f"to build nonempty residual bands, got analysis_resolution={analysis_resolution}, "
            f"num_candidates={len(candidates)}"
        )
    finest = max(2, max(candidates))
    scale = float(analysis_resolution) / float(finest)
    raw = [int(round(max(2, r) * scale)) for r in candidates]

    # Residual bands are computed as consecutive low-pass differences. If two
    # candidates map to the same analysis resolution, the later residual is
    # exactly zero and the scheduler will never choose that candidate. Keep the
    # proportional map but reserve one distinct analysis resolution per band.
    out: list[int] = []
    num = len(raw)
    for i, val in enumerate(raw):
        min_allowed = 2 + i
        max_allowed = int(analysis_resolution) - (num - 1 - i)
        val = int(np.clip(val, min_allowed, max_allowed))
        if out and val <= out[-1]:
            val = out[-1] + 1
        out.append(val)
    return out


def _sdf_on_grid(mesh, resolution: int, chunk_size: int, device: torch.device) -> torch.Tensor:
    sdf_fn = pysdf.SDF(mesh.vertices, mesh.faces)
    xs = torch.linspace(-1.0, 1.0, resolution)
    grid = torch.stack(torch.meshgrid(xs, xs, xs, indexing="ij"), dim=-1).reshape(-1, 3)
    sdf = torch.empty((grid.shape[0], 1), dtype=torch.float32)
    for start in range(0, grid.shape[0], chunk_size):
        pts = grid[start : start + chunk_size].numpy()
        sdf[start : start + chunk_size, 0] = torch.from_numpy(-sdf_fn(pts).astype(np.float32))
    # pysdf's mesh-distance query is CPU-only (no GPU backend); everything downstream
    # of this point (pooling, interpolation, quantization, entropy) moves to `device`.
    return sdf.reshape(resolution, resolution, resolution).to(device)


def _pad_to_cube_grid(volume: torch.Tensor, cube_size: int) -> tuple[torch.Tensor, tuple[int, int, int]]:
    cube_dims = tuple(int(np.ceil(s / cube_size)) for s in volume.shape[-3:])
    padded_shape = tuple(v * cube_size for v in cube_dims)
    pad = []
    for size, padded in reversed(list(zip(volume.shape[-3:], padded_shape))):
        pad.extend([0, padded - size])
    return F.pad(volume[None, None], pad, mode="replicate")[0, 0], cube_dims


def _cube_energy(volume: torch.Tensor, cube_size: int) -> torch.Tensor:
    padded, cube_dims = _pad_to_cube_grid(volume, cube_size)
    pooled = F.avg_pool3d(padded[None, None], kernel_size=cube_size, stride=cube_size)[0, 0]
    return pooled.reshape(-1)


def _cube_entropy(volume: torch.Tensor, cube_size: int, bins: int = 256, eps: float = 1e-12) -> torch.Tensor:
    """Empirical entropy per cube: quantize the (high-resolution) residual into `bins`
    levels by round(), then take the Shannon entropy of each cube's empirical histogram."""
    device = volume.device
    padded, cube_dims = _pad_to_cube_grid(volume, cube_size)
    v_min = padded.min()
    v_max = padded.max()
    num_cubes = int(np.prod(cube_dims))
    if float((v_max - v_min).abs().item()) < 1e-12:
        return torch.zeros(num_cubes, dtype=torch.float32, device=device)
    q = ((padded - v_min) * ((bins - 1) / (v_max - v_min))).round().clamp(0, bins - 1).to(torch.int64)
    cx, cy, cz = cube_dims
    q = q.view(cx, cube_size, cy, cube_size, cz, cube_size).permute(0, 2, 4, 1, 3, 5)
    q = q.reshape(num_cubes, cube_size * cube_size * cube_size)

    # Vectorized per-cube histogram via a single scatter_add over all cubes at once --
    # a Python loop over cubes would serialize into thousands of tiny kernel launches
    # and defeat the point of running this on the GPU.
    cube_ids = torch.arange(num_cubes, device=device).unsqueeze(1).expand_as(q)
    flat_idx = cube_ids.reshape(-1) * bins + q.reshape(-1)
    counts = torch.zeros(num_cubes * bins, dtype=torch.float32, device=device)
    counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
    counts = counts.view(num_cubes, bins)
    probs = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    return -(probs * torch.log2(probs.clamp_min(eps))).sum(dim=1)


def _save_debug_slices(info_volume: torch.Tensor, output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    mids = [s // 2 for s in info_volume.shape]
    for axis, slc in [
        ("x", info_volume[mids[0], :, :]),
        ("y", info_volume[:, mids[1], :]),
        ("z", info_volume[:, :, mids[2]]),
    ]:
        plt.figure(figsize=(4, 4))
        plt.imshow(slc.detach().cpu().numpy(), cmap="magma")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(output_dir, f"rim_info_slice_{axis}.png"), dpi=160)
        plt.close()


def estimate_sdf3d_information(
    mesh,
    analysis_resolution: int,
    cube_size: int,
    candidate_resolutions: Iterable[int],
    output_dir: str | None = None,
    save_debug: bool = False,
    chunk_size: int = 262144,
    information_metric: str = "entropy",
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Estimate 3D cube-band information from multiscale SDF residuals.

    Default is empirical (quantized-histogram) entropy of each level's residual,
    matching RIM's 2D block-entropy scheme. Residual energy (with a zero-level-set
    prior) is kept available as an alternative metric via `information_metric`.
    """
    R = int(analysis_resolution)
    cube_size = int(cube_size)
    candidates = sorted(set(int(v) for v in candidate_resolutions))
    if R < 4:
        raise ValueError("--rim_analysis_resolution must be at least 4")
    if cube_size < 1:
        raise ValueError("--rim_cube_size must be positive")
    if information_metric not in {"energy", "entropy", "energy_entropy"}:
        raise ValueError("information_metric must be energy, entropy, or energy_entropy")

    dev = torch.device(device)
    sdf_grid = _sdf_on_grid(mesh, R, int(chunk_size), dev)
    grid_5d = sdf_grid[None, None]
    surface_tau = max(2.0 / R, 0.015)
    surface_weight = torch.exp(-sdf_grid.abs() / surface_tau)
    band_resolutions = _candidate_to_analysis_resolutions(candidates, R)
    prev_low = None
    cube_values = []
    info_volume = torch.zeros_like(sdf_grid)

    for band_res in band_resolutions:
        low_small = F.adaptive_avg_pool3d(grid_5d, output_size=(band_res, band_res, band_res))
        low = F.interpolate(low_small, size=(R, R, R), mode="trilinear", align_corners=True)[0, 0]
        residual = low if prev_low is None else low - prev_low
        prev_low = low
        weighted_energy = residual.pow(2) * (1.0 + 2.0 * surface_weight) + 1e-3 * surface_weight
        if information_metric == "energy":
            cube_band = _cube_energy(weighted_energy, cube_size)
        elif information_metric == "entropy":
            cube_band = _cube_entropy(residual, cube_size)
        else:
            cube_band = _cube_energy(weighted_energy, cube_size) * (1.0 + _cube_entropy(residual, cube_size))
        cube_values.append(cube_band)
        info_volume += weighted_energy

    A = torch.stack(cube_values, dim=1).cpu().numpy().astype(np.float32)
    cube_dims = tuple(int(np.ceil(R / cube_size)) for _ in range(3))
    metadata = SDF3DInfoMetadata(
        analysis_resolution=R,
        cube_size=cube_size,
        cube_dims=cube_dims,
        candidate_resolutions=candidates,
        analysis_band_resolutions=band_resolutions,
        num_cubes=int(np.prod(cube_dims)),
        sdf_min=float(sdf_grid.min().item()),
        sdf_max=float(sdf_grid.max().item()),
        sdf_mean_abs=float(sdf_grid.abs().mean().item()),
        information_metric=information_metric,
    )
    stats = {
        "A_min": float(A.min()),
        "A_max": float(A.max()),
        "A_mean": float(A.mean()),
        "A_nonzero_fraction": float((A > 0).mean()),
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "rim_info_A.npy"), A)
        np.save(os.path.join(output_dir, "rim_info_volume.npy"), info_volume.cpu().numpy().astype(np.float32))
        with open(os.path.join(output_dir, "rim_info_metadata.json"), "w") as f:
            json.dump(asdict(metadata), f, indent=2)
        if save_debug:
            _save_debug_slices(info_volume, output_dir)
    return {
        "A": A,
        "metadata": asdict(metadata),
        "candidate_resolutions": candidates,
        "analysis_band_resolutions": band_resolutions,
        "stats": stats,
    }
