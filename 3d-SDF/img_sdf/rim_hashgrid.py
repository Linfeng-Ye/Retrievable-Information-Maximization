from __future__ import annotations

from typing import Sequence

import math
import numpy as np
import torch
from torch import nn


class RIMHashGrid3DEncoder(nn.Module):
    """3D multiresolution hash encoder with cube-wise RIM gates."""

    def __init__(
        self,
        *,
        levels: int,
        features: int,
        log_hash_size: int,
        resolutions: Sequence[int],
        cube_dims: Sequence[int],
        level_cube_gates: torch.Tensor | np.ndarray | None = None,
        gate_mode: str = "trainable_sigmoid",
        fallback_mode: str = "blockwise",
        fallback_init_scale: float = 1e-4,
        gate_init_logit: float = 2.5,
    ) -> None:
        super().__init__()
        if features <= 0:
            raise ValueError("features must be positive")
        self.levels = int(levels)
        self.features = int(features)
        log_hash_size_f = float(log_hash_size)
        if log_hash_size_f < 0:
            raise ValueError(f"log_hash_size must be non-negative, got {log_hash_size}")
        self.hash_size = int(math.floor((2.0 ** log_hash_size_f) + 0.5))
        if self.hash_size <= 0:
            raise ValueError(f"log_hash_size={log_hash_size} gives invalid hash_size={self.hash_size}")
        self.n_output_dims = self.levels * self.features
        self.output_dim = self.n_output_dims
        self.gate_mode = str(gate_mode)
        if self.gate_mode not in {"trainable_sigmoid", "fixed"}:
            raise ValueError("gate_mode must be trainable_sigmoid or fixed")
        self.fallback_mode = str(fallback_mode)
        if self.fallback_mode not in {"blockwise", "global_shared"}:
            raise ValueError("fallback_mode must be blockwise or global_shared")

        res_list = [int(r) for r in resolutions]
        if len(res_list) != self.levels:
            raise ValueError(f"resolutions length {len(res_list)} must equal levels={self.levels}")
        if any(r <= 0 for r in res_list):
            raise ValueError(f"all resolutions must be positive, got {res_list}")
        self.register_buffer("resolutions", torch.tensor(res_list, dtype=torch.int32))

        self.cube_dims = tuple(int(v) for v in cube_dims)
        if len(self.cube_dims) != 3 or any(v <= 0 for v in self.cube_dims):
            raise ValueError(f"cube_dims must be three positive ints, got {cube_dims}")
        self.num_cubes = int(np.prod(self.cube_dims))

        table = torch.empty((self.levels, self.hash_size, self.features), dtype=torch.float32)
        nn.init.uniform_(table, a=-1e-4, b=1e-4)
        self.table = nn.Parameter(table)
        if self.fallback_mode == "blockwise":
            fallback = torch.empty((self.levels, self.num_cubes, self.features), dtype=torch.float32)
        else:
            fallback = torch.empty((self.levels, self.features), dtype=torch.float32)
        nn.init.uniform_(fallback, a=-float(fallback_init_scale), b=float(fallback_init_scale))
        self.fallback_table = nn.Parameter(fallback)

        if level_cube_gates is None:
            gate_init = torch.ones((self.levels, self.num_cubes), dtype=torch.float32)
        else:
            gate_init = torch.as_tensor(level_cube_gates, dtype=torch.float32)
        if tuple(gate_init.shape) != (self.levels, self.num_cubes):
            raise ValueError(
                f"level_cube_gates must have shape {(self.levels, self.num_cubes)}, got {tuple(gate_init.shape)}"
            )
        gate_init = gate_init.clamp(0.0, 1.0).contiguous()
        self.register_buffer("level_cube_gate_init", gate_init)
        if self.gate_mode == "trainable_sigmoid":
            # Initialize logits at a moderate fixed magnitude derived from the solver's
            # on/off decision (gate_init is binary from solve_gate_prefix_fallback), not
            # from clamping the raw value to eps=1e-4 (logit ~= +-9.2). That saturates
            # sigmoid'(logit) to ~1e-4 and freezes gradients for the rest of training --
            # verified empirically (gate_grad_norm ~1e-6, no measurable movement over
            # 1000+ real training steps). +-2.5 (sigmoid'(2.5) ~= 0.07, ~700x larger)
            # matches the working reference gate init (RIM 2D: GATE_INIT_LOGIT=2.5),
            # whose gates demonstrably keep moving throughout training.
            init_logit = float(gate_init_logit)
            on_mask = gate_init >= 0.5
            logits = torch.where(
                on_mask,
                torch.full_like(gate_init, init_logit),
                torch.full_like(gate_init, -init_logit),
            )
            self.gate_logits = nn.Parameter(logits)
        else:
            self.register_buffer("fixed_gates", gate_init)

        is_pow2 = self.hash_size & (self.hash_size - 1) == 0
        self._use_bitmask = bool(is_pow2)
        if self._use_bitmask:
            self.register_buffer("hash_mask", torch.tensor(self.hash_size - 1, dtype=torch.int64))
        self.register_buffer("primes", torch.tensor([1_540_863, 1_006_721, 1_250_159], dtype=torch.int64))
        self.register_buffer(
            "offsets",
            torch.tensor(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1],
                ],
                dtype=torch.int64,
            ),
        )

    def extra_repr(self) -> str:
        rs = self.resolutions.detach().cpu().tolist()
        return (
            f"levels={self.levels}, features={self.features}, hash_size={self.hash_size}, "
            f"resolutions={rs[:4]}{'...' if len(rs) > 4 else ''}, "
            f"cube_dims={self.cube_dims}, gate_mode={self.gate_mode}, "
            f"fallback_mode={self.fallback_mode}"
        )

    def current_gates(self) -> torch.Tensor:
        if self.gate_mode == "trainable_sigmoid":
            return torch.sigmoid(self.gate_logits)
        return self.fixed_gates

    def _coords_to_cube_ids(self, coords: torch.Tensor) -> torch.Tensor:
        c = coords
        if torch.min(c) < 0.0 or torch.max(c) > 1.0:
            c01 = (c + 1.0) * 0.5
        else:
            c01 = c
        dims = torch.tensor(self.cube_dims, device=coords.device, dtype=coords.dtype)
        cube = torch.floor(c01.clamp(0.0, 1.0 - 1e-7) * dims).to(torch.int64)
        cube[:, 0].clamp_(0, self.cube_dims[0] - 1)
        cube[:, 1].clamp_(0, self.cube_dims[1] - 1)
        cube[:, 2].clamp_(0, self.cube_dims[2] - 1)
        return cube[:, 0] * (self.cube_dims[1] * self.cube_dims[2]) + cube[:, 1] * self.cube_dims[2] + cube[:, 2]

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[-1] != 3:
            raise ValueError(f"coords must have shape (B,3), got {tuple(coords.shape)}")
        if not torch.is_floating_point(coords):
            coords = coords.float()
        coords = coords.clamp(0.0, 1.0)
        B = coords.shape[0]
        device = coords.device
        L = self.levels

        res_f = self.resolutions.to(device=device, dtype=coords.dtype)
        uvw = coords.unsqueeze(0) * res_f.view(L, 1, 1)
        uvw0 = torch.floor(uvw).to(torch.int64)
        frac = uvw - uvw0.to(uvw.dtype)
        idxs = uvw0.unsqueeze(2) + self.offsets.view(1, 1, 8, 3)

        hx = (idxs[..., 0] * self.primes[0]) ^ (idxs[..., 1] * self.primes[1]) ^ (idxs[..., 2] * self.primes[2])
        if self._use_bitmask:
            hashed = hx & self.hash_mask
        else:
            hashed = torch.remainder(hx, self.hash_size)

        lvl = torch.arange(L, device=device, dtype=torch.int64).view(L, 1, 1)
        corner_feats = self.table[lvl, hashed]
        fx = frac[..., 0:1]
        fy = frac[..., 1:2]
        fz = frac[..., 2:3]
        weights = torch.stack(
            [
                (1 - fx) * (1 - fy) * (1 - fz),
                fx * (1 - fy) * (1 - fz),
                (1 - fx) * fy * (1 - fz),
                fx * fy * (1 - fz),
                (1 - fx) * (1 - fy) * fz,
                fx * (1 - fy) * fz,
                (1 - fx) * fy * fz,
                fx * fy * fz,
            ],
            dim=2,
        )
        feat = (corner_feats * weights).sum(dim=2)

        cube_ids = self._coords_to_cube_ids(coords)
        gates = self.current_gates().to(device=device, dtype=feat.dtype)[:, cube_ids].unsqueeze(-1)
        if self.fallback_mode == "blockwise":
            fallback = self.fallback_table[:, cube_ids, :]
        else:
            fallback = self.fallback_table[:, None, :].expand(-1, B, -1)
        fallback = fallback.to(dtype=feat.dtype)
        feat = fallback * (1.0 - gates) + feat * gates
        return feat.permute(1, 0, 2).reshape(B, L * self.features)

    def gate_regularization(self) -> torch.Tensor:
        return self.current_gates().mean()

    def get_gate_stats(self) -> dict[str, object]:
        with torch.no_grad():
            gate = self.current_gates().detach().float()
            per_level_mean = gate.mean(dim=1)
            per_level_min = gate.min(dim=1).values
            per_level_max = gate.max(dim=1).values
            per_level_on = (gate > 0.5).float().mean(dim=1)
            return {
                "gate_mean": float(gate.mean().item()),
                "gate_min": float(gate.min().item()),
                "gate_max": float(gate.max().item()),
                "gate_frac_gt_0p5": float((gate > 0.5).float().mean().item()),
                "gate_mean_per_level": [float(v) for v in per_level_mean.cpu()],
                "gate_min_per_level": [float(v) for v in per_level_min.cpu()],
                "gate_max_per_level": [float(v) for v in per_level_max.cpu()],
                "gate_frac_gt_0p5_per_level": [float(v) for v in per_level_on.cpu()],
                "effective_active_hash_capacity": self.effective_active_hash_capacity(),
            }

    def effective_active_hash_capacity(self) -> float:
        gate = self.current_gates().detach().float().cpu()
        resolutions = self.resolutions.detach().cpu().tolist()
        total = 0.0
        for ell, r in enumerate(resolutions):
            total += min(float(self.hash_size), float(int(r)) ** 3) * float(gate[ell].mean().item())
        return float(total * self.features)

    def gate_grad_norm(self) -> float | None:
        if not hasattr(self, "gate_logits") or self.gate_logits.grad is None:
            return None
        return float(self.gate_logits.grad.detach().norm().item())

    def gates_numpy(self) -> np.ndarray:
        return self.current_gates().detach().cpu().numpy().astype(np.float32)

    def gate_logits_numpy(self) -> np.ndarray | None:
        if not hasattr(self, "gate_logits"):
            return None
        return self.gate_logits.detach().cpu().numpy().astype(np.float32)
