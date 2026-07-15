#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""hash_encoder_2d.py

Fast 2D multi-resolution hash encoder (Instant-NGP style) with **per-level**
custom grid resolutions.

Why this exists (repo-local context)
----------------------------------
The original `train_ultra_image*.py` scripts implemented the encoder as a
Python loop over levels with one `nn.Embedding` per level. That makes it hard
to (a) specify an explicit resolution schedule `{N_l}` and (b) keep per-step
overhead low.

This module provides a drop-in encoder that:
* Accepts either geometric progression (base_resolution + growth_factor) OR an
  explicit per-level `resolutions=[N_0, ..., N_{L-1}]`.
* Vectorizes computation across levels to remove the Python loop.
* Stores all embedding tables in one contiguous parameter tensor for faster
  indexing and fewer kernel launches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import nn


def _parse_resolutions_arg(s: Optional[str]) -> Optional[List[int]]:
    """Parse CLI string for per-level resolutions.

    Accepts:
      * "16,32,64,128"
      * "[16, 32, 64, 128]"
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if s.startswith("["):
        # Safer than eval; supports Python literal lists.
        import ast

        out = ast.literal_eval(s)
        if not isinstance(out, (list, tuple)):
            raise ValueError("--resolutions must be a list/tuple literal")
        vals = [int(x) for x in out]
    else:
        parts = [p for p in s.replace(" ", "").split(",") if p]
        vals = [int(p) for p in parts]
    if len(vals) == 0:
        return None
    return vals


@dataclass(frozen=True)
class HashGrid2DConfig:
    levels: int = 16
    features: int = 2
    base_resolution: int = 16
    growth_factor: float = 1.5
    hash_size: int = 2**19
    resolutions: Optional[Sequence[int]] = None
    strict_increasing: bool = True


class MultiResolutionHashEncoder2D(nn.Module):
    """2D multi-resolution hash encoder with explicit per-level resolutions.

    Parameters
    ----------
    levels / base_resolution / growth_factor:
        Used only when `resolutions` is None.
    resolutions:
        Optional explicit list `[N_0, ..., N_{L-1}]` (grid resolution per level).
        If provided, `levels` is inferred as `len(resolutions)`.
    hash_size:
        Hash table size *per level* (T). Must be > 0.
    features:
        Feature dimension per level (F).
    strict_increasing:
        If True, enforces `N_l < N_{l+1}` to avoid degenerate repeated levels.
    """

    def __init__(
        self,
        *,
        levels: int = 16,
        features: int = 2,
        base_resolution: int = 16,
        growth_factor: float = 1.5,
        log_hash_size: int = 19,
        resolutions: Optional[Sequence[int]] = None,
        strict_increasing: bool = True,
        image_hw: Optional[Sequence[int]] = None,
        block_size: int = 0,
        level_block_mask: Optional[torch.Tensor] = None,
        level_block_gates: Optional[torch.Tensor] = None,
        gate_mode: str = "trainable_sigmoid",
        fallback_mode: str = "blockwise",
        fallback_init_scale: float = 1e-4,
        trainable_mask_init_logit: float = 4.0,
        trainable_gate_init_logit: Optional[float] = None,
        trainable_mask_temperature: float = 1.0,
        fixed_gate_threshold: float = 0.5,
    ) -> None:
        super().__init__()

        if log_hash_size <= 0:
            raise ValueError("hash_size must be positive")
        if features <= 0:
            raise ValueError("features must be positive")

        # Resolve per-level grid resolutions.
        if resolutions is not None:
            res_list = [int(r) for r in resolutions]
            if len(res_list) == 0:
                raise ValueError("resolutions must be a non-empty list")
            if any(r <= 0 for r in res_list):
                raise ValueError("all resolutions must be positive")
            # if strict_increasing and any(res_list[i] >= res_list[i + 1] for i in range(len(res_list) - 1)):
            #     raise ValueError("resolutions must be strictly increasing (N_l < N_{l+1})")
            self.levels = len(res_list)
            res_np = np.asarray(res_list, dtype=np.int64)
        else:
            self.levels = int(levels)
            if self.levels <= 0:
                raise ValueError("levels must be positive")
            if base_resolution <= 0:
                raise ValueError("base_resolution must be positive")
            if growth_factor <= 0:
                raise ValueError("growth_factor must be positive")

            res_np = np.round(base_resolution * (growth_factor ** np.arange(self.levels))).astype(np.int64)
            res_np = np.maximum(res_np, 1)
            if strict_increasing:
                # Make strictly increasing by enforcing N_{l+1} >= N_l + 1
                for i in range(1, len(res_np)):
                    if res_np[i] <= res_np[i - 1]:
                        res_np[i] = res_np[i - 1] + 1

        self.features = int(features)
        self.hash_size = int(2**log_hash_size)

        # Fast path: if T is a power of two, modulo can be replaced by bitmask.
        # (In these scripts hx is non-negative, so this matches `% T`.)
        is_pow2 = (self.hash_size & (self.hash_size - 1) == 0)
        self._use_bitmask = bool(is_pow2)
        if self._use_bitmask:
            self.register_buffer("hash_mask", torch.tensor(self.hash_size - 1, dtype=torch.int64))
        # Store resolutions as a buffer (int32 is enough for typical grids).
        self.register_buffer("resolutions", torch.from_numpy(res_np.astype(np.int32)))  # (L,)
        # One contiguous parameter tensor: (L, T, F)
        # This is equivalent to L separate nn.Embedding tables.
        table = torch.empty((self.levels, self.hash_size, self.features), dtype=torch.float32)
        nn.init.uniform_(table, a=-1e-4, b=1e-4)
        self.table = nn.Parameter(table)

        # Hash constants and corner offsets.
        self.register_buffer("primes", torch.tensor([1_540_863, 1_006_721], dtype=torch.int64))
        self.register_buffer("offsets", torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.int64))

        self.use_blockwise_selector = False
        self.block_size = int(block_size)
        self.image_hw = None
        self.num_blocks_y = 0
        self.num_blocks_x = 0
        self.trainable_mask_init_logit = float(trainable_mask_init_logit)
        self.trainable_mask_temperature = float(trainable_mask_temperature)
        if self.trainable_mask_temperature <= 0.0:
            raise ValueError("trainable_mask_temperature must be positive")
        self.gate_mode = str(gate_mode)
        self.fallback_mode = str(fallback_mode)
        if self.gate_mode not in {"trainable_sigmoid", "fixed_binary"}:
            raise ValueError("gate_mode must be trainable_sigmoid or fixed_binary")
        if self.fallback_mode not in {"blockwise", "global_shared"}:
            raise ValueError("fallback_mode must be blockwise or global_shared")

        if level_block_gates is not None and level_block_mask is not None:
            raise ValueError("Provide only one of level_block_gates or level_block_mask")
        gate_init_source = level_block_gates if level_block_gates is not None else level_block_mask
        if gate_init_source is not None:
            if image_hw is None:
                raise ValueError("image_hw must be provided when block gates are used")
            if self.block_size <= 0:
                raise ValueError("block_size must be positive when block gates are used")
            H_img, W_img = int(image_hw[0]), int(image_hw[1])
            if H_img <= 0 or W_img <= 0:
                raise ValueError("image_hw must be positive")
            gate_init = gate_init_source.to(dtype=torch.float32)
            if gate_init.ndim != 2:
                raise ValueError(f"block gate tensor must have shape (L, B), got {tuple(gate_init.shape)}")
            if int(gate_init.shape[0]) != self.levels:
                raise ValueError(
                    f"block gate first dim must equal levels={self.levels}, got {int(gate_init.shape[0])}"
                )
            gate_init = gate_init.clamp(0.0, 1.0).contiguous()
            mask = gate_init >= float(fixed_gate_threshold)
            self.register_buffer("level_block_mask_init", mask.contiguous())
            self.register_buffer("level_block_gate_init", gate_init.contiguous())
            self.image_hw = (H_img, W_img)
            self.num_blocks_y = int((H_img + self.block_size - 1) // self.block_size)
            self.num_blocks_x = int((W_img + self.block_size - 1) // self.block_size)
            expected_blocks = self.num_blocks_y * self.num_blocks_x
            if int(gate_init.shape[1]) != expected_blocks:
                raise ValueError(
                    f"block gate second dim must equal num_blocks={expected_blocks}, got {int(gate_init.shape[1])}"
                )
            if self.fallback_mode == "global_shared":
                fallback = torch.empty((self.levels, self.features), dtype=torch.float32)
            else:
                fallback = torch.empty((self.levels, expected_blocks, self.features), dtype=torch.float32)
            nn.init.uniform_(fallback, a=-float(fallback_init_scale), b=float(fallback_init_scale))
            self.fallback_table = nn.Parameter(fallback)

            if self.gate_mode == "trainable_sigmoid":
                if level_block_gates is None:
                    init_logit = float(self.trainable_mask_init_logit)
                    logits = torch.full((self.levels, expected_blocks), -init_logit, dtype=torch.float32)
                    logits = torch.where(mask, torch.full_like(logits, init_logit), logits)
                elif trainable_gate_init_logit is not None:
                    init_logit = abs(float(trainable_gate_init_logit))
                    logits = torch.full((self.levels, expected_blocks), -init_logit, dtype=torch.float32)
                    logits = torch.where(mask, torch.full_like(logits, init_logit), logits)
                else:
                    eps = 1e-4
                    g0 = gate_init.clamp(eps, 1.0 - eps)
                    logits = float(self.trainable_mask_temperature) * torch.log(g0 / (1.0 - g0))
                self.mask_logits = nn.Parameter(logits)
            else:
                self.register_buffer("gates", mask.to(torch.float32).contiguous())
            self.use_blockwise_selector = True

    def extra_repr(self) -> str:
        rs = self.resolutions.detach().cpu().tolist()
        extra = ""
        if self.use_blockwise_selector:
            extra = (
                f", block_size={self.block_size}, image_hw={self.image_hw}, "
                f"num_blocks={self.num_blocks_y * self.num_blocks_x}, gate_mode={self.gate_mode}, "
                f"fallback_mode={self.fallback_mode}, mask_temp={self.trainable_mask_temperature:g}"
            )
        return f"levels={self.levels}, features={self.features}, hash_size={self.hash_size}, resolutions={rs[:4]}{'...' if len(rs) > 4 else ''}{extra}"

    def _coords_to_block_ids(self, coords: torch.Tensor) -> torch.Tensor:
        if not self.use_blockwise_selector:
            raise RuntimeError("Blockwise selector is not enabled")
        c = coords
        if torch.min(c) < 0.0 or torch.max(c) > 1.0:
            c01 = (c + 1.0) * 0.5
        else:
            c01 = c
        H_img, W_img = int(self.image_hw[0]), int(self.image_hw[1])
        if W_img > 1:
            x = torch.round(c01[:, 0].clamp(0.0, 1.0) * float(W_img - 1)).to(torch.int64)
        else:
            x = torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.int64)
        if H_img > 1:
            y = torch.round(c01[:, 1].clamp(0.0, 1.0) * float(H_img - 1)).to(torch.int64)
        else:
            y = torch.zeros((coords.shape[0],), device=coords.device, dtype=torch.int64)
        bx = torch.clamp(x // self.block_size, min=0, max=self.num_blocks_x - 1)
        by = torch.clamp(y // self.block_size, min=0, max=self.num_blocks_y - 1)
        return by * self.num_blocks_x + bx

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode 2D coords in [0,1] into (B, L*F) features."""
        if coords.ndim != 2 or coords.shape[-1] != 2:
            raise ValueError(f"coords must have shape (B,2), got {tuple(coords.shape)}")

        # Ensure float dtype for interpolation math.
        if not torch.is_floating_point(coords):
            coords = coords.float()

        B = coords.shape[0]
        device = coords.device
        L = self.levels

        # (L,) in float for scaling.
        res_f = self.resolutions.to(device=device, dtype=coords.dtype)

        # uv: (L, B, 2)
        uv = coords.unsqueeze(0) * res_f.view(L, 1, 1)
        uv0 = torch.floor(uv).to(torch.int64)  # (L, B, 2)
        frac = uv - uv0.to(uv.dtype)          # (L, B, 2)

        # 4 corners: (L, B, 4, 2)
        idxs = uv0.unsqueeze(2) + self.offsets.view(1, 1, 4, 2)

        # Hash: (L, B, 4)
        hx = (idxs[..., 0] * self.primes[0]) ^ (idxs[..., 1] * self.primes[1])
        if self._use_bitmask:
            hashed = hx & self.hash_mask
        else:
            hashed = torch.remainder(hx, self.hash_size)

        # Gather: table is (L, T, F) => corner_feats (L, B, 4, F)
        lvl = torch.arange(L, device=device, dtype=torch.int64).view(L, 1, 1)
        corner_feats = self.table[lvl, hashed]  # advanced indexing

        # Bilinear weights: (L, B, 4, 1)
        fx = frac[..., 0:1]
        fy = frac[..., 1:2]
        w00 = (1 - fx) * (1 - fy)
        w10 = fx * (1 - fy)
        w01 = (1 - fx) * fy
        w11 = fx * fy
        w = torch.stack([w00, w10, w01, w11], dim=2)  # (L, B, 4, 1)

        # Interpolate corners: (L, B, F)
        feat = (corner_feats * w).sum(dim=2)

        if self.use_blockwise_selector:
            block_ids = self._coords_to_block_ids(coords)
            if self.gate_mode == "trainable_sigmoid":
                gate_logits = self.mask_logits[:, block_ids].unsqueeze(-1)
                gate = torch.sigmoid(gate_logits / float(self.trainable_mask_temperature))
            else:
                gate = self.gates[:, block_ids].unsqueeze(-1).to(dtype=feat.dtype)
            if self.fallback_mode == "global_shared":
                fallback = self.fallback_table[:, None, :].expand(-1, B, -1)
            else:
                fallback = self.fallback_table[:, block_ids, :]
            feat = fallback * (1.0 - gate) + feat * gate

        # (B, L*F)
        return feat.permute(1, 0, 2).reshape(B, L * self.features)

    def get_mask_gate_stats(self) -> Optional[dict[str, float]]:
        if not self.use_blockwise_selector:
            return None
        with torch.no_grad():
            if self.gate_mode == "trainable_sigmoid":
                gate = torch.sigmoid(self.mask_logits / float(self.trainable_mask_temperature))
            else:
                gate = self.gates
            stats: dict[str, float] = {
                "gate_mean": float(gate.mean().item()),
                "gate_min": float(gate.min().item()),
                "gate_max": float(gate.max().item()),
            }
            if hasattr(self, "level_block_mask_init"):
                init_mask = self.level_block_mask_init
                if bool(init_mask.any()):
                    stats["gate_mean_init_on"] = float(gate[init_mask].mean().item())
                if bool((~init_mask).any()):
                    stats["gate_mean_init_off"] = float(gate[~init_mask].mean().item())
            return stats


__all__ = [
    "MultiResolutionHashEncoder2D",
    "HashGrid2DConfig",
    "_parse_resolutions_arg",
]
