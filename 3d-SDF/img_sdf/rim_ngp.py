from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from termcolor import colored

import util_misc
from img_sdf.ngp import MertricEmbedding
from img_sdf.rim_hashgrid import RIMHashGrid3DEncoder
from img_sdf.siren import SirenLayer, Sine


class modulator(nn.Module):
    def __init__(self, in_dim, out_dim, use_bias=True, w0=30.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.act_fn = Sine(w0=w0)

    def forward(self, x):
        return self.act_fn(self.linear(x))


class RIMSDFNGP(torch.nn.Module):
    """MetricGrids-style SDF network backed by RIMHashGrid3DEncoder."""

    def __init__(
        self,
        cmin,
        cmax,
        *,
        selected_resolutions: Sequence[int],
        cube_dims: Sequence[int],
        gates,
        gate_mode: str = "trainable_sigmoid",
        fallback_mode: str = "blockwise",
        num_layers=5,
        hidden_dim=64,
        in_dim=3,
        out_dim=1,
        num_levels=15,
        level_dim=2,
        log2_hashmap_size=15,
        gate_init_logit=2.5,
    ):
        super().__init__()
        if in_dim != 3:
            raise ValueError("RIMSDFNGP only supports 3D SDF coordinates")
        if not isinstance(cmin, list):
            cmin = [cmin] * in_dim
        if not isinstance(cmax, list):
            cmax = [cmax] * in_dim
        self.register_buffer("cmin", torch.tensor(cmin, dtype=torch.float32))
        self.register_buffer("cmax", torch.tensor(cmax, dtype=torch.float32))

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.n_levels = int(num_levels)
        self.F = int(level_dim)
        self.num_layers = int(num_layers)
        self.dim_hidden = int(hidden_dim)
        self.encoder = RIMHashGrid3DEncoder(
            levels=num_levels,
            features=level_dim,
            log_hash_size=log2_hashmap_size,
            resolutions=selected_resolutions,
            cube_dims=cube_dims,
            level_cube_gates=gates,
            gate_mode=gate_mode,
            fallback_mode=fallback_mode,
            gate_init_logit=gate_init_logit,
        )
        self.embedding = MertricEmbedding(self.in_dim)

        self.w0 = 30.0
        self.use_bias = True
        self.omegas = 30
        layers = []
        for ind in range(self.num_layers - 1):
            is_first = ind == 0
            layer_in_dim = self.encoder.n_output_dims if is_first else self.dim_hidden
            layers.append(
                SirenLayer(
                    in_dim=layer_in_dim,
                    out_dim=self.dim_hidden,
                    w0=self.w0,
                    use_bias=self.use_bias,
                    is_first=is_first,
                )
            )
        self.net = nn.Sequential(*layers)
        self.last_layer = SirenLayer(
            in_dim=self.dim_hidden,
            out_dim=self.out_dim,
            w0=self.w0,
            use_bias=self.use_bias,
            is_last=True,
        )
        self.modulators = nn.ModuleList(
            [
                modulator(
                    in_dim=self.in_dim if i == 0 else self.encoder.n_output_dims,
                    out_dim=self.dim_hidden,
                    w0=self.omegas,
                )
                for i in range(self.num_layers - 1)
            ]
        )
        print(colored("INFO:RIM SDF Model Information", "cyan", None, ["bold"]))
        print(self)

    def count_params(self):
        decoder = [
            a + b + c
            for a, b, c in zip(
                util_misc.count_parameters(self.net),
                util_misc.count_parameters(self.modulators),
                util_misc.count_parameters(self.last_layer),
            )
        ]
        hash_grid = np.array([self.encoder.table.numel(), 0, self.encoder.table.numel()])
        fallback = np.array([self.encoder.fallback_table.numel(), 0, self.encoder.fallback_table.numel()])
        gate_n = self.encoder.gate_logits.numel() if hasattr(self.encoder, "gate_logits") else 0
        gates = np.array([gate_n, 0, gate_n])
        params = {
            "hash_grid": hash_grid,
            "decoder": decoder,
            "fallback": fallback,
            "gates": gates,
            "effective_active_hash_capacity": np.array([0, 0, self.encoder.effective_active_hash_capacity()]),
        }
        params["total"] = sum(v for k, v in params.items() if k != "effective_active_hash_capacity")
        for k, v in params.items():
            print(f"{k} params: {v}")
        return params

    def gate_regularization(self):
        return self.encoder.gate_regularization()

    def get_gate_stats(self):
        return self.encoder.get_gate_stats()

    def gate_grad_norm(self):
        return self.encoder.gate_grad_norm()

    def gates_numpy(self):
        return self.encoder.gates_numpy()

    def gate_logits_numpy(self):
        return self.encoder.gate_logits_numpy()

    # MertricEmbedding folds each coordinate into 3 branches with native ranges
    # (0, 1), (-1, 0), (-1, -0.5) (see img_sdf/ngp.py). The original MetricGrids
    # NGP decoder feeds those straight into a tcnn hash grid, which hashes
    # unbounded voxel coords without clamping. RIMHashGrid3DEncoder.forward()
    # clamps its input to [0, 1], so branches 2 and 3 would otherwise collapse
    # to the same clamped corner and carry no signal -- each branch's known
    # analytic range is affinely folded into [0, 1] here first to keep the
    # 3-branch embedding meaningful under RIM's stricter encoder contract.
    _EMBEDDING_BRANCH_RANGES = ((0.0, 1.0), (-1.0, 0.0), (-1.0, -0.5))

    def forward(self, x, step=None, exp_name=None, **kwargs):
        out_other = {}
        x_norm = x / self.cmax.flip(-1)[None]
        x_flatten = ((x_norm.reshape(-1, self.in_dim)) + 1.0) / 2.0
        x_flatten = x_flatten.clamp(0.0, 1.0)

        coord_pe = self.embedding(x_norm.reshape(-1, self.in_dim))
        encoded_outputs = torch.stack([
            self.encoder(((coord_pe[i] - lo) / (hi - lo)).clamp(0.0, 1.0))
            for i, (lo, hi) in enumerate(self._EMBEDDING_BRANCH_RANGES)
        ], dim=1)
        encoded = torch.sum(encoded_outputs, dim=1).float()
        encoded_outputs = encoded_outputs.permute(1, 0, 2).contiguous()

        x_dec = encoded
        for i, layer in enumerate(self.net):
            if i == 0:
                modulate = self.modulators[i](x_flatten)
            elif encoded_outputs.shape[0] >= i:
                modulate = self.modulators[i](encoded_outputs[i - 1].float())
            else:
                modulate = self.modulators[i](encoded_outputs[-1].float())
            backbone = layer(x_dec)
            x_dec = (modulate * modulate) * backbone
        out = self.last_layer(x_dec)
        return out, out_other
