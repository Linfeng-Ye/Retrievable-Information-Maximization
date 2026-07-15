from __future__ import annotations

import torch

from .common3d import q_survival


@torch.no_grad()
def solve_gate_prefix_fallback(
    information: torch.Tensor,
    kappa: torch.Tensor,
    table_size: int,
) -> torch.Tensor:
    """RIM-style prefix gate solver for max q(sum k g / T) * sum g I."""
    I = information.to(torch.float64).clamp_min(0.0)
    K = kappa.to(device=I.device, dtype=torch.float64).clamp_min(1e-12)
    if I.ndim != 1 or K.shape != I.shape:
        raise ValueError(f"information and kappa must be vectors with same shape, got {I.shape} and {K.shape}")
    B = int(I.numel())
    if B == 0:
        raise ValueError("Cannot solve gates for zero cubes")
    if float(I.sum().item()) <= 0.0:
        return torch.zeros_like(I, dtype=torch.float32)

    density = I / K
    order = torch.argsort(density, descending=True, stable=True)
    I_s = I[order]
    K_s = K[order]
    prefix_I = torch.cat([I_s.new_zeros(1), torch.cumsum(I_s, dim=0)])
    prefix_K = torch.cat([K_s.new_zeros(1), torch.cumsum(K_s, dim=0)])
    alpha = prefix_K / max(1.0, float(table_size))
    scores = q_survival(alpha) * prefix_I
    scores = torch.nan_to_num(scores, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))
    best_n = int(torch.argmax(scores).item())
    gates_sorted = torch.zeros(B, dtype=torch.float64, device=I.device)
    if best_n > 0:
        gates_sorted[:best_n] = 1.0
    gates = torch.zeros_like(gates_sorted)
    gates[order] = gates_sorted
    return gates.to(torch.float32)


def solve_gate_exact_fractional(information: torch.Tensor, kappa: torch.Tensor, table_size: int) -> torch.Tensor:
    # The 2D RIM repo exposes this name while currently using the robust prefix
    # solver. Keep the same compatibility point for the 3D port.
    return solve_gate_prefix_fallback(information, kappa, table_size)

