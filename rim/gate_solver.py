from __future__ import annotations

import torch

from .common import q_survival


@torch.no_grad()
def solve_gate_prefix_fallback(
    information: torch.Tensor,
    kappa: torch.Tensor,
    table_size: int,
) -> torch.Tensor:
    """Robust prefix solver for max q(sum k g / T) * sum g I.

    The solution is binary over density-sorted blocks. This keeps the API ready
    for a later exact fractional boundary solver while avoiding NaNs and edge
    cases in the initial implementation.
    """
    I = information.to(torch.float64).clamp_min(0.0)
    K = kappa.to(device=I.device, dtype=torch.float64).clamp_min(1e-12)
    if I.ndim != 1 or K.shape != I.shape:
        raise ValueError(f"information and kappa must be vectors with same shape, got {I.shape}, {K.shape}")
    B = int(I.numel())
    if B == 0:
        raise ValueError("Cannot solve gates for zero blocks")
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
    """Compatibility entry point; currently uses the robust prefix solver."""
    return solve_gate_prefix_fallback(information, kappa, table_size)
