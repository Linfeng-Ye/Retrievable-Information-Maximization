from __future__ import annotations

from typing import Callable, Sequence

import torch

from .common import q_survival, validate_partition


KappaFn = Callable[[int, int], torch.Tensor]


@torch.no_grad()
def segment_score_fixed_gate(
    A: torch.Tensor,
    gates_l: torch.Tensor,
    s: int,
    e: int,
    level: int,
    resolution: int,
    table_size: int,
    kappa_fn: KappaFn,
) -> torch.Tensor:
    I = A[:, int(s) : int(e) + 1].sum(dim=1)
    kappa = kappa_fn(int(level), int(resolution)).to(device=A.device, dtype=torch.float64)
    g = gates_l.to(device=A.device, dtype=torch.float64).clamp(0.0, 1.0)
    load = (kappa * g).sum() / max(1.0, float(table_size))
    return q_survival(load) * (g * I.to(torch.float64)).sum()


@torch.no_grad()
def solve_scheduler_dp_fixed_gates(
    A: torch.Tensor,
    gates: torch.Tensor,
    candidate_resolutions: Sequence[int],
    table_sizes: Sequence[int],
    kappa_fn: KappaFn,
) -> list[tuple[int, int]]:
    A64 = A.to(torch.float64)
    if A64.ndim != 2:
        raise ValueError(f"A must have shape [B,M], got {tuple(A64.shape)}")
    L, B = int(gates.shape[0]), int(gates.shape[1])
    if B != int(A64.shape[0]):
        raise ValueError(f"Gate block count {B} does not match A blocks {int(A64.shape[0])}")
    M = int(A64.shape[1])
    if M < L:
        raise ValueError(f"Need at least L={L} bands, got M={M}")

    score = torch.full((L, M, M), -float("inf"), dtype=torch.float64, device=A64.device)
    prefix_starts = torch.arange(M, device=A64.device)
    valid_intervals = prefix_starts[:, None] <= prefix_starts[None, :]
    for ell in range(L):
        g = gates[ell].to(device=A64.device, dtype=torch.float64).clamp(0.0, 1.0)
        band_info = (A64 * g[:, None]).sum(dim=0)
        prefix_info = torch.cat([band_info.new_zeros(1), torch.cumsum(band_info, dim=0)])
        interval_info = prefix_info[1:][None, :] - prefix_info[:-1][:, None]

        kappa_by_e = torch.stack(
            [
                kappa_fn(int(ell), int(candidate_resolutions[e])).to(device=A64.device, dtype=torch.float64)
                for e in range(M)
            ],
            dim=0,
        )
        loads = (kappa_by_e * g[None, :]).sum(dim=1) / max(1.0, float(table_sizes[ell]))
        score_ell = interval_info * q_survival(loads)[None, :]
        score[ell] = torch.where(valid_intervals, score_ell, score_ell.new_full((), -float("inf")))

    D = torch.full((L, M), -float("inf"), dtype=torch.float64, device=A64.device)
    Prev = torch.full((L, M), -1, dtype=torch.long, device=A64.device)
    for e in range(0, M - (L - 1)):
        D[0, e] = score[0, 0, e]

    for ell in range(1, L):
        min_e = ell
        max_e = M - (L - ell)
        for e in range(min_e, max_e + 1):
            k_vals = torch.arange(ell - 1, e, device=A64.device)
            vals = D[ell - 1, k_vals] + score[ell, k_vals + 1, e]
            best_idx = torch.argmax(vals)
            D[ell, e] = vals[best_idx]
            Prev[ell, e] = k_vals[best_idx]

    partition: list[tuple[int, int]] = []
    e = M - 1
    for ell in range(L - 1, -1, -1):
        k = int(Prev[ell, e].item())
        s = 0 if ell == 0 else k + 1
        partition.append((s, e))
        e = k
    partition.reverse()
    validate_partition(partition, M, L)
    return partition


@torch.no_grad()
def evaluate_total_objective(
    A: torch.Tensor,
    partition: Sequence[tuple[int, int]],
    gates: torch.Tensor,
    candidate_resolutions: Sequence[int],
    table_sizes: Sequence[int],
    kappa_fn: KappaFn,
) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64, device=A.device)
    for ell, (s, e) in enumerate(partition):
        total = total + segment_score_fixed_gate(
            A,
            gates[ell],
            int(s),
            int(e),
            ell,
            int(candidate_resolutions[int(e)]),
            int(table_sizes[ell]),
            kappa_fn,
        )
    return total
