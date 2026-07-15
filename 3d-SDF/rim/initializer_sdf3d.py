from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .common3d import (
    assert_strictly_increasing,
    cube_kappa,
    geometric_resolutions,
    init_partition,
    validate_partition,
)
from .gate_solver import solve_gate_exact_fractional, solve_gate_prefix_fallback
from .scheduler_dp import evaluate_total_objective, solve_scheduler_dp_fixed_gates


@dataclass
class RIMSDFInitResult:
    partition: list[tuple[int, int]]
    resolutions: list[int]
    gates: torch.Tensor
    objective_history: list[float]
    details: dict[str, Any]


def _as_A(A: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A)
    if A.ndim != 2:
        raise ValueError(f"A must have shape [B,M], got {tuple(A.shape)}")
    A = A.to(torch.float64).clamp_min(0.0)
    if not bool(torch.isfinite(A).all()):
        raise ValueError("A contains non-finite values")
    return A


@torch.no_grad()
def initialize_rim_sdf3d(
    A: torch.Tensor | np.ndarray,
    candidate_resolutions: Sequence[int],
    *,
    num_levels: int,
    table_sizes: Sequence[int],
    cube_dims: Sequence[int],
    max_iters: int = 20,
    tol: float = 1e-6,
    init_scheduler: str = "geometric",
    gate_solver: str = "prefix_fallback",
    mode: str = "rim_full",
    base_resolution: int = 16,
    desired_resolution: int = 2048,
    device: str | torch.device = "cpu",
    verbose: bool = False,
    output_dir: str | None = None,
) -> RIMSDFInitResult:
    cand = [int(r) for r in candidate_resolutions]
    assert_strictly_increasing(cand, name="candidate_resolutions")
    L = int(num_levels)
    if int(max_iters) < 1 and mode in {"rim_resolution", "rim_full"}:
        raise ValueError(f"max_iters must be at least 1 for mode={mode}, got {max_iters}")
    if len(table_sizes) != L:
        raise ValueError(f"table_sizes length must equal num_levels={L}, got {len(table_sizes)}")
    if mode not in {"fixed_hashgrid", "rim_gate", "rim_resolution", "rim_full"}:
        raise ValueError(f"Unknown RIM mode: {mode}")

    dev = torch.device(device)
    A_global = _as_A(A).to(dev)
    B, M = int(A_global.shape[0]), int(A_global.shape[1])
    if M != len(cand):
        raise ValueError(f"A has M={M} bands but candidate_resolutions has {len(cand)} entries")
    if M < L and mode != "fixed_hashgrid":
        raise ValueError(f"Need M >= L for contiguous partition, got M={M}, L={L}")
    if B != int(np.prod([int(v) for v in cube_dims])):
        raise ValueError(f"A cube count {B} does not match cube_dims={tuple(cube_dims)}")

    kappa_cache: dict[int, torch.Tensor] = {}

    def kappa_fn(_ell: int, resolution: int) -> torch.Tensor:
        key = int(resolution)
        if key not in kappa_cache:
            kappa_cache[key] = cube_kappa(key, cube_dims=cube_dims, device=dev)
        return kappa_cache[key]

    # The scheduler objective must use the same hash-table budget as the actual
    # encoder. Inflating this internal table size makes q_survival nearly flat,
    # so the DP ignores collisions and chooses high resolutions from the first
    # level onward; the fixed-gate RIM-resolution run then loses the coarse
    # levels that make the INGP baseline converge quickly.
    collision_calibration = 1.0
    table_sizes_obj = [float(t) for t in table_sizes]

    if mode == "fixed_hashgrid":
        resolutions = geometric_resolutions(L, base_resolution, desired_resolution)
        partition = init_partition(max(M, L), L, mode="geometric") if M >= L else [(0, 0)] * L
        gates = torch.ones((L, B), dtype=torch.float32, device=dev)
        details = {
            "init_method": "fixed_hashgrid_geometric",
            "candidate_resolutions": cand,
            "table_sizes": [int(t) for t in table_sizes],
            "num_levels": L,
            "num_cubes": B,
            "cube_dims": [int(v) for v in cube_dims],
            "mode": mode,
        }
        return RIMSDFInitResult(partition, resolutions, gates.cpu(), [], details)

    partition = init_partition(M, L, mode=init_scheduler)
    validate_partition(partition, M, L)
    objective_history: list[float] = []
    solver = solve_gate_exact_fractional if gate_solver == "exact_fractional" else solve_gate_prefix_fallback
    if gate_solver not in {"exact_fractional", "prefix_fallback"}:
        raise ValueError(f"Unknown gate_solver={gate_solver!r}")

    def _partition_resolutions(part: Sequence[tuple[int, int]]) -> list[int]:
        return [int(cand[int(e)]) for _, e in part]

    if verbose:
        print(
            "[RIM-SDF] init setup: "
            f"cubes={B} dims={tuple(int(v) for v in cube_dims)}, bands={M}, levels={L}, "
            f"candidates={len(cand)}, max_iters={int(max_iters)}, tol={float(tol):.3g}, "
            f"scheduler={init_scheduler}, gate_solver={gate_solver}, mode={mode}, device={dev}",
            flush=True,
        )
        print(
            f"[RIM-SDF] init start partition={partition} resolutions={_partition_resolutions(partition)}",
            flush=True,
        )

    def update_gates(part: Sequence[tuple[int, int]]) -> torch.Tensor:
        rows = []
        for ell, (s, e) in enumerate(part):
            I = A_global[:, int(s) : int(e) + 1].sum(dim=1)
            kappa = kappa_fn(ell, cand[int(e)])
            rows.append(solver(I, kappa, int(table_sizes_obj[ell])).to(dev))
        return torch.stack(rows, dim=0)

    gates = torch.ones((L, B), dtype=torch.float32, device=dev)
    converged = False
    stop_reason = "max_iters"
    iterations_run = 0
    for _it in range(int(max_iters)):
        it = _it + 1
        iterations_run = it
        old_partition = list(partition)
        if mode in {"rim_gate", "rim_full"}:
            if verbose:
                print(f"[RIM-SDF] init iter {it}/{int(max_iters)}: solving gates", flush=True)
            gates = update_gates(partition)
        else:
            gates = torch.ones((L, B), dtype=torch.float32, device=dev)

        if verbose:
            gate_mean = float(gates.float().mean().item())
            gate_min = float(gates.float().min().item())
            gate_max = float(gates.float().max().item())
            print(
                f"[RIM-SDF] init iter {it}/{int(max_iters)}: "
                f"gate mean/min/max={gate_mean:.4f}/{gate_min:.4f}/{gate_max:.4f}; solving scheduler DP",
                flush=True,
            )

        if mode in {"rim_resolution", "rim_full"}:
            partition = solve_scheduler_dp_fixed_gates(A_global, gates, cand, table_sizes_obj, kappa_fn)
        obj = evaluate_total_objective(A_global, partition, gates, cand, table_sizes_obj, kappa_fn)
        obj_f = float(obj.item())
        if not torch.isfinite(obj):
            raise RuntimeError("RIM-SDF objective became non-finite")
        objective_history.append(obj_f)
        improvement = None if len(objective_history) < 2 else objective_history[-1] - objective_history[-2]
        if improvement is not None:
            numerical_tol = 1e-8 * max(1.0, abs(objective_history[-2]))
            if improvement < -numerical_tol:
                raise RuntimeError(f"RIM-SDF objective decreased unexpectedly: {objective_history}")
        if verbose:
            improvement_text = "n/a" if improvement is None else f"{improvement:.6g}"
            print(
                f"[RIM-SDF] init iter {it}/{int(max_iters)}: "
                f"objective={obj_f:.6g}, improvement={improvement_text}, "
                f"partition={partition}, resolutions={_partition_resolutions(partition)}",
                flush=True,
            )
        if partition == old_partition:
            converged = True
            stop_reason = "partition_unchanged"
            if verbose:
                print(f"[RIM-SDF] init converged at iter {it}: partition unchanged", flush=True)
            break
        if improvement is not None:
            threshold = float(tol) * max(1.0, abs(objective_history[-2]))
            if improvement < threshold:
                stop_reason = "objective_tolerance"
                if verbose:
                    print(
                        f"[RIM-SDF] init converged at iter {it}: "
                        f"improvement {improvement:.6g} < threshold {threshold:.6g}",
                        flush=True,
                    )
                break

    if verbose:
        print("[RIM-SDF] init final: recomputing gates for selected partition", flush=True)
    if mode in {"rim_gate", "rim_full"}:
        gates = update_gates(partition)
    else:
        gates = torch.ones((L, B), dtype=torch.float32, device=dev)
    final_obj = evaluate_total_objective(A_global, partition, gates, cand, table_sizes_obj, kappa_fn)
    objective_history.append(float(final_obj.item()))
    if len(objective_history) >= 2:
        numerical_tol = 1e-8 * max(1.0, abs(objective_history[-2]))
        if objective_history[-1] < objective_history[-2] - numerical_tol:
            raise RuntimeError(f"RIM-SDF objective decreased after final gate update: {objective_history}")

    # A hard iteration cap or tolerance stop can leave the final gates one step
    # ahead of the partition. Record that explicitly instead of silently calling
    # the initializer converged.
    final_partition_consistent = True
    if mode in {"rim_resolution", "rim_full"}:
        verified_partition = solve_scheduler_dp_fixed_gates(
            A_global, gates, cand, table_sizes_obj, kappa_fn
        )
        final_partition_consistent = verified_partition == partition
        converged = final_partition_consistent
        if converged and stop_reason != "partition_unchanged":
            stop_reason = "verified_fixed_point"
        if verbose and not final_partition_consistent:
            print(
                "[RIM-SDF] init warning: final gates would change the scheduler partition; "
                f"increase --rim_iters above {int(max_iters)}",
                flush=True,
            )
    resolutions = [cand[e] for _, e in partition]
    if mode == "rim_gate":
        resolutions = geometric_resolutions(L, base_resolution, desired_resolution)

    details = {
        "init_method": "iterative_fixed_gate_dp",
        "candidate_resolutions": cand,
        "table_sizes": [int(t) for t in table_sizes],
        "collision_calibration": float(collision_calibration),
        "table_sizes_for_objective": [float(t) for t in table_sizes_obj],
        "num_levels": L,
        "num_cubes": B,
        "cube_dims": [int(v) for v in cube_dims],
        "gate_solver": gate_solver,
        "init_scheduler": init_scheduler,
        "mode": mode,
        "objective_history": objective_history,
        "iterations_run": int(iterations_run),
        "converged": bool(converged),
        "stop_reason": stop_reason,
        "final_partition_consistent": bool(final_partition_consistent),
    }
    if verbose:
        print(
            f"[RIM-SDF] init final: objective={float(final_obj.item()):.6g}, "
            f"partition={partition}, resolutions={resolutions}",
            flush=True,
        )

    result = RIMSDFInitResult(
        partition=[(int(s), int(e)) for s, e in partition],
        resolutions=[int(r) for r in resolutions],
        gates=gates.detach().cpu(),
        objective_history=objective_history,
        details=details,
    )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "rim_cube_gates.npy"), result.gates.numpy().astype(np.float32))
        np.save(os.path.join(output_dir, "rim_gates_init.npy"), result.gates.numpy().astype(np.float32))
        with open(os.path.join(output_dir, "rim_selected_resolutions.json"), "w") as f:
            json.dump({"selected_resolutions": result.resolutions, "partition": result.partition}, f, indent=2)
        with open(os.path.join(output_dir, "rim_scheduler_stats.json"), "w") as f:
            json.dump(details, f, indent=2)
    return result
