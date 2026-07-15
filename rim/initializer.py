from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .common import assert_strictly_increasing, block_kappa, init_partition, parse_pair, validate_partition
from .gate_solver import solve_gate_exact_fractional, solve_gate_prefix_fallback
from .scheduler_dp import evaluate_total_objective, solve_scheduler_dp_fixed_gates


@dataclass
class RimInitResult:
    partition: list[tuple[int, int]]
    resolutions: list[int]
    gates: torch.Tensor
    objective_history: list[float]
    details: dict[str, Any]


def _as_global_A(A: torch.Tensor) -> torch.Tensor:
    if A.ndim == 3:
        A = A.mean(dim=0)
    if A.ndim != 2:
        raise ValueError(f"A must have shape [B,M] or [P,B,M], got {tuple(A.shape)}")
    A = A.to(torch.float64).clamp_min(0.0)
    if not bool(torch.isfinite(A).all()):
        raise ValueError("A contains non-finite values")
    return A


@torch.no_grad()
def initialize_rim_iterative(
    A: torch.Tensor,
    candidate_resolutions: Sequence[int],
    *,
    num_levels: int,
    table_sizes: Sequence[int],
    image_hw: tuple[int, int],
    block_size: int | Sequence[int],
    num_blocks_h: int,
    num_blocks_w: int,
    max_iters: int = 20,
    tol: float = 1e-6,
    init_scheduler: str = "geometric",
    gate_solver: str = "exact_fractional",
    device: str | torch.device = "cpu",
    verbose: bool = False,
) -> RimInitResult:
    cand = [int(r) for r in candidate_resolutions]
    assert_strictly_increasing(cand, name="candidate_resolutions")
    L = int(num_levels)
    if len(table_sizes) != L:
        raise ValueError(f"table_sizes length must equal num_levels={L}, got {len(table_sizes)}")

    dev = torch.device(device)
    A_global = _as_global_A(A).to(dev)
    B, M = int(A_global.shape[0]), int(A_global.shape[1])
    if M != len(cand):
        raise ValueError(f"A has M={M} bands but candidate_resolutions has {len(cand)} entries")
    if M < L:
        raise ValueError(f"Need M >= L for contiguous partition, got M={M}, L={L}")
    if B != int(num_blocks_h) * int(num_blocks_w):
        raise ValueError(f"A block count {B} does not match layout {num_blocks_h}x{num_blocks_w}")

    block_hw = parse_pair(block_size, name="block_size")
    kappa_cache: dict[int, torch.Tensor] = {}

    def kappa_fn(_ell: int, resolution: int) -> torch.Tensor:
        key = int(resolution)
        if key not in kappa_cache:
            kappa_cache[key] = block_kappa(
                int(resolution),
                image_hw=image_hw,
                block_size=block_hw,
                num_blocks_h=int(num_blocks_h),
                num_blocks_w=int(num_blocks_w),
                device=dev,
            )
        return kappa_cache[key]

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
            "[B-RIM] init setup: "
            f"blocks={B} ({int(num_blocks_h)}x{int(num_blocks_w)}), "
            f"bands={M}, levels={L}, candidates={len(cand)}, "
            f"max_iters={int(max_iters)}, tol={float(tol):.3g}, "
            f"scheduler={init_scheduler}, gate_solver={gate_solver}, device={dev}",
            flush=True,
        )
        print(
            f"[B-RIM] init start partition={partition} resolutions={_partition_resolutions(partition)}",
            flush=True,
        )

    def update_gates(part: Sequence[tuple[int, int]]) -> torch.Tensor:
        rows = []
        for ell, (s, e) in enumerate(part):
            I = A_global[:, int(s) : int(e) + 1].sum(dim=1)
            kappa = kappa_fn(ell, cand[int(e)])
            rows.append(solver(I, kappa, int(table_sizes[ell])).to(dev))
        return torch.stack(rows, dim=0)

    gates = torch.ones((L, B), dtype=torch.float32, device=dev)
    for _it in range(int(max_iters)):
        it = _it + 1
        old_partition = list(partition)
        if verbose:
            print(f"[B-RIM] init iter {it}/{int(max_iters)}: solving gates", flush=True)
        gates = update_gates(partition)
        if verbose:
            gate_mean = float(gates.float().mean().item())
            gate_min = float(gates.float().min().item())
            gate_max = float(gates.float().max().item())
            print(
                f"[B-RIM] init iter {it}/{int(max_iters)}: "
                f"gate mean/min/max={gate_mean:.4f}/{gate_min:.4f}/{gate_max:.4f}; solving scheduler DP",
                flush=True,
            )
        partition = solve_scheduler_dp_fixed_gates(A_global, gates, cand, table_sizes, kappa_fn)
        obj = evaluate_total_objective(A_global, partition, gates, cand, table_sizes, kappa_fn)
        obj_f = float(obj.item())
        if not torch.isfinite(obj):
            raise RuntimeError("B-RIM objective became non-finite")
        objective_history.append(obj_f)
        improvement = None
        if len(objective_history) >= 2:
            improvement = objective_history[-1] - objective_history[-2]
        if verbose:
            improvement_text = "n/a" if improvement is None else f"{improvement:.6g}"
            print(
                f"[B-RIM] init iter {it}/{int(max_iters)}: "
                f"objective={obj_f:.6g}, improvement={improvement_text}, "
                f"partition={partition}, resolutions={_partition_resolutions(partition)}",
                flush=True,
            )
        if partition == old_partition:
            if verbose:
                print(f"[B-RIM] init converged at iter {it}: partition unchanged", flush=True)
            break
        if improvement is not None:
            threshold = float(tol) * max(1.0, abs(objective_history[-2]))
            if improvement < threshold:
                if verbose:
                    print(
                        f"[B-RIM] init converged at iter {it}: "
                        f"improvement {improvement:.6g} < threshold {threshold:.6g}",
                        flush=True,
                    )
                break

    if verbose:
        print("[B-RIM] init final: recomputing gates for selected partition", flush=True)
    gates = update_gates(partition)
    final_obj = evaluate_total_objective(A_global, partition, gates, cand, table_sizes, kappa_fn)
    objective_history.append(float(final_obj.item()))
    if any(objective_history[i + 1] + 1e-8 < objective_history[i] for i in range(len(objective_history) - 1)):
        # The alternating method can occasionally tie-shuffle due to fixed-gate DP,
        # but a material drop usually means a shape or numerical bug.
        raise RuntimeError(f"B-RIM objective decreased unexpectedly: {objective_history}")
    resolutions = [cand[e] for _, e in partition]
    if verbose:
        print(
            f"[B-RIM] init final: objective={float(final_obj.item()):.6g}, "
            f"partition={partition}, resolutions={resolutions}",
            flush=True,
        )
    details = {
        "init_method": "iterative_fixed_gate_dp",
        "candidate_resolutions": cand,
        "table_sizes": [int(t) for t in table_sizes],
        "num_levels": L,
        "num_blocks": B,
        "num_blocks_h": int(num_blocks_h),
        "num_blocks_w": int(num_blocks_w),
        "block_size": list(block_hw),
        "image_hw": [int(image_hw[0]), int(image_hw[1])],
        "gate_solver": gate_solver,
        "init_scheduler": init_scheduler,
    }
    return RimInitResult(
        partition=[(int(s), int(e)) for s, e in partition],
        resolutions=[int(r) for r in resolutions],
        gates=gates.detach().cpu(),
        objective_history=objective_history,
        details=details,
    )


def load_block_info_for_training(path: str, *, expected_block_size: int, expected_hw: tuple[int, int]) -> dict[str, Any]:
    info = torch.load(path, map_location="cpu")
    A = info.get("A")
    if not torch.is_tensor(A):
        raise ValueError(f"{path} does not contain tensor key 'A'")
    if not bool(torch.isfinite(A).all()):
        raise ValueError("Block info A contains non-finite entries")
    if float(A.min().item()) < -1e-8:
        raise ValueError("Block info A contains negative entries")
    block_size = info.get("block_size")
    if isinstance(block_size, (list, tuple)):
        if int(block_size[0]) != int(expected_block_size) or int(block_size[1]) != int(expected_block_size):
            raise ValueError(f"Block size mismatch: info has {block_size}, training expects {expected_block_size}")
    elif int(block_size) != int(expected_block_size):
        raise ValueError(f"Block size mismatch: info has {block_size}, training expects {expected_block_size}")
    block_stride = info.get("block_stride", block_size)
    if isinstance(block_stride, (list, tuple)):
        stride_vals = [int(block_stride[0]), int(block_stride[1])]
    else:
        stride_vals = [int(block_stride), int(block_stride)]
    if stride_vals != [int(expected_block_size), int(expected_block_size)]:
        raise ValueError(
            "Training currently supports non-overlapping B-RIM blocks only; "
            f"info has block_stride={block_stride}, expected {expected_block_size}."
        )
    patch_size = info.get("patch_size")
    if patch_size is not None:
        if isinstance(patch_size, (list, tuple)):
            info_hw = (int(patch_size[0]), int(patch_size[1]))
        else:
            info_hw = (int(patch_size), int(patch_size))
        if tuple(info_hw) != tuple(int(v) for v in expected_hw):
            raise ValueError(f"Block info patch_size/HW {info_hw} does not match training grid {expected_hw}")
    return info
