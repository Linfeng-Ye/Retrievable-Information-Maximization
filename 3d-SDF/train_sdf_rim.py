#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pprint
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import trimesh

import img_sdf.utils as utils
import util_misc
from img_sdf.provider import SDFDataset
from img_sdf.rim_ngp import RIMSDFNGP
from rim.common3d import geometric_resolutions, parse_candidate_resolutions
from rim.information_sdf3d import estimate_sdf3d_information
from rim.initializer_sdf3d import initialize_rim_sdf3d


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def normalize_mesh_to_unit_sphere(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    mesh = mesh.copy()
    vs = np.asarray(mesh.vertices, dtype=np.float64)
    vmin = vs.min(0)
    vmax = vs.max(0)
    center = (vmin + vmax) / 2.0
    shifted = vs - center[None, :]
    radius = float(np.sqrt(np.sum(shifted**2, axis=-1)).max())
    scale = 1.0 if radius <= 0.0 else 0.99 / radius
    mesh.vertices = shifted * scale
    return mesh, {
        "source_bounds_min": vmin.tolist(),
        "source_bounds_max": vmax.tolist(),
        "center": center.tolist(),
        "scale": float(scale),
        "target_domain": "[-1,1]^3 unit sphere radius 0.99",
    }


def log2_hashmap_size_to_table_size(value: float | int | str) -> int:
    value_f = float(value)
    if value_f < 0:
        raise ValueError(f"--log2_hashmap_size must be non-negative, got {value}")
    table_size = int(math.floor((2.0 ** value_f) + 0.5))
    if table_size <= 0:
        raise ValueError(f"--log2_hashmap_size={value} gives invalid table size {table_size}")
    return table_size


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MetricGrids SDF with fixed hash-grid or RIM-Hash initialization.")
    p.add_argument("--mesh", "--path", dest="mesh", required=True, help="Input .ply/.obj mesh path.")
    p.add_argument("--alias", default=None)
    p.add_argument("--out_dir", default="outputs/sdf_rim")
    p.add_argument("--encoder", default="rim_full", choices=["fixed_hashgrid", "rim_gate", "rim_resolution", "rim_full"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--ds_device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mesh_already_normalized", action="store_true")

    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=2**18)
    p.add_argument("--train_epoch_size", type=int, default=1000)
    p.add_argument("--train_presample", action="store_true")
    p.add_argument("--val_resolution", type=int, default=1024)
    p.add_argument("--val_num_samples", type=int, default=2**16)
    p.add_argument("--clip_sdf", type=float, default=None)
    p.add_argument("--loss", default="mape", choices=["mape", "l1", "mse"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.set_defaults(eval_mesh_metrics=True, save_pred=True, save_ckpt=True)
    p.add_argument("--eval_mesh_metrics", dest="eval_mesh_metrics", action="store_true")
    p.add_argument("--no_eval_mesh_metrics", dest="eval_mesh_metrics", action="store_false")
    p.add_argument("--save_pred", dest="save_pred", action="store_true")
    p.add_argument("--no_save_pred", dest="save_pred", action="store_false")
    p.add_argument("--save_ckpt", dest="save_ckpt", action="store_true")
    p.add_argument("--no_save_ckpt", dest="save_ckpt", action="store_false")

    p.add_argument("--num_levels", type=int, default=15)
    p.add_argument("--feature_dim", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--base_resolution", type=int, default=16)
    p.add_argument("--desired_resolution", type=int, default=2048)
    p.add_argument("--log2_hashmap_size", type=float, default=15)

    p.add_argument("--rim_enable", action="store_true", default=True)
    p.add_argument("--rim_analysis_resolution", type=int, default=128)
    p.add_argument("--rim_cube_size", type=int, default=16)
    p.add_argument("--rim_num_candidates", type=int, default=64)
    p.add_argument("--rim_iters", type=int, default=20)
    p.add_argument("--rim_init_tol", type=float, default=1e-6)
    p.add_argument("--rim_gate_solver", default="prefix_fallback", choices=["exact_fractional", "prefix_fallback"])
    p.add_argument("--rim_candidate_resolutions", default=None)
    p.add_argument("--rim_save_debug", action="store_true")
    p.add_argument("--rim_information_metric", default="entropy", choices=["energy", "entropy", "energy_entropy"])
    p.add_argument(
        "--rim_gate_init_logit",
        type=float,
        default=2.5,
        help=(
            "Fixed logit magnitude used to initialize trainable gate logits from the "
            "solver's binary on/off decision (matches the validated RIM 2D reference "
            "GATE_INIT_LOGIT default). A too-large magnitude (e.g. deriving logits from "
            "eps=1e-4 clamping) saturates sigmoid'(logit) and freezes gate gradients."
        ),
    )
    p.add_argument("--rim_gate_lr", type=float, default=1e-4)
    p.add_argument("--rim_gate_reg", type=float, default=0.0)
    p.add_argument("--rim_gate_sparsity_weight", type=float, default=0.0)
    p.add_argument("--rim_fallback_mode", default="blockwise", choices=["blockwise", "global_shared"])
    p.add_argument("--rim_train_resolution", action="store_true", help="Reserved; selected resolutions are fixed in this implementation.")
    p.set_defaults(rim_gate_trainable=True)
    p.add_argument("--rim_gate_trainable", dest="rim_gate_trainable", action="store_true")
    p.add_argument("--rim_gate_fixed", dest="rim_gate_trainable", action="store_false")
    return p


def make_hparams(args: argparse.Namespace, run_name: str) -> SimpleNamespace:
    opt = SimpleNamespace()
    opt.config_fp = __file__
    opt.task = "sdf"
    opt.workspace = args.out_dir
    opt.exp_name = run_name
    opt.test = False
    opt.seed = args.seed
    opt.cfactor = 128
    opt.train_shuffle_mode = 1
    opt.train_num_samples = int(args.batch_size)
    opt.train_presample = bool(args.train_presample)
    opt.train_epoch_size = int(args.train_epoch_size)
    opt.val_resolution = int(args.val_resolution)
    opt.clip_sdf = args.clip_sdf
    opt.record_training = False
    opt.arch = "rim_metric"
    opt.num_levels = int(args.num_levels)
    opt.level_dim = int(args.feature_dim)
    opt.base_resolution = int(args.base_resolution)
    opt.desired_resolution = int(args.desired_resolution)
    opt.log2_hashmap_size = float(args.log2_hashmap_size)
    opt.hash_table_size = log2_hashmap_size_to_table_size(args.log2_hashmap_size)
    opt.num_layers = int(args.num_layers)
    opt.hidden_dim = int(args.hidden_dim)
    opt.max_steps = int(args.steps)
    opt.val_freq = 1.0
    opt.val_first = False
    opt.train_val_same_points = False
    opt.log_train = True
    opt.log_img = False
    opt.log_kernel_img = False
    opt.log_sdf_slice = False
    opt.save_pred = bool(args.save_pred)
    opt.save_pred_pt = False
    opt.save_ckpt = bool(args.save_ckpt)
    opt.train_metric_list = ["psnr", "mae"]
    opt.val_metric_list = ["psnr", "iou", "mae", "mse"]
    if args.eval_mesh_metrics:
        opt.val_metric_list.append("mesh")
    opt.metric_config = {"n_surface_samples": 1000000, "fscore_tau": [1e-2, 2e-3, 1e-3, 2e-4, 1e-4]}
    opt.ema_decay = None
    opt.fp16 = False
    opt.vmin = -(2**2 * 3) ** 0.5
    opt.vmax = (2**2 * 3) ** 0.5
    opt.rim_gate_sparsity_weight = float(args.rim_gate_sparsity_weight)
    eps = 1e-15
    opt.optims = {
        "dec": {"type": "Adam", "lr": float(args.lr), "betas": (0.9, 0.99), "eps": eps, "wd": 0},
        "hg": {"type": "Adam", "lr": float(args.lr), "betas": (0.9, 0.99), "eps": eps, "wd": 0},
        "gate": {"type": "Adam", "lr": float(args.rim_gate_lr), "betas": (0.9, 0.99), "eps": eps, "wd": float(args.rim_gate_reg)},
    }
    opt.lr_schs = {
        "dec": {"type": "cosine", "T_max": max(1, int(args.steps)), "gamma": 10},
        "hg": {"type": "cosine", "T_max": max(1, int(args.steps)), "gamma": 10},
        "gate": {"type": "cosine", "T_max": max(1, int(args.steps)), "gamma": 10},
    }
    return opt


def main() -> None:
    args = build_argparser().parse_args()
    log2_hashmap_size_raw = float(args.log2_hashmap_size)
    hash_table_size = log2_hashmap_size_to_table_size(log2_hashmap_size_raw)
    print(
        f"[RIM-SDF] --log2_hashmap_size={log2_hashmap_size_raw:g} "
        f"=> hash table size=round(2^{log2_hashmap_size_raw:g})={hash_table_size}",
        flush=True,
    )
    if args.rim_train_resolution:
        raise NotImplementedError("--rim_train_resolution is reserved; resolutions are fixed after RIM initialization.")
    if args.seed is not None:
        utils.seed_everything(int(args.seed))
    if int(args.batch_size) % 8 != 0:
        raise ValueError("--batch_size must be divisible by 8 for the existing SDF sampler")
    if int(args.val_num_samples) % 8 != 0:
        raise ValueError("--val_num_samples must be divisible by 8")

    mesh_path = os.path.abspath(args.mesh)
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(
            f"Mesh not found: {mesh_path}. Run scripts/prepare_sdf_stanford.sh or place Stanford meshes under data/sdf/raw."
        )
    alias = args.alias or os.path.basename(mesh_path).rsplit(".", 1)[0].replace("_nrml", "")
    run_name = f"{alias}-{args.encoder}"
    os.makedirs(os.path.join(args.out_dir, "run", run_name), exist_ok=True)
    debug_dir = os.path.join(args.out_dir, "run", run_name, "rim_debug")
    os.makedirs(debug_dir, exist_ok=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    ds_device = "cpu" if args.ds_device == "cpu" or device.type == "cpu" else 0
    print(f"Using device: {device}; dataset device: {ds_device}")

    raw_mesh = trimesh.load(mesh_path, force="mesh")
    if args.mesh_already_normalized or "_nrml" in os.path.basename(mesh_path):
        gt_mesh = raw_mesh
        norm_meta = {"mesh_already_normalized": True, "target_domain": "[-1,1]^3"}
    else:
        gt_mesh, norm_meta = normalize_mesh_to_unit_sphere(raw_mesh)
    if not gt_mesh.is_watertight:
        print("[WARN] mesh is not watertight; SDF signs may be noisy.")

    cmin = [-1, -1, -1]
    cmax = [1, 1, 1]
    s_dims = [int(args.val_resolution)] * 3

    candidate_resolutions = parse_candidate_resolutions(
        args.rim_candidate_resolutions,
        int(args.rim_num_candidates),
        int(args.base_resolution),
        int(args.desired_resolution),
        num_levels=int(args.num_levels),
    )
    table_sizes = [int(hash_table_size) for _ in range(int(args.num_levels))]
    cube_dim = int(math.ceil(int(args.rim_analysis_resolution) / int(args.rim_cube_size)))
    cube_dims = (cube_dim, cube_dim, cube_dim)

    if args.encoder == "fixed_hashgrid":
        selected_resolutions = geometric_resolutions(int(args.num_levels), int(args.base_resolution), int(args.desired_resolution))
        gates_init = torch.ones((int(args.num_levels), int(np.prod(cube_dims))), dtype=torch.float32)
        rim_details = {
            "mode": "fixed_hashgrid",
            "selected_resolutions": selected_resolutions,
            "cube_dims": cube_dims,
            "normalization": norm_meta,
        }
    else:
        print("[RIM-SDF] estimating 3D cube-band information", flush=True)
        info = estimate_sdf3d_information(
            gt_mesh,
            analysis_resolution=int(args.rim_analysis_resolution),
            cube_size=int(args.rim_cube_size),
            candidate_resolutions=candidate_resolutions,
            output_dir=debug_dir,
            save_debug=bool(args.rim_save_debug),
            information_metric=str(args.rim_information_metric),
            device=str(device),
        )
        cube_dims = tuple(int(v) for v in info["metadata"]["cube_dims"])
        print(
            f"[RIM-SDF] info tensor: A={info['A'].shape} cube_dims={cube_dims} "
            f"candidates={len(candidate_resolutions)} levels={int(args.num_levels)}",
            flush=True,
        )
        init_result = initialize_rim_sdf3d(
            info["A"],
            candidate_resolutions,
            num_levels=int(args.num_levels),
            table_sizes=table_sizes,
            cube_dims=cube_dims,
            max_iters=int(args.rim_iters),
            tol=float(args.rim_init_tol),
            gate_solver=str(args.rim_gate_solver),
            mode=str(args.encoder),
            base_resolution=int(args.base_resolution),
            desired_resolution=int(args.desired_resolution),
            device=str(device),
            verbose=True,
            output_dir=debug_dir,
        )
        if args.encoder == "rim_full" and not bool(init_result.details.get("converged", False)):
            raise RuntimeError(
                "RIM gate/scheduler initialization did not reach a consistent fixed point; "
                f"increase --rim_iters above {int(args.rim_iters)}"
            )
        selected_resolutions = init_result.resolutions
        gates_init = init_result.gates
        rim_details = {
            "partition": init_result.partition,
            "selected_resolutions": init_result.resolutions,
            "objective_history": init_result.objective_history,
            "details": init_result.details,
            "info_stats": info["stats"],
            "info_metadata": info["metadata"],
            "normalization": norm_meta,
        }
        print(f"[RIM-SDF] selected resolutions: {selected_resolutions}", flush=True)
        print(f"[RIM-SDF] avg gate per level: {[round(float(v), 4) for v in gates_init.float().mean(dim=1)]}", flush=True)

    gate_mode = "fixed"
    if args.encoder in {"rim_gate", "rim_full"} and bool(args.rim_gate_trainable):
        gate_mode = "trainable_sigmoid"
    fallback_mode = str(args.rim_fallback_mode)
    if args.encoder in {"fixed_hashgrid", "rim_resolution"}:
        # These modes have all-one gates, so fallback features are unreachable.
        # Keep the tiny legacy allocation instead of adding unused block params
        # to the baseline parameter count.
        fallback_mode = "global_shared"
    model = RIMSDFNGP(
        cmin,
        cmax,
        selected_resolutions=selected_resolutions,
        cube_dims=cube_dims,
        gates=gates_init,
        gate_mode=gate_mode,
        fallback_mode=fallback_mode,
        num_layers=int(args.num_layers),
        hidden_dim=int(args.hidden_dim),
        in_dim=3,
        out_dim=1,
        num_levels=int(args.num_levels),
        level_dim=int(args.feature_dim),
        log2_hashmap_size=log2_hashmap_size_raw,
        gate_init_logit=float(args.rim_gate_init_logit),
    )
    model.count_params()
    print(
        f"[RIM-SDF] gate_mode={gate_mode} fallback_mode={fallback_mode} "
        f"gate_trainable={hasattr(model.encoder, 'gate_logits')}",
        flush=True,
    )
    gates_initial_model = model.gates_numpy()
    np.save(os.path.join(debug_dir, "rim_gates_init.npy"), gates_initial_model)
    logits_initial = model.gate_logits_numpy()
    if logits_initial is not None:
        np.save(os.path.join(debug_dir, "rim_gate_logits_init.npy"), logits_initial)

    print("Prepare dataset and dataloader")
    train_dataset = SDFDataset(
        gt_mesh,
        cmin,
        cmax,
        s_dims,
        num_samples=int(args.batch_size),
        size=int(args.train_epoch_size),
        presample=bool(args.train_presample),
        shuffle_mode=1,
        clip_sdf=args.clip_sdf,
        mesh_fp=mesh_path,
        device=ds_device,
    )
    valid_dataset = SDFDataset(
        gt_mesh,
        cmin,
        cmax,
        s_dims,
        num_samples=int(args.val_num_samples),
        is_grid=True,
        clip_sdf=args.clip_sdf,
        mesh_fp=mesh_path,
        device=ds_device,
    )
    train_dataset.prepare()
    valid_dataset.prepare()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    if args.loss == "mape":
        from thirdparty.torch_ngp.loss import mape_loss

        criterion = mape_loss
    elif args.loss == "l1":
        criterion = torch.nn.L1Loss(reduction="mean")
    else:
        criterion = torch.nn.MSELoss(reduction="mean")

    opt = make_hparams(args, run_name)
    eval_interval = math.ceil(math.ceil(int(args.steps) / len(train_loader)) * opt.val_freq)
    trainer = utils.Trainer(
        run_name,
        model,
        hparams=opt,
        workspace=args.out_dir,
        criterion=criterion,
        ema_decay=None,
        fp16=False,
        use_checkpoint="scratch",
        eval_interval=eval_interval,
        local_rank=0,
        device=device,
    )

    t0 = time.time()
    trainer.train(train_loader, valid_loader, max_steps=int(args.steps))
    train_time = time.time() - t0

    metrics = {"grid": trainer.val_metrics}
    eval_points_path = mesh_path.rsplit(".", 1)[0] + "_eval_points.pt"
    if os.path.exists(eval_points_path):
        try:
            # Stanford eval-point files contain NumPy arrays. PyTorch 2.6 changed
            # torch.load's default to weights_only=True, which rejects that trusted
            # local dataset format and previously dropped all eval-point metrics.
            eval_points = torch.load(eval_points_path, map_location="cpu", weights_only=False)
            metrics["eval_points"] = trainer.evaluate_points(eval_points)
        except Exception as exc:
            metrics["eval_points_error"] = str(exc)

    final_gates = model.gates_numpy()
    np.save(os.path.join(debug_dir, "rim_gates_final.npy"), final_gates)
    logits = model.gate_logits_numpy()
    if logits is not None:
        np.save(os.path.join(debug_dir, "rim_gate_logits_final.npy"), logits)
    gate_stats = model.get_gate_stats()
    gate_grad_norm = model.gate_grad_norm()
    gate_delta = float(np.abs(final_gates - gates_initial_model).mean())
    with open(os.path.join(debug_dir, "rim_gate_stats_final.json"), "w") as f:
        json.dump({**gate_stats, "gate_grad_norm": gate_grad_norm, "mean_abs_gate_delta": gate_delta}, f, indent=2)

    stats = {
        "time": {"t_train": train_time},
        "misc": {
            "n_epoch": trainer.epoch,
            "n_iter": trainer.global_step,
            "batch_size": int(args.batch_size),
            "encoder": args.encoder,
            "log2_hashmap_size_input": log2_hashmap_size_raw,
            "hash_table_size": int(hash_table_size),
            "gate_mode": gate_mode,
            "fallback_mode": fallback_mode,
            "gate_grad_norm": gate_grad_norm,
            "mean_abs_gate_delta": gate_delta,
        },
        "n_param": {f"n_param_{k}": _json_safe(v) for k, v in model.count_params().items()},
        "metrics": metrics,
        "rim": rim_details,
        "gate_stats": gate_stats,
        "outputs": {"debug_dir": debug_dir},
    }
    pprint.pprint(stats, sort_dicts=False)
    results_dir = os.path.join(args.out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    result_base = os.path.join(results_dir, run_name)
    with open(f"{result_base}.json", "w") as f:
        json.dump(_json_safe(stats), f, indent=2)
    with open(f"{result_base}.txt", "w") as f:
        pprint.pprint(stats, f, sort_dicts=False)

    if args.save_pred:
        pred_latest = getattr(trainer, "pred_latest", None)
        if pred_latest is not None:
            trainer.save_mesh(pred_latest, f"{result_base}.ply")
        else:
            print("[RIM-SDF] no validation prediction available; skipped mesh export.")

    if args.save_ckpt:
        trainer.save_checkpoint({"metrics": _json_safe(metrics), "gate_stats": _json_safe(gate_stats)})
    print(f"[RIM-SDF] metrics JSON: {result_base}.json")
    print(f"[RIM-SDF] debug dir: {debug_dir}")
    if gate_mode == "trainable_sigmoid":
        print(f"[RIM-SDF] gate grad norm: {gate_grad_norm}; mean abs gate delta: {gate_delta:.6g}")


if __name__ == "__main__":
    main()
