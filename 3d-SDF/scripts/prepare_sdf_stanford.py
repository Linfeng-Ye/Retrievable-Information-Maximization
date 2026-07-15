#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pysdf
import torch
import trimesh


SCENES = ["Armadillo", "Bunny", "Dragon", "Buddha", "Lucy", "XYZDragon", "Statuette"]


def normalize_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict]:
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


def write_manifest(path: Path, scenes: list[dict], note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({"dataset": "stanford_3d_scanning_repository_sdf", "assumption": note, "scenes": scenes}, f, indent=2)


def default_manifest(repo: Path, manifest: Path) -> None:
    scenes = []
    for name in SCENES:
        raw = repo / "data" / "sdf" / "raw" / f"{name}.ply"
        processed = repo / "data" / "sdf" / "processed" / f"{name}_nrml.obj"
        scenes.append(
            {
                "name": name,
                "raw_mesh_path": str(raw),
                "mesh_path": str(raw if raw.exists() else processed),
                "processed_mesh_path": str(processed),
                "eval_points_path": str(processed).rsplit(".", 1)[0] + "_eval_points.pt",
                "available": bool(raw.exists() or processed.exists()),
            }
        )
    write_manifest(
        manifest,
        scenes,
        "Scene list is reused from MetricGrids/sdf.sh. Download meshes manually from the Stanford repository.",
    )


def sample_eval_points(mesh: trimesh.Trimesh, out_path: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    sdf_fn = pysdf.SDF(mesh.vertices, mesh.faces)
    n_uniform = max(1, n // 2)
    n_near = max(1, n - n_uniform)
    pts_uniform = (rng.random((n_uniform, 3), dtype=np.float32) * 2.0 - 1.0).astype(np.float32)
    pts_near = mesh.sample(n_near).view(np.ndarray).astype(np.float32)
    pts_near += (0.01 * rng.standard_normal(pts_near.shape)).astype(np.float32)
    sdfs_uniform = -sdf_fn(pts_uniform)[:, None].astype(np.float32)
    sdfs_near = -sdf_fn(pts_near)[:, None].astype(np.float32)
    torch.save(
        {"points": {"unif": pts_uniform, "near": pts_near}, "sdfs": {"unif": sdfs_uniform, "near": sdfs_near}},
        out_path,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare Stanford SDF meshes for MetricGrids/RIM experiments.")
    p.add_argument("--mesh", default=None, help="Path to a .ply/.obj mesh to add/normalize.")
    p.add_argument("--name", default=None, help="Scene name for --mesh, e.g. Armadillo.")
    p.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--eval-samples", type=int, default=200000)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    repo = Path(args.repo).resolve()
    raw_dir = repo / "data" / "sdf" / "raw"
    processed_dir = repo / "data" / "sdf" / "processed"
    manifest = repo / "data" / "sdf" / "manifest_stanford.json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not args.mesh:
        default_manifest(repo, manifest)
        print(f"Created/updated manifest: {manifest}")
        print("Place Stanford meshes in data/sdf/raw or run with --mesh /path/to/mesh.ply --name Armadillo")
        return

    mesh_path = Path(args.mesh).expanduser().resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    name = args.name or mesh_path.stem.replace("_nrml", "")
    raw_copy = raw_dir / f"{name}{mesh_path.suffix}"
    if mesh_path != raw_copy:
        shutil.copy2(mesh_path, raw_copy)

    mesh = trimesh.load(mesh_path, force="mesh")
    if not mesh.is_watertight:
        print("[WARN] mesh is not watertight; SDF signs may be noisy.")
    mesh_norm, norm = normalize_mesh(mesh)
    processed = processed_dir / f"{name}_nrml.obj"
    mesh_norm.export(processed)
    eval_path = processed_dir / f"{name}_nrml_eval_points.pt"
    if not args.skip_eval:
        sample_eval_points(mesh_norm, eval_path, int(args.eval_samples), int(args.seed))

    default_manifest(repo, manifest)
    data = json.load(open(manifest))
    updated = False
    for scene in data["scenes"]:
        if scene["name"].lower() == name.lower():
            scene.update(
                {
                    "name": name,
                    "raw_mesh_path": str(raw_copy),
                    "mesh_path": str(raw_copy),
                    "processed_mesh_path": str(processed),
                    "eval_points_path": str(eval_path),
                    "normalization": norm,
                    "available": True,
                }
            )
            updated = True
    if not updated:
        data["scenes"].append(
            {
                "name": name,
                "raw_mesh_path": str(raw_copy),
                "mesh_path": str(raw_copy),
                "processed_mesh_path": str(processed),
                "eval_points_path": str(eval_path),
                "normalization": norm,
                "available": True,
            }
        )
    with open(manifest, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Raw mesh: {raw_copy}")
    print(f"Normalized mesh: {processed}")
    if not args.skip_eval:
        print(f"Eval points: {eval_path}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()

