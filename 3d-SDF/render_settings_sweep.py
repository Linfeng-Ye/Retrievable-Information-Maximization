"""
Batch version of render_hq.py: walks one or more outputs*/ roots, finds every
(setting, checkpoint, scene) reconstruction under them, and renders GT +
recon HQ images for each — same view/up-axis/dpi defaults as vis_hq.sh.

Each (root, setting, checkpoint) gets its own output subdirectory, so scenes
that share a name across different settings/checkpoints (e.g. Armadillo in
both a log2 sweep and the metricfix_seq run) never overwrite each other, and
nothing lands in vis_out/hq (owned by vis_hq.sh).

--shard/--num-shards split the discovered job list deterministically, so
several invocations (e.g. one per GPU) can each render a disjoint slice in
parallel.

    python render_settings_sweep.py \\
        --root outputs_pre --root outputs_secondbest \\
        --view 15,60 --view 10,225 \\
        --out-dir vis_out/hq_sweep \\
        --shard 0 --num-shards 4
"""

import argparse
import os
import re

import visualize_sdf as vis
from render_hq import render_one

REPO_ROOT = vis.REPO_ROOT

# Canonical scene names, as used by data/sdf/<Scene>_nrml.obj — path
# components and filenames found on disk don't always match this casing
# (e.g. "armadillo-fixed_hashgrid.ply"), so lookups below are case-insensitive.
KNOWN_SCENES = ["Bunny", "Armadillo", "Dragon", "Lucy", "Buddha", "Statuette", "XYZDragon"]
SCENE_BY_LOWER = {s.lower(): s for s in KNOWN_SCENES}
CKPT_RE = re.compile(r"log2_\d+p?\d*")


def canonical_scene(name):
    return SCENE_BY_LOWER.get(name.lower())


def discover_jobs(root):
    """Find every *.ply under `root`, yielding (setting, checkpoint, scene, ply_path).

    Two directory layouts coexist under outputs_pre/outputs_secondbest:
      A) <setting>/<checkpoint>/<Scene>/<variant>/results/<file>.ply
      B) <setting>/results/<file>.ply   (single checkpoint, folded into the
         setting name itself, e.g. full_baseline_log2_15; scene comes from
         the filename prefix, e.g. armadillo-fixed_hashgrid.ply)
    """
    root_abs = os.path.join(REPO_ROOT, root)
    jobs = []
    for dirpath, _, filenames in os.walk(root_abs):
        for fn in filenames:
            if not fn.endswith(".ply"):
                continue
            ply_path = os.path.join(dirpath, fn)
            parts = os.path.relpath(ply_path, root_abs).split(os.sep)
            setting = parts[0]

            ckpt_part = next((p for p in parts if CKPT_RE.fullmatch(p)), None)
            scene_part = next((p for p in parts if canonical_scene(p)), None)
            if ckpt_part and scene_part:
                checkpoint = ckpt_part
                scene = canonical_scene(scene_part)
            else:
                m = CKPT_RE.search(setting)
                checkpoint = m.group(0) if m else "unknown"
                scene = canonical_scene(fn.split("-")[0])

            if scene is None:
                print(f"[skip] couldn't identify scene for {os.path.relpath(ply_path, REPO_ROOT)}")
                continue
            jobs.append((setting, checkpoint, scene, ply_path))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", action="append", required=True,
                         help="outputs* root to sweep, repeatable (e.g. outputs_pre)")
    parser.add_argument("--setting", action="append",
                         help="restrict to this top-level setting dir under --root, repeatable "
                              "(e.g. sdf_stanford7_log2_sweep_part1); default: every setting found")
    parser.add_argument("--which", choices=["gt", "recon", "both"], default="both")
    parser.add_argument("--view", action="append", type=vis.parse_view, required=True)
    parser.add_argument("--up", choices=["x", "y", "z"], default="y")
    parser.add_argument("--gt-color", type=vis.parse_color, default=(0.75, 0.68, 0.55))
    parser.add_argument("--recon-color", type=vis.parse_color, default=(0.75, 0.68, 0.55))
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--size", type=float, default=12)
    parser.add_argument("--out-dir", default="vis_out/hq_sweep")
    parser.add_argument("--shard", type=int, default=0, help="which slice of the job list this invocation renders")
    parser.add_argument("--num-shards", type=int, default=1, help="total number of parallel invocations")
    args = parser.parse_args()

    setting_filter = set(args.setting) if args.setting else None

    all_jobs = []
    for root in args.root:
        for setting, checkpoint, scene, ply_path in discover_jobs(root):
            if setting_filter is not None and setting not in setting_filter:
                continue
            all_jobs.append((root, setting, checkpoint, scene, ply_path))
    all_jobs.sort()  # deterministic, so --shard slices are stable across processes/roots

    if setting_filter is not None:
        missing = setting_filter - {j[1] for j in all_jobs}
        if missing:
            print(f"[warn] no reconstructions found for setting(s): {', '.join(sorted(missing))}")

    my_jobs = all_jobs[args.shard::args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] {len(my_jobs)} of {len(all_jobs)} total jobs")

    gt_cache = {}  # scene -> (verts, faces); GT doesn't depend on setting/checkpoint

    for root, setting, checkpoint, scene, ply_path in my_jobs:
        out_dir = os.path.join(REPO_ROOT, args.out_dir, root, setting, checkpoint)
        os.makedirs(out_dir, exist_ok=True)
        print(f"--- {root}/{setting}/{checkpoint}/{scene} ---")

        gt_v = gt_f = rc_v = rc_f = None
        if args.which in ("gt", "both"):
            if scene not in gt_cache:
                gt_path = vis.gt_path_for_scene(scene)
                gt_cache[scene] = vis.load_mesh(gt_path, max_faces=None, up_axis=args.up)[:2]
            gt_v, gt_f = gt_cache[scene]
        if args.which in ("recon", "both"):
            rc_v, rc_f, _ = vis.load_mesh(ply_path, max_faces=None, up_axis=args.up)
            print(f"Recon: {os.path.relpath(ply_path, REPO_ROOT)} ({len(rc_f)} faces)")

        for elev, azim in args.view:
            if gt_v is not None:
                out = os.path.join(out_dir, f"{scene}_gt_elev{elev:g}_azim{azim:g}.png")
                render_one(gt_v, gt_f, args.gt_color, elev, azim, out, args.dpi, args.size)
            if rc_v is not None:
                out = os.path.join(out_dir, f"{scene}_recon_elev{elev:g}_azim{azim:g}.png")
                render_one(rc_v, rc_f, args.recon_color, elev, azim, out, args.dpi, args.size)


if __name__ == "__main__":
    main()
