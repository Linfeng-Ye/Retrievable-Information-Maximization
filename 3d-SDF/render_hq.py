"""
Render a single trained SDF reconstruction (and/or its ground truth) as
individual, full-resolution images — one PNG per (mesh, view), no grid, no
decimation. For a side-by-side comparison figure, use visualize_sdf.py
instead; this is for pulling the sharpest possible single shot of a chosen
viewpoint, e.g. for a paper figure.

Requires trimesh and matplotlib:
    python render_hq.py ...

Typical use, after picking angles with `visualize_sdf.py explore`:
    python render_hq.py --scene Bunny --view 15,60 --view 10,225 \\
        --out-dir vis_out/hq

Point at an exact reconstruction instead of scene auto-discovery:
    python render_hq.py --scene Bunny \\
        --recon outputs_secondbest/.../Bunny-log2_15-rim_full.ply \\
        --view 15,60 --out-dir vis_out/hq
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import visualize_sdf as vis


def render_one(verts, faces, color, elev, azim, out_path, dpi, size):
    xlim, ylim, zlim = vis.bounds_with_pad(verts)

    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(projection="3d")
    colors = vis.shaded_colors(verts, faces, color)
    # antialiased=True (unlike visualize_sdf.py's grid views) since this is
    # a one-off full-quality shot, not something rendered 8-to-a-page.
    coll = Poly3DCollection(verts[faces], facecolor=colors, edgecolor="none", antialiased=True)
    ax.add_collection3d(coll)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"wrote {out_path}  ({len(faces)} faces, {dpi} dpi, {size}in)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    vis.add_mesh_selection_args(parser)
    parser.add_argument("--recon", help="explicit reconstruction .ply path")
    parser.add_argument("--pick", type=int, help="index from `visualize_sdf.py list` output, if --scene has multiple matches")
    parser.add_argument("--which", choices=["gt", "recon", "both"], default="both",
                         help="which mesh(es) to render (default: both)")
    parser.add_argument("--view", action="append", type=vis.parse_view, required=True,
                         help="elev,azim — repeatable, one image pair written per view")
    parser.add_argument("--up", choices=["x", "y", "z"], default="y",
                         help="mesh axis that should point up on screen (default: y, true for the Stanford scans)")
    parser.add_argument("--gt-color", type=vis.parse_color, default=(0.75, 0.68, 0.55))
    parser.add_argument("--recon-color", type=vis.parse_color, default=(0.75, 0.68, 0.55))
    parser.add_argument("--dpi", type=int, default=400,
                         help="default 400 — full (non-decimated) mesh at high dpi is slow; lower it for a quick look")
    parser.add_argument("--size", type=float, default=12,
                         help="figure size in inches, square; --size * --dpi = output pixel dimensions (default 12in -> 4800px at dpi=400)")
    parser.add_argument("--out-dir", default="vis_out/hq")
    args = parser.parse_args()

    if args.scene is None and args.gt is None:
        parser.error("pass --scene or --gt")
    if args.which in ("recon", "both") and args.scene is None and args.recon is None:
        parser.error("rendering the reconstruction needs --scene (to auto-find one) or --recon")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.scene or os.path.splitext(os.path.basename(args.gt))[0]

    gt_v = gt_f = rc_v = rc_f = None
    if args.which in ("gt", "both"):
        gt_path = args.gt or vis.gt_path_for_scene(args.scene)
        gt_v, gt_f, _ = vis.load_mesh(gt_path, max_faces=None, up_axis=args.up)
        print(f"GT:    {os.path.relpath(gt_path, vis.REPO_ROOT)} ({len(gt_f)} faces)")
    if args.which in ("recon", "both"):
        recon_path = vis.resolve_recon(args)
        rc_v, rc_f, _ = vis.load_mesh(recon_path, max_faces=None, up_axis=args.up)
        print(f"Recon: {os.path.relpath(recon_path, vis.REPO_ROOT)} ({len(rc_f)} faces)")

    for elev, azim in args.view:
        if gt_v is not None:
            out = os.path.join(args.out_dir, f"{tag}_gt_elev{elev:g}_azim{azim:g}.png")
            render_one(gt_v, gt_f, args.gt_color, elev, azim, out, args.dpi, args.size)
        if rc_v is not None:
            out = os.path.join(args.out_dir, f"{tag}_recon_elev{elev:g}_azim{azim:g}.png")
            render_one(rc_v, rc_f, args.recon_color, elev, azim, out, args.dpi, args.size)


if __name__ == "__main__":
    main()
