"""
Compare a ground-truth mesh against a trained SDF's marching-cubes
reconstruction, rendered from angles you choose.

Requires trimesh and matplotlib:
    python visualize_sdf.py ...

Typical workflow
-----------------
1) Find candidate reconstructions for a scene:
     python visualize_sdf.py list --scene Bunny

2) Explore viewpoints cheaply (fast, decimated) to find angles you like:
     python visualize_sdf.py explore --scene Bunny

   This dumps a grid of (elev, azim) options -> bunny_explore.png. Pick the
   pair that looks best.

3) Render the final comparison at those angles, full quality:
     python visualize_sdf.py render --scene Bunny --view 10,45 --view 10,225 --full

You can always bypass scene auto-discovery and point at exact files:
     python visualize_sdf.py render --gt data/sdf/Bunny_nrml.obj \\
         --recon outputs_100k_eval/.../Bunny-log2_14p5-rim_full.ply \\
         --view 15,60 --full
"""

import argparse
import glob
import os
import textwrap

import numpy as np
import trimesh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LIGHT_DIR = np.array([0.4, 0.5, 0.8])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)


# ---------------------------------------------------------------------------
# Mesh discovery
# ---------------------------------------------------------------------------

def gt_path_for_scene(scene):
    path = os.path.join(REPO_ROOT, "data", "sdf", f"{scene}_nrml.obj")
    if not os.path.exists(path):
        raise SystemExit(f"no ground-truth mesh at {path}")
    return path


def find_reconstructions(scene):
    """All *.ply under any outputs*/ root whose name contains the scene,
    newest first. Multiple training sweeps/output roots coexist on this
    machine, so there is rarely a single unambiguous answer."""
    pattern = os.path.join(REPO_ROOT, "outputs*", "**", f"*{scene}*.ply")
    matches = glob.glob(pattern, recursive=True)
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches


def resolve_recon(args):
    if args.recon:
        return args.recon
    matches = find_reconstructions(args.scene)
    if not matches:
        raise SystemExit(f"no reconstructed .ply found for scene '{args.scene}' under outputs*/")
    if args.pick is not None:
        return matches[args.pick]
    if len(matches) > 1:
        print(f"[{len(matches)} matches for '{args.scene}', using the newest — pass --recon or --pick N to choose another]")
        for i, m in enumerate(matches):
            marker = "->" if i == 0 else "  "
            print(f"  {marker} [{i}] {os.path.relpath(m, REPO_ROOT)}")
    return matches[0]


# ---------------------------------------------------------------------------
# Loading / shading
# ---------------------------------------------------------------------------

def load_mesh(path, max_faces, up_axis):
    mesh = trimesh.load(path, force="mesh", process=False)
    orig_face_count = len(mesh.faces)
    if max_faces and orig_face_count > max_faces:
        # Quadric edge-collapse: merges neighboring triangles into fewer,
        # larger ones that still tile the whole surface. Randomly dropping
        # faces instead leaves a speckled, gappy look at the same budget.
        mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
    perm = {"x": [1, 2, 0], "y": [0, 2, 1], "z": [0, 1, 2]}[up_axis]
    verts = mesh.vertices[:, perm]  # rotate so the chosen mesh axis is plot-Z (up)
    return verts, mesh.faces, orig_face_count


def shaded_colors(verts, faces, color):
    tris = verts[faces]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-12, None)
    # two-sided lambertian + ambient floor so backfaces aren't pure black
    intensity = np.clip(np.abs(n @ LIGHT_DIR) * 0.75 + 0.25, 0, 1)[:, None]
    return np.clip(np.array(color)[None, :] * intensity, 0, 1)


def plot_mesh(ax, verts, faces, color, elev, azim, xlim, ylim, zlim, title):
    colors = shaded_colors(verts, faces, color)
    coll = Poly3DCollection(verts[faces], facecolor=colors, edgecolor="none", antialiased=False)
    ax.add_collection3d(coll)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)


def bounds_with_pad(*vert_arrays, pad=0.03):
    all_v = np.concatenate(vert_arrays, axis=0)
    lims = []
    for i in range(3):
        lims.append((all_v[:, i].min() - pad, all_v[:, i].max() + pad))
    return lims


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_list(args):
    matches = find_reconstructions(args.scene)
    if not matches:
        print(f"no reconstructions found for '{args.scene}'")
        return
    print(f"{len(matches)} match(es) for '{args.scene}', newest first:")
    for i, m in enumerate(matches):
        print(f"  [{i}] {os.path.relpath(m, REPO_ROOT)}")


def cmd_explore(args):
    path = args.gt or gt_path_for_scene(args.scene)
    if args.target == "recon":
        path = resolve_recon(args)
    verts, faces, _ = load_mesh(path, max_faces=args.max_faces, up_axis=args.up)
    xlim, ylim, zlim = bounds_with_pad(verts)

    combos = [(10, a) for a in range(0, 360, 45)]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={"projection": "3d"})
    for ax, (elev, azim) in zip(axes.flat, combos):
        plot_mesh(ax, verts, faces, args.color, elev, azim, xlim, ylim, zlim,
                   f"elev={elev} azim={azim}")
    fig.suptitle(f"{os.path.basename(path)} — pick a (elev, azim) for --view", fontsize=12)
    plt.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out} — rerun with `render --view ELEV,AZIM` using the pair you like")


def cmd_render(args):
    recon_path = resolve_recon(args)
    max_faces = None if args.full else args.max_faces
    rc_v, rc_f, rc_orig = load_mesh(recon_path, max_faces, args.up)

    if args.recon_only:
        # Skip loading/decimating the (often much larger) GT mesh entirely —
        # it's never plotted, so there's no reason to pay for it.
        gt_path, gt_v, gt_f, gt_orig = None, None, None, None
        xlim, ylim, zlim = bounds_with_pad(rc_v)
    else:
        gt_path = args.gt or gt_path_for_scene(args.scene)
        gt_v, gt_f, gt_orig = load_mesh(gt_path, max_faces, args.up)
        xlim, ylim, zlim = bounds_with_pad(gt_v, rc_v)

    views = args.view or [(10, 45), (10, 225)]
    nrows = 1 if args.recon_only else 2

    # Header/footer text bands are fixed-inch, not a fraction of figure
    # height — otherwise their y-fractions (tuned for the 2-row layout)
    # overlap the plot when recon-only collapses this to 1 row.
    header_in, footer_in = 1.3, 0.45
    plot_h_in = 5 * nrows
    fig_height_in = header_in + plot_h_in + footer_in
    fig_width_in = 5 * len(views)

    fig, axes = plt.subplots(nrows, len(views), figsize=(fig_width_in, fig_height_in),
                               subplot_kw={"projection": "3d"}, squeeze=False)
    for col, (elev, azim) in enumerate(views):
        if not args.recon_only:
            plot_mesh(axes[0, col], gt_v, gt_f, args.gt_color, elev, azim, xlim, ylim, zlim,
                       f"Ground truth (elev={elev}, azim={azim})")
        plot_mesh(axes[nrows - 1, col], rc_v, rc_f, args.recon_color, elev, azim, xlim, ylim, zlim,
                   f"Reconstruction (elev={elev}, azim={azim})")

    wrap_chars = max(40, int(fig_width_in * 11))
    title_y = 1 - (0.35 / fig_height_in)
    path_y = 1 - (0.85 / fig_height_in)
    footer_y = (0.5 * footer_in) / fig_height_in

    fig.suptitle(args.scene or os.path.basename(gt_path or recon_path), fontsize=13, y=title_y)
    fig.text(0.5, path_y, textwrap.fill(os.path.relpath(recon_path, REPO_ROOT), wrap_chars),
              ha="center", fontsize=8, color="gray")
    footer = f"Recon: {len(rc_f)}/{rc_orig} faces"
    if not args.recon_only:
        footer = f"GT: {len(gt_f)}/{gt_orig} faces  |  " + footer
    footer += "" if args.full else " (decimated for speed — pass --full for exact geometry)"
    fig.text(0.5, footer_y, textwrap.fill(footer, wrap_chars), ha="center", fontsize=8, color="gray")
    plt.tight_layout(rect=(0, footer_in / fig_height_in, 1, 1 - header_in / fig_height_in))
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_view(s):
    elev, azim = s.split(",")
    return (float(elev), float(azim))


def parse_color(s):
    if s.startswith("#"):
        s = s.lstrip("#")
        return tuple(int(s[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return tuple(float(x) for x in s.split(","))


def add_mesh_selection_args(p):
    p.add_argument("--scene", help="e.g. Bunny, Armadillo, Dragon, Lucy, Buddha, Statuette, XYZDragon")
    p.add_argument("--gt", help="explicit ground-truth mesh path, overrides --scene lookup")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list candidate reconstructions found for a scene")
    p_list.add_argument("--scene", required=True)
    p_list.set_defaults(func=cmd_list)

    p_explore = sub.add_parser("explore", help="render a grid of angles to help you pick --view")
    add_mesh_selection_args(p_explore)
    p_explore.add_argument("--recon", help="explicit reconstruction .ply path")
    p_explore.add_argument("--pick", type=int, help="index from `list` output, if --scene has multiple matches")
    p_explore.add_argument("--target", choices=["gt", "recon"], default="gt",
                            help="which mesh to explore angles on (default: gt)")
    p_explore.add_argument("--up", choices=["x", "y", "z"], default="y",
                            help="mesh axis that should point up on screen (default: y, true for the Stanford scans)")
    p_explore.add_argument("--color", type=parse_color, default=(0.75, 0.68, 0.55))
    p_explore.add_argument("--max-faces", type=int, default=150_000)
    p_explore.add_argument("--out", default="explore.png")
    p_explore.set_defaults(func=cmd_explore)

    p_render = sub.add_parser("render", help="render ground truth vs. reconstruction side by side")
    add_mesh_selection_args(p_render)
    p_render.add_argument("--recon", help="explicit reconstruction .ply path")
    p_render.add_argument("--pick", type=int, help="index from `list` output, if --scene has multiple matches")
    p_render.add_argument("--recon-only", action="store_true",
                           help="skip loading/plotting the ground-truth mesh — output is the reconstruction alone")
    p_render.add_argument("--view", action="append", type=parse_view,
                           help="elev,azim — repeatable, default two views: 10,45 and 10,225")
    p_render.add_argument("--up", choices=["x", "y", "z"], default="y",
                           help="mesh axis that should point up on screen (default: y, true for the Stanford scans)")
    p_render.add_argument("--gt-color", type=parse_color, default=(0.75, 0.68, 0.55))
    p_render.add_argument("--recon-color", type=parse_color, default=(0.75, 0.68, 0.55))
    p_render.add_argument("--max-faces", type=int, default=300_000,
                           help="decimation target for speed (default 300k); ignored with --full")
    p_render.add_argument("--full", action="store_true", help="no decimation — exact reconstructed geometry, slower")
    p_render.add_argument("--dpi", type=int, default=150, help="output resolution (default 150; use e.g. 300+ for print quality)")
    p_render.add_argument("--out", default="compare.png")
    p_render.set_defaults(func=cmd_render)

    args = parser.parse_args()
    if getattr(args, "scene", None) is None and getattr(args, "gt", None) is None and args.cmd != "list":
        parser.error("pass --scene or --gt")
    if getattr(args, "scene", None) is None and getattr(args, "recon", None) is None and args.cmd == "render":
        parser.error("render needs --scene (to auto-find a reconstruction) or --recon")
    args.func(args)


if __name__ == "__main__":
    main()
