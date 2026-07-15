#!/usr/bin/env python3
"""
Visualize reconstructed SDF surfaces produced by MetricGrids training runs.

Each training run writes its result mesh (the zero level-set of the learned
SDF, extracted via marching cubes) to:

    outputs/<run_name>/log2_*/<mesh>/<encoder>/results/<mesh>-<log2_*>-<encoder>.ply

That .ply is what this script visualizes. It is a binary_little_endian PLY
(float32 x,y,z vertices + uchar/int face index lists), as exported by
trimesh. Metrics for the same run live alongside it in the matching .txt/.json.

Backends (auto-picked, best available first): open3d > plotly > matplotlib.
A minimal built-in PLY reader is used so the script also works with nothing
but numpy installed; if `trimesh` or `plyfile` are present they're used
instead (more robust for edge-case PLY files).

Examples
--------
# Visualize one result directly
python visualize_sdf_results.py outputs/.../results/Armadillo-log2_14-rim_full.ply

# Find results by mesh name under an experiment folder and pick one
python visualize_sdf_results.py --search outputs/sdf_stanford7_allkinds_part1_metricfix_seq --name Armadillo

# Force a backend / cap face count for smoother interaction
python visualize_sdf_results.py result.ply --backend plotly --max-faces 300000
"""

import argparse
import struct
import sys
import webbrowser
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# PLY loading
# --------------------------------------------------------------------------

def _read_ply_builtin(path: Path):
    """Minimal PLY reader covering the binary_little_endian / ascii meshes
    that MetricGrids writes (float vertices + polygon face lists)."""
    with open(path, "rb") as f:
        raw = f.read()

    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:header_end].decode("ascii", errors="replace")
    body = raw[header_end:]

    lines = [l.strip() for l in header.splitlines()]
    fmt = None
    elements = []  # list of dicts: name, count, properties [(type, name)]
    cur = None
    for line in lines:
        if line.startswith("format"):
            fmt = line.split()[1]  # ascii | binary_little_endian | binary_big_endian
        elif line.startswith("element"):
            _, name, count = line.split()
            cur = {"name": name, "count": int(count), "properties": []}
            elements.append(cur)
        elif line.startswith("property list"):
            _, _, count_t, val_t, name = line.split()
            cur["properties"].append(("list", count_t, val_t, name))
        elif line.startswith("property"):
            _, t, name = line.split()
            cur["properties"].append(("scalar", t, name))

    type_map = {
        "float": ("f", 4), "float32": ("f", 4),
        "double": ("d", 8), "float64": ("d", 8),
        "int": ("i", 4), "int32": ("i", 4),
        "uint": ("I", 4), "uint32": ("I", 4),
        "short": ("h", 2), "int16": ("h", 2),
        "ushort": ("H", 2), "uint16": ("H", 2),
        "char": ("b", 1), "int8": ("b", 1),
        "uchar": ("B", 1), "uint8": ("B", 1),
    }
    endian = "<" if fmt != "binary_big_endian" else ">"

    vertices = None
    faces = []
    offset = 0

    if fmt == "ascii":
        text = body.decode("ascii", errors="replace").split()
        ptr = 0
        for elem in elements:
            if elem["name"] == "vertex":
                cols = len(elem["properties"])
                verts = np.array(text[ptr:ptr + elem["count"] * cols], dtype=np.float64)
                verts = verts.reshape(elem["count"], cols)[:, :3]
                vertices = verts.astype(np.float32)
                ptr += elem["count"] * cols
            elif elem["name"] == "face":
                for _ in range(elem["count"]):
                    n = int(text[ptr]); idx = list(map(int, text[ptr + 1:ptr + 1 + n]))
                    ptr += 1 + n
                    for k in range(1, n - 1):  # fan-triangulate polygons
                        faces.append((idx[0], idx[k], idx[k + 1]))
            else:
                cols = len(elem["properties"])
                ptr += elem["count"] * cols
    else:
        for elem in elements:
            if elem["name"] == "vertex" and all(p[0] == "scalar" for p in elem["properties"]):
                names = [p[2] for p in elem["properties"]]
                fmt_str = endian + "".join(type_map[p[1]][0] for p in elem["properties"])
                size = struct.calcsize(fmt_str)
                arr = np.frombuffer(body, dtype=np.dtype(
                    [(p[2], endian + type_map[p[1]][0]) for p in elem["properties"]]
                ), count=elem["count"], offset=offset)
                offset += size * elem["count"]
                xi, yi, zi = names.index("x"), names.index("y"), names.index("z")
                vertices = np.stack([arr[names[xi]], arr[names[yi]], arr[names[zi]]], axis=1).astype(np.float32)
            elif elem["name"] == "face":
                count_t = None
                for p in elem["properties"]:
                    if p[0] == "list":
                        count_t = p
                count_code, val_code = type_map[count_t[1]][0], type_map[count_t[2]][0]
                count_size, val_size = type_map[count_t[1]][1], type_map[count_t[2]][1]
                for _ in range(elem["count"]):
                    n = struct.unpack_from(endian + count_code, body, offset)[0]
                    offset += count_size
                    idx = struct.unpack_from(endian + val_code * n, body, offset)
                    offset += val_size * n
                    for k in range(1, n - 1):  # fan-triangulate polygons
                        faces.append((idx[0], idx[k], idx[k + 1]))
            else:
                # skip unknown element block using its fixed-size scalar properties
                if all(p[0] == "scalar" for p in elem["properties"]):
                    rec_size = sum(type_map[p[1]][1] for p in elem["properties"])
                    offset += rec_size * elem["count"]

    if vertices is None:
        raise ValueError(f"Could not find a 'vertex' element in {path}")
    faces = np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64)
    return vertices, faces


def load_mesh(path: Path):
    """Returns (vertices Nx3 float32, faces Mx3 int64). Prefers trimesh/plyfile
    if installed, falls back to the built-in reader."""
    try:
        import trimesh
        m = trimesh.load(path, process=False, force="mesh")
        return np.asarray(m.vertices, dtype=np.float32), np.asarray(m.faces, dtype=np.int64)
    except ImportError:
        pass
    except Exception as e:
        print(f"[warn] trimesh failed to load ({e}), falling back to built-in reader", file=sys.stderr)

    try:
        from plyfile import PlyData
        ply = PlyData.read(str(path))
        v = ply["vertex"]
        vertices = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
        faces = np.asarray([f[0] for f in ply["face"].data], dtype=object)
        tris = []
        for f in faces:
            for k in range(1, len(f) - 1):
                tris.append((f[0], f[k], f[k + 1]))
        return vertices, np.asarray(tris, dtype=np.int64)
    except ImportError:
        pass
    except Exception as e:
        print(f"[warn] plyfile failed to load ({e}), falling back to built-in reader", file=sys.stderr)

    return _read_ply_builtin(path)


def decimate_faces(faces: np.ndarray, max_faces: int, seed: int = 0):
    """Randomly subsample faces (fast, dependency-free) to keep interactive
    backends responsive. Not a true quadric decimation -- if you need that,
    install trimesh/open3d and use its simplify_quadric_decimation instead."""
    if max_faces <= 0 or len(faces) <= max_faces:
        return faces
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(faces), size=max_faces, replace=False)
    keep.sort()
    return faces[keep]


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

def show_open3d(vertices, faces, title):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.65, 0.7, 0.75])
    o3d.visualization.draw_geometries([mesh], window_name=title, mesh_show_back_face=True)


def show_plotly(vertices, faces, title, out_html: Path):
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="lightsteelblue",
        flatshading=False,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.6),
        lightposition=dict(x=100, y=200, z=150),
        showscale=False,
    )])
    fig.update_layout(
        title=f"{title}  ({len(vertices):,} verts / {len(faces):,} faces)",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    print(f"Wrote interactive viewer to: {out_html}")
    webbrowser.open(f"file://{out_html.resolve()}")


def show_matplotlib(vertices, faces, title):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    mesh_faces = vertices[faces]  # (M, 3, 3)
    coll = Poly3DCollection(mesh_faces, facecolor="lightsteelblue", edgecolor=None, linewidths=0, alpha=1.0)
    ax.add_collection3d(coll)

    mins, maxs = vertices.min(axis=0), vertices.max(axis=0)
    center, radius = (mins + maxs) / 2, (maxs - mins).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(f"{title}  ({len(vertices):,} verts / {len(faces):,} faces)")
    plt.tight_layout()
    plt.show()


BACKENDS = {"open3d": show_open3d, "plotly": show_plotly, "matplotlib": show_matplotlib}


def pick_backend(preferred: str):
    if preferred != "auto":
        return preferred
    for name in ("open3d", "plotly", "matplotlib"):
        try:
            __import__(name)
            return name
        except ImportError:
            continue
    return None


# --------------------------------------------------------------------------
# Search helper (results are deeply nested under outputs/)
# --------------------------------------------------------------------------

def search_results(root: Path, name_filter: str):
    hits = sorted(root.rglob("results/*.ply"))
    if name_filter:
        hits = [h for h in hits if name_filter.lower() in h.name.lower()]
    return hits


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ply", nargs="?", type=Path, help="Path to a MetricGrids result .ply file")
    p.add_argument("--search", type=Path, help="Search this directory tree for result .ply files instead")
    p.add_argument("--name", default="", help="Substring filter on mesh name when using --search (e.g. Armadillo)")
    p.add_argument("--backend", choices=["auto", "open3d", "plotly", "matplotlib"], default="auto",
                   help="Rendering backend (default: auto-detect best installed)")
    p.add_argument("--max-faces", type=int, default=500_000,
                   help="Randomly subsample faces above this count for responsiveness (0 = no limit)")
    p.add_argument("--out-html", type=Path, default=None,
                   help="Output path for the plotly backend's HTML viewer")
    args = p.parse_args()

    if args.search:
        hits = search_results(args.search, args.name)
        if not hits:
            print(f"No result .ply files found under {args.search} (filter={args.name!r})")
            return
        if len(hits) == 1 or args.ply is None:
            print(f"Found {len(hits)} matching result(s):")
            for i, h in enumerate(hits):
                print(f"  [{i}] {h.relative_to(args.search)}")
            if len(hits) > 1 and args.ply is None:
                choice = input(f"Pick an index [0-{len(hits) - 1}]: ").strip()
                ply_path = hits[int(choice)]
            else:
                ply_path = hits[0]
        else:
            ply_path = args.ply
    elif args.ply is not None:
        ply_path = args.ply
    else:
        p.error("Provide a .ply path, or use --search <dir> [--name <substr>]")
        return

    if not ply_path.exists():
        print(f"File not found: {ply_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading mesh: {ply_path}")
    vertices, faces = load_mesh(ply_path)
    print(f"  {len(vertices):,} vertices, {len(faces):,} faces")

    faces = decimate_faces(faces, args.max_faces)
    if args.max_faces > 0:
        print(f"  using {len(faces):,} faces for display (--max-faces {args.max_faces}, 0 = no limit)")

    backend = pick_backend(args.backend)
    if backend is None:
        print(
            "No visualization backend is installed.\n"
            "Install one of the following and re-run:\n"
            "  pip install plotly        # easiest: renders to a browser-viewable HTML file\n"
            "  pip install open3d        # best for interactive local viewing\n"
            "  pip install matplotlib    # simplest, but slow for meshes this large",
            file=sys.stderr,
        )
        sys.exit(1)

    title = ply_path.stem
    print(f"Rendering with backend: {backend}")
    if backend == "plotly":
        out_html = args.out_html or ply_path.with_suffix(".html")
        show_plotly(vertices, faces, title, out_html)
    elif backend == "open3d":
        show_open3d(vertices, faces, title)
    else:
        show_matplotlib(vertices, faces, title)


if __name__ == "__main__":
    main()
