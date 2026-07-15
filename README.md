# Retrievable Information Maximization (RIM)

RIM allocates a fixed capacity budget (e.g. hash-grid table slots, block features)
across spatial regions by solving a gated prefix-selection problem that maximizes
retrievable information under a resource constraint. This repository contains two
applications of the method:

- **2D images** (repository root): block-wise gating over image patches for
  implicit neural representation / compression-style fitting (Kodak, gigapixel,
  and whole-slide histology images).
- **3D signed distance fields** (`3d-SDF/`): cube-wise gating over a hash-grid
  encoder for SDF reconstruction of the Stanford 3D scanning repository meshes.

## Repository layout

- `rim/` — core RIM library: block/cube information scoring, the prefix gate
  solver, and the dynamic-programming resolution scheduler.
- `utils/` — exact and block-wise information estimators (`rim_exact.py`,
  `rim_blockwise.py`) and supporting metrics.
- `tools/` — offline block-info precomputation.
- `tests/` — smoke tests for the `rim` package.
- `thirdparty/torch_ngp/` — vendored hash-grid encoder (see `install.sh`).
- `hash_encoder_2d.py`, `train_giga_1.py`, `resize.py`, `shuffle.py` — 2D
  training/data-prep entry points; `run_rim_*.sh`, `fit_rim_*.sh`,
  `precompute_rim_*.sh`, `train_rim_*.sh` are the corresponding launch scripts
  for Kodak, gigapixel, and whole-slide-image experiments.
- `3d-SDF/` — the 3D SDF port of RIM.

## 3d-SDF

The code in `3d-SDF/` is heavily based on
[MetricGrids](https://github.com/wangshu31/MetricGrids) (CVPR 2025), which is
itself heavily based on [NeuRBF](https://github.com/oppo-us-research/NeuRBF).
It adds a RIM-gated hash-grid path (`3d-SDF/rim/`, `3d-SDF/img_sdf/rim_hashgrid.py`,
`3d-SDF/train_sdf_rim.py`) on top of the existing MetricGrids SDF pipeline,
porting the 2D RIM structure from this repository's root. See
`3d-SDF/README.md` for the original MetricGrids setup instructions and
`3d-SDF/README_RIM_SDF.md` for the RIM-specific SDF usage, and note MetricGrids'
own MIT license and attribution in `3d-SDF/LICENSE`.

## Notes

- No checkpoints or datasets are included in this repository. Scripts default
  to local dataset paths (e.g. Kodak, whole-slide images, Stanford meshes) that
  you will need to supply yourself; see the individual run scripts for the
  expected inputs, or the `--input`/`INPUT_IMAGE`/env-var overrides they expose.
- This is a research snapshot pulled from an internal working copy; some
  scripts still contain machine-specific fallback paths (conda env locations,
  default dataset paths) intended to be overridden via CLI args or environment
  variables.
