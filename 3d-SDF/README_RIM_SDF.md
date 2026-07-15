# RIM-Hash for 3D signed distance fields

This directory extends the MetricGrids SDF pipeline with RIM-Hash. The 2D image
blocks become 3D cubes, while the same information-aware resolution selection and
gating strategy is retained.

The paper configuration fixes the selected gates and resolutions before fitting.
The launchers use that behavior by default; `RIM_GATE_MODE=trainable` is available
for exploratory runs.

## Setup

From the repository root:

```bash
pip install -r requirements-sdf.txt
```

## Prepare the Stanford shapes

The experiments use Armadillo, Bunny, Dragon, Buddha, Lucy, XYZDragon, and
Statuette from the
[Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/).
Place downloaded meshes under `3d-SDF/data/sdf/raw/`, then prepare one mesh with:

```bash
cd 3d-SDF
bash scripts/prepare_sdf_stanford.sh \
  --mesh data/sdf/raw/Armadillo.ply --name Armadillo
```

Running `bash scripts/prepare_sdf_stanford.sh` without a mesh creates the expected
directories and dataset manifest. The helper `scripts/download_sdf_stanford.sh`
prints the source locations when manual download is required.

## Train

Full RIM-Hash:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sdf_rim.sh \
  data/sdf/Armadillo_nrml.obj armadillo
```

Fixed-layout hash-grid baseline:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sdf_baseline.sh \
  data/sdf/Armadillo_nrml.obj armadillo
```

Use `DEBUG=1` before either command for a short smoke run. Results, checkpoints,
and selected RIM layouts are written under `outputs/`.

## Seven-shape sweep

The complete hash-budget sweep is split into four launchers:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sdf_stanford7_allkinds_part1.sh &
CUDA_VISIBLE_DEVICES=1 bash scripts/run_sdf_stanford7_allkinds_part2.sh &
CUDA_VISIBLE_DEVICES=2 bash scripts/run_sdf_stanford7_allkinds_part3.sh &
CUDA_VISIBLE_DEVICES=3 bash scripts/run_sdf_stanford7_allkinds_part4.sh &
wait
```

Set `KINDS="baseline rim_resolution rim_gate rim_full"` to include the paper's
component ablations in the same sweep.

## Attribution

The surrounding SDF pipeline is derived from MetricGrids and NeuRBF. Their MIT
license and the original MetricGrids README are included in this directory.
