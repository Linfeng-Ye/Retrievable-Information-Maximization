# RIM SDF in MetricGrids

This adds a 3D SDF-only RIM-Hash path to the existing MetricGrids SDF pipeline.
It does not implement NeRF and does not replace the original 2D image code.

The implementation ports the 2D RIM structure from the repository root (`../rim`, `../utils`):

- block/cube information tensor `A[b, j]`;
- RIM-style prefix gate solve using `q(alpha) = (1 - exp(-alpha)) / alpha`;
- fixed-gate dynamic-programming schedule over contiguous candidate bands;
- gated hash-grid lookup with learned blockwise fallback features.

For SDF, 2D blocks become 3D cubes and bilinear hash interpolation becomes trilinear interpolation. The default information metric is residual entropy; residual energy and a zero-level-set-weighted hybrid are available with `--rim_information_metric energy` or `energy_entropy`.

Intentional deviation from the RIM paper: gates are trainable by default after initialization:

```bash
--encoder rim_full --rim_gate_trainable
```

Gates are parameterized as logits:

```text
g_l,b = sigmoid(a_l,b)
```

Use `--rim_gate_fixed` to freeze initialized gates exactly. Selected level resolutions are fixed after initialization; `--rim_train_resolution` is reserved and currently raises `NotImplementedError`.

## Dataset

MetricGrids already encodes the seven SDF scenes in `sdf.sh`: Armadillo, Bunny, Dragon, Buddha, Lucy, XYZDragon, and Statuette. The manifest at `data/sdf/manifest_stanford.json` reuses that scene list.

Create directories and manifest:

```bash
bash scripts/prepare_sdf_stanford.sh
```

Prepare one mesh:

```bash
bash scripts/prepare_sdf_stanford.sh --mesh /path/to/Armadillo.ply --name Armadillo
```

If automatic download is needed, run:

```bash
bash scripts/download_sdf_stanford.sh
```

It prints manual Stanford repository instructions because stable direct mesh URLs are not assumed.

## Runs

Baseline fixed hash grid:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sdf_baseline.sh data/sdf/raw/Armadillo.ply armadillo
```

RIM full with trainable gates:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_sdf_rim.sh data/sdf/raw/Armadillo.ply armadillo
```

The default full method uses trainable block gates, blockwise fallback features,
and up to 20 alternating gate/scheduler updates. Run the seven-scene hash-size
sweep across four GPUs with:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sdf_stanford7_allkinds_part1.sh &
CUDA_VISIBLE_DEVICES=1 bash scripts/run_sdf_stanford7_allkinds_part2.sh &
CUDA_VISIBLE_DEVICES=2 bash scripts/run_sdf_stanford7_allkinds_part3.sh &
CUDA_VISIBLE_DEVICES=3 bash scripts/run_sdf_stanford7_allkinds_part4.sh &
wait
```

These split scripts run `LOG2_HASHMAP_SIZES="14 14.5 15"` by default.

Quick smoke mode:

```bash
DEBUG=1 bash scripts/run_sdf_rim.sh data/sdf/raw/Armadillo.ply armadillo
```

Useful overrides:

```bash
STEPS=500 BATCH_SIZE=8192 LR=1e-4 LOG2_HASHMAP_SIZE=12 NUM_LEVELS=4 FEATURE_DIM=2 \
RIM_ANALYSIS_RESOLUTION=32 RIM_CUBE_SIZE=8 RIM_NUM_CANDIDATES=8 RIM_ITERS=20 \
RIM_INFORMATION_METRIC=entropy RIM_FALLBACK_MODE=blockwise \
OUT_DIR=outputs/debug bash scripts/run_sdf_rim.sh data/sdf/raw/Armadillo.ply armadillo
```

Outputs include:

- `outputs/.../results/<run>.json`;
- `outputs/.../checkpoints/<run>_ep....pth`;
- `outputs/.../run/<run>/rim_debug/rim_info_A.npy`;
- `rim_gates_init.npy`, `rim_gates_final.npy`, `rim_gate_logits_final.npy`;
- `rim_selected_resolutions.json`, `rim_scheduler_stats.json`, `rim_gate_stats_final.json`.
