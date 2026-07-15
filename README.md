# Retrievable Information Maximization

Code for **Retrievable Information Maximization for Efficient Multiresolution
Hash Encoding**.

RIM-Hash makes multiresolution hash encodings data-aware. It selects the grid
resolutions and the spatial blocks that should use each hash level, concentrating
capacity where signal information is most useful and recoverable after collisions.

The repository covers the experiments in the paper:

- 2D image fitting on Kodak, gigapixel natural images, and whole-slide images.
- 3D signed-distance-field fitting on shapes from the Stanford 3D Scanning Repository.

## Setup

The reference environment uses Python 3.10 and a CUDA-enabled PyTorch build. Create
an environment and install the lightweight 2D dependencies with:

```bash
conda create -n rim python=3.10 -y
conda activate rim
pip install -r requirements.txt
```

For the 3D SDF experiments, also install:

```bash
pip install -r requirements-sdf.txt
```

## Data

Datasets and checkpoints are not redistributed. Place local data under `data/`;
this directory is ignored by Git.

- Download the 24 images from the [Kodak image suite](https://r0k.us/graphics/kodak/)
  into `data/Kodak/`.
- Supply the Pluto, Tokyo, Girl, or WSI image directly to the image launcher.
- Follow [`3d-SDF/README_RIM_SDF.md`](3d-SDF/README_RIM_SDF.md) to prepare the
  seven Stanford meshes.

## Run the image experiments

The image workflow first measures block-wise information, then fits RIM-Hash. For
one image:

```bash
CONDA_ENV=rim WIDTH=8000 HEIGHT=8000 RUN_NAME=pluto \
  bash scripts/run_rim_iterative.sh data/gigapixel/pluto.jpg
```

For the Kodak sweep:

```bash
CONDA_ENV=rim KODAK_DIR=data/Kodak GPUS="0" \
  bash scripts/run_rim_kodak_grid.sh
```

Outputs are written to `runs/` and logs to `logs/`. The launchers expose the main
settings as environment variables so that hash budgets and GPU assignments can be
changed without editing source files.

## Run the 3D SDF experiments

After preparing a mesh, run the full RIM-Hash model with:

```bash
cd 3d-SDF
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sdf_rim.sh \
  data/sdf/Armadillo_nrml.obj armadillo
```

A matching fixed-layout hash-grid baseline is available through
`scripts/run_sdf_baseline.sh`. The full seven-shape sweep is documented in the SDF
README linked above.

## Repository layout

- `rim/`: RIM information allocation, gate solver, and resolution scheduler.
- `hash_encoder_2d.py` and `train_giga_1.py`: 2D RIM-Hash model and trainer.
- `tools/` and `scripts/`: information precomputation and experiment launchers.
- `3d-SDF/`: 3D SDF implementation, preparation scripts, and evaluation pipeline.
- `tests/`: fast CPU smoke tests for the core 2D implementation.

## Testing

```bash
pytest -q
```

## Acknowledgements

The SDF pipeline builds on
[MetricGrids](https://github.com/wangshu31/MetricGrids), which builds on
[NeuRBF](https://github.com/oppo-us-research/NeuRBF). Vendored third-party code
retains its original license and attribution.

## License

The RIM code is released under the MIT License. See [`LICENSE`](LICENSE) and the
license files inside third-party directories.
