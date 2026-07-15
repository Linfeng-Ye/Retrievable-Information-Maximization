#!/usr/bin/env bash
set -euo pipefail

cat <<'TXT'
Automatic Stanford 3D Scanning Repository download URLs are not stable enough to
hard-code safely here.

Manual workflow:
1. Download the meshes used by MetricGrids/sdf.sh:
   Armadillo, Bunny, Dragon, Buddha, Lucy, XYZDragon, Statuette
   from https://graphics.stanford.edu/data/3Dscanrep/
2. Place them under data/sdf/raw, for example:
   data/sdf/raw/Armadillo.ply
3. Prepare one mesh:
   bash scripts/prepare_sdf_stanford.sh --mesh data/sdf/raw/Armadillo.ply --name Armadillo

The training scripts can also consume raw meshes directly; preparation mainly
writes normalized copies, eval samples, and manifest metadata.
TXT

