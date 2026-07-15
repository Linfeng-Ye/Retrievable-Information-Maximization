#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_sdf_complete.sh [mesh_path] [alias]

Examples:
  DEBUG=1 MODE=rim bash scripts/train_sdf_complete.sh data/sdf/raw/Armadillo.ply armadillo
  MODE=both CUDA_VISIBLE_DEVICES=0 bash scripts/train_sdf_complete.sh /path/to/Armadillo.ply armadillo

Environment overrides:
  MODE=baseline|rim|both        Default: rim
  ENCODER=rim_full|rim_resolution|rim_gate|fixed_hashgrid
                                Default for RIM mode: rim_full
  RIM_GATE_MODE=trainable|fixed Default: trainable
  PREPARE=auto|1|0             Default: auto
  PREP_SKIP_EVAL=1|0           Default: 1
  PYTHON=/path/to/python       Default: auto-detected
  DEBUG=1                      Short smoke run

Training overrides:
  STEPS BATCH_SIZE LR LOG2_HASHMAP_SIZE NUM_LEVELS FEATURE_DIM HIDDEN_DIM
  BASE_RESOLUTION DESIRED_RESOLUTION SAVE_PRED EVAL_MESH_METRICS
  RIM_ANALYSIS_RESOLUTION RIM_CUBE_SIZE RIM_NUM_CANDIDATES RIM_ITERS
  RIM_GATE_LR RIM_GATE_REG RIM_GATE_SPARSITY_WEIGHT RIM_FALLBACK_MODE
  RIM_INFORMATION_METRIC
  VAL_RESOLUTION VAL_NUM_SAMPLES TRAIN_EPOCH_SIZE
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MESH_IN="${1:-data/sdf/raw/Armadillo.ply}"
ALIAS="${2:-$(basename "${MESH_IN%.*}")}"
SCENE_NAME="${SCENE_NAME:-${ALIAS}}"
MODE="${MODE:-rim}"
PREPARE="${PREPARE:-auto}"
PREP_SKIP_EVAL="${PREP_SKIP_EVAL:-1}"

case "${MODE}" in
  baseline|rim|both) ;;
  *)
    echo "Unknown MODE=${MODE}; expected baseline, rim, or both." >&2
    exit 2
    ;;
esac

if [[ -z "${PYTHON:-}" ]]; then
  for cand in python /home/l44ye/.conda/envs/INR/bin/python3 /home/l44ye/.conda/envs/py310/bin/python3 python3; do
    if command -v "${cand}" >/dev/null 2>&1 || [[ -x "${cand}" ]]; then
      PYTHON="${cand}"
      break
    fi
  done
fi

if [[ -z "${PYTHON:-}" ]]; then
  echo "Could not find a Python executable. Set PYTHON=/path/to/python." >&2
  exit 1
fi

if [[ ! -f "${MESH_IN}" ]]; then
  echo "Mesh not found: ${MESH_IN}" >&2
  echo "Put the mesh under data/sdf/raw or pass /path/to/mesh.ply." >&2
  exit 1
fi

should_prepare=0
if [[ "${PREPARE}" == "1" || "${PREPARE}" == "true" ]]; then
  should_prepare=1
elif [[ "${PREPARE}" == "auto" ]]; then
  if [[ ! "${MESH_IN}" == data/sdf/raw/* ]]; then
    should_prepare=1
  fi
fi

if [[ "${should_prepare}" == "1" ]]; then
  prep_args=(--mesh "${MESH_IN}" --name "${SCENE_NAME}")
  if [[ "${PREP_SKIP_EVAL}" == "1" || "${PREP_SKIP_EVAL}" == "true" ]]; then
    prep_args+=(--skip-eval)
  fi
  echo "[SDF] Preparing mesh ${MESH_IN} as ${SCENE_NAME}"
  PYTHON="${PYTHON}" bash scripts/prepare_sdf_stanford.sh "${prep_args[@]}"
  ext="${MESH_IN##*.}"
  MESH="data/sdf/raw/${SCENE_NAME}.${ext}"
else
  MESH="${MESH_IN}"
fi

echo "[SDF] Python: ${PYTHON}"
echo "[SDF] Mesh: ${MESH}"
echo "[SDF] Alias: ${ALIAS}"
echo "[SDF] Mode: ${MODE}"

if [[ "${MODE}" == "baseline" || "${MODE}" == "both" ]]; then
  echo "[SDF] Running fixed hash-grid baseline"
  PYTHON="${PYTHON}" bash scripts/run_sdf_baseline.sh "${MESH}" "${ALIAS}"
fi

if [[ "${MODE}" == "rim" || "${MODE}" == "both" ]]; then
  echo "[SDF] Running RIM SDF with trainable sigmoid gates"
  PYTHON="${PYTHON}" bash scripts/run_sdf_rim.sh "${MESH}" "${ALIAS}"
fi

echo "[SDF] Done."
