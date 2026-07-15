#!/usr/bin/env bash
set -euo pipefail

# Full Stanford SDF experiment batch for MetricGrids.
#
# Launch from anywhere:
#   CUDA_VISIBLE_DEVICES=0 bash /home/l44ye/MetricGrids/downloadSDF.sh
#
# Quick smoke run:
#   DEBUG=1 CUDA_VISIBLE_DEVICES=0 bash /home/l44ye/MetricGrids/downloadSDF.sh
#
# Defaults run the same ~900K-effective RIM-resolution setup for all 7 scenes:
#   MODE=rim
#   ENCODER=rim_resolution
#   RIM_GATE_MODE=fixed
#   LOG2_HASHMAP_SIZE=15
#   NUM_LEVELS=15
#   FEATURE_DIM=2
#
# Useful overrides:
#   MODE=both bash downloadSDF.sh
#   OUT_ROOT=outputs/my_sdf7 bash downloadSDF.sh
#   DO_DOWNLOAD=1 DO_PREPROC=1 RUN_EXPERIMENTS=0 bash downloadSDF.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"
cd "${REPO_DIR}"

DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/sdf}"
RAW_DIR="${RAW_DIR:-${DATA_DIR}/_raw}"

DO_DOWNLOAD="${DO_DOWNLOAD:-0}"
DO_PREPROC="${DO_PREPROC:-0}"
RUN_EXPERIMENTS="${RUN_EXPERIMENTS:-1}"

MODE="${MODE:-rim}" # rim, baseline, or both
ENCODER="${ENCODER:-rim_resolution}"
RIM_GATE_MODE="${RIM_GATE_MODE:-fixed}"
LOG2_HASHMAP_SIZE="${LOG2_HASHMAP_SIZE:-15}"
NUM_LEVELS="${NUM_LEVELS:-15}"
FEATURE_DIM="${FEATURE_DIM:-2}"
BASE_RESOLUTION="${BASE_RESOLUTION:-16}"
DESIRED_RESOLUTION="${DESIRED_RESOLUTION:-2048}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/outputs/sdf_stanford7_rim900k}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"

SCENES=(Armadillo Bunny Dragon Buddha Lucy XYZDragon Statuette)

if [[ -z "${PYTHON:-}" ]]; then
  for cand in python /home/l44ye/.conda/envs/INR/bin/python3 /home/l44ye/.conda/envs/py310/bin/python3 python3; do
    if command -v "${cand}" >/dev/null 2>&1 || [[ -x "${cand}" ]]; then
      PYTHON="${cand}"
      break
    fi
  done
fi

if [[ -z "${PYTHON:-}" ]]; then
  echo "[ERROR] Could not find Python. Set PYTHON=/path/to/python." >&2
  exit 1
fi

mkdir -p "${DATA_DIR}" "${RAW_DIR}" "${LOG_ROOT}" "${OUT_ROOT}"

download() {
  local url="$1"
  local out="$2"
  echo "[download] ${out}"
  wget -c -O "${out}" "${url}"
}

extract_tar_pick_ply() {
  local name="$1"
  local url="$2"
  local preferred="$3"
  local tarfile="${RAW_DIR}/${name}.tar.gz"
  local tmpdir="${RAW_DIR}/extract_${name}"

  download "${url}" "${tarfile}"
  rm -rf "${tmpdir}"
  mkdir -p "${tmpdir}"
  tar -xzf "${tarfile}" -C "${tmpdir}"

  local src=""
  src="$(find "${tmpdir}" -type f -iname "${preferred}" | head -n 1 || true)"
  if [[ -z "${src}" ]]; then
    src="$(find "${tmpdir}" -type f -iname "*.ply" -printf "%s\t%p\n" | sort -nr | head -n 1 | cut -f2- || true)"
  fi
  if [[ -z "${src}" ]]; then
    echo "[WARN] Cannot find .ply for ${name}; skipping download for this scene." >&2
    return 1
  fi

  cp "${src}" "${DATA_DIR}/${name}.ply"
  echo "[ok] ${DATA_DIR}/${name}.ply"
}

download_gz_ply() {
  local name="$1"
  local url="$2"
  local gzfile="${RAW_DIR}/${name}.ply.gz"

  download "${url}" "${gzfile}"
  gzip -cd "${gzfile}" > "${DATA_DIR}/${name}.ply"
  echo "[ok] ${DATA_DIR}/${name}.ply"
}

download_all() {
  command -v wget >/dev/null || { echo "[ERROR] Please install wget." >&2; exit 1; }
  command -v tar >/dev/null || { echo "[ERROR] Please install tar." >&2; exit 1; }
  command -v gzip >/dev/null || { echo "[ERROR] Please install gzip." >&2; exit 1; }

  extract_tar_pick_ply "Bunny" "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz" "bun_zipper.ply" || true
  download_gz_ply "Armadillo" "http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz" || true
  extract_tar_pick_ply "Dragon" "http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz" "dragon_vrip.ply" || true
  extract_tar_pick_ply "Buddha" "http://graphics.stanford.edu/pub/3Dscanrep/happy/happy_recon.tar.gz" "happy_vrip.ply" || true
  extract_tar_pick_ply "Lucy" "http://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz" "lucy.ply" || true
  download_gz_ply "XYZDragon" "http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_dragon.ply.gz" || true
  download_gz_ply "Statuette" "http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_statuette.ply.gz" || true
}

preprocess_scene_if_requested() {
  local name="$1"
  local raw_mesh="${DATA_DIR}/${name}.ply"
  local mesh="${DATA_DIR}/${name}_nrml.obj"

  if [[ -f "${mesh}" ]]; then
    return 0
  fi
  if [[ "${DO_PREPROC}" != "1" && "${DO_PREPROC}" != "true" ]]; then
    return 0
  fi
  if [[ ! -f "${raw_mesh}" ]]; then
    echo "[WARN] ${name}: missing raw mesh ${raw_mesh}; cannot preprocess." >&2
    return 1
  fi

  echo "[preprocess] ${name}: ${raw_mesh}"
  "${PYTHON}" preproc_mesh.py --path "${raw_mesh}"
}

run_with_log() {
  local name="$1"
  local kind="$2"
  local mesh="$3"
  local out_dir="$4"
  local log_file="$5"

  mkdir -p "$(dirname "${log_file}")" "${out_dir}"
  echo "[RUN] ${name} ${kind}"
  echo "[LOG] ${log_file}"

  set +e
  (
    set -euo pipefail
    export PYTHON
    export OUT_DIR="${out_dir}"
    export LOG2_HASHMAP_SIZE NUM_LEVELS FEATURE_DIM BASE_RESOLUTION DESIRED_RESOLUTION
    export ENCODER RIM_GATE_MODE
    echo "[SCENE] ${name}"
    echo "[KIND] ${kind}"
    echo "[MESH] ${mesh}"
    echo "[OUT_DIR] ${OUT_DIR}"
    echo "[CONFIG] ENCODER=${ENCODER} RIM_GATE_MODE=${RIM_GATE_MODE} LOG2_HASHMAP_SIZE=${LOG2_HASHMAP_SIZE} NUM_LEVELS=${NUM_LEVELS} FEATURE_DIM=${FEATURE_DIM}"
    date
    if [[ "${kind}" == "baseline" ]]; then
      bash scripts/run_sdf_baseline.sh "${mesh}" "${name}"
    else
      bash scripts/run_sdf_rim.sh "${mesh}" "${name}"
    fi
    date
    echo "[DONE] ${name} ${kind}"
  ) 2>&1 | tee "${log_file}"
  local status="${PIPESTATUS[0]}"
  set -e

  if [[ "${status}" != "0" ]]; then
    echo "[WARN] ${name} ${kind}: failed with status ${status}; continuing batch." >&2
    return 1
  fi
  return 0
}

run_scene() {
  local name="$1"
  local mesh="${DATA_DIR}/${name}_nrml.obj"
  local eval_points="${DATA_DIR}/${name}_nrml_eval_points.pt"
  local scene_out="${OUT_ROOT}/${name}"
  local failed=0

  preprocess_scene_if_requested "${name}" || true

  if [[ ! -f "${mesh}" ]]; then
    echo "[WARN] ${name}: required normalized mesh missing: ${mesh}. Skipping this scene." >&2
    return 0
  fi
  if [[ ! -f "${eval_points}" ]]; then
    echo "[WARN] ${name}: eval points missing: ${eval_points}. Training will run, but eval-point metrics may be absent." >&2
  fi
  if [[ "${mesh}" != *.obj && "${mesh}" != *.ply ]]; then
    echo "[WARN] ${name}: unexpected mesh format ${mesh}. Skipping this scene." >&2
    return 0
  fi

  if [[ "${MODE}" == "baseline" || "${MODE}" == "both" ]]; then
    run_with_log "${name}" "baseline" "${mesh}" "${scene_out}/baseline" "${LOG_ROOT}/${name}_baseline.txt" || failed=1
  fi
  if [[ "${MODE}" == "rim" || "${MODE}" == "both" ]]; then
    run_with_log "${name}" "rim" "${mesh}" "${scene_out}/rim" "${LOG_ROOT}/${name}_rim_${ENCODER}.txt" || failed=1
  fi

  return "${failed}"
}

case "${MODE}" in
  baseline|rim|both) ;;
  *)
    echo "[ERROR] Unknown MODE=${MODE}; expected baseline, rim, or both." >&2
    exit 2
    ;;
esac

case "${ENCODER}" in
  rim_full|rim_gate|rim_resolution|fixed_hashgrid) ;;
  *)
    echo "[ERROR] Unknown ENCODER=${ENCODER}; expected rim_full, rim_gate, rim_resolution, or fixed_hashgrid." >&2
    exit 2
    ;;
esac

echo "[INFO] Repo: ${REPO_DIR}"
echo "[INFO] Data dir: ${DATA_DIR}"
echo "[INFO] Output root: ${OUT_ROOT}"
echo "[INFO] Log root: ${LOG_ROOT}"

if [[ "${DO_DOWNLOAD}" == "1" || "${DO_DOWNLOAD}" == "true" ]]; then
  download_all
fi

if [[ "${RUN_EXPERIMENTS}" != "1" && "${RUN_EXPERIMENTS}" != "true" ]]; then
  echo "[DONE] RUN_EXPERIMENTS=${RUN_EXPERIMENTS}; no experiments launched."
  exit 0
fi

skipped_or_failed=0
for scene in "${SCENES[@]}"; do
  run_scene "${scene}" || skipped_or_failed=1
done

echo
echo "[DONE] Batch complete."
echo "[DONE] Results: ${OUT_ROOT}"
echo "[DONE] Logs: ${LOG_ROOT}"
if [[ "${skipped_or_failed}" != "0" ]]; then
  echo "[DONE] One or more scenes failed or were skipped; see warnings/logs above." >&2
fi
