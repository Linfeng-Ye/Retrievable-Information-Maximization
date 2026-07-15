#!/usr/bin/env bash
set -euo pipefail

# Sequential runner for the 16-level Stanford7 variant. All four scene splits
# share one output root and run one at a time on the selected GPU.
#
# Usage:
#   bash scripts/run_sdf_stanford7_allkinds_16levels_metricgrids_decoder.sh
#
# Override the GPU or output root:
#   GPU=5 OUT_ROOT=outputs/my_variant bash scripts/run_sdf_stanford7_allkinds_16levels_metricgrids_decoder.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

GPU="${GPU:-0}"
export OUT_ROOT="${OUT_ROOT:-outputs/sdf_stanford7_allkinds_16levels_metricgrids_decoder}"
export NUM_LEVELS="${NUM_LEVELS:-16}"

LOG_DIR="logs/rerun_16levels_metricgrids_decoder_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

FAILURES=0

for part_num in 1 2 3 4; do
  script="scripts/run_sdf_stanford7_allkinds_part${part_num}.sh"
  log_file="${LOG_DIR}/part${part_num}.log"

  echo "[RUN] part${part_num} on GPU ${GPU} -> ${OUT_ROOT} (NUM_LEVELS=${NUM_LEVELS})"
  echo "[LOG] ${log_file}"
  if CUDA_VISIBLE_DEVICES="${GPU}" OUT_ROOT="${OUT_ROOT}" NUM_LEVELS="${NUM_LEVELS}" \
       bash "${script}" 2>&1 | tee "${log_file}"; then
    echo "[OK] part${part_num}"
  else
    echo "[FAIL] part${part_num} -- see ${log_file}" >&2
    FAILURES=$((FAILURES + 1))
  fi
done

echo "[DONE] logs in ${LOG_DIR}"
if (( FAILURES > 0 )); then
  echo "[ERROR] ${FAILURES} part(s) failed." >&2
  exit 1
fi
