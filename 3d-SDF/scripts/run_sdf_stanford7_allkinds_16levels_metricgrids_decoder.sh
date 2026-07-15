#!/usr/bin/env bash
set -euo pipefail

# Re-run the Stanford7 "allkinds" rim_full sweep (log2 hashmap sizes 14/14.5/15,
# all 7 scenes) with:
#   - NUM_LEVELS=16 (was 15)
#   - RIMSDFNGP's decoder now includes the MertricEmbedding 3-branch positional
#     pre-transform ported from img_sdf/ngp.py's NGP class (the original
#     MetricGrids decoder), instead of feeding raw coordinates straight into
#     the encoder. This applies unconditionally to every RIMSDFNGP run now,
#     so there's no flag to toggle it back off.
#
# All 4 scene-split part scripts write into the SAME output root (matching how
# the earlier "_metricfix_seq" results were actually produced -- see
# outputs/sdf_stanford7_allkinds_part1_metricfix_seq/logs/log2_*/*.txt, which
# show all 7 scenes sharing that one OUT_ROOT despite the "part1" name), one
# part at a time (no backgrounding, no multi-GPU split) so only one experiment
# is training at any given moment.
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
