#!/usr/bin/env bash
set -euo pipefail

# GPU split 3/4 -- full RIM sweep; see part1 for details.
#
# Example:
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_sdf_stanford7_allkinds_part3.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

SCENES=(Statuette)
PART_LABEL="part3/4"
DEFAULT_OUT_ROOT="outputs/sdf_stanford7_allkinds_part3"

export KINDS="${KINDS:-rim_full}"
export LOG2_HASHMAP_SIZES="${LOG2_HASHMAP_SIZES:-14 14.5 15}"
export BATCH_SIZE="${BATCH_SIZE:-2097152}"
export RIM_ITERS="${RIM_ITERS:-20}"
export RIM_GATE_MODE="${RIM_GATE_MODE:-trainable}"
export RIM_FALLBACK_MODE="${RIM_FALLBACK_MODE:-blockwise}"

source scripts/run_sdf_stanford7_log2_sweep_common.sh
run_log2_sweep
