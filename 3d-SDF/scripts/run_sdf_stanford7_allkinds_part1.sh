#!/usr/bin/env bash
set -euo pipefail

# GPU split 1/4 -- full RIM method with trainable block gates and iterative
# gate/scheduler updates at log2 hash sizes 14, 14.5, and 15.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_sdf_stanford7_allkinds_part1.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

SCENES=(Bunny)
PART_LABEL="part1/4"
DEFAULT_OUT_ROOT="outputs/sdf_stanford7_allkinds_part1"

export KINDS="${KINDS:-rim_full}"
export LOG2_HASHMAP_SIZES="${LOG2_HASHMAP_SIZES:-14 14.5 15}"
export BATCH_SIZE="${BATCH_SIZE:-1048576}"
export RIM_ITERS="${RIM_ITERS:-20}"
export RIM_GATE_MODE="${RIM_GATE_MODE:-trainable}"
export RIM_FALLBACK_MODE="${RIM_FALLBACK_MODE:-blockwise}"

source scripts/run_sdf_stanford7_log2_sweep_common.sh
run_log2_sweep


SCENES=(Armadillo)
PART_LABEL="part1/4"
DEFAULT_OUT_ROOT="outputs/sdf_stanford7_allkinds_part1"

export KINDS="${KINDS:-rim_full}"
export LOG2_HASHMAP_SIZES="${LOG2_HASHMAP_SIZES:-14 14.5 15}"
export BATCH_SIZE="${BATCH_SIZE:-524288}"
export RIM_ITERS="${RIM_ITERS:-20}"
export RIM_GATE_MODE="${RIM_GATE_MODE:-trainable}"
export RIM_FALLBACK_MODE="${RIM_FALLBACK_MODE:-blockwise}"

source scripts/run_sdf_stanford7_log2_sweep_common.sh
run_log2_sweep
