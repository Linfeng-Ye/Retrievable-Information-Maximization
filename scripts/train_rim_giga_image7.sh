#!/usr/bin/env bash
set -euo pipefail

# Train train_giga_1.py with an existing B-RIM block-info .pt file.
# Run precompute_rim_block_info.sh first, or set INFO_PT to an existing file.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_ARG="${1:-./tokyo.jpg}"
if [[ "${INPUT_ARG}" = /* ]]; then
  INPUT_IMAGE="${INPUT_ARG}"
else
  INPUT_IMAGE="$(pwd)/${INPUT_ARG}"
fi
cd "${SCRIPT_DIR}"

CONDA_ENV="${CONDA_ENV:-py310}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export CUDA_VISIBLE_DEVICES

RUN_NAME="${RUN_NAME:-rim_iterative}"
OUT_DIR="${OUT_DIR:-runs/${RUN_NAME}}"
WIDTH="${WIDTH:-56718}"
HEIGHT="${HEIGHT:-21450}"
K="${K:-1}"

LEVELS="${LEVELS:-16}"
FEATURES="${FEATURES:-2}"
D_HIDDEN="${D_HIDDEN:-128}"
BASE_RESOLUTION="${BASE_RESOLUTION:-64}"
LOG_HASH_SIZE="${LOG_HASH_SIZE:-18}"
BLOCK_SIZE="${BLOCK_SIZE:-400}"

GATE_MODE="${GATE_MODE:-trainable_sigmoid}"       # trainable_sigmoid or fixed_binary
FALLBACK_MODE="${FALLBACK_MODE:-blockwise}"       # blockwise or global_shared
GATE_TEMPERATURE="${GATE_TEMPERATURE:-1.0}"
GATE_INIT_LOGIT="${GATE_INIT_LOGIT:-2.5}"          # 2.0 initializes off/on near sigmoid(-/+2)=0.119/0.881; set none to preserve raw gates
GATE_REG_WEIGHT="${GATE_REG_WEIGHT:-0.0}"
FIXED_GATE_THRESHOLD="${FIXED_GATE_THRESHOLD:-0.5}"

STEPS="${STEPS:-31000}"
BATCH_SIZE="${BATCH_SIZE:-4005536}"
EVAL_EVERY="${EVAL_EVERY:-50}"
LR="${LR:-1e-3}"
VAL_SAMPLES="${VAL_SAMPLES:-10000}"
RIM_MAX_INIT_ITERS="${RIM_MAX_INIT_ITERS:-20}"
RIM_INIT_TOL="${RIM_INIT_TOL:-1e-6}"
EVAL_ONLY="${EVAL_ONLY:-1}"
METRICS="${METRICS:-0}"

mkdir -p "${OUT_DIR}"
INFO_PT="${INFO_PT:-${OUT_DIR}/${RUN_NAME}_block_info.pt}"
OUT_PREFIX="${OUT_PREFIX:-${OUT_DIR}/${RUN_NAME}}"

if [[ ! -f "${INFO_PT}" ]]; then
  echo "ERROR: block info file not found: ${INFO_PT}" >&2
  echo "Run ./precompute_rim_block_info.sh first, or set INFO_PT=/path/to/block_info.pt." >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ "${EVAL_ONLY}" == "1" || "${EVAL_ONLY}" == "true" ]]; then
  EXTRA_ARGS+=(--eval_only --eval_full_psnr)
fi
if [[ "${METRICS}" == "1" || "${METRICS}" == "true" ]]; then
  EXTRA_ARGS+=(--metrics)
fi
GATE_INIT_ARGS=()
if [[ -n "${GATE_INIT_LOGIT}" && "${GATE_INIT_LOGIT}" != "none" ]]; then
  GATE_INIT_ARGS+=(--rim_gate_init_logit "${GATE_INIT_LOGIT}")
fi

conda run --no-capture-output -n "${CONDA_ENV}" python train_giga_1.py \
  --input_image "${INPUT_IMAGE}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --K "${K}" \
  --steps "${STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --eval_every "${EVAL_EVERY}" \
  --levels "${LEVELS}" \
  --features "${FEATURES}" \
  --d_hidden "${D_HIDDEN}" \
  --base_resolution "${BASE_RESOLUTION}" \
  --log_hash_size "${LOG_HASH_SIZE}" \
  --lr "${LR}" \
  --device cuda \
  --out_prefix "${OUT_PREFIX}" \
  --val_samples "${VAL_SAMPLES}" \
  --rim_enabled \
  --rim_info_tensor_path "${INFO_PT}" \
  --rim_max_init_iters "${RIM_MAX_INIT_ITERS}" \
  --rim_init_tol "${RIM_INIT_TOL}" \
  --rim_gate_mode "${GATE_MODE}" \
  --rim_gate_temperature "${GATE_TEMPERATURE}" \
  "${GATE_INIT_ARGS[@]}" \
  --rim_gate_regularization_weight "${GATE_REG_WEIGHT}" \
  --rim_fixed_gate_threshold "${FIXED_GATE_THRESHOLD}" \
  --rim_fallback_mode "${FALLBACK_MODE}" \
  --rim_save_init_state \
  --rim_block_size "${BLOCK_SIZE}" \
  --reconst_dir "${OUT_DIR}" \
  --reconst_name "${RUN_NAME}.npy" \
  "${EXTRA_ARGS[@]}"
