#!/usr/bin/env bash
set -euo pipefail

# Precompute reusable B-RIM block/band information only.
# Example:
#   LEVELS=64 NUM_CANDIDATES=128 WIDTH=8000 HEIGHT=8000 ./precompute_rim_block_info.sh ./pluto.jpg

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INPUT_ARG="${1:-./tokyo.jpg}"
if [[ "${INPUT_ARG}" = /* ]]; then
  INPUT_IMAGE="${INPUT_ARG}"
else
  INPUT_IMAGE="$(pwd)/${INPUT_ARG}"
fi
cd "${REPO_DIR}"

CONDA_ENV="${CONDA_ENV:-py310}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export CUDA_VISIBLE_DEVICES

RUN_NAME="${RUN_NAME:-rim_iterative}"
OUT_DIR="${OUT_DIR:-runs/${RUN_NAME}}"
WIDTH="${WIDTH:-56718}"
HEIGHT="${HEIGHT:-21450}"
PRECOMPUTE_PATCH_SIZE="${PRECOMPUTE_PATCH_SIZE:-}"
K="${K:-1}"

LEVELS="${LEVELS:-200}"
BASE_RESOLUTION="${BASE_RESOLUTION:-64}"
MAX_RESOLUTION="${MAX_RESOLUTION:-auto}"
BLOCK_SIZE="${BLOCK_SIZE:-400}"

NUM_CANDIDATES="${NUM_CANDIDATES:-${LEVELS}}"
CANDIDATES="${CANDIDATES:-}"
INFO_BATCH_SIZE="${INFO_BATCH_SIZE:-1}"  # compatibility only; precompute processes images one by one
PRECOMPUTE_DEVICE="${PRECOMPUTE_DEVICE:-cuda}"
PRECOMPUTE_DTYPE="${PRECOMPUTE_DTYPE:-float16}"
MAX_DOWNSAMPLE_FACTOR="${MAX_DOWNSAMPLE_FACTOR:-8}"
LOWPASS_CHANNELWISE="${LOWPASS_CHANNELWISE:-1}"
ENTROPY_BLOCK_CHUNK_SIZE="${ENTROPY_BLOCK_CHUNK_SIZE:-32}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
ALLOW_LARGE_IMAGE="${ALLOW_LARGE_IMAGE:-0}"

mkdir -p "${OUT_DIR}"
INFO_PT="${INFO_PT:-${OUT_DIR}/${RUN_NAME}_block_info.pt}"

PATCH_ARGS=()
if [[ -n "${PRECOMPUTE_PATCH_SIZE}" && "${PRECOMPUTE_PATCH_SIZE}" != "none" ]]; then
  PATCH_ARGS+=(--patch-size "${PRECOMPUTE_PATCH_SIZE}")
elif [[ -n "${WIDTH}" && -n "${HEIGHT}" ]]; then
  PATCH_ARGS+=(--patch-size "${HEIGHT}" "${WIDTH}")
fi

PROGRESS_ARGS=()
if [[ "${SHOW_PROGRESS}" == "0" || "${SHOW_PROGRESS}" == "false" ]]; then
  PROGRESS_ARGS+=(--no-progress)
fi

LARGE_IMAGE_ARGS=()
if [[ "${ALLOW_LARGE_IMAGE}" == "1" || "${ALLOW_LARGE_IMAGE}" == "true" ]]; then
  LARGE_IMAGE_ARGS+=(--allow-large-image)
fi

LOWPASS_ARGS=()
if [[ "${LOWPASS_CHANNELWISE}" == "1" || "${LOWPASS_CHANNELWISE}" == "true" ]]; then
  LOWPASS_ARGS+=(--lowpass-channelwise)
fi

CANDIDATE_ARGS=()
if [[ -n "${CANDIDATES}" ]]; then
  read -r -a CANDIDATE_VALUES <<< "${CANDIDATES}"
  if (( ${#CANDIDATE_VALUES[@]} < LEVELS )); then
    echo "ERROR: explicit CANDIDATES has ${#CANDIDATE_VALUES[@]} values but LEVELS=${LEVELS}. Need candidates >= levels." >&2
    exit 1
  fi
  CANDIDATE_ARGS+=(--candidate-resolutions "${CANDIDATE_VALUES[@]}")
else
  if (( NUM_CANDIDATES < LEVELS )); then
    echo "ERROR: NUM_CANDIDATES=${NUM_CANDIDATES} but LEVELS=${LEVELS}. Need NUM_CANDIDATES >= LEVELS." >&2
    exit 1
  fi
  CANDIDATE_ARGS+=(--num-candidate-levels "${NUM_CANDIDATES}")
  CANDIDATE_ARGS+=(--base-resolution "${BASE_RESOLUTION}")
  if [[ "${MAX_RESOLUTION}" != "auto" && -n "${MAX_RESOLUTION}" ]]; then
    CANDIDATE_ARGS+=(--max-resolution "${MAX_RESOLUTION}")
  fi
fi

if [[ "${RECOMPUTE_INFO:-1}" != "1" && -f "${INFO_PT}" ]]; then
  echo "Using existing block info: ${INFO_PT}"
  exit 0
fi

conda run --no-capture-output -n "${CONDA_ENV}" python tools/precompute_block_info.py \
  --input "${INPUT_IMAGE}" \
  --output "${INFO_PT}" \
  "${PATCH_ARGS[@]}" \
  --space-to-depth-k "${K}" \
  --block-size "${BLOCK_SIZE}" \
  "${CANDIDATE_ARGS[@]}" \
  --batch-size "${INFO_BATCH_SIZE}" \
  --device "${PRECOMPUTE_DEVICE}" \
  --dtype "${PRECOMPUTE_DTYPE}" \
  --max-downsample-factor "${MAX_DOWNSAMPLE_FACTOR}" \
  --entropy-block-chunk-size "${ENTROPY_BLOCK_CHUNK_SIZE}" \
  "${LOWPASS_ARGS[@]}" \
  "${PROGRESS_ARGS[@]}" \
  "${LARGE_IMAGE_ARGS[@]}"

echo "Block info saved to: ${INFO_PT}"
