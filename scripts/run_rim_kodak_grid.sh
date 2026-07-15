#!/usr/bin/env bash
set -euo pipefail

# Kodak B-RIM grid:
#   images: Kodak/*.png
#   GPUs: all 8 by default (0 1 2 3 4 5 6 7)
#   pixel shuffle K: 1 (disabled, as in the paper)
#   block size: 16 32 64 128
#   log_hash_size: 10..16
#
# Usage:
#   bash scripts/run_rim_kodak_grid.sh              # precompute, fit, summarize
#   bash scripts/run_rim_kodak_grid.sh precompute
#   bash scripts/run_rim_kodak_grid.sh fit
#   bash scripts/run_rim_kodak_grid.sh summarize
#
# Per-GPU scripts:
#   ./run_rim_kodak_gpu0.sh              # worker shard 0/8 on GPU 0
#   ./run_rim_kodak_gpu1.sh              # worker shard 1/8 on GPU 1
#   ...
#   ./run_rim_kodak_gpu7.sh              # worker shard 7/8 on GPU 7
#
# Common overrides:
#   GPUS="0 1 2 3 4 5 6 7" STEPS=30000 ./run_rim_kodak_grid.sh
#   KODAK_IMAGES="Kodak/1.png Kodak/2.png" ./run_rim_kodak_grid.sh fit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

STAGE="${1:-all}"

CONDA_ENV="${CONDA_ENV:-py310}"
RUN_PREFIX="${RUN_PREFIX:-rim_kodak}"
KODAK_DIR="${KODAK_DIR:-${REPO_DIR}/data/Kodak}"
LOG_DIR="${LOG_DIR:-logs/${RUN_PREFIX}}"
mkdir -p "${LOG_DIR}"

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
K_VALUES=(${K_VALUES:-1})
BLOCK_SIZES=(${BLOCK_SIZES:-16 32 64 128})
LOG_HASH_SIZES=(${LOG_HASH_SIZES:-10 11 12 13 14 15 16})

# Kodak fitting defaults from the command in the request.
WIDTH="${WIDTH:-768}"
HEIGHT="${HEIGHT:-512}"
TRAIN_LEVELS="${TRAIN_LEVELS:-13}"
FEATURES="${FEATURES:-2}"
D_HIDDEN="${D_HIDDEN:-128}"
BASE_RESOLUTION="${BASE_RESOLUTION:-16}"
GROWTH_FACTOR="${GROWTH_FACTOR:-1.1}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-393216}"
EVAL_EVERY="${EVAL_EVERY:-100000}"
LR="${LR:-1e-3}"
VAL_SAMPLES="${VAL_SAMPLES:-100000}"

# The paper uses 30 candidate frequency bands on Kodak, then selects 13 levels.
PRECOMPUTE_LEVELS="${PRECOMPUTE_LEVELS:-30}"
PRECOMPUTE_NUM_CANDIDATES="${PRECOMPUTE_NUM_CANDIDATES:-${PRECOMPUTE_LEVELS}}"
PRECOMPUTE_DTYPE="${PRECOMPUTE_DTYPE:-float16}"
MAX_RESOLUTION="${MAX_RESOLUTION:-auto}"
MAX_DOWNSAMPLE_FACTOR="${MAX_DOWNSAMPLE_FACTOR:-8}"
LOWPASS_CHANNELWISE="${LOWPASS_CHANNELWISE:-1}"
ENTROPY_BLOCK_CHUNK_SIZE="${ENTROPY_BLOCK_CHUNK_SIZE:-32}"

RIM_MAX_INIT_ITERS="${RIM_MAX_INIT_ITERS:-20}"
RIM_INIT_TOL="${RIM_INIT_TOL:-1e-6}"
RIM_INIT_SCHEDULER="${RIM_INIT_SCHEDULER:-geometric}"
RIM_GATE_SOLVER="${RIM_GATE_SOLVER:-exact_fractional}"
GATE_MODE="${GATE_MODE:-fixed_binary}"
FALLBACK_MODE="${FALLBACK_MODE:-blockwise}"
GATE_TEMPERATURE="${GATE_TEMPERATURE:-1.0}"
GATE_INIT_LOGIT="${GATE_INIT_LOGIT:-2.5}"
GATE_REG_WEIGHT="${GATE_REG_WEIGHT:-0.0}"
FIXED_GATE_THRESHOLD="${FIXED_GATE_THRESHOLD:-0.5}"

EVAL_ONLY="${EVAL_ONLY:-1}"
METRICS="${METRICS:-1}"
METRICS_TILES="${METRICS_TILES:-32}"
METRICS_TILE_SIZE="${METRICS_TILE_SIZE:-256}"
METRICS_SEED="${METRICS_SEED:-0}"
HALF_RECON="${HALF_RECON:-0}"
RECON_CHUNK_PIXELS="${RECON_CHUNK_PIXELS:-1000000}"

# Resume-friendly defaults.
RECOMPUTE_INFO="${RECOMPUTE_INFO:-0}"
SKIP_DONE="${SKIP_DONE:-1}"

if (( ${#GPUS[@]} == 0 )); then
  echo "ERROR: GPUS is empty." >&2
  exit 1
fi

is_true() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

load_images() {
  IMAGES=()
  if [[ -n "${KODAK_IMAGES:-}" ]]; then
    local item
    for item in ${KODAK_IMAGES}; do
      if [[ "${item}" = /* ]]; then
        IMAGES+=("${item}")
      else
        IMAGES+=("${REPO_DIR}/${item#./}")
      fi
    done
  else
    if [[ ! -d "${KODAK_DIR}" ]]; then
      echo "ERROR: Kodak directory not found: ${KODAK_DIR}" >&2
      echo "Download the dataset as described in README.md or set KODAK_DIR." >&2
      exit 1
    fi
    mapfile -t IMAGES < <(find "${KODAK_DIR}" -maxdepth 1 -type f -name "*.png" | sort -V)
  fi

  if (( ${#IMAGES[@]} == 0 )); then
    echo "ERROR: no Kodak png files found in ${KODAK_DIR}" >&2
    exit 1
  fi
}

image_tag() {
  local image_path="$1"
  local stem="${image_path##*/}"
  stem="${stem%.*}"
  if [[ "${stem}" =~ ^[0-9]+$ ]]; then
    printf "i%02d" "${stem}"
  else
    echo "${stem}" | tr -c "A-Za-z0-9_" "_"
  fi
}

info_run_name() {
  local image_path="$1"
  local k="$2"
  local block_size="$3"
  printf "%s_%s_k%s_b%s" "${RUN_PREFIX}" "$(image_tag "${image_path}")" "${k}" "${block_size}"
}

train_run_name() {
  local image_path="$1"
  local k="$2"
  local block_size="$3"
  local log_hash_size="$4"
  printf "%s_log%s" "$(info_run_name "${image_path}" "${k}" "${block_size}")" "${log_hash_size}"
}

is_finished_log() {
  local log_file="$1"
  [[ -f "${log_file}" ]] || return 1
  tr '\r' '\n' <"${log_file}" | grep -q "^Done\\." || return 1
  tr '\r' '\n' <"${log_file}" | grep -q "^Estimated active hash parameters:" || return 1
  tr '\r' '\n' <"${log_file}" | grep -q "^MLP trainable parameters:" || return 1
  if is_true "${METRICS}"; then
    tr '\r' '\n' <"${log_file}" | grep -q "'ssim'" || return 1
  else
    tr '\r' '\n' <"${log_file}" | grep -q "psnr_full" || return 1
  fi
}

run_precompute_job() {
  local image_path="$1"
  local k="$2"
  local block_size="$3"
  local gpu="$4"
  local info_run
  local info_dir
  local info_pt
  local log_file

  info_run="$(info_run_name "${image_path}" "${k}" "${block_size}")"
  info_dir="runs/${info_run}"
  info_pt="${info_dir}/${info_run}_block_info.pt"
  log_file="${LOG_DIR}/${info_run}_precompute.txt"

  if ! is_true "${RECOMPUTE_INFO}" && [[ -f "${info_pt}" ]]; then
    echo "[precompute][gpu ${gpu}] skip existing ${info_pt}"
    return 0
  fi

  echo "[precompute][gpu ${gpu}] ${info_run}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  CONDA_ENV="${CONDA_ENV}" \
  RUN_NAME="${info_run}" \
  OUT_DIR="${info_dir}" \
  INFO_PT="${info_pt}" \
  WIDTH="${WIDTH}" \
  HEIGHT="${HEIGHT}" \
  K="${k}" \
  LEVELS="${PRECOMPUTE_LEVELS}" \
  NUM_CANDIDATES="${PRECOMPUTE_NUM_CANDIDATES}" \
  BASE_RESOLUTION="${BASE_RESOLUTION}" \
  MAX_RESOLUTION="${MAX_RESOLUTION}" \
  BLOCK_SIZE="${block_size}" \
  PRECOMPUTE_DTYPE="${PRECOMPUTE_DTYPE}" \
  MAX_DOWNSAMPLE_FACTOR="${MAX_DOWNSAMPLE_FACTOR}" \
  LOWPASS_CHANNELWISE="${LOWPASS_CHANNELWISE}" \
  ENTROPY_BLOCK_CHUNK_SIZE="${ENTROPY_BLOCK_CHUNK_SIZE}" \
  ALLOW_LARGE_IMAGE=1 \
  RECOMPUTE_INFO=1 \
  "${SCRIPT_DIR}/precompute_rim_block_info.sh" "${image_path}" >"${log_file}" 2>&1
}

run_fit_job() {
  local image_path="$1"
  local k="$2"
  local block_size="$3"
  local log_hash_size="$4"
  local gpu="$5"
  local info_run
  local info_pt
  local train_run
  local train_dir
  local log_file
  local -a extra_args
  local -a gate_init_args

  info_run="$(info_run_name "${image_path}" "${k}" "${block_size}")"
  info_pt="runs/${info_run}/${info_run}_block_info.pt"
  train_run="$(train_run_name "${image_path}" "${k}" "${block_size}" "${log_hash_size}")"
  train_dir="runs/${train_run}"
  log_file="${LOG_DIR}/${train_run}.txt"

  if [[ ! -f "${info_pt}" ]]; then
    echo "ERROR: missing block info file: ${info_pt}" >&2
    echo "Run bash scripts/run_rim_kodak_grid.sh precompute first." >&2
    return 1
  fi

  if is_true "${SKIP_DONE}" && is_finished_log "${log_file}"; then
    echo "[fit][gpu ${gpu}] skip finished ${train_run}"
    return 0
  fi

  mkdir -p "${train_dir}"
  {
    echo "run_name: ${train_run}"
    echo "input_image: ${image_path}"
    echo "image_size: ${WIDTH}x${HEIGHT}"
    echo "K: ${k}"
    echo "block_size: ${block_size}"
    echo "log_hash_size: ${log_hash_size}"
    echo "gpu: ${gpu}"
    echo "block_info: ${info_pt}"
    echo "run_dir: ${train_dir}"
    echo
  } >"${log_file}"

  extra_args=()
  if is_true "${EVAL_ONLY}"; then
    extra_args+=(--eval_only --eval_full_psnr)
  fi
  if is_true "${METRICS}"; then
    extra_args+=(
      --metrics
      --metrics_tiles "${METRICS_TILES}"
      --metrics_tile_size "${METRICS_TILE_SIZE}"
      --metrics_device cuda
      --metrics_seed "${METRICS_SEED}"
    )
  fi
  if is_true "${HALF_RECON}"; then
    extra_args+=(--half_recon)
  fi

  gate_init_args=()
  if [[ -n "${GATE_INIT_LOGIT}" && "${GATE_INIT_LOGIT}" != "none" ]]; then
    gate_init_args+=(--rim_gate_init_logit "${GATE_INIT_LOGIT}")
  fi

  echo "[fit][gpu ${gpu}] ${train_run}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  conda run --no-capture-output -n "${CONDA_ENV}" python train_giga_1.py \
    --input_image "${image_path}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --K "${k}" \
    --steps "${STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_every "${EVAL_EVERY}" \
    --levels "${TRAIN_LEVELS}" \
    --features "${FEATURES}" \
    --d_hidden "${D_HIDDEN}" \
    --base_resolution "${BASE_RESOLUTION}" \
    --growth_factor "${GROWTH_FACTOR}" \
    --log_hash_size "${log_hash_size}" \
    --lr "${LR}" \
    --device cuda \
    --out_prefix "${train_dir}/${train_run}" \
    --val_samples "${VAL_SAMPLES}" \
    --rim_enabled \
    --rim_info_tensor_path "${info_pt}" \
    --rim_max_init_iters "${RIM_MAX_INIT_ITERS}" \
    --rim_init_tol "${RIM_INIT_TOL}" \
    --rim_init_scheduler "${RIM_INIT_SCHEDULER}" \
    --rim_gate_solver "${RIM_GATE_SOLVER}" \
    --rim_gate_mode "${GATE_MODE}" \
    --rim_gate_temperature "${GATE_TEMPERATURE}" \
    "${gate_init_args[@]}" \
    --rim_gate_regularization_weight "${GATE_REG_WEIGHT}" \
    --rim_fixed_gate_threshold "${FIXED_GATE_THRESHOLD}" \
    --rim_fallback_mode "${FALLBACK_MODE}" \
    --rim_save_init_state \
    --rim_block_size "${block_size}" \
    --recon_chunk_pixels "${RECON_CHUNK_PIXELS}" \
    --reconst_dir "${train_dir}" \
    --reconst_name "${train_run}.npy" \
    "${extra_args[@]}" >>"${log_file}" 2>&1
}

run_worker() {
  local stage="$1"
  local worker_idx="$2"
  local gpu="$3"
  local num_workers="$4"
  local job_idx=0
  local image_path
  local k
  local block_size
  local log_hash_size

  for image_path in "${IMAGES[@]}"; do
    for k in "${K_VALUES[@]}"; do
      for block_size in "${BLOCK_SIZES[@]}"; do
        if [[ "${stage}" == "precompute" ]]; then
          if (( job_idx % num_workers == worker_idx )); then
            run_precompute_job "${image_path}" "${k}" "${block_size}" "${gpu}"
          fi
          ((job_idx += 1))
        else
          for log_hash_size in "${LOG_HASH_SIZES[@]}"; do
            if (( job_idx % num_workers == worker_idx )); then
              run_fit_job "${image_path}" "${k}" "${block_size}" "${log_hash_size}" "${gpu}"
            fi
            ((job_idx += 1))
          done
        fi
      done
    done
  done
}

run_parallel_stage() {
  local stage="$1"
  local total_jobs="$2"
  local -a pids
  local status=0
  local num_workers="${#GPUS[@]}"
  local i

  echo
  echo "== ${stage}: ${total_jobs} jobs across GPUs: ${GPUS[*]} =="
  pids=()
  for i in "${!GPUS[@]}"; do
    run_worker "${stage}" "${i}" "${GPUS[$i]}" "${num_workers}" &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if (( status != 0 )); then
    echo "ERROR: ${stage} failed. Check logs in ${LOG_DIR}." >&2
    exit "${status}"
  fi
}

run_single_gpu_stage() {
  local stage="$1"
  local worker_idx="${WORKER_INDEX:-}"
  local worker_count="${WORKER_COUNT:-8}"
  local gpu="${GPU_ID:-${GPU:-${CUDA_VISIBLE_DEVICES:-${worker_idx}}}}"
  local total_jobs

  if [[ -z "${worker_idx}" ]]; then
    echo "ERROR: WORKER_INDEX is required for ${stage} worker mode." >&2
    exit 1
  fi
  if [[ ! "${worker_idx}" =~ ^[0-9]+$ || ! "${worker_count}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: WORKER_INDEX and WORKER_COUNT must be non-negative integers." >&2
    exit 1
  fi
  if (( worker_count <= 0 || worker_idx >= worker_count )); then
    echo "ERROR: invalid worker shard ${worker_idx}/${worker_count}." >&2
    exit 1
  fi

  if [[ "${stage}" == "precompute" ]]; then
    total_jobs="${TOTAL_PRECOMPUTE}"
  else
    total_jobs="${TOTAL_FIT}"
  fi

  echo
  echo "== ${stage}: worker ${worker_idx}/${worker_count} on GPU ${gpu}; total grid jobs=${total_jobs} =="
  run_worker "${stage}" "${worker_idx}" "${gpu}" "${worker_count}"
}

run_summary() {
  local python_bin="${PYTHON_BIN:-python3}"
  "${python_bin}" "${REPO_DIR}/summarize_rim_kodak_results.py" \
    --log-dir "${LOG_DIR}" \
    --run-prefix "${RUN_PREFIX}" \
    --expected-runs "${#IMAGES[@]}"
}

print_config() {
  echo "Kodak B-RIM grid"
  echo "  images: ${#IMAGES[@]} from ${KODAK_DIR}"
  echo "  GPUs: ${GPUS[*]}"
  echo "  K values: ${K_VALUES[*]}"
  echo "  block sizes: ${BLOCK_SIZES[*]}"
  echo "  log hash sizes: ${LOG_HASH_SIZES[*]}"
  echo "  size: ${WIDTH}x${HEIGHT}"
  echo "  train levels/base/features/hidden: ${TRAIN_LEVELS}/${BASE_RESOLUTION}/${FEATURES}/${D_HIDDEN}"
  echo "  steps/batch/eval_every: ${STEPS}/${BATCH_SIZE}/${EVAL_EVERY}"
  echo "  log dir: ${LOG_DIR}"
}

load_images
TOTAL_PRECOMPUTE=$(( ${#IMAGES[@]} * ${#K_VALUES[@]} * ${#BLOCK_SIZES[@]} ))
TOTAL_FIT=$(( TOTAL_PRECOMPUTE * ${#LOG_HASH_SIZES[@]} ))
print_config

case "${STAGE}" in
  all)
    run_parallel_stage precompute "${TOTAL_PRECOMPUTE}"
    run_parallel_stage fit "${TOTAL_FIT}"
    run_summary
    ;;
  precompute)
    run_parallel_stage precompute "${TOTAL_PRECOMPUTE}"
    ;;
  fit)
    run_parallel_stage fit "${TOTAL_FIT}"
    ;;
  summarize|summary)
    run_summary
    ;;
  worker-all|gpu-all)
    run_single_gpu_stage precompute
    run_single_gpu_stage fit
    ;;
  worker-precompute|gpu-precompute)
    run_single_gpu_stage precompute
    ;;
  worker-fit|gpu-fit)
    run_single_gpu_stage fit
    ;;
  *)
    echo "ERROR: unknown stage '${STAGE}'. Use all, precompute, fit, or summarize." >&2
    exit 1
    ;;
esac
