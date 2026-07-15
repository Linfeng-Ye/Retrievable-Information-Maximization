#!/usr/bin/env bash

# Shared runner for Stanford SDF split scripts.
# The caller must set:
#   SCENES=(...)
#   PART_LABEL="partX/4"
#   DEFAULT_OUT_ROOT="outputs/..."
#
# Log guarantee:
#   one wrapper log per scene, per LOG2_HASHMAP_SIZE setting, per experiment kind.
#   Example:
#     ${OUT_ROOT}/logs/log2_14p5/Armadillo_log2_14p5_rim_full.txt
#
# Trainer-internal logs are also written inside each run directory:
#   ${OUT_ROOT}/log2_14p5/Armadillo/rim_full/log_Armadillo-log2_14p5-rim_full.txt
#
# KINDS (optional): space-separated list of named kinds to run per scene/log2 setting.
# Takes precedence over the legacy MODE/ENCODER selection below when set.
#   baseline       -> fixed_hashgrid (INGP-style geometric resolutions, gates fixed at 1)
#   rim_resolution -> RIM resolution schedule, gates fixed at 1 (no gating; isolates the
#                     resolution-selection DP on its own). Runs a single DP pass
#                     (RIM_ITERS forced to 1) since fixed gates make the alternating
#                     gate/resolution loop a no-op.
#   rim_gate       -> geometric resolutions + selected fixed RIM gates
#   rim_full       -> RIM resolution schedule + selected fixed RIM gates ("our method")
# Example: KINDS="baseline rim_resolution rim_gate rim_full"
# If KINDS is unset, falls back to the legacy MODE-driven behavior (MODE=baseline|rim|both,
# ENCODER/RIM_GATE_MODE picked for the single "rim" kind) for backward compatibility with
# existing callers (e.g. run_sdf_stanford7_rim900k_part*.sh).

export MODE="${MODE:-rim}"
export ENCODER="${ENCODER:-rim_resolution}"
export RIM_GATE_MODE="${RIM_GATE_MODE:-fixed}"
export KINDS="${KINDS:-}"
export LOG2_HASHMAP_SIZES="${LOG2_HASHMAP_SIZES:-14 14.5 15}"
export NUM_LEVELS="${NUM_LEVELS:-15}"
export FEATURE_DIM="${FEATURE_DIM:-2}"
export BASE_RESOLUTION="${BASE_RESOLUTION:-16}"
export DESIRED_RESOLUTION="${DESIRED_RESOLUTION:-2048}"
export RIM_ITERS="${RIM_ITERS:-20}"
export RIM_INFORMATION_METRIC="${RIM_INFORMATION_METRIC:-entropy}"
export RIM_FALLBACK_MODE="${RIM_FALLBACK_MODE:-blockwise}"
export DATA_DIR="${DATA_DIR:-data/sdf}"
export OUT_ROOT="${OUT_ROOT:-${DEFAULT_OUT_ROOT}}"
export LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"
RUN_FAILURES=0

case "${MODE}" in
  baseline|rim|both) ;;
  *)
    echo "[ERROR] Unknown MODE=${MODE}; expected baseline, rim, or both." >&2
    exit 2
    ;;
esac

if [[ -z "${PYTHON:-}" ]]; then
  for cand in python python3; do
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

log2_label() {
  local value="$1"
  printf '%s' "${value}" | tr '.' 'p'
}

# Resolve a named kind to (run_script, encoder, gate_mode) and run it.
run_named_kind_log2() {
  local scene="$1"
  local log2="$2"
  local kind="$3"
  local mesh="$4"
  local label="$5"
  local run_script kind_encoder kind_gate_mode kind_rim_iters

  case "${kind}" in
    baseline)
      run_script="scripts/run_sdf_baseline.sh"
      kind_encoder="fixed_hashgrid"
      kind_gate_mode="fixed"
      kind_rim_iters=""
      ;;
    rim_resolution)
      run_script="scripts/run_sdf_rim.sh"
      kind_encoder="rim_resolution"
      kind_gate_mode="fixed"
      # Gates are fixed at 1 in this mode, so the resolution DP never sees a
      # changing gate between passes -- one solve already yields the final
      # partition. Force a single pass instead of the default alternating loop.
      kind_rim_iters="1"
      ;;
    rim_gate)
      run_script="scripts/run_sdf_rim.sh"
      kind_encoder="rim_gate"
      kind_gate_mode="fixed"
      kind_rim_iters=""
      ;;
    rim_full)
      run_script="scripts/run_sdf_rim.sh"
      kind_encoder="rim_full"
      kind_gate_mode="fixed"
      kind_rim_iters=""
      ;;
    rim)
      # Legacy alias: whatever ENCODER/RIM_GATE_MODE are currently exported to.
      run_script="scripts/run_sdf_rim.sh"
      kind_encoder="${ENCODER}"
      kind_gate_mode="${RIM_GATE_MODE}"
      kind_rim_iters=""
      ;;
    *)
      echo "[ERROR] Unknown kind=${kind}; expected baseline, rim_resolution, rim_gate, rim_full, or (legacy) rim." >&2
      exit 2
      ;;
  esac

  local run_alias="${scene}-log2_${label}"
  local run_out="${OUT_ROOT}/log2_${label}/${scene}/${kind}"
  local log_dir="${LOG_ROOT}/log2_${label}"
  local log_file="${log_dir}/${scene}_log2_${label}_${kind}.txt"

  mkdir -p "${log_dir}" "${run_out}"
  echo "[LOG] ${scene} log2=${log2} ${kind}: ${log_file}"
  set +e
  (
    set -euo pipefail
    export PYTHON
    export LOG2_HASHMAP_SIZE="${log2}"
    export OUT_DIR="${run_out}"
    export NUM_LEVELS FEATURE_DIM BASE_RESOLUTION DESIRED_RESOLUTION
    export ENCODER="${kind_encoder}"
    export RIM_GATE_MODE="${kind_gate_mode}"
    if [[ -n "${kind_rim_iters}" ]]; then
      export RIM_ITERS="${kind_rim_iters}"
    fi
    echo "[SCENE] ${scene}"
    echo "[KIND] ${kind}"
    echo "[MESH] ${mesh}"
    echo "[LOG2_HASHMAP_SIZE_INPUT] ${LOG2_HASHMAP_SIZE}"
    echo "[OUT_DIR] ${OUT_DIR}"
    echo "[ENCODER] ${ENCODER}"
    echo "[RIM_GATE_MODE] ${RIM_GATE_MODE}"
    echo "[RIM_ITERS] ${RIM_ITERS:-<default>}"
    echo "[RIM_INFORMATION_METRIC] ${RIM_INFORMATION_METRIC}"
    echo "[RIM_FALLBACK_MODE] ${RIM_FALLBACK_MODE}"
    echo "[RUN_ALIAS] ${run_alias}"
    date
    bash "${run_script}" "${mesh}" "${run_alias}"
    date
    echo "[DONE] ${scene} log2=${log2} kind=${kind}"
  ) 2>&1 | tee "${log_file}"
  local status="${PIPESTATUS[0]}"
  set -e

  if [[ "${status}" != "0" ]]; then
    echo "[WARN] ${scene} log2=${log2} ${kind}: failed with status ${status}; continuing." >&2
    RUN_FAILURES=$((RUN_FAILURES + 1))
  fi
}

run_scene_log2() {
  local scene="$1"
  local log2="$2"
  local label
  label="$(log2_label "${log2}")"
  local mesh="${DATA_DIR}/${scene}_nrml.obj"
  local eval_points="${DATA_DIR}/${scene}_nrml_eval_points.pt"

  if [[ ! -f "${mesh}" ]]; then
    echo "[WARN] ${scene}: missing normalized mesh ${mesh}; skipping." >&2
    RUN_FAILURES=$((RUN_FAILURES + 1))
    return 0
  fi
  if [[ ! -f "${eval_points}" ]]; then
    echo "[WARN] ${scene}: missing eval points ${eval_points}; training still runs." >&2
  fi

  if [[ -n "${KINDS}" ]]; then
    for kind in ${KINDS}; do
      run_named_kind_log2 "${scene}" "${log2}" "${kind}" "${mesh}" "${label}"
    done
    return 0
  fi

  # Legacy MODE-driven behavior (unchanged for existing callers).
  if [[ "${MODE}" == "baseline" || "${MODE}" == "both" ]]; then
    run_named_kind_log2 "${scene}" "${log2}" "baseline" "${mesh}" "${label}"
  fi
  if [[ "${MODE}" == "rim" || "${MODE}" == "both" ]]; then
    run_named_kind_log2 "${scene}" "${log2}" "rim" "${mesh}" "${label}"
  fi
}

run_log2_sweep() {
  echo "[RUN ${PART_LABEL}] scenes=${SCENES[*]}"
  echo "[RUN ${PART_LABEL}] MODE=${MODE}"
  echo "[RUN ${PART_LABEL}] KINDS=${KINDS:-<legacy MODE-driven: ${MODE}>}"
  echo "[RUN ${PART_LABEL}] LOG2_HASHMAP_SIZES=${LOG2_HASHMAP_SIZES}"
  echo "[RUN ${PART_LABEL}] RIM_ITERS=${RIM_ITERS}"
  echo "[RUN ${PART_LABEL}] RIM_INFORMATION_METRIC=${RIM_INFORMATION_METRIC}"
  echo "[RUN ${PART_LABEL}] RIM_FALLBACK_MODE=${RIM_FALLBACK_MODE}"
  echo "[RUN ${PART_LABEL}] OUT_ROOT=${OUT_ROOT}"
  echo "[RUN ${PART_LABEL}] LOG_ROOT=${LOG_ROOT}"

  for scene in "${SCENES[@]}"; do
    for log2 in ${LOG2_HASHMAP_SIZES}; do
      run_scene_log2 "${scene}" "${log2}"
    done
  done

  if (( RUN_FAILURES > 0 )); then
    echo "[ERROR] ${RUN_FAILURES} run(s) failed or were skipped because a mesh was missing." >&2
    return 1
  fi
}
