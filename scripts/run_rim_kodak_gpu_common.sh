#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${GPU_ID:-}" || -z "${WORKER_INDEX:-}" ]]; then
  echo "ERROR: GPU_ID and WORKER_INDEX must be set before sourcing this file." >&2
  exit 1
fi

STAGE="${1:-all}"
case "${STAGE}" in
  all)
    MASTER_STAGE="worker-all"
    ;;
  precompute)
    MASTER_STAGE="worker-precompute"
    ;;
  fit)
    MASTER_STAGE="worker-fit"
    ;;
  worker-all|worker-precompute|worker-fit|gpu-all|gpu-precompute|gpu-fit)
    MASTER_STAGE="${STAGE}"
    ;;
  summarize|summary)
    MASTER_STAGE="summarize"
    ;;
  *)
    echo "ERROR: unknown stage '${STAGE}'. Use all, precompute, fit, or summarize." >&2
    exit 1
    ;;
esac

export GPU_ID
export WORKER_INDEX
export WORKER_COUNT="${WORKER_COUNT:-8}"
export GPUS="${GPU_ID}"

exec "${SCRIPT_DIR}/run_rim_kodak_grid.sh" "${MASTER_STAGE}"
