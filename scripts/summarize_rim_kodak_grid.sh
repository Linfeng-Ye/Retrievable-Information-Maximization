#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PREFIX="${RUN_PREFIX:-rim_kodak}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/${RUN_PREFIX}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_rim_kodak_results.py" \
  --log-dir "${LOG_DIR}" \
  --run-prefix "${RUN_PREFIX}" \
  "$@"
