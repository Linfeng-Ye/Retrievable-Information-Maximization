#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_PREFIX="${RUN_PREFIX:-rim_kodak}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs/${RUN_PREFIX}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "${PYTHON_BIN}" "${REPO_DIR}/summarize_rim_kodak_results.py" \
  --log-dir "${LOG_DIR}" \
  --run-prefix "${RUN_PREFIX}" \
  "$@"
