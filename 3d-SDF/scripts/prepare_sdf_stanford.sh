#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

if [[ -z "${PYTHON:-}" ]]; then
  for cand in python python3; do
    if command -v "${cand}" >/dev/null 2>&1 || [[ -x "${cand}" ]]; then
      PYTHON="${cand}"
      break
    fi
  done
fi

mkdir -p data/sdf/raw data/sdf/processed
"${PYTHON}" scripts/prepare_sdf_stanford.py "$@"
