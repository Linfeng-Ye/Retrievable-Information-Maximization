#!/usr/bin/env bash
set -euo pipefail

# Convenience preset for the 8000 x 8000 Pluto image used in the paper.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-pluto_k1_200}"
export WIDTH="${WIDTH:-8000}"
export HEIGHT="${HEIGHT:-8000}"
export K="${K:-1}"
export LEVELS="${LEVELS:-200}"
export NUM_CANDIDATES="${NUM_CANDIDATES:-200}"
export BASE_RESOLUTION="${BASE_RESOLUTION:-64}"
export BLOCK_SIZE="${BLOCK_SIZE:-200}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "${SCRIPT_DIR}/precompute_rim_block_info.sh" "${1:-./data/gigapixel/pluto.jpg}"
