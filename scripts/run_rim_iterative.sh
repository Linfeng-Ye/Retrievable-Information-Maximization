#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper: precompute B-RIM block info, then train.
# The two steps can also be run separately:
#   ./precompute_rim_block_info.sh ./pluto.jpg
#   ./train_rim_giga_image.sh ./pluto.jpg

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_IMAGE="${1:-./pluto.jpg}"

"${SCRIPT_DIR}/precompute_rim_block_info.sh" "${INPUT_IMAGE}"
"${SCRIPT_DIR}/train_rim_giga_image.sh" "${INPUT_IMAGE}"
