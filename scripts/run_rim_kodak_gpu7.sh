#!/usr/bin/env bash
set -euo pipefail

GPU_ID=7
WORKER_INDEX=7
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_rim_kodak_gpu_common.sh"
