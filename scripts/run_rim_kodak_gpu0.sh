#!/usr/bin/env bash
set -euo pipefail

GPU_ID=0
WORKER_INDEX=0
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_rim_kodak_gpu_common.sh"
