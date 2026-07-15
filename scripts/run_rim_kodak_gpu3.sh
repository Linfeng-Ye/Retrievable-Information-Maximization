#!/usr/bin/env bash
set -euo pipefail

GPU_ID=3
WORKER_INDEX=3
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_rim_kodak_gpu_common.sh"
