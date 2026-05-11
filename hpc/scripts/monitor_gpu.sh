#!/bin/bash
set -euo pipefail

OUT_FILE="${1:?Usage: monitor_gpu.sh OUT_FILE [INTERVAL_SECONDS]}"
INTERVAL="${2:-5}"

mkdir -p "$(dirname "${OUT_FILE}")"

nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,nounits \
  --loop="${INTERVAL}" > "${OUT_FILE}"
