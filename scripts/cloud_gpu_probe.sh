#!/usr/bin/env bash
# Cheapest possible check of what the scheduler actually grants for a given --gpus.
set -u
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
echo "GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
df -h /home/jovyan | tail -1
echo "EXIT=0"
exit 0
