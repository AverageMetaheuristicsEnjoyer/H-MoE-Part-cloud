#!/usr/bin/env bash
set -euo pipefail

arm=${1:?usage: node207_monarch_pretrain.sh adamw|muon BLOCKS smoke|reload|bench|full GPU}
blocks=${2:?usage: node207_monarch_pretrain.sh adamw|muon BLOCKS smoke|reload|bench|full GPU}
mode=${3:?usage: node207_monarch_pretrain.sh adamw|muon BLOCKS smoke|reload|bench|full GPU}
gpu=${4:?usage: node207_monarch_pretrain.sh adamw|muon BLOCKS smoke|reload|bench|full GPU}
root=$(cd "$(dirname "$0")/.." && pwd)
work_root=${NODE207_MONARCH_ROOT:-/var/tmp/user1-monarch-pretrain}
python_bin=${MONARCH_PYTHON:-$root/scripts/node207_monarch_python.sh}
runtime_python=${NODE207_MONARCH_PYTHON:-$work_root/venv-py312/bin/python}

[[ $gpu =~ ^[0-7]$ ]] || { echo "GPU must be a physical index from 0 to 7" >&2; exit 2; }
[[ -x $python_bin ]] || { echo "missing Python: $python_bin" >&2; exit 2; }
[[ -x $runtime_python ]] || { echo "missing Python runtime: $runtime_python" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES=$gpu
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-$((29600 + gpu))}
export MONARCH_RUNTIME=node207
export MONARCH_PYTHON=$python_bin
export NODE207_MONARCH_PYTHON=$runtime_python
export MONARCH_BASE_DATA=${MONARCH_BASE_DATA:-$work_root/data/fineweb-edu-gpt2-megatron/data}
export MONARCH_CKPT_ROOT=${MONARCH_CKPT_ROOT:-$work_root/checkpoints}
export MONARCH_LOG_ROOT=${MONARCH_LOG_ROOT:-$work_root/logs}
export MONARCH_DATA_CACHE_PATH=${MONARCH_DATA_CACHE_PATH:-$work_root/dataset-cache/hmoe}
export TMPDIR=${TMPDIR:-$work_root/tmp}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$work_root/triton-cache/gpu-$gpu-$arm}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$work_root/torchinductor-cache/gpu-$gpu-$arm}
export WANDB_DIR=${WANDB_DIR:-$work_root/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$work_root/wandb-cache}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-$work_root/wandb-config}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-$work_root/wandb-data}
export PATH="$(dirname "$python_bin"):$PATH"

mkdir -p "$MONARCH_CKPT_ROOT" "$MONARCH_LOG_ROOT" "$MONARCH_DATA_CACHE_PATH" \
  "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"
df -h "$work_root"
df -i "$work_root"
nvidia-smi -i "$gpu" --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits

exec "$root/scripts/cloud_monarch_pretrain.sh" hmoe "$arm" "$blocks" ddp "$mode"
