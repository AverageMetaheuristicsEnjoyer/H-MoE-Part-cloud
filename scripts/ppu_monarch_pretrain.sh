#!/usr/bin/env bash
set -euo pipefail

model=${1:?usage: ppu_monarch_pretrain.sh hmoe|dense adamw|muon BLOCKS ddp|ep|pp smoke|reload|bench|full}
arm=${2:?usage: ppu_monarch_pretrain.sh hmoe|dense adamw|muon BLOCKS ddp|ep|pp smoke|reload|bench|full}
blocks=${3:?usage: ppu_monarch_pretrain.sh hmoe|dense adamw|muon BLOCKS ddp|ep|pp smoke|reload|bench|full}
parallelism=${4:?usage: ppu_monarch_pretrain.sh hmoe|dense adamw|muon BLOCKS ddp|ep|pp smoke|reload|bench|full}
mode=${5:?usage: ppu_monarch_pretrain.sh hmoe|dense adamw|muon BLOCKS ddp|ep|pp smoke|reload|bench|full}

[[ -n ${SLURM_JOB_ID:-} ]] || { echo "run inside a Slurm GPU allocation" >&2; exit 2; }
[[ -n ${CUDA_VISIBLE_DEVICES:-} ]] || { echo "Slurm did not expose any PPU" >&2; exit 2; }

root=$(cd "$(dirname "$0")/.." && pwd)
# The training-24.09 image has torch 2.4, which cannot import this vendored MCore.
image=${MONARCH_PPU_IMAGE:-/bmcp_lvm_fs/apptainer/sif/asllm.sif}
python_bin=${MONARCH_PYTHON:-python3}
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
nproc=${#devices[@]}
case "$nproc" in 1|2|4) ;; *) echo "expected 1, 2, or 4 allocated PPUs, got $nproc" >&2; exit 2 ;; esac

export MONARCH_RUNTIME=ppu
export MONARCH_PYTHON="$python_bin"
export MONARCH_BASE_DATA=${MONARCH_BASE_DATA:-/bmcp_lvm_fs/data/datasets/fineweb-edu-gpt2-megatron/data}
export MONARCH_OLD_EXTENSION=${MONARCH_OLD_EXTENSION:-/bmcp_lvm_fs/data/datasets/fineweb-edu-time-match-extension/data/train}
export MONARCH_DENSE_EXTENSION=${MONARCH_DENSE_EXTENSION:-/bmcp_lvm_fs/data/datasets/fineweb-edu-dense-1c-extension/data/train}
export MONARCH_CKPT_ROOT=${MONARCH_CKPT_ROOT:-/bmcp_lvm_fs/scratch/$USER/monarch-pretrain}
export MONARCH_DATA_CACHE_PATH=${MONARCH_DATA_CACHE_PATH:-$MONARCH_CKPT_ROOT/data-cache/${model}-1c}
export MONARCH_LOG_ROOT=${MONARCH_LOG_ROOT:-$HOME/logs/monarch-pretrain}

mkdir -p "$MONARCH_CKPT_ROOT" "$MONARCH_DATA_CACHE_PATH" "$MONARCH_LOG_ROOT"
df -h "$HOME" "$MONARCH_CKPT_ROOT" "$MONARCH_BASE_DATA"
df -i "$HOME" "$MONARCH_CKPT_ROOT" "$MONARCH_BASE_DATA"

echo "PPU_MONARCH_LAUNCH slurm_job=$SLURM_JOB_ID nproc=$nproc visible=$CUDA_VISIBLE_DEVICES image=$image python=$python_bin"
apptainer exec \
  --bind /bmcp_lvm_fs:/bmcp_lvm_fs \
  "$image" \
  "$python_bin" -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc-per-node "$nproc" \
    --max-restarts 0 \
    --no-python \
    "$root/scripts/cloud_monarch_pretrain.sh" \
    "$model" "$arm" "$blocks" "$parallelism" "$mode"
