#!/usr/bin/env bash
set -euo pipefail

arm=${1:?usage: cloud_dense_monarch_smoke.sh adamw|muon BLOCKS [ddp|pp]}
blocks=${2:?usage: cloud_dense_monarch_smoke.sh adamw|muon BLOCKS [ddp|pp]}
parallelism=${3:-ddp}
root=$(cd "$(dirname "$0")/.." && pwd)
source "$root/configs/dense-1p028b.sh"
source "$root/configs/stage3-moe-1p029b.sh"

[[ ${MLSUB_IMAGE:-} == torch28 ]] || { echo "requires --image torch28" >&2; exit 2; }
[[ -z ${TORCHELASTIC_RUN_ID:-} ]] || { echo "nested torchrun is not allowed" >&2; exit 2; }

export RANK=${OMPI_COMM_WORLD_RANK:?missing OMPI_COMM_WORLD_RANK}
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:?missing OMPI_COMM_WORLD_SIZE}
export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:?missing OMPI_COMM_WORLD_LOCAL_RANK}
local_world_size=${OMPI_COMM_WORLD_LOCAL_SIZE:?missing OMPI_COMM_WORLD_LOCAL_SIZE}
visible_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( WORLD_SIZE != local_world_size || local_world_size != visible_gpus )); then
  echo "one MPI process per visible GPU is required: world=$WORLD_SIZE local_world=$local_world_size visible_gpus=$visible_gpus" >&2
  exit 2
fi
export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-29543}
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
export CUDA_DEVICE_MAX_CONNECTIONS=1

nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
  -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: - || true)
export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDNN_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cudnn
export CURAND_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/curand
export NVRTC_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cuda_nvrtc

make -s -C "$root/third_party/Megatron-LM/megatron/core/datasets"

case "$arm" in
  adamw) optimizer=(--optimizer adam) ;;
  muon) optimizer=(--optimizer muon "${STAGE3_MOE_MUON_ARGS[@]}") ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

case "$parallelism" in
  ddp) pipeline_parallel=1 ;;
  pp)
    (( WORLD_SIZE > 1 )) || { echo "PP requires more than one rank" >&2; exit 2; }
    (( 16 % WORLD_SIZE == 0 )) || {
      echo "dense layers must be divisible by world size" >&2
      exit 2
    }
    pipeline_parallel=$WORLD_SIZE
    ;;
  *) echo "unknown parallelism: $parallelism" >&2; exit 2 ;;
esac

gpu_uuid=$(nvidia-smi -i "$LOCAL_RANK" --query-gpu=uuid --format=csv,noheader)
echo "DENSE_MONARCH_PROCESS rank=$RANK world_size=$WORLD_SIZE local_rank=$LOCAL_RANK local_world_size=$local_world_size pid=$$ gpu_uuid=$gpu_uuid parallelism=$parallelism pp=$pipeline_parallel nested_torchrun=false"

global_batch=${MONARCH_SMOKE_GLOBAL_BATCH:-$((2 * WORLD_SIZE))}
python "$root/stage3_moe/pretrain_monarch.py" \
  --monarch-blocks "$blocks" \
  "${DENSE_1P028B_MODEL_ARGS[@]}" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size "$pipeline_parallel" \
  --context-parallel-size 1 \
  --transformer-impl transformer_engine \
  --bf16 \
  "${optimizer[@]}" \
  --adam-beta1 0.9 \
  --adam-beta2 0.99 \
  --adam-eps 1e-8 \
  --lr 1e-3 \
  --min-lr 1e-3 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --clip-grad 1 \
  --micro-batch-size 2 \
  --global-batch-size "$global_batch" \
  --train-iters 6 \
  --eval-iters 0 \
  --eval-interval 1000000 \
  --tokenizer-type NullTokenizer \
  --vocab-size 50257 \
  --null-tokenizer-eod-id 50256 \
  --null-tokenizer-pad-id -1 \
  --mock-data \
  --num-workers 0 \
  --no-create-attention-mask-in-dataloader \
  --seed 1234 \
  --log-interval 1 \
  --log-throughput \
  --no-gradient-accumulation-fusion
