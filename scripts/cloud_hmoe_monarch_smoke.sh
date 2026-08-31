#!/usr/bin/env bash
set -euo pipefail

arm=${1:?usage: cloud_hmoe_monarch_smoke.sh baseline|adamw|muon BLOCKS [ddp|ep]}
blocks=${2:?usage: cloud_hmoe_monarch_smoke.sh baseline|adamw|muon BLOCKS [ddp|ep]}
parallelism=${3:-ddp}
root=$(cd "$(dirname "$0")/.." && pwd)
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
export MASTER_PORT=${MASTER_PORT:-29541}
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
export CUDA_DEVICE_MAX_CONNECTIONS=1

nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
  -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: - || true)
export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDNN_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cudnn
export CURAND_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/curand
export NVRTC_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cuda_nvrtc

flock "$root/third_party/Megatron-LM/megatron/core/datasets/.helpers-build.lock" \
  make -s -C "$root/third_party/Megatron-LM/megatron/core/datasets"

case "$arm" in
  baseline)
    optimizer=(--optimizer adam)
    runner="$root/third_party/Megatron-LM/pretrain_gpt.py"
    monarch_args=()
    ;;
  adamw)
    optimizer=(--optimizer adam)
    runner="$root/stage3_moe/pretrain_monarch.py"
    monarch_args=(--monarch-blocks "$blocks")
    ;;
  muon)
    optimizer=(--optimizer muon "${STAGE3_MOE_MUON_ARGS[@]}")
    runner="$root/stage3_moe/pretrain_monarch.py"
    monarch_args=(--monarch-blocks "$blocks")
    ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

case "$parallelism" in
  ddp) expert_parallel=1 ;;
  ep)
    (( WORLD_SIZE > 1 )) || { echo "EP requires more than one rank" >&2; exit 2; }
    (( STAGE3_MOE_ROUTED_EXPERTS % WORLD_SIZE == 0 )) || {
      echo "experts must be divisible by world size" >&2
      exit 2
    }
    expert_parallel=$WORLD_SIZE
    ;;
  *) echo "unknown parallelism: $parallelism" >&2; exit 2 ;;
esac

gpu_uuid=$(nvidia-smi -i "$LOCAL_RANK" --query-gpu=uuid --format=csv,noheader)
echo "HMOE_MONARCH_PROCESS rank=$RANK world_size=$WORLD_SIZE local_rank=$LOCAL_RANK local_world_size=$local_world_size pid=$$ gpu_uuid=$gpu_uuid parallelism=$parallelism ep=$expert_parallel nested_torchrun=false"

global_batch=${MONARCH_SMOKE_GLOBAL_BATCH:-$((2 * WORLD_SIZE))}
python "$runner" \
  "${monarch_args[@]}" \
  "${STAGE3_MOE_MODEL_ARGS[@]}" \
  "${STAGE3_MOE_ROUTER_ARGS[@]}" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size "$expert_parallel" \
  --expert-tensor-parallel-size 1 \
  --transformer-impl transformer_engine \
  --bf16 \
  "${optimizer[@]}" \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-8 \
  --lr 1.63e-3 \
  --min-lr 1.63e-3 \
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
