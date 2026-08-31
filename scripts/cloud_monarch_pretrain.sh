#!/usr/bin/env bash
set -euo pipefail

model=${1:?usage: cloud_monarch_pretrain.sh hmoe|dense adamw|muon BLOCKS ddp|ep|pp smoke|reload|bench|full}
arm=${2:?usage: cloud_monarch_pretrain.sh hmoe|dense adamw|muon BLOCKS ddp|ep|pp smoke|reload|bench|full}
blocks=${3:?usage: cloud_monarch_pretrain.sh hmoe|dense adamw|muon BLOCKS ddp|ep|pp smoke|reload|bench|full}
parallelism=${4:?usage: cloud_monarch_pretrain.sh hmoe|dense adamw|muon BLOCKS ddp|ep|pp smoke|reload|bench|full}
mode=${5:?usage: cloud_monarch_pretrain.sh hmoe|dense adamw|muon BLOCKS ddp|ep|pp smoke|reload|bench|full}
root=$(cd "$(dirname "$0")/.." && pwd)
source "$root/configs/stage3-moe-1p029b.sh"
source "$root/configs/dense-1p028b.sh"

[[ ${MLSUB_IMAGE:-} == torch28 ]] || { echo "requires --image torch28" >&2; exit 2; }
[[ -z ${TORCHELASTIC_RUN_ID:-} ]] || { echo "nested torchrun is not allowed" >&2; exit 2; }
[[ $blocks == 2 || $blocks == 4 ]] || { echo "BLOCKS must be 2 or 4" >&2; exit 2; }

export RANK=${OMPI_COMM_WORLD_RANK:?missing OMPI_COMM_WORLD_RANK}
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:?missing OMPI_COMM_WORLD_SIZE}
export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:?missing OMPI_COMM_WORLD_LOCAL_RANK}
local_world_size=${OMPI_COMM_WORLD_LOCAL_SIZE:?missing OMPI_COMM_WORLD_LOCAL_SIZE}
visible_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
case "$WORLD_SIZE" in 1|2|4) ;; *) echo "WORLD_SIZE must be 1, 2, or 4" >&2; exit 2 ;; esac
if (( WORLD_SIZE != local_world_size || local_world_size != visible_gpus )); then
  echo "one MPI process per visible GPU is required: world=$WORLD_SIZE local_world=$local_world_size visible_gpus=$visible_gpus" >&2
  exit 2
fi

export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-29547}
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
export CUDA_DEVICE_MAX_CONNECTIONS=1
unset PYTHONNOUSERSITE
nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
  -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: - || true)
export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDNN_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cudnn
export CURAND_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/curand
export NVRTC_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cuda_nvrtc

case "$arm" in
  adamw) optimizer=(--optimizer adam) ;;
  muon) optimizer=(--optimizer muon "${STAGE3_MOE_MUON_ARGS[@]}") ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

tensor_parallel=1
pipeline_parallel=1
expert_parallel=1
case "$model:$parallelism" in
  hmoe:ddp|dense:ddp) ;;
  hmoe:ep)
    (( WORLD_SIZE > 1 && STAGE3_MOE_ROUTED_EXPERTS % WORLD_SIZE == 0 )) || {
      echo "H-MoE EP requires 2 or 4 ranks dividing $STAGE3_MOE_ROUTED_EXPERTS experts" >&2
      exit 2
    }
    expert_parallel=$WORLD_SIZE
    ;;
  dense:pp)
    (( WORLD_SIZE > 1 && 16 % WORLD_SIZE == 0 )) || {
      echo "dense PP requires 2 or 4 ranks dividing 16 layers" >&2
      exit 2
    }
    pipeline_parallel=$WORLD_SIZE
    ;;
  *) echo "unsupported model/parallelism pair: $model/$parallelism" >&2; exit 2 ;;
esac

global_batch=${MONARCH_GLOBAL_BATCH:-208}
micro_batch=${MONARCH_MICRO_BATCH:-4}
data_parallel=$((WORLD_SIZE / tensor_parallel / pipeline_parallel))
if (( global_batch % (micro_batch * data_parallel) != 0 )); then
  echo "global batch must be divisible by micro batch times data-parallel size" >&2
  exit 2
fi

base_data=${MONARCH_BASE_DATA:-/home/jovyan/data/fineweb-edu-gpt2-megatron/data}
old_extension=${MONARCH_OLD_EXTENSION:-/workspace-SR006.nfs2/hmoe-data/fineweb-edu-time-match-extension/data/train}
new_extension=${MONARCH_DENSE_EXTENSION:-/workspace-SR006.nfs2/hmoe-data/fineweb-edu-dense-1c-extension/data/train}
case "$model" in
  hmoe)
    model_args=("${STAGE3_MOE_MODEL_ARGS[@]}" "${STAGE3_MOE_ROUTER_ARGS[@]}")
    train_data=("$base_data/train")
    data_manifests=("$base_data/../artifact-manifest.json")
    beta2=0.95
    peak_lr=1.63e-3
    min_lr=1.63e-4
    target_iters=17242
    decay_iters=3448
    warmup_iters=173
    decay_style=exponential
    save_interval=363
    retain_interval=13794
    storage_root=${MONARCH_CKPT_ROOT:-/workspace-SR006.nfs3/monarch-pretrain}
    ;;
  dense)
    model_args=("${DENSE_1P028B_MODEL_ARGS[@]}")
    train_data=("$base_data/train" "$old_extension" "$new_extension")
    data_manifests=(
      "$base_data/../artifact-manifest.json"
      "${old_extension%/data/train}/artifact-manifest.json"
      "${new_extension%/data/train}/artifact-manifest.json"
    )
    beta2=0.99
    peak_lr=1e-3
    min_lr=0
    target_iters=48265
    decay_iters=4827
    warmup_iters=615
    decay_style=cosine
    save_interval=587
    retain_interval=43438
    storage_root=${MONARCH_CKPT_ROOT:-/workspace-SR006.nfs2/monarch-pretrain}
    ;;
  *) echo "unknown model: $model" >&2; exit 2 ;;
esac

for prefix in "${train_data[@]}" "$base_data/development" "$base_data/final"; do
  [[ -s ${prefix}.bin && -s ${prefix}.idx ]] || { echo "missing indexed dataset: $prefix" >&2; exit 2; }
done
if [[ $model == dense ]]; then
  [[ -s ${new_extension%/data/train}/artifact-manifest.json ]] || {
    echo "missing dense extension manifest: ${new_extension%/data/train}/artifact-manifest.json" >&2
    exit 2
  }
fi

suffix=${MONARCH_RUN_SUFFIX:+-$MONARCH_RUN_SUFFIX}
case "$mode" in
  smoke)
    run_phase=smoke
    train_iters=12
    mode_save_interval=6
    min_free_gb=${MONARCH_MIN_FREE_GB:-20}
    scheduler_override=()
    eval_interval=1000000
    eval_iters=2
    log_interval=1
    ;;
  reload)
    run_phase=smoke
    train_iters=14
    mode_save_interval=6
    min_free_gb=${MONARCH_MIN_FREE_GB:-20}
    scheduler_override=(--override-opt_param-scheduler)
    eval_interval=1000000
    eval_iters=2
    log_interval=1
    ;;
  bench)
    run_phase=bench
    train_iters=${MONARCH_BENCH_ITERS:-25}
    mode_save_interval=0
    min_free_gb=${MONARCH_MIN_FREE_GB:-5}
    scheduler_override=()
    eval_interval=1000000
    eval_iters=0
    log_interval=1
    ;;
  full)
    run_phase=1c
    train_iters=$target_iters
    mode_save_interval=$save_interval
    min_free_gb=${MONARCH_MIN_FREE_GB:-30}
    scheduler_override=()
    eval_interval=250
    eval_iters=32
    log_interval=10
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

run_id="monarch-${model}-${arm}-n${blocks}-${parallelism}${WORLD_SIZE}-${run_phase}${suffix}"
ckpt_dir="$storage_root/$run_id"
log_root=${MONARCH_LOG_ROOT:-/home/jovyan/hmoe-cloud/monarch-pretrain}
rank_log="$log_root/$run_id/rank-${RANK}-$(date -u +%Y%m%dT%H%M%SZ).log"
data_cache_args=()
if [[ -n ${MONARCH_DATA_CACHE_PATH:-} ]]; then
  data_cache=$MONARCH_DATA_CACHE_PATH
elif [[ $model == dense ]]; then
  data_cache="$storage_root/data-cache/dense-1c"
else
  data_cache=dataset-default
fi
mkdir -p "$log_root/$run_id"
if [[ $data_cache != dataset-default ]]; then
  mkdir -p "$data_cache"
  data_cache_args=(--data-cache-path "$data_cache")
fi

if [[ $mode == reload && ! -s $ckpt_dir/latest_checkpointed_iteration.txt ]]; then
  echo "reload checkpoint is missing: $ckpt_dir/latest_checkpointed_iteration.txt" >&2
  exit 2
fi
if [[ $mode != bench ]]; then
  mkdir -p "$ckpt_dir"
  available_kb=$(df -Pk "$storage_root" | awk 'NR==2 {print $4}')
  if (( available_kb < min_free_gb * 1024 * 1024 )); then
    echo "need at least ${min_free_gb} GiB free on $storage_root" >&2
    df -h "$storage_root"
    exit 2
  fi
fi

export WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb-radfan.ru}
if [[ -z ${WANDB_API_KEY:-} && -f /home/jovyan/.wandb-key ]]; then
  export WANDB_API_KEY=$(< /home/jovyan/.wandb-key)
fi
logger_args=()
if [[ -n ${WANDB_API_KEY:-} ]] && python -c 'import wandb, torch.utils.tensorboard' >/dev/null 2>&1; then
  export WANDB_RUN_ID=${WANDB_RUN_ID:-$run_id}
  export WANDB_RESUME=${WANDB_RESUME:-allow}
  export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-monarch-1b-pretrain}
  export WANDB_TAGS=${WANDB_TAGS:-"monarch,nblocks${blocks},${model},${arm},${parallelism}${WORLD_SIZE}"}
  logger_args=(
    --tensorboard-dir "$log_root/$run_id/tensorboard"
    --tensorboard-log-interval 10
    --wandb-project "${MONARCH_WANDB_PROJECT:-muon-variations}"
    --wandb-entity "${MONARCH_WANDB_ENTITY:-efficient-muon}"
    --wandb-exp-name "$run_id"
    --wandb-save-dir "$log_root/$run_id"
  )
  wandb_status="online host=$WANDB_BASE_URL"
elif [[ $mode == full ]]; then
  echo "full mode requires W&B credentials and importable wandb/tensorboard" >&2
  exit 2
else
  export WANDB_MODE=offline
  wandb_status=offline
fi

flock "$root/third_party/Megatron-LM/megatron/core/datasets/.helpers-build.lock" \
  make -s -C "$root/third_party/Megatron-LM/megatron/core/datasets"

save_args=()
if (( mode_save_interval > 0 )); then
  save_args=(--save "$ckpt_dir" --load "$ckpt_dir" --save-interval "$mode_save_interval")
  if [[ $mode == full ]]; then
    save_args+=(--save-retain-interval "$retain_interval")
  fi
fi

gpu_uuid=$(nvidia-smi -i "$LOCAL_RANK" --query-gpu=uuid --format=csv,noheader)
echo "MONARCH_TRAIN_PROCESS model=$model arm=$arm blocks=$blocks rank=$RANK world_size=$WORLD_SIZE local_rank=$LOCAL_RANK local_world_size=$local_world_size pid=$$ gpu_uuid=$gpu_uuid parallelism=$parallelism tp=$tensor_parallel pp=$pipeline_parallel ep=$expert_parallel dp=$data_parallel nested_torchrun=false"
echo "MONARCH_TRAIN_CONFIG run_id=$run_id mode=$mode micro_batch=$micro_batch global_batch=$global_batch target_iters=$target_iters train_iters=$train_iters warmup=$warmup_iters decay=$decay_iters lr=$peak_lr min_lr=$min_lr wd=0.1 wandb=$wandb_status"
echo "MONARCH_DATA train=${train_data[*]} valid=$base_data/development test=$base_data/final cache=$data_cache"
echo "MONARCH_STORAGE checkpoint=${ckpt_dir:-none} log=$rank_log"
echo "MONARCH_CODE commit=$(git -C "$root" rev-parse HEAD)"
for manifest in "${data_manifests[@]}"; do
  [[ -f $manifest ]] && echo "MONARCH_DATA_MANIFEST $(sha256sum "$manifest")"
done

set +e
python "$root/stage3_moe/pretrain_monarch.py" \
  --monarch-blocks "$blocks" \
  "${model_args[@]}" \
  --tensor-model-parallel-size "$tensor_parallel" \
  --pipeline-model-parallel-size "$pipeline_parallel" \
  --context-parallel-size 1 \
  --expert-model-parallel-size "$expert_parallel" \
  --expert-tensor-parallel-size 1 \
  --transformer-impl transformer_engine \
  --bf16 \
  "${optimizer[@]}" \
  --adam-beta1 0.9 \
  --adam-beta2 "$beta2" \
  --adam-eps 1e-8 \
  --lr "$peak_lr" \
  --min-lr "$min_lr" \
  --lr-decay-style WSD \
  --lr-decay-iters "$target_iters" \
  --lr-wsd-decay-iters "$decay_iters" \
  --lr-wsd-decay-style "$decay_style" \
  --lr-warmup-iters "$warmup_iters" \
  --weight-decay 0.1 \
  --clip-grad 1 \
  --micro-batch-size "$micro_batch" \
  --global-batch-size "$global_batch" \
  --train-iters "$train_iters" \
  --tokenizer-type NullTokenizer \
  --vocab-size 50257 \
  --null-tokenizer-eod-id 50256 \
  --null-tokenizer-pad-id -1 \
  --train-data-path "${train_data[@]}" \
  --valid-data-path "$base_data/development" \
  --test-data-path "$base_data/final" \
  "${data_cache_args[@]}" \
  --dataloader-type single \
  --num-workers 2 \
  --no-create-attention-mask-in-dataloader \
  --seed 1234 \
  --eval-interval "${MONARCH_EVAL_INTERVAL:-$eval_interval}" \
  --eval-iters "${MONARCH_EVAL_ITERS:-$eval_iters}" \
  --log-interval "${MONARCH_LOG_INTERVAL:-$log_interval}" \
  --log-throughput \
  --timing-log-level 1 \
  --no-gradient-accumulation-fusion \
  --ckpt-format torch \
  "${logger_args[@]}" \
  "${save_args[@]}" \
  "${scheduler_override[@]}" \
  2>&1 | tee "$rank_log"
train_exit=${PIPESTATUS[0]}
set -e
echo "TRAIN_EXIT=$train_exit rank=$RANK run_id=$run_id log=$rank_log"
exit "$train_exit"
