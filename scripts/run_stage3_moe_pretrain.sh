#!/usr/bin/env bash
# Stage 3 MoE matched pretraining, WSD trunk-and-branch.
#
#   run_stage3_moe_pretrain.sh ARM trunk|decay-1p2b|smoke|bench
#
# smoke exercises save and resume; bench measures throughput and peak memory with no
# checkpoint traffic. STAGE3_MOE_RUN_SUFFIX keeps concurrent variants from sharing a
# run directory or a checkpoint.
#
# The schedule follows docs/design.md:182 -- linear warmup on steps 1-173,
# constant peak LR through step 13,794, exponential decay over the final 20% to
# 0.1x peak at step 17,242 (7,344,816,128 tokens, the Ling MoE-optimal budget).
# Because the stable phase is at constant LR, one trunk serves every budget: a
# shorter budget just branches off earlier and runs its own decay tail.
set -euo pipefail

arm=${1:?usage: run_stage3_moe_pretrain.sh ARM trunk|decay-1p2b}
mode=${2:?usage: run_stage3_moe_pretrain.sh ARM trunk|decay-1p2b}
root=$(cd "$(dirname "$0")/.." && pwd)
source "$root/configs/stage3-moe-1p029b.sh"

# --- budget, in steps of 208 x 2048 = 425,984 loss tokens ---
full_iters=17242          # 7,344,816,128 tokens = 1C for this MoE
full_decay_iters=3448     # final 20%, starts at step 13,795
warmup_iters=173          # first 1%
short_iters=2818          # 1,200,422,912 tokens
short_decay_iters=564     # final 20% of the short budget
short_branch=$((short_iters - short_decay_iters))   # 2254

global_batch=208
micro_batch=${STAGE3_MOE_MICRO_BATCH:-4}            # 4 x DP2 x accum26 = 208
data_root=${STAGE3_MOE_DATA_ROOT:-/home/jovyan/data/fineweb-edu-gpt2-megatron/data}
ckpt_root=${STAGE3_MOE_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3}
log_root=${STAGE3_MOE_LOG_ROOT:-/home/jovyan/hmoe-cloud/pretrain}

case "$arm" in
  adamw_bf16_state_fp32) optimizer=adam; state_precision=fp32; compute=() ;;
  adamw_bf16_state_fp8)  optimizer=adam; state_precision=fp8;  compute=() ;;
  muon_bf16_state_fp32)  optimizer=muon; state_precision=fp32; compute=() ;;
  muon_bf16_state_fp8)   optimizer=muon; state_precision=fp8;  compute=() ;;
  adamw_fp8gemm_state_fp32) optimizer=adam; state_precision=fp32; compute=(--fp8-format hybrid --fp8-recipe delayed) ;;
  muon_fp8gemm_state_fp32)  optimizer=muon; state_precision=fp32; compute=(--fp8-format hybrid --fp8-recipe delayed) ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac
probe_warmup=20
probe_measure=100
optimizer_args=()
[[ $optimizer == muon ]] && optimizer_args=("${STAGE3_MOE_MUON_ARGS[@]}")

trunk_dir="$ckpt_root/trunk/$arm"
decay_dir="$ckpt_root/1p2b/$arm"

case "$mode" in
  trunk)
    # Stable phase only: stop at the branch point, never reaching decay.
    train_iters=$short_branch
    target_iters=$full_iters
    decay_iters=$full_decay_iters
    # Keep the latest for crash resume; retain the branch point permanently.
    save_args=(--save "$trunk_dir" --save-interval 322 --save-retain-interval "$short_branch")
    load_args=(--load "$trunk_dir")
    ;;
  decay-1p2b)
    train_iters=$short_iters
    target_iters=$short_iters
    decay_iters=$short_decay_iters
    mkdir -p "$decay_dir"
    # Seed the branch from the trunk with hardlinks: same filesystem, no copy,
    # no extra space, and the run can then load and save in one directory.
    if [[ ! -f "$decay_dir/latest_checkpointed_iteration.txt" ]]; then
      src=$(printf '%s/iter_%07d' "$trunk_dir" "$short_branch")
      [[ -d $src ]] || { echo "trunk branch point missing: $src" >&2; exit 2; }
      cp -al "$src" "$decay_dir/" 2>/dev/null || cp -a "$src" "$decay_dir/"
      echo "$short_branch" > "$decay_dir/latest_checkpointed_iteration.txt"
    fi
    # Only the endpoint is kept, and without optimizer state: we never resume
    # from it, it feeds validation and downstream evaluation.
    save_args=(--save "$decay_dir" --save-interval "$short_decay_iters" --no-save-optim --no-save-rng)
    load_args=(--load "$decay_dir" --override-opt_param-scheduler)
    ;;
  smoke)
    # Exercises the whole path cheaply: real data, 2 GPUs, checkpoint save and
    # resume on the alternate volume, loggers. Re-run it to test resume.
    train_iters=25
    target_iters=$full_iters
    decay_iters=$full_decay_iters
    smoke_dir="$ckpt_root/smoke/$arm${STAGE3_MOE_RUN_SUFFIX:+-$STAGE3_MOE_RUN_SUFFIX}"
    save_args=(--save "$smoke_dir" --save-interval 10)
    load_args=(--load "$smoke_dir")
    probe_warmup=5
    probe_measure=10
    ;;
  bench)
    # Throughput and peak memory only. No checkpoint traffic, so an NFS write never
    # lands inside the measured window and topologies stay comparable.
    train_iters=25
    target_iters=$full_iters
    decay_iters=$full_decay_iters
    save_args=()
    load_args=()
    probe_warmup=5
    probe_measure=10
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

run_id="stage3-$arm-$mode${STAGE3_MOE_RUN_SUFFIX:+-$STAGE3_MOE_RUN_SUFFIX}"
mkdir -p "$log_root/$run_id" "$trunk_dir"
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
# Transformer Engine loads cudart and friends from the pip nvidia packages; without
# these the run dies with "cudart shared object not found".
unset PYTHONNOUSERSITE
nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
  -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDNN_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cudnn
export CURAND_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/curand
export NVRTC_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cuda_nvrtc
# Without credentials wandb.init() would abort the run, so fall back to offline
# logging: the run still records everything and can be `wandb sync`ed later.
if [[ -f /home/jovyan/.wandb-key ]]; then
  export WANDB_API_KEY=$(cat /home/jovyan/.wandb-key)
else
  export WANDB_MODE=offline
  echo "WANDB=offline (no /home/jovyan/.wandb-key); sync later with 'wandb sync'"
fi

# A missing logger must not kill a multi-hour run: drop the loggers instead.
logger_args=()
if python -c 'import wandb, torch.utils.tensorboard' >/dev/null 2>&1; then
  logger_args=(
    --tensorboard-dir "$log_root/$run_id/tensorboard"
    --tensorboard-log-interval 10
    --wandb-project "${STAGE3_MOE_WANDB_PROJECT:-hmoe-stage3}"
    --wandb-exp-name "$run_id"
    --wandb-save-dir "$log_root/$run_id"
  )
else
  echo "LOGGERS=absent (wandb/tensorboard not importable); running without them"
fi

# Apex's fused kernel is absent from the cloud image. Transformer Engine implements the
# fusion itself, so only the LM head (a plain ColumnParallelLinear) actually needs Apex,
# and the patched layer now degrades on its own instead of aborting the run. Asking for
# the fusion therefore buys it for every TE layer, including the expert GEMMs.
fusion_args=()
if [[ ${STAGE3_MOE_WGRAD_FUSION:-0} == 1 ]]; then
  echo "GRADIENT_ACCUMULATION_FUSION=requested (TE layers fuse, LM head falls back)"
elif ! python -c 'import fused_weight_gradient_mlp_cuda' >/dev/null 2>&1; then
  fusion_args=(--no-gradient-accumulation-fusion)
  echo "GRADIENT_ACCUMULATION_FUSION=disabled"
fi

# Throughput levers, kept off until a paired measurement justifies each one.
speed_args=()
if [[ ${STAGE3_MOE_OVERLAP_GRAD_REDUCE:-0} == 1 ]]; then
  speed_args+=(--overlap-grad-reduce)
fi
if [[ ${STAGE3_MOE_MOE_FUSIONS:-0} == 1 ]]; then
  speed_args+=(--moe-permute-fusion --moe-shared-expert-overlap)
fi
if [[ ${#speed_args[@]} -gt 0 ]]; then
  echo "SPEED_ARGS=${speed_args[*]}"
fi

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
# The result writer reads these; on real data the manifest hash is mandatory.
export STAGE3_MOE_RUN_ID="$run_id"
export STAGE3_MOE_SITE=cloudru
export STAGE3_MOE_IMAGE="${MLSUB_IMAGE:-torch28}"
export STAGE3_MOE_CONFIG_SHA256=$(sha256sum "$root/configs/stage3-moe-1p029b.sh" | awk '{print $1}')
export STAGE3_MOE_DATA_MANIFEST_SHA256=$(
  if [[ -f "$data_root/../artifact-manifest.json" ]]; then
    sha256sum "$data_root/../artifact-manifest.json" | awk '{print $1}'
  else
    printf '%s' 'AverageMetaheuristicsEnjoyer/fineweb-edu-gpt2-megatron' | sha256sum | awk '{print $1}'
  fi)
export STAGE3_MOE_MCORE_COMMIT="$STAGE3_MOE_MCORE_COMMIT"
export STAGE3_MOE_EO_COMMIT="$STAGE3_MOE_EO_COMMIT"
export STAGE3_MOE_GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)
export STAGE3_MOE_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
# Ask TE for the cuBLASLt version; the writer's ctypes fallback cannot find the
# shared object in this image.
export STAGE3_MOE_CUBLASLT=$(python -c 'import transformer_engine_torch as t; print(t.get_cublasLt_version())' 2>/dev/null || echo unknown)
export STAGE3_MOE_GPU_CLEAN_BEFORE=1 STAGE3_MOE_GPU_CLEAN_DURING=1 STAGE3_MOE_GPU_CLEAN_AFTER=1
echo "ARM=$arm MODE=$mode GPUS=$gpu_count micro_batch=$micro_batch global_batch=$global_batch"
echo "FP8_DEQUANT_CHUNK=${STAGE3_MOE_FP8_DEQUANT_CHUNK:-0} (0 = every state in FP32 at once)"
echo "SCHEDULE target_iters=$target_iters decay_iters=$decay_iters warmup=$warmup_iters train_iters=$train_iters"
echo "CKPT save=${save_args[*]} load=${load_args[*]}"

train_log="$log_root/$run_id/train-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "TRAIN_LOG=$train_log"
set +e
python -m torch.distributed.run --standalone --nproc-per-node "$gpu_count" \
  stage3_moe/pretrain_gpt.py \
  --stage3-arm "$arm" \
  --stage3-result-path "$log_root/$run_id/results.jsonl" \
  --stage3-warmup-steps "$probe_warmup" \
  --stage3-measure-steps "$probe_measure" \
  --optimizer-state-precision "$state_precision" \
  "${STAGE3_MOE_MODEL_ARGS[@]}" \
  "${STAGE3_MOE_ROUTER_ARGS[@]}" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size "${STAGE3_MOE_EP:-1}" \
  --expert-tensor-parallel-size 1 \
  --transformer-impl transformer_engine \
  --bf16 \
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --lr 1.63e-3 \
  --min-lr 1.63e-4 \
  --lr-decay-style WSD \
  --lr-decay-iters "$target_iters" \
  --lr-wsd-decay-iters "$decay_iters" \
  --lr-wsd-decay-style exponential \
  --lr-warmup-iters "$warmup_iters" \
  --weight-decay 0.1 \
  --clip-grad 1 \
  --micro-batch-size "$micro_batch" \
  --global-batch-size "$global_batch" \
  --train-iters "$train_iters" \
  --tokenizer-type NullTokenizer --vocab-size 50257 \
  --null-tokenizer-eod-id 50256 --null-tokenizer-pad-id -1 \
  --train-data-path "$data_root/train" \
  --valid-data-path "$data_root/development" \
  --test-data-path "$data_root/final" \
  --dataloader-type single \
  --num-workers 2 \
  --no-create-attention-mask-in-dataloader \
  --seed 1234 \
  --eval-interval 250 \
  --eval-iters 32 \
  --log-interval 10 \
  --log-throughput \
  --timing-log-level 1 \
  "${logger_args[@]}" \
  --optimizer "$optimizer" \
  "${optimizer_args[@]}" \
  "${compute[@]}" \
  "${fusion_args[@]}" \
  "${speed_args[@]}" \
  --ckpt-format torch \
  "${save_args[@]}" \
  "${load_args[@]}" \
  >"$train_log" 2>&1
code=$?
set -e
echo "TRAIN_EXIT=$code"
tail -n 150 "$train_log"
exit 0
