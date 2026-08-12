#!/usr/bin/env bash
# Stage 3 MoE matched pretraining, WSD trunk-and-branch.
#
#   run_stage3_moe_pretrain.sh ARM trunk|decay-1p2b
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
    smoke_dir="$ckpt_root/smoke/$arm"
    save_args=(--save "$smoke_dir" --save-interval 10)
    load_args=(--load "$smoke_dir")
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

run_id="stage3-$arm-$mode"
mkdir -p "$log_root/$run_id" "$trunk_dir"
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
# Without credentials wandb.init() would abort the run, so fall back to offline
# logging: the run still records everything and can be `wandb sync`ed later.
if [[ -f /home/jovyan/.wandb-key ]]; then
  export WANDB_API_KEY=$(cat /home/jovyan/.wandb-key)
else
  export WANDB_MODE=offline
  echo "WANDB=offline (no /home/jovyan/.wandb-key); sync later with 'wandb sync'"
fi

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "ARM=$arm MODE=$mode GPUS=$gpu_count micro_batch=$micro_batch global_batch=$global_batch"
echo "SCHEDULE target_iters=$target_iters decay_iters=$decay_iters warmup=$warmup_iters train_iters=$train_iters"
echo "CKPT save=${save_args[*]} load=${load_args[*]}"

exec python -m torch.distributed.run --standalone --nproc-per-node "$gpu_count" \
  stage3_moe/pretrain_gpt.py \
  --stage3-arm "$arm" \
  --stage3-result-path "$log_root/$run_id/results.jsonl" \
  --stage3-warmup-steps 20 \
  --stage3-measure-steps 100 \
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
  --tensorboard-dir "$log_root/$run_id/tensorboard" \
  --tensorboard-log-interval 10 \
  --wandb-project "${STAGE3_MOE_WANDB_PROJECT:-hmoe-stage3}" \
  --wandb-exp-name "$run_id" \
  --wandb-save-dir "$log_root/$run_id" \
  --optimizer "$optimizer" \
  "${optimizer_args[@]}" \
  "${compute[@]}"
