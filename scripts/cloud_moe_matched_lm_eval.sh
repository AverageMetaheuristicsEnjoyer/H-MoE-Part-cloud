#!/usr/bin/env bash
# Evaluate the three AdamW endpoints twice on identical validation/test windows.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
log_root=${STAGE3_MOE_LOG_ROOT:-/workspace-SR006.nfs2/hmoe-cloud/pretrain}
tag=${STAGE3_MOE_MATCHED_EVAL_TAG:-matched-lm-v1}
repeats=${STAGE3_MOE_MATCHED_EVAL_REPEATS:-2}
stage_dir=/tmp/stage3-matched-lm-eval

bf16=${STAGE3_MOE_MATCHED_BF16:-/workspace-SR006.nfs2/hmoe-checkpoints/stage3-1c-mb4/1c/adamw_bf16_state_fp32}
original=${STAGE3_MOE_MATCHED_ORIGINAL:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3/trunk/adamw_fp8gemm_state_fp32}
time_match=${STAGE3_MOE_MATCHED_TIME_MATCH:-/home/jovyan/hmoe-checkpoints/stage3-time-match/time-match/adamw_fp8gemm_state_fp32}
control=${STAGE3_MOE_MATCHED_CONTROL:-}
skip_references=${STAGE3_MOE_MATCHED_SKIP_REFERENCES:-0}

nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
echo "MATCHED_LM_EVAL tag=$tag repeats=$repeats micro_batch=16 eval_iters=32"

eval_one() {
  label=$1
  arm=$2
  source=$3
  expected=$4
  tracker=$source/latest_checkpointed_iteration.txt
  [[ -f $tracker ]] || { echo "MISSING tracker=$tracker"; return 2; }
  iteration=$(cat "$tracker")
  [[ $iteration == "$expected" ]] || {
    echo "WRONG_ITERATION label=$label expected=$expected actual=$iteration"
    return 2
  }
  endpoint=$(printf '%s/iter_%07d' "$source" "$iteration")
  [[ -d $endpoint ]] || { echo "MISSING endpoint=$endpoint"; return 2; }
  echo "SOURCE label=$label arm=$arm iteration=$iteration endpoint=$endpoint"

  rm -rf "$stage_dir"
  ln -s "$source" "$stage_dir"
  repeat=1
  while [[ $repeat -le $repeats ]]; do
    suffix="$tag-$label-r$repeat"
    echo "=== EVAL label=$label repeat=$repeat suffix=$suffix ==="
    STAGE3_MOE_PROPAGATE_EXIT=1 \
    STAGE3_MOE_EVAL_LOAD="$stage_dir" \
    STAGE3_MOE_RUN_SUFFIX="$suffix" \
    STAGE3_MOE_LOG_ROOT="$log_root" \
    STAGE3_MOE_MICRO_BATCH=16 \
    WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb-radfan.ru} \
      "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" eval-lm-fixed
    code=$?
    echo "EVAL_EXIT=$code label=$label repeat=$repeat"
    [[ $code == 0 ]] || return "$code"

    run_dir="$log_root/stage3-$arm-eval-lm-fixed-$suffix"
    newest=$(ls -1t "$run_dir"/train-*.log 2>/dev/null | head -1)
    [[ -n $newest ]] || { echo "MISSING_LOG label=$label repeat=$repeat"; return 2; }
    grep -aE "loss at iteration .* on (validation|test) set" "$newest" \
      | sed "s/^/MATCHED_RESULT label=$label repeat=$repeat /"
    repeat=$((repeat + 1))
  done
}

if [[ $skip_references != 1 ]]; then
  eval_one bf16 adamw_bf16_state_fp32 "$bf16" 17242 || exit $?
  eval_one original adamw_fp8gemm_state_fp32 "$original" 17242 || exit $?
  eval_one time_match adamw_fp8gemm_state_fp32 "$time_match" 19570 || exit $?
fi
if [[ -n $control ]]; then
  eval_one extension_decay adamw_fp8gemm_state_fp32 "$control" 17242 || exit $?
fi

echo "EXIT=0"
