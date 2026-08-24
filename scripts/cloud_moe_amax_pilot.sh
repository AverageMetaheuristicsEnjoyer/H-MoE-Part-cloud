#!/usr/bin/env bash
# Run a bounded trajectory comparison on one GPU without writing checkpoints.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
optimizer=${1:-adamw}
if (($#)); then
  shift
fi
case $optimizer in
  adamw|muon) ;;
  *) echo "optimizer must be adamw or muon" >&2; exit 2 ;;
esac
if (($#)); then
  variants=("$@")
else
  variants=(h16max current bf16)
fi
tag=${STAGE3_MOE_RUN_SUFFIX:-amax-pilot-$optimizer-20260824}
export STAGE3_MOE_BENCH_ITERS=${STAGE3_MOE_AMAX_PILOT_ITERS:-300}
export STAGE3_MOE_MICRO_BATCH=16
export STAGE3_MOE_LOG_ROOT=${STAGE3_MOE_LOG_ROOT:-/workspace-SR006.nfs2/hmoe-cloud/pretrain}
export STAGE3_MOE_CKPT_ROOT="/tmp/stage3-amax-pilot-$$"
export STAGE3_MOE_DATA_CACHE_PATH="/tmp/stage3-amax-pilot-cache-$$"

for variant in "${variants[@]}"; do
  case $variant in
    h16max|current|bf16) ;;
    *) echo "variant must be h16max, current, or bf16" >&2; exit 2 ;;
  esac
  unset STAGE3_MOE_FP8_AMAX_HISTORY_LEN STAGE3_MOE_FP8_AMAX_COMPUTE_ALGO
  arm=${optimizer}_fp8gemm_state_fp32
  if [[ $variant == h16max ]]; then
    export STAGE3_MOE_FP8_AMAX_HISTORY_LEN=16
    export STAGE3_MOE_FP8_AMAX_COMPUTE_ALGO=max
  elif [[ $variant == bf16 ]]; then
    arm=${optimizer}_bf16_state_fp32
  fi
  export STAGE3_MOE_RUN_SUFFIX="$tag-$variant"
  echo "=== VARIANT $variant arm=$arm iters=$STAGE3_MOE_BENCH_ITERS ==="
  "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" bench
  code=$?
  echo "VARIANT_EXIT=$code variant=$variant arm=$arm"
  run_dir="$STAGE3_MOE_LOG_ROOT/stage3-$arm-bench-$STAGE3_MOE_RUN_SUFFIX"
  newest=$(ls -1t "$run_dir"/train-*.log 2>/dev/null | head -1)
  if [[ -n $newest ]]; then
    grep -aE "iteration +[0-9]+/|validation loss|lm loss|nan iterations|Traceback|Error" "$newest" | tail -40
  fi
done

rm -rf "$STAGE3_MOE_CKPT_ROOT"
rm -rf "$STAGE3_MOE_DATA_CACHE_PATH"
echo "EXIT=0"
exit 0
