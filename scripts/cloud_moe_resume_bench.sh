#!/usr/bin/env bash
# Steady-state timing for the compute axis: resume the trunk branch point in each arm
# and time a short window.  Usage:
#   mlsub run ... --entry scripts/cloud_moe_resume_bench.sh --args "ARM [ARM ...]"
#
# Both arms of a comparison run in ONE job, sequentially, so their ratio is measured on
# one physical GPU: the same arm at the same config differs 8-16 % between jobs
# ([[stage3-wct-window-contaminated]]), which would swamp the effect being measured.
# Always exits 0 so the platform keeps the logs; real status is in ARM_EXIT.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
log_root=${STAGE3_MOE_LOG_ROOT:-/home/jovyan/hmoe-cloud/pretrain}

nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
echo "BENCH micro_batch=${STAGE3_MOE_MICRO_BATCH:-4} suffix=${STAGE3_MOE_RUN_SUFFIX:-none} warmup=${STAGE3_MOE_RESUME_WARMUP:-default} measured=${STAGE3_MOE_BENCH_ITERS:-150}"

for arm in "$@"; do
  if [[ ! $arm =~ ^[a-z0-9_]+$ ]]; then
    echo "SKIP invalid arm name: $arm"
    continue
  fi
  echo "=== ARM $arm ==="
  "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" resume-bench
  echo "ARM_EXIT=$? arm=$arm"
  run_dir="$log_root/stage3-$arm-resume-bench${STAGE3_MOE_RUN_SUFFIX:+-$STAGE3_MOE_RUN_SUFFIX}"
  newest=$(ls -1t "$run_dir"/train-*.log 2>/dev/null | head -1)
  if [[ -n $newest ]]; then
    echo "--- LOAD $arm"
    grep -iE "loading checkpoint|will not load|at iteration|does not match the optimizer" "$newest" | head -4
    echo "--- STEPS $arm"
    grep -E "iteration +[0-9]+/" "$newest" |
      sed -E 's/.*iteration +([0-9]+)\/ *([0-9]+).*elapsed time per iteration \(ms\): ([0-9.]+).*/iter=\1\/\2 ms=\3/'
  fi
  [[ -f "$run_dir/results.jsonl" ]] && { echo "--- RESULT_JSONL $arm"; tail -1 "$run_dir/results.jsonl"; }
  shadow="$run_dir/muon-fp8-shadow.jsonl"
  if [[ -s $shadow ]]; then
    echo "--- MUON_FP8_SHADOW rows=$(wc -l < "$shadow") sha256=$(sha256sum "$shadow" | awk '{print $1}')"
    head -3 "$shadow"
    tail -3 "$shadow"
  fi
done
echo "EXIT=0"
exit 0
