#!/usr/bin/env bash
# Acceptance test for the checkpoint-save memory regression.
#
#   mlsub run ... --entry scripts/cloud_moe_memaudit.sh --gpus 1 --image torch28
#
# Reproduces the smoke configuration that measured 1.0969 FAIL (mb=4, chunk=0,
# 1 GPU) and reports the peak it reaches now; a fixed run peaks where a bench run
# does, at 21,117,359,104 B. Pass --env STAGE3_MOE_MEM_AUDIT=1 to also bracket
# every save with the live-FP8 census. The launcher hides the run behind its own
# log, so the MEMAUDIT lines are pulled back out at the end. The checkpoints are
# throwaway and are removed on both sides of the run.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
ckpt_root=${STAGE3_MOE_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3}
log_root=${STAGE3_MOE_LOG_ROOT:-/home/jovyan/hmoe-cloud/pretrain}
arm=muon_bf16_state_fp8

export STAGE3_MOE_RUN_SUFFIX=memaudit
export STAGE3_MOE_MEM_AUDIT=${STAGE3_MOE_MEM_AUDIT:-0}
export STAGE3_MOE_MICRO_BATCH=${STAGE3_MOE_MICRO_BATCH:-4}
export STAGE3_MOE_FP8_DEQUANT_CHUNK=${STAGE3_MOE_FP8_DEQUANT_CHUNK:-0}

smoke_dir="$ckpt_root/smoke/$arm-$STAGE3_MOE_RUN_SUFFIX"
run_dir="$log_root/stage3-$arm-smoke-$STAGE3_MOE_RUN_SUFFIX"

echo "=== volume before ==="
df -h /workspace-SR006.nfs3 | tail -1
df -h /home/jovyan | tail -1
ls -ld /home/user/conda/lib/python3.12/site-packages/nvidia 2>&1
rm -rf "$smoke_dir"

# The launcher runs under `set -euo pipefail` and silences several commands, so a
# failure before its first echo says nothing at all. Keep a trace for that case.
trace=/tmp/launcher-trace.log
bash -x "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" smoke 2>"$trace"
code=$?
echo "ARM_EXIT=$code"
if [[ $code -ne 0 ]]; then
  echo "=== launcher trace (tail) ==="
  tail -n 60 "$trace"
fi

echo "=== MEMAUDIT ==="
newest=$(ls -1t "$run_dir"/train-*.log 2>/dev/null | head -1)
if [[ -n $newest ]]; then
  grep -aE "MEMAUDIT|saving checkpoint at iteration|successfully saved checkpoint" "$newest" || echo "(no MEMAUDIT lines)"
else
  echo "(no train log under $run_dir)"
fi

echo "=== RESULT ==="
cat "$run_dir/results.jsonl" 2>/dev/null || echo "(no results.jsonl)"

rm -rf "$smoke_dir"
echo "=== volume after ==="; df -h /workspace-SR006.nfs3 | tail -1
echo "EXIT=0"
exit 0
