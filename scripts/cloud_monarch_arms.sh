#!/bin/bash
# mlsub entry for the expert-bank calibration runs: Monarch-Muon n2 H-MoE with
# the routed experts as Monarch factors (the control), Monarch with a dense
# down projection, or low rank at the Monarch parameter count.
#
#   mlsub run --repo <public mirror> --branch monarch-moe/expert-gemm-bench --image torch28 --no-pip \
#     --entry scripts/cloud_monarch_arms.sh --gpus 1 --args="lowrank full 2000" \
#     --env MONARCH_CKPT_ROOT=/home/jovyan/monarch-pretrain
#
#   EXPERTS  monarch | monarch_dense_down | lowrank
#   MODE     bench (25 steps, no eval, no checkpoint) | smoke | full
#   EXIT     full only: stop after this iteration on the unchanged 1C schedule;
#            resubmitting the same job resumes from the last checkpoint
#
#   mlsub run ... --gpus cpu --args=peek      progress of every arm, from the volume
#
# Always exits zero, because a Failed mlsub job keeps no log; the real status is
# the ARM_EXIT line, and the full output stays in the per-run log on the volume.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
logs=
for volume in /workspace-SR006.nfs3 /workspace-SR006.nfs2 /home/jovyan; do
    if mkdir -p "$volume/monarch-pretrain-logs" 2>/dev/null; then
        logs="$volume/monarch-pretrain-logs"
        break
    fi
done

if [ "${1:-}" = "peek" ]; then
    for run in "$logs"/*/; do
        newest=$(ls -t "$run"rank-0-*.log 2>/dev/null | head -1)
        [ -n "$newest" ] || continue
        echo "=== $(basename "$run")"
        grep -h "validation loss at iteration\|lm loss validation" "$run"rank-0-*.log | tail -8
        grep "elapsed time per iteration" "$newest" | tail -1 | cut -c1-200
        grep -h "TRAIN_EXIT\|Traceback\|Error" "$newest" | tail -3
    done
    exit 0
fi

experts=${1:?usage: cloud_monarch_arms.sh EXPERTS MODE [EXIT] | peek}
mode=${2:?usage: cloud_monarch_arms.sh EXPERTS MODE [EXIT] | peek}
export MONARCH_EXPERTS=$experts
export MONARCH_LOG_ROOT=${MONARCH_LOG_ROOT:-$logs}
# the node207 Monarch runs used micro-batch 16 (global batch 208 either way)
export MONARCH_MICRO_BATCH=${MONARCH_MICRO_BATCH:-16}
[ -n "${3:-}" ] && export MONARCH_EXIT_INTERVAL=$3
if [ "$mode" = full ]; then
    # The volumes hold ~45 GB between them and a checkpoint is 8-9 GB, so save
    # only on exit (the retain interval is the save interval), and put each arm
    # where it fits with --env MONARCH_CKPT_ROOT=... . No W&B key lives on the
    # cluster; the rank log carries every train and validation loss.
    export MONARCH_SAVE_INTERVAL=${MONARCH_SAVE_INTERVAL:-13794}
    export MONARCH_MIN_FREE_GB=${MONARCH_MIN_FREE_GB:-12}
    export MONARCH_ALLOW_NO_WANDB=${MONARCH_ALLOW_NO_WANDB:-1}
fi

"$root/scripts/cloud_monarch_pretrain.sh" hmoe muon 2 ddp "$mode" >"$logs/arm-$experts-$mode.out" 2>&1
code=$?
grep -E "MONARCH_TRAIN_CONFIG|MONARCH_MODEL_CHECK|HMOE_MONARCH|MONARCH_DATA_MANIFEST|validation loss at iteration|TRAIN_EXIT" \
    "$logs/arm-$experts-$mode.out" | cut -c1-400
tail -40 "$logs/arm-$experts-$mode.out"
echo "ARM_EXIT=$code experts=$experts mode=$mode"
exit 0
