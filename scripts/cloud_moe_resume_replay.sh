#!/usr/bin/env bash
# Replay steps 13,795-13,800 on the original data and schedule without saving.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
arm=adamw_fp8gemm_state_fp32
source_dir=${STAGE3_MOE_REPLAY_SOURCE:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3-time-match-source/$arm}
stage_dir=/tmp/stage3-resume-replay
source="$source_dir/iter_0013794"

[[ -d $source ]] || { echo "replay source missing: $source" >&2; exit 2; }
rm -rf "$stage_dir"
mkdir -p "$stage_dir"
ln -s "$source" "$stage_dir/iter_0013794"
echo 13794 > "$stage_dir/latest_checkpointed_iteration.txt"
trap 'rm -rf "$stage_dir"' EXIT

export STAGE3_MOE_MICRO_BATCH=16
export STAGE3_MOE_PROPAGATE_EXIT=1
export STAGE3_MOE_REPLAY_LOAD="$stage_dir"
export STAGE3_MOE_RUN_SUFFIX=${STAGE3_MOE_RUN_SUFFIX:-resume-replay-v1}
export STAGE3_MOE_LOG_INTERVAL=1
export STAGE3_MOE_LOG_ROOT=${STAGE3_MOE_LOG_ROOT:-/workspace-SR006.nfs2/hmoe-cloud/pretrain}
export WANDB_MODE=disabled

"$root/scripts/run_stage3_moe_pretrain.sh" "$arm" resume-replay
code=$?
echo "REPLAY_EXIT=$code"
exit "$code"
