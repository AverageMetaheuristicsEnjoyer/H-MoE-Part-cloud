#!/usr/bin/env bash
# Preserve the original 1C data sequence through 17,242, then use unseen extension data.
set -euo pipefail

mode=${1:-run}
root=$(cd "$(dirname "$0")/.." && pwd)
arm=adamw_fp8gemm_state_fp32
source_dir=${STAGE3_MOE_CORRECTED_SOURCE:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3-time-match-source/$arm}
dst_root=${STAGE3_MOE_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3-corrected-time-match-v1}
base_root=${STAGE3_MOE_ORIGINAL_DATA_ROOT:-/home/jovyan/data/fineweb-edu-gpt2-megatron}
extension_root=${STAGE3_MOE_EXTENSION_ROOT:-/workspace-SR006.nfs2/hmoe-data/fineweb-edu-time-match-extension}
dst="$dst_root/corrected-time-match/$arm"

for path in "$source_dir/latest_checkpointed_iteration.txt" "$source_dir/iter_0013794" \
  "$base_root/data/train.bin" "$base_root/data/train.idx" "$base_root/artifact-manifest.json" \
  "$extension_root/data/train.bin" "$extension_root/data/train.idx" \
  "$extension_root/artifact-manifest.json"; do
  [[ -e $path ]] || { echo "corrected time-match prerequisite missing: $path" >&2; exit 2; }
done
[[ $(cat "$source_dir/latest_checkpointed_iteration.txt") == 13794 ]] || {
  echo "source tracker does not select iteration 13794" >&2
  exit 2
}
if [[ -f $dst/latest_checkpointed_iteration.txt ]]; then
  iteration=$(cat "$dst/latest_checkpointed_iteration.txt")
  [[ $iteration == 19570 ]] || { echo "unexpected destination iteration: $iteration" >&2; exit 2; }
  echo "ALREADY_COMPLETE iteration=$iteration"
  exit 0
fi

echo "CORRECTED_TIME_MATCH source=$source_dir destination=$dst"
echo "DATA_PHASE base_start=13794 base_end=17242 extension_start=17242 target=19570"
echo "SCHEDULE plateau_end=16122 decay_start=16123 decay_steps=3448"
sha256sum "$base_root/artifact-manifest.json" "$extension_root/artifact-manifest.json"
df -h /tmp /workspace-SR006.nfs2 /workspace-SR006.nfs3 /home/jovyan 2>/dev/null | grep -v '^Filesystem'
tmp_available_kb=$(df -Pk /tmp | awk 'END {print $4}')
[[ $tmp_available_kb -ge 20000000 ]] || {
  echo "temporary checkpoint needs at least 20,000,000 KiB free: available=$tmp_available_kb" >&2
  exit 2
}
dst_available_kb=$(df -Pk /workspace-SR006.nfs3 | awk 'END {print $4}')
[[ $dst_available_kb -ge 3000000 ]] || {
  echo "final checkpoint needs at least 3,000,000 KiB free: available=$dst_available_kb" >&2
  exit 2
}

if [[ $mode == preflight ]]; then
  echo "EXIT=0"
  exit 0
fi
[[ $mode == run ]] || { echo "usage: cloud_moe_corrected_time_match.sh [run|preflight]" >&2; exit 2; }
[[ -s /home/jovyan/.wandb-key || -n ${WANDB_API_KEY:-} ]] || {
  echo "online W&B credential is missing" >&2
  exit 2
}

transition_root=$(mktemp -d /tmp/stage3-corrected-time-match.XXXXXX)
trap 'rm -rf "$transition_root"' EXIT
transition_dir="$transition_root/$arm"

export STAGE3_MOE_MICRO_BATCH=16
export STAGE3_MOE_PROPAGATE_EXIT=1
export STAGE3_MOE_CKPT_ROOT="$dst_root"
export STAGE3_MOE_CORRECTED_SOURCE="$source_dir"
export STAGE3_MOE_CORRECTED_TRANSITION_DIR="$transition_dir"
export STAGE3_MOE_RUN_SUFFIX=${STAGE3_MOE_RUN_SUFFIX:-data-continuity-v1}
export STAGE3_MOE_LOG_ROOT=${STAGE3_MOE_LOG_ROOT:-/workspace-SR006.nfs3/hmoe-cloud/pretrain}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb-radfan.ru}
export WANDB_RUN_ID="stage3-$arm-corrected-time-match-$STAGE3_MOE_RUN_SUFFIX"
export WANDB_RESUME=allow

export STAGE3_MOE_CORRECTED_PHASE=base
export STAGE3_MOE_TRAIN_DATA_PREFIX="$base_root/data/train"
export STAGE3_MOE_DATA_MANIFEST_PATH="$base_root/artifact-manifest.json"
"$root/scripts/run_stage3_moe_pretrain.sh" "$arm" corrected-time-match
[[ -f $transition_dir/latest_checkpointed_iteration.txt ]] || {
  echo "transition checkpoint tracker is missing" >&2
  exit 2
}
[[ $(cat "$transition_dir/latest_checkpointed_iteration.txt") == 17242 ]] || {
  echo "transition checkpoint is not iteration 17242" >&2
  exit 2
}

export STAGE3_MOE_CORRECTED_PHASE=tail
export STAGE3_MOE_TRAIN_DATA_PREFIX="$extension_root/data/train"
export STAGE3_MOE_DATA_MANIFEST_PATH="$extension_root/artifact-manifest.json"
"$root/scripts/run_stage3_moe_pretrain.sh" "$arm" corrected-time-match

[[ -f $dst/latest_checkpointed_iteration.txt ]] || {
  echo "final checkpoint tracker is missing" >&2
  exit 2
}
[[ $(cat "$dst/latest_checkpointed_iteration.txt") == 19570 ]] || {
  echo "final checkpoint is not iteration 19570" >&2
  exit 2
}
echo "CORRECTED_TIME_MATCH_EXIT=0"
