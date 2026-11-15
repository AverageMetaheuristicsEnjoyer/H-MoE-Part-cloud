#!/usr/bin/env bash
# Resume AdamW FP8-GEMM at 13,794 and decay continuously through the time-matched endpoint.
set -u

mode=${1:-run}
root=$(cd "$(dirname "$0")/.." && pwd)
arm=adamw_fp8gemm_state_fp32
source_dir=${STAGE3_MOE_STRETCHED_SOURCE:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3-time-match-source/$arm}
dst_root=${STAGE3_MOE_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3-time-match-stretched-v1}
extension_root=${STAGE3_MOE_EXTENSION_ROOT:-/workspace-SR006.nfs2/hmoe-data/fineweb-edu-time-match-extension}
dst="$dst_root/time-match-stretched/$arm"
source_tracker="$source_dir/latest_checkpointed_iteration.txt"
dst_tracker="$dst/latest_checkpointed_iteration.txt"

for path in "$source_tracker" "$source_dir/iter_0013794" \
  "$extension_root/data/train.bin" "$extension_root/data/train.idx" \
  "$extension_root/artifact-manifest.json"; do
  [[ -e $path ]] || { echo "stretched-decay prerequisite missing: $path" >&2; exit 2; }
done
[[ $(cat "$source_tracker") == 13794 ]] || {
  echo "source tracker does not select iteration 13794: $source_tracker" >&2
  exit 2
}

echo "STRETCHED source=$source_dir destination=$dst data=$extension_root/data/train"
echo "STRETCHED_SCHEDULE phase_boundary=13794 sampler_offset_steps=0 decay_steps=5776 target=19570"
df -h "$source_dir" "$dst_root" "$extension_root" 2>/dev/null | grep -v '^Filesystem'
if [[ -f $dst_tracker ]]; then
  iteration=$(cat "$dst_tracker")
  echo "STRETCHED_DESTINATION iteration=$iteration"
  if [[ $iteration == 19570 ]]; then
    echo "ALREADY_COMPLETE iteration=$iteration"
    exit 0
  fi
  echo "incomplete model-only destination cannot be resumed: $dst" >&2
  exit 2
else
  echo "STRETCHED_DESTINATION empty"
fi

if [[ $mode == preflight ]]; then
  echo "EXIT=0"
  exit 0
fi
[[ $mode == run ]] || { echo "usage: cloud_moe_time_match_stretched_decay.sh [run|preflight]" >&2; exit 2; }
[[ -s /home/jovyan/.wandb-key || -n ${WANDB_API_KEY:-} ]] || {
  echo "online W&B credential is missing" >&2
  exit 2
}

export STAGE3_MOE_MICRO_BATCH=16
export STAGE3_MOE_PROPAGATE_EXIT=1
export STAGE3_MOE_CKPT_ROOT="$dst_root"
export STAGE3_MOE_TRAIN_DATA_PREFIX="$extension_root/data/train"
export STAGE3_MOE_DATA_MANIFEST_PATH="$extension_root/artifact-manifest.json"
export STAGE3_MOE_STRETCHED_LOAD="$source_dir"
export STAGE3_MOE_RUN_SUFFIX=${STAGE3_MOE_RUN_SUFFIX:-stretched-decay-v1}
export STAGE3_MOE_LOG_ROOT=${STAGE3_MOE_LOG_ROOT:-/workspace-SR006.nfs2/hmoe-cloud/pretrain}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb-radfan.ru}

"$root/scripts/run_stage3_moe_pretrain.sh" "$arm" time-match-stretched-decay
code=$?
echo "STRETCHED_EXIT=$code"
find "$dst" -maxdepth 1 -type d -name 'iter_*' -print | sort
exit "$code"
