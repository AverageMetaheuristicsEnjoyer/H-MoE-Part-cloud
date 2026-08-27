#!/usr/bin/env bash
# Resume AdamW FP8-GEMM at 13,794 on extension data and start WSD decay immediately.
set -u

mode=${1:-run}
root=$(cd "$(dirname "$0")/.." && pwd)
arm=adamw_fp8gemm_state_fp32
source_dir=${STAGE3_MOE_EXTENSION_DECAY_SOURCE:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3-time-match-source/$arm}
dst_root=${STAGE3_MOE_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3-extension-decay-control}
extension_root=${STAGE3_MOE_EXTENSION_ROOT:-/workspace-SR006.nfs2/hmoe-data/fineweb-edu-time-match-extension}
dst="$dst_root/extension-decay-control/$arm"
source_tracker="$source_dir/latest_checkpointed_iteration.txt"
dst_tracker="$dst/latest_checkpointed_iteration.txt"

for path in "$source_tracker" "$source_dir/iter_0013794" \
  "$extension_root/data/train.bin" "$extension_root/data/train.idx" \
  "$extension_root/artifact-manifest.json"; do
  [[ -e $path ]] || { echo "control prerequisite missing: $path" >&2; exit 2; }
done
[[ $(cat "$source_tracker") == 13794 ]] || {
  echo "source tracker does not select iteration 13794: $source_tracker" >&2
  exit 2
}

echo "CONTROL source=$source_dir destination=$dst data=$extension_root/data/train"
df -h "$source_dir" "$dst_root" "$extension_root" 2>/dev/null | grep -v '^Filesystem'
if [[ -f $dst_tracker ]]; then
  iteration=$(cat "$dst_tracker")
  echo "CONTROL_DESTINATION iteration=$iteration"
  [[ $iteration -ge 13794 && $iteration -le 17242 ]] || exit 2
  load=$dst
else
  echo "CONTROL_DESTINATION empty"
  load=$source_dir
fi

if [[ $mode == preflight ]]; then
  echo "EXIT=0"
  exit 0
fi
[[ $mode == run ]] || { echo "usage: cloud_moe_extension_decay_control.sh [run|preflight]" >&2; exit 2; }
[[ -s /home/jovyan/.wandb-key || -n ${WANDB_API_KEY:-} ]] || {
  echo "online W&B credential is missing" >&2
  exit 2
}

export STAGE3_MOE_MICRO_BATCH=16
export STAGE3_MOE_PROPAGATE_EXIT=1
export STAGE3_MOE_CKPT_ROOT="$dst_root"
export STAGE3_MOE_TRAIN_DATA_PREFIX="$extension_root/data/train"
export STAGE3_MOE_DATA_MANIFEST_PATH="$extension_root/artifact-manifest.json"
export STAGE3_MOE_EXTENSION_DECAY_LOAD="$load"
export STAGE3_MOE_RUN_SUFFIX=${STAGE3_MOE_RUN_SUFFIX:-extension-decay-v1}
export STAGE3_MOE_LOG_ROOT=${STAGE3_MOE_LOG_ROOT:-/workspace-SR006.nfs2/hmoe-cloud/pretrain}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb-radfan.ru}

"$root/scripts/run_stage3_moe_pretrain.sh" "$arm" extension-decay-control
code=$?
echo "CONTROL_EXIT=$code"
find "$dst" -maxdepth 1 -type d -name 'iter_*' -print | sort
exit "$code"
