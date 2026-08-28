#!/usr/bin/env bash
# Continue AdamW FP8-GEMM from 13,794 to 16,122 at peak LR on the original data sequence.
set -u

mode=${1:-run}
root=$(cd "$(dirname "$0")/.." && pwd)
arm=adamw_fp8gemm_state_fp32
source_dir=${STAGE3_MOE_ORIGINAL_PLATEAU_SOURCE:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3-time-match-source/$arm}
dst_root=${STAGE3_MOE_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3-original-data-plateau-control}
data_root=${STAGE3_MOE_ORIGINAL_DATA_ROOT:-/home/jovyan/data/fineweb-edu-gpt2-megatron}
dst="$dst_root/original-data-plateau-control/$arm"
source_tracker="$source_dir/latest_checkpointed_iteration.txt"
dst_tracker="$dst/latest_checkpointed_iteration.txt"
plateau_end=16122

for path in "$source_tracker" "$source_dir/iter_0013794" \
  "$data_root/data/train.bin" "$data_root/data/train.idx" \
  "$data_root/artifact-manifest.json"; do
  [[ -e $path ]] || { echo "original-data control prerequisite missing: $path" >&2; exit 2; }
done
[[ $(cat "$source_tracker") == 13794 ]] || {
  echo "source tracker does not select iteration 13794: $source_tracker" >&2
  exit 2
}

echo "ORIGINAL_DATA_CONTROL source=$source_dir destination=$dst data=$data_root/data/train"
echo "CONTROL_MATCH start=13794 end=$plateau_end plateau_steps=2328 lr=0.00163 decay_start=16123"
sha256sum "$data_root/artifact-manifest.json"
stat -c 'DATA_FILE size=%s inode=%i path=%n' "$data_root/data/train.bin" "$data_root/data/train.idx"
df -h /workspace-SR006.nfs2 /workspace-SR006.nfs3 /home/jovyan 2>/dev/null | grep -v '^Filesystem'
df -i /workspace-SR006.nfs2 /workspace-SR006.nfs3 /home/jovyan 2>/dev/null | grep -v '^Filesystem'
if [[ -f $dst_tracker ]]; then
  iteration=$(cat "$dst_tracker")
  echo "CONTROL_DESTINATION iteration=$iteration"
  [[ $iteration -ge 13794 && $iteration -le $plateau_end ]] || exit 2
  if [[ $iteration == "$plateau_end" ]]; then
    echo "ALREADY_COMPLETE iteration=$iteration"
    exit 0
  fi
  load=$dst
else
  echo "CONTROL_DESTINATION empty"
  load=$source_dir
fi

if [[ $mode == preflight ]]; then
  echo "EXIT=0"
  exit 0
fi
[[ $mode == run ]] || { echo "usage: cloud_moe_original_data_plateau_control.sh [run|preflight]" >&2; exit 2; }
[[ -s /home/jovyan/.wandb-key || -n ${WANDB_API_KEY:-} ]] || {
  echo "online W&B credential is missing" >&2
  exit 2
}

export STAGE3_MOE_MICRO_BATCH=16
export STAGE3_MOE_PROPAGATE_EXIT=1
export STAGE3_MOE_CKPT_ROOT="$dst_root"
export STAGE3_MOE_TRAIN_DATA_PREFIX="$data_root/data/train"
export STAGE3_MOE_DATA_MANIFEST_PATH="$data_root/artifact-manifest.json"
export STAGE3_MOE_ORIGINAL_PLATEAU_LOAD="$load"
export STAGE3_MOE_RUN_SUFFIX=${STAGE3_MOE_RUN_SUFFIX:-v1}
export STAGE3_MOE_LOG_ROOT=${STAGE3_MOE_LOG_ROOT:-/workspace-SR006.nfs2/hmoe-cloud/pretrain}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb-radfan.ru}

"$root/scripts/run_stage3_moe_pretrain.sh" "$arm" original-data-plateau-control
code=$?
echo "CONTROL_EXIT=$code"
find "$dst" -maxdepth 1 -type d -name 'iter_*' -print | sort
exit "$code"
