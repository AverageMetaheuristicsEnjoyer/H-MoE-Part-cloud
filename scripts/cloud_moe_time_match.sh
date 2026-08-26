#!/usr/bin/env bash
# Resume one FP8-GEMM arm from the retained pre-decay checkpoint onto the
# independent extension data phase. Re-running resumes the isolated destination.
set -u

arm=${1:?usage: cloud_moe_time_match.sh ARM}
case "$arm" in
  adamw_fp8gemm_state_fp32) target=19570 ;;
  muon_fp8gemm_state_fp32) target=22208 ;;
  *) echo "unsupported time-match arm: $arm" >&2; exit 2 ;;
esac

root=$(cd "$(dirname "$0")/.." && pwd)
src_root=${STAGE3_MOE_SRC_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3}
dst_root=${STAGE3_MOE_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3-time-match}
extension_root=${STAGE3_MOE_EXTENSION_ROOT:-/workspace-SR006.nfs2/hmoe-data/fineweb-edu-time-match-extension}
src="$src_root/trunk/$arm/iter_0013794"
dst="$dst_root/time-match/$arm"
tracker="$dst/latest_checkpointed_iteration.txt"

for path in "$extension_root/data/train.bin" "$extension_root/data/train.idx" "$extension_root/artifact-manifest.json"; do
  [[ -f $path ]] || { echo "extension data is incomplete: $path" >&2; exit 2; }
done
[[ -d $src ]] || { echo "pre-decay checkpoint is missing: $src" >&2; exit 2; }
[[ -s /home/jovyan/.wandb-key || -n ${WANDB_API_KEY:-} ]] || {
  echo "online W&B credential is missing" >&2
  exit 2
}

if [[ -f $tracker ]]; then
  iteration=$(cat "$tracker")
  echo "RESUME arm=$arm iteration=$iteration destination=$dst"
  if [[ $iteration == "$target" ]]; then
    echo "ALREADY_COMPLETE arm=$arm iteration=$iteration"
    exit 0
  fi
else
  mkdir -p "$dst"
  if [[ $(stat -c %d "$src") == $(stat -c %d "$dst") ]]; then
    cp -al "$src" "$dst/" || exit 2
  else
    cp -a "$src" "$dst/.iter_0013794.partial" || exit 2
    mv "$dst/.iter_0013794.partial" "$dst/iter_0013794" || exit 2
  fi
  echo 13794 > "$tracker"
  echo "SEEDED arm=$arm source=$src destination=$dst"
fi

export STAGE3_MOE_MICRO_BATCH=16
export STAGE3_MOE_PROPAGATE_EXIT=1
export STAGE3_MOE_TRAIN_DATA_PREFIX="$extension_root/data/train"
export STAGE3_MOE_DATA_MANIFEST_PATH="$extension_root/artifact-manifest.json"
export STAGE3_MOE_RUN_SUFFIX=${STAGE3_MOE_RUN_SUFFIX:-wallclock-v1}
export STAGE3_MOE_LOG_ROOT=${STAGE3_MOE_LOG_ROOT:-/workspace-SR006.nfs2/hmoe-cloud/pretrain}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb-radfan.ru}

"$root/scripts/run_stage3_moe_pretrain.sh" "$arm" time-match
code=$?
echo "ARM_EXIT=$code arm=$arm"
find "$dst" -maxdepth 1 -type d -name 'iter_*' -print | sort
df -h "$dst_root" "$extension_root" 2>/dev/null | grep -v '^Filesystem'
exit "$code"
