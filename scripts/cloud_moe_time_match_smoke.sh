#!/usr/bin/env bash
# One real post-transition optimizer step from the retained pre-decay checkpoint.
set -u

arm=${1:-adamw_fp8gemm_state_fp32}
case "$arm" in
  adamw_fp8gemm_state_fp32|muon_fp8gemm_state_fp32) ;;
  *) echo "unsupported time-match arm: $arm" >&2; exit 2 ;;
esac

root=$(cd "$(dirname "$0")/.." && pwd)
src_root=${STAGE3_MOE_SRC_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3}
smoke_root=${STAGE3_MOE_SMOKE_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3-time-match-smoke}
extension_root=${STAGE3_MOE_EXTENSION_ROOT:-/workspace-SR006.nfs2/hmoe-data/fineweb-edu-time-match-extension}
source="$src_root/trunk/$arm/iter_0013794"
source_dir="$smoke_root/$arm"

for path in "$source" "$extension_root/data/train.bin" "$extension_root/data/train.idx" "$extension_root/artifact-manifest.json"; do
  [[ -e $path ]] || { echo "smoke prerequisite missing: $path" >&2; exit 2; }
done
mkdir -p "$source_dir"
if [[ ! -d $source_dir/iter_0013794 ]]; then
  cp -al "$source" "$source_dir/" || exit 2
fi
echo 13794 > "$source_dir/latest_checkpointed_iteration.txt"

export STAGE3_MOE_MICRO_BATCH=16
export STAGE3_MOE_PROPAGATE_EXIT=1
export STAGE3_MOE_TRAIN_DATA_PREFIX="$extension_root/data/train"
export STAGE3_MOE_DATA_MANIFEST_PATH="$extension_root/artifact-manifest.json"
export STAGE3_MOE_TIME_MATCH_SOURCE="$source_dir"
export STAGE3_MOE_RUN_SUFFIX=${STAGE3_MOE_RUN_SUFFIX:-wallclock-smoke}
export STAGE3_MOE_LOG_ROOT=${STAGE3_MOE_LOG_ROOT:-/workspace-SR006.nfs2/hmoe-cloud/pretrain}
export WANDB_MODE=disabled

"$root/scripts/run_stage3_moe_pretrain.sh" "$arm" time-match-smoke
code=$?
echo "SMOKE_EXIT=$code arm=$arm"
rm -rf "$source_dir"
exit "$code"
