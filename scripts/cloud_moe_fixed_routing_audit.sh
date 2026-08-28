#!/usr/bin/env bash
# Read-only routing comparison on identical extension and base-corpus batches.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
mode=${1:-full}
case "$mode" in
  smoke) eval_iters=1 ;;
  full) eval_iters=16 ;;
  *) echo "usage: cloud_moe_fixed_routing_audit.sh [smoke|full]" >&2; exit 2 ;;
esac
arm=adamw_fp8gemm_state_fp32
extension_root=${STAGE3_MOE_EXTENSION_ROOT:-/workspace-SR006.nfs2/hmoe-data/fineweb-edu-time-match-extension}
output_root=${STAGE3_MOE_ROUTING_OUTPUT_ROOT:-/workspace-SR006.nfs2/hmoe-cloud/routing-audit/fixed-extension-v1}
log_root=${STAGE3_MOE_LOG_ROOT:-/workspace-SR006.nfs2/hmoe-cloud/pretrain}
data_cache=$output_root/data-cache
stage_dir=/tmp/stage3-fixed-routing-checkpoint
base_data=/home/jovyan/data/fineweb-edu-gpt2-megatron/data
trap 'rm -f "$stage_dir"' EXIT

for path in "$extension_root/data/train.bin" "$extension_root/data/train.idx" \
  "$extension_root/artifact-manifest.json" "$base_data/final.bin" "$base_data/final.idx"; do
  [[ -f $path ]] || { echo "routing audit prerequisite missing: $path" >&2; exit 2; }
done

mkdir -p "$output_root" "$data_cache"
stat -c 'DATA_FILE size=%s inode=%i path=%n' \
  "$extension_root/data/train.bin" "$extension_root/data/train.idx" \
  "$base_data/final.bin" "$base_data/final.idx"
sha256sum "$extension_root/data/train.idx" "$extension_root/artifact-manifest.json" \
  "$base_data/final.idx"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
echo "ROUTING_AUDIT mode=$mode eval_iters=$eval_iters eval_global_batch=208 eval_micro_batch=16 sampler_offset=0 seed=1234 bias_replay=1e-3"

audit_one() {
  label=$1
  source=$2
  expected=$3
  required=$4
  tracker=$source/latest_checkpointed_iteration.txt
  if [[ ! -f $tracker ]]; then
    [[ $required == 0 ]] && { echo "OPTIONAL_MISSING label=$label tracker=$tracker"; return 0; }
    echo "MISSING label=$label tracker=$tracker" >&2
    return 2
  fi
  iteration=$(cat "$tracker")
  if [[ $iteration != "$expected" ]]; then
    [[ $required == 0 ]] && { echo "OPTIONAL_NOT_READY label=$label expected=$expected actual=$iteration"; return 0; }
    echo "WRONG_ITERATION label=$label expected=$expected actual=$iteration" >&2
    return 2
  fi
  endpoint=$(printf '%s/iter_%07d' "$source" "$iteration")
  [[ -d $endpoint ]] || { echo "MISSING label=$label endpoint=$endpoint" >&2; return 2; }

  rm -f "$stage_dir"
  ln -s "$source" "$stage_dir"
  output="$output_root/$label.json"
  echo "AUDIT_START label=$label iteration=$iteration source=$source output=$output"
  STAGE3_MOE_PROPAGATE_EXIT=1 \
  STAGE3_MOE_EVAL_LOAD="$stage_dir" \
  STAGE3_MOE_RUN_SUFFIX="fixed-routing-v1-$label" \
  STAGE3_MOE_LOG_ROOT="$log_root" \
  STAGE3_MOE_VALID_DATA_PREFIX="$extension_root/data/train" \
  STAGE3_MOE_TEST_DATA_PREFIX="$base_data/final" \
  STAGE3_MOE_ROUTING_AUDIT_PATH="$output" \
  STAGE3_MOE_DATA_CACHE_PATH="$data_cache" \
  STAGE3_ROUTING_CHECKPOINT_LABEL="$label" \
  STAGE3_MOE_EVAL_ITERS="$eval_iters" \
  WANDB_MODE=disabled \
    "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" eval-routing-fixed
  code=$?
  echo "AUDIT_EXIT=$code label=$label"
  [[ $code == 0 ]] || return "$code"
  python - "$output" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for e in d["evaluations"]:
    print(
        f"ROUTING_RESULT label={d['checkpoint_label']} split={e['split']} "
        f"loss={e['loss']:.6f} worst_cv={e['worst_actual_cv']:.6f} "
        f"frozen_cv={e['worst_frozen_cv']:.6f} "
        f"unbiased_cv={e['worst_unbiased_cv']:.6f} "
        f"min_mean={e['worst_actual_minimum_to_mean']:.6f}"
    )
PY
}

audit_one source-13794 \
  "/workspace-SR006.nfs3/hmoe-checkpoints/stage3-time-match-source/$arm" 13794 1 || exit $?
[[ $mode == smoke ]] && { echo EXIT=0; exit 0; }
audit_one original-17242 \
  "/workspace-SR006.nfs3/hmoe-checkpoints/stage3/trunk/$arm" 17242 1 || exit $?
audit_one extension-control-17242 \
  "/workspace-SR006.nfs3/hmoe-checkpoints/stage3-extension-decay-control/extension-decay-control/$arm" 17242 1 || exit $?
audit_one old-time-match-19570 \
  "/home/jovyan/hmoe-checkpoints/stage3-time-match/time-match/$arm" 19570 1 || exit $?
audit_one stretched-19570 \
  "/workspace-SR006.nfs3/hmoe-checkpoints/stage3-time-match-stretched-v1/time-match-stretched/$arm" 19570 0 || exit $?

echo EXIT=0
