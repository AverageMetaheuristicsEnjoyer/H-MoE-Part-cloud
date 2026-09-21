#!/usr/bin/env bash
# Re-score the time-match / extension family on the WHOLE held-out splits.
#
#   mlsub run --entry scripts/cloud_moe_fullsplit_eval.sh --gpus 1 --image torch28 \
#     --note fullsplit-eval
#
# Every comparison in the August extension-data investigation was scored with
# --eval-iters 32, i.e. 6,656 sequences. Measured 2026-09-21: `development` holds 8,000,266
# tokens = 3,906 sequences and `final` holds 100,001,217 = 48,828, so that window is 1.7x
# the whole dev split and 13.6 % of test. The 0.28 % penalty the investigation carries is a
# single draw on it, against a per-point spread of 0.165 % measured on the 1C plateau.
#
# 240 iterations x 208 = 49,920 sequences covers test once (48,828) and dev 12.8 times.
# Neither is an exact epoch, but eval-lm-fixed starts both samplers at zero, so every
# checkpoint is scored on a byte-identical window and the comparison is exact even where
# the absolute number carries a fractional-epoch bias.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
log_root=${STAGE3_MOE_LOG_ROOT:-/tmp/hmoe-fullsplit-eval/logs}
stage_root=${STAGE3_MOE_FULLSPLIT_STAGE:-/tmp/hmoe-fullsplit-eval}
arm=${STAGE3_MOE_FULLSPLIT_ARM:-adamw_fp8gemm_state_fp32}
hf_repo=${STAGE3_MOE_PROBE_HF_REPO:-AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints}
export HF_HOME=${STAGE3_MOE_HF_HOME:-/tmp/hmoe-fullsplit-eval/hf}
export STAGE3_MOE_EVAL_ITERS=${STAGE3_MOE_EVAL_ITERS:-240}
export STAGE3_MOE_MICRO_BATCH=${STAGE3_MOE_MICRO_BATCH:-16}
export STAGE3_MOE_LOG_ROOT="$log_root"
export STAGE3_MOE_CKPT_ROOT=${STAGE3_MOE_CKPT_ROOT:-/tmp/hmoe-fullsplit-eval/ckpt}
export STAGE3_MOE_DATA_CACHE=${STAGE3_MOE_DATA_CACHE:-/tmp/hmoe-fullsplit-eval/data-cache}
export STAGE3_MOE_PROPAGATE_EXIT=1

echo "IMAGE=${MLSUB_IMAGE:-unset} EVAL_ITERS=$STAGE3_MOE_EVAL_ITERS ARM=$arm"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /tmp | tail -1

# The 1C reference is the baseline every other row is measured against and it is the one
# endpoint no longer on the volumes -- it went to the HF archive with the rest of 1C.
ref_dir="$stage_root/1c-mb16/$arm"
if [[ ! -f "$ref_dir/iter_0017242/mp_rank_00/model_optim_rng.pt" ]]; then
  echo "=== staging 1c-mb16/$arm/iter_0017242 ==="
  mkdir -p "$ref_dir"
  pip install --user --no-cache-dir hf_transfer 2>&1 | tail -1
  python -c 'import hf_transfer' >/dev/null 2>&1 && export HF_HUB_ENABLE_HF_TRANSFER=1
  python - "$hf_repo" "1c-mb16/$arm" "$ref_dir" iter_0017242 <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo, remote, dest, name = sys.argv[1:5]
snapshot_download(repo_id=repo, allow_patterns=[f"{remote}/{name}/**"], local_dir=dest, max_workers=8)
staged, final = os.path.join(dest, remote, name), os.path.join(dest, name)
if os.path.isdir(staged) and not os.path.isdir(final):
    os.renames(staged, final)
print("STAGED", final)
PY
  echo 17242 > "$ref_dir/latest_checkpointed_iteration.txt"
fi

nfs3=/workspace-SR006.nfs3/hmoe-checkpoints
# label : directory : expected iteration
targets=(
  "1c_original:$ref_dir:17242"
  "extension_decay_matched:$nfs3/stage3-extension-decay-control/extension-decay-control/$arm:17242"
  "corrected_time_match:$nfs3/stage3-corrected-time-match-v1/corrected-time-match/$arm:19570"
  "stretched_decay:$nfs3/stage3-time-match-stretched-v1/time-match-stretched/$arm:19570"
  "time_match_wallclock:/home/jovyan/hmoe-checkpoints/stage3-time-match/time-match/$arm:19570"
)
[[ -n ${STAGE3_MOE_FULLSPLIT_TARGETS:-} ]] && read -ra targets <<<"$STAGE3_MOE_FULLSPLIT_TARGETS"

summary="$log_root/fullsplit-eval-summary.txt"
mkdir -p "$log_root"
: > "$summary"

for spec in "${targets[@]}"; do
  label=${spec%%:*}; rest=${spec#*:}; dir=${rest%:*}; want=${rest##*:}
  echo "=== EVAL label=$label dir=$dir expect=$want ==="
  tracker="$dir/latest_checkpointed_iteration.txt"
  if [[ ! -f $tracker ]]; then echo "MISSING_TRACKER=$tracker"; continue; fi
  have=$(cat "$tracker")
  if [[ $have != "$want" ]]; then echo "WRONG_ITERATION label=$label want=$want have=$have"; continue; fi
  if [[ ! -d $(printf '%s/iter_%07d' "$dir" "$have") ]]; then echo "MISSING_ENDPOINT label=$label"; continue; fi

  STAGE3_MOE_EVAL_LOAD="$dir" \
  STAGE3_MOE_RUN_SUFFIX="fullsplit-$label" \
    "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" eval-lm-fixed
  echo "EVAL_EXIT=$? label=$label"

  newest=$(ls -1t "$log_root/stage3-$arm-eval-lm-fixed-fullsplit-$label"/train-*.log 2>/dev/null | head -1)
  if [[ -z $newest ]]; then echo "MISSING_LOG label=$label"; continue; fi
  grep -aE "loss at iteration .* on (validation|test) set" "$newest" \
    | sed "s/^/FULLSPLIT label=$label /" | tee -a "$summary"
done

echo "=== FULLSPLIT EVAL SUMMARY ==="
cat "$summary"
exit 0
