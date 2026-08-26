#!/usr/bin/env bash
# Score finished endpoints with the requested downstream suite.
#   mlsub run ... --entry scripts/cloud_moe_eval.sh --gpus 1 --image torch28 --args "ARM [ARM ...]"
#
# Both arms of a comparison belong in ONE job: pair_results requires the two records to
# carry the same GPU identity, and it compares the effective MCore argv wholesale, so the
# checkpoint is staged at a fixed path and every arm is loaded from there.
# Always exits 0 so the platform keeps the logs; real status is in ARM_EXIT.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
ckpt_root=${STAGE3_MOE_EVAL_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3}
# Where to look for each arm's endpoint, colon separated and searched in order. The two
# 1C waves live on different volumes -- the mb=4 arms under nfs2, the fp8gemm arms under
# nfs3 -- and the compute-axis pair takes its baseline from one and its treatment from the
# other, so a single job has to be able to reach both. (mlsub refuses an environment value
# containing a space, hence the colon.)
IFS=: read -ra eval_roots <<<"${STAGE3_MOE_EVAL_ROOTS:-$ckpt_root/1p2b}"
stage_dir=${STAGE3_MOE_EVAL_STAGE:-/tmp/stage3-eval-ckpt}
# The datasets are the bulk of this and /home/jovyan runs chronically near full, so the
# HF cache goes on the volume with room; the packages stay in the image's own user base.
export HF_HOME=${STAGE3_MOE_HF_HOME:-/workspace-SR006.nfs2/hmoe-hf-cache}
export HF_DATASETS_TRUST_REMOTE_CODE=1
mkdir -p "$HF_HOME"

nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h "$HF_HOME" | tail -1
echo "EVAL roots=${eval_roots[*]} tasks=${STAGE3_MOE_EVAL_TASKS:-basic_v2 default} limit=${STAGE3_MOE_EVAL_LIMIT:-none}"

if ! python -c 'import lm_eval; assert lm_eval.__version__ == "0.4.11"' >/dev/null 2>&1; then
  echo "=== installing lm-eval ==="
  # Pin torch so pip cannot pull a different build over the image's; the local +cu128
  # segment still satisfies the pin.
  printf 'torch==2.8.0\n' > /tmp/eval-constraints.txt
  pip install --user --no-cache-dir --constraint /tmp/eval-constraints.txt "lm-eval==0.4.11" 2>&1 | tail -5
fi
python - <<'PY'
import lm_eval, torch, transformers
print("lm_eval=", lm_eval.__version__, "torch=", torch.__version__, "transformers=", transformers.__version__)
PY
echo "INSTALL_EXIT=$?"

for arm in "$@"; do
  if [[ ! $arm =~ ^[a-z0-9_]+$ ]]; then
    echo "SKIP invalid arm name: $arm"
    continue
  fi
  src=""
  for base in "${eval_roots[@]}"; do
    tracker="$base/$arm/latest_checkpointed_iteration.txt"
    [[ -f $tracker ]] || continue
    # The offloaded waves left trackers behind pointing at iteration directories that
    # were deleted, so the tracker alone does not mean the weights are here.
    [[ -d $(printf '%s/%s/iter_%07d' "$base" "$arm" "$(cat "$tracker")") ]] || continue
    src="$base/$arm"
    break
  done
  if [[ -z $src ]]; then
    echo "SKIP $arm: no endpoint under ${eval_roots[*]}"
    continue
  fi
  rm -rf "$stage_dir"
  ln -s "$src" "$stage_dir"
  echo "=== ARM $arm endpoint=$(cat "$src/latest_checkpointed_iteration.txt") src=$src staged=$stage_dir ==="
  STAGE3_MOE_EVAL_LOAD="$stage_dir" "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" eval-downstream
  echo "ARM_EXIT=$? arm=$arm"
  # The launcher only tails the last 150 lines of the train log and MCore's own
  # evaluation fills them, so the scores have to be pulled out explicitly.
  run_dir="${STAGE3_MOE_LOG_ROOT:-/home/jovyan/hmoe-cloud/pretrain}/stage3-$arm-eval-downstream${STAGE3_MOE_RUN_SUFFIX:+-$STAGE3_MOE_RUN_SUFFIX}"
  newest=$(ls -1t "$run_dir"/train-*.log 2>/dev/null | head -1)
  if [[ -n $newest ]]; then
    grep -aE "DOWNSTREAM|lm_eval|Traceback|Error" "$newest" | head -30
  fi
  if [[ -f "$run_dir/downstream/downstream.json" ]]; then
    echo "--- DOWNSTREAM_JSON $arm"
    cat "$run_dir/downstream/downstream.json"
  else
    echo "--- NO DOWNSTREAM_JSON for $arm"
  fi
done
echo "EXIT=0"
exit 0
