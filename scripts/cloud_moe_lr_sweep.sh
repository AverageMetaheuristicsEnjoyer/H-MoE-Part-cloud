#!/usr/bin/env bash
# Is the peak learning rate too high, and does lowering it narrow the FP8 gap?
#
#   mlsub run --entry scripts/cloud_moe_lr_sweep.sh --gpus 1 --image torch28 \
#     --env STAGE3_MOE_LR=8.15e-4 --note lr-sweep-adamw-half --args adamw_bf16_state_fp32
#
# The peak LR is the ONLY learning-rate argument where this MoE diverges from the dense
# code base: 1.63e-3 here against 3e-4 in run_stage4_dense.sh, 5.4x, while the decay style,
# the 3448/17242 split, the warmup fraction and min_lr = 0.1 x lr are identical. Lengthening
# the decay was measured and moves both arms equally (the FP8-state gap goes 0.667 % ->
# 0.684 %), so it is a quality lever and not a gap lever. A peak that is too high is a
# candidate for both: it inflates activations and gradients, which is exactly what exhausts
# FP8 dynamic range in the arm that spikes to grad norm 14-29 against a 0.08 median.
#
# Swept on the 1.2B budget -- 2,818 steps, ~5-6 h on one H100 -- because that round already
# reproduced the state-axis result to within 0.05 pp, so it is a validated proxy for 1C.
# The run is from scratch, not branched from the trunk: the LR being swept acts through the
# plateau, and branching would hold it fixed precisely where it matters.
#
# Warmup stays at 173 steps for every point. That is part of the recipe under test, so a
# high-LR arm that destabilises in warmup is a result, not a confound to be tuned away.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
arm=${1:-${STAGE3_MOE_LR_SWEEP_ARM:-adamw_bf16_state_fp32}}
log_root=${STAGE3_MOE_LOG_ROOT:-/tmp/hmoe-lr-sweep/logs}

export STAGE3_MOE_MICRO_BATCH=${STAGE3_MOE_MICRO_BATCH:-16}
export STAGE3_MOE_GLOBAL_BATCH=${STAGE3_MOE_GLOBAL_BATCH:-208}
export STAGE3_MOE_LOG_ROOT="$log_root"
export STAGE3_MOE_CKPT_ROOT=${STAGE3_MOE_CKPT_ROOT:-/tmp/hmoe-lr-sweep/ckpt}
export STAGE3_MOE_DATA_CACHE=${STAGE3_MOE_DATA_CACHE:-/tmp/hmoe-lr-sweep/data-cache}
export STAGE3_MOE_PROPAGATE_EXIT=1
# Every point is one job, so the run id has to carry the LR or two of them collide on one
# W&B run and interleave -- the same trap the recipe probe hit.
lr_tag=$(LC_ALL=C awk -v l="${STAGE3_MOE_LR:-1.63e-3}" 'BEGIN{printf "%g", l}' | tr '.+' 'p_')
export STAGE3_MOE_RUN_SUFFIX="lrsweep-${lr_tag}"

unset PYTHONNOUSERSITE
nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
  -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: - || true)
[[ -n $nvidia_lib_path ]] && export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

echo "IMAGE=${MLSUB_IMAGE:-unset} ARM=$arm LR=${STAGE3_MOE_LR:-1.63e-3} SUFFIX=$STAGE3_MOE_RUN_SUFFIX"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /tmp | tail -1

if [[ -z ${STAGE3_MOE_DATA_ROOT:-} ]]; then
  for candidate in \
    /home/jovyan/data/fineweb-edu-gpt2-megatron/data \
    /workspace-SR006.nfs2/hmoe-data/fineweb-edu-gpt2-megatron/data \
    /workspace-SR006.nfs3/hmoe-data/fineweb-edu-gpt2-megatron/data; do
    if compgen -G "$candidate/train*.bin" >/dev/null; then
      export STAGE3_MOE_DATA_ROOT=$candidate
      break
    fi
  done
fi
if [[ -z ${STAGE3_MOE_DATA_ROOT:-} ]]; then
  echo "LR_SWEEP_ABORT=no corpus found"
  exit 0
fi
echo "DATA_ROOT=$STAGE3_MOE_DATA_ROOT"

start=$SECONDS
"$root/scripts/run_stage3_moe_pretrain.sh" "$arm" lr-sweep
echo "TRAIN_EXIT=$? ARM=$arm LR=${STAGE3_MOE_LR:-1.63e-3} SECONDS=$((SECONDS - start))"

newest=$(ls -1t "$log_root/stage3-$arm-lr-sweep-$STAGE3_MOE_RUN_SUFFIX"/train-*.log 2>/dev/null | head -1)
if [[ -z $newest ]]; then
  echo "MISSING_LOG arm=$arm"
  exit 0
fi
# The endpoint is the whole readout, plus enough of the trajectory to tell a high-LR arm
# that merely ended badly from one that was unstable all along.
grep -aE "loss at iteration .* on (validation|test) set" "$newest" \
  | sed "s/^/LR_SWEEP_RESULT arm=$arm lr=${STAGE3_MOE_LR:-1.63e-3} /"
python - "$arm" "${STAGE3_MOE_LR:-1.63e-3}" "$newest" <<'PY'
import re, sys, statistics
arm, lr, path = sys.argv[1:4]
step = re.compile(r"iteration\s+(\d+)/")
gnorm = re.compile(r"grad norm:\s*([0-9.]+)")
skipped = re.compile(r"number of skipped iterations:\s*(\d+)")
nans = re.compile(r"number of nan iterations:\s*(\d+)")
norms, last_skipped, last_nan = [], 0, 0
for line in open(path, errors="replace"):
    if step.search(line):
        m = gnorm.search(line)
        if m:
            norms.append(float(m.group(1)))
        s, n = skipped.search(line), nans.search(line)
        if s:
            last_skipped = max(last_skipped, int(s.group(1)))
        if n:
            last_nan = max(last_nan, int(n.group(1)))
if norms:
    print(f"LR_SWEEP_STABILITY arm={arm} lr={lr} steps={len(norms)} "
          f"gnorm_med={statistics.median(norms):.4f} gnorm_max={max(norms):.3f} "
          f"n>0.3={sum(x > 0.3 for x in norms)} n>1.0={sum(x > 1.0 for x in norms)} "
          f"skipped={last_skipped} nan={last_nan}")
PY
exit 0
