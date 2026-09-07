#!/usr/bin/env bash
# Bounded Frugal CoordAdamW / SlimAdam checkpoint and calibration gates.
# Usage: cloud_moe_optimizer_gate.sh resume|stability|lr-screen [RECIPE]
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
mode=${1:?usage: cloud_moe_optimizer_gate.sh resume|stability|lr-screen [RECIPE]}
recipe=${2:-}
ckpt_root=${STAGE3_MOE_CKPT_ROOT:-/workspace-SR006.nfs2/hmoe-checkpoints/frugal-slimadam-gates}
log_root=${STAGE3_MOE_LOG_ROOT:-/workspace-SR006.nfs2/hmoe-cloud/pretrain}
export STAGE3_MOE_CKPT_ROOT=$ckpt_root
export STAGE3_MOE_LOG_ROOT=$log_root
export STAGE3_MOE_MICRO_BATCH=${STAGE3_MOE_MICRO_BATCH:-4}
export STAGE3_MOE_WANDB_PROJECT=${STAGE3_MOE_WANDB_PROJECT:-hmoe-stage3-frugal-slimadam}
export STAGE3_MOE_PROPAGATE_EXIT=1
mkdir -p "$ckpt_root" "$log_root"

if [[ ${MLSUB_IMAGE:-} != torch28 ]]; then
  echo "GATE_FAIL reason=image MLSUB_IMAGE=${MLSUB_IMAGE:-unset} expected=torch28"
  echo "EXIT=1"
  exit 0
fi

echo "=== SOURCE AND DISK ==="
git -C "$root" status --short --branch
git -C "$root" rev-parse HEAD
df -h "$ckpt_root" "$log_root" | awk 'NR == 1 || !seen[$1]++'
available_kb=$(df -Pk "$ckpt_root" | awk 'NR == 2 {print $4}')
if (( available_kb < 20971520 )); then
  echo "GATE_FAIL reason=disk available_kb=$available_kb required_kb=20971520"
  echo "EXIT=1"
  exit 0
fi

echo "=== ALLOCATED RUNTIME ==="
nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version --format=csv,noheader
unset PYTHONNOUSERSITE
nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
  -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: - || true)
export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
python - <<'PY'
import torch
import transformer_engine
from megatron.core.package_info import __shortversion__ as mcore_version

try:
    import fused_weight_gradient_mlp_cuda  # noqa: F401
except ImportError:
    apex = "absent"
else:
    apex = "present"
print(f"torch={torch.__version__} cuda={torch.version.cuda} te={transformer_engine.__version__} mcore={mcore_version} apex_wgrad={apex}")
PY
if [[ $? -ne 0 ]]; then
  echo "GATE_FAIL reason=imports"
  echo "EXIT=1"
  exit 0
fi

run_launcher() {
  "$root/scripts/run_stage3_moe_pretrain.sh" "$1" "$2"
}

validate_calibration_result() {
  python - "$1" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
record = records[-1]
routing = record["measurement"]["routing"]
training_loss = record["measurement"]["loss"]["training"]
if record["status"] != "completed":
    raise SystemExit("result status is not completed")
if training_loss is None or not math.isfinite(training_loss):
    raise SystemExit("training loss is absent or non-finite")
if routing["minimum_to_mean"] is None or routing["minimum_to_mean"] < 0.1:
    raise SystemExit(f"routing minimum/mean failed: {routing['minimum_to_mean']}")
if routing["coefficient_of_variation"] is None or routing["coefficient_of_variation"] >= 0.2:
    raise SystemExit(f"routing CV failed: {routing['coefficient_of_variation']}")
if routing["dropped_tokens"] != 0:
    raise SystemExit(f"dropped tokens failed: {routing['dropped_tokens']}")
print(
    f"RESULT_PASS training_loss={training_loss:.6f} "
    f"route_min_mean={routing['minimum_to_mean']:.6f} "
    f"route_cv={routing['coefficient_of_variation']:.6f} dropped=0"
)
PY
}

run_resume() {
  local arm=$1
  local suffix=optimizer-resume-v1
  local checkpoint_dir="$ckpt_root/resume-gate/$arm-$suffix"
  local tracker="$checkpoint_dir/latest_checkpointed_iteration.txt"
  local run_dir="$log_root/stage3-$arm-resume-gate-$suffix"
  export STAGE3_MOE_RUN_SUFFIX=$suffix
  export STAGE3_MOE_LR=1.63e-3
  export STAGE3_MOE_MIN_LR=1.63e-4
  export STAGE3_MOE_ADAM_BETA2=0.95

  if [[ ! -f $tracker ]]; then
    echo "=== RESUME SAVE arm=$arm target=50 ==="
    run_launcher "$arm" resume-gate
    if [[ $? -ne 0 || ! -f $tracker ]]; then
      echo "GATE_FAIL arm=$arm phase=save tracker=missing"
      return 1
    fi
    local iteration
    iteration=$(cat "$tracker")
    if [[ $iteration != 50 ]]; then
      echo "GATE_FAIL arm=$arm phase=save tracker=$iteration expected=50"
      return 1
    fi
  fi

  local iteration
  iteration=$(cat "$tracker")
  if [[ $iteration == 50 ]]; then
    echo "=== RESUME LOAD arm=$arm source=50 target=52 ==="
    run_launcher "$arm" resume-gate
    if [[ $? -ne 0 ]]; then
      echo "GATE_FAIL arm=$arm phase=load"
      return 1
    fi
  fi

  iteration=$(cat "$tracker")
  if [[ $iteration != 52 ]]; then
    echo "GATE_FAIL arm=$arm phase=final tracker=$iteration expected=52"
    return 1
  fi
  if ! grep -qE 'successfully loaded checkpoint.*iteration +50' "$run_dir"/train-*.log; then
    echo "GATE_FAIL arm=$arm phase=verify reason=no_iteration_50_load"
    return 1
  fi
  if ! grep -q 'number of nan iterations: 0' "$run_dir"/train-*.log; then
    echo "GATE_FAIL arm=$arm phase=verify reason=nan_summary_missing"
    return 1
  fi
  echo "GATE_PASS arm=$arm gate=resume checkpoint=50 final=52"
  if [[ ${STAGE3_MOE_KEEP_GATE_CKPT:-0} != 1 ]]; then
    rm -rf -- "$checkpoint_dir"
    echo "GATE_CKPT_REMOVED=$checkpoint_dir"
  fi
}

run_calibration() {
  local launcher_mode=$1
  local arm=$2
  local label=$3
  local lr=$4
  local min_lr=$5
  local beta2=$6
  local target=$7
  local suffix="$launcher_mode-$label-v1"
  local checkpoint_dir="$ckpt_root/$launcher_mode/$arm-$suffix"
  local tracker="$checkpoint_dir/latest_checkpointed_iteration.txt"
  local run_dir="$log_root/stage3-$arm-$launcher_mode-$suffix"

  if [[ -e $checkpoint_dir ]]; then
    echo "GATE_FAIL arm=$arm gate=$launcher_mode reason=checkpoint_path_exists path=$checkpoint_dir"
    return 1
  fi
  export STAGE3_MOE_RUN_SUFFIX=$suffix
  export STAGE3_MOE_LR=$lr
  export STAGE3_MOE_MIN_LR=$min_lr
  export STAGE3_MOE_ADAM_BETA2=$beta2
  export STAGE3_MOE_EVAL_INTERVAL=$target

  echo "=== CALIBRATION gate=$launcher_mode arm=$arm recipe=$label lr=$lr beta2=$beta2 target=$target ==="
  run_launcher "$arm" "$launcher_mode"
  if [[ $? -ne 0 || ! -f $tracker ]]; then
    echo "GATE_FAIL arm=$arm gate=$launcher_mode recipe=$label"
    return 1
  fi
  local iteration
  iteration=$(cat "$tracker")
  if [[ $iteration != "$target" ]]; then
    echo "GATE_FAIL arm=$arm gate=$launcher_mode recipe=$label tracker=$iteration expected=$target"
    return 1
  fi
  if ! grep -q 'number of nan iterations: 0' "$run_dir"/train-*.log; then
    echo "GATE_FAIL arm=$arm gate=$launcher_mode recipe=$label reason=nan_summary_missing"
    return 1
  fi
  if ! grep -q 'validation loss at iteration' "$run_dir"/train-*.log; then
    echo "GATE_FAIL arm=$arm gate=$launcher_mode recipe=$label reason=validation_missing"
    return 1
  fi
  if ! validate_calibration_result "$run_dir/results.jsonl"; then
    echo "GATE_FAIL arm=$arm gate=$launcher_mode recipe=$label reason=result_or_routing"
    return 1
  fi
  echo "GATE_PASS arm=$arm gate=$launcher_mode recipe=$label target=$target"
  grep -E 'validation loss at iteration|loss at iteration|number of nan iterations' "$run_dir"/train-*.log | tail -8
  if [[ $launcher_mode == stability && ${STAGE3_MOE_KEEP_GATE_CKPT:-0} != 1 ]]; then
    rm -rf -- "$checkpoint_dir"
    echo "GATE_CKPT_REMOVED=$checkpoint_dir"
  fi
}

status=0
case "$mode" in
  resume)
    [[ -z $recipe ]] || status=1
    if (( status == 0 )); then
      run_resume frugal_coord_bf16_state_fp32 || status=1
    fi
    if (( status == 0 )); then
      run_resume slimadam_bf16_state_fp32 || status=1
    fi
    ;;
  stability)
    [[ -z $recipe ]] || status=1
    if (( status == 0 )); then
      run_calibration stability frugal_coord_bf16_state_fp32 matched 1.63e-3 1.63e-4 0.95 235 || status=1
    fi
    if (( status == 0 )); then
      run_calibration stability slimadam_bf16_state_fp32 matched 1.63e-3 1.63e-4 0.95 235 || status=1
    fi
    ;;
  lr-screen)
    case "$recipe" in
      matched)
        run_calibration lr-screen frugal_coord_bf16_state_fp32 matched 1.63e-3 1.63e-4 0.95 587 || status=1
        ;;
      efficient-training-1e3)
        run_calibration lr-screen frugal_coord_bf16_state_fp32 efficient-training-1e3 1e-3 1e-4 0.999 587 || status=1
        ;;
      efficient-training-2e3)
        run_calibration lr-screen frugal_coord_bf16_state_fp32 efficient-training-2e3 2e-3 2e-4 0.999 587 || status=1
        ;;
      *) status=1 ;;
    esac
    ;;
  *) status=1 ;;
esac

if (( status != 0 )); then
  echo "GATE_FAIL mode=$mode recipe=${recipe:-none}"
  echo "EXIT=1"
else
  echo "GATE_PASS mode=$mode recipe=${recipe:-none}"
  echo "EXIT=0"
fi
exit 0
