#!/usr/bin/env bash
# Run one or more Stage 3 MoE arms sequentially inside a single Cloud job, so every
# arm sees the same physical GPU and matched pairs stay valid.  Usage:
#   mlsub run ... --entry scripts/cloud_moe_arms.sh --args "smoke ARM [ARM ...]"
# The first token is the protocol (smoke|probe).  Always exits 0 so the platform
# keeps the logs; real status is reported with explicit ARM_EXIT/EXIT markers.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
persist=/home/jovyan/hmoe-cloud
log_dir=$persist/logs
mkdir -p "$log_dir"

protocol=${1:-smoke}
shift || true
arms=("$@")

# Optional bounded-timing overrides, supplied with `mlsub run --env`.
step_args=()
if [[ -n ${STAGE3_MOE_WARMUP_STEPS:-} && -n ${STAGE3_MOE_MEASURE_STEPS:-} ]]; then
  step_args=(--warmup-steps "$STAGE3_MOE_WARMUP_STEPS" --measure-steps "$STAGE3_MOE_MEASURE_STEPS")
fi
data_mode=${STAGE3_MOE_DATA_MODE:-mock}
batch_tag="mb${STAGE3_MOE_MICRO_BATCH:-1}gb${STAGE3_MOE_GLOBAL_BATCH:-1}"
echo "BATCH micro=${STAGE3_MOE_MICRO_BATCH:-1} global=${STAGE3_MOE_GLOBAL_BATCH:-1} steps=${step_args[*]:-default} data=$data_mode"

echo "=== PRE-EXISTING ARTIFACTS ==="
if [[ -d $persist/artifacts/stage3-moe-probes ]]; then
  find $persist/artifacts/stage3-moe-probes -name results.jsonl -print | sort | while read -r f; do
    echo "--- ARTIFACT $f"
    cat "$f"
  done
else
  echo "(none)"
fi

echo "=== ENVIRONMENT ==="
unset PYTHONNOUSERSITE
nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
  -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)
export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDNN_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cudnn
export CURAND_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/curand
export NVRTC_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cuda_nvrtc
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export STAGE3_MOE_GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d ' ')
export STAGE3_MOE_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
export STAGE3_MOE_CUBLASLT=$(python -c 'import transformer_engine_torch as t; print(t.get_cublasLt_version())')
nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version --format=csv,noheader

python - <<'PY'
import sys, torch, transformer_engine, transformer_engine.pytorch as te, transformer_engine_torch
from transformer_engine.common.recipe import DelayedScaling, Format
print("python=", sys.version.split()[0])
print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("te=", transformer_engine.__version__)
print("cublaslt=", transformer_engine_torch.get_cublasLt_version())
recipe = DelayedScaling(fp8_format=Format.HYBRID)
linear = te.Linear(256, 256, bias=False, params_dtype=torch.bfloat16, device="cuda")
x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
with te.autocast(enabled=True, recipe=recipe):
    y = linear(x)
y.float().square().mean().backward()
assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
assert torch.isfinite(linear.weight.grad).all()
print("te_delayed_hybrid_linear_fwd_bwd=PASS")
from megatron.core.package_info import __shortversion__ as mcore_version
print("mcore_version=", mcore_version)
PY
echo "ENV_EXIT=$?"

declare -a result_paths=()
for arm in "${arms[@]}"; do
  if [[ ! $arm =~ ^[a-z0-9_]+$ ]]; then
    echo "SKIP invalid arm name: $arm"
    continue
  fi
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  run_id="$stamp-$arm-$data_mode-$protocol-$batch_tag"
  result_path="$persist/artifacts/stage3-moe-probes/$run_id/results.jsonl"
  arm_log="$log_dir/stage3-$run_id.log"
  echo "=== ARM $arm protocol=$protocol run_id=$run_id ==="
  STAGE3_MOE_RUN_ID="$run_id" STAGE3_MOE_RESULTS_JSONL="$result_path" \
    "$root/scripts/run_stage3_moe_probe.sh" "$arm" "$data_mode" --protocol "$protocol" "${step_args[@]}" \
    >"$arm_log" 2>&1
  code=$?
  echo "ARM_EXIT=$code arm=$arm log=$arm_log"
  grep -E "GPU_PREFLIGHT|GPU_POSTFLIGHT|E2E_WCT_SECONDS|DENOMINATORS|GRADIENT_ACCUMULATION_FUSION|lm loss|nan iterations|max allocated|Total number of|Error|Traceback|assert" "$arm_log" | tail -25
  if [[ -f $result_path ]]; then
    echo "--- RESULT_JSONL $result_path"
    cat "$result_path"
    result_paths+=("$result_path")
  else
    echo "--- NO RESULT JSONL for $arm"
    echo "--- last 40 log lines:"; tail -40 "$arm_log"
  fi
done

echo "=== PAIRS ==="
cd "$root"
for ((i = 0; i < ${#result_paths[@]}; i++)); do
  for ((j = 0; j < ${#result_paths[@]}; j++)); do
    [[ $i == "$j" ]] && continue
    out=$(python -m stage3_moe.pair_results "${result_paths[$i]}" "${result_paths[$j]}" 2>&1)
    if [[ $? -eq 0 ]]; then
      echo "--- PAIR ${result_paths[$i]} -> ${result_paths[$j]}"
      echo "$out"
    fi
  done
done

echo "EXIT=0"
exit 0
