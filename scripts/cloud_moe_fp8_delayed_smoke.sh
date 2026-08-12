#!/usr/bin/env bash
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
log_dir=/home/jovyan/hmoe-cloud/logs
log=$log_dir/stage3-moe-fp8-delayed-$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p "$log_dir"

(
  set -euo pipefail
  if [[ ${MLSUB_IMAGE:-} != torch28 ]]; then
    echo "this smoke requires mlsub --image torch28; MLSUB_IMAGE=${MLSUB_IMAGE:-unset}" >&2
    exit 2
  fi

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
  export STAGE3_MOE_RUN_ID=${STAGE3_MOE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-adamw_fp8gemm_state_fp32-mock-smoke}
  export STAGE3_MOE_RESULTS_JSONL=${STAGE3_MOE_RESULTS_JSONL:-/home/jovyan/hmoe-cloud/artifacts/stage3-moe-probes/$STAGE3_MOE_RUN_ID/results.jsonl}

  nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version --format=csv,noheader
  python - <<'PY'
import sys

import torch
import transformer_engine
import transformer_engine.pytorch as te
import transformer_engine_torch
from transformer_engine.common.recipe import DelayedScaling, Format

assert sys.version_info >= (3, 12)
assert torch.cuda.is_available()
recipe = DelayedScaling(fp8_format=Format.HYBRID)
linear = te.Linear(256, 256, bias=False, params_dtype=torch.bfloat16, device="cuda")
x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
with te.autocast(enabled=True, recipe=recipe):
    y = linear(x)
y.float().square().mean().backward()
assert torch.isfinite(y).all()
assert torch.isfinite(x.grad).all()
assert torch.isfinite(linear.weight.grad).all()
print("python=", sys.version.split()[0])
print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("te=", transformer_engine.__version__)
print("cublaslt=", transformer_engine_torch.get_cublasLt_version())
print("te_delayed_hybrid_linear_fwd_bwd=PASS")

import megatron.core.optimizer.emerging_optimizers
import megatron.training.arguments
from megatron.core.package_info import __shortversion__ as mcore_version
print("mcore_version=", mcore_version)
print("mcore_declared_commit=571370c829ca768fe37244f4e2e7f28d8accc4ab")
print("mcore_moe_source_import=PASS")
PY

  "$root/scripts/run_stage3_moe_probe.sh" \
    adamw_fp8gemm_state_fp32 mock --protocol smoke
) >"$log" 2>&1
code=$?

echo "EXIT=$code"
echo "LOG=$log"
tail -n 240 "$log"
exit "$code"
