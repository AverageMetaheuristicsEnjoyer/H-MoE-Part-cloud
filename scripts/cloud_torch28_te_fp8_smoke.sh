#!/usr/bin/env bash
set -eu

unset PYTHONNOUSERSITE
nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
    -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)
export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDNN_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cudnn
export CURAND_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/curand
export NVRTC_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cuda_nvrtc
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1

python - <<'PY'
import sys
import torch
import transformer_engine
import transformer_engine.pytorch as te
import transformer_engine_torch
from transformer_engine.common.recipe import Float8BlockScaling, Format

assert sys.version_info >= (3, 12)
assert torch.cuda.is_available()
available, reason = te.is_fp8_block_scaling_available(return_reason=True)
assert available, reason
recipe = Float8BlockScaling(fp8_format=Format.E4M3)
linear = te.Linear(256, 256, bias=False, params_dtype=torch.bfloat16, device="cuda")
x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
with te.autocast(enabled=True, recipe=recipe):
    y = linear(x)
y.float().square().mean().backward()
assert torch.isfinite(y).all()
assert torch.isfinite(x.grad).all()
assert torch.isfinite(linear.weight.grad).all()
print("torch=", torch.__version__)
print("cuda=", torch.version.cuda)
print("te=", transformer_engine.__version__)
print("cublaslt=", transformer_engine_torch.get_cublasLt_version())
print("fp8_block_e4m3_linear=PASS")
PY
