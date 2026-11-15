#!/usr/bin/env bash
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
log_dir=/home/jovyan/hmoe-cloud/logs
log=$log_dir/stage4-import-$(date +%F_%H%M%S).log
mkdir -p "$log_dir"

(
    set -eu
    if [[ ${MLSUB_IMAGE:-} == torch28 ]]; then
        unset PYTHONNOUSERSITE
        nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
            -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)
        export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        export CUDNN_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cudnn
        export CURAND_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/curand
        export NVRTC_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cuda_nvrtc
    else
        export PYTHONNOUSERSITE=1
    fi
    export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
    python - <<'PY'
import torch
import transformer_engine
import transformer_engine.pytorch
import transformer_engine_torch

print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("gpu=", torch.cuda.get_device_name(0))
print("te=", transformer_engine.__version__)
print("cublaslt=", transformer_engine_torch.get_cublasLt_version())
import megatron.core.optimizer.emerging_optimizers
import megatron.training.arguments
from stage4.fp8_optimizer_states import (
    dequantize_fp8_state,
    init_fp8_state,
    quantize_fp8_state_,
)

value = torch.linspace(-4, 4, 383, device="cuda")
state = {}
init_fp8_state(state, "moment", value, group_size=128)
quantize_fp8_state_(state, "moment", value, signed=True, group_size=128)
restored = dequantize_fp8_state(state, "moment", signed=True, group_size=128)
assert torch.isfinite(restored).all()
assert (restored - value).abs().mean() / value.abs().mean() < 0.05
print("fp8_optimizer_state_roundtrip=PASS")
print("stage4_source_import=PASS")
PY
) >"$log" 2>&1
code=$?

echo "EXIT=$code"
echo "LOG=$log"
tail -n 240 "$log"
exit "$code"
