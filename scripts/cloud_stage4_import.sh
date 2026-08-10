#!/usr/bin/env bash
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
log_dir=/home/jovyan/hmoe-cloud/logs
log=$log_dir/stage4-import-$(date +%F_%H%M%S).log
mkdir -p "$log_dir"

(
    set -eu
    export PYTHONNOUSERSITE=1
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
import stage4.fp8_optimizer_states
print("stage4_source_import=PASS")
PY
) >"$log" 2>&1
code=$?

echo "EXIT=$code"
echo "LOG=$log"
tail -n 240 "$log"
exit "$code"
