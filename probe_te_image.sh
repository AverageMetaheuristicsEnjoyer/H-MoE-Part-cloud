#!/usr/bin/env bash
set -u

log_dir=/home/jovyan/logs
log=$log_dir/te_image_probe-$(date +%F_%H%M%S).log
mkdir -p "$log_dir"

(
    set -eu
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    nvcc --version
    python - <<'PY'
import torch
import transformer_engine
import transformer_engine.pytorch as te
print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("te=", transformer_engine.__version__)
print("te_module=", te.__file__)
PY
) >"$log" 2>&1
code=$?

echo "EXIT=$code"
echo "LOG=$log"
tail -n 100 "$log"
exit 0
