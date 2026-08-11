#!/usr/bin/env bash
set -eu

python - <<'PY'
import shutil
import subprocess
import torch

print("python=", __import__("sys").version.split()[0])
print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("gpu=", torch.cuda.get_device_name(0))
print("nvcc=", shutil.which("nvcc"))
if shutil.which("nvcc"):
    print(subprocess.check_output(["nvcc", "--version"], text=True).strip())
print("cudnn_h=", __import__("os").path.exists("/usr/local/cuda/include/cudnn.h"))
print("cudnn_path=", __import__("os").environ.get("CUDNN_PATH"))
PY
