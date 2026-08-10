#!/usr/bin/env bash
set -eu

venv=/home/jovyan/hmoe-cloud/torch251-cu121
du -sh "$venv" 2>/dev/null || true
find "$venv/lib" -maxdepth 3 -type d -name 'torch*' -printf '%p\n' 2>/dev/null || true
if [ -x "$venv/bin/python" ]; then
    timeout 30 "$venv/bin/python" - <<'PY'
import torch
print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
PY
fi
