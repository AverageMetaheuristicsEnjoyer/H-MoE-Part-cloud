#!/usr/bin/env bash
set -u

venv=/home/jovyan/hmoe-cloud/torch251-cu121
log_dir=/home/jovyan/hmoe-cloud/logs
log=$log_dir/bootstrap-torch251-$(date +%F_%H%M%S).log
mkdir -p "$log_dir"

(
    set -eu
    if [ ! -x "$venv/bin/python" ]; then
        /home/user/conda/bin/python -m venv "$venv"
    fi
    "$venv/bin/python" -m pip install --upgrade pip
    "$venv/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121
    "$venv/bin/python" - <<'PY'
import torch
print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("torch251=PASS")
PY
) >"$log" 2>&1
code=$?

echo "EXIT=$code"
echo "LOG=$log"
tail -n 240 "$log"
exit 0
