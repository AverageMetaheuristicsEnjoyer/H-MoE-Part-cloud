#!/usr/bin/env bash
set -u

log_dir=/home/jovyan/logs
log=$log_dir/te_install.log
mkdir -p "$log_dir"

python - <<'PY' >"$log" 2>&1
import torch
print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
PY

pip install --user --no-build-isolation "transformer_engine[pytorch]" >>"$log" 2>&1
install_code=$?

echo "EXIT=$install_code"
tail -n 60 "$log"
python - <<'PY' 2>&1 | tail -n 4
import transformer_engine.pytorch as te
import transformer_engine
print("TE OK", getattr(transformer_engine, "__version__", "unknown"))
print("TE module", te.__file__)
PY
exit 0
