#!/usr/bin/env bash
# Run a subset of the test suite on a real H100.
#
#   mlsub run --entry scripts/cloud_run_tests.sh --gpus 1 --image torch28 \
#     --args tests/stage3_moe/test_optimizer_contract.py
#
# The FP8 optimizer-state codec is Triton, so everything that exercises it is guarded by
# `torch.cuda.is_available()` and silently skips on a workstation. There is no torch on the
# workstation at all, so those tests have never run anywhere but here.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
targets=("$@")
[[ ${#targets[@]} -gt 0 ]] || targets=(tests/stage3_moe)

export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
unset PYTHONNOUSERSITE
nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
  -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: - || true)
[[ -n $nvidia_lib_path ]] && export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

echo "IMAGE=${MLSUB_IMAGE:-unset} TARGETS=${targets[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c 'import sys, torch; print(f"PY={sys.version.split()[0]} TORCH={torch.__version__} CUDA={torch.cuda.is_available()}")'
python -m pytest --version 2>/dev/null || pip install --user --no-cache-dir pytest 2>&1 | tail -1

cd "$root"
# -rs so a skip is visible: a CUDA-guarded test that skipped here would mean the job ran
# without a GPU, which looks identical to a pass in the summary line.
python -m pytest -v -rs "${targets[@]}"
echo "PYTEST_EXIT=$?"
exit 0
