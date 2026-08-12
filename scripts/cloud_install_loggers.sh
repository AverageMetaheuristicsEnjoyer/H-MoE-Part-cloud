#!/usr/bin/env bash
# One-off: put wandb and tensorboard into the persistent per-image user base so
# training jobs can keep using --no-pip.
set -u
echo "PYTHONUSERBASE=${PYTHONUSERBASE:-unset}"
unset PYTHONNOUSERSITE
pip install --user -q wandb tensorboard
echo "PIP_EXIT=$?"
python - <<'PY'
for m in ("wandb", "tensorboard", "torch.utils.tensorboard"):
    try:
        __import__(m); print(f"{m}=OK")
    except Exception as e:
        print(f"{m}=MISSING ({e})")
PY
df -h /home/jovyan | tail -1
echo "EXIT=0"
exit 0
