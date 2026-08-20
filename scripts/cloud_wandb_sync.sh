#!/usr/bin/env bash
# Push every offline W&B run this project has accumulated to the self-hosted server, then
# leave the directories in place -- `wandb sync` marks a run as synced, so re-running this
# is cheap and idempotent.  Usage:
#   mlsub run ... --entry scripts/cloud_wandb_sync.sh --gpus cpu
#     --env WANDB_BASE_URL=https://... --env WANDB_API_KEY=...
# Training ran with WANDB_MODE=offline for months because no credentials existed in the
# job; the runs recorded everything and this is the catch-up.
set -u

: "${WANDB_API_KEY:?set WANDB_API_KEY with mlsub run --env}"
echo "WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}"

unset PYTHONNOUSERSITE
export WANDB_MODE=online
python -c 'import wandb' 2>/dev/null || pip install --user -q wandb
python -c 'import wandb; print("wandb=", wandb.__version__)'

mapfile -t runs < <(find /home/jovyan/hmoe-cloud -maxdepth 5 -type d -name 'offline-run-*' 2>/dev/null | sort)
echo "OFFLINE_RUNS=${#runs[@]}"
if [[ ${#runs[@]} -eq 0 ]]; then
  echo "nothing to sync"; echo "EXIT=0"; exit 0
fi

ok=0; failed=0
for d in "${runs[@]}"; do
  # .synced is wandb's own marker; skip what has already gone across.
  if [[ -f "$d/.synced" ]]; then
    echo "SKIP already synced: $d"
    continue
  fi
  echo "--- SYNC $d"
  if wandb sync --no-include-synced --mark-synced "$d" 2>&1 | tail -5; then
    ok=$((ok + 1))
  else
    failed=$((failed + 1))
    echo "SYNC_FAILED $d"
  fi
done
echo "SYNCED=$ok FAILED=$failed"
echo "EXIT=0"
exit 0
