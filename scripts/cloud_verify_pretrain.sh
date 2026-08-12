#!/usr/bin/env bash
# Confirm what a pretraining run actually left behind: checkpoints, result record,
# logger output. Read-only.
set -u
ck=${STAGE3_MOE_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3}
lg=${STAGE3_MOE_LOG_ROOT:-/home/jovyan/hmoe-cloud/pretrain}
echo "=== checkpoints ==="
du -sh "$ck"/* 2>/dev/null | head
find "$ck" -maxdepth 3 -name "latest_checkpointed_iteration.txt" -exec sh -c 'echo "$1 -> $(cat "$1")"' _ {} \; 2>/dev/null | head
find "$ck" -maxdepth 3 -type d -name "iter_*" 2>/dev/null | head
echo
echo "=== volume ==="; df -h /workspace-SR006.nfs3 | tail -1
echo
echo "=== run dirs ==="; ls -la "$lg" 2>/dev/null | head
for d in "$lg"/*/; do
  [ -d "$d" ] || continue
  echo "--- $d"
  ls -la "$d" | head -8
  if [ -f "$d/results.jsonl" ]; then echo "  RESULT:"; cat "$d/results.jsonl"; fi
done
echo "EXIT=0"
exit 0
