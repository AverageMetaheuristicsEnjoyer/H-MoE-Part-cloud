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
  if [ -f "$d/results.jsonl" ]; then
    python - "$d/results.jsonl" <<'PYX'
import json, sys
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    m, n = d["measurement"], d["denominators"]
    print(f"  SUMMARY {d['arm_id']} mb={n['micro_batch_sequences_per_gpu']}"
          f" gb={n['global_batch_sequences']}"
          f" max_alloc={m['memory']['max_allocated_bytes']}"
          f" persistent={d['optimizer_state']['persistent_total_bytes']}"
          f" tok_s={m['timing']['tokens_per_second']:.0f}")
PYX
  fi
done
echo "EXIT=0"
exit 0
