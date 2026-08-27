#!/usr/bin/env bash
set -u

run=/home/jovyan/hmoe-cloud/pretrain/stage3-adamw_bf16_state_fp8-eval-downstream-broad-v2-1c
ls -l "$run/results.jsonl" "$run/downstream/downstream.json"
du -sh /home/jovyan/hmoe-cloud/pretrain/*broad-v2-1c 2>/dev/null
df -h /home/jovyan /workspace-SR006.nfs3

for snapshot in \
  /home/jovyan/.snapshot \
  /home/jovyan/hmoe-cloud/.snapshot \
  /home/jovyan/hmoe-cloud/pretrain/.snapshot; do
  echo "=== $snapshot ==="
  if [[ -d $snapshot ]]; then
    ls -la "$snapshot" | head -30
  else
    echo absent
  fi
done

echo EXIT=0
