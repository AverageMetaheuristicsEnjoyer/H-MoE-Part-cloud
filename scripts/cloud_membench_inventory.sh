#!/usr/bin/env bash
# What is on the volumes, and which of it is reproducible rather than load-bearing.
#
#   mlsub run ... --entry scripts/cloud_membench_inventory.sh --gpus cpu --no-pip
#
# /home/jovyan reached 0 bytes free on 2026-08-27, which stops every job: the platform
# cannot even create its own log symlinks there. Read-only; deletes nothing.
set -u

echo "=== free ==="
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1

for root in /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3; do
  echo
  echo "=== $root (shares/ skipped: it remounts the other volumes) ==="
  for entry in "$root"/* "$root"/.[!.]*; do
    [ -e "$entry" ] || continue
    case "$entry" in */shares) continue ;; esac
    du -sh "$entry" 2>/dev/null
  done | sort -h | tail -20
done

echo
echo "=== every Megatron corpus, wherever it lives ==="
# The audited build is archived as a public HF dataset
# (AverageMetaheuristicsEnjoyer/fineweb-edu-gpt2-megatron) and scripts/cloud_fetch_data.sh
# restores it with a size check, so a local copy is a cache, not the only copy.
for root in /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3; do
  find "$root" -xdev -name '*.idx' -size +1M -printf '%10s %p\n' 2>/dev/null
done | sort -rn | head -20

echo
echo "=== what the two hmoe directories on the full volume actually hold ==="
# These are this project's own; everything else large there belongs to other work.
for dir in /home/jovyan/hmoe-checkpoints /home/jovyan/hmoe-cloud; do
  echo "--- $dir  ($(du -sh "$dir" 2>/dev/null | cut -f1))"
  du -sh "$dir"/* 2>/dev/null | sort -h | tail -10
done

echo
echo "=== the same names on the volumes with room, for comparison ==="
du -sh /workspace-SR006.nfs2/hmoe-checkpoints/* /workspace-SR006.nfs3/hmoe-checkpoints/* 2>/dev/null | sort -h | tail -10
echo "EXIT=0"
exit 0
