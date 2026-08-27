#!/usr/bin/env bash
# What is on the volumes, at a granularity the disk audit does not reach.
#
#   mlsub run ... --entry scripts/cloud_membench_inventory.sh --gpus cpu --no-pip
#
# /home/jovyan reached 0 bytes free on 2026-08-27, which stops every job: the
# platform cannot even create its own log symlinks there. This lists what holds
# the space so the reclaim is a decision and not a guess. Read-only.
set -u

echo "=== free ==="
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1
echo "--- inodes ---"
df -i /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1

for root in /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3; do
  echo
  echo "=== $root (top level, shares/ excluded: it remounts the other volumes) ==="
  for entry in "$root"/* "$root"/.[!.]*; do
    [ -e "$entry" ] || continue
    case "$entry" in */shares) echo "  (skipped $entry)"; continue ;; esac
    du -sh "$entry" 2>/dev/null
  done | sort -h | tail -25
done

echo
echo "=== MoE training data (the bench needs it; mock data breaks MoE timing) ==="
for candidate in /home/jovyan/data/fineweb-edu-gpt2-megatron/data \
                 /workspace-SR006.nfs2/hmoe-data /workspace-SR006.nfs3/hmoe-data; do
  echo "--- $candidate"
  ls -la "$candidate" 2>&1 | head -12
done

echo
echo "=== dense fineweb shards (colleague's cloud runs read these) ==="
ls -la /workspace-SR006.nfs2/dimativator/fineweb-edu-100BT-16shards 2>&1 | head -8

echo
echo "=== pip caches per image ==="
du -sh /home/jovyan/.local-* /home/jovyan/.cache 2>/dev/null | sort -h
echo "EXIT=0"
exit 0
