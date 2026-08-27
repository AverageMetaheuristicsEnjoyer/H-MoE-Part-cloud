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
echo "=== every Megatron .bin/.idx pair reachable, wherever it lives ==="
# The MoE bench reads one corpus. If a second copy of it already sits on a volume
# with room, the 15 GB on the full volume is reclaimable rather than load-bearing.
for root in /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3; do
  find "$root" -maxdepth 6 -name '*.bin' -size +100M -printf '%10s %p\n' 2>/dev/null
done | sort -rn | head -20

echo
echo "=== dimativator (dense fineweb shards live somewhere under here) ==="
for root in /workspace-SR006.nfs2/dimativator /workspace-SR006.nfs3/dimativator; do
  echo "--- $root"
  du -sh "$root"/* 2>/dev/null | sort -h | tail -12
done

echo
echo "=== pip caches per image ==="
du -sh /home/jovyan/.local-* /home/jovyan/.cache 2>/dev/null | sort -h
echo "EXIT=0"
exit 0
