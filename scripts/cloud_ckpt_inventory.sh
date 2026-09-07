#!/usr/bin/env bash
# What is still on the two spare NFS volumes: free space, every checkpoint directory
# that survived, and the iteration each arm would resume from.  The platform keeps no
# logs once a job ends, so the same output is also left on /home/jovyan.
set -u
log=/home/jovyan/logs/ckpt-inventory-$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p "$(dirname "$log")"
exec > >(tee "$log") 2>&1

echo "=== free space ==="
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1

for vol in /workspace-SR006.nfs2 /workspace-SR006.nfs3; do
  echo
  echo "=== $vol ==="
  [ -d "$vol" ] || { echo "  absent"; continue; }
  du -sh "$vol"/* 2>/dev/null | sort -h | tail -10
  find "$vol" -maxdepth 6 -name latest_checkpointed_iteration.txt 2>/dev/null | sort | while read -r t; do
    echo "  TRACKER $(dirname "$t") -> $(cat "$t" 2>/dev/null)"
  done
  find "$vol" -maxdepth 6 -type d -name 'iter_*' 2>/dev/null | sort | while read -r d; do
    echo "  CKPT $(du -sh "$d" 2>/dev/null | cut -f1)  $d"
  done
done

echo
echo "LOG_SAVED $log"
echo "EXIT=0"
exit 0
