#!/usr/bin/env bash
# Probe the alternate NFS volumes for write access and cross-job persistence.
set -u
for d in /workspace-SR006.nfs2 /workspace-SR006.nfs3; do
  echo "=== $d ==="
  [ -e "$d" ] || { echo "  absent"; continue; }
  df -h "$d" | tail -1
  marker="$d/hmoe-persistence-marker.txt"
  if [ -f "$marker" ]; then
    echo "  PERSISTED, previous marker: $(cat "$marker")"
  else
    echo "  no previous marker (first visit)"
  fi
  if echo "written by $(hostname) at $(date -u +%FT%TZ)" > "$marker" 2>/dev/null; then
    echo "  WRITABLE, wrote marker"
    mkdir -p "$d/hmoe-checkpoints" 2>/dev/null && echo "  mkdir hmoe-checkpoints OK"
  else
    echo "  NOT WRITABLE"
  fi
  ls -la "$d" 2>/dev/null | head -6
done
echo "EXIT=0"
exit 0
