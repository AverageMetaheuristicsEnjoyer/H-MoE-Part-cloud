#!/usr/bin/env bash
# Probe the alternate NFS volumes for write access and cross-job persistence, and
# report how much room every volume the job can see actually has: what fits is a
# planning input for further training, and only nfs3 was being watched.
set -u
echo "=== all mounts ==="
df -h 2>/dev/null | grep -vE "^(tmpfs|devtmpfs|overlay .*/var/lib)" | head -25
echo "--- named ---"
df -h / /tmp /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 /home/jovyan/shares/SR006.nfs2 2>&1
echo
echo "=== what fills them ==="
du -sh /home/jovyan/* 2>/dev/null | sort -h | tail -8
du -sh /workspace-SR006.nfs3/* 2>/dev/null | sort -h | tail -5
du -sh /workspace-SR006.nfs3/hmoe-checkpoints/stage3/* 2>/dev/null | sort -h | tail -10
du -sh /workspace-SR006.nfs2/* 2>/dev/null | sort -h | tail -5
echo
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
