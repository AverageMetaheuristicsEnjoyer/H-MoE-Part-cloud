#!/usr/bin/env bash
# Where can we write, and how much room is there? Read-only except for one probe file.
set -u
echo "=== ALL MOUNTS ==="
df -hT | grep -vE "^(tmpfs|devtmpfs|overlay .*/var/lib)" | head -30
echo
echo "=== CANDIDATE ROOTS ==="
for d in /home/jovyan /home/jovyan/shares /home/jovyan/shares/SR006.nfs2 \
         /home/jovyan/shares/SR006.nfs2/"$MLSUB_STUDENT" /workspace /data /mnt /shares; do
  [ -e "$d" ] || continue
  printf '%-52s ' "$d"
  df -h "$d" 2>/dev/null | tail -1 | awk '{printf "size=%s used=%s avail=%s (%s)  ", $2,$3,$4,$5}'
  probe="$d/.write-probe-$$"
  if touch "$probe" 2>/dev/null; then echo "WRITABLE"; rm -f "$probe"; else echo "read-only"; fi
done
echo
echo "=== jovyan top level ==="
du -sh /home/jovyan/* 2>/dev/null | sort -rh | head -12
echo
echo "=== shares namespace (ours only) ==="
ls -la /home/jovyan/shares/SR006.nfs2/ 2>/dev/null | head -15
echo "EXIT=0"
exit 0
