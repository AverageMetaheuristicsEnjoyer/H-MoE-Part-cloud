#!/usr/bin/env bash
# Is a shared volume full or wedged? Every NFS touch is wrapped in a timeout, so this
# answers even when the thing it is asking about is the reason other jobs are hanging.
# Writes nothing anywhere.
set -u

for mount in /tmp /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3; do
  line=$(timeout 20 df -h "$mount" 2>&1 | tail -1)
  code=$?
  if [[ $code -eq 124 ]]; then
    echo "DF $mount = TIMEOUT (mount is wedged, not merely full)"
  else
    echo "DF $mount = $line"
  fi
done

# The failed 09-15 smoke downloaded into this cache before dying on ENOSPC; an
# .incomplete blob left behind would be holding the volume at zero.
for dir in /workspace-SR006.nfs2/hmoe-hf-cache /tmp/hmoe-hf-cache; do
  echo "=== $dir ==="
  timeout 30 du -sh "$dir" 2>&1 | tail -3
  timeout 30 find "$dir" -type f -size +100M -printf '%s\t%p\n' 2>/dev/null | sort -rn | head -10
  [[ $? -eq 124 ]] && echo "FIND $dir = TIMEOUT"
done

echo "DF_PROBE=DONE"
exit 0
