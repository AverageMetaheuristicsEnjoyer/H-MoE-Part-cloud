#!/usr/bin/env bash
# Reclaim the nfs2 space that the 2026-09-15 failed checkpoint download took.
#
#   mlsub run --entry scripts/cloud_reclaim_nfs2.sh --gpus cpu --image torch28
#
# snapshot_download without local_dir fills HF_HOME first and copies out of it afterwards.
# HF_HOME was on nfs2, the file was 14.4 GB and the volume had 6.8 GB, so it wrote until
# ENOSPC and left an .incomplete blob behind -- taking the volume to 100 %. That volume
# also held the platform's own logs_dir, so every job in the workspace stopped being able
# to start. Only removes our own regenerable cache; the datasets in it re-download.
#
# du -x does not work across these paths: shares/SR006.nfs{1,2,3} are separate mounts.
set -u

target=/workspace-SR006.nfs2/hmoe-hf-cache

echo "=== before ==="
df -h /workspace-SR006.nfs2 /workspace-SR006.nfs3 /home/jovyan 2>&1 | grep -v '^Filesystem'

echo "=== what is on nfs2 ==="
du -hd1 /workspace-SR006.nfs2 2>/dev/null | sort -h | tail -15

if [[ ! -d $target ]]; then
  echo "RECLAIM_ALREADY_ABSENT=$target"
else
  echo "=== $target ==="
  du -sh -- "$target" 2>/dev/null
  find "$target" -type f -size +100M -printf '%10s\t%p\n' 2>/dev/null | sort -rn | head -10
  # Structural guard: this script may only ever remove that one cache directory.
  if [[ $target != /workspace-SR006.nfs2/hmoe-hf-cache ]]; then
    echo "RECLAIM_REFUSED=unexpected target $target"
    exit 0
  fi
  rm -rf -- "$target"
  echo "RECLAIM_REMOVED=$target"
fi

echo "=== after ==="
df -h /workspace-SR006.nfs2 2>&1 | tail -1
echo "RECLAIM_DONE"
exit 0
