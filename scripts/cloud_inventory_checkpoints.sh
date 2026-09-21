#!/usr/bin/env bash
# What endpoints of the time-match / extension family still exist, and on which volume?
#
#   mlsub run --entry scripts/cloud_inventory_checkpoints.sh --gpus cpu --image torch28
#
# The August extension-data investigation compared runs whose job logs are gone; W&B kept
# their loss curves but every comparison in it was scored on --eval-iters 32, i.e. 6,656
# sequences, and a single 1C run wobbles ~0.3 % between adjacent eval points on that window.
# Re-scoring the surviving endpoints on the whole split is what turns that spread into a
# number, so the first question is which endpoints survived.
#
# Reads only directory metadata -- no payload, no du of the big trees -- so it is seconds.
set -u

roots=(
  /home/jovyan/hmoe-checkpoints
  /workspace-SR006.nfs2/hmoe-checkpoints
  /workspace-SR006.nfs3/hmoe-checkpoints
)

echo "=== free space ==="
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1 | grep -v "^Filesystem"

for root in "${roots[@]}"; do
  echo
  echo "=== $root ==="
  [[ -d $root ]] || { echo "  absent"; continue; }
  # An arm directory is one holding a tracker; print it with the iteration it names and
  # every iter_* actually present, which is what says whether the endpoint is loadable.
  find "$root" -maxdepth 4 -name latest_checkpointed_iteration.txt -print 2>/dev/null | sort | while read -r tracker; do
    arm_dir=$(dirname "$tracker")
    tracked=$(cat "$tracker" 2>/dev/null)
    iters=$(find "$arm_dir" -maxdepth 1 -type d -name 'iter_*' -printf '%f\n' 2>/dev/null | sort | tr '\n' ' ')
    size=$(du -sh --apparent-size "$arm_dir" 2>/dev/null | cut -f1)
    echo "  ${arm_dir#$root/}"
    echo "      tracker=$tracked size=${size:-?} present: ${iters:-none}"
  done
done
echo "INVENTORY=DONE"
exit 0
