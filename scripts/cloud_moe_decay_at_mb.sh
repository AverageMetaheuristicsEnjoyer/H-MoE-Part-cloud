#!/usr/bin/env bash
# Repeat the 1.2B decay branch at a different micro-batch, in its own checkpoint root,
# so the mb=4 deliverables under stage3/1p2b are never touched.  Usage:
#   mlsub run ... --entry scripts/cloud_moe_decay_at_mb.sh --args "ARM [ARM ...]"
#     --env STAGE3_MOE_CKPT_ROOT=/workspace-SR006.nfs3/hmoe-checkpoints/stage3-mb16
#     --env STAGE3_MOE_MICRO_BATCH=16
# The new root must sit on the same volume as the source: the branch point is seeded with
# hardlinks, which cost nothing there and would be a 7.3 GB copy anywhere else.
# Always exits 0 so the platform keeps the logs; real status is in ARM_EXIT.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
src_root=${STAGE3_MOE_SRC_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3}
dst_root=${STAGE3_MOE_CKPT_ROOT:?set STAGE3_MOE_CKPT_ROOT to the new checkpoint root}
branch=2254
branch_dir=$(printf 'iter_%07d' "$branch")

if [[ $dst_root == "$src_root" ]]; then
  echo "refusing to run: STAGE3_MOE_CKPT_ROOT is the source root, that would overwrite the mb=4 branches" >&2
  exit 2
fi
df -h "$src_root" | tail -1

for arm in "$@"; do
  if [[ ! $arm =~ ^[a-z0-9_]+$ ]]; then
    echo "SKIP invalid arm name: $arm"
    continue
  fi
  src="$src_root/trunk/$arm/$branch_dir"
  dst="$dst_root/trunk/$arm"
  if [[ ! -d $src ]]; then
    echo "SKIP $arm: no branch point at $src"
    continue
  fi
  mkdir -p "$dst"
  if [[ ! -d "$dst/$branch_dir" ]]; then
    cp -al "$src" "$dst/" || { echo "SKIP $arm: could not hardlink the branch point"; continue; }
  fi
  echo "$branch" > "$dst/latest_checkpointed_iteration.txt"
  echo "=== ARM $arm seeded from $src ==="
  "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" decay-1p2b
  echo "ARM_EXIT=$? arm=$arm"
  du -sh "$dst_root"/*/"$arm" 2>/dev/null
done
df -h "$dst_root" | tail -1
echo "EXIT=0"
exit 0
