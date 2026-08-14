#!/usr/bin/env bash
# Remove the stray iter_0002256 checkpoints the decay-1p2b runs leave behind.
#
# 2256 = 4 x 564, so --save-interval $short_decay_iters lands two steps after the
# branch point and writes a checkpoint nobody wants: ~2 GB per arm, and the volume
# needs that headroom for the fp8gemm trunks running alongside.
#
# Safe while a decay run is in flight: the run loaded its state at startup and never
# reads 2256 again. The tracker is rewound to 2254 so that a crashed run resubmitted
# later still finds the checkpoint it names; a run that reaches the end overwrites the
# tracker with 2818 itself.
#
#   mlsub run ... --entry scripts/cloud_clean_decay_stub.sh --gpus cpu
set -u
root=${STAGE3_MOE_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3}
stub=2256
seed=2254

echo "=== before ==="; df -h /workspace-SR006.nfs3 | tail -1

for d in "$root"/1p2b/*/; do
  [[ -d $d ]] || continue
  arm=$(basename "$d")
  stub_dir=$(printf '%s/iter_%07d' "${d%/}" "$stub")
  tracker="${d%/}/latest_checkpointed_iteration.txt"

  # Never touch a branch whose trunk point is missing: that would leave the arm
  # with no way back to 2254.
  if [[ ! -d "$root/trunk/$arm/$(printf 'iter_%07d' "$seed")" ]]; then
    echo "SKIP $arm: trunk branch point missing"
    continue
  fi
  if [[ ! -d $stub_dir ]]; then
    echo "SKIP $arm: no iter_$(printf '%07d' "$stub")"
    continue
  fi

  echo "REMOVE $arm $(du -sh "$stub_dir" 2>/dev/null | cut -f1)"
  rm -rf "$stub_dir"
  if [[ -f $tracker && $(cat "$tracker") == "$stub" ]]; then
    echo "$seed" > "$tracker"
    echo "  tracker $stub -> $seed"
  fi
done

echo "=== after ==="; df -h /workspace-SR006.nfs3 | tail -1
echo "=== remaining ==="
find "$root" -maxdepth 3 -name 'iter_*' -type d | sort
echo "EXIT=0"
exit 0
