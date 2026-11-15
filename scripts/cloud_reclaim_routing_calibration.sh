#!/usr/bin/env bash
# Remove the spent frugal/slimadam routing-calibration checkpoints from nfs2.
#
#   mlsub run --entry scripts/cloud_reclaim_routing_calibration.sh --gpus cpu --image torch28
#
# These come from the 2026-09-09 campaign that answered whether the early "routing collapse"
# of Frugal CoordAdamW and SlimAdam was real. It was a short-window artefact: at 235 and 587
# steps both read min/mean 0.0 with CV 1.39 / 1.82, and by step 2,750 Frugal was at min/mean
# 0.792, CV 0.0435, zero dropped tokens. That conclusion is recorded, the 1C Frugal run it
# authorised has completed, and its endpoints are in the HF archive -- so the calibration
# checkpoints themselves are spent.
#
# They are also what is keeping nfs2 at 100 %, which is not cosmetic: MCore writes a dataset
# index next to the corpus when --data-cache-path is absent, and the extension corpus lives
# here, so a full nfs2 kills any job that needs a new index size.
#
# Deleting the SlimAdam one is the irreversible part: SlimAdam never reached 1C, so step
# 2,254 was the only point it could have been continued from. Removed on explicit approval
# 2026-09-21.
set -u

root=/workspace-SR006.nfs2/hmoe-checkpoints
targets=(
  "$root/frugal-slimadam-gates/routing-calibration/frugal_coord_bf16_state_fp32-routing-calibration-matched-v1"
  "$root/frugal-slimadam-routing-2254/adamw_bf16_state_fp32-routing-calibration-2254-matched-v1"
  "$root/frugal-slimadam-routing-2254/slimadam_bf16_state_fp32-routing-calibration-2254-matched-v1"
)

echo "=== before ==="
df -h /workspace-SR006.nfs2 /workspace-SR006.nfs3 /home/jovyan 2>&1 | grep -v '^Filesystem'

for target in "${targets[@]}"; do
  echo
  echo "=== $target ==="
  # Structural guard: only ever a frugal/slimadam routing-calibration directory on nfs2.
  case "$target" in
    "$root"/frugal-slimadam-*/*routing-calibration*) ;;
    *) echo "RECLAIM_REFUSED=unexpected target"; continue ;;
  esac
  if [[ ! -d $target ]]; then
    echo "ALREADY_ABSENT"
    continue
  fi
  # Record what is being destroyed before destroying it.
  du -sh -- "$target" 2>/dev/null
  find "$target" -maxdepth 1 -mindepth 1 -printf '    %f\n' 2>/dev/null | sort
  rm -rf -- "$target"
  if [[ -d $target ]]; then echo "REMOVE_FAILED"; else echo "REMOVED"; fi
done

echo
echo "=== after ==="
df -h /workspace-SR006.nfs2 2>&1 | tail -1
echo "=== what still sits on nfs2 ==="
du -hd2 /workspace-SR006.nfs2/hmoe-checkpoints 2>/dev/null | sort -h | tail -10
echo "RECLAIM_DONE"
exit 0
