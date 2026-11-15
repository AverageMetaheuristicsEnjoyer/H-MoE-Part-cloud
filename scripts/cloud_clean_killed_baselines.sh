#!/usr/bin/env bash
# Remove the two mb=16 bf16 baselines that were killed on 2026-08-20 when the wave
# was corrected from eight arms to six. They are 25 GB of nfs3, and nfs3 needs that
# room: from iter_0014157 the two live fp8gemm arms each hold retained + rolling,
# three copies while a save is in flight, which does not fit beside these.
#
# A paired resume-bench WCT replicate can load its bf16 baseline from the live mb=4
# wave on nfs2 instead -- a job sees both volumes and check_checkpoint_args never
# compares micro_batch_size -- so nothing is lost that cannot be re-measured.
#
#   mlsub run ... --entry scripts/cloud_clean_killed_baselines.sh --gpus cpu
set -u
root=/workspace-SR006.nfs3/hmoe-checkpoints/stage3/trunk

# arm:expected iteration. The tracker must match exactly: any other value means the
# directory is not the dead run this script was written for -- most likely a live
# one -- and it is left alone.
targets=(
  "adamw_bf16_state_fp32:5082"
  "muon_bf16_state_fp32:4356"
)

echo "=== before ==="; df -h "$root" | tail -1

for t in "${targets[@]}"; do
  arm=${t%%:*}; want=${t##*:}
  d="$root/$arm"
  tracker="$d/latest_checkpointed_iteration.txt"
  if [[ ! -d $d ]]; then echo "SKIP $arm: absent"; continue; fi
  if [[ ! -f $tracker ]]; then echo "SKIP $arm: no tracker"; continue; fi
  have=$(cat "$tracker")
  if [[ $have != "$want" ]]; then
    echo "SKIP $arm: tracker is $have, expected $want -- not the dead run, leaving it"
    continue
  fi
  echo "REMOVE $arm at iteration $have"
  rm -rf "$d"
done

echo "=== after ==="; df -h "$root" | tail -1
echo "EXIT=0"
exit 0
