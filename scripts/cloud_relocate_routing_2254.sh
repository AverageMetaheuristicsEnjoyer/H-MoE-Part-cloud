#!/usr/bin/env bash
set -euo pipefail

arm=${1:?usage: cloud_relocate_routing_2254.sh ARM}
case "$arm" in
  adamw_bf16_state_fp32|frugal_coord_bf16_state_fp32|slimadam_bf16_state_fp32) ;;
  *) echo "unsupported arm: $arm" >&2; exit 2 ;;
esac

nfs2_root=/workspace-SR006.nfs2/hmoe-checkpoints
nfs3_root=/workspace-SR006.nfs3/hmoe-checkpoints
name=$arm-routing-calibration-2254-matched-v1
source_587=$nfs2_root/frugal-slimadam-gates/routing-calibration/$arm-routing-calibration-matched-v1
source_2254=$nfs3_root/frugal-slimadam-routing-2254/$name
destination_root=$nfs2_root/frugal-slimadam-routing-2254
destination_2254=$destination_root/$name
partial=$destination_2254.partial
run_dir=/workspace-SR006.nfs2/hmoe-cloud/pretrain/stage3-$arm-routing-calibration-2254-routing-calibration-2254-matched-v1

test "$(cat "$source_587/latest_checkpointed_iteration.txt")" = 587
test -d "$source_587/iter_0000587"
test "$(cat "$source_2254/latest_checkpointed_iteration.txt")" = 2254
test -d "$source_2254/iter_0002254"
grep -qE 'successfully loaded checkpoint.*iteration +587' "$run_dir"/train-*.log
grep -qE 'successfully saved checkpoint from iteration +2254' "$run_dir"/train-*.log
grep -qE 'number of nan iterations: +0' "$run_dir"/train-*.log
grep -q 'validation loss at iteration 2254' "$run_dir"/train-*.log
test ! -e "$destination_2254"
test ! -e "$partial"

rm -rf -- "$source_587"
echo "SOURCE_587_REMOVED=$source_587"

mkdir -p "$destination_root"
cp -a -- "$source_2254" "$partial"
diff -u \
  <(cd "$source_2254" && find . -type f -printf '%P\t%s\n' | LC_ALL=C sort) \
  <(cd "$partial" && find . -type f -printf '%P\t%s\n' | LC_ALL=C sort)
test "$(cat "$partial/latest_checkpointed_iteration.txt")" = 2254
test -d "$partial/iter_0002254"
mv -- "$partial" "$destination_2254"
rm -rf -- "$source_2254"

echo "CHECKPOINT_2254_RELOCATED=$destination_2254"
du -sh "$destination_2254"
df -h /workspace-SR006.nfs2 /workspace-SR006.nfs3
