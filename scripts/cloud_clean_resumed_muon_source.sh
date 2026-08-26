#!/usr/bin/env bash
# Drop the hardlinked 13,794 staging copy after Muon has a newer time-match resume point.
set -u

root=/workspace-SR006.nfs3/hmoe-checkpoints
source_dir="$root/stage3-time-match-source/muon_fp8gemm_state_fp32"
live_dir="$root/stage3/time-match/muon_fp8gemm_state_fp32"
archived_link="$root/stage3/trunk/muon_fp8gemm_state_fp32/iter_0013794"
source_iteration=$(cat "$source_dir/latest_checkpointed_iteration.txt" 2>/dev/null || true)
live_iteration=$(cat "$live_dir/latest_checkpointed_iteration.txt" 2>/dev/null || true)

echo "=== before ==="
df -h "$root" | tail -1
if [[ $source_iteration == 13794 && $live_iteration =~ ^[0-9]+$ && $live_iteration -ge 14883 && ! -e $archived_link ]]; then
  echo "REMOVE $source_dir: live time-match checkpoint is $live_iteration"
  rm -rf "$source_dir"
else
  echo "SKIP source=$source_iteration live=$live_iteration archived_link_present=$([[ -e $archived_link ]] && echo yes || echo no)"
fi
echo "=== after ==="
df -h "$root" | tail -1
echo "EXIT=0"
exit 0
