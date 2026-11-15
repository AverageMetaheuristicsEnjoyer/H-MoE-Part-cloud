#!/usr/bin/env bash
# Read-only availability check for the six original final and pre-decay 1C checkpoints.
set -u

failed=0
for arm in \
  adamw_bf16_state_fp32 adamw_bf16_state_fp8 \
  muon_bf16_state_fp32 muon_bf16_state_fp8; do
  base="/workspace-SR006.nfs2/hmoe-checkpoints/stage3-1c-mb4/1c/$arm"
  tracker="$base/latest_checkpointed_iteration.txt"
  iteration=$(cat "$tracker" 2>/dev/null || true)
  endpoint="$base/iter_0017242"
  if [[ $iteration == 17242 && -d $endpoint ]]; then
    echo "OK arm=$arm iteration=$iteration files=$(find "$endpoint" -type f | wc -l) bytes=$(du -sb "$endpoint" | cut -f1)"
  else
    echo "MISSING arm=$arm tracker=${iteration:-none} endpoint=$endpoint"
    failed=1
  fi
  retained="$base/iter_0013794"
  if [[ -d $retained ]]; then
    echo "PREDECAY_OK arm=$arm files=$(find "$retained" -type f | wc -l) bytes=$(du -sb "$retained" | cut -f1)"
  else
    echo "PREDECAY_MISSING arm=$arm checkpoint=$retained"
  fi
done

for arm in adamw_fp8gemm_state_fp32 muon_fp8gemm_state_fp32; do
  base="/workspace-SR006.nfs3/hmoe-checkpoints/stage3/trunk/$arm"
  tracker="$base/latest_checkpointed_iteration.txt"
  iteration=$(cat "$tracker" 2>/dev/null || true)
  endpoint="$base/iter_0017242"
  if [[ $iteration == 17242 && -d $endpoint ]]; then
    echo "OK arm=$arm iteration=$iteration files=$(find "$endpoint" -type f | wc -l) bytes=$(du -sb "$endpoint" | cut -f1)"
  else
    echo "MISSING arm=$arm tracker=${iteration:-none} endpoint=$endpoint"
    failed=1
  fi
  retained="$base/iter_0013794"
  if [[ -d $retained ]]; then
    echo "PREDECAY_OK arm=$arm files=$(find "$retained" -type f | wc -l) bytes=$(du -sb "$retained" | cut -f1)"
  else
    echo "PREDECAY_MISSING arm=$arm checkpoint=$retained"
  fi
done

df -h /workspace-SR006.nfs2 /workspace-SR006.nfs3
echo "EXIT=$failed"
exit "$failed"
