#!/usr/bin/env bash
# Read-only preflight for the Stage 3 FP8-GEMM time-matched extensions.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
src_root=${STAGE3_MOE_SRC_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3}
dst_root=${STAGE3_MOE_CKPT_ROOT:-/workspace-SR006.nfs2/hmoe-checkpoints/stage3-time-match}
data_root=${STAGE3_MOE_EXTENSION_ROOT:-/workspace-SR006.nfs2/hmoe-data/fineweb-edu-time-match-extension}

echo "=== FILESYSTEMS ==="
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1
df -i /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1

echo "=== DATA ==="
du -sh /home/jovyan/data/fineweb-edu-gpt2-megatron 2>/dev/null || echo "base-data-missing"
du -sh "$data_root" 2>/dev/null || echo "extension-data-not-built"
for path in "$data_root/data/train.bin" "$data_root/data/train.idx" "$data_root/artifact-manifest.json"; do
  if [[ -f $path ]]; then stat -c '%s %n' "$path"; else echo "MISSING $path"; fi
done

echo "=== CHECKPOINTS ==="
for arm in adamw_fp8gemm_state_fp32 muon_fp8gemm_state_fp32; do
  src="$src_root/trunk/$arm/iter_0013794"
  tracker="$src_root/trunk/$arm/latest_checkpointed_iteration.txt"
  if [[ -d $src ]]; then
    du -sh "$src"
    find "$src" -type f -printf '%s %p\n' | sort -n | tail -3
  else
    echo "MISSING $src"
  fi
  if [[ -f $tracker ]]; then echo "$tracker -> $(cat "$tracker")"; else echo "MISSING $tracker"; fi
done
du -sh "$dst_root" 2>/dev/null || echo "destination-not-created"

echo "=== CREDENTIAL PRESENCE ==="
[[ -s /home/jovyan/.wandb-key ]] && echo "wandb-key-present" || echo "wandb-key-missing"
[[ -s /home/jovyan/.cache/huggingface/token ]] && echo "hf-token-present" || echo "hf-token-missing"
echo "=== PHASE CONTRACT ==="
cd "$root"
python scripts/phase_transition_smoke.py
echo "PHASE_TEST_EXIT=$?"
echo "EXIT=0"
exit 0
