#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
base=${STAGE3_MOE_ORIGINAL_DATA_ROOT:-/home/jovyan/data/fineweb-edu-gpt2-megatron}
extension=${STAGE3_MOE_EXTENSION_ROOT:-/workspace-SR006.nfs2/hmoe-data/fineweb-edu-time-match-extension}
output=${STAGE3_MOE_SLICE_AUDIT_OUTPUT:-/workspace-SR006.nfs2/hmoe-cloud/data-audit/training-slice-overlap-v1.json}

python "$root/scripts/audit_training_slice_overlap.py" \
  --base-root "$base" \
  --extension-root "$extension" \
  --output "$output"

echo "EXIT=0"
