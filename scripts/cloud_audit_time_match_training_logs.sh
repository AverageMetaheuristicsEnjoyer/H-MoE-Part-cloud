#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
output=${STAGE3_MOE_TRAINING_LOG_AUDIT_OUTPUT:-/workspace-SR006.nfs2/hmoe-cloud/data-audit/time-match-training-logs-v1.json}

python "$root/scripts/audit_time_match_training_logs.py" \
  --run "original:/workspace-SR006.nfs3/hmoe-cloud/pretrain/stage3-adamw_fp8gemm_state_fp32-full-mb16:13794" \
  --run "old_time_match:/workspace-SR006.nfs2/hmoe-cloud/pretrain/stage3-adamw_fp8gemm_state_fp32-time-match-wallclock-v1:16122" \
  --run "corrected:/workspace-SR006.nfs3/hmoe-cloud/pretrain/stage3-adamw_fp8gemm_state_fp32-corrected-time-match-data-continuity-v1:16122" \
  --output "$output"

echo "EXIT=0"
