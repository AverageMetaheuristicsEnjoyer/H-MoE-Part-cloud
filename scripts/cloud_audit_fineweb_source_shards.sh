#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
tools_root=${STAGE3_MOE_DATA_TOOLS_ROOT:-/workspace-SR006.nfs2/hmoe-data/tools}
output=${STAGE3_MOE_SOURCE_AUDIT_OUTPUT:-/workspace-SR006.nfs2/hmoe-cloud/data-audit/source-shards-0-11-v1.json}

export HF_HOME="$tools_root/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$tools_root/cache/xdg"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

"$tools_root/venv/bin/python" "$root/scripts/audit_fineweb_source_shards.py" \
  --rows-per-shard "${STAGE3_MOE_SOURCE_AUDIT_ROWS_PER_SHARD:-50000}" \
  --output "$output"

echo "EXIT=0"
