#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
tools_root=${STAGE3_MOE_DATA_TOOLS_ROOT:-/workspace-SR006.nfs2/hmoe-data/tools}
shard_start=${STAGE3_MOE_MINHASH_SHARD_START:?set STAGE3_MOE_MINHASH_SHARD_START}
shard_end=${STAGE3_MOE_MINHASH_SHARD_END:?set STAGE3_MOE_MINHASH_SHARD_END}
documents=${STAGE3_MOE_MINHASH_DOCUMENTS_PER_SHARD:-100000}
output=${STAGE3_MOE_MINHASH_OUTPUT:?set STAGE3_MOE_MINHASH_OUTPUT}

export HF_HOME="$tools_root/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$tools_root/cache/xdg"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

"$tools_root/venv/bin/python" "$root/scripts/audit_fineweb_source_minhash.py" \
  --datatrove-root "$tools_root/datatrove" \
  --shard-start "$shard_start" \
  --shard-end "$shard_end" \
  --documents-per-shard "$documents" \
  --output "$output"

echo "EXIT=0"
