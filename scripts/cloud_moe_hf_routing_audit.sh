#!/usr/bin/env bash
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
arm=${1:?usage: cloud_moe_hf_routing_audit.sh ARM}
iteration=${STAGE3_MOE_ROUTING_CANDIDATE_ITERATION:-17242}
repo=${STAGE3_MOE_HF_SOURCE_REPO:-AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints}

case "$arm" in
  adamw_bf16_state_fp32|adamw_bf16_state_fp8|muon_bf16_state_fp32|muon_bf16_state_fp8)
    prefix=1c-mb4
    ;;
  adamw_fp8gemm_state_fp32|muon_fp8gemm_state_fp32)
    prefix=1c-mb16
    ;;
  *)
    echo "unsupported routing-audit arm: $arm" >&2
    exit 2
    ;;
esac

work=$(mktemp -d /tmp/stage3-hf-routing.XXXXXX)
trap 'rm -rf "$work"' EXIT
name=$(printf 'iter_%07d' "$iteration")
remote="$prefix/$arm/$name"
source_dir="$work/source/$prefix/$arm"
label=${STAGE3_MOE_ROUTING_CANDIDATE_LABEL:-final-$iteration-$arm}

available_kb=$(df -Pk /tmp | awk 'END {print $4}')
[[ $available_kb -ge 25000000 ]] || {
  echo "GPU-local /tmp needs at least 25,000,000 KiB free: available=$available_kb" >&2
  exit 2
}
df -h /tmp

unset PYTHONNOUSERSITE
python -c 'import huggingface_hub' 2>/dev/null || pip install --user -q huggingface_hub
python - "$repo" "$remote" "$work/source" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

repo, remote, destination = sys.argv[1:]
snapshot_download(
    repo_id=repo,
    repo_type="model",
    allow_patterns=f"{remote}/**",
    local_dir=destination,
)
entries = list(
    HfApi().list_repo_tree(repo, path_in_repo=remote, repo_type="model", recursive=True)
)
remote_files = {
    entry.path.removeprefix(remote + "/"): entry.size
    for entry in entries
    if getattr(entry, "size", None) is not None
}
if not remote_files:
    raise RuntimeError(f"HF checkpoint is empty: {remote}")
local = Path(destination) / remote
local_files = {
    str(path.relative_to(local)): path.stat().st_size
    for path in local.rglob("*")
    if path.is_file()
}
if remote_files != local_files:
    raise RuntimeError(f"download size mismatch: remote={remote_files} local={local_files}")
print(
    f"HF_DOWNLOAD_VERIFIED path={remote} files={len(local_files)} "
    f"bytes={sum(local_files.values())} source={local}"
)
PY

echo "$iteration" > "$source_dir/latest_checkpointed_iteration.txt"
STAGE3_MOE_ROUTING_ARM="$arm" \
STAGE3_MOE_ROUTING_CANDIDATE="$source_dir" \
STAGE3_MOE_ROUTING_CANDIDATE_LABEL="$label" \
STAGE3_MOE_ROUTING_CANDIDATE_ITERATION="$iteration" \
  "$root/scripts/cloud_moe_fixed_routing_audit.sh" candidate
