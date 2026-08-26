#!/usr/bin/env bash
# Build the independent FineWeb-Edu phase used by the longest (Muon) time-matched arm.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
output_root=${STAGE3_MOE_EXTENSION_ROOT:-/workspace-SR006.nfs2/hmoe-data/fineweb-edu-time-match-extension}
tools_root=${STAGE3_MOE_DATA_TOOLS_ROOT:-/workspace-SR006.nfs2/hmoe-data/tools}
plan="$root/configs/fineweb_edu_time_match_extension_plan.json"
venv="$tools_root/venv"
datatrove="$tools_root/datatrove"
megatron="$tools_root/Megatron-LM"
datatrove_commit=87f7bad5c4a56ec648265fbf0b91d7d226bad428
megatron_commit=571370c829ca768fe37244f4e2e7f28d8accc4ab

cleanup_intermediates() {
  python - "$output_root/shards" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
removed = 0
for path in root.glob("train-shard-*/tokens/*"):
    if path.is_file() and path.suffix in {".bin", ".idx"}:
        removed += path.stat().st_size
        path.unlink()
print(f"removed_intermediate_bytes={removed}")
PY
}

echo "=== PREFLIGHT ==="
df -h /workspace-SR006.nfs2 /workspace-SR006.nfs3 /home/jovyan 2>&1
df -i /workspace-SR006.nfs2 /workspace-SR006.nfs3 /home/jovyan 2>&1
mkdir -p "$tools_root" "$output_root"
unset PYTHONNOUSERSITE
nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
  -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: - || true)
export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

if [[ -f $output_root/artifact-manifest.json ]]; then
  python "$root/scripts/finalize_time_match_data.py" --output-root "$output_root" --plan "$plan"
  cleanup_intermediates
  echo "DATA_ALREADY_READY"
  exit 0
fi

if [[ ! -x $venv/bin/python ]]; then
  python -m venv --system-site-packages "$venv"
fi
requirements_sha=$(sha256sum "$root/requirements-data.txt" | awk '{print $1}')
if [[ ! -f $venv/.requirements-sha256 || $(cat "$venv/.requirements-sha256") != "$requirements_sha" ]]; then
  "$venv/bin/python" -m pip install -q -r "$root/requirements-data.txt"
  echo "$requirements_sha" > "$venv/.requirements-sha256"
fi

if [[ ! -d $datatrove/.git ]]; then
  git clone --filter=blob:none https://github.com/huggingface/datatrove.git "$datatrove"
fi
git -C "$datatrove" fetch --depth 1 origin "$datatrove_commit"
git -C "$datatrove" checkout --detach "$datatrove_commit"

if [[ ! -d $megatron/.git ]]; then
  git clone --filter=blob:none https://github.com/NVIDIA/Megatron-LM.git "$megatron"
fi
git -C "$megatron" fetch --depth 1 origin "$megatron_commit"
git -C "$megatron" checkout --detach "$megatron_commit"

"$venv/bin/python" "$root/scripts/source_plan_smoke.py" --plan "$plan"
"$venv/bin/python" "$root/scripts/build_fineweb_edu_train.py" \
  --plan "$plan" \
  --datatrove-root "$datatrove" \
  --output-root "$output_root/shards"

mkdir -p "$output_root/data"
if [[ ! -f $output_root/data/train.bin || ! -f $output_root/data/train.idx || ! -f $output_root/data/train.manifest.json ]]; then
  "$venv/bin/python" "$root/scripts/merge_indexed_data.py" \
    --megatron-root "$megatron" \
    --selection "$output_root/shards/train-selection.json" \
    --output-prefix "$output_root/data/train"
fi

"$venv/bin/python" "$root/scripts/gpt_dataset_smoke.py" \
  --megatron-root "$megatron" \
  --data-prefix "$output_root/data/train" \
  --samples 1750112 \
  > "$output_root/gpt-dataset-smoke.log"
cat "$output_root/gpt-dataset-smoke.log"
"$venv/bin/python" "$root/scripts/finalize_time_match_data.py" \
  --output-root "$output_root" \
  --plan "$plan"
cleanup_intermediates

echo "=== COMPLETE ==="
du -sh "$output_root" "$tools_root" 2>/dev/null
df -h "$output_root" | tail -1
echo "EXIT=0"
