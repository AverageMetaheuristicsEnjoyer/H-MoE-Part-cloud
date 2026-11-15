#!/usr/bin/env bash
# One-off: pull the audited Megatron-indexed FineWeb-Edu build onto the persistent
# volume so training jobs can use --data-path instead of --mock-data.  Idempotent:
# re-running only fills gaps.  CPU job; ~15.5 GB.
set -u
target=/home/jovyan/data/fineweb-edu-gpt2-megatron
repo=AverageMetaheuristicsEnjoyer/fineweb-edu-gpt2-megatron

echo "=== BEFORE ==="
df -h /home/jovyan | tail -1
mkdir -p "$target"

python - "$repo" "$target" <<'PY'
import sys, subprocess
try:
    import huggingface_hub  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--user", "-q", "huggingface_hub"], check=True)

from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=sys.argv[1],
    repo_type="dataset",
    local_dir=sys.argv[2],
    max_workers=8,
)
print("snapshot_download ->", path)
PY
echo "FETCH_EXIT=$?"

echo
echo "=== INVENTORY ==="
find "$target" -type f -printf '%10s  %p\n' 2>/dev/null | sort -k2 | head -30
echo
echo "expected from node207 build:"
echo "  train.bin       15113107020"
echo "  train.idx         145805762"
echo "  development.bin    16000532"
echo "  development.idx      152022"
echo "  final.bin         200002434"
echo "  final.idx           1936642"
echo
echo "=== SIZE CHECK ==="
python - "$target" <<'PY'
import sys
from pathlib import Path
expected = {
    "train.bin": 15113107020, "train.idx": 145805762,
    "development.bin": 16000532, "development.idx": 152022,
    "final.bin": 200002434, "final.idx": 1936642,
}
root = Path(sys.argv[1])
ok = True
for name, size in sorted(expected.items()):
    found = list(root.rglob(name))
    if not found:
        print(f"MISSING {name}"); ok = False; continue
    actual = found[0].stat().st_size
    match = actual == size
    ok &= match
    print(f"{'OK  ' if match else 'DIFF'} {name:18s} {actual:>13,} (expected {size:>13,})  {found[0].relative_to(root)}")
print("ALL_SIZES_MATCH" if ok else "SIZE_MISMATCH")
PY

echo
echo "=== AFTER ==="
df -h /home/jovyan | tail -1
du -sh "$target" 2>/dev/null
echo "EXIT=0"
exit 0
