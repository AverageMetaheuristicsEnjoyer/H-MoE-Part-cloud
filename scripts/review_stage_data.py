"""Stage the pinned FineWeb-Edu GPT-2 Megatron corpus for reviewer runs (public HF dataset)."""
import hashlib
import subprocess
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import snapshot_download

REVISION = "a2b8660e317ea2449c6ef57d9710a6e66c914751"
root = Path(__file__).resolve().parents[1]
dest = root / "data" / "fineweb-edu-gpt2-megatron"
if not (dest / "data" / "train.idx").exists():
    snapshot_download(repo_id="AverageMetaheuristicsEnjoyer/fineweb-edu-gpt2-megatron",
                      repo_type="dataset", revision=REVISION, local_dir=str(dest))
sha = hashlib.sha256((dest / "artifact-manifest.json").read_bytes()).hexdigest()
if not sha.startswith("7a7cf91633"):
    raise RuntimeError(f"corpus manifest sha {sha} does not match runs' 7a7cf91633...")
print(dest)
