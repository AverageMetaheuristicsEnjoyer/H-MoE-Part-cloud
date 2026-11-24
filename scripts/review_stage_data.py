"""Stage the pinned corpus into the SHARED path the CI-eval runner reads (/home/jovyan/data).

Idempotent like cloud_fetch_data.sh; the runner additionally gates exact sizes + full manifest sha.
No pip fallback: huggingface_hub ships in the job image; absence is a fail-closed DATA_FAIL.
"""
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

REVISION = "a2b8660e317ea2449c6ef57d9710a6e66c914751"
dest = Path("/home/jovyan/data/fineweb-edu-gpt2-megatron")
if not (dest / "data" / "train.idx").exists():
    snapshot_download(repo_id="AverageMetaheuristicsEnjoyer/fineweb-edu-gpt2-megatron",
                      repo_type="dataset", revision=REVISION, local_dir=str(dest))
sha = hashlib.sha256((dest / "artifact-manifest.json").read_bytes()).hexdigest()
if not sha.startswith("7a7cf91633"):
    raise RuntimeError(f"corpus manifest sha {sha} does not match runs' 7a7cf91633...")
print(dest)
