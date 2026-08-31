#!/usr/bin/env bash
set -euo pipefail

repo=${STAGE3_MOE_HF_SOURCE_REPO:-AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints}
token_file=${STAGE3_MOE_HF_TOKEN_FILE:-/home/jovyan/.cache/huggingface/token}

python - "$repo" "$token_file" <<'PY'
import os
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi

repo, token_path = sys.argv[1:]
token = os.environ.get("HF_TOKEN") or (
    Path(token_path).read_text().strip() if Path(token_path).is_file() else None
)
api = HfApi(token=token)
info = api.model_info(repo)
files = [
    entry
    for entry in api.list_repo_tree(repo, repo_type="model", recursive=True)
    if getattr(entry, "size", None) is not None
]
totals = defaultdict(lambda: [0, 0])
for entry in files:
    top = entry.path.split("/", 1)[0]
    totals[top][0] += 1
    totals[top][1] += entry.size
print(
    f"HF_ARCHIVE repo={repo} private={info.private} "
    f"access={'authenticated' if token else 'anonymous'} files={len(files)}"
)
for top, (count, size) in sorted(totals.items()):
    print(f"HF_PREFIX prefix={top} files={count} bytes={size}")
for entry in sorted(files, key=lambda item: item.path):
    if entry.size > 1 << 30:
        print(f"HF_LARGE_FILE bytes={entry.size} path={entry.path}")
print("HF_ARCHIVE_VERIFY=pass")
PY
