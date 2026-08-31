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

remote_sizes = {entry.path: entry.size for entry in files}
mappings = (
    (
        Path("/workspace-SR006.nfs2/hmoe-checkpoints/stage3-1c-mb4/1c"),
        "1c-mb4",
    ),
    (
        Path("/workspace-SR006.nfs3/hmoe-checkpoints/stage3/trunk"),
        "1c-mb16",
    ),
    (
        Path("/workspace-SR006.nfs3/hmoe-checkpoints/stage3-time-match-source"),
        "1c-mb16",
    ),
)
for root, prefix in mappings:
    for checkpoint in sorted(root.glob("*/iter_*")):
        arm = checkpoint.parent.name
        remote = f"{prefix}/{arm}/{checkpoint.name}"
        local_files = {
            str(path.relative_to(checkpoint)): path.stat().st_size
            for path in checkpoint.rglob("*")
            if path.is_file()
        }
        archived_files = {
            path.removeprefix(remote + "/"): size
            for path, size in remote_sizes.items()
            if path.startswith(remote + "/")
        }
        if local_files != archived_files:
            print(
                f"LOCAL_ARCHIVE_MISMATCH local={checkpoint} remote={remote} "
                f"local_files={len(local_files)} remote_files={len(archived_files)}"
            )
            continue
        print(
            f"LOCAL_ARCHIVE_MATCH local={checkpoint} remote={remote} "
            f"files={len(local_files)} bytes={sum(local_files.values())}"
        )
print("HF_ARCHIVE_VERIFY=pass")
PY
