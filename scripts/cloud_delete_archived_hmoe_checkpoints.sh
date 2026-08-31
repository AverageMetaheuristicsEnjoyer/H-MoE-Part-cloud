#!/usr/bin/env bash
set -euo pipefail

repo=${STAGE3_MOE_HF_SOURCE_REPO:-AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints}
token_file=${STAGE3_MOE_HF_TOKEN_FILE:-/home/jovyan/.cache/huggingface/token}

df -h /workspace-SR006.nfs2 /workspace-SR006.nfs3

python - "$repo" "$token_file" <<'PY'
import os
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi

repo, token_path = sys.argv[1:]
token = os.environ.get("HF_TOKEN") or (
    Path(token_path).read_text().strip() if Path(token_path).is_file() else None
)
files = [
    entry
    for entry in HfApi(token=token).list_repo_tree(
        repo, repo_type="model", recursive=True
    )
    if getattr(entry, "size", None) is not None
]
remote_sizes = {entry.path: entry.size for entry in files}
checkpoints = (
    (
        Path("/workspace-SR006.nfs2/hmoe-checkpoints/stage3-1c-mb4/1c/adamw_bf16_state_fp32/iter_0017242"),
        "1c-mb4/adamw_bf16_state_fp32/iter_0017242",
    ),
    (
        Path("/workspace-SR006.nfs2/hmoe-checkpoints/stage3-1c-mb4/1c/adamw_bf16_state_fp8/iter_0017242"),
        "1c-mb4/adamw_bf16_state_fp8/iter_0017242",
    ),
    (
        Path("/workspace-SR006.nfs2/hmoe-checkpoints/stage3-1c-mb4/1c/muon_bf16_state_fp32/iter_0017242"),
        "1c-mb4/muon_bf16_state_fp32/iter_0017242",
    ),
    (
        Path("/workspace-SR006.nfs2/hmoe-checkpoints/stage3-1c-mb4/1c/muon_bf16_state_fp8/iter_0017242"),
        "1c-mb4/muon_bf16_state_fp8/iter_0017242",
    ),
    (
        Path("/workspace-SR006.nfs3/hmoe-checkpoints/stage3/trunk/adamw_fp8gemm_state_fp32/iter_0013794"),
        "1c-mb16/adamw_fp8gemm_state_fp32/iter_0013794",
    ),
    (
        Path("/workspace-SR006.nfs3/hmoe-checkpoints/stage3/trunk/adamw_fp8gemm_state_fp32/iter_0017242"),
        "1c-mb16/adamw_fp8gemm_state_fp32/iter_0017242",
    ),
    (
        Path("/workspace-SR006.nfs3/hmoe-checkpoints/stage3/trunk/muon_fp8gemm_state_fp32/iter_0017242"),
        "1c-mb16/muon_fp8gemm_state_fp32/iter_0017242",
    ),
    (
        Path("/workspace-SR006.nfs3/hmoe-checkpoints/stage3-time-match-source/adamw_fp8gemm_state_fp32/iter_0013794"),
        "1c-mb16/adamw_fp8gemm_state_fp32/iter_0013794",
    ),
)

local_files = []
for checkpoint, remote in checkpoints:
    local = {
        str(path.relative_to(checkpoint)): path.stat().st_size
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    archived = {
        path.removeprefix(remote + "/"): size
        for path, size in remote_sizes.items()
        if path.startswith(remote + "/")
    }
    if not local or local != archived:
        raise RuntimeError(
            f"archive mismatch: local={checkpoint} remote={remote} "
            f"local_files={len(local)} remote_files={len(archived)}"
        )
    local_files.extend(path for path in checkpoint.rglob("*") if path.is_file())
    print(
        f"DELETE_ARCHIVE_MATCH local={checkpoint} remote={remote} "
        f"files={len(local)} bytes={sum(local.values())}"
    )

selected_links = Counter((path.stat().st_dev, path.stat().st_ino) for path in local_files)
for path in local_files:
    stat = path.stat()
    selected = selected_links[(stat.st_dev, stat.st_ino)]
    if stat.st_nlink != selected:
        raise RuntimeError(
            f"unselected hardlink: path={path} nlink={stat.st_nlink} selected={selected}"
        )

unique = {}
for path in local_files:
    stat = path.stat()
    unique[(stat.st_dev, stat.st_ino)] = stat.st_size
print(
    f"DELETE_ARCHIVE_PREFLIGHT=pass paths={len(local_files)} "
    f"unique_files={len(unique)} bytes={sum(unique.values())}"
)

for path in local_files:
    path.unlink()
for checkpoint, _ in checkpoints:
    for directory in sorted(
        (path for path in checkpoint.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.rmdir()
    checkpoint.rmdir()
print("DELETE_ARCHIVE_RESULT=pass")
PY

df -h /workspace-SR006.nfs2 /workspace-SR006.nfs3
