"""HF checkpoint archive transfer for fullrun-016 (reuses the October mechanism).

October offload (history/coat-tools/hf_offload.py, scripts/cloud_hf_offload_retained.sh) used
huggingface_hub HfApi: upload_file per file, then list_repo_tree(expand=True) to verify the
server's own LFS sha256 and size before any local deletion. This module reuses exactly that
pattern for full-state checkpoints and weights-only exports. HF_TOKEN comes from the job env
(submit.py --hf-token, redacted); it is never printed or written to artifacts. Weights travel
Cloud<->HF only, never through Mac or the small relay.
"""
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_hub(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    receipts = {}
    try:
        import huggingface_hub  # noqa: F401
        receipts["huggingface_hub"] = "present"
    except ImportError:
        with (log_dir / "pip-hub.stdout.log").open("w") as out, (log_dir / "pip-hub.stderr.log").open("w") as err:
            code = subprocess.call([sys.executable, "-m", "pip", "install", "--user", "-q",
                                    "--disable-pip-version-check", "huggingface_hub"], stdout=out, stderr=err, timeout=300)
        receipts["huggingface_hub"] = f"pip_install_exit={code}"
        if code != 0:
            raise RuntimeError("huggingface_hub unavailable and pip install failed")
        import huggingface_hub  # noqa: F401
    try:
        import hf_transfer  # noqa: F401
        receipts["hf_transfer"] = "present"
    except ImportError:
        with (log_dir / "pip-hftransfer.stdout.log").open("w") as out, (log_dir / "pip-hftransfer.stderr.log").open("w") as err:
            code = subprocess.call([sys.executable, "-m", "pip", "install", "--user", "-q",
                                    "--disable-pip-version-check", "hf_transfer"], stdout=out, stderr=err, timeout=180)
        receipts["hf_transfer"] = f"pip_install_exit={code} (optional)"
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    except ImportError:
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    import huggingface_hub
    receipts["version"] = huggingface_hub.__version__
    return receipts


def make_api(token):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def preflight(api, repo):
    info = api.model_info(repo)
    return {"repo": repo, "repo_sha": getattr(info, "sha", None),
            "last_modified": str(getattr(info, "last_modified", None)), "reachable": True}


def remote_entries(api, repo, prefix):
    return {entry.path: entry for entry in api.list_repo_tree(repo, path_in_repo=prefix, expand=True)}


def build_manifest(local_dir, iteration, arm, kind):
    local_dir = Path(local_dir)
    files = []
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            files.append({"relative_path": str(path.relative_to(local_dir)),
                          "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"iteration": iteration, "arm": arm, "kind": kind, "files": files,
            "total_bytes": sum(item["bytes"] for item in files)}


def verify_remote(api, repo, prefix, manifest):
    entries = remote_entries(api, repo, prefix)
    receipts = []
    for item in manifest["files"]:
        remote = f"{prefix}/{item['relative_path']}"
        entry = entries.get(remote)
        if entry is None:
            raise RuntimeError(f"HF missing uploaded file {remote}")
        lfs = getattr(entry, "lfs", None)
        server_sha = getattr(lfs, "sha256", None) if lfs is not None else None
        server_size = getattr(entry, "size", None)
        # Small non-LFS files (manifests) may not carry an LFS sha256; verify size for those.
        sha_ok = (server_sha == item["sha256"]) if server_sha is not None else (item["bytes"] < 5 * 2**20)
        if not sha_ok or server_size != item["bytes"]:
            raise RuntimeError(f"HF post-upload identity mismatch for {remote}: sha={server_sha} size={server_size}")
        receipts.append({"remote": remote, "bytes": server_size, "lfs_sha256": server_sha,
                         "local_sha256": item["sha256"], "verified": True})
    return receipts


def upload_directory(api, repo, local_dir, prefix, manifest, log=print):
    local_dir = Path(local_dir)
    started = time.monotonic()
    for item in manifest["files"]:
        local = local_dir / item["relative_path"]
        remote = f"{prefix}/{item['relative_path']}"
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=repo,
                        commit_message=f"fullrun-016 {manifest['arm']} iter{manifest['iteration']} {item['relative_path']}")
        log(f"HF_UPLOAD {remote} bytes={item['bytes']}", flush=True)
    receipts = verify_remote(api, repo, prefix, manifest)
    return {"prefix": prefix, "files": receipts, "upload_seconds": round(time.monotonic() - started, 3),
            "verified_before_local_deletion": True}


def fetch_directory(api, repo, prefix, local_dir, manifest, log=print):
    from huggingface_hub import hf_hub_download
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    token = os.environ.get("HF_TOKEN")
    for item in manifest["files"]:
        remote = f"{prefix}/{item['relative_path']}"
        target = local_dir / item["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(repo_id=repo, filename=remote, repo_type="model",
                                     local_dir=str(local_dir / "_hf_stage"), token=token)
        downloaded = Path(downloaded)
        size = downloaded.stat().st_size
        digest = sha256_file(downloaded)
        if size != item["bytes"] or digest != item["sha256"]:
            raise RuntimeError(f"HF fetch identity mismatch BEFORE load for {remote}: bytes={size} sha={digest}")
        if target.exists():
            target.unlink()
        os.replace(downloaded, target)
        log(f"HF_FETCH_VERIFIED {remote} bytes={size}", flush=True)
    return {"prefix": prefix, "files": manifest["files"], "fetch_seconds": round(time.monotonic() - started, 3),
            "verified_before_load": True}
