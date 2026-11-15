"""Checkpoints of the COAT series on the brain_lab gateway, or on HF with COAT_ARCHIVE_HF_PREFIX (the node's /tmp dies
with the job). COAT_FETCH_HF_PREFIX picks the HF folder `fetch` falls back to (default coat-20261002).

  coat_archive.py watch CKPT_DIR ARM_ID DONE_FLAG ITER:KIND...   in the job, next to the trainer
  coat_archive.py probe                                           upload and re-download a small file
  coat_archive.py fetch NAME SHA256 DEST                          bring an archived checkpoint back into a job
  coat_archive.py --receive DIRECTORY                             on the gateway, as the key's forced command

KIND is `full` (the checkpoint as written, optimizer state included) or `model` (optimizer state
dropped, ~2 GB). MCore keeps only the latest checkpoint and the retained pre-decay one, so `watch`
hard-links each wanted iteration the moment the tracker reaches it and uploads from the link.
The key (COAT_ARCHIVE_KEY_B64) may only run the receiver; the route from Cloud.ru to the gateway
drops full-size SSH packets, hence the MSS-limited ProxyCommand (as in monarch-muon-qwen-sft).
"""
import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HOST, PORT, USER = "proxy2.cod.phystech.edu", "10210", "xandi281"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mss_proxy(host, port):
    """stdin/stdout <-> TCP with a 1200-byte MSS."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1200)
    sock.settimeout(30)
    sock.connect((host, int(port)))
    sock.settimeout(None)
    readers = [sock, 0]
    while readers:
        ready, _, _ = select.select(readers, [], [])
        for reader in ready:
            if reader == 0:
                data = os.read(0, 65536)
                if data:
                    sock.sendall(data)
                else:
                    readers.remove(0)
                    sock.shutdown(socket.SHUT_WR)
            else:
                data = sock.recv(65536)
                if not data:
                    raise SystemExit(0)
                while data:
                    data = data[os.write(1, data):]


@contextmanager
def connection():
    with tempfile.TemporaryDirectory(prefix="coat-archive-auth-") as directory:
        key, hosts = Path(directory) / "key", Path(directory) / "known_hosts"
        key.write_bytes(base64.urlsafe_b64decode(os.environ["COAT_ARCHIVE_KEY_B64"]))
        key.chmod(0o600)
        hosts.write_bytes(base64.urlsafe_b64decode(os.environ["COAT_ARCHIVE_HOSTS_B64"]))
        proxy = shlex.join([sys.executable, str(Path(__file__).resolve()), "--mss-proxy", "%h", "%p"])
        yield ["ssh", "-T", "-F", "/dev/null", "-i", str(key), "-p", PORT, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
               "-o", f"ProxyCommand={proxy}", "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={hosts}",
               "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6", f"{USER}@{HOST}"]


def upload(path, name):
    if os.environ.get("COAT_ARCHIVE_HF_PREFIX"):
        return upload_hf(path, name)
    path = Path(path)
    expected, size = file_sha256(path), path.stat().st_size
    with connection() as command:
        command = command + [shlex.join(["upload", name, expected, str(size)])]
        for attempt in range(4):
            with path.open("rb") as stream:
                result = subprocess.run(command, stdin=stream, capture_output=True, text=True, timeout=6 * 3600)
            if result.returncode == 0:
                break
            time.sleep(30 * 2 ** attempt)
        else:
            raise RuntimeError(f"archive transfer failed: {result.stderr[-1500:]}")
    receipt = json.loads(result.stdout)
    if receipt["sha256"] != expected or receipt["bytes"] != size:
        raise RuntimeError("archive receipt does not match the local file")
    return receipt


HF_REPO, HF_PREFIX = "AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints", "coat-20261002"


def upload_hf(path, name):
    """SOW Stage 4 (user 2026-10-19): checkpoints go to HF_REPO under COAT_ARCHIVE_HF_PREFIX (e.g. stage4), one file per
    checkpoint, never replaced or deleted (no LFS history growth). Done only when HF reports the same sha256 and size.
    The quota error of 2026-09-26 (403) is retried for hours, like the other transient failures."""
    from huggingface_hub import HfApi

    path = Path(path)
    expected, size = file_sha256(path), path.stat().st_size
    api = HfApi(token=os.environ["HF_TOKEN"])
    remote = f"{os.environ['COAT_ARCHIVE_HF_PREFIX']}/{name}"
    error = None
    for attempt in range(10):
        try:
            api.upload_file(path_or_fileobj=str(path), path_in_repo=remote, repo_id=HF_REPO, commit_message=f"{remote}")
            info = api.get_paths_info(HF_REPO, [remote], expand=True)[0]
            if info.lfs is not None and info.lfs.sha256 == expected and info.size == size:
                return {"path": f"hf:{HF_REPO}/{remote}", "sha256": expected, "bytes": size}
            error = RuntimeError(f"HF reports {info.lfs.sha256 if info.lfs else None} / {info.size} for {remote}")
        except Exception as caught:
            error = caught
        print(f"COAT_ARCHIVE_HF_RETRY name={name} attempt={attempt} error={error!r}"[:500], flush=True)
        time.sleep(min(3600, 60 * 2 ** attempt))
    raise RuntimeError(f"HF upload of {remote} failed: {error!r}")


def download_hf(name, expected, path):
    """The series' checkpoints were offloaded to HF on 2026-10-06 and the gateway copies deleted after
    verification (archive-manifest.json `hf` field); the public repo is the fallback source."""
    # evalfetch jobs run with HF_HUB_OFFLINE=1 for the cached datasets; the checkpoint itself must come online.
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(key, None)
    from huggingface_hub import hf_hub_download

    token_file = Path(os.environ.get("STAGE3_MOE_HF_TOKEN_FILE", "/home/jovyan/.cache/huggingface/token"))
    token = os.environ.get("HF_TOKEN") or (token_file.read_text().strip() if token_file.is_file() else None)
    with tempfile.TemporaryDirectory(prefix="coat-hf-fetch-", dir=Path(path).parent) as directory:
        # A name with a folder (stage4/NAME) is taken as the repo path, so one job can mix series.
        remote = name if "/" in name else f"{os.environ.get('COAT_FETCH_HF_PREFIX', HF_PREFIX)}/{name}"
        fetched = hf_hub_download(HF_REPO, remote, local_dir=directory, token=token)
        if file_sha256(fetched) != expected:
            raise RuntimeError(f"HF download of {name} does not match the expected sha256")
        shutil.move(fetched, path)


def download(name, expected, path):
    if "COAT_ARCHIVE_KEY_B64" in os.environ:
        try:
            with connection() as command, Path(path).open("wb") as stream:
                result = subprocess.run(command + [shlex.join(["download", name, expected])], stdout=stream, stderr=subprocess.PIPE, timeout=6 * 3600)
            if result.returncode == 0 and file_sha256(path) == expected:
                print("COAT_FETCH_SOURCE=gateway", flush=True)
                return
            print("COAT_FETCH_GATEWAY_MISS " + result.stderr.decode()[-300:].strip().replace("\n", " "), flush=True)
        except Exception as error:  # the gateway copy may be gone: fall back to HF
            print(f"COAT_FETCH_GATEWAY_MISS {error!r}"[:400], flush=True)
    download_hf(name, expected, path)
    print("COAT_FETCH_SOURCE=hf", flush=True)


def receive(root):
    fields = shlex.split(os.environ.get("SSH_ORIGINAL_COMMAND", ""))
    if not fields or fields[0] not in ("upload", "download") or len(fields) != (4 if fields[0] == "upload" else 3):
        raise ValueError("only artifact transfers are allowed")
    name, expected = fields[1:3]
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,120}\.(pt|json|jsonl|bin)", name) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("invalid artifact name or checksum")
    target = Path(root) / name
    if fields[0] == "download":
        if file_sha256(target) != expected:
            raise ValueError("stored artifact checksum mismatch")
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
                sys.stdout.buffer.write(chunk)
        return
    size = int(fields[3])
    if not 0 <= size <= 40 * 1024**3:
        raise ValueError("artifact size exceeds this receiver's limit")
    if target.exists():
        if file_sha256(target) != expected:
            raise FileExistsError("a different artifact already has this name")
        sys.stdin.buffer.read()  # already archived: drain and acknowledge
        print(json.dumps({"path": str(target), "sha256": expected, "bytes": size}), flush=True)
        return
    digest, temporary = hashlib.sha256(), None
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix=".upload-", delete=False) as stream:
            temporary = Path(stream.name)
            remaining = size
            while remaining:
                chunk = sys.stdin.buffer.read(min(4 * 1024**2, remaining))
                if not chunk:
                    raise EOFError("incomplete artifact")
                stream.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if digest.hexdigest() != expected:
            raise ValueError("received checksum mismatch")
        temporary.replace(target)
        print(json.dumps({"path": str(target), "sha256": expected, "bytes": size}), flush=True)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def watch(checkpoints, arm_id, done_flag, wanted):
    """Archive each wanted iteration; ends once the trainer is done and nothing wanted can still appear."""
    checkpoints, done_flag = Path(checkpoints), Path(done_flag)
    staging = checkpoints.parent / f"archive-staging-{arm_id}"
    staging.mkdir(parents=True, exist_ok=True)
    pending = dict(wanted)
    while pending:
        finished = done_flag.exists()
        tracker = checkpoints / "latest_checkpointed_iteration.txt"
        latest = int(tracker.read_text().strip() or 0) if tracker.is_file() else 0
        for iteration, kind in sorted(pending.items()):
            source = checkpoints / f"iter_{iteration:07d}" / "mp_rank_00" / "model_optim_rng.pt"
            if latest < iteration or not source.is_file():
                continue
            staged = staging / f"iter{iteration:07d}.pt"
            stripped = staging / f"iter{iteration:07d}-model.pt"
            name = f"{arm_id}-iter{iteration:07d}-{kind}.pt"
            try:
                if not staged.exists():
                    os.link(source, staged)  # survives MCore dropping the checkpoint
                if kind == "model":
                    import torch

                    state = torch.load(staged, map_location="cpu", weights_only=False)
                    state.pop("optimizer", None)
                    torch.save(state, stripped)
                    del state
                    receipt = upload(stripped, name)
                    stripped.unlink()
                else:
                    receipt = upload(staged, name)
                staged.unlink()
                (staging / f"done-{iteration}").touch()
                print("COAT_ARCHIVE " + json.dumps({"arm_id": arm_id, "iteration": iteration, "kind": kind, **receipt}), flush=True)
            except Exception as error:
                # The job's /tmp dies with it: keep the file on the shared volume so a failed transfer loses nothing.
                kept = None
                fallback = os.environ.get("COAT_ARCHIVE_FALLBACK_DIR")
                if fallback:
                    for candidate in ((stripped, staged) if kind == "model" else (staged,)):
                        if candidate.exists():
                            Path(fallback).mkdir(parents=True, exist_ok=True)
                            kept = Path(fallback) / name
                            shutil.copyfile(candidate, kept)
                            break
                print("COAT_ARCHIVE_FAILED " + json.dumps({"arm_id": arm_id, "iteration": iteration, "kind": kind, "error": repr(error)[:600], "kept": str(kept) if kept else None}), flush=True)
            pending.pop(iteration)
        if finished:
            break
        time.sleep(30)
    for iteration, kind in sorted(pending.items()):
        print("COAT_ARCHIVE_MISSED " + json.dumps({"arm_id": arm_id, "iteration": iteration, "kind": kind}), flush=True)


def probe():
    with tempfile.TemporaryDirectory(prefix="coat-archive-probe-") as directory:
        path = Path(directory) / "probe.bin"
        path.write_bytes(os.urandom(64 * 1024**2))
        started = time.time()
        receipt = upload(path, f"probe-{int(started)}.bin")
        seconds = time.time() - started
        download(Path(receipt["path"]).name, receipt["sha256"], Path(directory) / "back.bin")
        print("COAT_ARCHIVE_PROBE " + json.dumps({**receipt, "upload_mb_per_s": round(64 / seconds, 2)}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--mss-proxy":
        mss_proxy(*sys.argv[2:4])
    elif sys.argv[1] == "--receive":
        receive(sys.argv[2])
    elif sys.argv[1] == "probe":
        probe()
    elif sys.argv[1] == "fetch":
        started = time.time()
        download(sys.argv[2], sys.argv[3], sys.argv[4])
        print("COAT_FETCH " + json.dumps({"name": sys.argv[2], "bytes": Path(sys.argv[4]).stat().st_size,
                                           "mb_per_s": round(Path(sys.argv[4]).stat().st_size / 2**20 / (time.time() - started), 2)}), flush=True)
    elif sys.argv[1] == "watch":
        watch(sys.argv[2], sys.argv[3], sys.argv[4], {int(i): k for i, k in (item.split(":") for item in sys.argv[5:])})
    else:
        raise SystemExit(__doc__)
