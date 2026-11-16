import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ["HMOE_CONTRACT_ROOT"])
OUTPUT.mkdir(parents=True, exist_ok=True)
ENV002 = Path(os.environ.get("HMOE_ENV002_ROOT", "/workspace-SR006.nfs3/hmoe-lionmoe-20261116/environment-002"))
STARTED = datetime.datetime.now(datetime.timezone.utc)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with Path(path).open("w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def bounded_error(error):
    text = str(error)
    token_file = Path("/home/jovyan/.cache/huggingface/token")
    for secret in (os.environ.get("HF_TOKEN"), token_file.read_text().strip() if token_file.is_file() else None):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return {"type": type(error).__name__, "message": text[:500]}


test_stdout = OUTPUT / "tests.stdout.log"
test_stderr = OUTPUT / "tests.stderr.log"
test_started = datetime.datetime.now(datetime.timezone.utc)
try:
    with test_stdout.open("w") as stdout, test_stderr.open("w") as stderr:
        tests = subprocess.run([sys.executable, str(ROOT / "tests/stage3_moe/test_lionmuon_energycal.py")],
                               cwd=ROOT, env=dict(os.environ, HMOE_CONTRACT_ROOT=str(OUTPUT)),
                               stdout=stdout, stderr=stderr, timeout=500)
        test_exit = tests.returncode
except Exception as error:
    test_exit = 125
    with test_stderr.open("a") as stream:
        stream.write(f"TEST_PARENT_ERROR {bounded_error(error)}\n")
print(f"CONTRACT003_TEST_EXIT={test_exit}", flush=True)
for label, path in (("STDOUT", test_stdout), ("STDERR", test_stderr)):
    print(f"CONTRACT003_TEST_{label}_BEGIN", flush=True)
    print(path.read_text(errors="replace") if path.exists() else "missing", flush=True)
    print(f"CONTRACT003_TEST_{label}_END", flush=True)
test_report_path = OUTPUT / "contract-results.json"
test_report = json.loads(test_report_path.read_text()) if test_report_path.is_file() else None

previous = json.loads((ENV002 / "preflight-report.json").read_text())
receipts = []
raw_artifacts = []


def capture(path, expected_sha256, expected_bytes, label):
    path = Path(path)
    data = path.read_bytes()
    actual_sha256 = sha256_bytes(data)
    receipt = {"label": label, "path": str(path), "bytes": len(data), "sha256": actual_sha256,
               "expected_bytes": expected_bytes, "expected_sha256": expected_sha256,
               "matches": len(data) == expected_bytes and actual_sha256 == expected_sha256}
    receipts.append(receipt)
    raw_artifacts.append({"label": label, "bytes": len(data), "sha256": actual_sha256,
                          "content_base64": base64.b64encode(data).decode()})
    return receipt


capture(ENV002 / "preflight-report.json", os.environ["HMOE_ENV002_REPORT_SHA256"],
        int(os.environ["HMOE_ENV002_REPORT_BYTES"]), "environment002-preflight-report")
tiny = previous["storage"]["tiny_round_trip"]
capture(ENV002 / "nfs-roundtrip.bin", tiny["sha256"], tiny["bytes"], "environment002-nfs-roundtrip")
capture(ENV002 / "payload-exit.json", os.environ["HMOE_ENV002_PAYLOAD_EXIT_SHA256"],
        int(os.environ["HMOE_ENV002_PAYLOAD_EXIT_BYTES"]), "environment002-payload-exit")
nonpersistent = []
for kind in ("wandb", "tensorboard"):
    for item in previous["logging"][kind]["files"]:
        path = Path(item["path"])
        try:
            path.resolve().relative_to(ENV002.resolve())
        except ValueError:
            nonpersistent.append({"path": str(path), "expected_bytes": item["bytes"],
                                  "expected_sha256": item["sha256"],
                                  "status": "outside_persistent_root_or_symlink_not_read"})
            continue
        capture(path, item["sha256"], item["bytes"], f"environment002-{kind}-{path.name}")
crosspod = {
    "previous_job": "lm-mpi-job-4eb3fcdb-88da-400c-8696-12952010927a",
    "current_host": socket.gethostname(),
    "current_job": os.environ.get("MLSUB_JOB_NAME"),
    "receipt_count": len(receipts),
    "nonpersistent_count": len(nonpersistent),
    "nonpersistent": nonpersistent,
    "all_bytes_and_hashes_match": bool(receipts) and all(item["matches"] for item in receipts),
    "receipts": receipts,
}
write_json(OUTPUT / "crosspod-receipts.json", crosspod)
print("CONTRACT003_CROSSPOD " + json.dumps({"passed": crosspod["all_bytes_and_hashes_match"],
                                           "receipts": len(receipts),
                                           "nonpersistent": len(nonpersistent)}), flush=True)

capacity = []
for candidate in ("/home/jovyan", "/workspace-SR006.nfs2", "/workspace-SR006.nfs3"):
    path = Path(candidate)
    item = {"path": candidate, "exists": path.exists()}
    if path.exists():
        usage = os.statvfs(path)
        item.update({"free_bytes": usage.f_bavail * usage.f_frsize, "free_files": usage.f_favail,
                     "writable_by_process": os.access(path, os.W_OK)})
    capacity.append(item)
capacity_report = {"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "volumes": capacity}
write_json(OUTPUT / "capacity.json", capacity_report)
print("CONTRACT003_CAPACITY " + json.dumps(capacity_report, sort_keys=True), flush=True)

hf_report = {"status": "unavailable", "reason": "no existing HF token in environment or standard user cache",
             "upload_attempted": False}
token_file = Path("/home/jovyan/.cache/huggingface/token")
token = os.environ.get("HF_TOKEN") or (token_file.read_text().strip() if token_file.is_file() else None)
if token:
    nonce = os.urandom(64)
    nonce_hash = sha256_bytes(nonce)
    remote = f"hmoe-lionmoe-20261116/probes/contract-003-{nonce.hex()[:16]}.bin"
    local = OUTPUT / "hf-nonce.bin"
    local.write_bytes(nonce)
    hf_report = {"status": "attempted", "upload_attempted": True, "repo": "AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints",
                 "path": remote, "bytes": len(nonce), "sha256": nonce_hash}
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=token)
        uploaded = api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                                   repo_id=hf_report["repo"], commit_message="contract003 tiny nonce roundtrip")
        info = api.get_paths_info(hf_report["repo"], [remote], expand=True)[0]
        remote_hash = info.lfs.sha256 if info.lfs else None
        restored = Path(hf_hub_download(hf_report["repo"], remote, token=token,
                                        local_dir=str(OUTPUT / "hf-nonce-download")))
        restored_hash = sha256_bytes(restored.read_bytes())
        hf_report.update({"status": "verified", "commit": getattr(uploaded, "oid", None),
                          "commit_url": getattr(uploaded, "commit_url", None), "remote_sha256": remote_hash,
                          "restored_sha256": restored_hash,
                          "round_trip_verified": remote_hash == restored_hash == nonce_hash and restored.stat().st_size == len(nonce)})
    except Exception as error:
        hf_report.update({"status": "unavailable_or_failed", "error": bounded_error(error)})
write_json(OUTPUT / "hf-nonce-receipt.json", hf_report)
print("CONTRACT003_HF_NONCE " + json.dumps(hf_report, sort_keys=True), flush=True)

for label, path in (("contract-results", test_report_path), ("crosspod-receipts", OUTPUT / "crosspod-receipts.json"),
                    ("capacity", OUTPUT / "capacity.json"), ("hf-nonce-receipt", OUTPUT / "hf-nonce-receipt.json")):
    if path.is_file():
        data = path.read_bytes()
        raw_artifacts.append({"label": label, "bytes": len(data), "sha256": sha256_bytes(data),
                              "content_base64": base64.b64encode(data).decode()})
sealed = {"schema": "hmoe-lionmoe-contract003-small-artifacts-v1", "nonsecret": True, "artifacts": raw_artifacts}
sealed_path = OUTPUT / "small-artifacts.json"
write_json(sealed_path, sealed)
sealed_bytes = sealed_path.read_bytes()
if len(sealed_bytes) > 256 * 1024:
    raise RuntimeError(f"Small artifact envelope exceeds 256 KiB: {len(sealed_bytes)}")
encoded = base64.b64encode(sealed_bytes).decode()
print(f"CONTRACT003_ARTIFACT path={sealed_path} bytes={len(sealed_bytes)} sha256={sha256_bytes(sealed_bytes)} "
      f"base64_bytes={len(encoded)}", flush=True)
chunk_size = 12000
chunks = [encoded[index:index + chunk_size] for index in range(0, len(encoded), chunk_size)]
for index, chunk in enumerate(chunks):
    print(f"CONTRACT003_B64 {index}/{len(chunks)} {chunk}", flush=True)
print("CONTRACT003_B64_END", flush=True)

blockers = []
if test_exit != 0 or not test_report or test_report["passed"] != test_report["test_count"]:
    blockers.append("synthetic_contract_failed")
if not crosspod["all_bytes_and_hashes_match"]:
    blockers.append("crosspod_mismatch")
if hf_report.get("status") != "verified":
    blockers.append("hf_small_artifact_channel_unavailable")
finished = datetime.datetime.now(datetime.timezone.utc)
result = {"schema": "hmoe-lionmoe-contract003-result-v1",
          "job_note": os.environ.get("HMOE_JOB_NOTE", "hmoe-lionmoe-20261116-contract-003"),
          "training_invoked": False, "full_model_constructed": False, "real_checkpoint_downloaded": False,
          "test_exit": test_exit, "test_report": str(test_report_path),
          "tests_passed": test_report["passed"] if test_report else 0,
          "test_count": test_report["test_count"] if test_report else 0,
          "crosspod_passed": crosspod["all_bytes_and_hashes_match"], "crosspod_receipts": len(receipts),
          "hf_channel": hf_report.get("status"), "small_artifact_path": str(sealed_path),
          "small_artifact_bytes": len(sealed_bytes), "small_artifact_sha256": sha256_bytes(sealed_bytes),
          "blockers": blockers, "started_at": STARTED.isoformat(), "finished_at": finished.isoformat(),
          "duration_seconds": round((finished - STARTED).total_seconds(), 3)}
write_json(OUTPUT / "contract003-result.json", result)
print("CONTRACT003_RESULT " + json.dumps(result, sort_keys=True), flush=True)
raise SystemExit(1 if "synthetic_contract_failed" in blockers or "crosspod_mismatch" in blockers else 0)
