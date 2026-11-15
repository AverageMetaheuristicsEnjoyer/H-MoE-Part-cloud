import datetime
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
output_root = Path(os.environ["HMOE_PREFLIGHT_ROOT"])
output_root.mkdir(parents=True, exist_ok=True)
started = datetime.datetime.now(datetime.timezone.utc)
report = {
    "schema": "hmoe-lionmoe-preflight-001-v1",
    "started_at": started.isoformat(),
    "training_invoked": False,
    "identity": {
        key: os.environ.get(key)
        for key in (
            "MLSUB_JOB_NAME",
            "MLSUB_IMAGE",
            "BUNDLE_ID",
            "EXPECTED_SOURCE_COMMIT",
            "EXPECTED_CARRIER_COMMIT",
            "EXPECTED_PAYLOAD_SHA256",
            "HMOE_PREFLIGHT_ROOT",
            "STAGE3_MOE_NPROC",
            "STAGE3_MOE_MICRO_BATCH",
            "STAGE3_MOE_GLOBAL_BATCH",
        )
    },
    "blockers": [],
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_error(error):
    text = str(error)
    secrets = [os.environ.get(name) for name in ("HF_TOKEN", "BUNDLE_KEY", "COAT_ARCHIVE_KEY_B64", "COAT_ARCHIVE_HOSTS_B64")]
    token_file = Path(os.environ.get("STAGE3_MOE_HF_TOKEN_FILE", "/home/jovyan/.cache/huggingface/token"))
    if token_file.is_file():
        secrets.append(token_file.read_text().strip())
    for value in secrets:
        if value:
            text = text.replace(value, "[REDACTED]")
    return {"type": type(error).__name__, "message": text[:500]}


def run_bounded(command, timeout=15):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {
            "exit": result.returncode,
            "stdout": result.stdout.strip().splitlines()[-30:],
            "stderr": result.stderr.strip().splitlines()[-10:],
        }
    except Exception as error:
        return {"error": safe_error(error)}


def file_metadata(path):
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": stat.st_size,
        "mtime_utc": datetime.datetime.fromtimestamp(stat.st_mtime, datetime.timezone.utc).isoformat(),
    }


report["environment"] = {
    "python": sys.version,
    "packages": {},
    "nvidia_smi": run_bounded([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader",
    ]),
}
for package in (
    "torch",
    "transformer-engine",
    "transformer-engine-torch",
    "huggingface-hub",
    "wandb",
    "tensorboard",
    "cryptography",
    "lm-eval",
):
    try:
        report["environment"]["packages"][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        report["environment"]["packages"][package] = None
try:
    torch = importlib.import_module("torch")
    report["environment"]["torch"] = {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "cudnn_version": torch.backends.cudnn.version(),
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ],
    }
except Exception as error:
    report["environment"]["torch_error"] = safe_error(error)
    report["blockers"].append("torch_or_cuda_unavailable")
try:
    transformer_engine = importlib.import_module("transformer_engine")
    report["environment"]["transformer_engine"] = {"version": getattr(transformer_engine, "__version__", None)}
    transformer_engine_torch = importlib.import_module("transformer_engine_torch")
    report["environment"]["transformer_engine"]["cublaslt_version"] = transformer_engine_torch.get_cublasLt_version()
except Exception as error:
    report["environment"]["transformer_engine_error"] = safe_error(error)
    report["blockers"].append("transformer_engine_unavailable")
try:
    importlib.import_module("wandb")
    importlib.import_module("torch.utils.tensorboard")
    report["environment"]["logger_imports"] = True
except Exception as error:
    report["environment"]["logger_imports"] = False
    report["environment"]["logger_error"] = safe_error(error)
    report["blockers"].append("logging_dependencies_missing")

report["storage"] = {"volumes": []}
for candidate in ("/home/jovyan", "/workspace-SR006.nfs2", "/workspace-SR006.nfs3"):
    path = Path(candidate)
    item = {"path": candidate, "exists": path.exists()}
    if path.exists():
        usage = os.statvfs(path)
        item.update({
            "free_bytes": usage.f_bavail * usage.f_frsize,
            "free_files": usage.f_favail,
            "writable_by_process": os.access(path, os.W_OK),
        })
    report["storage"]["volumes"].append(item)
probe_path = output_root / "nfs-roundtrip.bin"
probe_payload = b"hmoe-lionmoe-20261116-preflight-001\n" + started.isoformat().encode() + b"\n"
try:
    probe_path.write_bytes(probe_payload)
    restored = probe_path.read_bytes()
    report["storage"]["tiny_round_trip"] = {
        "path": str(probe_path),
        "bytes": len(probe_payload),
        "sha256": hashlib.sha256(probe_payload).hexdigest(),
        "readback_equal": restored == probe_payload,
    }
    if restored != probe_payload:
        report["blockers"].append("nfs_readback_mismatch")
except Exception as error:
    report["storage"]["tiny_round_trip"] = {"error": safe_error(error)}
    report["blockers"].append("nfs_write_failed")

corpus_roots = []
for candidate in (
    os.environ.get("STAGE3_MOE_DATA_ROOT"),
    "/home/jovyan/data/fineweb-edu-gpt2-megatron/data",
    "/workspace-SR006.nfs2/hmoe-data/fineweb-edu-gpt2-megatron/data",
    "/workspace-SR006.nfs3/hmoe-data/fineweb-edu-gpt2-megatron/data",
):
    if candidate and candidate not in corpus_roots:
        corpus_roots.append(candidate)
report["corpus"] = {
    "historical_manifest_sha256": "7a7cf91633081dd080f3cff96407bd61d4c237d88a5e926cffe36fc5b8cbfcc8",
    "historical_tree_sha256": "fe0dc4130d96371f39dadada4356a8f745ac6a1e1ff09ae79d11fa664d24d297",
    "payload_hashes_recomputed": False,
    "payload_hash_note": "Bounded preflight compares metadata and the small manifest; full corpus hashes remain unverified.",
    "candidates": [],
}
for candidate in corpus_roots:
    data_root = Path(candidate)
    entry = {"data_root": candidate, "files": {}}
    for split in ("train", "development", "final"):
        for extension in ("bin", "idx"):
            entry["files"][f"{split}.{extension}"] = file_metadata(data_root / f"{split}.{extension}")
    manifest_path = data_root.parent / "artifact-manifest.json"
    entry["manifest"] = file_metadata(manifest_path)
    if manifest_path.is_file():
        manifest_bytes = manifest_path.read_bytes()
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        entry["manifest"]["sha256"] = manifest_hash
        entry["manifest"]["matches_historical"] = manifest_hash == report["corpus"]["historical_manifest_sha256"]
        try:
            manifest = json.loads(manifest_bytes)
            records = []

            def visit(value):
                if isinstance(value, dict):
                    if "sha256" in value and any(key in value for key in ("path", "name", "filename")):
                        records.append(value)
                    for child in value.values():
                        visit(child)
                elif isinstance(value, list):
                    for child in value:
                        visit(child)

            visit(manifest)
            entry["recorded_payload_entries"] = [
                record for record in records
                if any(split in str(record) for split in ("train.bin", "train.idx", "development.bin", "development.idx", "final.bin", "final.idx"))
            ]
        except Exception as error:
            entry["manifest_error"] = safe_error(error)
    report["corpus"]["candidates"].append(entry)
selected = next((entry for entry in report["corpus"]["candidates"] if entry["files"]["train.bin"]["exists"]), None)
report["corpus"]["selected_data_root"] = selected["data_root"] if selected else None
if selected is None:
    report["blockers"].append("corpus_not_found")
elif not all(item["exists"] for item in selected["files"].values()):
    report["blockers"].append("corpus_incomplete")
elif not selected["manifest"]["exists"]:
    report["blockers"].append("corpus_manifest_missing")
elif not selected["manifest"].get("matches_historical", False):
    report["blockers"].append("corpus_manifest_mismatch")

checkpoint_name = "stage4/muon-ctrl-lm5-iter0017242-model.pt"
checkpoint_sha = "3cf8ec64542f262b05ecabdeac7a6f530a76a5907d80d117247f2ff49b4584b1"
report["checkpoint"] = {
    "repo": "AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints",
    "path": checkpoint_name,
    "expected_sha256": checkpoint_sha,
    "download_attempted": False,
}
checkpoint_program = r'''
import json, os, pathlib, sys
from huggingface_hub import HfApi
token_file = pathlib.Path(os.environ.get("STAGE3_MOE_HF_TOKEN_FILE", "/home/jovyan/.cache/huggingface/token"))
token = os.environ.get("HF_TOKEN") or (token_file.read_text().strip() if token_file.is_file() else None)
info = HfApi(token=token).get_paths_info(sys.argv[1], [sys.argv[2]], expand=True)[0]
print(json.dumps({"bytes": info.size, "remote_sha256": info.lfs.sha256 if info.lfs else None}))
'''
metadata_result = run_bounded(
    [sys.executable, "-c", checkpoint_program, report["checkpoint"]["repo"], checkpoint_name],
    timeout=30,
)
if metadata_result.get("exit") == 0 and metadata_result.get("stdout"):
    try:
        metadata = json.loads(metadata_result["stdout"][-1])
        report["checkpoint"].update({
            "reachable": True,
            **metadata,
            "sha256_matches": metadata["remote_sha256"] == checkpoint_sha,
        })
        if metadata["remote_sha256"] != checkpoint_sha:
            report["blockers"].append("checkpoint_hash_mismatch")
    except Exception as error:
        report["checkpoint"].update({"reachable": False, "error": safe_error(error)})
        report["blockers"].append("checkpoint_metadata_invalid")
else:
    if "stderr" in metadata_result:
        metadata_result["stderr"] = [safe_error(Exception(line))["message"] for line in metadata_result["stderr"]]
    report["checkpoint"].update({"reachable": False, "metadata_result": metadata_result})
    report["blockers"].append("checkpoint_metadata_unreachable")

report["transfer"] = {
    "contract": "forced-command SSH, pinned host keys, TCP_MAXSEG=1200, size+SHA256 receipt, then download and local hash comparison",
    "credentials_supplied": bool(os.environ.get("COAT_ARCHIVE_KEY_B64") and os.environ.get("COAT_ARCHIVE_HOSTS_B64")),
    "attempted": False,
}
if report["transfer"]["credentials_supplied"]:
    sys.path.insert(0, str(root / "scripts"))
    import coat_archive

    transfer_path = output_root / "cloud-brainlab-roundtrip.bin"
    transfer_payload = b"hmoe-lionmoe-transfer-preflight-001\n" + started.isoformat().encode() + b"\n"
    transfer_path.write_bytes(transfer_payload)
    transfer_sha = hashlib.sha256(transfer_payload).hexdigest()
    transfer_name = f"hmoe-lionmoe-20261116-preflight-{int(started.timestamp())}.bin"
    report["transfer"].update({"attempted": True, "name": transfer_name, "bytes": len(transfer_payload), "sha256": transfer_sha})
    try:
        with coat_archive.connection() as connection:
            upload_command = connection + [shlex.join(["upload", transfer_name, transfer_sha, str(len(transfer_payload))])]
            with transfer_path.open("rb") as stream:
                uploaded = subprocess.run(upload_command, stdin=stream, capture_output=True, text=True, timeout=60)
            uploaded.check_returncode()
            receipt = json.loads(uploaded.stdout)
            if receipt.get("sha256") != transfer_sha or receipt.get("bytes") != len(transfer_payload):
                raise RuntimeError("upload receipt mismatch")
            restored_path = output_root / "cloud-brainlab-restored.bin"
            download_command = connection + [shlex.join(["download", transfer_name, transfer_sha])]
            with restored_path.open("wb") as stream:
                downloaded = subprocess.run(download_command, stdout=stream, stderr=subprocess.PIPE, timeout=60)
            downloaded.check_returncode()
            restored_sha = sha256(restored_path)
            report["transfer"].update({
                "receipt_sha256": receipt["sha256"],
                "receipt_bytes": receipt["bytes"],
                "restored_sha256": restored_sha,
                "round_trip_verified": restored_sha == transfer_sha and restored_path.stat().st_size == len(transfer_payload),
            })
            if not report["transfer"]["round_trip_verified"]:
                report["blockers"].append("transfer_readback_mismatch")
    except Exception as error:
        report["transfer"]["error"] = safe_error(error)
        report["blockers"].append("transfer_roundtrip_failed")
else:
    report["transfer"]["status"] = "unverified_missing_credentials"
    report["blockers"].append("transfer_credentials_not_supplied")

report["status"] = "completed_with_blockers" if report["blockers"] else "passed"
report["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
report_path = output_root / "preflight-report.json"
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print("PREFLIGHT_JSON " + json.dumps(report, sort_keys=True), flush=True)
print(f"PREFLIGHT_REPORT={report_path}", flush=True)
print(f"PREFLIGHT_STATUS={report['status']} blockers={','.join(report['blockers']) or 'none'}", flush=True)
print("PREFLIGHT_EXIT=0 (probe completed; this is not a readiness claim)", flush=True)
