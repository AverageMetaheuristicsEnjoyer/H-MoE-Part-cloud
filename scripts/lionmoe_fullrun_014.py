import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ["HMOE_CONTRACT_ROOT"])
VERSION = os.environ.get("HMOE_FULLRUN_VERSION", "026")
EXPECTED_NOTE = f"hmoe-lionmoe-20261117-fullrun-014-v{VERSION}"
EXPECTED_ROOT = f"/workspace-SR006.nfs3/hmoe-lionmoe-20261117/fullrun-014-v{VERSION}"
EXPECTED_BUNDLE = f"hmoe-lionmoe-20261116-v{VERSION}"
if os.environ.get("HMOE_JOB_NOTE") != EXPECTED_NOTE:
    raise RuntimeError("Fullrun014 note mismatch")
if os.environ.get("HMOE_CONTRACT_ROOT") != EXPECTED_ROOT:
    raise RuntimeError("Fullrun014 root mismatch")
if os.environ.get("EXPECTED_BUNDLE_ID") != EXPECTED_BUNDLE:
    raise RuntimeError("Fullrun014 bundle mismatch")
for key, length in (("EXPECTED_SOURCE_COMMIT", 40), ("EXPECTED_CARRIER_COMMIT", 40), ("EXPECTED_PAYLOAD_SHA256", 64)):
    value = os.environ.get(key, "")
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"Missing/invalid runtime pin: {key}")
OUTPUT.mkdir(parents=True, exist_ok=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def mount_table():
    rows = []
    path = Path("/proc/self/mountinfo")
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 5 or len(right_fields) < 2:
            continue
        mountpoint = left_fields[4].replace("\\040", " ")
        rows.append({"mountpoint": mountpoint, "filesystem": right_fields[0], "source": right_fields[1]})
    return rows


def matching_mount(path, mounts):
    matches = [row for row in mounts if str(path) == row["mountpoint"] or str(path).startswith(row["mountpoint"].rstrip("/") + "/")]
    return max(matches, key=lambda row: len(row["mountpoint"])) if matches else None


def quota_evidence():
    executable = shutil.which("quota")
    result = {"available": bool(executable)}
    if executable:
        try:
            process = subprocess.run([executable, "-w"], capture_output=True, text=True, timeout=15)
            result.update(exit=process.returncode, stdout=process.stdout[-8000:], stderr=process.stderr[-4000:])
        except Exception as error:
            result.update(error=f"{type(error).__name__}: {error}")
    return result


def capacity():
    mounts = mount_table()
    specs = [
        ("/workspace-SR006.nfs3", "durable_cloud_nfs_candidate"),
        ("/workspace-SR006.nfs2", "durable_cloud_nfs_candidate"),
        ("/home/jovyan", "container_home_unless_mount_proves_durable"),
        ("/tmp", "ephemeral_job_local"),
        ("/dev/shm", "ephemeral_shared_memory"),
    ]
    rows = []
    for path, classification in specs:
        root = Path(path)
        row = {"path": path, "classification": classification, "exists": root.exists(),
               "writable_by_process": os.access(path, os.W_OK)}
        if root.exists():
            usage = os.statvfs(root)
            row.update(device=root.stat().st_dev, block_size=usage.f_frsize,
                       total_bytes=usage.f_blocks * usage.f_frsize,
                       free_bytes=usage.f_bfree * usage.f_frsize,
                       available_bytes=usage.f_bavail * usage.f_frsize,
                       total_inodes=usage.f_files, free_inodes=usage.f_ffree,
                       available_inodes=usage.f_favail,
                       mount=matching_mount(path, mounts))
            probe = root / "hmoe-lionmoe-20261117" / f"fullrun-014-v{VERSION}-capacity-probe"
            try:
                probe.mkdir(parents=True, exist_ok=True)
                probe_file = probe / "fsync-roundtrip.bin"
                payload = f"fullrun014-{VERSION}-{path}\n".encode()
                with probe_file.open("wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                if probe_file.read_bytes() != payload:
                    raise RuntimeError("tiny readback mismatch")
                row["tiny_fsync_readback"] = artifact(probe_file)
            except Exception as error:
                row["tiny_fsync_error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    master = 4_115_707_904
    optimizer = 4_532_039_680
    bf16_model = master // 2
    metadata_allowance = 64 * 2**20
    logical = master + optimizer + bf16_model + metadata_allowance
    checkpoint_budget = (logical * 110 + 99) // 100
    logs_caches = 4 * 2**30
    safety = max(4 * 2**30, checkpoint_budget)
    study_peak = 6 * checkpoint_budget + logs_caches + safety
    active_save_guard = 3 * checkpoint_budget + logs_caches + safety
    brainlab = [
        {"path": "/home/xandi281", "available_bytes": int(os.environ["HMOE_BRAINLAB_HOME_AVAILABLE_BYTES"])},
        {"path": "/tmp", "available_bytes": int(os.environ["HMOE_BRAINLAB_TMP_AVAILABLE_BYTES"])},
    ]
    for row in brainlab:
        row.update(durable_archive_candidate=False, estimated_peak_deficit_bytes=max(0, study_peak - row["available_bytes"]),
                   reason="not a verified durable full-state archive; capacity also below planned peak")
    candidates = []
    for row in rows:
        mount = row.get("mount") or {}
        durable = row["classification"] == "durable_cloud_nfs_candidate" and "nfs" in mount.get("filesystem", "").lower()
        available = row.get("available_bytes", 0)
        candidates.append({"path": row["path"], "durable_candidate": durable, "available_bytes": available,
                           "estimated_study_peak_deficit_bytes": max(0, study_peak - available),
                           "estimated_capacity_sufficient": durable and available >= study_peak and row.get("available_inodes", 0) >= 300000,
                           "not_an_admission": "actual serialized full-state save/new-allocation restore still required"})
    return {
        "schema": "hmoe-fullrun014-capacity-v1",
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "volumes": rows, "brainlab": brainlab, "quota": quota_evidence(),
        "ledger_estimate": {
            "fp32_master_bytes": master, "persistent_optimizer_bytes": optimizer,
            "bf16_model_bytes_derived_from_master_numel": bf16_model,
            "rng_scheduler_iteration_controller_config_allowance_bytes": metadata_allowance,
            "serialization_overhead_percent": 10,
            "one_full_state_budget_bytes": checkpoint_budget,
            "retained_full_state_equivalents_at_study_peak": 6,
            "logs_caches_bytes": logs_caches, "additional_safety_bytes": safety,
            "study_peak_budget_bytes": study_peak,
            "active_save_guard_budget_bytes": active_save_guard,
            "actual_serialized_size_verified": False,
        },
        "candidates": candidates,
        "reservations": {"explicit_quota_or_reservation_api": "not established",
                         "free_space_is_not_reserved": True,
                         "other_active_projects_outputs_excluded": True,
                         "runtime_guard_required_before_every_save": True},
        "hf_direct_hmoe_archive": "existing H-MoE credentials/route not supplied or verified; not used",
        "long_run_admitted": False,
    }


def data_identity():
    corpus = Path("/home/jovyan/data/fineweb-edu-gpt2-megatron/data")
    expected_sizes = {"train.bin": 15113107020, "train.idx": 145805762,
                      "development.bin": 16000532, "development.idx": 152022,
                      "final.bin": 200002434, "final.idx": 1936642}
    corpus_files = {}
    for name, expected in expected_sizes.items():
        path = corpus / name
        if not path.is_file() or path.stat().st_size != expected:
            raise RuntimeError(f"Corpus identity mismatch: {name}")
        corpus_files[name] = {"bytes": expected}
    manifest = corpus.parent / "artifact-manifest.json"
    if sha256(manifest) != "7a7cf91633081dd080f3cff96407bd61d4c237d88a5e926cffe36fc5b8cbfcc8":
        raise RuntimeError("Corpus manifest identity mismatch")
    versions = ["postwarmup-012-v019", "overnight-013-v020", "overnight-013-v023",
                "overnight-013-v024", "overnight-013-v025"]
    cache_receipts = []
    run_identities = []
    total_hashed = 0
    for version in versions:
        root = Path("/workspace-SR006.nfs3/hmoe-lionmoe-20261116") / version
        for arm in sorted(root.glob("arms/*")):
            cache = arm / "data-cache"
            cache_row = {"version": version, "arm": arm.name, "path": str(cache), "exists": cache.is_dir()}
            cache_receipts.append(cache_row)
            if cache.is_dir():
                files = []
                for path in sorted(cache.rglob("*")):
                    if path.is_symlink() or not path.is_file():
                        continue
                    size = path.stat().st_size
                    if size > 2**30 or total_hashed + size > 4 * 2**30:
                        raise RuntimeError("Data-cache hashing bound exceeded")
                    digest = sha256(path)
                    total_hashed += size
                    files.append({"relative_path": str(path.relative_to(cache)), "bytes": size, "sha256": digest})
                cache_row["files"] = files
            for result in sorted((arm / "logs").glob("*/results.jsonl")):
                record = json.loads(result.read_text().splitlines()[-1])
                argv = record.get("provenance", {}).get("argv", [])
                selected = {}
                for option in ("--train-iters", "--lr-decay-iters", "--lr-warmup-iters", "--lr-wsd-decay-iters",
                               "--eval-interval", "--eval-iters", "--seed"):
                    if option in argv:
                        selected[option] = argv[argv.index(option) + 1]
                run_identities.append({"version": version, "arm": arm.name, "result": artifact(result), "selected_argv": selected})
    identity_groups = {}
    for row in cache_receipts:
        for item in row.get("files", []):
            key = item["relative_path"]
            identity_groups.setdefault(key, {}).setdefault(item["sha256"], []).append(f"{row['version']}:{row['arm']}")
    return {
        "schema": "hmoe-fullrun014-data-identity-v1",
        "corpus_root": str(corpus), "corpus_files": corpus_files,
        "corpus_manifest": artifact(manifest), "cache_receipts": cache_receipts,
        "cache_identity_groups": identity_groups, "run_identities": run_identities,
        "total_cache_bytes_hashed": total_hashed,
        "scope": "metadata and index-cache bytes only; no raw corpus/tensors exported",
        "actual_fullrun_batch_fingerprints_verified": False,
    }


def run_telemetry():
    stdout_path = OUTPUT / "telemetry.stdout.log"
    stderr_path = OUTPUT / "telemetry.stderr.log"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.run([sys.executable, str(ROOT / "tests/stage3_moe/test_fullrun_014_telemetry.py")],
                                 cwd=ROOT, env=dict(os.environ), stdout=stdout, stderr=stderr, timeout=240)
    result_path = OUTPUT / "telemetry-contracts.json"
    if process.returncode != 0 or not result_path.is_file():
        raise RuntimeError("Telemetry contract failed: " + stderr_path.read_text(errors="replace")[-8000:])
    result = json.loads(result_path.read_text())
    if not result.get("passed"):
        raise RuntimeError("Telemetry contract did not pass")
    return result


def seal(result, paths):
    artifacts = []
    for label, path in paths:
        data = Path(path).read_bytes()
        artifacts.append({"label": label, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                          "content_base64": base64.b64encode(data).decode()})
    sealed_path = OUTPUT / "small-artifacts.json"
    write_json(sealed_path, {"schema": "hmoe-fullrun014-small-artifacts-v1", "nonsecret": True, "artifacts": artifacts})
    sealed = sealed_path.read_bytes()
    if len(sealed) > 256 * 1024:
        raise RuntimeError("Fullrun014 compact report exceeds256KiB")
    encoded = base64.b64encode(sealed).decode()
    print(f"FULLRUN014_ARTIFACT path={sealed_path} bytes={len(sealed)} sha256={hashlib.sha256(sealed).hexdigest()} base64_bytes={len(encoded)}", flush=True)
    chunks = [encoded[index:index + 12000] for index in range(0, len(encoded), 12000)]
    for index, chunk in enumerate(chunks):
        print(f"FULLRUN014_B64 {index}/{len(chunks)} {chunk}", flush=True)
    print("FULLRUN014_B64_END", flush=True)
    print("FULLRUN014_RESULT " + json.dumps(result, sort_keys=True), flush=True)


def main_checked():
    capacity_path = OUTPUT / "capacity-report.json"
    data_path = OUTPUT / "data-identity.json"
    write_json(capacity_path, capacity())
    write_json(data_path, data_identity())
    telemetry = run_telemetry()
    capacity_result = json.loads(capacity_path.read_text())
    summary = {
        "schema": "hmoe-fullrun014-admission-audit-v1", "audit_completed": True,
        "capacity_candidates": capacity_result["candidates"],
        "estimated_study_peak_bytes": capacity_result["ledger_estimate"]["study_peak_budget_bytes"],
        "telemetry_passed": telemetry["passed"], "data_identity_collected": True,
        "long_run_admitted": False, "training_invoked": False, "model_constructed_for_training": False,
        "checkpoint_saved": False, "restore_verified": False,
        "identity": {key: os.environ[key] for key in ("EXPECTED_SOURCE_COMMIT", "EXPECTED_CARRIER_COMMIT",
                                                       "EXPECTED_BUNDLE_ID", "EXPECTED_PAYLOAD_SHA256")},
    }
    summary_path = OUTPUT / "admission-audit.json"
    write_json(summary_path, summary)
    seal(summary, [("admission-audit", summary_path), ("capacity-report", capacity_path),
                   ("telemetry-contracts", OUTPUT / "telemetry-contracts.json"), ("data-identity", data_path)])
    return 0


def main():
    try:
        return main_checked()
    except Exception as error:
        partial = {"schema": "hmoe-fullrun014-error-v1", "audit_completed": False,
                   "long_run_admitted": False, "training_invoked": False,
                   "error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc()[-8000:]}
        path = OUTPUT / "partial-error.json"
        write_json(path, partial)
        available = []
        for label, candidate in (("capacity-report", OUTPUT / "capacity-report.json"),
                                 ("data-identity", OUTPUT / "data-identity.json"),
                                 ("telemetry-contracts", OUTPUT / "telemetry-contracts.json")):
            if candidate.is_file():
                available.append((label, candidate))
        partial["available_artifacts"] = [artifact(candidate) for _, candidate in available]
        write_json(path, partial)
        seal(partial, [("partial-error", path), *available])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
