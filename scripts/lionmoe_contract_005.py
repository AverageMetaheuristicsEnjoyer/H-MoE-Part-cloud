import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ["HMOE_CONTRACT_ROOT"])
OUTPUT.mkdir(parents=True, exist_ok=True)
STARTED = datetime.datetime.now(datetime.timezone.utc)


def write_json(path, value):
    with Path(path).open("w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_suite(script, label):
    stdout_path = OUTPUT / f"{label}.stdout.log"
    stderr_path = OUTPUT / f"{label}.stderr.log"
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT,
                                    env=dict(os.environ, HMOE_CONTRACT_ROOT=str(OUTPUT)),
                                    stdout=stdout, stderr=stderr, timeout=250)
            code = result.returncode
    except Exception as error:
        code = 125
        with stderr_path.open("a") as stream:
            stream.write(f"SUITE_PARENT_ERROR {type(error).__name__}: {error}\n")
    print(f"CONTRACT005_SUITE label={label} exit={code}", flush=True)
    for stream_label, path in (("STDOUT", stdout_path), ("STDERR", stderr_path)):
        print(f"CONTRACT005_{label}_{stream_label}_BEGIN", flush=True)
        print(path.read_text(errors="replace") if path.exists() else "missing", flush=True)
        print(f"CONTRACT005_{label}_{stream_label}_END", flush=True)
    return code


controlled_exit = run_suite("tests/stage3_moe/test_lionmuon_energycal.py", "controlled")
production_exit = run_suite("tests/stage3_moe/test_lionmuon_energycal_production.py", "production")
controlled_path = OUTPUT / "contract-results.json"
production_path = OUTPUT / "production-results.json"
controlled = json.loads(controlled_path.read_text()) if controlled_path.is_file() else None
production = json.loads(production_path.read_text()) if production_path.is_file() else None
passed = bool(controlled_exit == 0 and production_exit == 0 and controlled and production
              and controlled["test_count"] == 9 and production["test_count"] == 5
              and controlled["passed"] == controlled["test_count"]
              and production["passed"] == production["test_count"])
coverage = {
    "controlled_injected_direction": bool(controlled and controlled["passed"] == controlled["test_count"]),
    "real_ns5_and_pinned_original": bool(production and production["passed"] == production["test_count"]),
    "real_mcore_grouping_and_wrapper": bool(production and production["passed"] == production["test_count"]),
    "full_mcore_trainer_resume": False,
    "full_model_quality_speed_memory": False,
    "hf_receipt_fixtures_cloud_only": bool(controlled and any(
        item["name"] == "test_hf_receipt_fixtures" and item["status"] == "passed" for item in controlled["results"])),
}
raw_artifacts = []
for label, path in (("controlled-results", controlled_path), ("production-results", production_path)):
    if path.is_file():
        data = path.read_bytes()
        raw_artifacts.append({"label": label, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                              "content_base64": base64.b64encode(data).decode()})
sealed = {"schema": "hmoe-lionmoe-contract005-small-artifacts-v1", "nonsecret": True, "artifacts": raw_artifacts}
sealed_path = OUTPUT / "small-artifacts.json"
write_json(sealed_path, sealed)
sealed_bytes = sealed_path.read_bytes()
if len(sealed_bytes) > 256 * 1024:
    raise RuntimeError(f"Small artifact envelope exceeds 256 KiB: {len(sealed_bytes)}")
encoded = base64.b64encode(sealed_bytes).decode()
print(f"CONTRACT005_ARTIFACT path={sealed_path} bytes={len(sealed_bytes)} "
      f"sha256={hashlib.sha256(sealed_bytes).hexdigest()} base64_bytes={len(encoded)}", flush=True)
chunk_size = 12000
chunks = [encoded[index:index + chunk_size] for index in range(0, len(encoded), chunk_size)]
for index, chunk in enumerate(chunks):
    print(f"CONTRACT005_B64 {index}/{len(chunks)} {chunk}", flush=True)
print("CONTRACT005_B64_END", flush=True)
finished = datetime.datetime.now(datetime.timezone.utc)
result = {"schema": "hmoe-lionmoe-contract005-result-v1", "job_note": "hmoe-lionmoe-20261116-contract-004-v005",
          "training_invoked": False, "full_model_constructed": False, "real_checkpoint_downloaded": False,
          "controlled_exit": controlled_exit, "production_exit": production_exit,
          "controlled_passed": controlled["passed"] if controlled else 0,
          "controlled_count": controlled["test_count"] if controlled else 0,
          "production_passed": production["passed"] if production else 0,
          "production_count": production["test_count"] if production else 0,
          "passed": passed, "coverage": coverage, "small_artifact_path": str(sealed_path),
          "small_artifact_bytes": len(sealed_bytes), "small_artifact_sha256": hashlib.sha256(sealed_bytes).hexdigest(),
          "started_at": STARTED.isoformat(), "finished_at": finished.isoformat(),
          "duration_seconds": round((finished - STARTED).total_seconds(), 3)}
write_json(OUTPUT / "contract005-result.json", result)
print("CONTRACT005_RESULT " + json.dumps(result, sort_keys=True), flush=True)
raise SystemExit(0 if passed else 1)
