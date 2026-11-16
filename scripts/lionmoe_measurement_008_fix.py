import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics

OUTPUT = Path(os.environ["HMOE_CONTRACT_ROOT"])
OUTPUT.mkdir(parents=True, exist_ok=True)
SOURCE = Path("/workspace-SR006.nfs3/hmoe-lionmoe-20261116/measurement-008-v010")
EXPECTED_NOTE = "hmoe-lionmoe-20261116-measurement-008-v011"
if os.environ.get("HMOE_JOB_NOTE") != EXPECTED_NOTE:
    raise RuntimeError("Incorrect corrective note")
if OUTPUT != Path("/workspace-SR006.nfs3/hmoe-lionmoe-20261116/measurement-008-v011"):
    raise RuntimeError("Incorrect corrective root")
if os.environ.get("EXPECTED_BUNDLE_ID") != "hmoe-lionmoe-20261116-v011":
    raise RuntimeError("Incorrect corrective bundle")
for key, length in (("EXPECTED_SOURCE_COMMIT", 40), ("EXPECTED_CARRIER_COMMIT", 40), ("EXPECTED_PAYLOAD_SHA256", 64)):
    value = os.environ.get(key, "")
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"Missing/invalid runtime pin: {key}")
STARTED = datetime.datetime.now(datetime.timezone.utc)


def write_json(path, value):
    with Path(path).open("w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase_summary(values, phases, wanted):
    selected = [value for value, phase in zip(values, phases) if phase == wanted]
    return {"count": len(selected), "sum_seconds": sum(selected),
            "mean_seconds": statistics.mean(selected) if selected else None,
            "median_seconds": statistics.median(selected) if selected else None,
            "min_seconds": min(selected) if selected else None, "max_seconds": max(selected) if selected else None}


source_summary = json.loads((SOURCE / "measurement-summary.json").read_text())
source_summary["runtime_gates_detail"] = json.loads((SOURCE / "runtime-gates.json").read_text())
source_hashes = {}
for arm in source_summary["arms"]:
    source_path = SOURCE / "arms" / arm["preset"] / "analysis.json"
    source_hashes[arm["preset"]] = {"path": str(source_path), "sha256": sha256(source_path)}
    if sha256(source_path) != hashlib.sha256(json.dumps(json.loads(source_path.read_text()), indent=2, sort_keys=True).encode() + b"\n").hexdigest():
        raise RuntimeError("Source analysis canonical hash mismatch")
    timing = arm["timing"]
    full = timing["full_step_samples_seconds"]
    optimizer = timing["optimizer_step_samples_seconds"]
    if len(full) != 40 or len(optimizer) != 40:
        raise RuntimeError("Expected 40 actual samples")
    phases = ["spectral" if (iteration - 1) % 5 == 0 else "sign" for iteration in range(41, 81)]
    timing["phase_full_step"] = {name: phase_summary(full, phases, name) for name in ("spectral", "sign")}
    timing["phase_optimizer_step"] = {name: phase_summary(optimizer, phases, name) for name in ("spectral", "sign")}
    if timing["phase_full_step"]["spectral"]["count"] != 8 or timing["phase_full_step"]["sign"]["count"] != 32:
        raise RuntimeError("Incorrect corrected phase counts")
source_summary["identity"] = {key: os.environ[key] for key in ("EXPECTED_SOURCE_COMMIT", "EXPECTED_CARRIER_COMMIT",
                                                                  "EXPECTED_BUNDLE_ID", "EXPECTED_PAYLOAD_SHA256")}
source_summary["job_note"] = os.environ["HMOE_JOB_NOTE"]
source_summary["analysis_only_correction"] = True
source_summary["corrective_job_training_invoked"] = False
source_summary["source_samples"] = {"job": "lm-mpi-job-c0353d15-09fd-4b5d-bd45-11a449f3e34e",
                                    "root": str(SOURCE), "analysis_sha256": source_hashes}
source_summary["correction_reason"] = "v010 phase_summary lacked a phase filter; v011 recomputes only summaries from preserved samples"
source_summary["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
source_summary["duration_seconds"] = round((datetime.datetime.now(datetime.timezone.utc) - STARTED).total_seconds(), 3)
write_json(OUTPUT / "measurement-summary.json", source_summary)
data = (OUTPUT / "measurement-summary.json").read_bytes()
sealed = {"schema": "hmoe-lionmoe-measurement008-v011-small-artifacts-v1", "nonsecret": True,
          "artifacts": [{"label": "measurement-summary", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                         "content_base64": base64.b64encode(data).decode()}]}
sealed_path = OUTPUT / "small-artifacts.json"
write_json(sealed_path, sealed)
sealed_bytes = sealed_path.read_bytes()
if len(sealed_bytes) > 256 * 1024:
    raise RuntimeError("Corrective sealed report exceeds 256 KiB")
encoded = base64.b64encode(sealed_bytes).decode()
print(f"MEASUREMENT008FIX_ARTIFACT path={sealed_path} bytes={len(sealed_bytes)} "
      f"sha256={hashlib.sha256(sealed_bytes).hexdigest()} base64_bytes={len(encoded)}", flush=True)
chunks = [encoded[index:index + 12000] for index in range(0, len(encoded), 12000)]
for index, chunk in enumerate(chunks):
    print(f"MEASUREMENT008FIX_B64 {index}/{len(chunks)} {chunk}", flush=True)
print("MEASUREMENT008FIX_B64_END", flush=True)
print("MEASUREMENT008FIX_RESULT " + json.dumps({"passed": True, "analysis_only": True,
                                                 "note": os.environ["HMOE_JOB_NOTE"],
                                                 "source_samples": source_summary["source_samples"]}), flush=True)
