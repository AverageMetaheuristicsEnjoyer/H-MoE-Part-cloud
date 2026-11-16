import base64
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ["HMOE_CONTRACT_ROOT"])
OUTPUT.mkdir(parents=True, exist_ok=True)
STARTED = datetime.datetime.now(datetime.timezone.utc)
EXPECTED_NOTE = "hmoe-lionmoe-20261116-measurement-008-v010"
EXPECTED_ROOT = "/workspace-SR006.nfs3/hmoe-lionmoe-20261116/measurement-008-v010"
EXPECTED_BUNDLE = "hmoe-lionmoe-20261116-v010"
if os.environ.get("HMOE_JOB_NOTE") != EXPECTED_NOTE:
    raise RuntimeError("HMOE_JOB_NOTE does not match measurement008/v010")
if os.environ.get("HMOE_CONTRACT_ROOT") != EXPECTED_ROOT:
    raise RuntimeError("HMOE_CONTRACT_ROOT does not match measurement008/v010")
if os.environ.get("EXPECTED_BUNDLE_ID") != EXPECTED_BUNDLE:
    raise RuntimeError("Unexpected bundle for measurement008")
for key, length in (("EXPECTED_SOURCE_COMMIT", 40), ("EXPECTED_CARRIER_COMMIT", 40), ("EXPECTED_PAYLOAD_SHA256", 64)):
    value = os.environ.get(key, "")
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"Missing/invalid runtime pin: {key}")


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


def run_process(command, env, stdout_path, stderr_path, timeout):
    with Path(stdout_path).open("w") as stdout, Path(stderr_path).open("w") as stderr:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=timeout)
            return result.returncode
        except Exception as error:
            stderr.write(f"PROCESS_PARENT_ERROR {type(error).__name__}: {error}\n")
            return 125


def runtime_gates():
    import importlib.metadata
    import torch

    transformer_engine_version = importlib.metadata.version("transformer-engine")
    volumes = []
    for candidate in ("/home/jovyan", "/workspace-SR006.nfs2", "/workspace-SR006.nfs3"):
        path = Path(candidate)
        item = {"path": candidate, "exists": path.exists()}
        if path.exists():
            usage = os.statvfs(path)
            item.update({"free_bytes": usage.f_bavail * usage.f_frsize, "free_files": usage.f_favail,
                         "writable_by_process": os.access(path, os.W_OK)})
        volumes.append(item)
    nfs3 = next(item for item in volumes if item["path"] == "/workspace-SR006.nfs3")
    if nfs3["free_bytes"] < 2 * 2**30:
        raise RuntimeError("Less than 2 GiB available for checkpoint-free pilot artifacts")
    tiny = OUTPUT / "runtime-roundtrip.bin"
    payload = b"hmoe-lionmoe-measurement-008\n"
    with tiny.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if tiny.read_bytes() != payload:
        raise RuntimeError("Runtime tiny readback mismatch")
    corpus = Path("/home/jovyan/data/fineweb-edu-gpt2-megatron/data")
    expected_sizes = {"train.bin": 15113107020, "train.idx": 145805762,
                      "development.bin": 16000532, "development.idx": 152022,
                      "final.bin": 200002434, "final.idx": 1936642}
    files = {}
    for name, expected in expected_sizes.items():
        path = corpus / name
        if not path.is_file() or path.stat().st_size != expected:
            raise RuntimeError(f"Corpus metadata mismatch: {name}")
        files[name] = {"bytes": expected}
    manifest = corpus.parent / "artifact-manifest.json"
    manifest_sha = sha256(manifest)
    if manifest_sha != "7a7cf91633081dd080f3cff96407bd61d4c237d88a5e926cffe36fc5b8cbfcc8":
        raise RuntimeError("Corpus manifest hash mismatch")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip().splitlines()
    if len(gpu) != 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Pilot requires exactly one visible GPU")
    return {"volumes": volumes, "tiny_roundtrip": artifact(tiny), "corpus_root": str(corpus),
            "corpus_files": files, "corpus_manifest_sha256": manifest_sha,
            "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformer_engine": transformer_engine_version,
            "cuda_available": True, "device_count": 1,
            "weights_saved_or_downloaded": False, "heavy_channel_required_for_pilot": False}


def parse_training_log(path):
    text = Path(path).read_text(errors="replace")
    iterations = [int(value) for value in re.findall(r"iteration\s+(\d+)/", text)]
    losses = [float(value) for value in re.findall(r"lm loss:\s*([0-9.eE+-]+)", text)]
    return {"reached": max(iterations, default=0), "loss_values": losses,
            "all_losses_finite": all(math.isfinite(value) for value in losses),
            "nan_nonzero_lines": len(re.findall(r"nan iterations:\s+[1-9]", text)),
            "skipped_nonzero_lines": len(re.findall(r"skipped iterations:\s+[1-9]", text))}


def phase_summary(values, phase):
    selected = [value for value, phase_name in zip(values, phase)]
    return {"count": len(selected), "sum_seconds": sum(selected),
            "mean_seconds": statistics.mean(selected) if selected else None,
            "median_seconds": statistics.median(selected) if selected else None,
            "min_seconds": min(selected) if selected else None, "max_seconds": max(selected) if selected else None}


def arm_analysis(preset, result_path, bench_log, train_log, init_receipt, arm_root, wrapper_code, duration):
    result = json.loads(result_path.read_text().splitlines()[-1])
    bench = parse_training_log(bench_log)
    training = parse_training_log(train_log)
    argv = result["provenance"]["argv"]
    assert "--save" not in argv and "--load" not in argv
    assert argv[argv.index("--eval-iters") + 1] == "0"
    assert argv[argv.index("--train-iters") + 1] == "90"
    assert argv[argv.index("--lr-decay-iters") + 1] == "17242"
    assert argv[argv.index("--lr-warmup-iters") + 1] == "173"
    assert argv[argv.index("--lr-wsd-decay-iters") + 1] == "3448"
    denominators = result["denominators"]
    assert denominators["micro_batch_sequences_per_gpu"] == 16
    assert denominators["global_batch_sequences"] == 208
    assert denominators["gpu_count"] == 1
    assert denominators["total_parameters"] == 1028926976
    assert wrapper_code == 0 and bench["reached"] == 90 and training["reached"] == 90
    assert result["status"] == "completed"
    assert result["measurement"]["protocol"]["measured_steps"] == 40
    assert result["measurement"]["protocol"]["e2e_train_steps"] == 80
    assert training["all_losses_finite"] and not training["nan_nonzero_lines"] and not training["skipped_nonzero_lines"]
    timing = result["measurement"]["timing"]
    full = timing["full_step_samples_seconds"]
    optimizer = timing["optimizer_step_samples_seconds"]
    assert len(full) == len(optimizer) == 40
    iterations = list(range(41, 81))
    spectral = ["spectral" if (iteration - 1) % 5 == 0 else "sign" for iteration in iterations]
    assert spectral.count("spectral") == 8 and spectral.count("sign") == 32
    loss_tokens = result["denominators"]["loss_tokens_per_step"]
    throughput = 40 * loss_tokens / sum(full)
    init = json.loads(init_receipt.read_text())
    assert init["load"] is None and init["iteration"] == 0 and init["consumed_train_samples"] == 0
    assert init["parameters"] == 1028926976 and init["seed"] == 1234
    observer_path = result_path.parent / "energycal_diagnostics.jsonl"
    observer = [json.loads(line) for line in observer_path.read_text().splitlines()]
    assert len(observer) == 20
    assert not any(item["missing_selected"] for item in observer if item["spectral"])
    diagnostic_callbacks = sum(item["observer_callback_seconds"] for item in observer)
    diagnostic_windows = sum(item["observer_window_seconds"] for item in observer)
    q_summary = json.loads((result_path.parent / "energycal_summary.json").read_text())
    tokens_path = result_path.parent / "tokens_per_expert.json"
    tokens = json.loads(tokens_path.read_text())
    zero_experts = [sum(value == 0 for value in row) for row in tokens["tokens_per_expert"]]
    artifact_paths = [result_path, bench_log, train_log, init_receipt, observer_path,
                      result_path.parent / "energycal_blocks.jsonl", result_path.parent / "energycal_summary.json",
                      tokens_path, result_path.parent / "routing_telemetry.jsonl"]
    for base in (result_path.parent,):
        for path in base.rglob("*"):
            if path.is_file() and path not in artifact_paths:
                artifact_paths.append(path)
    for suffix in (".pt", ".pth", ".safetensors", ".ckpt"):
        assert not list(arm_root.rglob(f"*{suffix}"))
    return {"preset": preset, "status": "valid", "wrapper_exit": wrapper_code,
            "train_exit": 0, "reached": 90, "measured_steps": 40, "probe_e2e_steps": 80,
            "duration_seconds": duration, "result_path": str(result_path),
            "init_receipt": init, "data_fingerprint": {"manifest_sha256": init["data_manifest_sha256"],
                                                       "seed": 1234, "data_order": "same seed/order contract; batch content not separately hashed"},
            "timing": {"full_step_samples_seconds": full, "optimizer_step_samples_seconds": optimizer,
                       "full_step_sum_seconds": sum(full), "full_step_mean_seconds": statistics.mean(full),
                       "full_step_median_seconds": statistics.median(full),
                       "optimizer_step_mean_seconds": statistics.mean(optimizer),
                       "optimizer_step_median_seconds": statistics.median(optimizer),
                       "primary_tokens_per_second": throughput,
                       "probe_tokens_per_second_median_derived": timing["tokens_per_second"],
                       "probe_e2e_wct_seconds": timing["e2e_wct_seconds"],
                       "phase_full_step": {name: phase_summary(full, spectral) for name in ("spectral", "sign")},
                       "phase_optimizer_step": {name: phase_summary(optimizer, spectral) for name in ("spectral", "sign")},
                       "phase_counts": {"spectral": 8, "sign": 32},
                       "external_window_wallclock": None,
                       "wallclock_scope": "sum(train_step) is not total wall clock; Probe e2e reported separately"},
            "memory": result["measurement"]["memory"], "optimizer_state": result["optimizer_state"],
            "loss": result["measurement"]["loss"], "training_log": training,
            "routing": result["measurement"]["routing"], "zero_experts_per_layer": zero_experts,
            "diagnostics": {"summary": q_summary, "observer_callback_seconds": diagnostic_callbacks,
                            "observer_window_seconds": diagnostic_windows,
                            "timing_note": "observer windows include normal optimizer work; no incremental overhead claim"},
            "artifact_receipts_path": str(arm_root / "artifact-receipts.json")}, [artifact(path) for path in artifact_paths]


def main():
    runtime = runtime_gates()
    write_json(OUTPUT / "runtime-gates.json", runtime)
    env = dict(os.environ)
    env.pop("STAGE3_MOE_LIONMUON_ENERGYCAL_DIAGNOSTICS", None)
    env["STAGE3_MOE_MEASUREMENT_ENERGYCAL_DIAGNOSTICS"] = "1"
    transparency_code = run_process([sys.executable, str(ROOT / "tests/stage3_moe/test_lionmuon_measurement.py")],
                                    env, OUTPUT / "transparency.stdout.log", OUTPUT / "transparency.stderr.log", 120)
    transparency_path = OUTPUT / "transparency/transparency-results.json"
    if transparency_code != 0 or not transparency_path.is_file():
        raise RuntimeError(f"Tiny transparency gate failed: exit={transparency_code}")
    transparency = json.loads(transparency_path.read_text())
    if not transparency["passed"] or not all(item["selected_records"] for item in transparency["results"]):
        raise RuntimeError("Transparency gate did not exercise selected records")
    specs = [("lm5", "100", False), ("lm5r50", "50", False), ("lm5r50ec", "50", True)]
    arms = []
    all_receipts = []
    failed = None
    for preset, ratio, energycal in specs:
        arm_root = OUTPUT / "arms" / preset
        arm_root.mkdir(parents=True, exist_ok=True)
        arm_env = dict(env)
        for key in ("STAGE3_MOE_LIONMUON_PERIOD_EXPERTS", "STAGE3_MOE_LIONMUON_PROB", "STAGE3_MOE_LIONMUON_POWER",
                    "STAGE3_MOE_FP8_COMPUTE_ARGS", "STAGE3_MOE_FP8_STATE_DTYPES", "STAGE3_MOE_DEQUANT_ON_LOAD"):
            arm_env.pop(key, None)
        unique = f"hmoe-lionmoe-20261116-measurement-008-v010-{preset}"
        arm_env.update({
            "COAT_ARM_MODE": "bench", "COAT_BENCH_PROPAGATE_EXIT": "1",
            "COAT_BENCH_WARMUP": "40", "COAT_BENCH_MEASURE": "40", "COAT_BENCH_TAG": preset,
            "COAT_CAMPAIGN_ROOT": str(OUTPUT / "campaign"), "COAT_STAGE_ROOT": str(OUTPUT / "stage"),
            "COAT_RUN_TAG": unique, "COAT_ARM_ID_PREFIX": unique + "-", "COAT_ROUND": "measurement008v010",
            "STAGE3_MOE_LOG_ROOT": str(arm_root / "logs"), "STAGE3_MOE_CKPT_ROOT": str(arm_root / "no-checkpoints"),
            "STAGE3_MOE_DATA_CACHE_PATH": str(arm_root / "data-cache"),
            "STAGE3_MOE_COLDSTART_RECEIPT": str(arm_root / "init-receipt.json"),
            "STAGE3_MOE_DATA_ROOT": runtime["corpus_root"], "STAGE3_MOE_EVAL_ITERS": "0",
            "STAGE3_MOE_EVAL_INTERVAL": "1000000", "STAGE3_MOE_LOG_INTERVAL": "1",
            "STAGE3_MOE_ROUTING_TELEMETRY_INTERVAL": "10", "STAGE3_MOE_NPROC": "1",
            "STAGE3_MOE_MICRO_BATCH": "16", "STAGE3_MOE_EP": "1", "STAGE3_MOE_SEED": "1234",
            "STAGE3_MOE_MUON_PRESET": preset, "STAGE3_MOE_MUON_VARIANT": "lionmuon",
            "STAGE3_MOE_LIONMUON_PERIOD": "5", "STAGE3_MOE_LIONMUON_LR_RATIO": ratio,
            "STAGE3_MOE_LIONMUON_ENERGYCAL": "1" if energycal else "0",
            "STAGE3_MOE_LIONMUON_ENERGYCAL_RHO_REF": "0.8",
            "STAGE3_MOE_LIONMUON_ENERGYCAL_EXPONENT": "1", "STAGE3_MOE_LIONMUON_ENERGYCAL_QMIN": "0",
            "WANDB_MODE": "offline", "WANDB_RUN_ID": unique,
        })
        arm_started = time.perf_counter()
        wrapper_code = run_process(["bash", str(ROOT / "scripts/run_coat_arm.sh"), "muon-ctrl"], arm_env,
                                   arm_root / "wrapper.stdout.log", arm_root / "wrapper.stderr.log", 1200)
        duration = time.perf_counter() - arm_started
        arm_id = f"{unique}-muon-ctrl-{preset}"
        bench_log = OUTPUT / "stage" / arm_id / "bench.log"
        run_id = f"stage3-muon_bf16_state_fp32-bench-{unique}-bench-{arm_id}-{preset}"
        result_path = arm_root / "logs" / run_id / "results.jsonl"
        if not result_path.is_file():
            failed = {"preset": preset, "wrapper_exit": wrapper_code, "reason": "missing result JSONL"}
            break
        train_log_text = bench_log.read_text(errors="replace")
        match = re.search(r"^TRAIN_LOG=(.*)$", train_log_text, re.MULTILINE)
        if not match:
            failed = {"preset": preset, "wrapper_exit": wrapper_code, "reason": "missing TRAIN_LOG"}
            break
        analysis, receipts = arm_analysis(preset, result_path, bench_log, match.group(1),
                                          arm_root / "init-receipt.json", arm_root, wrapper_code, duration)
        write_json(arm_root / "analysis.json", analysis)
        write_json(arm_root / "artifact-receipts.json", receipts)
        arms.append(analysis)
        all_receipts.extend(receipts)
    if len({arm["init_receipt"]["parameter_sha256"] for arm in arms}) != 1:
        raise RuntimeError("Fresh init hashes differ across arms")
    if len({arm["init_receipt"]["data_manifest_sha256"] for arm in arms}) != 1:
        raise RuntimeError("Data manifest differs across arms")
    passed = failed is None and len(arms) == 3
    cost = {}
    if passed:
        by_name = {arm["preset"]: arm for arm in arms}
        candidate = by_name["lm5r50ec"]["timing"]["primary_tokens_per_second"]
        for baseline in ("lm5", "lm5r50"):
            base = by_name[baseline]["timing"]["primary_tokens_per_second"]
            cost[baseline] = {"candidate_over_baseline_throughput": candidate / base,
                              "candidate_time_overhead": base / candidate - 1}
        activity = by_name["lm5r50ec"]["diagnostics"]["summary"]["groups"].get("all_routed", {})
        q90 = (activity.get("hypothetical_q") or {}).get("p90")
        overhead = max(item["candidate_time_overhead"] for item in cost.values())
        recommendation = {"q90": q90, "max_time_overhead": overhead,
                          "activity": "attenuation_present" if q90 is not None and q90 < 0.99 else "mostly_inactive",
                          "cost": "investigate_or_reject_before_longer_run" if overhead > 0.1 else "preliminary_cost_within_10pct",
                          "next": "review only; no automatic longer run, ABBA or scalar sweep"}
    else:
        recommendation = {"next": "diagnose partial failure; do not repeat valid arms blindly"}
    result = {"schema": "hmoe-lionmoe-measurement008-result-v1", "job_note": os.environ["HMOE_JOB_NOTE"],
              "identity": {key: os.environ[key] for key in ("EXPECTED_SOURCE_COMMIT", "EXPECTED_CARRIER_COMMIT",
                                                              "EXPECTED_BUNDLE_ID", "EXPECTED_PAYLOAD_SHA256")},
              "training_invoked": True, "training_scope": "checkpoint-free 90-step bench only; not full training",
              "weights_saved_or_downloaded": False, "test_evaluated": False,
              "transparency_exit": transparency_code, "runtime_gates": "passed",
              "arms_completed": len(arms), "arms": arms, "failed": failed, "passed": passed,
              "cost_vs_plain": cost, "ordered_allocation_not_abba": True,
              "recommendation": recommendation,
              "fresh_init_and_data_manifest_equal": passed,
              "quality_speed_memory_benefit_established": False,
              "started_at": STARTED.isoformat(), "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "duration_seconds": round((datetime.datetime.now(datetime.timezone.utc) - STARTED).total_seconds(), 3),
              "raw_artifact_receipts": all_receipts}
    write_json(OUTPUT / "measurement-summary.json", result)
    raw = []
    for label, path in (("measurement-summary", OUTPUT / "measurement-summary.json"),
                        ("transparency-results", transparency_path)):
        data = path.read_bytes()
        raw.append({"label": label, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "content_base64": base64.b64encode(data).decode()})
    sealed_path = OUTPUT / "small-artifacts.json"
    write_json(sealed_path, {"schema": "hmoe-lionmoe-measurement008-small-artifacts-v1", "nonsecret": True,
                             "artifacts": raw})
    sealed = sealed_path.read_bytes()
    if len(sealed) > 256 * 1024:
        raise RuntimeError("Sealed measurement report exceeds 256 KiB")
    encoded = base64.b64encode(sealed).decode()
    print(f"MEASUREMENT008_ARTIFACT path={sealed_path} bytes={len(sealed)} "
          f"sha256={hashlib.sha256(sealed).hexdigest()} base64_bytes={len(encoded)}", flush=True)
    chunks = [encoded[index:index + 12000] for index in range(0, len(encoded), 12000)]
    for index, chunk in enumerate(chunks):
        print(f"MEASUREMENT008_B64 {index}/{len(chunks)} {chunk}", flush=True)
    print("MEASUREMENT008_B64_END", flush=True)
    print("MEASUREMENT008_RESULT " + json.dumps({key: value for key, value in result.items() if key != "arms"},
                                                sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
