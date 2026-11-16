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
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ["HMOE_CONTRACT_ROOT"])
OUTPUT.mkdir(parents=True, exist_ok=True)
STARTED = datetime.datetime.now(datetime.timezone.utc)
STAGES = []
VERSION = os.environ.get("HMOE_OVERNIGHT_VERSION", "020")
MODE = os.environ.get("HMOE_OVERNIGHT_MODE", "primary")
if MODE not in ("primary", "confirm"):
    raise RuntimeError("Unknown overnight mode")
EXPECTED_NOTE = f"hmoe-lionmoe-20261116-overnight-013-v{VERSION}"
EXPECTED_ROOT = f"/workspace-SR006.nfs3/hmoe-lionmoe-20261116/overnight-013-v{VERSION}"
EXPECTED_BUNDLE = f"hmoe-lionmoe-20261116-v{VERSION}"
SEED = 1234 if MODE == "primary" else 1235
TRAINING_DEADLINE = datetime.datetime(2026, 11, 17, 5, 50, tzinfo=datetime.timezone.utc)
if os.environ.get("HMOE_JOB_NOTE") != EXPECTED_NOTE:
    raise RuntimeError("HMOE_JOB_NOTE does not match overnight013/version")
if os.environ.get("HMOE_CONTRACT_ROOT") != EXPECTED_ROOT:
    raise RuntimeError("HMOE_CONTRACT_ROOT does not match overnight013/version")
if os.environ.get("EXPECTED_BUNDLE_ID") != EXPECTED_BUNDLE:
    raise RuntimeError("Unexpected bundle for overnight013")
for key, length in (("EXPECTED_SOURCE_COMMIT", 40), ("EXPECTED_CARRIER_COMMIT", 40), ("EXPECTED_PAYLOAD_SHA256", 64)):
    value = os.environ.get(key, "")
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"Missing/invalid runtime pin: {key}")


def mark_stage(name):
    if name not in STAGES:
        STAGES.append(name)
    write_json(OUTPUT / "progress.json", {"completed_stages": STAGES})


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
    payload = b"hmoe-lionmoe-overnight-013\n"
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


def freeze_alpha():
    if MODE == "confirm":
        alpha = float(os.environ["HMOE_FROZEN_ALPHA"])
        if not 0 < alpha <= 1:
            raise RuntimeError("Invalid frozen confirmation alpha")
        return {"alpha": alpha, "source": "frozen from primary seed1234; not recalibrated", "recalibrated": False}
    summary_path = Path("/workspace-SR006.nfs3/hmoe-lionmoe-20261116/postwarmup-012-v019/measurement-summary.json")
    if summary_path.stat().st_size != 156762 or sha256(summary_path) != "b9aa784de06c1b6eb2f32eca6d5814dd11706f1dbe2e39224e7c80af67cd4793":
        raise RuntimeError("Task012 summary provenance mismatch")
    summary = json.loads(summary_path.read_text())
    receipt = next(item for item in summary["raw_artifact_receipts"]
                   if item["path"].endswith("lm5r50ec/energycal_blocks.jsonl") or
                   ("/lm5r50ec/" in item["path"] and item["path"].endswith("/energycal_blocks.jsonl")))
    path = Path(receipt["path"])
    if path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["sha256"]:
        raise RuntimeError("Task012 actual-q block provenance mismatch")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    q = [record["actual_q"] for record in records if record["routed"] and record["iteration"] in (181, 186, 191, 196)]
    if len(q) != 108 or not all(math.isfinite(value) for value in q):
        raise RuntimeError("Expected108 finite actual q observations")
    raw_mean = statistics.mean(q)
    alpha = round(raw_mean, 6)
    if not 0 < alpha <= 1:
        raise RuntimeError("Degenerate fixed alpha")
    return {"alpha": alpha, "raw_mean": raw_mean, "rounded_decimals": 6, "count": len(q),
            "window": [181, 200], "spectral_steps": [181, 186, 191, 196],
            "source_summary": {"path": str(summary_path), "bytes": summary_path.stat().st_size, "sha256": sha256(summary_path)},
            "source_blocks": receipt, "frozen_before_new_validation_results": True, "recalibrated": False}


def parse_training_log(path):
    text = Path(path).read_text(errors="replace")
    iterations = [int(value) for value in re.findall(r"iteration\s+(\d+)/", text)]
    losses = [float(value) for value in re.findall(r"lm loss:\s*([0-9.eE+-]+)", text)]
    validation_matches = re.findall(r"validation loss at iteration\s+(\d+) on validation set \| lm loss value: ([0-9.eE+-]+)", text)
    scheduled = {}
    for iteration, value in validation_matches:
        scheduled.setdefault(int(iteration), float(value))
    return {"reached": max(iterations, default=0), "loss_values": losses,
            "all_losses_finite": all(math.isfinite(value) for value in losses),
            "validation_losses": [{"iteration": iteration, "loss": value} for iteration, value in validation_matches],
            "scheduled_validation": scheduled,
            "skipped_test_evaluation_calls": len(re.findall(r"STAGE3_TEST_EVALUATION_SKIPPED", text)),
            "nan_nonzero_lines": len(re.findall(r"nan iterations:\s+[1-9]", text)),
            "skipped_nonzero_lines": len(re.findall(r"skipped iterations:\s+[1-9]", text))}


def phase_summary(values, phases, wanted):
    selected = [value for value, phase_name in zip(values, phases) if phase_name == wanted]
    return {"count": len(selected), "sum_seconds": sum(selected),
            "mean_seconds": statistics.mean(selected) if selected else None,
            "median_seconds": statistics.median(selected) if selected else None,
            "min_seconds": min(selected) if selected else None, "max_seconds": max(selected) if selected else None}


def quantile(values, fraction):
    values = sorted(values)
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def distribution(values):
    if not values:
        return None
    return {"min": min(values), "p10": quantile(values, 0.1), "p50": quantile(values, 0.5),
            "p90": quantile(values, 0.9), "max": max(values)}


def diagnostic_analysis(blocks, preset):
    if len(blocks) != 124:
        raise RuntimeError(f"Expected124 late block records, got{len(blocks)}")
    keys = [(record["iteration"], record["layer"], record["expert"], record["fc_type"]) for record in blocks]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate late block key, including control halves")
    if any(record["nonfinite"] for record in blocks):
        raise RuntimeError("Nonfinite late block record")
    q_field = "actual_q" if preset == "lm5r50ec" else "hypothetical_q"
    profiles = []
    for iteration in (181, 186, 191, 196):
        step = [record for record in blocks if record["iteration"] == iteration]
        routed = [record for record in step if record["routed"]]
        if len(step) != 31 or len(routed) != 27 or len(step) - len(routed) != 4:
            raise RuntimeError(f"Late coverage mismatch at{iteration}")
        groups = [("all_routed", routed)] + [(f"layer_{layer:02d}/{fc}", [record for record in routed
                  if record["layer"] == layer and record["fc_type"] == fc])
                  for layer in (1, 9, 17) for fc in ("fc1_gate", "fc1_up", "fc2")]
        for name, records in groups:
            q = [record[q_field] for record in records if record.get(q_field) is not None]
            rho = [record["rho"] for record in records if record.get("rho") is not None]
            profiles.append({"iteration": iteration, "group": name, "samples": len(records),
                             "q_field": q_field, "q": distribution(q), "rho": distribution(rho),
                             "q_mean": statistics.mean(q) if q else None,
                             "fraction_q_lt_0_99": sum(value < 0.99 for value in q) / len(q) if q else None,
                             "fraction_q_lt_0_9": sum(value < 0.9 for value in q) / len(q) if q else None,
                             "fraction_q_lt_0_5": sum(value < 0.5 for value in q) / len(q) if q else None,
                             "fraction_q_zero": sum(value == 0 for value in q) / len(q) if q else None})
    routed = [record for record in blocks if record["routed"]]
    if preset == "lm5r50ec":
        for record in routed:
            ratio = record["post_update_norm"] / record["pre_update_norm"] if record["pre_update_norm"] else None
            if not record["applied"] or ratio is None or not math.isclose(record["actual_q"], ratio, abs_tol=1e-6, rel_tol=1e-6):
                raise RuntimeError("Candidate actual q/post-pre mismatch")
    elif any(record["applied"] or record["actual_q"] is not None for record in blocks):
        raise RuntimeError("Plain observer applied calibration")
    norms = {}
    for label, selected in (("affected_q_lt_0_99", [record for record in routed if record[q_field] < 0.99]),
                            ("unaffected_q_gte_0_99", [record for record in routed if record[q_field] >= 0.99])):
        metrics = {}
        for field in ("grad_norm", "direction_norm", "old_momentum_norm", "pre_update_norm",
                      "post_update_norm", "update_over_grad_alignment"):
            values = [record[field] for record in selected if record.get(field) is not None]
            metrics[field] = {"n": len(values), "null": len(selected) - len(values),
                              "mean": statistics.mean(values) if values else None,
                              "median": statistics.median(values) if values else None,
                              "min": min(values) if values else None, "max": max(values) if values else None}
        norms[label] = {"samples": len(selected), "metrics": metrics}
    return {"per_step": profiles, "norm_context": norms,
            "angle_scope": "positive scalar preserves update/G angle; no EnergyCal angular improvement claim"}


def diagnostic_routing_analysis(path):
    data = json.loads(path.read_text())
    if data["window"] != [181, 200] or len(data["layers"]) != 17:
        raise RuntimeError("Diagnostic routing window/layers mismatch")
    if len(data["tokens_per_expert"]) != 17 or any(len(row) != 64 for row in data["tokens_per_expert"]):
        raise RuntimeError("Diagnostic routing matrix schema mismatch")
    selected = []
    for layer in (1, 9, 17):
        rows = [index for index, name in enumerate(data["layers"])
                if re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name) and
                int(re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name).group(1)) == layer]
        if len(rows) != 1:
            raise RuntimeError(f"Ambiguous/missing diagnostic router layer{layer}")
        for expert in (0, 31, 63):
            selected.append({"layer": layer, "matrix_row": rows[0], "expert": expert,
                             "tokens": data["tokens_per_expert"][rows[0]][expert]})
    zero_counts = [sum(value == 0 for value in row) for row in data["tokens_per_expert"]]
    means = [statistics.mean(row) for row in data["tokens_per_expert"]]
    cv = [statistics.pstdev(row) / mean if mean else None for row, mean in zip(data["tokens_per_expert"], means)]
    return {"window": [181, 200], "scope": data["scope"], "selected_experts": selected,
            "zero_experts_per_layer_accumulated": zero_counts,
            "coefficient_of_variation_max": max(value for value in cv if value is not None),
            "same_window_as_block_diagnostics": True,
            "dropped_tokens_scope": "zero only because configured dropless; not an independent event counter"}


def arm_analysis(preset, result_path, bench_log, train_log, init_receipt, arm_root, wrapper_code, duration):
    result = json.loads(result_path.read_text().splitlines()[-1])
    bench = parse_training_log(bench_log)
    training = parse_training_log(train_log)
    argv = result["provenance"]["argv"]
    assert "--save" not in argv and "--load" not in argv
    assert argv[argv.index("--train-iters") + 1] == "600"
    assert argv[argv.index("--eval-iters") + 1] == "32"
    assert argv[argv.index("--eval-interval") + 1] == "300"
    assert argv[argv.index("--seed") + 1] == str(SEED)
    assert argv[argv.index("--lr-decay-iters") + 1] == "17242"
    assert argv[argv.index("--lr-warmup-iters") + 1] == "173"
    assert argv[argv.index("--lr-wsd-decay-iters") + 1] == "3448"
    denominators = result["denominators"]
    assert denominators["micro_batch_sequences_per_gpu"] == 16
    assert denominators["global_batch_sequences"] == 208
    assert denominators["gpu_count"] == 1
    assert denominators["total_parameters"] == 1028926976
    assert wrapper_code == 0 and bench["reached"] == 600 and training["reached"] == 600
    assert result["status"] == "completed"
    assert result["measurement"]["protocol"]["measured_steps"] == 40
    assert result["measurement"]["protocol"]["e2e_train_steps"] == 590
    assert training["all_losses_finite"] and not training["nan_nonzero_lines"] and not training["skipped_nonzero_lines"]
    assert training["skipped_test_evaluation_calls"] == 1
    assert result["evaluation"]["skipped_test_calls"] in (0, 1)
    test_output = [line for line in Path(train_log).read_text(errors="replace").splitlines()
                   if "on test set" in line and "STAGE3_TEST_EVALUATION_SKIPPED" not in line]
    assert not test_output, f"Unexpected real test output: {test_output[-3:]}"
    if set(training["scheduled_validation"]) != {300, 600}:
        raise RuntimeError(f"Missing scheduled development endpoints: {training['scheduled_validation']}")
    if not all(math.isfinite(value) for value in training["scheduled_validation"].values()):
        raise RuntimeError("Nonfinite development loss")
    timing = result["measurement"]["timing"]
    full = timing["full_step_samples_seconds"]
    optimizer = timing["optimizer_step_samples_seconds"]
    assert len(full) == len(optimizer) == 40
    iterations = list(range(551, 591))
    spectral = ["spectral" if (iteration - 1) % 5 == 0 else "sign" for iteration in iterations]
    assert spectral.count("spectral") == 8 and spectral.count("sign") == 32
    loss_tokens = result["denominators"]["loss_tokens_per_step"]
    throughput = 40 * loss_tokens / sum(full)
    init = json.loads(init_receipt.read_text())
    assert init["load"] is None and init["iteration"] == 0 and init["consumed_train_samples"] == 0
    assert init["parameters"] == 1028926976 and init["seed"] == SEED
    observer_path = result_path.parent / "energycal_diagnostics.jsonl"
    observer = [json.loads(line) for line in observer_path.read_text().splitlines()]
    assert len(observer) == 20
    assert [item["iteration"] for item in observer] == list(range(181, 201))
    assert [item["iteration"] for item in observer if item["spectral"]] == [181, 186, 191, 196]
    assert all(item["selected_records"] == 31 and item["missing_selected"] == 0
               for item in observer if item["spectral"])
    assert all(item["selected_records"] == 0 for item in observer if not item["spectral"])
    diagnostic_callbacks = sum(item["observer_callback_seconds"] for item in observer)
    diagnostic_windows = sum(item["observer_window_seconds"] for item in observer)
    q_summary = json.loads((result_path.parent / "energycal_summary.json").read_text())
    if q_summary["window"] != [181, 200] or q_summary["spectral_steps"] != [181, 186, 191, 196]:
        raise RuntimeError("Observer summary did not derive the actual late window")
    blocks_path = result_path.parent / "energycal_blocks.jsonl"
    blocks = [json.loads(line) for line in blocks_path.read_text().splitlines() if line.strip()]
    block_analysis = diagnostic_analysis(blocks, preset)
    diagnostic_routing_path = result_path.parent / "routing_diagnostics_181_200.json"
    diagnostic_routing = diagnostic_routing_analysis(diagnostic_routing_path)
    tokens_path = result_path.parent / "tokens_per_expert.json"
    tokens = json.loads(tokens_path.read_text())
    if tokens["window"] != "measured_steps_only":
        raise RuntimeError("Timing routing window label mismatch")
    zero_experts = [sum(value == 0 for value in row) for row in tokens["tokens_per_expert"]]
    artifact_paths = [result_path, bench_log, train_log, init_receipt, observer_path,
                      blocks_path, result_path.parent / "energycal_summary.json",
                      tokens_path, result_path.parent / "routing_telemetry.jsonl", diagnostic_routing_path]
    chronological = []
    for start, end in ((551, 560), (561, 570), (571, 580), (581, 590), (551, 570), (571, 590)):
        selected = [(iteration, value, phase) for iteration, value, phase in zip(iterations, full, spectral)
                    if start <= iteration <= end]
        values = [item[1] for item in selected]
        chronological.append({"start": start, "end": end, "count": len(values), "sum_seconds": sum(values),
                              "mean_seconds": statistics.mean(values), "median_seconds": statistics.median(values),
                              "min_seconds": min(values), "max_seconds": max(values),
                              "primary_tokens_per_second": len(values) * loss_tokens / sum(values),
                              "spectral_mean_seconds": statistics.mean([item[1] for item in selected if item[2] == "spectral"]),
                              "sign_mean_seconds": statistics.mean([item[1] for item in selected if item[2] == "sign"])})
    for base in (result_path.parent,):
        for path in base.rglob("*"):
            if path.is_file() and path not in artifact_paths:
                artifact_paths.append(path)
    for suffix in (".pt", ".pth", ".safetensors", ".ckpt"):
        assert not list(arm_root.rglob(f"*{suffix}"))
    return {"preset": preset, "status": "valid", "wrapper_exit": wrapper_code,
            "train_exit": 0, "reached": 600, "measured_steps": 40, "probe_e2e_steps": 590,
            "duration_seconds": duration, "result_path": str(result_path),
            "init_receipt": init, "data_fingerprint": {"manifest_sha256": init["data_manifest_sha256"],
                                                       "seed": SEED, "data_order": "same seed/order contract; batch content not separately hashed"},
            "timing": {"full_step_samples_seconds": full, "optimizer_step_samples_seconds": optimizer,
                       "full_step_sum_seconds": sum(full), "full_step_mean_seconds": statistics.mean(full),
                       "full_step_median_seconds": statistics.median(full),
                       "optimizer_step_mean_seconds": statistics.mean(optimizer),
                       "optimizer_step_median_seconds": statistics.median(optimizer),
                       "primary_tokens_per_second": throughput,
                       "probe_tokens_per_second_median_derived": timing["tokens_per_second"],
                       "probe_e2e_wct_seconds": timing["e2e_wct_seconds"],
                       "phase_full_step": {name: phase_summary(full, spectral, name) for name in ("spectral", "sign")},
                       "phase_optimizer_step": {name: phase_summary(optimizer, spectral, name) for name in ("spectral", "sign")},
                       "phase_counts": {"spectral": 8, "sign": 32},
                       "chronological_windows": chronological,
                       "max_sample": {"iteration": iterations[max(range(40), key=full.__getitem__)], "seconds": max(full)},
                       "min_sample": {"iteration": iterations[min(range(40), key=full.__getitem__)], "seconds": min(full)},
                       "external_window_wallclock": None,
                       "wallclock_scope": "sum(train_step) is not total wall clock; Probe e2e reported separately"},
            "memory": result["measurement"]["memory"], "optimizer_state": result["optimizer_state"],
            "loss": result["measurement"]["loss"], "training_log": training,
            "development": {"split": "development", "eval_iters": 32, "global_batch": 208,
                            "primary_iteration": 600, "secondary_iteration": 300,
                            "loss_at_300": training["scheduled_validation"][300],
                            "loss_at_600": training["scheduled_validation"][600],
                            "selection_scope": "first scheduled log per iteration; final duplicate not averaged",
                            "test_evaluated": False, "skipped_test_calls": training["skipped_test_evaluation_calls"]},
            "routing": result["measurement"]["routing"], "zero_experts_per_layer": zero_experts,
            "diagnostics": {"summary": q_summary, "observer_callback_seconds": diagnostic_callbacks,
                            "observer_window_seconds": diagnostic_windows, **block_analysis,
                            "timing_note": "observer windows include normal optimizer work; no incremental overhead claim"},
            "diagnostic_routing": diagnostic_routing,
            "artifact_receipts_path": str(arm_root / "artifact-receipts.json")}, [artifact(path) for path in artifact_paths]


def recover_v020_plain():
    source_root = Path("/workspace-SR006.nfs3/hmoe-lionmoe-20261116/overnight-013-v020")
    if os.environ.get("HMOE_RECOVER_V020_PLAIN") != "1":
        return None
    if MODE != "primary":
        raise RuntimeError("v020 plain recovery is primary-only")
    preset = "lm5r50"
    unique = f"hmoe-lionmoe-20261116-overnight-013-v020-{preset}-seed1234"
    arm_root = source_root / "arms" / preset
    arm_id = f"{unique}-muon-ctrl-{preset}"
    bench_log = source_root / "stage" / arm_id / "bench.log"
    run_id = f"stage3-muon_bf16_state_fp32-bench-{unique}-bench-{arm_id}-{preset}"
    result_path = arm_root / "logs" / run_id / "results.jsonl"
    wrapper_path = arm_root / "wrapper.stdout.log"
    wrapper_text = wrapper_path.read_text(errors="replace")
    wrapper_matches = re.findall(r"^COAT_BENCH (\{.*\})$", wrapper_text, re.MULTILINE)
    if len(wrapper_matches) != 1:
        raise RuntimeError("v020 plain recovery requires exactly one COAT_BENCH result")
    wrapper_result = json.loads(wrapper_matches[0])
    bench_text = bench_log.read_text(errors="replace")
    train_exits = re.findall(r"^TRAIN_EXIT=(\d+)$", bench_text, re.MULTILINE)
    train_logs = re.findall(r"^TRAIN_LOG=(.*)$", bench_text, re.MULTILINE)
    if train_exits != ["0"] or len(train_logs) != 1 or wrapper_result.get("exit") != 0:
        raise RuntimeError(f"v020 plain recovery exits invalid: train={train_exits}, wrapper={wrapper_result.get('exit')}")
    analysis, receipts = arm_analysis(preset, result_path, bench_log, train_logs[0],
                                      arm_root / "init-receipt.json", arm_root, 0, None)
    analysis["recovery"] = {"source_root": str(source_root), "source_job": "lm-mpi-job-a6c2e8eb-900e-40db-bd79-1444653d4572",
                            "training_repeated": False, "train_exit": 0, "wrapper_exit": 0,
                            "timing_scope": "cross-allocation quality comparison; do not claim paired speed"}
    return analysis, receipts


def main_checked():
    runtime = runtime_gates()
    write_json(OUTPUT / "runtime-gates.json", runtime)
    mark_stage("runtime")
    alpha_receipt = freeze_alpha()
    write_json(OUTPUT / "alpha-receipt.json", alpha_receipt)
    mark_stage("alpha_frozen")
    env = dict(os.environ)
    env.pop("STAGE3_MOE_LIONMUON_ENERGYCAL_DIAGNOSTICS", None)
    env["STAGE3_MOE_MEASUREMENT_ENERGYCAL_DIAGNOSTICS"] = "1"
    transparency_code = run_process([sys.executable, str(ROOT / "tests/stage3_moe/test_lionmuon_overnight.py")],
                                    env, OUTPUT / "transparency.stdout.log", OUTPUT / "transparency.stderr.log", 120)
    transparency_path = OUTPUT / "overnight-contracts/contract-results.json"
    if transparency_code != 0 or not transparency_path.is_file():
        stdout_detail = (OUTPUT / "transparency.stdout.log").read_text(errors="replace")[-4000:]
        stderr_detail = (OUTPUT / "transparency.stderr.log").read_text(errors="replace")[-4000:]
        print("OVERNIGHT013_TRANSPARENCY_STDOUT " + stdout_detail, flush=True)
        print("OVERNIGHT013_TRANSPARENCY_STDERR " + stderr_detail, flush=True)
        raise RuntimeError(f"Tiny transparency gate failed: exit={transparency_code}; stderr={stderr_detail}")
    transparency = json.loads(transparency_path.read_text())
    if not transparency["passed"] or not transparency["alpha_identity"]["nonzero_params_and_momentum_equal"] or not transparency["no_test_guard"]["test_prefix_blocked"]:
        raise RuntimeError("Overnight contracts did not exercise alpha identity and no-test guard")
    mark_stage("overnight_contracts")
    if MODE == "primary":
        specs = [("lm5r50", False, 1.0), ("lm5r50ec", True, 1.0), ("lm5r50fixed", False, alpha_receipt["alpha"])]
    else:
        modified = os.environ.get("HMOE_CONFIRM_MODIFIED", "")
        if modified not in ("lm5r50ec", "lm5r50fixed"):
            raise RuntimeError("Confirmation requires selected modified arm")
        specs = [(modified, modified == "lm5r50ec", alpha_receipt["alpha"] if modified == "lm5r50fixed" else 1.0),
                 ("lm5r50", False, 1.0)]
    arms = []
    all_receipts = []
    failed = None
    recovered = recover_v020_plain()
    remaining_training_arms = len(specs) - (1 if recovered else 0)
    initial_remaining = (TRAINING_DEADLINE - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    if initial_remaining < remaining_training_arms * 7200:
        raise RuntimeError(f"Insufficient time for full matched stage: {initial_remaining:.1f}s")
    if recovered:
        analysis, receipts = recovered
        recovered_root = OUTPUT / "arms" / analysis["preset"]
        write_json(recovered_root / "analysis.json", analysis)
        write_json(recovered_root / "artifact-receipts.json", receipts)
        analysis["artifact_receipts_path"] = str(recovered_root / "artifact-receipts.json")
        arms.append(analysis)
        all_receipts.extend(receipts)
        mark_stage("arm_lm5r50_valid_recovered")
    for preset, energycal, fixed_alpha in specs:
        if recovered and preset == "lm5r50":
            continue
        arm_root = OUTPUT / "arms" / preset
        arm_root.mkdir(parents=True, exist_ok=True)
        arm_env = dict(env)
        for key in ("STAGE3_MOE_LIONMUON_PERIOD_EXPERTS", "STAGE3_MOE_LIONMUON_PROB", "STAGE3_MOE_LIONMUON_POWER",
                    "STAGE3_MOE_FP8_COMPUTE_ARGS", "STAGE3_MOE_FP8_STATE_DTYPES", "STAGE3_MOE_DEQUANT_ON_LOAD"):
            arm_env.pop(key, None)
        unique = f"hmoe-lionmoe-20261116-overnight-013-v{VERSION}-{preset}-seed{SEED}"
        arm_env.update({
            "COAT_ARM_MODE": "bench", "COAT_BENCH_PROPAGATE_EXIT": "1",
            "COAT_BENCH_WARMUP": "550", "COAT_BENCH_MEASURE": "40", "COAT_BENCH_TAG": preset,
            "COAT_CAMPAIGN_ROOT": str(OUTPUT / "campaign"), "COAT_STAGE_ROOT": str(OUTPUT / "stage"),
            "COAT_RUN_TAG": unique, "COAT_ARM_ID_PREFIX": unique + "-", "COAT_ROUND": f"overnight013v{VERSION}",
            "STAGE3_MOE_LOG_ROOT": str(arm_root / "logs"), "STAGE3_MOE_CKPT_ROOT": str(arm_root / "no-checkpoints"),
            "STAGE3_MOE_DATA_CACHE_PATH": str(arm_root / "data-cache"),
            "STAGE3_MOE_COLDSTART_RECEIPT": str(arm_root / "init-receipt.json"),
            "STAGE3_MOE_DATA_ROOT": runtime["corpus_root"], "STAGE3_MOE_EVAL_ITERS": "32",
            "STAGE3_MOE_DISABLE_TEST_EVAL": "1", "STAGE3_MOE_EVAL_INTERVAL": "300", "STAGE3_MOE_LOG_INTERVAL": "1",
            "STAGE3_MOE_ROUTING_TELEMETRY_INTERVAL": "10", "STAGE3_MOE_ROUTING_DIAGNOSTICS": "1",
            "STAGE3_MOE_ENERGYCAL_DIAGNOSTICS_START": "181", "STAGE3_MOE_ENERGYCAL_DIAGNOSTICS_END": "200",
            "STAGE3_MOE_NPROC": "1",
            "STAGE3_MOE_MICRO_BATCH": "16", "STAGE3_MOE_EP": "1", "STAGE3_MOE_SEED": str(SEED),
            "STAGE3_MOE_MUON_PRESET": preset, "STAGE3_MOE_MUON_VARIANT": "lionmuon",
            "STAGE3_MOE_LIONMUON_PERIOD": "5", "STAGE3_MOE_LIONMUON_LR_RATIO": "50",
            "STAGE3_MOE_LIONMUON_FIXED_ROUTED_ALPHA": str(fixed_alpha),
            "STAGE3_MOE_LIONMUON_ENERGYCAL": "1" if energycal else "0",
            "STAGE3_MOE_LIONMUON_ENERGYCAL_RHO_REF": "0.8",
            "STAGE3_MOE_LIONMUON_ENERGYCAL_EXPONENT": "1", "STAGE3_MOE_LIONMUON_ENERGYCAL_QMIN": "0",
            "WANDB_MODE": "offline", "WANDB_RUN_ID": unique,
        })
        remaining = (TRAINING_DEADLINE - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        if remaining < 7200:
            failed = {"preset": preset, "reason": f"deadline reserve insufficient for next arm: {remaining:.1f}s"}
            break
        arm_started = time.perf_counter()
        wrapper_code = run_process(["bash", str(ROOT / "scripts/run_coat_arm.sh"), "muon-ctrl"], arm_env,
                                   arm_root / "wrapper.stdout.log", arm_root / "wrapper.stderr.log", 7200)
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
        mark_stage(f"arm_{preset}_valid")
    if len({arm["init_receipt"]["parameter_sha256"] for arm in arms}) != 1:
        raise RuntimeError("Fresh init hashes differ across arms")
    if len({arm["init_receipt"]["data_manifest_sha256"] for arm in arms}) != 1:
        raise RuntimeError("Data manifest differs across arms")
    passed = failed is None and len(arms) == len(specs)
    cost = {}
    if passed:
        by_name = {arm["preset"]: arm for arm in arms}
        modified = next(name for name in by_name if name != "lm5r50")
        candidate = by_name[modified]["timing"]["primary_tokens_per_second"]
        base = by_name["lm5r50"]["timing"]["primary_tokens_per_second"]
        cost[modified] = {"candidate_over_baseline_throughput": candidate / base,
                          "candidate_time_overhead": base / candidate - 1,
                          "cross_allocation_recovery": recovered is not None,
                          "speed_claim": "ordered probe only; not ABBA or paired speed acceptance"}
        recommendation = {"activity": "see per-step/layer/FC actual q; no single q90 verdict",
                          "cost": "ordered post-warmup probe only; inspect phase and chronological drift",
                          "next": "review only; no automatic quality/longer run or scalar sweep"}
    else:
        recommendation = {"next": "diagnose partial failure; do not repeat valid arms blindly"}
    confirmation = {"eligible": False}
    development = []
    if passed:
        by_name = {arm["preset"]: arm for arm in arms}
        for arm in arms:
            development.append({"preset": arm["preset"], "seed": SEED,
                                "loss_at_300": arm["development"]["loss_at_300"],
                                "loss_at_600": arm["development"]["loss_at_600"]})
        if MODE == "primary":
            ec, fixed, plain = (by_name[name]["development"]["loss_at_600"] for name in ("lm5r50ec", "lm5r50fixed", "lm5r50"))
            selected = "lm5r50ec" if ec < fixed else "lm5r50fixed"
            reason = "lower development600 loss; exact tie chooses simpler constant"
            if ec >= plain and fixed >= plain:
                selected = "lm5r50ec"
                reason = "both modified arms worse than plain; confirm EnergyCal disadvantage without erasing it"
            confirmation = {"eligible": True, "selected_modified": selected, "alpha": alpha_receipt["alpha"],
                            "reason": reason, "seed": 1235, "order": [selected, "lm5r50"],
                            "requires_full_pair_time_reserve": True}
    result = {"schema": "hmoe-lionmoe-overnight013-result-v1", "job_note": os.environ["HMOE_JOB_NOTE"],
              "mode": MODE, "seed": SEED, "alpha_receipt": alpha_receipt,
              "development": development, "confirmation": confirmation,
              "identity": {key: os.environ[key] for key in ("EXPECTED_SOURCE_COMMIT", "EXPECTED_CARRIER_COMMIT",
                                                              "EXPECTED_BUNDLE_ID", "EXPECTED_PAYLOAD_SHA256")},
              "training_invoked": True, "training_scope": "checkpoint-free from-scratch 600-step development-validation screen only; not full training",
              "weights_saved_or_downloaded": False, "test_evaluated": False,
              "transparency_exit": transparency_code, "runtime_gates": "passed",
              "arms_completed": len(arms), "arms": arms, "failed": failed, "passed": passed,
              "cost_vs_plain": cost, "ordered_allocation_not_abba": True,
              "recovered_v020_plain_without_retraining": recovered is not None,
              "recommendation": recommendation,
              "fresh_init_and_data_manifest_equal": passed,
              "quality_speed_memory_benefit_established": False,
              "started_at": STARTED.isoformat(), "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "duration_seconds": round((datetime.datetime.now(datetime.timezone.utc) - STARTED).total_seconds(), 3),
              "raw_artifact_receipts": all_receipts}
    write_json(OUTPUT / "measurement-summary.json", result)
    compact_arms = []
    for arm in arms:
        compact = {key: value for key, value in arm.items() if key != "artifact_receipts_path"}
        compact["training_log"] = {key: value for key, value in arm["training_log"].items() if key != "loss_values"}
        compact["timing"] = {key: value for key, value in arm["timing"].items()
                             if key not in ("full_step_samples_seconds", "optimizer_step_samples_seconds")}
        compact["diagnostics"] = {key: value for key, value in arm["diagnostics"].items() if key != "summary"}
        compact["diagnostics"]["per_step"] = [item for item in compact["diagnostics"]["per_step"]
                                              if item["group"] == "all_routed" or item["iteration"] == 196]
        compact["diagnostics"]["relay_per_step_scope"] = "all_routed at four spectral steps; all groups at196; full records remain on NFS"
        compact["artifact_receipts_file"] = artifact(arm["artifact_receipts_path"])
        compact_arms.append(compact)
    relay_result = {key: value for key, value in result.items() if key not in ("arms", "raw_artifact_receipts")}
    relay_result["arms"] = compact_arms
    relay_result["raw_artifact_receipt_count"] = len(all_receipts)
    relay_result["detailed_summary_path"] = str(OUTPUT / "measurement-summary.json")
    write_json(OUTPUT / "relay-summary.json", relay_result)
    raw = []
    for label, path in (("measurement-summary", OUTPUT / "relay-summary.json"),
                        ("transparency-results", transparency_path), ("alpha-receipt", OUTPUT / "alpha-receipt.json")):
        data = path.read_bytes()
        raw.append({"label": label, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "content_base64": base64.b64encode(data).decode()})
    sealed_path = OUTPUT / "small-artifacts.json"
    write_json(sealed_path, {"schema": "hmoe-lionmoe-overnight013-small-artifacts-v1", "nonsecret": True,
                             "artifacts": raw})
    sealed = sealed_path.read_bytes()
    if len(sealed) > 256 * 1024:
        raise RuntimeError("Sealed measurement report exceeds 256 KiB")
    encoded = base64.b64encode(sealed).decode()
    print(f"OVERNIGHT013_ARTIFACT path={sealed_path} bytes={len(sealed)} "
          f"sha256={hashlib.sha256(sealed).hexdigest()} base64_bytes={len(encoded)}", flush=True)
    chunks = [encoded[index:index + 12000] for index in range(0, len(encoded), 12000)]
    for index, chunk in enumerate(chunks):
        print(f"OVERNIGHT013_B64 {index}/{len(chunks)} {chunk}", flush=True)
    print("OVERNIGHT013_B64_END", flush=True)
    print("OVERNIGHT013_RESULT " + json.dumps({key: value for key, value in result.items() if key != "arms"},
                                                sort_keys=True), flush=True)
    return 0 if passed else 1


def main():
    try:
        return main_checked()
    except Exception as error:
        available = []
        for path in (OUTPUT / "runtime-gates.json", OUTPUT / "alpha-receipt.json",
                     OUTPUT / "overnight-contracts/contract-results.json",
                     OUTPUT / "progress.json", *OUTPUT.glob("arms/*/analysis.json"), *OUTPUT.glob("arms/*/artifact-receipts.json")):
            if path.is_file():
                available.append(artifact(path))
        partial = {"schema": "hmoe-lionmoe-overnight013-error-v1", "passed": False,
                   "job_note": os.environ["HMOE_JOB_NOTE"], "completed_stages": STAGES,
                   "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()[-4000:],
                   "available_artifacts": available, "quality_speed_memory_benefit_established": False,
                   "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write_json(OUTPUT / "partial-error.json", partial)
        data = (OUTPUT / "partial-error.json").read_bytes()
        sealed_path = OUTPUT / "small-artifacts.json"
        write_json(sealed_path, {"schema": "hmoe-lionmoe-overnight013-small-artifacts-v1", "nonsecret": True,
                   "artifacts": [{"label": "partial-error", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                                  "content_base64": base64.b64encode(data).decode()}]})
        sealed = sealed_path.read_bytes()
        if len(sealed) > 256 * 1024:
            raise RuntimeError("Sealed partial report exceeds256KiB")
        encoded = base64.b64encode(sealed).decode()
        print(f"OVERNIGHT013_ARTIFACT path={sealed_path} bytes={len(sealed)} "
              f"sha256={hashlib.sha256(sealed).hexdigest()} base64_bytes={len(encoded)}", flush=True)
        chunks = [encoded[index:index + 12000] for index in range(0, len(encoded), 12000)]
        for index, chunk in enumerate(chunks):
            print(f"OVERNIGHT013_B64 {index}/{len(chunks)} {chunk}", flush=True)
        print("OVERNIGHT013_B64_END", flush=True)
        print("OVERNIGHT013_RESULT " + json.dumps({"passed": False, "completed_stages": STAGES}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
