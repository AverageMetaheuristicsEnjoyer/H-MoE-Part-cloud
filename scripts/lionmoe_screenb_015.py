"""Screenb-015 v029: fresh capacity/data/telemetry audit + matched development screen B.

Stage 0 reuses the fullrun014 audit (exact bytes/inodes/mount/persistence/fsync, data/index identity,
INT64 telemetry fixture) plus the overnight013 alpha-identity/no-test contracts. Audit artifacts are
sealed and relayed BEFORE training (the v027 pattern) and survive a later training failure.

Stage 1 runs two checkpoint-free from-scratch 600-step arms, seed1234, in order:
  A lm5r50      control   PERIOD=5, LR_RATIO=50 (realized Muon steps exactly 120)
  B lmlinrevr50 candidate PROB=0:0.35,1:0.05, LR_RATIO=50 (realized count reported, ~207 expected)
The 600-step window is NOT budget-matched; the schedules match in expectation only over the full
17242 horizon, which is kept unchanged (WSD warmup173/decay3448, mb16/gb208, 1 GPU, BF16 compute /
FP32 states, Muon LR 1.63e-3, WD 0.1, NS5, betas 0.9/0.99). Development eval at 300/600, eval_iters32,
identical evaluator/order; explicit no-test guard; no save/load/checkpoints/weights export.
Realized Muon-step counts come from the passive default-off LIONMUON_STEP print in LionMuonMixin.step
(no optimizer math/dtype/RNG/return change) and are cross-checked against the declared PROB semantics.
EnergyCal and fixed routed alpha are off (alpha=1) in both arms. Timing is descriptive only (ordered,
cross-arm; not a speed claim). The routing observer keeps the INT64 windows 181-200 of overnight013.
"""
import base64
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
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
MONO_START = time.monotonic()
STAGES = []
VERSION = os.environ.get("HMOE_SCREENB_VERSION", "029")
EXPECTED_NOTE = f"hmoe-lionmoe-20261123-screenb-015-v{VERSION}"
EXPECTED_ROOT = f"/workspace-SR006.nfs3/hmoe-lionmoe-20261123/screenb-015-v{VERSION}"
EXPECTED_BUNDLE = f"hmoe-lionmoe-20261123-v{VERSION}"
SEED = 1234
SCHEDULE_ITERS = 17242
TRAIN_STEPS = 600
PAYLOAD_CAP_SECONDS = int(os.environ.get("HMOE_SCREENB_PAYLOAD_CAP_SECONDS", "16200"))
ARM_TIMEOUT_SECONDS = int(os.environ.get("HMOE_SCREENB_ARM_TIMEOUT_SECONDS", "7200"))
SEAL_RESERVE_SECONDS = 300
if os.environ.get("HMOE_JOB_NOTE") != EXPECTED_NOTE:
    raise RuntimeError("HMOE_JOB_NOTE does not match screenb015/version")
if os.environ.get("HMOE_CONTRACT_ROOT") != EXPECTED_ROOT:
    raise RuntimeError("HMOE_CONTRACT_ROOT does not match screenb015/version")
if os.environ.get("EXPECTED_BUNDLE_ID") != EXPECTED_BUNDLE:
    raise RuntimeError("Unexpected bundle for screenb015")
for key, length in (("EXPECTED_SOURCE_COMMIT", 40), ("EXPECTED_CARRIER_COMMIT", 40), ("EXPECTED_PAYLOAD_SHA256", 64)):
    value = os.environ.get(key, "")
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"Missing/invalid runtime pin: {key}")
if PAYLOAD_CAP_SECONDS < 2 * ARM_TIMEOUT_SECONDS + 600:
    raise RuntimeError("Payload cap cannot hold two arms plus audit/seal reserve")


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


def seal_envelope(filename, schema, labeled_paths, marker_prefix):
    raw = []
    for label, path in labeled_paths:
        data = Path(path).read_bytes()
        raw.append({"label": label, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "content_base64": base64.b64encode(data).decode()})
    sealed_path = OUTPUT / filename
    write_json(sealed_path, {"schema": schema, "nonsecret": True, "artifacts": raw})
    sealed = sealed_path.read_bytes()
    if len(sealed) > 256 * 1024:
        raise RuntimeError(f"Sealed envelope {filename} exceeds 256 KiB")
    encoded = base64.b64encode(sealed).decode()
    print(f"{marker_prefix}_ARTIFACT path={sealed_path} bytes={len(sealed)} "
          f"sha256={hashlib.sha256(sealed).hexdigest()} base64_bytes={len(encoded)}", flush=True)
    chunks = [encoded[index:index + 12000] for index in range(0, len(encoded), 12000)]
    for index, chunk in enumerate(chunks):
        print(f"{marker_prefix}_B64 {index}/{len(chunks)} {chunk}", flush=True)
    print(f"{marker_prefix}_B64_END", flush=True)
    return {"path": str(sealed_path), "bytes": len(sealed), "sha256": hashlib.sha256(sealed).hexdigest()}


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
        raise RuntimeError("Less than 2 GiB available for checkpoint-free screen artifacts")
    tiny = OUTPUT / "runtime-roundtrip.bin"
    payload = b"hmoe-lionmoe-screenb-015\n"
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
        raise RuntimeError("Screen requires exactly one visible GPU")
    return {"volumes": volumes, "tiny_roundtrip": artifact(tiny), "corpus_root": str(corpus),
            "corpus_files": files, "corpus_manifest_sha256": manifest_sha,
            "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformer_engine": transformer_engine_version,
            "cuda_available": True, "device_count": 1,
            "weights_saved_or_downloaded": False, "heavy_channel_required_for_screen": False}


def run_screen_contracts(env):
    code = run_process([sys.executable, str(ROOT / "tests/stage3_moe/test_lionmuon_overnight.py")],
                       env, OUTPUT / "contracts.stdout.log", OUTPUT / "contracts.stderr.log", 120)
    path = OUTPUT / "overnight-contracts/contract-results.json"
    if code != 0 or not path.is_file():
        detail = (OUTPUT / "contracts.stderr.log").read_text(errors="replace")[-4000:]
        raise RuntimeError(f"Screen contracts failed: exit={code}; stderr={detail}")
    result = json.loads(path.read_text())
    if not result["passed"] or not result["alpha_identity"]["nonzero_params_and_momentum_equal"] \
            or not result["no_test_guard"]["test_prefix_blocked"]:
        raise RuntimeError("Contracts did not exercise alpha identity and no-test guard")
    return code, result


def parse_training_log(path):
    text = Path(path).read_text(errors="replace")
    iterations = [int(value) for value in re.findall(r"iteration\s+(\d+)/", text)]
    losses = [float(value) for value in re.findall(r"lm loss:\s*([0-9.eE+-]+)", text)]
    validation_matches = re.findall(r"validation loss at iteration\s+(\d+)(?: on validation set)? \| lm loss value: ([0-9.eE+-]+)", text)
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


STEP_TELEMETRY_RE = re.compile(
    r"^LIONMUON_STEP t=(\d+) muon=(\d) muon_steps=(\d+) optimizer_steps=(\d+) rule=(\w+) value=([0-9.eE+-]+) power=([0-9.eE+-]+)$",
    re.MULTILINE)


def parse_step_telemetry(train_log):
    text = Path(train_log).read_text(errors="replace")
    rows = {}
    for match in STEP_TELEMETRY_RE.finditer(text):
        t = int(match.group(1))
        if t in rows:
            raise RuntimeError(f"Duplicate LIONMUON_STEP line at t={t}; expected one LionMuon instance")
        rows[t] = {"muon": int(match.group(2)), "muon_steps": int(match.group(3)),
                   "optimizer_steps": int(match.group(4)), "rule": match.group(5),
                   "value": float(match.group(6)), "power": float(match.group(7))}
    if set(rows) != set(range(1, TRAIN_STEPS + 1)):
        raise RuntimeError(f"LIONMUON_STEP coverage mismatch: {len(rows)} of {TRAIN_STEPS} steps")
    running = 0
    for t in range(1, TRAIN_STEPS + 1):
        row = rows[t]
        running += row["muon"]
        if row["optimizer_steps"] != t or row["muon_steps"] != running or row["power"] != 0.0:
            raise RuntimeError(f"LIONMUON_STEP counter inconsistency at t={t}")
    return rows


def prob_choice_checker():
    # Reuse the production schedule/draw semantics for a passive cross-check of the logged choices.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "third_party" / "emerging-optimizers"))
    from stage3_moe.lionmuon import parse_schedule, piecewise_linear

    points = parse_schedule("0:0.35,1:0.05", float)

    def expected(t):
        p = piecewise_linear(points, t / SCHEDULE_ITERS)
        return random.Random(f"lionmuon-{SEED}-{t}").random() < p, p

    return expected


def check_telemetry_semantics(preset, rows):
    if preset == "lm5r50":
        for t, row in rows.items():
            if row["rule"] != "period" or row["value"] != 5.0 or row["muon"] != int((t - 1) % 5 == 0):
                raise RuntimeError(f"Control realized choice deviates from P5 at t={t}")
        realized = rows[TRAIN_STEPS]["muon_steps"]
        if realized != 120:
            raise RuntimeError(f"Control realized Muon steps {realized} != 120")
        return {"realized_muon_steps": realized, "expected_muon_steps": 120, "rule": "period",
                "per_step_cross_check": "realized == (t-1)%5==0 for all 600 steps"}
    expected = prob_choice_checker()
    expected_total = 0
    for t, row in rows.items():
        choice, p = expected(t)
        expected_total += int(choice)
        if row["rule"] != "prob" or row["muon"] != int(choice) or not math.isclose(row["value"], p, rel_tol=1e-5, abs_tol=1e-6):
            raise RuntimeError(f"Candidate realized choice deviates from declared PROB semantics at t={t}")
    realized = rows[TRAIN_STEPS]["muon_steps"]
    return {"realized_muon_steps": realized, "expected_muon_steps": expected_total, "rule": "prob",
            "per_step_cross_check": "realized == Random('lionmuon-1234-t').random() < piecewise_linear(0:0.35,1:0.05, t/17242) for all 600 steps",
            "budget_scope": "600-step window is NOT budget-matched (P5 gives 120; lmlinrev ~207); matched in expectation only over the full 17242 horizon"}


def phase_summary(values, phases, wanted):
    selected = [value for value, phase_name in zip(values, phases) if phase_name == wanted]
    return {"count": len(selected), "sum_seconds": sum(selected),
            "mean_seconds": statistics.mean(selected) if selected else None,
            "median_seconds": statistics.median(selected) if selected else None,
            "min_seconds": min(selected) if selected else None, "max_seconds": max(selected) if selected else None}


def routing_int64_analysis(path):
    data = json.loads(path.read_text())
    if data["window"] != [181, 200] or len(data["layers"]) != 17:
        raise RuntimeError("Diagnostic routing window/layers mismatch")
    matrix = data["tokens_per_expert"]
    if len(matrix) != 17 or any(len(row) != 64 for row in matrix):
        raise RuntimeError("Diagnostic routing matrix schema mismatch")
    if any(not isinstance(value, int) or value < 0 for row in matrix for value in row):
        raise RuntimeError("Routing counts must be nonnegative integers (INT64 observer)")
    sums = [sum(row) for row in matrix]
    if any(total <= 0 or total % 8 for total in sums):
        raise RuntimeError("Routing row sums must be positive multiples of top8")
    valid_tokens = {total // 8 for total in sums}
    if len(valid_tokens) != 1:
        raise RuntimeError("Per-layer valid-token counts differ inside one window")
    tokens = valid_tokens.pop()
    if tokens > 20 * 208 * 2048:
        raise RuntimeError("Window valid-token count exceeds 20 steps of gb208 x seq2048")
    cv_max = 0.0
    for row in matrix:
        live = sum(value > 0 for value in row)
        if live < 8 or max(row) > tokens:
            raise RuntimeError("top8 dropless invariants violated (live>=8, per-expert<=valid tokens)")
        mean = statistics.mean(row)
        cv = statistics.pstdev(row) / mean
        if cv > 7**0.5 + 1e-9:
            raise RuntimeError("Routing CV exceeds the exact-top8 bound sqrt(7)")
        cv_max = max(cv_max, cv)
    return {"window": [181, 200], "scope": data["scope"], "valid_tokens_per_layer": tokens,
            "assignments_per_layer": 8 * tokens, "live_experts_min": min(sum(value > 0 for value in row) for row in matrix),
            "coefficient_of_variation_max": cv_max,
            "invariants": "sum=8*valid, per-expert<=valid, live>=8, CV<=sqrt7; INT64 exact accumulation",
            "dropped_tokens_scope": "zero only because configured dropless; not an independent event counter"}


def observer_analysis(preset, result_parent, telemetry):
    observer_path = result_parent / "energycal_diagnostics.jsonl"
    observer = [json.loads(line) for line in observer_path.read_text().splitlines()]
    if len(observer) != 20 or [item["iteration"] for item in observer] != list(range(181, 201)):
        raise RuntimeError("Observer window records mismatch")
    grid = [t for t in range(181, 201) if (t - 1) % 5 == 0]
    covered = [t for t in grid if telemetry[t]["muon"] == 1]
    for item in observer:
        t = item["iteration"]
        if item["spectral"] != (t in grid):
            raise RuntimeError("Observer spectral flag is not the period-grid label")
        expected_records = 31 if t in covered else 0
        if item["selected_records"] != expected_records:
            raise RuntimeError(f"Observer selected records mismatch at t={t}")
    blocks_path = result_parent / "energycal_blocks.jsonl"
    blocks = [json.loads(line) for line in blocks_path.read_text().splitlines() if line.strip()]
    if len(blocks) != 31 * len(covered):
        raise RuntimeError(f"Expected {31 * len(covered)} block records, got {len(blocks)}")
    keys = [(record["iteration"], record["layer"], record["expert"], record["fc_type"]) for record in blocks]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate block key")
    if any(record["nonfinite"] for record in blocks):
        raise RuntimeError("Nonfinite block record")
    if any(record["applied"] or record["actual_q"] is not None for record in blocks):
        raise RuntimeError("EnergyCal/fixed alpha applied while both are off")
    for t in covered:
        step = [record for record in blocks if record["iteration"] == t]
        if len(step) != 31 or sum(record["routed"] for record in step) != 27:
            raise RuntimeError(f"Block coverage mismatch at covered step {t}")
    q_summary = json.loads((result_parent / "energycal_summary.json").read_text())
    if q_summary["window"] != [181, 200] or q_summary["spectral_steps"] != covered:
        raise RuntimeError("Observer summary did not derive the covered window steps")
    callbacks = sum(item["observer_callback_seconds"] for item in observer)
    windows = sum(item["observer_window_seconds"] for item in observer)
    return {"window": [181, 200], "period_grid_steps": grid, "covered_realized_spectral_steps": covered,
            "block_records": len(blocks), "routed_blocks_per_covered_step": 27,
            "energycal_applied": False, "fixed_alpha_applied": False,
            "observer_callback_seconds": callbacks, "observer_window_seconds": windows,
            "grid_label_scope": ("observer 'spectral' is the (t-1)%5 grid label; under PROB it is not the realized choice; "
                                 "realized choices come from LIONMUON_STEP telemetry" if preset != "lm5r50" else
                                 "period arm: grid label equals realized choice"),
            "timing_note": "observer windows include normal optimizer work; no incremental overhead claim"}, [observer_path, blocks_path, result_parent / "energycal_summary.json"]


def arm_analysis(preset, result_path, bench_log, train_log, init_receipt, arm_root, wrapper_code, duration, wrapper_stdout):
    result = json.loads(result_path.read_text().splitlines()[-1])
    bench_text = Path(bench_log).read_text(errors="replace")
    bench = parse_training_log(bench_log)
    training = parse_training_log(train_log)
    argv = result["provenance"]["argv"]
    assert "--save" not in argv and "--load" not in argv
    assert not any(argument.startswith("--fp8") for argument in argv)
    assert argv[argv.index("--train-iters") + 1] == "600"
    assert argv[argv.index("--eval-iters") + 1] == "32"
    assert argv[argv.index("--eval-interval") + 1] == "300"
    assert argv[argv.index("--seed") + 1] == str(SEED)
    assert argv[argv.index("--lr-decay-iters") + 1] == "17242"
    assert argv[argv.index("--lr-warmup-iters") + 1] == "173"
    assert argv[argv.index("--lr-wsd-decay-iters") + 1] == "3448"
    assert argv[argv.index("--muon-num-ns-steps") + 1] == "5"
    assert argv[argv.index("--optimizer") + 1] == "muon"
    assert "--bf16" in argv
    denominators = result["denominators"]
    assert denominators["micro_batch_sequences_per_gpu"] == 16
    assert denominators["global_batch_sequences"] == 208
    assert denominators["gpu_count"] == 1
    assert denominators["total_parameters"] == 1028926976
    train_exits = re.findall(r"^TRAIN_EXIT=(\d+)$", bench_text, re.MULTILINE)
    assert train_exits == ["0"], f"TRAIN_EXIT lines: {train_exits}"
    assert re.search(r"^SCHEDULE target_iters=17242 decay_iters=3448 warmup=173 train_iters=600$", bench_text, re.MULTILINE)
    assert re.search(r"^SCHEDULE_ITERS=17242$", bench_text, re.MULTILINE)
    assert re.search(r"^OPTIMIZER_HPARAMS lr=1\.63e-3 .*weight_decay=0\.1$", bench_text, re.MULTILINE)
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
    preset_lines = [line for line in Path(wrapper_stdout).read_text(errors="replace").splitlines()
                    if line.startswith(f"COAT_ARM_MUON_PRESET name={preset} ")]
    if len(preset_lines) != 1:
        raise RuntimeError("Wrapper preset echo missing/duplicated")
    preset_line = preset_lines[0]
    assert "STAGE3_MOE_LIONMUON_LR_RATIO=50" in preset_line
    if preset == "lm5r50":
        assert "STAGE3_MOE_LIONMUON_PERIOD=5" in preset_line and "PROB=" not in preset_line
    else:
        assert "STAGE3_MOE_LIONMUON_PROB=0:0.35,1:0.05" in preset_line and "PERIOD=" not in preset_line
    first_config = re.search(r"^LIONMUON step=1 .*beta1=([0-9.]+) beta2=([0-9.]+) lr_ratio=([0-9.]+) energycal=(\d) .*fixed_routed_alpha=([0-9.]+)",
                             Path(train_log).read_text(errors="replace"), re.MULTILINE)
    if not first_config:
        raise RuntimeError("Missing LIONMUON step=1 configuration line")
    assert first_config.group(1) == "0.9" and first_config.group(2) == "0.99"
    assert float(first_config.group(3)) == 50.0
    assert first_config.group(4) == "0" and float(first_config.group(5)) == 1.0
    telemetry = parse_step_telemetry(train_log)
    realized = check_telemetry_semantics(preset, telemetry)
    timing = result["measurement"]["timing"]
    full = timing["full_step_samples_seconds"]
    optimizer = timing["optimizer_step_samples_seconds"]
    assert len(full) == len(optimizer) == 40
    iterations = list(range(551, 591))
    phases = ["spectral" if telemetry[t]["muon"] else "sign" for t in iterations]
    loss_tokens = denominators["loss_tokens_per_step"]
    throughput = 40 * loss_tokens / sum(full)
    init = json.loads(init_receipt.read_text())
    assert init["load"] is None and init["iteration"] == 0 and init["consumed_train_samples"] == 0
    assert init["parameters"] == 1028926976 and init["seed"] == SEED
    diagnostics, diagnostic_paths = observer_analysis(preset, result_path.parent, telemetry)
    routing_path = result_path.parent / "routing_diagnostics_181_200.json"
    routing = routing_int64_analysis(routing_path)
    tokens_path = result_path.parent / "tokens_per_expert.json"
    tokens = json.loads(tokens_path.read_text())
    if tokens["window"] != "measured_steps_only":
        raise RuntimeError("Timing routing window label mismatch")
    artifact_paths = [result_path, bench_log, train_log, init_receipt, routing_path, tokens_path,
                      result_path.parent / "routing_telemetry.jsonl", *diagnostic_paths]
    chronological = []
    for start, end in ((551, 560), (561, 570), (571, 580), (581, 590), (551, 570), (571, 590)):
        selected = [(iteration, value, phase) for iteration, value, phase in zip(iterations, full, phases)
                    if start <= iteration <= end]
        values = [item[1] for item in selected]
        spectral_values = [item[1] for item in selected if item[2] == "spectral"]
        sign_values = [item[1] for item in selected if item[2] == "sign"]
        chronological.append({"start": start, "end": end, "count": len(values), "sum_seconds": sum(values),
                              "mean_seconds": statistics.mean(values), "median_seconds": statistics.median(values),
                              "min_seconds": min(values), "max_seconds": max(values),
                              "descriptive_tokens_per_second": len(values) * loss_tokens / sum(values),
                              "spectral_mean_seconds": statistics.mean(spectral_values) if spectral_values else None,
                              "sign_mean_seconds": statistics.mean(sign_values) if sign_values else None})
    for path in result_path.parent.rglob("*"):
        if path.is_file() and path not in artifact_paths:
            artifact_paths.append(path)
    for suffix in (".pt", ".pth", ".safetensors", ".ckpt"):
        assert not list(arm_root.rglob(f"*{suffix}"))
    eval_argv = {option: argv[argv.index(option) + 1] for option in
                 ("--eval-iters", "--eval-interval", "--valid-data-path", "--test-data-path",
                  "--dataloader-type", "--seed", "--global-batch-size", "--micro-batch-size", "--seq-length")}
    return {"preset": preset, "status": "valid", "wrapper_exit": wrapper_code,
            "train_exit": 0, "reached": 600, "measured_steps": 40, "probe_e2e_steps": 590,
            "duration_seconds": duration, "result_path": str(result_path),
            "init_receipt": init, "data_fingerprint": {"manifest_sha256": init["data_manifest_sha256"],
                                                       "seed": SEED, "data_order": "same seed/order contract; batch content not separately hashed"},
            "realized_muon_steps": realized, "eval_argv": eval_argv,
            "timing": {"full_step_samples_seconds": full, "optimizer_step_samples_seconds": optimizer,
                       "full_step_sum_seconds": sum(full), "full_step_mean_seconds": statistics.mean(full),
                       "full_step_median_seconds": statistics.median(full),
                       "optimizer_step_mean_seconds": statistics.mean(optimizer),
                       "optimizer_step_median_seconds": statistics.median(optimizer),
                       "descriptive_tokens_per_second": throughput,
                       "probe_tokens_per_second_median_derived": timing["tokens_per_second"],
                       "probe_e2e_wct_seconds": timing["e2e_wct_seconds"],
                       "phase_full_step": {name: phase_summary(full, phases, name) for name in ("spectral", "sign")},
                       "phase_optimizer_step": {name: phase_summary(optimizer, phases, name) for name in ("spectral", "sign")},
                       "phase_counts": {"spectral": phases.count("spectral"), "sign": phases.count("sign")},
                       "phase_label_source": "realized LIONMUON_STEP choices, not the period grid",
                       "chronological_windows": chronological,
                       "wallclock_scope": "sum(train_step) is not total wall clock; ordered cross-arm timing is descriptive only, not a speed claim"},
            "memory": result["measurement"]["memory"], "optimizer_state": result["optimizer_state"],
            "loss": result["measurement"]["loss"], "training_log": training,
            "development": {"split": "development", "eval_iters": 32, "global_batch": 208,
                            "primary_iteration": 600, "secondary_iteration": 300,
                            "loss_at_300": training["scheduled_validation"][300],
                            "loss_at_600": training["scheduled_validation"][600],
                            "selection_scope": "first scheduled log per iteration; final duplicate not averaged",
                            "test_evaluated": False, "skipped_test_calls": training["skipped_test_evaluation_calls"]},
            "routing": result["measurement"]["routing"],
            "diagnostics": diagnostics, "diagnostic_routing": routing,
            "artifact_receipts_path": str(arm_root / "artifact-receipts.json")}, [artifact(path) for path in artifact_paths]


def arm_environment(base_env, preset, arm_root, unique, corpus_root):
    arm_env = dict(base_env)
    for key in ("STAGE3_MOE_LIONMUON_PERIOD_EXPERTS", "STAGE3_MOE_LIONMUON_PROB", "STAGE3_MOE_LIONMUON_POWER",
                "STAGE3_MOE_LIONMUON_PERIOD", "STAGE3_MOE_LIONMUON_BETA1", "STAGE3_MOE_LIONMUON_BETA2",
                "STAGE3_MOE_LIONMUON_ENERGYCAL_DIAGNOSTICS", "STAGE3_MOE_MUON_NS_STEPS",
                "STAGE3_MOE_FP8_COMPUTE_ARGS", "STAGE3_MOE_FP8_STATE_DTYPES", "STAGE3_MOE_DEQUANT_ON_LOAD",
                "STAGE3_MOE_FP32_MODEL", "STAGE3_MOE_EXTRA_ARGS", "STAGE3_MOE_PROBE_BASE_ITER",
                "STAGE3_MOE_FULL_TRAIN_ITERS", "COAT_STOP_ITERS"):
        arm_env.pop(key, None)
    arm_env.update({
        "COAT_ARM_MODE": "bench", "COAT_BENCH_PROPAGATE_EXIT": "1",
        "COAT_BENCH_WARMUP": "550", "COAT_BENCH_MEASURE": "40", "COAT_BENCH_TAG": preset,
        "COAT_CAMPAIGN_ROOT": str(OUTPUT / "campaign"), "COAT_STAGE_ROOT": str(OUTPUT / "stage"),
        "COAT_RUN_TAG": unique, "COAT_ARM_ID_PREFIX": unique + "-", "COAT_ROUND": f"screenb015v{VERSION}",
        "STAGE3_MOE_LOG_ROOT": str(arm_root / "logs"), "STAGE3_MOE_CKPT_ROOT": str(arm_root / "no-checkpoints"),
        "STAGE3_MOE_DATA_CACHE_PATH": str(arm_root / "data-cache"),
        "STAGE3_MOE_COLDSTART_RECEIPT": str(arm_root / "init-receipt.json"),
        "STAGE3_MOE_DATA_ROOT": corpus_root, "STAGE3_MOE_EVAL_ITERS": "32",
        "STAGE3_MOE_DISABLE_TEST_EVAL": "1", "STAGE3_MOE_EVAL_INTERVAL": "300", "STAGE3_MOE_LOG_INTERVAL": "1",
        "STAGE3_MOE_ROUTING_TELEMETRY_INTERVAL": "10", "STAGE3_MOE_ROUTING_DIAGNOSTICS": "1",
        "STAGE3_MOE_ENERGYCAL_DIAGNOSTICS_START": "181", "STAGE3_MOE_ENERGYCAL_DIAGNOSTICS_END": "200",
        "STAGE3_MOE_NPROC": "1",
        "STAGE3_MOE_MICRO_BATCH": "16", "STAGE3_MOE_EP": "1", "STAGE3_MOE_SEED": str(SEED),
        "STAGE3_MOE_SCHEDULE_ITERS": str(SCHEDULE_ITERS),
        "STAGE3_MOE_MUON_PRESET": preset, "STAGE3_MOE_MUON_VARIANT": "lionmuon",
        "STAGE3_MOE_LIONMUON_LR_RATIO": "50", "STAGE3_MOE_LIONMUON_SEED": str(SEED),
        "STAGE3_MOE_LIONMUON_BETA1": "0.9", "STAGE3_MOE_LIONMUON_BETA2": "0.99",
        "STAGE3_MOE_LIONMUON_FIXED_ROUTED_ALPHA": "1", "STAGE3_MOE_LIONMUON_ENERGYCAL": "0",
        "STAGE3_MOE_LIONMUON_STEP_TELEMETRY": "1",
        "WANDB_MODE": "offline", "WANDB_RUN_ID": unique,
    })
    if preset == "lm5r50":
        arm_env["STAGE3_MOE_LIONMUON_PERIOD"] = "5"
    return arm_env


def main_checked():
    runtime = runtime_gates()
    write_json(OUTPUT / "runtime-gates.json", runtime)
    mark_stage("runtime")
    # Stage 0: fresh audit, reusing the fullrun014 functions under this task's own pins/namespace.
    sys.path.insert(0, str(ROOT / "scripts"))
    import lionmoe_fullrun_014 as audit014

    audit014.VERSION = VERSION
    audit014.OUTPUT = OUTPUT
    capacity_path = OUTPUT / "capacity-report.json"
    data_path = OUTPUT / "data-identity.json"
    write_json(capacity_path, audit014.capacity(probe_namespace="hmoe-lionmoe-20261123",
                                                probe_label="screenb-015", probe_marker="screenb015"))
    write_json(data_path, audit014.data_identity())
    telemetry = audit014.run_telemetry()
    mark_stage("audit_collected")
    env = dict(os.environ)
    env.pop("STAGE3_MOE_LIONMUON_ENERGYCAL_DIAGNOSTICS", None)
    env["STAGE3_MOE_MEASUREMENT_ENERGYCAL_DIAGNOSTICS"] = "1"
    contracts_code, contracts = run_screen_contracts(env)
    mark_stage("screen_contracts")
    audit_summary = {
        "schema": "hmoe-lionmoe-screenb015-audit-v1", "audit_completed": True,
        "capacity_candidates": json.loads(capacity_path.read_text())["candidates"],
        "brainlab": json.loads(capacity_path.read_text())["brainlab"],
        "telemetry_passed": telemetry["passed"], "contracts_passed": contracts["passed"],
        "data_identity_collected": True,
        "scope": "checkpoint-free 600-step development screen; full-state checkpoint admission is NOT claimed",
        "long_run_admitted": False, "training_invoked": False, "checkpoint_saved": False,
        "identity": {key: os.environ[key] for key in ("EXPECTED_SOURCE_COMMIT", "EXPECTED_CARRIER_COMMIT",
                                                      "EXPECTED_BUNDLE_ID", "EXPECTED_PAYLOAD_SHA256")},
    }
    audit_summary_path = OUTPUT / "audit-summary.json"
    write_json(audit_summary_path, audit_summary)
    audit_artifacts = [("audit-summary", audit_summary_path), ("capacity-report", capacity_path),
                       ("data-identity", data_path), ("telemetry-contracts", OUTPUT / "telemetry-contracts.json"),
                       ("screen-contracts", OUTPUT / "overnight-contracts/contract-results.json")]
    audit_envelope = seal_envelope("audit-small-artifacts.json", "hmoe-lionmoe-screenb015-audit-small-artifacts-v1",
                                   audit_artifacts, "SCREENB015_AUDIT")
    mark_stage("audit_sealed")
    # Stage 1: matched development screen B, control first then candidate.
    specs = ["lm5r50", "lmlinrevr50"]
    arms = []
    all_receipts = []
    failed = None
    for preset in specs:
        remaining = PAYLOAD_CAP_SECONDS - (time.monotonic() - MONO_START) - SEAL_RESERVE_SECONDS
        if remaining < ARM_TIMEOUT_SECONDS:
            failed = {"preset": preset, "reason": f"payload cap reserve insufficient for next arm: {remaining:.1f}s"}
            break
        arm_root = OUTPUT / "arms" / preset
        arm_root.mkdir(parents=True, exist_ok=True)
        unique = f"hmoe-lionmoe-20261123-screenb-015-v{VERSION}-{preset}-seed{SEED}"
        arm_env = arm_environment(env, preset, arm_root, unique, runtime["corpus_root"])
        arm_started = time.perf_counter()
        wrapper_code = run_process(["bash", str(ROOT / "scripts/run_coat_arm.sh"), "muon-ctrl"], arm_env,
                                   arm_root / "wrapper.stdout.log", arm_root / "wrapper.stderr.log", ARM_TIMEOUT_SECONDS)
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
        try:
            analysis, receipts = arm_analysis(preset, result_path, bench_log, match.group(1),
                                              arm_root / "init-receipt.json", arm_root, wrapper_code, duration,
                                              arm_root / "wrapper.stdout.log")
        except Exception as error:
            failed = {"preset": preset, "wrapper_exit": wrapper_code,
                      "reason": f"arm validation failed: {type(error).__name__}: {error}"}
            break
        write_json(arm_root / "analysis.json", analysis)
        write_json(arm_root / "artifact-receipts.json", receipts)
        arms.append(analysis)
        all_receipts.extend(receipts)
        mark_stage(f"arm_{preset}_valid")
    if len({arm["init_receipt"]["parameter_sha256"] for arm in arms}) != 1:
        raise RuntimeError("Fresh init hashes differ across arms")
    if len({arm["init_receipt"]["data_manifest_sha256"] for arm in arms}) != 1:
        raise RuntimeError("Data manifest differs across arms")
    if len({json.dumps(arm["eval_argv"], sort_keys=True) for arm in arms}) != 1:
        raise RuntimeError("Evaluator argv differs across arms")
    passed = failed is None and len(arms) == len(specs)
    development = [{"preset": arm["preset"], "seed": SEED,
                    "loss_at_300": arm["development"]["loss_at_300"],
                    "loss_at_600": arm["development"]["loss_at_600"]} for arm in arms]
    result = {"schema": "hmoe-lionmoe-screenb015-result-v1", "job_note": os.environ["HMOE_JOB_NOTE"],
              "version": VERSION, "seed": SEED, "schedule_iters": SCHEDULE_ITERS, "arm_order": specs,
              "development": development,
              "realized_muon_steps": {arm["preset"]: arm["realized_muon_steps"] for arm in arms},
              "muon_step_budget_scope": "600-step window is NOT budget-matched; P5 gives exactly 120 Muon steps, lmlinrev (0:0.35,1:0.05 over t/17242) gives ~207; matched in expectation only over the full horizon",
              "identity": {key: os.environ[key] for key in ("EXPECTED_SOURCE_COMMIT", "EXPECTED_CARRIER_COMMIT",
                                                              "EXPECTED_BUNDLE_ID", "EXPECTED_PAYLOAD_SHA256")},
              "training_invoked": True,
              "training_scope": "checkpoint-free from-scratch 600-step development screen B only; not full training",
              "weights_saved_or_downloaded": False, "test_evaluated": False,
              "energycal_enabled": False, "fixed_routed_alpha": 1.0,
              "runtime_gates": "passed", "audit_envelope": audit_envelope,
              "contracts_exit": contracts_code,
              "arms_completed": len(arms), "arms": arms, "failed": failed, "passed": passed,
              "fresh_init_and_data_manifest_equal": passed,
              "timing_scope": "ordered cross-arm descriptive timing only; not ABBA, not a speed claim",
              "quality_speed_memory_benefit_established": False,
              "recommendation": {"next": "review only; no automatic longer run, seed sweep or test evaluation"},
              "started_at": STARTED.isoformat(), "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "duration_seconds": round((datetime.datetime.now(datetime.timezone.utc) - STARTED).total_seconds(), 3),
              "payload_cap_seconds": PAYLOAD_CAP_SECONDS, "arm_timeout_seconds": ARM_TIMEOUT_SECONDS,
              "raw_artifact_receipts": all_receipts}
    write_json(OUTPUT / "measurement-summary.json", result)
    compact_arms = []
    for arm in arms:
        compact = {key: value for key, value in arm.items() if key != "artifact_receipts_path"}
        compact["training_log"] = {key: value for key, value in arm["training_log"].items() if key != "loss_values"}
        compact["timing"] = {key: value for key, value in arm["timing"].items()
                             if key not in ("full_step_samples_seconds", "optimizer_step_samples_seconds")}
        compact["artifact_receipts_file"] = artifact(arm["artifact_receipts_path"])
        compact_arms.append(compact)
    relay_result = {key: value for key, value in result.items() if key not in ("arms", "raw_artifact_receipts")}
    relay_result["arms"] = compact_arms
    relay_result["raw_artifact_receipt_count"] = len(all_receipts)
    relay_result["detailed_summary_path"] = str(OUTPUT / "measurement-summary.json")
    write_json(OUTPUT / "relay-summary.json", relay_result)
    final_artifacts = [("measurement-summary", OUTPUT / "relay-summary.json"), *audit_artifacts]
    try:
        envelope = seal_envelope("small-artifacts.json", "hmoe-lionmoe-screenb015-small-artifacts-v1",
                                 final_artifacts, "SCREENB015")
    except RuntimeError:
        # Audit content was already relayed before training; keep its receipts and relay the screen summary alone.
        relay_result["audit_content_scope"] = "audit artifact contents relayed in the stage-0 SCREENB015_AUDIT envelope; receipts only here"
        relay_result["audit_artifact_receipts"] = [artifact(path) for _, path in audit_artifacts]
        write_json(OUTPUT / "relay-summary.json", relay_result)
        envelope = seal_envelope("small-artifacts.json", "hmoe-lionmoe-screenb015-small-artifacts-v1",
                                 [("measurement-summary", OUTPUT / "relay-summary.json")], "SCREENB015")
    result["final_envelope"] = envelope
    write_json(OUTPUT / "measurement-summary.json", result)
    print("SCREENB015_RESULT " + json.dumps({key: value for key, value in result.items()
                                             if key not in ("arms", "raw_artifact_receipts")}, sort_keys=True), flush=True)
    return 0 if passed else 1


def main():
    try:
        return main_checked()
    except Exception as error:
        available = []
        for path in (OUTPUT / "runtime-gates.json", OUTPUT / "capacity-report.json", OUTPUT / "data-identity.json",
                     OUTPUT / "telemetry-contracts.json", OUTPUT / "overnight-contracts/contract-results.json",
                     OUTPUT / "audit-summary.json", OUTPUT / "audit-small-artifacts.json",
                     OUTPUT / "progress.json", *OUTPUT.glob("arms/*/analysis.json"), *OUTPUT.glob("arms/*/artifact-receipts.json")):
            if path.is_file():
                available.append(artifact(path))
        partial = {"schema": "hmoe-lionmoe-screenb015-error-v1", "passed": False,
                   "job_note": os.environ.get("HMOE_JOB_NOTE"), "completed_stages": STAGES,
                   "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()[-4000:],
                   "available_artifacts": available, "quality_speed_memory_benefit_established": False,
                   "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write_json(OUTPUT / "partial-error.json", partial)
        labeled = [("partial-error", OUTPUT / "partial-error.json")]
        for label, name in (("capacity-report", "capacity-report.json"), ("data-identity", "data-identity.json"),
                            ("telemetry-contracts", "telemetry-contracts.json"), ("audit-summary", "audit-summary.json")):
            if (OUTPUT / name).is_file():
                labeled.append((label, OUTPUT / name))
        try:
            seal_envelope("small-artifacts.json", "hmoe-lionmoe-screenb015-small-artifacts-v1", labeled, "SCREENB015")
        except RuntimeError:
            seal_envelope("small-artifacts.json", "hmoe-lionmoe-screenb015-small-artifacts-v1",
                          [("partial-error", OUTPUT / "partial-error.json")], "SCREENB015")
        print("SCREENB015_RESULT " + json.dumps({"passed": False, "completed_stages": STAGES}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
