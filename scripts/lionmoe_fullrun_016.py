"""Fullrun-016 v031: lean-retention full-run machinery + HF archive + cross-allocation restore.

Implements the user-authorized 2026-11-23 lean-retention revision of research/fullrun-014/PLAN.md
(2 full arms: EnergyCal R50 then lmlinrev R50; the October plain R50 baseline is reused, re-evaluated
by baseline-eval-030). Modes (HMOE_FULLRUN016_MODE):
  gate-save     first 50 steps of the REAL EnergyCal arm (lm5r50ec), exit control (train_iters stays
                17242), full-state save + fsync/atomic per-file SHA256 manifest + HF upload + API verify.
  gate-restore  NEW allocation: fetch that full-state from HF, hash-verify BEFORE load, FULL load
                (optimizer+scheduler+RNG), verify model/master/momentum/scheduler/iteration/consumed/
                RNG/data-position/P5 phase BEFORE new updates (stage3_moe/fullrun_state_verify.py),
                then a few finite nonzero updates, sealed verification report.
  continue      resume the EnergyCal arm to 17242 with rolling full-state saves (interval configurable,
                default 726), periodic HF upload+verify+local rotation, dev eval every 1000 (eval_iters32),
                pre-decay 13794 full-state retained+uploaded, final 17242 WEIGHTS-ONLY export+upload,
                then the identical eval-full-dev endpoint evaluation (v030 contract).
  arm-lmlinrev  the same protocol from scratch with preset lmlinrevr50 (PROB 0:0.35,1:0.05, R50).

Lean retention: rolling latest full-state; pre-decay 13794 full-state; final weights-only. Previous-good
NOT kept. Local deletion ONLY after a hash-verified durable HF successor. Runtime free-space guard before
each segment; heavy staging in job-local /tmp; HF is the primary durable archive (5.2 TB headroom verified).
Storage failure => safe stop at the last good checkpoint, never fill the volume. No production math/dtype
change; the only optimizer-adjacent additions are passive verification hashing and the v029 realized-choice
telemetry. A strict-load gate (skeptic v030 major) refuses any segment/eval whose log shows MCore's
silent strict=False fallback ('load_return:' marker) and requires the 'successfully loaded checkpoint
... at iteration N' success marker; this is in addition to the pre-update model/master/momentum hash
checks. Resume traps handled: full load (no --no-load-optim so scheduler/RNG restore), param-group
collapse (MCore _filter_and_reorder_param_groups keeps same-key groups in order; structural check in the
verify hook), EnergyCal metadata mismatch (LionMuon.load_state_dict raises explicitly).
"""
import base64
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUTPUT = Path(os.environ["HMOE_CONTRACT_ROOT"])
OUTPUT.mkdir(parents=True, exist_ok=True)
STARTED = datetime.datetime.now(datetime.timezone.utc)
MONO_START = time.monotonic()
STAGES = []
VERSION = os.environ.get("HMOE_FULLRUN016_VERSION", "031")
MODE = os.environ.get("HMOE_FULLRUN016_MODE", "")
if MODE not in ("gate-save", "gate-restore", "continue", "arm-lmlinrev"):
    raise RuntimeError("HMOE_FULLRUN016_MODE must be gate-save|gate-restore|continue|arm-lmlinrev")
ARM = "lmlinrevr50" if MODE == "arm-lmlinrev" else os.environ.get("HMOE_FULLRUN016_ARM", "lm5r50ec")
if ARM not in ("lm5r50ec", "lmlinrevr50"):
    raise RuntimeError("Unsupported fullrun016 arm")
EXPECTED_NOTE = f"hmoe-lionmoe-20261123-fullrun-016-v{VERSION}"
EXPECTED_ROOT = f"/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v{VERSION}"
EXPECTED_BUNDLE = f"hmoe-lionmoe-20261123-v{VERSION}"
SEED = 1234
FULL_ITERS = 17242
PRE_DECAY = 13794
SAVE_INTERVAL = int(os.environ.get("HMOE_FULLRUN016_SAVE_INTERVAL", "726"))
SEGMENT_STEPS = int(os.environ.get("HMOE_FULLRUN016_SEGMENT_STEPS", "2178"))
CAP_SECONDS = int(os.environ.get("HMOE_FULLRUN016_CAP_SECONDS", "3600"))
SEAL_RESERVE = 300
SEC_PER_STEP = float(os.environ.get("HMOE_FULLRUN016_SEC_PER_STEP", "8"))
HF_REPO = "AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints"
HF_PREFIX = f"fullrun-016/{ARM}"
CORPUS_MANIFEST_SHA = "7a7cf91633081dd080f3cff96407bd61d4c237d88a5e926cffe36fc5b8cbfcc8"
TMP = Path(f"/tmp/hmoe-lionmoe-20261123-fullrun-016-v{VERSION}/{ARM}")
CKPT_ROOT = TMP / "ckpt"
FULL_DIR = CKPT_ROOT / "1c" / "muon_bf16_state_fp32"
ARM_NFS = OUTPUT / "arms" / ARM
DUMP_DIR = ARM_NFS / "state-manifests"
if os.environ.get("HMOE_JOB_NOTE") != EXPECTED_NOTE:
    raise RuntimeError("HMOE_JOB_NOTE does not match fullrun016/version")
if os.environ.get("HMOE_CONTRACT_ROOT") != EXPECTED_ROOT:
    raise RuntimeError("HMOE_CONTRACT_ROOT does not match fullrun016/version")
if os.environ.get("EXPECTED_BUNDLE_ID") != EXPECTED_BUNDLE:
    raise RuntimeError("Unexpected bundle for fullrun016")
for key, length in (("EXPECTED_SOURCE_COMMIT", 40), ("EXPECTED_CARRIER_COMMIT", 40), ("EXPECTED_PAYLOAD_SHA256", 64)):
    value = os.environ.get(key, "")
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"Missing/invalid runtime pin: {key}")

import hf_checkpoint_archive as hfarch  # noqa: E402


class FullrunStop(RuntimeError):
    """Safe stop at a good checkpoint (storage/budget); not a scientific failure."""


def mark_stage(name):
    if name not in STAGES:
        STAGES.append(name)
    write_json(OUTPUT / "progress.json", {"completed_stages": STAGES, "mode": MODE, "arm": ARM})


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def sha256(path):
    return hfarch.sha256_file(path)


def artifact(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def redact(text):
    token = os.environ.get("HF_TOKEN", "")
    return text.replace(token, "[REDACTED]") if token else text


def remaining():
    return CAP_SECONDS - (time.monotonic() - MONO_START) - SEAL_RESERVE


def run_process(command, env, stdout_path, stderr_path, timeout):
    with Path(stdout_path).open("w") as stdout, Path(stderr_path).open("w") as stderr:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        except Exception as error:
            stderr.write(f"PROCESS_PARENT_ERROR {type(error).__name__}: {error}\n")
            return 125
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    continue
            stderr.write(f"PROCESS_TIMEOUT_KILLED seconds={timeout}\n")
            stderr.flush()
            os.fsync(stderr.fileno())
            return 124
        except Exception as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stderr.write(f"PROCESS_PARENT_ERROR {type(error).__name__}: {error}\n")
            return 125


def seal_envelope(filename, schema, labeled_paths, marker_prefix="FULLRUN016"):
    raw = []
    for label, path in labeled_paths:
        data = Path(path).read_bytes()
        raw.append({"label": label, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "content_base64": base64.b64encode(data).decode()})
    sealed_path = OUTPUT / filename
    write_json(sealed_path, {"schema": schema, "nonsecret": True, "mode": MODE, "arm": ARM, "artifacts": raw})
    sealed = sealed_path.read_bytes()
    if len(sealed) > 256 * 1024:
        receipts = [{"label": label, **artifact(path)} for label, path in labeled_paths]
        write_json(sealed_path, {"schema": schema, "nonsecret": True, "mode": MODE, "arm": ARM,
                                 "artifacts": [], "degraded": "receipts-only: contents exceed 256 KiB; artifacts remain on NFS",
                                 "artifact_receipts": receipts})
        sealed = sealed_path.read_bytes()
        if len(sealed) > 256 * 1024:
            write_json(sealed_path, {"schema": schema, "nonsecret": True, "artifacts": [],
                                     "degraded": "receipts-only-truncated", "artifact_receipts": receipts[:8]})
            sealed = sealed_path.read_bytes()
    encoded = base64.b64encode(sealed).decode()
    print(f"{marker_prefix}_ARTIFACT path={sealed_path} bytes={len(sealed)} "
          f"sha256={hashlib.sha256(sealed).hexdigest()} base64_bytes={len(encoded)}", flush=True)
    chunks = [encoded[index:index + 12000] for index in range(0, len(encoded), 12000)]
    for index, chunk in enumerate(chunks):
        print(f"{marker_prefix}_B64 {index}/{len(chunks)} {chunk}", flush=True)
    print(f"{marker_prefix}_B64_END", flush=True)
    return {"path": str(sealed_path), "bytes": len(sealed), "sha256": hashlib.sha256(sealed).hexdigest()}


CLEANUP_KEYS = ("STAGE3_MOE_LR_SCALE", "STAGE3_MOE_LR_CONSTANT", "STAGE3_MOE_COLD_SWITCH",
                "STAGE3_MOE_TRAIN_DATA_PREFIX", "STAGE3_MOE_VALID_DATA_PREFIX", "STAGE3_MOE_TEST_DATA_PREFIX",
                "STAGE3_MOE_LR", "STAGE3_MOE_MIN_LR", "STAGE3_MOE_ADAM_BETA2",
                "STAGE3_MOE_WGRAD_FUSION", "STAGE3_MOE_MOE_FUSIONS", "STAGE3_MOE_OVERLAP_GRAD_REDUCE",
                "STAGE3_MOE_FP32_MODEL", "STAGE3_MOE_EXTRA_ARGS", "STAGE3_MOE_FP8_COMPUTE_ARGS",
                "STAGE3_MOE_FP8_STATE_DTYPES", "STAGE3_MOE_DEQUANT_ON_LOAD", "STAGE3_MOE_PROBE_BASE_ITER",
                "STAGE3_MOE_FULL_TRAIN_ITERS", "COAT_STOP_ITERS", "STAGE3_MOE_LIONMUON_PERIOD_EXPERTS",
                "STAGE3_MOE_LIONMUON_PROB", "STAGE3_MOE_LIONMUON_POWER", "STAGE3_MOE_MUON_NS_STEPS",
                "STAGE3_MOE_BF16_FUSED_EXPERTS", "STAGE3_MOE_COAT_STRICT", "STAGE3_MOE_COAT_FUSED_EXPERTS",
                "STAGE3_MOE_HESSIAN_PATH", "STAGE3_MOE_DIAGNOSTIC_PATH", "STAGE3_MOE_REPLAY_PATH",
                "STAGE3_MOE_M2_TRACE_PATH", "STAGE3_MOE_ACTIVATION_STATS_PATH", "STAGE3_MOE_COLDSTART_RECEIPT",
                "STAGE3_MOE_FULL_DEV_EVAL", "STAGE3_MOE_FULL_DEV_CAPTURE_PATH",
                "STAGE3_MOE_LIONMUON_ENERGYCAL_DIAGNOSTICS", "STAGE3_MOE_LIONMUON_PERIOD")


def base_arm_env(unique):
    env = dict(os.environ)
    removed = [key for key in CLEANUP_KEYS if key in env]
    for key in CLEANUP_KEYS:
        env.pop(key, None)
    env.update({
        "STAGE3_MOE_LOG_ROOT": str(ARM_NFS / "logs"),
        "STAGE3_MOE_CKPT_ROOT": str(CKPT_ROOT),
        "STAGE3_MOE_DATA_CACHE_PATH": str(TMP / "data-cache"),
        "STAGE3_MOE_DATA_ROOT": "/home/jovyan/data/fineweb-edu-gpt2-megatron/data",
        "STAGE3_MOE_RUN_SUFFIX": unique,
        "STAGE3_MOE_SEED": str(SEED), "STAGE3_MOE_LIONMUON_SEED": str(SEED),
        "STAGE3_MOE_NPROC": "1", "STAGE3_MOE_MICRO_BATCH": "16", "STAGE3_MOE_EP": "1",
        "STAGE3_MOE_EVAL_ITERS": "32", "STAGE3_MOE_EVAL_INTERVAL": "1000", "STAGE3_MOE_LOG_INTERVAL": "10",
        "STAGE3_MOE_DISABLE_TEST_EVAL": "1", "STAGE3_MOE_ROUTING_TELEMETRY_INTERVAL": "10",
        "STAGE3_MOE_SCHEDULE_ITERS": str(FULL_ITERS),
        "STAGE3_MOE_MUON_PRESET": ARM, "STAGE3_MOE_MUON_VARIANT": "lionmuon",
        "STAGE3_MOE_LIONMUON_LR_RATIO": "50", "STAGE3_MOE_LIONMUON_BETA1": "0.9", "STAGE3_MOE_LIONMUON_BETA2": "0.99",
        "STAGE3_MOE_LIONMUON_FIXED_ROUTED_ALPHA": "1",
        "STAGE3_MOE_LIONMUON_STEP_TELEMETRY": "1",
        "STAGE3_MOE_FULLRUN_STATE_DUMP": "1", "STAGE3_MOE_FULLRUN_DUMP_DIR": str(DUMP_DIR),
        "STAGE3_MOE_PROPAGATE_EXIT": "1",
        "STAGE3_MOE_ROUTING_DIAGNOSTICS": "1", "STAGE3_MOE_MEASUREMENT_ENERGYCAL_DIAGNOSTICS": "1",
        "STAGE3_MOE_ENERGYCAL_DIAGNOSTICS_START": "181", "STAGE3_MOE_ENERGYCAL_DIAGNOSTICS_END": "200",
        "WANDB_MODE": "offline", "WANDB_RUN_ID": unique,
    })
    if ARM == "lm5r50ec":
        env.update({"STAGE3_MOE_LIONMUON_PERIOD": "5", "STAGE3_MOE_LIONMUON_ENERGYCAL": "1",
                    "STAGE3_MOE_LIONMUON_ENERGYCAL_RHO_REF": "0.8", "STAGE3_MOE_LIONMUON_ENERGYCAL_EXPONENT": "1",
                    "STAGE3_MOE_LIONMUON_ENERGYCAL_QMIN": "0"})
    elif ARM == "lmlinrevr50":
        # The direct launcher path (run_stage3_moe_pretrain.sh full-lean) never sources
        # muon_presets.sh — only run_coat_arm.sh does. The PROB schedule MUST be set here
        # explicitly; otherwise lionmuon falls back to default PERIOD=2 and the arm silently
        # trains the wrong schedule (skeptic v031 blocker B1). Commas are fine in process env
        # (this is not an mlsub --env value).
        env.update({"STAGE3_MOE_LIONMUON_ENERGYCAL": "0",
                    "STAGE3_MOE_LIONMUON_PROB": "0:0.35,1:0.05"})
    else:
        raise RuntimeError(f"Unknown fullrun arm: {ARM}")
    return env, removed


def runtime_gates():
    import importlib.metadata
    import torch
    te = importlib.metadata.version("transformer-engine")
    volumes = []
    for candidate in ("/workspace-SR006.nfs3", "/tmp"):
        path = Path(candidate)
        usage = os.statvfs(path)
        volumes.append({"path": candidate, "free_bytes": usage.f_bavail * usage.f_frsize,
                        "free_files": usage.f_favail, "writable": os.access(path, os.W_OK)})
    nfs3 = next(v for v in volumes if v["path"] == "/workspace-SR006.nfs3")
    if nfs3["free_bytes"] < 1 * 2**30:
        raise RuntimeError("NFS3 < 1 GiB for receipts/logs")
    tmp = next(v for v in volumes if v["path"] == "/tmp")
    if tmp["free_bytes"] < 40 * 2**30:
        raise RuntimeError("/tmp < 40 GiB for full-state staging")
    corpus = Path("/home/jovyan/data/fineweb-edu-gpt2-megatron/data")
    expected = {"train.bin": 15113107020, "train.idx": 145805762, "development.bin": 16000532,
                "development.idx": 152022, "final.bin": 200002434, "final.idx": 1936642}
    for name, size in expected.items():
        path = corpus / name
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"Corpus metadata mismatch: {name}")
    manifest_sha = sha256(corpus.parent / "artifact-manifest.json")
    if manifest_sha != CORPUS_MANIFEST_SHA:
        raise RuntimeError("Corpus manifest hash mismatch")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip().splitlines()
    if len(gpu) != 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("fullrun016 requires exactly one visible GPU")
    return {"volumes": volumes, "corpus_manifest_sha256": manifest_sha, "gpu": gpu,
            "torch": torch.__version__, "cuda": torch.version.cuda, "transformer_engine": te}


def hf_setup():
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN not provided (submit.py --hf-token); HF is the primary durable archive")
    receipts = hfarch.ensure_hub(OUTPUT / "hf-logs")
    api = hfarch.make_api(token)
    preflight = hfarch.preflight(api, HF_REPO)
    write_json(OUTPUT / "hf-preflight.json", {**preflight, "dependency_receipts": receipts, "token_recorded": False})
    mark_stage("hf_preflight")
    return api


def parse_train_log(path):
    text = Path(path).read_text(errors="replace")
    iterations = [int(v) for v in re.findall(r"iteration\s+(\d+)/", text)]
    losses = [float(v) for v in re.findall(r"lm loss:\s*([0-9.eE+-]+)", text)]
    grad_norms = [float(v) for v in re.findall(r"grad norm:\s*([0-9.eE+-]+)", text)]
    validation = re.findall(r"validation loss at iteration\s+(\d+)(?: on validation set)? \| lm loss value: ([0-9.eE+-]+)", text)
    scheduled = {}
    for it, value in validation:
        scheduled.setdefault(int(it), float(value))
    return {"reached": max(iterations, default=0), "all_losses_finite": all(math.isfinite(v) for v in losses),
            "loss_count": len(losses), "grad_norm_nonzero_count": sum(v > 0 for v in grad_norms),
            "scheduled_validation": scheduled,
            "nan_nonzero_lines": len(re.findall(r"nan iterations:\s+[1-9]", text)),
            "skipped_nonzero_lines": len(re.findall(r"skipped iterations:\s+[1-9]", text)),
            "skipped_test_markers": len(re.findall(r"STAGE3_TEST_EVALUATION_SKIPPED", text))}


def check_realized_schedule(log_text, arm):
    """Per-step gate: the realized optimizer schedule must match the declared arm semantics.

    Catches wrong-schedule runs that would otherwise pass every other gate silently
    (skeptic v031 blocker B1). Uses the production schedule/draw functions; t is the global
    iteration (continues across resume segments), optimizer_steps/muon_steps are per-process
    counters (restart at 0 on resume), so coverage is checked as a contiguous t-range.
    """
    import random
    rows = re.findall(r"^LIONMUON_STEP t=(\d+) muon=(\d+) muon_steps=(\d+) optimizer_steps=(\d+) "
                      r"rule=(\w+) value=([0-9.eE+-]+) power=([0-9.eE+-]+)$", log_text, re.MULTILINE)
    if not rows:
        raise RuntimeError("No LIONMUON_STEP telemetry lines; realized schedule unverifiable")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "third_party" / "emerging-optimizers"))
    from stage3_moe.lionmuon import parse_schedule, piecewise_linear
    points = parse_schedule("0:0.35,1:0.05", float)
    seen = set()
    local_steps = []
    muon_total = 0
    for t_s, muon_s, msteps_s, osteps_s, rule, value_s, _power_s in rows:
        t = int(t_s)
        muon = int(muon_s)
        if t in seen:
            raise RuntimeError(f"Duplicate schedule telemetry at t={t}")
        seen.add(t)
        local_steps.append(int(osteps_s))
        value = float(value_s)
        if arm == "lm5r50ec":
            if rule != "period" or value != 5.0 or muon != int((t - 1) % 5 == 0):
                raise RuntimeError(f"EnergyCal arm realized schedule deviates from P5 at t={t}")
        elif arm == "lmlinrevr50":
            p = piecewise_linear(points, t / FULL_ITERS)
            expected = int(random.Random(f"lionmuon-{SEED}-{t}").random() < p)
            if rule != "prob" or not math.isclose(value, p, rel_tol=1e-5, abs_tol=1e-6) or muon != expected:
                raise RuntimeError(f"lmlinrev arm realized schedule deviates from PROB semantics at t={t}")
        else:
            raise RuntimeError(f"Unknown arm for schedule gate: {arm}")
        muon_total = int(msteps_s)
    if sorted(seen) != list(range(min(seen), max(seen) + 1)):
        raise RuntimeError("Schedule telemetry t-coverage incomplete")
    if local_steps != list(range(1, len(local_steps) + 1)):
        raise RuntimeError("Per-process optimizer_steps coverage incomplete")
    return {"telemetry_steps": len(seen), "t_range": [min(seen), max(seen)],
            "realized_muon_steps_this_segment": muon_total,
            "rule": "period5" if arm == "lm5r50ec" else "prob_0.35_to_0.05",
            "per_step_cross_check": "passed"}


def launch_segment(segment_tag, save_interval, exit_interval, timeout, verify_manifest=None, loaded_manifest=None):
    unique = f"hmoe-lionmoe-20261123-fullrun-016-v{VERSION}-{ARM}-{segment_tag}"
    env, removed = base_arm_env(unique)
    env["STAGE3_MOE_SAVE_INTERVAL"] = str(save_interval)
    env["STAGE3_MOE_SAVE_RETAIN_INTERVAL"] = "1"
    if exit_interval:
        env["STAGE3_MOE_EXIT_INTERVAL"] = str(exit_interval)
    else:
        env.pop("STAGE3_MOE_EXIT_INTERVAL", None)
    if verify_manifest:
        env["STAGE3_MOE_FULLRUN_VERIFY_MANIFEST"] = str(verify_manifest)
    if loaded_manifest:
        env["STAGE3_MOE_FULLRUN_LOADED_MANIFEST_PATH"] = str(loaded_manifest)
    stdout = ARM_NFS / f"segment-{segment_tag}.stdout.log"
    stderr = ARM_NFS / f"segment-{segment_tag}.stderr.log"
    started = time.perf_counter()
    code = run_process(["bash", str(ROOT / "scripts/run_stage3_moe_pretrain.sh"),
                        "muon_bf16_state_fp32", "full-lean"], env, stdout, stderr, timeout)
    duration = time.perf_counter() - started
    text = stdout.read_text(errors="replace")
    train_logs = re.findall(r"^TRAIN_LOG=(.*)$", text, re.MULTILINE)
    train_exits = re.findall(r"^TRAIN_EXIT=(\d+)$", text, re.MULTILINE)
    train_log = train_logs[-1] if train_logs else None
    parsed = parse_train_log(train_log) if train_log else {"reached": 0}
    # Strict-load gate (skeptic v030 major, extended to full-state restore): MCore
    # load_model_state_dict silently falls back to strict=False on a key mismatch and prints
    # "load_return:" (checkpointing.py:1893-1901); the "successfully loaded checkpoint ... at
    # iteration N" line prints in BOTH cases. Refuse any restore/eval that fell back.
    log_text = Path(train_log).read_text(errors="replace") if train_log else ""
    load_return_fallback = "load_return:" in log_text
    success = re.search(r"successfully loaded checkpoint.*at iteration (\d+)", log_text)
    loaded_iteration = int(success.group(1)) if success else None
    # Relay segment wrapper log tails for diagnosability (the underlying launcher/MCore error is
    # otherwise only on NFS). The realized-schedule gate applies only to SUCCESSFUL training; on a
    # wrapper failure the exit/reached gates surface the real error instead of masking it with
    # "no telemetry lines" (gate-save attempt1 failed opaque this way).
    training_ok = (code == 0 and train_exits == ["0"])
    seg_stdout_tail = redact(text[-16000:])
    seg_stderr_tail = redact(stderr.read_text(errors="replace")[-8000:]) if stderr.is_file() else ""
    (OUTPUT / f"segment-{segment_tag}.stdout.tail").write_text(seg_stdout_tail)
    (OUTPUT / f"segment-{segment_tag}.stderr.tail").write_text(seg_stderr_tail)
    schedule_check = check_realized_schedule(log_text, ARM) if (train_log and training_ok) else None
    return {"segment": segment_tag, "wrapper_exit": code, "train_exit": train_exits,
            "train_log": train_log, "duration_seconds": round(duration, 3),
            "removed_inherited_keys": removed, "parsed": parsed, "stdout": str(stdout),
            "load_return_fallback": load_return_fallback, "loaded_iteration": loaded_iteration,
            "training_ok": training_ok, "schedule_check": schedule_check,
            "segment_stdout_tail": seg_stdout_tail, "segment_stderr_tail": seg_stderr_tail}


def assert_strict_load(result, expected_iteration):
    if result.get("load_return_fallback"):
        raise RuntimeError(f"segment {result['segment']}: MCore strict=False fallback (load_return: marker); "
                           f"full-state restore refused")
    if expected_iteration is not None and result.get("loaded_iteration") != expected_iteration:
        raise RuntimeError(f"segment {result['segment']}: missing strict-load success marker at iteration "
                           f"{expected_iteration} (got {result.get('loaded_iteration')})")


def iter_dirs():
    if not FULL_DIR.is_dir():
        return []
    result = []
    for path in FULL_DIR.glob("iter_*"):
        match = re.fullmatch(r"iter_(\d+)", path.name)
        if match and path.is_dir():
            result.append((int(match.group(1)), path))
    return sorted(result)


def read_tracker():
    tracker = FULL_DIR / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        return int(tracker.read_text().strip())
    return 0


def upload_save(api, iteration, kind="full-state"):
    iter_dir = FULL_DIR / f"iter_{iteration:07d}"
    if not iter_dir.is_dir():
        raise RuntimeError(f"Missing save directory {iter_dir}")
    state_manifest_src = DUMP_DIR / f"iter_{iteration:07d}" / "state-manifest.json"
    if state_manifest_src.is_file():
        shutil.copy2(state_manifest_src, iter_dir / "state-manifest.json")
    manifest = hfarch.build_manifest(iter_dir, iteration, ARM, kind)
    write_json(iter_dir / "checkpoint-manifest.json", manifest)
    manifest = hfarch.build_manifest(iter_dir, iteration, ARM, kind)  # include checkpoint-manifest.json itself
    prefix = f"{HF_PREFIX}/iter_{iteration:07d}"
    receipt = hfarch.upload_directory(api, HF_REPO, iter_dir, prefix, manifest)
    write_json(ARM_NFS / f"hf-receipt-iter{iteration:07d}.json", receipt)
    return receipt, manifest


def fetch_save(api, iteration):
    prefix = f"{HF_PREFIX}/iter_{iteration:07d}"
    from huggingface_hub import hf_hub_download
    manifest_local = hf_hub_download(repo_id=HF_REPO, filename=f"{prefix}/checkpoint-manifest.json",
                                     repo_type="model", local_dir=str(TMP / "_manifests"), token=os.environ.get("HF_TOKEN"))
    manifest = json.loads(Path(manifest_local).read_text())
    iter_dir = FULL_DIR / f"iter_{iteration:07d}"
    receipt = hfarch.fetch_directory(api, HF_REPO, prefix, iter_dir, manifest)
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    (FULL_DIR / "latest_checkpointed_iteration.txt").write_text(str(iteration) + "\n")
    write_json(ARM_NFS / f"hf-fetch-receipt-iter{iteration:07d}.json", receipt)
    return iter_dir, manifest


def latest_hf_iteration(api):
    entries = hfarch.remote_entries(api, HF_REPO, HF_PREFIX)
    iterations = []
    for path in entries:
        match = re.match(rf"{re.escape(HF_PREFIX)}/iter_(\d+)$", path)
        if match:
            iterations.append(int(match.group(1)))
    return max(iterations, default=0)


def free_space_guard(needed_bytes):
    usage = os.statvfs(TMP)
    free = usage.f_bavail * usage.f_frsize
    if free < needed_bytes:
        raise FullrunStop(f"/tmp free {free} < required {needed_bytes}; safe stop at last good checkpoint")
    return free


def rotate_local(verified_latest, keep):
    # Delete ONLY task-owned rolling saves older than the verified HF successor, never pre-decay/latest.
    removed = []
    for iteration, path in iter_dirs():
        if iteration in keep or iteration >= verified_latest:
            continue
        receipt = ARM_NFS / f"hf-receipt-iter{iteration:07d}.json"
        if not receipt.is_file():
            continue  # not yet verified durable -> never delete
        shutil.rmtree(path)
        removed.append(iteration)
    return removed


def data_identity_snapshot(tag, consumed_samples):
    # fullrun-014 §3 deterministic index/sample identity branch: corpus manifest + seed + consumed
    # data position + bounded SHA256 of the train/dev index-cache files (no raw corpus/tensors export).
    cache = TMP / "data-cache"
    files = []
    total = 0
    if cache.is_dir():
        for path in sorted(cache.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            size = path.stat().st_size
            if size > 2**30 or total + size > 4 * 2**30:
                break
            total += size
            files.append({"relative_path": str(path.relative_to(cache)), "bytes": size, "sha256": sha256(path)})
    return {"tag": tag, "corpus_manifest_sha256": CORPUS_MANIFEST_SHA, "seed": SEED,
            "consumed_train_samples": consumed_samples, "data_cache_bytes_hashed": total,
            "index_cache_files": files,
            "scope": "deterministic index/sample identity; batch-content hashes not separately computed"}


def segment_exit_target(current):
    # Next exit boundary aligned to SAVE_INTERVAL; never beyond FULL_ITERS.
    nxt = ((current // SAVE_INTERVAL) + 1) * SAVE_INTERVAL
    return min(nxt, FULL_ITERS)


def run_to_endpoint(api, start_iteration):
    current = start_iteration
    uploads = []
    rotations = []
    dev_curve = {}
    while current < FULL_ITERS:
        if remaining() < SEC_PER_STEP * SAVE_INTERVAL + 600:
            raise FullrunStop(f"budget low at iteration {current}; safe stop (resume via continue)")
        full_state_bytes = 12 * 2**30
        free_space_guard(3 * full_state_bytes)
        target = segment_exit_target(current)
        # Bound the segment by remaining budget (whole save intervals only).
        affordable = int((remaining() - 600) / SEC_PER_STEP)
        affordable = (affordable // SAVE_INTERVAL) * SAVE_INTERVAL
        if affordable < SAVE_INTERVAL:
            raise FullrunStop(f"budget cannot fit one save interval at {current}; safe stop")
        target = min(target, current + affordable, FULL_ITERS)
        exit_interval = target if target < FULL_ITERS else 0
        seg_tag = f"seg{current:07d}-{target:07d}"
        timeout = max(600, int((target - current) * SEC_PER_STEP * 1.6) + 900)
        timeout = min(timeout, int(remaining()) - 120)
        loaded_manifest = ARM_NFS / f"loaded-state-iter{current:07d}.json"
        result = launch_segment(seg_tag, SAVE_INTERVAL, exit_interval, timeout, loaded_manifest=loaded_manifest)
        assert_strict_load(result, current)  # resume segment must strictly load iteration 'current'
        if result["wrapper_exit"] != 0 or result["train_exit"] != ["0"]:
            raise RuntimeError(f"segment {seg_tag} failed: wrapper={result['wrapper_exit']} train={result['train_exit']}")
        parsed = result["parsed"]
        if not parsed["all_losses_finite"] or parsed["nan_nonzero_lines"] or parsed["skipped_nonzero_lines"]:
            raise RuntimeError(f"segment {seg_tag} nonfinite/nan/skip")
        dev_curve.update(parsed["scheduled_validation"])
        reached = parsed["reached"]
        if reached <= current:
            raise RuntimeError(f"segment {seg_tag} made no progress ({current}->{reached})")
        new_saves = [it for it, _ in iter_dirs() if it > current]
        verified_latest = current
        for it in new_saves:
            receipt, _ = upload_save(api, it, "full-state")
            uploads.append({"iteration": it, "receipt": receipt["prefix"]})
            verified_latest = it
        keep = {PRE_DECAY, verified_latest}
        rotations.extend(rotate_local(verified_latest, keep))
        current = reached
        mark_stage(f"segment_{seg_tag}_uploaded")
        identity = data_identity_snapshot(seg_tag, current * 208)
        write_json(ARM_NFS / f"data-identity-{seg_tag}.json", identity)
        write_json(ARM_NFS / "continue-progress.json",
                   {"current_iteration": current, "uploads": uploads, "rotations": rotations,
                    "dev_curve": dev_curve, "verified_latest": verified_latest,
                    "data_identity": identity,
                    "elapsed_seconds": round(time.monotonic() - MONO_START, 3),
                    "payload_cap_seconds": CAP_SECONDS})
    return {"current": current, "uploads": uploads, "rotations": rotations, "dev_curve": dev_curve}


def export_weights_only(api):
    endpoint = FULL_DIR / f"iter_{FULL_ITERS:07d}"
    import torch
    state = torch.load(endpoint / "mp_rank_00" / "model_optim_rng.pt", map_location="cpu", weights_only=False)
    for key in ("optimizer", "opt_param_scheduler", "rng_state", "rerun_state_machine"):
        state.pop(key, None)
    weights_dir = TMP / "final-weights" / f"iter_{FULL_ITERS:07d}" / "mp_rank_00"
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, weights_dir / "model_optim_rng.pt")
    del state
    gc.collect()
    base = TMP / "final-weights" / f"iter_{FULL_ITERS:07d}"
    marker = {"weights_only": True, "resumable": False, "iteration": FULL_ITERS, "arm": ARM,
              "dropped_keys": ["optimizer", "opt_param_scheduler", "rng_state", "rerun_state_machine"],
              "purpose": "endpoint evaluation/downstream only; NOT optimizer/scheduler/RNG resume"}
    write_json(base / "WEIGHTS-ONLY-MARKER.json", marker)
    manifest = hfarch.build_manifest(base, FULL_ITERS, ARM, "weights-only")
    write_json(base / "checkpoint-manifest.json", manifest)
    manifest = hfarch.build_manifest(base, FULL_ITERS, ARM, "weights-only")
    prefix = f"{HF_PREFIX}/final-{FULL_ITERS}-weights"
    receipt = hfarch.upload_directory(api, HF_REPO, base, prefix, manifest)
    write_json(ARM_NFS / "hf-receipt-final-weights.json", receipt)
    return base, receipt


def eval_endpoint(weights_base, tag):
    # Identical eval-full-dev contract to baseline-eval-030 (v030 COMMANDS-DRAFT §4).
    eval_load = TMP / f"eval-load-{tag}"
    target = eval_load / f"iter_{FULL_ITERS:07d}" / "mp_rank_00" / "model_optim_rng.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    src = weights_base / f"iter_{FULL_ITERS:07d}" / "mp_rank_00" / "model_optim_rng.pt"
    if target.exists():
        target.unlink()
    os.link(src, target)
    (eval_load / "latest_checkpointed_iteration.txt").write_text(str(FULL_ITERS) + "\n")
    unique = f"hmoe-lionmoe-20261123-fullrun-016-v{VERSION}-{ARM}-finaleval-{tag}"
    env, removed = base_arm_env(unique)
    for key in ("STAGE3_MOE_FULLRUN_STATE_DUMP", "STAGE3_MOE_FULLRUN_DUMP_DIR", "STAGE3_MOE_ROUTING_DIAGNOSTICS",
                "STAGE3_MOE_MEASUREMENT_ENERGYCAL_DIAGNOSTICS", "STAGE3_MOE_LIONMUON_STEP_TELEMETRY"):
        env.pop(key, None)
    capture_path = ARM_NFS / f"final-full-dev-capture-{tag}.jsonl"
    env.update({"STAGE3_MOE_EVAL_LOAD": str(eval_load), "STAGE3_MOE_FULL_DEV_EVAL": "1",
                "STAGE3_MOE_FULL_DEV_CAPTURE_PATH": str(capture_path),
                "STAGE3_MOE_EVAL_INTERVAL": "300", "STAGE3_MOE_LOG_INTERVAL": "1",
                "STAGE3_MOE_CKPT_ROOT": str(TMP / f"eval-no-ckpt-{tag}")})
    stdout = ARM_NFS / f"final-eval-{tag}.stdout.log"
    code = run_process(["bash", str(ROOT / "scripts/run_stage3_moe_pretrain.sh"),
                        "muon_bf16_state_fp32", "eval-full-dev"], env, stdout,
                       ARM_NFS / f"final-eval-{tag}.stderr.log", max(600, int(remaining()) - 120))
    text = stdout.read_text(errors="replace")
    train_log = re.search(r"^TRAIN_LOG=(.*)$", text, re.MULTILINE).group(1)
    log_text = Path(train_log).read_text(errors="replace")
    if "load_return:" in log_text:
        raise RuntimeError("final eval: MCore strict=False fallback (load_return: marker); endpoint number refused")
    if "successfully loaded checkpoint" not in log_text or "at iteration 17242" not in log_text:
        raise RuntimeError("final eval: missing strict-load success marker at iteration 17242")
    aggregates = re.findall(r"validation loss at iteration 17242 on validation set \| lm loss value: ([0-9.eE+-]+)", log_text)
    if code != 0 or len(aggregates) != 1:
        raise RuntimeError(f"final eval failed: exit={code} aggregates={len(aggregates)}")
    aggregate = float(aggregates[0])
    samples = int(re.findall(r"Evaluating on (\d+) samples", log_text)[0])
    if len(re.findall(r"STAGE3_TEST_EVALUATION_SKIPPED", log_text)) != 1:
        raise RuntimeError("final eval skipped-test marker count != 1")
    if re.findall(r"iteration\s+\d+/\s*\d+ \|.*?lm loss:", log_text):
        raise RuntimeError("final eval observed training updates")
    records = [json.loads(line) for line in capture_path.read_text().splitlines() if line.strip()]
    lm = [r for r in records if r["key"] == "lm loss"]
    denominator = sum(r["num_tokens"] for r in lm)
    numerator = sum(r["loss_sum"] for r in lm)
    if len(lm) != samples or denominator <= 0 or not math.isclose(numerator / denominator, aggregate, rel_tol=1e-4):
        raise RuntimeError("final eval capture cross-check failed")
    return {"aggregate_full_dev_loss": aggregate, "eval_samples": samples,
            "valid_token_denominator": denominator, "eval_mode": "eval-full-dev (identical to baseline-eval-030)",
            "test_evaluated": False, "capture_path": str(capture_path), "train_log": train_log}


def gate_save(api):
    if read_tracker() != 0 or iter_dirs():
        raise RuntimeError("gate-save requires a fresh arm root (no existing checkpoint)")
    result = launch_segment("gate50", 50, 50, max(900, int(remaining()) - 120))
    assert_strict_load(result, None)  # from scratch: no checkpoint load expected
    if result["wrapper_exit"] != 0 or result["train_exit"] != ["0"] or result["parsed"]["reached"] != 50:
        raise RuntimeError(f"gate-save training invalid: {result['train_exit']} reached={result['parsed']['reached']}")
    if not result["parsed"]["all_losses_finite"]:
        raise RuntimeError("gate-save nonfinite losses")
    receipt, manifest = upload_save(api, 50, "full-state")
    state_manifest = json.loads((DUMP_DIR / "iter_0000050" / "state-manifest.json").read_text())
    summary = {"mode": "gate-save", "arm": ARM, "iteration": 50, "consumed_train_samples": 50 * 208,
               "train_exit": 0, "reached": 50, "hf_upload": receipt, "file_manifest": manifest,
               "state_manifest_counts": state_manifest["counts"], "next_step_phase": state_manifest["next_step_phase"],
               "scheduler_num_steps": (state_manifest.get("scheduler_state") or {}).get("num_steps"),
               "full_state_save": True, "resumable": True, "passed": True}
    write_json(OUTPUT / "gate-save-summary.json", summary)
    mark_stage("gate_save_uploaded")
    seal_envelope("small-artifacts.json", "hmoe-fullrun016-gate-save-v1",
                  [("gate-save-summary", OUTPUT / "gate-save-summary.json"),
                   ("hf-preflight", OUTPUT / "hf-preflight.json"),
                   ("hf-receipt", ARM_NFS / "hf-receipt-iter0000050.json")])
    print("FULLRUN016_RESULT " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


def gate_restore(api):
    iter_dir, manifest = fetch_save(api, 50)
    state_manifest_path = iter_dir / "state-manifest.json"
    if not state_manifest_path.is_file():
        raise RuntimeError("fetched gate-save checkpoint lacks state-manifest.json")
    loaded = ARM_NFS / "gate-restore-loaded-manifest.json"
    result = launch_segment("restore60", 50, 60, max(900, int(remaining()) - 120),
                            verify_manifest=state_manifest_path, loaded_manifest=loaded)
    assert_strict_load(result, 50)  # full-state restore must be strict, success marker at iteration 50
    verification = json.loads((state_manifest_path.parent / "gate-restore-verification.json").read_text())
    if not verification["verified"]:
        raise RuntimeError(f"gate-restore pre-update verification failed: {verification['mismatch_count']} mismatches")
    if result["wrapper_exit"] != 0 or result["train_exit"] != ["0"] or result["parsed"]["reached"] != 60:
        raise RuntimeError(f"gate-restore continuation invalid: {result['train_exit']} reached={result['parsed']['reached']}")
    parsed = result["parsed"]
    if not parsed["all_losses_finite"] or parsed["grad_norm_nonzero_count"] == 0:
        raise RuntimeError("gate-restore updates not finite/nonzero")
    loaded_manifest = json.loads(loaded.read_text())
    post = json.loads((DUMP_DIR / "iter_0000060" / "state-manifest.json").read_text())
    pre_hashes = {item["name"]: item["sha256"] for item in loaded_manifest["model_param_hashes"]}
    post_hashes = {item["name"]: item["sha256"] for item in post["model_param_hashes"]}
    changed = sum(pre_hashes[name] != post_hashes[name] for name in pre_hashes.keys() & post_hashes.keys())
    if changed == 0:
        raise RuntimeError("gate-restore produced no parameter change (zero updates?)")
    receipt, _ = upload_save(api, 60, "full-state")
    summary = {"mode": "gate-restore", "arm": ARM, "fetched_iteration": 50, "continued_to": 60,
               "pre_update_verification": verification, "hash_verify_before_load": manifest["files"],
               "full_load_optimizer_scheduler_rng": True, "strict_load_verified": True,
               "load_return_fallback": result["load_return_fallback"], "loaded_iteration": result["loaded_iteration"],
               "finite_nonzero_updates": True,
               "model_params_changed": changed, "hf_upload_iter60": receipt["prefix"],
               "data_position_consumed": loaded_manifest["consumed_train_samples"],
               "p5_phase_at_load": loaded_manifest["next_step_phase"], "passed": True}
    write_json(OUTPUT / "gate-restore-summary.json", summary)
    mark_stage("gate_restore_verified")
    seal_envelope("small-artifacts.json", "hmoe-fullrun016-gate-restore-v1",
                  [("gate-restore-summary", OUTPUT / "gate-restore-summary.json"),
                   ("gate-restore-verification", state_manifest_path.parent / "gate-restore-verification.json"),
                   ("hf-preflight", OUTPUT / "hf-preflight.json")])
    print("FULLRUN016_RESULT " + json.dumps({k: v for k, v in summary.items() if k != "hash_verify_before_load"}, sort_keys=True), flush=True)
    return 0


def continue_arm(api, from_scratch):
    if from_scratch:
        if read_tracker() != 0:
            raise RuntimeError("arm-lmlinrev expects fresh root")
        anchor = launch_segment("anchor50", 50, 50, max(900, int(remaining()) - 120))
        assert_strict_load(anchor, None)
        if anchor["wrapper_exit"] != 0 or anchor["train_exit"] != ["0"] or anchor["parsed"]["reached"] != 50:
            raise RuntimeError("arm anchor segment failed")
        upload_save(api, 50, "full-state")
        start = 50
        mark_stage("anchor50_uploaded")
    else:
        start = latest_hf_iteration(api)
        if start == 0:
            raise RuntimeError("continue found no HF full-state for arm; run gate-save first")
        fetch_save(api, start)
        mark_stage(f"resumed_from_hf_iter{start}")
    progress = run_to_endpoint(api, start)
    weights_base, weights_receipt = export_weights_only(api)
    final_eval = eval_endpoint(weights_base, "endpoint")
    summary = {"mode": MODE, "arm": ARM, "reached": progress["current"], "endpoint_iteration": FULL_ITERS,
               "uploads": progress["uploads"], "local_rotations_after_hf_verify": progress["rotations"],
               "dev_curve_periodic_1000": progress["dev_curve"], "final_full_dev": final_eval,
               "weights_only_upload": weights_receipt["prefix"], "pre_decay_retained": PRE_DECAY,
               "previous_good_kept": False, "hf_primary_durable": True,
               "duration_seconds": round((datetime.datetime.now(datetime.timezone.utc) - STARTED).total_seconds(), 3),
               "started_at": STARTED.isoformat(), "passed": progress["current"] == FULL_ITERS}
    write_json(OUTPUT / "continue-summary.json", summary)
    mark_stage("endpoint_evaluated")
    seal_envelope("small-artifacts.json", "hmoe-fullrun016-continue-v1",
                  [("continue-summary", OUTPUT / "continue-summary.json"),
                   ("hf-preflight", OUTPUT / "hf-preflight.json"),
                   ("hf-receipt-final-weights", ARM_NFS / "hf-receipt-final-weights.json")])
    print("FULLRUN016_RESULT " + json.dumps({k: v for k, v in summary.items() if k != "uploads"}, sort_keys=True), flush=True)
    return 0 if summary["passed"] else 1


def main_checked():
    ARM_NFS.mkdir(parents=True, exist_ok=True)
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    runtime = runtime_gates()
    write_json(OUTPUT / "runtime-gates.json", runtime)
    mark_stage("runtime")
    api = hf_setup()
    if MODE == "gate-save":
        return gate_save(api)
    if MODE == "gate-restore":
        return gate_restore(api)
    if MODE == "continue":
        return continue_arm(api, from_scratch=False)
    return continue_arm(api, from_scratch=True)


def main():
    try:
        return main_checked()
    except Exception as error:
        stopped = isinstance(error, FullrunStop)
        available = [artifact(path) for path in
                     (OUTPUT / "runtime-gates.json", OUTPUT / "hf-preflight.json",
                      OUTPUT / "gate-save-summary.json", OUTPUT / "gate-restore-summary.json",
                      OUTPUT / "continue-summary.json", ARM_NFS / "continue-progress.json", OUTPUT / "progress.json")
                     if path.is_file()]
        segment_tails = {}
        for tail in sorted(OUTPUT.glob("segment-*.tail")):
            segment_tails[tail.name] = tail.read_text(errors="replace")[-16000:]
            available.append(artifact(tail))
        partial = {"schema": "hmoe-fullrun016-error-v1", "passed": False, "safe_stop": stopped,
                   "mode": MODE, "arm": ARM, "job_note": os.environ.get("HMOE_JOB_NOTE"),
                   "completed_stages": STAGES, "error_type": type(error).__name__, "error": redact(str(error)),
                   "traceback": redact(traceback.format_exc())[-4000:], "available_artifacts": available,
                   "segment_log_tails": segment_tails,
                   "last_good_checkpoint": read_tracker(),
                   "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write_json(OUTPUT / "partial-error.json", partial)
        labeled = [("partial-error", OUTPUT / "partial-error.json")]
        for label, path in (("runtime-gates", OUTPUT / "runtime-gates.json"), ("hf-preflight", OUTPUT / "hf-preflight.json"),
                            ("continue-progress", ARM_NFS / "continue-progress.json")):
            if path.is_file():
                labeled.append((label, path))
        for tail in sorted(OUTPUT.glob("segment-*.tail")):
            labeled.append((tail.stem, tail))
        seal_envelope("small-artifacts.json", "hmoe-fullrun016-error-v1", labeled)
        print("FULLRUN016_RESULT " + json.dumps({"passed": False, "safe_stop": stopped, "mode": MODE,
                                                 "completed_stages": STAGES, "last_good_checkpoint": read_tracker()}, sort_keys=True), flush=True)
        return 0 if stopped else 1


if __name__ == "__main__":
    raise SystemExit(main())
