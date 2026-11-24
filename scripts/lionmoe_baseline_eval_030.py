"""Baseline-eval v030: dev-only full-development re-evaluation of the October plain R50 endpoint.

Mandatory condition 1 of the user-confirmed 3->2 full-arm revision (research/fullrun-014/PLAN.md
2026-11-23): download stage4/muon-ctrl-lm5r50-iter0017242-model.pt from the HF archive, verify
size/SHA256 BEFORE use, and re-evaluate the endpoint with the CURRENT identical full-development
evaluator (the same eval-full-dev mode/argv the new full arms' final evaluation will use).
October 2.621174 is context only until this re-evaluation.

Format finding (static): the October final-model export (run_coat_arm.sh full branch) torch.loads
the full MCore checkpoint and pops ONLY 'optimizer'; model.pt is therefore a standard MCore torch
checkpoint (checkpoint_version 3.0, keys args/iteration/model/opt_param_scheduler/rng_state/
rerun_state_machine/num_floating_point_operations_so_far) minus optimizer state. The minimal
adapter is layout-only: latest_checkpointed_iteration.txt + iter_0017242/mp_rank_00/model_optim_rng.pt,
loaded with --no-load-optim --no-load-rng --override-opt_param-scheduler. No optimizer/scheduler/RNG
state is fabricated; the load is explicitly marked EVAL-ONLY / NOT RESUMABLE.

Eval-only semantics: the launcher mode eval-full-dev does NOT pass --skip-train (it returns
optimizer=None and the result writer needs a real object, the documented eval-downstream pattern);
train_iters equals the endpoint iteration 17242, so ZERO training updates run -- asserted from the
log and the probe record. --full-validation evaluates the entire development partition once
(single-epoch sampler, eval batch sizes 1/1); the test call is blocked by the no-test guard with
exactly one skipped-test marker. A passive capture (stage3_moe/full_dev_capture.py, installed only
for this mode) records the evaluator's own [loss_sum, num_tokens] reports: exact valid-token
denominator + per-iteration losses, cross-checked against the printed aggregate. No production
math/dtype change. HF unreachable => sealed blocker report, NO Mac fallback (weights never on Mac).
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
OUTPUT = Path(os.environ["HMOE_CONTRACT_ROOT"])
OUTPUT.mkdir(parents=True, exist_ok=True)
STARTED = datetime.datetime.now(datetime.timezone.utc)
MONO_START = time.monotonic()
STAGES = []
VERSION = os.environ.get("HMOE_BASELINE_EVAL_VERSION", "030")
EXPECTED_NOTE = f"hmoe-lionmoe-20261123-baseline-eval-v{VERSION}"
EXPECTED_ROOT = f"/workspace-SR006.nfs3/hmoe-lionmoe-20261123/baseline-eval-v{VERSION}"
EXPECTED_BUNDLE = f"hmoe-lionmoe-20261123-v{VERSION}"
SEED = 1234
ENDPOINT_ITERATION = 17242
HF_REPO = "AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints"
# Reviewer MoE CI: allow re-evaluating any archived endpoint (default = October plain R50, unchanged).
HF_FILE = os.environ.get("HMOE_BASELINE_EVAL_HF_FILE", "stage4/muon-ctrl-lm5r50-iter0017242-model.pt")
EXPECTED_BYTES = int(os.environ.get("HMOE_BASELINE_EVAL_BYTES", "2058827231"))
EXPECTED_SHA256 = os.environ.get("HMOE_BASELINE_EVAL_SHA256", "a90d286c10aac5ce91769f789b6c48c6a9f1d284356ddb4fd340844714fa6ab0")
EVAL_LABEL = os.environ.get("HMOE_BASELINE_EVAL_LABEL", "October from-scratch plain R50 control (muon-ctrl-lm5r50, 17242, with legitimate own-checkpoint restarts); endpoint re-evaluated under the current full-development protocol")
CORPUS_MANIFEST_SHA = "7a7cf91633081dd080f3cff96407bd61d4c237d88a5e926cffe36fc5b8cbfcc8"
CAP_SECONDS = int(os.environ.get("HMOE_BASELINE_EVAL_CAP_SECONDS", "3600"))
SEAL_RESERVE_SECONDS = 180
TMP = Path(f"/tmp/hmoe-lionmoe-20261123-baseline-eval-v{VERSION}")
if os.environ.get("HMOE_JOB_NOTE") != EXPECTED_NOTE:
    raise RuntimeError("HMOE_JOB_NOTE does not match baseline-eval/version")
if os.environ.get("HMOE_CONTRACT_ROOT") != EXPECTED_ROOT:
    raise RuntimeError("HMOE_CONTRACT_ROOT does not match baseline-eval/version")
if os.environ.get("EXPECTED_BUNDLE_ID") != EXPECTED_BUNDLE:
    raise RuntimeError("Unexpected bundle for baseline-eval")
for key, length in (("EXPECTED_SOURCE_COMMIT", 40), ("EXPECTED_CARRIER_COMMIT", 40), ("EXPECTED_PAYLOAD_SHA256", 64)):
    value = os.environ.get(key, "")
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"Missing/invalid runtime pin: {key}")


class BaselineBlocker(RuntimeError):
    """A sealed, honest stop (e.g. HF unreachable from Cloud); no Mac fallback exists."""


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


def redact(text):
    token = os.environ.get("HF_TOKEN", "")
    if token:
        text = text.replace(token, "[REDACTED]")
    return text


def run_process(command, env, stdout_path, stderr_path, timeout):
    # M2: own session + process-group kill on timeout, so torchrun children cannot survive.
    with Path(stdout_path).open("w") as stdout, Path(stderr_path).open("w") as stderr:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
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


def seal_envelope(filename, schema, labeled_paths, marker_prefix):
    # M3: degrade to receipts-only with an explicit note instead of failing the seal itself.
    raw = []
    for label, path in labeled_paths:
        data = Path(path).read_bytes()
        raw.append({"label": label, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "content_base64": base64.b64encode(data).decode()})
    sealed_path = OUTPUT / filename
    write_json(sealed_path, {"schema": schema, "nonsecret": True, "artifacts": raw})
    sealed = sealed_path.read_bytes()
    if len(sealed) > 256 * 1024:
        receipts = [{"label": label, **artifact(path)} for label, path in labeled_paths]
        write_json(sealed_path, {"schema": schema, "nonsecret": True, "artifacts": [],
                                 "degraded": "receipts-only: full contents exceed 256 KiB; artifacts remain on NFS",
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


def remaining_seconds():
    return CAP_SECONDS - (time.monotonic() - MONO_START) - SEAL_RESERVE_SECONDS


def runtime_gates():
    import importlib.metadata
    import torch

    transformer_engine_version = importlib.metadata.version("transformer-engine")
    volumes = []
    for candidate in ("/home/jovyan", "/workspace-SR006.nfs2", "/workspace-SR006.nfs3", "/tmp"):
        path = Path(candidate)
        item = {"path": candidate, "exists": path.exists()}
        if path.exists():
            usage = os.statvfs(path)
            item.update({"free_bytes": usage.f_bavail * usage.f_frsize, "free_files": usage.f_favail,
                         "writable_by_process": os.access(path, os.W_OK)})
        volumes.append(item)
    nfs3 = next(item for item in volumes if item["path"] == "/workspace-SR006.nfs3")
    if nfs3["free_bytes"] < 1 * 2**30:
        raise RuntimeError("Less than 1 GiB available for eval artifacts on NFS3")
    tmp = next(item for item in volumes if item["path"] == "/tmp")
    if tmp["free_bytes"] < 12 * 2**30:
        raise RuntimeError("Less than 12 GiB job-local /tmp for the 2 GB download + layout + index cache")
    tiny = OUTPUT / "runtime-roundtrip.bin"
    payload = b"hmoe-lionmoe-baseline-eval-030\n"
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
    if manifest_sha != CORPUS_MANIFEST_SHA:
        raise RuntimeError("Corpus manifest hash mismatch")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip().splitlines()
    if len(gpu) != 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Baseline eval requires exactly one visible GPU")
    return {"volumes": volumes, "tiny_roundtrip": artifact(tiny), "corpus_root": str(corpus),
            "corpus_files": files, "corpus_manifest_sha256": manifest_sha,
            "gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformer_engine": transformer_engine_version, "cuda_available": True, "device_count": 1}


def ensure_huggingface_hub():
    receipts = {}
    try:
        import huggingface_hub  # noqa: F401
        receipts["huggingface_hub"] = "present"
    except ImportError:
        code = run_process([sys.executable, "-m", "pip", "install", "--user", "-q", "--disable-pip-version-check",
                            "huggingface_hub"], dict(os.environ), OUTPUT / "pip-hub.stdout.log",
                           OUTPUT / "pip-hub.stderr.log", 240)
        receipts["huggingface_hub"] = f"pip_install_exit={code}"
        if code != 0:
            raise BaselineBlocker("huggingface_hub unavailable and pip install failed; HF route blocked from Cloud")
        import huggingface_hub  # noqa: F401
    try:
        import hf_transfer  # noqa: F401
        receipts["hf_transfer"] = "present"
    except ImportError:
        code = run_process([sys.executable, "-m", "pip", "install", "--user", "-q", "--disable-pip-version-check",
                            "hf_transfer"], dict(os.environ), OUTPUT / "pip-hftransfer.stdout.log",
                           OUTPUT / "pip-hftransfer.stderr.log", 120)
        receipts["hf_transfer"] = f"pip_install_exit={code} (optional; python downloader fallback)"
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    except ImportError:
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    import huggingface_hub
    receipts["huggingface_hub_version"] = huggingface_hub.__version__
    return receipts


def hf_preflight():
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise BaselineBlocker("HF_TOKEN not provided to the job (submit.py --hf-token); no Mac fallback for weights")
    receipts = ensure_huggingface_hub()
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        info = api.model_info(HF_REPO)
        entries = {entry.path: entry for entry in api.list_repo_tree(HF_REPO, path_in_repo="stage4", expand=True)}
    except Exception as error:
        raise BaselineBlocker(f"HF unreachable/unauthorized from Cloud: {type(error).__name__}: {redact(str(error))}")
    entry = entries.get(HF_FILE)
    if entry is None:
        raise BaselineBlocker(f"HF repo reachable but {HF_FILE} absent from stage4/ listing")
    lfs = getattr(entry, "lfs", None)
    server_sha = getattr(lfs, "sha256", None) if lfs is not None else None
    server_size = getattr(entry, "size", None)
    if server_sha != EXPECTED_SHA256 or server_size != EXPECTED_BYTES:
        raise BaselineBlocker(f"HF server-side identity mismatch: sha={server_sha} size={server_size}")
    result = {"reachable": True, "repo": HF_REPO, "repo_sha": getattr(info, "sha", None),
              "last_modified": str(getattr(info, "last_modified", None)), "file": HF_FILE,
              "server_lfs_sha256": server_sha, "server_size": server_size,
              "identity_matches_pin_before_download": True, "dependency_receipts": receipts,
              "token_recorded": False}
    write_json(OUTPUT / "hf-preflight.json", result)
    mark_stage("hf_preflight")
    return api, result


def download_and_verify(api):
    from huggingface_hub import hf_hub_download

    TMP.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    local = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="model",
                            local_dir=str(TMP / "hf-download"), token=os.environ.get("HF_TOKEN"))
    local = Path(local)
    size = local.stat().st_size
    digest = sha256(local)
    if size != EXPECTED_BYTES or digest != EXPECTED_SHA256:
        raise RuntimeError(f"Downloaded checkpoint identity mismatch: bytes={size} sha256={digest}")
    receipt = {"source_repo": HF_REPO, "source_file": HF_FILE, "local_path": str(local),
               "bytes": size, "sha256": digest, "verified_before_use": True,
               "download_seconds": round(time.monotonic() - started, 3)}
    write_json(OUTPUT / "download-receipt.json", receipt)
    mark_stage("checkpoint_downloaded_verified")
    return local, receipt


def inspect_and_stage(local):
    import torch

    state = torch.load(local, map_location="cpu", weights_only=False)
    keys = sorted(state.keys())
    if "model" not in state or "optimizer" in state:
        raise RuntimeError(f"Unexpected October export structure: keys={keys}")
    if state.get("iteration") != ENDPOINT_ITERATION:
        raise RuntimeError(f"Checkpoint iteration {state.get('iteration')} != {ENDPOINT_ITERATION}")
    model_sd = state["model"]
    tensor_numel = sum(value.numel() for value in model_sd.values() if torch.is_tensor(value))
    nontensor_keys = [key for key, value in model_sd.items() if not torch.is_tensor(value)]
    ckpt_args = state.get("args")
    inspection = {"top_level_keys": keys, "checkpoint_version": state.get("checkpoint_version"),
                  "iteration": state.get("iteration"),
                  "model_tensor_numel": tensor_numel, "model_nontensor_keys": nontensor_keys,
                  "model_state_entries": len(model_sd),
                  "optimizer_state_present": False,
                  "opt_param_scheduler_present": "opt_param_scheduler" in state,
                  "rng_state_present": "rng_state" in state,
                  "rerun_state_machine_present": "rerun_state_machine" in state,
                  "consumed_train_samples_from_args": getattr(ckpt_args, "consumed_train_samples", None),
                  "format": "MCore torch checkpoint minus optimizer (October final-model export); eval-only, NOT resumable"}
    if tensor_numel != 1_028_926_976:
        raise RuntimeError(f"Model tensor numel {tensor_numel} != 1028926976")
    del state, model_sd
    gc.collect()
    eval_load = TMP / "eval-load"
    target = eval_load / f"iter_{ENDPOINT_ITERATION:07d}" / "mp_rank_00" / "model_optim_rng.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(local, target)
        layout = "hardlink"
    except OSError:
        shutil.copy2(local, target)
        layout = "copy"
    (eval_load / "latest_checkpointed_iteration.txt").write_text(str(ENDPOINT_ITERATION) + "\n")
    marker = {"eval_only": True, "resumable": False,
              "reason": "optimizer state absent (popped by the October export); scheduler/RNG present but NOT loaded (--no-load-optim --no-load-rng); no state fabricated",
              "source_repo": HF_REPO, "source_file": HF_FILE, "source_sha256": EXPECTED_SHA256,
              "iteration": ENDPOINT_ITERATION, "layout": layout, "staged_path": str(target)}
    write_json(eval_load / "EVAL-ONLY-MARKER.json", marker)
    inspection["eval_load_dir"] = str(eval_load)
    inspection["marker"] = marker
    write_json(OUTPUT / "checkpoint-inspection.json", inspection)
    mark_stage("checkpoint_staged_eval_only")
    return eval_load, inspection


def october_data_identity():
    # Bounded search of surviving October NFS logs; honest negative if nothing survives.
    roots = [Path("/workspace-SR006.nfs3/hmoe-cloud/coat-20261002/runs-s4full"),
             Path("/workspace-SR006.nfs2/hmoe-cloud/coat-20261002/runs-s4full"),
             Path("/home/jovyan/hmoe-cloud/pretrain")]
    found = []
    scanned_bytes = 0
    for root in roots:
        if not root.is_dir():
            continue
        for arm_dir in sorted(root.glob("*lm5r50*"))[:8]:
            if not arm_dir.is_dir():
                continue
            results = sorted(set(arm_dir.glob("*/results.jsonl")) | set(arm_dir.glob("logs/*/results.jsonl")))[:4]
            for result in results:
                try:
                    record = json.loads(result.read_text(errors="replace").splitlines()[-1])
                    provenance = record.get("provenance", {})
                    found.append({"kind": "results_provenance", "path": str(result),
                                  "data_manifest_sha256": provenance.get("data_manifest_sha256"),
                                  "git_commit": provenance.get("git_commit"),
                                  "selected_argv": {opt: provenance.get("argv", [])[provenance.get("argv", []).index(opt) + 1]
                                                    for opt in ("--train-iters", "--seed", "--eval-iters", "--eval-interval")
                                                    if opt in provenance.get("argv", [])}})
                except Exception as error:
                    found.append({"kind": "results_provenance", "path": str(result), "error": str(error)})
            logs = sorted(set(arm_dir.glob("train-*.log")) | set(arm_dir.glob("logs/*/train-*.log")))[:4]
            for log in logs:
                try:
                    with log.open("rb") as stream:
                        head = stream.read(2_000_000)
                    scanned_bytes += len(head)
                    text = head.decode(errors="replace")
                    data_lines = [line for line in text.splitlines() if line.startswith("DATA train=")][:2]
                    manifest_hit = CORPUS_MANIFEST_SHA in text
                    found.append({"kind": "train_log_head", "path": str(log), "bytes_scanned": len(head),
                                  "data_lines": data_lines, "current_manifest_sha_present_in_head": manifest_hit})
                except Exception as error:
                    found.append({"kind": "train_log_head", "path": str(log), "error": str(error)})
            if scanned_bytes > 64 * 2**30:
                break
    manifests = {item.get("data_manifest_sha256") for item in found if item.get("data_manifest_sha256")}
    if manifests:
        status = "matches_current" if manifests == {CORPUS_MANIFEST_SHA} else "differs_from_current"
    elif found:
        status = "surviving_logs_found_but_no_manifest_hash"
    else:
        status = "not re-verifiable from surviving logs"
    result = {"schema": "hmoe-baseline-eval-030-october-data-identity-v1", "status": status,
              "current_corpus_manifest_sha256": CORPUS_MANIFEST_SHA,
              "october_manifest_hashes_found": sorted(manifests), "evidence": found,
              "searched_roots": [str(root) for root in roots], "scanned_bytes": scanned_bytes,
              "scope": "bounded head/provenance reads only; current corpus gate enforced separately in runtime gates"}
    write_json(OUTPUT / "october-data-identity.json", result)
    mark_stage("october_data_identity_searched")
    return result


CLEANUP_KEYS = ("STAGE3_MOE_LR_SCALE", "STAGE3_MOE_LR_CONSTANT", "STAGE3_MOE_COLD_SWITCH",
                "STAGE3_MOE_TRAIN_DATA_PREFIX", "STAGE3_MOE_VALID_DATA_PREFIX", "STAGE3_MOE_TEST_DATA_PREFIX",
                "STAGE3_MOE_LR", "STAGE3_MOE_MIN_LR", "STAGE3_MOE_ADAM_BETA2",
                "STAGE3_MOE_WGRAD_FUSION", "STAGE3_MOE_MOE_FUSIONS", "STAGE3_MOE_OVERLAP_GRAD_REDUCE",
                "STAGE3_MOE_FP32_MODEL", "STAGE3_MOE_EXTRA_ARGS", "STAGE3_MOE_FP8_COMPUTE_ARGS",
                "STAGE3_MOE_FP8_STATE_DTYPES", "STAGE3_MOE_DEQUANT_ON_LOAD", "STAGE3_MOE_PROBE_BASE_ITER",
                "STAGE3_MOE_FULL_TRAIN_ITERS", "COAT_STOP_ITERS", "STAGE3_MOE_LIONMUON_PERIOD_EXPERTS",
                "STAGE3_MOE_LIONMUON_PROB", "STAGE3_MOE_LIONMUON_POWER", "STAGE3_MOE_LIONMUON_ENERGYCAL_DIAGNOSTICS",
                "STAGE3_MOE_MUON_NS_STEPS", "STAGE3_MOE_BF16_FUSED_EXPERTS", "STAGE3_MOE_COAT_STRICT",
                "STAGE3_MOE_COAT_FUSED_EXPERTS", "STAGE3_MOE_HESSIAN_PATH", "STAGE3_MOE_DIAGNOSTIC_PATH",
                "STAGE3_MOE_REPLAY_PATH", "STAGE3_MOE_M2_TRACE_PATH", "STAGE3_MOE_ACTIVATION_STATS_PATH",
                "STAGE3_MOE_COLDSTART_RECEIPT", "STAGE3_MOE_MEASUREMENT_ENERGYCAL_DIAGNOSTICS",
                "STAGE3_MOE_ROUTING_DIAGNOSTICS")


def eval_environment(eval_load, corpus_root, unique):
    env = dict(os.environ)
    removed = [key for key in CLEANUP_KEYS if key in env]
    for key in CLEANUP_KEYS:
        env.pop(key, None)
    capture_path = OUTPUT / "eval" / "full-dev-capture.jsonl"
    env.update({
        "STAGE3_MOE_EVAL_LOAD": str(eval_load),
        "STAGE3_MOE_LOG_ROOT": str(OUTPUT / "eval" / "logs"),
        "STAGE3_MOE_CKPT_ROOT": str(OUTPUT / "eval" / "no-checkpoints"),
        "STAGE3_MOE_DATA_CACHE_PATH": str(TMP / "data-cache"),
        "STAGE3_MOE_RUN_SUFFIX": unique,
        "STAGE3_MOE_DATA_ROOT": corpus_root,
        "STAGE3_MOE_SEED": str(SEED), "STAGE3_MOE_NPROC": "1",
        "STAGE3_MOE_MICRO_BATCH": "16", "STAGE3_MOE_EP": "1",
        "STAGE3_MOE_EVAL_ITERS": "32", "STAGE3_MOE_EVAL_INTERVAL": "300", "STAGE3_MOE_LOG_INTERVAL": "1",
        "STAGE3_MOE_DISABLE_TEST_EVAL": "1", "STAGE3_MOE_ROUTING_TELEMETRY_INTERVAL": "10",
        "STAGE3_MOE_SCHEDULE_ITERS": str(ENDPOINT_ITERATION),
        "STAGE3_MOE_MUON_PRESET": "lm5r50", "STAGE3_MOE_MUON_VARIANT": "lionmuon",
        "STAGE3_MOE_LIONMUON_PERIOD": "5", "STAGE3_MOE_LIONMUON_LR_RATIO": "50",
        "STAGE3_MOE_LIONMUON_SEED": str(SEED), "STAGE3_MOE_LIONMUON_BETA1": "0.9", "STAGE3_MOE_LIONMUON_BETA2": "0.99",
        "STAGE3_MOE_LIONMUON_ENERGYCAL": "0", "STAGE3_MOE_LIONMUON_FIXED_ROUTED_ALPHA": "1",
        "STAGE3_MOE_FULL_DEV_EVAL": "1", "STAGE3_MOE_FULL_DEV_CAPTURE_PATH": str(capture_path),
        "STAGE3_MOE_PROPAGATE_EXIT": "1",
        "WANDB_MODE": "offline", "WANDB_RUN_ID": unique,
    })
    return env, removed, capture_path


def analyze_evaluation(wrapper_stdout, unique, capture_path, aggregate_from_record):
    text = Path(wrapper_stdout).read_text(errors="replace")
    train_exits = re.findall(r"^TRAIN_EXIT=(\d+)$", text, re.MULTILINE)
    train_logs = re.findall(r"^TRAIN_LOG=(.*)$", text, re.MULTILINE)
    if train_exits != ["0"] or len(train_logs) != 1:
        raise RuntimeError(f"Eval launcher exits invalid: TRAIN_EXIT={train_exits} logs={len(train_logs)}")
    train_log = train_logs[0]
    log_text = Path(train_log).read_text(errors="replace")
    # Strict-load gate (skeptic v030 review, major): MCore load_model_state_dict silently
    # falls back to strict=False on any key mismatch and prints "load_return:"; the
    # "successfully loaded checkpoint" line is printed in BOTH cases. Refuse the baseline
    # number unless the load was strict (no fallback marker, success marker present).
    if "load_return:" in log_text:
        raise RuntimeError("MCore strict=False fallback detected (load_return: marker); baseline number refused")
    if "successfully loaded checkpoint" not in log_text or "at iteration 17242" not in log_text:
        raise RuntimeError("Missing strict checkpoint-load success marker at iteration 17242")
    aggregates = re.findall(r"validation loss at iteration 17242 on validation set \| lm loss value: ([0-9.eE+-]+)", log_text)
    if len(aggregates) != 1:
        raise RuntimeError(f"Expected exactly one full-development aggregate, got {len(aggregates)}")
    aggregate = float(aggregates[0])
    if not math.isfinite(aggregate):
        raise RuntimeError("Nonfinite full-development loss")
    samples_matches = re.findall(r"Evaluating on (\d+) samples", log_text)
    if len(samples_matches) != 1:
        raise RuntimeError(f"Expected one 'Evaluating on N samples' line, got {len(samples_matches)}")
    eval_samples = int(samples_matches[0])
    iter_progress = re.findall(r"Evaluating iter (\d+)/(\d+)", log_text)
    if iter_progress and (len(iter_progress) != eval_samples or int(iter_progress[-1][0]) != eval_samples):
        raise RuntimeError("Per-iteration evaluation progress incomplete")
    skipped = len(re.findall(r"STAGE3_TEST_EVALUATION_SKIPPED", log_text))
    if skipped != 1:
        raise RuntimeError(f"Expected exactly one skipped-test marker, got {skipped}")
    test_output = [line for line in log_text.splitlines()
                   if "on test set" in line and "STAGE3_TEST_EVALUATION_SKIPPED" not in line]
    if test_output:
        raise RuntimeError(f"Unexpected real test output: {test_output[-3:]}")
    training_lines = re.findall(r"iteration\s+\d+/\s*\d+ \|.*?lm loss:", log_text)
    if training_lines:
        raise RuntimeError(f"Training progress lines present during eval-only job: {len(training_lines)}")
    result_path = OUTPUT / "eval" / "logs" / f"stage3-muon_bf16_state_fp32-eval-full-dev-{unique}" / "results.jsonl"
    record = json.loads(result_path.read_text().splitlines()[-1])
    argv = record["provenance"]["argv"]
    assert "--full-validation" in argv
    assert "--no-load-optim" in argv and "--no-load-rng" in argv and "--override-opt_param-scheduler" in argv
    assert "--skip-train" not in argv and "--save" not in argv
    assert not any(argument.startswith("--fp8") for argument in argv)
    assert argv[argv.index("--train-iters") + 1] == "17242"
    assert argv[argv.index("--lr-decay-iters") + 1] == "17242"
    assert argv[argv.index("--lr-warmup-iters") + 1] == "173"
    assert argv[argv.index("--lr-wsd-decay-iters") + 1] == "3448"
    assert argv[argv.index("--eval-micro-batch-size") + 1] == "1"
    assert argv[argv.index("--eval-global-batch-size") + 1] == "1"
    assert argv[argv.index("--seed") + 1] == str(SEED)
    assert argv[argv.index("--muon-num-ns-steps") + 1] == "5"
    assert "--bf16" in argv
    denominators = record["denominators"]
    assert denominators["micro_batch_sequences_per_gpu"] == 16
    assert denominators["global_batch_sequences"] == 208
    assert denominators["gpu_count"] == 1
    assert denominators["total_parameters"] == 1028926976
    assert record["status"] == "completed"
    assert record["measurement"]["protocol"]["kind"] == "evaluation"
    assert record["measurement"]["protocol"]["e2e_train_steps"] == 0
    assert record["evaluation"]["skipped_test_calls"] in (0, 1)
    record_loss = record["measurement"]["loss"]["validation"]
    if record_loss is None or not math.isclose(record_loss, aggregate, rel_tol=1e-6):
        raise RuntimeError(f"Record validation loss {record_loss} != printed aggregate {aggregate}")
    if not math.isclose(aggregate, aggregate_from_record, rel_tol=1e-6):
        raise RuntimeError("Aggregate mismatch between log and capture cross-check input")
    return {"train_log": train_log, "result_path": str(result_path), "record": record,
            "aggregate_full_dev_loss": aggregate, "eval_samples": eval_samples,
            "per_iteration_progress_records": len(iter_progress),
            "skipped_test_calls": skipped, "training_updates_observed": 0,
            "eval_argv": {opt: argv[argv.index(opt) + 1] for opt in
                          ("--eval-iters", "--eval-interval", "--eval-micro-batch-size",
                           "--eval-global-batch-size", "--valid-data-path", "--test-data-path",
                           "--dataloader-type", "--seed", "--global-batch-size", "--micro-batch-size",
                           "--seq-length", "--train-iters", "--lr-decay-iters", "--lr-warmup-iters",
                           "--lr-wsd-decay-iters")
                          if opt in argv},
            "eval_argv_flags": [opt for opt in ("--full-validation", "--no-load-optim", "--no-load-rng",
                                                "--override-opt_param-scheduler", "--bf16") if opt in argv]}


def analyze_capture(capture_path, eval_samples, aggregate):
    records = [json.loads(line) for line in Path(capture_path).read_text().splitlines() if line.strip()]
    lm = [record for record in records if record["key"] == "lm loss"]
    if len(lm) != eval_samples:
        raise RuntimeError(f"Capture records {len(lm)} != eval samples {eval_samples}")
    numerator = sum(record["loss_sum"] for record in lm)
    denominator = sum(record["num_tokens"] for record in lm)
    if denominator <= 0 or any(record["num_tokens"] <= 0 or record["num_tokens"] > 2048 for record in lm):
        raise RuntimeError("Invalid per-sample token counts in capture")
    ratio = numerator / denominator
    if not math.isclose(ratio, aggregate, rel_tol=1e-4):
        raise RuntimeError(f"Capture ratio {ratio} != printed aggregate {aggregate}")
    per_iteration = [record["loss_sum"] / record["num_tokens"] for record in lm]
    return {"capture_records": len(lm), "loss_numerator": numerator,
            "valid_token_denominator": denominator,
            "denominator_scope": "exact evaluator loss-mask token count summed over the full development single epoch",
            "capture_ratio": ratio, "printed_aggregate": aggregate,
            "cross_check_rel_tol": 1e-4,
            "tokens_per_sample": {"min": min(record["num_tokens"] for record in lm),
                                  "max": max(record["num_tokens"] for record in lm),
                                  "mean": statistics.mean(record["num_tokens"] for record in lm)},
            "per_iteration_loss": {"count": len(per_iteration), "min": min(per_iteration),
                                   "max": max(per_iteration), "mean": statistics.mean(per_iteration),
                                   "first3": per_iteration[:3], "last3": per_iteration[-3:],
                                   "full_records_on_nfs": str(capture_path)},
            "per_iteration_scope": "eval gb/mb 1/1: one captured microbatch = one eval iteration = one dev sample"}


def main_checked():
    runtime = runtime_gates()
    write_json(OUTPUT / "runtime-gates.json", runtime)
    mark_stage("runtime")
    api, preflight = hf_preflight()
    local, download_receipt = download_and_verify(api)
    eval_load, inspection = inspect_and_stage(local)
    october_identity = october_data_identity()
    unique = f"hmoe-lionmoe-20261123-baseline-eval-v{VERSION}-lm5r50-october-endpoint"
    env, removed_keys, capture_path = eval_environment(eval_load, runtime["corpus_root"], unique)
    write_json(OUTPUT / "env-cleanup.json", {"removed_inherited_keys": removed_keys,
                                             "scope": "M1 hardening: post-echo/argv-gate env levers cleared before the eval launch"})
    timeout = remaining_seconds()
    if timeout < 900:
        raise RuntimeError(f"Insufficient payload cap remaining for evaluation: {timeout:.1f}s")
    eval_started = time.perf_counter()
    code = run_process(["bash", str(ROOT / "scripts/run_stage3_moe_pretrain.sh"),
                        "muon_bf16_state_fp32", "eval-full-dev"], env,
                       OUTPUT / "eval-wrapper.stdout.log", OUTPUT / "eval-wrapper.stderr.log", timeout)
    eval_duration = time.perf_counter() - eval_started
    if code != 0:
        raise RuntimeError(f"Eval launcher wrapper exit {code}; see eval-wrapper logs")
    mark_stage("eval_launched")
    # Capture cross-check first, then the full log/record analysis against the same aggregate.
    text = (OUTPUT / "eval-wrapper.stdout.log").read_text(errors="replace")
    train_log = re.search(r"^TRAIN_LOG=(.*)$", text, re.MULTILINE).group(1)
    log_text = Path(train_log).read_text(errors="replace")
    aggregate = float(re.findall(r"validation loss at iteration 17242 on validation set \| lm loss value: ([0-9.eE+-]+)", log_text)[0])
    eval_samples = int(re.findall(r"Evaluating on (\d+) samples", log_text)[0])
    capture = analyze_capture(capture_path, eval_samples, aggregate)
    evaluation = analyze_evaluation(OUTPUT / "eval-wrapper.stdout.log", unique, capture_path, aggregate)
    mark_stage("eval_valid")
    record = evaluation["record"]
    result = {"schema": "hmoe-lionmoe-baseline-eval-030-result-v1", "job_note": os.environ["HMOE_JOB_NOTE"],
              "version": VERSION, "seed": SEED,
              "protocol_label": EVAL_LABEL,
              "endpoint": {"repo": HF_REPO, "file": HF_FILE, "iteration": ENDPOINT_ITERATION,
                           "bytes": EXPECTED_BYTES, "sha256": EXPECTED_SHA256,
                           "download_receipt": download_receipt, "inspection": inspection},
              "eval_only": True, "resumable": False, "training_updates": 0,
              "full_development": {"aggregate_loss": evaluation["aggregate_full_dev_loss"],
                                   "eval_samples": evaluation["eval_samples"],
                                   "valid_token_denominator": capture["valid_token_denominator"],
                                   "loss_numerator": capture["loss_numerator"],
                                   "denominator_scope": capture["denominator_scope"],
                                   "per_iteration": capture["per_iteration_loss"],
                                   "eval_argv": evaluation["eval_argv"], "eval_argv_flags": evaluation["eval_argv_flags"],
                                   "evaluator_identity": "identical eval-full-dev mode/argv to be used for the new full arms' final full-development evaluation",
                                   "october_context_loss_2_621174": "context only; not directly comparable until this re-evaluation",
                                   "test_evaluated": False, "skipped_test_calls": evaluation["skipped_test_calls"]},
              "data_identity": {"current_corpus_manifest_sha256": CORPUS_MANIFEST_SHA,
                                "current_corpus_gate": "passed", "october_search": october_identity},
              "hf_preflight": preflight,
              "identity": {key: os.environ[key] for key in ("EXPECTED_SOURCE_COMMIT", "EXPECTED_CARRIER_COMMIT",
                                                              "EXPECTED_BUNDLE_ID", "EXPECTED_PAYLOAD_SHA256")},
              "runtime_gates": "passed", "env_cleanup_removed": removed_keys,
              "eval_wrapper_exit": code, "train_exit": 0,
              "eval_duration_seconds": round(eval_duration, 3),
              "memory": record["measurement"]["memory"],
              "result_record_path": evaluation["result_path"], "train_log_path": evaluation["train_log"],
              "cross_era_caveat": "plain-path code math identity October->current verified statically (gated additions only); hardware/allocation era drift remains a limitation",
              "passed": True, "blocked": False,
              "started_at": STARTED.isoformat(), "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "duration_seconds": round((datetime.datetime.now(datetime.timezone.utc) - STARTED).total_seconds(), 3),
              "cap_seconds": CAP_SECONDS,
              "raw_artifact_receipts": [artifact(path) for path in
                                        (OUTPUT / "runtime-gates.json", OUTPUT / "hf-preflight.json",
                                         OUTPUT / "download-receipt.json", OUTPUT / "checkpoint-inspection.json",
                                         OUTPUT / "october-data-identity.json", OUTPUT / "env-cleanup.json",
                                         capture_path, Path(evaluation["result_path"]), Path(evaluation["train_log"]),
                                         OUTPUT / "eval-wrapper.stdout.log")]}
    write_json(OUTPUT / "measurement-summary.json", result)
    relay = {key: value for key, value in result.items() if key != "raw_artifact_receipts"}
    relay["raw_artifact_receipt_count"] = len(result["raw_artifact_receipts"])
    relay["detailed_summary_path"] = str(OUTPUT / "measurement-summary.json")
    write_json(OUTPUT / "relay-summary.json", relay)
    envelope = seal_envelope("small-artifacts.json", "hmoe-lionmoe-baseline-eval-030-small-artifacts-v1",
                             [("measurement-summary", OUTPUT / "relay-summary.json"),
                              ("hf-preflight", OUTPUT / "hf-preflight.json"),
                              ("download-receipt", OUTPUT / "download-receipt.json"),
                              ("checkpoint-inspection", OUTPUT / "checkpoint-inspection.json"),
                              ("october-data-identity", OUTPUT / "october-data-identity.json")],
                             "BASELINE_EVAL_V030")
    result["final_envelope"] = envelope
    write_json(OUTPUT / "measurement-summary.json", result)
    print("BASELINE_EVAL_V030_RESULT " + json.dumps({key: value for key, value in result.items()
                                                     if key not in ("raw_artifact_receipts", "endpoint")}, sort_keys=True), flush=True)
    return 0


def main():
    try:
        return main_checked()
    except Exception as error:
        blocked = isinstance(error, BaselineBlocker)
        available = []
        for path in (OUTPUT / "runtime-gates.json", OUTPUT / "hf-preflight.json", OUTPUT / "download-receipt.json",
                     OUTPUT / "checkpoint-inspection.json", OUTPUT / "october-data-identity.json",
                     OUTPUT / "env-cleanup.json", OUTPUT / "progress.json"):
            if path.is_file():
                available.append(artifact(path))
        partial = {"schema": "hmoe-lionmoe-baseline-eval-030-error-v1", "passed": False, "blocked": blocked,
                   "job_note": os.environ.get("HMOE_JOB_NOTE"), "completed_stages": STAGES,
                   "error_type": type(error).__name__, "error": redact(str(error)),
                   "traceback": redact(traceback.format_exc())[-4000:],
                   "available_artifacts": available,
                   "mac_fallback": "none; weights never travel through Mac",
                   "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write_json(OUTPUT / "partial-error.json", partial)
        labeled = [("partial-error", OUTPUT / "partial-error.json")]
        for label, name in (("hf-preflight", "hf-preflight.json"), ("download-receipt", "download-receipt.json"),
                            ("checkpoint-inspection", "checkpoint-inspection.json"),
                            ("october-data-identity", "october-data-identity.json")):
            if (OUTPUT / name).is_file():
                labeled.append((label, OUTPUT / name))
        seal_envelope("small-artifacts.json", "hmoe-lionmoe-baseline-eval-030-small-artifacts-v1",
                      labeled, "BASELINE_EVAL_V030")
        print("BASELINE_EVAL_V030_RESULT " + json.dumps({"passed": False, "blocked": blocked,
                                                         "completed_stages": STAGES}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
