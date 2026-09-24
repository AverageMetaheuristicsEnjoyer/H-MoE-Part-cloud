#!/usr/bin/env python3
"""Run a corrected 1C arm, archiving complete pre-decay and final checkpoints."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
ARM = sys.argv[1]
ARMS = {
    f"{optimizer}_{precision}"
    for optimizer in ("adamw", "muon", "frugal_coord")
    for precision in ("fp8gemm_state_fp32", "bf16_state_fp8", "fp8gemm_state_fp8")
}
if ARM not in ARMS:
    raise ValueError(f"unsupported corrected arm: {ARM}")

WAVE = os.environ.get("STAGE3_MOE_WAVE", "fp8-corrected-20260922")
REPO = "AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints"
PREFIX = f"{WAVE}/{ARM}"
CHECKPOINT_ROOT = Path(f"/tmp/hmoe-{WAVE}")
CHECKPOINT = CHECKPOINT_ROOT / "1c" / ARM
LOG_ROOT = Path(f"/workspace-SR006.nfs2/hmoe-cloud/{WAVE}")
env = os.environ.copy()
env.update(
    LC_ALL="C",
    WANDB_MODE="online",
    WANDB_BASE_URL="https://wandb-radfan.ru",
    STAGE3_MOE_WANDB_PROJECT="hmoe-stage3",
    STAGE3_MOE_WANDB_ENTITY="andrey",
    STAGE3_MOE_LR="0.00163",
    STAGE3_MOE_MIN_LR="0.000163",
    STAGE3_MOE_MICRO_BATCH="16",
    STAGE3_MOE_RUN_SUFFIX=WAVE,
    STAGE3_MOE_CKPT_ROOT=str(CHECKPOINT_ROOT),
    STAGE3_MOE_LOG_ROOT=str(LOG_ROOT),
    STAGE3_MOE_DATA_CACHE_PATH=f"/tmp/hmoe-{WAVE}-data-cache",
    STAGE3_MOE_PROPAGATE_EXIT="1",
    STAGE3_MOE_FP8_DEQUANT_CHUNK="0",
    NVTE_FP8_BLOCK_SCALING_FP32_SCALES="1",
)
if "_fp8gemm_" in ARM:
    env["STAGE3_MOE_FP8_COMPUTE_ARGS"] = "--fp8-format e4m3 --fp8-recipe blockwise"
if ARM.endswith("_state_fp8"):
    env["STAGE3_MOE_FP8_STATE_DTYPES"] = "e4m3:e4m3"


def archive(iteration, api):
    directory = CHECKPOINT / f"iter_{iteration:07d}"
    remote = f"{PREFIX}/{directory.name}"
    staging = CHECKPOINT.parent / f".{ARM}-archive-{iteration}"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(directory, staging, copy_function=os.link)
    files = {}
    for path in staging.rglob("*.pt"):
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        files[str(path.relative_to(staging))] = (path.stat().st_size, digest)
    if not files or any(size == 0 for size, _ in files.values()):
        raise RuntimeError(f"empty checkpoint: {directory}")
    api.upload_folder(repo_id=REPO, folder_path=str(staging), path_in_repo=remote,
                      repo_type="model", allow_patterns=["*.pt"])
    entries = {p.path.removeprefix(remote + "/"): p for p in api.list_repo_tree(
        REPO, path_in_repo=remote, repo_type="model", recursive=True
    ) if getattr(p, "size", None) is not None}
    for name, (size, digest) in files.items():
        item = entries.get(name)
        lfs = getattr(item, "lfs", None)
        remote_digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if item is None or item.size != size or remote_digest != digest:
            raise RuntimeError(f"checkpoint verification failed: {remote}/{name}")
    print(f"HF_VERIFIED iteration={iteration} files={len(files)} path={remote}", flush=True)
    shutil.rmtree(staging)
    return files


def restore(api):
    tracker = CHECKPOINT / "latest_checkpointed_iteration.txt"
    local = int(tracker.read_text().strip()) if tracker.is_file() else 0
    pointer = f"{PREFIX}/resume.json"
    remote = None
    if api.file_exists(REPO, pointer, repo_type="model"):
        from huggingface_hub import hf_hub_download
        filename = hf_hub_download(REPO, pointer, token=env["HF_TOKEN"], force_download=True,
                                   cache_dir=f"/tmp/hmoe-{WAVE}-hf-cache")
        remote = json.loads(Path(filename).read_text())
        iteration = remote["iteration"]
        if iteration > local:
            for name, (size, digest) in remote["files"].items():
                filename = hf_hub_download(REPO, f"{PREFIX}/iter_{iteration:07d}/{name}",
                                           token=env["HF_TOKEN"], cache_dir=f"/tmp/hmoe-{WAVE}-hf-cache")
                source = Path(filename)
                with source.open("rb") as handle:
                    actual = hashlib.file_digest(handle, "sha256").hexdigest()
                if source.stat().st_size != size or actual != digest:
                    raise RuntimeError(f"restore verification failed: {name}")
                target = CHECKPOINT / f"iter_{iteration:07d}" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            tracker.write_text(str(iteration))
            local = iteration
            print(f"HF_RESTORED iteration={iteration}", flush=True)
    prior_logs = LOG_ROOT / f"stage3-{ARM}-full-{WAVE}"
    if not local and any(prior_logs.glob("train-*.log")):
        raise RuntimeError("prior full run exists but no recoverable checkpoint; refusing fresh restart")
    return remote


def save_resume_pointer(iteration, files, api, previous=None):
    retained = set((previous or {}).get("verified_retained", []))
    if iteration in (13794, 17242):
        retained.add(iteration)
    pointer = {"iteration": iteration, "files": files, "verified_retained": sorted(retained)}
    api.upload_file(repo_id=REPO, path_or_fileobj=json.dumps(pointer).encode(),
                    path_in_repo=f"{PREFIX}/resume.json", repo_type="model")
    return pointer


def main():
    if not env.get("HF_TOKEN") or not env.get("WANDB_API_KEY"):
        raise RuntimeError("HF_TOKEN and WANDB_API_KEY are required")
    if env.get("MLSUB_IMAGE") != "te4":
        raise RuntimeError("corrected wave requires the verified te4 image")
    subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"], check=True)
    for path in (Path("/tmp"), LOG_ROOT.parent):
        disk = shutil.disk_usage(path)
        print(f"DISK path={path} free_gib={disk.free / 2**30:.2f}", flush=True)
        required = 70 if path == Path("/tmp") else 2
        if disk.free < required * 2**30:
            raise RuntimeError(f"insufficient space at {path}: need {required} GiB")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = f"/tmp/hmoe-{WAVE}-hf-cache"
    os.environ["HF_HOME"] = env["HF_HOME"]
    api = HfApi(token=env["HF_TOKEN"])
    info = api.model_info(REPO)
    print(f"HF_REPO={info.id} private={info.private}", flush=True)
    subprocess.run(["python", "-c", "import wandb, torch.utils.tensorboard"], env=env, check=True)
    manifest = {
        "arm": ARM, "wave": WAVE, "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "image": env["MLSUB_IMAGE"], "lr": 0.00163, "micro_batch": 16,
        "global_batch": 208, "iterations": 17242, "retained_iterations": [13794, 17242],
        "fp8_compute": env.get("STAGE3_MOE_FP8_COMPUTE_ARGS"),
        "fp8_state_dtypes": env.get("STAGE3_MOE_FP8_STATE_DTYPES"),
    }
    api.upload_file(repo_id=REPO, path_or_fileobj=json.dumps(manifest, indent=2).encode(),
                    path_in_repo=f"{PREFIX}/launch.json", repo_type="model")
    (LOG_ROOT / f"{ARM}-launch.json").write_text(json.dumps(manifest, indent=2))
    remote_resume = restore(api)
    launcher = ROOT / "scripts/run_stage3_moe_pretrain.sh"
    smoke_env = {**env, "STAGE3_MOE_KEEP_SMOKE_CKPT": "1"}
    for steps in (() if (CHECKPOINT / "latest_checkpointed_iteration.txt").is_file() else (51, 52)):
        smoke_env["STAGE3_MOE_SMOKE_ITERS"] = str(steps)
        code = subprocess.call([str(launcher), ARM, "smoke"], env=smoke_env, cwd=ROOT)
        print(f"SMOKE_EXIT={code} arm={ARM} steps={steps}", flush=True)
        if code:
            print(f"FULL_RESULT=FAIL reason=smoke TRAIN_EXIT={code}", flush=True)
            return
    smoke_dir = CHECKPOINT_ROOT / "smoke" / f"{ARM}-{WAVE}"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    print(f"FULL_START arm={ARM} target=17242", flush=True)
    process = subprocess.Popen([str(launcher), ARM, "full"], env=env, cwd=ROOT)
    verified = set((remote_resume or {}).get("verified_retained", []))
    while True:
        tracker = CHECKPOINT / "latest_checkpointed_iteration.txt"
        latest = tracker.read_text().strip() if tracker.is_file() else ""
        iteration = int(latest) if latest.isdigit() else 0
        code = process.poll()
        targets = [13794, 17242]
        if iteration:
            targets.append(iteration)
        if code is not None and code != 0 and iteration:
            targets.append(iteration)
        for target in dict.fromkeys(targets):
            if target in verified or iteration < target or not (CHECKPOINT / f"iter_{target:07d}").is_dir():
                continue
            try:
                files = archive(target, api)
                if target >= (remote_resume or {}).get("iteration", 0):
                    previous = remote_resume
                    remote_resume = save_resume_pointer(target, files, api, previous)
                    if previous and previous["iteration"] not in (target, 13794, 17242):
                        api.delete_folder(repo_id=REPO, path_in_repo=f"{PREFIX}/iter_{previous['iteration']:07d}", repo_type="model")
            except Exception as exc:
                print(f"HF_RETRY iteration={target} error_type={type(exc).__name__}", flush=True)
                time.sleep(60)
                continue
            verified.add(target)
        if 13794 in verified and iteration > 13794:
            retained = CHECKPOINT / "iter_0013794"
            if retained.exists():
                shutil.rmtree(retained)
                print("REMOVED_LOCAL_VERIFIED iteration=13794", flush=True)
        if code is not None:
            if code == 0 and {13794, 17242} <= verified:
                print(f"FULL_RESULT=PASS TRAIN_EXIT=0 arm={ARM} HF_CHECKPOINTS=13794,17242", flush=True)
                return
            if code != 0 and (not iteration or iteration in verified):
                print(f"FULL_RESULT=FAIL TRAIN_EXIT={code} arm={ARM} rescued_iteration={iteration}", flush=True)
                return
            if code == 0 and iteration != 17242:
                print(f"FULL_RESULT=FAIL reason=incomplete_training iteration={iteration}", flush=True)
                return
        time.sleep(30)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        message = str(exc)
        for key in ("HF_TOKEN", "WANDB_API_KEY"):
            if env.get(key):
                message = message.replace(env[key], "[REDACTED]")
        print(f"FULL_RESULT=FAIL error_type={type(exc).__name__} message={message}", flush=True)
