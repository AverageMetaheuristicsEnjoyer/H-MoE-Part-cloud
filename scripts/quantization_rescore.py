#!/usr/bin/env python3
"""Score the three finished Muon checkpoints on one identical BF16 window."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from huggingface_hub import hf_hub_download


REPO = "AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints"
# Metadata verified against the HF API on 2026-10-01 before this campaign.
REVISION = "bf38e335691db9b7e87f905e2d8b538a6c88c880"
TARGETS = {
    "baseline": {
        "prefix": "1c-mb4", "arm": "muon_bf16_state_fp32", "training_micro_batch": 4,
        "training_compute": "BF16", "bytes": 10708281947,
        "sha256": "686598d965510c6fb33a4f8d508d737492e6014e165f106c139eab959aa4b95a",
    },
    "delayed": {
        "prefix": "1c-mb16", "arm": "muon_fp8gemm_state_fp32", "training_micro_batch": 16,
        "training_compute": "delayed HYBRID", "bytes": 10708622491,
        "sha256": "0190459aa8a5d52e105ec2c6d7a9b819805dcfa7a33e8dea3c6e7526ba3e435d",
    },
    "blockwise": {
        "prefix": "fp8-corrected-20260922-r1", "arm": "muon_fp8gemm_state_fp32", "training_micro_batch": 16,
        "training_compute": "blockwise E4M3", "bytes": 10708345243,
        "sha256": "c8bd4aabdbafae5ebcf3ab3a65d46b222fd893f4b6c2507aaf003850c7ffb075",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.environ.get("STAGE3_MOE_RESCORE_OUTPUT", "/tmp/hmoe-quantization-rescore/logs"))
    parser.add_argument("--scratch", default=os.environ.get("STAGE3_MOE_RESCORE_SCRATCH", "/tmp"))
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--eval-iters", type=int, default=240)
    parser.add_argument("--tag", default=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if args.eval_iters < 1 or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
        parser.error("eval-iters must be positive and tag must be a path-safe identifier")
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True
    ).strip()
    if len(gpu.splitlines()) != 1:
        raise RuntimeError("Rescore requires exactly one visible GPU")
    data_root = os.environ.get("STAGE3_MOE_DATA_ROOT")
    if not data_root:
        for candidate in (
            "/home/jovyan/data/fineweb-edu-gpt2-megatron/data",
            "/workspace-SR006.nfs2/hmoe-data/fineweb-edu-gpt2-megatron/data",
            "/workspace-SR006.nfs3/hmoe-data/fineweb-edu-gpt2-megatron/data",
        ):
            if all(Path(candidate, split + ".bin").is_file() for split in ("development", "final")):
                data_root = candidate
                break
    if not data_root:
        raise RuntimeError("Set STAGE3_MOE_DATA_ROOT to the original Stage 3 corpus")
    sources = ("scripts/quantization_rescore.py", "scripts/run_stage3_moe_pretrain.sh",
               "stage3_moe/pretrain_gpt.py", "configs/stage3-moe-1p029b.sh")
    manifest = {
        "hf_repo": REPO, "hf_revision": REVISION, "targets": {key: TARGETS[key] for key in args.targets},
        "evaluation_compute": "BF16", "evaluation_arm": "muon_bf16_state_fp32", "iteration": 17242,
        "sampler_start": 0, "eval_iters": args.eval_iters, "sequences_per_split": args.eval_iters * 208,
        "micro_batch": 16, "global_batch": 208, "dp": 1, "training": False,
        "load_optimizer": False, "load_rng": False, "data_root": data_root, "gpu": gpu,
        "image": os.environ.get("MLSUB_IMAGE"), "tag": args.tag,
        "source_files_sha256": {name: sha256(root / name) for name in sources},
        "interpretation": "Fixed equal windows, not exact single epochs. Historical baseline trained at mb4; FP8 arms at mb16.",
    }
    (output / f"manifest-{args.tag}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    token = os.environ.get("HF_TOKEN")
    token_file = Path(os.environ.get("STAGE3_MOE_HF_TOKEN_FILE", "/home/jovyan/.cache/huggingface/token"))
    if not token and token_file.is_file():
        token = token_file.read_text().strip()
    results = []
    print(f"EVAL_PROTOCOL=BF16 skip_train=true optimizer_load=false rng_load=false GPU={gpu}", flush=True)
    for label in args.targets:
        target = TARGETS[label]
        remote = f"{target['prefix']}/{target['arm']}/iter_0017242/mp_rank_00/model_optim_rng.pt"
        with tempfile.TemporaryDirectory(prefix=f"quant-rescore-{label}-", dir=scratch) as directory:
            work = Path(directory)
            available = os.statvfs(work)
            if available.f_bavail * available.f_frsize < target["bytes"] + 4 * 1024**3:
                raise RuntimeError("Insufficient node-local space for next checkpoint")
            downloaded = Path(hf_hub_download(
                repo_id=REPO, filename=remote, revision=REVISION, local_dir=work / "download", token=token,
            ))
            digest = sha256(downloaded)
            if downloaded.stat().st_size != target["bytes"] or digest != target["sha256"]:
                raise RuntimeError(f"Checkpoint integrity mismatch: {label}")
            checkpoint = work / "checkpoint"
            weights = checkpoint / "iter_0017242/mp_rank_00/model_optim_rng.pt"
            weights.parent.mkdir(parents=True)
            os.link(downloaded, weights)
            (checkpoint / "latest_checkpointed_iteration.txt").write_text("17242\n")
            print(f"CHECKPOINT_VERIFIED label={label} path={remote} sha256={digest}", flush=True)
            suffix = f"quant-rescore-{args.tag}-{label}"
            run_id = f"stage3-muon_bf16_state_fp32-eval-lm-fixed-{suffix}"
            env = os.environ.copy()
            env.update(
                STAGE3_MOE_EVAL_LOAD=str(checkpoint), STAGE3_MOE_RUN_SUFFIX=suffix,
                STAGE3_MOE_MICRO_BATCH="16", STAGE3_MOE_GLOBAL_BATCH="208", STAGE3_MOE_EP="1",
                STAGE3_MOE_EVAL_ITERS=str(args.eval_iters), STAGE3_MOE_LOG_ROOT=str(output),
                STAGE3_MOE_CKPT_ROOT=str(work / "unused-checkpoints"),
                STAGE3_MOE_DATA_CACHE_PATH=str(work / "data-cache"), STAGE3_MOE_DATA_ROOT=data_root,
                STAGE3_MOE_PROPAGATE_EXIT="1", STAGE3_MOE_MUON_STATE_CODEC="fp32",
                STAGE3_MOE_ADAM_STATE_CODEC="fp32", STAGE3_MOE_FP8_COMPUTE_ARGS="",
                WANDB_MODE="offline", WANDB_RUN_ID=run_id,
            )
            subprocess.run(
                ["bash", str(root / "scripts/run_stage3_moe_pretrain.sh"), "muon_bf16_state_fp32", "eval-lm-fixed"],
                cwd=root, env=env, check=True,
            )
            run = output / run_id
            log = max(run.glob("train-*.log"), key=lambda path: path.stat().st_mtime)
            contents = log.read_text(errors="replace")
            metrics = {}
            for split in ("validation", "test"):
                matches = re.findall(
                    rf"loss at iteration 17242 on {split} set\s*\| lm loss value:\s*([0-9.eE+\-]+)", contents,
                )
                if len(matches) != 1 or not math.isfinite(float(matches[0])):
                    raise RuntimeError(f"Expected one finite final {split} result for {label}")
                metrics[split] = float(matches[0])
            results.append({
                "label": label, "training_arm": target["arm"], "training_compute": target["training_compute"],
                "training_micro_batch": target["training_micro_batch"], "source": remote,
                "sha256": digest, "hf_revision": REVISION, "gpu": gpu, "evaluation_compute": "BF16",
                "evaluation_arm": "muon_bf16_state_fp32", "eval_iters": args.eval_iters,
                "micro_batch": 16, "global_batch": 208, "sampler_start": 0, "log": str(log), **metrics,
            })
            (output / f"results-{args.tag}.json").write_text(json.dumps(results, indent=2) + "\n")
            print("BF16_RESCORE_RESULT " + json.dumps(results[-1]), flush=True)
        print(f"SCRATCH_RELEASED={label}", flush=True)
    print(f"BF16_RESCORE_COMPLETE={len(results)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("BF16_RESCORE_EXIT=1", flush=True)
        raise
    else:
        print("BF16_RESCORE_EXIT=0", flush=True)
