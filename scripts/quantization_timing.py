#!/usr/bin/env python3
"""Measure paired FP8 compute recipes on one GPU from one FP32-state checkpoint."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess


VARIANTS = ("bf16", "delayed_h1", "current", "current_e4m3", "blockwise")
BASE_ITERATION = 13794
TOTAL_STEPS = 120
WARMUP_STEPS = 20


def gpu_snapshot():
    fields = ("name", "uuid", "memory.total", "memory.used", "temperature.gpu", "clocks.sm", "power.draw")
    lines = subprocess.check_output([
        "nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError("Timing requires exactly one visible GPU")
    snapshot = dict(zip(fields, (part.strip() for part in lines[0].split(","))))
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    snapshot["compute_processes"] = [line for line in processes if not line.startswith("No running processes")]
    return snapshot


def read_step_timings(path):
    steps = []
    for line in Path(path).read_text(errors="replace").splitlines():
        iteration = re.search(r"\biteration\s+(\d+)\s*/", line)
        elapsed = re.search(r"elapsed time per iteration \(ms\):\s*([0-9.eE+\-]+)", line)
        if iteration and elapsed:
            steps.append({"iteration": int(iteration[1]), "elapsed_ms": float(elapsed[1])})
    expected = list(range(BASE_ITERATION + 1, BASE_ITERATION + TOTAL_STEPS + 1))
    if [step["iteration"] for step in steps] != expected:
        raise RuntimeError("Expected exactly 120 consecutive logged iterations from checkpoint 13794")
    if any(not math.isfinite(step["elapsed_ms"]) or step["elapsed_ms"] <= 0 for step in steps):
        raise RuntimeError("Nonpositive or nonfinite step timing")
    return steps


def timing_statistics(steps):
    values = [step["elapsed_ms"] for step in steps[WARMUP_STEPS:]]
    return {
        "count": len(values), "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": sorted(values)[math.ceil(0.95 * len(values)) - 1],
        "p95_method": "nearest_rank",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("optimizer", choices=("adamw", "muon"))
    parser.add_argument("--output", default=os.environ.get("STAGE3_MOE_TIMING_OUTPUT", os.environ.get("STAGE3_MOE_LOG_ROOT")))
    parser.add_argument("--stage", default=os.environ.get("STAGE3_MOE_PROBE_STAGE"))
    parser.add_argument("--tag", default=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if not args.output:
        parser.error("--output or STAGE3_MOE_TIMING_OUTPUT must identify persistent storage")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
        parser.error("tag must be a path-safe identifier")
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stage = Path(args.stage or f"/tmp/hmoe-quantization-timing-{args.optimizer}").resolve()
    stage.mkdir(parents=True, exist_ok=True)
    result_path = output / f"timing-{args.optimizer}-{args.tag}.json"
    if result_path.exists():
        raise RuntimeError("Timing tag already exists; use a new tag to preserve its results")
    gpu = gpu_snapshot()
    protocol = {
        "optimizer": args.optimizer, "base_iteration": BASE_ITERATION,
        "total_steps": TOTAL_STEPS, "discarded_warmup_steps": WARMUP_STEPS,
        "measured_steps": TOTAL_STEPS - WARMUP_STEPS,
        "measured_local_steps": [21, 120], "evaluation_iterations": 0,
        "order": list(VARIANTS) + list(reversed(VARIANTS)),
        "gpu_uuid": gpu["uuid"], "initial_gpu": gpu,
        "micro_batch": 16, "global_batch": 208, "dp": 1,
        "optimizer_states": "FP32", "restore_optimizer": True, "restore_rng": True,
        "checkpoint_per_branch": False, "diagnostics": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing_source": "MCore elapsed time per iteration (ms), log_interval=1",
        "interpretation": "Common starting checkpoint and GPU; subsequent training trajectories may diverge. Only steps 21-120 are timed; startup, downloads, and first 20 steps are excluded. No cold-start or memory-saving claim.",
    }
    result = {"status": "running", "protocol": protocol, "branches": []}
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    checkpoint_signature = None
    for position, variant in enumerate(protocol["order"]):
        order = "forward" if position < len(VARIANTS) else "reverse"
        tag = f"timing-{args.tag}-{position:02d}"
        env = os.environ.copy()
        env.pop("STAGE3_MOE_DIAGNOSTIC_PATH", None)
        env.pop("STAGE3_MOE_PROBE_PHASE_TRANSITION", None)
        env.update(
            STAGE3_MOE_PROBE_STAGE=str(stage), STAGE3_MOE_PROBE_TAG=tag,
            STAGE3_MOE_LOG_ROOT=str(output), STAGE3_MOE_PROBE_BASE_ITER=str(BASE_ITERATION),
            STAGE3_MOE_PROBE_ITERS=str(TOTAL_STEPS), STAGE3_MOE_MICRO_BATCH="16",
            STAGE3_MOE_GLOBAL_BATCH="208", STAGE3_MOE_EVAL_ITERS="0",
            STAGE3_MOE_EVAL_INTERVAL="2000000000", STAGE3_MOE_MEM_AUDIT="0",
            STAGE3_MOE_MUON_STATE_CODEC="fp32", STAGE3_MOE_ADAM_STATE_CODEC="fp32",
            WANDB_MODE="offline",
        )
        before = gpu_snapshot()
        if before["uuid"] != protocol["gpu_uuid"]:
            raise RuntimeError("GPU changed during paired timing")
        if before["compute_processes"]:
            raise RuntimeError("GPU has another compute process before timing branch")
        driver_log = output / f"driver-{args.optimizer}-{tag}-{variant}.log"
        print(f"TIMING_BEGIN position={position} order={order} variant={variant} GPU={before['uuid']}", flush=True)
        with driver_log.open("w") as handle:
            with subprocess.Popen(
                ["bash", str(root / "scripts/cloud_quantization_probe.sh"), args.optimizer, variant],
                cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            ) as process:
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    print(line, end="", flush=True)
                returncode = process.wait()
        after = gpu_snapshot()
        summary_path = output / f"summary-{args.optimizer}-{tag}.jsonl"
        manifest_path = output / f"manifest-{args.optimizer}-{tag}.json"
        branch = {
            "position": position, "order": order, "variant": variant,
            "gpu_uuid": before["uuid"], "gpu_before": before, "gpu_after": after,
            "driver_log": str(driver_log), "summary": str(summary_path),
            "manifest": str(manifest_path), "validated": False,
        }
        result["branches"].append(branch)
        try:
            rows = [json.loads(line) for line in summary_path.read_text().splitlines() if line.strip()]
            if returncode != 0 or len(rows) != 1 or not rows[0]["branch_validated"]:
                raise RuntimeError("Training branch failed validation")
            if after["uuid"] != before["uuid"] or after["compute_processes"]:
                raise RuntimeError("GPU changed or remained occupied after the branch")
            row = rows[0]
            if row["muon_codec"] != "fp32" or row["adam_codec"] != "fp32":
                raise RuntimeError("Timing requires FP32 optimizer states")
            manifest = json.loads(manifest_path.read_text())
            current_signature = manifest["checkpoint"]
            if checkpoint_signature is None:
                checkpoint_signature = current_signature
                protocol["checkpoint"] = current_signature
                protocol["checkpoint_source_revision"] = manifest["checkpoint_source_revision"]
            elif checkpoint_signature != current_signature:
                raise RuntimeError("Initial checkpoint changed between timing branches")
            steps = read_step_timings(row["train_log"])
            branch.update(
                validated=True, train_log=row["train_log"], steps=steps,
                warmup_steps=steps[:WARMUP_STEPS], measured_steps=steps[WARMUP_STEPS:],
                statistics=timing_statistics(steps), compute_args=row["compute_args"],
            )
        except Exception as error:
            branch["error"] = str(error)
            result["status"] = "failed"
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            raise
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print("TIMING_BRANCH=" + json.dumps({
            "position": position, "order": order, "variant": variant,
            "gpu_uuid": before["uuid"], **branch["statistics"],
        }), flush=True)
    result["status"] = "completed"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"TIMING_COMPLETE={result_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("QUANTIZATION_TIMING_EXIT=1", flush=True)
        raise
    else:
        print("QUANTIZATION_TIMING_EXIT=0", flush=True)
