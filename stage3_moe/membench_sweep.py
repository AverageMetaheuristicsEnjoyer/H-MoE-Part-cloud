#!/usr/bin/env python3
"""Sweep the MoE memory/time table: model shape x arm x micro-batch, one GPU.

The protocol is docs/membench.md. Each point is a separate process, because a run
that exhausts the GPU is a result of the sweep and not a failure of it: the record
is written, every larger micro-batch of that model and arm is skipped, and the
sweep carries on.

The measurement itself is the Stage 3 probe (stage3_moe/result_writer.py), which
already resets peak memory after warmup and writes the memory, timing and
optimizer-state ledgers this table needs. Nothing is re-derived here.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage3_moe import ARMS, MODEL_SHAPES  # noqa: E402

# Membench trades accumulation against the micro-batch so every point of the table
# processes the same tokens per optimizer step. 16 sequences x 2,048 = 32,768.
DEFAULT_GLOBAL_BATCH = 16
DEFAULT_MICRO_BATCHES = (1, 2, 4, 8, 16)

OOM_SIGNATURES = (
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    "out of memory",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def point_path(results_root: Path, model: str, arm: str, micro_batch: int) -> Path:
    return results_root / "runs" / f"{model}-{arm}-mb{micro_batch}.json"


def controls(args, model: str, micro_batch: int) -> dict:
    """What a stored point must reproduce to be reused instead of rerun."""
    return {
        "harness_revision": 1,
        "model": model,
        "micro_batch": micro_batch,
        "global_batch": args.global_batch,
        "accumulation_steps": args.global_batch // micro_batch,
        "sequence_length": 2048,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "gpu_count": args.gpu_count,
    }


def stored_point(path: Path, expected: dict) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if payload.get("controls") != expected:
        return None
    return payload if payload.get("status") in ("complete", "oom") else None


def write_point(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def newest_train_log(run_dir: Path) -> Path | None:
    logs = sorted(run_dir.glob("train-*.log"), key=lambda item: item.stat().st_mtime)
    return logs[-1] if logs else None


def looks_like_oom(text: str) -> bool:
    return any(signature in text for signature in OOM_SIGNATURES)


def run_point(args, model: str, arm: str, micro_batch: int) -> dict:
    suffix = f"{model}-mb{micro_batch}"
    run_dir = Path(args.log_root) / f"stage3-{arm}-membench-{suffix}"
    # The probe appends, so a rerun of the same point would leave two records and the
    # reader could not tell which one it just produced.
    record_path = run_dir / "results.jsonl"
    if record_path.exists():
        record_path.unlink()

    environment = os.environ.copy()
    environment.update(
        {
            "STAGE3_MOE_MODEL": model,
            "STAGE3_MOE_MICRO_BATCH": str(micro_batch),
            "STAGE3_MOE_GLOBAL_BATCH": str(args.global_batch),
            "STAGE3_MOE_RUN_SUFFIX": suffix,
            "STAGE3_MOE_MEMBENCH_WARMUP": str(args.warmup_steps),
            "STAGE3_MOE_MEMBENCH_MEASURE": str(args.measured_steps),
            "STAGE3_MOE_LOG_ROOT": str(args.log_root),
            "STAGE3_MOE_PROPAGATE_EXIT": "1",
            # A membench point is one process on one card; a W&B run per point would
            # be 60 runs of 17 steps each and tells nobody anything.
            "WANDB_MODE": "offline",
        }
    )
    started = time.time()
    completed = subprocess.run(
        [str(ROOT / "scripts/run_stage3_moe_pretrain.sh"), arm, "membench"],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    elapsed = time.time() - started

    identity = {
        "model": model,
        "arm": arm,
        "micro_batch": micro_batch,
        "controls": controls(args, model, micro_batch),
        "launcher_exit": completed.returncode,
        "wall_seconds": elapsed,
    }

    if record_path.is_file():
        lines = [line for line in record_path.read_text().splitlines() if line.strip()]
        if lines:
            record = json.loads(lines[-1])
            if record.get("status") == "completed":
                return {**identity, "status": "complete", "record": record}

    train_log = newest_train_log(run_dir)
    tail = ""
    if train_log is not None:
        tail = train_log.read_text(errors="replace")[-20000:]
    if looks_like_oom(tail):
        log(f"OOM: {model}/{arm}/mb{micro_batch}")
        return {**identity, "status": "oom", "log_tail": tail[-4000:]}
    return {**identity, "status": "failed", "log_tail": tail[-4000:]}


def selected(values, requested, label):
    if not requested:
        return list(values)
    wanted = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = [item for item in wanted if item not in values]
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}")
    return wanted


def export_row(payload: dict) -> str:
    """One line per point, so `mlsub logs` carries the table even when the disk does not."""
    fields = [payload["model"], payload["arm"], str(payload["micro_batch"]), payload["status"]]
    if payload["status"] == "complete":
        measurement = payload["record"]["measurement"]
        state = payload["record"]["optimizer_state"]
        timing = measurement["timing"]
        fields += [
            f"{measurement['memory']['max_allocated_bytes'] / 1e9:.3f}",
            f"{measurement['memory']['max_reserved_bytes'] / 1e9:.3f}",
            f"{(state.get('persistent_total_bytes') or 0) / 1e9:.3f}",
            f"{(timing['full_step_seconds'] or 0) * 1000:.2f}",
            f"{(timing['optimizer_step_seconds'] or 0) * 1000:.2f}",
            f"{timing['tokens_per_second'] or 0:.1f}",
        ]
    else:
        fields += [""] * 6
    return "PT\t" + "\t".join(fields)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=None, help="comma-separated subset of model shapes")
    parser.add_argument("--arms", default=None, help="comma-separated subset of arms")
    parser.add_argument("--micro-batches", default=None, help="comma-separated micro-batch sizes")
    parser.add_argument("--global-batch", type=int, default=DEFAULT_GLOBAL_BATCH)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=12)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--export-only", action="store_true",
                        help="print the recorded points and exit, running nothing")
    args = parser.parse_args()

    models = selected(MODEL_SHAPES, args.models, "model shape")
    arms = selected(ARMS, args.arms, "arm")
    micro_batches = (
        sorted(int(value) for value in args.micro_batches.split(","))
        if args.micro_batches
        else list(DEFAULT_MICRO_BATCHES)
    )
    for micro_batch in micro_batches:
        if args.global_batch % micro_batch:
            raise ValueError(
                f"global batch {args.global_batch} is not divisible by micro-batch {micro_batch}"
            )

    results_root = args.results_root.resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    if args.export_only:
        print("PT\tmodel\tarm\tmicro_batch\tstatus\tpeak_gb\treserved_gb\tstate_gb\tstep_ms\topt_ms\ttokens_per_second")
        for path in sorted((results_root / "runs").glob("*.json")):
            print(export_row(json.loads(path.read_text())))
        return 0

    exhausted: set[tuple[str, str]] = set()
    # A broken arm -- a bad config, a missing optimizer -- fails identically at every
    # micro-batch, and each attempt costs a process start and a Megatron init. Stop it
    # here; a resubmission retries, because a failed point is never reused.
    broken: set[tuple[str, str]] = set()
    for model in models:
        for micro_batch in micro_batches:
            for arm in arms:
                key = (model, arm)
                path = point_path(results_root, model, arm, micro_batch)
                expected = controls(args, model, micro_batch)
                if key in exhausted or key in broken:
                    reason = "ran out of memory" if key in exhausted else "failed"
                    status = "skipped_after_oom" if key in exhausted else "skipped_after_failure"
                    log(f"skip {model}/{arm}/mb{micro_batch}: a smaller micro-batch {reason}")
                    write_point(path, {"model": model, "arm": arm, "micro_batch": micro_batch,
                                       "controls": expected, "status": status})
                    continue
                existing = None if args.rerun else stored_point(path, expected)
                if existing is not None:
                    log(f"reuse {path.name} ({existing['status']})")
                    if existing["status"] == "oom":
                        exhausted.add(key)
                    continue
                log(f"run {model}/{arm}/mb{micro_batch} "
                    f"(accumulation {args.global_batch // micro_batch})")
                payload = run_point(args, model, arm, micro_batch)
                write_point(path, payload)
                log(f"  -> {payload['status']} in {payload['wall_seconds']:.0f}s")
                if payload["status"] == "oom":
                    exhausted.add(key)
                elif payload["status"] == "failed":
                    broken.add(key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
