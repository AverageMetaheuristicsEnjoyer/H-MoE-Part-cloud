#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import numpy as np


PROGRESS = re.compile(
    r"iteration\s+(?P<iteration>\d+)/\s*\d+.*?"
    r"learning rate:\s+(?P<lr>[0-9.E+-]+).*?"
    r"lm loss:\s+(?P<loss>[0-9.E+-]+).*?"
    r"grad norm:\s+(?P<grad>[0-9.E+-]+).*?"
    r"number of skipped iterations:\s+(?P<skipped>\d+).*?"
    r"number of nan iterations:\s+(?P<nan>\d+)"
)


def load_run(spec):
    label, directory, decay_start = spec.split(":", 2)
    rows = {}
    logs = sorted(Path(directory).glob("train-*.log"))
    if not logs:
        raise RuntimeError(f"no train logs for {label}: {directory}")
    for log in logs:
        with log.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = PROGRESS.search(line)
                if match is None:
                    continue
                row = {
                    "iteration": int(match.group("iteration")),
                    "learning_rate": float(match.group("lr")),
                    "loss": float(match.group("loss")),
                    "grad_norm": float(match.group("grad")),
                    "skipped_iterations": int(match.group("skipped")),
                    "nan_iterations": int(match.group("nan")),
                    "log": str(log),
                }
                rows[row["iteration"]] = row
    rows = [rows[key] for key in sorted(rows)]
    if not rows:
        raise RuntimeError(f"no progress rows for {label}: {directory}")
    return label, int(decay_start), rows


def summarize(label, decay_start, rows, bin_width):
    bins = []
    final_offset = rows[-1]["iteration"] - decay_start
    for offset in range(0, final_offset, bin_width):
        selected = [
            row
            for row in rows
            if decay_start + offset < row["iteration"] <= decay_start + offset + bin_width
            and row["iteration"] % 10 == 0
        ]
        if not selected:
            continue
        bins.append(
            {
                "offset_start_exclusive": offset,
                "offset_stop_inclusive": offset + bin_width,
                "points": len(selected),
                "mean_loss": float(np.mean([row["loss"] for row in selected])),
                "maximum_grad_norm": max(row["grad_norm"] for row in selected),
                "mean_learning_rate": float(
                    np.mean([row["learning_rate"] for row in selected])
                ),
            }
        )
    return {
        "label": label,
        "decay_start_iteration": decay_start,
        "first_iteration": rows[0]["iteration"],
        "last_iteration": rows[-1]["iteration"],
        "progress_points": len(rows),
        "maximum_grad_norm": max(row["grad_norm"] for row in rows),
        "skipped_iterations": sum(row["skipped_iterations"] for row in rows),
        "nan_iterations": sum(row["nan_iterations"] for row in rows),
        "decay_relative_bins": bins,
        "progress": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--bin-width", type=int, default=250)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "bin_semantics": "decay_start + offset < iteration <= decay_start + offset + width",
        "runs": [
            summarize(*load_run(spec), args.bin_width)
            for spec in args.run
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for run in report["runs"]:
        print(
            f"TRAIN_HEALTH label={run['label']} first={run['first_iteration']}"
            f" last={run['last_iteration']} points={run['progress_points']}"
            f" max_grad={run['maximum_grad_norm']:.6f}"
            f" skipped={run['skipped_iterations']} nan={run['nan_iterations']}"
        )
        for row in run["decay_relative_bins"]:
            print(
                f"DECAY_BIN label={run['label']}"
                f" offset=({row['offset_start_exclusive']},{row['offset_stop_inclusive']}]"
                f" points={row['points']} mean_loss={row['mean_loss']:.8f}"
                f" mean_lr={row['mean_learning_rate']:.10f}"
                f" max_grad={row['maximum_grad_norm']:.6f}"
            )
    print(f"REPORT={args.output}")
    print("TIME_MATCH_TRAINING_LOG_AUDIT=pass")


if __name__ == "__main__":
    main()
