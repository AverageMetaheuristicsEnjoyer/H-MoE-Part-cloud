"""Build the one-sided paired 95% intervals the Stage 3 gates read.

`pair_results` never looks at a point estimate: the memory and WCT gates read
`measurement.inference.*_ci95` on the treatment record, which nothing populated.
This module discovers matched replicates, computes the intervals across them, and
writes the block back.

A replicate is one (baseline, treatment) pair measured under identical conditions.
`compare_runs` already refuses pairs that differ in GPU, host, image, revisions,
config, denominators or protocol, so any ordered pair it accepts is a valid
replicate and pairs from different jobs cannot be mixed up.

Ratios are multiplicative, so the interval is built on log(ratio) with a Student-t
quantile and transformed back; that keeps it positive and correctly asymmetric.
Both bounds are one-sided 95% bounds, matching how the gate uses them: the lower
one to declare a pass, the upper one to declare a fail.
"""

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from stage3_moe.pair_results import PAIR_ARMS, compare_runs, load_run

# Student-t, one-sided 95%, by degrees of freedom; the normal limit beyond the table.
_T_ONE_SIDED_95 = {
    1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895,
    8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761,
    15: 1.753, 16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729, 20: 1.725,
    21: 1.721, 22: 1.717, 23: 1.714, 24: 1.711, 25: 1.708, 26: 1.706,
    27: 1.703, 28: 1.701, 29: 1.699, 30: 1.697,
}


def t_quantile(degrees_of_freedom):
    if degrees_of_freedom in _T_ONE_SIDED_95:
        return _T_ONE_SIDED_95[degrees_of_freedom]
    return 1.645


def log_ratio_interval(ratios):
    """One-sided 95% bounds for the mean ratio, computed on the log scale."""
    usable = [r for r in ratios if r is not None and r > 0 and math.isfinite(r)]
    if len(usable) < 2:
        return None
    logs = [math.log(r) for r in usable]
    mean = statistics.mean(logs)
    spread = statistics.stdev(logs)
    margin = t_quantile(len(usable) - 1) * spread / math.sqrt(len(usable))
    return [math.exp(mean - margin), math.exp(mean + margin)]


def _degradation(baseline, treatment):
    if baseline is None or treatment is None or baseline == 0:
        return None
    return max(0.0, (treatment - baseline) / abs(baseline))


def _interval_for_degradations(values):
    usable = [v for v in values if v is not None and math.isfinite(v)]
    if len(usable) < 2:
        return None
    mean = statistics.mean(usable)
    margin = t_quantile(len(usable) - 1) * statistics.stdev(usable) / math.sqrt(len(usable))
    return [mean - margin, mean + margin]


def discover_replicates(paths):
    """Return {(axis, optimizer): [(baseline_path, treatment_path, baseline, treatment)]}."""
    runs = {}
    for path in paths:
        try:
            runs[path] = load_run(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    groups = defaultdict(list)
    for baseline_path, baseline in runs.items():
        for treatment_path, treatment in runs.items():
            if baseline_path == treatment_path:
                continue
            arms = (baseline["arm_id"], treatment["arm_id"])
            keys = [key for key, value in PAIR_ARMS.items() if value == arms]
            if not keys:
                continue
            try:
                compare_runs(baseline, treatment)
            except ValueError:
                continue
            groups[keys[0]].append((baseline_path, treatment_path, baseline, treatment))
    return groups


def build_inference(replicates):
    memory, wct, degradation = [], [], []
    for _, _, baseline, treatment in replicates:
        base_measure, treat_measure = baseline["measurement"], treatment["measurement"]
        memory.append(
            base_measure["memory"]["max_allocated_bytes"]
            / treat_measure["memory"]["max_allocated_bytes"]
        )
        base_wct = base_measure["timing"]["e2e_wct_seconds"]
        treat_wct = treat_measure["timing"]["e2e_wct_seconds"]
        wct.append(base_wct / treat_wct if base_wct and treat_wct else None)
        degradation.append(
            _degradation(base_measure["loss"]["validation"], treat_measure["loss"]["validation"])
        )
    return {
        "method": (
            f"paired log-ratio Student-t bound over {len(replicates)} matched replicates"
        ),
        "confidence_level": 0.95,
        "sidedness": "one-sided",
        "memory_allocated_ratio_ci95": log_ratio_interval(memory),
        "e2e_wct_ratio_ci95": log_ratio_interval(wct),
        "validation_loss_degradation_ci95": _interval_for_degradations(degradation),
        "downstream": [],
        "_ratios": {"memory_allocated": memory, "e2e_wct": wct},
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="directory holding <run_id>/results.jsonl")
    parser.add_argument("--filter", default="", help="substring the run directory must contain")
    parser.add_argument("--write", action="store_true", help="persist the inference block")
    args = parser.parse_args(argv)

    paths = sorted(
        p for p in Path(args.root).glob("*/results.jsonl") if args.filter in str(p)
    )
    print(f"scanned {len(paths)} result files matching {args.filter!r}")
    groups = discover_replicates(paths)
    if not groups:
        print("no valid replicates found")
        return 1

    for (axis, optimizer), replicates in sorted(groups.items()):
        summary = build_inference(replicates)
        ratios = summary.pop("_ratios")
        print(f"\naxis={axis} optimizer={optimizer} replicates={len(replicates)}")
        for _, treatment_path, baseline, _ in replicates:
            print(f"  {baseline['run_id']}  ->  {Path(treatment_path).parent.name}")
        for name, values in ratios.items():
            shown = ", ".join("null" if v is None else f"{v:.4f}" for v in values)
            print(f"  {name} ratios: {shown}")
        for name in ("memory_allocated_ratio_ci95", "e2e_wct_ratio_ci95"):
            interval = summary[name]
            text = "null (needs >=2 replicates)" if interval is None else (
                f"[{interval[0]:.4f}, {interval[1]:.4f}]"
            )
            print(f"  {name} = {text}")
        if not args.write:
            continue
        for _, treatment_path, baseline, treatment in replicates:
            block = dict(summary, baseline_run_id=baseline["run_id"])
            treatment["measurement"]["inference"] = block
            Path(treatment_path).write_text(json.dumps(treatment, sort_keys=True) + "\n")
        print(f"  wrote inference into {len(replicates)} treatment records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
