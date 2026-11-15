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
import hashlib
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


def _load_per_example(result_path, task, expected_sha256):
    """Per-example rows for one task, refused unless they hash to what the record claims."""
    artifact = Path(result_path).parent / "downstream" / f"{task}.jsonl"
    if not artifact.exists():
        return None
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected_sha256:
        return None
    rows = {}
    for line in artifact.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["doc_id"]] = row["metrics"]
    return rows


def _bootstrap_aggregates(values, index, metric):
    import numpy as np

    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        means = values[index].mean(axis=1)
        return np.exp(-means) if metric == "perplexity" else means
    if values.ndim != 2 or values.shape[1] != 2:
        return None
    weighted_means = (
        values[index, 0].sum(axis=1) / values[index, 1].sum(axis=1)
    )
    if metric in ("word_perplexity", "byte_perplexity"):
        return np.exp(-weighted_means)
    if metric == "bits_per_byte":
        return -weighted_means / math.log(2)
    return None


def _paired_bootstrap(baseline_values, treatment_values, higher_is_better,
                      metric=None, iterations=2000, seed=1234):
    """One-sided 95% bounds on relative degradation, resampling whole documents.

    The pairing key is the document (`pair_key: task_and_doc_id`), so baseline and
    treatment are resampled through the same index and their per-document correlation
    stays intact -- which is the whole point of scoring both arms on the same examples.
    """
    import numpy as np

    if len(baseline_values) < 2:
        return None
    generator = np.random.default_rng(seed)
    index = generator.integers(
        0, len(baseline_values), size=(iterations, len(baseline_values))
    )
    baseline_means = _bootstrap_aggregates(baseline_values, index, metric)
    treatment_means = _bootstrap_aggregates(treatment_values, index, metric)
    if baseline_means is None or treatment_means is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        if higher_is_better:
            degradation = (baseline_means - treatment_means) / np.abs(baseline_means)
        else:
            degradation = (treatment_means - baseline_means) / np.abs(baseline_means)
    degradation = degradation[np.isfinite(degradation)]
    if degradation.size == 0:
        return None
    # Not clamped at zero: an improvement is a real outcome and the gate reads both bounds.
    return [float(np.percentile(degradation, 5)), float(np.percentile(degradation, 95))]


def mcnemar(baseline_values, treatment_values):
    """Exact two-sided McNemar p-value for paired 0/1 outcomes, plus the discordant counts."""
    only_baseline = sum(1 for b, t in zip(baseline_values, treatment_values) if b > t)
    only_treatment = sum(1 for b, t in zip(baseline_values, treatment_values) if t > b)
    discordant = only_baseline + only_treatment
    if discordant == 0:
        return 1.0, only_baseline, only_treatment
    tail = sum(math.comb(discordant, k) for k in range(min(only_baseline, only_treatment) + 1))
    return min(1.0, 2 * tail / 2 ** discordant), only_baseline, only_treatment


def downstream_intervals(replicates):
    """Paired per-example intervals for every (task, metric) the records report.

    This one is not an interval across replicates: the comparison is paired by document,
    so one replicate carrying per-example artifacts is enough. The first replicate that
    has readable, hash-matching artifacts for a metric supplies it. Returns an empty list
    unless every reported metric could be covered -- `_downstream_effects` refuses a
    baseline/treatment key set that does not match exactly, so a partial block would
    abort the whole comparison rather than weaken it.
    """
    intervals = {}
    secondary = {}
    for baseline_path, treatment_path, baseline, treatment in replicates:
        reported = {
            (item["task"], item["metric"]): item
            for item in treatment["measurement"]["downstream"]
        }
        baseline_reported = {
            (item["task"], item["metric"]): item
            for item in baseline["measurement"]["downstream"]
        }
        for key, treatment_item in sorted(reported.items()):
            if key in intervals or key not in baseline_reported:
                continue
            task, metric = key
            baseline_item = baseline_reported[key]
            baseline_rows = _load_per_example(
                baseline_path, task, baseline_item["per_example_artifact_sha256"]
            )
            treatment_rows = _load_per_example(
                treatment_path, task, treatment_item["per_example_artifact_sha256"]
            )
            if not baseline_rows or not treatment_rows:
                continue
            shared = sorted(set(baseline_rows) & set(treatment_rows))
            paired = [
                (baseline_rows[doc][metric], treatment_rows[doc][metric])
                for doc in shared
                if metric in baseline_rows[doc] and metric in treatment_rows[doc]
            ]
            if len(paired) < 2:
                continue
            baseline_values = [value for value, _ in paired]
            treatment_values = [value for _, value in paired]
            interval = _paired_bootstrap(
                baseline_values,
                treatment_values,
                treatment_item["higher_is_better"],
                metric=metric,
            )
            if interval is None:
                continue
            intervals[key] = interval
            if all(value in (0.0, 1.0) for value in baseline_values) and all(
                value in (0.0, 1.0) for value in treatment_values
            ):
                secondary[key] = mcnemar(baseline_values, treatment_values)
    expected = {
        (item["task"], item["metric"])
        for _, _, _, treatment in replicates
        for item in treatment["measurement"]["downstream"]
    }
    if not expected or set(intervals) != expected:
        return [], secondary
    block = [
        {"task": task, "metric": metric, "degradation_ci95": intervals[(task, metric)]}
        for task, metric in sorted(intervals)
    ]
    return block, secondary


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
    downstream, _ = downstream_intervals(replicates)
    memory, wct, degradation = [], [], []
    for _, _, baseline, treatment in replicates:
        base_measure, treat_measure = baseline["measurement"], treatment["measurement"]
        base_memory = base_measure["memory"]["max_allocated_bytes"]
        treat_memory = treat_measure["memory"]["max_allocated_bytes"]
        memory.append(base_memory / treat_memory if base_memory and treat_memory else None)
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
        "downstream": downstream,
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
        _, secondary = downstream_intervals(replicates)
        for item in summary["downstream"]:
            key = (item["task"], item["metric"])
            low, high = item["degradation_ci95"]
            line = f"  downstream {item['task']}/{item['metric']} = [{low:+.4f}, {high:+.4f}]"
            if key in secondary:
                probability, only_baseline, only_treatment = secondary[key]
                line += f"  mcnemar p={probability:.4f} (b={only_baseline}, c={only_treatment})"
            print(line)
        if not summary["downstream"]:
            print("  downstream: no paired per-example artifacts, gate stays inconclusive")
        if not args.write:
            continue
        for _, treatment_path, baseline, treatment in replicates:
            block = dict(summary, baseline_run_id=baseline["run_id"])
            treatment["measurement"]["inference"] = block
            target = Path(treatment_path)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_text(json.dumps(treatment, sort_keys=True) + "\n")
            temporary.replace(target)
        print(f"  wrote inference into {len(replicates)} treatment records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
