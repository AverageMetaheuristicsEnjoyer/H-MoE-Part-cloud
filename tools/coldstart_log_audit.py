#!/usr/bin/env python3
"""Parse raw per-run train logs, including v3 summaries with mixed val/test labels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

NUMBER = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf))"
STEP = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)\s*\|")
EVAL = re.compile(r"validation.*?loss at iteration\s+(\d+)(?: on (validation|test) set)?\s*\| lm loss value:\s*(" + NUMBER + r")", re.I)
FIELDS = {
    "loss": r"lm loss:", "grad_norm": r"grad norm:", "learning_rate": r"learning rate:",
    "elapsed_ms": r"elapsed time per iteration \(ms\):", "consumed_samples": r"consumed samples:",
    "skipped": r"number of skipped iterations:", "nans": r"number of nan iterations:",
}
FIELD_RE = {k: re.compile(v + r"\s*(" + NUMBER + r")", re.I) for k, v in FIELDS.items()}


def read_jsonl(path):
    if not path.is_file():
        return [], 0
    rows, incomplete = [], 0
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            incomplete += 1
    return rows, incomplete


def parse_log(path, base, expected, require_evaluation):
    text = path.read_text(errors="replace")
    steps, evaluations, duplicate_conflicts = {}, [], []
    for line in text.splitlines():
        step = STEP.search(line)
        if step and "lm loss:" in line:
            row = {"iteration": int(step[1]), "target_iteration": int(step[2])}
            for key, pattern in FIELD_RE.items():
                match = pattern.search(line)
                row[key] = float(match[1]) if match else None
            old = steps.get(row["iteration"])
            if old is not None and old != row:
                duplicate_conflicts.append(row["iteration"])
            steps[row["iteration"]] = row
        evaluation = EVAL.search(line)
        if evaluation:
            row = {"iteration": int(evaluation[1]), "split": evaluation[2] or "validation",
                   "final": evaluation[2] is not None, "loss": float(evaluation[3])}
            if row not in evaluations:
                evaluations.append(row)
    steps = sorted(steps.values(), key=lambda row: row["iteration"])
    evaluations.sort(key=lambda row: (row["iteration"], row["split"], row["final"]))
    endpoint = base + expected
    final = {split: next((row for row in reversed(evaluations)
                         if row["split"] == split and row["final"] and row["iteration"] == endpoint), None)
             for split in ("validation", "test")}
    nonfinite = [{"iteration": row["iteration"], "field": key} for row in steps
                 for key in ("loss", "grad_norm") if row[key] is not None and not math.isfinite(row[key])]
    skipped = max((row["skipped"] or 0 for row in steps), default=0)
    nans = max((row["nans"] or 0 for row in steps), default=0)
    complete_steps = [row["iteration"] for row in steps] == list(range(base + 1, endpoint + 1))
    finite_eval = all(row is not None and math.isfinite(row["loss"]) for row in final.values())
    healthy = complete_steps and not duplicate_conflicts and not nonfinite and skipped == nans == 0
    if require_evaluation:
        healthy = healthy and finite_eval
    routing, routing_parse_errors = read_jsonl(path.parent / "routing_telemetry.jsonl")
    routing = sorted(routing, key=lambda row: row["iteration"])
    records, record_parse_errors = read_jsonl(path.parent / "results.jsonl")
    run_records = [row for row in records if row.get("record_type") == "run"]
    record = run_records[-1] if run_records else {}
    manifests = sorted(path.parent.parent.glob("manifest-*.json"))
    manifest = json.loads(manifests[0].read_text()) if len(manifests) == 1 else None
    norms = [row["grad_norm"] for row in steps if row["grad_norm"] is not None and math.isfinite(row["grad_norm"])]
    losses = [row["loss"] for row in steps[-100:] if row["loss"] is not None and math.isfinite(row["loss"])]
    summary = {
        "raw_log": str(path), "raw_log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "run_id": path.parent.name, "step_count": len(steps), "first_iteration": steps[0]["iteration"] if steps else None,
        "last_iteration": steps[-1]["iteration"] if steps else None, "expected_endpoint": endpoint,
        "complete_contiguous_steps": complete_steps, "numerical_and_endpoint_checks_pass": healthy,
        "trainer_exit_verified": False,
        "completion_note": "Raw train log alone does not establish wrapper/process exit status; combine with VARIANT_EXIT and BRANCH_VALIDATED.",
        "skipped_max": skipped, "nans_max": nans, "nonfinite": nonfinite,
        "duplicate_step_conflicts": sorted(set(duplicate_conflicts)), "final": final,
        "train_tail100_mean": statistics.mean(losses) if losses else None,
        "grad_norm_median": statistics.median(norms) if norms else None,
        "grad_norm_max": max(norms) if norms else None,
        "grad_norm_gt_0p3": sum(x > .3 for x in norms), "grad_norm_gt_1": sum(x > 1 for x in norms),
        "steps": steps, "evaluations": evaluations,
        "routing": {"records": len(routing), "parse_errors": routing_parse_errors,
                    "first_iteration": routing[0]["iteration"] if routing else None,
                    "last": routing[-1] if routing else None, "artifact": str(path.parent / "routing_telemetry.jsonl")},
        "run_record": {"parse_errors": record_parse_errors, "denominators": record.get("denominators"),
                       "provenance": record.get("provenance"), "environment": record.get("environment"),
                       "measurement_protocol": record.get("measurement", {}).get("protocol")},
        "manifest": manifest,
        "memory_claim": "No steady-state memory claim extracted. v3 result record measures the first step only.",
        "timing_claim": "Per-step elapsed_ms retained as raw data; this output makes no paired speedup claim.",
    }
    return summary


def matching_audit(runs):
    groups = {}
    for run in runs:
        manifest = run["manifest"]
        if not manifest:
            groups.setdefault("missing_manifest", []).append({"run_id": run["run_id"], "fingerprint": None})
            continue
        env = manifest["environment"]
        fingerprint = {"optimizer": manifest["optimizer"], "checkpoint": manifest["checkpoint"],
                       "git_commit": manifest["git_commit"],
                       "source_sha256": manifest["source_sha256"], "packages": manifest["packages"],
                       "protocol": manifest["protocol"],
                       "environment": {k: v for k, v in env.items() if k != "STAGE3_MOE_DATA_CACHE_PATH"}}
        digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        groups.setdefault(manifest["optimizer"], []).append({"run_id": run["run_id"], "fingerprint": digest})
    return {key: {"consistent_recorded_protocol": all(row["fingerprint"] for row in rows)
                   and len({row["fingerprint"] for row in rows}) == 1, "runs": rows,
                   "scope": "Metadata/checkpoint/source consistency, not proof of tensorwise deterministic execution or independent seeds."}
            for key, rows in groups.items()}


def clean_nonfinite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: clean_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_nonfinite(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Raw train log files or roots containing per-run train-*.log")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-iteration", type=int, default=13794)
    parser.add_argument("--expected-steps", type=int, default=400)
    parser.add_argument("--allow-no-evaluation", action="store_true")
    args = parser.parse_args()
    paths = sorted({p.resolve() for root in args.paths for p in ([root] if root.is_file() else root.rglob("train-*.log"))})
    runs = [parse_log(path, args.base_iteration, args.expected_steps, not args.allow_no_evaluation) for path in paths]
    result = {"schema_version": 1, "runs": runs, "matching_audit": matching_audit(runs)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(clean_nonfinite(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "runs": len(runs),
                      "endpoint_checks_pass": sum(row["numerical_and_endpoint_checks_pass"] for row in runs)}))


if __name__ == "__main__":
    main()
