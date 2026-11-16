import base64
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

OUTPUT = Path(os.environ["HMOE_CONTRACT_ROOT"])
OUTPUT.mkdir(parents=True, exist_ok=True)
SOURCE = Path("/workspace-SR006.nfs3/hmoe-lionmoe-20261116/measurement-008-v010")
CORRECTED = Path("/workspace-SR006.nfs3/hmoe-lionmoe-20261116/measurement-008-v011")
EXPECTED_NOTE = "hmoe-lionmoe-20261116-analysis-009-v012"
EXPECTED_ROOT = "/workspace-SR006.nfs3/hmoe-lionmoe-20261116/analysis-009-v012"
EXPECTED_BUNDLE = "hmoe-lionmoe-20261116-v012"
if os.environ.get("HMOE_JOB_NOTE") != EXPECTED_NOTE:
    raise RuntimeError("Incorrect analysis009 note")
if OUTPUT != Path(EXPECTED_ROOT):
    raise RuntimeError("Incorrect analysis009 root")
if os.environ.get("EXPECTED_BUNDLE_ID") != EXPECTED_BUNDLE:
    raise RuntimeError("Incorrect analysis009 bundle")
for key, length in (("EXPECTED_SOURCE_COMMIT", 40), ("EXPECTED_CARRIER_COMMIT", 40), ("EXPECTED_PAYLOAD_SHA256", 64)):
    value = os.environ.get(key, "")
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"Missing/invalid runtime pin: {key}")
STARTED = datetime.datetime.now(datetime.timezone.utc)
ARMS = ("lm5", "lm5r50", "lm5r50ec")
STEPS = (1, 6, 11, 16)
TEMPORAL = ("step_1", "step_6", "step_11", "step_16", "excluding_step_1")
NORM_FIELDS = ("grad_norm", "direction_norm", "old_momentum_norm", "pre_update_norm",
               "post_update_norm", "update_over_grad_alignment")
ABS_TOLERANCE = 1e-12
REL_TOLERANCE = 1e-12
Q_ABS_TOLERANCE = 1e-6
Q_REL_TOLERANCE = 1e-6


def write_json(path, value, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        if compact:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        else:
            json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def quantile(values, fraction):
    values = sorted(values)
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def distribution(records, field):
    values = [record[field] for record in records if isinstance(record.get(field), (int, float))
              and not isinstance(record.get(field), bool) and math.isfinite(record[field])]
    if not values:
        return None
    return {"min": min(values), "p10": quantile(values, 0.1), "p50": quantile(values, 0.5),
            "p90": quantile(values, 0.9), "max": max(values)}


def summary_group(records):
    hypothetical = [record["hypothetical_q"] for record in records
                    if isinstance(record.get("hypothetical_q"), (int, float)) and math.isfinite(record["hypothetical_q"])]
    actual = [record["actual_q"] for record in records
              if isinstance(record.get("actual_q"), (int, float)) and math.isfinite(record["actual_q"])]
    return {"samples": len(records), "rho": distribution(records, "rho"),
            "hypothetical_q": distribution(records, "hypothetical_q"), "actual_q": distribution(records, "actual_q"),
            "fraction_hypothetical_q_lt_0_99": sum(value < 0.99 for value in hypothetical) / len(hypothetical) if hypothetical else None,
            "fraction_hypothetical_q_lt_0_9": sum(value < 0.9 for value in hypothetical) / len(hypothetical) if hypothetical else None,
            "fraction_hypothetical_q_lt_0_5": sum(value < 0.5 for value in hypothetical) / len(hypothetical) if hypothetical else None,
            "fraction_hypothetical_q_zero": sum(value == 0 for value in hypothetical) / len(hypothetical) if hypothetical else None,
            "actual_q_samples": len(actual), "nonfinite": sum(record["nonfinite"] for record in records)}


def phase_summary(values, phases, wanted):
    selected = [value for value, phase in zip(values, phases) if phase == wanted]
    return {"count": len(selected), "sum_seconds": sum(selected),
            "mean_seconds": statistics.mean(selected) if selected else None,
            "median_seconds": statistics.median(selected) if selected else None,
            "min_seconds": min(selected) if selected else None,
            "max_seconds": max(selected) if selected else None}


def temporal_name(iteration):
    return f"step_{iteration}"


def selected_records(records, temporal):
    if temporal == "excluding_step_1":
        return [record for record in records if record["iteration"] in (6, 11, 16)]
    wanted = int(temporal.removeprefix("step_"))
    return [record for record in records if record["iteration"] == wanted]


def numeric_summary(records, field):
    present = [record[field] for record in records if isinstance(record.get(field), (int, float))
               and not isinstance(record.get(field), bool)]
    nonfinite = sum(not math.isfinite(value) for value in present)
    values = [value for value in present if math.isfinite(value)]
    return {"n": len(values), "null": len(records) - len(present), "nonfinite": nonfinite,
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def ratio_record(record, numerator, denominator):
    top, bottom = record.get(numerator), record.get(denominator)
    if top is None or bottom is None:
        return None
    if bottom == 0:
        return None
    return top / bottom


def pearson(records, x_field, y_field):
    pairs = [(record.get(x_field), record.get(y_field)) for record in records]
    pairs = [(x, y) for x, y in pairs if isinstance(x, (int, float)) and isinstance(y, (int, float))
             and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return {"n": len(pairs), "r": None, "reason": "fewer_than_two_finite_pairs"}
    xs, ys = zip(*pairs)
    x_mean, y_mean = statistics.mean(xs), statistics.mean(ys)
    xx = sum((x - x_mean) ** 2 for x in xs)
    yy = sum((y - y_mean) ** 2 for y in ys)
    if xx == 0 or yy == 0:
        return {"n": len(pairs), "r": None, "reason": "zero_variance"}
    return {"n": len(pairs), "r": sum((x - x_mean) * (y - y_mean) for x, y in pairs) / math.sqrt(xx * yy),
            "reason": None}


def compare_json(actual, expected, path=""):
    differences = []
    if isinstance(expected, dict):
        for key, value in expected.items():
            differences.extend(compare_json(actual.get(key) if isinstance(actual, dict) else None, value, f"{path}.{key}"))
    elif isinstance(expected, float):
        if actual is None or not math.isclose(actual, expected, abs_tol=ABS_TOLERANCE, rel_tol=REL_TOLERANCE):
            differences.append({"path": path, "actual": actual, "expected": expected})
    elif actual != expected:
        differences.append({"path": path, "actual": actual, "expected": expected})
    return differences


def q_profile(records, arm):
    q_field = "actual_q" if arm == "lm5r50ec" else "hypothetical_q"
    q_values = [record[q_field] for record in records if record.get(q_field) is not None]
    return {"samples": len(records), "q_samples": len(q_values), "q_field": q_field,
            "rho": distribution(records, "rho"), "q": distribution(records, q_field),
            "q_mean": statistics.mean(q_values) if q_values else None,
            "q_median": statistics.median(q_values) if q_values else None,
            "fraction_q_lt_0_99": sum(value < 0.99 for value in q_values) / len(q_values) if q_values else None,
            "fraction_q_lt_0_9": sum(value < 0.9 for value in q_values) / len(q_values) if q_values else None,
            "fraction_q_lt_0_5": sum(value < 0.5 for value in q_values) / len(q_values) if q_values else None,
            "fraction_q_zero": sum(value == 0 for value in q_values) / len(q_values) if q_values else None,
            "nonfinite": sum(record["nonfinite"] for record in records)}


def norm_profile(records):
    result = {field: numeric_summary(records, field) for field in NORM_FIELDS}
    for name, numerator, denominator in (("post_over_pre", "post_update_norm", "pre_update_norm"),
                                         ("pre_over_grad", "pre_update_norm", "grad_norm"),
                                         ("direction_over_grad", "direction_norm", "grad_norm"),
                                         ("old_momentum_over_grad", "old_momentum_norm", "grad_norm")):
        values = [ratio_record(record, numerator, denominator) for record in records]
        result[name] = numeric_summary([{"value": value} for value in values], "value")
        result[name]["zero_or_missing_denominator"] = sum(record.get(denominator) in (None, 0) for record in records)
    return result


def fixture_checks():
    fixture = [
        {"iteration": 1, "layer": 1, "expert": 0, "fc_type": "fc2", "routed": True,
         "rho": 0.4, "hypothetical_q": 0.5, "actual_q": 0.5, "pre_update_norm": 2.0,
         "post_update_norm": 1.0, "grad_norm": 2.0, "nonfinite": False},
        {"iteration": 6, "layer": 1, "expert": 0, "fc_type": "fc2", "routed": True,
         "rho": 0.8, "hypothetical_q": 1.0, "actual_q": 1.0, "pre_update_norm": 1.0,
         "post_update_norm": 1.0, "grad_norm": 0.0, "nonfinite": False},
    ]
    assert q_profile(selected_records(fixture, "step_1"), "lm5r50ec")["q"]["p50"] == 0.5
    assert q_profile(selected_records(fixture, "step_6"), "lm5r50ec")["q"]["p50"] == 1.0
    assert ratio_record(fixture[1], "post_update_norm", "grad_norm") is None
    assert norm_profile([fixture[1]])["pre_over_grad"]["n"] == 0
    values = list(range(1, 11))
    phases = ["spectral" if index % 5 == 0 else "sign" for index in range(10)]
    assert phase_summary(values, phases, "spectral")["sum_seconds"] == 7
    assert phase_summary(values, phases, "spectral")["count"] == 2
    assert phase_summary(values, phases, "sign")["sum_seconds"] == 48
    try:
        validate_coverage(fixture, "fixture", expected_per_step=1, require_controls=False)
    except RuntimeError as error:
        assert "Missing block keys" in str(error)
    else:
        raise AssertionError("Missing-record fixture was not rejected")
    duplicate = fixture + [dict(fixture[0])]
    try:
        validate_coverage(duplicate, "fixture", expected_per_step=1, require_controls=False)
    except RuntimeError as error:
        assert "Duplicate routed key" in str(error)
    else:
        raise AssertionError("Duplicate-record fixture was not rejected")
    return {"passed": True, "step_groups_differ": True, "phase_contents_differ": True,
            "zero_denominator_null": True, "missing_guard_executed": True, "duplicate_guard_executed": True}


def validate_coverage(records, arm, expected_per_step=31, require_controls=True):
    keys = [(record["iteration"], record["layer"], record["expert"], record["fc_type"]) for record in records]
    duplicate_count = len(keys) - len(set(keys))
    routed_keys = [key for key, record in zip(keys, records) if record["routed"]]
    if len(routed_keys) != len(set(routed_keys)):
        raise RuntimeError(f"Duplicate routed key in {arm}")
    missing = []
    for iteration in STEPS:
        rows = [record for record in records if record["iteration"] == iteration]
        routed = [record for record in rows if record["routed"]]
        controls = [record for record in rows if not record["routed"]]
        expected_routed = {(iteration, layer, expert, fc) for layer in (1, 9, 17) for expert in (0, 31, 63)
                           for fc in ("fc1_gate", "fc1_up", "fc2")}
        actual_routed = {(record["iteration"], record["layer"], record["expert"], record["fc_type"]) for record in routed}
        missing.extend(sorted(expected_routed - actual_routed))
        if arm != "fixture" and (len(rows) != expected_per_step or len(routed) != 27 or (require_controls and len(controls) != 4)):
            raise RuntimeError(f"Incorrect coverage in {arm} step{iteration}: {len(rows)}/{len(routed)}/{len(controls)}")
    if missing:
        raise RuntimeError(f"Missing block keys in {arm}: {missing[:5]}")
    return duplicate_count


def runtime_gate():
    usage = os.statvfs(OUTPUT)
    free_bytes = usage.f_bavail * usage.f_frsize
    if free_bytes < 64 * 1024 * 1024 or usage.f_favail < 100:
        raise RuntimeError("Insufficient small-analysis output space/inodes")
    path = OUTPUT / "runtime-roundtrip.bin"
    payload = b"hmoe-lionmoe-analysis-009\n"
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != payload:
        raise RuntimeError("Tiny readback mismatch")
    return {"free_bytes": free_bytes, "free_inodes": usage.f_favail, "tiny_roundtrip": receipt(path),
            "cuda_called": False, "model_constructed": False, "training_invoked": False}


def main():
    runtime = runtime_gate()
    fixture = fixture_checks()
    source = json.loads((SOURCE / "measurement-summary.json").read_text())
    corrected = json.loads((CORRECTED / "measurement-summary.json").read_text())
    if source["job_note"] != "hmoe-lionmoe-20261116-measurement-008-v010":
        raise RuntimeError("Unexpected source job")
    if not corrected["analysis_only_correction"] or corrected["corrective_job_training_invoked"]:
        raise RuntimeError("Canonical correction provenance mismatch")
    receipt_results = []
    for expected in source["raw_artifact_receipts"]:
        path = Path(expected["path"])
        path.relative_to(SOURCE)
        if not path.is_file():
            raise RuntimeError(f"Missing receipt artifact: {path}")
        actual = receipt(path)
        actual["match"] = actual["bytes"] == expected["bytes"] and actual["sha256"] == expected["sha256"]
        receipt_results.append(actual)
        if not actual["match"]:
            raise RuntimeError(f"Receipt mismatch: {path}")
    write_json(OUTPUT / "source-receipt-verification.json", receipt_results)
    arms = {}
    profiles = []
    norms = []
    pooled_comparisons = []
    candidate_q_checks = {"records": 0, "max_abs_actual_hypothetical": 0.0,
                          "max_abs_actual_post_pre": 0.0, "mismatches": 0}
    for arm in ARMS:
        original_arm = next(item for item in source["arms"] if item["preset"] == arm)
        corrected_arm = next(item for item in corrected["arms"] if item["preset"] == arm)
        analysis_path = SOURCE / "arms" / arm / "analysis.json"
        expected_analysis_sha = corrected["source_samples"]["analysis_sha256"][arm]["sha256"]
        if sha256(analysis_path) != expected_analysis_sha:
            raise RuntimeError(f"Analysis source hash mismatch: {arm}")
        directory = Path(original_arm["result_path"]).parent
        blocks_path = directory / "energycal_blocks.jsonl"
        records = [json.loads(line) for line in blocks_path.read_text().splitlines() if line.strip()]
        if len(records) != 124:
            raise RuntimeError(f"Unexpected block record count: {arm}/{len(records)}")
        duplicate_count = validate_coverage(records, arm)
        diagnostics = [json.loads(line) for line in (directory / "energycal_diagnostics.jsonl").read_text().splitlines()]
        spectral_diagnostics = [row for row in diagnostics if row["spectral"]]
        if len(diagnostics) != 20 or [row["iteration"] for row in spectral_diagnostics] != list(STEPS):
            raise RuntimeError(f"Diagnostic iteration coverage mismatch: {arm}")
        if any(row["selected_records"] != 31 or row["missing_selected"] != 0 for row in spectral_diagnostics):
            raise RuntimeError(f"Per-step diagnostic coverage mismatch: {arm}")
        if any(row["selected_records"] != 0 for row in diagnostics if not row["spectral"]):
            raise RuntimeError(f"Unexpected sign-step block records: {arm}")
        saved_summary = json.loads((directory / "energycal_summary.json").read_text())
        recomputed_groups = {}
        for record in records:
            key = f"layer_{record['layer']:02d}/{record['fc_type']}"
            recomputed_groups.setdefault(key, []).append(record)
            recomputed_groups.setdefault("all_routed" if record["routed"] else "all_controls", []).append(record)
        differences = []
        for key, rows in recomputed_groups.items():
            differences.extend(compare_json(summary_group(rows), saved_summary["groups"][key], key))
        if differences:
            raise RuntimeError(f"Pooled summary mismatch: {arm}: {differences[:3]}")
        pooled_comparisons.append({"arm": arm, "groups": len(recomputed_groups), "differences": 0,
                                   "abs_tolerance": ABS_TOLERANCE, "rel_tolerance": REL_TOLERANCE})
        if arm == "lm5r50ec":
            for record in records:
                if record["routed"]:
                    candidate_q_checks["records"] += 1
                    if not record["applied"] or record["actual_q"] is None:
                        raise RuntimeError("Candidate routed record lacks applied actual q")
                    delta = abs(record["actual_q"] - record["hypothetical_q"])
                    candidate_q_checks["max_abs_actual_hypothetical"] = max(candidate_q_checks["max_abs_actual_hypothetical"], delta)
                    ratio = ratio_record(record, "post_update_norm", "pre_update_norm")
                    if ratio is None or not math.isclose(record["actual_q"], ratio, abs_tol=Q_ABS_TOLERANCE, rel_tol=Q_REL_TOLERANCE):
                        candidate_q_checks["mismatches"] += 1
                    candidate_q_checks["max_abs_actual_post_pre"] = max(candidate_q_checks["max_abs_actual_post_pre"],
                                                                         abs(record["actual_q"] - (ratio or 0.0)))
                    if not math.isclose(record["actual_q"], record["hypothetical_q"], abs_tol=Q_ABS_TOLERANCE, rel_tol=Q_REL_TOLERANCE):
                        candidate_q_checks["mismatches"] += 1
        elif any(record["actual_q"] is not None or record["applied"] for record in records):
            raise RuntimeError(f"Plain arm applied calibration: {arm}")
        routed = [record for record in records if record["routed"]]
        group_names = ["all_routed"] + [f"layer_{layer:02d}/{fc}" for layer in (1, 9, 17)
                                       for fc in ("fc1_gate", "fc1_up", "fc2")]
        for temporal in TEMPORAL:
            temporal_records = selected_records(routed, temporal)
            for group in group_names:
                selected = temporal_records if group == "all_routed" else [
                    record for record in temporal_records if f"layer_{record['layer']:02d}/{record['fc_type']}" == group]
                profiles.append({"arm": arm, "temporal": temporal, "group": group, **q_profile(selected, arm)})
            for activity in ("affected_q_lt_0_99", "unaffected_q_gte_0_99"):
                q_field = "actual_q" if arm == "lm5r50ec" else "hypothetical_q"
                selected = [record for record in temporal_records
                            if (record[q_field] < 0.99) == (activity == "affected_q_lt_0_99")]
                norms.append({"arm": arm, "temporal": temporal, "activity": activity,
                              "samples": len(selected), "metrics": norm_profile(selected)})
        arms[arm] = {"records": records, "analysis": corrected_arm, "directory": directory,
                     "duplicate_full_keys": duplicate_count, "diagnostics": diagnostics,
                     "saved_summary": saved_summary}
    if candidate_q_checks["mismatches"]:
        raise RuntimeError("Candidate q/norm consistency mismatch")
    trajectories = []
    candidate_records = [record for record in arms["lm5r50ec"]["records"] if record["routed"]]
    affected_keys = sorted({(record["layer"], record["expert"], record["fc_type"])
                            for record in candidate_records if record["iteration"] > 1 and record["actual_q"] < 0.99})
    affected_keys = sorted(affected_keys, key=lambda key: min(record["actual_q"] for record in candidate_records
                           if (record["layer"], record["expert"], record["fc_type"]) == key))[:8]
    for key in affected_keys:
        item = {"layer": key[0], "expert": key[1], "fc_type": key[2], "steps": []}
        for iteration in STEPS:
            step = {"iteration": iteration}
            for arm in ARMS:
                matches = [record for record in arms[arm]["records"] if record["routed"]
                           and (record["layer"], record["expert"], record["fc_type"], record["iteration"])
                           == (key[0], key[1], key[2], iteration)]
                if len(matches) != 1:
                    raise RuntimeError("Matched trajectory is not unique")
                record = matches[0]
                step[arm] = {"q": record["actual_q"] if arm == "lm5r50ec" else record["hypothetical_q"],
                             "rho": record["rho"], "pre_update_norm": record["pre_update_norm"],
                             "grad_norm": record["grad_norm"], "direction_norm": record["direction_norm"],
                             "old_momentum_norm": record["old_momentum_norm"],
                             "update_over_grad_alignment": record["update_over_grad_alignment"]}
            item["steps"].append(step)
        trajectories.append(item)
    correlations = []
    for temporal in TEMPORAL:
        records = selected_records(candidate_records, temporal)
        augmented = [dict(record, pre_over_grad=ratio_record(record, "pre_update_norm", "grad_norm"),
                          direction_over_grad=ratio_record(record, "direction_norm", "grad_norm")) for record in records]
        for field in ("grad_norm", "direction_norm", "old_momentum_norm", "pre_update_norm",
                      "update_over_grad_alignment", "pre_over_grad", "direction_over_grad"):
            correlations.append({"temporal": temporal, "field": field, **pearson(augmented, "actual_q", field)})
    routing = {}
    for arm in ARMS:
        directory = arms[arm]["directory"]
        token_data = json.loads((directory / "tokens_per_expert.json").read_text())
        telemetry = [json.loads(line) for line in (directory / "routing_telemetry.jsonl").read_text().splitlines()]
        selected_tokens = []
        for layer in (1, 9, 17):
            for expert in (0, 31, 63):
                selected_tokens.append({"layer": layer, "expert": expert,
                                        "accumulated_tokens": token_data["tokens_per_expert"][layer][expert]})
        routing[arm] = {"telemetry_iterations": [row["iteration"] for row in telemetry],
                        "selected_accumulated_tokens": selected_tokens,
                        "summary": arms[arm]["analysis"]["routing"],
                        "zero_experts_per_layer_accumulated": arms[arm]["analysis"]["zero_experts_per_layer"],
                        "exact_per_block_temporal_join": False,
                        "limitation": "Only iteration1 overlaps spectral1/6/11/16; telemetry has aggregate layer summaries, tokens are accumulated. No per-step expert causal join."}
    timing = []
    for arm in ARMS:
        analysis = arms[arm]["analysis"]
        values = analysis["timing"]["full_step_samples_seconds"]
        optimizer = analysis["timing"]["optimizer_step_samples_seconds"]
        iterations = list(range(41, 81))
        phases = ["spectral" if (iteration - 1) % 5 == 0 else "sign" for iteration in iterations]
        if len(values) != 40 or len(optimizer) != 40:
            raise RuntimeError("Timing sample length mismatch")
        phase_differences = []
        for name, samples in (("phase_full_step", values), ("phase_optimizer_step", optimizer)):
            for phase in ("spectral", "sign"):
                actual = phase_summary(samples, phases, phase)
                phase_differences.extend(compare_json(actual, analysis["timing"][name][phase], f"{name}.{phase}"))
        if phase_differences:
            raise RuntimeError(f"Canonical phase mismatch: {arm}: {phase_differences[:2]}")
        windows = []
        for start, end in ((41, 50), (51, 60), (61, 70), (71, 80), (41, 60), (61, 80)):
            selected = [(iteration, value, phase) for iteration, value, phase in zip(iterations, values, phases)
                        if start <= iteration <= end]
            window_values = [item[1] for item in selected]
            windows.append({"start": start, "end": end, "count": len(window_values),
                            "sum_seconds": sum(window_values), "mean_seconds": statistics.mean(window_values),
                            "median_seconds": statistics.median(window_values), "min_seconds": min(window_values),
                            "max_seconds": max(window_values),
                            "primary_tokens_per_second": len(window_values) * analysis["init_receipt"]["global_batch"] * 2048 / sum(window_values),
                            "spectral_mean_seconds": statistics.mean([item[1] for item in selected if item[2] == "spectral"]),
                            "sign_mean_seconds": statistics.mean([item[1] for item in selected if item[2] == "sign"])})
        maximum_index = max(range(len(values)), key=values.__getitem__)
        minimum_index = min(range(len(values)), key=values.__getitem__)
        timing.append({"arm": arm, "phase_verified_against_v011": True,
                       "phase_counts": {"spectral": phases.count("spectral"), "sign": phases.count("sign")},
                       "primary_tokens_per_second": analysis["timing"]["primary_tokens_per_second"],
                       "windows": windows, "max_sample": {"iteration": iterations[maximum_index], "seconds": values[maximum_index]},
                       "min_sample": {"iteration": iterations[minimum_index], "seconds": values[minimum_index]},
                       "max_over_median": values[maximum_index] / statistics.median(values),
                       "scope": "one ordered allocation; chronological sensitivity, not ABBA or speed acceptance"})
    persistence = [record for record in candidate_records if record["iteration"] > 1 and record["actual_q"] < 0.99]
    conditional_plan = {
        "submit_now": False, "condition_met_persistent_attenuation": bool(persistence),
        "type": "from-scratch checkpoint-free post-warmup comparison; not a warm switch",
        "arms": ["lm5r50", "lm5r50ec"], "original_wsd": {"decay": 17242, "warmup": 173, "wsd_decay": 3448},
        "proposed_window": "train through warmup without observer; diagnostics on steps181–200; clean timing201–240 (40 steps/eight P5 cycles); stop250",
        "per_arm_train_steps": 250, "allocation": "te4/1GPU, both arms in one bounded allocation if admitted",
        "estimated_payload_budget_seconds": 6000, "persistence": "small logs/receipts only; no weights; fresh space/inode gates",
        "stop_gates": ["any failed init/data match", "nonfinite/NaN/skips", "observer missing coverage", "hard payload cap", "no checkpoint/save/load"],
        "timing": "primary total loss tokens/sum actual full steps; report spectral/sign and chronological drift separately; not ABBA",
        "fixed_scalar_control": "future separate implementation/run only; one fixed routed spectral multiplier chosen from training-only diagnostics, unchanged other branches/WD; no test tuning or sweep; matching mean q does not match update dynamics"
    }
    result = {
        "schema": "hmoe-lionmoe-analysis009-v1", "job_note": EXPECTED_NOTE,
        "identity": {key: os.environ[key] for key in ("EXPECTED_SOURCE_COMMIT", "EXPECTED_CARRIER_COMMIT",
                                                       "EXPECTED_BUNDLE_ID", "EXPECTED_PAYLOAD_SHA256")},
        "source": {"root": str(SOURCE), "job": "lm-mpi-job-c0353d15-09fd-4b5d-bd45-11a449f3e34e",
                   "canonical_phase_root": str(CORRECTED), "receipts_verified": len(receipt_results),
                   "receipt_mismatches": 0, "receipt_detail_path": str(OUTPUT / "source-receipt-verification.json")},
        "runtime": runtime, "fixture": fixture, "passed": True,
        "coverage": {arm: {"block_records": len(arms[arm]["records"]), "routed_per_spectral_step": 27,
                           "controls_per_spectral_step": 4, "routed_total": 108,
                           "routed_duplicate_keys": 0, "full_key_duplicate_records": arms[arm]["duplicate_full_keys"],
                           "nonfinite_records": sum(record["nonfinite"] for record in arms[arm]["records"]),
                           "limitation": "Two dense FC1 control halves share the requested key; routed keys are unique and primary analysis is unaffected."}
                     for arm in ARMS},
        "pooled_summary_comparisons": pooled_comparisons, "candidate_q_consistency": candidate_q_checks,
        "profiles": profiles, "norm_alignment_profiles": norms, "matched_trajectories": trajectories,
        "descriptive_correlations": correlations, "routing": routing, "timing": timing,
        "semantics": {"positive_scalar_preserves_update_gradient_angle": True,
                      "alignment_scope": "cosine of observed update direction with gradient; not EnergyCal angular improvement",
                      "sampling": "selected logical blocks; repeated time/layer observations are dependent, not independent replicates",
                      "quality_speed_memory_benefit_established": False},
        "persistent_attenuation_samples_excluding_step1": len(persistence),
        "conditional_next_plan": conditional_plan,
        "started_at": STARTED.isoformat(), "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "duration_seconds": round((datetime.datetime.now(datetime.timezone.utc) - STARTED).total_seconds(), 3)
    }
    write_json(OUTPUT / "analysis.json", result)
    compact_path = OUTPUT / "analysis-summary.json"
    write_json(compact_path, result, compact=True)
    data = compact_path.read_bytes()
    sealed = {"schema": "hmoe-lionmoe-analysis009-small-artifacts-v1", "nonsecret": True,
              "artifacts": [{"label": "analysis-summary", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                             "content_base64": base64.b64encode(data).decode()}]}
    sealed_path = OUTPUT / "small-artifacts.json"
    write_json(sealed_path, sealed, compact=True)
    sealed_bytes = sealed_path.read_bytes()
    if len(sealed_bytes) > 256 * 1024:
        raise RuntimeError(f"Sealed analysis exceeds256KiB: {len(sealed_bytes)}")
    encoded = base64.b64encode(sealed_bytes).decode()
    print(f"ANALYSIS009_ARTIFACT path={sealed_path} bytes={len(sealed_bytes)} "
          f"sha256={hashlib.sha256(sealed_bytes).hexdigest()} base64_bytes={len(encoded)}", flush=True)
    chunks = [encoded[index:index + 12000] for index in range(0, len(encoded), 12000)]
    for index, chunk in enumerate(chunks):
        print(f"ANALYSIS009_B64 {index}/{len(chunks)} {chunk}", flush=True)
    print("ANALYSIS009_B64_END", flush=True)
    print("ANALYSIS009_RESULT " + json.dumps({"passed": True, "receipts": len(receipt_results),
          "persistent_attenuation_samples": len(persistence), "sealed_bytes": len(sealed_bytes)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
