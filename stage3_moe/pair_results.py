import argparse
import json
import math
import sys
from pathlib import Path


PAIR_ARMS = {
    ("optimizer_state", "adamw"): (
        "adamw_bf16_state_fp32",
        "adamw_bf16_state_fp8",
    ),
    ("optimizer_state", "muon"): (
        "muon_bf16_state_fp32",
        "muon_bf16_state_fp8",
    ),
    ("fp8_gemm", "adamw"): (
        "adamw_bf16_state_fp32",
        "adamw_fp8gemm_state_fp32",
    ),
    ("fp8_gemm", "muon"): (
        "muon_bf16_state_fp32",
        "muon_fp8gemm_state_fp32",
    ),
}


def load_run(path):
    records = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                record = json.loads(line)
                if record.get("record_type") == "run":
                    records.append(record)
    if len(records) != 1:
        raise ValueError(f"{path}: expected exactly one run record, found {len(records)}")
    return records[0]


def _require(mapping, keys, label):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{label}: missing {', '.join(missing)}")


def _positive_or_none(value, label):
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label}: expected a number or null")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label}: expected a finite positive number")


# Options whose value names where a run reads or writes, never what it computes. Their
# values are per-arm by construction -- each arm must load its own checkpoint and log under
# its own name -- so comparing them verbatim rejects every honest pair. Masking the arm id
# instead keeps the check strict: a path that differs by anything else still fails.
PER_ARM_VALUE_OPTIONS = (
    "--load",
    "--save",
    "--tensorboard-dir",
    "--wandb-exp-name",
    "--wandb-save-dir",
)

ARM_PLACEHOLDER = "<arm>"


def mask_arm_in_paths(argv, arm):
    if not arm:
        return list(argv)
    masked = list(argv)
    for index, token in enumerate(masked[:-1]):
        if token in PER_ARM_VALUE_OPTIONS:
            masked[index + 1] = masked[index + 1].replace(arm, ARM_PLACEHOLDER)
    return masked


def _normalized_pair_argv(baseline_argv, treatment_argv, axis,
                          baseline_arm=None, treatment_arm=None):
    for label, argv in (("baseline", baseline_argv), ("treatment", treatment_argv)):
        for option in ("--fp8-format", "--fp8-recipe"):
            if option in argv and (axis != "fp8_gemm" or label != "treatment"):
                raise ValueError(f"{label} argv contains forbidden {option}")
    baseline_argv = mask_arm_in_paths(baseline_argv, baseline_arm)
    treatment_argv = mask_arm_in_paths(treatment_argv, treatment_arm)
    if axis != "fp8_gemm":
        return list(baseline_argv), list(treatment_argv)

    normalized = list(treatment_argv)
    for option, expected in (("--fp8-format", "hybrid"), ("--fp8-recipe", "delayed")):
        if normalized.count(option) != 1:
            raise ValueError(f"treatment argv must contain exactly one {option}")
        index = normalized.index(option)
        if index + 1 >= len(normalized) or normalized[index + 1] != expected:
            raise ValueError(f"treatment argv requires {option} {expected}")
        del normalized[index:index + 2]
    return list(baseline_argv), normalized


def validate_run(run):
    _require(
        run,
        [
            "schema_version",
            "record_type",
            "run_id",
            "arm_id",
            "status",
            "comparison",
            "denominators",
            "provenance",
            "environment",
            "parameter_groups",
            "optimizer_state",
            "measurement",
        ],
        "run",
    )
    if run["schema_version"] != 1 or run["record_type"] != "run":
        raise ValueError("run: unsupported record schema")
    comparison = run["comparison"]
    _require(
        comparison,
        ["optimizer", "gemm_mode", "optimizer_state_mode", "match_key_sha256"],
        "comparison",
    )
    if comparison["optimizer"] not in {"adamw", "muon"}:
        raise ValueError("comparison: unsupported optimizer")
    _require(
        run["provenance"],
        [
            "git_commit",
            "git_dirty",
            "git_diff_sha256",
            "config_sha256",
            "data_manifest_sha256",
            "argv_scope",
            "argv",
        ],
        "provenance",
    )
    if run["provenance"]["argv_scope"] != "effective_mcore_argv":
        raise ValueError("provenance argv must be effective MCore argv")
    measurement = run["measurement"]
    _require(measurement, ["protocol", "memory", "timing", "loss", "routing", "downstream", "inference"], "measurement")
    _require(measurement["protocol"], ["kind", "warmup_steps", "measured_steps", "e2e_train_steps"], "protocol")
    _require(measurement["memory"], ["max_allocated_bytes", "max_reserved_bytes"], "memory")
    _require(
        measurement["timing"],
        [
            "tokens_per_second",
            "optimizer_step_seconds",
            "full_step_seconds",
            "e2e_wct_seconds",
            "e2e_wct_scope",
            "optimizer_step_samples_seconds",
            "full_step_samples_seconds",
        ],
        "timing",
    )
    _require(measurement["loss"], ["training", "validation"], "loss")
    if measurement["timing"]["e2e_wct_scope"] not in {
        "process_start_to_result_write",
        "launcher_start_to_process_exit",
    }:
        raise ValueError("timing.e2e_wct_scope: unsupported scope")
    _require(
        measurement["routing"],
        [
            "scope",
            "tokens_per_expert_artifact_sha256",
            "minimum_to_mean",
            "maximum_to_mean",
            "coefficient_of_variation",
            "dropped_tokens",
        ],
        "routing",
    )
    if measurement["routing"]["scope"] != "global_unpadded":
        raise ValueError("routing: counts must be global and unpadded")
    for name in ("max_allocated_bytes", "max_reserved_bytes"):
        _positive_or_none(measurement["memory"][name], f"memory.{name}")
    for section, names in (
        (measurement["timing"], ("tokens_per_second", "optimizer_step_seconds", "full_step_seconds", "e2e_wct_seconds")),
        (measurement["loss"], ("training", "validation")),
    ):
        for name in names:
            _positive_or_none(section[name], name)
    for name in ("optimizer_step_samples_seconds", "full_step_samples_seconds"):
        samples = measurement["timing"][name]
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError(f"timing.{name}: samples must be finite positive numbers")
        if run["status"] == "completed" and len(samples) != measurement["protocol"]["measured_steps"]:
            raise ValueError(f"timing.{name}: sample count must equal measured_steps")
    _positive_or_none(run["optimizer_state"]["persistent_total_bytes"], "optimizer_state.persistent_total_bytes")


def _ratio(numerator, denominator):
    if numerator is None or denominator is None:
        return None
    return numerator / denominator


def _effect(baseline, treatment, ratio_direction):
    ratio = _ratio(baseline, treatment) if ratio_direction == "baseline_over_treatment" else _ratio(treatment, baseline)
    return {
        "baseline": baseline,
        "treatment": treatment,
        "ratio": ratio,
        "ratio_direction": ratio_direction,
        "ratio_ci95": None,
    }


def _gate(status, criterion):
    return {"status": status, "criterion": criterion}


def _validate_ci(ci, label):
    if ci is None:
        return
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError(f"{label}: expected [lower, upper] or null")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in ci):
        raise ValueError(f"{label}: bounds must be finite numbers")
    if ci[0] > ci[1]:
        raise ValueError(f"{label}: lower bound exceeds upper bound")


def _routing_gate(run):
    if "--mock-data" in run["provenance"]["argv"]:
        # Random tokens make the per-expert distribution unrepresentative at any step
        # count, so the counts are recorded but cannot decide the gate either way.
        return "inconclusive"
    routing = run["measurement"]["routing"]
    fields = (
        routing["tokens_per_expert_artifact_sha256"],
        routing["minimum_to_mean"],
        routing["coefficient_of_variation"],
        routing["dropped_tokens"],
    )
    if any(value is None for value in fields):
        return "inconclusive"
    if (
        routing["minimum_to_mean"] >= 0.10
        and routing["coefficient_of_variation"] < 0.20
        and routing["dropped_tokens"] == 0
    ):
        return "pass"
    return "fail"


def _degradation(baseline, treatment, higher_is_better):
    if baseline == 0:
        return None
    signed = baseline - treatment if higher_is_better else treatment - baseline
    return max(0.0, signed / abs(baseline))


def _downstream_effects(baseline, treatment, inference):
    baseline_items = {(item["task"], item["metric"]): item for item in baseline}
    treatment_items = {(item["task"], item["metric"]): item for item in treatment}
    if not baseline_items and not treatment_items:
        return [], "inconclusive"
    if baseline_items.keys() != treatment_items.keys():
        raise ValueError("downstream: task/metric sets differ")
    effects = []
    statuses = []
    inferred = {}
    if inference is not None:
        inferred = {
            (item["task"], item["metric"]): item["degradation_ci95"]
            for item in inference["downstream"]
        }
        if inferred.keys() != baseline_items.keys():
            raise ValueError("downstream: paired inference task/metric sets differ")
    for key in sorted(baseline_items):
        base = baseline_items[key]
        treat = treatment_items[key]
        if base["higher_is_better"] != treat["higher_is_better"]:
            raise ValueError(f"downstream {key}: metric direction differs")
        degradation = _degradation(base["value"], treat["value"], base["higher_is_better"])
        evidence = base.get("per_example_artifact_sha256") and treat.get("per_example_artifact_sha256")
        ci = inferred.get(key)
        if degradation is None or not evidence or ci is None:
            status = "inconclusive"
        elif ci[1] < 0.01:
            status = "pass"
        elif ci[0] >= 0.01:
            status = "fail"
        else:
            status = "inconclusive"
        statuses.append(status)
        effects.append(
            {
                "task": key[0],
                "metric": key[1],
                "higher_is_better": base["higher_is_better"],
                "baseline": base["value"],
                "treatment": treat["value"],
                "degradation_fraction_of_baseline": degradation,
                "degradation_ci95": ci,
                "status": status,
            }
        )
    if "fail" in statuses:
        return effects, "fail"
    if all(status == "pass" for status in statuses):
        return effects, "pass"
    return effects, "inconclusive"


def compare_runs(baseline, treatment):
    validate_run(baseline)
    validate_run(treatment)
    base_cmp = baseline["comparison"]
    treat_cmp = treatment["comparison"]
    if base_cmp["optimizer"] != treat_cmp["optimizer"]:
        raise ValueError("pair must use the same optimizer")
    arm_pair = (baseline["arm_id"], treatment["arm_id"])
    matching_pairs = [key for key, arms in PAIR_ARMS.items() if arms == arm_pair]
    if len(matching_pairs) != 1:
        raise ValueError("unsupported baseline/treatment arm pair")
    pair_key = matching_pairs[0]
    axis, optimizer = pair_key
    if base_cmp["optimizer"] != optimizer:
        raise ValueError("arm pair and optimizer disagree")
    expected_modes = {
        "optimizer_state": (("bf16", "fp32"), ("bf16", "fp8_hybrid")),
        "fp8_gemm": (("bf16", "fp32"), ("fp8_delayed_hybrid", "fp32")),
    }[axis]
    actual_modes = (
        (base_cmp["gemm_mode"], base_cmp["optimizer_state_mode"]),
        (treat_cmp["gemm_mode"], treat_cmp["optimizer_state_mode"]),
    )
    if actual_modes != expected_modes:
        raise ValueError("pair changes more than its declared comparison axis")
    if base_cmp["match_key_sha256"] != treat_cmp["match_key_sha256"]:
        raise ValueError("pair controlled-factor hashes differ")
    for run in (baseline, treatment):
        if run["provenance"].get("argv_scope") != "effective_mcore_argv":
            raise ValueError("provenance argv must be effective MCore argv")
    if baseline["provenance"]["config_sha256"] != treatment["provenance"]["config_sha256"]:
        raise ValueError("pair matched config hashes differ")
    normalized_base_argv, normalized_treat_argv = _normalized_pair_argv(
        baseline["provenance"]["argv"],
        treatment["provenance"]["argv"],
        axis,
        baseline_arm=baseline["arm_id"],
        treatment_arm=treatment["arm_id"],
    )
    if normalized_base_argv != normalized_treat_argv:
        raise ValueError("pair normalized effective MCore argv differ")
    if baseline["denominators"] != treatment["denominators"]:
        raise ValueError("pair denominators differ")
    if baseline["environment"]["gpus"] != treatment["environment"]["gpus"]:
        raise ValueError("pair GPU identities differ")
    if baseline["measurement"]["protocol"] != treatment["measurement"]["protocol"]:
        raise ValueError("pair measurement protocols differ")
    if (
        baseline["measurement"]["timing"]["e2e_wct_scope"]
        != treatment["measurement"]["timing"]["e2e_wct_scope"]
    ):
        raise ValueError("pair end-to-end WCT scopes differ")
    for key in (
        "site",
        "host",
        "image",
        "python",
        "torch",
        "cuda_runtime",
        "driver",
        "cublaslt",
        "triton",
        "transformer_engine",
        "megatron_core",
        "emerging_optimizers",
        "nccl",
    ):
        if baseline["environment"][key] != treatment["environment"][key]:
            raise ValueError(f"pair environment field {key} differs")
    for key in ("git_commit", "git_dirty", "git_diff_sha256", "data_manifest_sha256"):
        if baseline["provenance"][key] != treatment["provenance"][key]:
            raise ValueError(f"pair provenance field {key} differs")
    if baseline["parameter_groups"] != treatment["parameter_groups"]:
        raise ValueError("pair optimizer parameter assignments differ")

    base_measurement = baseline["measurement"]
    treat_measurement = treatment["measurement"]
    if base_measurement["inference"] is not None:
        raise ValueError("baseline run must not contain treatment inference")
    inference = treat_measurement["inference"]
    if inference is not None:
        _require(
            inference,
            [
                "baseline_run_id",
                "method",
                "confidence_level",
                "sidedness",
                "memory_allocated_ratio_ci95",
                "e2e_wct_ratio_ci95",
                "validation_loss_degradation_ci95",
                "downstream",
            ],
            "inference",
        )
        if inference["baseline_run_id"] != baseline["run_id"]:
            raise ValueError("paired inference names a different baseline run")
        if "paired" not in inference["method"]:
            raise ValueError("inference method must contain lowercase paired")
        if inference["confidence_level"] != 0.95 or inference["sidedness"] != "one-sided":
            raise ValueError("inference must use a one-sided 95% confidence bound")
        for key in (
            "memory_allocated_ratio_ci95",
            "e2e_wct_ratio_ci95",
            "validation_loss_degradation_ci95",
        ):
            _validate_ci(inference[key], f"inference.{key}")
        for item in inference["downstream"]:
            _validate_ci(item["degradation_ci95"], "inference.downstream.degradation_ci95")
    memory_allocated = _effect(
        base_measurement["memory"]["max_allocated_bytes"],
        treat_measurement["memory"]["max_allocated_bytes"],
        "baseline_over_treatment",
    )
    memory_reserved = _effect(
        base_measurement["memory"]["max_reserved_bytes"],
        treat_measurement["memory"]["max_reserved_bytes"],
        "baseline_over_treatment",
    )
    state_bytes = _effect(
        baseline["optimizer_state"]["persistent_total_bytes"],
        treatment["optimizer_state"]["persistent_total_bytes"],
        "baseline_over_treatment",
    )
    tokens_per_second = _effect(
        base_measurement["timing"]["tokens_per_second"],
        treat_measurement["timing"]["tokens_per_second"],
        "treatment_over_baseline",
    )
    optimizer_step = _effect(
        base_measurement["timing"]["optimizer_step_seconds"],
        treat_measurement["timing"]["optimizer_step_seconds"],
        "baseline_over_treatment",
    )
    full_step = _effect(
        base_measurement["timing"]["full_step_seconds"],
        treat_measurement["timing"]["full_step_seconds"],
        "baseline_over_treatment",
    )
    e2e_wct = _effect(
        base_measurement["timing"]["e2e_wct_seconds"],
        treat_measurement["timing"]["e2e_wct_seconds"],
        "baseline_over_treatment",
    )
    memory_allocated["ratio_ci95"] = (
        None if inference is None else inference["memory_allocated_ratio_ci95"]
    )
    e2e_wct["ratio_ci95"] = None if inference is None else inference["e2e_wct_ratio_ci95"]
    validation_degradation = None
    validation_ci = None if inference is None else inference["validation_loss_degradation_ci95"]
    if base_measurement["loss"]["validation"] is not None and treat_measurement["loss"]["validation"] is not None:
        validation_degradation = _degradation(
            base_measurement["loss"]["validation"],
            treat_measurement["loss"]["validation"],
            False,
        )
    downstream, downstream_status = _downstream_effects(
        base_measurement["downstream"], treat_measurement["downstream"], inference
    )

    clean = all(baseline["environment"]["gpu_clean"].values()) and all(
        treatment["environment"]["gpu_clean"].values()
    )
    health_status = "pass" if baseline["status"] == treatment["status"] == "completed" and clean else "fail"
    routing_statuses = (_routing_gate(baseline), _routing_gate(treatment))
    routing_status = "fail" if "fail" in routing_statuses else ("pass" if routing_statuses == ("pass", "pass") else "inconclusive")
    memory_status = "not_applicable"
    wct_status = "not_applicable"
    if axis == "optimizer_state":
        memory_ci = None if inference is None else inference["memory_allocated_ratio_ci95"]
        if memory_ci is None:
            memory_status = "inconclusive"
        elif memory_ci[0] >= 1.10:
            memory_status = "pass"
        elif memory_ci[1] < 1.10:
            memory_status = "fail"
        else:
            memory_status = "inconclusive"
    else:
        timing_ready = all(
            run["measurement"]["protocol"]["warmup_steps"] >= 20
            and run["measurement"]["protocol"]["measured_steps"] >= 100
            and run["measurement"]["timing"]["e2e_wct_scope"]
            == "launcher_start_to_process_exit"
            for run in (baseline, treatment)
        )
        wct_ci = None if inference is None else inference["e2e_wct_ratio_ci95"]
        if e2e_wct["ratio"] is None or wct_ci is None or not timing_ready:
            wct_status = "inconclusive"
        elif wct_ci[0] >= 1.10:
            wct_status = "pass"
        elif wct_ci[1] < 1.10:
            wct_status = "fail"
        else:
            wct_status = "inconclusive"
    if validation_degradation is None or validation_ci is None:
        validation_status = "inconclusive"
    elif validation_ci[1] < 0.01:
        validation_status = "pass"
    elif validation_ci[0] >= 0.01:
        validation_status = "fail"
    else:
        validation_status = "inconclusive"

    gates = {
        "health": _gate(health_status, "both runs completed on clean matched GPUs"),
        "routing": _gate(routing_status, "global unpadded min/mean >= 0.10, CV < 0.20, dropped tokens = 0; not decidable on mock data"),
        "memory": _gate(memory_status, "optimizer-state axis: one-sided paired 95% CI lower bound for max-allocated baseline/treatment >= 1.10"),
        "wct": _gate(wct_status, "FP8-GEMM axis: one-sided paired 95% CI lower bound for launcher-start-to-process-exit WCT baseline/treatment >= 1.10 after >=20 warmups and >=100 measured steps"),
        "validation": _gate(validation_status, "one-sided paired 95% CI upper bound for relative validation-loss degradation < 1%"),
        "downstream": _gate(downstream_status, "every matched metric has baseline-relative degradation < 1% and per-example artifacts"),
    }
    primary_status = memory_status if axis == "optimizer_state" else wct_status
    required_statuses = [health_status, routing_status, primary_status, validation_status, downstream_status]
    short_smoke = any(run["measurement"]["protocol"]["kind"] == "smoke" for run in (baseline, treatment))
    if short_smoke:
        verdict = "inconclusive"
    elif "fail" in required_statuses:
        verdict = "fail"
    elif all(status == "pass" for status in required_statuses):
        verdict = "pass"
    else:
        verdict = "inconclusive"

    return {
        "schema_version": 1,
        "record_type": "pair",
        "pair_id": f"{baseline['run_id']}__vs__{treatment['run_id']}",
        "axis": axis,
        "optimizer": optimizer,
        "baseline_run_id": baseline["run_id"],
        "treatment_run_id": treatment["run_id"],
        "denominators": baseline["denominators"],
        "effects": {
            "max_memory_allocated_bytes": memory_allocated,
            "max_memory_reserved_bytes": memory_reserved,
            "persistent_optimizer_state_bytes": state_bytes,
            "tokens_per_second": tokens_per_second,
            "optimizer_step_seconds": optimizer_step,
            "full_step_seconds": full_step,
            "e2e_wct_seconds": e2e_wct,
            "validation_loss_degradation_fraction_of_baseline": validation_degradation,
            "validation_loss_degradation_ci95": validation_ci,
            "downstream": downstream,
        },
        "gates": gates,
        "verdict": verdict,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_jsonl")
    parser.add_argument("treatment_jsonl")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        result = compare_runs(load_run(args.baseline_jsonl), load_run(args.treatment_jsonl))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 2
    encoded = json.dumps(result, sort_keys=True)
    if args.output:
        Path(args.output).write_text(encoded + "\n")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
