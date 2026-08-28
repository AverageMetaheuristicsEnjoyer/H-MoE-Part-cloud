#!/usr/bin/env bash
set -eu

root=${STAGE3_MOE_ROUTING_OUTPUT_ROOT:-/workspace-SR006.nfs2/hmoe-cloud/routing-audit/fixed-extension-v1}
python - "$root" <<'PY'
import glob
import hashlib
import json
import statistics
import sys

for path in sorted(glob.glob(sys.argv[1] + "/*.json")):
    data = json.load(open(path))
    for evaluation in data["evaluations"]:
        layers = evaluation["layers"]
        actual = [x["actual_balance"]["coefficient_of_variation"] for x in layers]
        frozen = [x["frozen_balance"]["coefficient_of_variation"] for x in layers]
        unbiased = [x["unbiased_balance"]["coefficient_of_variation"] for x in layers]
        bias_values = [v for x in layers for v in x["checkpoint_expert_bias"]]
        bias_hash = hashlib.sha256(
            "".join(x["checkpoint_expert_bias_sha256"] for x in layers).encode()
        ).hexdigest()
        improved = sum(a < f for a, f in zip(actual, frozen))
        trajectory = max(
            layers,
            key=lambda x: x["actual_balance"]["coefficient_of_variation"],
        ).get("actual_cumulative_cv_by_batch", [])
        print(
            f"ROUTING_SUMMARY label={data['checkpoint_label']} "
            f"iteration={data['checkpoint_iteration']} split={evaluation['split']} "
            f"loss={evaluation['loss']:.6f} layers={len(layers)} "
            f"actual_worst={max(actual):.6f} actual_median={statistics.median(actual):.6f} "
            f"frozen_worst={max(frozen):.6f} frozen_median={statistics.median(frozen):.6f} "
            f"unbiased_worst={max(unbiased):.6f} unbiased_median={statistics.median(unbiased):.6f} "
            f"controller_improved_layers={improved}/{len(layers)} "
            f"actual_min_mean={evaluation['worst_actual_minimum_to_mean']:.6f} "
            f"bias_updates={min(x['bias_updates'] for x in layers)}-{max(x['bias_updates'] for x in layers)} "
            f"max_bias_delta={max(x['maximum_absolute_bias_change'] for x in layers):.6f} "
            f"max_adapted_fraction={max(x['assignment_fraction_changed_by_adaptation'] for x in layers):.6f} "
            f"max_changed_by_bias={max(x['assignment_fraction_changed_by_bias'] for x in layers):.6f} "
            f"max_mismatch={max(x['computed_routing_mismatch_fraction'] for x in layers):.6f} "
            f"checkpoint_bias_nonzero={sum(v != 0 for v in bias_values)}/{len(bias_values)} "
            f"checkpoint_bias_min={min(bias_values):.6f} "
            f"checkpoint_bias_max={max(bias_values):.6f} "
            f"checkpoint_bias_mean={statistics.mean(bias_values):.6f} "
            f"checkpoint_bias_hash={bias_hash}"
        )
        if trajectory:
            points = sorted({0, 1, 3, 7, 15, 31, 63, len(trajectory) - 1})
            print(
                f"ROUTING_TRAJECTORY label={data['checkpoint_label']} "
                f"split={evaluation['split']} worst_final_layer_cumulative_cv="
                + ",".join(f"{i + 1}:{trajectory[i]:.6f}" for i in points if i >= 0)
            )
print("EXIT=0")
PY
