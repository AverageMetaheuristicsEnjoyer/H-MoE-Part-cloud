#!/usr/bin/env bash
set -u

repo_root=$(cd "$(dirname "$0")/.." && pwd)
results_root=/workspace-SR006.nfs3/hmoe-cloud/pretrain

cd "$repo_root"
PYTHONPATH="$repo_root" python - "$results_root" <<'PY'
import copy
import json
import sys
from pathlib import Path

from stage3_moe.pair_results import compare_runs, load_run

root = Path(sys.argv[1])
baseline_arm = "adamw_bf16_state_fp32"
treatment_arm = "adamw_bf16_state_fp8"
suffix = "eval-downstream-broad-v2-1c"
baseline_path = root / f"stage3-{baseline_arm}-{suffix}" / "results.jsonl"
treatment_dir = root / f"stage3-{treatment_arm}-{suffix}"
treatment_path = treatment_dir / "results.jsonl"

baseline = load_run(baseline_path)
treatment = copy.deepcopy(baseline)
treatment["run_id"] = treatment_dir.name
treatment["arm_id"] = treatment_arm
treatment["comparison"]["optimizer_state_mode"] = "fp8_hybrid"
treatment["provenance"]["argv"] = [
    value.replace(baseline_arm, treatment_arm)
    for value in treatment["provenance"]["argv"]
]
treatment["measurement"]["memory"] = {
    "max_allocated_bytes": None,
    "max_reserved_bytes": None,
}
treatment["measurement"]["timing"] = {
    "tokens_per_second": None,
    "optimizer_step_seconds": None,
    "full_step_seconds": None,
    "e2e_wct_seconds": None,
    "e2e_wct_scope": "process_start_to_result_write",
    "optimizer_step_samples_seconds": [],
    "full_step_samples_seconds": [],
}
treatment["measurement"]["loss"] = {
    "training": None,
    "validation": 2.694221,
}
treatment["measurement"]["routing"] = {
    "scope": "global_unpadded",
    "tokens_per_expert_artifact_sha256": None,
    "minimum_to_mean": None,
    "maximum_to_mean": None,
    "coefficient_of_variation": None,
    "dropped_tokens": None,
}
treatment["measurement"]["downstream"] = json.loads(
    (treatment_dir / "downstream" / "downstream.json").read_text()
)
treatment["measurement"]["inference"] = None
treatment["recovery"] = {
    "reason": "original results.jsonl was truncated by a non-atomic inference write on a full volume",
    "source": "matched baseline record plus preserved downstream.json from the completed treatment run",
    "unrecoverable_fields_set_to_null": ["memory", "timing", "routing"],
}

compare_runs(baseline, treatment)
temporary = treatment_path.with_suffix(".jsonl.tmp")
temporary.write_text(json.dumps(treatment, sort_keys=True) + "\n")
temporary.replace(treatment_path)
print(f"RECOVERED {treatment_path}")
PY
code=$?
echo "PY_EXIT=$code"
echo "EXIT=$code"
exit "$code"
