#!/usr/bin/env bash
# Gate one optimizer's full Wave 2 evaluation behind an all-task GPU smoke.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
optimizer=${1:?usage: cloud_moe_eval_wave2_pipeline.sh adamw|muon}
case "$optimizer" in
  adamw)
    arms=(adamw_bf16_state_fp32 adamw_bf16_state_fp8 adamw_fp8gemm_state_fp32)
    ;;
  muon)
    arms=(muon_bf16_state_fp32 muon_bf16_state_fp8 muon_fp8gemm_state_fp32)
    ;;
  *)
    echo "unknown optimizer: $optimizer"
    echo "EXIT=0"
    exit 0
    ;;
esac

export STAGE3_MOE_LOG_ROOT=${STAGE3_MOE_LOG_ROOT:-/workspace-SR006.nfs2/hmoe-cloud/pretrain}

validate_artifacts() {
  python - "$1" "$2" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
kind = sys.argv[2]
expected = {
    ("c4", "bits_per_byte"),
    ("mmlu", "acc"),
    ("openbookqa", "acc"),
    ("openbookqa", "acc_norm"),
    ("wikitext", "bits_per_byte"),
    ("winogrande", "acc"),
}
downstream = json.loads((run_dir / "downstream" / "downstream.json").read_text())
assert {(item["task"], item["metric"]) for item in downstream} == expected
assert all(math.isfinite(item["value"]) for item in downstream)

row_counts = {}
for item in downstream:
    artifact = run_dir / "downstream" / f"{item['task']}.jsonl"
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == item["per_example_artifact_sha256"]
    rows = [json.loads(line) for line in artifact.read_text().splitlines()]
    row_counts[item["task"]] = len(rows)
    values = [row["metrics"][item["metric"]] for row in rows]
    if item["metric"] == "bits_per_byte":
        assert all(len(value) == 2 and value[1] > 0 for value in values)
    else:
        assert all(value in (0.0, 1.0) for value in values)

expected_rows = (
    {"c4": 2, "mmlu": 114, "openbookqa": 2, "wikitext": 2, "winogrande": 2}
    if kind == "smoke"
    else {"c4": 45576, "mmlu": 14042, "openbookqa": 500, "wikitext": 62, "winogrande": 1267}
)
assert row_counts == expected_rows, row_counts
records = [
    json.loads(line)
    for line in (run_dir / "results.jsonl").read_text().splitlines()
    if line.strip() and json.loads(line).get("record_type") == "run"
]
assert len(records) == 1
assert {
    (item["task"], item["metric"])
    for item in records[0]["measurement"]["downstream"]
} == expected
print(f"ARTIFACT_GATE=PASS kind={kind} run={run_dir.name} rows={row_counts}")
PY
}

export STAGE3_MOE_EVAL_LIMIT=2
export STAGE3_MOE_RUN_SUFFIX="wave2-smoke-$optimizer-v1"
"$root/scripts/cloud_moe_eval_wave2.sh" "${arms[0]}"
smoke_dir="$STAGE3_MOE_LOG_ROOT/stage3-${arms[0]}-eval-downstream-$STAGE3_MOE_RUN_SUFFIX"
if ! validate_artifacts "$smoke_dir" smoke; then
  echo "SMOKE_GATE=FAIL optimizer=$optimizer"
  echo "EXIT=0"
  exit 0
fi
echo "SMOKE_GATE=PASS optimizer=$optimizer"

unset STAGE3_MOE_EVAL_LIMIT
export STAGE3_MOE_RUN_SUFFIX=wave2-1c-v1
"$root/scripts/cloud_moe_eval_wave2.sh" "${arms[@]}"

failed=0
for arm in "${arms[@]}"; do
  run_dir="$STAGE3_MOE_LOG_ROOT/stage3-$arm-eval-downstream-$STAGE3_MOE_RUN_SUFFIX"
  if ! validate_artifacts "$run_dir" full; then
    echo "FULL_GATE=FAIL arm=$arm"
    failed=1
  fi
done
if [[ $failed -eq 0 ]]; then
  echo "PIPELINE_STATUS=COMPLETE optimizer=$optimizer"
  state_dir="$STAGE3_MOE_LOG_ROOT/wave2-1c-v1-state"
  mkdir -p "$state_dir"
  touch "$state_dir/$optimizer.done"
  if [[ -f $state_dir/adamw.done && -f $state_dir/muon.done ]] && mkdir "$state_dir/finalize.lock" 2>/dev/null; then
    echo "=== FINALIZING BOTH OPTIMIZERS ==="
    STAGE3_MOE_RESULTS_ROOT="$STAGE3_MOE_LOG_ROOT" \
      "$root/scripts/cloud_finalize.sh" eval-downstream-wave2-1c-v1
    python - "$STAGE3_MOE_LOG_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
treatments = (
    "adamw_bf16_state_fp8",
    "adamw_fp8gemm_state_fp32",
    "muon_bf16_state_fp8",
    "muon_fp8gemm_state_fp32",
)
for arm in treatments:
    path = root / f"stage3-{arm}-eval-downstream-wave2-1c-v1" / "results.jsonl"
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip() and json.loads(line).get("record_type") == "run"
    ]
    assert len(records) == 1
    inference = records[0]["measurement"]["inference"]
    assert inference is not None and len(inference["downstream"]) == 6
print("FINALIZE_GATE=PASS pairs=4 metrics_per_pair=6")
PY
    if [[ $? -eq 0 ]]; then
      touch "$state_dir/finalized.done"
    else
      echo "FINALIZE_GATE=FAIL"
    fi
  fi
else
  echo "PIPELINE_STATUS=FAILED optimizer=$optimizer"
fi
echo "EXIT=0"
exit 0
