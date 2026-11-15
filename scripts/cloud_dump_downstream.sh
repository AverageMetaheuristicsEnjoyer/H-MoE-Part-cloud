#!/usr/bin/env bash
# Print every scoring record's numbers as one compact JSON line per arm, so a round of
# downstream results can be lifted off the volume and committed. The scores exist nowhere
# else: run_suite writes them to the run directory and nothing pushes them to W&B.
#   mlsub run ... --entry scripts/cloud_dump_downstream.sh --gpus cpu --args "FILTER"
# One line per arm keeps the whole round inside the platform's log window; the pretty-printed
# downstream.json files would not fit.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
persist=${STAGE3_MOE_RESULTS_ROOT:-/home/jovyan/hmoe-cloud/pretrain}
filter=${1:-}

cd "$root"
python - "$persist" "$filter" <<'PY'
import json, sys
from pathlib import Path

root, filter_ = Path(sys.argv[1]), sys.argv[2]
paths = sorted(p for p in root.glob("*/results.jsonl") if filter_ in str(p))
print(f"MATCHED {len(paths)}")
for path in paths:
    record = json.loads(path.read_text().splitlines()[0])
    measurement = record["measurement"]
    inference = measurement["inference"] or {}
    print("RECORD " + json.dumps({
        "run_id": record["run_id"],
        "arm_id": record["arm_id"],
        "endpoint": record["measurement"]["protocol"],
        "loss": measurement["loss"],
        "downstream": measurement["downstream"],
        "inference_baseline_run_id": inference.get("baseline_run_id"),
        "inference_downstream": inference.get("downstream"),
    }, sort_keys=True))
PY
echo "EXIT=0"
exit 0
