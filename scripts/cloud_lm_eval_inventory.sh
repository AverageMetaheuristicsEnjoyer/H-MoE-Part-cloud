#!/usr/bin/env bash
# Inventory the task definitions available in the pinned lm-eval image without loading datasets.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)

if ! python -c 'import lm_eval; assert lm_eval.__version__ == "0.4.11"' >/dev/null 2>&1; then
  printf 'torch==2.8.0\n' > /tmp/eval-constraints.txt
  pip install --user --no-cache-dir --constraint /tmp/eval-constraints.txt "lm-eval==0.4.11" 2>&1 | tail -5
fi

cd "$root"
PYTHONPATH="$root" python - <<'PY'
import json
from collections import Counter

from lm_eval.tasks import TaskManager


manager = TaskManager(include_path="stage4/eval_tasks")
counts = Counter()
compatible = []
for name in manager.all_subtasks:
    entry = manager.task_index[name]
    if entry["type"] == "python_task":
        counts["python_task"] += 1
        continue
    config = manager._get_config(name)
    output_type = config.get("output_type", "unknown")
    counts[output_type] += 1
    if output_type not in {"multiple_choice", "loglikelihood_rolling"}:
        continue
    compatible.append(
        {
            "task": name,
            "output_type": output_type,
            "dataset_path": config.get("dataset_path"),
            "dataset_name": config.get("dataset_name"),
            "validation_split": config.get("validation_split"),
            "test_split": config.get("test_split"),
            "num_fewshot": config.get("num_fewshot"),
            "metrics": [item["metric"] for item in config.get("metric_list", [])],
        }
    )

print(
    "INVENTORY "
    + json.dumps(
        {
            "groups": len(manager.all_groups),
            "tags": len(manager.all_tags),
            "subtasks": len(manager.all_subtasks),
            "output_types": dict(sorted(counts.items())),
            "compatible": len(compatible),
        },
        sort_keys=True,
    )
)
for item in compatible:
    print("TASK " + json.dumps(item, sort_keys=True))
PY
code=$?
echo "EXIT=$code"
exit "$code"
