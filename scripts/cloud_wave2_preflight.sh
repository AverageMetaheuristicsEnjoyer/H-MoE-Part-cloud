#!/usr/bin/env bash
# Load the frozen Wave 2 task group on CPU, warm its dataset cache, and run paired tests.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME=${STAGE3_MOE_HF_HOME:-/workspace-SR006.nfs2/hmoe-hf-cache}
export HF_DATASETS_TRUST_REMOTE_CODE=1
mkdir -p "$HF_HOME"

if ! python -c 'import lm_eval; assert lm_eval.__version__ == "0.4.11"' >/dev/null 2>&1; then
  printf 'torch==2.8.0\n' > /tmp/eval-constraints.txt
  pip install --user --no-cache-dir --constraint /tmp/eval-constraints.txt "lm-eval==0.4.11" 2>&1 | tail -5
fi

cd "$root"
python -m unittest discover -s tests/stage3_moe -p 'test_*downstream*.py'
test_code=$?
PYTHONPATH="$root" python - <<'PY'
from lm_eval.tasks import TaskManager


def leaves(tasks):
    for value in tasks.values():
        if isinstance(value, dict):
            yield from leaves(value)
        else:
            yield value


manager = TaskManager(include_path="stage4/eval_tasks")
tasks = list(leaves(manager.load_task_or_group(["stage3_wave2"])))
assert len(tasks) == 61, len(tasks)
for task in tasks:
    name = task.task_name
    expected_shots = 5 if name in ("winogrande", "openbookqa") or name.startswith("mmlu_") else 0
    assert task.get_config("num_fewshot") == expected_shots, (name, task.get_config("num_fewshot"))
    metrics = sorted(task.higher_is_better())
    if name in ("wikitext", "c4"):
        assert metrics == ["bits_per_byte"], (name, metrics)
    print(f"TASK name={name} docs={len(task.eval_docs)} fewshot={expected_shots} metrics={','.join(metrics)}")
print(f"WAVE2_TASKS={len(tasks)}")
PY
task_code=$?
echo "TEST_EXIT=$test_code TASK_EXIT=$task_code"
if [[ $test_code -ne 0 || $task_code -ne 0 ]]; then
  echo "EXIT=1"
  exit 1
fi
echo "EXIT=0"
