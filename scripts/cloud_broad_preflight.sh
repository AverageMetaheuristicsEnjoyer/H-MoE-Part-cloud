#!/usr/bin/env bash
# Load and enumerate the frozen broad exploratory task group on CPU.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME=${STAGE3_MOE_HF_HOME:-/home/jovyan/hmoe-hf-cache-broad-v1}
export HF_DATASETS_TRUST_REMOTE_CODE=1

if ! python -c 'import lm_eval; assert lm_eval.__version__ == "0.4.11"' >/dev/null 2>&1; then
  printf 'torch==2.8.0\n' > /tmp/eval-constraints.txt
  pip install --user --no-cache-dir --constraint /tmp/eval-constraints.txt "lm-eval==0.4.11" 2>&1 | tail -5
fi

cd "$root"
PYTHONPATH="$root" python - <<'PY'
from lm_eval.tasks import TaskManager


def leaves(tasks):
    for value in tasks.values():
        if isinstance(value, dict):
            yield from leaves(value)
        else:
            yield value


expected = {
    "swag": (0, ["acc", "acc_norm"]),
    "mnli": (0, ["acc"]),
    "mnli_mismatch": (0, ["acc"]),
    "qnli": (0, ["acc"]),
    "qqp": (0, ["acc"]),
    "prost": (0, ["acc", "acc_norm"]),
    "toxigen": (0, ["acc", "acc_norm"]),
    "moral_stories": (0, ["acc", "acc_norm"]),
    "boolq": (0, ["acc"]),
    "race": (0, ["acc"]),
    "lambada_openai": (0, ["acc"]),
    "pile_10k": (0, ["bits_per_byte"]),
    "leaderboard_mmlu_pro": (5, ["acc"]),
}

manager = TaskManager(include_path="stage4/eval_tasks")
tasks = list(leaves(manager.load_task_or_group(["stage3_broad_v1"])))
assert len(tasks) == 80, len(tasks)
blimp = 0
for task in tasks:
    name = task.task_name
    if name.startswith("blimp_"):
        shots, metrics = 0, ["acc"]
        blimp += 1
    else:
        shots, metrics = expected.pop(name)
    actual_metrics = sorted(task.higher_is_better())
    assert task.get_config("num_fewshot") == shots, (name, task.get_config("num_fewshot"))
    assert actual_metrics == metrics, (name, actual_metrics)
    print(f"TASK name={name} docs={len(task.eval_docs)} fewshot={shots} metrics={','.join(metrics)}")
assert blimp == 67, blimp
assert not expected, expected
print(f"BROAD_TASKS={len(tasks)} BLIMP_LEAVES={blimp}")
PY
code=$?
echo "TASK_EXIT=$code"
echo "EXIT=$code"
exit "$code"
