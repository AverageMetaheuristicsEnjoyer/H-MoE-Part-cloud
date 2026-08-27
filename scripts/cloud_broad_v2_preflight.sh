#!/usr/bin/env bash
# Check dataset availability and sizes for the next frozen English lm-eval screen.
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
from lm_eval.tasks import TaskManager, get_task_dict


def leaves(tasks):
    for value in tasks.values():
        if isinstance(value, dict):
            yield from leaves(value)
        else:
            yield value


candidates = [
    "anli_r1", "anli_r2", "anli_r3", "commonsense_qa",
    "mrpc", "rte", "sst2", "wnli",
    "logiqa", "logiqa2", "mathqa", "mc_taco", "openbookqa",
    "pubmedqa", "sciq", "social_iqa", "storycloze_2016", "storycloze_2018",
    "cb", "copa", "multirc", "sglue_rte", "wic", "wsc",
    "truthfulqa_mc1", "truthfulqa_mc2", "winogender_all", "winogrande", "wsc273",
    "mmlu", "leaderboard_bbh", "wikitext",
    "paloma_4chan_meta_sep", "paloma_c4_en", "paloma_dolma-v1_5",
    "paloma_falcon-refinedweb", "paloma_gab", "paloma_m2d2_s2orc_unsplit",
    "paloma_m2d2_wikipedia_unsplit", "paloma_manosphere_meta_sep", "paloma_ptb",
    "paloma_redpajama", "paloma_twitterAAE_HELM_fixed", "paloma_wikitext_103",
]
manager = TaskManager(include_path="stage4/eval_tasks")
passed = 0
for candidate in candidates:
    try:
        tasks = list(leaves(get_task_dict([candidate], manager)))
        docs = sum(len(task.eval_docs) for task in tasks)
        metrics = sorted({metric for task in tasks for metric in task.higher_is_better()})
        output_types = sorted({task.get_config("output_type") for task in tasks})
        print(
            f"PASS candidate={candidate} leaves={len(tasks)} docs={docs} "
            f"output={','.join(output_types)} metrics={','.join(metrics)}"
        )
        passed += 1
    except Exception as error:
        print(f"FAIL candidate={candidate} error={type(error).__name__}:{error}")
print(f"PREFLIGHT candidates={len(candidates)} passed={passed} failed={len(candidates) - passed}")
PY
code=$?
echo "EXIT=$code"
exit "$code"
