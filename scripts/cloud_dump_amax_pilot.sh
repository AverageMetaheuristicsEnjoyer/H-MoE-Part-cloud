#!/usr/bin/env bash
# Print compact train and validation curves from a bounded amax pilot. Read-only.
set -u

log_root=${STAGE3_MOE_LOG_ROOT:-/workspace-SR006.nfs2/hmoe-cloud/pretrain}
filter=${1:?pass the pilot run suffix}

python - "$log_root" "$filter" <<'PY'
import json
import re
import sys
from pathlib import Path

root, filter_ = Path(sys.argv[1]), sys.argv[2]
for run_dir in sorted(path for path in root.glob(f"*{filter_}*") if path.is_dir()):
    logs = sorted(run_dir.glob("train-*.log"), key=lambda path: path.stat().st_mtime)
    if not logs:
        continue
    print(f"RUN {run_dir.name}")
    result_path = run_dir / "results.jsonl"
    if result_path.exists():
        record = json.loads(result_path.read_text().splitlines()[0])
        argv = record["provenance"]["argv"]
        recipe = []
        for option in ("--fp8-format", "--fp8-recipe", "--fp8-amax-history-len", "--fp8-amax-compute-algo"):
            if option in argv:
                index = argv.index(option)
                recipe.append(f"{option[2:]}={argv[index + 1]}")
        print("RECIPE " + (" ".join(recipe) if recipe else "bf16"))
    text = logs[-1].read_text(errors="replace")
    for line in text.splitlines():
        match = re.search(
            r"iteration\s+(\d+)/\s*(\d+).*?lm loss:\s*([0-9.Ee+-]+).*?"
            r"number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)",
            line,
        )
        if match:
            step, total, loss, skipped, nan = match.groups()
            print(f"STEP {step}/{total} loss={loss} skipped={skipped} nan={nan}")
        match = re.search(
            r"validation loss at iteration\s+(\d+).*?lm loss value:\s*([0-9.Ee+-]+)",
            line,
            re.IGNORECASE,
        )
        if match:
            print(f"VALID step={match.group(1)} loss={match.group(2)}")
    exit_match = re.findall(r"TRAIN_EXIT=(\d+)", text)
    if exit_match:
        print(f"TRAIN_EXIT={exit_match[-1]}")
print("EXIT=0")
PY
