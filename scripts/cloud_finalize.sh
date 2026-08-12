#!/usr/bin/env bash
# Build the paired intervals across every stored replicate, write them into the
# treatment records, then re-evaluate the pairs.  CPU only.
#   mlsub run ... --entry scripts/cloud_finalize.sh --gpus cpu --args "FILTER"
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
persist=/home/jovyan/hmoe-cloud/artifacts/stage3-moe-probes
filter=${1:-}

cd "$root"
export PYTHONPATH="$root"

echo "=== PAIRED INFERENCE ==="
python -m stage3_moe.paired_inference "$persist" --filter "$filter" --write
echo "PAIRED_INFERENCE_EXIT=$?"

echo
echo "=== PAIR VERDICTS ==="
python - "$persist" "$filter" <<'PY'
import json, subprocess, sys
from itertools import permutations
from pathlib import Path

root, filter_ = Path(sys.argv[1]), sys.argv[2]
files = sorted(p for p in root.glob("*/results.jsonl") if filter_ in str(p))
seen = set()
for a, b in permutations(files, 2):
    out = subprocess.run([sys.executable, "-m", "stage3_moe.pair_results", str(a), str(b)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        continue
    d = json.loads(out.stdout)
    key = (d["axis"], d["optimizer"], d["baseline_run_id"], d["treatment_run_id"])
    if key in seen:
        continue
    seen.add(key)
    e = d["effects"]
    den = d["denominators"]
    print(f"\naxis={d['axis']} opt={d['optimizer']} VERDICT={d['verdict']}"
          f"  mb={den['micro_batch_sequences_per_gpu']} tokens/step={den['loss_tokens_per_step']}")
    print(f"  {d['baseline_run_id']}\n  -> {d['treatment_run_id']}")
    for k in ("max_memory_allocated_bytes", "persistent_optimizer_state_bytes",
              "full_step_seconds", "e2e_wct_seconds"):
        v = e[k]
        ci = v.get("ratio_ci95")
        text = "" if ci is None else f"  ci95=[{ci[0]:.4f}, {ci[1]:.4f}]"
        print(f"  {k:32s} ratio={v['ratio']:.4f}{text}")
    print("  gates: " + ", ".join(f"{g}={d['gates'][g]['status']}" for g in d["gates"]))
PY
echo "EXIT=0"
exit 0
