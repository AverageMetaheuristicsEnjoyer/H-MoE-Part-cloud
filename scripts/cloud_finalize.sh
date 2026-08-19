#!/usr/bin/env bash
# Build the paired intervals across every stored replicate, write them into the
# treatment records, then re-evaluate the pairs.  CPU only.
#   mlsub run ... --entry scripts/cloud_finalize.sh --gpus cpu --args "FILTER"
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
# Probe records live under artifacts/; scoring records live beside their run in pretrain/.
persist=${STAGE3_MOE_RESULTS_ROOT:-/home/jovyan/hmoe-cloud/artifacts/stage3-moe-probes}
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
rejected = {}
for a, b in permutations(files, 2):
    out = subprocess.run([sys.executable, "-m", "stage3_moe.pair_results", str(a), str(b)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        # Every rejection was silent, so a run that paired nothing looked the same as a
        # run with nothing to pair. One line per distinct reason is enough to act on.
        reason = (out.stderr or "").strip().splitlines()
        reason = reason[-1] if reason else "unknown"
        if reason not in rejected:
            rejected[reason] = (a.parent.name, b.parent.name)
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
        # An evaluation record has no optimizer state and takes no training step, so
        # these are legitimately null there.
        if v["ratio"] is None:
            print(f"  {k:32s} ratio=n/a")
            continue
        ci = v.get("ratio_ci95")
        text = "" if ci is None else f"  ci95=[{ci[0]:.4f}, {ci[1]:.4f}]"
        print(f"  {k:32s} ratio={v['ratio']:.4f}{text}")
    val = e["validation_loss_degradation_fraction_of_baseline"]
    val_ci = e["validation_loss_degradation_ci95"]
    val_ci_text = "" if val_ci is None else f"  ci95=[{val_ci[0]:+.4f}, {val_ci[1]:+.4f}]"
    print(f"  {'validation_loss':32s} degradation="
          f"{'n/a' if val is None else format(val, '+.4%')}{val_ci_text}")
    for item in e["downstream"]:
        ci = item["degradation_ci95"]
        ci_text = "" if ci is None else f"  ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        deg = item["degradation_fraction_of_baseline"]
        deg_text = "n/a" if deg is None else format(deg, "+.4%")
        print(f"  {item['task']:32s} {item['metric']:12s} "
              f"{item['baseline']:.4f} -> {item['treatment']:.4f}  "
              f"deg={deg_text}{ci_text}  {item['status']}")
    print("  gates: " + ", ".join(f"{g}={d['gates'][g]['status']}" for g in d["gates"]))

for reason, (left, right) in rejected.items():
    print(f"\nREJECTED {reason}\n  example: {left} -> {right}")
PY
echo "EXIT=0"
exit 0
