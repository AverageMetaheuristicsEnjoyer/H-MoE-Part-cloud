#!/usr/bin/env bash
# Re-pair result JSONLs already sitting on the persistent volume and print a compact
# summary.  mlsub only returns a truncated log tail, so verbose entrypoints lose their
# earlier pairs; this one prints a handful of lines per pair and needs no GPU.
#   mlsub run ... --entry scripts/cloud_dump_pairs.sh --gpus cpu --args "FILTER"
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
persist=/home/jovyan/hmoe-cloud/artifacts/stage3-moe-probes
filter=${1:-}

cd "$root"
export PYTHONPATH="$root"
python - "$persist" "$filter" <<'PY'
import json, subprocess, sys
from itertools import permutations
from pathlib import Path

root, filter_ = Path(sys.argv[1]), sys.argv[2]
files = sorted(p for p in root.glob("*/results.jsonl") if filter_ in str(p))
print(f"MATCHED {len(files)} result files for filter={filter_!r}")
for p in files:
    record = json.loads(p.read_text().splitlines()[0])
    m = record["measurement"]
    r = m["routing"]
    print(f"   {p.parent.name}")
    print(f"     kind={m['protocol']['kind']} warmup={m['protocol']['warmup_steps']}"
          f" measured={m['protocol']['measured_steps']}"
          f" inference={'set' if m['inference'] else 'null'}")
    print(f"     routing min/mean={r['minimum_to_mean']} max/mean={r['maximum_to_mean']}"
          f" cv={r['coefficient_of_variation']} dropped={r['dropped_tokens']}"
          f" artifact={(r['tokens_per_expert_artifact_sha256'] or 'null')[:12]}")

for a, b in permutations(files, 2):
    out = subprocess.run([sys.executable, "-m", "stage3_moe.pair_results", str(a), str(b)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        continue
    d = json.loads(out.stdout)
    e = d["effects"]
    print(f"\nPAIR axis={d['axis']} opt={d['optimizer']} verdict={d['verdict']}")
    print(f"  baseline={d['baseline_run_id']}")
    print(f"  treatment={d['treatment_run_id']}")
    print(f"  denom mb={d['denominators']['micro_batch_sequences_per_gpu']}"
          f" gb={d['denominators']['global_batch_sequences']}"
          f" tokens/step={d['denominators']['loss_tokens_per_step']}")
    for k in ("max_memory_allocated_bytes", "max_memory_reserved_bytes",
              "persistent_optimizer_state_bytes", "tokens_per_second",
              "full_step_seconds", "optimizer_step_seconds", "e2e_wct_seconds"):
        v = e[k]
        print(f"  {k:32s} base={v['baseline']:>16,.4f} treat={v['treatment']:>16,.4f} ratio={v['ratio']:.4f}")
PY
echo "EXIT=0"
exit 0
