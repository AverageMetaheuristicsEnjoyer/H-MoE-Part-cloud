#!/usr/bin/env bash
# Confirm what a pretraining run actually left behind: checkpoints, result record,
# logger output. Read-only.
#
#   ... --entry scripts/cloud_verify_pretrain.sh --args "SUBSTRING"
#
# With every run dir reported the output outgrew the platform's log limit and rows
# were silently dropped, so pass a substring to report only the matching runs.
set -u
want=${1:-}
ck=${STAGE3_MOE_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3}
lg=${STAGE3_MOE_LOG_ROOT:-/home/jovyan/hmoe-cloud/pretrain}
if [[ -z $want ]]; then
  echo "=== checkpoints ==="
  du -sh "$ck"/* 2>/dev/null | head
  find "$ck" -maxdepth 3 -name "latest_checkpointed_iteration.txt" -exec sh -c 'echo "$1 -> $(cat "$1")"' _ {} \; 2>/dev/null | head
  find "$ck" -maxdepth 3 -type d -name "iter_*" 2>/dev/null | head
  echo
fi
echo "=== volume ==="; df -h "$ck" | tail -1
echo
echo
if [ -f /home/jovyan/.wandb-key ]; then echo "=== wandb key: present ==="; else echo "=== wandb key: absent -> runs are offline ==="; fi
echo
echo "=== run dirs ==="; ls -la "$lg" 2>/dev/null | head
for d in "$lg"/*/; do
  [ -d "$d" ] || continue
  case "$d" in *"$want"*) ;; *) continue ;; esac
  echo "--- $d"
  ls -la "$d" | head -8
  # Whether a resume actually loaded is decided at startup, so it never shows up
  # in the tail the launcher prints.
  newest=$(ls -1t "$d"/train-*.log 2>/dev/null | head -1)
  if [ -n "$newest" ]; then
    echo "  LOG $newest"
    grep -iE "loading checkpoint|could not find|will not load|checkpoint .*at iteration|setting training iteration|does not match the optimizer|live group sizes" \
      "$newest" 2>/dev/null | head -6 | sed 's/^/    /'
    # A running job's own log is unreadable through the platform API, so the last
    # progress line here is the only way to see how far a live run has got.
    grep -E "iteration +[0-9]+/" "$newest" 2>/dev/null | tail -1 | cut -c1-420 | sed 's/^/    /'
    tr '\r' '\n' <"$newest" | grep -aE "Running (loglikelihood|loglikelihood_rolling) requests" \
      | tail -1 | cut -c1-420 | sed 's/^/    /'
    grep -aE "loss at iteration .* on (validation|test) set" "$newest" 2>/dev/null \
      | sed 's/^/    EVAL_RESULT /'
  fi
  # wandb names an offline run dir offline-run-*, an online one run-*; its size says
  # whether history is actually being recorded.
  du -sh "$d/wandb"/*run-* 2>/dev/null | tail -1 | sed 's/^/  WANDB /'
  du -sh "$d/tensorboard" 2>/dev/null | sed 's/^/  TB /'
  # A scoring run's numbers live here: the platform log window is far too small to hold
  # two arms' worth of MCore startup and still show them.
  if [ -f "$d/downstream/downstream.json" ]; then
    echo "  DOWNSTREAM_JSON $d"
    python - "$d/downstream/downstream.json" <<'PYD'
import json, sys
for item in json.load(open(sys.argv[1])):
    print(f"    {item['task']} {item['metric']}={item['value']:.6f}")
PYD
  fi
  # A run that left neither a record nor scores failed somewhere the launcher's own
  # tail never showed, because it exits 0 whatever the training process did.
  if [ ! -f "$d/results.jsonl" ] && [ ! -f "$d/downstream/downstream.json" ] && [ -n "$newest" ]; then
    echo "  NO OUTPUT, tail of $newest:"
    grep -aiE "error|traceback|assert|raise |killed|out of memory|fp8|cublas|invalid|dimension|align" "$newest" | tail -18 | sed 's/^/    /'
    tail -12 "$newest" | sed 's/^/    /'
  fi
  if [ -f "$d/results.jsonl" ]; then
    python - "$d/results.jsonl" <<'PYX'
import json, sys
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    m, n = d["measurement"], d["denominators"]
    t = m["timing"]
    step, full = t["optimizer_step_seconds"], t["full_step_seconds"]
    share = f"{step / full:.3f}" if step and full else "na"
    tok_s = "na" if t["tokens_per_second"] is None else f'{t["tokens_per_second"]:.0f}'
    print(f"  SUMMARY {d['arm_id']} mb={n['micro_batch_sequences_per_gpu']}"
          f" gb={n['global_batch_sequences']}"
          f" max_alloc={m['memory']['max_allocated_bytes']}"
          f" max_resvd={m['memory']['max_reserved_bytes']}"
          f" persistent={d['optimizer_state']['persistent_total_bytes']}"
          f" tok_s={tok_s}"
          f" opt_s={step} full_s={full} opt_share={share}")
PYX
  fi
done
echo "EXIT=0"
exit 0
