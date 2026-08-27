#!/usr/bin/env bash
# Entry point for the MoE memory/time sweep on Cloud.ru.
#
#   mlsub run --repo https://github.com/AverageMetaheuristicsEnjoyer/H-MoE-Part-cloud \
#     --branch bench/fp8-membench --entry scripts/cloud_membench.sh --image torch28 \
#     --no-pip --gpus 1 --note membench-moe \
#     --args "--models 1p029b --arms adamw_bf16_state_fp32,adamw_bf16_state_fp8 --micro-batches 1,2,4"
#
# First argument may instead be:
#   export   print one line per recorded point, for `mlsub logs`
#   peek     print the newest log and which points exist so far
#   disk     report free space and probe every volume for a write
#
# A failed mlsub job shows no logs at all, so everything is teed to a persistent
# volume and this script always exits zero; the real status is the EXIT= line.
set -u

# /home/jovyan reached 0 bytes free on 2026-08-27 and the platform could not even
# create its own log symlinks there. nfs3 is the volume with room, so the sweep
# writes there and only reads the training data from /home/jovyan.
ROOT=${MEMBENCH_ROOT:-/workspace-SR006.nfs3/hmoe-membench}
RESULTS="$ROOT/results"
LOGS="$ROOT/logs"
RUNS="$ROOT/pretrain"
mkdir -p "$LOGS" "$RESULTS" "$RUNS"

case "${1:-}" in
  peek)
    echo "=== recorded points ==="
    ls -1 "$RESULTS/runs" 2>/dev/null | sort || echo "none yet"
    newest=$(ls -t "$LOGS"/*.log 2>/dev/null | head -1)
    echo "=== tail of ${newest:-no log} ==="
    [ -n "$newest" ] && tail -"${2:-150}" "$newest"
    exit 0
    ;;
  export)
    python3 stage3_moe/membench_sweep.py --export-only \
      --results-root "$RESULTS" --log-root "$RUNS"
    exit 0
    ;;
  disk)
    df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1
    df -i /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1
    echo "recorded points: $(ls -1 "$RESULTS/runs" 2>/dev/null | wc -l)"
    for target in "$RESULTS" /workspace-SR006.nfs2 /workspace-SR006.nfs3 /home/jovyan; do
      if probe=$(mktemp "$target/.membench-probe.XXXXXX" 2>&1); then
        echo "writable: $target"
        rm -f "$probe"
      else
        echo "NOT writable: $target ($probe)"
      fi
    done
    exit 0
    ;;
esac

LOG="$LOGS/$(date -u +%F_%H%M%S)-$$.log"

{
  echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "python: $(python3 -V 2>&1)"
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader || \
    echo "nvidia-smi unavailable"
  df -h "$ROOT" | tail -1

  # The launcher reads the corpus from here; a MoE step timed on mock data measures a
  # collapsed router instead of the model, so the real corpus is not optional.
  data_root=${STAGE3_MOE_DATA_ROOT:-/home/jovyan/data/fineweb-edu-gpt2-megatron/data}
  if [ ! -f "$data_root/train.bin" ]; then
    echo "FATAL: no training corpus at $data_root"
    exit 3
  fi
  echo "data: $data_root ($(stat -c %s "$data_root/train.bin") bytes)"

  export STAGE3_MOE_DATA_ROOT="$data_root"
  python3 stage3_moe/membench_sweep.py \
    --results-root "$RESULTS" \
    --log-root "$RUNS" \
    "$@"
} 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}

echo "EXIT=$status"
echo "log: $LOG"
echo "=== points ==="
python3 stage3_moe/membench_sweep.py --export-only \
  --results-root "$RESULTS" --log-root "$RUNS" 2>/dev/null || true
exit 0
