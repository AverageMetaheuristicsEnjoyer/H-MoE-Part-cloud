#!/bin/bash
# CI eval entry: set the launcher PYTHONPATH then run the proven baseline-eval runner.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root/third_party/coat:$root"
if [ "${1:-}" = probe ]; then
  D=${HMOE_PROBE_DIR:?probe dir}
  echo "== ls $D"; ls -la "$D" 2>&1 | head -30
  for f in partial-error.json eval-wrapper.stderr.log eval-wrapper.stdout.log progress.json result.json; do
    echo "== $f"; tail -40 "$D/$f" 2>&1
  done
  exit 0
fi
python "$root/scripts/review_stage_data.py" >/dev/null || { echo "DATA_FAIL"; exit 0; }
exec python scripts/lionmoe_baseline_eval_030.py "$@"
