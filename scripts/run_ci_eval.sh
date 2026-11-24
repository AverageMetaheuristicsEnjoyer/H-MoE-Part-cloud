#!/bin/bash
# CI eval entry: set the launcher PYTHONPATH then run the proven baseline-eval runner.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root/third_party/coat:$root"
exec python scripts/lionmoe_baseline_eval_030.py "$@"
