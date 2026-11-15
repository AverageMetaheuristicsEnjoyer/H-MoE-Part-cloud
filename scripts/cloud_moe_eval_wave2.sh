#!/usr/bin/env bash
# Exploratory Wave 2 on the six original final 1C endpoints.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
export STAGE3_MOE_EVAL_ROOTS=${STAGE3_MOE_EVAL_ROOTS:-/workspace-SR006.nfs2/hmoe-checkpoints/stage3-1c-mb4/1c:/workspace-SR006.nfs3/hmoe-checkpoints/stage3/trunk}
export STAGE3_MOE_EVAL_BUDGET=${STAGE3_MOE_EVAL_BUDGET:-1c}
export STAGE3_MOE_EVAL_TASKS=${STAGE3_MOE_EVAL_TASKS:-stage3_wave2}
export STAGE3_MOE_RUN_SUFFIX=${STAGE3_MOE_RUN_SUFFIX:-wave2-1c}
export STAGE3_MOE_LOG_ROOT=${STAGE3_MOE_LOG_ROOT:-/workspace-SR006.nfs2/hmoe-cloud/pretrain}

exec "$root/scripts/cloud_moe_eval.sh" "$@"
