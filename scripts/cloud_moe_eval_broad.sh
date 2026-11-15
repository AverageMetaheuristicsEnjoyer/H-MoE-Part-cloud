#!/usr/bin/env bash
# Broad exploratory screening on the six original final 1C endpoints.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
export STAGE3_MOE_EVAL_ROOTS=${STAGE3_MOE_EVAL_ROOTS:-/workspace-SR006.nfs2/hmoe-checkpoints/stage3-1c-mb4/1c:/workspace-SR006.nfs3/hmoe-checkpoints/stage3/trunk}
export STAGE3_MOE_EVAL_BUDGET=${STAGE3_MOE_EVAL_BUDGET:-1c}
export STAGE3_MOE_EVAL_TASKS=${STAGE3_MOE_EVAL_TASKS:-stage3_broad_v1}
export STAGE3_MOE_RUN_SUFFIX=${STAGE3_MOE_RUN_SUFFIX:-broad-v1-1c}
export STAGE3_MOE_LOG_ROOT=${STAGE3_MOE_LOG_ROOT:-/home/jovyan/hmoe-cloud/pretrain}
export STAGE3_MOE_HF_HOME=${STAGE3_MOE_HF_HOME:-/home/jovyan/hmoe-hf-cache-broad-v1}

exec "$root/scripts/cloud_moe_eval.sh" "$@"
