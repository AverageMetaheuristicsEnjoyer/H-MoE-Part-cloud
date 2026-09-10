#!/usr/bin/env bash
# Prepare, run, or clean a read-only Frugal checkpoint snapshot used to compare
# gradient-accumulation fusion within one Cloud image.
set -u

mode=${1:?usage: cloud_moe_wgrad_image_bench.sh prepare|bench|cleanup}
root=$(cd "$(dirname "$0")/.." && pwd)
arm=frugal_coord_bf16_state_fp32
iteration=2254
iter_dir=$(printf 'iter_%07d' "$iteration")
source_dir=${STAGE3_MOE_BENCH_SOURCE:-/workspace-SR006.nfs3/hmoe-checkpoints/frugal-slimadam-1c/1c/$arm}
snapshot_dir=${STAGE3_MOE_BENCH_SNAPSHOT:-/workspace-SR006.nfs3/hmoe-checkpoints/frugal-wgrad-bench/$arm}
log_root=${STAGE3_MOE_LOG_ROOT:-/workspace-SR006.nfs2/hmoe-cloud/pretrain}

validate_snapshot() {
  [[ -f $snapshot_dir/latest_checkpointed_iteration.txt ]] || return 1
  [[ $(cat "$snapshot_dir/latest_checkpointed_iteration.txt") == "$iteration" ]] || return 1
  [[ -d $snapshot_dir/$iter_dir ]] || return 1
}

case "$mode" in
  prepare)
    if validate_snapshot; then
      echo "SNAPSHOT_READY existing=$snapshot_dir iteration=$iteration"
      echo "EXIT=0"
      exit 0
    fi
    source_tracker="$source_dir/latest_checkpointed_iteration.txt"
    [[ -f $source_tracker && $(cat "$source_tracker") == "$iteration" && -d $source_dir/$iter_dir ]] || {
      echo "SNAPSHOT_FAIL source is not a complete iteration-$iteration checkpoint: $source_dir"
      echo "EXIT=1"
      exit 0
    }
    mkdir -p "$(dirname "$snapshot_dir")"
    partial="$snapshot_dir.partial"
    [[ ! -e $snapshot_dir && ! -e $partial ]] || {
      echo "SNAPSHOT_FAIL incomplete destination exists: $snapshot_dir"
      echo "EXIT=1"
      exit 0
    }
    mkdir "$partial"
    cp -al "$source_dir/$iter_dir" "$partial/"
    echo "$iteration" > "$partial/latest_checkpointed_iteration.txt"
    mv "$partial" "$snapshot_dir"
    validate_snapshot || { echo "SNAPSHOT_FAIL validation failed"; echo "EXIT=1"; exit 0; }
    echo "SNAPSHOT_READY created=$snapshot_dir iteration=$iteration"
    du -sh "$snapshot_dir"
    df -h "$snapshot_dir" | tail -1
    ;;
  bench)
    validate_snapshot || { echo "BENCH_FAIL snapshot missing: $snapshot_dir"; echo "EXIT=1"; exit 0; }
    echo "IMAGE=${MLSUB_IMAGE:-unknown}"
    nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
    nvidia_lib_path=$(find /home/user/conda/lib/python*/site-packages/nvidia \
      /home/jovyan/.local-torch28/lib/python*/site-packages/nvidia \
      -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: - || true)
    export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    python - <<'PY'
import importlib.util
import platform
import sys

import torch
import transformer_engine
import emerging_optimizers

print(f"PYTHON={platform.python_version()}")
print(f"TORCH={torch.__version__} CUDA={torch.version.cuda}")
print(f"TE={transformer_engine.__version__}")
print(f"APEX={importlib.util.find_spec('apex') is not None}")
print(f"AMP_C={importlib.util.find_spec('amp_C') is not None}")
print(f"FUSED_WGRAD={importlib.util.find_spec('fused_weight_gradient_mlp_cuda') is not None}")
print(f"EXECUTABLE={sys.executable}")
print(f"EMERGING_OPTIMIZERS={emerging_optimizers.__version__}")
PY
    env_exit=$?
    if (( env_exit != 0 )); then
      echo "BENCH_FAIL environment imports exit=$env_exit"
      echo "EXIT=1"
      exit 0
    fi
    find /home/user/conda /usr/local /opt -maxdepth 7 -type f \
      \( -name 'fused_weight_gradient_mlp_cuda*.so' -o -name 'amp_C*.so' \) \
      -print 2>/dev/null | sed 's/^/EXTENSION_FILE=/' | head -20

    export STAGE3_MOE_BENCH_LOAD=$snapshot_dir
    export STAGE3_MOE_BENCH_ITERS=${STAGE3_MOE_BENCH_ITERS:-50}
    export STAGE3_MOE_EVAL_ITERS=1
    export STAGE3_MOE_LOG_ROOT=$log_root
    export WANDB_MODE=offline
    result_files=()
    for fusion in 0 1; do
      if (( fusion == 0 )); then label=unfused; else label=fused; fi
      export STAGE3_MOE_WGRAD_FUSION=$fusion
      export STAGE3_MOE_RUN_SUFFIX="wgrad-${MLSUB_IMAGE:-unknown}-$label-v3"
      echo "=== BENCH image=${MLSUB_IMAGE:-unknown} fusion=$fusion label=$label ==="
      "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" resume-bench
      run_dir="$log_root/stage3-$arm-resume-bench-$STAGE3_MOE_RUN_SUFFIX"
      newest=$(ls -1t "$run_dir"/train-*.log 2>/dev/null | head -1)
      if [[ -n $newest ]]; then
        grep -E "successfully loaded checkpoint|iteration +[0-9]+/|Traceback|Error" "$newest" | tail -35
      fi
      result="$run_dir/results.jsonl"
      if [[ -f $result ]]; then
        result_files+=("$result")
      else
        echo "BENCH_FAIL no result for image=${MLSUB_IMAGE:-unknown} label=$label"
        echo "EXIT=1"
        exit 0
      fi
    done

    if (( ${#result_files[@]} != 2 )); then
      echo "BENCH_FAIL expected two result files, got ${#result_files[@]}"
      echo "EXIT=1"
      exit 0
    fi
    python - "${result_files[@]}" <<'PY'
import json
import sys
from pathlib import Path

rows = []
for path in map(Path, sys.argv[1:]):
    record = json.loads(path.read_text().splitlines()[-1])
    timing = record["measurement"]["timing"]
    rows.append((path, timing["full_step_seconds"], timing["tokens_per_second"]))
for path, seconds, tokens in rows:
    print(f"BENCH_RESULT path={path} step_seconds={seconds} tokens_per_second={tokens}")
speedup = rows[0][1] / rows[1][1]
print(f"FUSION_SPEEDUP={speedup:.6f} FUSION_STEP_REDUCTION={(1 - 1 / speedup) * 100:.3f}%")
PY
    ;;
  cleanup)
    if [[ ! -e $snapshot_dir ]]; then
      echo "SNAPSHOT_ALREADY_ABSENT=$snapshot_dir"
    elif validate_snapshot && [[ $(find "$snapshot_dir" -mindepth 1 -maxdepth 1 | wc -l) == 2 ]]; then
      rm -rf -- "$snapshot_dir"
      echo "SNAPSHOT_REMOVED=$snapshot_dir"
    else
      echo "SNAPSHOT_FAIL refusing unexpected layout: $snapshot_dir"
      echo "EXIT=1"
      exit 0
    fi
    ;;
  *)
    echo "unknown mode: $mode"
    echo "EXIT=1"
    exit 0
    ;;
esac

echo "EXIT=0"
exit 0
