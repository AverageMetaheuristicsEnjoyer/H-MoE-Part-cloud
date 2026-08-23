#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: CUDA_VISIBLE_DEVICES=N scripts/run_stage3_moe_probe.sh ARM mock|real [--protocol smoke|probe] [--warmup-steps N --measure-steps N]" >&2
}

arm=${1:-}
data_mode=${2:-}
if [[ -z "$arm" || -z "$data_mode" ]]; then
  usage
  exit 2
fi
shift 2

protocol=probe
warmup_override=
measure_override=
while (($#)); do
  case "$1" in
    --protocol)
      protocol=${2:?missing value for --protocol}
      shift 2
      ;;
    --warmup-steps)
      warmup_override=${2:?missing value for --warmup-steps}
      shift 2
      ;;
    --measure-steps)
      measure_override=${2:?missing value for --measure-steps}
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$protocol" in
  smoke)
    if [[ -n "$warmup_override" || -n "$measure_override" ]]; then
      echo "smoke protocol is fixed at 0 warmup + 1 measured step" >&2
      exit 2
    fi
    warmup_steps=0
    measure_steps=1
    ;;
  probe)
    warmup_steps=${warmup_override:-3}
    measure_steps=${measure_override:-5}
    ;;
  *)
    echo "unknown protocol: $protocol" >&2
    exit 2
    ;;
esac

if [[ ! "$warmup_steps" =~ ^[0-9]+$ || ! "$measure_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "warmup steps must be non-negative and measured steps must be positive" >&2
  exit 2
fi
train_iters=$((warmup_steps + measure_steps))

case "$data_mode" in
  mock|real) ;;
  *) echo "unknown data mode: $data_mode" >&2; exit 2 ;;
esac

root=$(cd "$(dirname "$0")/.." && pwd)
source "$root/configs/stage3-moe-1p029b.sh"

optimizer_args=()
compute_args=()
case "$arm" in
  adamw_bf16_state_fp32)
    optimizer=adam
    state_precision=fp32
    ;;
  adamw_bf16_state_fp8)
    optimizer=adam
    state_precision=fp8
    ;;
  muon_bf16_state_fp32)
    optimizer=muon
    state_precision=fp32
    optimizer_args+=("${STAGE3_MOE_MUON_ARGS[@]}")
    ;;
  muon_bf16_state_fp8)
    optimizer=muon
    state_precision=fp8
    optimizer_args+=("${STAGE3_MOE_MUON_ARGS[@]}")
    ;;
  adamw_fp8gemm_state_fp32)
    optimizer=adam
    state_precision=fp32
    compute_args+=(--fp8-format hybrid --fp8-recipe delayed)
    ;;
  muon_fp8gemm_state_fp32)
    optimizer=muon
    state_precision=fp32
    optimizer_args+=("${STAGE3_MOE_MUON_ARGS[@]}")
    compute_args+=(--fp8-format hybrid --fp8-recipe delayed)
    ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

if ((${#compute_args[@]})); then
  if [[ -n ${STAGE3_MOE_FP8_AMAX_HISTORY_LEN:-} ]]; then
    compute_args+=(--fp8-amax-history-len "$STAGE3_MOE_FP8_AMAX_HISTORY_LEN")
  fi
  if [[ -n ${STAGE3_MOE_FP8_AMAX_COMPUTE_ALGO:-} ]]; then
    compute_args+=(--fp8-amax-compute-algo "$STAGE3_MOE_FP8_AMAX_COMPUTE_ALGO")
  fi
fi

data_args=()
case "$data_mode" in
  mock)
    data_args+=(--mock-data --split 100,0,0)
    ;;
  real)
    data_root=${STAGE3_MOE_DATA_ROOT:-/workspace/data/fineweb-edu-public/data}
    data_args+=(--data-path 1 "$data_root/train" --split 100,0,0)
    ;;
esac

if [[ ! ${CUDA_VISIBLE_DEVICES:-} =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "set CUDA_VISIBLE_DEVICES to one numeric device index, or a comma-separated list" >&2
  exit 2
fi
IFS=',' read -r -a selected_devices <<<"$CUDA_VISIBLE_DEVICES"
gpu_count=${#selected_devices[@]}

python_bin=${STAGE3_MOE_PYTHON:-python}
runtime_prefix=(env)
if [[ ${MLSUB_IMAGE:-} == torch28 ]]; then
  stage3_site=cloudru
  stage3_image=torch28
  export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root${PYTHONPATH:+:$PYTHONPATH}"
elif [[ -x "$root/scripts/node207_env.sh" ]]; then
  stage3_site=node207
  stage3_image=node207-pytorch-26.04-be06a21b
  runtime_prefix=(
    "$root/scripts/node207_env.sh" env
    TRITON_LIBCUDA_PATH=/.singularity.d/libs
    TRITON_CACHE_DIR=/tmp/triton-stage3
    PYTHONPATH=/workspace/third_party/Megatron-LM:/workspace/third_party/emerging-optimizers:/workspace
  )
else
  stage3_site=${STAGE3_MOE_SITE:-direct}
  stage3_image=${STAGE3_MOE_IMAGE:-direct-source-runtime}
  export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root${PYTHONPATH:+:$PYTHONPATH}"
fi

if [[ -e "$root/third_party/emerging-optimizers/.git" ]]; then
  eo_commit=$(git -C "$root/third_party/emerging-optimizers" rev-parse HEAD)
  if [[ "$eo_commit" != "$STAGE3_MOE_EO_COMMIT" ]]; then
    echo "Emerging Optimizers commit mismatch: $eo_commit != $STAGE3_MOE_EO_COMMIT" >&2
    exit 2
  fi
  eo_commit_status=verified
else
  vendored_eo_tree=$(git -C "$root" rev-parse HEAD:third_party/emerging-optimizers)
  if [[ "$vendored_eo_tree" != "$STAGE3_MOE_VENDORED_EO_TREE" ]]; then
    echo "vendored Emerging Optimizers tree mismatch: $vendored_eo_tree != $STAGE3_MOE_VENDORED_EO_TREE" >&2
    exit 2
  fi
  eo_commit_status=vendored-tree-verified
fi

if [[ -e "$root/third_party/Megatron-LM/.git" ]]; then
  mcore_commit=$(git -C "$root/third_party/Megatron-LM" rev-parse HEAD)
  if [[ "$mcore_commit" != "$STAGE3_MOE_MCORE_COMMIT" ]]; then
    echo "MCore commit mismatch: $mcore_commit != $STAGE3_MOE_MCORE_COMMIT" >&2
    exit 2
  fi
  mcore_commit_status=verified
else
  vendored_mcore_tree=$(git -C "$root" rev-parse HEAD:third_party/Megatron-LM)
  if [[ "$vendored_mcore_tree" != "$STAGE3_MOE_VENDORED_MCORE_TREE" ]]; then
    echo "vendored MCore tree mismatch: $vendored_mcore_tree != $STAGE3_MOE_VENDORED_MCORE_TREE" >&2
    exit 2
  fi
  mcore_commit_status=vendored-tree-verified
fi

gpu_clean_before=0
gpu_clean_during=0
selected_gpu_uuid=${STAGE3_MOE_GPU_UUID:-unverified}
if [[ ${STAGE3_MOE_DRY_RUN:-0} == 1 ]]; then
  echo "GPU_PREFLIGHT=dry-run"
else
  # Every requested device must clear the gate, not just the first one.
  selected_uuids=()
  for device in "${selected_devices[@]}"; do
    gpu_line=$(nvidia-smi --query-gpu=index,uuid,memory.used,driver_version --format=csv,noheader,nounits | awk -F, -v selected="$device" '
      {for (i=1; i<=NF; i++) {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)}}
      $1 == selected {print $2, $3, $4}
    ')
    read -r device_uuid device_memory_mib driver_version <<<"$gpu_line"
    if [[ -z ${device_uuid:-} ]]; then
      echo "selected GPU index $device was not reported by nvidia-smi" >&2
      exit 2
    fi
    compute_pids=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F, -v selected="$device_uuid" '
      {for (i=1; i<=NF; i++) {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)}}
      $1 == selected {print $2}
    ')
    echo "GPU_PREFLIGHT index=$device uuid=$device_uuid memory_used_mib=$device_memory_mib driver=$driver_version"
    if [[ -n "$compute_pids" ]]; then
      echo "refusing to use GPU $device; active compute PIDs: ${compute_pids//$'\n'/,}" >&2
      exit 3
    fi
    if ((device_memory_mib > 1024)); then
      echo "refusing to use GPU $device; idle memory threshold exceeded: ${device_memory_mib} MiB > 1024 MiB" >&2
      exit 3
    fi
    selected_uuids+=("$device_uuid")
  done
  selected_gpu_uuid=$(IFS=,; echo "${selected_uuids[*]}")
  gpu_clean_before=1
  gpu_clean_during=1
fi

fusion_args=()
if "${runtime_prefix[@]}" "$python_bin" -c 'import fused_weight_gradient_mlp_cuda' >/dev/null 2>&1; then
  grad_accum_fusion=enabled
else
  grad_accum_fusion=disabled
  fusion_args+=(--no-gradient-accumulation-fusion)
fi

run_id=${STAGE3_MOE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${arm}-${data_mode}-${protocol}}
result_path=${STAGE3_MOE_RESULTS_JSONL:-artifacts/stage3-moe-probes/$run_id/results.jsonl}
inprocess_result_path=${result_path}.inprocess.$$
config_sha256=$(sha256sum "$root/configs/stage3-moe-1p029b.sh" | awk '{print $1}')
if [[ "$data_mode" == mock ]]; then
  data_manifest_sha256=$(printf '%s' 'mcore-mock-data-seed-1234-vocab-50257-seq-2048' | sha256sum | awk '{print $1}')
else
  data_manifest_sha256=${STAGE3_MOE_DATA_MANIFEST_SHA256:?set STAGE3_MOE_DATA_MANIFEST_SHA256 for real data}
fi
match_key_sha256=$(printf '%s\0' \
  stage3-moe-1p029b-v1 "$data_mode" "$protocol" "$warmup_steps" "$measure_steps" \
  "$STAGE3_MOE_TOTAL_PARAMETERS" "$STAGE3_MOE_ACTIVE_PARAMETERS" "$config_sha256" \
  "$data_manifest_sha256" "$grad_accum_fusion" | sha256sum | awk '{print $1}')
driver_version=${driver_version:-${STAGE3_MOE_DRIVER:-unknown}}

cmd=(
  "${runtime_prefix[@]}"
  "STAGE3_MOE_RUN_ID=$run_id"
  "STAGE3_MOE_RESULTS_JSONL=$result_path"
  "STAGE3_MOE_SITE=$stage3_site"
  "STAGE3_MOE_IMAGE=$stage3_image"
  "MLSUB_IMAGE=${MLSUB_IMAGE:-$stage3_image}"
  "STAGE3_MOE_MATCH_KEY_SHA256=$match_key_sha256"
  "STAGE3_MOE_CONFIG_SHA256=$config_sha256"
  "STAGE3_MOE_DATA_MANIFEST_SHA256=$data_manifest_sha256"
  "STAGE3_MOE_DRIVER=$driver_version"
  "STAGE3_MOE_MCORE_COMMIT=$STAGE3_MOE_MCORE_COMMIT"
  "STAGE3_MOE_EO_COMMIT=$STAGE3_MOE_EO_COMMIT"
  "STAGE3_MOE_GPU_UUID=$selected_gpu_uuid"
  "STAGE3_MOE_GPU_CLEAN_BEFORE=$gpu_clean_before"
  "STAGE3_MOE_GPU_CLEAN_DURING=$gpu_clean_during"
  "STAGE3_MOE_GPU_CLEAN_AFTER=0"
  "$python_bin" -m torch.distributed.run --standalone --nproc-per-node "$gpu_count"
  stage3_moe/pretrain_gpt.py
  --stage3-arm "$arm"
  --stage3-result-path "$inprocess_result_path"
  --stage3-warmup-steps "$warmup_steps"
  --stage3-measure-steps "$measure_steps"
  --optimizer-state-precision "$state_precision"
  "${STAGE3_MOE_MODEL_ARGS[@]}"
  "${STAGE3_MOE_ROUTER_ARGS[@]}"
  "${STAGE3_MOE_PARALLEL_ARGS[@]}"
  "${STAGE3_MOE_TRAINING_ARGS[@]}"
  --optimizer "$optimizer"
  "${optimizer_args[@]}"
  "${compute_args[@]}"
  "${fusion_args[@]}"
  "${data_args[@]}"
  --train-iters "$train_iters"
)

echo "STAGE3_MOE_RUN_ID=$run_id"
echo "STAGE3_MOE_RESULTS_JSONL=$result_path"
echo "STAGE3_MOE_SITE=$stage3_site STAGE3_MOE_IMAGE=$stage3_image"
echo "MCORE_COMMIT=$STAGE3_MOE_MCORE_COMMIT STATUS=$mcore_commit_status"
echo "EO_COMMIT=$STAGE3_MOE_EO_COMMIT STATUS=$eo_commit_status"
echo "ARM=$arm DATA=$data_mode PROTOCOL=$protocol WARMUP_STEPS=$warmup_steps MEASURE_STEPS=$measure_steps TRAIN_ITERS=$train_iters"
echo "DENOMINATORS gpu=$gpu_count micro_batch=${STAGE3_MOE_MICRO_BATCH:-1} global_batch=${STAGE3_MOE_GLOBAL_BATCH:-1} sequence_length=2048 total_parameters=$STAGE3_MOE_TOTAL_PARAMETERS active_parameters=$STAGE3_MOE_ACTIVE_PARAMETERS"
echo "GRADIENT_ACCUMULATION_FUSION=$grad_accum_fusion"

cd "$root"
if [[ ${STAGE3_MOE_DRY_RUN:-0} == 1 ]]; then
  printf 'COMMAND='
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

if [[ -e "$result_path" || -e "$inprocess_result_path" ]]; then
  echo "refusing to overwrite an existing result: $result_path" >&2
  exit 2
fi

host_python=${STAGE3_MOE_HOST_PYTHON:-python3}
wct_file=$(mktemp "${TMPDIR:-/tmp}/stage3-moe-wct.XXXXXX")
set +e
"$host_python" - "$wct_file" "${cmd[@]}" <<'PY'
import subprocess
import sys
import time
from pathlib import Path

started = time.perf_counter()
completed = subprocess.run(sys.argv[2:])
elapsed = time.perf_counter() - started
Path(sys.argv[1]).write_text(f"{completed.returncode} {elapsed:.9f}\n")
raise SystemExit(completed.returncode)
PY
outer_code=$?
set -e
if ! read -r training_code e2e_wct_seconds <"$wct_file"; then
  echo "outer launcher failed to record child exit and monotonic WCT" >&2
  rm -f -- "$wct_file"
  exit "$outer_code"
fi
rm -f -- "$wct_file"

if ! post_gpu_line=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits | awk -F, -v selected="$CUDA_VISIBLE_DEVICES" '
    {for (i=1; i<=NF; i++) {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)}}
    $1 == selected {print $2, $3}
  '); then
  post_gpu_line=
fi
read -r post_gpu_uuid post_gpu_memory_mib <<<"$post_gpu_line"
if ! post_compute_pids=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F, -v selected="$selected_gpu_uuid" '
    {for (i=1; i<=NF; i++) {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)}}
    $1 == selected {print $2}
  '); then
  post_compute_pids=query-failed
fi
gpu_clean_after=0
if [[ "$post_gpu_uuid" == "$selected_gpu_uuid" && -z "$post_compute_pids" && "$post_gpu_memory_mib" =~ ^[0-9]+$ ]] && ((post_gpu_memory_mib <= 1024)); then
  gpu_clean_after=1
fi
echo "GPU_POSTFLIGHT index=$CUDA_VISIBLE_DEVICES uuid=${post_gpu_uuid:-unreported} memory_used_mib=${post_gpu_memory_mib:-unreported} compute_pids=${post_compute_pids//$'\n'/,} clean=$gpu_clean_after"
echo "E2E_WCT_SECONDS=$e2e_wct_seconds SCOPE=launcher_start_to_process_exit"

finalize_code=0
if [[ -f "$inprocess_result_path" ]]; then
  "$host_python" - "$inprocess_result_path" "$result_path" "$e2e_wct_seconds" "$gpu_clean_after" <<'PY' || finalize_code=$?
import json
import os
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
if len(records) != 1 or records[0].get("record_type") != "run":
    raise SystemExit(f"expected one run record in {source}, found {len(records)}")
record = records[0]
record["measurement"]["timing"]["e2e_wct_seconds"] = float(sys.argv[3])
record["measurement"]["timing"]["e2e_wct_scope"] = "launcher_start_to_process_exit"
record["environment"]["gpu_clean"]["after"] = sys.argv[4] == "1"
temporary = target.with_name(f".{target.name}.finalizing.{os.getpid()}")
with temporary.open("x") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
try:
    os.link(temporary, target)
finally:
    temporary.unlink(missing_ok=True)
source.unlink()
PY
else
  echo "training produced no in-process result: $inprocess_result_path" >&2
  finalize_code=1
fi

if ((training_code != 0)); then
  exit "$training_code"
fi
if ((finalize_code != 0 || gpu_clean_after != 1)); then
  exit 3
fi
