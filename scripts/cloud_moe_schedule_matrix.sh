#!/usr/bin/env bash
# AdamW BF16-GEMM schedule controls from the native iteration-13,794 checkpoints.
set -uo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
mode=${1:?usage: cloud_moe_schedule_matrix.sh inventory|preflight|run [ARM] [short|long]}
arm=${2:-}
schedule=${3:-}

hf_repo=${STAGE3_MOE_HF_REPO:-AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints}
hf_source_prefix=${STAGE3_MOE_HF_SOURCE_PREFIX:-1c-mb4}
hf_output_prefix=${STAGE3_MOE_HF_OUTPUT_PREFIX:-schedule-matrix-adamw-v1}
token_file=${STAGE3_MOE_HF_TOKEN_FILE:-/home/jovyan/.cache/huggingface/token}
extension_root=${STAGE3_MOE_EXTENSION_ROOT:-/workspace-SR006.nfs2/hmoe-data/fineweb-edu-time-match-extension}
log_root=${STAGE3_MOE_LOG_ROOT:-/workspace-SR006.nfs3/hmoe-cloud/pretrain}
evidence_root=${STAGE3_MOE_SCHEDULE_EVIDENCE_ROOT:-/workspace-SR006.nfs3/hmoe-cloud/schedule-matrix-adamw-v1}

check_arm() {
  case "$arm" in
    adamw_bf16_state_fp32|adamw_bf16_state_fp8) ;;
    *) echo "unsupported schedule-matrix arm: $arm" >&2; return 2 ;;
  esac
}

inventory() {
  echo "SCHEDULE_MATRIX_INVENTORY commit=$(git -C "$root" rev-parse HEAD)"
  echo "=== FILESYSTEM BYTES ==="
  df -h /tmp /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3
  echo "=== FILESYSTEM INODES ==="
  df -i /tmp /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3
  echo "=== CREDENTIAL PRESENCE ==="
  [[ -s $token_file ]] && echo "hf-token-present path=$token_file" || echo "hf-token-missing path=$token_file"
  [[ -s /home/jovyan/.wandb-key || -n ${WANDB_API_KEY:-} ]] && echo "wandb-key-present" || echo "wandb-key-missing"
  echo "=== EXTENSION DATA ==="
  for path in "$extension_root/data/train.bin" "$extension_root/data/train.idx" "$extension_root/artifact-manifest.json"; do
    [[ -f $path ]] || { echo "MISSING $path"; return 2; }
    stat -c 'DATA_FILE size=%s inode=%i path=%n' "$path"
  done
  sha256sum "$extension_root/data/train.idx" "$extension_root/artifact-manifest.json"
  echo "=== LOCAL CHECKPOINTS ==="
  for name in adamw_bf16_state_fp32 adamw_bf16_state_fp8; do
    local_path="/workspace-SR006.nfs2/hmoe-checkpoints/stage3-1c-mb4/1c/$name/iter_0013794"
    [[ -d $local_path ]] && du -sh "$local_path" || echo "LOCAL_SOURCE_MISSING arm=$name path=$local_path"
  done
  local_fp8gemm=/workspace-SR006.nfs3/hmoe-checkpoints/stage3/trunk/adamw_fp8gemm_state_fp32/iter_0013794
  [[ -d $local_fp8gemm ]] && du -sh "$local_fp8gemm" || echo "LOCAL_FP8GEMM_SOURCE_MISSING path=$local_fp8gemm"
  echo "=== HF SOURCE INVENTORY ==="
  python - "$hf_repo" "$hf_source_prefix" "$token_file" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo, prefix, token_path = sys.argv[1:]
token = Path(token_path).read_text().strip() if Path(token_path).is_file() else None
api = HfApi(token=token)
info = api.model_info(repo)
print(f"HF_REPO repo={repo} private={info.private} access={'authenticated' if token else 'anonymous'}")
for arm in ("adamw_bf16_state_fp32", "adamw_bf16_state_fp8"):
    remote = f"{prefix}/{arm}/iter_0013794"
    entries = list(api.list_repo_tree(repo, path_in_repo=remote, repo_type="model", recursive=True))
    files = [(entry.path, entry.size) for entry in entries if getattr(entry, "size", None) is not None]
    if not files:
        raise RuntimeError(f"HF source is empty: {remote}")
    print(f"HF_SOURCE arm={arm} path={remote} files={len(files)} bytes={sum(size for _, size in files)}")
    for path, size in files:
        print(f"HF_FILE arm={arm} size={size} path={path}")
PY
  echo "INVENTORY_EXIT=0"
}

prepare_workdir() {
  work=$(mktemp -d /tmp/stage3-schedule-matrix.XXXXXX)
  source_dir="$work/source/$hf_source_prefix/$arm"
  checkpoint_file="$source_dir/iter_0013794/mp_rank_00/model_optim_rng.pt"
  mkdir -p "$work/checkpoints"
  echo "WORKDIR=$work"
}

check_gpu_prerequisites() {
  nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
  unset PYTHONNOUSERSITE
  nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
    -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: - || true)
  export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  export CUDNN_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cudnn
  export CURAND_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/curand
  export NVRTC_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cuda_nvrtc
  for path in "$extension_root/data/train.bin" "$extension_root/data/train.idx" "$extension_root/artifact-manifest.json"; do
    [[ -f $path ]] || { echo "schedule-matrix prerequisite missing: $path" >&2; return 2; }
  done
  available_kb=$(df -Pk /tmp | awk 'END {print $4}')
  [[ $available_kb -ge 25000000 ]] || {
    echo "GPU-local /tmp needs at least 25,000,000 KiB free: available=$available_kb" >&2
    return 2
  }
  echo "STORAGE checkpoint_root=$work/checkpoints log_root=$log_root evidence_root=$evidence_root"
  df -h /tmp /workspace-SR006.nfs2 /workspace-SR006.nfs3
  df -i /tmp /workspace-SR006.nfs2 /workspace-SR006.nfs3
}

download_source() {
  mkdir -p "$work/source"
  python - "$hf_repo" "$hf_source_prefix" "$arm" "$token_file" "$work/source" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

repo, prefix, arm, token_path, destination = sys.argv[1:]
token = Path(token_path).read_text().strip() if Path(token_path).is_file() else None
remote = f"{prefix}/{arm}/iter_0013794"
snapshot_download(
    repo_id=repo,
    repo_type="model",
    allow_patterns=f"{remote}/**",
    local_dir=destination,
    token=token,
)
api = HfApi(token=token)
entries = list(api.list_repo_tree(repo, path_in_repo=remote, repo_type="model", recursive=True))
remote_files = {
    entry.path.removeprefix(remote + "/"): entry.size
    for entry in entries
    if getattr(entry, "size", None) is not None
}
local = Path(destination) / remote
local_files = {str(path.relative_to(local)): path.stat().st_size for path in local.rglob("*") if path.is_file()}
if remote_files != local_files:
    raise RuntimeError(f"download size mismatch: remote={remote_files} local={local_files}")
print(
    f"HF_DOWNLOAD_VERIFIED arm={arm} access={'authenticated' if token else 'anonymous'} "
    f"files={len(local_files)} bytes={sum(local_files.values())} source={local}"
)
PY
  [[ -f $checkpoint_file ]] || { echo "downloaded checkpoint file is missing: $checkpoint_file" >&2; return 2; }
  echo 13794 > "$source_dir/latest_checkpointed_iteration.txt"
}

audit_source_checkpoint() {
  source_audit="$work/source-audit.json"
  PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root" \
    python - "$checkpoint_file" "$arm" "$source_audit" "$hf_repo" "$hf_source_prefix" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

path, arm, output, hf_repo, hf_prefix = sys.argv[1:]
checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
required = {"model", "optimizer", "opt_param_scheduler", "rng_state"}
missing = sorted(required - checkpoint.keys())
if missing:
    raise RuntimeError(f"checkpoint keys missing: {missing}")
iteration = checkpoint.get("iteration")
consumed = getattr(checkpoint["args"], "consumed_train_samples", None)
if iteration != 13794 or consumed != 13794 * 208:
    raise RuntimeError(f"wrong checkpoint position: iteration={iteration} consumed={consumed}")
model = checkpoint["model"]
bias_keys = sorted(key for key in model if key.endswith("expert_bias"))
biases = [model[key].detach().float().reshape(-1) for key in bias_keys]
flat = torch.cat(biases)
if len(bias_keys) != 17 or flat.numel() != 1088 or torch.count_nonzero(flat).item() != 1088:
    raise RuntimeError("router expert-bias contract failed")

def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)

dtypes = Counter(str(tensor.dtype) for tensor in tensors(checkpoint["optimizer"]))
float8 = sum(count for dtype, count in dtypes.items() if "float8" in dtype)
if arm.endswith("_state_fp8") and not float8:
    raise RuntimeError("FP8-state checkpoint has no FP8 optimizer tensors")
if arm.endswith("_state_fp32") and float8:
    raise RuntimeError("FP32-state checkpoint contains FP8 optimizer tensors")
scheduler = checkpoint["opt_param_scheduler"]
record = {
    "arm": arm,
    "hf_repo": hf_repo,
    "hf_path": f"{hf_prefix}/{arm}/iter_0013794",
    "checkpoint_file": str(Path(path)),
    "checkpoint_bytes": Path(path).stat().st_size,
    "iteration": iteration,
    "consumed_train_samples": consumed,
    "optimizer_present": True,
    "scheduler_present": True,
    "rng_present": True,
    "scheduler_num_steps": scheduler.get("num_steps"),
    "scheduler_max_lr": scheduler.get("max_lr"),
    "scheduler_min_lr": scheduler.get("min_lr"),
    "router_bias_keys": len(bias_keys),
    "router_bias_values": flat.numel(),
    "router_bias_nonzero": torch.count_nonzero(flat).item(),
    "router_bias_sha256": hashlib.sha256(flat.numpy().tobytes()).hexdigest(),
    "optimizer_tensor_dtypes": dict(sorted(dtypes.items())),
}
Path(output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(
    f"SOURCE_CHECKPOINT arm={arm} iteration={iteration} consumed={consumed} "
    f"optimizer=1 scheduler=1 rng=1 router_bias_keys={len(bias_keys)} "
    f"router_bias_nonzero={record['router_bias_nonzero']} "
    f"router_bias_sha256={record['router_bias_sha256']} dtypes={dict(dtypes)}"
)
PY
}

export_run_environment() {
  mkdir -p "$work/triton-cache" "$work/data-cache"
  export TRITON_CACHE_DIR="$work/triton-cache"
  export STAGE3_MOE_DATA_CACHE_PATH="$work/data-cache"
  export STAGE3_MOE_MICRO_BATCH=16
  export STAGE3_MOE_PROPAGATE_EXIT=1
  export STAGE3_MOE_CKPT_ROOT="$work/checkpoints"
  export STAGE3_MOE_TRAIN_DATA_PREFIX="$extension_root/data/train"
  export STAGE3_MOE_DATA_MANIFEST_PATH="$extension_root/artifact-manifest.json"
  export STAGE3_MOE_SCHEDULE_LOAD="$source_dir"
  export STAGE3_MOE_LOG_ROOT="$log_root"
  export STAGE3_MOE_WANDB_ENTITY=andrey
  export WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb-radfan.ru}
}

validate_smoke() {
  local smoke_schedule=$1
  local suffix="schedule-matrix-preflight-$smoke_schedule-v1"
  local run_dir="$log_root/stage3-$arm-schedule-tail-smoke-$suffix"
  local output="$evidence_root/preflight/$arm-$smoke_schedule.json"
  python - "$run_dir" "$arm" "$smoke_schedule" "$source_audit" "$output" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

run_dir, arm, schedule, source_audit, output = sys.argv[1:]
logs = sorted(Path(run_dir).glob("train-*.log"))
if not logs:
    raise RuntimeError(f"preflight log missing: {run_dir}")
text = logs[-1].read_text(errors="replace")
if "successfully loaded checkpoint" not in text or "at iteration 13794" not in text:
    raise RuntimeError("runtime checkpoint load was not confirmed")
for failure in ("Unable to load optimizer", "Unable to load rng state"):
    if failure in text:
        raise RuntimeError(failure)
if "train:      208" not in text:
    raise RuntimeError("one-step preflight did not build exactly one extension batch")
contracts = {
    "global_batch_size": "208",
    "phase_transition_iterations": "[13794]",
    "train_iters": "13795",
    "eval_interval": "1",
    "lr_decay_iters": "19570",
    "lr_wsd_decay_iters": "3448" if schedule == "short" else "5776",
}
for name, expected_value in contracts.items():
    match = re.search(rf"^\s*{re.escape(name)}\s+\.{{2,}}\s+(.+?)\s*$", text, re.MULTILINE)
    if match is None or match.group(1) != expected_value:
        raise RuntimeError(f"runtime contract mismatch for {name}: {None if match is None else match.group(1)!r}")
matches = re.findall(r"iteration\s+(\d+)/\s*\d+.*?learning rate:\s*([0-9.E+-]+)", text)
first = next(((int(step), float(lr)) for step, lr in matches if int(step) >= 13795), None)
if first is None or first[0] != 13795:
    raise RuntimeError(f"first resumed progress row is wrong: {first}")
decay = 3448 if schedule == "short" else 5776
start = 19570 - decay
if first[0] <= start:
    expected = 1.63e-3
else:
    ratio = (first[0] - start) / decay
    expected = 1.63e-4 + (2.0 * math.pow(0.5, ratio) - 1.0) * (1.63e-3 - 1.63e-4)
if not math.isclose(first[1], expected, rel_tol=2e-7, abs_tol=1e-10):
    raise RuntimeError(f"first LR mismatch: actual={first[1]} expected={expected}")
record = {
    "arm": arm,
    "schedule": schedule,
    "source": json.loads(Path(source_audit).read_text()),
    "runtime_checkpoint_load": True,
    "runtime_optimizer_load": True,
    "runtime_rng_load": True,
    "phase_transition_iteration": 13794,
    "extension_local_consumed_samples": 0,
    "extension_phase_samples": 5776 * 208,
    "micro_batch": 16,
    "global_batch": 208,
    "first_iteration": first[0],
    "first_learning_rate": first[1],
    "expected_first_learning_rate": expected,
    "target_iteration": 19570,
    "final_learning_rate": 1.63e-4,
    "log": str(logs[-1]),
}
Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(
    f"PREFLIGHT_RESULT arm={arm} schedule={schedule} iteration={first[0]} "
    f"first_lr={first[1]:.10g} final_target=19570 final_lr=0.000163 "
    f"local_consumed_samples=0 optimizer_rng_load=pass output={output}"
)
PY
}

preflight() {
  check_arm
  prepare_workdir
  check_gpu_prerequisites
  download_source
  audit_source_checkpoint
  export_run_environment
  for smoke_schedule in short long; do
    suffix="schedule-matrix-preflight-$smoke_schedule-v1"
    STAGE3_MOE_SCHEDULE="$smoke_schedule" \
    STAGE3_MOE_RUN_SUFFIX="$suffix" \
    STAGE3_MOE_LOG_INTERVAL=1 \
    STAGE3_MOE_EVAL_INTERVAL=1 \
    STAGE3_MOE_EVAL_ITERS=1 \
    WANDB_MODE=disabled \
      "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" schedule-tail-smoke
    validate_smoke "$smoke_schedule"
  done
  if find "$work/checkpoints" -type d -name 'iter_*' -print -quit | grep -q .; then
    echo "preflight unexpectedly wrote a checkpoint" >&2
    return 2
  fi
  rm -rf "$work"
  echo "PREFLIGHT_EXIT=0 arm=$arm"
}

upload_endpoint() {
  local endpoint=$1
  local archive_json=$2
  python - "$hf_repo" "$hf_output_prefix" "$arm" "$schedule" "$token_file" "$endpoint" "$archive_json" <<'PY'
import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

repo, prefix, arm, schedule, token_path, endpoint, output = sys.argv[1:]
token = Path(token_path).read_text().strip()
api = HfApi(token=token)
local = Path(endpoint)
remote = f"{prefix}/{arm}/{schedule}/iter_0019570"
files = {str(path.relative_to(local)): path.stat().st_size for path in local.rglob("*") if path.is_file()}
if not files:
    raise RuntimeError(f"endpoint is empty: {endpoint}")
started = time.time()
api.upload_folder(folder_path=str(local), path_in_repo=remote, repo_id=repo, repo_type="model")
entries = list(api.list_repo_tree(repo, path_in_repo=remote, repo_type="model", recursive=True))
remote_files = {
    entry.path.removeprefix(remote + "/"): entry.size
    for entry in entries
    if getattr(entry, "size", None) is not None
}
if files != remote_files:
    raise RuntimeError(f"HF endpoint verification failed: local={files} remote={remote_files}")
record = {
    "repo": repo,
    "remote_path": remote,
    "files": files,
    "bytes": sum(files.values()),
    "upload_seconds": time.time() - started,
    "verified": True,
}
Path(output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(f"HF_ENDPOINT_VERIFIED arm={arm} schedule={schedule} path={remote} files={len(files)} bytes={sum(files.values())}")
PY
}

write_final_report() {
  local health_json=$1
  local eval_log=$2
  local routing_json=$3
  local archive_json=$4
  local report=$5
  local wandb_run_id=$6
  python - "$arm" "$schedule" "$source_audit" "$health_json" "$eval_log" "$routing_json" "$archive_json" "$report" "$wandb_run_id" <<'PY'
import json
import re
import sys
from pathlib import Path

arm, schedule, source, health, eval_log, routing, archive, output, wandb_run_id = sys.argv[1:]
text = Path(eval_log).read_text(errors="replace")
values = {"validation": [], "test": []}
pattern = re.compile(r"MATCHED_RESULT .*?repeat=(\d+).*?on (validation|test) set.*?lm loss value:\s*([0-9.E+-]+)")
for _, split, value in pattern.findall(text):
    values[split].append(float(value))
for split in values:
    if len(values[split]) != 2:
        raise RuntimeError(f"fixed evaluation count for {split} is {len(values[split])}, expected 2")
    if abs(values[split][0] - values[split][1]) > 1e-9:
        raise RuntimeError(f"fixed evaluation repeats differ for {split}: {values[split]}")
health_record = json.loads(Path(health).read_text())["runs"][0]
if health_record["skipped_iterations"] or health_record["nan_iterations"]:
    raise RuntimeError("training health contains skipped or NaN iterations")
record = {
    "schema_version": 1,
    "arm": arm,
    "schedule": schedule,
    "precision_regime": "BF16 GEMM + FP8 states" if arm.endswith("state_fp8") else "BF16 GEMM + FP32 states",
    "source": json.loads(Path(source).read_text()),
    "schedule_contract": {
        "source_iteration": 13794,
        "target_iteration": 19570,
        "post_checkpoint_steps": 5776,
        "extension_local_consumed_samples": 0,
        "micro_batch": 16,
        "global_batch": 208,
        "peak_lr": 1.63e-3,
        "final_lr": 1.63e-4,
        "plateau_steps": 2328 if schedule == "short" else 0,
        "decay_steps": 3448 if schedule == "short" else 5776,
    },
    "wandb": {
        "mode": "online",
        "run_id": wandb_run_id,
        "url": f"https://wandb-radfan.ru/andrey/hmoe-stage3/runs/{wandb_run_id}",
    },
    "fixed_lm": values,
    "training_health": health_record,
    "routing": json.loads(Path(routing).read_text()),
    "archive": json.loads(Path(archive).read_text()),
}
Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(
    f"FINAL_RESULT arm={arm} schedule={schedule} validation={values['validation'][0]:.6f} "
    f"test={values['test'][0]:.6f} max_grad={health_record['maximum_grad_norm']:.6f} "
    f"wandb={record['wandb']['url']} report={output}"
)
PY
}

run_tail() {
  check_arm
  case "$schedule" in short|long) ;; *) echo "schedule must be short or long" >&2; return 2 ;; esac
  [[ -s /home/jovyan/.wandb-key || -n ${WANDB_API_KEY:-} ]] || {
    echo "online W&B credential is missing" >&2
    return 2
  }
  [[ -s $token_file ]] || {
    echo "safe HF upload credential is missing: $token_file" >&2
    return 2
  }
  prepare_workdir
  check_gpu_prerequisites
  download_source
  audit_source_checkpoint
  export_run_environment

  suffix="schedule-matrix-$schedule-v1"
  train_run_id="stage3-$arm-schedule-tail-$suffix"
  echo "WANDB_RUN url=https://wandb-radfan.ru/andrey/hmoe-stage3/runs/$train_run_id"
  STAGE3_MOE_SCHEDULE="$schedule" \
  STAGE3_MOE_RUN_SUFFIX="$suffix" \
  WANDB_RUN_ID="$train_run_id" \
  WANDB_RESUME=allow \
    "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" schedule-tail

  endpoint_root="$work/checkpoints/schedule-matrix/$schedule/$arm"
  [[ -f $endpoint_root/latest_checkpointed_iteration.txt ]] || { echo "endpoint tracker missing" >&2; return 2; }
  [[ $(cat "$endpoint_root/latest_checkpointed_iteration.txt") == 19570 ]] || { echo "endpoint is not iteration 19570" >&2; return 2; }
  endpoint="$endpoint_root/iter_0019570"
  [[ -d $endpoint ]] || { echo "endpoint directory missing: $endpoint" >&2; return 2; }

  mkdir -p "$evidence_root/eval" "$evidence_root/health" "$evidence_root/routing"
  label="$arm-$schedule"
  eval_log="$evidence_root/eval/$label.log"
  WANDB_MODE=disabled \
  STAGE3_MOE_MATCHED_SKIP_REFERENCES=1 \
  STAGE3_MOE_MATCHED_CANDIDATE="$endpoint_root" \
  STAGE3_MOE_MATCHED_CANDIDATE_LABEL="$label" \
  STAGE3_MOE_MATCHED_CANDIDATE_ARM="$arm" \
  STAGE3_MOE_MATCHED_CANDIDATE_ITERATION=19570 \
  STAGE3_MOE_MATCHED_EVAL_TAG="schedule-matrix-$schedule-v1" \
  STAGE3_MOE_MATCHED_EVAL_REPEATS=2 \
    "$root/scripts/cloud_moe_matched_lm_eval.sh" | tee "$eval_log"

  routing_log="$evidence_root/routing/$label.log"
  routing_output="$evidence_root/routing/$label-artifacts"
  STAGE3_MOE_ROUTING_ARM="$arm" \
  STAGE3_MOE_ROUTING_CANDIDATE="$endpoint_root" \
  STAGE3_MOE_ROUTING_CANDIDATE_LABEL="$label" \
  STAGE3_MOE_ROUTING_CANDIDATE_ITERATION=19570 \
  STAGE3_MOE_ROUTING_OUTPUT_ROOT="$routing_output" \
  STAGE3_MOE_LOG_ROOT="$log_root" \
    "$root/scripts/cloud_moe_fixed_routing_audit.sh" candidate | tee "$routing_log"
  routing_json="$routing_output/$label.json"

  decay_start=13794
  [[ $schedule == short ]] && decay_start=16122
  health_json="$evidence_root/health/$label.json"
  python "$root/scripts/audit_time_match_training_logs.py" \
    --run "$label:$log_root/$train_run_id:$decay_start" \
    --output "$health_json"

  archive_json="$work/archive.json"
  upload_endpoint "$endpoint" "$archive_json"
  report="$evidence_root/$label.json"
  write_final_report "$health_json" "$eval_log" "$routing_json" "$archive_json" "$report" "$train_run_id"

  rm -rf "$work"
  echo "RUN_EXIT=0 arm=$arm schedule=$schedule"
}

main() {
  case "$mode" in
    inventory) inventory ;;
    preflight) preflight ;;
    run) run_tail ;;
    *) echo "unknown mode: $mode" >&2; return 2 ;;
  esac
}

(
  set -e
  main
)
code=$?
echo "SCHEDULE_MATRIX_EXIT=$code mode=$mode arm=${arm:-none} schedule=${schedule:-none}"
exit 0
