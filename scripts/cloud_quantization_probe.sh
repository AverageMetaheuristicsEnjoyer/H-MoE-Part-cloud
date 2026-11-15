#!/usr/bin/env bash
# Matched short continuations. Usage: cloud_quantization_probe.sh adamw|muon VARIANT...
# Mxy: momentum {fp32,maxabs,dre} x fallback {fp32,dre,maxabs}; M13 keeps E4/E5.
# A0/A1/A2/A3: fp32 / DRE E4/E4 / maxabs E4/E4 / DRE E4/E5.
# Compute variants: bf16 delayed_h1 delayed_h1024max current current_e4m3 blockwise (FP32 states).
# COAT variants: coatact (COAT activations, fused expert path) coatact_loop (per-expert loop)
# coatopt (BF16 GEMMs, COAT optimizer states) coat (both).
# bf16_fused: BF16 on the fused expert path (stage3_moe/bf16_fused_experts.py), FP32 states.
# VARIANT+NAME runs VARIANT with the Muon preset NAME of scripts/muon_presets.sh for that run only (e.g. bf16+lm2,
# coatopt+nm), so one job can interleave optimizer variants on the same allocation.
# Real success is QUANTIZATION_PROBE_EXIT=0 and BRANCH_VALIDATED=true, not platform exit.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
source "$root/scripts/muon_presets.sh"
optimizer=${1:?usage: cloud_quantization_probe.sh adamw\|muon VARIANT...}
shift
[[ $optimizer == adamw || $optimizer == muon || $optimizer == ademamix || $optimizer == slimadam ]] || { echo "PROBE_ABORT=unknown_optimizer"; exit 2; }
[[ $# -gt 0 ]] || { echo "PROBE_ABORT=no_variants"; exit 2; }
variants=("$@")
tag=${STAGE3_MOE_PROBE_TAG:-quant-$(date -u +%Y%m%dT%H%M%SZ)-$$}
[[ $tag =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo "PROBE_ABORT=invalid_tag"; exit 2; }
stage_root=${STAGE3_MOE_PROBE_STAGE:-/tmp/hmoe-quantization-$optimizer}
log_root=${STAGE3_MOE_LOG_ROOT:-$stage_root/logs}
mkdir -p "$stage_root" "$log_root"
manifest="$log_root/manifest-$optimizer-$tag.json"
summary="$log_root/summary-$optimizer-$tag.jsonl"
status=1
trap 'echo "QUANTIZATION_PROBE_EXIT=$status MANIFEST=$manifest SUMMARY=$summary"; exit 0' EXIT

select_variant() {
  local variant=${1%%+*}
  preset=()
  if [[ $1 == *+* ]]; then
    read -ra preset <<<"$(muon_preset "${1#*+}")"
    [[ ${#preset[@]} -gt 0 ]] || return 1
  fi
  muon_codec=fp32; adam_codec=fp32; formats=e4m3:e4m3; flags=""; coat_arm=""; fused=0; bf16_fused=0; extra=""; grouped=0; grouped_sw=0; m2=""
  case "$variant" in
    M[012][012])
      [[ $optimizer == muon ]] || return 1
      case ${variant:1:1} in 0) muon_codec=fp32;; 1) muon_codec=maxabs;; 2) muon_codec=dre;; esac
      case ${variant:2:1} in 0) adam_codec=fp32;; 1) adam_codec=dre;; 2) adam_codec=maxabs;; esac ;;
    M13) [[ $optimizer == muon ]] || return 1; muon_codec=maxabs; adam_codec=dre; formats=e4m3:e5m2 ;;
    A0) [[ $optimizer == adamw ]] || return 1 ;;
    A1) [[ $optimizer == adamw ]] || return 1; adam_codec=dre ;;
    A2) [[ $optimizer == adamw ]] || return 1; adam_codec=maxabs ;;
    A3) [[ $optimizer == adamw ]] || return 1; adam_codec=dre; formats=e4m3:e5m2 ;;
    bf16) ;;
    bf16_fused) bf16_fused=1 ;;
    # Megatron's own fused expert MLP (TE operation fuser, needs TE >= 2.14).
    bf16_opfuser) extra="--use-transformer-engine-op-fuser" ;;
    bf16_permfusion) extra="--moe-permute-fusion" ;;
    blockwise_permfusion) flags="--fp8-format e4m3 --fp8-recipe blockwise"; extra="--moe-permute-fusion" ;;
    delayed_h1024max_permfusion) flags="--fp8-format hybrid --fp8-recipe delayed --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"; extra="--moe-permute-fusion" ;;
    delayed_h1024max_opfuser) flags="--fp8-format hybrid --fp8-recipe delayed --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"; extra="--use-transformer-engine-op-fuser" ;;
    current_opfuser) flags="--fp8-format hybrid --fp8-recipe tensorwise"; extra="--use-transformer-engine-op-fuser" ;;
    blockwise_opfuser) flags="--fp8-format e4m3 --fp8-recipe blockwise"; extra="--use-transformer-engine-op-fuser" ;;
    delayed_h1) flags="--fp8-format hybrid --fp8-recipe delayed --fp8-amax-history-len 1 --fp8-amax-compute-algo most_recent" ;;
    delayed_h1024max) flags="--fp8-format hybrid --fp8-recipe delayed --fp8-amax-history-len 1024 --fp8-amax-compute-algo max" ;;
    coatact) coat_arm=${optimizer}_coat_state_fp32; fused=1 ;;
    coatact_loop) coat_arm=${optimizer}_coat_state_fp32 ;;
    coatact_grouped) coat_arm=${optimizer}_coat_state_fp32; grouped=1 ;;
    coatact_grouped_sw) coat_arm=${optimizer}_coat_state_fp32; grouped=1; grouped_sw=1 ;;
    coatopt) muon_codec=coat; adam_codec=coat ;;
    # AdEMAMix with COAT states and the slow EMA m2 stochastically rounded / kept in FP32 (stage3_moe/ademamix.py).
    coatopt_m2sr) [[ $optimizer == ademamix ]] || return 1; muon_codec=coat; adam_codec=coat; m2=sr ;;
    coatopt_m2fp32) [[ $optimizer == ademamix ]] || return 1; muon_codec=coat; adam_codec=coat; m2=fp32 ;;
    coat) coat_arm=${optimizer}_coat_state_fp8; muon_codec=coat; adam_codec=coat; fused=1 ;;
    current) flags="--fp8-format hybrid --fp8-recipe tensorwise" ;;
    current_e4m3) flags="--fp8-format e4m3 --fp8-recipe tensorwise" ;;
    blockwise) flags="--fp8-format e4m3 --fp8-recipe blockwise" ;;
    *) return 1 ;;
  esac
  arm=${optimizer}_bf16_state_fp32
  [[ $muon_codec == fp32 && $adam_codec == fp32 ]] || arm=${optimizer}_bf16_state_fp8
  [[ -z $flags ]] || arm=${optimizer}_fp8gemm_state_fp32
  [[ -z $coat_arm ]] || arm=$coat_arm
}
for variant in "${variants[@]}"; do
  select_variant "$variant" || { echo "PROBE_ABORT=unknown_variant:$variant"; exit 2; }
done

# STAGE3_MOE_PROBE_COLD_SWITCH=1: fresh optimizer states at the base checkpoint (--no-load-optim) while the
# LR/WD schedule continues from its iteration (stage3_moe/pretrain_gpt.py advances the scheduler, which
# MCore skips together with the optimizer). The optimizer-switching protocol of SOW Stage 3 item 3.
cold_switch=${STAGE3_MOE_PROBE_COLD_SWITCH:-0}
[[ $cold_switch == 1 ]] && export STAGE3_MOE_COLD_SWITCH=1
export STAGE3_MOE_PROBE_BASE_ITER=${STAGE3_MOE_PROBE_BASE_ITER:-13794}
# Every base is a 1C trunk: the Muon variants' training-fraction schedules keep its length in a branch.
export STAGE3_MOE_SCHEDULE_ITERS=${STAGE3_MOE_SCHEDULE_ITERS:-17242}
# A decay branch runs to its target unless told otherwise.
default_iters=400
[[ -z ${STAGE3_MOE_PROBE_TARGET_ITERS:-} ]] || default_iters=$((STAGE3_MOE_PROBE_TARGET_ITERS - STAGE3_MOE_PROBE_BASE_ITER))
export STAGE3_MOE_PROBE_ITERS=${STAGE3_MOE_PROBE_ITERS:-$default_iters}
export STAGE3_MOE_MICRO_BATCH=${STAGE3_MOE_MICRO_BATCH:-16}
export STAGE3_MOE_GLOBAL_BATCH=208
export STAGE3_MOE_LOG_INTERVAL=1
export STAGE3_MOE_EVAL_INTERVAL=${STAGE3_MOE_EVAL_INTERVAL:-100}
export STAGE3_MOE_EVAL_ITERS=${STAGE3_MOE_EVAL_ITERS:-32}
export STAGE3_MOE_DATA_CACHE_PATH=${STAGE3_MOE_DATA_CACHE_PATH:-$stage_root/data-cache}
export STAGE3_MOE_CKPT_ROOT=$stage_root/no-save-checkpoints
export STAGE3_MOE_LOG_ROOT=$log_root
export STAGE3_MOE_PROPAGATE_EXIT=1
export STAGE3_MOE_EP=1
export HF_HOME=${STAGE3_MOE_HF_HOME:-$stage_root/hf-cache}
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1
export WANDB_MODE=${WANDB_MODE:-offline}
unset PYTHONNOUSERSITE
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[[ $gpu_count == "${STAGE3_MOE_PROBE_GPUS:-1}" ]] || { echo "PROBE_ABORT=unexpected_visible_gpus:got_$gpu_count"; exit 2; }
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h "$stage_root"

if [[ -z ${STAGE3_MOE_DATA_ROOT:-} ]]; then
  for candidate in /home/jovyan/data/fineweb-edu-gpt2-megatron/data \
    /workspace-SR006.nfs2/hmoe-data/fineweb-edu-gpt2-megatron/data \
    /workspace-SR006.nfs3/hmoe-data/fineweb-edu-gpt2-megatron/data; do
    if [[ -f $candidate/train.bin ]]; then export STAGE3_MOE_DATA_ROOT=$candidate; break; fi
  done
fi
[[ -n ${STAGE3_MOE_DATA_ROOT:-} ]] || { echo "PROBE_ABORT=no_corpus"; exit 2; }
# STAGE3_MOE_PROBE_BASE_ARM names the arm the base checkpoint belongs to when it is not this optimizer's
# (an optimizer switch, item 3, starts from an AdamW checkpoint).
base_arm=${STAGE3_MOE_PROBE_BASE_ARM:-${optimizer}_bf16_state_fp32}
base_dir=${STAGE3_MOE_PROBE_LOAD:-$stage_root/checkpoint/$base_arm}
iter_name=$(printf 'iter_%07d' "$STAGE3_MOE_PROBE_BASE_ITER")
mkdir -p "$base_dir"
export QUANT_PROBE_CHECKPOINT_REVISION=local
# COAT_FETCH_CKPT=NAME:SHA256 starts from an archived checkpoint of the series (gateway) instead of the HF base.
if [[ ! -f $base_dir/$iter_name/mp_rank_00/model_optim_rng.pt && -n ${COAT_FETCH_CKPT:-} ]]; then
  mkdir -p "$base_dir/$iter_name/mp_rank_00"
  python "$root/scripts/coat_archive.py" fetch "${COAT_FETCH_CKPT%%:*}" "${COAT_FETCH_CKPT##*:}" \
    "$base_dir/$iter_name/mp_rank_00/model_optim_rng.pt" || { echo "PROBE_ABORT=fetch_failed"; exit 2; }
  export QUANT_PROBE_CHECKPOINT_REVISION=${COAT_FETCH_CKPT%%:*}
fi
if [[ ! -f $base_dir/$iter_name/mp_rank_00/model_optim_rng.pt ]]; then
  free_kb=$(df -Pk "$stage_root" | awk 'NR==2 {print $4}')
  [[ ${free_kb:-0} -ge 20000000 ]] || { echo "PROBE_ABORT=insufficient_local_disk"; exit 2; }
  hf_repo=${STAGE3_MOE_PROBE_HF_REPO:-AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints}
  hf_prefix=${STAGE3_MOE_PROBE_HF_PREFIX:-1c-mb4}
  python - "$hf_repo" "$hf_prefix/$base_arm" "$base_dir" "$iter_name" <<'PY'
import os, pathlib, sys
from huggingface_hub import HfApi, snapshot_download
repo, remote, dest, name = sys.argv[1:]
token = os.environ.get("HF_TOKEN")
token_file = pathlib.Path(os.environ.get("STAGE3_MOE_HF_TOKEN_FILE", "/home/jovyan/.cache/huggingface/token"))
if not token and token_file.is_file():
    token = token_file.read_text().strip()
revision = HfApi(token=token).model_info(repo).sha
snapshot_download(repo_id=repo, revision=revision, allow_patterns=[f"{remote}/{name}/**"],
                  local_dir=dest, max_workers=4, token=token)
staged, final = pathlib.Path(dest) / remote / name, pathlib.Path(dest) / name
if staged.is_dir() and not final.exists():
    staged.rename(final)
(pathlib.Path(dest) / "source-revision.txt").write_text(revision + "\n")
print("CHECKPOINT_HF_REVISION=" + revision)
PY
fi
[[ -f $base_dir/$iter_name/mp_rank_00/model_optim_rng.pt ]] || { echo "PROBE_ABORT=missing_checkpoint"; exit 2; }
if [[ -f $base_dir/latest_checkpointed_iteration.txt ]]; then
  [[ $(cat "$base_dir/latest_checkpointed_iteration.txt") == "$STAGE3_MOE_PROBE_BASE_ITER" ]] || {
    echo "PROBE_ABORT=checkpoint_tracker_mismatch"; exit 2;
  }
else
  printf '%s\n' "$STAGE3_MOE_PROBE_BASE_ITER" > "$base_dir/latest_checkpointed_iteration.txt"
fi
export STAGE3_MOE_PROBE_LOAD=$base_dir
python - "$manifest" "$root" "$base_dir/$iter_name" "$optimizer" "$tag" "${variants[@]}" <<'PY'
import datetime, hashlib, importlib.metadata, json, os, pathlib, subprocess, sys
path, root, checkpoint, optimizer, tag, *variants = sys.argv[1:]
def digest(p):
    if os.environ.get("STAGE3_MOE_PROBE_HASH", "1") == "0":
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()
files = [{"path": str(p.relative_to(checkpoint)), "bytes": p.stat().st_size,
          "sha256": digest(p)} for p in sorted(pathlib.Path(checkpoint).rglob("*")) if p.is_file()]
source_files = ["scripts/cloud_quantization_probe.sh", "scripts/run_stage3_moe_pretrain.sh",
                "configs/stage3-moe-1p029b.sh", "stage3_moe/optimizer_states.py", "stage3_moe/muon.py"]
revisions = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
packages = {}
for name in ("torch", "numpy", "transformer-engine", "triton"):
    try: packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: packages[name] = None
source_revision = pathlib.Path(checkpoint).parent / "source-revision.txt"
payload = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "optimizer": optimizer,
    "tag": tag, "variants": variants,
    "presets": {v: v.split("+", 1)[1] for v in variants if "+" in v}, "git_commit": revisions.stdout.strip(), "packages": packages,
    "source_sha256": {f: digest(pathlib.Path(root) / f) for f in source_files},
    "checkpoint": files, "checkpoint_source_revision": source_revision.read_text().strip() if source_revision.exists() else "local",
    "protocol": {"schedule": "WSD target %s decay %s" % (os.environ.get("STAGE3_MOE_PROBE_TARGET_ITERS", "17242"), os.environ.get("STAGE3_MOE_PROBE_DECAY_ITERS", "3448")),
                 "save_checkpoints": False,
                 "restore_optimizer": os.environ.get("STAGE3_MOE_COLD_SWITCH") != "1", "restore_rng": True,
                 "dp": int(os.environ.get("STAGE3_MOE_PROBE_GPUS", "1")), "global_batch": 208},
    "environment": {k: v for k, v in os.environ.items() if k in (
        "STAGE3_MOE_PROBE_BASE_ITER", "STAGE3_MOE_PROBE_ITERS", "STAGE3_MOE_MICRO_BATCH", "STAGE3_MOE_PROBE_GPUS",
        "STAGE3_MOE_PROBE_BASE_ARM", "STAGE3_MOE_COLD_SWITCH", "COAT_FETCH_CKPT",
        "STAGE3_MOE_ADEMAMIX_BETA3", "STAGE3_MOE_ADEMAMIX_ALPHA", "STAGE3_MOE_ADEMAMIX_WARMUP", "STAGE3_MOE_SLIM_SKIP",
        "STAGE3_MOE_ADEMAMIX_M2_FP32", "STAGE3_MOE_ADEMAMIX_M2_SR", "STAGE3_MOE_M2_TRACE_PATH", "STAGE3_MOE_M2_TRACE_STEPS",
        "STAGE3_MOE_MUON_VARIANT", "STAGE3_MOE_LIONMUON_PERIOD", "STAGE3_MOE_LIONMUON_BETA1", "STAGE3_MOE_LIONMUON_BETA2",
        "STAGE3_MOE_LIONMUON_LR_RATIO", "STAGE3_MOE_SOFTMUON_START_ITER", "STAGE3_MOE_SOFTMUON_END_ITER", "STAGE3_MOE_SOFTMUON_EPS",
        "STAGE3_MOE_LIONMUON_PROB", "STAGE3_MOE_LIONMUON_SEED", "STAGE3_MOE_LIONMUON_POWER", "STAGE3_MOE_SCHEDULE_ITERS", "STAGE3_MOE_NORMUON_BETA2", "STAGE3_MOE_NORMUON_EPS",
        "STAGE3_MOE_MUON2_BETA2", "STAGE3_MOE_MUON2_EPS", "STAGE3_MOE_MUON_NS_STEPS", "STAGE3_MOE_PROBE_TARGET_ITERS",
        "STAGE3_MOE_PROBE_DECAY_ITERS", "STAGE3_MOE_PROBE_WARMUP", "STAGE3_MOE_PROBE_MEASURE",
        "STAGE3_MOE_DEQUANT_ON_LOAD", "STAGE3_MOE_SEED",
        "STAGE3_MOE_EVAL_INTERVAL", "STAGE3_MOE_EVAL_ITERS", "STAGE3_MOE_DATA_ROOT",
        "STAGE3_MOE_DATA_CACHE_PATH", "STAGE3_MOE_LR", "STAGE3_MOE_MIN_LR", "STAGE3_MOE_LR_SCALE", "STAGE3_MOE_LR_CONSTANT",
        "STAGE3_MOE_PROBE_SAVE_INTERVAL", "STAGE3_MOE_PROBE_SAVE_ROOT", "STAGE3_MOE_ADAM_BETA2", "MLSUB_IMAGE")}}
pathlib.Path(path).write_text(json.dumps(payload, indent=2) + "\n")
print("CHECKPOINT_SHA256=" + json.dumps(files, sort_keys=True))
PY

status=0
# The m2 variants set their flag for their own run only; other variants keep the job-level values.
m2_sr_job=${STAGE3_MOE_ADEMAMIX_M2_SR:-}; m2_fp32_job=${STAGE3_MOE_ADEMAMIX_M2_FP32:-}
for variant in "${variants[@]}"; do
  select_variant "$variant"
  case $m2 in
    sr) export STAGE3_MOE_ADEMAMIX_M2_SR=1 STAGE3_MOE_ADEMAMIX_M2_FP32= ;;
    fp32) export STAGE3_MOE_ADEMAMIX_M2_SR= STAGE3_MOE_ADEMAMIX_M2_FP32=1 ;;
    *) export STAGE3_MOE_ADEMAMIX_M2_SR=$m2_sr_job STAGE3_MOE_ADEMAMIX_M2_FP32=$m2_fp32_job ;;
  esac
  export STAGE3_MOE_MUON_STATE_CODEC=$muon_codec
  export STAGE3_MOE_ADAM_STATE_CODEC=$adam_codec
  export STAGE3_MOE_FP8_STATE_DTYPES=$formats
  export STAGE3_MOE_FP8_COMPUTE_ARGS=$flags
  export STAGE3_MOE_COAT_FUSED_EXPERTS=$fused
  export STAGE3_MOE_COAT_GROUPED_GEMM=$grouped
  export STAGE3_MOE_COAT_GROUPED_STACKED_WEIGHTS=$grouped_sw
  export STAGE3_MOE_BF16_FUSED_EXPERTS=$bf16_fused
  [[ $cold_switch == 1 ]] && extra="${extra:+$extra }--no-load-optim"
  export STAGE3_MOE_EXTRA_ARGS=$extra
  # COAT momentum is dequantized 256 parameters at a time, as in the COAT series (run_coat_arm.sh).
  if [[ $muon_codec == coat ]]; then export STAGE3_MOE_FP8_DEQUANT_CHUNK=256; else unset STAGE3_MOE_FP8_DEQUANT_CHUNK; fi
  export STAGE3_MOE_RUN_SUFFIX=$tag-${variant//+/-}
  run_id=stage3-$arm-recipe-probe-$STAGE3_MOE_RUN_SUFFIX
  run_dir=$log_root/$run_id
  # STAGE3_MOE_PROBE_SAVE_FINAL=1: save the endpoint (weights only) and archive it on the gateway as
  # <STAGE3_MOE_PROBE_ARCHIVE_ID or tag>-<variant>-iterNNNNNNN-model.pt (scripts/coat_archive.py).
  # STAGE3_MOE_PROBE_SAVE_ROOT (WSM): save into this persistent directory instead and keep it after the run;
  # STAGE3_MOE_PROBE_ARCHIVE_EVERY=1 also archives every periodic save, not only the endpoint.
  unset STAGE3_MOE_PROBE_SAVE_DIR
  if [[ ${STAGE3_MOE_PROBE_SAVE_FINAL:-0} == 1 ]]; then
    export STAGE3_MOE_PROBE_SAVE_DIR=${STAGE3_MOE_PROBE_SAVE_ROOT:-$stage_root/save/$run_id}
    mkdir -p "$STAGE3_MOE_PROBE_SAVE_DIR"
    df -h "$STAGE3_MOE_PROBE_SAVE_DIR" | sed 's/^/PROBE_SAVE_DF /'
  fi
  export WANDB_RUN_ID=$run_id
  echo "VARIANT_BEGIN=$variant ARM=$arm MUON_CODEC=$muon_codec ADAM_CODEC=$adam_codec FORMATS=$formats COMPUTE=${flags:-bf16} PRESET=${preset[*]:-none}"
  started=$SECONDS
  code=0
  if [[ ${STAGE3_MOE_PROBE_MODE:-} == eval-lm-fixed || ${STAGE3_MOE_PROBE_MODE:-} == eval-routing-fixed ]]; then
    # Model-only validation/test loss of the base checkpoint under this variant's compute (item 2: how much of
    # the FP8-GEMM gap is the evaluating forward itself, and how much is in the weights). eval-routing-fixed
    # also writes the routing audit (STAGE3_MOE_ROUTING_AUDIT_PATH); its loss replays the expert-bias updates.
    STAGE3_MOE_EVAL_LOAD=$base_dir "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" "$STAGE3_MOE_PROBE_MODE" || code=$?
    echo "VARIANT_EXIT=$code VARIANT=$variant SECONDS=$((SECONDS - started))"
    newest=$(ls -1t "$log_root/stage3-$arm-$STAGE3_MOE_PROBE_MODE-$STAGE3_MOE_RUN_SUFFIX"/train-*.log 2>/dev/null | head -1)
    [[ -n $newest ]] && grep -aE "loss at iteration .* on (validation|test) set" "$newest" | sed "s/^/COAT_EVALLM variant=$variant arm=$arm base=${COAT_FETCH_CKPT%%:*} /"
    [[ -n $newest ]] && grep -aE "^COAT_HESSIAN|Traceback|Error" "$newest" | head -40
    continue
  fi
  env "${preset[@]}" "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" recipe-probe || code=$?
  echo "VARIANT_EXIT=$code VARIANT=$variant SECONDS=$((SECONDS - started))"
  if [[ -n ${STAGE3_MOE_PROBE_SAVE_DIR:-} && ( -n ${COAT_ARCHIVE_KEY_B64:-} || ( -n ${COAT_ARCHIVE_HF_PREFIX:-} && -n ${HF_TOKEN:-} ) ) ]]; then
    touch "$STAGE3_MOE_PROBE_SAVE_DIR/training-done"
    export COAT_ARCHIVE_FALLBACK_DIR=${COAT_CAMPAIGN_ROOT:-/workspace-SR006.nfs3/hmoe-cloud/coat-20261002}/archive-failed
    endpoint=$((STAGE3_MOE_PROBE_BASE_ITER + STAGE3_MOE_PROBE_ITERS))
    endpoint_kind=model; [[ ${STAGE3_MOE_PROBE_SAVE_OPTIM:-0} == 1 ]] && endpoint_kind=full
    wanted=("$endpoint:$endpoint_kind")
    if [[ ${STAGE3_MOE_PROBE_ARCHIVE_EVERY:-0} == 1 && -n ${STAGE3_MOE_PROBE_SAVE_INTERVAL:-} ]]; then
      for ((i = (STAGE3_MOE_PROBE_BASE_ITER / STAGE3_MOE_PROBE_SAVE_INTERVAL + 1) * STAGE3_MOE_PROBE_SAVE_INTERVAL; i < endpoint; i += STAGE3_MOE_PROBE_SAVE_INTERVAL)); do
        wanted+=("$i:model")
      done
    fi
    python "$root/scripts/coat_archive.py" watch "$STAGE3_MOE_PROBE_SAVE_DIR" "${STAGE3_MOE_PROBE_ARCHIVE_ID:-$tag}-${variant//+/-}" \
      "$STAGE3_MOE_PROBE_SAVE_DIR/training-done" "${wanted[@]}" || echo "ARCHIVE_EXIT=$?"
    [[ -n ${STAGE3_MOE_PROBE_SAVE_ROOT:-} ]] || rm -rf "$STAGE3_MOE_PROBE_SAVE_DIR"
  fi
  if ! python - "$summary" "$run_dir" "$variant" "$code" "$((SECONDS - started))" "$manifest" <<'PY'
import json, math, os, pathlib, re, statistics, sys
summary, run_dir, variant, code, elapsed, manifest = sys.argv[1:]
logs = sorted(pathlib.Path(run_dir).glob("train-*.log"))
lines = logs[-1].read_text(errors="replace").splitlines() if logs else []
step_re = re.compile(r"iteration\s+(\d+)/")
loss_re = re.compile(r"lm loss:\s*([0-9.eE+\-]+)")
norm_re = re.compile(r"grad norm:\s*([0-9.eE+\-]+)")
lr_re = re.compile(r"learning rate:\s*([0-9.eE+\-]+)")
val_re = re.compile(r"validation.*?loss at iteration\s+(\d+)(?: on (validation|test) set)?\s*\| lm loss value:\s*([0-9.eE+\-]+)")
expected = int(os.environ["STAGE3_MOE_PROBE_ITERS"])
endpoint = int(os.environ["STAGE3_MOE_PROBE_BASE_ITER"]) + expected
steps, vals, tests = [], [], []
skipped = nans = 0
for line in lines:
    step, loss, norm = step_re.search(line), loss_re.search(line), norm_re.search(line)
    if step and loss:
        steps.append({"iteration": int(step[1]), "loss": float(loss[1]), "grad_norm": float(norm[1]) if norm else None})
        if (lr := lr_re.search(line)) and (len(steps) == 1 or int(step[1]) % 250 == 0 or int(step[1]) == endpoint):
            steps[-1]["lr"] = float(lr[1])
    val = val_re.search(line)
    if val:
        destination = tests if val[2] == "test" else vals
        destination.append({"iteration": int(val[1]), "loss": float(val[3])})
    for label, pattern in (("skip", r"number of skipped iterations:\s*(\d+)"), ("nan", r"number of nan iterations:\s*(\d+)")):
        match = re.search(pattern, line)
        if match:
            if label == "skip": skipped = max(skipped, int(match[1]))
            else: nans = max(nans, int(match[1]))
valid = int(code) == 0 and len(steps) == expected and steps[-1]["iteration"] == endpoint and skipped == nans == 0 and all(math.isfinite(x["loss"]) for x in steps)
if int(os.environ["STAGE3_MOE_EVAL_ITERS"]) > 0:
    valid = valid and all(values and values[-1]["iteration"] == endpoint and all(math.isfinite(x["loss"]) for x in values) for values in (vals, tests))
row = {"variant": variant, "trainer_exit": int(code), "seconds": int(elapsed), "branch_validated": valid,
       "lr_first": steps[0].get("lr") if steps else None,
       "manifest": manifest, "run_dir": run_dir, "train_log": str(logs[-1]) if logs else None,
       "steps": steps, "validation": vals, "test": tests, "skipped": skipped, "nans": nans,
       "muon_codec": os.environ["STAGE3_MOE_MUON_STATE_CODEC"], "adam_codec": os.environ["STAGE3_MOE_ADAM_STATE_CODEC"],
       "state_formats": os.environ["STAGE3_MOE_FP8_STATE_DTYPES"], "compute_args": os.environ["STAGE3_MOE_FP8_COMPUTE_ARGS"]}
with open(summary, "a") as f: f.write(json.dumps(row, allow_nan=False) + "\n")
compact = {k: v for k, v in row.items() if k != "steps"}
compact["step_count"] = len(steps)
compact["tail_loss_mean"] = statistics.mean(x["loss"] for x in steps[-100:]) if steps else None
print("BRANCH_RESULT=" + json.dumps(compact))
print("BRANCH_VALIDATED=" + str(valid).lower() + " VARIANT=" + variant)
sys.exit(0 if valid else 1)
PY
  then status=1; fi
  [[ $status == 0 || ${STAGE3_MOE_PROBE_KEEP_GOING:-0} == 1 ]] || break
done
