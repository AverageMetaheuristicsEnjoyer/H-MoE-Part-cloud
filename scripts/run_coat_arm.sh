#!/usr/bin/env bash
# One from-scratch 1C pretraining arm of the COAT series on the GPUs in CUDA_VISIBLE_DEVICES.
#   run_coat_arm.sh ARM_ID
# Checkpoints live on the node's /tmp: they serve restarts inside this job, nothing else.
# Real success is COAT_ARM_RESULT ... reached=17242.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
arm_id=${1:?usage: run_coat_arm.sh ARM_ID}
campaign=${COAT_CAMPAIGN_ROOT:-/workspace-SR006.nfs3/hmoe-cloud/coat-20261002}
# COAT_STOP_ITERS=N: stop a from-scratch run at step N on the unchanged 17,242-step WSD schedule (the BF16
# noise-floor replicates of 2026-10-08: 5 seeds x 3,000 steps). No archive, no final-model copy.
full_iters=${COAT_STOP_ITERS:-17242}
[[ -z ${COAT_STOP_ITERS:-} ]] || export STAGE3_MOE_FULL_TRAIN_ITERS=$COAT_STOP_ITERS

case "$arm_id" in
  # adamw-ctrl2: the same control restarted from scratch if its first allocation has to be released.
  adamw-ctrl|adamw-ctrl2) arm=adamw_bf16_state_fp32 ;;
  muon-ctrl) arm=muon_bf16_state_fp32 ;;
  adamw-coatopt) arm=adamw_bf16_state_fp8; export STAGE3_MOE_ADAM_STATE_CODEC=coat ;;
  muon-coatopt) arm=muon_bf16_state_fp8; export STAGE3_MOE_ADAM_STATE_CODEC=coat STAGE3_MOE_MUON_STATE_CODEC=coat STAGE3_MOE_FP8_DEQUANT_CHUNK=${COAT_MUON_CHUNK:-256} ;;
  adamw-coatact) arm=adamw_coat_state_fp32 ;;
  muon-coatact) arm=muon_coat_state_fp32 ;;
  # Replicates (user 2026-10-02 09:42 MSK): the AdamW controls differ by ~0.8 % at matched steps, so
  # every comparison needs its own noise floor. Same seed and configuration as the originals.
  muon-ctrl2) arm=muon_bf16_state_fp32 ;;
  # Seed replicates (feedback 2026-10-05): the same control with another seed (init + data order).
  adamw-ctrl-s[0-9]*) arm=adamw_bf16_state_fp32; export STAGE3_MOE_SEED=${arm_id##*-s} ;;
  muon-ctrl-s[0-9]*) arm=muon_bf16_state_fp32; export STAGE3_MOE_SEED=${arm_id##*-s} ;;
  adamw-coatopt2) arm=adamw_bf16_state_fp8; export STAGE3_MOE_ADAM_STATE_CODEC=coat ;;
  # COAT activations with q/k/v scales and memory-efficient weights (stage3_moe/coat_activation.py, STRICT).
  adamw-coatact-strict) arm=adamw_coat_state_fp32; export STAGE3_MOE_COAT_STRICT=1 ;;
  muon-coatact-strict) arm=muon_coat_state_fp32; export STAGE3_MOE_COAT_STRICT=1 ;;
  adamw-coat) arm=adamw_coat_state_fp8; export STAGE3_MOE_ADAM_STATE_CODEC=coat ;;
  # Chunk 256: measured 2026-10-02 (muonchunk job) +0.10 s/step over FP32 states against +0.46 s at 1, same peak memory.
  muon-coat) arm=muon_coat_state_fp8; export STAGE3_MOE_ADAM_STATE_CODEC=coat STAGE3_MOE_MUON_STATE_CODEC=coat STAGE3_MOE_FP8_DEQUANT_CHUNK=${COAT_MUON_CHUNK:-256} ;;
  # Full COAT (activations + optimizer states) on the fused expert path (stage3_moe/coat_fused_experts.py).
  # COAT activations only (FP32 states) on the fused path: the FP8-GEMM-only base of the switching matrix.
  adamw-coatact-fused) arm=adamw_coat_state_fp32; export STAGE3_MOE_COAT_FUSED_EXPERTS=1 ;;
  muon-coatact-fused) arm=muon_coat_state_fp32; export STAGE3_MOE_COAT_FUSED_EXPERTS=1 ;;
  adamw-coat-fused) arm=adamw_coat_state_fp8; export STAGE3_MOE_ADAM_STATE_CODEC=coat STAGE3_MOE_COAT_FUSED_EXPERTS=1 ;;
  muon-coat-fused) arm=muon_coat_state_fp8; export STAGE3_MOE_ADAM_STATE_CODEC=coat STAGE3_MOE_MUON_STATE_CODEC=coat STAGE3_MOE_FP8_DEQUANT_CHUNK=${COAT_MUON_CHUNK:-256} STAGE3_MOE_COAT_FUSED_EXPERTS=1 ;;
  # The FP8 schemes Megatron ships (TransformerEngine GEMMs, FP32 states), in the protocol of the COAT
  # arms: blockwise E4M3 is TE's most accurate recipe, delayed is what --fp8-recipe gives by default.
  adamw-te-blockwise|muon-te-blockwise) arm=${arm_id%%-*}_fp8gemm_state_fp32; export STAGE3_MOE_FP8_COMPUTE_ARGS="--fp8-format e4m3 --fp8-recipe blockwise" ;;
  adamw-te-delayed|muon-te-delayed) arm=${arm_id%%-*}_fp8gemm_state_fp32; export STAGE3_MOE_FP8_COMPUTE_ARGS="--fp8-format hybrid --fp8-recipe delayed" ;;
  # Delayed scaling as NVIDIA configures it (TE's own default and examples/llama/train_llama3_8b_h100_fp8.sh):
  # amax history 1024, max. MCore's dataclass default is history 1, most_recent -- the arm above.
  adamw-te-delayed1024|muon-te-delayed1024) arm=${arm_id%%-*}_fp8gemm_state_fp32
    export STAGE3_MOE_FP8_COMPUTE_ARGS="--fp8-format hybrid --fp8-recipe delayed --fp8-amax-history-len 1024 --fp8-amax-compute-algo max" ;;
  *) echo "COAT_ARM_ABORT=unknown_arm:$arm_id"; exit 2 ;;
esac

# STAGE3_MOE_MUON_PRESET=NAME: a Muon variant of scripts/muon_presets.sh for the whole run (SOW Stage 4 arms).
if [[ -n ${STAGE3_MOE_MUON_PRESET:-} ]]; then
  source "$root/scripts/muon_presets.sh"
  preset=$(muon_preset "$STAGE3_MOE_MUON_PRESET") || { echo "COAT_ARM_ABORT=unknown_muon_preset:$STAGE3_MOE_MUON_PRESET"; exit 2; }
  for assignment in $preset; do export "$assignment"; done
  echo "COAT_ARM_MUON_PRESET name=$STAGE3_MOE_MUON_PRESET $preset"
  # The preset is part of the run's identity: stage dir, log dir, W&B id and archive names.
  arm_id=$arm_id-$STAGE3_MOE_MUON_PRESET
fi

stage=/tmp/hmoe-coat/$arm_id
export STAGE3_MOE_CKPT_ROOT=$stage/ckpt
export STAGE3_MOE_LOG_ROOT=$campaign/runs-${COAT_ROUND:-r2}/$arm_id
export STAGE3_MOE_DATA_CACHE_PATH=$stage/data-cache
export STAGE3_MOE_RUN_SUFFIX=c1002-$arm_id
export STAGE3_MOE_MICRO_BATCH=${STAGE3_MOE_MICRO_BATCH:-13}
export STAGE3_MOE_NPROC=${STAGE3_MOE_NPROC:-4}
export STAGE3_MOE_EP=1
export STAGE3_MOE_EVAL_INTERVAL=${STAGE3_MOE_EVAL_INTERVAL:-250}
export STAGE3_MOE_EVAL_ITERS=${STAGE3_MOE_EVAL_ITERS:-32}
export STAGE3_MOE_LOG_INTERVAL=${STAGE3_MOE_LOG_INTERVAL:-10}
export STAGE3_MOE_PROPAGATE_EXIT=1
export HF_HOME=$stage/hf-cache
export WANDB_MODE=offline
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1
export WANDB_RUN_ID=stage3-$arm-full-$STAGE3_MOE_RUN_SUFFIX
unset PYTHONNOUSERSITE
mkdir -p "$stage" "$STAGE3_MOE_LOG_ROOT"

if [[ -z ${STAGE3_MOE_DATA_ROOT:-} ]]; then
  for candidate in /home/jovyan/data/fineweb-edu-gpt2-megatron/data \
    /workspace-SR006.nfs2/hmoe-data/fineweb-edu-gpt2-megatron/data \
    /workspace-SR006.nfs3/hmoe-data/fineweb-edu-gpt2-megatron/data; do
    if [[ -f $candidate/train.bin ]]; then export STAGE3_MOE_DATA_ROOT=$candidate; break; fi
  done
fi
[[ -n ${STAGE3_MOE_DATA_ROOT:-} ]] || { echo "COAT_ARM_ABORT=no_corpus"; exit 2; }

if [[ ${COAT_ARM_MODE:-full} == bench ]]; then
  export STAGE3_MOE_RUN_SUFFIX=c1002-bench-$arm_id-${COAT_BENCH_TAG:-0} STAGE3_MOE_BENCH_WARMUP=${COAT_BENCH_WARMUP:-20} STAGE3_MOE_BENCH_MEASURE=${COAT_BENCH_MEASURE:-40}
  "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" bench > "$stage/bench.log" 2>&1
  code=$?
  python - "$STAGE3_MOE_LOG_ROOT/stage3-$arm-bench-$STAGE3_MOE_RUN_SUFFIX/results.jsonl" "$arm_id" "$code" <<'PY'
import json, sys
path, arm_id, code = sys.argv[1:]
try:
    record = json.loads(open(path).read().splitlines()[-1])
except (OSError, IndexError) as error:
    print("COAT_BENCH " + json.dumps({"arm_id": arm_id, "exit": int(code), "error": str(error)}), flush=True)
    raise SystemExit(0)
m = record["measurement"]
state = record["optimizer_state"]
print("COAT_BENCH " + json.dumps({"arm_id": arm_id, "exit": int(code),
      "measured_steps": m["protocol"]["measured_steps"], "full_step_s_median": m["timing"]["full_step_seconds"],
      "optimizer_step_s_median": m["timing"]["optimizer_step_seconds"], "tokens_per_second": m["timing"]["tokens_per_second"],
      "max_allocated_bytes": m["memory"]["max_allocated_bytes"], "max_reserved_bytes": m["memory"]["max_reserved_bytes"],
      "optimizer_state": {k: state.get(k) for k in ("persistent_data_bytes", "metadata_bytes", "master_parameter_bytes")},
      "train_loss": m["loss"]["training"]}), flush=True)
PY
  grep -E 'Traceback|Error' "$stage/bench.log" | head -3
  exit 0
fi

if [[ ${COAT_ARM_MODE:-full} == screen ]]; then
  # SOW Stage 4 screen: a complete 1.2B-token WSD run from scratch (launcher mode lr-sweep: 2,818 steps, decay over the
  # last 564, nothing saved), so schedules that act in the decay are seen. Real success is COAT_ARM_SCREEN ... reached=2818.
  export STAGE3_MOE_RUN_SUFFIX=c1002-screen-$arm_id
  "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" lr-sweep > "$stage/screen.log" 2>&1
  code=$?
  train_log=$(grep -oP '^TRAIN_LOG=\K.*' "$stage/screen.log")
  python - "$train_log" "$arm_id" "${STAGE3_MOE_MUON_PRESET:-none}" "$code" <<'PY'
import json, re, statistics, sys
log, arm_id, preset, code = sys.argv[1:]
try:
    lines = open(log, errors="replace").read().splitlines()
except OSError as error:
    print("COAT_ARM_SCREEN " + json.dumps({"arm_id": arm_id, "preset": preset, "exit": int(code), "error": str(error)}), flush=True)
    raise SystemExit(0)
evals = {}
for line in lines:
    # Periodic evaluations print "validation loss at iteration N | ...", the final ones add "on validation/test set".
    match = re.search(r"loss at iteration\s+(\d+)(?: on (validation|test) set)?\s*\| lm loss value:\s*([0-9.eE+\-]+)", line)
    if match:
        evals.setdefault(match[2] or "validation", []).append((int(match[1]), float(match[3])))
iterations = [int(m[1]) for m in (re.search(r"iteration\s+(\d+)/", line) for line in lines) if m]
elapsed = [float(m[1]) for m in (re.search(r"elapsed time per iteration \(ms\):\s*([0-9.]+)", line) for line in lines) if m]
print("COAT_ARM_SCREEN " + json.dumps({"arm_id": arm_id, "preset": preset, "exit": int(code),
      "reached": max(iterations, default=0), "validation": evals.get("validation", [])[-12:], "test": evals.get("test", [])[-1:],
      "step_ms_median_last_half": statistics.median(elapsed[len(elapsed) // 2:]) if elapsed else None}), flush=True)
PY
  [[ -n $train_log ]] && grep -E 'LIONMUON|NORMUON|MUON2|Traceback|Error' "$train_log" | tail -8
  grep -E 'Traceback|Error' "$stage/screen.log" | head -3
  exit 0
fi

if [[ ${COAT_ARM_MODE:-full} == smoke ]]; then
  # Save at 10 and 20, then resume from 20 in a fresh process and run to 40.
  export STAGE3_MOE_CKPT_ROOT=$stage/smoke-ckpt STAGE3_MOE_KEEP_SMOKE_CKPT=1 STAGE3_MOE_RUN_SUFFIX=c1002-smoke-$arm_id STAGE3_MOE_LOG_INTERVAL=1
  rm -rf "$STAGE3_MOE_CKPT_ROOT"
  for iters in 20 40; do
    STAGE3_MOE_SMOKE_ITERS=$iters "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" smoke > "$stage/smoke-$iters.log" 2>&1
    code=$?
    log=$(grep -oP '^TRAIN_LOG=\K.*' "$stage/smoke-$iters.log")
    steps=$(grep -cE 'iteration +[0-9]+/' "$log" 2>/dev/null)
    last=$(grep -E 'iteration +[0-9]+/' "$log" 2>/dev/null | tail -1)
    echo "COAT_SMOKE arm_id=$arm_id iters=$iters exit=$code logged_steps=$steps imports=$(grep -c COAT_STATE_IMPORT "$log") loaded=$(grep -c 'successfully loaded checkpoint' "$log") nan=$(grep -cE 'nan iterations: +[1-9]|skipped iterations: +[1-9]' "$log")"
    echo "COAT_SMOKE_LAST $last"
    grep -E 'Traceback|Error|error:' "$log" | head -5
    grep -E 'mem|memory' "$log" | tail -3
  done
  ls -la "$STAGE3_MOE_CKPT_ROOT"/smoke/*/ 2>/dev/null | head
  rm -rf "$STAGE3_MOE_CKPT_ROOT"
  exit 0
fi

echo "COAT_ARM_BEGIN arm_id=$arm_id arm=$arm devices=${CUDA_VISIBLE_DEVICES:-all} nproc=$STAGE3_MOE_NPROC micro_batch=$STAGE3_MOE_MICRO_BATCH bundle=${BUNDLE_ID:-none} utc=$(date -u +%FT%TZ)"
tracker=$STAGE3_MOE_CKPT_ROOT/1c/$arm/latest_checkpointed_iteration.txt
reached=0
# Four checkpoints per run go to the gateway (scripts/coat_archive.py): early, middle and pre-decay
# (13,794, the last step at peak LR) with their optimizer state, the final one as model weights.
# 2,178 = 6 x 363 and 8,712 = 24 x 363 are save iterations of the launcher's 363-step interval.
done_flag=$stage/training-done
archived=$STAGE3_MOE_CKPT_ROOT/1c/archive-staging-$arm_id/done-$full_iters
rm -f "$done_flag"
archiver=
# COAT_RESUME_CKPT=NAME:SHA256 (with COAT_FETCH_HF_PREFIX for an HF copy): a new job continues an archived full checkpoint
# of this run, whose pod and /tmp died (s4-full-muon, 2026-10-20).
resume_iter=0
if [[ -n ${COAT_RESUME_CKPT:-} && ! -f $tracker ]]; then
  resume_name=${COAT_RESUME_CKPT%%:*}
  resume_iter=$((10#$(grep -oP 'iter\K[0-9]{7}' <<<"$resume_name")))
  resume_file=$STAGE3_MOE_CKPT_ROOT/1c/$arm/iter_$(printf '%07d' "$resume_iter")/mp_rank_00/model_optim_rng.pt
  mkdir -p "$(dirname "$resume_file")"
  python "$root/scripts/coat_archive.py" fetch "$resume_name" "${COAT_RESUME_CKPT##*:}" "$resume_file" || { echo "COAT_ARM_ABORT=resume_fetch_failed"; exit 2; }
  echo "$resume_iter" > "$tracker"
  echo "COAT_ARM_RESUME arm_id=$arm_id iteration=$resume_iter"
fi
# COAT_ARCHIVE_HF_PREFIX=stage4 (with HF_TOKEN): the same four checkpoints go to HF instead (SOW Stage 4, user 2026-10-19).
if [[ -n ${COAT_ARCHIVE_KEY_B64:-} || ( -n ${COAT_ARCHIVE_HF_PREFIX:-} && -n ${HF_TOKEN:-} ) ]]; then
  wanted=()
  # Every 2178 = 6 x 363 steps (user 2026-10-20, after a pod died at 8290 with only 2178 archived): a lost pod costs at most
  # ~2178 steps, and the plateau points double as decay-branch starts for the time-to-AdamW-loss measurement.
  for item in 2178:full 4356:full 6534:full 8712:full 10890:full 13794:full 15972:full "$full_iters:model"; do
    (( ${item%%:*} > resume_iter )) && wanted+=("$item")
  done
  python "$root/scripts/coat_archive.py" watch "$STAGE3_MOE_CKPT_ROOT/1c/$arm" "$arm_id" "$done_flag" "${wanted[@]}" &
  archiver=$!
fi
for attempt in 1 2 3; do
  "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" full > "$stage/launcher-attempt$attempt.log" 2>&1
  code=$?
  grep -E '^(TRAIN_EXIT|TRAIN_LOG|ARM=|SCHEDULE|OPTIMIZER_HPARAMS|CKPT)' "$stage/launcher-attempt$attempt.log"
  last=$(grep -oE 'iteration +[0-9]+/' "$(grep -oP '^TRAIN_LOG=\K.*' "$stage/launcher-attempt$attempt.log")" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
  reached=${last:-0}
  echo "COAT_ARM_ATTEMPT arm_id=$arm_id attempt=$attempt exit=$code reached=$reached checkpoint=$(cat "$tracker" 2>/dev/null || echo none)"
  [[ $code == 0 && $reached == "$full_iters" ]] && break
  # A restart is only worth it when there is a checkpoint to continue from.
  [[ -f $tracker ]] || break
done
echo "COAT_ARM_RESULT arm_id=$arm_id reached=$reached target=$full_iters utc=$(date -u +%FT%TZ)"
touch "$done_flag"
[[ -z $archiver ]] || wait "$archiver"

# Without the gateway archive (or if the final upload failed) keep the final model weights on a persistent
# volume that still has room; the shared volumes are nearly full, so a volume with less than 6 GiB free is skipped.
if [[ -z ${COAT_STOP_ITERS:-} && $reached == "$full_iters" && -f $tracker && ! -f $archived ]]; then
  python - "$STAGE3_MOE_CKPT_ROOT/1c/$arm/iter_$(printf '%07d' "$(cat "$tracker")")/mp_rank_00/model_optim_rng.pt" "$arm_id" \
    "$campaign/final-models" /workspace-SR006.nfs2/hmoe-cloud/coat-20261002-final-models <<'PY'
import hashlib, pathlib, shutil, sys, torch
source, arm_id, *roots = sys.argv[1:]
try:
    state = torch.load(source, map_location="cpu", weights_only=False)
    state.pop("optimizer", None)
    for root in map(pathlib.Path, roots):
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < 6 * 2**30:
            continue
        target = root / arm_id / pathlib.Path(source).parent.parent.name / "mp_rank_00" / "model_optim_rng.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        print(f"COAT_ARM_FINAL_MODEL arm_id={arm_id} path={target} bytes={target.stat().st_size} sha256={digest}", flush=True)
        break
    else:
        print(f"COAT_ARM_FINAL_MODEL_SKIPPED arm_id={arm_id} reason=no_volume_with_6GiB_free", flush=True)
except Exception as error:
    print(f"COAT_ARM_FINAL_MODEL_SKIPPED arm_id={arm_id} reason={error!r}", flush=True)
PY
fi
