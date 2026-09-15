#!/usr/bin/env bash
# A: which FP8 recipe costs the least, measured against bf16 on identical weights.
#
#   mlsub run --repo ... --branch stage3/fp8-recipe-probe \
#     --entry scripts/cloud_moe_recipe_probe.sh --gpus 1 --image torch28 \
#     --note fp8-recipe-probe-adamw \
#     --env STAGE3_MOE_LOG_ROOT=/workspace-SR006.nfs2/hmoe-cloud/pretrain \
#     --args "bf16 delayed_h1 delayed_h1024max current current_nowgrad"
#
# Every variant resumes the SAME bf16 checkpoint at iteration 13,794 -- deep in the 1C
# plateau, where the FP8-GEMM arms exceeded grad norm 0.3 on ~3.7 % of logged steps while
# no bf16 arm ever passed 0.15 in the whole run -- and runs the same few hundred steps on
# the same data. A bf16 base is deliberate: it carries no TE FP8 metadata, so no variant
# inherits another recipe's amax history.
#
# Readout per variant: grad-norm spike count (the fast one), train loss over the window,
# and the end-of-training validation loss. Nothing is saved.
# Always exits 0 so the platform keeps the logs; real status is in VARIANT_EXIT.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
log_root=${STAGE3_MOE_LOG_ROOT:-/workspace-SR006.nfs2/hmoe-cloud/pretrain}
stage_root=${STAGE3_MOE_PROBE_STAGE:-/workspace-SR006.nfs2/hmoe-checkpoints/recipe-probe}
base_arm=${STAGE3_MOE_PROBE_BASE_ARM:-adamw_bf16_state_fp32}
base_iter=${STAGE3_MOE_PROBE_BASE_ITER:-13794}
hf_repo=${STAGE3_MOE_PROBE_HF_REPO:-AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints}
hf_prefix=${STAGE3_MOE_PROBE_HF_PREFIX:-1c-mb4}
export HF_HOME=${STAGE3_MOE_HF_HOME:-/workspace-SR006.nfs2/hmoe-hf-cache}

# The compute axis was measured at micro-batch 16 and mb is worth ~0.01 % of loss, so the
# probe keeps 16: it is 2.5x cheaper per step at the same global batch.
export STAGE3_MOE_MICRO_BATCH=${STAGE3_MOE_MICRO_BATCH:-16}
export STAGE3_MOE_GLOBAL_BATCH=${STAGE3_MOE_GLOBAL_BATCH:-208}
export STAGE3_MOE_PROBE_ITERS=${STAGE3_MOE_PROBE_ITERS:-400}
# Spikes are the readout, and they last one step, so every step has to be logged.
export STAGE3_MOE_LOG_INTERVAL=${STAGE3_MOE_LOG_INTERVAL:-1}
export STAGE3_MOE_LOG_ROOT="$log_root"

variants=("$@")
[[ ${#variants[@]} -gt 0 ]] || variants=(bf16 delayed_h1 delayed_h1024max current)

# Each variant is (arm, extra MCore flags). Anything not in this table is skipped rather
# than guessed at -- mlsub argv reaches here as bare words with no quoting.
variant_arm() {
  case "$1" in
    bf16) echo "$base_arm" ;;
    *) echo "${base_arm/_bf16_state_fp32/_fp8gemm_state_fp32}" ;;
  esac
}

variant_flags() {
  case "$1" in
    # What every 1C FP8-GEMM arm actually ran: MCore's dataclass defaults, i.e. an amax
    # history of ONE with most_recent. Kept as the FP8-side control.
    bf16)              echo "" ;;
    delayed_h1)        echo "--fp8-format hybrid --fp8-recipe delayed" ;;
    # TE's own default and the setting in NVIDIA's reference FP8 pretraining scripts.
    delayed_h1024max)  echo "--fp8-format hybrid --fp8-recipe delayed --fp8-amax-history-len 1024 --fp8-amax-compute-algo max" ;;
    # ... plus one binade of headroom against the overflow the history is there to predict.
    delayed_h1024max_m1) echo "--fp8-format hybrid --fp8-recipe delayed --fp8-amax-history-len 1024 --fp8-amax-compute-algo max --fp8-margin 1" ;;
    # Per-tensor CURRENT scaling: the scale comes from the tensor being quantized, which
    # is what the dense COAT path does and what design.md:735-740 specified.
    current)           echo "--fp8-format hybrid --fp8-recipe tensorwise" ;;
    # E4M3 on the backward operands too -- more mantissa, less range, on gradients that
    # Muon then orthogonalizes.
    current_e4m3)      echo "--fp8-format e4m3 --fp8-recipe tensorwise" ;;
    # NVIDIA's standard mitigation. MCore forbids it under delayed scaling, so it can only
    # ride on a current-scaling variant. Layer 0 is the dense MLP layer here.
    current_flbf16)    echo "--fp8-format hybrid --fp8-recipe tensorwise --first-last-layers-bf16 --num-layers-at-start-in-bf16 1 --num-layers-at-end-in-bf16 1" ;;
    # Keep the weight-gradient GEMM out of FP8: that is the one that consumes E5M2
    # gradients, and it is the cheapest place to buy accuracy back.
    current_nowgrad)   echo "--fp8-format hybrid --fp8-recipe tensorwise --no-fp8-wgrad" ;;
    # 1x128 activations / 128x128 weights, the design's actual target. Needs CUDA >= 12.9,
    # so it only runs on the te3 image.
    blockwise)         echo "--fp8-format e4m3 --fp8-recipe blockwise" ;;
    *) return 1 ;;
  esac
}

unset PYTHONNOUSERSITE
nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
  -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: - || true)
if [[ -n $nvidia_lib_path ]]; then
  export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=${NVTE_FP8_BLOCK_SCALING_FP32_SCALES:-1}

echo "IMAGE=${MLSUB_IMAGE:-unset} VARIANTS=${variants[*]} ITERS=$STAGE3_MOE_PROBE_ITERS mb=$STAGE3_MOE_MICRO_BATCH"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

# --- data ------------------------------------------------------------------------------
# The corpus has moved between volumes more than once; take the first root that has one.
if [[ -z ${STAGE3_MOE_DATA_ROOT:-} ]]; then
  for candidate in \
    /home/jovyan/data/fineweb-edu-gpt2-megatron/data \
    /workspace-SR006.nfs2/hmoe-data/fineweb-edu-gpt2-megatron/data \
    /workspace-SR006.nfs3/hmoe-data/fineweb-edu-gpt2-megatron/data; do
    if compgen -G "$candidate/train*.bin" >/dev/null; then
      export STAGE3_MOE_DATA_ROOT=$candidate
      break
    fi
  done
fi
if [[ -z ${STAGE3_MOE_DATA_ROOT:-} ]]; then
  echo "PROBE_ABORT=no corpus found; set STAGE3_MOE_DATA_ROOT"
  exit 0
fi
echo "DATA_ROOT=$STAGE3_MOE_DATA_ROOT"
export STAGE3_MOE_DATA_CACHE=${STAGE3_MOE_DATA_CACHE:-$log_root/data-cache}

# --- base checkpoint -------------------------------------------------------------------
base_dir="$stage_root/$base_arm"
iter_dir=$(printf '%s/iter_%07d' "$base_dir" "$base_iter")
if [[ ! -f "$iter_dir/mp_rank_00/model_optim_rng.pt" ]]; then
  echo "=== staging $hf_prefix/$base_arm/iter_$base_iter from HF ==="
  df -h "$stage_root" 2>/dev/null | tail -1 || df -h "$(dirname "$stage_root")" | tail -1
  mkdir -p "$base_dir"
  pip install --user --no-cache-dir hf_transfer 2>&1 | tail -2
  HF_HUB_ENABLE_HF_TRANSFER=1 python - "$hf_repo" "$hf_prefix/$base_arm/iter_$(printf '%07d' "$base_iter")" "$base_dir" <<'PY'
import sys, os, shutil, time
from huggingface_hub import snapshot_download

repo, remote, dest = sys.argv[1], sys.argv[2], sys.argv[3]
start = time.time()
# The repo is public, so no token: an unauthenticated download is the whole point of
# having flipped it public when the private-storage quota blocked the last offload.
local = snapshot_download(repo_id=repo, allow_patterns=[f"{remote}/**"],
                          max_workers=8)
src = os.path.join(local, remote)
name = os.path.basename(remote)
target = os.path.join(dest, name)
os.makedirs(target, exist_ok=True)
for dirpath, _, files in os.walk(src):
    rel = os.path.relpath(dirpath, src)
    out = os.path.join(target, rel) if rel != "." else target
    os.makedirs(out, exist_ok=True)
    for f in files:
        dst = os.path.join(out, f)
        if os.path.exists(dst):
            continue
        # The snapshot is a symlink farm into the HF cache; copy so the cache can be
        # cleared without pulling the checkpoint out from under a running job.
        shutil.copyfile(os.path.join(dirpath, f), dst)
        print("STAGED", dst, os.path.getsize(dst))
print(f"STAGE_SECONDS={time.time() - start:.0f}")
PY
  echo "STAGE_EXIT=$?"
fi
if [[ ! -f "$iter_dir/mp_rank_00/model_optim_rng.pt" ]]; then
  echo "PROBE_ABORT=base checkpoint missing at $iter_dir"
  exit 0
fi
# MCore resumes from whatever the tracker names, and the HF copy carries no tracker.
echo "$base_iter" > "$base_dir/latest_checkpointed_iteration.txt"
export STAGE3_MOE_PROBE_LOAD="$base_dir"
du -sh "$base_dir" 2>/dev/null | tail -1

# --- the variants ----------------------------------------------------------------------
summary="$log_root/recipe-probe-summary-$(date -u +%Y%m%dT%H%M%SZ).txt"
mkdir -p "$log_root"
: > "$summary"

for variant in "${variants[@]}"; do
  if ! flags=$(variant_flags "$variant"); then
    echo "SKIP unknown variant: $variant"
    continue
  fi
  arm=$(variant_arm "$variant")
  export STAGE3_MOE_RUN_SUFFIX="recipe-$variant"
  export STAGE3_MOE_FP8_COMPUTE_ARGS="$flags"
  echo "=== VARIANT=$variant ARM=$arm FLAGS=${flags:-none} ==="
  start=$SECONDS
  "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" recipe-probe
  code=$?
  echo "VARIANT_EXIT=$code VARIANT=$variant SECONDS=$((SECONDS - start))"

  run_dir="$log_root/stage3-$arm-recipe-probe-recipe-$variant"
  train_log=$(ls -t "$run_dir"/train-*.log 2>/dev/null | head -1)
  if [[ -z $train_log ]]; then
    echo "NO_TRAIN_LOG variant=$variant"
    continue
  fi
  python - "$variant" "$train_log" "$summary" <<'PY'
import re, sys, statistics

variant, path, summary = sys.argv[1], sys.argv[2], sys.argv[3]
step = re.compile(r"iteration\s+(\d+)/")
loss = re.compile(r"lm loss:\s*([0-9.]+E[+-]\d+|[0-9.]+)")
gnorm = re.compile(r"grad norm:\s*([0-9.]+)")
# ' validation loss at iteration 14194 | lm loss value: 2.724416E+00 | lm loss PPL: ... '
val = re.compile(r"validation.*?loss at .*?lm loss value:\s*([0-9.]+E[+-]\d+|[0-9.]+)")
skipped = re.compile(r"number of skipped iterations:\s*(\d+)")
nans = re.compile(r"number of nan iterations:\s*(\d+)")

losses, norms, vals = [], [], []
last_skipped = last_nan = 0
for line in open(path, errors="replace"):
    m = gnorm.search(line)
    if m and step.search(line):
        norms.append(float(m.group(1)))
        lm = loss.search(line)
        losses.append(float(lm.group(1)) if lm else float("nan"))
        s, n = skipped.search(line), nans.search(line)
        if s:
            last_skipped = max(last_skipped, int(s.group(1)))
        if n:
            last_nan = max(last_nan, int(n.group(1)))
    v = val.search(line)
    if v:
        vals.append(float(v.group(1)))

if not norms:
    row = f"{variant:20s} NO_STEPS_PARSED"
else:
    tail = [x for x in losses[-200:] if x == x]
    loss_tail = statistics.mean(tail) if tail else float("nan")
    row = (f"{variant:20s} steps={len(norms):4d} "
           f"gnorm_med={statistics.median(norms):.4f} gnorm_max={max(norms):8.3f} "
           f"n>0.3={sum(n > 0.3 for n in norms):4d} n>1.0={sum(n > 1.0 for n in norms):4d} "
           f"skipped={last_skipped} nan={last_nan} "
           f"loss_tail={loss_tail:.5f} "
           + (f"val={vals[-1]:.6f}" if vals else "val=none"))
print("SUMMARY_ROW", row)
with open(summary, "a") as fh:
    fh.write(row + "\n")
PY
done

echo "=== RECIPE PROBE SUMMARY ==="
cat "$summary"
echo "SUMMARY_FILE=$summary"
exit 0
