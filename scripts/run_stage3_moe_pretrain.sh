#!/usr/bin/env bash
# Stage 3 MoE matched pretraining, WSD trunk-and-branch.
#
#   run_stage3_moe_pretrain.sh ARM trunk|decay-1p2b|smoke|bench|resume-bench|eval-downstream
#
# smoke exercises save and resume; bench measures throughput and peak memory with no
# checkpoint traffic; resume-bench does the same from the trunk branch point, so the
# measured steps sit in the phase training actually runs in. STAGE3_MOE_RUN_SUFFIX keeps
# concurrent variants from sharing a run directory or a checkpoint.
#
# The schedule follows docs/design.md:182 -- linear warmup on steps 1-173,
# constant peak LR through step 13,794, exponential decay over the final 20% to
# 0.1x peak at step 17,242 (7,344,816,128 tokens, the Ling MoE-optimal budget).
# Because the stable phase is at constant LR, one trunk serves every budget: a
# shorter budget just branches off earlier and runs its own decay tail.
set -euo pipefail

arm=${1:?usage: run_stage3_moe_pretrain.sh ARM full|trunk|decay-1p2b|smoke|bench|resume-bench}
mode=${2:?usage: run_stage3_moe_pretrain.sh ARM full|trunk|decay-1p2b|smoke|bench|resume-bench}
root=$(cd "$(dirname "$0")/.." && pwd)
source "$root/configs/stage3-moe-1p029b.sh"

# --- budget, in steps of 208 x 2048 = 425,984 loss tokens ---
full_iters=17242          # 7,344,816,128 tokens = 1C for this MoE
full_decay_iters=3448     # final 20%, starts at step 13,795
warmup_iters=173          # first 1%
short_iters=2818          # 1,200,422,912 tokens
short_decay_iters=564     # final 20% of the short budget
short_branch=$((short_iters - short_decay_iters))   # 2254

global_batch=208
micro_batch=${STAGE3_MOE_MICRO_BATCH:-4}            # 4 x DP2 x accum26 = 208
data_root=${STAGE3_MOE_DATA_ROOT:-/home/jovyan/data/fineweb-edu-gpt2-megatron/data}
ckpt_root=${STAGE3_MOE_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3}
log_root=${STAGE3_MOE_LOG_ROOT:-/home/jovyan/hmoe-cloud/pretrain}
data_cache_args=()
if [[ -n ${STAGE3_MOE_DATA_CACHE_PATH:-} ]]; then
  data_cache_args=(--data-cache-path "$STAGE3_MOE_DATA_CACHE_PATH")
fi

case "$arm" in
  adamw_bf16_state_fp32) optimizer=adam; state_precision=fp32; compute=() ;;
  adamw_bf16_state_fp8)  optimizer=adam; state_precision=fp8;  compute=() ;;
  muon_bf16_state_fp32)  optimizer=muon; state_precision=fp32; compute=() ;;
  muon_bf16_state_fp8)   optimizer=muon; state_precision=fp8;  compute=() ;;
  adamw_fp8gemm_state_fp32) optimizer=adam; state_precision=fp32; compute=(--fp8-format hybrid --fp8-recipe delayed) ;;
  muon_fp8gemm_state_fp32)  optimizer=muon; state_precision=fp32; compute=(--fp8-format hybrid --fp8-recipe delayed) ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac
if ((${#compute[@]})); then
  if [[ -n ${STAGE3_MOE_FP8_AMAX_HISTORY_LEN:-} ]]; then
    compute+=(--fp8-amax-history-len "$STAGE3_MOE_FP8_AMAX_HISTORY_LEN")
  fi
  if [[ -n ${STAGE3_MOE_FP8_AMAX_COMPUTE_ALGO:-} ]]; then
    compute+=(--fp8-amax-compute-algo "$STAGE3_MOE_FP8_AMAX_COMPUTE_ALGO")
  fi
fi
probe_warmup=20
probe_measure=100
optimizer_args=()
[[ $optimizer == muon ]] && optimizer_args=("${STAGE3_MOE_MUON_ARGS[@]}")

trunk_dir="$ckpt_root/trunk/$arm"
decay_dir="$ckpt_root/1p2b/$arm"

run_id_eval="stage3-$arm-$mode${STAGE3_MOE_RUN_SUFFIX:+-$STAGE3_MOE_RUN_SUFFIX}"
eval_args=()
full_dir="$ckpt_root/${STAGE3_MOE_FULL_DIR:-1c}/$arm"

case "$mode" in
  full)
    # The 1C budget: 17,242 steps = 7,344,816,128 tokens, decay tail included.
    # This is not a new run. `trunk` was already launched on exactly this schedule
    # (--lr-decay-iters 17242, --lr-wsd-decay-iters 3448) and stopped at 2254 inside the
    # constant-LR plateau, so continuing it reproduces the single 1C curve. train_iters is
    # the only argument that changes, and it is precisely what makes the parameter
    # scheduler assert (wd_incr_steps = train_iters * global_batch), hence the override --
    # weight decay is constant here (start_wd = end_wd = 0.1) and every LR argument
    # matches, so the rebuilt schedule is identical and num_steps still comes from the
    # checkpoint.
    train_iters=$full_iters
    target_iters=$full_iters
    decay_iters=$full_decay_iters
    mkdir -p "$full_dir"
    # 363 x 38 = 13,794 = lr_decay_iters - lr_wsd_decay_iters, which the scheduler makes
    # the last step at peak LR (optimizer_param_scheduler.py:263-266: coeff is 1.0 while
    # num_steps <= wsd_anneal_start_). So a checkpoint lands exactly on the end of the
    # stable phase. The trunk's 322 cannot: it divides 2254, which is why it was chosen,
    # but 13,794 = 2*3*11^2*19 shares only a factor of 2 with it, and the nearest saves
    # would straddle the boundary at 13,524 and 13,846.
    #
    # Retaining that one iteration is the whole point of a WSD run. The stable phase is
    # constant-LR, so every shorter budget, every alternative decay tail and any later WSM
    # merge branch from it -- exactly as the 1.2B deliverables branched from the trunk at
    # 2254. Without it those cost a 13,794-step re-run. 13,794 is the only multiple of
    # itself below 17,242, so precisely one checkpoint is ever retained; everything else is
    # dropped as soon as its successor is on disk and the tracker points at it.
    save_args=(--save "$full_dir" --save-interval 363 --save-retain-interval 13794)
    load_args=(--load "$full_dir" --override-opt_param-scheduler)
    ;;
  trunk)
    # Stable phase only: stop at the branch point, never reaching decay.
    train_iters=$short_branch
    target_iters=$full_iters
    decay_iters=$full_decay_iters
    # Keep the latest for crash resume; retain the branch point permanently.
    save_args=(--save "$trunk_dir" --save-interval 322 --save-retain-interval "$short_branch")
    load_args=(--load "$trunk_dir")
    ;;
  decay-1p2b)
    train_iters=$short_iters
    target_iters=$short_iters
    decay_iters=$short_decay_iters
    mkdir -p "$decay_dir"
    # Seed the branch from the trunk with hardlinks: same filesystem, no copy,
    # no extra space, and the run can then load and save in one directory.
    if [[ ! -f "$decay_dir/latest_checkpointed_iteration.txt" ]]; then
      src=$(printf '%s/iter_%07d' "$trunk_dir" "$short_branch")
      [[ -d $src ]] || { echo "trunk branch point missing: $src" >&2; exit 2; }
      cp -al "$src" "$decay_dir/" 2>/dev/null || cp -a "$src" "$decay_dir/"
      echo "$short_branch" > "$decay_dir/latest_checkpointed_iteration.txt"
    fi
    # Only the endpoint is kept, and without optimizer state: we never resume
    # from it, it feeds validation and downstream evaluation.
    save_args=(--save "$decay_dir" --save-interval "$short_decay_iters" --no-save-optim --no-save-rng)
    load_args=(--load "$decay_dir" --override-opt_param-scheduler)
    ;;
  smoke)
    # Exercises the whole path cheaply: real data, 2 GPUs, checkpoint save and
    # resume on the alternate volume, loggers. Three checkpoints of ~7.3 GB per
    # run filled the 100 GB volume twice, so they are removed when the run ends;
    # set STAGE3_MOE_KEEP_SMOKE_CKPT=1 to keep them and re-run to test resume.
    train_iters=25
    target_iters=$full_iters
    decay_iters=$full_decay_iters
    smoke_dir="$ckpt_root/smoke/$arm${STAGE3_MOE_RUN_SUFFIX:+-$STAGE3_MOE_RUN_SUFFIX}"
    save_args=(--save "$smoke_dir" --save-interval 10)
    load_args=(--load "$smoke_dir")
    probe_warmup=5
    probe_measure=10
    ;;
  bench)
    # Throughput and peak memory only. No checkpoint traffic, so an NFS write never
    # lands inside the measured window and topologies stay comparable.
    train_iters=${STAGE3_MOE_BENCH_ITERS:-25}
    target_iters=$full_iters
    decay_iters=$full_decay_iters
    save_args=()
    load_args=()
    probe_warmup=5
    probe_measure=10
    ;;
  resume-bench)
    # Steady-state timing. `bench` measures 25 cold iterations of an untrained model,
    # which is a different regime: at iteration 2254 the router is balanced and the
    # expert GEMMs are small, and FP8 delayed scaling measures very differently there.
    # Resume the branch point, run a short window, never write a checkpoint.
    resume_bench_iters=${STAGE3_MOE_BENCH_ITERS:-150}
    if [[ ${STAGE3_MOE_MUON_SHADOW:-0} == 1 ]]; then
      [[ $arm == muon_bf16_state_fp32 ]] || {
        echo "Muon FP8 shadow diagnostic requires muon_bf16_state_fp32" >&2
        exit 2
      }
      resume_bench_iters=${STAGE3_MOE_BENCH_ITERS:-20}
      probe_warmup=0
      probe_measure=$resume_bench_iters
    fi
    train_iters=$((short_branch + resume_bench_iters))
    target_iters=$full_iters
    decay_iters=$full_decay_iters
    save_args=()
    # The scheduler derives wd_incr_steps from train_iters, which no longer equals the
    # trunk's, and refuses to load on the mismatch. Weight decay is constant here
    # (start_wd = end_wd = 0.1) and every LR argument matches the trunk, so overriding
    # rebuilds the identical schedule; num_steps still comes from the checkpoint.
    load_args=(--load "$trunk_dir" --override-opt_param-scheduler)
    ;;
  eval-downstream)
    # Score a finished checkpoint. --skip-train makes MCore build the model, load the
    # weights and go straight to evaluation, which is what the decay endpoints support:
    # they are saved with --no-save-optim, so there is no optimizer state to restore.
    # The budget has to be the one the endpoint was trained on. MCore restores
    # consumed_train_samples from the checkpoint and sizes the train dataset from
    # train_iters, and MegatronPretrainingSampler refuses a dataloader whose samples are
    # already consumed -- so a 1C endpoint (17,242) cannot be scored on the 1.2B schedule.
    case ${STAGE3_MOE_EVAL_BUDGET:-1p2b} in
      1p2b) train_iters=$short_iters; decay_iters=$short_decay_iters ;;
      1c)   train_iters=$full_iters;  decay_iters=$full_decay_iters ;;
      *) echo "unknown eval budget: $STAGE3_MOE_EVAL_BUDGET" >&2; exit 2 ;;
    esac
    target_iters=$train_iters
    eval_load=${STAGE3_MOE_EVAL_LOAD:?set STAGE3_MOE_EVAL_LOAD to the checkpoint directory}
    case ${STAGE3_MOE_EVAL_COMPUTE_MODE:-native} in
      native)
        eval_compute_mode=bf16
        [[ ${#compute[@]} -gt 0 ]] && eval_compute_mode=fp8_delayed_hybrid
        ;;
      bf16)
        [[ -n ${STAGE3_MOE_RUN_SUFFIX:-} ]] || {
          echo "BF16 eval override requires STAGE3_MOE_RUN_SUFFIX" >&2
          exit 2
        }
        compute=()
        eval_compute_mode=bf16
        ;;
      *) echo "unknown eval compute mode: $STAGE3_MOE_EVAL_COMPUTE_MODE" >&2; exit 2 ;;
    esac
    save_args=()
    # No --skip-train: it returns optimizer=None, and both the FP8-state bootstrap check
    # and the record's optimizer ledgers need a real optimizer object. Training is skipped
    # anyway because the checkpoint's iteration already equals train_iters.
    load_args=(--load "$eval_load" --no-load-optim --no-load-rng
               --override-opt_param-scheduler)
    probe_warmup=0
    probe_measure=1
    eval_args=(
      --stage3-eval-downstream "${STAGE3_MOE_EVAL_TASKS:-basic_v2_hellaswag,basic_v2_arc_easy,basic_v2_arc_challenge,basic_v2_piqa,basic_v2_gsm8k_gold_bpb_5shot}"
      --stage3-eval-artifact-dir "$log_root/$run_id_eval/downstream"
      --stage3-eval-batch-size "${STAGE3_MOE_EVAL_BATCH:-8}"
      --stage3-eval-compute-mode "$eval_compute_mode"
    )
    if [[ -n ${STAGE3_MOE_EVAL_LIMIT:-} ]]; then
      eval_args+=(--stage3-eval-limit "$STAGE3_MOE_EVAL_LIMIT")
    fi
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

run_id="stage3-$arm-$mode${STAGE3_MOE_RUN_SUFFIX:+-$STAGE3_MOE_RUN_SUFFIX}"
mkdir -p "$log_root/$run_id" "$trunk_dir"
if [[ ${STAGE3_MOE_MUON_SHADOW:-0} == 1 ]]; then
  export STAGE3_MOE_MUON_SHADOW_PATH="$log_root/$run_id/muon-fp8-shadow.jsonl"
  echo "MUON_FP8_SHADOW=$STAGE3_MOE_MUON_SHADOW_PATH"
else
  unset STAGE3_MOE_MUON_SHADOW_PATH
fi
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
# Transformer Engine loads cudart and friends from the pip nvidia packages; without
# these the run dies with "cudart shared object not found".
unset PYTHONNOUSERSITE
# That path exists only in the torch28 image. On any other one find exits 1, and
# under `set -eo pipefail` with stderr silenced the run used to die here without
# printing anything at all; let it carry on and fail loudly further down instead.
nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
  -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: - || true)
export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDNN_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cudnn
export CURAND_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/curand
export NVRTC_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cuda_nvrtc
# Without credentials wandb.init() would abort the run, so fall back to offline
# logging: the run still records everything and can be `wandb sync`ed later.
# A key in the environment (mlsub run --env WANDB_API_KEY=...) wins: that is how the
# self-hosted server is reached, and WANDB_BASE_URL has to travel with it.
if [[ -n ${WANDB_API_KEY:-} ]]; then
  echo "WANDB=online host=${WANDB_BASE_URL:-https://api.wandb.ai}"
elif [[ -f /home/jovyan/.wandb-key ]]; then
  export WANDB_API_KEY=$(cat /home/jovyan/.wandb-key)
else
  export WANDB_MODE=offline
  echo "WANDB=offline (no key in the environment and no /home/jovyan/.wandb-key); sync later with 'wandb sync'"
fi
# A 1C arm outlives one job, so every segment must land in the same W&B run instead of
# opening a new one. MCore calls wandb.init() without an id, so the env vars decide.
export WANDB_RUN_ID="${WANDB_RUN_ID:-$run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

# A missing logger must not kill a multi-hour run: drop the loggers instead.
logger_args=()
if python -c 'import wandb, torch.utils.tensorboard' >/dev/null 2>&1; then
  logger_args=(
    --tensorboard-dir "$log_root/$run_id/tensorboard"
    --tensorboard-log-interval 10
    --wandb-project "${STAGE3_MOE_WANDB_PROJECT:-hmoe-stage3}"
    --wandb-exp-name "$run_id"
    --wandb-save-dir "$log_root/$run_id"
  )
else
  echo "LOGGERS=absent (wandb/tensorboard not importable); running without them"
fi

# Apex's fused kernel is absent from the cloud image. Transformer Engine implements the
# fusion itself, so only the LM head (a plain ColumnParallelLinear) actually needs Apex,
# and the patched layer now degrades on its own instead of aborting the run. Asking for
# the fusion therefore buys it for every TE layer, including the expert GEMMs.
fusion_args=()
if [[ ${STAGE3_MOE_WGRAD_FUSION:-0} == 1 ]]; then
  echo "GRADIENT_ACCUMULATION_FUSION=requested (TE layers fuse, LM head falls back)"
elif ! python -c 'import fused_weight_gradient_mlp_cuda' >/dev/null 2>&1; then
  fusion_args=(--no-gradient-accumulation-fusion)
  echo "GRADIENT_ACCUMULATION_FUSION=disabled"
fi

# Throughput levers, kept off until a paired measurement justifies each one.
speed_args=()
if [[ ${STAGE3_MOE_OVERLAP_GRAD_REDUCE:-0} == 1 ]]; then
  speed_args+=(--overlap-grad-reduce)
fi
if [[ ${STAGE3_MOE_MOE_FUSIONS:-0} == 1 ]]; then
  speed_args+=(--moe-permute-fusion --moe-shared-expert-overlap)
fi
if [[ ${#speed_args[@]} -gt 0 ]]; then
  echo "SPEED_ARGS=${speed_args[*]}"
fi

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
# The result writer reads these; on real data the manifest hash is mandatory.
export STAGE3_MOE_RUN_ID="$run_id"
export STAGE3_MOE_SITE=cloudru
export STAGE3_MOE_IMAGE="${MLSUB_IMAGE:-torch28}"
export STAGE3_MOE_CONFIG_SHA256=$(sha256sum "$root/configs/stage3-moe-1p029b.sh" | awk '{print $1}')
export STAGE3_MOE_DATA_MANIFEST_SHA256=$(
  if [[ -f "$data_root/../artifact-manifest.json" ]]; then
    sha256sum "$data_root/../artifact-manifest.json" | awk '{print $1}'
  else
    printf '%s' 'AverageMetaheuristicsEnjoyer/fineweb-edu-gpt2-megatron' | sha256sum | awk '{print $1}'
  fi)
export STAGE3_MOE_MCORE_COMMIT="$STAGE3_MOE_MCORE_COMMIT"
export STAGE3_MOE_EO_COMMIT="$STAGE3_MOE_EO_COMMIT"
export STAGE3_MOE_GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)
export STAGE3_MOE_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
# Ask TE for the cuBLASLt version; the writer's ctypes fallback cannot find the
# shared object in this image.
export STAGE3_MOE_CUBLASLT=$(python -c 'import transformer_engine_torch as t; print(t.get_cublasLt_version())' 2>/dev/null || echo unknown)
export STAGE3_MOE_GPU_CLEAN_BEFORE=1 STAGE3_MOE_GPU_CLEAN_DURING=1 STAGE3_MOE_GPU_CLEAN_AFTER=1
echo "ARM=$arm MODE=$mode GPUS=$gpu_count micro_batch=$micro_batch global_batch=$global_batch"
echo "FP8_DEQUANT_CHUNK=${STAGE3_MOE_FP8_DEQUANT_CHUNK:-0} (0 = every state in FP32 at once)"
echo "SCHEDULE target_iters=$target_iters decay_iters=$decay_iters warmup=$warmup_iters train_iters=$train_iters"
echo "CKPT save=${save_args[*]} load=${load_args[*]}"

train_log="$log_root/$run_id/train-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "TRAIN_LOG=$train_log"
set +e
python -m torch.distributed.run --standalone --nproc-per-node "$gpu_count" \
  stage3_moe/pretrain_gpt.py \
  --stage3-arm "$arm" \
  --stage3-result-path "$log_root/$run_id/results.jsonl" \
  --stage3-warmup-steps "$probe_warmup" \
  --stage3-measure-steps "$probe_measure" \
  "${eval_args[@]}" \
  --optimizer-state-precision "$state_precision" \
  "${STAGE3_MOE_MODEL_ARGS[@]}" \
  "${STAGE3_MOE_ROUTER_ARGS[@]}" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size "${STAGE3_MOE_EP:-1}" \
  --expert-tensor-parallel-size 1 \
  --transformer-impl transformer_engine \
  --bf16 \
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --lr 1.63e-3 \
  --min-lr 1.63e-4 \
  --lr-decay-style WSD \
  --lr-decay-iters "$target_iters" \
  --lr-wsd-decay-iters "$decay_iters" \
  --lr-wsd-decay-style exponential \
  --lr-warmup-iters "$warmup_iters" \
  --weight-decay 0.1 \
  --clip-grad 1 \
  --micro-batch-size "$micro_batch" \
  --global-batch-size "$global_batch" \
  --train-iters "$train_iters" \
  --tokenizer-type NullTokenizer --vocab-size 50257 \
  --null-tokenizer-eod-id 50256 --null-tokenizer-pad-id -1 \
  --train-data-path "$data_root/train" \
  --valid-data-path "$data_root/development" \
  --test-data-path "$data_root/final" \
  "${data_cache_args[@]}" \
  --dataloader-type single \
  --num-workers 2 \
  --no-create-attention-mask-in-dataloader \
  --seed "${STAGE3_MOE_SEED:-1234}" \
  --eval-interval 250 \
  --eval-iters 32 \
  --log-interval 10 \
  --log-throughput \
  --timing-log-level 1 \
  "${logger_args[@]}" \
  --optimizer "$optimizer" \
  "${optimizer_args[@]}" \
  "${compute[@]}" \
  "${fusion_args[@]}" \
  "${speed_args[@]}" \
  --ckpt-format torch \
  "${save_args[@]}" \
  "${load_args[@]}" \
  >"$train_log" 2>&1
code=$?
set -e
echo "TRAIN_EXIT=$code"
tail -n 150 "$train_log"

if [[ $mode == smoke && ${STAGE3_MOE_KEEP_SMOKE_CKPT:-0} != 1 ]]; then
  echo "SMOKE_CKPT=removing $smoke_dir"
  rm -rf "$smoke_dir"
  df -h "$ckpt_root" | tail -1
fi
exit 0
