#!/usr/bin/env bash
set -euo pipefail

variant=${1:?usage: run_stage4_dense.sh VARIANT smoke|reload|profile|gate|pilot-predecay|pilot}
mode=${2:?usage: run_stage4_dense.sh VARIANT smoke|reload|profile|gate|pilot-predecay|pilot}
root=$(cd "$(dirname "$0")/.." && pwd)
extra_args=()

schedule_args=()
case "$variant" in
  adamw_fp32) optimizer=adam; state_precision=fp32 ;;
  adamw_fp8) optimizer=adam; state_precision=fp8 ;;
  muon_fp32) optimizer=muon; state_precision=fp32 ;;
  muon_fp8) optimizer=muon; state_precision=fp8 ;;
  *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac

case "$mode" in
  smoke)
    train_iters=3
    global_batch=8
    save_interval=3
    eval_interval=2
    eval_iters=1
    log_interval=1
    output_group=stage4-smoke
    ;;
  reload)
    train_iters=3
    global_batch=8
    eval_interval=2
    eval_iters=1
    log_interval=1
    output_group=stage4-smoke
    ;;
  profile)
    train_iters=8
    global_batch=208
    eval_interval=1000000
    eval_iters=0
    log_interval=1
    output_group=stage4-profile
    extra_args+=(--timing-log-level 2 --logging-level 20)
    ;;
  gate)
    train_iters=2348
    global_batch=208
    save_interval=2348
    eval_interval=500
    eval_iters=18
    log_interval=10
    output_group=stage4-gate
    ;;
  pilot-predecay)
    train_iters=13794
    global_batch=208
    save_interval=4311
    eval_interval=500
    eval_iters=18
    log_interval=10
    output_group=stage4-pilot
    ;;
  pilot)
    train_iters=17242
    global_batch=208
    save_interval=4311
    eval_interval=500
    eval_iters=18
    log_interval=10
    output_group=stage4-pilot
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

if [[ "$mode" == gate || "$mode" == pilot || "$mode" == pilot-predecay ]]; then
  schedule_args=(--lr-decay-style WSD --lr-decay-iters 17242 --lr-wsd-decay-iters 3448 --lr-wsd-decay-style exponential)
else
  schedule_args=(--lr-decay-style cosine --lr-decay-iters "$train_iters")
fi

checkpoint_dir="/workspace/checkpoints/$output_group/$variant"
checkpoint_args=()
if [[ "$mode" != profile && "$mode" != gate && "$mode" != pilot-predecay ]]; then
  checkpoint_args+=(--load "$checkpoint_dir")
fi
if [[ "$mode" != reload && "$mode" != profile ]]; then
  checkpoint_args+=(--save "$checkpoint_dir" --save-interval "$save_interval")
fi

cd "$root"
mkdir -p runtime-tmp/triton-cache runtime-tmp/torchinductor-cache
exec scripts/node207_env.sh env TRITON_LIBCUDA_PATH=/.singularity.d/libs \
  TRITON_CACHE_DIR=/workspace/runtime-tmp/triton-cache \
  TORCHINDUCTOR_CACHE_DIR=/workspace/runtime-tmp/torchinductor-cache \
  STAGE4_REPORT_PEAK_MEMORY="${STAGE4_REPORT_PEAK_MEMORY:-0}" \
  python -m torch.distributed.run --standalone --nproc-per-node 1 \
  stage4/pretrain_gpt.py \
  --optimizer-state-precision "$state_precision" \
  --num-layers 16 \
  --hidden-size 1536 \
  --ffn-hidden-size 4096 \
  --num-attention-heads 24 \
  --seq-length 2048 \
  --max-position-embeddings 2048 \
  --position-embedding-type rope \
  --rotary-percent 1.0 \
  --swiglu \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --disable-bias-linear \
  --hidden-dropout 0.0 \
  --attention-dropout 0.0 \
  --make-vocab-size-divisible-by 128 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --bf16 \
  --optimizer "$optimizer" \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-8 \
  --muon-momentum 0.95 \
  --muon-scale-mode spectral \
  --muon-extra-scale-factor 0.2 \
  --muon-coefficient-type quintic \
  --muon-num-ns-steps 5 \
  --muon-fp32-matmul-prec medium \
  --lr 3e-4 \
  --min-lr 3e-5 \
  "${schedule_args[@]}" \
  --lr-warmup-fraction 0.01 \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --micro-batch-size 4 \
  --global-batch-size "$global_batch" \
  --train-iters "$train_iters" \
  --train-data-path /workspace/data/fineweb-edu-public/data/train \
  --valid-data-path /workspace/data/fineweb-edu-public/data/development \
  --test-data-path /workspace/data/fineweb-edu-public/data/final \
  --dataloader-type single \
  --num-workers 2 \
  --tokenizer-type NullTokenizer \
  --vocab-size 50257 \
  --null-tokenizer-eod-id 50256 \
  --null-tokenizer-pad-id -1 \
  --seed 1234 \
  --init-method-std 0.02 \
  --eval-interval "$eval_interval" \
  --eval-iters "$eval_iters" \
  "${checkpoint_args[@]}" \
  --ckpt-format torch \
  --log-interval "$log_interval" \
  "${extra_args[@]}"
