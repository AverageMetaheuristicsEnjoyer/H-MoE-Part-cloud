#!/usr/bin/env bash
# The third MoE shape of the memory/time sweep -- docs/membench.md.
#
# Ling 2.0's own sparsity, at this budget: 256 routed experts plus one shared, top-8,
# an expert-bank activation ratio of 3.50%. Every other dimension is the 1.029B arm
# unchanged, 18 layers included, so the two form a controlled pair whose only
# difference is the size of the expert bank.
#
# Total parameters 3.5x the 1.029B arm; active parameters 1.2% higher. That is the
# axis the sweep exists to separate: optimizer state follows the total, compute
# follows the active, and FP8 state quantization acts on the first alone.
#
# Predicted from the two measured shapes (18.25 GB per billion parameters plus a
# 2.33 GB intercept): 68 GB peak for FP32 AdamW at micro-batch 1, and the 80 GB wall
# somewhere around micro-batch 16. Where exactly is a result, not a guess.

STAGE3_MOE_TOTAL_PARAMETERS=3599183360
STAGE3_MOE_ACTIVE_PARAMETERS=283586048
STAGE3_MOE_ROUTED_EXPERTS=256
STAGE3_MOE_ROUTER_TOPK=8
STAGE3_MOE_MCORE_COMMIT=571370c829ca768fe37244f4e2e7f28d8accc4ab
STAGE3_MOE_VENDORED_MCORE_TREE=e2d9e7f73d24f3e60527a9d18d441a6de9411fe4
STAGE3_MOE_EO_COMMIT=1effa026ff096b7fa1063ca2fba19d98be6e6cdf
STAGE3_MOE_VENDORED_EO_TREE=e6b6cfd986bc0af4cd4f8e2c4ebedad16144e856

STAGE3_MOE_MODEL_ARGS=(
  --num-layers 18
  --hidden-size 1024
  --ffn-hidden-size 2816
  --moe-ffn-hidden-size 256
  --num-attention-heads 8
  --kv-channels 128
  --group-query-attention
  --num-query-groups 2
  --qk-layernorm
  --seq-length 2048
  --max-position-embeddings 2048
  --position-embedding-type rope
  --rotary-percent 1.0
  --swiglu
  --normalization RMSNorm
  --norm-epsilon 1e-5
  --disable-bias-linear
  --hidden-dropout 0
  --attention-dropout 0
  --init-method-std 0.006
  --untie-embeddings-and-output-weights
  --make-vocab-size-divisible-by 1
)

STAGE3_MOE_ROUTER_ARGS=(
  --num-experts 256
  --moe-layer-freq "[0]+[1]*17"
  --moe-shared-expert-intermediate-size 256
  --moe-router-topk 8
  --moe-router-load-balancing-type none
  --moe-aux-loss-coeff 0
  --moe-router-score-function sigmoid
  --moe-router-pre-softmax
  --moe-router-topk-scaling-factor 2.5
  --moe-router-enable-expert-bias
  --moe-router-bias-update-rate 1e-3
  --moe-router-dtype fp32
  --moe-token-dispatcher-type alltoall
  --moe-grouped-gemm
)

STAGE3_MOE_PARALLEL_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size ${STAGE3_MOE_EP:-1}
  --expert-tensor-parallel-size 1
)

STAGE3_MOE_TRAINING_ARGS=(
  --transformer-impl transformer_engine
  --bf16
  --adam-beta1 0.9
  --adam-beta2 0.95
  --adam-eps 1e-8
  --lr 1.63e-3
  --min-lr 1.63e-3
  --lr-decay-style constant
  --weight-decay 0.1
  --clip-grad 1
  --micro-batch-size ${STAGE3_MOE_MICRO_BATCH:-1}
  --global-batch-size ${STAGE3_MOE_GLOBAL_BATCH:-1}
  --eval-iters 0
  --eval-interval 1000000
  --tokenizer-type NullTokenizer
  --vocab-size 50257
  --null-tokenizer-eod-id 50256
  --null-tokenizer-pad-id -1
  --num-workers 0
  --no-create-attention-mask-in-dataloader
  --seed 1234
  --log-interval 1
  --timing-log-level 2
  --timing-log-option minmax
  --log-throughput
)

STAGE3_MOE_MUON_ARGS=(
  --muon-momentum 0.95
  --muon-nesterov
  --muon-num-ns-steps 5
  --muon-coefficient-type quintic
  --muon-scale-mode spectral
  --muon-extra-scale-factor 0.2
  --muon-fp32-matmul-prec medium
  --muon-scalar-optimizer adam
)
