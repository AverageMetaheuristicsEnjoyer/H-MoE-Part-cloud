#!/usr/bin/env bash
# The second MoE shape of the memory/time sweep -- derived in docs/membench.md.
#
# Everything the 1.029B arm fixes stays fixed: hidden 1,024, expert width 256 (so
# Ling's granularity G = 2*d_model/d_expert = 8), top-8 routed plus one shared
# expert, head dim 128, GQA 8/2, SwiGLU 2,816, GPT-2 50,257, sequence 2,048. Only
# the expert bank and the depth move: 64 -> 128 routed experts and 18 -> 20 layers.
#
# That doubles total parameters (1.029B -> 2.094B) while active parameters rise
# 7.4% (280.2M -> 301.0M), which is the point: the optimizer state -- what FP8
# state quantization acts on -- grows with the total, and the compute does not.
# The expert-bank activation ratio falls 13.85% -> 6.98%, inside the 4.7-10.9%
# band Ling validated its fits over and toward Ling 2.0's own 3.5%.
#
# The learning rate is deliberately left at the 1.029B value. A membench point
# runs 17 iterations and never trains; carrying a re-derived LR here would imply
# this shape had a calibrated schedule, and it does not.

STAGE3_MOE_TOTAL_PARAMETERS=2094088192
STAGE3_MOE_ACTIVE_PARAMETERS=301023232
STAGE3_MOE_ROUTED_EXPERTS=128
STAGE3_MOE_ROUTER_TOPK=8
STAGE3_MOE_MCORE_COMMIT=571370c829ca768fe37244f4e2e7f28d8accc4ab
STAGE3_MOE_VENDORED_MCORE_TREE=e2d9e7f73d24f3e60527a9d18d441a6de9411fe4
STAGE3_MOE_EO_COMMIT=1effa026ff096b7fa1063ca2fba19d98be6e6cdf
STAGE3_MOE_VENDORED_EO_TREE=e6b6cfd986bc0af4cd4f8e2c4ebedad16144e856

STAGE3_MOE_MODEL_ARGS=(
  --num-layers 20
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
  --num-experts 128
  --moe-layer-freq "[0]+[1]*19"
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
