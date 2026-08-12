#!/usr/bin/env bash

STAGE3_MOE_TOTAL_PARAMETERS=1028926976
STAGE3_MOE_ACTIVE_PARAMETERS=280243712
STAGE3_MOE_ROUTED_EXPERTS=64
STAGE3_MOE_ROUTER_TOPK=8
STAGE3_MOE_MCORE_COMMIT=571370c829ca768fe37244f4e2e7f28d8accc4ab
STAGE3_MOE_VENDORED_MCORE_TREE=e56265e78f086c1ff831ed40c30e50395e236a83
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
  --num-experts 64
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
  --expert-model-parallel-size 1
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
  --micro-batch-size 1
  --global-batch-size 1
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
