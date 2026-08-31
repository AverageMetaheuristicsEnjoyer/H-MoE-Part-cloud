#!/usr/bin/env bash

DENSE_1P028B_PARAMETERS=1028003840

DENSE_1P028B_MODEL_ARGS=(
  --num-layers 16
  --hidden-size 2048
  --ffn-hidden-size 5632
  --num-attention-heads 16
  --kv-channels 128
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
  --init-method-std 0.02
  --untie-embeddings-and-output-weights
  --make-vocab-size-divisible-by 1
)
