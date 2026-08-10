# Stage 4: dense optimizer-state quantization

## Scope and pins

This is a one-seed controlled pilot. It compares optimizer-state storage within
matched optimizer pairs, not AdamW against Muon.

- Host: `node207-23`.
- Project branch: `stage4/dense-optimizer-state-quantization`.
- MCore: `571370c829ca768fe37244f4e2e7f28d8accc4ab` (`0.18.2`).
- TransformerEngine: `b9d690e042b1c4e455214e7dab65d6d3512c05d6`
  (`2.16.0`).
- emerging-optimizers:
  `1effa026ff096b7fa1063ca2fba19d98be6e6cdf` (`0.2.0`).
- Runtime: pinned `nvcr.io/nvidia/pytorch:26.04-py3` SIF with PyTorch
  `2.12.0a0+0291f960b6.nv26.04.48445190`.
- Data: `AverageMetaheuristicsEnjoyer/fineweb-edu-gpt2-megatron` at
  `a2b8660e317ea2449c6ef57d9710a6e66c914751`, using the verified local copy.

No FP8 weight, activation, gradient, or GEMM option is enabled. All arms use
MCore BF16 mixed-precision training with FP32 master parameters and FP32
gradient accumulation. Only persistent optimizer states differ within a pair.

## Dense model

The decoder has 16 layers, hidden size 1,536, SwiGLU FFN size 4,096, 24
attention heads, MHA, RMSNorm, RoPE, no linear biases, and sequence length
2,048. Input and output embeddings are tied. NullTokenizer has 50,257 tokens;
padding to a multiple of 128 gives 50,304 embedding rows.

| Component | Parameters |
|---|---:|
| Tied token embedding | 77,266,944 |
| Attention and MLP matrices, 16 layers | 452,984,832 |
| Two RMSNorm weights per layer | 49,152 |
| Final RMSNorm | 1,536 |
| **Total** | **530,302,464** |

The implemented MCore model must reproduce this count before launch.

## Optimizers and quantization

AdamW uses TransformerEngine FusedAdam with betas `(0.9, 0.95)`, epsilon
`1e-8`, and decoupled weight decay. Muon uses MCore TensorParallelMuon with
momentum `0.95`, no Nesterov term, five quintic Newton-Schulz iterations,
spectral scaling, FP32 matmul precision `medium`, and extra scale factor `0.2`.
MCore routes matrices other than tied embeddings to Muon. The tied embedding
and all one-dimensional norm weights use the same AdamW configuration as the
AdamW arm.

The FP8 arms reproduce the state representation audited at Efficient-Training
commit `8de86400c9365e47ed88427543a25c99be094494`:

- E4M3 (`torch.float8_e4m3fn`) state values;
- flat groups of 128 values, including a padded final group;
- FP32 scale, expansion, and square-root-min/max per group;
- signed transforms for AdamW first moments and Muon momentum;
- nonnegative transforms for AdamW second moments;
- dynamic-range expansion rounded down in increments of `1/16`, with minimum
  `1/16`;
- deterministic PyTorch casts;
- dequantize to FP32 immediately before ordinary optimizer mathematics and
  requantize after every update.

The unquantized arms retain ordinary FP32 persistent states. Quantized AdamW
wraps the same FusedAdam update. Quantized Muon wraps the same native MCore
Muon update, while its fallback group is handled by the quantized AdamW
wrapper.

## Training recipe

- Micro batch: 4 sequences per GPU.
- Global batch: 208 sequences = 425,984 loss tokens.
- Steps: 17,242.
- Planned consumption: exactly 7,344,816,128 loss tokens.
- Parallelism per run: TP=PP=CP=1, DP=1, one H100 80GB.
- Learning rate: `3e-4`, WSD with a 1% linear warmup, a stable plateau through step 13,794, and exponential decay to `3e-5` over the final 3,448 steps.
- Weight decay: `0.1`.
- Gradient clipping: `1.0`.
- Initialization/data seed: `1234`.
- Dropout: zero.
- Data loader: single-pass, two workers.
- Checkpoints: steps 4,311, 8,622, 12,933, 13,794, and final step 17,242.
- The launcher uses `pilot-predecay` for the first 13,794 steps, then
  `pilot` resumes from the latest checkpoint and completes step 17,242.
  This creates the exact checkpoint immediately before WSD decay without
  repeating any training steps.
- Development loss: 18 iterations every 500 training steps. A separate
  endpoint job will evaluate 235 iterations on the frozen final split.

The independent prefixes are `/workspace/data/fineweb-edu-public/data/train`,
`development`, and `final`. No `--data-path` or `--split` argument is used.
NullTokenizer uses vocabulary size 50,257, EOD 50,256, and pad ID -1.

Legacy `--ckpt-format torch` is deliberate for the DP=1 pilot. MCore's generic
`torch_dist` optimizer sharding requires per-parameter state tensors to match
parameter shapes, while the per-group metadata does not. Checkpoint/resume is
still required and tested, but topology-portable sharded resume is not claimed.

## Four-run matrix

| Arm | Matrix optimizer | Fallback optimizer | Persistent state |
|---|---|---|---|
| `adamw_fp32` | AdamW | AdamW | FP32 moments |
| `adamw_fp8` | AdamW | AdamW | FP8 moments plus metadata |
| `muon_fp32` | Muon | AdamW | FP32 momentum/moments |
| `muon_fp8` | Muon | AdamW | FP8 momentum/moments plus metadata |

Within each pair, architecture, initialization, data order, token count, batch
sizes, schedule, optimizer hyperparameters, clipping, layout, validation, and
checkpoint steps are identical.

## Profiling and memory

Serial timing uses one idle H100, CUDA-event synchronization, compilation and
autotuning warmup, and repeated measured iterations. NVTX-visible ranges mark
state dequantization, optimizer mathematics, and state quantization. Nsight
Systems attributes kernels; Nsight Compute is used only if attribution needs a
memory-throughput measurement.

Memory accounting traverses actual state storages without double-counting
aliases. It reports state data, metadata, FP32 masters, model, gradients,
temporary allocations, peak allocated memory during `step`, and reserved
allocator memory separately. Ratios are reported for optimizer state only,
state plus masters, all persistent training state, and optimizer-step peak.
Muon matrix and fallback groups are reported separately.

## Evaluation and decision rule

All checkpoints use an identical unquantized BF16 forward. The evaluator must
preserve per-example outputs and use the `basic_v2` definitions for HellaSwag,
ARC-easy, ARC-challenge, PIQA, and GSM8K gold BPB 5-shot. Metrics are
`acc_v2`, `len_norm_v2`, and `bpb_v2` as applicable.

The canonical `basic_v2` task definitions are project-owned lm-eval tasks in
`stage4/eval_tasks`, loaded with `--include_path` using
`lm-eval==0.4.11`. They preserve the OLMo prompt and scoring conventions for
HellaSwag, ARC-easy, ARC-challenge, and PIQA, plus a deterministic five-shot
GSM8K gold-continuation BPB task. The secondary standard-task set uses the
same `lm-eval==0.4.11`: Wikitext and C4 rolling log-likelihood with zero
shots, plus Winogrande, OpenBookQA, and the MMLU group with five shots.
OpenBookQA is selected instead of BoolQ to add a four-choice science-reasoning
task; both are available in the pinned lm-eval release.

The exact evaluator manifest is `configs/stage4-evaluation.json`, and its
dependency pins are in `requirements-stage4-eval.txt`. Both evaluator paths
must preserve per-example outputs under a stable `(task, doc_id)` key.

For each matched pair, report paired bootstrap 95% confidence intervals.
McNemar is secondary for accuracy metrics; GSM8K uses paired bootstrap over
per-example BPB. Pilot margins are 1 absolute percentage point for
accuracy-style metrics, 1% relative for GSM8K BPB, and 1% relative for
validation loss. A result is pass, fail, or inconclusive against its margin.
These four runs measure one training seed and do not estimate training-seed
uncertainty. A passing pilot should be followed by at least two, preferably
three, seeds per cell.
