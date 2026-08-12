# Stage 3 MoE: first handoff and matched experiment plan

Status: prepared 2026-08-12.  No long pretraining was launched.  Labels in this
document have the following meanings: **measured** means observed in a live
checkout, allocated job, or file; **estimated** means static arithmetic or a
duration estimate; **hypothesis** means a claim that still needs a matched run.

## 1. Live-verified scope and current state

- **Measured — contract.**  The source presentation has SHA-256
  `287921444371876282fd36dd5fa81ce8271b73764ff3941879015c43316acf2f`.
  Stage 3 is slide 3, table `Google Shape;89;p14`, row 2.  Red single-strike
  removes the optimizer-state-only `WCT >= 1.15x` sentence, the INT8 and FP4
  bullets, and the old word `training` in the loss criterion.  The active
  requirements are: state-mode GPU-memory ratio at least 1.1x; delayed/HYBRID
  linear/activation WCT ratio at least 1.1x; validation parity and downstream
  degradation below 1%; at least three dense optimizers on models >=500M and
  two MoE optimizers on models >=1B total, with throughput and WCT analysis.
  Evidence: the original PPTX and `ppt/slides/slide3.xml` inside it.
- **Measured — authoritative Git.**  The live checkout is
  `node207:/home/user1/xandi281/H-MoE-Part`, branch
  `stage4/dense-optimizer-state-quantization`, commit
  `3940ec988595153f6805e4354b9d8784d3abf859`.  It had no Git remote and was
  dirty before this work; the pre-existing changes remain unstaged and
  untouched.  The dependency pins are MCore
  [`571370c8`](https://github.com/NVIDIA/Megatron-LM/commit/571370c829ca768fe37244f4e2e7f28d8accc4ab),
  [TE `b9d690e0`](https://github.com/NVIDIA/TransformerEngine/commit/b9d690e042b1c4e455214e7dab65d6d3512c05d6), and
  [emerging-optimizers `1effa026`](https://github.com/NVIDIA/emerging-optimizers/commit/1effa026ff096b7fa1063ca2fba19d98be6e6cdf).  Evidence:
  `node207:/home/user1/xandi281/H-MoE-Part/AGENTS.md`, `git status --short
  --branch`, `git log`, the nested-repository `git rev-parse HEAD` output, and
  the [live-evidence ledger](stage3-moe-live-evidence-2026-08-12.md).
- **Measured — no Stage 2 MoE baseline.**  Stage 2 was an API smoke: TE
  Linear/GroupedLinear forward/backward and an isolated one-parameter Muon
  step.  It explicitly excludes MoE dispatch/combine, throughput, peak VRAM,
  and relative speed.  Therefore none of the six pairs below may be labelled a
  reproduced Stage 2 baseline.  Evidence:
  [`docs/stage2-node207.md`](stage2-node207.md),
  [`scripts/stage2_smoke.py`](../scripts/stage2_smoke.py), and
  `node207:/home/user1/xandi281/H-MoE-Part/logs/stage2-smoke-gpu3.log`; exact
  source ranges are in the [evidence ledger](stage3-moe-live-evidence-2026-08-12.md).
- **Measured — node207 unavailable for a valid probe.**  All eight H100 80GB GPUs had
  live compute processes and 14.7--56.1 GiB allocated when checked.  Foreign
  Ray/VLLM/Python processes were present, so no GPU import or training probe was
  submitted.  A later tiny codec-unit-test was mistakenly run on occupied GPU0;
  it is a disclosed protocol error and is excluded from MoE evidence.  The host
  reported driver `595.71.05`; `/dev/md0` had about 672 GiB free.  Evidence:
  live `nvidia-smi --query-compute-apps`, per-GPU memory query, `ps`, `df`, and
  the incident record in the
  [evidence ledger](stage3-moe-live-evidence-2026-08-12.md).
- **Measured — public Cloud repo differs.**  Public `main` was
  [`3eddfd74`](https://github.com/AverageMetaheuristicsEnjoyer/H-MoE-Part-cloud/commit/3eddfd74530bfcb336a73c3eba1d7f0265ceec01).
  It had neither a true-MoE launcher nor a delayed/HYBRID smoke; its existing
  FP8 script uses `Float8BlockScaling(E4M3)`.  The implementation prepared here
  is additive and must be reviewed as a diff before any synchronization.
- **Measured — queue gate.**  `mlsub-queue` showed two running foreign/global
  jobs and one pending job; `mlsub list --active` showed no job owned by this
  account.  No new Cloud GPU job was submitted while the global queue was busy.
  Evidence: [queue snapshot and job ledger](stage3-moe-live-evidence-2026-08-12.md).

## 2. Selected baseline

The frozen logical architecture is the design in
[`docs/design.md`](design.md): 18 layers, hidden size 1024, one dense layer and
17 MoE layers, 64 routed experts, top-8, one shared expert, routed/shared width
256, dense SwiGLU width 2816, 8 query heads, 2 KV heads, GPT-2 vocabulary 50,257,
untied input/output embeddings, and no linear bias.

- **Estimated from the frozen design arithmetic:** **1,028,926,976 total
  parameters** and **280,243,712 active parameters per token** (27.24%).
- **Formal topology:** 4 H100 GPUs, `DP=4`, `EP=4`, `TP=1`, `PP=1`, `CP=1`,
  `ETP=1`; 16 routed experts per EP rank; sequence length 2048; micro-batch 2
  sequences/GPU; global batch 208 sequences = 425,984 loss tokens; 26 gradient
  accumulation micro-batches per rank.  Expert dispatch is all-to-all and
  expert weights remain separate 2-D tensors.
- **Short-smoke projection:** one GPU, all parallel dimensions 1, micro-batch 1,
  global batch 1.  This projection verifies construction, optimizer assignment,
  state dtype, memory accounting, and finite steps.  It is not evidence for the
  formal 4-GPU throughput or WCT claims.

The formal topology uses EP=4 because it exercises sparse expert ownership and
all-to-all communication rather than measuring a replicated dense-like proxy.
Before a long run it must be compared against the matched 4-GPU `DP=4, EP=1`
layout; topology is held fixed inside every precision pair.

### Runnable router deviation

**Measured.**  The frozen memo requested softmax plus aux-loss-free expert bias,
but pinned MCore rejects that combination: expert bias accepts only sigmoid or
sqrt-softplus.  The bounded probe therefore uses sigmoid, pre-softmax top-8,
FP32 router arithmetic, expert bias update rate `1e-3`, and no auxiliary loss.
This is an explicit architecture deviation, not silent equivalence.  A matched
long run is blocked until sigmoid is approved or the centered softmax-bias
controller is implemented and tested.  Evidence:
[`TransformerConfig`](https://github.com/NVIDIA/Megatron-LM/blob/571370c829ca768fe37244f4e2e7f28d8accc4ab/megatron/core/transformer/transformer_config.py#L2060-L2067).

## 3. The six nearest bounded probes

All commands use mock data, one H100, sequence length 2048, micro/global batch
1, and 3 warm-up plus 5 measured steps.  `probe` is deliberately too short for
a Stage 3 WCT or quality verdict; a healthy result is recorded as
`probe_pass/stage3_inconclusive`.

| # | Goal and exact command | Matched baseline | Recorded metric | Probe pass/fail criterion | Estimated duration |
|---|---|---|---|---|---:|
| 1 | AdamW BF16/full-state baseline: `CUDA_VISIBLE_DEVICES=0 scripts/run_stage3_moe_probe.sh adamw_bf16_state_fp32 mock --protocol probe` | self/reference | total/active count; Adam group; allocated/reserved peak; persistent bytes; token/s; optimizer/full-step/E2E; finite loss | exit 0; exact logical count; only `exp_avg,exp_avg_sq` FP32; finite five measured steps | 6--10 min |
| 2 | AdamW state treatment: `CUDA_VISIBLE_DEVICES=0 scripts/run_stage3_moe_probe.sh adamw_bf16_state_fp8 mock --protocol probe` | #1, same seed/order/batches/GPU | same metrics plus per-state payload dtype and metadata | exit 0; `exp_avg=E4M3`, `exp_avg_sq=E5M2`; persistent `B_base/B_treat>=1.10`; finite; peak ratio reported but not promoted | 7--12 min |
| 3 | Muon BF16/full-state baseline: `CUDA_VISIBLE_DEVICES=0 scripts/run_stage3_moe_probe.sh muon_bf16_state_fp32 mock --protocol probe` | self/reference | Muon/fallback parameter and byte ledger plus common metrics | exit 0; router/embedding/norm in AdamW; matrix FC1 gate/up split; FP32 `momentum_buffer` and fallback moments; finite | 8--15 min |
| 4 | Muon state treatment: `CUDA_VISIBLE_DEVICES=0 scripts/run_stage3_moe_probe.sh muon_bf16_state_fp8 mock --protocol probe` | #3 only | same metrics plus payload codec | exit 0; Muon momentum E4M3 max-abs; fallback Adam `exp_avg=E4M3`, `exp_avg_sq=E5M2`; persistent ratio >=1.10; finite | 10--18 min |
| 5 | AdamW delayed/HYBRID compute: `CUDA_VISIBLE_DEVICES=0 scripts/run_stage3_moe_probe.sh adamw_fp8gemm_state_fp32 mock --protocol probe` | #1 only | TE recipe; token/s; optimizer/full-step/E2E; peak; finite loss | exit 0; actual `DelayedScaling(Format.HYBRID)`; both Adam states FP32; finite. WCT ratio is measured but remains inconclusive at 5 steps | 6--10 min |
| 6 | Muon delayed/HYBRID compute: `CUDA_VISIBLE_DEVICES=0 scripts/run_stage3_moe_probe.sh muon_fp8gemm_state_fp32 mock --protocol probe` | #3 only | TE recipe; full Muon/fallback ledger and common timing | exit 0; delayed/HYBRID; all states FP32; intended Muon assignment; finite. WCT ratio remains inconclusive | 8--15 min |

Before each command: re-run the site queue command, `nvidia-smi
--query-compute-apps`, and per-GPU allocated-memory query; proceed only on a
genuinely idle allocated GPU.  Run the two members of each pair sequentially on
the same GPU, counterbalance order across at least three replicates, and retain
raw JSONL plus stdout.  No checkpoint, W&B, HF, or long token budget is used.

For a stronger but still bounded timing diagnostic, use `--protocol probe
--warmup-steps 20 --measure-steps 100`.  That
diagnostic can gate engineering work; only fixed-token end-to-end pretraining
can satisfy the contract's WCT claim.

## 4. Optimizer assignment and persistent-state ledger

The table below is **estimated/accounted**, excludes FP32 master weights and
tiny step scalars, and must be replaced by a unique-storage runtime traversal.
Payload, quantization metadata, scalar metadata, and master weights are separate
schema fields.

| Arm/group | Total params | Active params/token | FP32 persistent state | FP8 payload + FP32 metadata | Actually quantized in treatment |
|---|---:|---:|---:|---:|---|
| AdamW, all parameters | 1,028,926,976 | 280,243,712 | 8,231,415,808 B | 2,250,777,760 B | `exp_avg`: DRE E4M3; `exp_avg_sq`: DRE E5M2; three FP32 metadata arrays/state/128 values |
| Muon matrix group | 924,844,032 | 176,160,768 | 3,699,376,128 B | 953,745,408 B | `momentum_buffer`: block E4M3 round-to-nearest, one FP32 max-abs scale/128; FP32 recurrence and NS |
| Muon AdamW fallback | 104,082,944 | 104,082,944 | 832,663,552 B | 227,681,440 B | fallback `exp_avg`: DRE E4M3; `exp_avg_sq`: DRE E5M2 |
| Muon arm total | 1,028,926,976 | 280,243,712 | 4,532,039,680 B | 1,181,426,848 B | union of the two rows above |

The intended Muon group contains attention, dense-FFN, and separate expert
matrices.  The AdamW fallback contains the input/output embeddings, all norms,
and 17 router weights.  Stock MCore instead sends the 2-D router weights to
Muon and orthogonalizes a fused SwiGLU FC1 jointly; the Stage 3 bootstrap adds
the narrow router override and split.  Therefore the counts above remain
estimated until the built-model audit asserts 924,844,032 / 104,082,944.

At the formal EP=4 topology, estimated local parameters are 283,115,520 Muon
matrix and 104,082,944 fallback parameters per rank.  Corresponding state-only
bytes per rank are 1,965,125,632 FP32 versus 519,644,320 quantized.  The result
file reports both logical-global and local physical storage; percentages always
use local training-process allocated/reserved memory on a named rank/GPU.

## 5. Matched comparison rules and gates

There are two independent axes.  No main pair may change both:

1. **State axis, BF16 GEMMs:** AdamW FP32 states vs AdamW FP8 states; Muon FP32
   states vs Muon FP8 states.
2. **Compute axis, full-precision states:** AdamW BF16 GEMMs vs AdamW
   delayed/HYBRID GEMMs; Muon BF16 GEMMs vs Muon delayed/HYBRID GEMMs.

Joint FP8 GEMM plus FP8 states is a later hypothesis only after both component
pairs pass.  AdamW and Muon results are never treated as an optimizer-quality
comparison.

Every result records `torch.cuda.max_memory_allocated`,
`torch.cuda.max_memory_reserved`, unique persistent state bytes, tokens/s,
optimizer-inner-step, full-step, subprocess E2E WCT, train/validation loss,
downstream scores, total/active parameters, micro/global batch, gradient
accumulation, exact GPU UUID/count, topology, source/image hashes, run order,
and all raw measured steps.  Tokens/s is
`global_batch_sequences * seq_len / full_step_seconds`; MCore's built-in
`--log-throughput` is TFLOP/s/GPU and is not substituted for it.

- State memory pass: the one-sided paired 95% CI lower bound for
  `allocated_base / allocated_treatment` is at least 1.10 at identical work and
  denominators.  Point estimate, reserved, and persistent-state ratios receive
  separate verdicts.  The historical `treatment <= 0.90 * baseline` gate is
  also shown as a stricter sensitivity, not substituted for the PPTX wording.
- Compute WCT pass: `WCT_base / WCT_treatment >= 1.10` over identical fixed
  tokens; the one-sided paired 95% CI lower bound must also be at least 1.10.
  Throughput, full-step, and optimizer-step effects are reported independently.
- Validation parity: the one-sided paired 95% upper bound on relative loss
  degradation is below 1% at identical tokens/checkpoints.
- Downstream pass: for every preregistered task, the one-sided paired-bootstrap
  95% upper bound on relative degradation is below 1%; accuracy percentage-point
  changes and paired McNemar diagnostics are reported alongside it.  The current
  dense task set is in [`configs/stage4-evaluation.json`](../configs/stage4-evaluation.json),
  but no committed MoE evaluator or MoE checkpoint exists yet.
- A mismatch in optimizer, seed/data order, batch, token budget, topology,
  kernels, non-target precision, or GPU class makes the pair `invalid`, not a
  pass/fail.  A bounded healthy probe with too few timing steps or no checkpoint
  quality is `inconclusive` for Stage 3.

## 6. Cloud image verdict

- **Measured.**  Job `lm-mpi-job-0bcb7d5c-78c2-44c0-b0cb-cb10688e3790`
  completed in 33 s on H100 80GB with torch `2.8.0+cu128`, TE 2.16.0 and
  cuBLASLt 120804; source import and the Triton state round-trip passed.  The
  entrypoint is
  [`cloud_stage4_import.sh`](https://github.com/AverageMetaheuristicsEnjoyer/H-MoE-Part-cloud/blob/3eddfd74530bfcb336a73c3eba1d7f0265ceec01/scripts/cloud_stage4_import.sh).
- **Measured.**  Job `lm-mpi-job-06803109-383b-4f85-8440-b14c34049ca1`
  failed its `Float8BlockScaling` smoke with the explicit CUDA >=12.9 assertion.
  That script tests a different recipe.
- **Source-verified verdict.**  A cu129 image is **not required** for the requested
  Hopper delayed/HYBRID path: TE maps HYBRID to E4M3 forward and E5M2 backward,
  and delayed scaling uses the ordinary Hopper FP8 support check.  CUDA >=12.9
  is specific to `Float8BlockScaling` in this TE version.  Primary evidence:
  [TE recipes](https://github.com/NVIDIA/TransformerEngine/blob/4220403e831d29e93868f7793693ea83f6b8b05b/transformer_engine/common/recipe/__init__.py#L27-L48)
  and [support checks](https://github.com/NVIDIA/TransformerEngine/blob/4220403e831d29e93868f7793693ea83f6b8b05b/transformer_engine/pytorch/quantization.py#L149-L185).
- **Unverified.**  The current torch28 driver, delayed Linear F/B, true-MoE
  construction, and one MoE training step still require the new bounded Cloud
  entrypoint on an idle queue.  Therefore there is no cu129 build request in
  this handoff.  If block scaling becomes a future branch, note that grouped
  block-scaled GEMM additionally requires cuBLAS 13.4 in TE 2.16
  ([source](https://github.com/NVIDIA/TransformerEngine/blob/4220403e831d29e93868f7793693ea83f6b8b05b/transformer_engine/common/gemm/cublaslt_grouped_gemm.cu#L923-L950)); merely adding a cu129 registry tag would not prove grouped-MoE readiness.

## 7. Why dense results do not transfer automatically

- Expert tensors add many small per-expert scale tails and thousands of Muon NS
  calls; a dense weighted-average codec error does not characterize singular
  direction error for these matrices.
- Routing can amplify small parameter/activation perturbations into top-k flips,
  load skew, padding, and different expert update histories.  Log globally
  reduced tokens/expert, min/mean/max, CV, top-k overlap, and dropped tokens.
- EP changes physical state ownership and introduces all-to-all dispatch/combine;
  memory ratios and full-step speed from replicated dense runs do not include
  this communication.
- Routed experts receive sparse, nonuniform token batches even though their
  parameters are stepped each iteration.  Measure per-expert gradient/state
  norms, zero-update fraction, codec cosine/relative-L2 error, and scale
  saturation by expert and layer.
- Muon's AdamW fallback is a material 104.08M-parameter component.  Its moments,
  metadata, and update time must be counted separately from Muon momentum.
- Dense Muon DRE and SOAP moments-only observations are diagnostics, not MoE
  evidence.  SOAP is only a future branch with an independent preconditioner
  audit.

Cheap diagnostics before any long run are: CPU static parameter arithmetic;
one-GPU built-model/optimizer ledger; synthetic per-expert FP8 round-trip error;
two-step resume equivalence; bounded BF16/FP8 routing-overlap replay; and the
4-GPU EP=1 versus EP=4 topology probe.  None consumes a 0.5C/1C/2C budget.

## 8. Prepared implementation

Public feature commit `dfc66c19932ac167e356153d84a3bce669d0cd7b` adds only
the Stage 3 paths: the frozen config, six-arm launcher, Cloud delayed/HYBRID
entrypoint, hybrid optimizer-state codecs, Muon assignment/split patch,
machine-readable JSONL schema/writer/pair validator, and their tests.  Existing
Stage 4 and `third_party` files are unchanged.  The exact paths are:

```text
configs/stage3-moe-1p029b.sh
scripts/run_stage3_moe_probe.sh
scripts/cloud_moe_fp8_delayed_smoke.sh
stage3_moe/{__init__,muon,optimizer_states,pair_results,pretrain_gpt,result_writer}.py
stage3_moe/result.schema.json
tests/stage3_moe/{test_launcher_contract,test_optimizer_contract,test_pair_results}.py
```

**Measured verification:** all six launcher dry-runs and shell/AST/JSON checks
pass; the pair/schema suite passes 11/11.  The final CPU-only run of the full
Stage 3 test directory in the pinned node207 container passes 28 tests and
skips four CUDA cases (plus five passing subtests).  The CUDA unit
incident disclosed in the evidence ledger is not a true-MoE smoke and is not
counted here.  No W&B/HF publication, checkpoint deletion, reset, or long run
was performed.

## 9. Long-pretraining blockers

1. No honest Stage 2 MoE checkpoint, logs, or paired performance baseline.
2. The six new arms have not yet passed a true-model GPU smoke; node207 and the
   Cloud global queue were occupied.
3. Runtime parameter/state enumeration must confirm total/active counts,
   router fallback, separate 2-D experts, SwiGLU split, and exact local/global
   unique-storage bytes.
4. The sigmoid router deviation needs approval or a tested centered softmax
   controller; routing health gates need data.
5. Formal EP=4 versus replicated EP=1 topology evidence is missing.
6. No committed MoE downstream runner or paired MoE checkpoints exist.
7. Validation/downstream parity requires matched training; the current probes
   are intentionally incapable of satisfying those gates.
