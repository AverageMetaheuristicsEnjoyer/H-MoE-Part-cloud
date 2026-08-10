# FP8 low-bit MoE pretraining: research memo and implementation plan

Status: design only, 2026-07-28. No training code has been written and no server
state has been changed.

## Executive decision

Use direct Megatron-LM/Megatron-Core with TransformerEngine (TE) and NVIDIA
Emerging-Optimizers (EO). Do not add Megatron Bridge. The target model is an
18-layer, 1.029B-total / 280.2M-active MoE with 64 routed experts, top-8
routing, one shared expert, expert granularity 8, and one initial dense layer.
The provisional compute-optimal budget is 7.345B GPT-2 tokens, not 20 times
total parameters.

The target Hopper compute recipe is TE blockwise E4M3: 1x128 activation and
gradient blocks, 128x128 weight blocks, FP32 scales and accumulation, and BF16
outputs. The AdamW state candidate is COAT-style E4M3 with 128-element groups
and Dynamic Range Expansion (DRE). Muon's persistent momentum is also stored in
blockwise E4M3, but its momentum update, normalization, Newton-Schulz (NS)
interface, and final master-weight update remain high precision. There is no
published, validated FP8 Muon-state recipe to copy; this is the main research
risk and likely the novel part of the project.

The design is feasible in memory without model parallelism on an H100 80GB.
The initial 2-4 GPU reference therefore uses data parallelism with TP=PP=CP=EP
= ETP = 1. Replicated versus EP-sharded routed experts remains an
evidence-gated systems decision: both will be benchmarked on clean GPUs before
the final run. The shared expert follows the dense MLP's DP layout in both.

## 1. What the literature does and does not establish

The two local papers were read in full:

- The [Ling 2.0 technical report](../tmp/literature/ling-moe.pdf) is a primary
  source for the architecture, FP8 recipe, and the plotted MoE hyperparameter
  and model-data fits. Its released models start at 16B total parameters, so it
  is not a reference configuration for a 1B-total model.
- NVIDIA's [Megatron-Core MoE report](../tmp/literature/nvidia-moe.pdf) is a
  primary systems source. It establishes the current Megatron/TE mechanisms and
  Hopper precision guidance. It is not a scaling-law paper and supplies neither
  a 1B reference model nor a fitted optimal token count.

The main supplemental scaling sources are [Towards Greater Leverage
(Ling)](https://arxiv.org/abs/2507.17702), which studies activation ratio,
granularity, sharing, and layer placement, and [Joint MoE Scaling
Laws](https://arxiv.org/abs/2502.05172), which fits active-parameter/data/expert
laws on more than 280 dense and Switch-style models. The latter uses top-1,
coarse experts and at most 32 experts; it is a useful independent check, not a
direct formula for the proposed fine-grained top-8 architecture. A 2025
[fine-grained MoE scale study](https://arxiv.org/abs/2506.02890) supports the
direction of using fine-grained experts, but its models are far larger and do
not determine the 1B configuration.

Five discrepancies must remain visible:

1. Ling's 64-routed/top-4/shared-expert fit, which used G=2, predicts a peak learning rate
   around 1.63e-3 here. Its hyperparameter law was validated only through a
   10.9% expert-bank activation ratio, while this proposal uses 13.85%.
   The Joint MoE formula predicts 3.24e-4 with the literal 64 experts or
   5.59e-4 if its top-1 expert count is heuristically replaced by the effective
   bank multiplier `65/9`. These are not equivalent experiments. Ling is the
   closer architecture, but neither formula settles the value; a short
   preregistered learning-rate sweep is mandatory.
2. Ling finds that lower expert activation ratios continue to help, including
   ratios as low as 1/128. That does not imply that a 1B-total model should
   mechanically copy Ling's 256-expert model. At this small total budget, the
   attention, vocabulary, and shared components set a floor on active
   parameters, and extremely narrow experts are poor H100 GEMMs.
3. Ling's Appendix-C attention equation omits the usual factor of two for
   projection multiply-adds while retaining it for FFNs and QK/PV. The memo
   treats this as a typesetting error, uses the physically correct count, and
   reports the literal-reading sensitivity separately. The two readings shift
   the projected token budget by about 8%.
4. Ling fits on its private multilingual corpus with a 126,464-token BPE,
   while Joint MoE uses a FineWeb-derived corpus and GPT-2 BPE but a top-1
   coarse architecture. Neither is a matched scaling law for FineWeb-100BT and
   this top-8 model. The token count below is therefore a literature-grounded
   prior, not a measured optimum for this dataset.
5. Ling's granularity sweep narrows experts while proportionally increasing
   their count to hold activation ratio and per-token compute fixed. `E=64,
   G=8` is not one of those matched points; it is a 1B/H100-constrained
   extrapolation chosen to keep expert dimensions divisible by 128.

## 2. Stack decision

### Criteria

The comparison is made on the requested axes:

- **State access:** how directly we can replace persistent optimizer-state
  storage while retaining FP32 update arithmetic and correct checkpointing.
- **Hopper FP8:** quality and maturity of FP8 compute specifically on H100,
  rather than Blackwell-only FP4/MXFP8 features.
- **MoE path:** grouped GEMM, token dispatch/combine, and sensible operation on
  2-4 GPUs.
- **setup risk:** amount of configuration and rapidly moving integration code
  that can obscure a numerical experiment.
- **nearby reference:** availability of a roughly 1B-total MoE recipe.

| Stack | State access | Hopper FP8 | 2-4 GPU MoE path | Setup risk | Nearby reference |
|---|---|---|---|---|---|
| **Direct Megatron-Core + TE + EO** | Medium. EO Muon is small Python code; Megatron wrappers and checkpointing are the complication. Stock precision-aware Adam exposes FP8 moments but requires top-level Adam and the distributed optimizer. | **Best fit.** TE 2.16 supports Hopper blockwise FP8 and grouped experts. | TE GroupedLinear/GroupedMLP, fused permutation/router paths, all-to-all, DeepEP; all can be introduced incrementally. | High, controlled by one narrow config and exact source pins. | No exact 1B model; basic MoE examples and much larger production recipes exist. |
| Megatron Bridge | Changes still land in MCore/EO. | Same engine. | Same engine. | Highest because it adds another recipe/config layer. | OLMoE 7B total / 1.3B active is still far away. |
| TorchTitan | Best local optimizer ergonomics and state-dict layout. | Good row-wise H100 FP8 for ordinary Linear; its current quantized grouped-MoE path centers on Blackwell MXFP8 rather than the required Hopper block recipe. | BF16 grouped GEMM, EP, and DeepEP exist, but the target combination needs more custom work. | Medium and tied to fast-moving PyTorch nightlies. | No close 1B-total MoE recipe. |
| NeMo AutoModel | Good optimizer factory and typed configs. | Strong TE-based current/block FP8. | Grouped experts, FSDP2, DeepEP. | Lower than raw MCore, but adds an HF-model layer that does not simplify the custom Muon state. | H100 Qwen3-MoE-30B and Moonlight-size configs, not 1B total. |
| MegaBlocks / LLM Foundry | State changes are possible. | No equally integrated current Hopper block-FP8 plus NVIDIA Muon path. | Good dropless sparse kernels. | Medium. | DBRX-scale rather than 1B. |
| DeepSpeed | State changes are possible. | No comparable current official Hopper block-FP8 grouped-expert recipe. | Older MoE path; Megatron-DeepSpeed is not the current NVIDIA path. | Medium-high. | No suitable current reference. |

### Recommendation

Keep the proposed default stack, but use it directly:

- Megatron-Core for the model and distributed training.
- TE for attention/linear/MoE kernels and Hopper blockwise FP8.
- EO Muon as the behavioral reference and the smallest code surface to modify.
- No Bridge, NeMo trainer, Hydra hierarchy, or general configuration framework.

The decisive reason is that this is the only current stack combining the
required NVIDIA Muon semantics, Hopper block-FP8 grouped experts, and mature EP
communication. TorchTitan would make an isolated optimizer edit more pleasant,
but it moves more of the requested MoE/FP8 integration into project-owned code.

The first path uses separate 2D expert weights
(`moe_single_grouped_weight=false`). A packed 3D grouped weight falls through
MCore's matrix-aware Muon selection and, more importantly, must not be
orthogonalized as one matrix. MCore splits fused QKV weights for Muon but does
not split a fused SwiGLU FC1 into its gate and up projections. The proposed
Muon semantics therefore treat gate, up, and down as three logical matrices:
update the persistent fused FC1 momentum elementwise, split gate/up before NS,
orthogonalize them independently, and concatenate their updates. This is
invariant to an implementation packing choice, unlike orthogonalizing
`[gate; up]` as one matrix. It creates 3,315 small routed/shared-expert logical
matrices (`17*65*3`), so this split is a user-visible decision and NS launch
overhead is a measured-risk item. If the optimizer exceeds 5% of steady-state
step time, batched expert NS becomes a material fork and will be brought back
for approval.

### Reproducible starting pins

Use the dependency graph of the MCore release, not a hand-assembled set:

| Component | Initial pin | Reason |
|---|---|---|
| Container | `nvcr.io/nvidia/pytorch:26.04-py3`, then record its immutable digest | This is MCore `core_v0.18.2`'s own dev image. It contains Python 3.12, PyTorch `2.12.0a0+0291f960b6`, and CUDA 13.2.1. |
| Megatron-LM/Core | `core_v0.18.2`, commit `571370c829ca768fe37244f4e2e7f28d8accc4ab` | Fixed release tag. |
| TransformerEngine | commit `b9d690e042b1c4e455214e7dab65d6d3512c05d6` | Exact MCore source pin on the TE 2.16 post-release branch; do not silently use the TE version preinstalled in the base image. |
| Emerging-Optimizers | `v0.2.0`, commit `1effa026ff096b7fa1063ca2fba19d98be6e6cdf` | Exact MCore pin. EO `v0.3.1` exists, but upgrading it is a separate change. |
| COAT reference only | commit `80ec99f47aaa09231b07ace1fd04c30a1e30ec18` | Exact source inspected for DRE discretization, metadata allocation, and fused-kernel behavior; project code will not import COAT as a training framework. |

The image, driver, CUDA compatibility, container runtime, NCCL, and H100
topology must be checked on node207 before installation. If its driver cannot
run CUDA 13.2, do not upgrade the host or improvise an environment: propose a
compatible pinned stack first. Sources for the pins are the
[`core_v0.18.2` project file](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.2/pyproject.toml),
the [NGC 26.04 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-04.html),
and the official [TE release notes](https://docs.nvidia.com/deeplearning/transformer-engine/release-notes/index.html).

## 3. Proposed model and training configuration

### Parameter table

| Parameter | Proposed value | Source/reason | Alternative rejected |
|---|---:|---|---|
| Total / active parameters | **1.029B / 280.2M**, 3.67x total/active | Constructed under the 1B-total constraint; Ling says expert activation ratio is the primary MoE leverage axis. | `20 x total params` token rule and Ling-mini's 16B/1.4B shape do not apply. |
| Layers / hidden size | **18 / 1,024** | Simple aspect ratio with H100-friendly multiples; parameter arithmetic below. | A copied 2,048-wide Ling model cannot fit the total budget with a useful expert pool. |
| Vocabulary | **GPT-2 50,257**, separate input and output matrices; `--make-vocab-size-divisible-by 1 --untie-embeddings-and-output-weights` | FineWeb's supplied `token_count` is defined by GPT-2; Joint MoE also uses it and counts separate embedding/unembedding. These flags prevent MCore's defaults from padding to 50,304 or tying the two matrices, either of which would invalidate the frozen parameter/state counts. | A new tokenizer adds an uncontrolled data axis; Ling's 156K multilingual vocabulary spends too much of a 1B budget. Accepting the padded/tied MCore defaults would make this a different model. |
| Attention | 8 query heads, 2 KV heads, head dim 128; GQA; pre-RMSNorm; QKNorm; full RoPE | Ling uses GQA, QKNorm, RMSNorm and head dim 128. QKNorm is specifically reported as helpful in low precision. | MHA wastes KV parameters; partial RoPE targets length extrapolation not needed at 2K. |
| Linear biases | **Disabled** (`--disable-bias-linear`, no QKV bias) | Makes the parameter arithmetic below exact and follows the bias-free modern GPT/MoE pattern. This is an explicit project choice rather than a scaling-law result. | MCore defaults to linear biases; silently retaining that default would invalidate the frozen count and optimizer assignment. |
| Initialization / dropout | MCore normal initialization with base std **0.006** and its standard scaled output-projection initialization; attention and hidden dropout **0** | Ling's scaling experiments report initialization std 0.006 but do not report dropout. Dropout 0 is an explicit baseline for a one-pass, non-repeated FineWeb subset; this is a 1x MoE-optimal run, not an established overtrained or token-rich regime. | MCore's generic 0.02 initialization is not the scaling-law initialization. Nonzero dropout adds an uncalibrated regularization axis and should be revisited only if pilots show a train/validation gap. |
| Dense FFN | SwiGLU width **2,816** | 2.75x expansion, divisible by 128. | Widths not divisible by 128 complicate Hopper block scaling; MTP is outside scope. |
| MoE placement | First layer dense, remaining **17 MoE** | Ling finds a small dense prefix has minor quality cost and helps early routing. | Every layer MoE is slightly more fragile; every-N placement gives up expert capacity without evidence at this budget. |
| Routed experts / top-k | **64 / 8** | Ling 2.0 combines top-8 with `d_expert=d_model/4` (`G=8`), and Greater Leverage places the useful granularity region around G=8–12, shifting toward roughly G=6–8 under imperfect balancing. With one shared expert, this gives 2,304 active FFN channels and 43.4%/48.8% attention share under the paper-literal/standard FLOP counts. | The matched 32-expert alternative is 32/top-4 with routed width 512 and shared width 256, not 32/top-4 with width 256. It preserves routed capacity and active FFN compute but has G=4. A 256/top-8 design at fixed 1B would force routed widths near 64. |
| Shared experts | **1**, width **256** (`--moe-shared-expert-intermediate-size 256`), ungated (`moe_shared_expert_gate=false`) | Ling finds one shared expert broadly near-optimal. It is 1/9 = 11.1% of active experts here, between the study's low- and high-compute preferences. MCore requires this separate width setting; `moe_ffn_hidden_size` controls only routed experts. | No shared expert loses the common path; multiple shared experts spend too much active compute. A learned shared gate adds an unstudied mechanism and changes the small-op/count assumptions. |
| Per-expert intermediate width / granularity | **256 / G=8**, where `G=2*d_model/d_expert` | Each expert is 4x narrower than `d_model=1024`. Ling 2.0 itself uses expert widths equal to one quarter of model width (for example 512 versus 2,048); its companion study finds an optimum around G=8-12, with imbalance shifting it toward G=6-8. G=8 is also a clean 128 multiple. | A per-expert expansion above model width is not proposed. G=12 gives a non-kernel-friendly width near 171; G=2 makes each expert as wide as the residual stream. |
| Expert-bank activation ratio | `(8+1)/(64+1) = 13.85%` | A deliberate systems/compute-allocation compromise at 1B; whole-model total/active remains 3.67x. | Top-4 with the current 64-by-256 bank gives 7.69%, which is inside Ling's validation range, but makes attention 56.8% of block FLOPs under the paper's convention. The selected top-8 value, 13.85%, is the value outside the 4.7–10.9% range and therefore requires LR calibration. |
| Router | BF16 gating-linear operands with FP32 logits/softmax/top-k/bias/counts; pre-softmax top-8, dropless; `--moe-router-score-function softmax --moe-router-pre-softmax --moe-router-topk-scaling-factor 2.5`; enable `moe_router_padding_for_quantization` only with block-FP8 experts | Ling production recipe and NVIDIA's selective-precision guidance. In pinned MCore the 2.5 factor is otherwise silently unused: it requires pre-softmax routing. MCore's padding makes each expert's dynamic token count legal for quantized grouped GEMM without capacity dropping. | A custom full-FP32 gating GEMM has no demonstrated need. Capacity dropping changes the objective; unpadded variable expert rows are not a valid block-FP8 assumption. |
| Load balance | **Aux-loss-free centered bias**, update rate **0.001**, globally aggregated counts; router z-loss **disabled initially** | Ling supplies the centered bias controller and 0.001 update rate but does not report z-loss. Joint MoE uses z-loss 0.001 only in a different top-1 Switch setup that also uses load-balancing loss 0.01, so 0.001 is only a fallback candidate here. | Aux loss 0.01 is a fresh-run fallback if the bias controller fails the routing gate. Router z-loss 0.001 is a separate fresh-run fallback for stability problems. Neither is enabled by mutating a running job. |
| Sequence length | **2,048** | With top-8, the paper-literal/standard attention shares are 43.4%/48.8%, inside Ling's broad 20–50% band. It is also a conventional FineWeb pretraining context. | At 4K the corresponding shares are 57.5%/60.6%; 1K would permit more sparsity but gives up useful context. |
| Global batch | **425,984 tokens = 208 sequences** | Ling fit gives 434,087 tokens at this compute; rounded to the closest sequence count divisible by both 2- and 4-GPU microbatch layouts. | Joint MoE's 256K and COAT's 4M batches come from different architectures/budgets. |
| Peak AdamW LR | **1.63e-3 prior**, subject to a short `{0.6, 1.0, 1.4, 1.8}e-3` sweep | Direct evaluation of Ling's law inside its compute range, but slightly outside its activation-ratio validation range. | Joint's literal/effective-expert predictions are 0.324e-3/0.559e-3. The disagreement is real; do not select after seeing final-run loss. |
| AdamW | beta1=0.9, beta2=0.95, epsilon=1e-8, clip norm 1.0; decoupled weight decay 0.1 on matrix weights and 0 on RMS/QK normalization scales | Ling and its companion architecture study supply the optimizer values; the no-decay normalization group is the standard explicit parameter grouping. | Framework defaults such as beta2=0.999 are not the MoE recipe studied here. Applying decay to norm scales adds an unrelated optimizer change. |
| Schedule | **Provisional reference: WSD.** Linear warmup on steps 1–173; constant peak LR on steps 174–13,794; exponential decay on steps 13,795–17,242 to 0.1x peak. **WSM remains an optional later user choice.** | The decay boundary is after 80% of total progress, so decay occupies the final 20%; the constant plateau spans about 79% because warmup uses the first 1%. Ling's matched scaling experiments use WSD, while the released Ling 2.0 recipe uses warmup, constant LR, and checkpoint merging. The final-20% WSD boundary is an explicit project choice, not a paper result. | Joint MoE specifies 130M warmup and final-20% linear decay to zero for top-1 Switch models. WSM is not rejected, but it will not be enabled or inferred from approval of this memo. |
| Muon | Same selected LR and weight decay; `--muon-momentum 0.95 --muon-nesterov --muon-num-ns-steps 5 --muon-coefficient-type quintic --muon-scale-mode spectral --muon-extra-scale-factor 0.2 --muon-fp32-matmul-prec medium --muon-scalar-optimizer adam`; leave `--muon-no-split-qkv` unset | [Muon is Scalable](https://arxiv.org/abs/2502.16982) derives the 0.2-spectral scaling to reuse AdamW LR/WD; EO supplies current NS semantics. Every non-default flag is pinned because MCore defaults are momentum 0.9, Nesterov off, and extra scale 1.0. The absent negative flag means QKV splitting remains enabled. | Framework defaults do not implement the selected paper recipe. Muon LR still receives a matched short-run check. |

Here `G=2*d_model/d_expert` is the companion paper's inverse-width
convention: at `d_model=1,024`, G=2/4/8 corresponds to expert widths
1,024/512/256. The LR/batch and model-data fits used
64-routed/top-4/one-shared/G=2. The separate matched granularity sweep reached
G=8 by using 256 routed, top-8, four shared experts, and width 256. Therefore
our 64-routed/top-8/one-shared/G=8 design uses the paper's granularity evidence
but is not its matched G=8 configuration; transferring the fitted LR and token
budget still requires calibration.

The hyperparameter projection is evaluated, not copied from a nearby model:

```text
C_fit = 4.4713e18
eta_opt = 1.1576 * C_fit^-0.1529 = 1.6289e-3
B_opt   = 0.0694 * C_fit^0.3644  = 434,087 tokens
```

Ling's fit has no explicit sparsity term: it was fitted at 7.8% expert-bank
activation and then validated without refitting over 4.7–10.9%. Its conclusion
is that MoE sparsity favors a larger batch and slightly lower LR than a dense
model at the same compute because each expert sees fewer tokens, not that one
should multiply LR by activation ratio. Our 13.85% point is outside that
validation range, so inserting an invented sparsity correction would be less
defensible than the declared LR sweep. The Joint MoE alternative does include
expert count,
`LR=exp(8.39-0.81 ln(N_active,nonembed)-0.25 ln(E))`; it yields 3.24e-4 for
literal `E=64`, or 5.59e-4 for the unmatched effective-bank heuristic
`E=65/9`. That architecture mismatch is why those values are calibration
context rather than the selected prior.

The calibration protocol uses the exact 1B shape, fixed initialization and data
order, BF16 compute, and FP32 states for exactly 587 steps = 250,052,608 tokens
per LR candidate. Use the same boundaries: 1% warmup, constant peak LR through
80% of total progress, and exponential decay over the final 20%; do not expose
candidates to different schedules. Reject a candidate on
non-finite values or routing-gate failure; otherwise rank by the fixed
validation loss at the end. If the best two differ by less than 0.01 nats,
advance only those two to the 2,348-step = 1,000,210,432-token gate before
freezing LR. Apply the same declared grid independently to AdamW and to the
0.2-scaled Muon recipe. These are calibration runs and still require approval
before launch.

### Optional WSM schedule

Ling 2.0 does use WSM for its released training recipe: after a 2,000-step
linear warmup it retains a constant learning rate and obtains the final
annealed model by averaging selected late checkpoints. This does not change
the fact that Ling's learning-rate, batch-size, and scaling-law experiments
were run with WSD. Their fitted hyperparameters therefore do not establish
that WSM is preferable for this model.

The accepted [WSM paper](https://openreview.net/forum?id=HhThhjKyfw) reports a
higher average downstream score for WSM than WSD on a
16.3B-total/1.43B-active MoE, including a smaller supporting Muon experiment.
Its strongest result starts from a model already trained for 10.2T tokens and
uses a specialized high-quality continuation mixture. In one reported MoE
comparison, WSM also has worse language-modeling loss, 0.697 versus 0.675 for
WSD, despite better downstream scores and routing balance. NVIDIA's
[Nemotron 3 Super report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
finds that WSM helps at one shorter merge horizon but not at its 1T and 1.5T
horizons. The literature therefore supports WSM as a credible candidate, not
as a general replacement for WSD, especially for a uniformly shuffled
FineWeb run whose primary metric is validation loss/bpb.

Checkpoint merging exactly reweights parameter increments along the realized
constant-LR trajectory, but it is not equivalent to rerunning WSD: later
gradients, weight decay, Adam moments, Muon momentum, and router-controller
state are path dependent. It can also smooth optimizer-state quantization
error, which would confound the central FP8-state comparison. Schedule choice
must consequently be frozen using FP32 optimizer states before comparing
state formats.

WSD remains the reference unless the user later selects WSM. If a matched
schedule gate is requested, use a common BF16-compute/FP32-state trajectory
through the 80% boundary and fork only the final 20%: one branch performs the
declared WSD decay and the other retains the peak LR and saves predeclared
checkpoints for one fixed offline merge. Hold LR, data, initialization, and
token budget fixed; report the WSD endpoint, raw constant-LR endpoint, and
merged WSM model. Do not copy Ling's validation-selected top-32 heuristic.
Merge only FP32 master weights, retain the unmerged checkpoint and optimizer
state for resumption, and regenerate rather than average TransformerEngine
FP8 scaling histories. Optimizer states and the aux-loss-free router bias are
not mergeable under the WSM derivation. This optional gate requires separate
approval and should be run independently for AdamW and Muon if both schedules
remain candidates.

Aux-loss-free routing is the target, not a claim that it is universally
superior. This is a pilot-selection gate, not an adaptive in-run switch. For
each MoE layer, globally reduce expert token counts on every step, sum them over
the fixed first 100 steps after warmup, and define
`CV = std_e(c_e) / mean_e(c_e)`. Every layer must satisfy
`min_e(c_e) >= 0.1*mean_e(c_e)` and `CV < 0.20`. If an aux-free pilot fails,
stop and log it, then launch a fresh run from the same initial conditions with
the bias controller disabled and load-balancing auxiliary-loss coefficient
0.01; do not mutate a running job. Continue logging the same routing metrics
throughout every run; an unexpected final-run failure stops the run for
investigation rather than changing its objective.

The 100-step window, 0.1 minimum-load threshold, CV threshold, and stop/relaunch
policy are project engineering gates, not paper prescriptions. The original
Loss-Free paper reports MaxVio and smooths its training curve over 100
neighboring steps only for visibility. Router z-loss is separate from load
balancing: it regularizes router-logit magnitude and contributes gradients.
Thus “aux-loss-free” here means no auxiliary load-balancing gradient, not no
router auxiliary term. It is disabled initially. If a pilot shows router
stability problems, stop and log that run, then relaunch from the same initial
conditions with z-loss 0.001; never add it mid-run. Keep the z-loss setting
fixed across aux-free and auxiliary-load-balancing arms in any matched
comparison.

“Centered” is a real implementation delta from pinned MCore, not a flag. For
globally reduced expert loads `l`, use Ling's controller
`b += 0.001*(sign(mean(l)-l) - mean(sign(mean(l)-l)))`. Stock MCore omits the
second, mean-centering term. Stage 4 therefore adds only this router update and
tests exact agreement with the formula, zero-sum bias increments, global-load
aggregation, and checkpoint/resume; using stock uncentered bias would be a
declared architecture change.

### Parameter arithmetic

Counts include separate GPT-2 embedding and LM-head matrices and omit biases.
QKNorm adds only 4,608 parameters.

| Component | Total parameters | Active parameters/token |
|---|---:|---:|
| Input embedding + output head | 102,926,336 | 102,926,336 |
| 18 GQA attention blocks | 47,185,920 | 47,185,920 |
| One dense SwiGLU | 8,650,752 | 8,650,752 |
| 17 x 65 routed/shared expert SwiGLUs | 869,007,360 | 120,324,096 (nine per token) |
| 17 routers | 1,114,112 | 1,114,112 |
| RMSNorm/final norm/QKNorm | 42,496 | 42,496 |
| **Total** | **1,028,926,976** | **280,243,712** |

The whole-model active fraction is 27.24%. The expert-bank activation fraction
is 13.85%. Non-embedding active parameters are 177,317,376. All three should be
logged; they answer different questions.

`d_expert=256` is the width of each independent SwiGLU, not their concatenated
width. Top-8 plus one shared expert has an aggregate active intermediate width
of `9*256=2,304`, still below the dense layer's 2,816. This mirrors the Ling
pattern of narrow individual experts whose active widths collectively approach
a dense FFN; it is not an expert with `moe_hidden_dim > hidden_dim`. In
Megatron terms the intended `moe_ffn_hidden_size` is therefore **256**, not
2,304. The latter is only a per-token sum over separately evaluated branches.

```text
--hidden-size 1024
--ffn-hidden-size 2816
--moe-ffn-hidden-size 256
--moe-shared-expert-intermediate-size 256
```

### Initial parallel layout

- 2 GPUs: DP=2, EP=TP=PP=CP=ETP=1; for a two-sequence microbatch per GPU,
  gradient accumulation is 52.
- 4 GPUs: DP=4 with the same other degrees; gradient accumulation is 26.
- 1 GPU is a smoke path only.
- Recompute is off initially and enabled only if measured peak VRAM requires it.
- The first controlled topology comparison is a replicated routed-expert bank
  `(DP=world, EP=1, EDP=world)` versus sharded experts
  `(DP=world, EP=world, EDP=1)`, with identical global batch and precision.
  Dense parameters remain data-parallel in both layouts; EP changes the expert
  data-parallel group for routed experts, not the global DP size. The shared
  expert remains dense-like/data-parallel and is not described by EDP.

This follows NVIDIA's explicit rule to minimize model parallelism and maximize
DP when the model fits. The literature cannot determine which layout wins for
this unusually small, fine-grained model; only node207 measurements can.
Plain all-to-all precedes DeepEP. TP, PP, CP, and ETP remain 1.

## 4. Compute and token budget

### FLOPs/token

Use the Ling companion paper's explicit forward convention to evaluate its
scaling law. For batch `B`, sequence `s`, hidden size `d`, query heads `n_h`,
and KV heads `n_kv`, its Appendix C prints:

```text
C_attn,paper / (B s) = d^2 (2 + 2 n_kv/n_h) + 4 s d
C_dense_ffn / (B s) = 6 d d_ff
C_moe_ffn / (B s) = 6 d (k + n_shared) d_expert
C_train ~= 3 C_forward
```

The projection term is internally inconsistent with the same appendix's FFN
count: it counts one operation per projection multiply-add, whereas `6*d*d_ff`
counts two. Standard hardware accounting therefore doubles only that
projection term:

```text
C_attn,standard / (B s) = 2 d^2 (2 + 2 n_kv/n_h) + 4 s d
```

At `s=2048`, the proposed model gives:

| Forward component | FLOPs/token |
|---|---:|
| GQA projections, 18 layers | 94,371,840 |
| QK/PV sequence term, 18 layers | 150,994,944 |
| Dense FFN | 17,301,504 |
| Active routed + shared FFNs | 240,648,192 |
| Router projections | 2,228,224 |
| Router softmax/top-k, routed weighting, and shared merge (explicit small-op estimate) | about 306,816 |
| LM-head projection | 102,926,336 |
| **Forward, standard hardware convention** | **608,777,856** |
| **Training, forward + backward (`3x`)** | **1,826,333,568** |

The small-op line uses `10*E` operations for router normalization/selection and
`2*k*d` for the probability-weighted routed combine, plus `d` additions for
MCore's default ungated shared-expert merge, per MoE layer. It is an analytic
estimate, not a kernel instruction count. Dispatch and memory movement are not
hidden inside the FLOP count.

Ling calls `M` “non-embedding” FLOPs, but its Equation 16 explicitly includes
the final vocabulary projection. The count above follows that equation:
input-embedding lookup cost is omitted, while the LM-head GEMM is included.

Using the paper's printed projection term instead gives **561,591,936**
forward FLOPs/token. Because the prose defines `M` as FLOPs and the FFN and
QK/PV terms use two operations per multiply-add, the memo treats the missing
projection factor as a typesetting error and uses the standard **608,777,856**
forward count both for the primary law projection and for hardware planning.
A causal kernel may execute a different physical instruction count, but every
run will use the fixed **1.826333568B training FLOPs/token** analytic numerator.
`6*N_active = 1.681B` also misses the exact attention/embedding asymmetry and
is 7.9% below the explicit training count (equivalently, the explicit count is
8.6% higher).

Under the paper's own attention convention, attention is 43.45% of
attention-plus-FFN FLOPs; under standard FLOPs it is 48.75%. Both are within
the paper's broad 20–50% allocation band. This check is why the initial
top-4/4K draft was rejected: its corresponding shares were about 70%/72%.

### MoE-optimal data

The Ling MoE fit is:

```text
M_opt(C) = 0.1915 C^0.5095
D_opt(C) = 5.2232 C^0.4905
C = M D
```

Here `M` is the corrected standard forward FLOPs/token convention above.
Solving the reported `D_opt` fit with `C=M*D`:

```text
D(M) = (5.2232 * M^0.4905)^(1/0.5095)
     = 7.3447e9 tokens
```

The corresponding fit-compute is `M*D = 4.4713e18`. This lies inside the
paper's fitted `3e17`–`3e20` range. Actual training accounting is:

```text
F_train = 3 * M * D = 1.3414e19 FLOPs.
```

At the rounded global batch, run
**17,242 steps = 7,344,816,128 tokens**.

As an independent check, the Joint MoE law at 280.2M active parameters predicts
6.89B tokens for 8 coarse top-1 experts and 10.48B for 16. The proposed
expert-bank capacity multiplier is `65/9 = 7.22`, closest to its 8-expert
curve; the two unrelated projections are within 6.7%. This is a useful scale
check but not a matched-law validation. The chosen budget is neither the dense
`20*N_total = 20.58B` rule nor `20*N_active = 5.60B`.

There is an unavoidable convention sensitivity. Taking the erroneous printed
projection term literally gives 6.796B tokens. The memo selects 7.345B because
it is the physically correct FLOP count, but records **6.80–7.35B** as the
source-induced uncertainty interval. More importantly, the law was fitted at
top-4, `G=2`, 4K and a lower activation ratio; `D`, LR, and batch are
projections for the proposed model, not exact optima.

### Wall-clock planning envelope

NVIDIA lists sparse FP8 peak with an asterisk. The dense H100 SXM bound used
here is half of that value: **1.979 PFLOP/s/GPU**. Hopper block scaling is
software-emulated and the model has many small expert GEMMs, so 5%, 10%, and
15% MFU are shown as sensitivity points, not forecasts.

| GPUs | 5% MFU | 10% MFU | 15% MFU |
|---:|---:|---:|---:|
| 2 x H100 SXM | 18.8 h | 9.4 h | 6.3 h |
| 4 x H100 SXM | 9.4 h | 4.7 h | 3.1 h |

Adding a planning allowance of 20% for evaluation, checkpointing, and input
stalls gives 7.5-22.6 hours on two GPUs and 3.8-11.3 hours on four. Node207's
H100 form factor, clocks, topology, and achieved MFU are unknown. These numbers
are estimates and must never be reported as measured throughput.

The full target is therefore reasonable rather than obviously infeasible. A
smaller model is not proposed as a substitute. The validation ladder still
uses short 100M- and 1B-token gates; those establish numerical behavior but do
not establish final loss, expert specialization, or end-to-end MFU. If clean
node measurements fall below 5% MFU, a 300M-total research run can test
optimizer stability, but it gives up the final model's routing occupancy,
expert-GEMM shape, and quality claim.

## 5. FP8 optimizer-state design

### Failure mechanisms

Persistent state is quantized and dequantized at every step, so quantization
noise becomes part of the recurrence rather than a one-off model perturbation.

- AdamW's first moment loses small signed momentum and direction. Its beta1
  recurrence damps old error, but deterministic rounding can introduce a
  persistent directional bias.
- The second moment is non-negative, often has a very narrow within-block
  dynamic range, and enters an inverse square root. Underflow or relative error
  near zero can turn into a large update error. The beta2 recurrence makes the
  corruption long-lived.
- Per-tensor scaling lets a small number of outliers consume most E4M3 codes.
  Per-element scaling defeats the storage objective. A 128-element group is the
  supported middle point with direct COAT evidence.
- Muon's polar/NS map discards singular-value magnitude but preserves singular
  directions. Momentum quantization that rotates weak singular directions can
  be amplified by orthogonalization; elementwise reconstruction error alone is
  not an adequate test.

### AdamW state: primary candidate

Use the [COAT](https://arxiv.org/abs/2410.19313) recipe as the first target:

1. Persistent `m` and `v`: E4M3, consecutive 128-element groups.
2. For each group, compute `a_max=max(abs(x))` and the minimum over nonzero
   magnitudes only, with
   `a_min=max(min_nonzero, min(a_max, 1e-30))`. The second term supplies a
   numerical floor without making `a_min>a_max` for uniformly subnormal
   groups. For a genuine range, set
   `k_raw=log(100352)/log(a_max/a_min)` and use
   `k=clamp(floor(16*k_raw)/16, 1/16, 16)`. Allowing `k<1` is essential:
   groups wider than the target range must be contracted, not left unchanged.
   Center by
   `c=sqrt(a_min)*sqrt(a_max)` and apply
   `f(x)=sign(x)*abs(x/c)^k`. Set `scale=(a_max/c)^k/448`, and store the
   round-to-nearest E4M3 payload of `f(x)/scale` plus FP32 `scale`, `k`, and
   `c`. The 1/16
   discretization and target `448^2/2=100352` reproduce the released COAT
   kernel. The paper's idealized derivation instead states the full E4M3 ratio
   229376; that discrepancy is recorded. The finite `1/16` and `16` bounds are
   project robustness rules and must be tested rather than attributed to the
   paper; any non-finite input/state is a hard failure.
3. Dequantize into FP32 as `y=float(q)*scale`, then invert with
   `x_hat=c*sign(y)*abs(y)^(1/k)`.
4. Compute moment recurrences, bias correction, `sqrt(v)+epsilon`, weight
   decay, and the FP32 master-weight update in FP32.
5. Recompute group metadata and requantize the updated states.
6. All-zero group: zero payload with `scale=k=c=1`; minimum reductions ignore
   zero entries. If `a_max/a_min` is numerically one, including a group with
   only one nonzero magnitude, bypass DRE and use ordinary max-abs E4M3 with
   `k=c=1`; this avoids COAT's released kernel's division by `log(1)`. Tail
   groups are masked, not padded into statistics.

COAT's paper says BF16 scaling factors. Its repository allocates the three
metadata arrays per state with the parameter dtype, while the released fused
CUDA kernel consumes `float*`, so that path is effectively tied to FP32
parameters. This project deliberately uses FP32 metadata for the first
implementation and accounts for it explicitly; this is a project/kernel
constraint, not a universal property of COAT:

```text
per Adam state = 1 + 3*4/128 = 1.09375 bytes/parameter
two states      = 2.1875 bytes/parameter
```

For 1.029B parameters, FP32 moments occupy 8.231 GB by tensor-size accounting;
the proposed payload plus metadata occupies 2.251 GB. The 5.981 GB difference
is an accounting prediction, not a peak-VRAM measurement.

[Scaling FP8 Training to Trillion-Token
LLMs](https://arxiv.org/abs/2409.12517) instead found that ordinary E4M3 first
moment plus E5M2 second moment was the only standard-format combination that
converged in its experiment. It is the required ablation because it was
validated to two trillion tokens, but on Gaudi2 with delayed scaling and
without COAT's per-128 DRE. It is not the primary Hopper implementation.
Earlier [FP8-LM](https://arxiv.org/abs/2310.18313) retained the second moment in
16-bit; that is a useful safety fallback, not the core objective.
[FlashOptim](https://arxiv.org/abs/2602.23349) is newer and uses nonlinear
companding, but its released optimizer payload is signed INT8 with FP16 scales,
not FP8. It is outside the requested format rather than a replacement for
COAT.

Decisions on the three requested mechanisms:

- **Dynamic range expansion:** yes for Adam `m` and `v`; it is the strongest
  direct FP8-state evidence for both E4M3 states.
- **Stochastic rounding:** not in the primary recipe. COAT uses
  round-to-nearest, and changing the quantizer while testing DRE would confound
  the first result. If mean signed quantization error or long-horizon step
  drift fails its gate, the first ablation is counter-based, float-domain
  stochastic rounding keyed per element by the run seed, global step, a stable
  hash of the parameter's fully qualified name, state kind (`m`, `v`, or Muon
  momentum), and global flattened element index. These inputs and the RNG
  counter are checkpointed; Python object IDs and block-shared random draws
  are forbidden because they break exact resume or correlate errors.
- **Error feedback:** no initially. A BF16/FP32 residual costs 2/4 extra bytes
  per parameter and can erase the state-memory result. It is a last escalation
  only if DRE plus stochastic rounding fails, with its storage included in all
  accounting and logs.

### Muon state: separate recipe

EO currently initializes `momentum_buffer = zeros_like(master_parameter)`;
under MCore's mixed-precision wrapper this is an FP32 buffer. EO's NS entry
rejects non-FP32 input. It transposes tall matrices so the smaller dimension is
orthogonalized, divides each full matrix by its FP32 Frobenius norm, casts the
working matrix to BF16 when `fp32_matmul_prec=medium`, performs the tensor-core
iterations, and casts the BF16 result back to FP32.

The first FP8 Muon implementation changes storage only:

1. Apply Muon only to hidden 2D attention, dense-FFN, and individual expert
   logical matrices. Use MCore's QKV split and independently orthogonalize the
   gate and up halves of every fused SwiGLU FC1; the down projection is the
   third matrix. Keep embeddings, output head, norms, biases, and router weight
   on AdamW; router decision tensors remain FP32 as specified below. The
   alternative of applying NS once to `[gate; up]` is
   rejected because optimizer semantics would depend on parameter packing.
2. Store the persistent pre-orthogonalization momentum in E4M3 with one FP32
   max-abs scale per contiguous row-aligned group of 128 elements. Per-matrix
   scaling is a required diagnostic baseline, not a second training recipe; it
   advances only if the blockwise choice fails the directional-error gate on
   real pilot states.
3. Dequantize the previous buffer to FP32 and reproduce EO's recurrence:
   `m_t=lerp(m_{t-1}, g_t, 1-beta)` and, with the selected Nesterov option,
   `u_t=lerp(g_t, m_t, beta)`. Both are computed in FP32.
4. Requantize `m_t` for the next step. Never store either the Nesterov blend
   `u_t` or the orthogonalized update as persistent momentum.
5. Pass each FP32 logical matrix of `u_t` into EO's unchanged transpose and
   whole-logical-matrix Frobenius normalization. For fused FC1, split the
   Nesterov blend first and concatenate the two post-NS updates afterward.
   Run five quintic NS iterations with the BF16 working tensor and cast the
   result back to FP32. Do not alter the polynomial, normalization, or
   epilogue while testing state storage. FP8 NS is a separate future ablation,
   not part of the state experiment.
6. Compute the `0.2*sqrt(max(m,n))` spectral scale and final decoupled weight
   decay/master-weight update in FP32.

Muon DRE is rejected initially: its sign-preserving power transform is
validated for Adam state error, not for preserving Muon's singular directions.
Round-to-nearest and no residual are the initial controlled recipe; stochastic
rounding is the same predeclared first escalation as for Adam.

[Effective Quantization of Muon Optimizer
States](https://arxiv.org/abs/2509.23106) provides strong evidence that
blockwise 8-bit momentum storage can match full-precision Muon up to 2.7B
parameters, but its payload is signed integer linear/dynamic quantization, not
FP8. [MuonQ](https://arxiv.org/abs/2605.11396) studies 4-bit companding and
direction preservation, again not FP8. They inform the directional diagnostics
but are not presented as FP8 recipes. An experimental repository,
`dtensor-muon`, advertises FP8 state through torchao but has no matched
pretraining evidence for this setup. No reviewed source or NVIDIA EO release
found in the 2025-2026 search implements validated FP8 Muon state.
[Mellum2](https://arxiv.org/abs/2605.31268) trains with Muon and hybrid FP8
model compute, but does not provide an FP8 momentum-state recipe. It supports
separating compute precision from state precision, not quantizing both at
once.

[Tang et al. (ICLR 2026)](https://openreview.net/forum?id=wwP1SCACee) prove
the full-precision convergence rate for an idealized exact-SVD Muon only when
the relative quantization errors for weights, gradients, and momentum shrink
as `O(T^-1/2)` under their prescribed schedules; fixed precision reaches only
a stationary-point neighborhood. Their experiments do numerically quantize
Adam moments, Muon momentum, and the hybrid auxiliary-Adam states by stochastic
mantissa truncation, but retain the FP32/FP64 exponent and expand back to the
high-precision dtype. CIFAR/nanoGPT experiments use finite NS, but not EO's
specific FP32 recurrence plus BF16-working-tensor path. They do not store E4M3,
use block scales, or report state bytes, memory, or speed. The work supports
treating Adam's second moment as especially sensitive and suggests Muon may
tolerate relative error; it is theory and mantissa-emulation evidence, not a
validated FP8-state recipe.

For the proposed parameter assignment, 924.84M parameters use Muon and 104.08M
use AdamW. Tensor-size accounting predicts:

| Hybrid persistent state | FP32-state reference | Proposed FP8 payload + FP32 metadata |
|---|---:|---:|
| Muon momentum | 3.699 GB | 0.954 GB |
| Adam moments on exclusions | 0.833 GB | 0.228 GB |
| **Total** | **4.532 GB** | **1.181 GB** |

Again, these are byte counts, not peak-VRAM results. Temporary FP32
dequantization, NS workspaces, allocator behavior, and sharding must be
measured.

### Checkpoint and distributed semantics

FP8 payloads, all metadata, step counters, quantizer mode, and stochastic
rounding seed/counters (if enabled), stable parameter-name mapping, and state
kind are first-class optimizer state. An interrupted and resumed run must be
bitwise identical to an uninterrupted run for round-to-nearest and must use
the identical counter stream for stochastic rounding. A command-line
`--resume` is not evidence of support.

Start without the distributed optimizer because the model fits. MCore's stock
precision-aware FusedAdam is retained as an additional reference, but it
requires `optimizer=adam` and the distributed optimizer and cannot be enabled
for the mixed Muon/Adam path. State sharding is introduced only after local
state serialization passes. If layer-wise distributed Muon is later enabled,
whole matrices and all their scale blocks must remain on the owning rank before
NS; elementwise ZeRO sharding cannot change the orthogonalization semantics.

NVIDIA also supports native FP8 primary model weights, generated directly from
FP32 master shards after a distributed-optimizer step. That removes the BF16
model-weight copy, but the stock path is not available to the mixed
Muon/AdamW optimizer. The first experiment therefore follows Ling's
BF16-source-to-FP8-GEMM flow. Native FP8 primary weights are deferred until
state correctness is established; enabling them in the first state experiment
would conflate two storage changes.

## 6. Precision map for everything else

The target is selective precision, not the slogan “everything FP8.”

| Tensor/operation | Target precision on H100 | Reason |
|---|---|---|
| Linear/expert forward operands | E4M3; activations 1x128, weights 128x128 | TE Hopper block recipe and Ling/NVIDIA production guidance. |
| Linear backward operands | E4M3; activation gradients 1x128 | Same block recipe. |
| GEMM accumulation/output | FP32 accumulation, BF16 output/residual | Protect accumulated sums; this is mixed precision even when the GEMM operands are FP8. |
| Expert grouped GEMM | Same E4M3 block recipe; routing-map padding enabled | Model dimensions are 128 multiples, but dropless routing still gives dynamic rows per expert. MCore's `moe_router_padding_for_quantization` pads those rows to the kernel-supported multiple; padded rows must be removed before combine and excluded from loss/load counts. |
| QKV/output projections | E4M3 block GEMMs | Bulk linear compute. |
| SDPA, attention scores/softmax | BF16 operands, FP32 softmax/reductions | NVIDIA's current MoE recipe normally keeps SDPA BF16; TE 2.16 removed the legacy delayed-scaling FP8 attention path. |
| RMSNorm/QKNorm | BF16 I/O, FP32 reductions/division | Reduction sensitivity. |
| Residual stream | BF16 | Accumulates across layers. |
| Router gating projection / logits/top-k/bias/counts | BF16 input and weight operands; FP32 output logits, score function, top-k, expert bias, and counters | This is what pinned MCore's `moe_router_dtype=fp32` implements. A full-FP32 gating GEMM would require a custom path; FP32 decision tensors protect the discontinuous part. |
| Embedding and LM head | BF16 model copy/GEMM | NVIDIA keeps them high precision; they are not the dominant expert compute. |
| Logits and cross entropy | FP32 | Protect loss and evaluation. |
| Master weights and final optimizer update | FP32 | Accumulate small updates; both FP8-LM and NVIDIA identify this as sensitive. |
| Main parameter-gradient accumulation | FP32 | Reference-quality optimizer signal. TE's internal FP8 gradient operands do not imply FP8 persistent main gradients. |
| DP gradient reduce/all-reduce | FP32 initially | NVIDIA's current selective recipe keeps main gradients high precision. FP8-LM's globally scaled FP8 all-reduce is an older separate system and would confound the state experiment. |
| Distributed-optimizer parameter all-gather | FP8 blockwise only if that optimizer is enabled | TE officially supports blockwise FP8 all-gather. Otherwise not applicable. |
| EP dispatch payload | BF16 reference, then MCore HybridEP E4M3 `fp8_dispatch` target | Enable only in the EP arm after matched convergence; no EP traffic exists in the initial EP=1 arm. This is the FP8 EP direction exposed by pinned MCore. |
| EP combine payload/accumulation | BF16 initially | Pinned MCore has no corresponding FP8-combine option. Record the selected unpermute/combine kernel and verify its accumulation behavior rather than promising FP32 accumulation that the dispatch flag does not provide. |
| Adam persistent state | E4M3+DRE per 128, FP32 metadata | Section 5. |
| Muon persistent momentum | E4M3 per 128, FP32 scale | Section 5. |
| Muon recurrence/Frobenius normalization/NS boundary/final update | FP32; EO's NS working tensor is BF16 and its result is cast back to FP32 | Preserve directions and retain current EO semantics. |
| Checkpoint | FP32 masters, exact FP8 state bytes/metadata; BF16 model export | Exact resume and portable model artifacts. |

Keeping FP32 masters is deliberate. The 2026
[ECO](https://arxiv.org/abs/2601.22101) paper demonstrates error-compensated
updates without master weights, including an FP8 2.1B sparse-MoE experiment.
It injects weight-quantization error into optimizer momentum, however, so using
it here would change the very recurrence whose FP8 storage is under study.
ECO is a credible later experiment, not part of the state-isolation baseline.

TE exposes per-tensor delayed and current scaling, blockwise FP8, MXFP8, and
NVFP4, but they are not interchangeable:

- Per-tensor **current** scaling is the first migration smoke. Delayed scaling
  is rejected because current values are more precise and NVIDIA no longer
  recommends delayed scaling for this path.
- [`Float8BlockScaling`](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html)
  is the Hopper target. TE 2.16 uses 1x128 activation/gradient and 128x128
  weight blocks with FP32 scales and supports quantized all-gather.
- On Hopper, set `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1` so those are
  unrestricted FP32 scales rather than TE's default power-of-two-constrained
  values. This is supported on SM90 and is the closer match to Ling's reported
  FP32-scale recipe; the exact environment setting is part of every run
  manifest.
- DeepSeek-V3-style fine-grained block scaling is available on Hopper through
  this TE recipe, but is software-emulated rather than a new Hopper tensor-core
  format.
- MXFP8 and NVFP4 are Blackwell recipes. They must not be imported into the
  H100 plan. TE's [2.16 release notes](https://docs.nvidia.com/deeplearning/transformer-engine/release-notes/index.html)
  explicitly add Hopper block-FP8 grouped GEMM support.

## 7. Data and evaluation

The official [FineWeb dataset
card](https://huggingface.co/datasets/HuggingFaceFW/fineweb) describes
`sample-100BT` as about 100B GPT-2 tokens and 277.4GB. It also states that the
`token_count` field is generated with the GPT-2 tokenizer. Use GPT-2 BPE
(50,257 tokens) to preserve that accounting, avoid a tokenizer-training
subproject, and make the Joint MoE comparison less ambiguous. Because tokenizer
quality affects token loss, **bits per UTF-8 byte (bpb) is the primary
cross-tokenizer metric**, with token validation loss reported alongside it.

Data preparation, after approval:

1. Pin the Hugging Face dataset revision and record every Parquet file's size
   and SHA-256. Check available node207 disk before downloading 277.4GB.
2. Pin and hash the GPT-2 tokenizer files. Add one EOS separator between
   documents; do not otherwise normalize text.
3. Split by a stable hash of FineWeb document ID before tokenization/packing.
   Reserve approximately 100M tokens (0.1%) for validation.
4. Deterministically choose enough remaining documents for 7.345B training
   tokens, then tokenize once into Megatron indexed shards. Record the document
   manifest, order, token count, byte count, tokenizer hash, packing rule, and
   shard hashes.
5. Never stream the network dataset in a measured run. Periodic evaluation uses
   a fixed 8M-token validation slice; final evaluation uses the full held-out
   set.
6. Report token NLL and
   `bpb = sum(non-padding NLL)/(ln(2)*sum(source UTF-8 bytes))`, including the
   inserted document-separator loss. The definition and byte count remain fixed
   for every comparison.

The sample is much larger than the selected training budget. Selecting a
deterministic subset is preferable to repeating data or silently training
beyond the derived optimum.

## 8. Validation and measurement contract

### Unit and optimizer tests

The following are pass/fail gates, not optional diagnostics:

1. **Quantize/dequantize:** zeros, constants, signed log-uniform values,
   non-negative second moments, outliers, subnormal-scale values, and tensor
   lengths not divisible by 128. Payload and metadata must match a simple FP32
   reference quantizer. DRE inversion must remain finite and `v >= 0`;
   constant and single-nonzero groups must exercise the explicit DRE bypass.
2. **Representative state error:** on synthetic distributions and saved
   unquantized pilot-state snapshots, normalized RMSE must be at most 0.03 for
   both Adam states and Muon momentum. Report mean signed error separately.
3. **AdamW one-step agreement:** against the same FP32-master implementation,
   initialize representative nonzero prior `m` and `v` and first compare the
   bias-corrected adaptive update before learning-rate multiplication and
   weight decay. Then compare the final parameter delta: relative L2 error at
   most 0.02 and cosine similarity at least 0.999, with finite output. Test
   first-step serialization separately, plus warmup LR=0, bias correction,
   weight decay, and late-step states; a zero-state/weight-decay-dominated test
   is not sufficient.
4. **Muon one-step agreement:** reproduce EO Nesterov and scaling exactly;
   relative Frobenius error of the post-NS update at most 0.02 and cosine at
   least 0.999 on well-conditioned representative matrices. Ill-conditioned
   matrices additionally report singular-vector/subspace error rather than
   hiding failures in elementwise MSE.
5. **Long-horizon recurrence:** at least 1,000 deterministic synthetic steps,
   logging state NRMSE, update cosine, signed bias, and finite/non-negative
   invariants. This catches error accumulation that one-step tests miss.
6. **Distributed and routing identity:** a two-rank test with identical global gradients
   must produce the same model/state as the single-rank reference within FP32
   `rtol=1e-6, atol=1e-7`; quantized state bytes are exact when rounding inputs
   are identical. Separately force non-multiple expert token counts, then
   verify padding/unpadding preserves token order, router weights, combined
   outputs, loss, and unpadded load statistics.
7. **Resume:** uninterrupted and save/reload paths are bitwise identical for
   round-to-nearest, including FP8 payload, metadata, step, router bias, and
   sampler position. Stochastic rounding, if added, includes its counter state.

The 0.02/0.999 step tolerances are engineering gates selected before seeing
training results, not claims from a paper.

### Training parity gates

#### Training-budget ladder

An LR candidate means one LR value for one optimizer. A parity arm means
one fixed compute/state-precision configuration. Budgets are charged per
candidate or arm, not once for the whole sweep.

| Budget | Exact size | Purpose |
|---|---:|---|
| Smoke | 235 steps = 100,106,240 tokens per arm | Numerical stability, routing, logging, and checkpoint/resume; coarse loss-parity screen. |
| LR screen | 587 steps = 250,052,608 tokens per LR candidate | Rank the four declared LRs on the exact 1B model with BF16 compute and FP32 states. |
| Calibration/parity gate | 2,348 steps = 1,000,210,432 tokens per surviving candidate or precision arm | Break an LR tie and establish the main state-quantization loss-parity result. |
| Optional WSD/WSM gate | 2,818 steps = 1,200,422,912 tokens total per optimizer | Only if requested: one shared 1,878-step prefix and two 470-step tails produce the WSD, raw constant-LR, and merged WSM endpoints at the same 1B-token horizon. |
| Final run | 17,242 steps = 7,344,816,128 tokens per approved arm | Run the provisional 1x MoE-optimal budget only after all gates and separate approval. |

Four 250M-token LR candidates cost 1,000,210,432 tokens per optimizer,
or 2,000,420,864 tokens for AdamW and Muon together. If the best two LRs for an
optimizer are tied and both are extended from 250M to 1B, that adds
1,500,315,648 tokens for that optimizer.

The optional schedule gate costs 200,212,480 tokens more than one ordinary
1B-token gate per optimizer; checkpoint storage and I/O are measured
separately. Every LR or precision comparison fixes initialization, document
order, global batch, schedule, token count, model, router, and evaluation set.
The WSD/WSM gate fixes the same variables except for the schedule and its
offline merge.

- 235-step = 100,106,240-token smoke: no NaN/Inf, no expert collapse, checkpoint/resume passes,
  and quantized versus reference final validation loss differs by at most 0.05
  nats with relative bpb gap at most 1.0%.
- 2,348-step = 1,000,210,432-token gate: final validation-loss gap no more than 0.02 nats and bpb gap
  no more than 0.5% relative for the state-quantized arm versus its identical
  FP32-state arm. Routing gates above must also pass.
- A failed or skipped arm is reported with its log and is not advanced.
- The full 7.345B run starts only after explicit approval.

### Clean-GPU performance protocol

Before and during every reported benchmark:

- Record GPU UUIDs/model/form factor, driver, clocks, power limit, topology,
  MIG state, container digest, source SHAs, and full config.
- Check `nvidia-smi` utilization, memory, and compute processes. Reserve exact
  GPU IDs. Any foreign process or unexplained memory use contaminates the run.
- Warm up at least 20 optimizer steps; measure at least 100 steady steps.
  Monitor occupancy/processes during the interval. Discard thermal throttling,
  clock changes, foreign-process overlap, or input stalls.
- Log raw per-step time, tokens/s, analytic MFU, allocated and reserved peak
  VRAM, optimizer-step time, routing loads, and communication time. Keep the
  raw log and machine-readable summary.
- Define FP8 MFU as
  `1.826333568e9 * measured_tokens_per_second /
  (num_gpus * 1.979e15)`, using the dense peak and the fixed FLOP convention.
  The BF16 denominator is 0.9895 PFLOP/s/GPU.
- A statement that one arm is faster or uses less peak memory is permitted
  only after a matched clean measurement supplies the number. Paper results and
  the arithmetic in this memo are labeled as external results or estimates.

## 9. Staged implementation plan

No stage begins until this memo is approved.

| Stage | Work | Validation and pass criterion |
|---|---|---|
| 0. Decisions | Approve architecture, tokenizer, balancing, comparison scope, and state recipes. | User approval; unresolved material forks recorded. |
| 1. Node preflight and lock | Read-only inspect node207 driver, H100 form factor/topology, container runtime, disk, CPU/RAM, GPU occupancy, and network. Resolve the absolute server path corresponding to `xandi281/H-MoE-Part`, without touching sibling projects; resolve the exact container digest and write a version manifest. | Two clean H100s visible; driver can run the pinned image; enough disk for source data, indexed subset, and checkpoints. Any driver/host change requires approval. |
| 2. Environment smoke | Build the narrow MCore tag environment with its exact TE/EO pins. Run upstream import, TE block-FP8 GEMM, grouped-GEMM, NCCL two-rank, and EO Muon tests. | All tests pass on H100; installed versions and build flags equal the manifest. |
| 3. FineWeb preparation | Download the pinned `sample-100BT`, hash it, make the deterministic split/subset, tokenize, and build indexed shards. This is a large action and needs approval. | Repeated preparation yields identical manifests/shard hashes; decoded samples and exact token/byte counts agree. |
| 4. BF16 model reference | Implement only the proposed GPT/MoE configuration, Ling-centered router-bias update, and logging. No FP8. Unit-test parameter/active counts and FLOPs; run one-GPU tiny and exact-model smokes. | Counts equal this memo; forward/backward finite; deterministic loss; centered-bias formula, global counts, zero-sum increments, resume, router, and packing tests pass. |
| 5. Distributed BF16 AdamW | Run two-rank then four-rank DP with FP32 masters/states/main grads. Establish the 100M-token reference and, after approval, the 250M-per-candidate LR calibration above. Run the optional matched WSD/WSM gate only if the user requests it after LR selection. | Distributed result agrees with single-rank; resume exact; routing gates pass; clean tok/s/MFU/VRAM log exists; LR is selected by the preregistered rule or the top-two 1B fork is returned for approval. If requested, the schedule gate follows the predeclared comparison in Section 3 and returns all three endpoints. |
| 6. FP8 compute only | Keep optimizer state FP32. First enable per-tensor current scaling for diagnosis, then Hopper block scaling as the target. Enable and test `moe_router_padding_for_quantization`; keep router/SDPA/sensitive paths as in the precision table. | Pad/unpad identity passes for deliberately uneven expert loads, then unit numerical checks and 100M/1B parity gates pass against Stage 5. No state quantization yet. |
| 7. FP8 Adam states | Implement the simple reference quantizer and COAT DRE state path; serialize all metadata. Only then fuse the proven operation. Compare COAT E4M3/E4M3 with the predeclared E4M3/E5M2 ablation. | All state/step/distributed/resume tests pass; 100M and 1B parity gates pass against the FP8-compute/FP32-state Adam arm. |
| 8. Muon reference | Route only approved 2D logical matrices to EO Muon; use AdamW for exclusions. Verify split-QKV, split gate/up, and per-expert matrices, with the exact Muon flags in Section 3. Run the same 250M-per-candidate LR protocol after approval and profile optimizer overhead. | FP32-state Muon passes packing-invariance tests and the stability/routing gates; LR follows the preregistered selection rule. Optimizer is at most 5% of step time, or the batching fork returns for approval. |
| 9. FP8 Muon state | Add storage-only block-E4M3 momentum while retaining EO's FP32/BF16 NS path. Do not also change NS, scaling, or communication. | Muon unit/step/resume gates, then 100M and 1B parity against identical FP32-state Muon. |
| 10. Topology/communication | On clean 2 and 4 GPU allocations compare DP-replicated versus EP-sharded routed experts; the shared expert stays DP. If EP wins, compare plain all-to-all and DeepEP, then BF16 versus FP8 **dispatch** payload while combine remains BF16. | Select only from matched logged tok/s, MFU, peak VRAM, optimizer time, routing, and 1B parity. No “expected faster” choice. |
| 11. Final runs | Freeze code/config/data/container manifests. Propose the exact run matrix, GPU IDs, time estimate, and checkpoint policy for approval. | Explicit approval before each long run; final artifacts include logs, hashes, clean-GPU evidence, checkpoints, bpb/loss, tok/s, MFU, and peak VRAM. |

The minimum scientifically interpretable full-run matrix is:

1. FP8 compute + FP32-state AdamW.
2. FP8 compute + FP8-state AdamW.
3. FP8 compute + FP32-state Muon/AdamW hybrid.
4. FP8 compute + FP8-state Muon/AdamW hybrid.

A BF16-compute AdamW run is the initial reference and can remain at the 1B-token
gate unless the user wants a full compute-precision baseline. Removing either
FP32-state arm from the full matrix prevents a clean claim about state
quantization; that scope decision should be made after the 1B-token results,
before long runs.

## 10. Open questions and decisions

### User decisions needed before code

1. **Approved:** 64 routed experts/top-8, GPT-2 tokenizer, one shared expert,
   2K context, and aux-loss-free balancing.
2. **Approved:** 7.345B tokens as the provisional 1x-MoE-optimal budget and
   the `{0.6, 1.0, 1.4, 1.8}e-3` Adam/Muon learning-rate calibration. WSD
   remains the provisional reference. The user may instead request the matched
   WSM gate in Section 3 and choose WSM later; this approval does not authorize
   that extra gate.
3. Approve COAT DRE as the primary Adam state recipe and plain block-E4M3
   storage as the primary Muon state recipe.
4. Approve logical gate/up splitting for Muon NS. The rejected alternative is
   stock MCore behavior, which orthogonalizes the fused `[gate; up]` FC1 as one
   matrix and is therefore dependent on weight packing.
5. After the 1B-token gates, choose whether all four state-isolation arms above
   receive full 7.345B runs. No long run will be inferred from approval of this
   memo alone.

### Evidence-gated, not silently chosen

- node207 driver compatibility, H100 SXM/NVL form factor, topology, and disk;
- replicated versus EP=2/4 routed experts (shared expert remains DP);
- plain all-to-all versus DeepEP and BF16 versus FP8 EP dispatch payload;
- final LR within the declared sweep;
- whether stochastic rounding is justified by signed-error measurements;
- whether expert Muon needs a batched NS implementation;
- whether distributed optimizer/state sharding is useful at all at this size.

## 11. Source ledger

The source type matters:

- **Primary local papers:** [Ling 2.0](../tmp/literature/ling-moe.pdf);
  [NVIDIA Megatron-Core MoE](../tmp/literature/nvidia-moe.pdf).
- **Primary papers/preprints:** [Ling architecture/scaling
  companion](https://arxiv.org/abs/2507.17702); [Joint MoE Scaling
  Laws](https://arxiv.org/abs/2502.05172); [fine-grained MoE at larger
  scale](https://arxiv.org/abs/2506.02890);
  [Auxiliary-Loss-Free Load
  Balancing](https://arxiv.org/abs/2408.15664);
  [ST-MoE](https://arxiv.org/abs/2202.08906);
  [COAT](https://arxiv.org/abs/2410.19313); [trillion-token FP8
  training](https://arxiv.org/abs/2409.12517);
  [FP8-LM](https://arxiv.org/abs/2310.18313);
  [ECO](https://arxiv.org/abs/2601.22101);
  [FlashOptim](https://arxiv.org/abs/2602.23349);
  [Muon is Scalable](https://arxiv.org/abs/2502.16982);
  [8-bit Muon](https://arxiv.org/abs/2509.23106);
  [MuonQ](https://arxiv.org/abs/2605.11396);
  [Mellum2](https://arxiv.org/abs/2605.31268);
  [WSM](https://openreview.net/forum?id=HhThhjKyfw);
  [pretraining model merging](https://arxiv.org/abs/2505.12082);
  [WSD analysis](https://arxiv.org/abs/2410.05192);
  [floating-point quantization convergence analysis (ICLR
  2026)](https://openreview.net/forum?id=wwP1SCACee).
- **Primary technical reports:** [NVIDIA Nemotron 3
  Super](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf).
- **Official documentation:** [TE blockwise
  FP8](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html);
  [TE current
  scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html);
  [TE 2.16 release
  notes](https://docs.nvidia.com/deeplearning/transformer-engine/release-notes/index.html);
  [MCore MoE
  guide](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.2/megatron/core/transformer/moe/README.md);
  [NGC 26.04](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-04.html);
  [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/);
  [FineWeb dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb).
- **Repository evidence:** [MCore optimizer
  arguments](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.2/megatron/training/arguments.py);
  [MCore optimizer
  constraints](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.2/megatron/core/optimizer/optimizer_config.py);
  [MCore EO
  integration](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.2/megatron/core/optimizer/emerging_optimizers.py);
  [EO Muon
  implementation](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/tree/v0.2.0/emerging_optimizers/orthogonalized_optimizers);
  [COAT optimizer
  code](https://github.com/NVlabs/COAT/tree/80ec99f47aaa09231b07ace1fd04c30a1e30ec18/coat/optimizer);
  [TorchTitan](https://github.com/pytorch/torchtitan);
  [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel).

No blog performance number is used to choose the design. Repository claims
without a matched paper or local measurement are labeled as implementation
availability, not quality or speed evidence.
