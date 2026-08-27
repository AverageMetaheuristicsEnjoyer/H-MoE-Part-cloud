# MoE memory and time benchmark

Peak memory and step time for every arm, across model shape and batch size, on one
H100. It answers a different question from the 1C training runs: those measured six
arms at one shape and one micro-batch, this measures the shape of the memory/time
surface the arms live on.

The protocol is the one used in `effective-muon-membench`
(`scripts/monarch_benchmark`), so the MoE and dense tables can sit next to each
other and be read the same way.

## Protocol

| | |
|---|---|
| Axes | model shape x arm x micro-batch |
| Model shapes | `1p029b`, `2p094b` (below) |
| Arms | the six of `stage3_moe.ARMS`: {AdamW, Muon} x {bf16 baseline, FP8 GEMM, FP8 state} |
| Micro-batch | 1, 2, 4, 8, 16 |
| Tokens per optimizer step | **fixed at 16 sequences x 2,048 = 32,768**, accumulation traded against the micro-batch |
| Window | 5 warmup steps, then 12 measured steps |
| Peak memory | `torch.cuda.max_memory_allocated`, reset after warmup |
| Step time | median of the 12 measured full steps; the optimizer step is timed separately |
| Out of memory | a **result**, recorded as such; every larger micro-batch of that model and arm is then skipped |

Holding tokens per optimizer step fixed is what makes the time columns comparable:
every cell of the table does the same work, so a step time is a step time and not a
batch size in disguise. Peak memory still moves with the micro-batch, which is the
point of that axis.

The global batch is 16 sequences, not the 208 the 1C runs trained at. 208 at
micro-batch 1 is 208 forward/backward passes per measured step, which buys no
information and costs hours per cell.

### What the numbers are not

A membench point trains a **freshly initialized** model for 17 steps. The router has
not learned anything, so token-to-expert assignment is close to uniform and the
expert GEMMs are evenly sized. A trained router is not uniform, and Stage 3 already
measured that the two regimes time differently -- which is why `resume-bench` exists
for the 1.029B arms and why its numbers, not these, belong in a throughput claim.
These numbers are a like-for-like surface across shapes and batch sizes; both shapes
are measured cold, because only one of them has a checkpoint to resume from.

### Which corpus, and why it does not matter much

Train is read from the extension corpus on nfs2 --
`/workspace-SR006.nfs2/hmoe-data/fineweb-edu-time-match-extension/data/train.{bin,idx}`,
FineWeb-Edu 100BT shards 8-11, the one the causal control reads. Validation and test
stay at `/home/jovyan/data/fineweb-edu-gpt2-megatron/data/{development,final}.{bin,idx}`,
216 MB together. The 15.1 GB copy of the original Stage 3 train (shards 0-7) on
`/home/jovyan` is therefore not a dependency of this sweep.

Which of the two corpora a point reads has no measurable effect on it: 17 steps, one
tokenizer, one sequence length, a router that has learned nothing either way. What the
nfs2 prefix buys is that the sweep survives a cleanup of the volume with no free space,
including the resubmission that finishes it. Each point's own record carries the
`data_manifest_sha256` it ran against, so a table assembled across a change of corpus
says so rather than hiding it.

Mock data is a different matter and is not used: on it the AdamW router collapses and
the step time drifts through the measured window.

## The two model shapes

`1p029b` is the shape all six 1C arms were trained at: 18 layers, hidden 1,024, 64
routed experts of width 256 plus one shared, top-8. `docs/design.md` derives it.

`2p094b` changes exactly two things: 64 -> 128 routed experts, 18 -> 20 layers.
Everything else is held: hidden 1,024, expert width 256, top-8, one shared expert,
head dim 128, GQA 8/2, dense SwiGLU 2,816, GPT-2 50,257, sequence 2,048.

| | `1p029b` | `2p094b` |
|---|---:|---:|
| Layers (dense + MoE) | 1 + 17 | 1 + 19 |
| Routed experts | 64 | 128 |
| Total parameters | 1,028,926,976 | 2,094,088,192 |
| Active parameters/token | 280,243,712 | 301,023,232 |
| Total / active | 3.67x | 6.96x |
| Expert-bank activation `(8+1)/(E+1)` | 13.85% | 6.98% |
| Granularity `G = 2*d_model/d_expert` | 8 | 8 |
| Muon matrix / AdamW fallback | 924,844,032 / 104,082,944 | 1,988,624,384 / 105,463,808 |

### Why that shape

Total parameters double while active parameters rise 7.4%. That separation is the
whole reason for the second point: **optimizer state scales with total parameters and
compute scales with active parameters**, and FP8 optimizer state acts on the first
only. At `1p029b` the AdamW state is 8.23 GB against a 28.16 GB peak; doubling the
state while holding the compute fixed is the cleanest available test of whether the
FP8-state memory gate widens with model size, with no compute confound to argue about
afterwards.

The literature supports the direction. Ling 2.0's own models are far sparser than
`1p029b`: 256 routed experts, top-8, one shared, an overall activation ratio of about
3.5%, and expert width exactly one quarter of the model width at every scale
(512/2,048, 1,024/4,096, 2,048/8,192 -- so `G = 8`). Its companion scaling study finds
that efficiency leverage is driven primarily by the activation ratio, rising as
sparsity rises and still holding at ratios as extreme as 1/128, while expert
granularity is a secondary log-polynomial modulation with an optimum around `G = 8-12`
and the arrangement of shared and MoE layers is third-order. Moving 64 -> 128 routed
experts at fixed `G = 8` is exactly the axis that study says matters, and it lands the
expert-bank activation ratio at 6.98%, inside the 4.7-10.9% band the Ling fits were
validated over -- where `1p029b`'s 13.85% sits outside it.

Two shapes were rejected:

- **256 routed experts at hidden 1,024** (3.60B total, 3.50% activation) reproduces
  Ling's own sparsity exactly, and is the shape to prefer if the card were larger.
  With the non-distributed optimizer its FP32 AdamW state alone is 28.8 GB and the
  parameter-proportional footprint is around 72 GB before a single activation, so the
  bf16/FP32-state baseline -- the arm every ratio is taken against -- would be at or
  past the 80 GB wall at the bottom of the batch ramp. A table whose baseline column
  is all OOM measures nothing.
- **Widening to hidden 1,536** keeps `G = 8` at expert width 384 but multiplies the
  expert bank so fast that a 64-expert model is already 2.96B total with 0.68B active;
  the active compute grows with it and the shape confounds the two axes it was chosen
  to separate.

The learning rate is left at the `1p029b` value. A membench point runs 17 steps and
never trains; re-deriving a schedule for a shape nothing is trained at would only
suggest the shape had one.

## Running it

```bash
ssh brain_lab mlsub run \
  --repo https://github.com/AverageMetaheuristicsEnjoyer/H-MoE-Part-cloud \
  --branch bench/fp8-membench --entry scripts/cloud_membench.sh \
  --image torch28 --no-pip --gpus 1 --note membench-moe \
  --args "--models 1p029b --arms adamw_bf16_state_fp32 --micro-batches 1,2"
```

`--image torch28` is mandatory; the default image kills the launcher with an empty
log. The sweep is resumable: a point is reused when the controls it was recorded at
match the ones requested, so a job that runs out of time is continued by resubmitting
it. `--args "export"` prints one line per recorded point, which is the only way to
read the table back out of a finished job -- `mlsub logs` keeps a tail and the
workspace disk is not otherwise reachable. `--args "peek"` shows the newest log and
`--args "disk"` reports free space.

### Where it writes

`/workspace-SR006.nfs3/hmoe-membench`. **Not `/home/jovyan`:** that volume reached 0
bytes free on 2026-08-27, to the point that the platform could no longer create its
own job-log symlinks. The corpus is still read from `/home/jovyan/data`, which needs
no free space. Reclaimable there, if it becomes necessary: `hmoe-cloud` (14 GB of
Stage 3 logs) and `hmoe-checkpoints` (14 GB, duplicated on nfs2 and nfs3). The other
large directories -- `rl_muon` 33 GB, `.cache` 23 GB -- are not this project's.
