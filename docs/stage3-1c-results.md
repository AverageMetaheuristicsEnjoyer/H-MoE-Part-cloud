# Stage 3 MoE — the 1C results ledger

The 1C budget is **17,242 steps x 425,984 loss tokens = 7,344,816,128 tokens** per arm
(`--lr-decay-iters 17242 --lr-wsd-decay-iters 3448 --lr-warmup-iters 173`, WSD). Six arms,
one GPU each, launched 2026-08-20 as a continuation of the trunk from step 2254.

**Training is COMPLETE. Do not relaunch it.** All six reached `TRAIN_EXIT=0` at 17,242
between 2026-08-21 15:53 and 2026-08-23 12:36 UTC, 0 skipped / 0 NaN iterations everywhere.

## Final loss, iteration 17,242

Read from each run's own job log (`mlsub logs <job> | grep "loss at iteration 17242"`) —
W&B carries only the last *train* `lm loss`, never the val/test evaluation.

| arm | micro-batch | val | test | job |
|---|---|---:|---:|---|
| `muon_bf16_state_fp8` | 4 | **2.648571** | **2.632795** | `a13ae7f7` |
| `muon_bf16_state_fp32` | 4 | 2.650184 | 2.634595 | `bbab6612` |
| `adamw_bf16_state_fp32` | 4 | 2.675462 | 2.659013 | `2343c75f` |
| `muon_fp8gemm_state_fp32` | 16 | 2.682054 | 2.666146 | `ccb19a7f` |
| `adamw_bf16_state_fp8` | 4 | 2.694214 | 2.677219 | `1d15556c` |
| `adamw_fp8gemm_state_fp32` | 16 | 2.724416 | 2.706251 | `5e0b10ac` |

Relative to each pair's baseline (val):

| axis | AdamW | Muon |
|---|---|---|
| optimizer state, FP32 -> FP8 | +0.7009 % | **-0.0609 %** (FP8 nominally better) |
| FP8 GEMM vs bf16 | +1.8297 % | +1.2026 % |

Muon beats AdamW by **0.9448 %** val at matched settings (bf16, FP32 states, mb=4).

The state-axis numbers reproduce the 1.2B round (+0.662 % / -0.093 %) to within 0.05 pp:
the cost of FP8 optimizer state is stable across a 6x token budget.

**The compute axis is confounded and the confound is not closed.** The `fp8gemm` arms
trained at micro-batch 16 and their baselines at micro-batch 4, because both mb=16 bf16
controls were deliberately killed at ~5k steps to save ~72 GPU-h. The "mb=4 and mb=16
reproduce val loss to five significant figures" evidence came from *decay tails* only, so an
unknown share of the +1.83 / +1.20 % is micro-batch rather than FP8 GEMM. Closing it needs a
finished mb=16 bf16 baseline.

## Memory, measured on the deliverable runs

`max_allocated` / persistent optimizer state, from each run's own `results.jsonl`:

| arm | mb | max_alloc | persistent |
|---|---|---:|---:|
| `adamw_bf16_state_fp32` | 4 | 28,158,190,080 | 8,231,415,808 |
| `adamw_bf16_state_fp8` | 4 | 22,205,281,792 | 2,250,777,760 |
| `muon_bf16_state_fp32` | 4 | 24,453,571,584 | 4,532,039,680 |
| `muon_bf16_state_fp8` | 4 | 21,120,808,960 | 1,181,426,848 |
| `adamw_fp8gemm_state_fp32` | 16 | 53,033,605,120 | 8,231,415,808 |
| `muon_fp8gemm_state_fp32` | 16 | 49,333,291,520 | 4,532,039,680 |

State-axis memory gate: AdamW **1.2681**, Muon **1.1578**, both >= 1.10 **PASS**. Optimizer
state alone shrinks 3.66x / 3.84x. At mb=16 FP8 GEMM uses 3.27 GB *less* than bf16 (ratios
1.062 / 1.066) where at mb=4 it used 0.6 GB more — that axis is `memory=not_applicable`, so
this is a bonus, not a gate.

## Downstream, wave 1 — scored 2026-08-23

Primary `basic_v2` suite only: `hellaswag`, `arc_easy`, `arc_challenge`, `piqa`,
`gsm8k_gold_bpb_5shot`; 9 metrics. Not scored: the secondary set (`wikitext` and `c4` as
`loglikelihood_rolling`, plus `winogrande` / `openbookqa` / `mmlu` at 5-shot). That is not
only a time decision — `lm_eval_mcore.run_suite` collects only metrics whose name ends in
`_v2`, so a secondary task would run and contribute nothing to the record.

How it ran: two 1-GPU jobs, `36818058` (three AdamW arms) and `fe297497` (three Muon arms),
~56-58 min each, ~15 min per arm; pairing by `2ae02dfd`. **Both arms of a pair must share one
job** — `compare_runs` requires identical GPU identity and host — which is why the split is by
optimizer and not by axis.

**Self-check**: MCore's own validation on the loaded endpoint reproduces the training run's
number to five decimals (`adamw_fp8gemm` 2.724413 vs 2.724416, `muon_fp8gemm` 2.682019 vs
2.682054). The model being scored is provably the model that was trained.

### State axis — FP8 optimizer states vs FP32

| metric | AdamW base -> treat | | Muon base -> treat | |
|---|---|---|---|---|
| arc_easy bpb | 1.1509 -> 1.1347 | better | 1.0917 -> **1.1755** (+7.68 %) | **fail** |
| arc_challenge bpb | 1.2607 -> 1.2581 | pass | 1.2214 -> 1.2631 (+3.41 %) | **fail** |
| piqa bpb | 1.1872 -> 1.1864 | pass | 1.1621 -> 1.1873 (+2.16 %) | **fail** |
| gsm8k bpb | 0.7751 -> 0.7914 (+2.11 %) | **fail** | 0.7680 -> 0.7721 (+0.54 %) | pass |
| hellaswag bpb | 0.8943 -> 0.9031 (+0.99 %) | inconcl. | 0.8934 -> 0.8905 | pass |
| hellaswag len_norm | 0.4277 -> 0.4173 (+2.42 %) | **fail** | 0.4335 -> 0.4354 | pass |
| arc_challenge len_norm | 0.3080 -> 0.2858 (+7.20 %) | **fail** | 0.2927 -> 0.2944 | inconcl. |
| arc_easy acc | 0.6208 -> 0.6048 (+2.58 %) | inconcl. | 0.6204 -> 0.6296 | pass |
| piqa len_norm | 0.6839 -> 0.6795 (+0.64 %) | inconcl. | 0.6921 -> 0.6806 (+1.65 %) | inconcl. |

AdamW's gsm8k bpb reproduces the 1.2B round exactly: +2.11 % against +2.14 %. Muon's failure
widened from one metric at 1.2B (piqa bpb +1.38 %) to three, worst arc_easy bpb +7.68 % —
while its validation loss says FP8 state is free. Loss and downstream disagree hardest here.

### Compute axis — FP8 GEMM vs bf16

| metric | AdamW base -> treat | | Muon base -> treat | |
|---|---|---|---|---|
| arc_easy bpb | 1.1509 -> 1.2183 (+5.86 %) | **fail** | 1.0917 -> 1.0861 | pass |
| arc_challenge bpb | 1.2607 -> 1.3198 (+4.69 %) | **fail** | 1.2214 -> 1.1911 | pass |
| gsm8k bpb | 0.7751 -> 0.8179 (+5.53 %) | **fail** | 0.7680 -> 0.8070 (+5.08 %) | **fail** |
| piqa bpb | 1.1872 -> 1.2062 (+1.60 %) | **fail** | 1.1621 -> 1.1879 (+2.21 %) | **fail** |
| hellaswag bpb | 0.8943 -> 0.9109 (+1.86 %) | **fail** | 0.8934 -> 0.9074 (+1.56 %) | **fail** |
| hellaswag len_norm | 0.4277 -> 0.4129 (+3.47 %) | **fail** | 0.4335 -> 0.4282 (+1.22 %) | inconcl. |
| arc_challenge len_norm | 0.3080 -> 0.2850 (+7.48 %) | **fail** | 0.2927 -> 0.2867 (+2.04 %) | inconcl. |
| arc_easy acc | 0.6208 -> 0.5968 (+3.86 %) | **fail** | 0.6204 -> 0.6023 (+2.92 %) | inconcl. |
| piqa len_norm | 0.6839 -> 0.6812 (+0.40 %) | inconcl. | 0.6921 -> 0.6768 (+2.20 %) | inconcl. |

**This reverses the 1.2B conclusion.** At 1.2B, FP8 GEMM looked safe for AdamW (every bpb
passed and arc_easy / arc_challenge bpb *improved* 4-5 %) and destructive for Muon (4 bpb
failures). At 1C, AdamW fails 8 of 9 including those two arc metrics, and Muon's arc bpb now
improves. Do not carry the 1.2B contrast into the write-up. The micro-batch confound above
applies to this whole table.

### How to read these

- **The accuracy metrics now carry signal.** At 1.2B the models sat near chance and a 1 %
  relative bound was finer than the data supported. At 1C: hellaswag len_norm 0.43 (chance
  0.25), arc_easy acc 0.62, piqa len_norm 0.68-0.69.
- **Ignore `memory`, `wct` and `routing` in the evaluation records.** A scoring run holds no
  optimizer state and takes no training step, so its printed ratios describe the scoring job.
  The memory verdict is the training measurement above.
- `VERDICT=fail` on all four pairs is the *downstream* gate alone, and it is strict by
  construction: all nine CIs must sit entirely below +1 %, with no multiple-comparison
  allowance. Report the per-metric table, not the gate word.
- Each pair is one replicate, so `memory_allocated_ratio_ci95` and `e2e_wct_ratio_ci95` are
  null. The downstream CIs are paired per document and need no replicates.

## The numbers themselves

[`stage3-1c-downstream.json`](stage3-1c-downstream.json) holds all 54 metric values at full
precision, each arm's validation loss, the paired 95 % degradation intervals on the four
treatment arms, and the SHA-256 of every per-example artifact they were computed from. It was
lifted off the volume with `scripts/cloud_dump_downstream.sh`; the tables above are the same
numbers rounded.

**Nothing logs these scores anywhere else.** `run_suite` writes them to the run directory and
no code path pushes them to W&B, so before this file the only copy was on nfs2.

## Where the artifacts are

| what | where |
|---|---|
| checkpoints, mb=4 arms | `/workspace-SR006.nfs2/hmoe-checkpoints/stage3-1c-mb4/1c/<arm>/iter_0017242` |
| checkpoints, mb=16 arms | `/workspace-SR006.nfs3/hmoe-checkpoints/stage3/trunk/<arm>/iter_0017242` (+ `iter_0013794`) |
| scoring records + per-example artifacts | `/workspace-SR006.nfs2/hmoe-cloud/pretrain/stage3-<arm>-eval-downstream-1c/` |
| training run dirs | `/home/jovyan/hmoe-cloud/pretrain/stage3-<arm>-full/` |
| W&B | `https://wandb-radfan.ru`, entity `andrey`, project `hmoe-stage3` |
| HF archive | private repo `hmoe-stage3-checkpoints` |

The `1p2b/` endpoints on nfs3 are **gone locally** — only their trackers remain after the HF
offload. Re-scoring the 1.2B round means downloading from HF first.

## Reproducing or extending

```bash
# score arms on the 1C endpoints (both arms of a pair in ONE job)
mlsub run --repo https://github.com/AverageMetaheuristicsEnjoyer/H-MoE-Part-cloud \
  --branch stage3/moe-short-probes --entry scripts/cloud_moe_eval.sh --gpus 1 --image torch28 \
  --note eval-1c-adamw \
  --env "STAGE3_MOE_EVAL_ROOTS=/workspace-SR006.nfs2/hmoe-checkpoints/stage3-1c-mb4/1c:/workspace-SR006.nfs3/hmoe-checkpoints/stage3/trunk" \
  --env STAGE3_MOE_EVAL_BUDGET=1c --env STAGE3_MOE_RUN_SUFFIX=1c \
  --env STAGE3_MOE_LOG_ROOT=/workspace-SR006.nfs2/hmoe-cloud/pretrain \
  --args "adamw_bf16_state_fp32 adamw_bf16_state_fp8 adamw_fp8gemm_state_fp32"

# then pair them (CPU)
mlsub run ... --entry scripts/cloud_finalize.sh --gpus cpu --image torch28 \
  --env STAGE3_MOE_RESULTS_ROOT=/workspace-SR006.nfs2/hmoe-cloud/pretrain \
  --args "eval-downstream-1c"
```

Three things that are easy to get wrong:

1. **`STAGE3_MOE_EVAL_BUDGET=1c` is mandatory** for these endpoints. The eval mode otherwise
   uses the 1.2B schedule, and MCore restores `consumed_train_samples` from the checkpoint
   while sizing the train dataset from `train_iters` — a 17,242 endpoint then trips
   `MegatronPretrainingSampler`'s "no samples left to consume".
2. **`STAGE3_MOE_LOG_ROOT` must point off `/home/jovyan`.** That volume refuses new
   directories (`mkdir: No space left on device`) while `df` still reports ~29 G free — an
   inode or quota limit, not bytes. It killed the first eval submission in 27 s.
3. **mlsub rejects an environment value containing a space**, which is why the endpoint root
   list is colon separated.

## Open work

- **A finished mb=16 bf16 baseline** — the only thing that separates FP8 GEMM from
  micro-batch in the compute-axis numbers above. ~32.5 h (AdamW) / ~42.7 h (Muon) on one GPU.
- **Wave 2 downstream is exploratory/post-hoc** and does not replace the preregistered
  wave-1 gate. Its protocol was frozen before scoring: the same six final 1C endpoints;
  `wikitext` and `c4` `bits_per_byte` at 0-shot; `winogrande` accuracy, `openbookqa`
  accuracy and normalized accuracy, and aggregate `mmlu` accuracy at 5-shot. Accuracy
  uses paired document bootstrap with McNemar as a secondary test; bpb resamples matched
  documents and recomputes the official pooled-by-byte aggregate. A limited run is smoke
  only and must not be reported as a Wave 2 result.
- The paired CI provenance caveat from the 1.2B round: `build_inference` computes one
  downstream block per replicate group. With one replicate per pair here, it does not bite.
