# SlimAdam fused versus split FC1 moments

Question: does sharing the compressed second moment across SwiGLU up/gate branches contribute to the persistent SlimAdam routing imbalance?

Two fresh runs, each 2254 optimizer steps (960,167,936 loss tokens), one allocated GPU per run, image `torch28`. One run keeps the original fused FC1 second moment; the other maintains independent compressed moments for the two FC1 halves. This covers 1106 FC1 weights: 1088 routed experts, 17 shared experts and one dense first layer. Both arms use the same new code revision. No AdamW/Frugal training is relaunched.

Controls: seed 1234, real Stage 3 data, BF16 computation, FP32 master parameters and optimizer state, micro-batch 4, global batch 208, sequence length 2048. LR 1.63e-3, beta1 0.9, beta2 0.95, epsilon 1e-8, weight decay 0.1, clipping 1. Full-run WSD schedule: warmup 173, total schedule 17242, decay 3448; the experiment ends on the plateau. Sigmoid top-8 of 64 experts, score scale 2.5, bias update 1e-3, no balancing aux loss. Router/QKV second moments remain full.

Expected additional split-state storage: `1106 * 1024 * 4 = 4,530,176 bytes` (4.32 MiB). This is a prediction to check against the runtime ledger, not a measured result.

`SlimAdamW` preserves a full first moment. Split FC1's second moment has shape `(2, 1, input_width)`; updates broadcast independently into the two matrix halves. The group flag is serialized in the ordinary torch checkpoint. Existing uncompressed and fused paths remain available. Runtime `slim_compression_manifest.json` records every parameter's actual rule.

Before either calibration, its smoke runs 12 steps, saves, restarts the process and resumes to 14. Only the disposable smoke checkpoints are removed after the successful reload; smoke logs/manifests remain. Training checkpoints rotate every 587 steps and retain the final 2254 checkpoint. Resubmission resumes from the variant's own directory. Do not submit a second active job for the same variant/mode.

Evaluation: retain batch and rolling-100 per-layer routing curves, expert bias and validation loss. At 2254, require all layers' rolling-100 minimum/mean >=0.10, CV <0.20, dropped=0. The final `endpoint.json` separates routing failure from execution failure: a clean run with poor routing is still a completed experiment. Compare curves and loss, not just one gate bit. This one-seed experiment is a mechanism screen, not a general quality or speed claim. Do not use generic `pair_results.py` for a cross-host speed/memory verdict.

Logs/checkpoints default under `/home/jovyan/hmoe-cloud/slimadam-fc1-ab-20260924-v1`. The runner requires 48 GiB free for overlapping paired checkpoint saves; `SLIM_AB_ROOT` can select another audited persistent volume without changing the experiment. Data cache is per variant. GPU identity, Torch/CUDA/TE versions, source commit and optimizer source hash are recorded. W&B is offline; persisted logs are authoritative.

Local validation: optimizer equivalence to independently optimized matrix halves, exact save/load continuation, FC1-only mapping, and an intercepted launcher test establishing that paired arguments differ only in the FC1 flag and artifact paths. Vendored-tree pins were refreshed to match the already-vendored dependencies; their source was not edited.

Cloud submission, after this branch has been published with permission:

```bash
mlsub run --repo https://github.com/AverageMetaheuristicsEnjoyer/H-MoE-Part-cloud \
  --branch codex/slimadam-swiglu-ab-20260924 --entry scripts/cloud_slimadam_ab.py \
  --gpus cpu --image torch28 --no-pip --note slim-ab-preflight --args 'preflight baseline'

# Submit baseline and split smokes separately, then verify SMOKE_RESULT=PASS for both.
mlsub run --repo https://github.com/AverageMetaheuristicsEnjoyer/H-MoE-Part-cloud \
  --branch codex/slimadam-swiglu-ab-20260924 --entry scripts/cloud_slimadam_ab.py \
  --gpus 1 --image torch28 --no-pip --note slim-ab-smoke-baseline --args 'smoke baseline'
mlsub run --repo https://github.com/AverageMetaheuristicsEnjoyer/H-MoE-Part-cloud \
  --branch codex/slimadam-swiglu-ab-20260924 --entry scripts/cloud_slimadam_ab.py \
  --gpus 1 --image torch28 --no-pip --note slim-ab-smoke-split --args 'smoke split'

# Only after those smokes, submit the two 2254-step runs.
mlsub run --repo https://github.com/AverageMetaheuristicsEnjoyer/H-MoE-Part-cloud \
  --branch codex/slimadam-swiglu-ab-20260924 --entry scripts/cloud_slimadam_ab.py \
  --gpus 1 --image torch28 --no-pip --note slim-ab-train-baseline --args 'train baseline'
mlsub run --repo https://github.com/AverageMetaheuristicsEnjoyer/H-MoE-Part-cloud \
  --branch codex/slimadam-swiglu-ab-20260924 --entry scripts/cloud_slimadam_ab.py \
  --gpus 1 --image torch28 --no-pip --note slim-ab-train-split --args 'train split'
```

Estimated training cost from historical 15–25 s/step: 9.4–15.7 GPU-hours per arm, about 19–31 GPU-hours total, plus smoke and evaluation/checkpoint overhead. This is not a current throughput measurement. Queueing can dominate wall time.
