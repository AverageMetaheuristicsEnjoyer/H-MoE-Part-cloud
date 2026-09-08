# Frugal CoordAdamW and SlimAdam pretraining gates

The primary arms keep the completed Stage 3 optimizer settings: peak LR `1.63e-3`,
`beta1=0.9`, `beta2=0.95`, epsilon `1e-8`, weight decay `0.1`, and BF16 compute with FP32
optimizer state. This is the matched comparison; it is not a claim that these are optimal
hyperparameters for either method.

Frugal uses the Efficient-Training CoordAdamW variant: column coordinates, density `0.25`,
coordinate refresh every 50 optimizer steps, reset moments on refresh, and signSGD on inactive
coordinates. Two separately labelled sensitivity recipes reproduce the earlier search points:

| label | LR | beta2 | interpretation |
|---|---:|---:|---|
| `matched` | `1.63e-3` | `0.95` | primary Stage 3 comparison |
| `efficient-training-1e3` | `1e-3` | `0.999` | historical recipe sensitivity |
| `efficient-training-2e3` | `2e-3` | `0.999` | historical recipe sensitivity |

The sensitivity runs do not replace the matched arm. If the objective later changes to the
best tuned optimizer, AdamW and Muon need the same declared tuning opportunity.

Gate order:

1. `resume`: save at step 50, start a new process from that checkpoint, and finish at step 52.
   This puts the first resumed Frugal optimizer step on the coordinate-refresh boundary. The
   unit test also compares uninterrupted and resumed optimizer state across a refresh with the
   RNG restored.
2. `stability`: run both matched arms for 235 steps (100,106,240 loss tokens), with a 2-step
   warmup and a 47-step exponential WSD tail. Require a clean exit, zero NaN iterations,
   healthy routing, and fixed-set final validation loss.
3. `lr-screen`: only after the stability gate, run the three declared Frugal recipes and the
   matched SlimAdam recipe as four independent 587-step jobs (250,052,608 loss tokens each),
   with a 6-step warmup and a 117-step exponential WSD tail. Rank the Frugal recipes by
   fixed-set final validation loss; compare SlimAdam only to its matched recipe. The matched
   results remain the primary comparison.
4. If the best two sensitivity results differ by less than `0.01` nats, extending them to the
   existing 2,348-step calibration gate requires a separate implementation and approval.

No 1C run is authorized by these gates. Cloud runs use `torch28`, one H100, micro-batch 4,
global batch 208, the real Stage 3 dataset, and disabled gradient-accumulation fusion when the
APEX extension is absent.

Speed, peak memory, and persistent optimizer-state bytes are recorded for description only.
They are not pass/fail criteria for Frugal CoordAdamW or SlimAdam.

## Routing telemetry

Every training run writes `routing_telemetry.jsonl` beside `results.jsonl` at iteration 1,
every 10 steps, and the endpoint. Counts are MCore's global, unpadded assignments after its
normal TP/CP/DP all-reduce. Each record keeps the full per-layer expert counts and bias vector,
plus per-layer and aggregate minimum/mean, maximum/mean, coefficient of variation, zero-expert
count, and DeepSeek's batch MaxVio. A 100-step sum is scored separately so the gate can see
whether early collapse recovered instead of averaging it into the endpoint. Bias range,
update-direction flips, and total-variation drift of the load distribution are also recorded.

The lr-screen routing verdict reads the final 100-step window: every layer must have
minimum/mean at least `0.10`, CV below `0.20`, and zero dropped tokens. Batch and rolling
MaxVio are also sent to TensorBoard/W&B, including a curve for every MoE layer. DeepSeek's
Loss-Free Balancing paper defines MaxVio and plots batch MaxVio averaged over 100 neighboring
steps; the bias trace matters because too-small update rates converge slowly and too-large
rates oscillate late. OLMoE's checkpoint-to-checkpoint top-k overlap remains a fixed-data
post-hoc audit: normal training batches differ, so their token-level route IDs must not be
presented as routing saturation.

Sources: [DeepSeek Loss-Free Balancing](https://arxiv.org/abs/2408.15664),
[OLMoE](https://arxiv.org/abs/2409.02060).
