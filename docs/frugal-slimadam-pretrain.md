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
3. `lr-screen`: only after the stability gate, run the three declared Frugal recipes for 587
   steps (250,052,608 loss tokens), with a 6-step warmup and a 117-step exponential WSD tail.
   Rank by fixed-set final validation loss. The matched result remains the primary comparison.
4. If the best two sensitivity results differ by less than `0.01` nats, extending them to the
   existing 2,348-step calibration gate requires a separate implementation and approval.

No 1C run is authorized by these gates. Cloud runs use `torch28`, one H100, micro-batch 4,
global batch 208, the real Stage 3 dataset, and disabled gradient-accumulation fusion when the
APEX extension is absent.
