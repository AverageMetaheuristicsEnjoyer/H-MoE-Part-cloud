# Stage 3: how FP8 optimizer states are done elsewhere

Our FP8 state codec (`stage3_moe/optimizer_states.py`) was written from scratch:
it dequantizes state into FP32 around a call to the wrapped optimizer's `step()`.
This note reads the reference implementations to decide whether that shape is
wrong. Everything below is read from source, not from abstracts; sources were
fetched on 2026-08-13 and function names are given because line numbers drift.

## 1. TransformerEngine `FusedAdam` — MCore's precision-aware optimizer

`transformer_engine/pytorch/optimizers/fused_adam.py`, reached from MCore through
`--use-precision-aware-optimizer` with `--exp-avg-dtype fp8`.

**Dequantization happens in Python, not in the kernel.** `get_unscaled_state`
holds, for `dtype == torch.uint8`:

```python
if dtype == torch.uint8:
    unscaled = unscaled_local_state.float()
```

`step()` calls it for every parameter of a group, appends the FP32 result to
`unscaled_lists[name]`, hands the lists to `multi_tensor_applier(self.multi_tensor_adam, ...)`,
and only then requantizes in `_apply_scale`. The FP32 copies are released at the
end of the group loop:

```python
# Try to reclaim the temporary fp32 buffers.
del unscaled_lists
```

So TE's FP32 working set is **every low-precision state in one param group, all
at once**. That is our `STAGE3_MOE_FP8_DEQUANT_CHUNK=0` behaviour, bounded by
group size rather than by a chunk.

What *is* fused:

| path | condition | fused? |
|---|---|---|
| `multi_tensor_adam_fp8` | the **parameter** is a delayed-scaling `Float8Tensor` | yes — FP8 *weights*, FP32 master + FP32 moments |
| `fuse_unscale` | `exp_avg_dtype == exp_avg_sq_dtype == torch.bfloat16` | yes — BF16 states updated in place, no FP32 copy |
| FP8 states | `exp_avg_dtype == torch.uint8` | **no** — Python `.float()`, FP32 kernel, Python requantize |

This corrects an earlier claim in this project that TE fuses
dequantize–update–requantize for FP8 states. It does not; `multi_tensor_adam_fp8`
is about FP8 *weights*.

**Scale granularity is per tensor**, and the codec asserts it:

```python
assert len(scaled_state._quantizer.scale) == 1, \
    "Only scaling with one scaling factor per tensor is supported by the FusedAdam."
```

The scale is recomputed each step from `torch.aminmax` over the whole FP32 state
(current scaling, not delayed), and both moments use E4M3.

**`store_param_remainders`.** A BF16 parameter already carries the top 16 bits of
its FP32 master weight. TE therefore stores only the *trailing* 16 bits, as
`int16`, and reconstructs the FP32 master inside
`multi_tensor_adam_param_remainder`. It is exact, not lossy: 2 bytes per
parameter saved with zero error. Enabled only when `master_weights=True`, the
master dtype is FP32 and the parameter dtype is BF16.

**It is not available to us.** `OptimizerConfig.__post_init__` asserts:

```python
assert self.optimizer == 'adam', '--use-precision-aware-optimizer only supported with adam'
assert self.use_distributed_optimizer, '--use-precision-aware-optimizer only supported with distributed optimizer'
```

Muon fails the first; the second requires `--use-distributed-optimizer`, which
shards optimizer state across DP ranks — the exact numerator the memory gate
stands on — and which `validate_axis` already forbids.

## 2. bitsandbytes 8-bit optimizers

`bitsandbytes/optim/optimizer.py`. State is `uint8`, one FP32 `absmax` per block
of **256** elements (2048 in the 2021 paper), plus a 256-entry codebook shared by
every parameter: `create_dynamic_map(signed=True)` for the first moment,
`signed=False` for the second. Values are indices into that codebook, not
floats — "dynamic tree" quantization, a non-uniform grid.

The update is a single CUDA kernel:

```python
elif state["state1"].dtype == torch.uint8:
    F.optimizer_update_8bit_blockwise(
        self.optimizer_name, grad, p, state["state1"], state["state2"], ...,
        state["qmap1"], state["qmap2"], state["absmax1"], state["absmax2"], ...)
```

Dequantize, update and requantize happen in registers. **No FP32 copy of the
state is ever materialized**, for any parameter. Tensors below 4096 elements are
kept in FP32.

## 3. torchao `torchao/optim`

`OptimStateFp8` is a tensor subclass holding `codes` (E4M3) and `scale`, with
plain absmax per block of **256**:

```python
scale = input.abs().amax(-1).clip(1e-12) / torch.finfo(DTYPE).max
```

The update is per parameter and compiled:

```python
torch.compile(single_param_adam, fullgraph=True, dynamic=False)(p.detach(), grad, ...)
```

`single_param_adam` writes `exp_avg.float().lerp(...)` and copies back through
the subclass's `aten.copy_`, so Inductor fuses the dequantize–update–requantize
into a small number of kernels and **one parameter's** FP32 state is live at a
time. Like bitsandbytes it skips small tensors:

```python
# follow bitsandbytes, only quantize tensors >= 4096 values
if local_p.numel() >= 4096 and local_p.numel() % self.block_size == 0:
```

## 4. COAT — where our DRE recipe comes from

`NVlabs/COAT`, `coat/optimizer/fp8_adamw.py`. Our state keys are literally theirs
(`scale_exp_avg`, `expand_exp_avg`, `sqrt_minmax_exp_avg`), and the defaults in
`coat/utils/_fp8_quantization_config.py` match what we implemented:

| COAT default | value | ours |
|---|---|---|
| `qgroup_size` | 128 | 128 |
| `expand_min` | 16 | `1.0 / 16` expansion floor |
| `first_order_bit` | E4M3 | E4M3 |
| `second_order_bit` | E4M3 (E4M3+E5M2 also offered) | E5M2 |

**Their update is one fused CUDA kernel per parameter**:

```python
qoptim_cuda.fp8_adamw_expand_step(
    param, grad,
    exp_avg, scale_exp_avg, expand_exp_avg, sqrt_minmax_exp_avg,
    exp_avg_sq, scale_exp_avg_sq, expand_exp_avg_sq, sqrt_minmax_exp_avg_sq,
    beta1, beta2, lr, weight_decay, eps, step, qgroup_size, expand_min)
```

No FP32 state buffer exists at any point. This is the same recipe we implemented,
with the fusion left in.

One detail worth noting: COAT allocates its three metadata arrays with
`dtype=p.dtype` — BF16 — while we allocate FP32. At group 128 that is
12 bytes per group versus 6, i.e. 0.094 vs 0.047 bytes per parameter.

## 5. DeepSpeed

Verified against `deepspeed/runtime/zero/config.py` and `runtime/config.py`:
DeepSpeed has **no quantized optimizer states**. `zero_quantized_weights`,
`zero_quantized_nontrainable_weights` and `zero_quantized_gradients` cover
weights and gradients (mostly for communication), and `quantize_training` is MoQ
for weights and activations. DeepSpeed's optimizer-state memory story is ZeRO
partitioning plus CPU/NVMe offload, which is a different axis from ours.

## 6. Muon specifically

*Effective Quantization of Muon Optimizer States*, arXiv 2509.23106, is the
closest published work to our Muon arm.

- Only the `momentum_buffer` of matrix parameters is quantized; non-matrix and
  input/output parameters stay on AdamW, and in the **recommended** variant that
  AdamW state stays FP32.
- Blockwise **linear (absmax)** quantization, block size 2048; they compare
  against dynamic-tree quantization and linear wins for Muon
  (dynamic shows a consistent degradation at larger scale, 2.508 vs 2.495
  validation loss at their XL scale).
- 62 % optimizer-state reduction at 2.7 B with validation-loss differences
  ≤ 0.002 against FP32 Muon.
- The reason linear suffices is structural: Muon keeps no second moment, so
  there is no `sqrt(v) + eps` denominator to amplify error, and Newton–Schulz
  orthogonalization equalizes the update spectrum. Their Theorem 3 bounds the
  weight error by the momentum's smallest singular value; the AdamW bound is
  unbounded as `eps → 0`.

This is independent confirmation of two choices we already made: `maxabs` rather
than DRE for Muon, and E4M3. Our group of 128 is 16× finer than their 2048.

Adam-mini, GaLore and friends reduce state by *structure* (one second moment per
block, low-rank projections) rather than by quantization; they are a different
axis and not comparable here.

## 7. Comparison

| | where the dequantize happens | FP32 state live at once | scale granularity | small tensors |
|---|---|---|---|---|
| bitsandbytes 8-bit | inside the CUDA kernel | none | block 256 + shared codebook | FP32 below 4096 |
| COAT FP8 AdamW | inside the CUDA kernel | none | group 128, 3 arrays in BF16 | all quantized |
| torchao | `torch.compile`, per parameter | one parameter | block 256 | FP32 below 4096 |
| TE `FusedAdam` | Python `.float()` | one whole param group | **per tensor** | all quantized |
| **ours** | Triton kernel before the wrapped `step()` | all params, or `chunk` of them | group 128, 3 arrays in FP32 | all quantized |

## 8. What this means for our mixin

**We are not behind TE.** On the two axes that matter we are ahead: group-128
scaling against TE's per-tensor scaling, and a tunable FP32 working set against
TE's whole-param-group one. The gap to bitsandbytes/COAT/torchao is real but is a
*fusion* gap, and the next two points are why closing it is not worth doing.

**A COAT-style fused kernel cannot exist for Muon.** COAT, bitsandbytes and
torchao all fuse an *element-wise* update. Muon's is not element-wise: after
`momentum_buffer.lerp_(grad, 1 - momentum)` the buffer is fed to Newton–Schulz,
which is a sequence of matmuls over the whole matrix. The dequantized momentum
has to exist as a full tensor for at least one parameter no matter what. The
tightest achievable design for Muon is therefore *exactly* torchao's — dequantize
one parameter, update, requantize — which is what `STAGE3_MOE_FP8_DEQUANT_CHUNK`
already does. The remaining question is only which chunk value is best, and
chunk=1 has not been measured.

**And the window is too small to be worth fusing.** Measured on the Muon arm,
mb=4, 1 GPU (`optimizer_step_seconds` against `full_step_seconds`, medians):

| arm | chunk | optimizer step | full step | share |
|---|---|---|---|---|
| `muon_bf16_state_fp32` | — | 0.938 s | 27.45 s | 3.4 % |
| `muon_bf16_state_fp8` | 0 | 1.122 s | 30.02 s | 3.7 % |
| `muon_bf16_state_fp8` | 32 | 1.163 s | 26.49 s | 4.4 % |
| `muon_bf16_state_fp8` | 128 | 1.047 s | 23.91 s | 4.4 % |

The whole optimizer step is 3–5 % of a training step, and the FP8 codec's share
of it is the ~0.1 s difference against the FP32 arm — **under 1 % of wall clock**.
A perfect fused kernel cannot recover more than that. The SOW has no WCT
requirement on the optimizer-state axis at all, which settles it.

**The chunking result needs re-reading.** `optimizer_step_seconds` is flat across
chunk 0/32/128 (1.12 / 1.16 / 1.05 s): chunking does not make the optimizer step
faster. The gain, whatever it is, is in forward and backward. Against the three
replicated chunk=0 runs (full step 30.02 / 29.15 / 26.06 s — a 15 % spread on an
identical configuration, `max_allocated` identical to the byte) chunk=128's
23.91 s is +9 % over the best and +19 % over the mean, not the +25.6 % that came
from comparing single runs.

`max_reserved_bytes` does support the allocator explanation, on the memory side
at least. Reserved minus allocated, mb=4:

| chunk | allocated | reserved | allocator overhead |
|---|---|---|---|
| 0 | 21,117,359,104 | 23,783,800,832 | 2.67 GB (12.6 %) |
| 32 | 21,117,156,352 | 22,871,539,712 | 1.75 GB (8.3 %) |
| 128 | 21,117,569,536 | 22,970,105,856 | 1.85 GB (8.8 %) |

Chunking gives back about **0.8 GB of reserved-but-unallocated memory** for the
same allocated peak, consistently. It does not explain the throughput spread
though: within chunk=0 the fastest run (16,344 tok/s) and the slowest
(14,188 tok/s) differ by 2.1 MB of reserved. So the memory effect is real and
measured; the speed claim still needs replicates before it is quoted.

**Worth taking from the prior art:**

1. *Skip small tensors.* bitsandbytes and torchao both leave tensors below 4096
   elements in FP32. We quantize everything, so every layernorm and bias pays 12
   bytes of metadata per 128 elements and takes quantization error for almost no
   saving.
2. *Metadata in BF16.* COAT stores `scale`/`expand`/`sqrt_minmax` in the
   parameter dtype, halving metadata from 0.094 to 0.047 bytes per parameter.
3. *`store_param_remainders` is the biggest untapped lever, and it is not on our
   axis.* It is exact, and at 1.029 B parameters it saves 2.06 GB of master
   weights. Applied to both arms it would move the mb=4 gate from
   24.454/21.117 = 1.158 to 22.394/19.057 = 1.175, because subtracting a constant
   from both sides of a ratio above 1 raises it. It needs either TE's FusedAdam
   (unavailable, see §1) or our own kernel plus a patch to MCore's master-weight
   storage.

**Not worth taking:** TE's per-tensor scaling (coarser than ours), and
bitsandbytes' dynamic-tree codebook for Muon (2509.23106 measures linear as
better there, and we measured the same in Stage 4).
