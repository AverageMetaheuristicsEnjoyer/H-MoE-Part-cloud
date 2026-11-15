"""COAT FP8 activation flow (NVlabs/COAT@80ec99f) inside MCore's MoE transformer layers.

Every quantization, normalization, nonlinearity and dense FP8 GEMM below is an upstream
function from `third_party/coat/coat/activation/real_quantization`, imported unmodified, and
called the way `coat/models/coat_llama.py` calls it. Only training forwards are replaced; in
eval mode the original BF16 modules run, exactly as COAT's modules fall back to BF16.
Parameters, their names, initialization and the checkpoint layout are MCore's own.

What changes per transformer layer (operations not listed run unchanged in BF16):
  attention  input RMSNorm + fused QKV projection  -> COAT BeforeAttention flow (one fused
             QKV linear instead of three q/k/v linears, so one weight scale instead of three)
             output projection                    -> COAT AfterAttention flow (linear part)
  dense MLP  RMSNorm + gate/up + SiLU + mul + down -> COAT MLPResidual flow (gate and up are the
             two halves of MCore's fused fc1 and keep separate scales, as COAT's two linears do)
  MoE        pre-MLP RMSNorm                       -> COAT RMSNorm (FP8 per-tensor output; the
             router reads that output dequantized to BF16, the router itself is unchanged)
             shared expert                         -> COAT MLP flow
             routed experts                        -> COAT MLP flow per expert on permuted FP8
             tokens; GEMMs via torch._scaled_mm (cuBLASLt, per-tensor scales) because COAT's
             Triton GEMM takes M as a constexpr and writes output scales unmasked, which breaks
             on per-expert token counts. Router probabilities multiply the SiLU*up product
             before the FP8 down projection -- where MCore applies them -- and their gradient
             is taken from the FP8 copy saved for that projection.
Unchanged BF16: embeddings, qk-layernorm, RoPE, core attention (FlashAttention, as in COAT),
router, residual additions, final norm, output layer and loss.

STAGE3_MOE_COAT_STRICT=1 (off by default; no run started before 2026-10-02 10:00 MSK uses it)
brings two more places in line with coat_llama.py:
  - q, k and v are three linears with their own weight scales and their own gradient scales
    (NormLinearQKV) instead of one fused QKV linear;
  - weights follow COAT's weight_memory_efficient=True, the setting of every published COAT config:
    only the per-tensor scale is kept from the first micro-batch after an optimizer step, and the
    FP8 copy (forward) or its transpose (backward) is rebuilt from it on each use. The FP8 values
    are the same as in the cached mode; peak memory and time are not.

Two places differ numerically from coat_llama.py by construction:
  - COAT quantizes the residual stream per 1x16 group inside its residual-add kernel and feeds
    that FP8 copy to the next RMSNorm. Here each RMSNorm quantizes its own BF16 input with the
    same upstream `fp8_quantize` (1x16, E4M3), i.e. from the BF16 sum rather than the FP32 sum.
  - Weights are quantized once per optimizer step (COAT's FP8CacheWeightModule, non
    memory-efficient mode). The cache is dropped after every optimizer step: MCore writes the
    updated weights with multi_tensor_applier, which does not bump autograd version counters.
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/coat/coat/activation/real_quantization"
QB = 16
STRICT = os.environ.get("STAGE3_MOE_COAT_STRICT") == "1"


def _load_real_quantization():
    name = "coat_upstream.activation.real_quantization"
    if name in sys.modules:
        return sys.modules[name]
    for package, directory in (("coat_upstream", ROOT / "third_party/coat/coat"),
                               ("coat_upstream.activation", UPSTREAM.parent)):
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__ = [str(directory)]
            sys.modules[package] = module
    spec = importlib.util.spec_from_file_location(name, UPSTREAM / "__init__.py", submodule_search_locations=[str(UPSTREAM)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rq = _load_real_quantization()
_mul_fwd = sys.modules["coat_upstream.activation.real_quantization.mul_fwd"]
_common = sys.modules["coat_upstream.activation.real_quantization.common"]


# ---------------------------------------------------------------- weights

_WEIGHT_CACHE = {}


def invalidate_weights():
    _WEIGHT_CACHE.clear()


def fp8_weight(weight, key=None):
    """COAT prepare_weight: per-tensor E4M3 copy and its scale for the forward GEMM. Built on first
    use after each optimizer step (see invalidate_weights); rebuilt per use when STRICT."""
    if STRICT:
        w_s, _ = _weight_scale(weight, key)
        with torch.no_grad():
            return rq.fp8_division(weight.detach(), QB, "E4M3", w_s)
    w_fp8, w_s, _, _ = _weight_entry(weight, key)
    return w_fp8, w_s


def fp8_weight_t(weight, key=None):
    """The transposed E4M3 copy and its scale, for the backward GEMMs."""
    if STRICT:
        w_s, _ = _weight_scale(weight, key)
        with torch.no_grad():
            return rq.fp8_division_transpose(weight.detach(), QB, "E4M3", w_s, only_transposed=True)
    _, w_s, w_t, _ = _weight_entry(weight, key)
    return w_t, w_s


def fp8_weight_f32(weight):
    """fp8_weight for torch._scaled_mm, which takes the scale as an FP32 scalar."""
    return fp8_weight(weight)[0], _scale_f32(weight)


def fp8_weight_t_f32(weight):
    return fp8_weight_t(weight)[0], _scale_f32(weight)


def _scale_f32(weight):
    return _weight_scale(weight)[1] if STRICT else _weight_entry(weight)[3]


def _weight_entry(weight, key=None):
    # `key` names a weight that is a fresh copy on every call (the q/k/v parts of a fused QKV).
    key = key or (weight.data_ptr(), tuple(weight.shape))
    entry = _WEIGHT_CACHE.get(key)
    if entry is None:
        with torch.no_grad():
            w_fp8, w_s, w_t = rq.fp8_division_transpose(weight.detach(), QB, "E4M3")
        entry = (w_fp8, w_s, w_t, w_s.float())
        _WEIGHT_CACHE[key] = entry
    return entry


def _weight_scale(weight, key=None):
    """COAT prepare_weight with weight_memory_efficient=True: only the scale is kept."""
    key = ("scale",) + (key or (weight.data_ptr(), tuple(weight.shape)))
    entry = _WEIGHT_CACHE.get(key)
    if entry is None:
        with torch.no_grad():
            _, w_s, _ = rq.fp8_division_transpose(weight.detach(), QB, "E4M3")
        entry = (w_s, w_s.float())
        _WEIGHT_CACHE[key] = entry
    return entry


def _halves(fc1_weight):
    rows = fc1_weight.shape[0] // 2
    return fc1_weight[:rows], fc1_weight[rows:]


# ---------------------------------------------------------------- shared pieces

def _norm_forward(x2d, norm_weight, eps):
    qx, sx = rq.fp8_quantize(x2d, QB, "E4M3")
    ln_x, ln_s, ln_t, (_, rstd, num_warps) = rq.fp8_rmsnorm_forward(qx, sx, norm_weight, QB, eps, transpose_output_2d=True)
    return qx, sx, ln_x, ln_s, ln_t, rstd, num_warps


def _mlp_forward(ln_x, ln_s, fc1_weight, fc2_weight):
    """coat_llama _CoatLlamaMLPResidual.forward from the normalized input to the down projection."""
    gate_w, up_w = _halves(fc1_weight)
    g_fp8, g_s = fp8_weight(gate_w)
    u_fp8, u_s = fp8_weight(up_w)
    d_fp8, d_s = fp8_weight(fc2_weight)
    gate_x, gate_s = rq.fp8_linear_forward(ln_x, ln_s, g_fp8, g_s, True, QB)
    up_x, up_s = rq.fp8_linear_forward(ln_x, ln_s, u_fp8, u_s, True, QB)
    silu_x, silu_s = rq.fp8_silu_forward(gate_x, gate_s, QB)
    mul_x, mul_s, mul_t = rq.fp8_mul_forward(silu_x, silu_s, up_x, up_s, QB, transpose_output_2d=True)
    y = rq.fp8_linear_forward(mul_x, mul_s, d_fp8, d_s, False, QB)
    return y, (gate_x, gate_s, up_x, up_s, silu_x, silu_s, mul_t, mul_s)


def _mlp_backward(grad, ln_t, ln_s, saved, fc1_weight, fc2_weight):
    """coat_llama _CoatLlamaMLPResidual.backward down to the gradient of the normalized input."""
    gate_x, gate_s, up_x, up_s, silu_x, silu_s, mul_t, mul_s = saved
    gate_w, up_w = _halves(fc1_weight)
    g_t, g_s = fp8_weight_t(gate_w)
    u_t, u_s = fp8_weight_t(up_w)
    d_t, d_s = fp8_weight_t(fc2_weight)
    out_g, out_gs, out_g_t = rq.fp8_quantize_pertensor_transpose(grad.contiguous(), QB, "E5M2", transpose_output_2d=True)
    fc3_g, fc2_grad = rq.fp8_linear_backward(mul_t, mul_s, out_g, out_gs, out_g_t, d_t, d_s, QB)
    mul_g1, (mul_g2, mul_gs2, mul_g2_t) = rq.fp8_mul_backward(silu_x, silu_s, up_x, up_s, fc3_g, QB, "E5M2",
                                                             output_quantized_transpose=True)
    silu_g, silu_gs, silu_g_t = rq.fp8_silu_backward(gate_x, gate_s, mul_g1, QB, "E5M2", output_quantized_transpose=True)
    fc1_g, gate_grad = rq.fp8_linear_backward(ln_t, ln_s, silu_g, silu_gs, silu_g_t, g_t, g_s, QB)
    fc2_g, up_grad = rq.fp8_linear_backward(ln_t, ln_s, mul_g2, mul_gs2, mul_g2_t, u_t, u_s, QB)
    return fc1_g + fc2_g, torch.cat([gate_grad, up_grad], dim=0), fc2_grad


# ---------------------------------------------------------------- autograd functions

class NormLinear(torch.autograd.Function):
    """RMSNorm + one linear (attention QKV): coat_llama BeforeAttentionResidual."""

    @staticmethod
    def forward(ctx, x, norm_weight, weight, eps):
        shape = x.shape
        qx, sx, ln_x, ln_s, ln_t, rstd, num_warps = _norm_forward(x.reshape(-1, shape[-1]), norm_weight, eps)
        w_fp8, w_s = fp8_weight(weight)
        y = rq.fp8_linear_forward(ln_x, ln_s, w_fp8, w_s, False, QB)
        ctx.save_for_backward(qx, sx, ln_t, ln_s, rstd, norm_weight, weight)
        ctx.num_warps, ctx.shape = num_warps, shape
        return y.view(*shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, grad):
        qx, sx, ln_t, ln_s, rstd, norm_weight, weight = ctx.saved_tensors
        w_t, w_s = fp8_weight_t(weight)
        g, gs, g_t = rq.fp8_quantize_pertensor_transpose(grad.reshape(-1, grad.shape[-1]).contiguous(), QB, "E5M2",
                                                         transpose_output_2d=True)
        fc_g, weight_grad = rq.fp8_linear_backward(ln_t, ln_s, g, gs, g_t, w_t, w_s, QB)
        dx, norm_grad = rq.fp8_rmsnorm_backward(qx, sx, fc_g, norm_weight, rstd, QB, ctx.num_warps)
        return dx.view(ctx.shape), norm_grad, weight_grad, None


def _qkv_parts(weight, groups, q_rows, kv_rows):
    """MCore's fused QKV weight, rows [group: q heads | k head | v head], as COAT's q, k, v weights."""
    w = weight.view(groups, q_rows + 2 * kv_rows, weight.shape[-1])
    return (w[:, :q_rows].reshape(-1, weight.shape[-1]),
            w[:, q_rows:q_rows + kv_rows].reshape(-1, weight.shape[-1]),
            w[:, q_rows + kv_rows:].reshape(-1, weight.shape[-1]))


class NormLinearQKV(torch.autograd.Function):
    """coat_llama BeforeAttentionResidual on MCore's fused QKV weight: q, k and v are separate
    COAT linears with their own weight and gradient scales; output and weight gradient keep MCore's
    interleaved layout. `layout` = (query groups, q rows per group, k/v rows per group)."""

    @staticmethod
    def forward(ctx, x, norm_weight, weight, eps, layout):
        groups, q_rows, kv_rows = layout
        shape = x.shape
        qx, sx, ln_x, ln_s, ln_t, rstd, num_warps = _norm_forward(x.reshape(-1, shape[-1]), norm_weight, eps)
        rows = ln_x.shape[0]
        outputs = []
        for name, part in zip("qkv", _qkv_parts(weight, *layout)):
            w_fp8, w_s = fp8_weight(part, key=(weight.data_ptr(), name))
            outputs.append(rq.fp8_linear_forward(ln_x, ln_s, w_fp8, w_s, False, QB))
        y = torch.cat([outputs[0].view(rows, groups, q_rows), outputs[1].view(rows, groups, kv_rows),
                       outputs[2].view(rows, groups, kv_rows)], dim=2)
        ctx.save_for_backward(qx, sx, ln_t, ln_s, rstd, norm_weight, weight)
        ctx.num_warps, ctx.shape, ctx.layout = num_warps, shape, layout
        return y.view(*shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad):
        qx, sx, ln_t, ln_s, rstd, norm_weight, weight = ctx.saved_tensors
        groups, q_rows, kv_rows = ctx.layout
        g = grad.reshape(-1, groups, q_rows + 2 * kv_rows)
        rows = g.shape[0]
        slices = (g[:, :, :q_rows], g[:, :, q_rows:q_rows + kv_rows], g[:, :, q_rows + kv_rows:])
        fc_g, weight_grads = None, []
        for name, part, part_grad in zip("qkv", _qkv_parts(weight, *ctx.layout), slices):
            # COAT quantizes the query, key and value gradients separately.
            pg, pgs, pg_t = rq.fp8_quantize_pertensor_transpose(part_grad.reshape(rows, -1).contiguous(), QB, "E5M2",
                                                                transpose_output_2d=True)
            w_t, w_s = fp8_weight_t(part, key=(weight.data_ptr(), name))
            d, wg = rq.fp8_linear_backward(ln_t, ln_s, pg, pgs, pg_t, w_t, w_s, QB)
            fc_g = d if fc_g is None else fc_g + d
            weight_grads.append(wg)
        weight_grad = torch.cat([weight_grads[0].view(groups, q_rows, -1), weight_grads[1].view(groups, kv_rows, -1),
                                 weight_grads[2].view(groups, kv_rows, -1)], dim=1).view_as(weight)
        dx, norm_grad = rq.fp8_rmsnorm_backward(qx, sx, fc_g, norm_weight, rstd, QB, ctx.num_warps)
        return dx.view(ctx.shape), norm_grad, weight_grad, None, None


class Linear(torch.autograd.Function):
    """Attention output projection: coat_llama AfterAttentionResidual without the residual add."""

    @staticmethod
    def forward(ctx, x, weight):
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        xq, xs, _ = rq.fp8_quantize_pertensor(x2d, QB, "E4M3")
        w_fp8, w_s = fp8_weight(weight)
        y = rq.fp8_linear_forward(xq, xs, w_fp8, w_s, False, QB)
        # Like COAT, keep the BF16 input (core attention saves it anyway) and re-derive its
        # transposed FP8 copy in backward.
        ctx.save_for_backward(x2d, xs, weight)
        ctx.shape = shape
        return y.view(*shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, grad):
        x2d, xs, weight = ctx.saved_tensors
        w_t, w_s = fp8_weight_t(weight)
        g, gs, g_t = rq.fp8_quantize_pertensor_transpose(grad.reshape(-1, grad.shape[-1]).contiguous(), QB, "E5M2",
                                                         transpose_output_2d=True)
        x_t, xs = rq.fp8_division_transpose(x2d, QB, "E4M3", xs, only_transposed=True)
        dx, weight_grad = rq.fp8_linear_backward(x_t, xs, g, gs, g_t, w_t, w_s, QB)
        return dx.view(ctx.shape), weight_grad


class NormMLP(torch.autograd.Function):
    """Dense layer: RMSNorm + gate/up + SiLU + mul + down, coat_llama MLPResidual."""

    @staticmethod
    def forward(ctx, x, norm_weight, fc1_weight, fc2_weight, eps):
        shape = x.shape
        qx, sx, ln_x, ln_s, ln_t, rstd, num_warps = _norm_forward(x.reshape(-1, shape[-1]), norm_weight, eps)
        y, saved = _mlp_forward(ln_x, ln_s, fc1_weight, fc2_weight)
        ctx.save_for_backward(qx, sx, ln_t, ln_s, rstd, norm_weight, fc1_weight, fc2_weight, *saved)
        ctx.num_warps, ctx.shape = num_warps, shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, grad):
        qx, sx, ln_t, ln_s, rstd, norm_weight, fc1_weight, fc2_weight, *saved = ctx.saved_tensors
        fc_g, fc1_grad, fc2_grad = _mlp_backward(grad.reshape(-1, grad.shape[-1]), ln_t, ln_s, saved, fc1_weight, fc2_weight)
        dx, norm_grad = rq.fp8_rmsnorm_backward(qx, sx, fc_g, norm_weight, rstd, QB, ctx.num_warps)
        return dx.view(ctx.shape), norm_grad, fc1_grad, fc2_grad, None


class Norm(torch.autograd.Function):
    """Pre-MoE RMSNorm. Returns the FP8 per-tensor output dequantized to BF16 (what the router
    reads) plus the FP8 tensors themselves, which the expert functions consume."""

    @staticmethod
    def forward(ctx, x, norm_weight, eps):
        shape = x.shape
        qx, sx, ln_x, ln_s, ln_t, rstd, num_warps = _norm_forward(x.reshape(-1, shape[-1]), norm_weight, eps)
        ln_bf16 = (ln_x.float() * ln_s.float()).to(torch.bfloat16).view(shape)
        ctx.save_for_backward(qx, sx, rstd, norm_weight)
        ctx.num_warps, ctx.shape = num_warps, shape
        ctx.mark_non_differentiable(ln_x, ln_s, ln_t)
        return ln_bf16, ln_x, ln_s, ln_t

    @staticmethod
    def backward(ctx, grad, *unused):
        qx, sx, rstd, norm_weight = ctx.saved_tensors
        dx, norm_grad = rq.fp8_rmsnorm_backward(qx, sx, grad.reshape(-1, grad.shape[-1]).contiguous(), norm_weight, rstd,
                                                QB, ctx.num_warps)
        return dx.view(ctx.shape), norm_grad, None


class SharedMLP(torch.autograd.Function):
    """Shared expert on the FP8 normalized input. `anchor` is the BF16 normalized tensor: it is
    not read, it only carries this function's input gradient back to the norm."""

    @staticmethod
    def forward(ctx, anchor, ln_x, ln_s, ln_t, fc1_weight, fc2_weight):
        y, saved = _mlp_forward(ln_x, ln_s, fc1_weight, fc2_weight)
        ctx.save_for_backward(ln_t, ln_s, fc1_weight, fc2_weight, *saved)
        ctx.shape = anchor.shape
        return y.view(anchor.shape)

    @staticmethod
    def backward(ctx, grad):
        ln_t, ln_s, fc1_weight, fc2_weight, *saved = ctx.saved_tensors
        fc_g, fc1_grad, fc2_grad = _mlp_backward(grad.reshape(-1, grad.shape[-1]), ln_t, ln_s, saved, fc1_weight, fc2_weight)
        return fc_g.view(ctx.shape), None, None, None, fc1_grad, fc2_grad


def _mul_forward_unquantized(x1, s_x1, x2, s_x2):
    """First half of upstream fp8_mul_forward: the kernel's BF16 product, before the per-tensor
    quantization the wrapper applies next (which the expert path applies after the router
    probabilities instead)."""
    M, N = x1.shape
    _, SN = s_x1.shape
    y = torch.empty_like(x1, dtype=torch.bfloat16)
    s_y = torch.empty_like(s_x1, dtype=s_x1.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _mul_fwd.fp8_mul_forward_kernel[grid](
        y, s_y, x1, s_x1, x2, s_x2, M, N, SN, QB, _common.FP8_MAX_VALUE[x1.dtype],
        x1.stride(0), x1.stride(1), s_x1.stride(0), s_x1.stride(1), x2.stride(0), x2.stride(1),
        s_x2.stride(0), s_x2.stride(1), y.stride(0), y.stride(1), s_y.stride(0), s_y.stride(1),
        SCALE_MIN_THRES=_common.SCALE_MIN_THRES,
    )
    return y


def _mm(a, b, scale_a, scale_b, out_dtype, out=None):
    """FP8 GEMM with FP32 scalar scales. Expert loops issue tens of thousands of these per step,
    so scales are converted once by the caller and results land in place through `out`."""
    if out is None:
        return torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype)
    return torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype, out=out)


ALIGN = 16  # cuBLASLt FP8: row counts and leading dimensions in multiples of 16


class Experts(torch.autograd.Function):
    """Routed experts on permuted FP8 tokens. Segments are padded to 16 rows inside a buffer of
    fixed height, so every upstream Triton kernel sees one shape per run (they autotune on it)."""

    @staticmethod
    def forward(ctx, anchor, probs, ln_x, ln_s, routing_map, topk, *weights):
        experts = routing_map.shape[1]
        fc1, fc2 = weights[:experts], weights[experts:]
        tokens, hidden = ln_x.shape
        ffn = fc1[0].shape[0] // 2
        counts = routing_map.sum(dim=0)
        padded = (counts + ALIGN - 1) // ALIGN * ALIGN
        starts = torch.cumsum(padded, 0) - padded
        rows = tokens * topk + experts * ALIGN
        expert_idx, token_idx = routing_map.t().nonzero(as_tuple=True)  # MCore's permute order
        rank = torch.arange(token_idx.numel(), device=token_idx.device) - (torch.cumsum(counts, 0) - counts)[expert_idx]
        dest = starts[expert_idx] + rank
        x = torch.zeros(rows, hidden, dtype=torch.uint8, device=ln_x.device)
        x[dest] = ln_x.view(torch.uint8)[token_idx]
        x = x.view(ln_x.dtype)
        perm_probs = torch.zeros(rows, dtype=torch.float32, device=ln_x.device)
        perm_probs[dest] = probs[token_idx, expert_idx].float()
        counts_cpu, starts_cpu = counts.tolist(), starts.tolist()
        if sum(counts_cpu) != tokens * topk:
            raise RuntimeError("routing map does not carry topk experts per token")

        gate = torch.zeros(rows, ffn, dtype=torch.float32, device=ln_x.device)
        up = torch.zeros(rows, ffn, dtype=torch.float32, device=ln_x.device)
        ln_s32 = ln_s.float()
        for e, (count, start) in enumerate(zip(counts_cpu, starts_cpu)):
            if count == 0:
                continue
            end = start + (count + ALIGN - 1) // ALIGN * ALIGN
            gate_w, up_w = _halves(fc1[e])
            g_fp8, g_s32 = fp8_weight_f32(gate_w)
            u_fp8, u_s32 = fp8_weight_f32(up_w)
            _mm(x[start:end], g_fp8.t(), ln_s32, g_s32, torch.float32, out=gate[start:end])
            _mm(x[start:end], u_fp8.t(), ln_s32, u_s32, torch.float32, out=up[start:end])
        # COAT quantizes these 1x16 inside its GEMM epilogue; same function of the FP32 result.
        gate_x, gate_s = rq.fp8_quantize(gate, QB, "E4M3")
        up_x, up_s = rq.fp8_quantize(up, QB, "E4M3")
        del gate, up
        silu_x, silu_s = rq.fp8_silu_forward(gate_x, gate_s, QB)
        product = _mul_forward_unquantized(silu_x, silu_s, up_x, up_s)
        weighted = (product * perm_probs.unsqueeze(-1)).to(torch.bfloat16)
        del product
        mul_x, mul_s, mul_t = rq.fp8_division_transpose(weighted, QB, "E4M3")
        del weighted
        out = torch.zeros(rows, hidden, dtype=torch.bfloat16, device=ln_x.device)
        mul_s32 = mul_s.float()
        for e, (count, start) in enumerate(zip(counts_cpu, starts_cpu)):
            if count == 0:
                continue
            end = start + (count + ALIGN - 1) // ALIGN * ALIGN
            d_fp8, d_s32 = fp8_weight_f32(fc2[e])
            _mm(mul_x[start:end], d_fp8.t(), mul_s32, d_s32, torch.bfloat16, out=out[start:end])
        combined = torch.zeros(tokens, hidden, dtype=torch.float32, device=ln_x.device)
        combined.index_add_(0, token_idx, out[dest].float())
        ctx.save_for_backward(x, ln_s, gate_x, gate_s, up_x, up_s, silu_x, silu_s, mul_t, mul_s,
                              perm_probs, dest, token_idx, expert_idx, *weights)
        ctx.meta = (counts_cpu, starts_cpu, experts, tokens, hidden, rows, anchor.shape, probs.dtype)
        return combined.to(torch.bfloat16).view(anchor.shape)

    @staticmethod
    def backward(ctx, grad):
        (x, ln_s, gate_x, gate_s, up_x, up_s, silu_x, silu_s, mul_t, mul_s,
         perm_probs, dest, token_idx, expert_idx, *weights) = ctx.saved_tensors
        counts, starts, experts, tokens, hidden, rows, shape, probs_dtype = ctx.meta
        fc1, fc2 = weights[:experts], weights[experts:]
        ffn = fc1[0].shape[0] // 2
        grad = grad.reshape(tokens, hidden)
        g_perm = torch.zeros(rows, hidden, dtype=torch.bfloat16, device=grad.device)
        g_perm[dest] = grad[token_idx].to(torch.bfloat16)
        g, gs, g_t = rq.fp8_quantize_pertensor_transpose(g_perm, QB, "E5M2", transpose_output_2d=True)
        del g_perm
        d_weighted = torch.zeros(rows, ffn, dtype=torch.bfloat16, device=grad.device)
        gs32, mul_s32 = gs.float(), mul_s.float()
        fc2_grads = []
        for e, (count, start) in enumerate(zip(counts, starts)):
            if count == 0:
                fc2_grads.append(torch.zeros_like(fc2[e]))
                continue
            end = start + (count + ALIGN - 1) // ALIGN * ALIGN
            d_t, d_s32 = fp8_weight_t_f32(fc2[e])
            _mm(g[start:end], d_t.t(), gs32, d_s32, torch.bfloat16, out=d_weighted[start:end])
            fc2_grads.append(_mm(g_t[:, start:end], mul_t[:, start:end].t(), gs32, mul_s32, torch.bfloat16))
        # d(prob) = <d(prob*h), prob*h> / prob, from the FP8 copy of prob*h the down GEMM used.
        weighted = mul_t.t().float() * mul_s.float()
        safe = torch.where(perm_probs > 0, perm_probs, torch.ones_like(perm_probs))
        d_perm_probs = (d_weighted.float() * weighted).sum(-1) / safe
        del weighted
        probs_grad = torch.zeros(tokens, experts, dtype=probs_dtype, device=grad.device)
        probs_grad[token_idx, expert_idx] = d_perm_probs[dest].to(probs_dtype)
        d_product = (d_weighted.float() * perm_probs.unsqueeze(-1)).to(torch.bfloat16)
        del d_weighted
        mul_g1, (mul_g2, mul_gs2, mul_g2_t) = rq.fp8_mul_backward(silu_x, silu_s, up_x, up_s, d_product, QB, "E5M2",
                                                                 output_quantized_transpose=True)
        del d_product
        silu_g, silu_gs, silu_g_t = rq.fp8_silu_backward(gate_x, gate_s, mul_g1, QB, "E5M2", output_quantized_transpose=True)
        del mul_g1
        x_t = x.view(torch.uint8).t().contiguous().view(x.dtype)
        d_x = torch.zeros(rows, hidden, dtype=torch.bfloat16, device=grad.device)
        silu_gs32, mul_gs2_32, ln_s32 = silu_gs.float(), mul_gs2.float(), ln_s.float()
        fc1_grads = []
        for e, (count, start) in enumerate(zip(counts, starts)):
            if count == 0:
                fc1_grads.append(torch.zeros_like(fc1[e]))
                continue
            end = start + (count + ALIGN - 1) // ALIGN * ALIGN
            gate_w, up_w = _halves(fc1[e])
            g_wt, g_s32 = fp8_weight_t_f32(gate_w)
            u_wt, u_s32 = fp8_weight_t_f32(up_w)
            _mm(silu_g[start:end], g_wt.t(), silu_gs32, g_s32, torch.bfloat16, out=d_x[start:end])
            d_x[start:end] += _mm(mul_g2[start:end], u_wt.t(), mul_gs2_32, u_s32, torch.bfloat16)
            fc1_grad = torch.empty_like(fc1[e])
            _mm(silu_g_t[:, start:end], x_t[:, start:end].t(), silu_gs32, ln_s32, torch.bfloat16, out=fc1_grad[:ffn])
            _mm(mul_g2_t[:, start:end], x_t[:, start:end].t(), mul_gs2_32, ln_s32, torch.bfloat16, out=fc1_grad[ffn:])
            fc1_grads.append(fc1_grad)
        d_ln = torch.zeros(tokens, hidden, dtype=torch.float32, device=grad.device)
        d_ln.index_add_(0, token_idx, d_x[dest].float())
        return (d_ln.to(torch.bfloat16).view(shape), probs_grad, None, None, None, None, *fc1_grads, *fc2_grads)


def _stacked_expert_weights(fc1, fc2):
    """Per-expert COAT weight quantization (as fp8_weight), stacked for the grouped kernels."""
    if os.environ.get("STAGE3_MOE_COAT_GROUPED_STACKED_WEIGHTS") == "1":
        # Same codes, scales and layouts from one launch per projection (the fused path's quantizer):
        # isolates the cost of the 192 per-expert launches in the speed decomposition.
        from stage3_moe.coat_fused_experts import _stacked_weights
        return _stacked_weights(fc1, fc2)
    key = ("stacked", fc1[0].data_ptr(), fc2[0].data_ptr(), len(fc1))
    entry = _WEIGHT_CACHE.get(key)
    if entry is None:
        parts = {name: ([], [], []) for name in ("gate", "up", "down")}
        with torch.no_grad():
            for w1, w2 in zip(fc1, fc2):
                for name, weight in zip(("gate", "up", "down"), (*_halves(w1), w2)):
                    w_fp8, w_s, w_t = rq.fp8_division_transpose(weight.detach(), QB, "E4M3")
                    parts[name][0].append(w_fp8.view(torch.uint8))
                    parts[name][1].append(w_t.view(torch.uint8))
                    parts[name][2].append(w_s.reshape(1))
        entry = {}
        for name, (data, data_t, scales) in parts.items():
            entry[name] = torch.stack(data).view(torch.float8_e4m3fn)
            entry[name + "_t"] = torch.stack(data_t).view(torch.float8_e4m3fn)
            entry[name + "_s"] = torch.cat(scales).float()
        _WEIGHT_CACHE[key] = entry
    return entry


class GroupedExperts(torch.autograd.Function):
    """Experts with the same COAT flow, but every expert GEMM of a kind in one Triton launch
    (stage3_moe.coat_grouped_gemm) and the permutation glue done by gathers: no atomics, no FP32
    copies of the permuted tensors, no host synchronization. Segments are padded to 128 rows."""

    @staticmethod
    def forward(ctx, anchor, probs, ln_x, ln_s, routing_map, topk, *weights):
        from stage3_moe import coat_grouped_gemm as gg

        align = gg.BLOCK_M
        experts = routing_map.shape[1]
        fc1, fc2 = weights[:experts], weights[experts:]
        tokens, hidden = ln_x.shape
        ffn = fc1[0].shape[0] // 2
        device = ln_x.device
        counts = routing_map.sum(dim=0)
        padded = (counts + align - 1) // align * align
        starts = torch.cumsum(padded, 0) - padded
        rows = tokens * topk + experts * align
        expert_idx, token_idx = routing_map.t().nonzero(as_tuple=True)  # MCore's permute order
        rank = torch.arange(token_idx.numel(), device=device) - (torch.cumsum(counts, 0) - counts)[expert_idx]
        dest = starts[expert_idx] + rank
        if token_idx.numel() != tokens * topk:
            raise RuntimeError("routing map does not carry topk experts per token")
        # Row -> token, with padding rows pointing at an appended zero row.
        row_token = torch.full((rows,), tokens, dtype=torch.long, device=device)
        row_token[dest] = token_idx
        row_flat = torch.full((rows,), tokens * experts, dtype=torch.long, device=device)
        row_flat[dest] = token_idx * experts + expert_idx
        token_rows = dest[torch.argsort(token_idx, stable=True)].view(tokens, topk)
        zero_row = torch.zeros(1, hidden, dtype=torch.uint8, device=device)
        x = torch.cat([ln_x.view(torch.uint8), zero_row])[row_token].view(ln_x.dtype)
        perm_probs = torch.cat([probs.reshape(-1).float(), torch.zeros(1, device=device)])[row_flat]
        tiles = gg.tile_experts(starts, padded, rows)
        w = _stacked_expert_weights(fc1, fc2)
        ln_s32 = ln_s.float()
        gate = torch.empty(rows, ffn, dtype=torch.float32, device=device)
        up = torch.empty(rows, ffn, dtype=torch.float32, device=device)
        gg.segment_gemm(x, w["gate"], ln_s32, w["gate_s"], tiles, gate)
        gg.segment_gemm(x, w["up"], ln_s32, w["up_s"], tiles, up)
        gate_x, gate_s = rq.fp8_quantize(gate, QB, "E4M3")
        up_x, up_s = rq.fp8_quantize(up, QB, "E4M3")
        del gate, up
        silu_x, silu_s = rq.fp8_silu_forward(gate_x, gate_s, QB)
        product = _mul_forward_unquantized(silu_x, silu_s, up_x, up_s)
        weighted = (product * perm_probs.unsqueeze(-1)).to(torch.bfloat16)
        del product
        mul_x, mul_s, mul_t = rq.fp8_division_transpose(weighted, QB, "E4M3")
        del weighted
        out = torch.empty(rows, hidden, dtype=torch.bfloat16, device=device)
        gg.segment_gemm(mul_x, w["down"], mul_s.float(), w["down_s"], tiles, out)
        combined = gg.gather_sum(out, token_rows, torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device))
        ctx.save_for_backward(x, ln_s, gate_x, gate_s, up_x, up_s, silu_x, silu_s, mul_t, mul_s,
                              perm_probs, row_token, row_flat, token_rows, starts, padded, tiles, *weights)
        ctx.meta = (experts, tokens, hidden, rows, ffn, anchor.shape, probs.dtype)
        return combined.view(anchor.shape)

    @staticmethod
    def backward(ctx, grad):
        from stage3_moe import coat_grouped_gemm as gg

        (x, ln_s, gate_x, gate_s, up_x, up_s, silu_x, silu_s, mul_t, mul_s,
         perm_probs, row_token, row_flat, token_rows, starts, padded, tiles, *weights) = ctx.saved_tensors
        experts, tokens, hidden, rows, ffn, shape, probs_dtype = ctx.meta
        fc1, fc2 = weights[:experts], weights[experts:]
        device = grad.device
        w = _stacked_expert_weights(fc1, fc2)
        grad = grad.reshape(tokens, hidden).to(torch.bfloat16)
        g_perm = torch.cat([grad, torch.zeros(1, hidden, dtype=torch.bfloat16, device=device)])[row_token]
        g, gs, g_t = rq.fp8_quantize_pertensor_transpose(g_perm, QB, "E5M2", transpose_output_2d=True)
        del g_perm
        gs32, mul_s32, ln_s32 = gs.float(), mul_s.float(), ln_s.float()
        d_weighted = torch.empty(rows, ffn, dtype=torch.bfloat16, device=device)
        gg.segment_gemm(g, w["down_t"], gs32, w["down_s"], tiles, d_weighted)
        fc2_grad = torch.empty(experts, hidden, ffn, dtype=fc2[0].dtype, device=device)
        gg.segment_wgrad(g_t, mul_t, gs32, mul_s32, starts, padded, fc2_grad)
        d_perm_probs = torch.empty(rows, dtype=torch.float32, device=device)
        d_product = torch.empty(rows, ffn, dtype=torch.bfloat16, device=device)
        gg.probs_backward(d_weighted, mul_t, mul_s32, perm_probs, d_perm_probs, d_product)
        del d_weighted
        probs_grad = torch.zeros(tokens * experts + 1, dtype=torch.float32, device=device)
        probs_grad.index_put_((row_flat,), d_perm_probs)  # every real (token, expert) appears once
        probs_grad = probs_grad[:-1].view(tokens, experts).to(probs_dtype)
        mul_g1, (mul_g2, mul_gs2, mul_g2_t) = rq.fp8_mul_backward(silu_x, silu_s, up_x, up_s, d_product, QB, "E5M2",
                                                                 output_quantized_transpose=True)
        del d_product
        silu_g, silu_gs, silu_g_t = rq.fp8_silu_backward(gate_x, gate_s, mul_g1, QB, "E5M2", output_quantized_transpose=True)
        del mul_g1
        silu_gs32, mul_gs2_32 = silu_gs.float(), mul_gs2.float()
        x_t = x.view(torch.uint8).t().contiguous().view(x.dtype)
        d_x = torch.empty(rows, hidden, dtype=torch.bfloat16, device=device)
        gg.segment_gemm(silu_g, w["gate_t"], silu_gs32, w["gate_s"], tiles, d_x)
        gg.segment_gemm(mul_g2, w["up_t"], mul_gs2_32, w["up_s"], tiles, d_x, accumulate=True)
        fc1_grad = torch.empty(experts, 2 * ffn, hidden, dtype=fc1[0].dtype, device=device)
        gg.segment_wgrad(silu_g_t, x_t, silu_gs32, ln_s32, starts, padded, fc1_grad[:, :ffn])
        gg.segment_wgrad(mul_g2_t, x_t, mul_gs2_32, ln_s32, starts, padded, fc1_grad[:, ffn:])
        d_ln = gg.gather_sum(d_x, token_rows, torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device))
        return (d_ln.view(shape), probs_grad, None, None, None, None, *fc1_grad.unbind(0), *fc2_grad.unbind(0))


# ---------------------------------------------------------------- MCore wiring

def _check(condition, message):
    if not condition:
        raise RuntimeError("COAT activation port: " + message)


def _patch_linear_qkv(module, eps, attention):
    original = module.forward
    _check(hasattr(module, "layer_norm_weight"), "linear_qkv must fuse the input RMSNorm")
    head = attention.hidden_size_per_attention_head
    groups = attention.num_query_groups_per_partition
    layout = (groups, attention.num_attention_heads_per_partition // groups * head, head)
    _check(module.weight.shape[0] == groups * (layout[1] + 2 * head), "unexpected fused QKV layout")

    def forward(inp, *args, **kwargs):
        if not module.training:
            return original(inp, *args, **kwargs)
        if STRICT:
            return NormLinearQKV.apply(inp, module.layer_norm_weight, module.weight, eps, layout), None
        return NormLinear.apply(inp, module.layer_norm_weight, module.weight, eps), None

    module.forward = forward


def _patch_linear(module):
    original = module.forward

    def forward(inp, *args, **kwargs):
        if not module.training:
            return original(inp, *args, **kwargs)
        return Linear.apply(inp, module.weight), None

    module.forward = forward


def _patch_dense_mlp(mlp, eps):
    original = mlp.forward
    fc1, fc2 = mlp.linear_fc1, mlp.linear_fc2
    _check(hasattr(fc1, "layer_norm_weight"), "dense MLP must fuse its RMSNorm into linear_fc1")

    def forward(hidden_states, per_token_scale=None, **kwargs):
        if not mlp.training:
            return original(hidden_states, per_token_scale, **kwargs)
        _check(per_token_scale is None, "per_token_scale is not ported")
        return NormMLP.apply(hidden_states, fc1.layer_norm_weight, fc1.weight, fc2.weight, eps), None

    mlp.forward = forward


def _patch_moe(layer, eps):
    norm, moe = layer.pre_mlp_layernorm, layer.mlp
    norm_original, moe_original = norm.forward, moe.forward
    experts = moe.experts
    count = experts.num_local_experts
    fc1 = [getattr(experts.linear_fc1, f"weight{i}") for i in range(count)]
    fc2 = [getattr(experts.linear_fc2, f"weight{i}") for i in range(count)]
    shared = moe.shared_experts
    _check(shared is not None and not getattr(shared, "use_shared_expert_gate", False), "expected an ungated shared expert")
    topk = moe.config.moe_router_topk
    experts_function = GroupedExperts if os.environ.get("STAGE3_MOE_COAT_GROUPED_GEMM") == "1" else Experts
    fused = os.environ.get("STAGE3_MOE_COAT_FUSED_EXPERTS") == "1"
    if fused:
        from stage3_moe.coat_fused_experts import FusedExperts
    # The grouped and fused paths stack cached FP8 copies of every expert; they have no memory-efficient mode.
    _check(not (STRICT and (fused or experts_function is GroupedExperts)), "STRICT does not cover the grouped expert paths")

    def norm_forward(hidden_states):
        if not norm.training:
            return norm_original(hidden_states)
        ln_bf16, ln_x, ln_s, ln_t = Norm.apply(hidden_states, norm.weight, eps)
        ln_bf16.coat_fp8 = (ln_x, ln_s, ln_t)
        return ln_bf16

    def moe_forward(hidden_states, intermediate_tensors=None, padding_mask=None):
        if not moe.training or not hasattr(hidden_states, "coat_fp8"):
            return moe_original(hidden_states, intermediate_tensors, padding_mask=padding_mask)
        _check(intermediate_tensors is None, "partial MoE execution is not ported")
        ln_x, ln_s, ln_t = hidden_states.coat_fp8
        del hidden_states.coat_fp8
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()
        probs, routing_map = moe.route(hidden_states, padding_mask)
        shared_out = SharedMLP.apply(hidden_states, ln_x, ln_s, ln_t, shared.linear_fc1.weight, shared.linear_fc2.weight)
        if fused:
            routed = FusedExperts.apply(hidden_states, probs, ln_x, ln_s, ln_t, routing_map, topk, *fc1, *fc2)
        else:
            routed = experts_function.apply(hidden_states, probs, ln_x, ln_s, routing_map, topk, *fc1, *fc2)
        return routed + shared_out, None

    norm.forward = norm_forward
    moe.forward = moe_forward


def install(models, optimizer):
    """Swap the training forwards of every transformer layer for the COAT flow."""
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.transformer_layer import TransformerLayer

    counts = {"attention": 0, "dense_mlp": 0, "moe": 0}
    for model in models:
        for module in model.modules():
            if not isinstance(module, TransformerLayer):
                continue
            config = module.config
            _check(config.normalization == "RMSNorm" and config.gated_linear_unit and not config.add_bias_linear,
                   "expected RMSNorm, SwiGLU and bias-free linears")
            _check(config.tensor_model_parallel_size == 1 and config.expert_model_parallel_size == 1, "TP=1 and EP=1 only")
            _check(not config.fp8, "COAT replaces TE FP8; do not combine them")
            eps = config.layernorm_epsilon
            attention = module.self_attention
            _check(not getattr(config, "attention_output_gate", False), "gated attention is not ported")
            _patch_linear_qkv(attention.linear_qkv, eps, attention)
            _patch_linear(attention.linear_proj)
            counts["attention"] += 1
            if isinstance(module.mlp, MoELayer):
                _check(not config.moe_apply_probs_on_input, "probabilities must apply after the activation")
                _patch_moe(module, eps)
                counts["moe"] += 1
            else:
                _patch_dense_mlp(module.mlp, eps)
                counts["dense_mlp"] += 1
    if optimizer is not None:  # eval-only modes (--skip-train) build no optimizer; weights never change
        step = optimizer.step

        def step_and_invalidate(*args, **kwargs):
            try:
                return step(*args, **kwargs)
            finally:
                invalidate_weights()

        optimizer.step = step_and_invalidate
    print("COAT_ACTIVATIONS_INSTALLED " + " ".join(f"{k}={v}" for k, v in counts.items()) + f" strict={int(STRICT)}", flush=True)
    return counts
