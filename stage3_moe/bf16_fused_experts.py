"""BF16 routed experts on the launch structure of coat_fused_experts (STAGE3_MOE_BF16_FUSED_EXPERTS=1).

The control that separates precision from implementation: the same grouped Triton GEMMs, one
launch per GEMM kind, token rows read through the row -> token index, no permuted copies -- but
BF16 operands and no quantization anywhere. COAT on the fused path against this arm is the effect
of FP8 alone; this arm against MCore's own BF16 experts is the effect of the expert-path design.

The arithmetic is MCore's SwiGLU expert MLP with router probabilities applied after the
activation: out[t] = sum_e p[t, e] * silu(gate) * up @ W2[e].T, [gate | up] = x[t] @ W1[e].T.
The activation is evaluated in FP32 and rounded to BF16 once, where MCore rounds after each
BF16 operation, so the two agree to BF16 rounding, not bit for bit.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from stage3_moe import coat_grouped_gemm as gg

BLOCK_M = gg.BLOCK_M
_WEIGHTS = {}


@triton.jit
def _gemm_kernel(A, ROW, B, C, tile_expert, N, K, stride_am, stride_bg, stride_b, stride_cm,
                 INDEXED: tl.constexpr, B_TRANSPOSED: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """C[m] = A[m'] @ W for the expert g of row tile m; m' = ROW[m] if INDEXED else m.
    W = B[g].T with B [G, N, K], or B[g] with B [G, K, N] when B_TRANSPOSED."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.maximum(tl.load(tile_expert + pid_m), 0).to(tl.int64)  # tail tiles hold only zero rows
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    rows = rm
    if INDEXED:
        rows = tl.load(ROW + rm)
    a_ptr = A + rows[:, None] * stride_am + rk[None, :]
    if B_TRANSPOSED:
        b_ptr = B + g * stride_bg + rk[:, None] * stride_b + rn[None, :]
        b_step = BLOCK_K * stride_b
    else:
        b_ptr = B + g * stride_bg + rn[None, :] * stride_b + rk[:, None]
        b_step = BLOCK_K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptr, mask=rk[None, :] < k_left, other=0.0)
        b = tl.load(b_ptr, mask=(rk[:, None] < k_left) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptr += BLOCK_K
        b_ptr += b_step
    tl.store(C + rm[:, None] * stride_cm + rn[None, :], acc.to(C.dtype.element_ty), mask=rn[None, :] < N)


def _gemm(a, row_token, weight, tile_expert, n, transposed=False):
    """[rows, n] = a[rows] @ weight[expert]; weight is [G, n, k], or [G, k, n] when transposed."""
    rows = tile_expert.numel() * BLOCK_M
    k = a.shape[1]
    out = torch.empty(rows, n, dtype=torch.bfloat16, device=a.device)
    assert a.stride(1) == 1 and weight.stride(2) == 1 and weight.shape[1:] == ((k, n) if transposed else (n, k))
    _gemm_kernel[(rows // BLOCK_M, triton.cdiv(n, 128))](
        a, row_token, weight, out, tile_expert, n, k, a.stride(0), weight.stride(0), weight.stride(1), out.stride(0),
        INDEXED=row_token is not None, B_TRANSPOSED=transposed, BLOCK_M=BLOCK_M, BLOCK_N=128, BLOCK_K=128,
        num_warps=8, num_stages=3)
    return out


@triton.jit
def _act_forward_kernel(H, P, OUT, stride_h, stride_o, F: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_F: tl.constexpr):
    """OUT[r] = p[r] * silu(gate) * up with H[r] = [gate | up], one BLOCK_F-column tile per program."""
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    f = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
    gate = tl.load(H + r[:, None] * stride_h + f[None, :]).to(tl.float32)
    up = tl.load(H + r[:, None] * stride_h + F + f[None, :]).to(tl.float32)
    p = tl.load(P + r)
    out = gate / (1.0 + libdevice.exp(-gate)) * up * p[:, None]
    tl.store(OUT + r[:, None] * stride_o + f[None, :], out.to(OUT.dtype.element_ty))


@triton.jit
def _act_backward_kernel(H, P, DW, DH, DP, stride_h, stride_dw, stride_dh, stride_dp,
                         F: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_F: tl.constexpr):
    """Gradients of OUT = p * silu(gate) * up: DH[r] = [d gate | d up], DP[r, tile] = this tile's part of d p."""
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    tile = tl.program_id(1)
    f = tile * BLOCK_F + tl.arange(0, BLOCK_F)
    gate = tl.load(H + r[:, None] * stride_h + f[None, :]).to(tl.float32)
    up = tl.load(H + r[:, None] * stride_h + F + f[None, :]).to(tl.float32)
    d = tl.load(DW + r[:, None] * stride_dw + f[None, :]).to(tl.float32)
    p = tl.load(P + r)
    sigmoid = 1 / (1.0 + libdevice.exp(-gate))
    silu = gate * sigmoid
    d_gate = d * p[:, None] * up * (sigmoid + silu * (1 - sigmoid))
    d_up = d * p[:, None] * silu
    tl.store(DH + r[:, None] * stride_dh + f[None, :], d_gate.to(DH.dtype.element_ty))
    tl.store(DH + r[:, None] * stride_dh + F + f[None, :], d_up.to(DH.dtype.element_ty))
    tl.store(DP + r * stride_dp + tile, tl.sum(d * silu * up, axis=1))


def _act_forward(h, probs, ffn):
    rows = h.shape[0]
    block_f = min(ffn, 256)
    assert ffn % block_f == 0 and rows % 32 == 0
    out = torch.empty(rows, ffn, dtype=torch.bfloat16, device=h.device)
    _act_forward_kernel[(rows // 32, ffn // block_f)](h, probs, out, h.stride(0), out.stride(0), F=ffn, BLOCK_R=32, BLOCK_F=block_f)
    return out


def _act_backward(h, probs, d_weighted, ffn):
    rows = h.shape[0]
    block_f = min(ffn, 256)
    d_h = torch.empty(rows, 2 * ffn, dtype=torch.bfloat16, device=h.device)
    partial = torch.empty(rows, ffn // block_f, dtype=torch.float32, device=h.device)
    _act_backward_kernel[(rows // 32, ffn // block_f)](h, probs, d_weighted, d_h, partial, h.stride(0), d_weighted.stride(0),
                                                       d_h.stride(0), partial.stride(0), F=ffn, BLOCK_R=32, BLOCK_F=block_f)
    return d_h, partial.sum(1)


@triton.jit
def _wgrad_kernel(G, X, ROW, D, seg_start, seg_rows, N, K, stride_g, stride_x, stride_dg, stride_dn,
                  INDEX_G: tl.constexpr, INDEX_X: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """D[g, n, k] = sum_{m in expert g} G[m', n] * X[m'', k]; an indexed operand is read through ROW."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    g = tl.program_id(2)
    start = tl.multiple_of(tl.load(seg_start + g), BLOCK_M)
    rows = tl.load(seg_rows + g)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rm = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for chunk in range(0, rows // BLOCK_M):
        r = start + chunk * BLOCK_M + rm
        g_rows = r
        x_rows = r
        if INDEX_G:
            g_rows = tl.load(ROW + r)
        if INDEX_X:
            x_rows = tl.load(ROW + r)
        a = tl.load(G + g_rows[:, None] * stride_g + rn[None, :], mask=rn[None, :] < N, other=0.0)
        b = tl.load(X + x_rows[:, None] * stride_x + rk[None, :], mask=rk[None, :] < K, other=0.0)
        acc = tl.dot(tl.trans(a), b, acc)
    d_ptr = D + g.to(tl.int64) * stride_dg + rn[:, None] * stride_dn + rk[None, :]
    tl.store(d_ptr, acc.to(D.dtype.element_ty), mask=(rn[:, None] < N) & (rk[None, :] < K))


def _wgrad(g, x, row_token, seg_start, seg_rows, out, index_g=False, index_x=False):
    experts, n, k = out.shape
    assert g.stride(1) == 1 and x.stride(1) == 1 and g.shape[1] == n and x.shape[1] == k
    _wgrad_kernel[(triton.cdiv(n, 128), triton.cdiv(k, 128), experts)](
        g, x, row_token, out, seg_start, seg_rows, n, k, g.stride(0), x.stride(0), out.stride(0), out.stride(1),
        INDEX_G=index_g, INDEX_X=index_x, BLOCK_M=BLOCK_M, BLOCK_N=128, BLOCK_K=128, num_warps=8, num_stages=3)
    return out


def _with_zero_row(x):
    return torch.cat([x, torch.zeros(1, x.shape[1], dtype=x.dtype, device=x.device)])


def _stacked(fc1, fc2):
    """The expert weights as [G, 2F, H] and [G, H, F]; stacked once per optimizer step."""
    key = (fc1[0].data_ptr(), fc2[0].data_ptr())
    entry = _WEIGHTS.get(key)
    if entry is None:
        with torch.no_grad():
            entry = _WEIGHTS[key] = (torch.stack([w.detach() for w in fc1]), torch.stack([w.detach() for w in fc2]))
    return entry


class FusedExpertsBF16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, probs, routing_map, topk, *weights):
        experts = routing_map.shape[1]
        fc1, fc2 = weights[:experts], weights[experts:]
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        tokens, hidden = x.shape
        ffn = fc1[0].shape[0] // 2
        device = x.device
        counts = routing_map.sum(dim=0)
        padded = (counts + BLOCK_M - 1) // BLOCK_M * BLOCK_M
        starts = torch.cumsum(padded, 0) - padded
        rows = tokens * topk + experts * BLOCK_M
        expert_idx, token_idx = routing_map.t().nonzero(as_tuple=True)  # MCore's permute order
        rank = torch.arange(token_idx.numel(), device=device) - (torch.cumsum(counts, 0) - counts)[expert_idx]
        dest = starts[expert_idx] + rank
        if token_idx.numel() != tokens * topk:
            raise RuntimeError("routing map does not carry topk experts per token")
        # Row -> token; padding rows point at an appended zero row.
        row_token = torch.full((rows,), tokens, dtype=torch.long, device=device)
        row_token[dest] = token_idx
        row_flat = torch.full((rows,), tokens * experts, dtype=torch.long, device=device)
        row_flat[dest] = token_idx * experts + expert_idx
        token_rows = dest[torch.argsort(token_idx, stable=True)].view(tokens, topk)
        perm_probs = torch.cat([probs.reshape(-1).float(), torch.zeros(1, device=device)])[row_flat]
        tiles = gg.tile_experts(starts, padded, rows)
        w1, w2 = _stacked(fc1, fc2)

        x0 = _with_zero_row(x)
        h = _gemm(x0, row_token, w1, tiles, 2 * ffn)
        weighted = _act_forward(h, perm_probs, ffn)
        out = _gemm(weighted, None, w2, tiles, hidden)
        combined = gg.gather_sum(out, token_rows, torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device))
        ctx.save_for_backward(x0, h, weighted, perm_probs, row_token, row_flat, token_rows, starts, padded, tiles, *weights)
        ctx.meta = (experts, tokens, hidden, rows, ffn, shape, probs.dtype)
        return combined.view(shape)

    @staticmethod
    def backward(ctx, grad):
        x0, h, weighted, perm_probs, row_token, row_flat, token_rows, starts, padded, tiles, *weights = ctx.saved_tensors
        experts, tokens, hidden, rows, ffn, shape, probs_dtype = ctx.meta
        fc1, fc2 = weights[:experts], weights[experts:]
        device = grad.device
        w1, w2 = _stacked(fc1, fc2)
        g0 = _with_zero_row(grad.reshape(tokens, hidden).to(torch.bfloat16))
        d_weighted = _gemm(g0, row_token, w2, tiles, ffn, transposed=True)
        d_h, d_perm_probs = _act_backward(h, perm_probs, d_weighted, ffn)
        probs_grad = torch.zeros(tokens * experts + 1, dtype=torch.float32, device=device)
        probs_grad.index_put_((row_flat,), d_perm_probs)  # every real (token, expert) appears once
        probs_grad = probs_grad[:-1].view(tokens, experts).to(probs_dtype)
        d_x = _gemm(d_h, None, w1, tiles, hidden, transposed=True)
        d_in = gg.gather_sum(d_x, token_rows, torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device))
        fc2_grad = torch.empty(experts, hidden, ffn, dtype=fc2[0].dtype, device=device)
        _wgrad(g0, weighted, row_token, starts, padded, fc2_grad, index_g=True)
        fc1_grad = torch.empty(experts, 2 * ffn, hidden, dtype=fc1[0].dtype, device=device)
        _wgrad(d_h, x0, row_token, starts, padded, fc1_grad, index_x=True)
        return (d_in.view(shape), probs_grad, None, None, *fc1_grad.unbind(0), *fc2_grad.unbind(0))


def install(models, optimizer):
    """Route the training forward of every MoE layer's routed experts through FusedExpertsBF16."""
    from megatron.core.transformer.moe.moe_layer import MoELayer

    count = 0
    for model in models:
        for moe in (m for m in model.modules() if isinstance(m, MoELayer)):
            config = moe.config
            if not (config.gated_linear_unit and not config.add_bias_linear and not config.moe_apply_probs_on_input
                    and config.expert_model_parallel_size == 1 and not config.fp8):
                raise RuntimeError("BF16 fused experts: expected bias-free SwiGLU experts, probabilities after the activation, EP=1, no TE FP8")
            _patch(moe)
            count += 1
    step = optimizer.step

    def step_and_restack(*args, **kwargs):
        try:
            return step(*args, **kwargs)
        finally:
            _WEIGHTS.clear()

    optimizer.step = step_and_restack
    print(f"BF16_FUSED_EXPERTS_INSTALLED moe={count}", flush=True)


def _patch(moe):
    original = moe.forward
    experts = moe.experts
    count = experts.num_local_experts
    fc1 = [getattr(experts.linear_fc1, f"weight{i}") for i in range(count)]
    fc2 = [getattr(experts.linear_fc2, f"weight{i}") for i in range(count)]
    shared = moe.shared_experts
    topk = moe.config.moe_router_topk

    def forward(hidden_states, intermediate_tensors=None, padding_mask=None):
        if not moe.training:
            return original(hidden_states, intermediate_tensors, padding_mask=padding_mask)
        if intermediate_tensors is not None:
            raise RuntimeError("BF16 fused experts: partial MoE execution is not supported")
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()
        probs, routing_map = moe.route(hidden_states, padding_mask)
        routed = FusedExpertsBF16.apply(hidden_states, probs, routing_map, topk, *fc1, *fc2)
        return (routed + shared(hidden_states) if shared is not None else routed), None

    moe.forward = forward
