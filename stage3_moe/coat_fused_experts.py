"""Routed experts of the COAT flow without permuted copies (STAGE3_MOE_COAT_FUSED_EXPERTS=1).

The arithmetic is coat_activation.GroupedExperts' own, operation for operation: the same FP8
operands, scales, accumulation order and rounding points. What changes is how often memory is
traversed. Top-k routing replicates every token topk times, so each extra pass over a permuted
tensor costs topk times what it costs in a dense layer:

  - the permuted FP8 input, its transpose, the permuted gradient and its two FP8 copies are never
    built: GEMMs read token rows through the row -> token index, and the per-tensor gradient
    quantization runs on the unpermuted gradient (same scale, same codes);
  - gate and up are one GEMM whose epilogue quantizes 1x16, as COAT's own GEMM does, instead of
    two FP32 results quantized by a second kernel;
  - SiLU, the product, the router probabilities and the maximum the next per-tensor scale needs
    are one kernel;
  - the down-projection data gradient applies the probabilities in its epilogue, and the two
    fc1 data gradients are one launch;
  - expert weights are quantized by one launch per projection instead of three per expert;
  - the weight-gradient kernel is told that segments start on tile boundaries, which lets Triton
    load its operands with pipelined cp.async instead of one byte at a time.
The elementwise backward kernels (mul, SiLU) and the per-tensor divisions are upstream's.
"""
import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from stage3_moe import coat_activation as ca
from stage3_moe import coat_grouped_gemm as gg

rq, QB = ca.rq, ca.QB
THRES = ca._common.SCALE_MIN_THRES
E4M3_MAX = ca._common.FP8_MAX_VALUE[torch.float8_e4m3fn]
BLOCK_M = gg.BLOCK_M
# Where the weight gradients read their operands from. The sums are the same in every mode;
# measured 2026-10-02 per layer and micro-batch (26,624 tokens), whole block forward+backward:
#   transposed  6.3 ms  both from transposed copies; the token operand's permuted transpose is built first
#   hybrid      7.1 ms  the permuted operand from upstream's transposed copy, the token operand row-major
#   rowmajor    8.2 ms  both operands row-major and transposed in registers: no transposed copy exists
#   reference  10.1 ms  as transposed, with the grouped path's own kernel (the check for the others)
WGRAD = os.environ.get("STAGE3_MOE_COAT_FUSED_WGRAD", "transposed")


@triton.jit
def _stacked_quantize_kernel(W, S, Q, QT, N, K, stride_wg, stride_wn, stride_qg, stride_qn, stride_tg, stride_tk,
                             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Q[g] = W[g] / S[g] in E4M3 and its transpose: fp8_division_transpose for every expert at once."""
    g = tl.program_id(0).to(tl.int64)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.program_id(2) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (rn[:, None] < N) & (rk[None, :] < K)
    w = tl.load(W + g * stride_wg + rn[:, None] * stride_wn + rk[None, :], mask=mask, other=0.0).to(tl.float32)
    q = tl.fdiv(w, tl.load(S + g).to(tl.float32)).to(Q.dtype.element_ty)
    tl.store(Q + g * stride_qg + rn[:, None] * stride_qn + rk[None, :], q, mask=mask)
    tl.store(QT + g * stride_tg + rk[None, :] * stride_tk + rn[:, None], q, mask=mask)


def _stacked_weights(fc1, fc2):
    """Per-expert COAT weight quantization (as ca.fp8_weight), stacked; one launch per projection."""
    key = ("stacked-fused", fc1[0].data_ptr(), fc2[0].data_ptr(), len(fc1))
    entry = ca._WEIGHT_CACHE.get(key)
    if entry is None:
        entry = {}
        with torch.no_grad():
            w1 = torch.stack([w.detach() for w in fc1])
            w2 = torch.stack([w.detach() for w in fc2])
            ffn = w1.shape[1] // 2
            for name, w in (("gate", w1[:, :ffn]), ("up", w1[:, ffn:]), ("down", w2)):
                experts, n, k = w.shape
                scale = (w.abs().amax(dim=(1, 2)) + THRES) / E4M3_MAX  # fp8_division_transpose's own scale
                q = torch.empty(experts, n, k, dtype=torch.float8_e4m3fn, device=w.device)
                q_t = torch.empty(experts, k, n, dtype=torch.float8_e4m3fn, device=w.device)
                _stacked_quantize_kernel[(experts, triton.cdiv(n, 64), triton.cdiv(k, 64))](
                    w, scale, q, q_t, n, k, w.stride(0), w.stride(1), q.stride(0), q.stride(1), q_t.stride(0), q_t.stride(1),
                    BLOCK_N=64, BLOCK_K=64)
                entry[name], entry[name + "_t"], entry[name + "_s"] = q, q_t, scale.float()
            entry["fc1"] = torch.cat([entry["gate"].view(torch.uint8), entry["up"].view(torch.uint8)], 1).view(torch.float8_e4m3fn)
            tiles = ffn // 128
            entry["fc1_s"] = torch.cat([entry["gate_s"][:, None].expand(-1, tiles), entry["up_s"][:, None].expand(-1, tiles)], 1).contiguous()
        ca._WEIGHT_CACHE[key] = entry
    return entry


@triton.jit
def _fc1_kernel(A, ROW, B, C, SC, tile_expert, scale_a, scale_b, K, fp8_max,
                stride_am, stride_bg, stride_bn, stride_cm, stride_scm,
                N_TILES: tl.constexpr, QB: tl.constexpr, SCALE_MIN_THRES: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """C[m] = quantize_1x16(scale_a * scale_b[g, tile] * A[ROW[m]] @ B[g].T) for the rows m of expert g.
    Rows past the last expert point at a zero row of A, so they quantize like the grouped path's zeros."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.maximum(tl.load(tile_expert + pid_m), 0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    token = tl.load(ROW + rm)
    a_ptr = A + token[:, None] * stride_am + rk[None, :]
    b_ptr = B + g.to(tl.int64) * stride_bg + rn[None, :] * stride_bn + rk[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptr, mask=rk[None, :] < k_left, other=0.0)
        b = tl.load(b_ptr, mask=rk[:, None] < k_left, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptr += BLOCK_K
        b_ptr += BLOCK_K
    acc = acc * (tl.load(scale_a) * tl.load(scale_b + g * N_TILES + pid_n))
    # fp8_quantize of the FP32 result (the epilogue of COAT's fp8matmul with output_quantize).
    acc = tl.reshape(acc, (BLOCK_M, BLOCK_N // QB, QB))
    max_val = tl.max(tl.abs(acc), axis=2) + SCALE_MIN_THRES
    scale_c = max_val / fp8_max
    acc = tl.fdiv(acc, tl.reshape(scale_c, (BLOCK_M, BLOCK_N // QB, 1)))
    acc = tl.reshape(acc, (BLOCK_M, BLOCK_N)).to(C.dtype.element_ty)
    tl.store(C + rm[:, None] * stride_cm + rn[None, :], acc)
    rs = pid_n * (BLOCK_N // QB) + tl.arange(0, BLOCK_N // QB)
    tl.store(SC + rm[:, None] * stride_scm + rs[None, :], scale_c.to(SC.dtype.element_ty))


def _fc1(x0, row_token, weight, scale_x, scale_w, tile_expert, out, out_scale):
    rows, n = out.shape
    k = x0.shape[1]
    assert rows % BLOCK_M == 0 and n % 128 == 0 and weight.stride(2) == 1 and scale_w.shape[1] == n // 128
    _fc1_kernel[(rows // BLOCK_M, n // 128)](
        x0, row_token, weight, out, out_scale, tile_expert, scale_x, scale_w, k, E4M3_MAX,
        x0.stride(0), weight.stride(0), weight.stride(1), out.stride(0), out_scale.stride(0),
        N_TILES=n // 128, QB=QB, SCALE_MIN_THRES=THRES, BLOCK_M=BLOCK_M, BLOCK_N=128, BLOCK_K=128,
        num_warps=8, num_stages=3)


@triton.jit
def _act_kernel(HX, HS, P, SILU, SILU_S, WEIGHTED, ROWMAX, fp8_max,
                stride_hx, stride_hs, stride_silu, stride_silus, stride_w,
                F: tl.constexpr, QB: tl.constexpr, SCALE_MIN_THRES: tl.constexpr, BLOCK_R: tl.constexpr):
    """fp8_silu_forward, the product of fp8_mul_forward and the router probabilities in one pass:
    SILU = quantize_1x16(silu(gate)); WEIGHTED = bf16(bf16(SILU * up) * p); ROWMAX = max |WEIGHTED| per row."""
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    f = tl.arange(0, F)
    sf = tl.arange(0, F // QB)
    gate = tl.load(HX + r[:, None] * stride_hx + f[None, :]).to(tl.float32)
    up = tl.load(HX + r[:, None] * stride_hx + F + f[None, :]).to(tl.float32)
    gate_s = tl.load(HS + r[:, None] * stride_hs + sf[None, :]).to(tl.float32)
    up_s = tl.load(HS + r[:, None] * stride_hs + F // QB + sf[None, :]).to(tl.float32)
    gate = tl.reshape(gate, (BLOCK_R, F // QB, QB)) * tl.reshape(gate_s, (BLOCK_R, F // QB, 1))
    up = tl.reshape(up, (BLOCK_R, F // QB, QB)) * tl.reshape(up_s, (BLOCK_R, F // QB, 1))

    sigmoid = 1 / (1.0 + libdevice.exp(-gate))
    silu = sigmoid * gate
    max_val = tl.max(tl.abs(silu), axis=2) + SCALE_MIN_THRES
    silu_scale = max_val / fp8_max
    silu_q = tl.fdiv(silu, tl.reshape(silu_scale, (BLOCK_R, F // QB, 1))).to(SILU.dtype.element_ty)
    silu_scale = silu_scale.to(SILU_S.dtype.element_ty)
    tl.store(SILU + r[:, None] * stride_silu + f[None, :], tl.reshape(silu_q, (BLOCK_R, F)))
    tl.store(SILU_S + r[:, None] * stride_silus + sf[None, :], silu_scale)

    # The product reads SiLU back the way fp8_mul_forward does: the FP8 codes times the BF16 scale.
    silu = silu_q.to(tl.float32) * tl.reshape(silu_scale.to(tl.float32), (BLOCK_R, F // QB, 1))
    product = tl.reshape((silu * up).to(tl.bfloat16), (BLOCK_R, F))
    p = tl.load(P + r)
    weighted = (product.to(tl.float32) * p[:, None]).to(tl.bfloat16)
    tl.store(WEIGHTED + r[:, None] * stride_w + f[None, :], weighted)
    tl.store(ROWMAX + r, tl.max(tl.abs(weighted.to(tl.float32)), axis=1))


@triton.jit
def _act_wide_kernel(HX, HS, P, SILU, SILU_S, WEIGHTED, TILEMAX, fp8_max,
                     stride_hx, stride_hs, stride_silu, stride_silus, stride_w, stride_tm,
                     F: tl.constexpr, QB: tl.constexpr, SCALE_MIN_THRES: tl.constexpr,
                     BLOCK_R: tl.constexpr, BLOCK_F: tl.constexpr):
    """_act_kernel for experts wider than one tile: the same arithmetic per BLOCK_F columns, with the
    maximum of |WEIGHTED| written per (row, tile)."""
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    pid_f = tl.program_id(1)
    f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
    sf = pid_f * (BLOCK_F // QB) + tl.arange(0, BLOCK_F // QB)
    gate = tl.load(HX + r[:, None] * stride_hx + f[None, :]).to(tl.float32)
    up = tl.load(HX + r[:, None] * stride_hx + F + f[None, :]).to(tl.float32)
    gate_s = tl.load(HS + r[:, None] * stride_hs + sf[None, :]).to(tl.float32)
    up_s = tl.load(HS + r[:, None] * stride_hs + F // QB + sf[None, :]).to(tl.float32)
    gate = tl.reshape(gate, (BLOCK_R, BLOCK_F // QB, QB)) * tl.reshape(gate_s, (BLOCK_R, BLOCK_F // QB, 1))
    up = tl.reshape(up, (BLOCK_R, BLOCK_F // QB, QB)) * tl.reshape(up_s, (BLOCK_R, BLOCK_F // QB, 1))

    sigmoid = 1 / (1.0 + libdevice.exp(-gate))
    silu = sigmoid * gate
    max_val = tl.max(tl.abs(silu), axis=2) + SCALE_MIN_THRES
    silu_scale = max_val / fp8_max
    silu_q = tl.fdiv(silu, tl.reshape(silu_scale, (BLOCK_R, BLOCK_F // QB, 1))).to(SILU.dtype.element_ty)
    silu_scale = silu_scale.to(SILU_S.dtype.element_ty)
    tl.store(SILU + r[:, None] * stride_silu + f[None, :], tl.reshape(silu_q, (BLOCK_R, BLOCK_F)))
    tl.store(SILU_S + r[:, None] * stride_silus + sf[None, :], silu_scale)

    silu = silu_q.to(tl.float32) * tl.reshape(silu_scale.to(tl.float32), (BLOCK_R, BLOCK_F // QB, 1))
    product = tl.reshape((silu * up).to(tl.bfloat16), (BLOCK_R, BLOCK_F))
    p = tl.load(P + r)
    weighted = (product.to(tl.float32) * p[:, None]).to(tl.bfloat16)
    tl.store(WEIGHTED + r[:, None] * stride_w + f[None, :], weighted)
    tl.store(TILEMAX + r * stride_tm + pid_f, tl.max(tl.abs(weighted.to(tl.float32)), axis=1))


def _act(h_x, h_s, probs):
    rows, ffn = h_x.shape[0], h_x.shape[1] // 2
    device = h_x.device
    silu_x = torch.empty(rows, ffn, dtype=h_x.dtype, device=device)
    silu_s = torch.empty(rows, ffn // QB, dtype=torch.bfloat16, device=device)
    weighted = torch.empty(rows, ffn, dtype=torch.bfloat16, device=device)
    assert rows % 32 == 0
    if ffn > 256:
        assert ffn % 256 == 0
        row_max = torch.empty(rows, ffn // 256, dtype=torch.float32, device=device)
        _act_wide_kernel[(rows // 32, ffn // 256)](
            h_x, h_s, probs, silu_x, silu_s, weighted, row_max, E4M3_MAX,
            h_x.stride(0), h_s.stride(0), silu_x.stride(0), silu_s.stride(0), weighted.stride(0), row_max.stride(0),
            F=ffn, QB=QB, SCALE_MIN_THRES=THRES, BLOCK_R=32, BLOCK_F=256, num_warps=4)
        return silu_x, silu_s, weighted, row_max
    row_max = torch.empty(rows, dtype=torch.float32, device=device)
    _act_kernel[(rows // 32,)](h_x, h_s, probs, silu_x, silu_s, weighted, row_max, E4M3_MAX,
                               h_x.stride(0), h_s.stride(0), silu_x.stride(0), silu_s.stride(0), weighted.stride(0),
                               F=ffn, QB=QB, SCALE_MIN_THRES=THRES, BLOCK_R=32, num_warps=4)
    return silu_x, silu_s, weighted, row_max


@triton.jit
def _down_dgrad_kernel(A, ROW, B, MULX, P, DP, DPROD, tile_expert, scale_a, scale_b, mul_scale, K,
                       stride_am, stride_bg, stride_bn, stride_mx, stride_dp,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """d = bf16(scale * A[ROW[m]] @ B[g].T), the gradient of p*h; then, as gg.probs_backward:
    DP[m] = <d, p*h> / p from the FP8 copy of p*h, and DPROD[m] = bf16(d * p)."""
    pid_m = tl.program_id(0)
    g = tl.maximum(tl.load(tile_expert + pid_m), 0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    token = tl.load(ROW + rm)
    a_ptr = A + token[:, None] * stride_am + rk[None, :]
    b_ptr = B + g.to(tl.int64) * stride_bg + rn[None, :] * stride_bn + rk[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptr, mask=rk[None, :] < k_left, other=0.0)
        b = tl.load(b_ptr, mask=rk[:, None] < k_left, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptr += BLOCK_K
        b_ptr += BLOCK_K
    acc = acc * (tl.load(scale_a) * tl.load(scale_b + g))
    d = acc.to(tl.bfloat16).to(tl.float32)
    weighted = tl.load(MULX + rm[:, None] * stride_mx + rn[None, :]).to(tl.float32)
    weighted = weighted * tl.load(mul_scale)
    p = tl.load(P + rm)
    safe = tl.where(p > 0, p, 1.0)
    tl.store(DP + rm, tl.sum(d * weighted, axis=1) / safe)
    tl.store(DPROD + rm[:, None] * stride_dp + rn[None, :], (d * p[:, None]).to(DPROD.dtype.element_ty))


@triton.jit
def _down_dgrad_wide_kernel(A, ROW, B, MULX, P, DP, DPROD, tile_expert, scale_a, scale_b, mul_scale, K,
                            stride_am, stride_bg, stride_bn, stride_mx, stride_dp, stride_dpt,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """_down_dgrad_kernel for experts wider than one tile; DP holds one partial sum per (row, tile)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.maximum(tl.load(tile_expert + pid_m), 0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    token = tl.load(ROW + rm)
    a_ptr = A + token[:, None] * stride_am + rk[None, :]
    b_ptr = B + g.to(tl.int64) * stride_bg + rn[None, :] * stride_bn + rk[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptr, mask=rk[None, :] < k_left, other=0.0)
        b = tl.load(b_ptr, mask=rk[:, None] < k_left, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptr += BLOCK_K
        b_ptr += BLOCK_K
    acc = acc * (tl.load(scale_a) * tl.load(scale_b + g))
    d = acc.to(tl.bfloat16).to(tl.float32)
    weighted = tl.load(MULX + rm[:, None] * stride_mx + rn[None, :]).to(tl.float32)
    weighted = weighted * tl.load(mul_scale)
    p = tl.load(P + rm)
    safe = tl.where(p > 0, p, 1.0)
    tl.store(DP + rm * stride_dpt + pid_n, tl.sum(d * weighted, axis=1) / safe)
    tl.store(DPROD + rm[:, None] * stride_dp + rn[None, :], (d * p[:, None]).to(DPROD.dtype.element_ty))


def _down_dgrad(g0, row_token, weight_t, scale_g, scale_w, tile_expert, mul_x, mul_scale, probs, d_probs, d_product):
    rows, ffn = d_product.shape
    assert rows % BLOCK_M == 0 and weight_t.stride(2) == 1 and mul_x.stride(1) == 1
    if ffn > 256:
        assert ffn % 256 == 0
        partial = torch.empty(rows, ffn // 256, dtype=torch.float32, device=d_probs.device)
        _down_dgrad_wide_kernel[(rows // BLOCK_M, ffn // 256)](
            g0, row_token, weight_t, mul_x, probs, partial, d_product, tile_expert, scale_g, scale_w, mul_scale, g0.shape[1],
            g0.stride(0), weight_t.stride(0), weight_t.stride(1), mul_x.stride(0), d_product.stride(0), partial.stride(0),
            BLOCK_M=BLOCK_M, BLOCK_N=256, BLOCK_K=128, num_warps=8, num_stages=3)
        d_probs.copy_(partial.sum(1))
        return
    assert ffn in (128, 256)
    _down_dgrad_kernel[(rows // BLOCK_M,)](
        g0, row_token, weight_t, mul_x, probs, d_probs, d_product, tile_expert, scale_g, scale_w, mul_scale, g0.shape[1],
        g0.stride(0), weight_t.stride(0), weight_t.stride(1), mul_x.stride(0), d_product.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=ffn, BLOCK_K=128, num_warps=8, num_stages=3)


@triton.jit
def _fc1_dgrad_kernel(A1, A2, B1, B2, C, tile_expert, scale_a1, scale_a2, scale_b1, scale_b2, N, K,
                      stride_a1, stride_a2, stride_bg, stride_bn, stride_cm,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """C = bf16(bf16(s1 * A1 @ B1[g].T) + s2 * A2 @ B2[g].T): the gate and up data gradients,
    rounded where two accumulating launches round."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.maximum(tl.load(tile_expert + pid_m), 0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    b_offset = g.to(tl.int64) * stride_bg + rn[None, :] * stride_bn + rk[:, None]
    b_mask = rn[None, :] < N

    a_ptr = A1 + rm[:, None] * stride_a1 + rk[None, :]
    b_ptr = B1 + b_offset
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptr, mask=rk[None, :] < k_left, other=0.0)
        b = tl.load(b_ptr, mask=(rk[:, None] < k_left) & b_mask, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptr += BLOCK_K
        b_ptr += BLOCK_K
    first = (acc * (tl.load(scale_a1) * tl.load(scale_b1 + g))).to(tl.bfloat16)

    a_ptr = A2 + rm[:, None] * stride_a2 + rk[None, :]
    b_ptr = B2 + b_offset
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptr, mask=rk[None, :] < k_left, other=0.0)
        b = tl.load(b_ptr, mask=(rk[:, None] < k_left) & b_mask, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptr += BLOCK_K
        b_ptr += BLOCK_K
    acc = acc * (tl.load(scale_a2) * tl.load(scale_b2 + g))
    acc += first.to(tl.float32)
    tl.store(C + rm[:, None] * stride_cm + rn[None, :], acc.to(C.dtype.element_ty), mask=b_mask)


def _fc1_dgrad(a1, a2, b1, b2, scale_a1, scale_a2, scale_b1, scale_b2, tile_expert, out):
    rows, k = a1.shape
    n = b1.shape[1]
    assert rows % BLOCK_M == 0 and a1.stride(1) == 1 and a2.stride(1) == 1 and b1.stride(2) == 1 and b1.stride() == b2.stride()
    _fc1_dgrad_kernel[(rows // BLOCK_M, triton.cdiv(n, 128))](
        a1, a2, b1, b2, out, tile_expert, scale_a1, scale_a2, scale_b1, scale_b2, n, k,
        a1.stride(0), a2.stride(0), b1.stride(0), b1.stride(1), out.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=128, BLOCK_K=128, num_warps=8, num_stages=3)


@triton.jit
def _wgrad_kernel(G, X, ROW, D, seg_start, seg_rows, scale_a, scale_b, N, K,
                  stride_g, stride_x, stride_dg, stride_dn,
                  G_LAYOUT: tl.constexpr, X_LAYOUT: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """D[g, n, k] = scale * sum_{m in expert g} G[m, n] * X[m, k], summed over the same 128-row
    chunks, in the same order, as gg._segment_wgrad_kernel. Operand layouts:
      0  [features, rows]: contiguous along the reduction, as FP8 MMA wants it; loaded by cp.async
      1  [rows, features]: loaded row-major and transposed in registers
      2  [tokens + 1, features] read through ROW (row -> token, padding rows -> the zero last row)."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    g = tl.program_id(2)
    start = tl.multiple_of(tl.load(seg_start + g), BLOCK_M)  # segments start on tile boundaries
    rows = tl.load(seg_rows + g)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rm = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for chunk in range(0, rows // BLOCK_M):
        r = start + chunk * BLOCK_M + rm
        g_rows = r
        x_rows = r
        if G_LAYOUT == 2:
            g_rows = tl.load(ROW + r)
        if X_LAYOUT == 2:
            x_rows = tl.load(ROW + r)
        if G_LAYOUT == 0:
            a = tl.load(G + rn[:, None] * stride_g + r[None, :], mask=rn[:, None] < N, other=0.0)
        else:
            a = tl.trans(tl.load(G + g_rows[:, None] * stride_g + rn[None, :], mask=rn[None, :] < N, other=0.0))
        if X_LAYOUT == 0:
            b = tl.load(X + rk[None, :] * stride_x + r[:, None], mask=rk[None, :] < K, other=0.0)
        else:
            b = tl.load(X + x_rows[:, None] * stride_x + rk[None, :], mask=rk[None, :] < K, other=0.0)
        acc = tl.dot(a, b, acc)
    acc = acc * (tl.load(scale_a) * tl.load(scale_b))
    d_ptr = D + g.to(tl.int64) * stride_dg + rn[:, None] * stride_dn + rk[None, :]
    tl.store(d_ptr, acc.to(D.dtype.element_ty), mask=(rn[:, None] < N) & (rk[None, :] < K))


def _wgrad(g, x, row_token, scale_a, scale_b, seg_start, seg_rows, out, g_layout, x_layout):
    """out[e] = g[rows of e].T @ x[rows of e]; see _wgrad_kernel for the operand layouts."""
    experts, n, k = out.shape
    assert g.stride(1) == 1 and x.stride(1) == 1 and out.stride(2) == 1
    assert g.shape[0 if g_layout == 0 else 1] == n and x.shape[0 if x_layout == 0 else 1] == k
    # 128 x 128 output tiles: 256 x 256 measured 1.3-1.9x slower in every layout.
    _wgrad_kernel[(triton.cdiv(n, 128), triton.cdiv(k, 128), experts)](
        g, x, row_token, out, seg_start, seg_rows, scale_a, scale_b, n, k,
        g.stride(0), x.stride(0), out.stride(0), out.stride(1),
        G_LAYOUT=g_layout, X_LAYOUT=x_layout, BLOCK_M=BLOCK_M, BLOCK_N=128, BLOCK_K=128, num_warps=8, num_stages=3)
    return out


def _permuted_transpose(x_t, row_token):
    """[features, tokens] -> [features, rows]: the transposed copy the grouped path's weight gradient reads."""
    return _with_zero_column(x_t).view(torch.uint8).index_select(1, row_token).view(x_t.dtype)


def _with_zero_row(x):
    return torch.cat([x.view(torch.uint8), torch.zeros(1, x.shape[1], dtype=torch.uint8, device=x.device)]).view(x.dtype)


def _with_zero_column(x):
    return torch.cat([x.view(torch.uint8), torch.zeros(x.shape[0], 1, dtype=torch.uint8, device=x.device)], 1).view(x.dtype)


class FusedExperts(torch.autograd.Function):
    """coat_activation.GroupedExperts with the same values and fewer passes over memory."""

    @staticmethod
    def forward(ctx, anchor, probs, ln_x, ln_s, ln_t, routing_map, topk, *weights):
        experts = routing_map.shape[1]
        fc1, fc2 = weights[:experts], weights[experts:]
        tokens, hidden = ln_x.shape
        ffn = fc1[0].shape[0] // 2
        device = ln_x.device
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
        w = _stacked_weights(fc1, fc2)

        h_x = torch.empty(rows, 2 * ffn, dtype=ln_x.dtype, device=device)  # [gate | up], 1x16 E4M3
        h_s = torch.empty(rows, 2 * ffn // QB, dtype=torch.bfloat16, device=device)
        _fc1(_with_zero_row(ln_x), row_token, w["fc1"], ln_s.float(), w["fc1_s"], tiles, h_x, h_s)
        silu_x, silu_s, weighted, row_max = _act(h_x, h_s, perm_probs)
        mul_s = (row_max.max().to(torch.bfloat16) + THRES) / E4M3_MAX  # fp8_division_transpose's own scale
        if WGRAD != "rowmajor":
            mul_x, mul_s, mul_t = rq.fp8_division_transpose(weighted, QB, "E4M3", mul_s)
        else:
            mul_x, mul_s = rq.fp8_division(weighted, QB, "E4M3", mul_s)
            mul_t = mul_x[:0]
        del weighted
        out = torch.empty(rows, hidden, dtype=torch.bfloat16, device=device)
        gg.segment_gemm(mul_x, w["down"], mul_s.float(), w["down_s"], tiles, out)
        combined = gg.gather_sum(out, token_rows, torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device))
        ctx.save_for_backward(ln_x, ln_s, ln_t, h_x, h_s, silu_x, silu_s, mul_x, mul_t, mul_s,
                              perm_probs, row_token, row_flat, token_rows, starts, padded, tiles, *weights)
        ctx.meta = (experts, tokens, hidden, rows, ffn, anchor.shape, probs.dtype)
        return combined.view(anchor.shape)

    @staticmethod
    def backward(ctx, grad):
        (ln_x, ln_s, ln_t, h_x, h_s, silu_x, silu_s, mul_x, mul_t, mul_s,
         perm_probs, row_token, row_flat, token_rows, starts, padded, tiles, *weights) = ctx.saved_tensors
        experts, tokens, hidden, rows, ffn, shape, probs_dtype = ctx.meta
        fc1, fc2 = weights[:experts], weights[experts:]
        device = grad.device
        transposed = WGRAD != "rowmajor"  # upstream's transposed copies of the permuted operands exist
        permuted_copies = WGRAD in ("transposed", "reference")
        w = _stacked_weights(fc1, fc2)
        # Every token reaches topk experts, so the per-tensor scale of the permuted gradient is the
        # scale of the gradient itself and its permuted FP8 codes are these rows.
        grad = grad.reshape(tokens, hidden).to(torch.bfloat16).contiguous()
        if permuted_copies:
            g, gs, g_t = rq.fp8_quantize_pertensor_transpose(grad, QB, "E5M2", transpose_output_2d=True)
        else:
            g, gs, _ = rq.fp8_quantize_pertensor(grad, QB, "E5M2")
        g0 = _with_zero_row(g)
        gs32, mul_s32, ln_s32 = gs.float(), mul_s.float(), ln_s.float()
        d_perm_probs = torch.empty(rows, dtype=torch.float32, device=device)
        d_product = torch.empty(rows, ffn, dtype=torch.bfloat16, device=device)
        _down_dgrad(g0, row_token, w["down_t"], gs32, w["down_s"], tiles, mul_x, mul_s32, perm_probs, d_perm_probs, d_product)
        probs_grad = torch.zeros(tokens * experts + 1, dtype=torch.float32, device=device)
        probs_grad.index_put_((row_flat,), d_perm_probs)  # every real (token, expert) appears once
        probs_grad = probs_grad[:-1].view(tokens, experts).to(probs_dtype)
        gate_x, up_x = h_x[:, :ffn], h_x[:, ffn:]
        gate_s, up_s = h_s[:, :ffn // QB], h_s[:, ffn // QB:]
        if transposed:
            mul_g1, (mul_g2, mul_gs2, mul_g2_t) = rq.fp8_mul_backward(silu_x, silu_s, up_x, up_s, d_product, QB, "E5M2",
                                                                     output_quantized_transpose=True)
            del d_product
            silu_g, silu_gs, silu_g_t = rq.fp8_silu_backward(gate_x, gate_s, mul_g1, QB, "E5M2", output_quantized_transpose=True)
            del mul_g1
        else:
            # The same kernels and the same per-tensor division, without the transposed copies.
            mul_g1, (mul_g2, mul_gs2) = rq.fp8_mul_backward(silu_x, silu_s, up_x, up_s, d_product, QB, "E5M2")
            del d_product
            mul_g2, mul_gs2 = rq.fp8_division(mul_g2, QB, "E5M2", mul_gs2.max())
            silu_g, silu_gs = rq.fp8_silu_backward(gate_x, gate_s, mul_g1, QB, "E5M2")
            del mul_g1
            silu_g, silu_gs = rq.fp8_division(silu_g, QB, "E5M2", silu_gs.max())
        silu_gs32, mul_gs2_32 = silu_gs.float(), mul_gs2.float()
        d_x = torch.empty(rows, hidden, dtype=torch.bfloat16, device=device)
        _fc1_dgrad(silu_g, mul_g2, w["gate_t"], w["up_t"], silu_gs32, mul_gs2_32, w["gate_s"], w["up_s"], tiles, d_x)
        d_ln = gg.gather_sum(d_x, token_rows, torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device))
        del d_x
        fc2_grad = torch.empty(experts, hidden, ffn, dtype=fc2[0].dtype, device=device)
        fc1_grad = torch.empty(experts, 2 * ffn, hidden, dtype=fc1[0].dtype, device=device)
        segments = (starts, padded)
        if WGRAD == "reference":
            x_t = _permuted_transpose(ln_t, row_token)
            gg.segment_wgrad(_permuted_transpose(g_t, row_token), mul_t, gs32, mul_s32, *segments, fc2_grad)
            gg.segment_wgrad(silu_g_t, x_t, silu_gs32, ln_s32, *segments, fc1_grad[:, :ffn])
            gg.segment_wgrad(mul_g2_t, x_t, mul_gs2_32, ln_s32, *segments, fc1_grad[:, ffn:])
        elif WGRAD == "transposed":
            x_t = _permuted_transpose(ln_t, row_token)
            _wgrad(_permuted_transpose(g_t, row_token), mul_t, row_token, gs32, mul_s32, *segments, fc2_grad, 0, 0)
            _wgrad(silu_g_t, x_t, row_token, silu_gs32, ln_s32, *segments, fc1_grad[:, :ffn], 0, 0)
            _wgrad(mul_g2_t, x_t, row_token, mul_gs2_32, ln_s32, *segments, fc1_grad[:, ffn:], 0, 0)
        elif WGRAD == "hybrid":
            x0 = _with_zero_row(ln_x)
            _wgrad(g0, mul_t, row_token, gs32, mul_s32, *segments, fc2_grad, 2, 0)
            _wgrad(silu_g_t, x0, row_token, silu_gs32, ln_s32, *segments, fc1_grad[:, :ffn], 0, 2)
            _wgrad(mul_g2_t, x0, row_token, mul_gs2_32, ln_s32, *segments, fc1_grad[:, ffn:], 0, 2)
        else:
            x0 = _with_zero_row(ln_x)
            _wgrad(g0, mul_x, row_token, gs32, mul_s32, *segments, fc2_grad, 2, 1)
            _wgrad(silu_g, x0, row_token, silu_gs32, ln_s32, *segments, fc1_grad[:, :ffn], 1, 2)
            _wgrad(mul_g2, x0, row_token, mul_gs2_32, ln_s32, *segments, fc1_grad[:, ffn:], 1, 2)
        return (d_ln.view(shape), probs_grad, None, None, None, None, None, *fc1_grad.unbind(0), *fc2_grad.unbind(0))
