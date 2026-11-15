"""Grouped FP8 GEMMs for the routed experts: one Triton launch per GEMM kind per layer.

The per-expert loop of torch._scaled_mm issues ~40k launches per training step from Python.
These kernels compute the same products -- FP8 operands, FP32 accumulation, per-tensor
activation/gradient scale times per-expert weight scale -- over every expert at once.
Expert segments must start on multiples of BLOCK_M so no tile straddles two experts.

Both dot operands are kept K-major, which Hopper's FP8 MMA requires: forward reads the weights
[N, K] as stored, data gradients read the transposed weight copies, weight gradients read the
transposed activation and gradient copies the COAT flow already produces.
"""
import torch
import triton
import triton.language as tl

BLOCK_M = 128


@triton.jit
def _segment_gemm_kernel(A, B, C, tile_expert, scale_a, scale_b, N, K,
                         stride_am, stride_ak, stride_bg, stride_bn, stride_bk, stride_cm, stride_cn,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                         ACCUMULATE: tl.constexpr):
    """C[m, n] (+)= scale_a * scale_b[g] * sum_k A[m, k] * B[g, n, k] for the rows m of expert g;
    tiles past the last expert are written as zeros, so C needs no initialization."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.load(tile_expert + pid_m)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    if g < 0:
        if not ACCUMULATE:
            tl.store(c_ptr, tl.zeros((BLOCK_M, BLOCK_N), dtype=C.dtype.element_ty), mask=rn[None, :] < N)
        return
    rk = tl.arange(0, BLOCK_K)
    a_ptr = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptr = B + g.to(tl.int64) * stride_bg + rn[None, :] * stride_bn + rk[:, None] * stride_bk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptr, mask=rk[None, :] < k_left, other=0.0)
        b = tl.load(b_ptr, mask=(rk[:, None] < k_left) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptr += BLOCK_K * stride_ak
        b_ptr += BLOCK_K * stride_bk
    acc = acc * (tl.load(scale_a) * tl.load(scale_b + g))
    if ACCUMULATE:
        acc += tl.load(c_ptr, mask=rn[None, :] < N, other=0.0).to(tl.float32)
    tl.store(c_ptr, acc.to(C.dtype.element_ty), mask=rn[None, :] < N)


@triton.jit
def _segment_wgrad_kernel(GT, XT, D, seg_start, seg_rows, scale_a, scale_b, N, K,
                          stride_gn, stride_gm, stride_xk, stride_xm, stride_dg, stride_dn, stride_dk,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """D[g, n, k] = scale_a * scale_b * sum_{m in expert g} GT[n, m] * XT[k, m]."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    g = tl.program_id(2)
    start = tl.load(seg_start + g)
    rows = tl.load(seg_rows + g)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rm = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m in range(0, rows, BLOCK_M):
        cols = start + m + rm
        a = tl.load(GT + rn[:, None] * stride_gn + cols[None, :] * stride_gm, mask=rn[:, None] < N, other=0.0)
        b = tl.load(XT + rk[None, :] * stride_xk + cols[:, None] * stride_xm, mask=rk[None, :] < K, other=0.0)
        acc = tl.dot(a, b, acc)
    acc = acc * (tl.load(scale_a) * tl.load(scale_b))
    d_ptr = D + g.to(tl.int64) * stride_dg + rn[:, None] * stride_dn + rk[None, :] * stride_dk
    tl.store(d_ptr, acc.to(D.dtype.element_ty), mask=(rn[:, None] < N) & (rk[None, :] < K))


def tile_experts(starts, padded, rows):
    """Expert of each BLOCK_M row tile, -1 for tiles past the last segment."""
    tiles = torch.arange(rows // BLOCK_M, device=starts.device) * BLOCK_M
    ends = starts + padded
    expert = torch.searchsorted(ends, tiles, right=True)
    return torch.where(tiles < ends[-1], expert, torch.full_like(expert, -1)).to(torch.int32)


def segment_gemm(a, b, scale_a, scale_b, tile_expert, out, accumulate=False):
    """out[m] (+)= a[m] @ b[g(m)].T with b stacked [G, N, K]; a [rows, K] row-major."""
    rows, k = a.shape
    n = b.shape[1]
    assert rows % BLOCK_M == 0 and a.stride(1) == 1 and b.stride(2) == 1
    grid = (rows // BLOCK_M, triton.cdiv(n, 128))
    _segment_gemm_kernel[grid](a, b, out, tile_expert, scale_a, scale_b, n, k,
                               a.stride(0), a.stride(1), b.stride(0), b.stride(1), b.stride(2),
                               out.stride(0), out.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=128, BLOCK_K=128,
                               ACCUMULATE=accumulate, num_warps=8, num_stages=3)
    return out


def segment_wgrad(gt, xt, scale_a, scale_b, seg_start, seg_rows, out):
    """out[g] = gt[:, seg g] @ xt[:, seg g].T; gt [N, rows], xt [K, rows]; out [G, N, K]."""
    n, k, experts = gt.shape[0], xt.shape[0], out.shape[0]
    assert gt.stride(1) == 1 and xt.stride(1) == 1
    grid = (triton.cdiv(n, 128), triton.cdiv(k, 128), experts)
    _segment_wgrad_kernel[grid](gt, xt, out, seg_start, seg_rows, scale_a, scale_b, n, k,
                                gt.stride(0), gt.stride(1), xt.stride(0), xt.stride(1),
                                out.stride(0), out.stride(1), out.stride(2),
                                BLOCK_M=BLOCK_M, BLOCK_N=128, BLOCK_K=128, num_warps=8, num_stages=3)
    return out


@triton.jit
def _gather_sum_kernel(src, token_rows, out, H, stride_src, stride_out, TOPK: tl.constexpr, BLOCK_H: tl.constexpr):
    """out[t] = sum_j src[token_rows[t, j]] in FP32, written in out's dtype."""
    t = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for j in tl.static_range(TOPK):
        row = tl.load(token_rows + t * TOPK + j).to(tl.int64)
        acc += tl.load(src + row * stride_src + cols, mask=cols < H, other=0.0).to(tl.float32)
    tl.store(out + t.to(tl.int64) * stride_out + cols, acc.to(out.dtype.element_ty), mask=cols < H)


def gather_sum(src, token_rows, out):
    tokens, topk = token_rows.shape
    hidden = src.shape[1]
    grid = (tokens, triton.cdiv(hidden, 1024))
    _gather_sum_kernel[grid](src, token_rows, out, hidden, src.stride(0), out.stride(0), TOPK=topk, BLOCK_H=1024)
    return out


@triton.jit
def _probs_backward_kernel(dw, mul_t, mul_scale, probs, d_probs, d_product, rows, F,
                           stride_dw, stride_mt, stride_dp, BLOCK_R: tl.constexpr, BLOCK_F: tl.constexpr):
    """For h = SiLU(gate)*up and the FP8 copy of p*h the down GEMM consumed:
    d_probs[r] = <dw[r], p*h[r]> / p[r] and d_product[r] = p[r] * dw[r]."""
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    f = tl.arange(0, BLOCK_F)
    mask = (r[:, None] < rows) & (f[None, :] < F)
    d = tl.load(dw + r[:, None].to(tl.int64) * stride_dw + f[None, :], mask=mask, other=0.0).to(tl.float32)
    weighted = tl.load(mul_t + f[None, :].to(tl.int64) * stride_mt + r[:, None], mask=mask, other=0.0).to(tl.float32)
    weighted = weighted * tl.load(mul_scale)
    p = tl.load(probs + r, mask=r < rows, other=0.0)
    safe = tl.where(p > 0, p, 1.0)
    tl.store(d_probs + r, tl.sum(d * weighted, axis=1) / safe, mask=r < rows)
    tl.store(d_product + r[:, None].to(tl.int64) * stride_dp + f[None, :], (d * p[:, None]).to(d_product.dtype.element_ty), mask=mask)


def probs_backward(dw, mul_t, mul_scale, probs, d_probs, d_product):
    rows, ffn = dw.shape
    grid = (triton.cdiv(rows, 64),)
    _probs_backward_kernel[grid](dw, mul_t, mul_scale, probs, d_probs, d_product, rows, ffn,
                                 dw.stride(0), mul_t.stride(0), d_product.stride(0),
                                 BLOCK_R=64, BLOCK_F=triton.next_power_of_2(ffn))
