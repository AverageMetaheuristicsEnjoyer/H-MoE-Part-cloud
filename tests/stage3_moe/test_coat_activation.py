"""GPU checks of the COAT activation port against FP32 references. Lines COAT_REPORT carry numbers."""
import json

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from stage3_moe import coat_activation as ca  # noqa: E402

T, H, FFN, MOE_FFN, E, TOPK, EPS = 4096, 1024, 2816, 256, 64, 8, 1e-5


@pytest.fixture(autouse=True)
def fresh_weight_cache():
    # Freed test tensors hand their addresses to new ones; training parameters never move.
    ca.invalidate_weights()
    yield
    ca.invalidate_weights()


def report(name, **values):
    print("COAT_REPORT " + json.dumps({"check": name, **values}), flush=True)


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _param(shape, seed, std=0.02):
    return torch.nn.Parameter((torch.randn(shape, device="cuda", generator=_gen(seed)) * std).to(torch.bfloat16))


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def _cos(a, b):
    return F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0).item()


def _rms(x, w):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w.float()


def _swiglu_mlp(x, w1, w2):
    gate, up = (x @ w1.float().t()).chunk(2, dim=-1)
    return (F.silu(gate) * up) @ w2.float().t()


def _compare(name, outputs, references, inputs):
    """Forward error plus gradient agreement for every input, from one shared upstream gradient."""
    torch.manual_seed(0)
    upstream = [torch.randn_like(o.float()) for o in outputs]
    grads = torch.autograd.grad([o.float() for o in outputs], inputs, upstream, allow_unused=True, retain_graph=True)
    ref_grads = torch.autograd.grad(references, [i.ref for i in inputs], upstream, allow_unused=True)
    values = {"forward_rel_l2": max(_rel(o, r) for o, r in zip(outputs, references))}
    for item, g, rg in zip(inputs, grads, ref_grads):
        values[f"{item.label}_grad_cos"] = _cos(g, rg)
        values[f"{item.label}_grad_rel_l2"] = _rel(g, rg)
    report(name, **values)
    assert values["forward_rel_l2"] < 0.15
    for item in inputs:
        assert values[f"{item.label}_grad_cos"] > 0.95, item.label
    return values


def _leaf(tensor, label):
    t = tensor.detach().clone().requires_grad_(True)
    t.label = label
    t.ref = tensor.detach().clone().float().requires_grad_(True)
    t.ref.label = label
    return t


def _x(seed):
    return (torch.randn(T // 2, 2, H, device="cuda", generator=_gen(seed))).to(torch.bfloat16)


def test_norm_linear_matches_fp32():
    x, nw, w = _leaf(_x(1), "x"), _leaf(1 + 0.1 * _param((H,), 2, 1.0).detach(), "norm"), _leaf(_param((1536, H), 3).detach(), "w")
    y = ca.NormLinear.apply(x, nw, w, EPS)
    ref = _rms(x.ref, nw.ref) @ w.ref.t()
    _compare("norm_linear", [y], [ref], [x, nw, w])


def test_linear_matches_fp32():
    x, w = _leaf(_x(4), "x"), _leaf(_param((H, H), 5).detach(), "w")
    y = ca.Linear.apply(x, w)
    _compare("linear", [y], [x.ref @ w.ref.t()], [x, w])


def test_dense_mlp_matches_fp32():
    x = _leaf(_x(6), "x")
    nw = _leaf(1 + 0.1 * _param((H,), 7, 1.0).detach(), "norm")
    w1, w2 = _leaf(_param((2 * FFN, H), 8).detach(), "fc1"), _leaf(_param((H, FFN), 9).detach(), "fc2")
    y = ca.NormMLP.apply(x, nw, w1, w2, EPS)
    ref = _swiglu_mlp(_rms(x.ref, nw.ref), w1.ref, w2.ref)
    _compare("dense_mlp", [y], [ref], [x, nw, w1, w2])


@pytest.mark.parametrize("skew", [0.0, 3.0])
def test_moe_block_matches_fp32(skew):
    """Norm -> router -> routed experts + shared expert, gradients through the probabilities."""
    x = _leaf(_x(10), "x")
    nw = _leaf(1 + 0.1 * _param((H,), 11, 1.0).detach(), "norm")
    router_w = _leaf(_param((E, H), 12, 0.05).detach().float(), "router")
    bias = torch.zeros(E, device="cuda")
    bias[: E // 4] = -skew  # starve a quarter of the experts; with skew 3 some receive no tokens
    fc1 = [_leaf(_param((2 * MOE_FFN, H), 100 + i).detach(), f"fc1_{i}") for i in range(E)]
    fc2 = [_leaf(_param((H, MOE_FFN), 200 + i).detach(), f"fc2_{i}") for i in range(E)]
    s1, s2 = _leaf(_param((2 * MOE_FFN, H), 13).detach(), "shared_fc1"), _leaf(_param((H, MOE_FFN), 14).detach(), "shared_fc2")

    ln_bf16, ln_x, ln_s, ln_t = ca.Norm.apply(x, nw, EPS)
    with torch.no_grad():
        scores = torch.sigmoid(ln_bf16.reshape(-1, H).float() @ router_w.float().t()) + bias
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, scores.topk(TOPK, dim=-1).indices, True)
    probs = torch.where(routing_map, torch.sigmoid(ln_bf16.reshape(-1, H).float() @ router_w.t()), torch.zeros(()))
    probs = probs / probs.sum(-1, keepdim=True) * 2.5
    routed = ca.Experts.apply(ln_bf16, probs, ln_x, ln_s, routing_map, TOPK, *fc1, *fc2)
    shared = ca.SharedMLP.apply(ln_bf16, ln_x, ln_s, ln_t, s1, s2)
    y = routed + shared

    ln_ref = _rms(x.ref, nw.ref).reshape(-1, H)
    probs_ref = torch.where(routing_map, torch.sigmoid(ln_ref @ router_w.ref.t()), torch.zeros(()))
    probs_ref = probs_ref / probs_ref.sum(-1, keepdim=True) * 2.5
    out = torch.zeros(T, H, device="cuda")
    for e in range(E):
        idx = routing_map[:, e].nonzero(as_tuple=True)[0]
        if idx.numel():
            gate, up = (ln_ref[idx] @ fc1[e].ref.t()).chunk(2, -1)
            out = out.index_add(0, idx, ((F.silu(gate) * up) * probs_ref[idx, e:e + 1]) @ fc2[e].ref.t())
    ref = (out + _swiglu_mlp(ln_ref, s1.ref, s2.ref)).view(T // 2, 2, H)

    counts = routing_map.sum(0)
    empty = (counts == 0).nonzero(as_tuple=True)[0].tolist()
    busy = [e for e in range(E) if counts[e] > 0]
    values = _compare(f"moe_block_skew{skew}", [y], [ref], [x, nw, router_w, s1, s2, fc1[busy[0]], fc2[busy[0]], fc1[busy[-1]], fc2[busy[-1]]])
    report("moe_routing", skew=skew, empty_experts=len(empty), min_tokens=int(counts.min()), max_tokens=int(counts.max()))
    if empty:
        grads = torch.autograd.grad(y.float().sum(), [fc1[empty[0]], fc2[empty[0]]])
        assert all(torch.count_nonzero(g) == 0 for g in grads)


def test_weight_cache_follows_optimizer_steps():
    w = _param((H, H), 30)
    scale = ca.fp8_weight(w)[1].item()
    with torch.no_grad():
        w.mul_(2)
    assert ca.fp8_weight(w)[1].item() == scale  # cached until the optimizer step invalidates it
    ca.invalidate_weights()
    assert abs(ca.fp8_weight(w)[1].item() / scale - 2) < 0.01


def test_saved_activation_bytes():
    """What the routed-expert function keeps for backward, against BF16 TE grouped MLP's set."""
    x = _x(40).reshape(-1, H)
    nw = torch.ones(H, device="cuda", dtype=torch.bfloat16)
    fc1 = [_param((2 * MOE_FFN, H), 300 + i) for i in range(E)]
    fc2 = [_param((H, MOE_FFN), 400 + i) for i in range(E)]
    ln_bf16, ln_x, ln_s, ln_t = ca.Norm.apply(x.requires_grad_(True), nw, EPS)
    scores = torch.rand(T, E, device="cuda", generator=_gen(41))
    routing_map = torch.zeros(T, E, dtype=torch.bool, device="cuda").scatter_(1, scores.topk(TOPK, -1).indices, True)
    probs = torch.where(routing_map, scores, torch.zeros(())).requires_grad_(True)
    saved = []

    def pack(t):
        saved.append(t)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        ca.Experts.apply(ln_bf16, probs, ln_x, ln_s, routing_map, TOPK, *fc1, *fc2)
    weights = {id(w) for w in fc1 + fc2}
    coat = sum(t.numel() * t.element_size() for t in saved if id(t) not in weights)
    rows = T * TOPK
    bf16_te = rows * H * 2 + rows * 2 * MOE_FFN * 2 + rows * MOE_FFN * 2 + rows * 4
    report("expert_saved_bytes", tokens=T, coat_bytes=coat, bf16_grouped_mlp_estimate=bf16_te, ratio=coat / bf16_te)


@pytest.mark.parametrize("skew", [0.0, 3.0])
def test_grouped_experts_match_loop(skew):
    """GroupedExperts changes only how expert GEMMs are launched; outputs and gradients must agree."""
    x = _x(50).reshape(-1, H)
    nw = torch.ones(H, device="cuda", dtype=torch.bfloat16)
    fc1 = [_param((2 * MOE_FFN, H), 500 + i).detach().requires_grad_(True) for i in range(E)]
    fc2 = [_param((H, MOE_FFN), 600 + i).detach().requires_grad_(True) for i in range(E)]
    scores = torch.rand(T, E, device="cuda", generator=_gen(51))
    scores[:, : E // 4] -= skew
    routing_map = torch.zeros(T, E, dtype=torch.bool, device="cuda").scatter_(1, scores.topk(TOPK, -1).indices, True)
    probs_base = torch.where(routing_map, scores.clamp_min(0.05), torch.zeros(()))
    upstream = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    results = []
    for fn in (ca.Experts, ca.GroupedExperts):
        ca.invalidate_weights()
        xi = x.clone().requires_grad_(True)
        probs = probs_base.clone().requires_grad_(True)
        ln_bf16, ln_x, ln_s, ln_t = ca.Norm.apply(xi, nw, EPS)
        y = fn.apply(ln_bf16, probs, ln_x, ln_s, routing_map, TOPK, *fc1, *fc2)
        grads = torch.autograd.grad(y, [xi, probs, *fc1, *fc2], upstream)
        results.append((y, grads))
    (y0, g0), (y1, g1) = results
    worst = max(_rel(a, b) for a, b in zip(g1, g0) if b.abs().sum() > 0)
    empty = (routing_map.sum(0) == 0).nonzero(as_tuple=True)[0].tolist()
    report("grouped_vs_loop", skew=skew, forward_rel_l2=_rel(y1, y0), worst_grad_rel_l2=worst, empty_experts=len(empty))
    assert _rel(y1, y0) < 2e-2 and worst < 5e-2
    for e in empty:
        assert torch.count_nonzero(g1[2 + e]) == 0 and torch.count_nonzero(g1[2 + E + e]) == 0


@pytest.mark.parametrize("wgrad", ["rowmajor", "hybrid", "transposed", "reference"])
@pytest.mark.parametrize("skew", [0.0, 3.0])
def test_fused_experts_match_grouped(skew, wgrad, monkeypatch):
    """FusedExperts changes how often memory is traversed, not the arithmetic of GroupedExperts."""
    from stage3_moe import coat_fused_experts as fe

    monkeypatch.setattr(fe, "WGRAD", wgrad)
    x = _x(50).reshape(-1, H)
    nw = torch.ones(H, device="cuda", dtype=torch.bfloat16)
    fc1 = [_param((2 * MOE_FFN, H), 500 + i).detach().requires_grad_(True) for i in range(E)]
    fc2 = [_param((H, MOE_FFN), 600 + i).detach().requires_grad_(True) for i in range(E)]
    scores = torch.rand(T, E, device="cuda", generator=_gen(51))
    scores[:, : E // 4] -= skew
    routing_map = torch.zeros(T, E, dtype=torch.bool, device="cuda").scatter_(1, scores.topk(TOPK, -1).indices, True)
    probs_base = torch.where(routing_map, scores.clamp_min(0.05), torch.zeros(()))
    upstream = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    results = []
    for fn, with_ln_t in ((ca.GroupedExperts, False), (fe.FusedExperts, True)):
        ca.invalidate_weights()
        xi = x.clone().requires_grad_(True)
        probs = probs_base.clone().requires_grad_(True)
        ln_bf16, ln_x, ln_s, ln_t = ca.Norm.apply(xi, nw, EPS)
        fp8 = (ln_x, ln_s, ln_t) if with_ln_t else (ln_x, ln_s)
        y = fn.apply(ln_bf16, probs, *fp8, routing_map, TOPK, *fc1, *fc2)
        grads = torch.autograd.grad(y, [xi, probs, *fc1, *fc2], upstream)
        results.append((y, grads))
    (y0, g0), (y1, g1) = results
    # d_probs is a sum over the expert width whose reduction tree differs between the two launches.
    exact = [torch.equal(y1, y0), torch.equal(g1[0], g0[0]), *(torch.equal(a, b) for a, b in zip(g1[2:], g0[2:]))]
    worst = max(_rel(a, b) for a, b in zip(g1, g0) if b.abs().sum() > 0)
    report("fused_vs_grouped", skew=skew, wgrad=wgrad, forward_bitwise=exact[0], input_grad_bitwise=exact[1],
           weight_grads_bitwise=sum(exact[2:]), weight_grads=len(exact) - 2, probs_grad_rel_l2=_rel(g1[1], g0[1]),
           forward_rel_l2=_rel(y1, y0), worst_grad_rel_l2=worst)
    assert all(exact) and _rel(g1[1], g0[1]) < 1e-5


def test_stacked_weight_quantizers_bitwise():
    """The fused path's one-launch weight quantizer yields the grouped path's per-expert codes, scales and layouts."""
    from stage3_moe import coat_fused_experts as fe

    fc1 = [_param((2 * MOE_FFN, H), 500 + i).detach() for i in range(E)]
    fc2 = [_param((H, MOE_FFN), 600 + i).detach() for i in range(E)]
    per_expert = ca._stacked_expert_weights(fc1, fc2)
    stacked = fe._stacked_weights(fc1, fc2)
    equal = {name: stacked[name].shape == per_expert[name].shape and stacked[name].stride() == per_expert[name].stride()
             and torch.equal(stacked[name].view(torch.uint8) if stacked[name].dtype == torch.float8_e4m3fn else stacked[name],
                             per_expert[name].view(torch.uint8) if per_expert[name].dtype == torch.float8_e4m3fn else per_expert[name])
             for name in per_expert}
    report("stacked_weight_quantizers", **equal)
    assert all(equal.values())


def test_segment_gemm_matches_scaled_mm_on_identical_fp8_inputs():
    """The grouped kernels alone, against per-expert torch._scaled_mm on the same FP8 operands."""
    from stage3_moe import coat_grouped_gemm as gg

    g = _gen(70)
    counts = torch.randint(0, 600, (E,), device="cuda", generator=g)
    counts[:3] = 0
    padded = (counts + 127) // 128 * 128
    starts = torch.cumsum(padded, 0) - padded
    rows = int(counts.sum()) + E * 128
    rows = (rows + 127) // 128 * 128
    tiles = gg.tile_experts(starts, padded, rows)
    x = (torch.randn(rows, H, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    grad = (torch.randn(rows, MOE_FFN, device="cuda", generator=g) * 2).to(torch.float8_e5m2)
    w = (torch.randn(E, MOE_FFN, H, device="cuda", generator=g)).to(torch.float8_e4m3fn)
    sa = torch.tensor(0.5, device="cuda"); sb = torch.rand(E, device="cuda", generator=g) + 0.5
    out = torch.empty(rows, MOE_FFN, dtype=torch.float32, device="cuda")
    gg.segment_gemm(x, w, sa, sb, tiles, out)
    din = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda")
    gg.segment_gemm(grad, w.transpose(1, 2).contiguous(), sa, sb, tiles, din)
    wg = torch.empty(E, MOE_FFN, H, dtype=torch.bfloat16, device="cuda")
    gg.segment_wgrad(grad.view(torch.uint8).t().contiguous().view(grad.dtype), x.view(torch.uint8).t().contiguous().view(x.dtype),
                     sa, sa, starts, padded, wg)
    worst = {"forward": 0.0, "dgrad": 0.0, "wgrad": 0.0}
    for e in range(E):
        s, n = int(starts[e]), int(padded[e])
        if n == 0:
            assert torch.count_nonzero(wg[e]) == 0
            continue
        ref = torch._scaled_mm(x[s:s + n], w[e].t(), scale_a=sa, scale_b=sb[e], out_dtype=torch.float32)
        worst["forward"] = max(worst["forward"], _rel(out[s:s + n], ref))
        ref = torch._scaled_mm(grad[s:s + n], w[e].t().contiguous().t(), scale_a=sa, scale_b=sb[e], out_dtype=torch.float32)
        worst["dgrad"] = max(worst["dgrad"], _rel(din[s:s + n], ref))
        ref = torch._scaled_mm(grad[s:s + n].t().contiguous(), x[s:s + n].t().contiguous().t(), scale_a=sa, scale_b=sa, out_dtype=torch.float32)
        worst["wgrad"] = max(worst["wgrad"], _rel(wg[e], ref))
    tail = int(starts[-1] + padded[-1])
    report("segment_gemm_vs_scaled_mm", **worst, tail_rows_zero=bool(out[tail:].abs().sum() == 0))
    assert max(worst.values()) < 1e-2 and out[tail:].abs().sum() == 0


GROUPS, Q_ROWS, KV_ROWS = 2, 512, 128  # 8 heads of 128 in 2 query groups, as in the 1.029B model


@pytest.fixture
def strict():
    ca.STRICT = True
    ca.invalidate_weights()
    yield
    ca.STRICT = False
    ca.invalidate_weights()


def _fused_qkv(q, k, v):
    """MCore's fused QKV rows: per query group, its q heads, then k, then v."""
    return torch.cat([q.view(GROUPS, Q_ROWS, H), k.view(GROUPS, KV_ROWS, H), v.view(GROUPS, KV_ROWS, H)], 1).view(-1, H)


def test_strict_qkv_matches_fp32(strict):
    x, nw = _leaf(_x(80), "x"), _leaf(1 + 0.1 * _param((H,), 81, 1.0).detach(), "norm")
    w = _leaf(_param((GROUPS * (Q_ROWS + 2 * KV_ROWS), H), 82).detach(), "w")
    y = ca.NormLinearQKV.apply(x, nw, w, EPS, (GROUPS, Q_ROWS, KV_ROWS))
    _compare("strict_qkv", [y], [_rms(x.ref, nw.ref) @ w.ref.t()], [x, nw, w])


def test_strict_qkv_is_three_coat_linears(strict):
    """Bit for bit what coat_llama's BeforeAttentionResidual does with separate q, k, v weights."""
    rq = ca.rq
    q, k, v = (_param((rows, H), seed).detach() for rows, seed in ((GROUPS * Q_ROWS, 83), (GROUPS * KV_ROWS, 84), (GROUPS * KV_ROWS, 85)))
    x = _x(86).reshape(-1, H)
    nw = (1 + 0.1 * _param((H,), 87, 1.0)).detach()
    upstream = torch.randn(x.shape[0], GROUPS * (Q_ROWS + 2 * KV_ROWS), device="cuda", generator=_gen(88)).to(torch.bfloat16)

    xi, wi, nwi = x.clone().requires_grad_(True), _fused_qkv(q, k, v).detach().requires_grad_(True), nw.clone().requires_grad_(True)
    y = ca.NormLinearQKV.apply(xi, nwi, wi, EPS, (GROUPS, Q_ROWS, KV_ROWS))
    dx, dnw, dw = torch.autograd.grad(y, [xi, nwi, wi], upstream)

    # coat_llama.py:221-347 with weight_memory_efficient=True, written out with upstream functions.
    qx, sx = rq.fp8_quantize(x, 16, "E4M3")
    ln_x, ln_s, ln_t, (_, rstd, warps) = rq.fp8_rmsnorm_forward(qx, sx, nw, 16, EPS, transpose_output_2d=True)
    outs, grads, fc_g = [], [], None
    g3 = upstream.view(-1, GROUPS, Q_ROWS + 2 * KV_ROWS)
    for weight, cols in ((q, slice(0, Q_ROWS)), (k, slice(Q_ROWS, Q_ROWS + KV_ROWS)), (v, slice(Q_ROWS + KV_ROWS, None))):
        _, w_s, _ = rq.fp8_division_transpose(weight, 16, "E4M3")
        w_fp8, w_s = rq.fp8_division(weight, 16, "E4M3", w_s)
        outs.append(rq.fp8_linear_forward(ln_x, ln_s, w_fp8, w_s, False, 16))
        g, gs, g_t = rq.fp8_quantize_pertensor_transpose(g3[:, :, cols].reshape(x.shape[0], -1).contiguous(), 16, "E5M2",
                                                        transpose_output_2d=True)
        w_t, w_s = rq.fp8_division_transpose(weight, 16, "E4M3", w_s, only_transposed=True)
        d, wg = rq.fp8_linear_backward(ln_t, ln_s, g, gs, g_t, w_t, w_s, 16)
        fc_g = d if fc_g is None else fc_g + d
        grads.append(wg)
    ref_dx, ref_dnw = rq.fp8_rmsnorm_backward(qx, sx, fc_g, nw, rstd, 16, warps)
    ref_y = torch.cat([outs[0].view(-1, GROUPS, Q_ROWS), outs[1].view(-1, GROUPS, KV_ROWS), outs[2].view(-1, GROUPS, KV_ROWS)], 2)
    checks = {"y": torch.equal(y, ref_y.view_as(y)), "dx": torch.equal(dx, ref_dx), "dnorm": torch.equal(dnw, ref_dnw),
              "dw": torch.equal(dw, _fused_qkv(*grads))}
    report("strict_qkv_bitwise", **checks)
    assert all(checks.values())


def _run(fn, *tensors, upstream_seed=90):
    leaves = [t.detach().clone().requires_grad_(t.is_floating_point()) for t in tensors]
    out = fn(*leaves)
    upstream = torch.randn(out.shape, device="cuda", generator=_gen(upstream_seed)).to(out.dtype)
    grads = torch.autograd.grad(out, [t for t in leaves if t.requires_grad], upstream, allow_unused=True)
    return out, grads


def test_strict_weights_change_nothing_but_memory():
    """weight_memory_efficient rebuilds the same FP8 copies from the same scale: bitwise equal."""
    x = _x(91)
    nw = (1 + 0.1 * _param((H,), 92, 1.0)).detach()
    w1, w2, wo = _param((2 * FFN, H), 93).detach(), _param((H, FFN), 94).detach(), _param((H, H), 95).detach()
    cases = {"linear": (lambda a, b: ca.Linear.apply(a, b), (x, wo)),
             "dense_mlp": (lambda a, b, c, d: ca.NormMLP.apply(a, b, c, d, EPS), (x, nw, w1, w2))}
    results = {}
    for mode in (False, True):
        ca.STRICT = mode
        ca.invalidate_weights()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        results[mode] = {name: _run(fn, *args) for name, (fn, args) in cases.items()}
        results[mode]["peak_extra_bytes"] = torch.cuda.max_memory_allocated() - base
        results[mode]["cached_bytes"] = sum(t.numel() * t.element_size() for entry in ca._WEIGHT_CACHE.values()
                                            for t in (entry.values() if isinstance(entry, dict) else entry))
    ca.STRICT = False
    ca.invalidate_weights()
    equal = {name: torch.equal(results[False][name][0], results[True][name][0])
             and all(torch.equal(a, b) for a, b in zip(results[False][name][1], results[True][name][1])) for name in cases}
    report("strict_weights_bitwise", **equal, cached_bytes={str(m): results[m]["cached_bytes"] for m in (False, True)},
           peak_extra_bytes={str(m): results[m]["peak_extra_bytes"] for m in (False, True)})
    assert all(equal.values())


@pytest.mark.parametrize("skew", [0.0, 3.0])
def test_strict_weights_experts_bitwise(skew):
    x = _x(96).reshape(-1, H)
    nw = torch.ones(H, device="cuda", dtype=torch.bfloat16)
    fc1 = [_param((2 * MOE_FFN, H), 700 + i).detach() for i in range(E)]
    fc2 = [_param((H, MOE_FFN), 800 + i).detach() for i in range(E)]
    scores = torch.rand(T, E, device="cuda", generator=_gen(97))
    scores[:, : E // 4] -= skew
    routing_map = torch.zeros(T, E, dtype=torch.bool, device="cuda").scatter_(1, scores.topk(TOPK, -1).indices, True)
    probs_base = torch.where(routing_map, scores.clamp_min(0.05), torch.zeros(()))
    upstream = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, generator=_gen(98))
    results = []
    for mode in (False, True):
        ca.STRICT = mode
        ca.invalidate_weights()
        w = [t.clone().requires_grad_(True) for t in fc1 + fc2]
        xi, probs = x.clone().requires_grad_(True), probs_base.clone().requires_grad_(True)
        ln_bf16, ln_x, ln_s, ln_t = ca.Norm.apply(xi, nw, EPS)
        y = ca.Experts.apply(ln_bf16, probs, ln_x, ln_s, routing_map, TOPK, *w)
        results.append((y, torch.autograd.grad(y, [xi, probs, *w], upstream)))
    ca.STRICT = False
    ca.invalidate_weights()
    equal = torch.equal(results[0][0], results[1][0]) and all(torch.equal(a, b) for a, b in zip(results[0][1], results[1][1]))
    report("strict_weights_experts_bitwise", skew=skew, equal=equal)
    assert equal
