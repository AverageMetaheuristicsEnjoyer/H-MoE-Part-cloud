"""Where the COAT activation flow spends its time, at the real layer shapes (one GPU).

Prints COAT_PROFILE json lines: forward+backward wall time of each ported block against its
BF16 Transformer Engine counterpart, and the top operators of the routed-expert block.
"""
import json
import sys

import torch
import transformer_engine.pytorch as te

from stage3_moe import coat_activation as ca

T, H, F, E, TOPK, EPS = 13 * 2048, 1024, 256, 64, 8, 1e-5


def report(name, **values):
    print("COAT_PROFILE " + json.dumps({"check": name, **values}), flush=True)


def timed(fn, repeat=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def param(*shape, std=0.02):
    return torch.nn.Parameter(torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * std)


def routing(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand(T, E, device="cuda", generator=g)
    routing_map = torch.zeros(T, E, dtype=torch.bool, device="cuda").scatter_(1, scores.topk(TOPK, -1).indices, True)
    probs = torch.where(routing_map, scores, torch.zeros((), device="cuda")).requires_grad_(True)
    return probs, routing_map


def experts():
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
    nw = torch.ones(H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fc1 = [param(2 * F, H) for _ in range(E)]
    fc2 = [param(H, F) for _ in range(E)]
    probs, routing_map = routing()
    grad = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

    def coat():
        ca.invalidate_weights()
        ln_bf16, ln_x, ln_s, ln_t = ca.Norm.apply(x, nw, EPS)
        out = ca.Experts.apply(ln_bf16, probs, ln_x, ln_s, routing_map, TOPK, *fc1, *fc2)
        out.backward(grad)

    grouped1 = te.GroupedLinear(E, H, 2 * F, bias=False, params_dtype=torch.bfloat16)
    grouped2 = te.GroupedLinear(E, F, H, bias=False, params_dtype=torch.bfloat16)
    counts = routing_map.sum(0).tolist()
    perm = torch.randn(T * TOPK, H, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
    pp = torch.rand(T * TOPK, 1, device="cuda")

    def bf16():
        h = grouped1(perm, counts)
        gate, up = h.chunk(2, -1)
        y = grouped2((torch.nn.functional.silu(gate) * up * pp).to(torch.bfloat16), counts)
        y.backward(torch.randn_like(y))

    def coat_grouped():
        ca.invalidate_weights()
        ln_bf16, ln_x, ln_s, ln_t = ca.Norm.apply(x, nw, EPS)
        out = ca.GroupedExperts.apply(ln_bf16, probs, ln_x, ln_s, routing_map, TOPK, *fc1, *fc2)
        out.backward(grad)

    report("experts_fwd_bwd_ms", coat_loop=timed(coat), coat_grouped=timed(coat_grouped), te_bf16_grouped=timed(bf16),
           tokens=T, experts=E, topk=TOPK, note="coat timings include per-call weight quantization (once per step in training)")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        coat_grouped()
        torch.cuda.synchronize()
    events = prof.key_averages()
    cuda_total = sum(e.self_device_time_total for e in events) / 1e3
    top = sorted(events, key=lambda e: e.self_device_time_total, reverse=True)[:14]
    report("experts_top_ops", cuda_ms_total=cuda_total,
           ops=[{"name": e.key[:60], "count": e.count, "cuda_ms": round(e.self_device_time_total / 1e3, 2),
                 "cpu_ms": round(e.self_cpu_time_total / 1e3, 2)} for e in top])
    calls = {e.key: e.count for e in events}
    report("experts_launch_counts", scaled_mm=calls.get("aten::_scaled_mm", 0), index_ops=calls.get("aten::index", 0),
           index_put=calls.get("aten::index_put_", 0), copy=calls.get("aten::copy_", 0))


def dense():
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
    nw = torch.ones(H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w_qkv, w_proj = param(1536, H), param(H, H)
    s1, s2 = param(2 * F, H), param(H, F)
    grad_qkv = torch.randn(T, 1536, device="cuda", dtype=torch.bfloat16)
    grad_h = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

    def coat_qkv():
        ca.invalidate_weights()
        ca.NormLinear.apply(x, nw, w_qkv, EPS).backward(grad_qkv)

    def coat_proj():
        ca.invalidate_weights()
        ca.Linear.apply(x, w_proj).backward(grad_h)

    def coat_shared():
        ca.invalidate_weights()
        ln_bf16, ln_x, ln_s, ln_t = ca.Norm.apply(x, nw, EPS)
        ca.SharedMLP.apply(ln_bf16, ln_x, ln_s, ln_t, s1, s2).backward(grad_h)

    te_qkv = te.LayerNormLinear(H, 1536, bias=False, normalization="RMSNorm", params_dtype=torch.bfloat16)
    te_proj = te.Linear(H, H, bias=False, params_dtype=torch.bfloat16)
    te_norm = te.RMSNorm(H, params_dtype=torch.bfloat16)
    te_fc1 = te.Linear(H, 2 * F, bias=False, params_dtype=torch.bfloat16)
    te_fc2 = te.Linear(F, H, bias=False, params_dtype=torch.bfloat16)

    def bf16_shared():
        gate, up = te_fc1(te_norm(x)).chunk(2, -1)
        te_fc2(torch.nn.functional.silu(gate) * up).backward(grad_h)

    report("dense_fwd_bwd_ms",
           qkv={"coat": timed(coat_qkv), "te_bf16": timed(lambda: te_qkv(x).backward(grad_qkv))},
           proj={"coat": timed(coat_proj), "te_bf16": timed(lambda: te_proj(x).backward(grad_h))},
           norm_plus_shared_expert={"coat": timed(coat_shared), "te_bf16": timed(bf16_shared)})


if __name__ == "__main__":
    for part in sys.argv[1:] or ["experts", "dense"]:
        {"experts": experts, "dense": dense}[part]()
