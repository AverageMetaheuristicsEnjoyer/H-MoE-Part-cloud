"""Stage-by-stage time of the COAT routed-expert paths at the real layer shapes (one GPU).

Prints COAT_STAGE json lines:
  stages     every upstream / grouped kernel call of one forward+backward, synchronized, and the
             remaining glue; with the weight cache cold (first micro-batch of a step) and warm.
             Weight quantization is not a stage of its own: cold minus warm is its cost.
  baselines  the BF16 block MCore runs (permute + TE grouped GEMMs + unpermute), with and without
             the permutation, and a plain PyTorch BF16 block as the dense code base runs it
  gemm       one forward GEMM in BF16 (cuBLAS), FP8 (cuBLASLt) and COAT's Triton kernel, at this
             model's shapes and at wider ones
"""
import json
import sys
import time

import torch
import transformer_engine.pytorch as te

from stage3_moe import coat_activation as ca
from stage3_moe import coat_fused_experts as fe
from stage3_moe import coat_grouped_gemm as gg

T, H, F, E, TOPK, EPS = 13 * 2048, 1024, 256, 64, 8, 1e-5
rq = ca.rq


def report(name, **values):
    print("COAT_STAGE " + json.dumps({"check": name, **values}), flush=True)


def timed(fn, repeat=7, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    return round(sorted(times)[len(times) // 2], 3)


class Stages:
    """Wraps functions so each call is timed between two synchronizations."""

    def __init__(self):
        self.ms, self.calls, self.originals = {}, {}, []

    def wrap(self, owner, name):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))

        def wrapped(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = original(*args, **kwargs)
            torch.cuda.synchronize()
            self.ms[name] = self.ms.get(name, 0.0) + (time.perf_counter() - start) * 1e3
            self.calls[name] = self.calls.get(name, 0) + 1
            return out

        setattr(owner, name, wrapped)

    def restore(self):
        for owner, name, original in self.originals:
            setattr(owner, name, original)


def param(*shape, std=0.02):
    return torch.nn.Parameter(torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * std)


def routing(seed=0, skew=0.0):
    """Uniform routing, or with `skew` most tokens on a few experts and some experts empty."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand(T, E, device="cuda", generator=g) + skew * torch.linspace(0, 1, E, device="cuda") ** 4
    routing_map = torch.zeros(T, E, dtype=torch.bool, device="cuda").scatter_(1, scores.topk(TOPK, -1).indices, True)
    probs = torch.where(routing_map, scores, torch.zeros((), device="cuda")).requires_grad_(True)
    return probs, routing_map


def stage_breakdown(function, name, x, nw, fc1, fc2, probs, routing_map, grad, cold, with_ln_t=False):
    def run():
        if cold:
            ca.invalidate_weights()
        ln_bf16, ln_x, ln_s, ln_t = ca.Norm.apply(x, nw, EPS)
        fp8 = (ln_x, ln_s, ln_t) if with_ln_t else (ln_x, ln_s)
        out = function.apply(ln_bf16, probs, *fp8, routing_map, TOPK, *fc1, *fc2)
        out.backward(grad)

    plain = timed(run)
    stages = Stages()
    for owner, names in ((rq, ("fp8_quantize", "fp8_silu_forward", "fp8_division_transpose", "fp8_quantize_pertensor_transpose",
                               "fp8_mul_backward", "fp8_silu_backward", "fp8_rmsnorm_forward", "fp8_rmsnorm_backward",
                               "fp8_division", "fp8_quantize_pertensor")),
                         (ca, ("_mul_forward_unquantized", "_mm")),
                         (gg, ("segment_gemm", "segment_wgrad", "gather_sum", "probs_backward", "tile_experts")),
                         (fe, ("_fc1", "_act", "_down_dgrad", "_fc1_dgrad", "_wgrad", "_stacked_weights"))):
        for item in names:
            stages.wrap(owner, item)
    repeat = 5
    for _ in range(2):
        run()
    stages.ms.clear()
    stages.calls.clear()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        run()
    torch.cuda.synchronize()
    total = (time.perf_counter() - start) * 1e3 / repeat
    stages.restore()
    per = {k: round(v / repeat, 3) for k, v in sorted(stages.ms.items(), key=lambda kv: -kv[1])}
    report("stages", path=name, weights="cold" if cold else "warm", fwd_bwd_ms=plain, fwd_bwd_ms_synchronized=round(total, 3),
           stage_ms=per, glue_ms=round(total - sum(per.values()), 3))
    report("stage_calls", path=name, weights="cold" if cold else "warm", calls={k: v // repeat for k, v in stages.calls.items()})


def experts():
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
    nw = torch.ones(H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fc1 = [param(2 * F, H) for _ in range(E)]
    fc2 = [param(H, F) for _ in range(E)]
    probs, routing_map = routing()
    grad = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    for function, name in ((ca.Experts, "loop"), (ca.GroupedExperts, "grouped")):
        for cold in (True, False):
            stage_breakdown(function, name, x, nw, fc1, fc2, probs, routing_map, grad, cold)

    # The BF16 block as MCore runs it: RMSNorm, permute, grouped fc1, SiLU*up*probs, grouped fc2, unpermute.
    norm = te.RMSNorm(H, params_dtype=torch.bfloat16)
    grouped1 = te.GroupedLinear(E, H, 2 * F, bias=False, params_dtype=torch.bfloat16)
    grouped2 = te.GroupedLinear(E, F, H, bias=False, params_dtype=torch.bfloat16)
    counts = routing_map.sum(0).tolist()
    expert_idx, token_idx = routing_map.t().nonzero(as_tuple=True)
    perm_probs = probs.detach()[token_idx, expert_idx].unsqueeze(-1)
    scatter_index = token_idx.unsqueeze(1).expand(-1, H)

    def mcore_like():
        permuted = norm(x).index_select(0, token_idx)
        gate, up = grouped1(permuted, counts).chunk(2, -1)
        y = grouped2((torch.nn.functional.silu(gate) * up * perm_probs).to(torch.bfloat16), counts)
        out = torch.zeros(T, H, dtype=y.dtype, device="cuda").scatter_add_(0, scatter_index, y)
        out.backward(grad)

    perm_input = torch.randn(T * TOPK, H, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
    perm_grad = torch.randn(T * TOPK, H, device="cuda", dtype=torch.bfloat16)

    def gemm_only():
        gate, up = grouped1(perm_input, counts).chunk(2, -1)
        grouped2((torch.nn.functional.silu(gate) * up * perm_probs).to(torch.bfloat16), counts).backward(perm_grad)

    def permutation_only():
        permuted = x.index_select(0, token_idx)
        torch.zeros(T, H, dtype=x.dtype, device="cuda").scatter_add_(0, scatter_index, permuted).backward(grad)

    report("baselines_fwd_bwd_ms", mcore_like_bf16=timed(mcore_like), te_grouped_gemm_and_activation_only=timed(gemm_only),
           permute_unpermute_only=timed(permutation_only), tokens=T, experts=E, topk=TOPK)

    # A dense SwiGLU MLP with the same GEMM work on unreplicated tokens: FFN width TOPK * F.
    wide1 = te.Linear(H, 2 * F * TOPK, bias=False, params_dtype=torch.bfloat16)
    wide2 = te.Linear(F * TOPK, H, bias=False, params_dtype=torch.bfloat16)
    d1, d2 = param(2 * F * TOPK, H), param(H, F * TOPK)

    def dense_te():
        gate, up = wide1(norm(x)).chunk(2, -1)
        wide2(torch.nn.functional.silu(gate) * up).backward(grad)

    def dense_coat():
        ca.NormMLP.apply(x, nw, d1, d2, EPS).backward(grad)

    report("same_flops_dense_mlp_fwd_bwd_ms", te_bf16=timed(dense_te), coat=timed(dense_coat), ffn=F * TOPK,
           note="coat: weight cache warm")


def fused():
    """FusedExperts against GroupedExperts on identical inputs: equality of every output, then time."""
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
    nw = torch.ones(H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fc1 = [param(2 * F, H) for _ in range(E)]
    fc2 = [param(H, F) for _ in range(E)]
    grad = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

    def outputs(function, with_ln_t, probs, routing_map):
        ca.invalidate_weights()
        for tensor in (x, nw, probs, *fc1, *fc2):
            tensor.grad = None
        ln_bf16, ln_x, ln_s, ln_t = ca.Norm.apply(x, nw, EPS)
        fp8 = (ln_x, ln_s, ln_t) if with_ln_t else (ln_x, ln_s)
        out = function.apply(ln_bf16, probs, *fp8, routing_map, TOPK, *fc1, *fc2)
        out.backward(grad)
        return {"out": out.detach(), "d_input": x.grad, "d_norm_weight": nw.grad, "d_probs": probs.grad,
                "d_fc1": torch.stack([w.grad for w in fc1]), "d_fc2": torch.stack([w.grad for w in fc2])}

    def compare(a, b):
        result = {}
        for name in a:
            u, v = a[name].float(), b[name].float()
            result[name] = {"bitwise": bool(torch.equal(a[name], b[name])),
                            "rel_l2": float((u - v).norm() / v.norm().clamp_min(1e-30)), "max_abs": float((u - v).abs().max())}
        return result

    variants = ("transposed", "hybrid", "rowmajor", "reference")

    def select(mode):
        fe.WGRAD = mode
        return mode

    for skew in (0.0, 1.0):
        probs, routing_map = routing(skew=skew)
        counts = routing_map.sum(0)
        try:
            reference = outputs(ca.GroupedExperts, False, probs, routing_map)
            again = outputs(ca.GroupedExperts, False, probs, routing_map)
            report("grouped_run_twice", skew=skew, **compare(again, reference))
            for mode in variants:
                label = select(mode)
                result = compare(outputs(fe.FusedExperts, True, probs, routing_map), reference)
                report("fused_vs_grouped", skew=skew, wgrad=label, empty_experts=int((counts == 0).sum()), max_count=int(counts.max()),
                       bitwise={k: v["bitwise"] for k, v in result.items()}, rel_l2={k: v["rel_l2"] for k, v in result.items()})
        except Exception as error:
            import traceback
            report("fused_vs_grouped", skew=skew, error=repr(error)[:400], trace=traceback.format_exc()[-1500:])
            return
        del reference, again

        def run(function, with_ln_t, cold):
            def step():
                if cold:
                    ca.invalidate_weights()
                ln_bf16, ln_x, ln_s, ln_t = ca.Norm.apply(x, nw, EPS)
                fp8 = (ln_x, ln_s, ln_t) if with_ln_t else (ln_x, ln_s)
                function.apply(ln_bf16, probs, *fp8, routing_map, TOPK, *fc1, *fc2).backward(grad)
            return timed(step)

        times = {"grouped_cold": run(ca.GroupedExperts, False, True), "grouped_warm": run(ca.GroupedExperts, False, False)}
        for mode in variants:
            label = select(mode)
            times[f"fused_{label}_warm"] = run(fe.FusedExperts, True, False)
            times[f"fused_{label}_cold"] = run(fe.FusedExperts, True, True)
        report("fused_fwd_bwd_ms", skew=skew, **times)
        for mode in variants:
            label = select(mode)
            stage_breakdown(fe.FusedExperts, f"fused_{label}_skew{skew}", x, nw, fc1, fc2, probs, routing_map, grad, False, with_ln_t=True)

    # The BF16 block MCore runs, on the last (skewed) routing: the baseline the uniform one has in `experts`.
    norm = te.RMSNorm(H, params_dtype=torch.bfloat16)
    grouped1 = te.GroupedLinear(E, H, 2 * F, bias=False, params_dtype=torch.bfloat16)
    grouped2 = te.GroupedLinear(E, F, H, bias=False, params_dtype=torch.bfloat16)
    expert_idx, token_idx = routing_map.t().nonzero(as_tuple=True)
    perm_probs = probs.detach()[token_idx, expert_idx].unsqueeze(-1)
    scatter_index = token_idx.unsqueeze(1).expand(-1, H)
    counts = counts.tolist()

    def mcore_like():
        permuted = norm(x).index_select(0, token_idx)
        gate, up = grouped1(permuted, counts).chunk(2, -1)
        y = grouped2((torch.nn.functional.silu(gate) * up * perm_probs).to(torch.bfloat16), counts)
        torch.zeros(T, H, dtype=y.dtype, device="cuda").scatter_add_(0, scatter_index, y).backward(grad)

    report("baseline_skewed_fwd_bwd_ms", mcore_like_bf16=timed(mcore_like), skew=1.0)


class TorchRMSNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim, device="cuda"))

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS)).type_as(x) * self.weight


def dense():
    """One dense transformer MLP block (norm + SwiGLU MLP) at three widths: COAT, TE BF16, PyTorch BF16."""
    for hidden, ffn, tokens in ((1024, 256, T), (1024, 4096, T), (4096, 11008, 4096)):
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
        grad = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        nw = torch.ones(hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w1, w2 = param(2 * ffn, hidden), param(hidden, ffn)
        te_norm = te.RMSNorm(hidden, params_dtype=torch.bfloat16)
        te1 = te.Linear(hidden, 2 * ffn, bias=False, params_dtype=torch.bfloat16)
        te2 = te.Linear(ffn, hidden, bias=False, params_dtype=torch.bfloat16)
        pt_norm = TorchRMSNorm(hidden)
        pt1 = torch.nn.Linear(hidden, 2 * ffn, bias=False, device="cuda")
        pt2 = torch.nn.Linear(ffn, hidden, bias=False, device="cuda")

        def coat():
            ca.NormMLP.apply(x, nw, w1, w2, EPS).backward(grad)

        def te_bf16():
            gate, up = te1(te_norm(x)).chunk(2, -1)
            te2(torch.nn.functional.silu(gate) * up).backward(grad)

        def torch_bf16():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gate, up = pt1(pt_norm(x)).chunk(2, -1)
                y = pt2(torch.nn.functional.silu(gate) * up)
            y.backward(grad)

        try:
            report("dense_mlp_fwd_bwd_ms", hidden=hidden, ffn=ffn, tokens=tokens, coat=timed(coat), te_bf16=timed(te_bf16),
                   torch_bf16=timed(torch_bf16), note="coat: weight cache warm")
        except Exception as error:  # a width one of the kernels cannot take is a result, not a failure
            report("dense_mlp_fwd_bwd_ms", hidden=hidden, ffn=ffn, tokens=tokens, error=repr(error)[:300])
        ca.invalidate_weights()
        del x, grad, w1, w2, te1, te2, pt1, pt2
        torch.cuda.empty_cache()


def gemm():
    """One forward GEMM y = x @ w.T, no gradient: TFLOP/s of BF16 cuBLAS, FP8 cuBLASLt, COAT Triton."""
    one = torch.ones((), device="cuda")
    for rows, k, n in ((T, 1024, 1536), (T, 1024, 512), (T * TOPK // E // 16 * 16, 1024, 512), (T, 1024, 8192),
                       (8192, 4096, 4096), (8192, 4096, 22016)):
        x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        x8, w8 = x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)
        scale = one.to(torch.bfloat16)
        flops = 2 * rows * k * n
        result = {}
        with torch.no_grad():
            for label, fn in (("bf16_cublas", lambda: x @ w.t()),
                              ("fp8_cublaslt", lambda: torch._scaled_mm(x8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)),
                              ("fp8_coat_triton", lambda: rq.fp8_linear_forward(x8, scale, w8, scale, False, 16)),
                              ("fp8_coat_triton_quantized_output", lambda: rq.fp8_linear_forward(x8, scale, w8, scale, True, 16))):
                try:
                    ms = timed(fn, repeat=9)
                    result[label] = {"ms": ms, "tflops": round(flops / ms / 1e9, 1)}
                except Exception as error:
                    result[label] = {"error": repr(error)[:200]}
        report("gemm_forward", rows=rows, k=k, n=n, **result)
        del x, w, x8, w8
        torch.cuda.empty_cache()


def bf16():
    """FusedExpertsBF16 against an FP32 autograd reference of the same expert MLP, then time."""
    from stage3_moe import bf16_fused_experts as bf

    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
    fc1 = [param(2 * F, H) for _ in range(E)]
    fc2 = [param(H, F) for _ in range(E)]
    grad = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    leaves = (x, *fc1, *fc2)
    for skew in (0.0, 1.0):
        probs, routing_map = routing(skew=skew)

        def gradients():
            values = {"d_input": x.grad, "d_probs": probs.grad, "d_fc1": torch.stack([w.grad for w in fc1]),
                      "d_fc2": torch.stack([w.grad for w in fc2])}
            for tensor in (*leaves, probs):
                tensor.grad = None
            return {k: v.float() for k, v in values.items()}

        bf._WEIGHTS.clear()
        for tensor in leaves:  # the timing runs of the previous routing left their gradients behind
            tensor.grad = None
        out = bf.FusedExpertsBF16.apply(x, probs, routing_map, TOPK, *fc1, *fc2)
        out.backward(grad)
        fused_out, fused = out.detach().float(), gradients()
        reference_out = torch.zeros(T, H, device="cuda")
        for e in range(E):
            tokens = routing_map[:, e].nonzero(as_tuple=True)[0]
            gate, up = (x.float()[tokens] @ fc1[e].float().t()).chunk(2, -1)
            y = (torch.nn.functional.silu(gate) * up * probs[tokens, e].float().unsqueeze(-1)) @ fc2[e].float().t()
            reference_out = reference_out.index_add(0, tokens, y)
        reference_out.backward(grad.float())
        reference = gradients()

        def rel(a, b):
            return float((a - b).norm() / b.norm().clamp_min(1e-30))

        report("bf16_fused_vs_fp32_reference", skew=skew, out_rel_l2=rel(fused_out, reference_out.detach()),
               **{k + "_rel_l2": rel(fused[k], reference[k]) for k in fused})

        norm = te.RMSNorm(H, params_dtype=torch.bfloat16)
        grouped1 = te.GroupedLinear(E, H, 2 * F, bias=False, params_dtype=torch.bfloat16)
        grouped2 = te.GroupedLinear(E, F, H, bias=False, params_dtype=torch.bfloat16)
        counts = routing_map.sum(0).tolist()
        expert_idx, token_idx = routing_map.t().nonzero(as_tuple=True)
        perm_probs = probs.detach()[token_idx, expert_idx].unsqueeze(-1)
        scatter_index = token_idx.unsqueeze(1).expand(-1, H)
        nw = torch.ones(H, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        def fused_step():
            bf.FusedExpertsBF16.apply(norm(x), probs, routing_map, TOPK, *fc1, *fc2).backward(grad)

        def fused_step_new_weights():
            bf._WEIGHTS.clear()
            fused_step()

        def mcore_like():
            permuted = norm(x).index_select(0, token_idx)
            gate, up = grouped1(permuted, counts).chunk(2, -1)
            y = grouped2((torch.nn.functional.silu(gate) * up * perm_probs).to(torch.bfloat16), counts)
            torch.zeros(T, H, dtype=y.dtype, device="cuda").scatter_add_(0, scatter_index, y).backward(grad)

        def coat_fused():
            ln_bf16, ln_x, ln_s, ln_t = ca.Norm.apply(x, nw, EPS)
            fe.FusedExperts.apply(ln_bf16, probs, ln_x, ln_s, ln_t, routing_map, TOPK, *fc1, *fc2).backward(grad)

        report("bf16_fused_fwd_bwd_ms", skew=skew, bf16_fused_warm=timed(fused_step), bf16_fused_cold=timed(fused_step_new_weights),
               mcore_like_bf16=timed(mcore_like), coat_fused_warm=timed(coat_fused))


def exact():
    """Which FP8 GEMM is exact? Both kernels against the float64 product of the same FP8 operands."""
    g = torch.Generator(device="cuda").manual_seed(0)
    one = torch.ones((), device="cuda")
    for rows, k, n in ((1024, 1024, 512), (1024, 256, 1024)):
        x8 = (torch.randn(rows, k, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
        w8 = torch.randn(n, k, device="cuda", generator=g).to(torch.float8_e4m3fn)
        reference = x8.double() @ w8.double().t()
        cublaslt = torch._scaled_mm(x8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.float32)
        tiles = torch.zeros(rows // gg.BLOCK_M, dtype=torch.int32, device="cuda")
        triton_out = gg.segment_gemm(x8, w8[None].contiguous(), one, torch.ones(1, device="cuda"), tiles,
                                     torch.empty(rows, n, dtype=torch.float32, device="cuda"))
        float32 = x8.float() @ w8.float().t()

        def error(y):
            return float((y.double() - reference).norm() / reference.norm())

        report("fp8_gemm_exactness", rows=rows, k=k, n=n, cublaslt_rel_error=error(cublaslt), triton_rel_error=error(triton_out),
               fp32_matmul_rel_error=error(float32), cublaslt_vs_triton_equal_share=float((cublaslt == triton_out).float().mean()),
               cublaslt_vs_triton_rel=float((cublaslt - triton_out).norm() / triton_out.norm()))


if __name__ == "__main__":
    for part in sys.argv[1:] or ["experts", "dense", "gemm", "fused"]:
        {"experts": experts, "dense": dense, "gemm": gemm, "fused": fused, "exact": exact, "bf16": bf16}[part]()
