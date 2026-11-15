"""One MoE layer's routed experts (norm, experts, combine), forward+backward, at several model sizes.

Does FP8 beat BF16 once the experts are wide? Prints MOE_LAYER json lines per shape:
  native_bf16          what MCore runs: permute, TE GroupedLinear fc1, SiLU*up*probs, GroupedLinear fc2, unpermute
  native_fp8_<recipe>  the same modules under TE's FP8 autocast (blockwise, tensorwise, delayed 1024/max)
  fused_bf16           stage3_moe.bf16_fused_experts
  fused_coat           stage3_moe.coat_fused_experts (COAT FP8 flow)
`warm` reuses the quantized weights of the previous call (micro-batches 2..n of a step), `cold` requantizes.
"""
import json
import sys
import time

import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe as te_recipe

from stage3_moe import bf16_fused_experts as bf
from stage3_moe import coat_activation as ca
from stage3_moe import coat_fused_experts as fe

EPS = 1e-5
# name: hidden, expert width, experts, top-k, tokens. Rows per expert = tokens * topk / experts.
SHAPES = {
    "stage3_moe_1b": (1024, 256, 64, 8, 26624),
    "qwen3_moe_30b_like": (2048, 768, 128, 8, 16384),
    "wide_experts_2048x2048": (2048, 2048, 64, 8, 8192),
    "deepseek_v3_like_64_of_256_experts": (7168, 2048, 64, 8, 8192),
}
_autocast = getattr(te, "autocast", None) or te.fp8_autocast


def autocast(recipe):
    try:
        return _autocast(enabled=True, recipe=recipe)
    except TypeError:
        return _autocast(enabled=True, fp8_recipe=recipe)


def report(name, **values):
    print("MOE_LAYER " + json.dumps({"check": name, **values}), flush=True)


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


def measure(fn):
    torch.cuda.reset_peak_memory_stats()
    try:
        ms = timed(fn)
        return {"ms": ms, "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    except Exception as error:
        return {"error": repr(error)[:300]}
    finally:
        torch.cuda.empty_cache()


def bench(name, hidden, ffn, experts, topk, tokens):
    per_expert = tokens * topk // experts
    assert tokens * topk % experts == 0 and per_expert % 128 == 0 and experts % topk == 0
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
    grad = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    # Token t goes to experts t + j * experts / topk: every expert receives exactly per_expert tokens.
    choice = (torch.arange(tokens, device="cuda")[:, None] + torch.arange(topk, device="cuda")[None, :] * (experts // topk)) % experts
    routing_map = torch.zeros(tokens, experts, dtype=torch.bool, device="cuda").scatter_(1, choice, True)
    probs = torch.where(routing_map, torch.rand(tokens, experts, device="cuda", generator=generator) + 0.05,
                        torch.zeros((), device="cuda")).requires_grad_(True)
    norm = te.RMSNorm(hidden, params_dtype=torch.bfloat16)
    grouped1 = te.GroupedLinear(experts, hidden, 2 * ffn, bias=False, params_dtype=torch.bfloat16)
    grouped2 = te.GroupedLinear(experts, ffn, hidden, bias=False, params_dtype=torch.bfloat16)
    fc1 = [getattr(grouped1, f"weight{i}") for i in range(experts)]
    fc2 = [getattr(grouped2, f"weight{i}") for i in range(experts)]
    nw = torch.ones(hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    counts = [per_expert] * experts
    expert_idx, token_idx = routing_map.t().nonzero(as_tuple=True)
    perm_probs = probs.detach()[token_idx, expert_idx].unsqueeze(-1)
    scatter_index = token_idx.unsqueeze(1).expand(-1, hidden)
    flops = 3 * 2 * tokens * topk * (2 * ffn * hidden + ffn * hidden)  # forward, data gradient, weight gradient

    def native(g1, g2, recipe=None, first=None):
        def step():
            permuted = norm(x).index_select(0, token_idx)
            if recipe is None:
                gate, up = g1(permuted, counts).chunk(2, -1)
                y = g2((torch.nn.functional.silu(gate) * up * perm_probs).to(torch.bfloat16), counts)
            else:
                with autocast(recipe):
                    gate, up = g1(permuted, counts, is_first_microbatch=first).chunk(2, -1)
                    y = g2((torch.nn.functional.silu(gate) * up * perm_probs).to(torch.bfloat16), counts, is_first_microbatch=first)
            torch.zeros(tokens, hidden, dtype=y.dtype, device="cuda").scatter_add_(0, scatter_index, y).backward(grad)
        return step

    def fused_bf16(cold):
        def step():
            if cold:
                bf._WEIGHTS.clear()
            bf.FusedExpertsBF16.apply(norm(x), probs, routing_map, topk, *fc1, *fc2).backward(grad)
        return step

    def fused_coat(cold):
        def step():
            if cold:
                ca.invalidate_weights()
            ln_bf16, ln_x, ln_s, ln_t = ca.Norm.apply(x, nw, EPS)
            fe.FusedExperts.apply(ln_bf16, probs, ln_x, ln_s, ln_t, routing_map, topk, *fc1, *fc2).backward(grad)
        return step

    result = {"native_bf16": measure(native(grouped1, grouped2))}
    recipes = {"blockwise": te_recipe.Float8BlockScaling(fp8_format=te_recipe.Format.E4M3),
               "tensorwise": te_recipe.Float8CurrentScaling(fp8_format=te_recipe.Format.HYBRID),
               "delayed1024": te_recipe.DelayedScaling(fp8_format=te_recipe.Format.HYBRID, amax_history_len=1024, amax_compute_algo="max")}
    for label, recipe in recipes.items():
        # A module keeps the FP8 metadata of its first recipe: fresh ones per recipe.
        g1 = te.GroupedLinear(experts, hidden, 2 * ffn, bias=False, params_dtype=torch.bfloat16)
        g2 = te.GroupedLinear(experts, ffn, hidden, bias=False, params_dtype=torch.bfloat16)
        result[f"native_fp8_{label}_cold"] = measure(native(g1, g2, recipe, True))
        result[f"native_fp8_{label}_warm"] = measure(native(g1, g2, recipe, False))
        del g1, g2
        torch.cuda.empty_cache()
    result["fused_bf16_warm"] = measure(fused_bf16(False))
    result["fused_bf16_cold"] = measure(fused_bf16(True))
    result["fused_coat_warm"] = measure(fused_coat(False))
    result["fused_coat_cold"] = measure(fused_coat(True))
    report("moe_layer_fwd_bwd", shape=name, hidden=hidden, expert_width=ffn, experts=experts, topk=topk, tokens=tokens,
           rows_per_expert=per_expert, gemm_tflop=round(flops / 1e12, 3), **result)

    # At widths the fused COAT path was not validated on: is it still the grouped path's arithmetic?
    if ffn != 256:
        try:
            outputs = {}
            for label, function, extra in (("grouped", ca.GroupedExperts, False), ("fused", fe.FusedExperts, True)):
                ca.invalidate_weights()
                for tensor in (x, probs, *fc1, *fc2):
                    tensor.grad = None
                ln_bf16, ln_x, ln_s, ln_t = ca.Norm.apply(x, nw, EPS)
                fp8 = (ln_x, ln_s, ln_t) if extra else (ln_x, ln_s)
                out = function.apply(ln_bf16, probs, *fp8, routing_map, topk, *fc1, *fc2)
                out.backward(grad)
                outputs[label] = {"out": out.detach(), "d_input": x.grad, "d_probs": probs.grad,
                                  "d_fc1": torch.stack([w.grad for w in fc1]), "d_fc2": torch.stack([w.grad for w in fc2])}
            report("fused_vs_grouped_wide", shape=name, bitwise={k: bool(torch.equal(outputs["fused"][k], outputs["grouped"][k])) for k in outputs["fused"]},
                   d_probs_rel=float((outputs["fused"]["d_probs"].float() - outputs["grouped"]["d_probs"].float()).norm()
                                     / outputs["grouped"]["d_probs"].float().norm()))
        except Exception as error:
            report("fused_vs_grouped_wide", shape=name, error=repr(error)[:300])
    ca.invalidate_weights()
    bf._WEIGHTS.clear()


if __name__ == "__main__":
    for shape in sys.argv[1:] or SHAPES:
        try:
            bench(shape, *SHAPES[shape])
        except Exception as error:
            report("moe_layer_fwd_bwd", shape=shape, error=repr(error)[:400])
        torch.cuda.empty_cache()
