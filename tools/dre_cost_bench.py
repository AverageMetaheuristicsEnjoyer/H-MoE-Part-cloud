"""What COAT's dynamic range expansion costs per optimizer step (customer feedback 2026-10-05, item 2).

    python tools/dre_cost_bench.py ADAM_FULL_CHECKPOINT [--muon MUON_FULL_CHECKPOINT] [--repeats 20]

Real Adam moments of the checkpoint (all parameters, 1.03 B) drive four optimizer steps on one GPU, timed with
CUDA events (median of `repeats` after 3 warm-up steps):
  fusedadam_fp32   TE FusedAdam, FP32 moments (the series' FP32-state reference)
  coat             MCoreCoatAdamW: the upstream fused kernel as shipped (decode -> Adam math -> stats, k -> encode)
  coat_nopow       same kernel compiled with powf(x, e) -> x: the range expansion / contraction itself removed,
                   everything else (group max/min, log2, k, scales) kept
  coat_nopow_nok   additionally log2f(x) -> 1 (k == 1, no exponent search): the max-abs-with-sqrt(min*max) structure
coat - coat_nopow is the price of the powers, coat_nopow - coat_nopow_nok the price of computing k.  The elementwise
block times a copy, abs().sqrt_() and abs().pow_(1/k) over one 1.03 B FP32 tensor (bandwidth floor vs the
transcendental), and the Muon block times the COAT momentum codec (quantize + dequantize, upstream kernels) against
the series' Triton max-abs codec on the Muon momentum of --muon.
The ablated kernels change the arithmetic (they are timing variants only); they are never used for training.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "Megatron-LM"))
sys.path.insert(0, str(ROOT / "third_party" / "emerging-optimizers"))

import torch

from stage3_moe import coat_kernels
from stage3_moe.optimizer_states import StateSpec, dequantize_fp8_state, init_fp8_state, quantize_fp8_state_
from tools import quantization_diagnostics as qd

LR, WD, BETAS, EPS, STEP = 1.63e-3, 0.1, (0.9, 0.95), 1e-8, 13794


def report(name, **values):
    print("COAT_DREBENCH " + json.dumps({"check": name, **values}), flush=True)


def timed(fn, repeats, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times), "repeats": repeats}


def adam_moments(checkpoint):
    """FP32 (or decoded) exp_avg / exp_avg_sq of every Adam-state parameter, in checkpoint order."""
    moments = []
    for block in qd.checkpoint_blocks(checkpoint["optimizer"]):
        for group in block["param_groups"]:
            for key in group["params"]:
                entry = block["state"].get(key)
                if isinstance(entry, dict) and "exp_avg" in entry and "exp_avg_sq" in entry:
                    moments.append((qd.materialize(entry, "exp_avg"), qd.materialize(entry, "exp_avg_sq")))
    return moments


def muon_momenta(checkpoint):
    out = []
    for block in qd.checkpoint_blocks(checkpoint["optimizer"]):
        for group in block["param_groups"]:
            for key in group["params"]:
                entry = block["state"].get(key)
                if isinstance(entry, dict) and "momentum_buffer" in entry:
                    out.append(qd.materialize(entry, "momentum_buffer"))
    return out


def build_optimizer(cls, moments, **kwargs):
    params = [torch.nn.Parameter(torch.zeros_like(m)) for m, _ in moments]
    for p in params:
        p.grad = torch.empty_like(p)
    optimizer = cls(params, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, **kwargs)
    optimizer.load_state_dict({
        "state": {i: {"exp_avg": m.clone(), "exp_avg_sq": v.clone()} for i, (m, v) in enumerate(moments)},
        "param_groups": [{**{k: v for k, v in optimizer.param_groups[0].items() if k != "params"},
                          "params": list(range(len(params))), "step": STEP}],
    })
    return optimizer, params


def set_grads(params, moments, generator):
    for p, (m, _) in zip(params, moments):
        p.grad.copy_(m).add_(torch.randn(m.shape, device=m.device, generator=generator), alpha=float(m.abs().mean()) * 0.5)


def kernel_variant(name):
    text = coat_kernels.KERNEL_FILE.read_text()
    if name == "coat":
        return None
    prologue = "#define COAT_POWF(x, y) (x)\n"
    text = text.replace("powf(", "COAT_POWF(")
    if name == "coat_nopow_nok":
        prologue += "#define COAT_LOG2F(x) (1.0f)\n"
        text = text.replace("log2f(", "COAT_LOG2F(")
    return prologue + text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("adam_checkpoint", type=Path)
    parser.add_argument("--muon", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--stored-codec", choices=("auto", "maxabs", "dre", "coat"), default="coat")
    args = parser.parse_args()
    qd.STORED_CODEC = args.stored_codec
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(0)
    started = time.monotonic()

    checkpoint = torch.load(args.adam_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    moments = adam_moments(checkpoint)
    numel = sum(m.numel() for m, _ in moments)
    report("adam_states", parameters=len(moments), elements=numel, gib_fp32_moments=round(numel * 8 / 2**30, 3),
           iteration=checkpoint.get("iteration"), load_seconds=round(time.monotonic() - started, 1))

    from megatron.core.optimizer import Adam as FusedAdam
    from stage3_moe.coat_adamw import MCoreCoatAdamW

    results = {}
    optimizer, params = build_optimizer(FusedAdam, moments, bias_correction=True, adam_w_mode=True)
    set_grads(params, moments, generator)
    results["fusedadam_fp32"] = timed(optimizer.step, args.repeats)
    report("step", optimizer="fusedadam_fp32", **results["fusedadam_fp32"])
    del optimizer, params
    torch.cuda.empty_cache()

    optimizer, params = build_optimizer(MCoreCoatAdamW, moments)
    set_grads(params, moments, generator)
    index = device.index
    for variant in ("coat", "coat_nopow", "coat_nopow_nok"):
        coat_kernels._KERNELS[index] = coat_kernels._Kernels(index, source_text=kernel_variant(variant))
        results[variant] = timed(optimizer.step, args.repeats)
        report("step", optimizer=variant, **results[variant])
    coat_kernels._KERNELS[index] = coat_kernels._Kernels(index)
    report("dre_price", pow_ms=round(results["coat"]["median_ms"] - results["coat_nopow"]["median_ms"], 3),
           k_ms=round(results["coat_nopow"]["median_ms"] - results["coat_nopow_nok"]["median_ms"], 3),
           coat_over_fp32_ms=round(results["coat"]["median_ms"] - results["fusedadam_fp32"]["median_ms"], 3),
           train_step_s_adamw_dp4=2.144, train_step_s_adamw_1gpu_mb16=5.80)
    del optimizer, params
    torch.cuda.empty_cache()

    # Elementwise floor: one tensor the size of all moments of one kind.
    x = torch.cat([m.reshape(-1) for m, _ in moments])
    y = torch.empty_like(x)
    groups = x.numel() // 128
    exponents = torch.empty(groups, device=device).uniform_(1.0, 3.0)
    elementwise = {
        "copy": lambda: y.copy_(x),
        "abs_sqrt": lambda: torch.abs(x, out=y).sqrt_(),
        "abs_pow_const": lambda: torch.abs(x, out=y).pow_(1 / 2.37),
        "abs_pow_group_exponent": lambda: torch.abs(x, out=y)[: groups * 128].view(groups, 128).pow_(exponents.unsqueeze(1)),
    }
    bytes_moved = x.numel() * 4 * 2
    for name, fn in elementwise.items():
        t = timed(fn, args.repeats)
        report("elementwise", op=name, elements=x.numel(), gb_per_s=round(bytes_moved / t["median_ms"] / 1e6, 1), **t)
    del x, y, moments
    torch.cuda.empty_cache()

    if args.muon is not None:
        muon_ckpt = torch.load(args.muon, map_location="cpu", weights_only=False, mmap=True)
        momenta = muon_momenta(muon_ckpt)
        total = sum(m.numel() for m in momenta)
        report("muon_states", parameters=len(momenta), elements=total)
        for recipe in ("coat", "maxabs", "dre"):
            spec = StateSpec("momentum_buffer", True, torch.float8_e4m3fn, recipe)
            states = []
            for m in momenta:
                state = {}
                init_fp8_state(state, spec, m, group_size=128)
                states.append(state)

            def quantize():
                for m, state in zip(momenta, states):
                    quantize_fp8_state_(state, spec, m, group_size=128)

            def dequantize():
                for state in states:
                    dequantize_fp8_state(state, spec, group_size=128)

            report("muon_codec", recipe=recipe, quantize=timed(quantize, args.repeats), dequantize=timed(dequantize, args.repeats))
            del states
            torch.cuda.empty_cache()
    report("done", elapsed_seconds=round(time.monotonic() - started, 1))
    print("DRE_COST_BENCH=PASS", flush=True)


if __name__ == "__main__":
    main()
