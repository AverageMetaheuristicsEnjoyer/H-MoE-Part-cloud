"""Outlier / dynamic-range statistics of optimizer states (customer feedback 2026-10-05, COAT Fig. 2 style).

    python tools/state_outliers.py CHECKPOINT --output OUT.json --tag ARM_ID [--stored-codec coat|maxabs|dre]

Per parameter role (shape-classified) and per state tensor (exp_avg, exp_avg_sq, exp_avg_slow, momentum_buffer):
  1. distribution of log10|x| (0.1-dex bins), zeros, quantiles, max/RMS, share of the squared norm in the
     top 1 % / 0.1 % elements;
  2. per group of 128: dynamic range R = max|x| / min_nonzero|x| and the DRE exponent
     k = floor(16 * log2(448^2 / 2) / log2 R) / 16 (upstream kernel), histograms (COAT Fig. 2c);
  3. E4M3 range utilisation: log2 of the code |q| = 448 |x| / max under max-abs scaling (Fig. 2a) and
     |q'| = 448 (|x| / max)^k after expansion (Fig. 2b);
  4. round-trip error of max-abs E4M3, DRE E4M3 (series Triton codec) and COAT E4M3 (upstream kernel):
     relative L2 and cosine per role, and the same for Adam's direction m_hat / (sqrt(v_hat) + eps).
States already stored in FP8 (COAT-states checkpoints) are decoded first; for them (4) measures a second
round trip, not the training-time error (the FP32 original is gone).
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "Megatron-LM"))
sys.path.insert(0, str(ROOT / "third_party" / "emerging-optimizers"))

import torch

from stage3_moe.optimizer_states import (
    FP8_DTYPES,
    StateSpec,
    dequantize_fp8_state,
    init_fp8_state,
    quantize_fp8_state_,
)
from tools import quantization_diagnostics as qd

STATE_NAMES = ("exp_avg", "exp_avg_sq", "exp_avg_slow", "momentum_buffer")
SIGNED = {"exp_avg": True, "exp_avg_sq": False, "exp_avg_slow": True, "momentum_buffer": True}
# Stage 3 MoE 1.03B (configs/stage3-moe-1p029b.sh): H 1024, 8 heads x 128, 2 KV groups, dense FFN 2816,
# experts 64 x 256 (+ shared 256), vocab 50257.
ROLE_BY_SHAPE = {
    (50257, 1024): "embedding_or_head",
    (1536, 1024): "attn_qkv",
    (1024, 1024): "attn_proj",
    (5632, 1024): "dense_fc1",
    (1024, 2816): "dense_fc2",
    (512, 1024): "expert_fc1_routed_or_shared",
    (1024, 256): "expert_fc2_routed_or_shared",
    (64, 1024): "router",
    (1024,): "norm",
}
E4M3_MAX = 448.0
RATIO_UPPER = E4M3_MAX * E4M3_MAX / 2.0
LOG10_EDGES = (-14.0, 2.0, 160)
LOG10R_EDGES = (0.0, 40.0, 160)
K_EDGES = (0.0, 32.0, 128)
CODE_EDGES = (-12.0, 9.0, 84)
QUANTILES = (0.5, 0.9, 0.99, 0.999, 0.9999)
CODECS = (("maxabs", torch.float8_e4m3fn), ("dre", torch.float8_e4m3fn), ("coat", torch.float8_e4m3fn))


def hist(values, edges):
    low, high, bins = edges
    return torch.histc(values.clamp(low, high - 1e-6), bins=bins, min=low, max=high).to(torch.int64)


class Accumulator:
    def __init__(self):
        self.count = 0
        self.zeros = 0
        self.tensors = 0
        self.sumsq = 0.0
        self.max_abs = 0.0
        self.top1_sq = 0.0
        self.top01_sq = 0.0
        self.hist_log10 = None
        self.hist_log10r = None
        self.hist_k = None
        self.hist_code_maxabs = None
        self.hist_code_dre = None
        self.groups = 0
        self.constant_groups = 0
        self.samples = []
        self.errors = defaultdict(lambda: {"err2": 0.0, "ref2": 0.0, "dot": 0.0, "hat2": 0.0, "new_zeros": 0})

    def add_hist(self, name, values):
        current = getattr(self, name)
        setattr(self, name, values if current is None else current + values)


def role_of(shape):
    return ROLE_BY_SHAPE.get(tuple(shape), "other" + "x".join(map(str, shape)))


def roundtrip(value, name, recipe, dtype):
    spec = StateSpec(name, SIGNED[name], dtype, recipe)
    state = {}
    init_fp8_state(state, spec, value, group_size=128)
    quantize_fp8_state_(state, spec, value, group_size=128)
    return dequantize_fp8_state(state, spec, group_size=128)


def accumulate_error(bucket, reference, restored):
    delta = (restored - reference).double()
    bucket["err2"] += float(delta.pow(2).sum())
    bucket["ref2"] += float(reference.double().pow(2).sum())
    bucket["dot"] += float((reference.double() * restored.double()).sum())
    bucket["hat2"] += float(restored.double().pow(2).sum())
    bucket["new_zeros"] += int(((reference != 0) & (restored == 0)).sum())


def analyse(value, name, acc):
    flat = value.reshape(-1)
    magnitude = flat.abs()
    nonzero = magnitude[magnitude > 0]
    acc.count += flat.numel()
    acc.zeros += flat.numel() - nonzero.numel()
    acc.tensors += 1
    acc.sumsq += float(flat.double().pow(2).sum())
    acc.max_abs = max(acc.max_abs, float(magnitude.max()))
    if nonzero.numel():
        acc.add_hist("hist_log10", hist(nonzero.log10(), LOG10_EDGES))
        stride = max(1, (nonzero.numel() + 65535) // 65536)
        acc.samples.append(nonzero[::stride].cpu())
    sq = magnitude.double().pow(2)
    for share, attr in ((0.01, "top1_sq"), (0.001, "top01_sq")):
        k = max(1, int(share * flat.numel()))
        setattr(acc, attr, getattr(acc, attr) + float(torch.topk(sq, k).values.sum()))
    groups = flat.numel() // 128
    if groups:
        g = magnitude[: groups * 128].view(groups, 128)
        gmax = g.max(dim=1).values
        gmin = g.masked_fill(g == 0, float("inf")).min(dim=1).values
        valid = (gmax > 0) & torch.isfinite(gmin)
        ratio = (gmax[valid] / gmin[valid]).double()
        acc.groups += int(valid.sum())
        constant = ratio <= 1.0
        acc.constant_groups += int(constant.sum())
        ratio = ratio[~constant]
        if ratio.numel():
            acc.add_hist("hist_log10r", hist(ratio.log10().float(), LOG10R_EDGES))
            k = torch.floor(16.0 * math.log2(RATIO_UPPER) / torch.log2(ratio)) / 16.0
            acc.add_hist("hist_k", hist(k.float(), K_EDGES))
        # Codes under max-abs scaling and after expansion, per group (k = 1 for constant groups).
        gv = g[valid]
        scaled = gv / gmax[valid].unsqueeze(1)
        code_max = scaled * E4M3_MAX
        k_full = torch.ones(valid.sum(), device=g.device, dtype=torch.float64)
        k_full[~constant] = k
        code_dre = scaled.double().pow(k_full.unsqueeze(1)) * E4M3_MAX
        positive = code_max > 0
        acc.add_hist("hist_code_maxabs", hist(code_max[positive].log2().float(), CODE_EDGES))
        acc.add_hist("hist_code_dre", hist(code_dre[positive].float().log2(), CODE_EDGES))
    for recipe, dtype in CODECS:
        accumulate_error(acc.errors[f"{recipe}_{str(dtype).split('.')[-1]}"], value, roundtrip(value, name, recipe, dtype))


def adam_direction(m, v, step, beta1=0.9, beta2=0.95, eps=1e-8):
    m_hat = m / (1 - beta1 ** step)
    v_hat = v / (1 - beta2 ** step)
    return m_hat / (v_hat.sqrt() + eps)


def finish(acc):
    out = {"count": acc.count, "tensors": acc.tensors, "zeros": acc.zeros, "zero_fraction": acc.zeros / max(acc.count, 1),
           "rms": math.sqrt(acc.sumsq / max(acc.count, 1)), "max_abs": acc.max_abs,
           "max_over_rms": acc.max_abs / math.sqrt(acc.sumsq / max(acc.count, 1)) if acc.sumsq > 0 else None,
           "top1pct_norm_share": acc.top1_sq / acc.sumsq if acc.sumsq > 0 else None,
           "top0p1pct_norm_share": acc.top01_sq / acc.sumsq if acc.sumsq > 0 else None,
           "groups": acc.groups, "constant_groups": acc.constant_groups}
    if acc.samples:
        sample = torch.cat(acc.samples).double()
        if sample.numel() > 1 << 22:
            sample = sample[:: (sample.numel() + (1 << 22) - 1) // (1 << 22)]
        q = torch.quantile(sample, torch.tensor(QUANTILES, dtype=torch.float64))
        out["abs_quantiles_nonzero_sample"] = dict(zip(map(str, QUANTILES), q.tolist()))
        out["sample_size"] = sample.numel()
    for key, edges in (("hist_log10", LOG10_EDGES), ("hist_log10r", LOG10R_EDGES), ("hist_k", K_EDGES),
                       ("hist_code_maxabs", CODE_EDGES), ("hist_code_dre", CODE_EDGES)):
        values = getattr(acc, key)
        out[key] = {"edges": edges, "counts": values.tolist() if values is not None else None}
    out["roundtrip"] = {codec: {"relative_l2": math.sqrt(b["err2"] / b["ref2"]) if b["ref2"] > 0 else None,
                                "cosine": b["dot"] / math.sqrt(b["ref2"] * b["hat2"]) if b["ref2"] > 0 and b["hat2"] > 0 else None,
                                "new_zeros": b["new_zeros"]}
                       for codec, b in acc.errors.items()}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--stored-codec", choices=("auto", "maxabs", "dre", "coat"), default="coat")
    args = parser.parse_args()
    qd.STORED_CODEC = args.stored_codec
    started = time.monotonic()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    iteration = checkpoint.get("iteration", checkpoint.get("itr"))
    stats = defaultdict(Accumulator)
    direction = defaultdict(lambda: defaultdict(lambda: {"err2": 0.0, "ref2": 0.0, "dot": 0.0, "hat2": 0.0, "new_zeros": 0}))
    seen = 0
    for block in qd.checkpoint_blocks(checkpoint["optimizer"]):
        for gi, group in enumerate(block["param_groups"]):
            group_step = group.get("step")
            for key in group["params"]:
                entry = block["state"].get(key)
                if not isinstance(entry, dict):
                    continue
                names = [n for n in STATE_NAMES if n in entry and torch.is_tensor(entry[n])]
                if not names:
                    continue
                role = role_of(entry[names[0]].shape)
                values = {}
                for name in names:
                    stored_fp8 = entry[name].dtype in FP8_DTYPES
                    value = qd.materialize(entry, name)
                    values[name] = value
                    acc = stats[(role, name)]
                    acc.stored_fp8 = stored_fp8
                    analyse(value, name, acc)
                if "exp_avg" in values and "exp_avg_sq" in values:
                    step = entry.get("step", group_step)
                    step = int(step.item() if torch.is_tensor(step) else (step or 1))
                    reference = adam_direction(values["exp_avg"], values["exp_avg_sq"], step)
                    for recipe, dtype in CODECS:
                        restored = adam_direction(roundtrip(values["exp_avg"], "exp_avg", recipe, dtype),
                                                  roundtrip(values["exp_avg_sq"], "exp_avg_sq", recipe, dtype), step)
                        accumulate_error(direction[role][f"{recipe}_{str(dtype).split('.')[-1]}"], reference, restored)
                seen += 1
                del values
    if not seen:
        raise RuntimeError("no optimizer state found in the checkpoint")
    result = {"tag": args.tag, "checkpoint": str(args.checkpoint.name), "iteration": iteration, "parameters": seen,
              "roles": {}, "adam_direction": {}, "elapsed_seconds": round(time.monotonic() - started, 1)}
    for (role, name), acc in sorted(stats.items()):
        result["roles"].setdefault(role, {})[name] = {"stored_fp8": acc.stored_fp8, **finish(acc)}
    for role, codecs in direction.items():
        result["adam_direction"][role] = {codec: {"relative_l2": math.sqrt(b["err2"] / b["ref2"]) if b["ref2"] > 0 else None,
                                                  "cosine": b["dot"] / math.sqrt(b["ref2"] * b["hat2"]) if b["ref2"] > 0 and b["hat2"] > 0 else None}
                                          for codec, b in codecs.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print("COAT_OUTLIERS_META " + json.dumps({k: v for k, v in result.items() if k not in ("roles",)}, separators=(",", ":")), flush=True)
    for role, states in result["roles"].items():
        for name, payload in states.items():
            print("COAT_OUTLIERS " + json.dumps({"tag": args.tag, "role": role, "state": name, **payload}, separators=(",", ":")), flush=True)
    print(f"STATE_OUTLIERS=PASS parameters={seen} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
