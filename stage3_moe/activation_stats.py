"""Activation quantization metrics on a training batch (SOW Stage 3 item 2, robustness).

STAGE3_MOE_ACTIVATION_STATS_PATH=<jsonl> installs forward hooks on every layer's expert linear_fc1 /
linear_fc2 inputs and attention linear_proj input (the GEMM operands) and on the linear_fc1 output (the
SwiGLU halves). For STAGE3_MOE_ACTIVATION_STATS_STEPS forward passes (default 2) each hook records:
  * relative RMSE of an E4M3 cast under per-tensor, 1x128 and 1x16 scaling (TE tensorwise / blockwise / COAT);
  * the outlier ratio max|x| / rms(x) and the fraction of 1x128 blocks with ratio > 16;
  * for the fc1 output: the mean per-token correlation between the gate and up halves and the outlier ratio of
    silu(gate) * up -- the activation-level precursor of Fishman et al. (ICLR 2025).
Rows are sampled (every 26th row of the permuted expert input) so the hooks cost little; nothing is modified.
"""
import json
import os

import torch

F_E4M3 = 448.0


def _quant_error(x, group):
    """Relative RMSE of a fake E4M3 cast with one FP32 scale per `group` columns (0 = whole tensor)."""
    if group:
        cols = x.shape[1] - x.shape[1] % group
        blocks = x[:, :cols].reshape(x.shape[0], cols // group, group)
        scale = (blocks.abs().amax(dim=2, keepdim=True) / F_E4M3).clamp_min(1e-30)
        restored = (blocks / scale).to(torch.float8_e4m3fn).float() * scale
        err = (restored - blocks).pow(2).sum()
        ref = blocks.pow(2).sum()
    else:
        scale = (x.abs().max() / F_E4M3).clamp_min(1e-30)
        restored = (x / scale).to(torch.float8_e4m3fn).float() * scale
        err = (restored - x).pow(2).sum()
        ref = x.pow(2).sum()
    return (err / ref.clamp_min(1e-30)).sqrt().item()


def _operand_stats(x):
    x = x.detach().float().reshape(-1, x.shape[-1])
    x = x[::26] if x.shape[0] > 16384 else x
    rms = x.pow(2).mean().sqrt().clamp_min(1e-30)
    cols = x.shape[1] - x.shape[1] % 128
    blocks = x[:, :cols].reshape(x.shape[0], cols // 128, 128)
    block_ratio = blocks.abs().amax(dim=2) / blocks.pow(2).mean(dim=2).sqrt().clamp_min(1e-30)
    return {"rows": x.shape[0], "rel_rmse_tensor": _quant_error(x, 0), "rel_rmse_1x128": _quant_error(x, 128),
            "rel_rmse_1x16": _quant_error(x, 16), "outlier_ratio": (x.abs().max() / rms).item(),
            "max_abs": x.abs().max().item(), "blocks_ratio_gt16": (block_ratio > 16).float().mean().item()}


def _swiglu_stats(y):
    y = y.detach().float().reshape(-1, y.shape[-1])
    y = y[::26] if y.shape[0] > 16384 else y
    gate, up = y.chunk(2, dim=1)
    g = gate - gate.mean(dim=1, keepdim=True)
    u = up - up.mean(dim=1, keepdim=True)
    corr = (g * u).sum(1) / (g.norm(dim=1) * u.norm(dim=1)).clamp_min(1e-12)
    product = torch.nn.functional.silu(gate) * up
    return {"gate_up_corr_mean": corr.mean().item(), "gate_up_corr_abs_p99": corr.abs().quantile(0.99).item(),
            "product_outlier_ratio": (product.abs().max() / product.pow(2).mean().sqrt().clamp_min(1e-30)).item(),
            "product_max_abs": product.abs().max().item()}


def install(models, path, steps):
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    handles = []
    counter = {"calls": 0}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = open(path, "a")

    def emit(row):
        out.write(json.dumps(row) + "\n")
        out.flush()

    def pre_hook(name, kind):
        def hook(module, args):
            if counter["calls"] >= steps * len(handles):
                return
            counter["calls"] += 1
            emit({"module": name, "kind": kind, **_operand_stats(args[0])})
        return hook

    def fc1_output_hook(name):
        def hook(module, args, output):
            if counter["calls"] >= steps * len(handles):
                return
            counter["calls"] += 1
            y = output[0] if isinstance(output, (tuple, list)) else output
            emit({"module": name, "kind": "fc1_output", **_swiglu_stats(y)})
        return hook

    for model in models:
        module = model
        while hasattr(module, "module"):
            module = module.module
        for name, child in module.named_modules():
            if name.endswith("mlp.experts.linear_fc1"):
                handles.append(child.register_forward_pre_hook(pre_hook(name, "fc1_input")))
                handles.append(child.register_forward_hook(fc1_output_hook(name)))
            elif name.endswith("mlp.experts.linear_fc2"):
                handles.append(child.register_forward_pre_hook(pre_hook(name, "fc2_input")))
            elif name.endswith("self_attention.linear_proj"):
                handles.append(child.register_forward_pre_hook(pre_hook(name, "proj_input")))
    emit({"kind": "protocol", "hooks": len(handles), "steps": steps, "row_stride": 26})
