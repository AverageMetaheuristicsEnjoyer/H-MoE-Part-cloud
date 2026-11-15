"""Official COAT AdamW (NVlabs/COAT@80ec99f) as MCore's Adam.

The optimizer is upstream `coat/optimizer/fp8_adamw.py`, imported unmodified from
`third_party/coat`; its `qoptim_cuda` extension module is bound to the NVRTC build in
`coat_kernels`. `MCoreCoatAdamW` adds only what MCore needs and upstream lacks:

  - MCore's constructor keywords (`adam_w_mode`, `capturable`), restricted to the values the
    upstream kernel implements (decoupled weight decay, no CUDA graphs);
  - loading a checkpoint whose Adam moments are FP32 (written by TE FusedAdam): per-parameter
    `step` comes from TE's per-group counter and the moments are quantized with COAT's own
    quantization code. A checkpoint written by this class loads through upstream unchanged.

Quantization settings are upstream's defaults: E4M3 for both moments, groups of 128,
dynamic range expansion on, expand_min 16.
"""
import importlib.util
import sys
import types
from pathlib import Path

import torch

from stage3_moe import coat_kernels

UPSTREAM = Path(__file__).resolve().parents[1] / "third_party/coat/coat"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_upstream():
    if "coat_upstream.optimizer.fp8_adamw" in sys.modules:
        return sys.modules["coat_upstream.utils._fp8_quantization_config"], sys.modules["coat_upstream.optimizer.fp8_adamw"]
    for package, directory in (("coat_upstream", UPSTREAM), ("coat_upstream.utils", UPSTREAM / "utils"),
                               ("coat_upstream.optimizer", UPSTREAM / "optimizer")):
        module = types.ModuleType(package)
        module.__path__ = [str(directory)]
        sys.modules[package] = module
    extension = types.ModuleType("qoptim_cuda")
    extension.fp8_adamw_expand_step = coat_kernels.fp8_adamw_expand_step

    def unported(*args, **kwargs):
        raise NotImplementedError("only COAT's dynamic-range-expansion kernel is ported")

    extension.fp8_adamw_step = unported
    if "qoptim_cuda" in sys.modules and sys.modules["qoptim_cuda"].fp8_adamw_expand_step is not coat_kernels.fp8_adamw_expand_step:
        raise RuntimeError("another qoptim_cuda is already imported")
    sys.modules["qoptim_cuda"] = extension
    # The config module imports transformers.PretrainedConfig and never uses it.
    stub = None
    if importlib.util.find_spec("transformers") is None:
        stub = sys.modules["transformers"] = types.ModuleType("transformers")
        stub.PretrainedConfig = object
    try:
        config = _load("coat_upstream.utils._fp8_quantization_config", UPSTREAM / "utils/_fp8_quantization_config.py")
    finally:
        if stub is not None:
            del sys.modules["transformers"]
    optimizer = _load("coat_upstream.optimizer.fp8_adamw", UPSTREAM / "optimizer/fp8_adamw.py")
    return config, optimizer


_config, _upstream = _load_upstream()
QuantizationConfig = _config.QuantizationConfig
CoatAdamW = _upstream.CoatAdamW


class MCoreCoatAdamW(CoatAdamW):
    def __init__(self, params, lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8,
                 capturable=False, adam_w_mode=True):
        if not adam_w_mode:
            raise ValueError("COAT's kernel implements decoupled weight decay (AdamW) only")
        if capturable:
            raise ValueError("COAT's optimizer is not CUDA-graph capturable")
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                         qargs=QuantizationConfig())
        if not self.qargs.first_order_expansion.lower() in ("expansion", "true"):
            raise ValueError("expected dynamic range expansion")

    @torch._disable_dynamo
    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        imported = {}
        groups = state_dict["param_groups"]
        state = dict(state_dict["state"])
        for group in groups:
            for key in group["params"]:
                entry = state.get(key)
                if not entry or "step" in entry:
                    continue
                # TE FusedAdam keeps one step counter per group; COAT keeps one per parameter.
                if set(entry) != {"exp_avg", "exp_avg_sq"} or "step" not in group:
                    raise ValueError(f"unrecognized FP32 Adam state for COAT import: keys {sorted(entry)}")
                step = group["step"]
                step = float(step.item() if torch.is_tensor(step) else step)
                imported[key] = (entry["exp_avg"], entry["exp_avg_sq"])
                state[key] = {"step": torch.tensor(step)}
        state_dict["state"] = state
        super().load_state_dict(state_dict)
        if not imported:
            return
        id_map = dict(zip((key for group in groups for key in group["params"]),
                          (p for group in self.param_groups for p in group["params"])))
        for key, (exp_avg, exp_avg_sq) in imported.items():
            parameter = id_map[key]
            target = self.state[parameter]
            groups_count = (parameter.numel() + self.qargs.qgroup_size - 1) // self.qargs.qgroup_size
            # The layout upstream `_init_group` creates for a new parameter.
            target["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float8_e4m3fn)
            target["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float8_e4m3fn)
            for name in ("scale_exp_avg", "expand_exp_avg", "sqrt_minmax_exp_avg",
                         "scale_exp_avg_sq", "expand_exp_avg_sq", "sqrt_minmax_exp_avg_sq"):
                target[name] = torch.zeros(groups_count, device=parameter.device, dtype=parameter.dtype)
            coat_kernels.import_fp32_moments(
                exp_avg.to(device=parameter.device, dtype=torch.float32).contiguous(),
                exp_avg_sq.to(device=parameter.device, dtype=torch.float32).contiguous(),
                target["exp_avg"], target["scale_exp_avg"], target["expand_exp_avg"], target["sqrt_minmax_exp_avg"],
                target["exp_avg_sq"], target["scale_exp_avg_sq"], target["expand_exp_avg_sq"], target["sqrt_minmax_exp_avg_sq"],
                self.qargs.expand_min, self.qargs.qgroup_size)
        print(f"COAT_STATE_IMPORT fp32_parameters={len(imported)}", flush=True)
