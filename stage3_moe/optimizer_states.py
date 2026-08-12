from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from typing import Any, DefaultDict, Dict, Iterable

import torch
import triton
import triton.language as tl
from triton.backends.nvidia.compiler import CUDAOptions
from triton.language.extra import libdevice


GROUP_SIZE = 128
GROUPS_PER_BLOCK = 8
NUM_WARPS = 4
QUANT_EPS = 1e-30
FLOAT32_MAX = float(torch.finfo(torch.float32).max)
FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}
FTZ_KWARGS = (
    {"enable_reflect_ftz": False}
    if "enable_reflect_ftz" in CUDAOptions.__dataclass_fields__
    else {}
)

_EPS = tl.constexpr(QUANT_EPS)
_ONE_PLUS_EPS = tl.constexpr(1.0 + QUANT_EPS)


@dataclass(frozen=True)
class StateSpec:
    name: str
    signed: bool
    dtype: torch.dtype
    recipe: str


ADAM_STATE_SPECS = (
    StateSpec("exp_avg", True, torch.float8_e4m3fn, "dre"),
    StateSpec("exp_avg_sq", False, torch.float8_e5m2, "dre"),
)
MUON_STATE_SPECS = (
    StateSpec("momentum_buffer", True, torch.float8_e4m3fn, "maxabs"),
)


def num_groups(numel: int, group_size: int = GROUP_SIZE) -> int:
    return (numel + group_size - 1) // group_size


@triton.jit
def _dre_dequantize_kernel(
    data_ptr,
    out_ptr,
    scale_ptr,
    expand_ptr,
    sqrt_minmax_ptr,
    numel,
    num_state_groups,
    SIGNED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    offsets = rows[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    mask = offsets < numel
    group_mask = rows < num_state_groups

    scale = tl.load(scale_ptr + rows, mask=group_mask, other=0.0)
    expansion = tl.maximum(
        tl.load(expand_ptr + rows, mask=group_mask, other=1.0), _EPS
    )
    sqrt_minmax = tl.maximum(
        tl.load(sqrt_minmax_ptr + rows, mask=group_mask, other=1.0), _EPS
    )
    raw = tl.load(data_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    raw *= scale[:, None]
    if SIGNED:
        magnitude = tl.abs(raw)
        sign = tl.where(raw > 0.0, 1.0, tl.where(raw < 0.0, -1.0, 0.0))
        restored = sign * libdevice.pow(
            tl.maximum(magnitude, _EPS),
            libdevice.div_rn(1.0, expansion)[:, None],
        ) * sqrt_minmax[:, None]
        restored = tl.where(magnitude > 0.0, restored, 0.0)
    else:
        raw = tl.maximum(raw, 0.0)
        restored = libdevice.pow(
            tl.maximum(raw, _EPS),
            libdevice.div_rn(1.0, expansion)[:, None],
        ) * sqrt_minmax[:, None]
        restored = tl.where(raw > 0.0, restored, 0.0)
    tl.store(out_ptr + offsets, restored, mask=mask)


@triton.jit
def _dre_quantize_kernel(
    value_ptr,
    data_ptr,
    scale_ptr,
    expand_ptr,
    sqrt_minmax_ptr,
    numel,
    num_state_groups,
    fp8_max,
    ratio_upper,
    float32_max,
    SIGNED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    offsets = rows[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    mask = offsets < numel
    group_mask = rows < num_state_groups

    value = tl.load(value_ptr + offsets, mask=mask, other=0.0)
    if SIGNED:
        source = value
        magnitude = tl.abs(value)
    else:
        source = tl.maximum(value, 0.0)
        magnitude = source
    nonzero = magnitude > 0.0
    absmax = tl.maximum(tl.max(magnitude, axis=1), _EPS)
    absmin = tl.min(tl.where(nonzero, magnitude, float32_max), axis=1)
    absmin = tl.where(
        tl.max(nonzero.to(tl.int32), axis=1) != 0,
        tl.maximum(absmin, _EPS),
        _EPS,
    )

    ratio = tl.maximum(libdevice.div_rn(absmax, absmin), _ONE_PLUS_EPS)
    raw_expansion = libdevice.div_rn(
        libdevice.floor(
            libdevice.div_rn(
                libdevice.log2(ratio_upper),
                tl.maximum(libdevice.log2(ratio), _EPS),
            )
            * 16
        ),
        16.0,
    )
    expansion = tl.where(
        ratio <= _ONE_PLUS_EPS,
        1.0,
        tl.maximum(raw_expansion, 1.0 / 16),
    )
    sqrt_minmax = tl.maximum(tl.sqrt_rn(absmax) * tl.sqrt_rn(absmin), _EPS)
    normalized = libdevice.pow(
        tl.maximum(libdevice.div_rn(magnitude, sqrt_minmax[:, None]), _EPS),
        expansion[:, None],
    )
    normalized = tl.where(nonzero, normalized, 0.0)
    if SIGNED:
        normalized *= tl.where(source > 0.0, 1.0, tl.where(source < 0.0, -1.0, 0.0))
    scale = tl.maximum(
        libdevice.div_rn(
            libdevice.pow(
                tl.maximum(libdevice.div_rn(absmax, sqrt_minmax), _EPS),
                expansion,
            ),
            fp8_max,
        ),
        _EPS,
    )
    tl.store(
        data_ptr + offsets,
        libdevice.div_rn(normalized, scale[:, None]).to(data_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(scale_ptr + rows, scale, mask=group_mask)
    tl.store(expand_ptr + rows, expansion, mask=group_mask)
    tl.store(sqrt_minmax_ptr + rows, sqrt_minmax, mask=group_mask)


@triton.jit
def _maxabs_dequantize_kernel(
    data_ptr,
    out_ptr,
    scale_ptr,
    numel,
    num_state_groups,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    offsets = rows[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    mask = offsets < numel
    group_mask = rows < num_state_groups
    scale = tl.load(scale_ptr + rows, mask=group_mask, other=0.0)
    raw = tl.load(data_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, raw * scale[:, None], mask=mask)


@triton.jit
def _maxabs_quantize_kernel(
    value_ptr,
    data_ptr,
    scale_ptr,
    numel,
    num_state_groups,
    fp8_max,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    offsets = rows[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    mask = offsets < numel
    group_mask = rows < num_state_groups
    value = tl.load(value_ptr + offsets, mask=mask, other=0.0)
    absmax = tl.max(tl.abs(value), axis=1)
    scale = tl.maximum(libdevice.div_rn(absmax, fp8_max), _EPS)
    tl.store(
        data_ptr + offsets,
        libdevice.div_rn(value, scale[:, None]).to(data_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(scale_ptr + rows, scale, mask=group_mask)


def init_fp8_state(
    state: Dict[str, Any],
    spec: StateSpec,
    reference: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
) -> None:
    groups = num_groups(reference.numel(), group_size)
    state[spec.name] = torch.empty(
        reference.shape, device=reference.device, dtype=spec.dtype
    )
    state[f"scale_{spec.name}"] = torch.empty(
        groups, device=reference.device, dtype=torch.float32
    )
    if spec.recipe == "dre":
        state[f"expand_{spec.name}"] = torch.empty(
            groups, device=reference.device, dtype=torch.float32
        )
        state[f"sqrt_minmax_{spec.name}"] = torch.empty(
            groups, device=reference.device, dtype=torch.float32
        )


def dequantize_fp8_state(
    state: Dict[str, Any],
    spec: StateSpec,
    *,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    data = state[spec.name]
    groups = num_groups(data.numel(), group_size)
    restored = torch.empty_like(data, dtype=torch.float32)
    grid = (triton.cdiv(groups, GROUPS_PER_BLOCK),)
    if spec.recipe == "dre":
        _dre_dequantize_kernel[grid](
            data,
            restored,
            state[f"scale_{spec.name}"],
            state[f"expand_{spec.name}"],
            state[f"sqrt_minmax_{spec.name}"],
            data.numel(),
            groups,
            SIGNED=spec.signed,
            BLOCK_SIZE=group_size,
            ROWS=GROUPS_PER_BLOCK,
            num_warps=NUM_WARPS,
            **FTZ_KWARGS,
        )
    else:
        _maxabs_dequantize_kernel[grid](
            data,
            restored,
            state[f"scale_{spec.name}"],
            data.numel(),
            groups,
            BLOCK_SIZE=group_size,
            ROWS=GROUPS_PER_BLOCK,
            num_warps=NUM_WARPS,
            **FTZ_KWARGS,
        )
    return restored


def quantize_fp8_state_(
    state: Dict[str, Any],
    spec: StateSpec,
    value: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
) -> None:
    data = state[spec.name]
    groups = num_groups(value.numel(), group_size)
    grid = (triton.cdiv(groups, GROUPS_PER_BLOCK),)
    flat = value.contiguous().view(-1)
    fp8_max = float(torch.finfo(spec.dtype).max)
    if spec.recipe == "dre":
        _dre_quantize_kernel[grid](
            flat,
            data,
            state[f"scale_{spec.name}"],
            state[f"expand_{spec.name}"],
            state[f"sqrt_minmax_{spec.name}"],
            value.numel(),
            groups,
            fp8_max,
            fp8_max * fp8_max / 2.0,
            FLOAT32_MAX,
            SIGNED=spec.signed,
            BLOCK_SIZE=group_size,
            ROWS=GROUPS_PER_BLOCK,
            num_warps=NUM_WARPS,
            **FTZ_KWARGS,
        )
    else:
        _maxabs_quantize_kernel[grid](
            flat,
            data,
            state[f"scale_{spec.name}"],
            value.numel(),
            groups,
            fp8_max,
            BLOCK_SIZE=group_size,
            ROWS=GROUPS_PER_BLOCK,
            num_warps=NUM_WARPS,
            **FTZ_KWARGS,
        )


class FP8StateDictMixin:
    def state_dict(self):
        return torch.optim.Optimizer.state_dict(self)

    @torch._disable_dynamo
    def load_state_dict(self, state_dict):
        state_dict = state_dict.copy()
        for hook in self._optimizer_load_state_dict_pre_hooks.values():
            result = hook(self, state_dict)
            if result is not None:
                state_dict = result

        groups = self.param_groups
        saved_groups = deepcopy(state_dict["param_groups"])
        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        if any(
            len(group["params"]) != len(saved["params"])
            for group, saved in zip(groups, saved_groups)
        ):
            raise ValueError("loaded state dict does not match the optimizer")

        id_map = dict(
            zip(
                chain.from_iterable(group["params"] for group in saved_groups),
                chain.from_iterable(group["params"] for group in groups),
            )
        )

        def move(param, value, key=None):
            if isinstance(value, torch.Tensor):
                if key == "step" and value.device.type == "cpu":
                    return value.clone()
                return value.to(device=param.device, copy=True)
            if isinstance(value, dict):
                return {k: move(param, v, k) for k, v in value.items()}
            if isinstance(value, (str, bytes)):
                return value
            if isinstance(value, Iterable):
                return type(value)(move(param, item) for item in value)
            return value

        state: DefaultDict[torch.Tensor, Dict[Any, Any]] = defaultdict(dict)
        for key, value in state_dict["state"].items():
            state[id_map[key] if key in id_map else key] = (
                move(id_map[key], value) if key in id_map else value
            )
        for group, saved in zip(groups, saved_groups):
            saved["params"] = group["params"]
        self.__setstate__({"state": state, "param_groups": saved_groups})
        for hook in self._optimizer_load_state_dict_post_hooks.values():
            hook(self)


class FP8StateOptimizerMixin(FP8StateDictMixin):
    state_specs = ()
    group_size = GROUP_SIZE

    def step(self, closure=None):
        active = [
            (parameter, self.state[parameter])
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        with torch.autograd.profiler.record_function("optimizer_state_dequantize"):
            for _, state in active:
                for spec in self.state_specs:
                    if spec.name in state and state[spec.name].dtype in FP8_DTYPES:
                        state[spec.name] = dequantize_fp8_state(
                            state, spec, group_size=self.group_size
                        )

        with torch.autograd.profiler.record_function("optimizer_math"):
            loss = super().step(closure)

        with torch.autograd.profiler.record_function("optimizer_state_quantize"):
            for _, state in active:
                for spec in self.state_specs:
                    if spec.name not in state:
                        continue
                    value = state[spec.name]
                    init_fp8_state(state, spec, value, group_size=self.group_size)
                    quantize_fp8_state_(
                        state, spec, value, group_size=self.group_size
                    )
        return loss


def make_fp8_adamw(base_class):
    class FP8StateAdamW(FP8StateOptimizerMixin, base_class):
        state_specs = ADAM_STATE_SPECS

    FP8StateAdamW.__name__ = "FP8StateAdamW"
    return FP8StateAdamW
