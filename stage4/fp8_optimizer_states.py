from collections import defaultdict
from copy import deepcopy
from itertools import chain
import os
from typing import Any, DefaultDict, Dict, Iterable

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}
QUANT_EPS = 1e-30
FLOAT32_MAX = float(torch.finfo(torch.float32).max)
GROUPS_PER_BLOCK = 8
NUM_WARPS = 4
USE_BATCHED_CONVERSIONS = os.environ.get("STAGE4_FP8_BATCHED", "0") != "0"

# Triton kernels can only close over globals declared as constexpr.
_EPS = tl.constexpr(QUANT_EPS)
_ONE_PLUS_EPS = tl.constexpr(1.0 + QUANT_EPS)


def _num_groups(numel: int, group_size: int) -> int:
    return (numel + group_size - 1) // group_size


@triton.jit
def _dequantize_kernel(
    data_ptr,
    out_ptr,
    scale_ptr,
    expand_ptr,
    sqrt_minmax_ptr,
    numel,
    num_groups,
    SIGNED: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    offsets = rows[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    mask = offsets < numel
    group_mask = rows < num_groups

    scale = tl.load(scale_ptr + rows, mask=group_mask, other=0.0)
    expansion = tl.load(expand_ptr + rows, mask=group_mask, other=1.0)
    sqrt_minmax = tl.load(sqrt_minmax_ptr + rows, mask=group_mask, other=1.0)
    expansion = tl.maximum(expansion, _EPS)
    sqrt_minmax = tl.maximum(sqrt_minmax, _EPS)
    inverse = libdevice.div_rn(1.0, expansion)[:, None]

    raw = tl.load(data_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    raw = raw * scale[:, None]
    if SIGNED:
        magnitude = tl.abs(raw)
        sign = tl.where(raw > 0.0, 1.0, tl.where(raw < 0.0, -1.0, 0.0))
        restored = (
            sign
            * libdevice.pow(tl.maximum(magnitude, _EPS), inverse)
            * sqrt_minmax[:, None]
        )
        restored = tl.where(magnitude > 0.0, restored, 0.0)
    else:
        raw = tl.maximum(raw, 0.0)
        restored = (
            libdevice.pow(tl.maximum(raw, _EPS), inverse) * sqrt_minmax[:, None]
        )
        restored = tl.where(raw > 0.0, restored, 0.0)
    tl.store(out_ptr + offsets, restored, mask=mask)


@triton.jit
def _quantize_kernel(
    value_ptr,
    data_ptr,
    scale_ptr,
    expand_ptr,
    sqrt_minmax_ptr,
    numel,
    num_groups,
    fp8_max,
    ratio_upper,
    float32_max,
    SIGNED: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    offsets = rows[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    mask = offsets < numel
    group_mask = rows < num_groups

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
        ratio <= _ONE_PLUS_EPS, 1.0, tl.maximum(raw_expansion, 1.0 / 16)
    )

    sqrt_minmax = tl.maximum(tl.sqrt_rn(absmax) * tl.sqrt_rn(absmin), _EPS)
    base = tl.maximum(libdevice.div_rn(magnitude, sqrt_minmax[:, None]), _EPS)
    normalized = libdevice.pow(base, expansion[:, None])
    normalized = tl.where(nonzero, normalized, 0.0)
    if SIGNED:
        sign = tl.where(source > 0.0, 1.0, tl.where(source < 0.0, -1.0, 0.0))
        normalized = sign * normalized

    scale = tl.maximum(
        libdevice.div_rn(
            libdevice.pow(
                tl.maximum(libdevice.div_rn(absmax, sqrt_minmax), _EPS), expansion
            ),
            fp8_max,
        ),
        _EPS,
    )
    quantized = libdevice.div_rn(normalized, scale[:, None])

    tl.store(
        data_ptr + offsets,
        quantized.to(data_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(scale_ptr + rows, scale, mask=group_mask)
    tl.store(expand_ptr + rows, expansion, mask=group_mask)
    tl.store(sqrt_minmax_ptr + rows, sqrt_minmax, mask=group_mask)


@triton.jit
def _batched_dequantize_kernel(
    data_ptrs,
    out_ptrs,
    scale_ptrs,
    expand_ptrs,
    sqrt_minmax_ptrs,
    group_ends,
    numels,
    total_groups,
    num_tensors,
    SIGNED: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    group_mask = rows < total_groups
    safe_rows = tl.minimum(rows, total_groups - 1).to(tl.int64)
    owner = tl.zeros((ROWS,), tl.int64)
    for shift in tl.static_range(7):
        increment = 64 // (1 << shift)
        candidate = owner + increment
        valid_candidate = candidate < num_tensors
        end = tl.load(group_ends + candidate - 1, mask=valid_candidate, other=total_groups)
        owner += tl.where(valid_candidate & (safe_rows >= end), increment, 0)
    start = tl.load(group_ends + owner - 1, mask=owner > 0, other=0)
    local_rows = safe_rows - start
    offsets = local_rows[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    numel = tl.load(numels + owner)
    mask = group_mask[:, None] & (offsets < numel[:, None])

    data = tl.load(data_ptrs + owner).to(tl.pointer_type(tl.float8e4nv))
    out = tl.load(out_ptrs + owner).to(tl.pointer_type(tl.float32))
    scale = tl.load(scale_ptrs + owner).to(tl.pointer_type(tl.float32))
    expand = tl.load(expand_ptrs + owner).to(tl.pointer_type(tl.float32))
    sqrt_minmax = tl.load(sqrt_minmax_ptrs + owner).to(tl.pointer_type(tl.float32))
    expansion = tl.maximum(tl.load(expand + local_rows, mask=group_mask, other=1.0), _EPS)
    minimum = tl.maximum(tl.load(sqrt_minmax + local_rows, mask=group_mask, other=1.0), _EPS)
    raw = tl.load(data[:, None] + offsets, mask=mask, other=0.0).to(tl.float32)
    raw *= tl.load(scale + local_rows, mask=group_mask, other=0.0)[:, None]
    if SIGNED:
        magnitude = tl.abs(raw)
        sign = tl.where(raw > 0.0, 1.0, tl.where(raw < 0.0, -1.0, 0.0))
        restored = sign * libdevice.pow(
            tl.maximum(magnitude, _EPS), libdevice.div_rn(1.0, expansion)[:, None]
        ) * minimum[:, None]
        restored = tl.where(magnitude > 0.0, restored, 0.0)
    else:
        raw = tl.maximum(raw, 0.0)
        restored = libdevice.pow(
            tl.maximum(raw, _EPS), libdevice.div_rn(1.0, expansion)[:, None]
        ) * minimum[:, None]
        restored = tl.where(raw > 0.0, restored, 0.0)
    tl.store(out[:, None] + offsets, restored, mask=mask)


@triton.jit
def _batched_quantize_kernel(
    value_ptrs,
    data_ptrs,
    scale_ptrs,
    expand_ptrs,
    sqrt_minmax_ptrs,
    group_ends,
    numels,
    total_groups,
    num_tensors,
    fp8_max,
    ratio_upper,
    float32_max,
    SIGNED: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    group_mask = rows < total_groups
    safe_rows = tl.minimum(rows, total_groups - 1).to(tl.int64)
    owner = tl.zeros((ROWS,), tl.int64)
    for shift in tl.static_range(7):
        increment = 64 // (1 << shift)
        candidate = owner + increment
        valid_candidate = candidate < num_tensors
        end = tl.load(group_ends + candidate - 1, mask=valid_candidate, other=total_groups)
        owner += tl.where(valid_candidate & (safe_rows >= end), increment, 0)
    start = tl.load(group_ends + owner - 1, mask=owner > 0, other=0)
    local_rows = safe_rows - start
    offsets = local_rows[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    numel = tl.load(numels + owner)
    mask = group_mask[:, None] & (offsets < numel[:, None])

    value = tl.load(value_ptrs + owner).to(tl.pointer_type(tl.float32))
    data = tl.load(data_ptrs + owner).to(tl.pointer_type(tl.float8e4nv))
    scale = tl.load(scale_ptrs + owner).to(tl.pointer_type(tl.float32))
    expand = tl.load(expand_ptrs + owner).to(tl.pointer_type(tl.float32))
    sqrt_minmax = tl.load(sqrt_minmax_ptrs + owner).to(tl.pointer_type(tl.float32))
    source = tl.load(value[:, None] + offsets, mask=mask, other=0.0)
    if SIGNED:
        magnitude = tl.abs(source)
    else:
        source = tl.maximum(source, 0.0)
        magnitude = source
    nonzero = magnitude > 0.0
    absmax = tl.maximum(tl.max(magnitude, axis=1), _EPS)
    absmin = tl.min(tl.where(nonzero, magnitude, float32_max), axis=1)
    absmin = tl.where(
        tl.max(nonzero.to(tl.int32), axis=1) != 0, tl.maximum(absmin, _EPS), _EPS
    )
    ratio = tl.maximum(libdevice.div_rn(absmax, absmin), _ONE_PLUS_EPS)
    raw_expansion = libdevice.div_rn(
        libdevice.floor(
            libdevice.div_rn(
                libdevice.log2(ratio_upper), tl.maximum(libdevice.log2(ratio), _EPS)
            ) * 16
        ),
        16.0,
    )
    expansion = tl.where(
        ratio <= _ONE_PLUS_EPS, 1.0, tl.maximum(raw_expansion, 1.0 / 16)
    )
    minimum = tl.maximum(tl.sqrt_rn(absmax) * tl.sqrt_rn(absmin), _EPS)
    normalized = libdevice.pow(
        tl.maximum(libdevice.div_rn(magnitude, minimum[:, None]), _EPS), expansion[:, None]
    )
    normalized = tl.where(nonzero, normalized, 0.0)
    if SIGNED:
        normalized *= tl.where(source > 0.0, 1.0, tl.where(source < 0.0, -1.0, 0.0))
    new_scale = tl.maximum(
        libdevice.div_rn(
            libdevice.pow(
                tl.maximum(libdevice.div_rn(absmax, minimum), _EPS), expansion
            ),
            fp8_max,
        ),
        _EPS,
    )
    tl.store(
        data[:, None] + offsets,
        libdevice.div_rn(normalized, new_scale[:, None]).to(tl.float8e4nv),
        mask=mask,
    )
    tl.store(scale + local_rows, new_scale, mask=group_mask)
    tl.store(expand + local_rows, expansion, mask=group_mask)
    tl.store(sqrt_minmax + local_rows, minimum, mask=group_mask)


def _batched_tables(entries):
    ends = []
    total_groups = 0
    for stored, _, _, _ in entries:
        total_groups += _num_groups(stored.numel(), 128)
        ends.append(total_groups)
    device = entries[0][0].device
    tensors = {
        "data": [stored for stored, _, _, _ in entries],
        "value": [value for _, value, _, _ in entries],
        "scale": [state[f"scale_{prefix}"] for _, _, state, prefix in entries],
        "expand": [state[f"expand_{prefix}"] for _, _, state, prefix in entries],
        "sqrt": [state[f"sqrt_minmax_{prefix}"] for _, _, state, prefix in entries],
    }
    tables = {
        key: torch.tensor([value.data_ptr() for value in values], device=device, dtype=torch.int64)
        for key, values in tensors.items()
    }
    tables["ends"] = torch.tensor(ends[:-1], device=device, dtype=torch.int64)
    tables["numels"] = torch.tensor(
        [stored.numel() for stored, _, _, _ in entries], device=device, dtype=torch.int64
    )
    for table in tables.values():
        table.record_stream(torch.cuda.current_stream())
    return tables, total_groups


def _batched_dequantize(entries, signed):
    tables, total_groups = _batched_tables(entries)
    _batched_dequantize_kernel[(triton.cdiv(total_groups, GROUPS_PER_BLOCK),)](
        tables["data"], tables["value"], tables["scale"], tables["expand"], tables["sqrt"],
        tables["ends"], tables["numels"], total_groups, len(entries), SIGNED=signed,
        GROUP_SIZE=128, ROWS=GROUPS_PER_BLOCK, num_warps=NUM_WARPS,
    )


def _batched_quantize(entries, signed):
    tables, total_groups = _batched_tables(entries)
    _batched_quantize_kernel[(triton.cdiv(total_groups, GROUPS_PER_BLOCK),)](
        tables["value"], tables["data"], tables["scale"], tables["expand"], tables["sqrt"],
        tables["ends"], tables["numels"], total_groups, len(entries), 448.0,
        448.0 * 448.0 / 2.0, FLOAT32_MAX, SIGNED=signed, GROUP_SIZE=128,
        ROWS=GROUPS_PER_BLOCK, num_warps=NUM_WARPS,
    )


def init_fp8_state(
    state: Dict[str, Any],
    prefix: str,
    reference: torch.Tensor,
    *,
    group_size: int,
) -> None:
    groups = _num_groups(reference.numel(), group_size)
    state[prefix] = torch.zeros(
        reference.shape,
        device=reference.device,
        dtype=torch.float8_e4m3fn,
    )
    state[f"scale_{prefix}"] = torch.zeros(groups, device=reference.device, dtype=torch.float32)
    state[f"expand_{prefix}"] = torch.ones(groups, device=reference.device, dtype=torch.float32)
    state[f"sqrt_minmax_{prefix}"] = torch.ones(
        groups, device=reference.device, dtype=torch.float32
    )


def dequantize_fp8_state(
    state: Dict[str, Any],
    prefix: str,
    *,
    signed: bool,
    group_size: int,
) -> torch.Tensor:
    data = state[prefix]
    numel = data.numel()
    num_groups = _num_groups(numel, group_size)
    restored = torch.empty_like(data, dtype=torch.float32)
    _dequantize_kernel[(triton.cdiv(num_groups, GROUPS_PER_BLOCK),)](
        data,
        restored,
        state[f"scale_{prefix}"],
        state[f"expand_{prefix}"],
        state[f"sqrt_minmax_{prefix}"],
        numel,
        num_groups,
        SIGNED=signed,
        GROUP_SIZE=group_size,
        ROWS=GROUPS_PER_BLOCK,
        num_warps=NUM_WARPS,
    )
    return restored


def quantize_fp8_state_(
    state: Dict[str, Any],
    prefix: str,
    value: torch.Tensor,
    *,
    signed: bool,
    group_size: int,
) -> None:
    stored = state[prefix]
    numel = value.numel()
    num_groups = _num_groups(numel, group_size)
    fp8_max = float(torch.finfo(stored.dtype).max)
    _quantize_kernel[(triton.cdiv(num_groups, GROUPS_PER_BLOCK),)](
        value.contiguous().view(-1),
        stored,
        state[f"scale_{prefix}"],
        state[f"expand_{prefix}"],
        state[f"sqrt_minmax_{prefix}"],
        numel,
        num_groups,
        fp8_max,
        fp8_max * fp8_max / 2.0,
        FLOAT32_MAX,
        SIGNED=signed,
        GROUP_SIZE=group_size,
        ROWS=GROUPS_PER_BLOCK,
        num_warps=NUM_WARPS,
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
            raise ValueError(
                "loaded state dict contains a parameter group that does not match the optimizer"
            )

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
                return {
                    item_key: move(param, item, item_key)
                    for item_key, item in value.items()
                }
            if isinstance(value, (str, bytes)):
                return value
            if isinstance(value, Iterable):
                return type(value)(move(param, item) for item in value)
            return value

        state: DefaultDict[torch.Tensor, Dict[Any, Any]] = defaultdict(dict)
        for key, value in state_dict["state"].items():
            if key in id_map:
                state[id_map[key]] = move(id_map[key], value)
            else:
                state[key] = value

        for group, saved in zip(groups, saved_groups):
            saved["params"] = group["params"]
        self.__setstate__({"state": state, "param_groups": saved_groups})

        for hook in self._optimizer_load_state_dict_post_hooks.values():
            hook(self)


class FP8StateOptimizerMixin(FP8StateDictMixin):
    state_specs = ()
    group_size = 128

    def step(self, closure=None):
        active = [
            (parameter, self.state[parameter])
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        stored_states = {}
        with torch.autograd.profiler.record_function("optimizer_state_dequantize"):
            batched = []
            for _, state in active:
                for prefix, signed in self.state_specs:
                    if prefix in state and state[prefix].dtype in FP8_DTYPES:
                        if USE_BATCHED_CONVERSIONS:
                            stored_states[(id(state), prefix)] = state[prefix]
                            state[prefix] = torch.empty_like(state[prefix], dtype=torch.float32)
                            batched.append(
                                (stored_states[(id(state), prefix)], state[prefix], state, prefix, signed)
                            )
                        else:
                            # Overwriting the entry drops the last reference to the FP8
                            # storage, so the two representations of a state coexist for
                            # one tensor at a time instead of for the whole step.
                            state[prefix] = dequantize_fp8_state(
                                state, prefix, signed=signed, group_size=self.group_size
                            )
            if USE_BATCHED_CONVERSIONS:
                for signed in (False, True):
                    entries = [entry[:-1] for entry in batched if entry[-1] == signed]
                    if entries:
                        _batched_dequantize(entries, signed)

        with torch.autograd.profiler.record_function("optimizer_math"):
            loss = super().step(closure)

        with torch.autograd.profiler.record_function("optimizer_state_quantize"):
            batched = []
            for _, state in active:
                for prefix, signed in self.state_specs:
                    if prefix not in state:
                        continue
                    value = state[prefix]
                    if USE_BATCHED_CONVERSIONS:
                        stored = stored_states.get((id(state), prefix))
                        if stored is None:
                            init_fp8_state(
                                state,
                                prefix,
                                value,
                                group_size=self.group_size,
                            )
                        else:
                            state[prefix] = stored
                        batched.append((state[prefix], value, state, prefix, signed))
                    else:
                        init_fp8_state(
                            state,
                            prefix,
                            value,
                            group_size=self.group_size,
                        )
                        quantize_fp8_state_(
                            state, prefix, value, signed=signed, group_size=self.group_size
                        )
                        del value
            if USE_BATCHED_CONVERSIONS:
                for signed in (False, True):
                    entries = [entry[:-1] for entry in batched if entry[-1] == signed]
                    if entries:
                        _batched_quantize(entries, signed)
        return loss


def make_fp8_adamw(base_class):
    class FP8StateAdamW(FP8StateOptimizerMixin, base_class):
        state_specs = (("exp_avg", True), ("exp_avg_sq", False))

    FP8StateAdamW.__name__ = "FP8StateAdamW"
    return FP8StateAdamW


def make_fp8_soap(base_class):
    class FP8StateSOAP(FP8StateOptimizerMixin, base_class):
        state_specs = (
            ("exp_avg", True),
            ("exp_avg_sq", False),
            ("L", True),
            ("R", True),
            ("Q_L", True),
            ("Q_R", True),
        )

    FP8StateSOAP.__name__ = "FP8StateSOAP"
    return FP8StateSOAP
