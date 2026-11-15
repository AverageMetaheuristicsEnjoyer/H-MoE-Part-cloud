import os
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
    # Unbiased (stochastic) rounding onto the E4M3 grid instead of round-to-nearest; COAT codec only.
    stochastic: bool = False


ADAM_STATE_SPECS = (
    StateSpec("exp_avg", True, torch.float8_e4m3fn, "dre"),
    StateSpec("exp_avg_sq", False, torch.float8_e5m2, "dre"),
)

FP8_DTYPE_BY_NAME = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


def adam_state_specs():
    """Resolve standalone Adam and Muon's Adam fallback storage independently of Muon."""
    recipe = os.environ.get("STAGE3_MOE_ADAM_STATE_CODEC", "dre").strip().lower()
    if recipe not in {"fp32", "maxabs", "dre", "coat"}:
        raise ValueError(f"unknown STAGE3_MOE_ADAM_STATE_CODEC: {recipe!r}")
    requested = os.environ.get("STAGE3_MOE_FP8_STATE_DTYPES", "").strip()
    if recipe == "coat":
        # Official COAT AdamW (stage3_moe.coat_adamw) stores both moments in E4M3.
        if requested not in ("", "e4m3:e4m3"):
            raise ValueError("the COAT codec stores both Adam moments as e4m3")
        requested = "e4m3:e4m3"
    names = (
        [name.strip().lower() for name in requested.split(":")]
        if requested else ["e4m3", "e5m2"]
    )
    if len(names) != len(ADAM_STATE_SPECS):
        raise ValueError(
            f"STAGE3_MOE_FP8_STATE_DTYPES needs {len(ADAM_STATE_SPECS)} names "
            f"({', '.join(spec.name for spec in ADAM_STATE_SPECS)}), got {requested!r}"
        )
    unknown = [name for name in names if name not in FP8_DTYPE_BY_NAME]
    if unknown:
        raise ValueError(f"unknown FP8 state dtype(s): {unknown}")
    return tuple(
        StateSpec(
            spec.name, spec.signed,
            torch.float32 if recipe == "fp32" else FP8_DTYPE_BY_NAME[name],
            recipe,
        )
        for spec, name in zip(ADAM_STATE_SPECS, names)
    )


MUON_STATE_SPECS = (
    StateSpec("momentum_buffer", True, torch.float8_e4m3fn, "maxabs"),
)


def muon_state_specs():
    recipe = os.environ.get("STAGE3_MOE_MUON_STATE_CODEC", "maxabs").strip().lower()
    if recipe not in {"fp32", "maxabs", "dre", "coat"}:
        raise ValueError(f"unknown STAGE3_MOE_MUON_STATE_CODEC: {recipe!r}")
    return (
        StateSpec(
            "momentum_buffer", True,
            torch.float32 if recipe == "fp32" else torch.float8_e4m3fn,
            recipe,
        ),
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
    if spec.recipe in ("dre", "coat"):
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
    if spec.recipe == "coat":
        from stage3_moe import coat_kernels

        # The momentum kernel is the first-order half of upstream's step kernel; an unsigned
        # second moment goes through the same statements (fabs, sign +1) as upstream's v.
        if group_size != 128:
            raise ValueError("the COAT codec uses groups of 128")
        coat_kernels.momentum_dequantize(
            data, state[f"scale_{spec.name}"], state[f"expand_{spec.name}"],
            state[f"sqrt_minmax_{spec.name}"], restored,
        )
        return restored
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


def _coat_stochastic_round_(value, data, scale, expand, sqrt_minmax, seed):
    """Re-round the payload the COAT kernel has just written, keeping its group metadata.

    The kernel maps x to y = sign(x) (|x| / sqrt_minmax)^expand / scale and casts y to E4M3 with
    round-to-nearest. Here y is recomputed from the same metadata and sent to the grid point above it
    with probability equal to its distance from the grid point below, in units of the E4M3 spacing
    (2^(e-3) in the binade [2^e, 2^(e+1)), 2^-9 below 2^-6), so E[decode] = x up to the metadata.
    The unchanged dequantize kernel decodes the result. The uniforms come from a generator seeded
    with `seed` (step and parameter ordinal), identical on every data-parallel rank.
    """
    groups = scale.numel()
    x = torch.zeros(groups * GROUP_SIZE, device=value.device, dtype=torch.float32)
    x[: value.numel()] = value.reshape(-1)
    x = x.view(groups, GROUP_SIZE)
    y = x.abs().div_(sqrt_minmax[:, None]).pow_(expand[:, None]).div_(scale[:, None])
    spacing = torch.exp2(torch.floor(torch.log2(y.clamp_min(2.0 ** -6))) - 3)
    position = y.div_(spacing)
    low = torch.floor(position)
    generator = torch.Generator(device=value.device)
    generator.manual_seed(seed)
    uniform = torch.rand(position.shape, generator=generator, device=value.device)
    rounded = low.add_((uniform < position.sub_(low)).float()).mul_(spacing)
    rounded = rounded.clamp_(max=float(torch.finfo(torch.float8_e4m3fn).max)).copysign_(x)
    data.view(-1).copy_(rounded.view(-1)[: data.numel()].to(torch.float8_e4m3fn))


def quantize_fp8_state_(
    state: Dict[str, Any],
    spec: StateSpec,
    value: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
    seed: int = 0,
) -> None:
    data = state[spec.name]
    groups = num_groups(value.numel(), group_size)
    grid = (triton.cdiv(groups, GROUPS_PER_BLOCK),)
    flat = value.contiguous().view(-1)
    fp8_max = float(torch.finfo(spec.dtype).max)
    if spec.stochastic and spec.recipe != "coat":
        raise ValueError("stochastic rounding is implemented for the COAT codec only")
    if spec.recipe == "coat":
        from stage3_moe import coat_kernels

        if group_size != 128:
            raise ValueError("the COAT codec uses groups of 128")
        coat_kernels.momentum_quantize(
            flat, data, state[f"scale_{spec.name}"], state[f"expand_{spec.name}"],
            state[f"sqrt_minmax_{spec.name}"],
        )
        if spec.stochastic:
            _coat_stochastic_round_(
                flat, data, state[f"scale_{spec.name}"], state[f"expand_{spec.name}"],
                state[f"sqrt_minmax_{spec.name}"], seed,
            )
        return
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
            live = [len(group["params"]) for group in groups]
            stored = [len(saved["params"]) for saved in saved_groups]
            raise ValueError(
                "loaded state dict does not match the optimizer: "
                f"live group sizes {live} vs stored {stored}"
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
    # Parameters whose state is held in FP32 at the same time. 0 means all of them,
    # which is what every measurement so far was taken with.
    dequantize_chunk = int(os.environ.get("STAGE3_MOE_FP8_DEQUANT_CHUNK", "0"))

    def _step_over(self, groups, closure):
        active = [
            (parameter, self.state[parameter])
            for group in groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        with torch.autograd.profiler.record_function("optimizer_state_dequantize"):
            for _, state in active:
                for spec in self.state_specs:
                    if spec.dtype == torch.float32:
                        continue
                    if spec.name in state and state[spec.name].dtype in FP8_DTYPES:
                        state[spec.name] = dequantize_fp8_state(
                            state, spec, group_size=self.group_size
                        )

        with torch.autograd.profiler.record_function("optimizer_math"):
            every_group = self.param_groups
            self.param_groups = groups
            try:
                loss = super().step(closure)
            finally:
                self.param_groups = every_group

        with torch.autograd.profiler.record_function("optimizer_state_quantize"):
            ordinals = self._parameter_ordinals()
            for parameter, state in active:
                for spec in self.state_specs:
                    if spec.dtype == torch.float32:
                        continue
                    if spec.name not in state:
                        continue
                    value = state[spec.name]
                    init_fp8_state(state, spec, value, group_size=self.group_size)
                    # Same seed on every rank: step and the parameter's position in the groups.
                    seed = int(state.get("step", 0)) * 1_000_003 + ordinals.get(id(parameter), 0)
                    quantize_fp8_state_(
                        state, spec, value, group_size=self.group_size, seed=seed
                    )
        return loss

    def _parameter_ordinals(self):
        if not hasattr(self, "_ordinals"):
            self._ordinals = {
                id(parameter): index
                for index, parameter in enumerate(
                    chain.from_iterable(group["params"] for group in self.param_groups)
                )
            }
        return self._ordinals

    def _param_group_chunks(self, chunk):
        """Yield views of ``param_groups`` holding at most *chunk* parameters.

        Each parameter's update reads only its own state and its group's
        hyper-parameters, and neither the wrapped Muon nor AdamW writes to the
        group dict during a step, so stepping a subset at a time is equivalent
        to stepping everything at once.
        """
        selection = []
        for index, group in enumerate(self.param_groups):
            for parameter in group["params"]:
                selection.append((index, parameter))
                if len(selection) == chunk:
                    yield self._narrowed(selection)
                    selection = []
        if selection:
            yield self._narrowed(selection)

    def _narrowed(self, selection):
        params_by_group = defaultdict(list)
        for index, parameter in selection:
            params_by_group[index].append(parameter)
        return [
            {**self.param_groups[index], "params": params}
            for index, params in params_by_group.items()
        ]

    def step(self, closure=None):
        if self.dequantize_chunk <= 0:
            return self._step_over(self.param_groups, closure)
        loss = None
        for groups in self._param_group_chunks(self.dequantize_chunk):
            loss = self._step_over(groups, closure)
            # Re-running a closure would repeat the forward and backward pass.
            closure = None
        return loss


def dequantize_loaded_state_dict(optimizer, state_dict):
    """Pre-hook for a warm FP8 -> FP32 state switch (customer feedback 2026-10-05: "resume" across precisions).

    The checkpoint holds the series' FP8 states (Muon momentum under the COAT codec, or the official COAT
    AdamW's two moments with their per-parameter step); the live optimizer keeps FP32 states. Every FP8
    entry is decoded with its own metadata and the metadata dropped, so torch's load sees plain FP32
    moments. COAT AdamW's per-parameter step becomes the group step TE FusedAdam keeps.
    """
    state = state_dict.get("state")
    groups = state_dict.get("param_groups")
    if not isinstance(state, dict) or not groups:
        return None
    group_of = {key: group for group in groups for key in group["params"]}
    converted = adam_entries = 0
    device = torch.device("cuda", torch.cuda.current_device())
    for key, entry in list(state.items()):
        if not isinstance(entry, dict) or key not in group_of:
            continue
        fp8_names = [name for name, value in entry.items() if torch.is_tensor(value) and value.dtype in FP8_DTYPES]
        if not fp8_names:
            continue
        rebuilt = {}
        for name in fp8_names:
            recipe = "coat" if f"sqrt_minmax_{name}" in entry else "maxabs"
            spec = StateSpec(name, name != "exp_avg_sq", entry[name].dtype, recipe)
            metadata = {
                k: v.to(device) for k, v in entry.items()
                if torch.is_tensor(v) and (k == name or k.endswith("_" + name))
            }
            rebuilt[name] = dequantize_fp8_state(metadata, spec, group_size=GROUP_SIZE)
            converted += 1
        for name, value in entry.items():
            if name in rebuilt or any(name.endswith("_" + fp8) for fp8 in fp8_names):
                continue
            if name == "step" and "exp_avg" in rebuilt:
                # COAT AdamW: one step per parameter -> FusedAdam: one per group.
                group_of[key]["step"] = int(value.item() if torch.is_tensor(value) else value)
                adam_entries += 1
                continue
            rebuilt[name] = value
        state[key] = rebuilt
    # torch replaces the live groups by the stored ones; a COAT AdamW group lacks FusedAdam's keys
    # (bias_correction, ...) and vice versa, so every key the live optimizer has stays in the loaded group.
    filled = 0
    for live, saved in zip(optimizer.param_groups, groups):
        for key, value in live.items():
            if key != "params" and key not in saved:
                saved[key] = value
                filled += 1
    print(f"STAGE3_DEQUANT_ON_LOAD fp8_tensors={converted} coat_adam_parameters={adam_entries} group_keys_filled={filled}", flush=True)
    return state_dict


def install_dequantize_on_load(optimizer):
    for wrapped in getattr(optimizer, "chained_optimizers", [optimizer]):
        raw = getattr(wrapped, "optimizer", wrapped)
        if raw is not None:
            raw.register_load_state_dict_pre_hook(dequantize_loaded_state_dict)


def make_fp8_adamw(base_class):
    class FP8StateAdamW(FP8StateOptimizerMixin, base_class):
        state_specs = adam_state_specs()

    FP8StateAdamW.__name__ = "FP8StateAdamW"
    return FP8StateAdamW


def make_fp8_frugal(base_class):
    """Frugal CoordAdamW with FP8 moments.

    Frugal keeps Adam's two moments, only narrowed to the active columns, so the Adam specs
    apply unchanged -- STAGE3_MOE_FP8_STATE_DTYPES included, which is why this resolves them
    at install time rather than at import.

    The coordinate refresh needs no special handling even though it calls ``state.clear()``
    and puts fresh FP32 moments back every ``update_gap`` steps: the mixin re-runs
    ``init_fp8_state`` after every wrapped step and sizes the holders from whatever tensor is
    there, so the quantised buffers are rebuilt on the same step they were dropped. The case
    it does not cover is a parameter that never receives a gradient -- the refresh
    re-initialises its moments but the quantise pass only walks parameters with a grad, so
    they stay FP32. That corrupts nothing: result_writer's precision contract asserts the
    state dtypes, so such a run fails loudly instead of quietly costing memory.
    """

    class FP8StateFrugalCoordAdamW(FP8StateOptimizerMixin, base_class):
        state_specs = adam_state_specs()

    FP8StateFrugalCoordAdamW.__name__ = "FP8StateFrugalCoordAdamW"
    return FP8StateFrugalCoordAdamW
