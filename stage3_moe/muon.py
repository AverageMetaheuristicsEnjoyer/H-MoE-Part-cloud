import os
from typing import Any

import torch

from megatron.core.optimizer.emerging_optimizers import (
    TensorParallelMuon,
    _EMERGING_OPTIMIZERS,
)
from megatron.core.optimizer.optimizer_config import ParamKey, ParamWithNamePredicate

from stage3_moe.optimizer_states import (
    FP8StateOptimizerMixin,
    MUON_STATE_SPECS,
    muon_state_specs,
)


SPLIT_SWIGLU_FC1 = "stage3_split_swiglu_fc1"
ROUTED_EXPERT = "stage3_routed_expert"


def is_router_weight(param: torch.Tensor, name: str) -> bool:
    return len(param.shape) == 2 and name.endswith(".router.weight")


def is_routed_expert_weight(param: torch.Tensor, name: str) -> bool:
    return len(param.shape) == 2 and ".mlp.experts." in name


def is_swiglu_fc1_weight(param: torch.Tensor, name: str) -> bool:
    return len(param.shape) == 2 and ".linear_fc1.weight" in name


class SplitSwiGLUTensorParallelMuon(TensorParallelMuon):
    def orthogonalize(
        self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        kwargs.pop(ROUTED_EXPERT, None)
        split_fc1 = kwargs.pop(SPLIT_SWIGLU_FC1, False)
        if not split_fc1:
            return super().orthogonalize(p, grad, **kwargs)
        if grad.ndim != 2 or grad.shape[0] % 2:
            raise ValueError(
                f"SwiGLU FC1 must be an even 2D matrix, got {tuple(grad.shape)}"
            )
        gate, up = grad.chunk(2, dim=0)
        parent = super()
        return torch.cat(
            (
                parent.orthogonalize(p, gate, **kwargs),
                parent.orthogonalize(p, up, **kwargs),
            ),
            dim=0,
        )


class FP8StateSplitSwiGLUTensorParallelMuon(
    FP8StateOptimizerMixin, SplitSwiGLUTensorParallelMuon
):
    state_specs = MUON_STATE_SPECS

    def __init__(self, *args, **kwargs):
        self.state_specs = muon_state_specs()
        super().__init__(*args, **kwargs)


def install_muon_contract(*, fp8_states: bool) -> None:
    from stage3_moe.lionmuon import apply_variant, remember_ns_settings, variant_name

    entry = _EMERGING_OPTIMIZERS["muon"]
    # STAGE3_MOE_MUON_NS_STEPS, and the Newton-Schulz settings SoftMuon / NorMuon need from TensorParallelMuon's closure.
    previous_config_to_kwargs = entry.config_to_kwargs

    def config_to_kwargs_with_variant(*args, **kwargs):
        return remember_ns_settings(previous_config_to_kwargs(*args, **kwargs))

    entry.config_to_kwargs = config_to_kwargs_with_variant
    overrides = dict(entry.default_param_overrides)
    overrides[
        ParamKey(
            with_name_predicate=ParamWithNamePredicate(
                name="stage3_router_adam_fallback", fn=is_router_weight
            )
        )
    ] = {"optimizer": "adam"}
    overrides[
        ParamKey(
            with_name_predicate=ParamWithNamePredicate(
                name="stage3_split_swiglu_fc1", fn=is_swiglu_fc1_weight
            )
        )
    ] = {SPLIT_SWIGLU_FC1: True}
    if (os.environ.get("STAGE3_MOE_NORMUON_SCOPE", "all") != "all" or os.environ.get("STAGE3_MOE_LIONMUON_PERIOD_EXPERTS")
            or os.environ.get("STAGE3_MOE_LIONMUON_ENERGYCAL") == "1"):
        # NorMuon diagnosis (2026-10-22): mark the routed experts' matrices so NorMuon can be limited to or kept off them.
        overrides[
            ParamKey(
                with_name_predicate=ParamWithNamePredicate(
                    name="stage3_routed_expert", fn=is_routed_expert_weight
                )
            )
        ] = {ROUTED_EXPERT: True}
    if os.environ.get("STAGE3_MOE_LIONMUON_BLOCK_FACTORS"):
        if variant_name() not in ("lionmuon", "lionmuon2f"):
            raise ValueError("LionMuon block factors require a LionMuon variant")
        from stage3_moe.lionmuon import (
            LIONMUON_BLOCK_KEY,
            lionmuon_block_for_name,
            parse_lionmuon_block_factors,
        )
        factors = parse_lionmuon_block_factors(os.environ["STAGE3_MOE_LIONMUON_BLOCK_FACTORS"])
        num_layers = int(os.environ.get("STAGE3_MOE_LIONMUON_NUM_LAYERS", "0"))
        if num_layers < 1:
            raise ValueError("STAGE3_MOE_LIONMUON_NUM_LAYERS must be positive for block probability")
        for block in range(3):
            def in_block(param, name, expected=block):
                return (len(param.shape) == 2 and not is_router_weight(param, name)
                        and lionmuon_block_for_name(name, num_layers) == expected)
            overrides[
                ParamKey(
                    with_name_predicate=ParamWithNamePredicate(
                        name=f"stage3_lionmuon_block{block}", fn=in_block
                    )
                )
            ] = {LIONMUON_BLOCK_KEY: block}
        print(f"stage3 lionmuon block probability: layers={num_layers} factors={factors}", flush=True)
    entry.default_param_overrides = overrides
    if variant_name() == "muon":
        entry.optimizer_cls = (
            FP8StateSplitSwiGLUTensorParallelMuon
            if fp8_states
            else SplitSwiGLUTensorParallelMuon
        )
        return
    bases = apply_variant(
        (FP8StateOptimizerMixin, SplitSwiGLUTensorParallelMuon) if fp8_states else (SplitSwiGLUTensorParallelMuon,)
    )
    namespace = {"state_specs": MUON_STATE_SPECS}
    if fp8_states:
        def __init__(self, *args, **kwargs):
            self.state_specs = muon_state_specs()
            super(variant_cls, self).__init__(*args, **kwargs)

        namespace["__init__"] = __init__
    variant_cls = type(f"{variant_name().capitalize()}SplitSwiGLUTensorParallelMuon", bases, namespace)
    entry.optimizer_cls = variant_cls
    print(f"stage3 muon variant: {variant_name()} -> {[c.__name__ for c in variant_cls.__mro__[:4]]}", flush=True)
