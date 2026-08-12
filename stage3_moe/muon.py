from typing import Any

import torch

from megatron.core.optimizer.emerging_optimizers import (
    TensorParallelMuon,
    _EMERGING_OPTIMIZERS,
)
from megatron.core.optimizer.optimizer_config import ParamKey, ParamWithNamePredicate

from stage3_moe.optimizer_states import FP8StateOptimizerMixin, MUON_STATE_SPECS


SPLIT_SWIGLU_FC1 = "stage3_split_swiglu_fc1"


def is_router_weight(param: torch.Tensor, name: str) -> bool:
    return len(param.shape) == 2 and name.endswith(".router.weight")


def is_swiglu_fc1_weight(param: torch.Tensor, name: str) -> bool:
    return len(param.shape) == 2 and ".linear_fc1.weight" in name


class SplitSwiGLUTensorParallelMuon(TensorParallelMuon):
    def orthogonalize(
        self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
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


def install_muon_contract(*, fp8_states: bool) -> None:
    entry = _EMERGING_OPTIMIZERS["muon"]
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
    entry.default_param_overrides = overrides
    entry.optimizer_cls = (
        FP8StateSplitSwiGLUTensorParallelMuon
        if fp8_states
        else SplitSwiGLUTensorParallelMuon
    )
