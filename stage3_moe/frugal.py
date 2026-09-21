# Adapted from https://github.com/fzmushko/frugal (Apache-2.0).

import math

import torch

from megatron.core.optimizer.emerging_optimizers import (
    EmergingOptimizerEntry,
    _EMERGING_OPTIMIZERS,
)
from megatron.core.optimizer.optimizer_config import ParamKey, ParamWithNamePredicate


FRUGAL_DENSITY = 0.25
FRUGAL_UPDATE_GAP = 50
FRUGAL_COORD_CHOICE = "columns"


def is_frugal_fallback(param: torch.Tensor, name: str) -> bool:
    return (
        len(param.shape) != 2
        or getattr(param, "is_embedding_or_output_parameter", False)
        or name.endswith(".router.weight")
    )


class FrugalCoordAdamW(torch.optim.Optimizer):
    optimizer_role = "frugal_matrix"

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        density=FRUGAL_DENSITY,
        update_gap=FRUGAL_UPDATE_GAP,
    ):
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "density": density,
                "update_gap": update_gap,
            },
        )
        for group in self.param_groups:
            group.setdefault("frugal_step", 0)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for state in self.state.values():
            state["coord_indices"] = state["coord_indices"].to(dtype=torch.long)

    @staticmethod
    def _init_state(param, state, density):
        state.clear()
        columns = torch.randperm(param.shape[1], device=param.device)[
            : int(param.shape[1] * density)
        ]
        shape = (param.shape[0], columns.numel())
        state["step"] = 0
        state["coord_indices"] = columns
        state["exp_avg"] = torch.zeros(shape, dtype=param.dtype, device=param.device)
        state["exp_avg_sq"] = torch.zeros(shape, dtype=param.dtype, device=param.device)

    @torch.no_grad()
    def _init_group(self, group, skip_non_grad_params=False):
        for param in group["params"]:
            if skip_non_grad_params and param.grad is None:
                continue
            self._init_state(param, self.state[param], group["density"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["frugal_step"] % group["update_gap"] == 0:
                self._init_group(group)
            group["frugal_step"] += 1

            beta1, beta2 = group["betas"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("FrugalCoordAdamW does not support sparse gradients")

                param.mul_(1 - group["lr"] * group["weight_decay"])
                state = self.state[param]
                columns = state["coord_indices"]
                active_grad = grad[:, columns]

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(active_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    active_grad, active_grad, value=1 - beta2
                )

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2))
                denom.add_(group["eps"])
                active_update = exp_avg / denom
                active_update.mul_(-group["lr"] / bias_correction1)

                update = grad.sign().mul_(-group["lr"])
                update[:, columns] = active_update
                param.add_(update)
        return loss


def _frugal_config_to_kwargs(config, model_chunks, pg_collection):
    return {
        "lr": config.lr,
        "betas": (config.adam_beta1, config.adam_beta2),
        "eps": config.adam_eps,
        "weight_decay": config.weight_decay,
        "density": FRUGAL_DENSITY,
        "update_gap": FRUGAL_UPDATE_GAP,
    }


def install_frugal_contract() -> None:
    fallback = ParamWithNamePredicate(
        name="stage3_frugal_adam_fallback", fn=is_frugal_fallback
    )
    _EMERGING_OPTIMIZERS["frugal"] = EmergingOptimizerEntry(
        optimizer_cls=FrugalCoordAdamW,
        config_to_kwargs=_frugal_config_to_kwargs,
        default_param_overrides={
            ParamKey(with_name_predicate=fallback): {"optimizer": "adam"}
        },
    )
