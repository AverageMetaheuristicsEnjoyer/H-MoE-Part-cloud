# Adapted from https://github.com/fzmushko/frugal (Apache-2.0).

import math

import torch

from megatron.core.optimizer.emerging_optimizers import (
    EmergingOptimizerEntry,
    _EMERGING_OPTIMIZERS,
)
from megatron.core.optimizer.optimizer_config import ParamKey, ParamWithNamePredicate


FRUGAL_DENSITY = 0.25
FRUGAL_UPDATE_GAP = 200


def is_frugal_fallback(param: torch.Tensor, name: str) -> bool:
    return (
        len(param.shape) != 2
        or getattr(param, "is_embedding_or_output_parameter", False)
        or name.endswith(".router.weight")
    )


class FrugalAdamW(torch.optim.Optimizer):
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
            group.setdefault("frugal_next_block_start", len(group["params"]) - 1)
            group.setdefault(
                "frugal_num_active_blocks", round(len(group["params"]) * density)
            )

    @staticmethod
    def _init_state(param, state):
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(param)
        state["exp_avg_sq"] = torch.zeros_like(param)

    @torch.no_grad()
    def _rotate_group(self, group):
        params = group["params"]
        if not params:
            return
        for param in params:
            self.state[param].clear()
            self.state[param]["active"] = False

        count = group["frugal_num_active_blocks"]
        start = group["frugal_next_block_start"]
        for offset in range(count):
            param = params[(start - offset) % len(params)]
            state = self.state[param]
            self._init_state(param, state)
            state["active"] = True
        group["frugal_next_block_start"] = (start - count) % len(params)

    @torch.no_grad()
    def _init_group(self, group, skip_non_grad_params=False):
        if not any("active" in self.state[param] for param in group["params"]):
            self._rotate_group(group)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["frugal_step"] % group["update_gap"] == 0:
                self._rotate_group(group)
            group["frugal_step"] += 1

            beta1, beta2 = group["betas"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("FrugalAdamW does not support sparse gradients")

                param.mul_(1 - group["lr"] * group["weight_decay"])
                state = self.state[param]
                if not state["active"]:
                    param.add_(grad.sign(), alpha=-group["lr"])
                    continue

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2))
                denom.add_(group["eps"])
                param.addcdiv_(
                    exp_avg,
                    denom,
                    value=-group["lr"] / bias_correction1,
                )
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
        optimizer_cls=FrugalAdamW,
        config_to_kwargs=_frugal_config_to_kwargs,
        default_param_overrides={
            ParamKey(with_name_predicate=fallback): {"optimizer": "adam"}
        },
    )
