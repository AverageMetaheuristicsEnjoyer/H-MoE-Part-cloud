# Adapted from https://github.com/dayal-kalra/low-memory-adam (MIT).

import math

import torch

from megatron.core.optimizer.emerging_optimizers import (
    EmergingOptimizerEntry,
    _EMERGING_OPTIMIZERS,
)
from megatron.core.optimizer.optimizer_config import ParamKey, ParamWithNamePredicate


SLIM_COMPRESS_DIMS = "slim_compress_dims"


def slim_compression_dims(param: torch.Tensor, name: str):
    if len(param.shape) != 2 or name.endswith(".router.weight"):
        return None
    if (
        getattr(param, "is_embedding_or_output_parameter", False)
        or ".embedding.word_embeddings.weight" in name
        or name.endswith(".output_layer.weight")
    ):
        return (1,)
    if ".linear_fc1.weight" in name or ".linear_fc2.weight" in name:
        return (0,)
    if ".self_attention.linear_proj.weight" in name:
        return (0,)
    return None


def _uses_slim_dims(dims):
    return ParamWithNamePredicate(
        name=f"stage3_slim_dims_{dims[0]}",
        fn=lambda param, name: slim_compression_dims(param, name) == dims,
    )


class SlimAdamW(torch.optim.Optimizer):
    optimizer_role = "slimadam_all"

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                SLIM_COMPRESS_DIMS: None,
            },
        )

    @staticmethod
    def _compressed_square(value, dims):
        if dims is None:
            return value.square()
        return value.square().mean(dim=dims, keepdim=True)

    def _init_group(self, group, skip_non_grad_params=False):
        dims = group[SLIM_COMPRESS_DIMS]
        for param in group["params"]:
            if skip_non_grad_params and param.grad is None:
                continue
            state = self.state[param]
            if state:
                continue
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param)
            shape = list(param.shape)
            if dims is not None:
                for dim in dims:
                    shape[dim] = 1
            state["exp_avg_sq"] = torch.zeros(
                shape, dtype=param.dtype, device=param.device
            )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            dims = group[SLIM_COMPRESS_DIMS]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("SlimAdamW does not support sparse gradients")

                state = self.state[param]
                if not state:
                    self._init_group(
                        {**group, "params": [param]}, skip_non_grad_params=False
                    )
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(
                    self._compressed_square(grad, dims), alpha=1 - beta2
                )

                if group["weight_decay"]:
                    param.mul_(1 - group["lr"] * group["weight_decay"])

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


def _slimadam_config_to_kwargs(config, model_chunks, pg_collection):
    return {
        "lr": config.lr,
        "betas": (config.adam_beta1, config.adam_beta2),
        "eps": config.adam_eps,
        "weight_decay": config.weight_decay,
    }


def install_slimadam_contract() -> None:
    _EMERGING_OPTIMIZERS["slimadam"] = EmergingOptimizerEntry(
        optimizer_cls=SlimAdamW,
        config_to_kwargs=_slimadam_config_to_kwargs,
        default_param_overrides={
            ParamKey(with_name_predicate=_uses_slim_dims((0,))): {
                SLIM_COMPRESS_DIMS: (0,)
            },
            ParamKey(with_name_predicate=_uses_slim_dims((1,))): {
                SLIM_COMPRESS_DIMS: (1,)
            },
        },
    )
