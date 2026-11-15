"""AdEMAMix (Pagliardini, Ablin, Grangier, ICLR 2025) for the switching experiments of SOW Stage 3 item 3.

Adam with a second, slow EMA of the gradient (beta3 ~ 0.9999) mixed into the update with weight alpha:
    m1 = b1 m1 + (1-b1) g;  m2 = b3(t) m2 + (1-b3(t)) g;  v = b2 v + (1-b2) g^2
    p -= lr * ((m1/(1-b1^t) + alpha(t) m2) / (sqrt(v/(1-b2^t)) + eps) + wd p)
alpha and beta3 are warmed up over T steps as in the paper's reference code (linear for alpha, linear in
the half-life for beta3); T counts from the optimizer's first step. The paper switches from AdamW without
schedulers (T = 0, Fig. 5 and App. C.1.9), inheriting the AdamW state and setting m2 = 0: that is the warm
path below (load_state_dict from a TE FusedAdam checkpoint). A cold switch (--no-load-optim) restarts m1 and v
as well.

States: exp_avg, exp_avg_slow, exp_avg_sq. With fp8_states they go through the optimizer_states codec
(STAGE3_MOE_ADAM_STATE_CODEC=coat -> the COAT quantizer, E4M3, groups of 128, range expansion).
"""
import math
import os

import torch

from megatron.core.optimizer.emerging_optimizers import (
    EmergingOptimizerEntry,
    _EMERGING_OPTIMIZERS,
)

from stage3_moe.optimizer_states import FP8StateOptimizerMixin, StateSpec, adam_state_specs

BETA3 = float(os.environ.get("STAGE3_MOE_ADEMAMIX_BETA3", "0.9999"))
ALPHA = float(os.environ.get("STAGE3_MOE_ADEMAMIX_ALPHA", "5"))
# Warmup of alpha and beta3, in optimizer steps; 0 = no schedulers (the paper's resume setting).
WARMUP = int(os.environ.get("STAGE3_MOE_ADEMAMIX_WARMUP", "0"))


def _alpha_at(step, alpha, warmup):
    if warmup <= 0 or step >= warmup:
        return alpha
    return alpha * step / warmup


def _beta3_at(step, beta_end, beta_start, warmup):
    """Linear warmup of the EMA half-life, as in the reference implementation."""
    if warmup <= 0 or step >= warmup:
        return beta_end

    def half_life(beta, eps=1e-8):
        return math.log(0.5) / math.log(beta + eps) - 1

    def from_half_life(t):
        return math.pow(0.5, 1 / (t + 1))

    a = step / warmup
    return from_half_life((1 - a) * half_life(beta_start) + a * half_life(beta_end))


class AdEMAMix(torch.optim.Optimizer):
    optimizer_role = "ademamix_all"

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, BETA3), alpha=ALPHA, warmup=WARMUP, eps=1e-8, weight_decay=0.0):
        super().__init__(params, {"lr": lr, "betas": betas, "alpha": alpha, "warmup": warmup, "eps": eps,
                                  "weight_decay": weight_decay})

    def _init_group(self, group, skip_non_grad_params=False):
        for param in group["params"]:
            if skip_non_grad_params and param.grad is None:
                continue
            state = self.state[param]
            if state:
                continue
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_slow"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)

    @torch._disable_dynamo
    def load_state_dict(self, state_dict):
        """Warm switch from AdamW (the paper's protocol): m1 and v are inherited, m2 = 0. TE FusedAdam keeps
        one step counter per group, AdEMAMix one per parameter; alpha, beta3 and warmup are not in the stored
        groups and torch would drop them, so they come from the defaults."""
        state_dict = dict(state_dict)
        groups = [dict(group) for group in state_dict["param_groups"]]
        state = dict(state_dict["state"])
        imported = 0
        for group in groups:
            group_step = group.pop("step", None)
            for key in group["params"]:
                entry = state.get(key)
                if entry and "exp_avg_slow" not in entry:
                    if set(entry) - {"step"} != {"exp_avg", "exp_avg_sq"}:
                        raise ValueError(f"unrecognized Adam state for AdEMAMix import: keys {sorted(entry)}")
                    step = entry["step"] if "step" in entry else group_step
                    state[key] = {"exp_avg": entry["exp_avg"], "exp_avg_sq": entry["exp_avg_sq"],
                                  "exp_avg_slow": torch.zeros_like(entry["exp_avg"]),
                                  "step": int(step.item() if torch.is_tensor(step) else step)}
                    imported += 1
            if len(group["betas"]) == 2:
                group["betas"] = (*group["betas"], self.defaults["betas"][2])
            group.setdefault("alpha", self.defaults["alpha"])
            group.setdefault("warmup", self.defaults["warmup"])
        state_dict["param_groups"], state_dict["state"] = groups, state
        super().load_state_dict(state_dict)
        if imported:
            print(f"ADEMAMIX_STATE_IMPORT fp32_parameters={imported}", flush=True)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                if not state:
                    self._init_group({**group, "params": [param]})
                state["step"] += 1
                step = state["step"]
                alpha = _alpha_at(step, group["alpha"], group["warmup"])
                beta3_t = _beta3_at(step, beta3, beta1, group["warmup"])
                exp_avg, exp_avg_slow, exp_avg_sq = state["exp_avg"], state["exp_avg_slow"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_slow.mul_(beta3_t).add_(grad, alpha=1 - beta3_t)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group["weight_decay"]:
                    param.mul_(1 - group["lr"] * group["weight_decay"])
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                denom = (exp_avg_sq / bias_correction2).sqrt_().add_(group["eps"])
                update = exp_avg / bias_correction1
                update.add_(exp_avg_slow, alpha=alpha)
                param.addcdiv_(update, denom, value=-group["lr"])
        return loss


def ademamix_state_specs():
    """The three moments under the Adam codec: m1 and m2 like exp_avg, v like exp_avg_sq.

    m2's per-step relative change is 1 - beta3 = 1e-4, far below the E4M3 spacing, so under
    round-to-nearest it stops integrating (S8712, 2026-10-04). STAGE3_MOE_ADEMAMIX_M2_FP32=1 keeps m2
    in FP32 (the codec skips FP32 specs, the precision contract reads these specs);
    STAGE3_MOE_ADEMAMIX_M2_SR=1 keeps it in E4M3 with stochastic rounding.
    """
    first, second = adam_state_specs()
    fp32 = os.environ.get("STAGE3_MOE_ADEMAMIX_M2_FP32") == "1"
    stochastic = os.environ.get("STAGE3_MOE_ADEMAMIX_M2_SR") == "1"
    if fp32 and stochastic:
        raise ValueError("STAGE3_MOE_ADEMAMIX_M2_FP32 and STAGE3_MOE_ADEMAMIX_M2_SR exclude each other")
    slow = StateSpec("exp_avg_slow", first.signed, torch.float32 if fp32 else first.dtype, first.recipe,
                     stochastic=stochastic)
    return (first, slow, second)


class FP8StateAdEMAMix(FP8StateOptimizerMixin, AdEMAMix):
    state_specs = ()

    def __init__(self, *args, **kwargs):
        self.state_specs = ademamix_state_specs()
        super().__init__(*args, **kwargs)


def _config_to_kwargs(config, model_chunks, pg_collection):
    return {"lr": config.lr, "betas": (config.adam_beta1, config.adam_beta2, BETA3), "alpha": ALPHA, "warmup": WARMUP,
            "eps": config.adam_eps, "weight_decay": config.weight_decay}


def install_ademamix_contract(*, fp8_states: bool) -> None:
    _EMERGING_OPTIMIZERS["ademamix"] = EmergingOptimizerEntry(
        optimizer_cls=FP8StateAdEMAMix if fp8_states else AdEMAMix,
        config_to_kwargs=_config_to_kwargs,
        default_param_overrides={},
    )
