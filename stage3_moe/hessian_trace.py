"""Hutchinson estimate of the loss Hessian trace of a loaded checkpoint (customer hypothesis 2026-10-05:
"the BF16 baseline has a higher Hessian trace than the FP8-state arm").

Installed by stage3_moe/pretrain_gpt.py when STAGE3_MOE_HESSIAN_PATH is set, in the model-only evaluation mode
(`eval-lm-fixed`, --skip-train).  MCore's `evaluate` is wrapped: on its first call (the validation window) the
first STAGE3_MOE_HESSIAN_MICROBATCHES micro-batches of the iterator are cached, and for each of
STAGE3_MOE_HESSIAN_PROBES Rademacher vectors z (one torch generator seed, so every checkpoint sees the same z)
the Hessian-vector product is taken by central finite differences of the gradient,
    H z ~= (grad L(theta + eps z) - grad L(theta - eps z)) / (2 eps),
with the gradient of the mean micro-batch loss from MCore's own forward-backward schedule.  The trace estimate is
mean_z z^T H z; per parameter block (attention / dense MLP / routed experts / shared expert / router / embeddings
and head / norms) the same sum restricted to the block's coordinates estimates the block-diagonal trace.
STAGE3_MOE_HESSIAN_EPS_CHECK=1 repeats the first two probes with eps / 2 (Richardson check of the step).

Run the model in FP32 (STAGE3_MOE_FP32_MODEL=1 drops --bf16): with BF16 parameters eps cannot be made smaller
than the parameter grid spacing (~1e-4 at |w| ~ 0.02) without the perturbation rounding away.  Top-k routing is
evaluated at every perturbed point, so a token that changes expert between theta - eps z and theta + eps z enters
the difference as a jump; eps is small against the weights (eps / rms(w) ~ 5 %) and the eps / 2 check bounds it.
"""
import json
import math
import os
import time
from pathlib import Path

import torch

ROLE_RULES = (
    (".router.", "router"),
    (".experts.", "experts_routed"),
    (".shared_experts.", "experts_shared"),
    (".self_attention.", "attention"),
    (".mlp.", "dense_mlp"),
    ("embedding", "embedding_or_head"),
    ("output_layer", "embedding_or_head"),
    ("norm", "norm"),
)


def role_of(name):
    for needle, role in ROLE_RULES:
        if needle in name:
            return role
    return "other"


class Replay:
    """MCore's RerunDataIterator has __next__ but no __iter__: replay the cached batches, then delegate."""

    def __init__(self, cached, source):
        self.cached, self.source = list(cached), source

    def __iter__(self):
        return self

    def __next__(self):
        if self.cached:
            return self.cached.pop(0)
        return next(self.source)


def install(output_path):
    import megatron.training.training as training
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.training import get_args

    probes = int(os.environ.get("STAGE3_MOE_HESSIAN_PROBES", "16"))
    microbatches = int(os.environ.get("STAGE3_MOE_HESSIAN_MICROBATCHES", "8"))
    eps = float(os.environ.get("STAGE3_MOE_HESSIAN_EPS", "1e-3"))
    eps_check = os.environ.get("STAGE3_MOE_HESSIAN_EPS_CHECK", "1") == "1"
    seed = int(os.environ.get("STAGE3_MOE_HESSIAN_SEED", "20261008"))
    previous_evaluate = training.evaluate
    done = {"ran": False}

    def gradient(forward_backward_func, forward_step_func, model, batches, args):
        # --skip-train builds the model without the DDP wrapper: grads land in p.grad, not main_grad.
        if hasattr(model[0], "zero_grad_buffer"):
            model[0].zero_grad_buffer()
        for p in model[0].parameters():
            p.grad = None
        losses = forward_backward_func(forward_step_func=forward_step_func, data_iterator=Replay(batches, None), model=model,
                                       num_microbatches=len(batches), seq_length=args.seq_length,
                                       micro_batch_size=args.micro_batch_size, decoder_seq_length=args.decoder_seq_length,
                                       forward_only=False)
        loss = sum(float(d["lm loss"][0] if torch.is_tensor(d["lm loss"]) and d["lm loss"].numel() > 1 else d["lm loss"])
                   for d in losses) / len(losses)
        grads = []
        for p in model[0].parameters():
            g = getattr(p, "main_grad", None)
            if g is None:
                g = p.grad
            grads.append(g.detach().clone().float() if g is not None else torch.zeros_like(p, dtype=torch.float32))
        return grads, loss

    def hessian_trace(forward_step_func, data_iterator, model, config):
        from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine

        args = get_args()
        started = time.monotonic()
        forward_backward_func = get_forward_backward_func()
        # As MCore's evaluate does: with the rerun machine disabled, loss_func's validate_result is a no-op
        # (otherwise it asserts that it runs inside a train-step forward-backward pass).
        rerun_state_machine = get_rerun_state_machine()
        rerun_mode = rerun_state_machine.get_mode()
        rerun_state_machine.set_mode(RerunMode.DISABLED)
        batches = [next(data_iterator) for _ in range(microbatches)]
        params = [p for p in model[0].parameters()]
        names = [n for n, _ in model[0].named_parameters()]
        roles = [role_of(n) for n in names]
        counts = {}
        for role, p in zip(roles, params):
            counts[role] = counts.get(role, 0) + p.numel()
        dtype = {str(p.dtype) for p in params}
        generator = torch.Generator(device=params[0].device).manual_seed(seed)
        base_grads, base_loss = gradient(forward_backward_func, forward_step_func, model, batches, args)
        grad_norm = math.sqrt(sum(float(g.double().pow(2).sum()) for g in base_grads))
        print(f"COAT_HESSIAN_BASE loss={base_loss:.6f} grad_norm={grad_norm:.6g} params={sum(p.numel() for p in params)} "
              f"dtypes={sorted(dtype)} microbatches={microbatches} probes={probes} eps={eps}", flush=True)

        def probe(step, z):
            with torch.no_grad():
                for p, zi in zip(params, z):
                    p.add_(zi, alpha=step)
            plus, _ = gradient(forward_backward_func, forward_step_func, model, batches, args)
            with torch.no_grad():
                for p, zi in zip(params, z):
                    p.add_(zi, alpha=-2 * step)
            minus, _ = gradient(forward_backward_func, forward_step_func, model, batches, args)
            with torch.no_grad():
                for p, zi in zip(params, z):
                    p.add_(zi, alpha=step)
            per_role = {}
            total = 0.0
            for role, zi, gp, gm in zip(roles, z, plus, minus):
                value = float((zi.double() * (gp.double() - gm.double())).sum()) / (2 * step)
                per_role[role] = per_role.get(role, 0.0) + value
                total += value
            return total, per_role

        estimates, per_role_estimates, checks = [], [], []
        for index in range(probes):
            z = [torch.empty_like(p, dtype=torch.float32).bernoulli_(0.5, generator=generator).mul_(2).sub_(1) for p in params]
            total, per_role = probe(eps, z)
            estimates.append(total)
            per_role_estimates.append(per_role)
            if eps_check and index < 2:
                half_total, _ = probe(eps / 2, z)
                checks.append({"probe": index, "eps": total, "eps_half": half_total,
                               "relative_difference": abs(total - half_total) / max(abs(total), 1e-30)})
            running = sum(estimates) / len(estimates)
            sd = (sum((e - running) ** 2 for e in estimates) / max(len(estimates) - 1, 1)) ** 0.5
            print(f"COAT_HESSIAN_PROBE index={index} zHz={total:.6g} running_mean={running:.6g} running_sd={sd:.4g} "
                  f"elapsed={time.monotonic() - started:.0f}s", flush=True)
            del z
        n = len(estimates)
        mean = sum(estimates) / n
        sd = (sum((e - mean) ** 2 for e in estimates) / max(n - 1, 1)) ** 0.5
        result = {"tag": os.environ.get("STAGE3_MOE_PROBE_ARCHIVE_ID") or os.environ.get("COAT_FETCH_CKPT", "").split(":")[0],
                  "iteration": int(args.iteration), "loss": base_loss, "grad_norm": grad_norm, "probes": n, "eps": eps,
                  "microbatches": microbatches, "micro_batch_size": args.micro_batch_size, "seq_length": args.seq_length,
                  "param_dtypes": sorted(dtype), "parameters": sum(counts.values()),
                  "trace": mean, "trace_se": sd / math.sqrt(n), "trace_sd": sd, "trace_per_param": mean / sum(counts.values()),
                  "blocks": {role: {"parameters": counts[role],
                                    "trace": sum(e.get(role, 0.0) for e in per_role_estimates) / n,
                                    "trace_se": (sum((e.get(role, 0.0) - sum(x.get(role, 0.0) for x in per_role_estimates) / n) ** 2
                                                     for e in per_role_estimates) / max(n - 1, 1)) ** 0.5 / math.sqrt(n),
                                    "trace_per_param": sum(e.get(role, 0.0) for e in per_role_estimates) / n / counts[role]}
                             for role in counts},
                  "eps_checks": checks, "elapsed_seconds": round(time.monotonic() - started, 1)}
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=1) + "\n")
        print("COAT_HESSIAN " + json.dumps(result, separators=(",", ":")), flush=True)
        if hasattr(model[0], "zero_grad_buffer"):
            model[0].zero_grad_buffer()
        for p in model[0].parameters():
            p.grad = None
        rerun_state_machine.set_mode(rerun_mode)
        return batches

    def evaluate_with_hessian(forward_step_func, data_iterator, model, *args, **kwargs):
        if not done["ran"]:
            done["ran"] = True
            batches = hessian_trace(forward_step_func, data_iterator, model, kwargs.get("config"))
            # MCore evaluates eval_iters batches from the iterator; hand the cached ones back first.
            data_iterator = Replay(batches, data_iterator)
        return previous_evaluate(forward_step_func, data_iterator, model, *args, **kwargs)

    training.evaluate = evaluate_with_hessian
