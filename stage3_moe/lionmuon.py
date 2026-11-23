"""Muon-family optimizer switching for SOW Stage 3 item 3 (customer feedback 2026-10-05, colleagues' proposals).

Ports of the authors' reference code into the series' Megatron Muon (stage3_moe/muon.py ->
megatron.core.optimizer.emerging_optimizers.TensorParallelMuon), selected with STAGE3_MOE_MUON_VARIANT:

* ``lionmuon`` -- LionMuon, github.com/brain-lab-research/lion-muon ``src/optim/lion_muon.py`` (arXiv 2605.19811):
      update = beta1 * m + (1 - beta1) * g            # Lion interpolation from the previous buffer
      Muon step (step_count % P == 0, counted from 0): p -= lr * (NS(update) * 0.2 * sqrt(max(m, n)) + wd * p)
      Lion step (otherwise):                           p -= lion_lr * (sign(update) + wd * p)
      m = beta2 * m + (1 - beta2) * g                  # every step, after the direction
  ``lion_lr = lr * lion_lr_ratio`` follows the scheduler.  Paper / scripts: P = 2, (beta1, beta2) = (0.9, 0.99),
  NS 5 steps, sign LR 2e-5 ... 3.75e-5 against Muon LR 7.5e-4 ... 3e-3 (124M / 355M), i.e. ratios 20 ... 150; the
  code's default ratio is 0.01.  Here: STAGE3_MOE_LIONMUON_PERIOD (2), _BETA1 (0.9), _BETA2 (0.99),
  _LR_RATIO (100: eta_L = 1.63e-5 at the series' Muon LR).  Deviations from the reference: the buffer is the
  series' ``momentum_buffer`` (a warm switch inherits Muon's 0.95-EMA as M, the paper's single shared buffer);
  the 0.2 * sqrt(max(m, n)) scale is the series' spectral scale x extra_scale_factor (identical numbers); the split
  SwiGLU fc1 is orthogonalized per half as in the series' Muon; the 1-D / embedding / router fallback is the
  series' Adam at the schedule LR (reference: AdamW at a fixed 1e-3).

* ``softmuon`` -- SoftMuon, github.com/brain-lab-research/softsign ``softmuon/softmuon.py`` and the Megatron variant
  ``softmuon/d_softmuon.py`` (arXiv 2605.31371): for the first ``sign_iters`` optimizer steps plain Muon; at step
  ``sign_iters + 1`` a temperature schedule of ``transition_iters`` values is built once from the singular values of
  the *momentum buffers* M_{k+1} of all Muon parameters, after that step's momentum update (paper Algorithm 3; the
  reference ``get_temperature_schedule`` reads ``p.grad``, which a co-author confirmed 2026-10-08 is a naming bug and
  the momentum is meant) (pooled; folded-Cauchy fit mu = median, sigma = MAD; quantiles by Newton,
  10 iterations; tau_p = phi^-1(1 - eps) / q_p with phi(x) = x / sqrt(1 + x^2), clamped to [1, 1e9]); from then on
  the update is ``muon_regularized_update``: ``(X X^T + I / tau^2)^(-1/2) X`` by a coupled Newton-Schulz inverse
  square root in BF16 (reference code verbatim, ns_steps iterations), scaled like the series' Muon update.
  Here: STAGE3_MOE_SOFTMUON_START_ITER (global iteration of the first soft step, paper alpha_sign = 0.9 of training
  -> 15518 of 17242), _END_ITER (17242), _EPS (1e-4), _NEWTON_ITERS (10), _TMAX (1e9); the optimizer counts its
  own steps from STAGE3_MOE_PROBE_BASE_ITER.  Deviations: singular values via torch.linalg.svdvals (reference:
  the CANS polar SVD, same quantity); gate / up halves of the split fc1 soft-signed separately; no DP broadcast
  needed (gradients are already all-reduced and identical on every rank).

* LionMuon schedules (SOW Stage 4, colleagues' proposal "first more Lion, then more Muon"): which steps are Muon
  steps is decided on the *global* training step t (1-based, MCore's ``args.curr_iteration + 1``), so a resumed job
  continues the same sequence.  Schedules are written over the training fraction x = t / STAGE3_MOE_SCHEDULE_ITERS
  (the launcher sets it to the run's WSD length, 17242 for 1C, 2818 for the screen; a branch from a 1C trunk keeps
  17242) as ``"x1:v1,x2:v2,..."`` (``_`` may replace the commas, which mlsub --env rejects; a repeated x is a jump).
  STAGE3_MOE_LIONMUON_PERIOD is a period P or a piecewise-constant schedule of P (Muon when
  (t - 1 - STAGE3_MOE_PROBE_BASE_ITER) % P == 0, the reference rule for a constant P).  STAGE3_MOE_LIONMUON_PROB, if
  set, replaces the period by a probability p(x) of a Muon step, piecewise linear, with one uniform draw per step from
  ``random.Random("lionmuon-<seed>-<t>")`` -- identical on every rank and across restarts; STAGE3_MOE_LIONMUON_SEED
  (default STAGE3_MOE_SEED, 1234).  scripts/muon_presets.sh names the schedules, e.g. a dense colleague's five of
  2026-10-19 (mean p = 0.2 each).

* Per-group period (2026-10-22): STAGE3_MOE_LIONMUON_PERIOD_EXPERTS, a schedule like STAGE3_MOE_LIONMUON_PERIOD, sets
  the period of the routed-expert matrices (marked by name in stage3_moe/muon.py); the rest keeps
  STAGE3_MOE_LIONMUON_PERIOD.  Experts carry most of the Newton-Schulz cost.

* ``lionmuon2f`` (2026-10-22): LionMuon whose spectral step is Muon^2-F's -- the Lion interpolation is divided by the
  factored second moment of the gradient before Newton-Schulz; the factors are updated on every step, Lion steps
  included, and seeded from M * M like muon2f's.  Sign steps are unchanged.

* EnergyCal (arXiv 2610.37062v1, sections 3.2--3.3): STAGE3_MOE_LIONMUON_ENERGYCAL=1 attenuates only routed-expert
  spectral updates by q=clamp((rho/rho_ref)^exponent,qmin,1), with rho=||raw_NS||^2/min(shape), reconstructed from the
  existing scaled update in FP32. FC1 gate/up are separate. Defaults: rho_ref=.8, exponent=1, qmin=0. The configuration
  is stored in optimizer param groups and checked on resume; sign steps, LR and weight decay are unchanged.

* Power steps (colleague's proposal 2026-10-19, "Frankenstein"): STAGE3_MOE_LIONMUON_POWER = a(x), piecewise linear.
  With a > 0 the spectral step is U diag(s)^a V^T of the LionMuon direction (SVD in FP32; a = 0 is Muon's U V^T) and
  the coordinate step sign(G) |G|^a (a = 0 is Lion's sign).  Our normalisation, so that the LR means the same at every
  a: the spectral step is scaled to the Frobenius norm of U V^T (sqrt(min(m, n))) and then by the series' Muon scale,
  per block as Muon splits them (SwiGLU halves, Q / K / V); the coordinate step to RMS 1, the RMS of sign.  The SVD
  runs per block, so a > 0 makes spectral steps much slower than Newton-Schulz.

* ``normuon`` -- NorMuon, arXiv 2510.05491 Algorithm 1: O = NS(M); v = beta2 * v + (1 - beta2) * mean_cols(O * O),
  one scalar per row (output neuron); O_hat = O / (sqrt(v) + eps); update = O_hat * 0.2 * sqrt(m n) / ||O_hat||_F
  (RMS 0.2, the series' Muon update RMS).  Here: STAGE3_MOE_NORMUON_BETA2 (0.95, the paper's 1.1B / 5.4B value),
  _EPS (1e-8, not stated in the paper); the 0.2 is the series' extra_scale_factor.  Deviations: the series' Nesterov
  direction and momentum 0.95 go into NS (the paper: plain M); the Frobenius rescale is per SwiGLU half, as the
  series orthogonalizes the halves separately.  v stays FP32 (m floats per matrix) when the momentum is FP8.

* ``muon2`` / ``muon2f`` -- Muon^2, arXiv 2604.09967 Algorithms 1 and 2: V = beta2 * V + (1 - beta2) * G * G (raw
  gradient, no bias correction); M_tilde = M / (sqrt(V) + eps); O = NS(M_tilde, K).  muon2f stores V factored as
  Adafactor row / column sums r (m) and c (n) of G * G and uses V_hat = r c^T / sum(r) (the paper's Algorithm 2
  writes V_t in its line 11; the factored V_hat is what its Section 3.6 describes).  Here: STAGE3_MOE_MUON2_BETA2
  (0.95) and _EPS (1e-8) -- the paper gives no values; K by STAGE3_MOE_MUON_NS_STEPS (the paper compares 3 and 5).
  Deviations: the update keeps the series' scale 0.2 * sqrt(max(m, n)) (the paper: sqrt(m / n)) so the series' LR
  carries over; the Nesterov direction is preconditioned; the split fc1 keeps one column factor per half; V starts
  from M * M of the momentum it finds (zero from scratch, as in the paper), so a warm switch from a Muon checkpoint
  does not divide an inherited momentum by a one-step second moment.

STAGE3_MOE_MUON_NS_STEPS overrides --muon-num-ns-steps for every variant, plain Muon included.

STAGE3_MOE_LIONMUON_STEP_TELEMETRY=1 (screenb-015, default off) prints one passive ``LIONMUON_STEP`` line per
optimizer step with the realized choice and the running ``_lionmuon_muon``/``_lionmuon_steps`` counters, so a
PROB arm's realized Muon-step count is exact in the train log. Print only: no math, dtype, RNG or return change.

All keep the series' Muon contract otherwise (param groups, split fc1, Adam fallback, optional COAT FP8
momentum through FP8StateOptimizerMixin, whose ``_step_over`` calls ``super().step()`` -> the variant's step).
"""
import math
import os
import random

import torch

from emerging_optimizers import utils
from emerging_optimizers.orthogonalized_optimizers.muon import get_muon_scale_factor


def _telemetry_rank0():
    """True on the rank that owns the shared train log (rank 0).

    Data-parallel runs would otherwise print every LIONMUON_STEP line once per rank and the
    realized-schedule checker (lionmoe_fullrun_016.check_realized_schedule) rejects duplicates.
    """
    try:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return True
        return torch.distributed.get_rank() == 0
    except Exception:
        return True


def _env_float(name, default):
    return float(os.environ.get(name, default))


# --------------------------------------------------------------------------------------------------- LionMuon


def parse_schedule(text, cast):
    """``"v"`` -> [(0, v)]; ``"x1:v1,x2:v2"`` (or ``_`` between points: mlsub --env rejects commas) -> points sorted by
    the training fraction x; equal x keep their order, so a repeated x is a jump."""
    text = text.strip()
    if ":" not in text:
        return [(0.0, cast(text))]
    return sorted(((float(x), cast(v)) for x, v in (item.split(":") for item in text.replace("_", ",").split(","))),
                  key=lambda point: point[0])


def _last_at_or_before(points, x):
    index = 0
    for i, (start, _) in enumerate(points):
        if start <= x:
            index = i
    return index


def piecewise_constant(points, x):
    return points[_last_at_or_before(points, x)][1]


def piecewise_linear(points, x):
    i = _last_at_or_before(points, x)
    x0, v0 = points[i]
    if x <= x0 or i == len(points) - 1:
        return v0
    x1, v1 = points[i + 1]
    return v0 + (v1 - v0) * (x - x0) / (x1 - x0)


def megatron_iteration():
    """Iterations completed before the current optimizer step (MCore's args.curr_iteration); None outside training."""
    try:
        from megatron.training.global_vars import get_args
        return getattr(get_args(), "curr_iteration", None)
    except AssertionError:
        return None


class LionMuonMixin:
    """Alternating sign / spectral steps on one momentum buffer (lion_muon.py of the reference repo)."""

    lion_period = parse_schedule(os.environ.get("STAGE3_MOE_LIONMUON_PERIOD", "2"), int)
    lion_prob = parse_schedule(os.environ["STAGE3_MOE_LIONMUON_PROB"], float) if os.environ.get("STAGE3_MOE_LIONMUON_PROB") else None
    lion_power = parse_schedule(os.environ["STAGE3_MOE_LIONMUON_POWER"], float) if os.environ.get("STAGE3_MOE_LIONMUON_POWER") else None
    lion_total = int(os.environ.get("STAGE3_MOE_SCHEDULE_ITERS", "17242"))
    lion_seed = int(os.environ.get("STAGE3_MOE_LIONMUON_SEED", os.environ.get("STAGE3_MOE_SEED", "1234")))
    lion_base_iter = int(os.environ.get("STAGE3_MOE_PROBE_BASE_ITER", "0"))
    lion_beta1 = _env_float("STAGE3_MOE_LIONMUON_BETA1", "0.9")
    lion_beta2 = _env_float("STAGE3_MOE_LIONMUON_BETA2", "0.99")
    lion_lr_ratio = _env_float("STAGE3_MOE_LIONMUON_LR_RATIO", "100")
    lion_period_experts = (parse_schedule(os.environ["STAGE3_MOE_LIONMUON_PERIOD_EXPERTS"], int)
                           if os.environ.get("STAGE3_MOE_LIONMUON_PERIOD_EXPERTS") else None)
    lion_energycal = os.environ.get("STAGE3_MOE_LIONMUON_ENERGYCAL", "0") == "1"
    lion_energycal_rho_ref = _env_float("STAGE3_MOE_LIONMUON_ENERGYCAL_RHO_REF", "0.8")
    lion_energycal_exponent = _env_float("STAGE3_MOE_LIONMUON_ENERGYCAL_EXPONENT", "1")
    lion_energycal_qmin = _env_float("STAGE3_MOE_LIONMUON_ENERGYCAL_QMIN", "0")
    lion_fixed_routed_alpha = _env_float("STAGE3_MOE_LIONMUON_FIXED_ROUTED_ALPHA", "1")
    lion_energycal_diagnostics = os.environ.get("STAGE3_MOE_LIONMUON_ENERGYCAL_DIAGNOSTICS", "0") == "1"
    # Passive realized-choice telemetry (screenb-015): a print only, default off. No math/dtype/RNG/return change.
    lion_step_telemetry = os.environ.get("STAGE3_MOE_LIONMUON_STEP_TELEMETRY", "0") == "1"

    def _lion_energycal_config(self):
        if self.lion_energycal_rho_ref <= 0 or self.lion_energycal_exponent <= 0 or not 0 <= self.lion_energycal_qmin <= 1:
            raise ValueError("Invalid LionMuon EnergyCal configuration")
        if not 0 < self.lion_fixed_routed_alpha <= 1:
            raise ValueError("Fixed routed alpha must be in(0,1]")
        if self.lion_energycal and self.lion_fixed_routed_alpha != 1:
            raise ValueError("Adaptive EnergyCal and fixed routed alpha are mutually exclusive")
        return {"enabled": self.lion_energycal, "scope": "routed_expert_spectral",
                "rho_ref": self.lion_energycal_rho_ref, "exponent": self.lion_energycal_exponent,
                "qmin": self.lion_energycal_qmin, "fixed_routed_alpha": self.lion_fixed_routed_alpha}

    def _lion_energycal_group(self, group):
        config = self._lion_energycal_config()
        saved = group.get("lionmuon_energycal")
        if saved is not None and saved != config:
            raise ValueError(f"LionMuon EnergyCal checkpoint mismatch: {saved} != {config}")
        group["lionmuon_energycal"] = config

    def load_state_dict(self, state_dict):
        config = self._lion_energycal_config()
        for group in state_dict["param_groups"]:
            saved = group.get("lionmuon_energycal")
            if saved is None and config["enabled"]:
                raise ValueError("EnergyCal resume from a baseline checkpoint without calibration metadata requires an explicit switch")
            if saved is not None and saved != config:
                raise ValueError(f"LionMuon EnergyCal checkpoint mismatch: {saved} != {config}")
        return super().load_state_dict(state_dict)

    def _lion_energycal_apply(self, update, split_fc1):
        blocks = update.chunk(2, dim=0) if split_fc1 else (update,)
        diagnostics = []
        for index, block in enumerate(blocks):
            scale = get_muon_scale_factor(block.size(-2), block.size(-1), mode=self._soft_scale_mode) * self._soft_extra_scale
            rho = block.float().square().sum() / (scale * scale * min(block.shape))
            q = (rho / self.lion_energycal_rho_ref).pow(self.lion_energycal_exponent).clamp(self.lion_energycal_qmin, 1)
            block.mul_(q)
            if self.lion_energycal_diagnostics:
                diagnostics.append({"split": index, "shape": list(block.shape), "rho": float(rho), "q": float(q)})
        if self.lion_energycal_diagnostics:
            self._lion_energycal_last.extend(diagnostics)
        return update

    def _lion_fixed_alpha_apply(self, update, split_fc1):
        blocks = update.chunk(2, dim=0) if split_fc1 else (update,)
        for block in blocks:
            block.mul_(self.lion_fixed_routed_alpha)
        return update

    @staticmethod
    def _period_rule(points, count, x):
        period = piecewise_constant(points, x)
        if period < 1:
            raise ValueError("STAGE3_MOE_LIONMUON_PERIOD must be >= 1")
        # Reference: ``is_muon = (step_count % k == 0)`` evaluated before ``step_count += 1`` -> the first step is Muon.
        return period <= 1 or count % period == 0, period

    def _lion_observe(self, p, split_fc1):
        """Called on every step before the direction; the lionmuon2f variant updates its second moment here."""

    def _lion_spectral(self, p, update, split_fc1):
        """The direction a spectral step orthogonalizes; lionmuon2f preconditions it."""
        return update

    def _lionmuon_choice(self):
        """(is_muon_step, global step t, schedule value, power a) -- on the global iteration, so restarts keep the sequence."""
        local = getattr(self, "_lionmuon_steps", 0)
        self._lionmuon_steps = local + 1
        completed = megatron_iteration()
        count = local if completed is None else completed - self.lion_base_iter
        t = self.lion_base_iter + count + 1
        x = t / self.lion_total
        power = piecewise_linear(self.lion_power, x) if self.lion_power is not None else 0.0
        if self.lion_prob is not None:
            p = piecewise_linear(self.lion_prob, x)
            return random.Random(f"lionmuon-{self.lion_seed}-{t}").random() < p, t, p, power
        self._lion_count, self._lion_x = count, x
        is_muon, period = self._period_rule(self.lion_period, count, x)
        return is_muon, t, period, power

    def _power_orthogonalize(self, p, direction, power, group_kwargs):
        """U diag(s)^a V^T per block Muon orthogonalizes separately, scaled like Muon's U V^T."""
        def power_fn(block, tp_group=None, partition_dim=None):
            u, s, vh = torch.linalg.svd(block.float(), full_matrices=False)
            y = (u * s.pow(power)) @ vh
            y = y * (math.sqrt(s.numel()) / y.norm().clamp_min(1e-30))
            return y * (get_muon_scale_factor(block.size(-2), block.size(-1), mode=self._soft_scale_mode) * self._soft_extra_scale)

        newton_schulz = self.scaled_orthogonalize_fn
        self.scaled_orthogonalize_fn = power_fn
        try:
            return self.orthogonalize(p, direction, **group_kwargs)
        finally:
            self.scaled_orthogonalize_fn = newton_schulz

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        default_muon, t, value, power = self._lionmuon_choice()
        self._lionmuon_muon = getattr(self, "_lionmuon_muon", 0) + int(default_muon)
        if self.lion_step_telemetry and _telemetry_rank0():
            print(f"LIONMUON_STEP t={t} muon={int(default_muon)} muon_steps={self._lionmuon_muon} "
                  f"optimizer_steps={self._lionmuon_steps} rule={'prob' if self.lion_prob is not None else 'period'} "
                  f"value={value:.6g} power={power:.4g}", flush=True)
        if self.lion_energycal_diagnostics:
            self._lion_energycal_last = []
        measurement_observer = getattr(self, "_lionmuon_measurement_observer", None)
        measure = measurement_observer is not None and measurement_observer.enabled(t)
        if measure:
            measurement_observer.start_step(self, t)
        for group in self.param_groups:
            self._lion_energycal_group(group)
            self._init_group(group)
            group_kwargs = {k: v for k, v in group.items() if k != "params"}
            split_fc1 = group_kwargs.get("stage3_split_swiglu_fc1", False)
            is_muon_step = default_muon
            if (self.lion_period_experts is not None and self.lion_prob is None
                    and group_kwargs.get("stage3_routed_expert", False)):
                is_muon_step = self._period_rule(self.lion_period_experts, self._lion_count, self._lion_x)[0]
            lr = group["lr"] if is_muon_step else group["lr"] / self.lion_lr_ratio
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                momentum = state["momentum_buffer"]
                self._lion_observe(p, split_fc1)
                update = momentum * self.lion_beta1 + grad * (1 - self.lion_beta1)
                if is_muon_step and power > 0:
                    update = self._power_orthogonalize(p, update, power, group_kwargs)
                elif is_muon_step:
                    update = self._lion_spectral(p, update, split_fc1)
                    with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                        update = self.orthogonalize(p, update, **group_kwargs)
                elif power > 0:
                    update = update.sign() * update.abs().pow(power)
                    update.mul_(math.sqrt(update.numel()) / update.norm().clamp_min(1e-30))
                else:
                    update = update.sign_()
                if measure and is_muon_step:
                    measurement_observer.before_calibration(self, p, update, group_kwargs, t)
                if is_muon_step and power == 0 and self.lion_energycal and group_kwargs.get("stage3_routed_expert", False):
                    update = self._lion_energycal_apply(update, split_fc1)
                elif is_muon_step and power == 0 and self.lion_fixed_routed_alpha != 1 and group_kwargs.get("stage3_routed_expert", False):
                    update = self._lion_fixed_alpha_apply(update, split_fc1)
                if measure and is_muon_step:
                    measurement_observer.after_calibration(self, p, update, group_kwargs, t)
                momentum.mul_(self.lion_beta2).add_(grad, alpha=1 - self.lion_beta2)
                self._apply_weight_decay_inplace(p, grad, lr, group["weight_decay"])
                self.pre_weight_update_fn_inplace(p, update)
                p.add_(update, alpha=-lr)
                self.post_weight_update_fn_inplace(p)
        steps = self._lionmuon_steps
        if steps == 1 or steps % 500 == 0:
            rule = f"prob={value:.4g}" if self.lion_prob is not None else f"period={value}"
            experts = f" experts_period={piecewise_constant(self.lion_period_experts, self._lion_x)}" if self.lion_period_experts and self.lion_prob is None else ""
            print(f"LIONMUON step={t} fraction={t / self.lion_total:.4f} muon_step={default_muon} {rule}{experts} power={power:.4g} "
                  f"muon_fraction_this_process={self._lionmuon_muon / steps:.4f} "
                  f"beta1={self.lion_beta1} beta2={self.lion_beta2} lr_ratio={self.lion_lr_ratio} "
                  f"energycal={int(self.lion_energycal)} energycal_scope=routed_spectral "
                  f"rho_ref={self.lion_energycal_rho_ref} exponent={self.lion_energycal_exponent} qmin={self.lion_energycal_qmin} "
                  f"fixed_routed_alpha={self.lion_fixed_routed_alpha}", flush=True)
        if measure:
            measurement_observer.end_step(self, t)
        return loss


# --------------------------------------------------------------------------------------------------- SoftMuon


def newton_quantile(p, mu, sigma, max_iter=10, tol=1e-8):
    """Quantiles of the folded Cauchy distribution by Newton's method (reference softmuon.py, verbatim)."""
    import torch.distributions as td

    distribution = td.Cauchy(0, 1)
    q = sigma * distribution.icdf((1 + p) / 2) + mu.abs()
    for _ in range(max_iter):
        z1 = (q - mu) / (sigma + tol)
        z2 = (-q - mu) / (sigma + tol)
        F = distribution.cdf(z1) - distribution.cdf(z2) - p
        pdf_term = (distribution.log_prob(z1).exp() + distribution.log_prob(z2).exp()) / sigma
        q = q - F / pdf_term
        q = torch.clamp(q, min=0.0)
    return q


def inverse_soft_clipping(y):
    y = torch.as_tensor(y)
    return y / (1 - y ** 2) ** 0.5


def temperature_schedule(momenta, transition_iters, eps=1e-4, newton_iters=10, tmax=1e9):
    """Reference get_temperature_schedule on the momentum buffers: singular values pooled, folded-Cauchy fit."""
    device = momenta[0].device
    all_sigmas = torch.cat([torch.linalg.svdvals(m.float().reshape(m.size(0), -1)).flatten() for m in momenta])
    mu = all_sigmas.median()
    sigma = (all_sigmas - mu).abs().median()
    p = torch.arange(transition_iters, device=device, dtype=torch.float64) / transition_iters
    quantiles_p = newton_quantile(p, mu.double(), sigma.double(), newton_iters)
    schedule = inverse_soft_clipping(1 - eps) / quantiles_p
    return schedule.clamp(1.0, tmax), float(mu), float(sigma), int(all_sigmas.numel())


def muon_regularized_update(G, steps=10, temp=1.0):
    """Reference muon_regularized_update, verbatim: (X X^T + I / temp^2)^(-1/2) X by Newton-Schulz in BF16."""
    assert G.ndim >= 2
    X = G.bfloat16()
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True
    A = X @ X.mT
    I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
    A = A + I / (temp ** 2)
    normA = A.norm(dim=(-2, -1), keepdim=True)
    A = A / normA
    Y = A
    Z = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
    for _ in range(steps):
        T = 0.5 * (3 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    out = Z @ X
    out = out / normA.sqrt()
    if transposed:
        out = out.mT
    return out


class SoftMuonMixin:
    """Spectral soft sign with the reference temperature schedule (softmuon.py / d_softmuon.py)."""

    soft_start_iter = int(os.environ.get("STAGE3_MOE_SOFTMUON_START_ITER", "15518"))
    soft_end_iter = int(os.environ.get("STAGE3_MOE_SOFTMUON_END_ITER", "17242"))
    soft_eps = _env_float("STAGE3_MOE_SOFTMUON_EPS", "1e-4")
    soft_newton_iters = int(os.environ.get("STAGE3_MOE_SOFTMUON_NEWTON_ITERS", "10"))
    soft_tmax = _env_float("STAGE3_MOE_SOFTMUON_TMAX", "1e9")
    soft_base_iter = int(os.environ.get("STAGE3_MOE_PROBE_BASE_ITER", "0"))

    def _soft_sign_iters(self):
        return max(0, self.soft_start_iter - 1 - self.soft_base_iter)

    def _soft_advance(self):
        """Counts this optimizer step; True on the schedule step (reference: current_iter == sign_iters + 1)."""
        current = getattr(self, "_softmuon_steps", 0) + 1
        self._softmuon_steps = current
        return current == self._soft_sign_iters() + 1

    def _soft_build_schedule(self, momenta):
        transition_iters = max(1, self.soft_end_iter - self.soft_start_iter + 1)
        schedule, mu, sigma, count = temperature_schedule(momenta, transition_iters, self.soft_eps,
                                                          self.soft_newton_iters, self.soft_tmax)
        self._softmuon_schedule = schedule
        print(f"SOFTMUON schedule iteration={self.soft_base_iter + self._softmuon_steps} source=momentum "
              f"singular_values={count} mu={mu:.4g} sigma={sigma:.4g} tau_first={float(schedule[0]):.4g} "
              f"tau_mid={float(schedule[len(schedule) // 2]):.4g} tau_last={float(schedule[-1]):.4g} "
              f"transition_iters={transition_iters}", flush=True)

    def _soft_temperature(self):
        """Reference step(): the schedule indexed by the steps since sign_iters, clamped at the end."""
        schedule = getattr(self, "_softmuon_schedule", None)
        current, sign_iters = self._softmuon_steps, self._soft_sign_iters()
        if schedule is None or current <= sign_iters:
            return None
        index = max(0, min(current - sign_iters - 1, schedule.numel() - 1))
        return float(schedule[index])

    def _soft_block(self, block, tau):
        rows, cols = block.shape[-2], block.shape[-1]
        soft = muon_regularized_update(block, steps=self._soft_ns_steps, temp=tau).to(torch.float32)
        return soft * (get_muon_scale_factor(rows, cols, mode=self._soft_scale_mode) * self._soft_extra_scale)

    def _soft_orthogonalize(self, p, direction, tau, group_kwargs):
        if tau is None:
            return self.orthogonalize(p, direction, **group_kwargs)
        if group_kwargs.get("stage3_split_swiglu_fc1", False):
            gate, up = direction.chunk(2, dim=0)
            return torch.cat((self._soft_block(gate, tau), self._soft_block(up, tau)), dim=0)
        return self._soft_block(direction, tau)

    @torch.no_grad()
    def step(self, closure=None):
        if getattr(self, "dequantize_chunk", 0) > 0:
            # One step() per parameter chunk would advance the schedule per chunk and fit it on one chunk's momenta.
            raise ValueError("SoftMuon does not support STAGE3_MOE_FP8_DEQUANT_CHUNK > 0")
        loss = closure() if closure is not None else None
        schedule_step = self._soft_advance()
        if schedule_step:
            # Paper Algorithm 3: the singular values of M_{k+1}, i.e. after this step's momentum update (weight decay
            # first, as in the per-parameter loop below, which then skips both for this step).
            momenta = []
            for group in self.param_groups:
                self._init_group(group)
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self._apply_weight_decay_inplace(p, p.grad, group["lr"], group["weight_decay"])
                    self.state[p]["momentum_buffer"].lerp_(p.grad, 1 - group["momentum"])
                    momenta.append(self.state[p]["momentum_buffer"])
            self._soft_build_schedule(momenta)
        tau = self._soft_temperature()
        for group in self.param_groups:
            self._init_group(group)
            group_kwargs = {k: v for k, v in group.items() if k != "params"}
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not schedule_step:
                    self._apply_weight_decay_inplace(p, grad, group["lr"], group["weight_decay"])
                    state["momentum_buffer"].lerp_(grad, 1 - group["momentum"])
                direction = grad.lerp(state["momentum_buffer"], group["momentum"]) if self.nesterov else state["momentum_buffer"]
                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    update = self._soft_orthogonalize(p, direction, tau, group_kwargs)
                self.pre_weight_update_fn_inplace(p, update)
                p.add_(update, alpha=-group["lr"])
                self.post_weight_update_fn_inplace(p)
        steps = self._softmuon_steps
        if tau is not None and (steps % 250 == 0 or steps == max(0, self.soft_start_iter - self.soft_base_iter)):
            print(f"SOFTMUON iteration={self.soft_base_iter + steps} tau={tau:.6g}", flush=True)
        return loss


# --------------------------------------------------------------------------------------------------- NorMuon / Muon^2


def _row_blocks(rows, split_fc1):
    """Row slices orthogonalized together: the two SwiGLU halves of fc1, else the whole matrix."""
    if split_fc1:
        return [slice(0, rows // 2), slice(rows // 2, rows)]
    return [slice(0, rows)]


class _MomentumStepMixin:
    """The series' Muon step up to the direction fed to NS: decoupled weight decay, momentum EMA, Nesterov."""

    def _direction(self, p, group):
        state = self.state[p]
        self._apply_weight_decay_inplace(p, p.grad, group["lr"], group["weight_decay"])
        state["momentum_buffer"].lerp_(p.grad, 1 - group["momentum"])
        if self.nesterov:
            return p.grad.lerp(state["momentum_buffer"], group["momentum"])
        return state["momentum_buffer"]

    def _apply(self, p, update, lr):
        self.pre_weight_update_fn_inplace(p, update)
        p.add_(update, alpha=-lr)
        self.post_weight_update_fn_inplace(p)


class NorMuonMixin(_MomentumStepMixin):
    """NorMuon (arXiv 2510.05491, Algorithm 1): row-wise second moment of the orthogonalized update."""

    normuon_beta2 = _env_float("STAGE3_MOE_NORMUON_BETA2", "0.95")
    normuon_eps = _env_float("STAGE3_MOE_NORMUON_EPS", "1e-8")
    # Diagnosis 2026-10-22: "experts" = NorMuon only on routed-expert matrices, "dense" = everywhere else; the rest gets Muon.
    normuon_scope = os.environ.get("STAGE3_MOE_NORMUON_SCOPE", "all")

    def _normuon_applies(self, group):
        if self.normuon_scope == "all":
            return True
        return group.get("stage3_routed_expert", False) == (self.normuon_scope == "experts")

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            self._init_group(group)
            group_kwargs = {k: v for k, v in group.items() if k != "params"}
            for p in group["params"]:
                if p.grad is None:
                    continue
                direction = self._direction(p, group)
                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    orth = self.orthogonalize(p, direction, **group_kwargs).float()
                if not self._normuon_applies(group):
                    self._apply(p, orth, group["lr"])
                    continue
                state = self.state[p]
                if "normuon_v" not in state:
                    state["normuon_v"] = torch.zeros(orth.shape[0], 1, dtype=torch.float32, device=orth.device)
                state["normuon_v"].lerp_(orth.square().mean(dim=1, keepdim=True), 1 - self.normuon_beta2)
                update = orth / (state["normuon_v"].sqrt() + self.normuon_eps)
                for rows in _row_blocks(update.shape[0], group_kwargs.get("stage3_split_swiglu_fc1", False)):
                    block = update[rows]
                    # An expert that has never received a token has a zero update; keep it zero instead of 0 / 0.
                    block.mul_(self._ns_extra_scale * math.sqrt(block.numel()) / block.norm().clamp_min(1e-30))
                self._apply(p, update, group["lr"])
        if not getattr(self, "_normuon_logged", False):
            self._normuon_logged = True
            routed = sum(len(g["params"]) for g in self.param_groups if g.get("stage3_routed_expert", False))
            print(f"NORMUON routed_expert_params={routed} of {sum(len(g['params']) for g in self.param_groups)} scope={self.normuon_scope} beta2={self.normuon_beta2} eps={self.normuon_eps} update_rms={self._ns_extra_scale} "
                  f"ns_steps={self._ns_steps}", flush=True)
        return loss


class Muon2Mixin(_MomentumStepMixin):
    """Muon^2 (arXiv 2604.09967): Adam second moment of the gradient applied before NS; factored = Muon^2-F."""

    muon2_factored = False
    muon2_beta2 = _env_float("STAGE3_MOE_MUON2_BETA2", "0.95")
    muon2_eps = _env_float("STAGE3_MOE_MUON2_EPS", "1e-8")

    def _init_second_moment(self, p, split_fc1):
        """Before this step's momentum update: V = M * M of the momentum found (zero from scratch)."""
        state = self.state[p]
        if "muon2_v" in state or "muon2_row" in state:
            return
        seed = state["momentum_buffer"].float().square()
        if not self.muon2_factored:
            state["muon2_v"] = seed
            return
        state["muon2_row"] = seed.sum(dim=1)
        state["muon2_col"] = torch.stack([seed[rows].sum(dim=0) for rows in _row_blocks(seed.shape[0], split_fc1)])

    def _m2_update(self, p, split_fc1):
        """EMA of the squared gradient: full V, or its row sums and per-block column sums."""
        state = self.state[p]
        squared = p.grad.float().square()
        if not self.muon2_factored:
            state["muon2_v"].lerp_(squared, 1 - self.muon2_beta2)
            return
        state["muon2_row"].lerp_(squared.sum(dim=1), 1 - self.muon2_beta2)
        for index, rows in enumerate(_row_blocks(squared.shape[0], split_fc1)):
            state["muon2_col"][index].lerp_(squared[rows].sum(dim=0), 1 - self.muon2_beta2)

    def _m2_divide(self, p, direction, split_fc1):
        state = self.state[p]
        if not self.muon2_factored:
            return direction / (state["muon2_v"].sqrt() + self.muon2_eps)
        preconditioned = torch.empty(direction.shape, dtype=torch.float32, device=direction.device)
        for index, rows in enumerate(_row_blocks(direction.shape[0], split_fc1)):
            row = state["muon2_row"][rows]
            v_hat = torch.outer(row, state["muon2_col"][index]) / row.sum().clamp_min(1e-30)
            preconditioned[rows] = direction[rows] / (v_hat.sqrt() + self.muon2_eps)
        return preconditioned

    def _precondition(self, p, direction, split_fc1):
        self._m2_update(p, split_fc1)
        return self._m2_divide(p, direction, split_fc1)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            self._init_group(group)
            group_kwargs = {k: v for k, v in group.items() if k != "params"}
            for p in group["params"]:
                if p.grad is None:
                    continue
                split_fc1 = group_kwargs.get("stage3_split_swiglu_fc1", False)
                self._init_second_moment(p, split_fc1)
                direction = self._direction(p, group)
                preconditioned = self._precondition(p, direction, split_fc1)
                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    update = self.orthogonalize(p, preconditioned, **group_kwargs)
                self._apply(p, update, group["lr"])
        if not getattr(self, "_muon2_logged", False):
            self._muon2_logged = True
            print(f"MUON2 factored={self.muon2_factored} beta2={self.muon2_beta2} eps={self.muon2_eps} "
                  f"ns_steps={self._ns_steps}", flush=True)
        return loss


class Muon2FMixin(Muon2Mixin):
    muon2_factored = True


class LionMuon2FMixin(LionMuonMixin, Muon2FMixin):
    """LionMuon with Muon^2-F's preconditioned spectral step (see the module docstring)."""

    def _lion_observe(self, p, split_fc1):
        self._init_second_moment(p, split_fc1)
        self._m2_update(p, split_fc1)

    def _lion_spectral(self, p, update, split_fc1):
        return self._m2_divide(p, update, split_fc1)


# --------------------------------------------------------------------------------------------------- wiring

VARIANTS = {"lionmuon": LionMuonMixin, "softmuon": SoftMuonMixin, "normuon": NorMuonMixin,
            "muon2": Muon2Mixin, "muon2f": Muon2FMixin, "lionmuon2f": LionMuon2FMixin}


def variant_name():
    name = os.environ.get("STAGE3_MOE_MUON_VARIANT", "muon").strip().lower()
    if name != "muon" and name not in VARIANTS:
        raise ValueError(f"unknown STAGE3_MOE_MUON_VARIANT: {name!r}")
    return name


def apply_variant(base_classes):
    """The class tuple with the selected variant's mixin in front (after the FP8 mixin if present)."""
    name = variant_name()
    if name == "muon":
        return base_classes
    classes = list(base_classes)
    position = 1 if classes and classes[0].__name__ == "FP8StateOptimizerMixin" else 0
    classes.insert(position, VARIANTS[name])
    return tuple(classes)


def remember_ns_settings(kwargs):
    """Apply STAGE3_MOE_MUON_NS_STEPS; SoftMuon and NorMuon need the settings TensorParallelMuon keeps in a closure."""
    if os.environ.get("STAGE3_MOE_MUON_NS_STEPS"):
        kwargs["num_ns_steps"] = int(os.environ["STAGE3_MOE_MUON_NS_STEPS"])
    SoftMuonMixin._soft_ns_steps = int(kwargs.get("num_ns_steps", 5))
    SoftMuonMixin._soft_scale_mode = kwargs.get("scale_mode", "spectral")
    SoftMuonMixin._soft_extra_scale = float(kwargs.get("extra_scale_factor", 1.0))
    _MomentumStepMixin._ns_steps = SoftMuonMixin._soft_ns_steps
    _MomentumStepMixin._ns_extra_scale = SoftMuonMixin._soft_extra_scale
    LionMuonMixin._soft_scale_mode = SoftMuonMixin._soft_scale_mode
    LionMuonMixin._soft_extra_scale = SoftMuonMixin._soft_extra_scale
    print(f"stage3 muon ns_steps={SoftMuonMixin._soft_ns_steps} scale_mode={SoftMuonMixin._soft_scale_mode} "
          f"extra_scale={SoftMuonMixin._soft_extra_scale}", flush=True)
    return kwargs
