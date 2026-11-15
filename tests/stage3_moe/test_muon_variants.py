"""CPU checks of the Stage 4 Muon variants in stage3_moe/lionmuon.py (LionMuon schedules, NorMuon, Muon^2, Muon^2-F)."""
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "Megatron-LM"))
sys.path.insert(0, str(ROOT / "third_party" / "emerging-optimizers"))

from stage3_moe import lionmuon  # noqa: E402
from stage3_moe.lionmuon import (  # noqa: E402
    LionMuon2FMixin,
    LionMuonMixin,
    Muon2FMixin,
    Muon2Mixin,
    NorMuonMixin,
    _MomentumStepMixin,
    parse_schedule,
    piecewise_constant,
    piecewise_linear,
)


class Host:
    """The part of the series' TensorParallelMuon the variant mixins use; orthogonalize is the identity."""

    nesterov = False
    fp32_matmul_prec = "highest"

    def __init__(self, params, momentum=0.9, weight_decay=0.0, lr=1.0, split_fc1=False):
        self.param_groups = [{"params": params, "lr": lr, "momentum": momentum, "weight_decay": weight_decay,
                              "stage3_split_swiglu_fc1": split_fc1}]
        self.state = {p: {} for p in params}

    def _init_group(self, group):
        for p in group["params"]:
            if not self.state[p]:
                self.state[p]["momentum_buffer"] = torch.zeros_like(p)

    def orthogonalize(self, p, grad, **kwargs):
        return grad.clone()

    def _apply_weight_decay_inplace(self, p, grad, lr, weight_decay):
        p.mul_(1 - lr * weight_decay)

    def pre_weight_update_fn_inplace(self, p, update):
        pass

    def post_weight_update_fn_inplace(self, p):
        pass


def make(mixin, params, **kwargs):
    cls = type(f"Test{mixin.__name__}", (mixin, Host), {})
    return cls(params, **kwargs)


def test_schedule_parsing_and_interpolation():
    assert parse_schedule("2", int) == [(0.0, 2)]
    points = parse_schedule("0.5:2,0:5,0.9:1", int)
    assert points == [(0.0, 5), (0.5, 2), (0.9, 1)]
    assert [piecewise_constant(points, x) for x in (0, 0.49, 0.5, 0.89, 0.9, 1.2)] == [5, 5, 2, 2, 1, 1]
    linear = parse_schedule("0:0.05,1:0.35", float)
    assert math.isclose(piecewise_linear(linear, 0), 0.05)
    assert math.isclose(piecewise_linear(linear, 0.5), 0.2)
    assert math.isclose(piecewise_linear(linear, 3), 0.35)
    jump = parse_schedule("0:0.2222,0.9:0.2222,0.9:0", float)
    assert math.isclose(piecewise_linear(jump, 0.8999), 0.2222) and piecewise_linear(jump, 0.9) == 0
    assert parse_schedule("0:0.05_1:0.35", float) == linear


def test_presets_parse_to_known_variants():
    import subprocess
    script = ROOT / "scripts" / "muon_presets.sh"
    names = ("lm2", "lm5", "sm2", "lmlin", "lmlinrev", "lmtail0", "lmtail8", "fklin", "fktail", "nm", "m2", "m2f", "m2f3", "ns3",
             "nmexp", "nmdense", "nmb999", "lmph", "lmphrev", "lmexp10", "lm5r50", "lm5r200", "lm5m2f3")
    for name in names:
        words = subprocess.run(["bash", "-c", f"source {script}; muon_preset {name}"], capture_output=True, text=True, check=True).stdout.split()
        values = dict(word.split("=", 1) for word in words)
        assert values.get("STAGE3_MOE_MUON_VARIANT", "muon") in ("muon", *lionmuon.VARIANTS)
        if "STAGE3_MOE_LIONMUON_PROB" in values:
            points = parse_schedule(values["STAGE3_MOE_LIONMUON_PROB"], float)
            for total in (2818, 17242):
                mean = sum(piecewise_linear(points, t / total) for t in range(1, total + 1)) / total
                assert abs(mean - 0.2) < 0.002, (name, total, mean)
    assert subprocess.run(["bash", "-c", f"source {script}; muon_preset nope"]).returncode == 1


def lion(**attributes):
    optimizer = make(LionMuonMixin, [torch.nn.Parameter(torch.zeros(2, 2))])
    optimizer.lion_period, optimizer.lion_prob, optimizer.lion_power = [(0.0, 2)], None, None
    optimizer.lion_base_iter, optimizer.lion_total, optimizer.lion_seed = 0, 17242, 1234
    for key, value in attributes.items():
        setattr(optimizer, key, value)
    return optimizer


def test_lionmuon_constant_period_matches_reference_rule(monkeypatch):
    monkeypatch.setattr(lionmuon, "megatron_iteration", lambda: None)
    optimizer = lion(lion_period=[(0.0, 3)])
    assert [optimizer._lionmuon_choice()[0] for _ in range(7)] == [True, False, False, True, False, False, True]


def test_lionmuon_follows_the_global_iteration_across_a_restart(monkeypatch):
    optimizer = lion()
    monkeypatch.setattr(lionmuon, "megatron_iteration", lambda: 5)  # a fresh process resumed after 5 iterations
    assert optimizer._lionmuon_choice()[:2] == (False, 6)
    monkeypatch.setattr(lionmuon, "megatron_iteration", lambda: 6)
    assert optimizer._lionmuon_choice()[:2] == (True, 7)


def test_lionmuon_probability_schedule_is_deterministic_and_has_its_mean(monkeypatch):
    clock = {"t": 0}
    monkeypatch.setattr(lionmuon, "megatron_iteration", lambda: clock["t"])

    def draws(seed):
        optimizer = lion(lion_prob=parse_schedule("0:0.05,1:0.35", float), lion_seed=seed, lion_total=20000)
        out = []
        for t in range(20000):
            clock["t"] = t
            out.append(optimizer._lionmuon_choice()[0])
        return out

    first, again, other = draws(1234), draws(1234), draws(99)
    assert first == again and first != other
    assert abs(sum(first) / len(first) - 0.2) < 0.01
    assert abs(sum(first[:2000]) / 2000 - 0.065) < 0.02 and abs(sum(first[-2000:]) / 2000 - 0.335) < 0.04


class SpectralHost(Host):
    """orthogonalize goes through scaled_orthogonalize_fn, as TensorParallelMuon's does."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaled_orthogonalize_fn = lambda grad, tp_group=None, partition_dim=None: grad * 0 + 7.0

    def orthogonalize(self, p, grad, **kwargs):
        return self.scaled_orthogonalize_fn(grad, None, None)


def test_power_steps(monkeypatch):
    monkeypatch.setattr(lionmuon, "megatron_iteration", lambda: None)
    cls = type("TestPowerLionMuon", (LionMuonMixin, SpectralHost), {})
    p = torch.nn.Parameter(torch.zeros(4, 8))
    optimizer = cls([p])
    optimizer.lion_period, optimizer.lion_prob, optimizer.lion_power = [(0.0, 1)], None, [(0.0, 0.5)]
    optimizer.lion_base_iter, optimizer.lion_total = 0, 100
    lionmuon.remember_ns_settings({"num_ns_steps": 5, "scale_mode": "spectral", "extra_scale_factor": 0.2})
    torch.manual_seed(2)
    u, _ = torch.linalg.qr(torch.randn(4, 4))
    v, _ = torch.linalg.qr(torch.randn(8, 4))
    singular = torch.tensor([4.0, 2.0, 1.0, 0.25])
    direction = (u * singular) @ v.T
    newton_schulz = optimizer.scaled_orthogonalize_fn
    out = optimizer._power_orthogonalize(p, direction, 0.5, {})
    assert optimizer.scaled_orthogonalize_fn is newton_schulz
    expected = (u * singular.sqrt()) @ v.T
    expected = expected * (2.0 / expected.norm()) * (math.sqrt(8) * 0.2)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)
    # The coordinate step at a = 0.5: sign(G) |G|^0.5 scaled to RMS 1, at lr / lr_ratio.
    q = torch.nn.Parameter(torch.zeros(3, 3))
    q.grad = torch.randn(3, 3)
    coordinate = cls([q])
    coordinate.lion_period, coordinate.lion_prob, coordinate.lion_power = [(0.0, 2)], None, [(0.0, 0.5)]
    coordinate.lion_base_iter, coordinate.lion_total, coordinate.lion_beta1, coordinate.lion_lr_ratio = 0, 100, 0.0, 1.0
    coordinate._lionmuon_steps = 1  # the second step of period 2: a coordinate step
    grad = q.grad.clone()
    coordinate.step()
    step = grad.sign() * grad.abs().sqrt()
    torch.testing.assert_close(-q.detach(), step * (3.0 / step.norm()))


def test_normuon_row_moment_and_update_rms():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(8, 6))
    p.grad = torch.randn(8, 6)
    optimizer = make(NorMuonMixin, [p], momentum=0.0)
    optimizer._ns_extra_scale, optimizer._ns_steps = 0.2, 5
    optimizer.normuon_beta2, optimizer.normuon_eps = 0.95, 0.0
    grad = p.grad.clone()
    optimizer.step()
    v = optimizer.state[p]["normuon_v"]
    assert v.shape == (8, 1)
    torch.testing.assert_close(v, 0.05 * grad.square().mean(dim=1, keepdim=True))
    expected = grad / v.sqrt()
    expected = expected * 0.2 * math.sqrt(expected.numel()) / expected.norm()
    torch.testing.assert_close(p.detach(), -expected)
    assert math.isclose(p.detach().square().mean().sqrt().item(), 0.2, rel_tol=1e-5)


def test_normuon_rescales_swiglu_halves_separately_and_keeps_dead_rows_finite():
    p = torch.nn.Parameter(torch.zeros(4, 3))
    p.grad = torch.cat([torch.zeros(2, 3), torch.randn(2, 3)])
    optimizer = make(NorMuonMixin, [p], momentum=0.0, split_fc1=True)
    optimizer._ns_extra_scale, optimizer._ns_steps = 0.2, 5
    optimizer.step()
    update = -p.detach()
    assert torch.isfinite(update).all()
    assert torch.equal(update[:2], torch.zeros(2, 3))
    assert math.isclose(update[2:].square().mean().sqrt().item(), 0.2, rel_tol=1e-5)


def _rank_one_abs_gradient(rows, cols, seed):
    torch.manual_seed(seed)
    a, b = torch.rand(rows) + 0.5, torch.rand(cols) + 0.5
    return torch.outer(a, b) * torch.randn(rows, cols).sign()


def test_muon2_preconditions_momentum_by_the_gradient_second_moment():
    p = torch.nn.Parameter(torch.zeros(4, 5))
    p.grad = torch.randn(4, 5)
    optimizer = make(Muon2Mixin, [p], momentum=0.0)
    optimizer._ns_steps, optimizer.muon2_beta2, optimizer.muon2_eps = 5, 0.95, 0.0
    grad = p.grad.clone()
    optimizer.step()
    torch.testing.assert_close(optimizer.state[p]["muon2_v"], 0.05 * grad.square())
    torch.testing.assert_close(-p.detach(), grad / (0.05 * grad.square()).sqrt())


def test_muon2_second_moment_starts_from_the_inherited_momentum():
    p = torch.nn.Parameter(torch.zeros(4, 6))
    p.grad = torch.randn(4, 6)
    optimizer = make(Muon2FMixin, [p], momentum=0.9, split_fc1=True)
    optimizer._ns_steps, optimizer.muon2_beta2, optimizer.muon2_eps = 5, 0.95, 0.0
    inherited = torch.randn(4, 6)
    optimizer.state[p]["momentum_buffer"] = inherited.clone()
    grad = p.grad.clone()
    optimizer.step()
    row = 0.95 * inherited.square().sum(dim=1) + 0.05 * grad.square().sum(dim=1)
    torch.testing.assert_close(optimizer.state[p]["muon2_row"], row)
    col_top = 0.95 * inherited[:2].square().sum(dim=0) + 0.05 * grad[:2].square().sum(dim=0)
    torch.testing.assert_close(optimizer.state[p]["muon2_col"][0], col_top)


def test_muon2f_equals_muon2_when_the_squared_gradient_is_rank_one():
    for split in (False, True):
        # Different factors per half: the whole matrix is rank 2, so only per-half column factors reproduce V exactly.
        grad = torch.cat([_rank_one_abs_gradient(3, 5, 1), _rank_one_abs_gradient(3, 5, 2)]) if split else _rank_one_abs_gradient(6, 5, 1)
        results = []
        for mixin in (Muon2Mixin, Muon2FMixin):
            p = torch.nn.Parameter(torch.zeros(6, 5))
            p.grad = grad.clone()
            optimizer = make(mixin, [p], momentum=0.5, split_fc1=split)
            optimizer._ns_steps, optimizer.muon2_beta2, optimizer.muon2_eps = 5, 0.95, 0.0
            optimizer.step()
            results.append(p.detach().clone())
            if mixin is Muon2FMixin:
                state = optimizer.state[p]
                assert state["muon2_row"].shape == (6,) and state["muon2_col"].shape == (2 if split else 1, 5)
        torch.testing.assert_close(results[0], results[1])


def test_variant_wiring_and_ns_override(monkeypatch):
    from stage3_moe.optimizer_states import FP8StateOptimizerMixin

    class Base:
        pass

    for name, mixin in (("normuon", NorMuonMixin), ("muon2", Muon2Mixin), ("muon2f", Muon2FMixin), ("lionmuon", LionMuonMixin)):
        monkeypatch.setenv("STAGE3_MOE_MUON_VARIANT", name)
        assert lionmuon.apply_variant((FP8StateOptimizerMixin, Base)) == (FP8StateOptimizerMixin, mixin, Base)
    monkeypatch.setenv("STAGE3_MOE_MUON_NS_STEPS", "3")
    kwargs = lionmuon.remember_ns_settings({"num_ns_steps": 5, "scale_mode": "spectral", "extra_scale_factor": 0.2})
    assert kwargs["num_ns_steps"] == 3 and _MomentumStepMixin._ns_steps == 3 and _MomentumStepMixin._ns_extra_scale == 0.2


def test_normuon_scope_keeps_plain_muon_off_scope():
    torch.manual_seed(3)
    p = torch.nn.Parameter(torch.zeros(8, 6))
    p.grad = torch.randn(8, 6)
    grad = p.grad.clone()
    optimizer = make(NorMuonMixin, [p], momentum=0.0)
    optimizer.param_groups[0]["stage3_routed_expert"] = False
    optimizer._ns_extra_scale, optimizer._ns_steps = 0.2, 5
    optimizer.normuon_scope = "experts"
    optimizer.step()
    torch.testing.assert_close(-p.detach(), grad)  # identity orthogonalize, no row normalization
    assert "normuon_v" not in optimizer.state[p]
    optimizer.normuon_scope = "dense"
    p.grad = grad.clone()
    optimizer.step()
    assert optimizer.state[p]["normuon_v"].shape == (8, 1)


def test_phase_schedule_switches_period_at_training_fractions(monkeypatch):
    clock = {"t": 0}
    monkeypatch.setattr(lionmuon, "megatron_iteration", lambda: clock["t"])
    optimizer = lion(lion_period=parse_schedule("0:10_0.4:5_0.8:2", int), lion_total=100)
    muon_steps = []
    for t in range(100):
        clock["t"] = t
        if optimizer._lionmuon_choice()[0]:
            muon_steps.append(t + 1)
    assert [s for s in muon_steps if s <= 40] == [1, 11, 21, 31]
    assert [s for s in muon_steps if 40 < s <= 80] == [41, 46, 51, 56, 61, 66, 71, 76]
    assert [s for s in muon_steps if s > 80] == list(range(81, 101, 2))


def test_routed_experts_follow_their_own_period(monkeypatch):
    monkeypatch.setattr(lionmuon, "megatron_iteration", lambda: None)
    cls = type("TestExpertLionMuon", (LionMuonMixin, SpectralHost), {})
    dense, expert = torch.nn.Parameter(torch.zeros(2, 2)), torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = cls([dense])
    optimizer.param_groups.append({"params": [expert], "lr": 1.0, "momentum": 0.9, "weight_decay": 0.0,
                                   "stage3_split_swiglu_fc1": False, "stage3_routed_expert": True})
    optimizer.state[expert] = {}
    optimizer.lion_period, optimizer.lion_prob, optimizer.lion_power = [(0.0, 2)], None, None
    optimizer.lion_period_experts = [(0.0, 3)]
    optimizer.lion_base_iter, optimizer.lion_total, optimizer.lion_beta1, optimizer.lion_lr_ratio = 0, 100, 0.0, 1.0
    spectral = {"dense": [], "expert": []}
    for step in range(6):
        dense.grad, expert.grad = torch.full((2, 2), -1.0), torch.full((2, 2), -1.0)
        before = (dense.detach().clone(), expert.detach().clone())
        optimizer.step()
        # the spectral step of SpectralHost writes 7.0; a sign step moves by exactly 1
        spectral["dense"].append(bool(torch.allclose(dense.detach() - before[0], torch.full((2, 2), -7.0))))
        spectral["expert"].append(bool(torch.allclose(expert.detach() - before[1], torch.full((2, 2), -7.0))))
    assert spectral["dense"] == [True, False, True, False, True, False]
    assert spectral["expert"] == [True, False, False, True, False, False]


def test_lionmuon2f_preconditions_only_spectral_steps(monkeypatch):
    monkeypatch.setattr(lionmuon, "megatron_iteration", lambda: None)
    cls = type("TestLionMuon2F", (LionMuon2FMixin, Host), {})
    p = torch.nn.Parameter(torch.zeros(3, 4))
    optimizer = cls([p])
    optimizer.lion_period, optimizer.lion_prob, optimizer.lion_power = [(0.0, 2)], None, None
    optimizer.lion_period_experts = None
    optimizer.lion_base_iter, optimizer.lion_total, optimizer.lion_beta1, optimizer.lion_beta2 = 0, 100, 0.0, 0.0
    optimizer.lion_lr_ratio, optimizer.muon2_beta2, optimizer.muon2_eps = 1.0, 0.0, 0.0
    grad = _rank_one_abs_gradient(3, 4, 5)
    p.grad = grad.clone()
    optimizer.step()  # spectral: identity orthogonalize of G / sqrt(V_hat), V_hat = G*G exactly for a rank-one |G|
    torch.testing.assert_close(-p.detach(), grad.sign())
    assert optimizer.state[p]["muon2_row"].shape == (3,)
    before = p.detach().clone()
    p.grad = 2 * grad
    optimizer.step()  # sign step, factors still updated
    torch.testing.assert_close(before - p.detach(), (2 * grad).sign())
    torch.testing.assert_close(optimizer.state[p]["muon2_row"], (2 * grad).square().sum(dim=1))
