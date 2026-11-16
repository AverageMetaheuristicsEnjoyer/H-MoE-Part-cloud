import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "Megatron-LM"))
sys.path.insert(0, str(ROOT / "third_party" / "emerging-optimizers"))

from emerging_optimizers.orthogonalized_optimizers.muon import get_muon_scale_factor
from stage3_moe import lionmuon
from stage3_moe.lionmuon import LionMuonMixin

DEVICE = "cuda"
CLOCK = {"iteration": None}
lionmuon.megatron_iteration = lambda: CLOCK["iteration"]


class ToyHost(torch.optim.Optimizer):
    def __init__(self, specs, lr=0.01, weight_decay=0.1):
        if not specs:
            raise ValueError("empty specs")
        super().__init__([specs[0][1]], {"lr": lr, "weight_decay": weight_decay})
        self.fp32_matmul_prec = "highest"
        self._soft_scale_mode = "spectral"
        self._soft_extra_scale = 0.2
        self.spectral_calls = 0
        self.last_updates = {}
        self.decay_calls = []
        for index, (label, parameter, split, routed) in enumerate(specs):
            if index:
                self.add_param_group({"params": [parameter]})
            group = self.param_groups[index]
            group.update({"label": label, "stage3_split_swiglu_fc1": split, "stage3_routed_expert": routed})
            self._init_group(group)

    def _init_group(self, group):
        for parameter in group["params"]:
            if "momentum_buffer" not in self.state[parameter]:
                self.state[parameter]["momentum_buffer"] = torch.zeros_like(parameter)

    def orthogonalize(self, parameter, direction, **kwargs):
        self.spectral_calls += 1
        blocks = direction.chunk(2, dim=0) if kwargs.get("stage3_split_swiglu_fc1") else (direction,)
        scaled = []
        for block in blocks:
            scale = get_muon_scale_factor(block.size(-2), block.size(-1), mode="spectral") * self._soft_extra_scale
            scaled.append(block * scale)
        return torch.cat(scaled) if len(scaled) > 1 else scaled[0]

    def _apply_weight_decay_inplace(self, parameter, grad, lr, weight_decay):
        self.decay_calls.append((lr, weight_decay))
        parameter.mul_(1 - lr * weight_decay)

    def pre_weight_update_fn_inplace(self, parameter, update):
        self.last_updates[parameter] = update.detach().clone()

    def post_weight_update_fn_inplace(self, parameter):
        pass


def make_optimizer(specs, enabled=True, lr=0.01, weight_decay=0.1, beta1=0.9, beta2=0.99,
                   rho_ref=0.8, diagnostics=True):
    cls = type("ToyLionMuon", (LionMuonMixin, ToyHost), {})
    optimizer = cls(specs, lr=lr, weight_decay=weight_decay)
    optimizer.lion_period = [(0.0, 5)]
    optimizer.lion_prob = None
    optimizer.lion_power = None
    optimizer.lion_period_experts = None
    optimizer.lion_base_iter = 0
    optimizer.lion_total = 100
    optimizer.lion_beta1 = beta1
    optimizer.lion_beta2 = beta2
    optimizer.lion_lr_ratio = 50
    optimizer.lion_energycal = enabled
    optimizer.lion_energycal_rho_ref = rho_ref
    optimizer.lion_energycal_exponent = 1.0
    optimizer.lion_energycal_qmin = 0.0
    optimizer.lion_energycal_diagnostics = diagnostics
    return optimizer


def parameter(values, shape):
    return torch.nn.Parameter(torch.tensor(values, dtype=torch.float32, device=DEVICE).reshape(shape))


def single(value=1.0, shape=(2, 4), routed=True, split=False, label="expert", enabled=True, rho_ref=0.8):
    p = parameter([value] * (shape[0] * shape[1]), shape)
    return p, make_optimizer([(label, p, split, routed)], enabled=enabled, rho_ref=rho_ref)


def set_gradient(optimizer, value):
    for group in optimizer.param_groups:
        for p in group["params"]:
            p.grad = torch.full_like(p, value) if value is not None else None


def state_momentum(optimizer, parameter):
    return optimizer.state[parameter]["momentum_buffer"]


def assert_equal(left, right):
    assert torch.equal(left, right), (left, right)


def test_disabled_and_q1_identity():
    p0 = parameter([1.0] * 8, (2, 4))
    p1 = parameter([1.0] * 8, (2, 4))
    plain = make_optimizer([("expert", p0, False, True)], enabled=False)
    candidate = make_optimizer([("expert", p1, False, True)], enabled=True)
    p0.grad = torch.full_like(p0, 2.0)
    p1.grad = torch.full_like(p1, 2.0)
    plain.step()
    candidate.step()
    assert candidate._lion_energycal_last[0]["q"] == 1.0
    assert_equal(p0, p1)
    assert_equal(state_momentum(plain, p0), state_momentum(candidate, p1))
    direction = torch.full((2, 4), 0.2, device=DEVICE)
    scale = 0.2 * 2.0
    expected = torch.full((2, 4), 0.999, device=DEVICE) - 0.01 * direction * scale
    torch.testing.assert_close(p0, expected)
    torch.testing.assert_close(state_momentum(plain, p0), torch.full((2, 4), 0.02, device=DEVICE))


def test_low_zero_scope_and_split():
    specs = []
    params = {}
    for label, split, routed in (("gate_up", True, True), ("fc2", False, True),
                                 ("dense", False, False), ("shared", False, False),
                                 ("router", False, False), ("zero", False, True)):
        shape = (4, 4) if split else (2, 4)
        p = parameter([0.0] * (shape[0] * shape[1]), shape)
        params[label] = p
        specs.append((label, p, split, routed))
    optimizer = make_optimizer(specs, lr=1.0, weight_decay=0.0, beta1=0.0, beta2=0.0)
    for label, p in params.items():
        p.grad = torch.zeros_like(p)
    params["gate_up"].grad = torch.cat([
        torch.full((2, 4), 0.1, device=DEVICE), torch.full((2, 4), 0.2, device=DEVICE)
    ])
    for label in ("fc2", "dense", "shared", "router"):
        params[label].grad = torch.full_like(params[label], 0.1)
    optimizer.step()
    torch.testing.assert_close(params["gate_up"][:2], torch.full((2, 4), -0.002, device=DEVICE))
    torch.testing.assert_close(params["gate_up"][2:], torch.full((2, 4), -0.008, device=DEVICE))
    torch.testing.assert_close(params["fc2"], torch.full((2, 4), -0.002, device=DEVICE))
    for label in ("dense", "shared", "router"):
        torch.testing.assert_close(params[label], torch.full((2, 4), -0.04, device=DEVICE))
    assert_equal(params["zero"], torch.zeros(2, 4, device=DEVICE))
    diagnostics = optimizer._lion_energycal_last
    torch.testing.assert_close(torch.tensor([item["q"] for item in diagnostics]),
                               torch.tensor([0.05, 0.2, 0.05, 0.0]))
    assert [item["shape"] for item in diagnostics[:2]] == [[2, 4], [2, 4]]
    assert all(item["shape"] == [2, 4] for item in diagnostics)


def test_lr_weight_decay_and_ns_sign_counts():
    p0 = parameter([1.0] * 8, (2, 4))
    p1 = parameter([1.0] * 8, (2, 4))
    plain = make_optimizer([("expert", p0, False, True)], enabled=False)
    candidate = make_optimizer([("expert", p1, False, True)], enabled=True)
    for iteration in range(10):
        CLOCK["iteration"] = iteration
        for p, optimizer in ((p0, plain), (p1, candidate)):
            p.grad = torch.full_like(p, 0.1 + iteration * 0.01)
        plain.step()
        before = p1.detach().clone()
        candidate.step()
        lr, wd = candidate.decay_calls[-1]
        update = candidate.last_updates[p1]
        torch.testing.assert_close(p1, before * (1 - lr * wd) - lr * update)
        assert torch.count_nonzero(update)
        assert len(candidate._lion_energycal_last) == (1 if iteration % 5 == 0 else 0)
    assert plain.decay_calls == candidate.decay_calls
    assert plain.spectral_calls == candidate.spectral_calls == 2
    assert_equal(state_momentum(plain, p0), state_momentum(candidate, p1))
    expected_lr = [0.01 if i % 5 == 0 else 0.01 / 50 for i in range(10)]
    assert [item[0] for item in candidate.decay_calls] == expected_lr
    plain_wd = 1.0
    candidate_wd = 1.0
    for (plain_lr, _), (candidate_lr, _) in zip(plain.decay_calls, candidate.decay_calls):
        plain_wd *= 1 - plain_lr * 0.1
        candidate_wd *= 1 - candidate_lr * 0.1
    assert plain_wd == candidate_wd
    CLOCK["iteration"] = None


def test_grad_none_and_materialized_zero():
    none = parameter([1.0] * 8, (2, 4))
    zero = parameter([1.0] * 8, (2, 4))
    optimizer = make_optimizer([("none", none, False, True), ("zero", zero, False, True)])
    none.grad = None
    zero.grad = torch.zeros_like(zero)
    state_momentum(optimizer, none).fill_(0.2)
    state_momentum(optimizer, zero).fill_(0.2)
    optimizer.step()
    assert_equal(none, torch.ones(2, 4, device=DEVICE))
    assert_equal(state_momentum(optimizer, none), torch.full((2, 4), 0.2, device=DEVICE))
    torch.testing.assert_close(state_momentum(optimizer, zero), torch.full((2, 4), 0.198, device=DEVICE))
    scale = 0.2 * 2.0
    expected = torch.full((2, 4), 0.999, device=DEVICE) - 0.01 * 0.18 * scale
    torch.testing.assert_close(zero, expected)


def test_nan_inf_are_not_sanitized():
    nan = parameter([1.0] * 8, (2, 4))
    inf = parameter([1.0] * 8, (2, 4))
    optimizer = make_optimizer([("nan", nan, False, True), ("inf", inf, False, True)])
    nan.grad = torch.full_like(nan, float("nan"))
    inf.grad = torch.full_like(inf, float("inf"))
    optimizer.step()
    assert not torch.isfinite(nan).all()
    assert not torch.isfinite(inf).all()
    assert not torch.isfinite(state_momentum(optimizer, nan)).all()
    assert not torch.isfinite(state_momentum(optimizer, inf)).all()
    assert not all(torch.isfinite(torch.tensor(item["q"])) for item in optimizer._lion_energycal_last)


def test_resume_all_five_offsets():
    output = Path(os.environ["HMOE_CONTRACT_ROOT"]) / "toy-checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for offset in range(5):
        reference_parameter = parameter([0.5] * 8, (2, 4))
        reference = make_optimizer([("expert", reference_parameter, False, True)])
        for iteration in range(offset):
            CLOCK["iteration"] = iteration
            reference_parameter.grad = torch.full_like(reference_parameter, 0.03 * (iteration + 1))
            reference.step()
        for group in reference.param_groups:
            reference._lion_energycal_group(group)
        checkpoint = {"model": reference_parameter.detach().cpu(), "optimizer": reference.state_dict(), "iteration": offset}
        path = output / f"offset-{offset}.pt"
        torch.save(checkpoint, path)
        CLOCK["iteration"] = offset
        reference_parameter.grad = torch.full_like(reference_parameter, 0.03 * (offset + 1))
        reference.step()
        restored_parameter = parameter([0.0] * 8, (2, 4))
        restored = make_optimizer([("expert", restored_parameter, False, True)])
        loaded = torch.load(path, weights_only=True)
        restored_parameter.data.copy_(loaded["model"].to(DEVICE))
        restored.load_state_dict(loaded["optimizer"])
        restored_parameter.grad = torch.full_like(restored_parameter, 0.03 * (offset + 1))
        restored.step()
        assert_equal(restored_parameter, reference_parameter)
        assert_equal(state_momentum(restored, restored_parameter), state_momentum(reference, reference_parameter))
        assert restored._lion_count == offset
        assert restored.spectral_calls == int(offset % 5 == 0)
        receipts.append({"offset": offset, "path": str(path), "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "next_spectral": offset % 5 == 0})
    CLOCK["iteration"] = None
    return receipts


def test_resume_config_mismatch_and_disabled_compatibility():
    old_parameter, old = single(enabled=False)
    old_parameter.grad = torch.ones_like(old_parameter)
    old.step()
    old_state = old.state_dict()
    for group in old_state["param_groups"]:
        group.pop("lionmuon_energycal")
    disabled_parameter, disabled = single(enabled=False)
    disabled.load_state_dict(old_state)
    CLOCK["iteration"] = 1
    old_parameter.grad = torch.full_like(old_parameter, 2.0)
    disabled_parameter.grad = torch.full_like(disabled_parameter, 2.0)
    old.step()
    disabled.step()
    assert_equal(disabled_parameter, old_parameter)
    assert_equal(state_momentum(disabled, disabled_parameter), state_momentum(old, old_parameter))
    CLOCK["iteration"] = None
    candidate_parameter, candidate = single(enabled=True)
    try:
        candidate.load_state_dict(old_state)
        raise AssertionError("EnergyCal accepted a baseline checkpoint without metadata")
    except ValueError as error:
        assert "explicit switch" in str(error)
    candidate_parameter.grad = torch.ones_like(candidate_parameter)
    candidate.step()
    saved = candidate.state_dict()
    mismatch_parameter, mismatch = single(enabled=True, rho_ref=0.7)
    try:
        mismatch.load_state_dict(saved)
        raise AssertionError("EnergyCal config mismatch was accepted")
    except ValueError as error:
        assert "mismatch" in str(error)
    try:
        disabled.load_state_dict(saved)
        raise AssertionError("disabled baseline accepted an enabled checkpoint")
    except ValueError as error:
        assert "mismatch" in str(error)


def test_preset_and_scope_predicates():
    script = ROOT / "scripts" / "muon_presets.sh"
    words = subprocess.run(["bash", "-c", f"source {script}; muon_preset lm5r50ec"],
                           capture_output=True, text=True, check=True).stdout.split()
    values = dict(word.split("=", 1) for word in words)
    assert values == {"STAGE3_MOE_MUON_VARIANT": "lionmuon", "STAGE3_MOE_LIONMUON_PERIOD": "5",
                      "STAGE3_MOE_LIONMUON_LR_RATIO": "50", "STAGE3_MOE_LIONMUON_ENERGYCAL": "1"}
    from stage3_moe.muon import is_routed_expert_weight
    matrix = torch.zeros(2, 4, device=DEVICE)
    assert is_routed_expert_weight(matrix, "decoder.layers.1.mlp.experts.linear_fc2.weight0")
    assert not is_routed_expert_weight(matrix, "decoder.layers.1.mlp.shared_experts.linear_fc2.weight")
    assert not is_routed_expert_weight(matrix, "decoder.layers.1.mlp.router.weight")


TESTS = [
    test_disabled_and_q1_identity,
    test_low_zero_scope_and_split,
    test_lr_weight_decay_and_ns_sign_counts,
    test_grad_none_and_materialized_zero,
    test_nan_inf_are_not_sanitized,
    test_resume_all_five_offsets,
    test_resume_config_mismatch_and_disabled_compatibility,
    test_preset_and_scope_predicates,
]


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Contract requires exactly one visible CUDA GPU")
    output = Path(os.environ["HMOE_CONTRACT_ROOT"])
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for test in TESTS:
        try:
            value = test()
            torch.cuda.synchronize()
            result = {"name": test.__name__, "status": "passed"}
            if value is not None:
                result["receipts"] = value
        except Exception as error:
            result = {"name": test.__name__, "status": "failed", "error": repr(error), "traceback": traceback.format_exc()}
        results.append(result)
        print("CONTRACT_TEST " + json.dumps(result, sort_keys=True), flush=True)
    passed = all(result["status"] == "passed" for result in results)
    report = {"schema": "hmoe-lionmoe-energycal-contract-v1", "training_invoked": False,
              "full_model_constructed": False, "real_checkpoint_downloaded": False,
              "test_count": len(results), "passed": sum(result["status"] == "passed" for result in results),
              "results": results}
    with (output / "contract-results.json").open("w") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("CONTRACT_RESULT " + json.dumps({"passed": passed, "test_count": len(results),
                                           "report": str(output / "contract-results.json")}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
