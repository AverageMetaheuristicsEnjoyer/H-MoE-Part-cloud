import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "stage3_moe"))
sys.path.insert(0, str(ROOT))

from test_lionmuon_energycal import CLOCK, make_optimizer, parameter, state_momentum
from stage3_moe.result_writer import skip_test_evaluation


def alpha_identity():
    p0, p1 = parameter([1.0] * 8, (2, 4)), parameter([1.0] * 8, (2, 4))
    baseline = make_optimizer([("expert", p0, False, True)], enabled=False)
    control = make_optimizer([("expert", p1, False, True)], enabled=False)
    baseline.lion_fixed_routed_alpha = control.lion_fixed_routed_alpha = 1.0
    for completed in (0, 4, 5, 9, 10):
        CLOCK["iteration"] = completed
        p0.grad = p1.grad = torch.full_like(p0, 0.1)
        baseline.step()
        control.step()
        assert torch.equal(p0, p1)
        assert torch.equal(state_momentum(baseline, p0), state_momentum(control, p1))
    return {"multiple_p5_cycles": True, "nonzero_params_and_momentum_equal": True}


def routed_scope():
    p0, p1 = parameter([1.0] * 8, (2, 4)), parameter([1.0] * 8, (2, 4))
    plain = make_optimizer([("expert", p0, False, True)], enabled=False)
    fixed = make_optimizer([("expert", p1, False, True)], enabled=False)
    fixed.lion_fixed_routed_alpha = 0.5
    CLOCK["iteration"] = 0
    p0.grad = p1.grad = torch.full_like(p0, 0.1)
    plain.step()
    fixed.step()
    assert torch.equal(fixed.last_updates[p1], plain.last_updates[p0] * 0.5)
    assert fixed.decay_calls == plain.decay_calls
    sign_plain = make_optimizer([("expert", p0, False, True)], enabled=False)
    sign_fixed = make_optimizer([("expert", p1, False, True)], enabled=False)
    sign_fixed.lion_fixed_routed_alpha = 0.5
    CLOCK["iteration"] = 4
    p0.grad = p1.grad = torch.full_like(p0, 0.1)
    sign_plain.step()
    sign_fixed.step()
    assert torch.equal(sign_fixed.last_updates[p1], sign_plain.last_updates[p0])
    shared_plain = make_optimizer([("shared", p0, False, False)], enabled=False)
    shared_fixed = make_optimizer([("shared", p1, False, False)], enabled=False)
    shared_fixed.lion_fixed_routed_alpha = 0.5
    CLOCK["iteration"] = 0
    p0.grad = p1.grad = torch.full_like(p0, 0.1)
    shared_plain.step()
    shared_fixed.step()
    assert torch.equal(shared_fixed.last_updates[p1], shared_plain.last_updates[p0])
    return {"routed_spectral_scaled_by_0_5": True, "sign_unchanged": True,
            "nonrouted_unchanged": True, "weight_decay_calls_unchanged": True}


def simultaneous_rejected():
    p = parameter([1.0] * 8, (2, 4))
    optimizer = make_optimizer([("expert", p, False, True)], enabled=True)
    optimizer.lion_fixed_routed_alpha = 0.5
    p.grad = torch.full_like(p, 0.1)
    try:
        optimizer.step()
    except ValueError as error:
        assert "mutually exclusive" in str(error)
        return True
    raise AssertionError("Adaptive plus fixed alpha was not rejected")


def no_test_guard():
    os.environ["STAGE3_MOE_DISABLE_TEST_EVAL"] = "1"
    assert skip_test_evaluation("iteration 600 on test set")
    assert not skip_test_evaluation("iteration 600 on validation set")
    os.environ["STAGE3_MOE_DISABLE_TEST_EVAL"] = "0"
    assert not skip_test_evaluation("iteration 600 on test set")
    os.environ["STAGE3_MOE_DISABLE_TEST_EVAL"] = "1"
    source = (ROOT / "third_party/Megatron-LM/megatron/training/training.py").read_text()
    assert "prefix = f'iteration {iteration} on test set'" in source
    return {"test_prefix_blocked": True, "validation_prefix_allowed": True,
            "default_off": True, "actual_mcore_prefix_match": True}


def main():
    output = Path(os.environ["HMOE_CONTRACT_ROOT"]) / "overnight-contracts"
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema": "hmoe-lionmoe-overnight013-contracts-v1", "passed": True,
              "alpha_identity": alpha_identity(), "fixed_scope": routed_scope(),
              "simultaneous_rejected": simultaneous_rejected(), "no_test_guard": no_test_guard(),
              "full_model_constructed": False}
    with (output / "contract-results.json").open("w") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("OVERNIGHT013_CONTRACTS " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
