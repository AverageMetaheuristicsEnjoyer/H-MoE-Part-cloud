import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "stage3_moe"))

from test_lionmuon_energycal import CLOCK, make_optimizer, parameter, state_momentum
from stage3_moe.energycal_diagnostics import EnergyCalObserver

NAME = "model_chunk0.decoder.layers.1.mlp.experts.linear_fc2.weight0"


def observed_optimizer(enabled, output):
    p = parameter([1.0] * 8, (2, 4))
    optimizer = make_optimizer([("expert", p, False, True)], enabled=enabled)
    optimizer.lion_energycal_diagnostics = False
    observer = EnergyCalObserver({id(p): NAME}, output)
    optimizer._lionmuon_measurement_observer = observer
    return p, optimizer, observer


def check_variant(enabled, output):
    observed_parameter, observed, observer = observed_optimizer(enabled, output / "observed")
    plain_parameter = parameter([1.0] * 8, (2, 4))
    plain = make_optimizer([("expert", plain_parameter, False, True)], enabled=enabled)
    CLOCK["iteration"] = 0
    observed_parameter.grad = torch.full_like(observed_parameter, 0.1)
    plain_parameter.grad = torch.full_like(plain_parameter, 0.1)
    observed.step()
    plain.step()
    assert torch.equal(observed_parameter, plain_parameter)
    assert torch.equal(state_momentum(observed, observed_parameter), state_momentum(plain, plain_parameter))
    assert len(observer.records) == 1
    record = observer.records[0]
    assert record["hypothetical_q"] < 1.0
    assert record["applied"] is enabled
    assert (record["actual_q"] is not None) is enabled
    if not enabled:
        assert record["post_update_norm"] == record["pre_update_norm"]
    before_records = len(observer.records)
    before_callbacks = observer.callback_seconds
    CLOCK["iteration"] = 20
    observed_parameter.grad = torch.full_like(observed_parameter, 0.2)
    assert not observer.enabled(21)
    observed.step()
    assert len(observer.records) == before_records
    assert observer.callback_seconds == before_callbacks
    assert not observed.lion_energycal_diagnostics
    return {"enabled": enabled, "selected_records": len(observer.records),
            "hypothetical_q": record["hypothetical_q"], "actual_q": record["actual_q"],
            "off_at_21": True, "params_and_momentum_equal": True}


def main():
    if os.environ.get("STAGE3_MOE_LIONMUON_ENERGYCAL_DIAGNOSTICS"):
        raise RuntimeError("Legacy optional per-expert diagnostics must be unset for the clean pilot")
    output = Path(os.environ["HMOE_CONTRACT_ROOT"]) / "transparency"
    output.mkdir(parents=True, exist_ok=True)
    results = [check_variant(False, output / "plain"), check_variant(True, output / "candidate")]
    report = {"schema": "hmoe-lionmoe-measurement-transparency-v1", "passed": True,
              "results": results, "full_model_constructed": False, "training_invoked": False}
    with (output / "transparency-results.json").open("w") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("MEASUREMENT_TRANSPARENCY " + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
