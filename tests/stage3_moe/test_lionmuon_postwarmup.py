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
LATE_STEPS = (181, 186, 191, 196)


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
    callback_by_step = {}
    for iteration in (180, 181, 186, 191, 196, 200, 201, 221):
        CLOCK["iteration"] = iteration
        observed_parameter.grad = torch.full_like(observed_parameter, 0.1)
        plain_parameter.grad = torch.full_like(plain_parameter, 0.1)
        observed.step()
        plain.step()
        assert torch.equal(observed_parameter, plain_parameter)
        assert torch.equal(state_momentum(observed, observed_parameter), state_momentum(plain, plain_parameter))
        callback_by_step[str(iteration)] = observer.callback_seconds
    assert not observer.enabled(180) and observer.enabled(181) and observer.enabled(200)
    assert not observer.enabled(201) and not observer.enabled(221)
    assert callback_by_step["180"] == 0
    assert callback_by_step["201"] == callback_by_step["200"] == callback_by_step["221"]
    assert observer.records == [] and observer.current == {} and observer.started is None
    block_path = observer.blocks_path
    records = [json.loads(line) for line in block_path.read_text().splitlines()]
    assert [record["iteration"] for record in records] == list(LATE_STEPS)
    assert all(record["applied"] is enabled for record in records)
    assert all((record["actual_q"] is not None) is enabled for record in records)
    if not enabled:
        assert all(record["post_update_norm"] == record["pre_update_norm"] for record in records)
    summary = json.loads(observer.summary_path.read_text())
    assert summary["window"] == [181, 200] and summary["spectral_steps"] == list(LATE_STEPS)
    return {"enabled": enabled, "selected_records": len(records), "spectral_steps": summary["spectral_steps"],
            "hypothetical_q": [record["hypothetical_q"] for record in records],
            "actual_q": [record["actual_q"] for record in records],
            "boundaries": {"180": False, "181": True, "200": True, "201": False, "221": False},
            "buffers_released": True, "params_and_momentum_equal": True}


def check_dense_control_identity(output):
    name = "model_chunk0.decoder.layers.0.mlp.linear_fc1.weight"
    p = parameter([1.0] * 16, (4, 4))
    optimizer = make_optimizer([("dense", p, True, False)], enabled=False)
    optimizer.lion_energycal_diagnostics = False
    observer = EnergyCalObserver({id(p): name}, output)
    optimizer._lionmuon_measurement_observer = observer
    CLOCK["iteration"] = 181
    p.grad = torch.full_like(p, 0.1)
    optimizer.step()
    labels = [record["fc_type"] for record in observer.records]
    assert labels == ["dense_fc1_gate", "dense_fc1_up"] and len(set(labels)) == 2
    CLOCK["iteration"] = 200
    p.grad = torch.full_like(p, 0.1)
    optimizer.step()
    summary = json.loads(observer.summary_path.read_text())
    assert summary["groups"]["layer_00/dense_fc1_gate"]["samples"] == 1
    assert summary["groups"]["layer_00/dense_fc1_up"]["samples"] == 1
    return {"unique_dense_fc1_halves": labels, "summary_groups_verified": True}


def main():
    if os.environ.get("STAGE3_MOE_LIONMUON_ENERGYCAL_DIAGNOSTICS"):
        raise RuntimeError("Legacy optional per-expert diagnostics must be unset")
    output = Path(os.environ["HMOE_CONTRACT_ROOT"]) / "transparency"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.pop("STAGE3_MOE_ENERGYCAL_DIAGNOSTICS_START", None)
    os.environ.pop("STAGE3_MOE_ENERGYCAL_DIAGNOSTICS_END", None)
    legacy = EnergyCalObserver({}, output / "legacy-defaults")
    assert legacy.window_start == 1 and legacy.window_end == 20 and legacy.enabled(1) and not legacy.enabled(21)
    os.environ["STAGE3_MOE_ENERGYCAL_DIAGNOSTICS_START"] = "181"
    os.environ["STAGE3_MOE_ENERGYCAL_DIAGNOSTICS_END"] = "200"
    results = [check_variant(False, output / "plain"), check_variant(True, output / "candidate")]
    dense = check_dense_control_identity(output / "dense")
    sys.path.insert(0, str(ROOT / "scripts"))
    from lionmoe_postwarmup_012 import phase_summary
    values = list(range(1, 11))
    phases = ["spectral" if index % 5 == 0 else "sign" for index in range(10)]
    spectral = phase_summary(values, phases, "spectral")
    sign = phase_summary(values, phases, "sign")
    assert spectral["count"] == 2 and spectral["sum_seconds"] == 7
    assert sign["count"] == 8 and sign["sum_seconds"] == 48
    report = {"schema": "hmoe-lionmoe-postwarmup-transparency-v1", "passed": True,
              "results": results, "legacy_default_window": [1, 20], "dense_controls": dense,
              "phase_contents": {"spectral_count": 2, "spectral_sum": 7, "sign_count": 8, "sign_sum": 48},
              "full_model_constructed": False, "training_invoked": False}
    with (output / "transparency-results.json").open("w") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("POSTWARMUP_TRANSPARENCY " + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
