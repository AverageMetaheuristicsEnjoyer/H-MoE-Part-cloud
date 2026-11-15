import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage3_moe.optimizer_states import (
    ADAM_STATE_SPECS,
    MUON_STATE_SPECS,
    FP8StateOptimizerMixin,
    adam_state_specs,
    make_fp8_adamw,
    muon_state_specs,
)


@pytest.fixture(autouse=True)
def clean_codec_environment(monkeypatch):
    for name in (
        "STAGE3_MOE_MUON_STATE_CODEC",
        "STAGE3_MOE_ADAM_STATE_CODEC",
        "STAGE3_MOE_FP8_STATE_DTYPES",
    ):
        monkeypatch.delenv(name, raising=False)


def test_historical_defaults_are_unchanged():
    assert adam_state_specs() == ADAM_STATE_SPECS
    assert muon_state_specs() == MUON_STATE_SPECS


@pytest.mark.parametrize("momentum", ["fp32", "maxabs", "dre"])
@pytest.mark.parametrize("fallback", ["fp32", "maxabs", "dre"])
def test_component_codecs_are_independent(monkeypatch, momentum, fallback):
    monkeypatch.setenv("STAGE3_MOE_MUON_STATE_CODEC", momentum)
    monkeypatch.setenv("STAGE3_MOE_ADAM_STATE_CODEC", fallback)
    monkeypatch.setenv("STAGE3_MOE_FP8_STATE_DTYPES", "e4m3:e4m3")
    m_spec, = muon_state_specs()
    a_specs = adam_state_specs()
    assert m_spec.recipe == momentum
    assert m_spec.dtype == (
        torch.float32 if momentum == "fp32" else torch.float8_e4m3fn
    )
    assert all(spec.recipe == fallback for spec in a_specs)
    assert all(spec.dtype == (
        torch.float32 if fallback == "fp32" else torch.float8_e4m3fn
    ) for spec in a_specs)
    monkeypatch.setenv("STAGE3_MOE_FP8_STATE_DTYPES", "e4m3:e5m2")
    assert muon_state_specs() == (m_spec,)
    assert adam_state_specs()[1].dtype == (
        torch.float32 if fallback == "fp32" else torch.float8_e5m2
    )


@pytest.mark.parametrize("name,resolver", [
    ("STAGE3_MOE_MUON_STATE_CODEC", muon_state_specs),
    ("STAGE3_MOE_ADAM_STATE_CODEC", adam_state_specs),
])
def test_unknown_codec_is_rejected(monkeypatch, name, resolver):
    monkeypatch.setenv(name, "dre2")
    with pytest.raises(ValueError, match=name):
        resolver()


@pytest.mark.parametrize("momentum", ["fp32", "maxabs", "dre"])
@pytest.mark.parametrize("fallback", ["fp32", "maxabs", "dre"])
def test_mixed_codec_ledger_and_bootstrap(monkeypatch, momentum, fallback):
    from stage3_moe.result_writer import assert_fp8_adam_bootstrap, optimizer_state_ledger

    monkeypatch.setenv("STAGE3_MOE_MUON_STATE_CODEC", momentum)
    monkeypatch.setenv("STAGE3_MOE_ADAM_STATE_CODEC", fallback)
    monkeypatch.setenv("STAGE3_MOE_FP8_STATE_DTYPES", "e4m3:e4m3")
    matrix = torch.nn.Parameter(torch.zeros(16, 16))
    vector = torch.nn.Parameter(torch.zeros(16))
    m_spec, = muon_state_specs()
    matrix_state = {m_spec.name: torch.zeros_like(matrix, dtype=m_spec.dtype)}
    if momentum == "dre":
        matrix_state["expand_momentum_buffer"] = torch.ones(2)
        matrix_state["sqrt_minmax_momentum_buffer"] = torch.ones(2)
    matrix_optimizer = SimpleNamespace(
        state_specs=(m_spec,), state={matrix: matrix_state}, param_groups=[{"params": [matrix]}]
    )
    fallback_optimizer = make_fp8_adamw(torch.optim.AdamW)([vector])
    fallback_optimizer.state[vector] = {
        spec.name: torch.zeros_like(vector, dtype=spec.dtype) for spec in adam_state_specs()
    }
    chained = SimpleNamespace(chained_optimizers=[
        SimpleNamespace(optimizer=matrix_optimizer),
        SimpleNamespace(optimizer=fallback_optimizer),
    ])
    assert_fp8_adam_bootstrap(chained)
    ledger = optimizer_state_ledger(chained, "muon_bf16_state_fp8")
    momentum_row, = [row for row in ledger["tensors"] if row["state_key"] == "momentum_buffer"]
    assert momentum_row["quantized"] == (momentum != "fp32")
    adam_rows = [row for row in ledger["tensors"] if row["state_key"] in {"exp_avg", "exp_avg_sq"}]
    assert all(row["quantized"] == (fallback != "fp32") for row in adam_rows)
    matrix_state[m_spec.name] = torch.zeros_like(matrix, dtype=(
        torch.float8_e4m3fn if momentum == "fp32" else torch.float32
    ))
    with pytest.raises(AssertionError, match="Muon momentum precision"):
        optimizer_state_ledger(chained, "muon_bf16_state_fp8")
    matrix_state[m_spec.name] = torch.zeros_like(matrix, dtype=m_spec.dtype)
    fallback_optimizer.state[vector]["exp_avg"] = torch.zeros_like(vector, dtype=(
        torch.float8_e4m3fn if fallback == "fp32" else torch.float32
    ))
    with pytest.raises(AssertionError, match="Adam state precision"):
        optimizer_state_ledger(chained, "muon_bf16_state_fp8")


def test_fp32_adam_wrapper_resumes_without_changing_math(monkeypatch):
    monkeypatch.setenv("STAGE3_MOE_ADAM_STATE_CODEC", "fp32")
    parameter = torch.nn.Parameter(torch.linspace(-1, 1, 257))
    reference = torch.optim.AdamW([parameter], lr=1e-3, betas=(0.9, 0.95))
    parameter.grad = torch.linspace(0.1, 0.3, 257)
    reference.step()
    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = make_fp8_adamw(torch.optim.AdamW)([resumed_parameter])
    resumed.load_state_dict(copy.deepcopy(reference.state_dict()))
    parameter.grad = torch.linspace(-0.3, 0.2, 257)
    resumed_parameter.grad = parameter.grad.clone()
    reference.step()
    resumed.step()
    assert torch.equal(parameter, resumed_parameter)
    assert reference.state[parameter].keys() == resumed.state[resumed_parameter].keys()
    for name, value in reference.state[parameter].items():
        assert torch.equal(value, resumed.state[resumed_parameter][name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("codec", ["maxabs", "dre"])
@pytest.mark.parametrize("component", ["adam", "momentum"])
def test_fp32_checkpoint_conversion_and_fp8_resume(monkeypatch, codec, component):
    monkeypatch.setenv("STAGE3_MOE_MUON_STATE_CODEC", codec)
    monkeypatch.setenv("STAGE3_MOE_ADAM_STATE_CODEC", codec)
    monkeypatch.setenv("STAGE3_MOE_FP8_STATE_DTYPES", "e4m3:e4m3")
    if component == "adam":
        base = torch.optim.AdamW
        wrapped = make_fp8_adamw(base)
        kwargs = {"lr": 1e-3, "betas": (0.9, 0.95)}
    else:
        base = torch.optim.SGD

        class WrappedMomentum(FP8StateOptimizerMixin, base):
            state_specs = muon_state_specs()

        wrapped = WrappedMomentum
        kwargs = {"lr": 1e-3, "momentum": 0.95}
    parameter = torch.nn.Parameter(torch.linspace(-1, 1, 257, device="cuda"))
    reference = base([parameter], **kwargs)
    parameter.grad = torch.linspace(0.1, 0.3, 257, device="cuda")
    reference.step()
    quantized_parameter = torch.nn.Parameter(parameter.detach().clone())
    quantized = wrapped([quantized_parameter], **kwargs)
    quantized.load_state_dict(copy.deepcopy(reference.state_dict()))
    parameter.grad = torch.linspace(-0.3, 0.2, 257, device="cuda")
    quantized_parameter.grad = parameter.grad.clone()
    reference.step()
    quantized.step()
    assert torch.equal(parameter, quantized_parameter)
    state = quantized.state[quantized_parameter]
    for spec in quantized.state_specs:
        assert state[spec.name].dtype == torch.float8_e4m3fn
        assert state[f"scale_{spec.name}"].shape == (3,)
        assert (f"expand_{spec.name}" in state) == (codec == "dre")
        assert (f"sqrt_minmax_{spec.name}" in state) == (codec == "dre")
    resumed_parameter = torch.nn.Parameter(quantized_parameter.detach().clone())
    resumed = wrapped([resumed_parameter], **kwargs)
    resumed.load_state_dict(copy.deepcopy(quantized.state_dict()))
    resumed_parameter.grad = quantized_parameter.grad.clone()
    quantized.step()
    resumed.step()
    assert torch.isfinite(resumed_parameter).all()
    assert torch.equal(quantized_parameter, resumed_parameter)
    for name, value in quantized.state[quantized_parameter].items():
        assert torch.equal(value.float(), resumed.state[resumed_parameter][name].float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("codec", ["fp32", "maxabs", "dre"])
def test_real_muon_codec_selection_and_resume(monkeypatch, codec):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "third_party" / "Megatron-LM"))
    sys.path.insert(0, str(root / "third_party" / "emerging-optimizers"))
    from stage3_moe.muon import (
        FP8StateSplitSwiGLUTensorParallelMuon,
        SPLIT_SWIGLU_FC1,
        SplitSwiGLUTensorParallelMuon,
    )

    monkeypatch.setenv("STAGE3_MOE_MUON_STATE_CODEC", codec)
    torch.manual_seed(7)
    parameter = torch.nn.Parameter(torch.randn(32, 16, device="cuda"))
    reference = SplitSwiGLUTensorParallelMuon(
        [{"params": [parameter], SPLIT_SWIGLU_FC1: True}], num_ns_steps=5
    )
    parameter.grad = torch.randn_like(parameter)
    reference.step()
    quantized_parameter = torch.nn.Parameter(parameter.detach().clone())
    wrapped = FP8StateSplitSwiGLUTensorParallelMuon
    quantized = wrapped([quantized_parameter], num_ns_steps=5)
    quantized.load_state_dict(copy.deepcopy(reference.state_dict()))
    assert quantized.state_specs[0].recipe == codec
    parameter.grad = torch.randn_like(parameter)
    quantized_parameter.grad = parameter.grad.clone()
    reference.step()
    quantized.step()
    assert torch.equal(parameter, quantized_parameter)
    resumed_parameter = torch.nn.Parameter(quantized_parameter.detach().clone())
    resumed = wrapped([resumed_parameter], num_ns_steps=5)
    resumed.load_state_dict(copy.deepcopy(quantized.state_dict()))
    resumed_parameter.grad = quantized_parameter.grad.clone()
    quantized.step()
    resumed.step()
    assert torch.isfinite(resumed_parameter).all()
    assert torch.equal(quantized_parameter, resumed_parameter)
