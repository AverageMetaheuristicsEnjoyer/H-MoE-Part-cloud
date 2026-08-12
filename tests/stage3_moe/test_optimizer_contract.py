import copy
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "Megatron-LM"))
sys.path.insert(0, str(ROOT / "third_party" / "emerging-optimizers"))

pytest.importorskip("triton")

from stage3_moe.muon import (
    SPLIT_SWIGLU_FC1,
    SplitSwiGLUTensorParallelMuon,
    install_muon_contract,
    is_router_weight,
    is_swiglu_fc1_weight,
)
from stage3_moe.optimizer_states import (
    ADAM_STATE_SPECS,
    MUON_STATE_SPECS,
    dequantize_fp8_state,
    init_fp8_state,
    make_fp8_adamw,
    num_groups,
    quantize_fp8_state_,
)
from stage3_moe.pretrain_gpt import take_stage3_args, validate_axis
from stage3_moe.result_writer import (
    _environment,
    _normalized_match_argv,
    assert_fp8_adam_bootstrap,
    parameter_group_ledger,
)


def test_state_formats_are_hybrid_and_separate():
    assert [(s.name, s.dtype, s.recipe) for s in ADAM_STATE_SPECS] == [
        ("exp_avg", torch.float8_e4m3fn, "dre"),
        ("exp_avg_sq", torch.float8_e5m2, "dre"),
    ]
    assert [(s.name, s.dtype, s.recipe) for s in MUON_STATE_SPECS] == [
        ("momentum_buffer", torch.float8_e4m3fn, "maxabs")
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("spec", (*ADAM_STATE_SPECS, *MUON_STATE_SPECS))
def test_codec_layout_and_round_trip(spec):
    value = torch.linspace(-4, 4, 383, device="cuda")
    if not spec.signed:
        value = value.square()
    state = {}
    init_fp8_state(state, spec, value)
    quantize_fp8_state_(state, spec, value)
    restored = dequantize_fp8_state(state, spec)

    assert state[spec.name].dtype == spec.dtype
    assert state[f"scale_{spec.name}"].shape == (num_groups(value.numel()),)
    if spec.recipe == "dre":
        assert f"expand_{spec.name}" in state
        assert f"sqrt_minmax_{spec.name}" in state
    else:
        assert f"expand_{spec.name}" not in state
        assert f"sqrt_minmax_{spec.name}" not in state
    assert torch.isfinite(restored).all()
    assert (restored - value).abs().mean() / value.abs().mean() < 0.08


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_adam_hybrid_state_and_checkpoint_resume():
    optimizer_class = make_fp8_adamw(torch.optim.AdamW)
    parameter = torch.nn.Parameter(torch.randn(257, device="cuda"))
    optimizer = optimizer_class([parameter], lr=1e-3, betas=(0.9, 0.95))
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    assert type(optimizer).__name__ == "FP8StateAdamW"
    assert state["exp_avg"].dtype == torch.float8_e4m3fn
    assert state["exp_avg_sq"].dtype == torch.float8_e5m2

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = optimizer_class([resumed_parameter], lr=1e-3, betas=(0.9, 0.95))
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert resumed.state[resumed_parameter]["exp_avg"].dtype == torch.float8_e4m3fn
    assert resumed.state[resumed_parameter]["exp_avg_sq"].dtype == torch.float8_e5m2


def test_router_and_grouped_expert_routing_predicates():
    matrix = torch.nn.Parameter(torch.empty(64, 1024))
    assert is_router_weight(matrix, "decoder.layers.1.mlp.router.weight")
    assert not is_router_weight(matrix, "decoder.layers.1.self_attention.linear_qkv.weight")

    fc1 = torch.nn.Parameter(torch.empty(512, 1024))
    assert is_swiglu_fc1_weight(fc1, "decoder.layers.1.mlp.experts.linear_fc1.weight0")
    assert is_swiglu_fc1_weight(fc1, "decoder.layers.0.mlp.linear_fc1.weight")
    assert not is_swiglu_fc1_weight(fc1, "decoder.layers.1.mlp.experts.linear_fc2.weight0")


def test_install_contract_routes_router_to_adam_and_marks_fc1():
    from megatron.core.optimizer.emerging_optimizers import _EMERGING_OPTIMIZERS

    entry = _EMERGING_OPTIMIZERS["muon"]
    old_cls = entry.optimizer_cls
    old_overrides = entry.default_param_overrides
    try:
        install_muon_contract(fp8_states=True)
        router = torch.nn.Parameter(torch.empty(64, 1024))
        fc1 = torch.nn.Parameter(torch.empty(512, 1024))
        router_values = [
            value
            for key, value in entry.default_param_overrides.items()
            if key.matches(router, "decoder.layers.1.mlp.router.weight")
        ]
        fc1_values = [
            value
            for key, value in entry.default_param_overrides.items()
            if key.matches(fc1, "decoder.layers.1.mlp.experts.linear_fc1.weight0")
        ]
        assert {"optimizer": "adam"} in router_values
        assert {SPLIT_SWIGLU_FC1: True} in fc1_values
        assert entry.optimizer_cls.state_specs == MUON_STATE_SPECS
    finally:
        entry.optimizer_cls = old_cls
        entry.default_param_overrides = old_overrides


def test_swiglu_halves_are_orthogonalized_independently():
    optimizer = object.__new__(SplitSwiGLUTensorParallelMuon)
    calls = []

    def scaled(grad, tp_group, partition_dim):
        calls.append(tuple(grad.shape))
        return grad + len(calls)

    optimizer.scaled_orthogonalize_fn = scaled
    optimizer.pg_collection = None
    optimizer.tp_mode = "duplicated"
    optimizer.split_qkv = False
    optimizer.is_qkv_fn = lambda _: False
    optimizer.qkv_split_shapes = None
    parameter = torch.nn.Parameter(torch.empty(8, 4))
    grad = torch.zeros_like(parameter)
    result = optimizer.orthogonalize(
        parameter, grad, **{SPLIT_SWIGLU_FC1: True}
    )
    assert calls == [(4, 4), (4, 4)]
    assert torch.equal(result[:4], torch.ones(4, 4))
    assert torch.equal(result[4:], torch.full((4, 4), 2.0))


def test_custom_args_are_consumed_and_axis_checked():
    args, remaining = take_stage3_args(
        [
            "pretrain_gpt.py",
            "--stage3-arm",
            "muon_bf16_state_fp8",
            "--stage3-result-path",
            "result.jsonl",
            "--stage3-warmup-steps",
            "0",
            "--stage3-measure-steps",
            "1",
            "--optimizer-state-precision",
            "fp8",
            "--optimizer",
            "muon",
            "--train-iters",
            "1",
        ]
    )
    assert "--stage3-arm" not in remaining
    assert "--optimizer-state-precision" not in remaining
    validate_axis(
        args.stage3_arm,
        args.optimizer_state_precision,
        remaining,
        args.stage3_warmup_steps,
        args.stage3_measure_steps,
    )


def test_axis_rejects_silent_fp32_state_fallback():
    with pytest.raises(ValueError, match="requires --optimizer-state-precision fp8"):
        validate_axis(
            "adamw_bf16_state_fp8",
            "fp32",
            ["pretrain_gpt.py", "--optimizer", "adam", "--train-iters", "1"],
            0,
            1,
        )


def test_bootstrap_assert_rejects_plain_adam():
    raw = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    wrapped = type("Wrapped", (), {"optimizer": raw})()
    chained = type("Chained", (), {"chained_optimizers": [wrapped]})()
    with pytest.raises(AssertionError, match="raw classes"):
        assert_fp8_adam_bootstrap(chained)


def test_environment_reports_source_pins(monkeypatch):
    monkeypatch.setenv("STAGE3_MOE_MCORE_COMMIT", "a" * 40)
    monkeypatch.setenv("STAGE3_MOE_EO_COMMIT", "b" * 40)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: type("P", (), {"uuid": "GPU-test"})()
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "test GPU")
    monkeypatch.setattr(torch.cuda.nccl, "version", lambda: (2, 29, 7))
    monkeypatch.setattr("stage3_moe.result_writer._cublaslt_version", lambda: "130400")
    environment = _environment()
    assert environment["megatron_core"] == f"source@{'a' * 40}"
    assert environment["emerging_optimizers"] == f"source@{'b' * 40}"


def test_match_key_normalization_removes_only_fp8_compute_axis():
    base = ["p", "--optimizer", "adam", "--train-iters", "8"]
    treatment = [
        "p",
        "--optimizer",
        "adam",
        "--fp8-format",
        "hybrid",
        "--fp8-recipe",
        "delayed",
        "--train-iters",
        "8",
    ]
    assert _normalized_match_argv(treatment) == base
    assert _normalized_match_argv(base + ["--micro-batch-size", "2"]) != base


def test_parameter_ledger_uses_stable_names(monkeypatch):
    from stage3_moe import result_writer

    regular = torch.nn.Parameter(torch.empty(10))
    routed = torch.nn.Parameter(torch.empty(64))
    raw = type("Raw", (), {})()
    raw.state_specs = ()
    raw.param_groups = [{"params": [regular, routed]}]
    wrapped = type("Wrapped", (), {"optimizer": raw})()
    chained = type("Chained", (), {"chained_optimizers": [wrapped]})()
    names = {
        id(regular): "model_chunk0.embedding.weight",
        id(routed): "model_chunk0.decoder.layers.1.mlp.experts.linear_fc2.weight0",
    }
    monkeypatch.setattr(result_writer, "TOTAL_PARAMETERS", 74)
    monkeypatch.setattr(result_writer, "ACTIVE_PARAMETERS", 18)
    rows = parameter_group_ledger(chained, "adamw_bf16_state_fp32", names)
    assert rows[0]["parameters"] == 74
    assert rows[0]["active_parameters_per_token"] == 18
    assert len(rows[0]["named_parameter_manifest_sha256"]) == 64
