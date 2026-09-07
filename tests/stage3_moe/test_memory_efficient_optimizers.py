import copy
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "Megatron-LM"))
sys.path.insert(0, str(ROOT / "third_party" / "emerging-optimizers"))

from stage3_moe.frugal import (
    FRUGAL_COORD_CHOICE,
    FRUGAL_DENSITY,
    FRUGAL_UPDATE_GAP,
    FrugalCoordAdamW,
    install_frugal_contract,
    is_frugal_fallback,
)
from stage3_moe.pretrain_gpt import take_stage3_args, validate_axis
from stage3_moe.slim_adam import (
    SLIM_COMPRESS_DIMS,
    SlimAdamW,
    install_slimadam_contract,
    slim_compression_dims,
)


def test_frugal_routes_only_hidden_matrices_to_the_coordinate_optimizer():
    matrix = torch.nn.Parameter(torch.empty(4, 4))
    matrix.is_embedding_or_output_parameter = True
    assert is_frugal_fallback(matrix, "embedding.word_embeddings.weight")
    del matrix.is_embedding_or_output_parameter
    assert is_frugal_fallback(matrix, "decoder.layers.1.mlp.router.weight")
    assert not is_frugal_fallback(
        matrix, "decoder.layers.1.mlp.experts.linear_fc2.weight0"
    )
    assert is_frugal_fallback(
        torch.nn.Parameter(torch.empty(4)), "decoder.layers.1.input_layernorm.weight"
    )


def test_frugal_projects_columns_and_resumes_exactly():
    parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(2, 4))
    optimizer = FrugalCoordAdamW(
        [{"params": [parameter]}],
        lr=0.1,
        betas=(0.5, 0.75),
        eps=1e-8,
        density=0.5,
        update_gap=2,
    )
    parameter.grad = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["coord_indices"].shape == (2,)
    assert state["exp_avg"].shape == (2, 2)
    assert state["exp_avg_sq"].shape == (2, 2)

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = FrugalCoordAdamW(
        [{"params": [resumed_parameter]}],
        lr=0.1,
        betas=(0.5, 0.75),
        eps=1e-8,
        density=0.5,
        update_gap=2,
    )
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))

    parameter.grad = torch.full_like(parameter, 2.0)
    resumed_parameter.grad = torch.full_like(resumed_parameter, 2.0)
    optimizer.step()
    resumed.step()
    assert torch.equal(parameter, resumed_parameter)

    rng_state = torch.random.get_rng_state()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    torch.random.set_rng_state(rng_state)
    resumed_parameter.grad = torch.ones_like(resumed_parameter)
    resumed.step()
    assert torch.equal(parameter, resumed_parameter)
    assert torch.equal(
        optimizer.state[parameter]["coord_indices"],
        resumed.state[resumed_parameter]["coord_indices"],
    )


def test_frugal_matches_efficient_training_coordinate_recipe():
    assert FRUGAL_COORD_CHOICE == "columns"
    assert FRUGAL_DENSITY == 0.25
    assert FRUGAL_UPDATE_GAP == 50


def test_slimadam_static_mcore_rules_leave_qkv_and_router_full():
    matrix = torch.nn.Parameter(torch.empty(8, 4))
    assert slim_compression_dims(
        matrix, "decoder.layers.1.self_attention.linear_proj.weight"
    ) == (0,)
    assert slim_compression_dims(
        matrix, "decoder.layers.1.mlp.experts.linear_fc1.weight0"
    ) == (0,)
    assert slim_compression_dims(
        matrix, "model_chunk0.embedding.word_embeddings.weight"
    ) == (1,)
    assert slim_compression_dims(matrix, "model_chunk0.output_layer.weight") == (1,)
    assert slim_compression_dims(
        matrix, "decoder.layers.1.self_attention.linear_qkv.weight"
    ) is None
    assert slim_compression_dims(
        matrix, "decoder.layers.1.mlp.router.weight"
    ) is None


def test_slimadam_compressed_second_moment_and_resume():
    start = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    gradient = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    parameter = torch.nn.Parameter(start.clone())
    optimizer = SlimAdamW(
        [{"params": [parameter], SLIM_COMPRESS_DIMS: (0,)}],
        lr=0.1,
        betas=(0.0, 0.0),
        eps=0.0,
    )
    parameter.grad = gradient.clone()
    optimizer.step()

    expected_second = gradient.square().mean(dim=0, keepdim=True)
    expected = start - 0.1 * gradient / expected_second.sqrt()
    assert torch.allclose(parameter, expected)
    assert optimizer.state[parameter]["exp_avg"].shape == parameter.shape
    assert optimizer.state[parameter]["exp_avg_sq"].shape == (1, 2)

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = SlimAdamW(
        [{"params": [resumed_parameter], SLIM_COMPRESS_DIMS: (0,)}],
        lr=0.1,
        betas=(0.0, 0.0),
        eps=0.0,
    )
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    parameter.grad = gradient.flip(0)
    resumed_parameter.grad = gradient.flip(0)
    optimizer.step()
    resumed.step()
    assert torch.equal(parameter, resumed_parameter)


def test_stage3_registry_entries_and_cli_contract():
    from megatron.core.optimizer.emerging_optimizers import _EMERGING_OPTIMIZERS

    old_frugal = _EMERGING_OPTIMIZERS.get("frugal")
    old_slimadam = _EMERGING_OPTIMIZERS.get("slimadam")
    try:
        install_frugal_contract()
        install_slimadam_contract()
        assert _EMERGING_OPTIMIZERS["frugal"].optimizer_cls is FrugalCoordAdamW
        assert _EMERGING_OPTIMIZERS["slimadam"].optimizer_cls is SlimAdamW
    finally:
        if old_frugal is None:
            _EMERGING_OPTIMIZERS.pop("frugal", None)
        else:
            _EMERGING_OPTIMIZERS["frugal"] = old_frugal
        if old_slimadam is None:
            _EMERGING_OPTIMIZERS.pop("slimadam", None)
        else:
            _EMERGING_OPTIMIZERS["slimadam"] = old_slimadam

    for arm, optimizer_name in (
        ("frugal_coord_bf16_state_fp32", "frugal"),
        ("slimadam_bf16_state_fp32", "slimadam"),
    ):
        args, remaining = take_stage3_args(
            [
                "pretrain_gpt.py",
                "--stage3-arm",
                arm,
                "--stage3-result-path",
                "result.jsonl",
                "--stage3-warmup-steps",
                "0",
                "--stage3-measure-steps",
                "1",
                "--optimizer-state-precision",
                "fp32",
                "--optimizer",
                optimizer_name,
                "--ckpt-format",
                "torch",
                "--train-iters",
                "1",
            ]
        )
        validate_axis(
            args.stage3_arm,
            args.optimizer_state_precision,
            remaining,
            args.stage3_warmup_steps,
            args.stage3_measure_steps,
        )


def test_memory_efficient_optimizers_reject_torch_dist_checkpointing():
    for arm, optimizer_name in (
        ("frugal_coord_bf16_state_fp32", "frugal"),
        ("slimadam_bf16_state_fp32", "slimadam"),
    ):
        try:
            validate_axis(
                arm,
                "fp32",
                [
                    "pretrain_gpt.py",
                    "--optimizer",
                    optimizer_name,
                    "--ckpt-format",
                    "torch_dist",
                    "--train-iters",
                    "1",
                ],
                0,
                1,
            )
        except ValueError as error:
            assert str(error) == f"{optimizer_name} requires --ckpt-format torch"
        else:
            raise AssertionError("torch_dist checkpointing was accepted")
