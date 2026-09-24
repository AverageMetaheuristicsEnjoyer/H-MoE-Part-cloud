"""scripts/expert_gemm_bench.py must compute what stage3_moe/monarch.py computes.

The benchmark re-implements the packed butterfly instead of importing it, so its
timings describe production only if outputs and gradients match
MonarchFactors.forward_packed. On CPU torch._grouped_mm is swapped for a
per-group loop; on an H100 the real kernel runs.

Runs under pytest or as `python tests/stage3_moe/test_expert_gemm_bench.py`.
"""

import importlib.util
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode


ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from stage3_moe.monarch import (  # noqa: E402
    DenseExperts,
    LowRankExperts,
    MonarchFactors,
    MonarchGroupedMLP,
    _packed_layout,
)

_spec = importlib.util.spec_from_file_location(
    "expert_gemm_bench", ROOT / "scripts" / "expert_gemm_bench.py"
)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
DTYPE = torch.bfloat16 if CUDA else torch.float32
EXPERTS = 8
# Stage 3 1C expert bank: fc1 is hidden -> 2 * expert_hidden, fc2 the reverse.
PROJECTIONS = ((1024, 512), (256, 1024))


def _loop_grouped_mm(a, b, offs):
    out = a.new_zeros(a.shape[0], b.shape[-1])
    start = 0
    for group, end in enumerate(offs.tolist()):
        out[start:end] = a[start:end] @ b[group]
        start = end
    return out


if not CUDA:
    torch._grouped_mm = _loop_grouped_mm


def _counts(kind):
    if kind == "uniform":
        return torch.full((EXPERTS,), 24, dtype=torch.long)
    # skewed, with one idle expert so the active-expert index_select path runs
    generator = torch.Generator().manual_seed(0)
    counts = torch.randint(1, 60, (EXPERTS,), generator=generator)
    counts[1] = 0
    return counts


def _cases():
    for in_features, out_features in PROJECTIONS:
        for blocks in (2, 4):
            for kind in ("uniform", "skewed"):
                yield in_features, out_features, blocks, _counts(kind).to(DEVICE)


def _run(forward, parameters, x, grad):
    x = x.clone().requires_grad_()
    y = forward(x)
    return (y, *torch.autograd.grad(y, [x, *parameters], grad))


def _relative_error(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


TOLERANCE = 1e-2 if CUDA else 1e-5


def test_bench_copy_is_bitwise_production():
    torch.manual_seed(0)
    for in_features, out_features, blocks, counts in _cases():
        for sharing in ("none", "hidden", "all"):
            copy = bench.MonarchBank(
                in_features, out_features, blocks, EXPERTS, DTYPE, DEVICE, sharing=sharing
            )
            production = MonarchFactors(
                in_features, out_features, blocks, EXPERTS, DTYPE, DEVICE,
                share1=copy.share1, share2=copy.share2,
            )
            assert copy.blkdiag1.shape == production.blkdiag1.shape
            assert copy.blkdiag2.shape == production.blkdiag2.shape
            with torch.no_grad():
                copy.blkdiag1.copy_(production.blkdiag1)
                copy.blkdiag2.copy_(production.blkdiag2)
            layout = _packed_layout(counts, blocks)
            x = torch.randn(int(counts.sum()), in_features, device=DEVICE, dtype=DTYPE)
            grad = torch.randn(x.shape[0], out_features, device=DEVICE, dtype=DTYPE)

            expected = _run(
                lambda v: production.forward_packed(v, layout),
                (production.blkdiag1, production.blkdiag2), x, grad,
            )
            actual = _run(lambda v: copy(v, layout), (copy.blkdiag1, copy.blkdiag2), x, grad)
            for name, a, e in zip(("y", "dx", "dw1", "dw2"), actual, expected):
                assert torch.equal(a, e), (
                    name, sharing, in_features, out_features, blocks, counts.tolist()
                )


def test_packed_path_matches_per_expert_butterfly():
    # independent of the copy: the packed path against MonarchFactors.forward,
    # the plain bmm butterfly production uses for attention and shared experts
    torch.manual_seed(1)
    for in_features, out_features, blocks, counts in _cases():
        factors = MonarchFactors(in_features, out_features, blocks, EXPERTS, DTYPE, DEVICE)
        layout = _packed_layout(counts, blocks)
        x = torch.randn(int(counts.sum()), in_features, device=DEVICE, dtype=DTYPE)
        packed = factors.forward_packed(x, layout)
        segments = x.split(counts.tolist())
        reference = torch.cat(
            [factors.forward(segment, group=e) for e, segment in enumerate(segments)]
        )
        error = _relative_error(packed, reference)
        assert error < TOLERANCE, (error, in_features, out_features, blocks)


def test_shared_variants_are_production_factors_with_a_tied_factor():
    torch.manual_seed(2)
    for in_features, out_features, blocks, counts in _cases():
        layout = _packed_layout(counts, blocks)
        x = torch.randn(int(counts.sum()), in_features, device=DEVICE, dtype=DTYPE)
        grad = torch.randn(x.shape[0], out_features, device=DEVICE, dtype=DTYPE)
        for sharing in ("hidden", "all"):
            shared = bench.MonarchBank(
                in_features, out_features, blocks, EXPERTS, DTYPE, DEVICE, sharing=sharing
            )
            # "hidden" must tie the factor that touches the model dimension
            hidden = max(in_features, out_features)
            tied = [p for p in (shared.blkdiag1, shared.blkdiag2) if p.shape[0] == 1]
            if sharing == "hidden":
                assert len(tied) == 1
                block_shape = tied[0].shape[-2:]
                assert hidden // blocks in block_shape, (block_shape, hidden, blocks)
            else:
                assert len(tied) == 2

            production = MonarchFactors(in_features, out_features, blocks, EXPERTS, DTYPE, DEVICE)
            with torch.no_grad():
                production.blkdiag1.copy_(shared.blkdiag1.expand_as(production.blkdiag1))
                production.blkdiag2.copy_(shared.blkdiag2.expand_as(production.blkdiag2))
            actual = _run(lambda v: shared(v, layout), (shared.blkdiag1, shared.blkdiag2), x, grad)
            expected = list(_run(
                lambda v: production.forward_packed(v, layout),
                (production.blkdiag1, production.blkdiag2), x, grad,
            ))
            # a tied factor's gradient is the sum of the per-expert gradients
            for index, parameter in ((2, shared.blkdiag1), (3, shared.blkdiag2)):
                if parameter.shape[0] == 1:
                    expected[index] = expected[index].float().sum(0, keepdim=True)
            for name, a, e in zip(("y", "dx", "dw1", "dw2"), actual, expected):
                error = _relative_error(a, e)
                assert error < TOLERANCE, (name, sharing, error, in_features, out_features, blocks)


def test_bench_expert_mlp_matches_production_module():
    if not CUDA:
        return "skip"  # MonarchGroupedMLP allocates on torch.cuda.current_device()
    torch.manual_seed(3)
    config = SimpleNamespace(
        hidden_size=1024,
        moe_ffn_hidden_size=256,
        expert_model_parallel_size=1,
        params_dtype=DTYPE,
    )
    for blocks, share, variant in (
        (2, "none", "monarch"), (4, "none", "monarch"), (2, "hidden", "shared"),
    ):
        os.environ["STAGE3_MONARCH_BLOCKS"] = str(blocks)
        os.environ["STAGE3_MONARCH_SHARE"] = share
        production = MonarchGroupedMLP(EXPERTS, config)
        copy = bench.ExpertMLP(variant, 1024, 256, EXPERTS, blocks, DTYPE, DEVICE)
        with torch.no_grad():
            for name in ("fc1", "fc2"):
                getattr(copy, name).blkdiag1.copy_(getattr(production, name).blkdiag1)
                getattr(copy, name).blkdiag2.copy_(getattr(production, name).blkdiag2)
        for kind in ("uniform", "skewed"):
            counts = _counts(kind)  # production receives tokens_per_expert on the CPU
            x = torch.randn(int(counts.sum()), 1024, device=DEVICE, dtype=DTYPE)
            probs = torch.ones(x.shape[0], device=DEVICE, dtype=DTYPE)
            expected, _ = production(x, counts, probs)
            layout = _packed_layout(counts.to(DEVICE), blocks)
            actual = copy(x, layout, None)
            assert torch.equal(actual, expected), (blocks, share, kind)


def test_dense_bank_matches_per_expert_matmul():
    torch.manual_seed(4)
    for in_features, out_features in PROJECTIONS:
        for kind in ("uniform", "skewed"):
            counts = _counts(kind).to(DEVICE)
            dense = bench.DenseBank(in_features, out_features, EXPERTS, DTYPE, DEVICE)
            x = torch.randn(int(counts.sum()), in_features, device=DEVICE, dtype=DTYPE)
            actual = dense(x, _packed_layout(counts, 1))
            reference = torch.cat([
                segment @ dense.weight[e].T
                for e, segment in enumerate(x.split(counts.tolist()))
            ])
            error = _relative_error(actual, reference)
            assert error < TOLERANCE, (error, in_features, out_features, kind)


def test_dense_and_lowrank_experts_match_per_expert_matmul():
    torch.manual_seed(5)
    for in_features, out_features in PROJECTIONS:
        rank = min(in_features, out_features) // 2
        banks = (
            (DenseExperts(in_features, out_features, EXPERTS, DTYPE, DEVICE),
             lambda bank, e: bank.weight[e]),
            (LowRankExperts(in_features, out_features, rank, EXPERTS, DTYPE, DEVICE),
             lambda bank, e: bank.up[e] @ bank.down[e]),
        )
        for bank, matrix in banks:
            params = tuple(bank.parameters())
            assert all(getattr(p, "monarch_factor", False) for p in params)
            for kind in ("uniform", "skewed"):
                counts = _counts(kind).to(DEVICE)
                x = torch.randn(int(counts.sum()), in_features, device=DEVICE, dtype=DTYPE)
                grad = torch.randn(x.shape[0], out_features, device=DEVICE, dtype=DTYPE)
                actual = _run(lambda v: bank.forward_packed(v, _packed_layout(counts, 1)),
                              params, x, grad)
                expected = _run(
                    lambda v: torch.cat([segment @ matrix(bank, e).T
                                         for e, segment in enumerate(v.split(counts.tolist()))]),
                    params, x, grad)
                for name, a, e in zip(("y", "dx", "dp0", "dp1"), actual, expected):
                    error = _relative_error(a, e)
                    assert error < TOLERANCE, (type(bank).__name__, name, kind, error)


def test_lowrank_experts_have_the_monarch_parameter_count():
    for in_features, out_features in PROJECTIONS:
        monarch = MonarchFactors(in_features, out_features, 2, 1, DTYPE, DEVICE)
        lowrank = LowRankExperts(in_features, out_features, min(in_features, out_features) // 2,
                                 1, DTYPE, DEVICE)
        count = lambda module: sum(p.numel() for p in module.parameters())
        assert count(lowrank) == count(monarch), (in_features, out_features)


def test_production_expert_kinds_run_and_match_their_parameters():
    if not CUDA:
        return "skip"  # MonarchGroupedMLP allocates on torch.cuda.current_device()
    torch.manual_seed(6)
    config = SimpleNamespace(hidden_size=1024, moe_ffn_hidden_size=256,
                             expert_model_parallel_size=1, params_dtype=DTYPE)
    os.environ["STAGE3_MONARCH_BLOCKS"] = "2"
    os.environ["STAGE3_MONARCH_SHARE"] = "none"
    counts = _counts("skewed")
    x = torch.randn(int(counts.sum()), 1024, device=DEVICE, dtype=DTYPE)
    probs = torch.rand(x.shape[0], device=DEVICE, dtype=torch.float32)
    sizes = {}
    for kind in ("monarch", "monarch_dense_down", "lowrank"):
        os.environ["STAGE3_MONARCH_EXPERTS"] = kind
        mlp = MonarchGroupedMLP(EXPERTS, config)
        sizes[kind] = sum(p.numel() for p in mlp.parameters())
        output, _ = mlp(x, counts, probs)
        output.float().pow(2).mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mlp.parameters())
        # the bank is a function of its own parameters: compare with a dense re-computation
        eye1 = torch.eye(1024, device=DEVICE, dtype=DTYPE)
        eye2 = torch.eye(256, device=DEVICE, dtype=DTYPE)
        segments = x.split(counts.tolist())
        weights = probs.split(counts.tolist())
        reference = []
        for e, (segment, p) in enumerate(zip(segments, weights)):
            if segment.shape[0] == 0:
                continue
            w1 = _single_expert(mlp.fc1, e, eye1)
            w2 = _single_expert(mlp.fc2, e, eye2)
            gate, value = (segment @ w1.T).chunk(2, dim=-1)
            reference.append((F.silu(gate) * value * p[:, None]).to(DTYPE) @ w2.T)
        error = _relative_error(output.detach(), torch.cat(reference))
        assert error < TOLERANCE, (kind, error)
    assert sizes["lowrank"] == sizes["monarch"] < sizes["monarch_dense_down"], sizes
    os.environ["STAGE3_MONARCH_EXPERTS"] = "monarch"


def _single_expert(bank, expert, eye):
    """Expert `expert`'s dense matrix, read off the bank by feeding it the identity."""
    with torch.no_grad():
        if isinstance(bank, MonarchFactors):
            return bank(eye, group=expert).T
        if isinstance(bank, DenseExperts):
            return bank.weight[expert]
        return bank.up[expert] @ bank.down[expert]


def test_parameter_and_flop_arithmetic():
    # params = min(in, out) * (in + out) / nb, so a square matrix at nb=2 saves nothing
    for in_features, out_features in (
        (1024, 512), (256, 1024), (1024, 1024), (1024, 1536), (1024, 5632), (2816, 1024),
    ):
        for blocks in (2, 4):
            shape1, shape2 = bench.monarch_shapes(in_features, out_features, blocks)
            params = math.prod(shape1) + math.prod(shape2)
            assert params * blocks == min(in_features, out_features) * (in_features + out_features)
    shape1, shape2 = bench.monarch_shapes(1024, 1024, 2)
    assert math.prod(shape1) + math.prod(shape2) == 1024 * 1024

    # the forward costs exactly 2 FLOPs per factor parameter per token
    tokens = 16
    for in_features, out_features in PROJECTIONS:
        for blocks in (2, 4):
            factors = MonarchFactors(in_features, out_features, blocks, 1, DTYPE, DEVICE)
            x = torch.randn(tokens, in_features, device=DEVICE, dtype=DTYPE)
            with FlopCounterMode(display=False) as counter:
                factors(x)
            params = factors.blkdiag1.numel() + factors.blkdiag2.numel()
            assert counter.get_total_flops() == 2 * tokens * params


if __name__ == "__main__":
    print(f"torch {torch.__version__} device {DEVICE} dtype {DTYPE}", flush=True)
    failures = 0
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                outcome = test()
                print(f"{'SKIP' if outcome == 'skip' else 'PASS'} {name}", flush=True)
            except AssertionError as error:
                failures += 1
                print(f"FAIL {name}: {error!r}", flush=True)
    print(f"TESTS_FAILED={failures}", flush=True)
    sys.exit(1 if failures else 0)
