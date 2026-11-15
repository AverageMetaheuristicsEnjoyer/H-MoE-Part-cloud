import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.quantization_diagnostics import (
    adam_direction,
    install_live_diagnostics,
    metrics,
    selected_parameters,
)


def test_strata_include_first_middle_last_and_distinct_shapes():
    states = {i: {"exp_avg": torch.zeros(2 if i < 7 else 3)} for i in range(9)}
    selected = list(selected_parameters([{"params": list(range(9))}], states, 3))
    assert [row[2] for row in selected] == [0, 3, 6, 7, 8]


def test_adam_direction_matches_pytorch_next_step():
    p = torch.nn.Parameter(torch.tensor([0.3, -0.2, 0.9], dtype=torch.float64))
    opt = torch.optim.AdamW([p], lr=0.1, betas=(0.8, 0.93), eps=0.001, weight_decay=0.0)
    for gradient in ([0.4, 0.7, -0.6], [0.1, -0.2, 0.5]):
        p.grad = torch.tensor(gradient, dtype=torch.float64)
        before = p.detach().clone()
        opt.step()
        expected = adam_direction(opt.state[p], opt.param_groups[0], int(opt.state[p]["step"]))
        assert torch.allclose((before - p) / 0.1, expected, atol=1e-12, rtol=1e-12)


def test_error_metrics_do_not_underflow_small_second_moments():
    reference = torch.tensor([1e-25, 2e-25, 4e-25], dtype=torch.float32)
    result = metrics(reference, reference * 0.5)
    assert result["relative_frobenius"] == pytest.approx(0.5)
    assert result["cosine"] == pytest.approx(1.0)
    assert result["reference_norm"] > 0
    zero = metrics(torch.zeros(3), torch.zeros(3))
    assert zero["relative_frobenius"] == 0
    assert zero["cosine"] is None
    assert zero["element_relative_error_quantiles"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_live_hook_leaves_adam_update_and_states_unchanged(tmp_path):
    initial = torch.linspace(-0.1, 0.1, 259, device="cuda")
    p, q = torch.nn.Parameter(initial.clone()), torch.nn.Parameter(initial.clone())
    kwargs = dict(lr=0.003, betas=(0.9, 0.95), weight_decay=0.1)
    base, diagnostic = torch.optim.AdamW([p], **kwargs), torch.optim.AdamW([q], **kwargs)
    for opt, param in ((base, p), (diagnostic, q)):
        param.grad = torch.cos(initial)
        opt.step()
    output = tmp_path / "diagnostic.jsonl"
    install_live_diagnostics(diagnostic, [], output, samples=1)
    p.grad, q.grad = torch.sin(initial), torch.sin(initial)
    base.step()
    diagnostic.step()
    assert torch.equal(p, q)
    for key, value in base.state[p].items():
        other = diagnostic.state[q][key]
        assert torch.equal(value, other) if torch.is_tensor(value) else value == other
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert sum(row["kind"] == "adam_next_gradient_direction" for row in rows) == 3
    assert sum(row["kind"] == "one_step_ema_resolution" for row in rows) == 4
    assert rows[-1]["kind"] == "live_block_complete"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_live_hook_leaves_split_muon_update_unchanged(tmp_path):
    from stage3_moe.muon import SPLIT_SWIGLU_FC1, SplitSwiGLUTensorParallelMuon

    initial = torch.linspace(-0.2, 0.2, 768, device="cuda").reshape(32, 24)
    p, q = torch.nn.Parameter(initial.clone()), torch.nn.Parameter(initial.clone())
    kwargs = dict(lr=0.003, momentum=0.95, nesterov=True, weight_decay=0.1, split_qkv=False, fp32_matmul_prec="medium", coefficient_type="quintic", num_ns_steps=5, scale_mode="spectral", extra_scale_factor=0.2, pg_collection=None)
    base = SplitSwiGLUTensorParallelMuon([{"params": [p], SPLIT_SWIGLU_FC1: True}], **kwargs)
    diagnostic = SplitSwiGLUTensorParallelMuon([{"params": [q], SPLIT_SWIGLU_FC1: True}], **kwargs)
    for opt, param in ((base, p), (diagnostic, q)):
        param.grad = torch.cos(initial * 17)
        opt.step()
    output = tmp_path / "muon.jsonl"
    install_live_diagnostics(diagnostic, [], output, samples=1)
    p.grad, q.grad = torch.sin(initial * 13), torch.sin(initial * 13)
    base.step()
    diagnostic.step()
    assert torch.equal(p, q)
    assert torch.equal(base.state[p]["momentum_buffer"], diagnostic.state[q]["momentum_buffer"])
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert sum(row["kind"] == "muon_next_gradient_orthogonalized_direction" for row in rows) == 2
