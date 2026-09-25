"""The sweep's own logic: what counts as a finished point, and what an OOM costs.

Running out of memory is a recorded result here, not an error, and it has to stop
the larger micro-batches of that same model and arm -- otherwise a sweep spends an
hour re-discovering the same wall five times.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from stage3_moe import membench_sweep  # noqa: E402


def args(tmp_path, **overrides):
    from argparse import Namespace

    base = dict(
        models=None,
        arms=None,
        micro_batches=None,
        global_batch=16,
        warmup_steps=5,
        measured_steps=12,
        gpu_count=1,
        results_root=tmp_path / "results",
        log_root=tmp_path / "runs",
        rerun=False,
        export_only=False,
    )
    base.update(overrides)
    return Namespace(**base)


def test_oom_is_recognized_from_the_train_log():
    assert membench_sweep.looks_like_oom("torch.cuda.OutOfMemoryError: CUDA out of memory.")
    assert membench_sweep.looks_like_oom("... tried to allocate 2.00 GiB, out of memory ...")
    assert not membench_sweep.looks_like_oom("iteration 7/17 | elapsed time per iteration (ms)")


def test_a_point_is_reused_only_at_the_controls_it_was_measured_at(tmp_path):
    parsed = args(tmp_path)
    path = membench_sweep.point_path(parsed.results_root, "1p029b", "muon_bf16_state_fp32", 4)
    expected = membench_sweep.controls(parsed, "1p029b", 4)
    membench_sweep.write_point(path, {"status": "complete", "controls": expected})

    assert membench_sweep.stored_point(path, expected) is not None
    # A different measured window is a different measurement.
    other = membench_sweep.controls(args(tmp_path, measured_steps=40), "1p029b", 4)
    assert membench_sweep.stored_point(path, other) is None


def test_a_failed_point_is_not_reused(tmp_path):
    parsed = args(tmp_path)
    path = membench_sweep.point_path(parsed.results_root, "1p029b", "adamw_bf16_state_fp32", 1)
    expected = membench_sweep.controls(parsed, "1p029b", 1)
    membench_sweep.write_point(path, {"status": "failed", "controls": expected})
    assert membench_sweep.stored_point(path, expected) is None


def test_accumulation_keeps_tokens_per_step_fixed(tmp_path):
    parsed = args(tmp_path)
    steps = {
        micro_batch: membench_sweep.controls(parsed, "1p029b", micro_batch)
        for micro_batch in (1, 2, 4, 8, 16)
    }
    tokens = {
        control["micro_batch"] * control["accumulation_steps"] * control["sequence_length"]
        for control in steps.values()
    }
    assert tokens == {16 * 2048}


def test_a_micro_batch_that_does_not_divide_the_global_batch_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["membench_sweep.py", "--micro-batches", "3",
         "--results-root", str(tmp_path / "r"), "--log-root", str(tmp_path / "l")],
    )
    with pytest.raises(ValueError, match="not divisible"):
        membench_sweep.main()


def test_oom_skips_every_larger_micro_batch_of_that_arm(tmp_path, monkeypatch):
    attempted = []

    def fake_run_point(parsed, model, arm, micro_batch):
        attempted.append((model, arm, micro_batch))
        status = "oom" if micro_batch >= 4 and arm.startswith("adamw") else "complete"
        return {
            "model": model,
            "arm": arm,
            "micro_batch": micro_batch,
            "controls": membench_sweep.controls(parsed, model, micro_batch),
            "status": status,
            "wall_seconds": 1.0,
            "record": {},
        }

    monkeypatch.setattr(membench_sweep, "run_point", fake_run_point)
    monkeypatch.setattr(
        sys, "argv",
        ["membench_sweep.py",
         "--models", "1p029b",
         "--arms", "adamw_bf16_state_fp32,muon_bf16_state_fp32",
         "--micro-batches", "1,2,4,8",
         "--results-root", str(tmp_path / "r"), "--log-root", str(tmp_path / "l")],
    )
    assert membench_sweep.main() == 0

    adamw = [batch for model, arm, batch in attempted if arm.startswith("adamw")]
    muon = [batch for model, arm, batch in attempted if arm.startswith("muon")]
    # AdamW hits the wall at 4 and is never launched at 8; Muon is unaffected.
    assert adamw == [1, 2, 4]
    assert muon == [1, 2, 4, 8]

    skipped = json.loads(
        (tmp_path / "r" / "runs" / "1p029b-adamw_bf16_state_fp32-mb8.json").read_text()
    )
    assert skipped["status"] == "skipped_after_oom"


def test_export_row_reports_memory_and_time_for_a_finished_point():
    payload = {
        "model": "2p094b",
        "arm": "muon_bf16_state_fp8",
        "micro_batch": 4,
        "status": "complete",
        "record": {
            "measurement": {
                "memory": {"max_allocated_bytes": 21_120_808_960,
                           "max_reserved_bytes": 23_000_000_000},
                "timing": {"full_step_seconds": 1.25,
                           "optimizer_step_seconds": 0.0312,
                           "tokens_per_second": 26214.4},
            },
            "optimizer_state": {"persistent_total_bytes": 1_181_426_848},
        },
    }
    row = membench_sweep.export_row(payload).split("\t")
    assert row[:5] == ["PT", "2p094b", "muon_bf16_state_fp8", "4", "complete"]
    assert row[5:] == ["21.121", "23.000", "1.181", "1250.00", "31.20", "26214.4"]


def test_export_row_leaves_an_oom_point_blank_rather_than_zero():
    row = membench_sweep.export_row(
        {"model": "2p094b", "arm": "adamw_bf16_state_fp32", "micro_batch": 16, "status": "oom"}
    ).split("\t")
    assert row[4] == "oom"
    assert row[5:] == [""] * 6


def test_a_broken_arm_is_abandoned_instead_of_retried_at_every_batch(tmp_path, monkeypatch):
    attempted = []

    def fake_run_point(parsed, model, arm, micro_batch):
        attempted.append((arm, micro_batch))
        return {
            "model": model,
            "arm": arm,
            "micro_batch": micro_batch,
            "controls": membench_sweep.controls(parsed, model, micro_batch),
            "status": "failed" if arm.startswith("muon") else "complete",
            "wall_seconds": 1.0,
            "record": {},
        }

    monkeypatch.setattr(membench_sweep, "run_point", fake_run_point)
    monkeypatch.setattr(
        sys, "argv",
        ["membench_sweep.py",
         "--models", "1p029b",
         "--arms", "adamw_bf16_state_fp32,muon_bf16_state_fp32",
         "--micro-batches", "1,2,4",
         "--results-root", str(tmp_path / "r"), "--log-root", str(tmp_path / "l")],
    )
    assert membench_sweep.main() == 0

    assert [batch for arm, batch in attempted if arm.startswith("muon")] == [1]
    assert [batch for arm, batch in attempted if arm.startswith("adamw")] == [1, 2, 4]
    skipped = json.loads(
        (tmp_path / "r" / "runs" / "1p029b-muon_bf16_state_fp32-mb4.json").read_text()
    )
    assert skipped["status"] == "skipped_after_failure"


def test_a_skipped_point_is_not_reused_as_a_measurement(tmp_path):
    parsed = args(tmp_path)
    path = membench_sweep.point_path(parsed.results_root, "1p029b", "muon_bf16_state_fp32", 4)
    expected = membench_sweep.controls(parsed, "1p029b", 4)
    membench_sweep.write_point(path, {"status": "skipped_after_failure", "controls": expected})
    assert membench_sweep.stored_point(path, expected) is None
