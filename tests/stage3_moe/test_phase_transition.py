import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "Megatron-LM"))

from megatron.training import training


def test_iteration_schedule_sizes_and_resets_extension_phase(monkeypatch):
    args = SimpleNamespace(
        train_samples=None,
        train_iters=22_208,
        global_batch_size=208,
        full_validation=False,
        skip_train=False,
        eval_interval=250,
        start_eval_at_iter=None,
        eval_iters=32,
        phase_transition_iterations=[13_794],
        iteration=13_794,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)

    train_samples, _, _ = training.get_train_valid_test_num_samples()

    assert train_samples == (22_208 - 13_794) * 208


def test_phase_local_sampler_offset_starts_at_zero():
    source = (ROOT / "third_party/Megatron-LM/megatron/training/training.py").read_text()

    assert "consumed_train_samples_in_current_phase = (args.iteration - last_transition) * args.global_batch_size" in source


def test_time_match_schedule_and_data_capacity_are_pinned():
    launcher = (ROOT / "scripts/run_stage3_moe_pretrain.sh").read_text()
    plan = (ROOT / "configs/fineweb_edu_time_match_extension_plan.json").read_text()

    assert "adamw_fp8gemm_state_fp32) target_iters=19570" in launcher
    assert "muon_fp8gemm_state_fp32) target_iters=22208" in launcher
    assert 'phase_args=(--phase-transition-iterations "$time_match_branch")' in launcher
    assert '"extension_target_indexed_tokens": 3584229377' in plan
    assert '"shard_start_inclusive": 8' in plan


def test_fixed_lm_eval_never_trains_and_resets_eval_samplers():
    launcher = (ROOT / "scripts/run_stage3_moe_pretrain.sh").read_text()
    cloud = (ROOT / "scripts/cloud_moe_matched_lm_eval.sh").read_text()

    block = launcher.split("  eval-lm-fixed)", 1)[1].split("  eval-downstream)", 1)[0]
    assert "train_iters=1" in block
    assert "--no-load-optim --no-load-rng --skip-train" in block
    assert 'STAGE3_MOE_MICRO_BATCH=16' in cloud
    assert 'STAGE3_MOE_MATCHED_EVAL_REPEATS:-2' in cloud
    assert 'STAGE3_MOE_MATCHED_SKIP_REFERENCES:-0' in cloud
    assert 'eval_one extension_decay adamw_fp8gemm_state_fp32 "$control" 17242' in cloud


def test_extension_decay_control_starts_decay_at_phase_boundary():
    launcher = (ROOT / "scripts/run_stage3_moe_pretrain.sh").read_text()
    cloud = (ROOT / "scripts/cloud_moe_extension_decay_control.sh").read_text()

    block = launcher.split("  extension-decay-control)", 1)[1].split("  eval-lm-fixed)", 1)[0]
    assert "target_iters=$full_iters" in block
    assert "decay_iters=$full_decay_iters" in block
    assert '--save-interval "$full_iters" --no-save-optim --no-save-rng' in block
    assert 'control_phase_start=$((time_match_branch - time_match_plateau_iters))' in block
    assert 'phase_args=(--phase-transition-iterations "$control_phase_start")' in block
    assert 'source_dir/iter_0013794' in cloud
    assert 'STAGE3_MOE_TRAIN_DATA_PREFIX="$extension_root/data/train"' in cloud
    assert "phase_boundary=11466 sampler_offset_steps=2328 decay_steps=3448" in cloud


def test_time_match_stretched_decay_uses_the_whole_extension_phase():
    launcher = (ROOT / "scripts/run_stage3_moe_pretrain.sh").read_text()
    cloud = (ROOT / "scripts/cloud_moe_time_match_stretched_decay.sh").read_text()

    block = launcher.split("  time-match-stretched-decay)", 1)[1].split("  eval-lm-fixed)", 1)[0]
    assert "train_iters=19570" in block
    assert 'decay_iters=$((train_iters - time_match_branch))' in block
    assert 'phase_args=(--phase-transition-iterations "$time_match_branch")' in block
    assert '--save-interval "$train_iters" --no-save-optim --no-save-rng' in block
    assert 'source_dir/iter_0013794' in cloud
    assert 'STAGE3_MOE_TRAIN_DATA_PREFIX="$extension_root/data/train"' in cloud
    assert "phase_boundary=13794 sampler_offset_steps=0 decay_steps=5776" in cloud
