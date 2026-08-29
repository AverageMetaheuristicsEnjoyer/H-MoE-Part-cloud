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


def test_schedule_matrix_tails_use_native_sources_and_one_extension_sequence():
    launcher = (ROOT / "scripts/run_stage3_moe_pretrain.sh").read_text()
    cloud = (ROOT / "scripts/cloud_moe_schedule_matrix.sh").read_text()

    block = launcher.split("  schedule-tail|schedule-tail-smoke)", 1)[1].split(
        "  eval-lm-fixed)", 1
    )[0]
    assert "adamw_bf16_state_fp32|adamw_bf16_state_fp8" in block
    assert "target_iters=19570" in block
    assert "short) decay_iters=$full_decay_iters" in block
    assert "long) decay_iters=$((19570 - time_match_branch))" in block
    assert 'phase_args=(--phase-transition-iterations "$time_match_branch")' in block
    assert 'train_iters=$((time_match_branch + 1))' in block
    assert '--save-interval "$target_iters"' in block
    assert "--no-save-optim --no-save-rng" in block

    assert "1c-mb4" in cloud
    assert "iter_0013794" in cloud
    assert "STAGE3_MOE_MICRO_BATCH=16" in cloud
    assert 'STAGE3_MOE_CKPT_ROOT="$work/checkpoints"' in cloud
    assert "STAGE3_MOE_EVAL_INTERVAL=1" in cloud
    assert "extension_local_consumed_samples\": 0" in cloud
    assert "STAGE3_MOE_MATCHED_EVAL_REPEATS=2" in cloud
    assert 'STAGE3_MOE_MATCHED_CANDIDATE_ARM="$arm"' in cloud
    assert 'STAGE3_MOE_ROUTING_ARM="$arm"' in cloud
    assert "/home/jovyan/.cache/huggingface/token" in cloud


def test_fixed_candidate_audits_accept_the_requested_arm():
    matched = (ROOT / "scripts/cloud_moe_matched_lm_eval.sh").read_text()
    routing = (ROOT / "scripts/cloud_moe_fixed_routing_audit.sh").read_text()

    assert "STAGE3_MOE_MATCHED_CANDIDATE_ARM" in matched
    assert 'eval_one "$candidate_label" "$candidate_arm"' in matched
    assert "STAGE3_MOE_ROUTING_ARM" in routing


def test_original_data_plateau_control_is_constant_lr_and_reuses_original_cache():
    launcher = (ROOT / "scripts/run_stage3_moe_pretrain.sh").read_text()
    cloud = (ROOT / "scripts/cloud_moe_original_data_plateau_control.sh").read_text()
    matched_eval = (ROOT / "scripts/cloud_moe_matched_lm_eval.sh").read_text()

    block = launcher.split("  original-data-plateau-control)", 1)[1].split("  eval-lm-fixed)", 1)[0]
    assert "plateau_end=$((time_match_branch + time_match_plateau_iters))" in block
    assert "train_iters=$full_iters" in block
    assert "target_iters=$((plateau_end + full_decay_iters))" in block
    assert 'exit_args=(--exit-interval "$plateau_end")' in block
    assert '--save-interval "$plateau_end"' in block
    assert "--no-save-optim --no-save-rng" in block
    assert "/home/jovyan/data/fineweb-edu-gpt2-megatron" in cloud
    assert "/workspace-SR006.nfs3/hmoe-cloud/pretrain" in cloud
    assert "start=13794 end=$plateau_end plateau_steps=2328" in cloud
    assert 'eval_one original_plateau_source adamw_fp8gemm_state_fp32 "$plateau_source" 13794' in matched_eval
