from pathlib import Path


ROOT = Path(__file__).parents[2]
LAUNCHER = ROOT / "scripts" / "run_stage3_moe_pretrain.sh"
CLOUD_GATE = ROOT / "scripts" / "cloud_moe_optimizer_gate.sh"


def test_resume_gate_crosses_frugal_refresh_boundary():
    launcher = LAUNCHER.read_text()

    block = launcher.split("  resume-gate)", 1)[1].split("  stability)", 1)[0]
    assert "train_iters=52" in block
    assert "--save-interval 50" in block
    assert "--exit-interval 50" in block
    assert 'load_args=(--load "$resume_dir")' in block


def test_calibration_budgets_and_optimizer_overrides_are_explicit():
    launcher = LAUNCHER.read_text()

    stability = launcher.split("  stability)", 1)[1].split("  lr-screen)", 1)[0]
    lr_screen = launcher.split("  lr-screen)", 1)[1].split("  bench)", 1)[0]
    assert "train_iters=235" in stability
    assert "decay_iters=47" in stability
    assert "warmup_iters=2" in stability
    assert "train_iters=587" in lr_screen
    assert "decay_iters=117" in lr_screen
    assert "warmup_iters=6" in lr_screen
    assert 'learning_rate=${STAGE3_MOE_LR:-1.63e-3}' in launcher
    assert 'adam_beta2=${STAGE3_MOE_ADAM_BETA2:-0.95}' in launcher
    assert '--adam-beta1 0.9 --adam-beta2 "$adam_beta2"' in launcher
    assert '--lr "$learning_rate"' in launcher


def test_cloud_gate_preregisters_only_the_three_frugal_recipes():
    gate = CLOUD_GATE.read_text()

    assert "matched 1.63e-3 1.63e-4 0.95 587" in gate
    assert "efficient-training-1e3 1e-3 1e-4 0.999 587" in gate
    assert "efficient-training-2e3 2e-3 2e-4 0.999 587" in gate
    assert "frugal_coord_bf16_state_fp32 matched 1.63e-3 1.63e-4 0.95 235" in gate
    assert "slimadam_bf16_state_fp32 matched 1.63e-3 1.63e-4 0.95 235" in gate
    assert 'routing["minimum_to_mean"] < 0.1' in gate
    assert 'routing["coefficient_of_variation"] >= 0.2' in gate
    assert "number of nan iterations: +0" in gate
    assert "GATE_CKPT_REMOVED" in gate
