from pathlib import Path


ROOT = Path(__file__).parents[2]
LAUNCHER = ROOT / "scripts" / "run_stage3_moe_pretrain.sh"
CLOUD_GATE = ROOT / "scripts" / "cloud_moe_optimizer_gate.sh"
CLOUD_FULL = ROOT / "scripts" / "cloud_moe_full.sh"
WGRAD_BENCH = ROOT / "scripts" / "cloud_moe_wgrad_image_bench.sh"


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
    lr_screen = launcher.split("  lr-screen)", 1)[1].split(
        "  routing-calibration)", 1
    )[0]
    routing_calibration = launcher.split("  routing-calibration)", 1)[1].split(
        "  routing-calibration-2254)", 1
    )[0]
    routing_calibration_2254 = launcher.split(
        "  routing-calibration-2254)", 1
    )[1].split("  bench)", 1)[0]
    assert "train_iters=235" in stability
    assert "decay_iters=47" in stability
    assert "warmup_iters=2" in stability
    assert "train_iters=587" in lr_screen
    assert "decay_iters=117" in lr_screen
    assert "warmup_iters=6" in lr_screen
    assert "train_iters=587" in routing_calibration
    assert "target_iters=$full_iters" in routing_calibration
    assert "decay_iters=$full_decay_iters" in routing_calibration
    assert "warmup_iters=" not in routing_calibration
    assert "train_iters=$short_branch" in routing_calibration_2254
    assert "target_iters=$full_iters" in routing_calibration_2254
    assert "decay_iters=$full_decay_iters" in routing_calibration_2254
    assert '--load "$routing_source" --override-opt_param-scheduler' in routing_calibration_2254
    assert 'learning_rate=${STAGE3_MOE_LR:-1.63e-3}' in launcher
    assert 'adam_beta2=${STAGE3_MOE_ADAM_BETA2:-0.95}' in launcher
    assert '--adam-beta1 0.9 --adam-beta2 "$adam_beta2"' in launcher
    assert '--lr "$learning_rate"' in launcher


def test_cloud_gate_preregisters_three_frugal_recipes_and_matched_slimadam():
    gate = CLOUD_GATE.read_text()

    assert "matched 1.63e-3 1.63e-4 0.95 587" in gate
    assert "efficient-training-1e3 1e-3 1e-4 0.999 587" in gate
    assert "efficient-training-2e3 2e-3 2e-4 0.999 587" in gate
    assert "slimadam_bf16_state_fp32 matched 1.63e-3 1.63e-4 0.95 587" in gate
    assert "frugal_coord_bf16_state_fp32 matched 1.63e-3 1.63e-4 0.95 235" in gate
    assert "slimadam_bf16_state_fp32 matched 1.63e-3 1.63e-4 0.95 235" in gate
    assert 'routing["minimum_to_mean_min"] < 0.1' in gate
    assert 'routing["coefficient_of_variation_max"] >= 0.2' in gate
    assert 'last["iteration"] != target' in gate
    assert 'routing["window_steps"] != 100' in gate
    assert "number of nan iterations: +0" in gate
    assert "GATE_CKPT_REMOVED" in gate


def test_routing_calibration_has_matched_adamw_control_and_prefix_schedule():
    gate = CLOUD_GATE.read_text()

    block = gate.split("  routing-calibration)", 1)[1].split(
        "\n  *) status=1 ;;\nesac", 1
    )[0]
    assert "adamw_bf16_state_fp32" in block
    assert "frugal_coord_bf16_state_fp32" in block
    assert "slimadam_bf16_state_fp32" in block
    assert (
        'run_calibration routing-calibration "$gate_arm" matched '
        "1.63e-3 1.63e-4 0.95 587" in block
    )


def test_routing_calibration_2254_resumes_then_removes_verified_source():
    gate = CLOUD_GATE.read_text()

    function = gate.split("run_routing_calibration_2254()", 1)[1].split(
        "\nstatus=0", 1
    )[0]
    assert "source_587_missing" in function
    assert "output_2254_missing" in function
    assert "successfully loaded checkpoint.*iteration +587" in function
    assert "successfully saved checkpoint from iteration +2254" in function
    assert function.index('rm -rf -- "$source"') > function.index("no_iteration_2254_save")
    assert '"$run_dir/results.jsonl" "$run_dir/routing_telemetry.jsonl" 2254' in function


def test_full_wave_accepts_and_moves_a_direct_branch_checkpoint():
    cloud = CLOUD_FULL.read_text()

    assert "STAGE3_MOE_BRANCH_CHECKPOINT_DIR" in cloud
    assert '$(cat "$source_tracker") != "$branch"' in cloud
    assert 'mv -- "$direct_source" "$dst"' in cloud
    assert 'FULL_PREFLIGHT_PASS arm=$arm checkpoint=$resume_iteration' in cloud
    assert "additional_checkpoints * checkpoint_kb + 2 * 1024 * 1024" in cloud
    assert "STAGE3_MOE_PREFLIGHT_ONLY" in cloud


def test_wgrad_image_bench_is_matched_and_checkpoint_read_only():
    launcher = LAUNCHER.read_text()
    bench = WGRAD_BENCH.read_text()

    assert 'bench_load=${STAGE3_MOE_BENCH_LOAD:-$trunk_dir}' in launcher
    assert "STAGE3_MOE_BENCH_ITERS" in bench
    assert 'for fusion in 0 1' in bench
    assert 'export STAGE3_MOE_WGRAD_FUSION=$fusion' in bench
    assert "FUSION_SPEEDUP=" in bench
    assert 'cp -al "$source_dir/$iter_dir" "$partial/"' in bench
    bench_mode = bench.split('  bench)', 1)[1].split('  cleanup)', 1)[0]
    assert "--save" not in bench_mode


def test_stability_cleanup_is_exact_and_requires_the_235_trackers():
    gate = CLOUD_GATE.read_text()

    cleanup = gate.split("if [[ $mode == cleanup-stability ]]", 1)[1].split(
        "if (( available_kb", 1
    )[0]
    assert "frugal_coord_bf16_state_fp32-stability-matched-v1" in cleanup
    assert "slimadam_bf16_state_fp32-stability-matched-v1" in cleanup
    assert "iter_0000235" in cleanup
    assert '$(cat "$tracker") != 235' in cleanup
    assert 'rm -rf -- "$path"' in cleanup


def test_lr_screen_cleanup_is_exact_and_requires_the_587_trackers():
    gate = CLOUD_GATE.read_text()

    cleanup = gate.split("if [[ $mode == cleanup-lr-screen ]]", 1)[1].split(
        "if (( available_kb", 1
    )[0]
    assert "frugal_coord_bf16_state_fp32-lr-screen-matched-v1" in cleanup
    assert "frugal_coord_bf16_state_fp32-lr-screen-efficient-training-1e3-v1" in cleanup
    assert "frugal_coord_bf16_state_fp32-lr-screen-efficient-training-2e3-v1" in cleanup
    assert "slimadam_bf16_state_fp32-lr-screen-matched-v1" in cleanup
    assert "iter_0000587" in cleanup
    assert '$(cat "$tracker") != 587' in cleanup
    assert 'rm -rf -- "$path"' in cleanup


def test_cpu_contract_exits_before_allocated_runtime_checks():
    gate = CLOUD_GATE.read_text()

    assert gate.index("if [[ $mode == cpu-contract ]]") < gate.index(
        'echo "=== ALLOCATED RUNTIME ==="'
    )


def test_gpu_smoke_requires_complete_17_by_64_routing_telemetry():
    gate = CLOUD_GATE.read_text()

    assert 'last["iteration"] != 25' in gate
    assert 'len(last["layers"]) != 17' in gate
    assert 'len(row) != 64' in gate
    assert 'run_smoke "$gate_arm"' in gate
