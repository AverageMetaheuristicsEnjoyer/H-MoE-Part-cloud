import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).parents[2]
LAUNCHER = ROOT / "scripts" / "run_stage3_moe_probe.sh"
CONFIG = ROOT / "configs" / "stage3-moe-1p029b.sh"
CLOUD = ROOT / "scripts" / "cloud_moe_fp8_delayed_smoke.sh"

ARMS = (
    "adamw_bf16_state_fp32",
    "adamw_bf16_state_fp8",
    "muon_bf16_state_fp32",
    "muon_bf16_state_fp8",
    "adamw_fp8gemm_state_fp32",
    "muon_fp8gemm_state_fp32",
)


def dry_run(arm, *extra, env_overrides=None):
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="0",
        STAGE3_MOE_DRY_RUN="1",
        STAGE3_MOE_PYTHON="/bin/false",
        STAGE3_MOE_RUN_ID="contract-test",
    )
    env.update(env_overrides or {})
    completed = subprocess.run(
        [str(LAUNCHER), arm, "mock", *extra],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    command_line = next(
        line.removeprefix("COMMAND=")
        for line in completed.stdout.splitlines()
        if line.startswith("COMMAND=")
    )
    return completed.stdout, shlex.split(command_line)


def value(args, flag):
    return args[args.index(flag) + 1]


def test_exact_architecture_and_forbidden_moe_flags():
    _, args = dry_run("adamw_bf16_state_fp32")

    expected = {
        "--num-layers": "18",
        "--hidden-size": "1024",
        "--ffn-hidden-size": "2816",
        "--moe-ffn-hidden-size": "256",
        "--num-attention-heads": "8",
        "--num-query-groups": "2",
        "--num-experts": "64",
        "--moe-layer-freq": "[0]+[1]*17",
        "--moe-shared-expert-intermediate-size": "256",
        "--moe-router-topk": "8",
        "--moe-router-score-function": "sigmoid",
        "--moe-token-dispatcher-type": "alltoall",
    }
    assert all(value(args, flag) == expected_value for flag, expected_value in expected.items())
    assert "--moe-grouped-gemm" in args
    assert "--moe-router-enable-expert-bias" in args
    assert "--moe-router-pre-softmax" in args
    assert "--moe-router-padding-for-fp8" not in args
    assert "--moe-router-padding-for-quantization" not in args
    assert "--moe-single-grouped-weight" not in args


def test_six_arms_keep_state_and_compute_axes_separate():
    commands = {arm: dry_run(arm)[1] for arm in ARMS}

    for arm in ARMS[:4]:
        assert "--fp8-format" not in commands[arm]
    for arm in ARMS[4:]:
        assert value(commands[arm], "--fp8-format") == "hybrid"
        assert value(commands[arm], "--fp8-recipe") == "delayed"
        assert value(commands[arm], "--optimizer-state-precision") == "fp32"

    assert value(commands["adamw_bf16_state_fp32"], "--optimizer-state-precision") == "fp32"
    assert value(commands["adamw_bf16_state_fp8"], "--optimizer-state-precision") == "fp8"
    assert value(commands["muon_bf16_state_fp32"], "--optimizer-state-precision") == "fp32"
    assert value(commands["muon_bf16_state_fp8"], "--optimizer-state-precision") == "fp8"
    assert value(commands["muon_bf16_state_fp32"], "--optimizer") == "muon"
    assert "--muon-nesterov" in commands["muon_bf16_state_fp32"]


def test_fp8_amax_overrides_are_opt_in_and_fp8_only():
    overrides = {
        "STAGE3_MOE_FP8_AMAX_HISTORY_LEN": "16",
        "STAGE3_MOE_FP8_AMAX_COMPUTE_ALGO": "max",
    }
    bf16 = dry_run("adamw_bf16_state_fp32", env_overrides=overrides)[1]
    fp8 = dry_run("adamw_fp8gemm_state_fp32", env_overrides=overrides)[1]

    assert "--fp8-amax-history-len" not in bf16
    assert "--fp8-amax-compute-algo" not in bf16
    assert value(fp8, "--fp8-amax-history-len") == "16"
    assert value(fp8, "--fp8-amax-compute-algo") == "max"


def test_probe_and_smoke_step_contract_and_denominators():
    output, probe = dry_run("adamw_bf16_state_fp32")
    _, smoke = dry_run("adamw_fp8gemm_state_fp32", "--protocol", "smoke")

    assert value(probe, "--stage3-warmup-steps") == "3"
    assert value(probe, "--stage3-measure-steps") == "5"
    assert value(probe, "--train-iters") == "8"
    assert value(smoke, "--stage3-warmup-steps") == "0"
    assert value(smoke, "--stage3-measure-steps") == "1"
    assert value(smoke, "--train-iters") == "1"
    assert "gpu=1 micro_batch=1 global_batch=1 sequence_length=2048" in output
    assert "total_parameters=1028926976 active_parameters=280243712" in output
    assert value(probe, "--stage3-result-path").startswith(
        "artifacts/stage3-moe-probes/contract-test/results.jsonl.inprocess."
    )
    assert "--split" in probe
    assert value(probe, "--split") == "100,0,0"


def test_gradient_accumulation_fusion_falls_back_only_when_import_is_absent():
    output, args = dry_run("adamw_bf16_state_fp32")

    assert "GRADIENT_ACCUMULATION_FUSION=disabled" in output
    assert "--no-gradient-accumulation-fusion" in args
    launcher = LAUNCHER.read_text()
    assert "import fused_weight_gradient_mlp_cuda" in launcher
    assert "grad_accum_fusion=enabled" in launcher


def test_config_counts_and_cloud_delayed_hybrid_route_are_pinned():
    config = CONFIG.read_text()
    cloud = CLOUD.read_text()

    assert "STAGE3_MOE_TOTAL_PARAMETERS=1028926976" in config
    assert "STAGE3_MOE_ACTIVE_PARAMETERS=280243712" in config
    assert "STAGE3_MOE_MCORE_COMMIT=571370c829ca768fe37244f4e2e7f28d8accc4ab" in config
    assert "STAGE3_MOE_VENDORED_MCORE_TREE=e56265e78f086c1ff831ed40c30e50395e236a83" in config
    assert "STAGE3_MOE_EO_COMMIT=1effa026ff096b7fa1063ca2fba19d98be6e6cdf" in config
    assert "STAGE3_MOE_VENDORED_EO_TREE=e6b6cfd986bc0af4cd4f8e2c4ebedad16144e856" in config
    assert "MLSUB_IMAGE:-} != torch28" in cloud
    assert "DelayedScaling" in cloud
    assert "Format.HYBRID" in cloud
    assert "adamw_fp8gemm_state_fp32 mock --protocol smoke" in cloud
    assert "Float8BlockScaling" not in cloud
    assert "mxfp8" not in cloud.lower()
    assert "uuid,compute_cap" in cloud
    assert "/home/jovyan/hmoe-cloud/artifacts/stage3-moe-probes/" in cloud


def test_node207_route_uses_one_runtime_for_preflight_and_training():
    launcher = LAUNCHER.read_text()

    assert 'runtime_prefix=(env)' in launcher
    assert '"$root/scripts/node207_env.sh" env' in launcher
    assert '"${runtime_prefix[@]}" "$python_bin" -c' in launcher
    assert 'nvidia-smi --query-compute-apps=gpu_uuid,pid' in launcher
    assert "active compute PIDs" in launcher
    assert "idle memory threshold exceeded" in launcher
    assert '"STAGE3_MOE_MATCH_KEY_SHA256=$match_key_sha256"' in launcher
    assert '"STAGE3_MOE_CONFIG_SHA256=$config_sha256"' in launcher
    assert '"STAGE3_MOE_DATA_MANIFEST_SHA256=$data_manifest_sha256"' in launcher
    assert 'git -C "$root/third_party/Megatron-LM" rev-parse HEAD' in launcher
    assert '"$mcore_commit" != "$STAGE3_MOE_MCORE_COMMIT"' in launcher
    assert 'TRITON_CACHE_DIR=/tmp/triton-stage3' in launcher
    assert 'git -C "$root" rev-parse HEAD:third_party/Megatron-LM' in launcher
    assert '"$vendored_mcore_tree" != "$STAGE3_MOE_VENDORED_MCORE_TREE"' in launcher
    assert 'git -C "$root" rev-parse HEAD:third_party/emerging-optimizers' in launcher
    assert '"$vendored_eo_tree" != "$STAGE3_MOE_VENDORED_EO_TREE"' in launcher


def test_outer_launcher_finalizes_wct_and_gpu_postflight():
    launcher = LAUNCHER.read_text()

    assert "started = time.perf_counter()" in launcher
    assert "completed = subprocess.run(sys.argv[2:])" in launcher
    assert 'read -r training_code e2e_wct_seconds' in launcher
    assert 'SCOPE=launcher_start_to_process_exit' in launcher
    assert '"e2e_wct_scope"] = "launcher_start_to_process_exit"' in launcher
    assert '"gpu_clean"]["after"] = sys.argv[4] == "1"' in launcher
    assert 'GPU_POSTFLIGHT' in launcher
    assert 'exec "${cmd[@]}"' not in launcher
