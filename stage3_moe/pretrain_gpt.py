import argparse
import os
import runpy
import sys
import time
import typing
from pathlib import Path


# `typing.override` is 3.12+, and the pinned MCore and Emerging Optimizers trees import it
# from `typing` in a dozen files. The torch28 image is 3.12 and never noticed; the te3
# image (torch 2.8+cu129, TE 2.5) is 3.10 and every arm died on the import before reaching
# a single step.
#
# Patching the vendored files is the obvious fix and the wrong one here: their tree hashes
# are pinned in configs/stage3-moe-1p029b.sh and verified by run_stage3_moe_probe.sh, and
# the whole point of a recipe probe is to stay comparable to the 1C arms that ran against
# those exact trees. So the backport goes here, in our own entry point, before anything
# imports them. typing_extensions.override is the same object on every version.
if not hasattr(typing, "override"):  # pragma: no cover - depends on the image's Python
    from typing_extensions import override as _override

    typing.override = _override


def _allow_cross_numpy_checkpoints():
    """Let an image read a checkpoint another image's numpy wrote.

    MCore allowlists numpy's unpickler as `from numpy.core.multiarray import _reconstruct`
    (`megatron/core/safe_globals.py`), which is the numpy 1.x path. A checkpoint written
    under numpy 2.x pickles that function as `numpy._core.multiarray._reconstruct`, so on
    an image carrying numpy 1.x the names do not match and `weights_only=True` refuses the
    load -- which is how every te3 variant died once the typing shim let it get that far.

    Allowlisting the function object does not fix it, which cost a round trip to learn:
    `add_safe_globals` derives the name from `__module__`, and numpy keeps that at
    `numpy.core.multiarray` for backward compatibility even under numpy 2.x, so the
    registered name still never matches the `numpy._core...` spelling in the pickle.

    So the default is relaxed instead -- never an explicit argument, so MCore's own
    `safe_load_from_bytes(weights_only=True)` is untouched -- and only for checkpoints
    this project wrote and published itself.
    """
    import numpy

    # Only numpy 1.x has the problem: under numpy 2.x the running spelling and the pickled
    # spelling agree, MCore's allowlist resolves, and weights_only stays True. The te4
    # image ships numpy 2.2.6, so this is a no-op there and the safety property is kept
    # wherever it can be.
    if int(numpy.__version__.split(".")[0]) >= 2:
        return

    import torch

    original_load = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load


_allow_cross_numpy_checkpoints()

PROGRAM_START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
MCORE_ROOT = ROOT / "third_party" / "Megatron-LM"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MCORE_ROOT))

from stage3_moe import ARMS


def take_stage3_args(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--stage3-arm", choices=ARMS, required=True)
    parser.add_argument("--stage3-result-path", type=Path, required=True)
    parser.add_argument("--stage3-warmup-steps", type=int, required=True)
    parser.add_argument("--stage3-measure-steps", type=int, required=True)
    parser.add_argument(
        "--optimizer-state-precision", choices=("fp32", "fp8"), required=True
    )
    # Downstream scoring runs inside the training process so the forward is the same one
    # the arm was trained with; see stage3_moe/lm_eval_mcore.py.
    parser.add_argument("--stage3-eval-downstream", default=None)
    parser.add_argument("--stage3-eval-artifact-dir", type=Path, default=None)
    parser.add_argument("--stage3-eval-batch-size", type=int, default=8)
    parser.add_argument("--stage3-eval-limit", type=int, default=None)
    parser.add_argument("--stage3-routing-audit-path", type=Path, default=None)
    args, remaining = parser.parse_known_args(argv[1:])
    if args.stage3_warmup_steps < 0 or args.stage3_measure_steps < 1:
        raise ValueError("stage3 warmup must be non-negative and measured steps positive")
    return args, [argv[0], *remaining]


def install_downstream_eval(probe, *, tasks, artifact_dir, batch_size, limit):
    """Score the loaded checkpoint the moment MCore finishes building and loading it."""
    import megatron.training.training as training

    previous = training.setup_model_and_optimizer
    probe.protocol_kind = "evaluation"

    def setup_and_evaluate(*args, **kwargs):
        model, optimizer, scheduler = previous(*args, **kwargs)
        from stage3_moe.lm_eval_mcore import run_suite

        module = model[0]
        module.eval()
        probe.downstream = run_suite(
            module,
            tasks=tasks,
            include_path=ROOT / "stage4" / "eval_tasks",
            artifact_dir=artifact_dir,
            batch_size=batch_size,
            limit=limit,
        )
        # MCore's own validation runs after this and puts the model back in train mode.
        module.eval()
        return model, optimizer, scheduler

    training.setup_model_and_optimizer = setup_and_evaluate


def option_value(argv, option):
    index = argv.index(option)
    return argv[index + 1]


def validate_axis(arm, state_precision, argv, warmup_steps, measure_steps):
    optimizer = {
        "adamw": "adam",
        "muon": "muon",
        "frugal": "frugal",
        "slimadam": "slimadam",
        "ademamix": "ademamix",
    }[arm.split("_", 1)[0]]
    if option_value(argv, "--optimizer") != optimizer:
        raise ValueError(f"{arm} requires --optimizer {optimizer}")
    # A bounded probe measures the whole run; a pretraining run measures its first
    # window and then keeps going, so the budget only has to cover that window.
    if int(option_value(argv, "--train-iters")) < warmup_steps + measure_steps:
        raise ValueError("--train-iters must cover warmup plus measured steps")
    expected_state_precision = "fp8" if arm.endswith("_state_fp8") else "fp32"
    if state_precision != expected_state_precision:
        raise ValueError(
            f"{arm} requires --optimizer-state-precision {expected_state_precision}"
        )

    compute_fp8 = "_fp8gemm_" in arm
    has_fp8 = "--fp8-format" in argv or "--fp8-recipe" in argv
    if compute_fp8:
        # The 1C arms ran hybrid/delayed and this check used to pin exactly that. Delayed
        # scaling turned out to be the wrong choice -- MCore's defaults make it
        # amax_history_len=1 / most_recent, and only the FP8-GEMM arms threw grad-norm
        # spikes of 14-29 against a 0.15 ceiling everywhere else -- so the recipe is now a
        # variable. What the check still guarantees is that an FP8-GEMM arm runs FP8 and
        # that the recipe is one we understand; the record carries which one it was.
        if option_value(argv, "--fp8-format") not in ("hybrid", "e4m3"):
            raise ValueError("FP8 GEMM arms require --fp8-format hybrid or e4m3")
        if option_value(argv, "--fp8-recipe") not in ("delayed", "tensorwise", "blockwise"):
            raise ValueError(
                "FP8 GEMM arms require --fp8-recipe delayed, tensorwise or blockwise"
            )
    elif has_fp8:
        raise ValueError("BF16-GEMM arms must not pass FP8 compute flags")
    if "--use-distributed-optimizer" in argv:
        raise ValueError("the first Stage 3 MoE probes use the non-distributed optimizer")
    if optimizer in {"frugal", "slimadam", "ademamix"}:
        checkpoint_format = (
            option_value(argv, "--ckpt-format")
            if "--ckpt-format" in argv
            else "torch_dist"
        )
        if checkpoint_format != "torch":
            raise ValueError(f"{optimizer} requires --ckpt-format torch")


def install_fp8_adamw():
    import torch
    import megatron.core.optimizer as mcore_optimizer

    from stage3_moe.optimizer_states import make_fp8_adamw

    mcore_base = mcore_optimizer.Adam
    torch_adamw_base = torch.optim.AdamW
    torch_adam_base = torch.optim.Adam
    mcore_optimizer.Adam = make_fp8_adamw(mcore_base)
    torch.optim.AdamW = (
        mcore_optimizer.Adam
        if mcore_base is torch_adamw_base
        else make_fp8_adamw(torch_adamw_base)
    )
    torch.optim.Adam = (
        mcore_optimizer.Adam
        if mcore_base is torch_adam_base
        else make_fp8_adamw(torch_adam_base)
    )


def install_coat_adamw():
    import megatron.core.optimizer as mcore_optimizer

    from stage3_moe.coat_adamw import MCoreCoatAdamW

    mcore_optimizer.Adam = MCoreCoatAdamW


def main():
    stage3_args, mcore_argv = take_stage3_args(sys.argv)
    validate_axis(
        stage3_args.stage3_arm,
        stage3_args.optimizer_state_precision,
        mcore_argv,
        stage3_args.stage3_warmup_steps,
        stage3_args.stage3_measure_steps,
    )
    sys.argv = mcore_argv

    state_fp8 = stage3_args.stage3_arm.endswith("_state_fp8")
    is_muon = stage3_args.stage3_arm.startswith("muon_")
    is_frugal = stage3_args.stage3_arm.startswith("frugal_")
    is_slimadam = stage3_args.stage3_arm.startswith("slimadam_")
    is_ademamix = stage3_args.stage3_arm.startswith("ademamix_")
    if state_fp8:
        if os.environ.get("STAGE3_MOE_ADAM_STATE_CODEC", "").strip().lower() == "coat":
            install_coat_adamw()
        else:
            install_fp8_adamw()
    if is_muon:
        from stage3_moe.muon import install_muon_contract

        install_muon_contract(fp8_states=state_fp8)
    if is_frugal:
        from stage3_moe.frugal import install_frugal_contract

        install_frugal_contract(fp8_states=state_fp8)
    if is_slimadam:
        from stage3_moe.slim_adam import install_slimadam_contract

        install_slimadam_contract()
    if is_ademamix:
        from stage3_moe.ademamix import install_ademamix_contract

        install_ademamix_contract(fp8_states=state_fp8)

    from stage3_moe.memory_audit import install as install_memory_audit
    from stage3_moe.result_writer import install_probe

    probe = install_probe(
        arm=stage3_args.stage3_arm,
        result_path=stage3_args.stage3_result_path,
        warmup_steps=stage3_args.stage3_warmup_steps,
        measured_steps=stage3_args.stage3_measure_steps,
        program_start=PROGRAM_START,
        argv=mcore_argv,
    )
    install_memory_audit()
    if os.environ.get("STAGE3_MOE_COLDSTART_RECEIPT"):
        from stage3_moe.coldstart import install_initialization_receipt
        install_initialization_receipt()

    if "_coat_" in stage3_args.stage3_arm:
        import megatron.training.training as training

        previous_coat_setup = training.setup_model_and_optimizer

        def setup_with_coat_activations(*args, **kwargs):
            models, optimizer, scheduler = previous_coat_setup(*args, **kwargs)
            from stage3_moe.coat_activation import install as install_coat_activations

            install_coat_activations(models, optimizer)
            return models, optimizer, scheduler

        training.setup_model_and_optimizer = setup_with_coat_activations
    elif os.environ.get("STAGE3_MOE_BF16_FUSED_EXPERTS") == "1":
        # BF16 on the expert path of the fused COAT flow: the control for precision alone.
        import megatron.training.training as training

        previous_bf16_setup = training.setup_model_and_optimizer

        def setup_with_bf16_fused_experts(*args, **kwargs):
            models, optimizer, scheduler = previous_bf16_setup(*args, **kwargs)
            from stage3_moe.bf16_fused_experts import install as install_bf16_fused_experts

            install_bf16_fused_experts(models, optimizer)
            return models, optimizer, scheduler

        training.setup_model_and_optimizer = setup_with_bf16_fused_experts
    if os.environ.get("STAGE3_MOE_COLD_SWITCH") == "1":
        # Optimizer switch at a checkpoint: --no-load-optim gives fresh states, but MCore then skips
        # the LR/WD scheduler too (num_steps stays 0, i.e. warmup). The schedule is meant to continue
        # from the checkpoint, so advance it; num_steps counts samples, not iterations.
        import megatron.training.training as training
        from megatron.training import get_args

        previous_cold_setup = training.setup_model_and_optimizer

        def setup_with_cold_switch(*args, **kwargs):
            models, optimizer, scheduler = previous_cold_setup(*args, **kwargs)
            mcore_args = get_args()
            if not mcore_args.no_load_optim or mcore_args.iteration <= 0:
                raise RuntimeError("STAGE3_MOE_COLD_SWITCH needs --no-load-optim and a loaded checkpoint")
            scheduler.step(increment=mcore_args.iteration * mcore_args.global_batch_size)
            print(
                f"stage3 cold switch: fresh optimizer states, schedule advanced to iteration "
                f"{mcore_args.iteration} lr={optimizer.param_groups[0]['lr']:.4e}",
                flush=True,
            )
            return models, optimizer, scheduler

        training.setup_model_and_optimizer = setup_with_cold_switch
    if os.environ.get("STAGE3_MOE_DEQUANT_ON_LOAD") == "1":
        # Warm precision switch FP8 states -> FP32 states: decode the checkpoint's FP8 entries while loading
        # (stage3_moe/optimizer_states.py). Registered on the raw optimizers before load_checkpoint runs.
        import megatron.training.training as training

        previous_get_optimizer = training.get_megatron_optimizer

        def get_optimizer_with_dequant(*args, **kwargs):
            optimizer = previous_get_optimizer(*args, **kwargs)
            from stage3_moe.optimizer_states import install_dequantize_on_load

            install_dequantize_on_load(optimizer)
            return optimizer

        training.get_megatron_optimizer = get_optimizer_with_dequant
    lr_scale = float(os.environ.get("STAGE3_MOE_LR_SCALE", "1"))
    # WSM (arXiv 2507.17634): continue at the peak LR with no decay; min_lr = max_lr makes the WSD
    # decay branch of the scheduler return max_lr.
    lr_constant = os.environ.get("STAGE3_MOE_LR_CONSTANT") == "1"
    if lr_scale != 1.0 or lr_constant:
        # LR stress / LR-on-decay runs continue from a checkpoint WITH its optimizer state, and the loaded
        # param groups carry the checkpoint's max_lr/min_lr, which the scheduler reads in preference to --lr
        # (so --lr + --override-opt_param-scheduler alone changes nothing). Scale the groups after the load.
        import megatron.training.training as training

        previous_lr_setup = training.setup_model_and_optimizer

        def setup_with_lr_scale(*args, **kwargs):
            models, optimizer, scheduler = previous_lr_setup(*args, **kwargs)
            for group in optimizer.param_groups:
                group["max_lr"] = group["max_lr"] * lr_scale
                if group.get("min_lr") is not None:
                    group["min_lr"] = group["min_lr"] * lr_scale
                if lr_constant:
                    group["min_lr"] = group["max_lr"]
            scheduler.step(increment=0)
            print(f"stage3 lr scale: x{lr_scale} constant={int(lr_constant)} -> lr={optimizer.param_groups[0]['lr']:.4e} "
                  f"max_lr={optimizer.param_groups[0]['max_lr']:.4e} min_lr={optimizer.param_groups[0]['min_lr']:.4e}", flush=True)
            return models, optimizer, scheduler

        training.setup_model_and_optimizer = setup_with_lr_scale
    activation_stats_path = os.environ.get("STAGE3_MOE_ACTIVATION_STATS_PATH")
    if activation_stats_path:
        import megatron.training.training as training

        previous_stats_setup = training.setup_model_and_optimizer

        def setup_with_activation_stats(*args, **kwargs):
            models, optimizer, scheduler = previous_stats_setup(*args, **kwargs)
            from stage3_moe.activation_stats import install as install_activation_stats

            install_activation_stats(models, activation_stats_path, int(os.environ.get("STAGE3_MOE_ACTIVATION_STATS_STEPS", "2")))
            return models, optimizer, scheduler

        training.setup_model_and_optimizer = setup_with_activation_stats
    diagnostic_path = os.environ.get("STAGE3_MOE_DIAGNOSTIC_PATH")
    replay_path = os.environ.get("STAGE3_MOE_REPLAY_PATH")
    m2_trace_path = os.environ.get("STAGE3_MOE_M2_TRACE_PATH")
    if diagnostic_path or replay_path or m2_trace_path:
        import megatron.training.training as training
        from tools.quantization_diagnostics import install_live_diagnostics, install_m2_trace

        previous_setup = training.setup_model_and_optimizer

        def setup_with_diagnostics(*args, **kwargs):
            models, optimizer, scheduler = previous_setup(*args, **kwargs)
            if diagnostic_path:
                install_live_diagnostics(optimizer, models, diagnostic_path)
            if m2_trace_path:
                steps = os.environ.get("STAGE3_MOE_M2_TRACE_STEPS", "10:20:30:50:100:200")
                install_m2_trace(optimizer, models, m2_trace_path, {int(s) for s in steps.split(":")})
            if replay_path:
                from tools.quantization_replay import install_replay
                install_replay(optimizer, models, replay_path, int(os.environ.get("STAGE3_MOE_REPLAY_STEPS", "128")))
            return models, optimizer, scheduler

        training.setup_model_and_optimizer = setup_with_diagnostics
    if os.environ.get("STAGE3_MOE_HESSIAN_PATH"):
        from stage3_moe.hessian_trace import install as install_hessian_trace

        install_hessian_trace(os.environ["STAGE3_MOE_HESSIAN_PATH"])
    if stage3_args.stage3_routing_audit_path:
        from stage3_moe.routing_audit import install as install_routing_audit

        install_routing_audit(stage3_args.stage3_routing_audit_path)
    if os.environ.get("STAGE3_MOE_FULL_DEV_EVAL") == "1":
        # Eval-only full-development re-evaluation (baseline-eval-030): mark the record as an
        # evaluation record and passively capture the evaluator's exact token denominator and
        # per-iteration losses, which the stock print path does not expose. No math change.
        probe.protocol_kind = "evaluation"
        from stage3_moe.full_dev_capture import install as install_full_dev_capture
        install_full_dev_capture(os.environ["STAGE3_MOE_FULL_DEV_CAPTURE_PATH"])
    if stage3_args.stage3_eval_downstream:
        install_downstream_eval(
            probe,
            tasks=[task for task in stage3_args.stage3_eval_downstream.split(",") if task],
            artifact_dir=stage3_args.stage3_eval_artifact_dir,
            batch_size=stage3_args.stage3_eval_batch_size,
            limit=stage3_args.stage3_eval_limit,
        )
    print(
        f"stage3 MoE arm={stage3_args.stage3_arm} "
        f"warmup={stage3_args.stage3_warmup_steps} "
        f"measured={stage3_args.stage3_measure_steps}",
        flush=True,
    )
    runpy.run_path(str(MCORE_ROOT / "pretrain_gpt.py"), run_name="__main__")


if __name__ == "__main__":
    main()
