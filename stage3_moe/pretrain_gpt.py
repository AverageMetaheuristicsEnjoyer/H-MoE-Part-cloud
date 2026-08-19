import argparse
import runpy
import sys
import time
from pathlib import Path


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
    args, remaining = parser.parse_known_args(argv[1:])
    if args.stage3_warmup_steps < 0 or args.stage3_measure_steps < 1:
        raise ValueError("stage3 warmup must be non-negative and measured steps positive")
    return args, [argv[0], *remaining]


def install_downstream_eval(*, tasks, artifact_dir, batch_size, limit):
    """Score the loaded checkpoint the moment MCore finishes building and loading it."""
    import megatron.training.training as training

    previous = training.setup_model_and_optimizer

    def setup_and_evaluate(*args, **kwargs):
        model, optimizer, scheduler = previous(*args, **kwargs)
        from stage3_moe.lm_eval_mcore import run_suite

        module = model[0]
        module.eval()
        run_suite(
            module,
            tasks=tasks,
            include_path=ROOT / "stage4" / "eval_tasks",
            artifact_dir=artifact_dir,
            batch_size=batch_size,
            limit=limit,
        )
        return model, optimizer, scheduler

    training.setup_model_and_optimizer = setup_and_evaluate


def option_value(argv, option):
    index = argv.index(option)
    return argv[index + 1]


def validate_axis(arm, state_precision, argv, warmup_steps, measure_steps):
    optimizer = "muon" if arm.startswith("muon_") else "adam"
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
        if option_value(argv, "--fp8-format") != "hybrid":
            raise ValueError("FP8 GEMM arms require --fp8-format hybrid")
        if option_value(argv, "--fp8-recipe") != "delayed":
            raise ValueError("FP8 GEMM arms require --fp8-recipe delayed")
    elif has_fp8:
        raise ValueError("BF16-GEMM arms must not pass FP8 compute flags")
    if "--use-distributed-optimizer" in argv:
        raise ValueError("the first Stage 3 MoE probes use the non-distributed optimizer")


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
    if state_fp8:
        install_fp8_adamw()
    if is_muon:
        from stage3_moe.muon import install_muon_contract

        install_muon_contract(fp8_states=state_fp8)

    from stage3_moe.memory_audit import install as install_memory_audit
    from stage3_moe.result_writer import install_probe

    install_probe(
        arm=stage3_args.stage3_arm,
        result_path=stage3_args.stage3_result_path,
        warmup_steps=stage3_args.stage3_warmup_steps,
        measured_steps=stage3_args.stage3_measure_steps,
        program_start=PROGRAM_START,
        argv=mcore_argv,
    )
    install_memory_audit()
    if stage3_args.stage3_eval_downstream:
        install_downstream_eval(
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
