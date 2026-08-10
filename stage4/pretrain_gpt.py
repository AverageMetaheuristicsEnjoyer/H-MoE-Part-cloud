import argparse
import atexit
import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MCORE_ROOT = ROOT / "third_party" / "Megatron-LM"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MCORE_ROOT))


def take_state_precision():
    flag = "--optimizer-state-precision"
    index = sys.argv.index(flag)
    precision = sys.argv[index + 1]
    del sys.argv[index : index + 2]
    if precision not in {"fp32", "fp8"}:
        raise ValueError(f"unsupported optimizer-state precision: {precision}")
    return precision


precision = take_state_precision()

import megatron.core.optimizer.emerging_optimizers as mcore_eopt
import megatron.training.arguments as mcore_arguments

# MCore registers SOAP through the generic fallback path, which leaves three gaps.
# Without the param overrides the tied 50,304x1,536 embedding is handed to SOAP,
# whose Kronecker factor and eigenbasis for that parameter are both 50,304x50,304
# (~10 GB each); SOAP's `eps` has no OptimizerConfig field, so `--adam-eps` never
# reaches it; and the three `soap_*` config fields have no command-line flags.


def _soap_config_to_kwargs(config, model_chunks, pg_collection):
    kwargs = mcore_eopt._default_adam_based_eopt_config_to_kwargs(
        "soap", config, model_chunks, pg_collection
    )
    kwargs["eps"] = config.adam_eps
    return kwargs


soap_entry = mcore_eopt._EMERGING_OPTIMIZERS["soap"]
soap_entry.config_to_kwargs = _soap_config_to_kwargs
soap_entry.default_param_overrides = mcore_eopt._default_param_overrides_factory()


def add_soap_args(parser):
    group = parser.add_argument_group(title="stage4 soap")
    group.add_argument("--soap-shampoo-beta", type=float, default=0.95)
    group.add_argument("--soap-precondition-frequency", type=int, default=1)
    group.add_argument(
        "--soap-use-kl-shampoo", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


mcore_parse_and_validate_args = mcore_arguments.parse_and_validate_args


def parse_and_validate_args(extra_args_provider=None, **kwargs):
    def provider(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        return add_soap_args(parser)

    return mcore_parse_and_validate_args(provider, **kwargs)


mcore_arguments.parse_and_validate_args = parse_and_validate_args

if precision == "fp8":
    import megatron.core.optimizer as mcore_optimizer
    from megatron.core.optimizer.emerging_optimizers import (
        TensorParallelMuon,
        _EMERGING_OPTIMIZERS,
    )

    from stage4.fp8_optimizer_states import (
        FP8StateOptimizerMixin,
        make_fp8_adamw,
        make_fp8_soap,
    )

    class FP8StateTensorParallelMuon(FP8StateOptimizerMixin, TensorParallelMuon):
        state_specs = (("momentum_buffer", True),)

    mcore_optimizer.Adam = make_fp8_adamw(mcore_optimizer.Adam)
    _EMERGING_OPTIMIZERS["muon"].optimizer_cls = FP8StateTensorParallelMuon
    _EMERGING_OPTIMIZERS["soap"].optimizer_cls = make_fp8_soap(
        _EMERGING_OPTIMIZERS["soap"].optimizer_cls
    )

print(f"stage4 optimizer-state precision: {precision}", flush=True)

if os.getenv("STAGE4_REPORT_PEAK_MEMORY") == "1":
    def report_peak_memory():
        import torch

        print(
            "stage4 peak memory bytes: "
            f"allocated={torch.cuda.max_memory_allocated()} "
            f"reserved={torch.cuda.max_memory_reserved()}",
            flush=True,
        )

    atexit.register(report_peak_memory)

runpy.run_path(str(MCORE_ROOT / "pretrain_gpt.py"), run_name="__main__")
