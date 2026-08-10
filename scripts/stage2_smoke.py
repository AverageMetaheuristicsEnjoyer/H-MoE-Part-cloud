import os
import sys

if os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES") != "1":
    raise RuntimeError("set NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1 before starting Python")

import torch
import transformer_engine as transformer_engine
import transformer_engine.pytorch as te
import transformer_engine_torch
from emerging_optimizers import __version__ as emerging_optimizers_version
from emerging_optimizers.orthogonalized_optimizers import Muon
import megatron.core as megatron_core
from megatron.core.optimizer.emerging_optimizers import HAVE_EMERGING_OPTIMIZERS
from transformer_engine.common.recipe import Float8BlockScaling, Format


def main() -> None:
    if not __debug__:
        raise RuntimeError("run smoke tests without Python optimization")

    assert sys.version_info[:2] == (3, 12), sys.version
    assert torch.__version__.startswith("2.12.0a0+0291f960b6"), torch.__version__
    assert torch.version.cuda is not None and torch.version.cuda.startswith("13.2"), torch.version.cuda
    assert megatron_core.__version__.startswith("0.18.2"), megatron_core.__version__
    assert transformer_engine.__version__ == "2.16.0+b9d690e", transformer_engine.__version__
    assert emerging_optimizers_version == "0.2.0", emerging_optimizers_version
    assert HAVE_EMERGING_OPTIMIZERS

    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    capability = torch.cuda.get_device_capability(0)
    assert capability == (9, 0), capability

    cublaslt_version = transformer_engine_torch.get_cublasLt_version()
    assert cublaslt_version >= 130400, cublaslt_version

    block_fp8_available, block_fp8_reason = te.is_fp8_block_scaling_available(return_reason=True)
    assert block_fp8_available, block_fp8_reason
    recipe = Float8BlockScaling(fp8_format=Format.E4M3)
    assert recipe.fp8_quant_fwd_inp.power_2_scale is False
    assert recipe.fp8_quant_fwd_weight.power_2_scale is False
    assert recipe.fp8_quant_bwd_grad.power_2_scale is False

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    linear = te.Linear(
        256,
        256,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
    )
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with te.autocast(enabled=True, recipe=recipe):
        y = linear(x)
    y.float().square().mean().backward()
    assert y.dtype == torch.bfloat16, y.dtype
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert linear.weight.grad is not None and torch.isfinite(linear.weight.grad).all()
    assert torch.isfinite(y).all()

    grouped = te.GroupedLinear(
        2,
        256,
        256,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
    )
    grouped_x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    splits = torch.tensor([128, 128], dtype=torch.int64)
    with te.autocast(enabled=True, recipe=recipe):
        grouped_y = grouped(grouped_x, splits)
    grouped_y.float().square().mean().backward()
    assert grouped_y.dtype == torch.bfloat16, grouped_y.dtype
    assert torch.isfinite(grouped_y).all()
    assert grouped_x.grad is not None and torch.isfinite(grouped_x.grad).all()
    assert grouped.weight0.grad is not None and torch.isfinite(grouped.weight0.grad).all()
    assert grouped.weight1.grad is not None and torch.isfinite(grouped.weight1.grad).all()

    parameter = torch.nn.Parameter(torch.randn(256, 256, device="cuda", dtype=torch.float32))
    parameter_before = parameter.detach().clone()
    parameter.grad = torch.randn_like(parameter).mul_(0.01)
    optimizer = Muon(
        [parameter],
        lr=1e-3,
        momentum=0.95,
        weight_decay=0.1,
        nesterov=True,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=0.2,
        fp32_matmul_prec="medium",
        use_syrk=False,
    )
    optimizer.step()
    momentum = optimizer.state[parameter]["momentum_buffer"]
    assert momentum.dtype == torch.float32, momentum.dtype
    assert momentum.shape == parameter.shape
    assert torch.isfinite(momentum).all()
    assert torch.isfinite(parameter).all()
    assert not torch.equal(parameter, parameter_before)

    print(
        "versions=pass"
        f" python={sys.version_info.major}.{sys.version_info.minor}"
        f" torch={torch.__version__}"
        f" cuda={torch.version.cuda}"
        f" mcore={megatron_core.__version__}"
        f" te={transformer_engine.__version__}"
        f" eo={emerging_optimizers_version}"
        f" sm={capability[0]}{capability[1]}"
        f" cublaslt={cublaslt_version}"
    )
    print("fp8_linear=pass recipe=block_scaling fp32_scales=true")
    print("grouped_linear=pass groups=2")
    print("muon=pass momentum_buffer=fp32 fp32_matmul_prec=medium")
    print("stage2_smoke=pass")


if __name__ == "__main__":
    main()
