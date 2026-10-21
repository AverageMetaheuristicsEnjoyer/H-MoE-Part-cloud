"""Check-only guard (copied as sitecustomize.py into the pod's copy of the repository root, which the launcher puts on PYTHONPATH): with STAGE3_MOE_FP8_BACKEND=emulated, any native FP8
GEMM (torch._scaled_mm) or Triton kernel that receives an FP8 tensor raises, so an emulated run on a Hopper GPU shows
that the emulated path never needs native FP8 support."""
import os

if os.environ.get("STAGE3_MOE_FP8_BACKEND") == "emulated":
    import torch
    import triton.runtime.jit as jit

    FP8 = {torch.float8_e4m3fn, torch.float8_e5m2}

    def _scaled_mm(*args, **kwargs):
        raise RuntimeError("EMU_GUARD: torch._scaled_mm called in emulated mode")

    torch._scaled_mm = _scaled_mm
    _run = jit.JITFunction.run

    def run(self, *args, **kwargs):
        for value in (*args, *kwargs.values()):
            if isinstance(value, torch.Tensor) and value.dtype in FP8:
                raise RuntimeError(f"EMU_GUARD: Triton kernel {self.fn.__name__} received an FP8 tensor in emulated mode")
        return _run(self, *args, **kwargs)

    jit.JITFunction.run = run
    import sys

    print(f"EMU_GUARD_ACTIVE pid={os.getpid()}", file=sys.stderr, flush=True)
