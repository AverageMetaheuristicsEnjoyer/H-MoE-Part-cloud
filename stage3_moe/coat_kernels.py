"""The official COAT optimizer kernel, compiled with NVRTC because the GPU image has no nvcc.

`third_party/coat` is NVlabs/COAT@80ec99f byte for byte. The device code of
`fp8_adamw_expand_cuda_kernel.cu` is compiled unchanged: only the host includes of that
translation unit (torch/extension.h, cuda runtime, stdio -- NVRTC compiles device code only,
and the kernel uses none of them) and the host wrapper `FP8_AdamW_expand_cuda` are left out.
`fp8_adamw_expand_step` replaces that wrapper with the same checks and launch geometry.

Three kernels are ours, and each reuses upstream text instead of restating it:
  - `coat_import_fp32_kernel` quantizes FP32 moments into COAT's format. Its quantization is
    the upstream kernel's own tail, cut from the file; only the prologue that would have
    dequantized and updated the states is replaced by loading the FP32 values.
  - `coat_momentum_dequantize_kernel` / `coat_momentum_quantize_kernel` give Muon's
    momentum the same storage (upstream has no Muon). They are the first-order halves of the
    upstream dequantize and quantize code, and the tests check them bit for bit against the
    upstream path.
"""
import ctypes
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_FILE = ROOT / "third_party/coat/coat/optimizer/kernels/fp8_adamw_expand_cuda_kernel.cu"
STEP_KERNEL = "fp8_adamw_cuda_expand_kernel<float>"
IMPORT_KERNEL = "coat_import_fp32_kernel"
MOMENTUM_DEQUANTIZE_KERNEL = "coat_momentum_dequantize_kernel"
MOMENTUM_QUANTIZE_KERNEL = "coat_momentum_quantize_kernel"
HOST_INCLUDES = (
    "#include <torch/extension.h>",
    "#include <cuda.h>",
    "#include <cuda_runtime.h>",
    "#include <cuda_runtime_api.h>",
    "#include <stdio.h>",
)
HOST_WRAPPER = "\nvoid FP8_AdamW_expand_cuda("
QUANTIZE_TAIL = "    //// quantize the first-order and second-order momentum"

IMPORT_PROLOGUE = r"""
extern "C" __global__ void coat_import_fp32_kernel(
    const float * __restrict__ src_exp_avg,
    const float * __restrict__ src_exp_avg_sq,
    __nv_fp8_e4m3 * __restrict__ exp_avg,
    float * __restrict__ scale_exp_avg,
    float * __restrict__ expand_exp_avg,
    float * __restrict__ sqrtminmax_exp_avg,
    __nv_fp8_e4m3 * __restrict__ exp_avg_sq,
    float * __restrict__ scale_exp_avg_sq,
    float * __restrict__ expand_exp_avg_sq,
    float * __restrict__ sqrtminmax_exp_avg_sq,
    int expand_min, int total_elements, int total_scale_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int scale_idx = blockIdx.x;

    float float_exp_avg, float_exp_avg_sq;

    if (idx < total_elements) {
        float_exp_avg = src_exp_avg[idx];
        float_exp_avg_sq = src_exp_avg_sq[idx];
    } else {
        float_exp_avg = 0.0f;
        float_exp_avg_sq = 0.0f;
    }

"""

# First-order halves of the upstream kernel, statement for statement: the dequantize lines of
# its prologue and the max/min reductions and quantization of its tail, with every
# second-order statement dropped.
MOMENTUM_SOURCE = r"""
extern "C" __global__ void coat_momentum_dequantize_kernel(
    const __nv_fp8_e4m3 * __restrict__ exp_avg,
    const float * __restrict__ scale_exp_avg,
    const float * __restrict__ expand_exp_avg,
    const float * __restrict__ sqrtminmax_exp_avg,
    float * __restrict__ out,
    int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int scale_idx = blockIdx.x;
    if (idx < total_elements) {
        float float_exp_avg = float(exp_avg[idx]) * scale_exp_avg[scale_idx];
        int sign_exp_avg = 1 - 2 * signbit(float_exp_avg);
        float_exp_avg = sign_exp_avg * powf(fabsf(float_exp_avg), 1 / expand_exp_avg[scale_idx]) * sqrtminmax_exp_avg[scale_idx];
        out[idx] = float_exp_avg;
    }
}

extern "C" __global__ void coat_momentum_quantize_kernel(
    const float * __restrict__ src,
    __nv_fp8_e4m3 * __restrict__ exp_avg,
    float * __restrict__ scale_exp_avg,
    float * __restrict__ expand_exp_avg,
    float * __restrict__ sqrtminmax_exp_avg,
    int expand_min, int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int scale_idx = blockIdx.x;
    float float_exp_avg = (idx < total_elements) ? src[idx] : 0.0f;

    int wid = threadIdx.x / WARPSIZE;
    __shared__ float sharedFirstMaxVal[32];
    __shared__ float sharedFirstMinVal[32];
    cg::thread_block_tile<32> warpTile = cg::tiled_partition<32>(cg::this_thread_block());
    float firstMaxVal = fabsf(float_exp_avg);
    float firstMinVal = fabsf(float_exp_avg);
    if (idx >= total_elements) {
        firstMinVal = __int_as_float(0x7f7fffff);
    }
    for (int i = warpTile.size() / 2; i > 0; i /= 2) {
        float reduceFirstMaxVal = warpTile.shfl_down(firstMaxVal, i);
        float reduceFirstMinVal = warpTile.shfl_down(firstMinVal, i);
        firstMaxVal = fmax(firstMaxVal, fabsf(reduceFirstMaxVal));
        float absReduceFirstMinVal = fabsf(reduceFirstMinVal);
        firstMinVal = (absReduceFirstMinVal > 0) ? fmin(firstMinVal, absReduceFirstMinVal) : firstMinVal;
    }
    int lane = warpTile.thread_rank();
    if (lane == 0) {
        sharedFirstMaxVal[wid] = firstMaxVal;
        sharedFirstMinVal[wid] = firstMinVal;
    }
    __syncthreads();
    __shared__ float shared_absmax_exp_avg;
    __shared__ float shared_absmin_exp_avg;
    firstMaxVal = (threadIdx.x < blockDim.x / warpSize) ? sharedFirstMaxVal[lane] : 0;
    firstMinVal = (threadIdx.x < blockDim.x / warpSize) ? sharedFirstMinVal[lane] : 1e9;
    if (wid == 0) {
        for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {
            float reduceFirstMaxVal = __shfl_down_sync(0xFFFFFFFF, firstMaxVal, offset);
            float reduceFirstMinVal = __shfl_down_sync(0xFFFFFFFF, firstMinVal, offset);
            firstMaxVal = fmax(firstMaxVal, fabsf(reduceFirstMaxVal));
            float absReduceFirstMinVal = fabsf(reduceFirstMinVal);
            firstMinVal = (absReduceFirstMinVal > 0) ? fmin(firstMinVal, absReduceFirstMinVal) : firstMinVal;
        }
        if (lane == 0) {
            shared_absmax_exp_avg = firstMaxVal;
            shared_absmin_exp_avg = firstMinVal;
        }
    }
    __syncthreads();
    if (idx < total_elements) {
        float fp8MaxVal = 448;
        firstMaxVal = shared_absmax_exp_avg + QUANT_MIN_VAL;
        firstMinVal = shared_absmin_exp_avg + QUANT_MIN_VAL;
        float firstRatio = firstMaxVal / firstMinVal;
        float firstSqrtMinMax = sqrt(firstMaxVal) * sqrt(firstMinVal);
        float ratioUpperBound = fp8MaxVal * fp8MaxVal / 2;
        float firstExp = floor((log2f(ratioUpperBound) / log2f(firstRatio)) * expand_min) / expand_min;
        int sign_exp_avg = 1 - 2 * signbit(float_exp_avg);
        float_exp_avg = sign_exp_avg * powf(fabsf(float_exp_avg) / firstSqrtMinMax, firstExp);
        float new_scale_exp_avg = powf(firstMaxVal / firstSqrtMinMax, firstExp) / fp8MaxVal;
        __nv_fp8_e4m3 exp_avg_new = static_cast<__nv_fp8_e4m3>(float_exp_avg / new_scale_exp_avg);
        exp_avg[idx] = exp_avg_new;
        scale_exp_avg[scale_idx] = new_scale_exp_avg;
        expand_exp_avg[scale_idx] = firstExp;
        sqrtminmax_exp_avg[scale_idx] = firstSqrtMinMax;
    }
}
"""


def device_source(text=None):
    """Upstream device code plus our three kernels, ready for NVRTC."""
    text = KERNEL_FILE.read_text() if text is None else text
    if text.count(HOST_WRAPPER) != 1 or text.count(QUANTIZE_TAIL) != 1:
        raise RuntimeError("upstream kernel layout changed; refusing to cut it")
    device = text[: text.index(HOST_WRAPPER)].rstrip() + "\n"
    lines = device.splitlines(keepends=True)
    for include in HOST_INCLUDES:
        if sum(line.strip() == include for line in lines) != 1:
            raise RuntimeError(f"expected exactly one {include!r} in the upstream kernel")
    device = "".join(line for line in lines if line.strip() not in HOST_INCLUDES)
    tail = device[device.index(QUANTIZE_TAIL):].rstrip()
    if not tail.endswith("}"):
        raise RuntimeError("upstream kernel does not end where expected")
    return device + IMPORT_PROLOGUE + tail + "\n" + MOMENTUM_SOURCE


# The te4 image ships the pip CUDA wheels under its conda site-packages; a process whose `nvidia` namespace does not
# reach them (2026-10-08: the standalone tools and the FP32-state trainer) looks through every site directory and,
# failing that, installs the two wheels it needs into /tmp (as _cccl_include does for CCCL).
_NVIDIA_WHEELS = {"cuda_nvrtc": "nvidia-cuda-nvrtc-cu12==12.9.86", "cuda_runtime": "nvidia-cuda-runtime-cu12==12.9.79"}


def _nvidia_roots():
    import site
    import sys

    spec = importlib.util.find_spec("nvidia")
    roots = [Path(base) for base in ((spec.submodule_search_locations or []) if spec else [])]
    candidates = [Path(p) / "nvidia" for p in sys.path if p]
    candidates += [Path(p) / "nvidia" for p in site.getsitepackages() + [site.getusersitepackages()]]
    candidates += sorted(Path("/").glob("home/user/conda/lib/python3*/site-packages/nvidia"))
    candidates += sorted(Path("/home/jovyan").glob(".local*/lib/python3*/site-packages/nvidia"))
    candidates.append(Path("/tmp/coat-cuda/nvidia"))
    for candidate in candidates:
        if candidate.is_dir() and candidate not in roots:
            roots.append(candidate)
    return roots


def _nvidia_dir(name):
    for base in _nvidia_roots():
        candidate = base / name
        if candidate.is_dir():
            return candidate
    if name in _NVIDIA_WHEELS:
        import subprocess
        import sys

        target = Path("/tmp/coat-cuda")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps",
                        "--only-binary=:all:", "--target", str(target), _NVIDIA_WHEELS[name]], check=True)
        candidate = target / "nvidia" / name
        if candidate.is_dir():
            print(f"COAT_KERNELS_WHEEL installed {_NVIDIA_WHEELS[name]} into {target}", flush=True)
            return candidate
    raise RuntimeError(f"nvidia/{name} not found in the Python environment")


def _cccl_include():
    """cooperative_groups needs CCCL's <nv/target>, which torch's CUDA wheels do not ship."""
    try:
        return _nvidia_dir("cuda_cccl") / "include"
    except RuntimeError:
        pass
    import subprocess
    import sys

    target = Path("/tmp/coat-cccl")
    include = target / "nvidia/cuda_cccl/include"
    if not include.is_dir():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps",
                        "--only-binary=:all:", "--target", str(target), "nvidia-cuda-cccl-cu12==12.9.27"], check=True)
    return include


def compile_cubin(arch, *, nvrtc_path=None, include_dirs=None, source_text=None):
    """Compile to SASS for `arch` (e.g. 90); returns (cubin bytes, {name: lowered name}, log).

    source_text: an edited copy of the upstream kernel file (tools/dre_cost_bench.py ablations); default = as shipped."""
    nvrtc_path = nvrtc_path or sorted(_nvidia_dir("cuda_nvrtc").glob("lib/libnvrtc.so*"))[0]
    include_dirs = include_dirs or [_nvidia_dir("cuda_runtime") / "include", _cccl_include()]
    nvrtc = ctypes.CDLL(str(nvrtc_path))

    def check(result):
        if result != 0:
            nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError("NVRTC: " + nvrtc.nvrtcGetErrorString(result).decode())

    source = device_source(source_text).encode()
    program = ctypes.c_void_p()
    check(nvrtc.nvrtcCreateProgram(ctypes.byref(program), source, b"fp8_adamw_expand_cuda_kernel.cu", 0, None, None))
    names = (STEP_KERNEL, IMPORT_KERNEL, MOMENTUM_DEQUANTIZE_KERNEL, MOMENTUM_QUANTIZE_KERNEL)
    for name in names:
        check(nvrtc.nvrtcAddNameExpression(program, name.encode()))
    # nvcc's device defaults (-O3, IEEE div/sqrt, no FTZ, FMA contraction) are NVRTC's too.
    options = [f"--gpu-architecture=sm_{arch}".encode(), b"-std=c++17", *(f"-I{d}".encode() for d in include_dirs)]
    result = nvrtc.nvrtcCompileProgram(program, len(options), (ctypes.c_char_p * len(options))(*options))
    size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(program, log)
    if result != 0:
        raise RuntimeError("COAT kernel compilation failed:\n" + log.value.decode(errors="replace"))
    lowered = {}
    for name in names:
        pointer = ctypes.c_char_p()
        check(nvrtc.nvrtcGetLoweredName(program, name.encode(), ctypes.byref(pointer)))
        lowered[name] = pointer.value.decode()
    check(nvrtc.nvrtcGetCUBINSize(program, ctypes.byref(size)))
    cubin = ctypes.create_string_buffer(size.value)
    check(nvrtc.nvrtcGetCUBIN(program, cubin))
    nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
    return cubin.raw, lowered, log.value.decode(errors="replace")


class _Kernels:
    """Per-device module of the compiled kernels, launched on torch's current stream."""

    def __init__(self, device, source_text=None):
        import torch

        self.torch = torch
        major, minor = torch.cuda.get_device_capability(device)
        cubin, lowered, _ = compile_cubin(f"{major}{minor}", source_text=source_text)
        self.cuda = ctypes.CDLL("libcuda.so.1")
        with torch.cuda.device(device):
            torch.cuda.current_stream().synchronize()  # the primary context is current from here on
            self.module = ctypes.c_void_p()
            self._check(self.cuda.cuModuleLoadData(ctypes.byref(self.module), cubin))
            self.functions = {}
            for name, symbol in lowered.items():
                function = ctypes.c_void_p()
                self._check(self.cuda.cuModuleGetFunction(ctypes.byref(function), self.module, symbol.encode()))
                self.functions[name] = function

    def _check(self, result):
        if result != 0:
            text = ctypes.c_char_p()
            self.cuda.cuGetErrorString(result, ctypes.byref(text))
            raise RuntimeError("CUDA driver: " + (text.value or b"unknown").decode())

    def launch(self, name, grid, block, args):
        holders = []
        for value in args:
            if isinstance(value, self.torch.Tensor):
                holders.append(ctypes.c_void_p(value.data_ptr()))
            elif isinstance(value, ctypes._SimpleCData):
                holders.append(value)
            else:
                raise TypeError(f"kernel argument needs an explicit C type, got {type(value)}")
        pointers = (ctypes.c_void_p * len(holders))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
        stream = ctypes.c_void_p(self.torch.cuda.current_stream().cuda_stream)
        self._check(self.cuda.cuLaunchKernel(self.functions[name], grid, 1, 1, block, 1, 1, 0, stream, pointers, None))


_KERNELS = {}


def kernels(device):
    index = device.index if device.index is not None else __import__("torch").cuda.current_device()
    if index not in _KERNELS:
        _KERNELS[index] = _Kernels(index)
    return _KERNELS[index]


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _check_layout(params, tensors, qgroup_size):
    import torch

    _require(qgroup_size == 128, "Only Support 128 per-group quantization currently")
    _require(params.numel() < 2**31, "the upstream kernel indexes with int")
    for tensor in tensors:
        _require(tensor.is_cuda and tensor.device == params.device, "all COAT tensors must share one CUDA device")
        # data_ptr<T>() in the extension reads raw memory, so it silently assumed this.
        _require(tensor.is_contiguous(), "COAT tensors must be contiguous")
    return (params.numel() + qgroup_size - 1) // 128


def fp8_adamw_expand_step(params, grads, exp_avg, scale_exp_avg, expand_exp_avg, sqrtminmax_exp_avg,
                          exp_avg_sq, scale_exp_avg_sq, expand_exp_avg_sq, sqrtminmax_exp_avg_sq,
                          beta1, beta2, lr, wd, eps, step, qgroup_size, expand_min):
    """`FP8_AdamW_expand_cuda` (upstream .cu, host part): same asserts, grid and block."""
    import torch

    grid_dim = _check_layout(params, (params, grads, exp_avg, scale_exp_avg, expand_exp_avg, sqrtminmax_exp_avg,
                                      exp_avg_sq, scale_exp_avg_sq, expand_exp_avg_sq, sqrtminmax_exp_avg_sq), qgroup_size)
    _require(grid_dim == scale_exp_avg.numel() and grid_dim == scale_exp_avg_sq.numel(), "grid_dim != scale numel")
    # AT_DISPATCH plus data_ptr<T>(): params and grads share the dispatched type. Only the FP32
    # instantiation is compiled -- MCore hands the optimizer FP32 main parameters.
    _require(params.dtype == torch.float32 and grads.dtype == torch.float32, "COAT step expects FP32 params and grads")
    _require(exp_avg.dtype == torch.float8_e4m3fn and exp_avg_sq.dtype == torch.float8_e4m3fn, "COAT moments must be E4M3")
    for tensor in (scale_exp_avg, expand_exp_avg, sqrtminmax_exp_avg, scale_exp_avg_sq, expand_exp_avg_sq, sqrtminmax_exp_avg_sq):
        _require(tensor.dtype == torch.float32, "COAT metadata must be FP32")
    f, i = ctypes.c_float, ctypes.c_int
    kernels(params.device).launch(STEP_KERNEL, grid_dim, 128, [
        params, grads, exp_avg, scale_exp_avg, expand_exp_avg, sqrtminmax_exp_avg,
        exp_avg_sq, scale_exp_avg_sq, expand_exp_avg_sq, sqrtminmax_exp_avg_sq,
        f(beta1), f(beta2), f(lr), f(wd), f(eps), i(step), i(qgroup_size), i(expand_min),
        i(params.numel()), i(scale_exp_avg.numel())])


def import_fp32_moments(src_exp_avg, src_exp_avg_sq, exp_avg, scale_exp_avg, expand_exp_avg, sqrtminmax_exp_avg,
                        exp_avg_sq, scale_exp_avg_sq, expand_exp_avg_sq, sqrtminmax_exp_avg_sq, expand_min, qgroup_size=128):
    import torch

    grid_dim = _check_layout(src_exp_avg, (src_exp_avg, src_exp_avg_sq, exp_avg, scale_exp_avg, expand_exp_avg,
                                           sqrtminmax_exp_avg, exp_avg_sq, scale_exp_avg_sq, expand_exp_avg_sq,
                                           sqrtminmax_exp_avg_sq), qgroup_size)
    _require(src_exp_avg.dtype == torch.float32 and src_exp_avg_sq.dtype == torch.float32, "import expects FP32 moments")
    _require(exp_avg.dtype == torch.float8_e4m3fn and exp_avg_sq.dtype == torch.float8_e4m3fn, "COAT moments must be E4M3")
    i = ctypes.c_int
    kernels(src_exp_avg.device).launch(IMPORT_KERNEL, grid_dim, 128, [
        src_exp_avg, src_exp_avg_sq, exp_avg, scale_exp_avg, expand_exp_avg, sqrtminmax_exp_avg,
        exp_avg_sq, scale_exp_avg_sq, expand_exp_avg_sq, sqrtminmax_exp_avg_sq,
        i(expand_min), i(src_exp_avg.numel()), i(scale_exp_avg.numel())])


def momentum_dequantize(data, scale, expand, sqrt_minmax, out):
    import torch

    grid_dim = _check_layout(out, (data, scale, expand, sqrt_minmax, out), 128)
    _require(data.dtype == torch.float8_e4m3fn and out.dtype == torch.float32, "momentum is E4M3 -> FP32")
    kernels(out.device).launch(MOMENTUM_DEQUANTIZE_KERNEL, grid_dim, 128,
                               [data, scale, expand, sqrt_minmax, out, ctypes.c_int(out.numel())])


def momentum_quantize(value, data, scale, expand, sqrt_minmax, expand_min=16):
    import torch

    grid_dim = _check_layout(value, (value, data, scale, expand, sqrt_minmax), 128)
    _require(value.dtype == torch.float32 and data.dtype == torch.float8_e4m3fn, "momentum is FP32 -> E4M3")
    _require(scale.numel() == grid_dim, "momentum metadata has the wrong number of groups")
    kernels(value.device).launch(MOMENTUM_QUANTIZE_KERNEL, grid_dim, 128,
                                 [value, data, scale, expand, sqrt_minmax, ctypes.c_int(expand_min), ctypes.c_int(value.numel())])
