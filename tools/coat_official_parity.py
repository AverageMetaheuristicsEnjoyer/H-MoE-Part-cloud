"""Bitwise parity of the NVRTC COAT optimizer kernels against upstream's own nvcc build.

  build    (CPU) build upstream `qoptim_cuda` with its unmodified setup.py and nvcc, compile our
           NVRTC source with the production NVRTC and with the NVRTC of the same CUDA release as
           that nvcc, and compare the SASS of the fused step kernel.
  compare  (GPU) run the official extension and our launcher on identical inputs and compare
           every output buffer bit for bit: the fused AdamW step over many steps, edge cases,
           and Muon's momentum codec through the official kernel (beta1 = 0 isolates its
           quantization, beta1 = 1/2 with zero gradient isolates its dequantization).

Lines starting PARITY carry results.
"""
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("COAT_OFFICIAL_ROOT", "/workspace-SR006.nfs3/hmoe-cloud/coat-20261002/official-qoptim"))
CUDA_HOME = Path(os.environ.get("COAT_OFFICIAL_CUDA_HOME", "/home/jovyan/hmoe-cloud/cuda-12.1.1"))
STEP_SYMBOL = "_Z28fp8_adamw_cuda_expand_kernelIfEvPT_S1_P13__nv_fp8_e4m3PfS4_S4_S3_S4_S4_S4_fffffiiiii"


def say(kind, **values):
    print("PARITY " + json.dumps({"kind": kind, **values}), flush=True)


def run(command, **kw):
    r = subprocess.run(command, capture_output=True, text=True, **kw)
    return r.returncode, (r.stdout + r.stderr)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sass_of(binary, cuobjdump):
    """Instruction text of the fused step kernel, without addresses and encodings."""
    code, text = run([str(cuobjdump), "-sass", str(binary)])
    if code:
        return None, text[-2000:]
    blocks = re.split(r"\n\s*Function : ", text)
    for block in blocks[1:]:
        name, _, body = block.partition("\n")
        if name.strip() == STEP_SYMBOL:
            lines = []
            for line in body.splitlines():
                m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s*(.*?)\s*;?\s*/\*.*$", line)
                if m and m[1]:
                    lines.append(m[1].rstrip(" ;"))
            return lines, None
    return None, "step kernel not found; functions: " + ", ".join(b.partition("\n")[0].strip() for b in blocks[1:])[:2000]


def build():
    from stage3_moe import coat_kernels

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = OUT / stamp
    out.mkdir(parents=True, exist_ok=True)
    nvcc = CUDA_HOME / "bin/nvcc"
    facts = {"cuda_home": str(CUDA_HOME), "nvcc_exists": nvcc.exists()}
    for label, command in (("nvcc", [str(nvcc), "--version"]), ("gxx", ["g++", "--version"]),
                           ("torch", [sys.executable, "-c", "import torch; print(torch.__version__, torch.version.cuda)"])):
        facts[label] = run(command)[1].strip().splitlines()[-1:] if Path(command[0]).exists() or shutil.which(command[0]) else "missing"
    facts["cuda_home_entries"] = sorted(p.name for p in CUDA_HOME.iterdir()) if CUDA_HOME.is_dir() else None
    facts["libcudart"] = [str(p) for p in CUDA_HOME.rglob("libcudart.so*")][:6]
    facts["libnvrtc"] = [str(p) for p in CUDA_HOME.rglob("libnvrtc.so*")][:6]
    facts["cuda_fp8_h"] = [str(p) for p in CUDA_HOME.rglob("cuda_fp8.h")][:4]
    facts["cuobjdump"] = [str(p) for p in CUDA_HOME.rglob("cuobjdump")][:2]
    say("build_environment", **facts)

    # Upstream's kernels directory, unmodified, built the way its README builds it.
    source = ROOT / "third_party/coat/coat/optimizer/kernels"
    work = Path("/tmp/qoptim-official-build")
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(source, work)
    env = dict(os.environ, CUDA_HOME=str(CUDA_HOME), PATH=f"{CUDA_HOME / 'bin'}:{os.environ['PATH']}",
               TORCH_CUDA_ARCH_LIST="9.0", MAX_JOBS=os.environ.get("COAT_OFFICIAL_MAX_JOBS", "8"))
    # conda's toolkit keeps libraries in lib/, not the lib64/ torch links against.
    lib_dirs = sorted({str(Path(p).parent) for p in facts["libcudart"]})
    if lib_dirs:
        env["LIBRARY_PATH"] = ":".join(lib_dirs + [os.environ.get("LIBRARY_PATH", "")])
    include_dirs = sorted({str(Path(p).parent) for p in facts["cuda_fp8_h"]})
    if include_dirs and not (CUDA_HOME / "include/cuda_fp8.h").exists():
        env["CPATH"] = ":".join(include_dirs + [os.environ.get("CPATH", "")])
    code, log = run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=work, env=env)
    (out / "build.log").write_text(log)
    built = sorted(work.glob("qoptim_cuda*.so"))
    say("build", returncode=code, built=[p.name for p in built], log_tail=log.splitlines()[-40:] if code or not built else log.splitlines()[-6:])
    receipt = {"stamp": stamp, "facts": facts, "returncode": code,
               "upstream_sources_sha256": {p.name: sha(p) for p in sorted(source.iterdir()) if p.is_file()}}
    if built:
        shutil.copy2(built[0], out / built[0].name)
        receipt["so"] = built[0].name
        receipt["so_sha256"] = sha(out / built[0].name)

    # Our NVRTC source: production NVRTC (pip, CUDA 12.9) and the NVRTC of the nvcc's release.
    cubins = {}
    try:
        cubin, lowered, _ = coat_kernels.compile_cubin("90")
        (out / "nvrtc-prod.cubin").write_bytes(cubin)
        cubins["nvrtc-prod"] = lowered
    except Exception as error:  # report and keep going: the GPU phase compiles on its own
        say("nvrtc_prod_failed", error=repr(error)[:2000])
    nvrtc_lib = sorted(CUDA_HOME.rglob("libnvrtc.so*"))
    nvrtc_lib = [p for p in nvrtc_lib if "builtins" not in p.name]
    if nvrtc_lib and include_dirs:
        try:
            cubin, lowered, _ = coat_kernels.compile_cubin("90", nvrtc_path=nvrtc_lib[0], include_dirs=[Path(d) for d in include_dirs])
            (out / "nvrtc-toolkit.cubin").write_bytes(cubin)
            cubins["nvrtc-toolkit"] = lowered
        except Exception as error:
            say("nvrtc_toolkit_failed", error=repr(error)[:2000])
    (out / "lowered.json").write_text(json.dumps(cubins, indent=1))
    receipt["cubins"] = sorted(cubins)

    cuobjdump = facts["cuobjdump"][0] if facts["cuobjdump"] else None
    if cuobjdump and built:
        listings = {"nvcc": sass_of(out / built[0].name, cuobjdump)}
        for name in cubins:
            listings[name] = sass_of(out / f"{name}.cubin", cuobjdump)
        summary = {name: (len(lines) if lines else err) for name, (lines, err) in listings.items()}
        same = {name: listings[name][0] == listings["nvcc"][0] for name in cubins if listings[name][0] and listings["nvcc"][0]}
        for name, (lines, _) in listings.items():
            if lines:
                (out / f"step-kernel-{name}.sass").write_text("\n".join(lines) + "\n")
        receipt["sass"] = {"instructions": summary, "identical_to_nvcc": same}
        say("sass", instructions=summary, identical_to_nvcc=same)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    if built:
        (OUT / "latest").write_text(stamp + "\n")
    say("build_done", out=str(out), so=receipt.get("so"), so_sha256=receipt.get("so_sha256"))
    return 0


# ---------------------------------------------------------------------------------------------
# GPU comparison


def load_official(directory):
    import torch  # noqa: F401  (the extension links against libtorch)

    so = next(directory.glob("qoptim_cuda*.so"))
    loader = importlib.machinery.ExtensionFileLoader("qoptim_cuda", str(so))
    spec = importlib.util.spec_from_file_location("qoptim_cuda", str(so), loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def kernels_from_cubin(cubin, lowered):
    """A coat_kernels._Kernels holding a prebuilt cubin instead of compiling one."""
    import ctypes

    import torch
    from stage3_moe import coat_kernels

    k = object.__new__(coat_kernels._Kernels)
    k.torch = torch
    k.cuda = ctypes.CDLL("libcuda.so.1")
    torch.cuda.current_stream().synchronize()
    k.module = ctypes.c_void_p()
    k._check(k.cuda.cuModuleLoadData(ctypes.byref(k.module), cubin))
    k.functions = {}
    for name, symbol in lowered.items():
        function = ctypes.c_void_p()
        k._check(k.cuda.cuModuleGetFunction(ctypes.byref(function), k.module, symbol.encode()))
        k.functions[name] = function
    return k


STATE = ("exp_avg", "scale_exp_avg", "expand_exp_avg", "sqrt_minmax_exp_avg",
         "exp_avg_sq", "scale_exp_avg_sq", "expand_exp_avg_sq", "sqrt_minmax_exp_avg_sq")


def fresh_state(p):
    """Upstream `_init_group` for a new parameter."""
    import torch

    groups = (p.numel() + 127) // 128
    return {"exp_avg": torch.zeros_like(p, dtype=torch.float8_e4m3fn),
            "scale_exp_avg": torch.zeros(groups, device=p.device, dtype=p.dtype),
            "expand_exp_avg": torch.ones(groups, device=p.device, dtype=p.dtype),
            "sqrt_minmax_exp_avg": torch.ones(groups, device=p.device, dtype=p.dtype),
            "exp_avg_sq": torch.zeros_like(p, dtype=torch.float8_e4m3fn),
            "scale_exp_avg_sq": torch.zeros(groups, device=p.device, dtype=p.dtype),
            "expand_exp_avg_sq": torch.ones(groups, device=p.device, dtype=p.dtype),
            "sqrt_minmax_exp_avg_sq": torch.ones(groups, device=p.device, dtype=p.dtype)}


def bits(t):
    import torch

    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int32)


def diff(a, b):
    """Number of bitwise-different elements and the first few of them."""
    import torch

    ne = (bits(a) != bits(b)).view(-1)
    count = int(ne.sum())
    first = []
    if count:
        for i in torch.nonzero(ne).view(-1)[:3].tolist():
            first.append([i, float(a.view(-1)[i].float()), float(b.view(-1)[i].float())])
    return count, first


def step_with(launch, p, st, g, hp, step):
    launch(p, g, *(st[k] for k in STATE), hp["beta1"], hp["beta2"], hp["lr"], hp["wd"], hp["eps"], step, 128, 16)


def edge_values(numel, device):
    """Groups that hit every branch of the upstream reduction and the ratio/expansion arithmetic."""
    import torch

    groups = []
    g = torch.Generator(device=device).manual_seed(7)
    rnd = lambda: torch.randn(128, device=device, generator=g)  # noqa: E731
    z = torch.zeros(128, device=device)
    groups.append(z.clone())                                   # all zero
    x = z.clone(); x[0] = 0.37; groups.append(x)               # only lane 0 nonzero: ratio 1
    x = z.clone(); x[5] = -0.37; groups.append(x)              # one nonzero elsewhere
    groups.append(torch.full((128,), 1.0, device=device))      # all equal
    groups.append(torch.full((128,), -3.0, device=device))
    x = rnd(); x[0] = 0.0; groups.append(x)                    # lane-0 zero: the min quirk
    x = rnd(); x[32] = 0.0; x[64] = 0.0; groups.append(x)      # zeros in other warps' lane 0
    x = rnd(); x[::2] = 0.0; groups.append(x)
    groups.append(rnd() * 1e-38)                               # near the FP32 normal limit
    groups.append(rnd() * 1e-42)                               # subnormal
    groups.append(rnd() * 1e30)
    x = rnd(); x[3] = 1e20; groups.append(x)                   # one huge outlier
    groups.append(rnd())
    v = torch.cat(groups)
    reps = (numel + v.numel() - 1) // v.numel()
    return v.repeat(reps)[:numel].contiguous()


def compare():
    import torch
    from stage3_moe import coat_kernels

    if not (OUT / "latest").exists():
        # No CPU build yet: GPU nodes have the cores to build it here.
        build()
    stamp = (OUT / "latest").read_text().strip()
    directory = OUT / stamp
    receipt = json.loads((directory / "receipt.json").read_text())
    official = load_official(directory)
    device = torch.device("cuda")
    say("compare_environment", build=stamp, so_sha256=receipt.get("so_sha256"), gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, nvcc=receipt["facts"].get("nvcc"))
    lowered = json.loads((directory / "lowered.json").read_text())
    launchers = {"nvrtc-prod": coat_kernels.fp8_adamw_expand_step}
    index = torch.cuda.current_device()
    if "nvrtc-toolkit" in lowered:
        alt = kernels_from_cubin((directory / "nvrtc-toolkit.cubin").read_bytes(), lowered["nvrtc-toolkit"])

        def toolkit_step(*args):
            saved = coat_kernels._KERNELS.get(index)
            coat_kernels._KERNELS[index] = alt
            try:
                coat_kernels.fp8_adamw_expand_step(*args)
            finally:
                if saved is None:
                    del coat_kernels._KERNELS[index]
                else:
                    coat_kernels._KERNELS[index] = saved
        launchers["nvrtc-toolkit"] = toolkit_step
    coat_kernels.kernels(device)  # compile the production module before anything else touches _KERNELS

    prod = dict(beta1=0.9, beta2=0.95, lr=1.63e-3, wd=0.1, eps=1e-8)
    gen = torch.Generator(device="cuda").manual_seed(1234)

    def gauss(numel, s):
        return torch.randn(numel, device=device, generator=gen) * s

    def heavy(numel, s):
        x = gauss(numel, s) * torch.exp(2 * torch.randn(numel, device=device, generator=gen))
        x[::7] = 0.0
        lane0 = x.view(-1)[: numel // 128 * 128].view(-1, 128)
        lane0[::2, 0] = 0.0
        return x

    cases = [
        ("gauss_tail", 128 * 4096 + 37, 40, 1, prod, lambda n, t: gauss(n, 1e-3 * (1 + t % 5))),
        ("heavy_zeros", 128 * 4096 + 37, 40, 1, prod, lambda n, t: heavy(n, 1e-3)),
        ("edge", 128 * 13 * 40 + 1, 12, 1, prod, lambda n, t: edge_values(n, device) * (1 if t % 2 else -0.5)),
        ("late_steps", 128 * 2048, 10, 9000, prod, lambda n, t: heavy(n, 1e-4)),
        ("matrix_1024x4096", 1024 * 4096, 6, 1, prod, lambda n, t: heavy(n, 1e-3)),
        ("embedding_50304x1024", 50304 * 1024, 3, 1, prod, lambda n, t: heavy(n, 1e-4)),
    ]
    totals = {}
    for name, numel, steps, first_step, hp, grads in cases:
        p0 = gauss(numel, 0.02)
        runs = {"official": (p0.clone(), fresh_state(p0))}
        for label in launchers:
            runs[label] = (p0.clone(), fresh_state(p0))
        worst = {label: {"steps_with_diff": 0, "elements": 0, "first": None} for label in launchers}
        nan_official = 0
        for t in range(steps):
            g = grads(numel, t).contiguous()
            step = first_step + t
            step_with(official.fp8_adamw_expand_step, *runs["official"], g, hp, step)
            for label, launch in launchers.items():
                step_with(launch, *runs[label], g, hp, step)
            torch.cuda.synchronize()
            ref_p, ref_st = runs["official"]
            nan_official = int(torch.isnan(ref_p).sum())
            for label in launchers:
                p, st = runs[label]
                total = 0
                first = None
                for key, a, b in [("param", ref_p, p)] + [(k, ref_st[k], st[k]) for k in STATE]:
                    count, where = diff(a, b)
                    total += count
                    if count and first is None:
                        first = {"step": step, "tensor": key, "count": count, "examples": where}
                if total:
                    worst[label]["steps_with_diff"] += 1
                    worst[label]["elements"] = max(worst[label]["elements"], total)
                    worst[label]["first"] = worst[label]["first"] or first
        for label in launchers:
            say("adamw_step", case=name, numel=numel, steps=steps, launcher=label, bitwise_equal=worst[label]["steps_with_diff"] == 0,
                nan_params_official=nan_official, **worst[label])
            totals[(name, label)] = worst[label]["steps_with_diff"] == 0
        del runs
        torch.cuda.empty_cache()

    # Muon momentum codec against the official kernel.
    for name, values in (("gauss", gauss(128 * 4096 + 37, 1e-3)), ("heavy_zeros", heavy(128 * 4096 + 37, 1e-3)),
                         ("edge", edge_values(128 * 13 * 40 + 1, device)), ("matrix_1024x4096", heavy(1024 * 4096, 1e-3))):
        numel = values.numel()
        # beta1 = 0: the kernel's first-moment update is 0 * m + 1 * g = g, so it stores quantize(g).
        p = torch.zeros(numel, device=device)
        ref = fresh_state(p)
        official.fp8_adamw_expand_step(p, values, *(ref[k] for k in STATE), 0.0, 0.5, 0.0, 0.0, 1.0, 1, 128, 16)
        ours = fresh_state(p)
        coat_kernels.momentum_quantize(values, ours["exp_avg"], ours["scale_exp_avg"], ours["expand_exp_avg"], ours["sqrt_minmax_exp_avg"])
        torch.cuda.synchronize()
        q = {k: diff(ref[k], ours[k]) for k in ("exp_avg", "scale_exp_avg", "expand_exp_avg", "sqrt_minmax_exp_avg")}
        # beta1 = 1/2, g = 0, eps = 1, step 1, zero second moment, lr = -1, wd = 0, p = 0:
        # m' = m/2 (exact), denom = (0 + 1) * (1 - 1/2) = 1/2, p' = 0 - (-1) * (m'/denom) = m.
        state = fresh_state(p)
        for k in ("exp_avg", "scale_exp_avg", "expand_exp_avg", "sqrt_minmax_exp_avg"):
            state[k] = ref[k].clone()
        p_out = torch.zeros(numel, device=device)
        official.fp8_adamw_expand_step(p_out, torch.zeros_like(values), *(state[k] for k in STATE), 0.5, 0.5, -1.0, 0.0, 1.0, 1, 128, 16)
        mine = torch.empty(numel, device=device)
        coat_kernels.momentum_dequantize(ref["exp_avg"], ref["scale_exp_avg"], ref["expand_exp_avg"], ref["sqrt_minmax_exp_avg"], mine)
        torch.cuda.synchronize()
        # Halving is exact unless it lands in the subnormal range; those elements are reported apart.
        tiny = mine.abs() < 2.0 ** -125
        dq_count, dq_where = diff(p_out[~tiny], mine[~tiny])
        say("muon_codec", case=name, numel=numel,
            quantize_bitwise_equal=all(c == 0 for c, _ in q.values()), quantize_diff={k: v for k, v in q.items() if v[0]},
            dequantize_bitwise_equal=dq_count == 0, dequantize_diff=dq_count, dequantize_examples=dq_where,
            dequantize_tiny_excluded=int(tiny.sum()), nan_dequantized=int(torch.isnan(mine).sum()))
        totals[("muon_" + name, "quantize")] = all(c == 0 for c, _ in q.values())
        totals[("muon_" + name, "dequantize")] = dq_count == 0

    verdict = all(totals.values())
    result = {"build": stamp, "verdict_all_bitwise_equal": verdict,
              "cases": {f"{a}/{b}": v for (a, b), v in totals.items()}}
    (directory / f"parity-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json").write_text(json.dumps(result, indent=1) + "\n")
    say("verdict", **result)
    return 0


if __name__ == "__main__":
    mode = sys.argv[1]
    raise SystemExit(build() if mode == "build" else compare())
