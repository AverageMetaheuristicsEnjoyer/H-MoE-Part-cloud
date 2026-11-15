#!/usr/bin/env bash
# A0: which FP8 recipes this image can actually run, on Linear AND on GroupedLinear.
#
#   mlsub run --entry scripts/cloud_te_recipe_capability.sh --gpus 1 --image te3
#
# The Stage 3 compute-axis arms ran per-tensor DELAYED scaling with MCore's bare
# defaults (amax_history_len=1, most_recent). docs/design.md:735-740 had rejected
# delayed scaling and specified Float8BlockScaling, which TE refused on the cu128
# image (CUDA >= 12.9). The policy now carries a cu129 image (te3), so the question
# is open again -- and it has to be asked about the *grouped* GEMM, because that is
# where the experts live and grouped block-scaled GEMM has its own cuBLAS floor.
#
# Prints one PASS/FAIL line per (recipe, layer) pair plus the amax-history geometry
# of a delayed-scaling GroupedLinear, which is what decides whether each expert gets
# its own scale or all 64 share one.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
log_dir=${STAGE3_MOE_LOG_DIR:-/home/jovyan/hmoe-cloud/logs}
mkdir -p "$log_dir" 2>/dev/null || log_dir=/tmp
log=$log_dir/te-recipe-capability-${MLSUB_IMAGE:-unknown}-$(date -u +%Y%m%dT%H%M%SZ).log

(
  set -uo pipefail

  # Only the torch28 image keeps the CUDA shared objects under conda's nvidia tree;
  # on any other image the find returns nothing and the export is a no-op.
  unset PYTHONNOUSERSITE
  nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
    -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: - || true)
  if [[ -n $nvidia_lib_path ]]; then
    export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export CUDNN_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cudnn
    export CURAND_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/curand
    export NVRTC_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cuda_nvrtc
  fi
  export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
  # Ling's FP32 scales rather than TE's power-of-two-constrained default, per design.md.
  export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=${NVTE_FP8_BLOCK_SCALING_FP32_SCALES:-1}

  echo "MLSUB_IMAGE=${MLSUB_IMAGE:-unset}"
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader

  python - <<'PY'
import sys, traceback
import torch

print("python=", sys.version.split()[0])
print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)

import transformer_engine
import transformer_engine.pytorch as te
print("te=", getattr(transformer_engine, "__version__", "unknown"))
try:
    import transformer_engine_torch as tex
    print("cublaslt=", tex.get_cublasLt_version())
    print("cudnn=", tex.get_cudnn_version())
except Exception as exc:
    print("cublaslt= unavailable:", exc)

from transformer_engine.common import recipe as te_recipe

# TE's own DelayedScaling defaults, to contrast with MCore's dataclass defaults
# (fp8_amax_history_len=1, fp8_amax_compute_algo='most_recent') that the arms ran on.
ds_default = te_recipe.DelayedScaling()
print("te_delayed_default_amax_history_len=", ds_default.amax_history_len)
print("te_delayed_default_amax_compute_algo=", ds_default.amax_compute_algo)
print("te_delayed_default_margin=", ds_default.margin)

# The images disagree on the TE API: torch28 carries TE 2.16 (te.autocast(recipe=...),
# te.is_*_available), te3 carries TE 2.5 (te.fp8_autocast(fp8_recipe=...), and the support
# checks still live in transformer_engine.pytorch.fp8 as check_*_support).
try:
    from transformer_engine.pytorch import fp8 as te_fp8
except Exception:
    te_fp8 = None

_LEGACY_CHECK = {
    "fp8": "check_fp8_support",
    "fp8_block_scaling": "check_fp8_block_scaling_support",
    "mxfp8": "check_mxfp8_support",
}

def availability(name):
    fn = getattr(te, f"is_{name}_available", None)
    if fn is None and te_fp8 is not None:
        fn = getattr(te_fp8, _LEGACY_CHECK[name], None)
    if fn is None:
        print(f"available_{name}= absent from this TE build")
        return
    for kwargs in ({"return_reason": True}, {}):
        try:
            out = fn(**kwargs)
            ok, reason = out if isinstance(out, tuple) else (out, "")
            print(f"available_{name}= {bool(ok)} {reason}")
            return
        except TypeError:
            continue
        except Exception as exc:
            print(f"available_{name}= error: {exc}")
            return
    print(f"available_{name}= no accepted signature")

for _name in ("fp8", "fp8_block_scaling", "mxfp8"):
    availability(_name)

# TE renamed fp8_autocast -> autocast during 2.x and renamed the recipe keyword with it.
_raw_autocast = getattr(te, "autocast", None) or te.fp8_autocast

def autocast(enabled=True, recipe=None):
    try:
        return _raw_autocast(enabled=enabled, recipe=recipe)
    except TypeError:
        return _raw_autocast(enabled=enabled, fp8_recipe=recipe)

def build_recipes():
    F = te_recipe.Format
    out = {}
    out["delayed_hybrid_h1"] = te_recipe.DelayedScaling(
        fp8_format=F.HYBRID, amax_history_len=1, amax_compute_algo="most_recent")
    out["delayed_hybrid_h1024max"] = te_recipe.DelayedScaling(
        fp8_format=F.HYBRID, amax_history_len=1024, amax_compute_algo="max")
    for name, cls, kwargs in [
        ("current_hybrid", "Float8CurrentScaling", {"fp8_format": F.HYBRID}),
        ("blockwise_e4m3", "Float8BlockScaling", {"fp8_format": F.E4M3}),
    ]:
        klass = getattr(te_recipe, cls, None)
        if klass is None:
            print(f"recipe_{name}= absent from this TE build")
            continue
        try:
            out[name] = klass(**kwargs)
        except Exception as exc:
            print(f"recipe_{name}= constructor failed: {exc}")
    return out

# 1024 and 2816/256 mirror the real shapes: hidden 1024, dense ffn 2816, expert ffn 256.
def make_linear():
    return te.Linear(1024, 2816, bias=False, params_dtype=torch.bfloat16, device="cuda")

def make_grouped(num_gemms=4):
    return te.GroupedLinear(num_gemms, 1024, 256, bias=False,
                            params_dtype=torch.bfloat16, device="cuda")

def run_linear(mod, recipe):
    x = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with autocast(enabled=True, recipe=recipe):
        y = mod(x)
    y.float().square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    assert torch.isfinite(mod.weight0.grad if hasattr(mod, "weight0") else mod.weight.grad).all()

def run_grouped(mod, recipe, num_gemms=4):
    # Rows per expert must stay multiples of 16 or TE refuses the quantized GEMM;
    # MCore pads with Fp8Padding for exactly this reason.
    splits = [128] * num_gemms
    x = torch.randn(sum(splits), 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with autocast(enabled=True, recipe=recipe):
        y = mod(x, splits)
    y.float().square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    assert torch.isfinite(mod.weight0.grad).all()

results = {}
for name, recipe in build_recipes().items():
    for layer, build, run in (("linear", make_linear, run_linear),
                              ("grouped", make_grouped, run_grouped)):
        key = f"{name}/{layer}"
        try:
            mod = build()
            run(mod, recipe)
            results[key] = "PASS"
        except Exception as exc:
            first = traceback.format_exception_only(type(exc), exc)[-1].strip()
            results[key] = f"FAIL {first[:220]}"
        finally:
            torch.cuda.empty_cache()
        print(f"recipe_matrix {key}= {results[key]}")

# Does a delayed-scaling GroupedLinear keep one amax per expert, or one for all of them?
# This is the MoE-specific question the 1C arms never answered.
try:
    mod = make_grouped(8)
    rec = te_recipe.DelayedScaling(fp8_format=te_recipe.Format.HYBRID)
    with autocast(enabled=True, recipe=rec):
        mod(torch.randn(8 * 128, 1024, device="cuda", dtype=torch.bfloat16), [128] * 8)
    meta = mod.fp8_meta["scaling_fwd"]
    print("grouped_num_gemms=", mod.fp8_meta.get("num_gemms"))
    print("grouped_amax_history_shape=", tuple(meta.amax_history.shape))
    print("grouped_scale_shape=", tuple(meta.scale.shape))
except Exception as exc:
    print("grouped_amax_geometry= error:", exc)

# Does our vendored MCore build each recipe on this image?
try:
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.fp8_utils import get_fp8_recipe
    from megatron.core.package_info import __shortversion__ as mcore_version
    print("mcore_version=", mcore_version)
    for r in ("delayed", "tensorwise", "blockwise"):
        try:
            cfg = TransformerConfig(
                num_layers=1, hidden_size=1024, num_attention_heads=8,
                fp8="e4m3" if r == "blockwise" else "hybrid", fp8_recipe=r)
            built = get_fp8_recipe(cfg)
            print(f"mcore_recipe_{r}= {type(built).__name__}")
        except Exception as exc:
            print(f"mcore_recipe_{r}= FAIL {exc}")
except Exception as exc:
    print("mcore_import= FAIL", exc)

print("CAPABILITY_PROBE=DONE")
PY
  echo "PY_EXIT=$?"
) >"$log" 2>&1
code=$?

echo "EXIT=$code"
echo "LOG=$log"
cat "$log"
exit 0
