#!/usr/bin/env bash
# tools/moe_scale_bench.py with the library environment of run_stage3_moe_pretrain.sh (needed on the torch28 image).
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
unset PYTHONNOUSERSITE
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1
nvidia_lib_path=$(python - <<'PYEOF' 2>/dev/null || true
import importlib.util, pathlib
spec = importlib.util.find_spec("nvidia")
roots = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
libs = [str(p) for root in roots for p in sorted(pathlib.Path(root).glob("*/lib")) if p.is_dir()]
print(":".join(libs))
PYEOF
)
[[ -n $nvidia_lib_path ]] && export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
IFS=: read -r -a nvidia_lib_dirs <<< "$nvidia_lib_path"
for library in "${nvidia_lib_dirs[@]}"; do
  case "$library" in
    */cudnn/lib) export CUDNN_HOME=${CUDNN_HOME:-${library%/lib}} ;;
    */curand/lib) export CURAND_HOME=${CURAND_HOME:-${library%/lib}} ;;
    */cuda_nvrtc/lib) export NVRTC_HOME=${NVRTC_HOME:-${library%/lib}} ;;
  esac
done
echo "SCALEBENCH_ENV python=$(command -v python) nvidia_libs=${#nvidia_lib_dirs[@]}"
cd "$root"
python tools/moe_scale_bench.py "$@"
