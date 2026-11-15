#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
unset PYTHONNOUSERSITE
nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: - || true)
if [[ -n $nvidia_lib_path ]]; then
  export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
output_root=${STAGE3_MOE_DIAGNOSTIC_ROOT:?set a fresh persistent diagnostic output directory}
python -u tools/dense_quantization_diagnostics.py --output-root "$output_root" "$@"
rc=$?
echo "DENSE_DIAGNOSTICS_EXIT=$rc"
exit 0
