#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
image=${MONARCH_PPU_IMAGE:-/bmcp_lvm_fs/apptainer/sif/asllm.sif}
base=${MONARCH_BASE_DATA_ROOT:-/bmcp_lvm_fs/data/datasets/fineweb-edu-gpt2-megatron}
tools=${STAGE3_MOE_DATA_TOOLS_ROOT:-$HOME/work/monarch-data-tools}
old=${MONARCH_OLD_EXTENSION_ROOT:-/bmcp_lvm_fs/data/datasets/fineweb-edu-time-match-extension}
dense=${MONARCH_DENSE_EXTENSION_ROOT:-/bmcp_lvm_fs/data/datasets/fineweb-edu-dense-1c-extension}

for path in \
  "$base/data/train.bin" "$base/data/train.idx" \
  "$base/data/development.bin" "$base/data/development.idx" \
  "$base/data/final.bin" "$base/data/final.idx" \
  "$base/artifact-manifest.json"; do
  [[ -s $path ]] || { echo "missing base dataset artifact: $path" >&2; exit 2; }
done

export MONARCH_RUNTIME=ppu
export STAGE3_MOE_DATA_TOOLS_ROOT=$tools
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}

run_build() {
  export STAGE3_MOE_EXTENSION_ROOT=$1
  export STAGE3_MOE_EXTENSION_PLAN=$2
  export STAGE3_MOE_EXTENSION_SAMPLES=$3
  apptainer exec \
    --bind /bmcp_lvm_fs:/bmcp_lvm_fs \
    "$image" \
    bash "$root/scripts/cloud_build_time_match_data.sh"
}

run_build \
  "$old" \
  "$root/configs/fineweb_edu_time_match_extension_plan.json" \
  1750112
run_build \
  "$dense" \
  "$root/configs/fineweb_edu_dense_1c_extension_plan.json" \
  4502020
