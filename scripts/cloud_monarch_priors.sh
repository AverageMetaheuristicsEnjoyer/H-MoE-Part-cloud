#!/bin/bash
# mlsub entry point for scripts/monarch_priors.py on a public HF checkpoint.
#
#   mlsub run --repo <public mirror> --branch monarch-moe/expert-gemm-bench --image torch28 --no-pip \
#     --entry scripts/cloud_monarch_priors.sh --gpus 1 \
#     --args=1c-mb4/muon_bf16_state_fp32/iter_0017242
#
# --gpus 1 is for the host RAM (the CPU flavour has 8 GiB), not the GPU. The
# checkpoint is downloaded to the first place with room, read through mmap and
# deleted again. Always exits zero; the real status is the EXIT line.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
checkpoint=${1:?usage: cloud_monarch_priors.sh HF_CHECKPOINT_DIR}
url="https://huggingface.co/AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints/resolve/main/$checkpoint/mp_rank_00/model_optim_rng.pt"

scratch=
for place in /tmp /workspace-SR006.nfs3 /workspace-SR006.nfs2; do
    free_kb=$(df -Pk "$place" 2>/dev/null | awk 'NR==2 {print $4}')
    if [ -n "$free_kb" ] && [ "$free_kb" -gt $((16 * 1024 * 1024)) ] &&
        mkdir -p "$place/monarch-moe/priors" 2>/dev/null; then
        scratch="$place/monarch-moe/priors"
        break
    fi
done
[ -n "$scratch" ] || { echo "EXIT=2 no volume with 16 GiB free"; exit 0; }
file="$scratch/$(echo "$checkpoint" | tr / _).pt"

export PYTHONUSERBASE="$scratch/userbase"
{
    echo "=== $checkpoint -> $file"
    python3 -c "import scipy" 2>/dev/null || python3 -m pip install --user -q scipy
    curl -sSfL --retry 3 -o "$file" "$url" && ls -l "$file" &&
        PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root" \
            python3 scripts/monarch_priors.py "$file"
    echo "EXIT=$?"
} 2>&1
rm -f "$file"
exit 0
