#!/bin/bash
# mlsub entry point for scripts/monarch_priors.py on a public HF checkpoint.
#
#   mlsub run --repo <public mirror> --branch monarch-moe/expert-gemm-bench --image torch28 --no-pip \
#     --entry scripts/cloud_monarch_priors.sh --gpus 1 \
#     --args=1c-mb4/muon_bf16_state_fp32/iter_0017242
#
# A second argument picks the script: monarch_priors (weight energy, the
# default) or monarch_functional_priors (output error on real inputs, which
# also reads the development split from /home/jovyan/data).
#
# --gpus 1 is for the host RAM (the CPU flavour has 8 GiB) and, for the
# functional version, the forward pass. The
# checkpoint is downloaded to the first place with room, read through mmap and
# deleted again. Always exits zero; the real status is the EXIT line.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

# a copy of the output on a volume that outlives the job: mlsub logs hangs on a
# running job, so `--args=peek` in a CPU job is the only way to watch one
logs=
for volume in /workspace-SR006.nfs3 /workspace-SR006.nfs2 /home/jovyan; do
    if mkdir -p "$volume/monarch-moe/logs" 2>/dev/null; then
        logs="$volume/monarch-moe/logs"
        break
    fi
done
if [ "${1:-}" = "peek" ]; then
    newest=$(ls -t "$logs"/priors_*.log 2>/dev/null | head -1)
    echo "=== ${newest:-no log}"
    [ -n "$newest" ] && tail -"${2:-60}" "$newest"
    exit 0
fi
checkpoint=${1:?usage: cloud_monarch_priors.sh HF_CHECKPOINT_DIR [SCRIPT]}
script=${2:-monarch_priors}
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
            python3 "scripts/$script.py" "$file"
    echo "EXIT=$?"
} 2>&1 | tee "$logs/priors_$(date +%F_%H%M%S).log"
rm -f "$file"
exit 0
