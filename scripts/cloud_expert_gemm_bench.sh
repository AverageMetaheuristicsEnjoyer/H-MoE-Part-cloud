#!/bin/bash
# mlsub entry point for the expert-bank GEMM benchmark.
#
#   mlsub run --repo <public mirror> --branch monarch-moe/expert-gemm-bench --image torch28 --no-pip \
#     --entry scripts/cloud_expert_gemm_bench.sh --gpus 1 \
#     --args "--sweep 128,256,512,1024,2048 --blocks 2"
#
#   mlsub run ... --gpus cpu --args "--selftest"    cheap rehearsal, no GPU
#
# A Failed mlsub job shows no logs at all, so everything is teed to a volume
# that survives the job and this script always exits zero.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

# /home/jovyan has hit its inode quota before while df still reported free
# gigabytes; take the first volume that actually accepts a directory.
workspace=${BENCH_WORKSPACE:-}
if [ -z "$workspace" ]; then
    for volume in /workspace-SR006.nfs3 /workspace-SR006.nfs2 /home/jovyan /tmp; do
        if mkdir -p "$volume/monarch-moe/logs" 2>/dev/null; then
            workspace="$volume/monarch-moe"
            break
        fi
    done
fi
mkdir -p "$workspace/logs" "$workspace/results"

export PYTHONUSERBASE="$workspace/userbase"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONUNBUFFERED=1
mkdir -p "$PYTHONUSERBASE"

if [ "${1:-}" = "peek" ]; then
    echo "workspace: $workspace"
    find "$workspace/results" -name '*.json' 2>/dev/null | sort
    newest=$(ls -t "$workspace"/logs/*.log 2>/dev/null | head -1)
    echo "=== tail of ${newest:-no log} ==="
    [ -n "$newest" ] && tail -"${2:-200}" "$newest"
    exit 0
fi

# --gpus 2+ starts this entry point once per MPI rank; only rank 0 measures.
rank=${OMPI_COMM_WORLD_RANK:-0}
if [ "$rank" != "0" ]; then
    echo "rank $rank stands down"
    exit 0
fi

stamp=$(date +%F_%H%M%S)
log="$workspace/logs/bench_$stamp.log"

if [ "${1:-}" = "--selftest" ]; then
    # CPU rehearsal: the clone, the interpreter and the import graph only.
    {
        echo "=== selftest $stamp ==="
        df -h "$workspace" | tail -1
        python3 - <<'PY'
import ast, torch
print("torch", torch.__version__)
print("grouped_mm", hasattr(torch, "_grouped_mm"))
print("cuda", torch.cuda.is_available())
ast.parse(open("scripts/expert_gemm_bench.py").read())
print("bench parses")
PY
    } >"$log" 2>&1
else
    python3 scripts/expert_gemm_bench.py \
        --json-out "$workspace/results/bench_$stamp.json" "$@" >"$log" 2>&1
fi

code=$?
echo "EXIT=$code"
echo "LOG=$log"
echo "=== full output ==="
cat "$log"
exit 0
