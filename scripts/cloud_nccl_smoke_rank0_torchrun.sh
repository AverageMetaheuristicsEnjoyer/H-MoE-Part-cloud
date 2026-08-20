#!/usr/bin/env bash
set -euo pipefail

: "${OMPI_COMM_WORLD_RANK:?Cloud MPI rank is missing}"
: "${OMPI_COMM_WORLD_SIZE:?Cloud MPI world size is missing}"
[[ $OMPI_COMM_WORLD_SIZE == 2 ]] || {
  echo "expected two Cloud MPI ranks, got $OMPI_COMM_WORLD_SIZE" >&2
  exit 2
}

if [[ $OMPI_COMM_WORLD_RANK != 0 ]]; then
  echo "LAUNCH=rank0-torchrun outer_rank=$OMPI_COMM_WORLD_RANK action=idle"
  exit 0
fi

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[[ $gpu_count == 2 ]] || {
  echo "expected two visible GPUs, got $gpu_count" >&2
  exit 2
}

echo "LAUNCH=rank0-torchrun outer_rank=0 workers=$gpu_count"
exec python -m torch.distributed.run --standalone --nproc-per-node "$gpu_count" \
  scripts/nccl_smoke.py
