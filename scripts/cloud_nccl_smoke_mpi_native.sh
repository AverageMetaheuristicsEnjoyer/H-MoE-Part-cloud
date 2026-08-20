#!/usr/bin/env bash
set -euo pipefail

: "${OMPI_COMM_WORLD_RANK:?Cloud MPI rank is missing}"
: "${OMPI_COMM_WORLD_LOCAL_RANK:?Cloud MPI local rank is missing}"
: "${OMPI_COMM_WORLD_SIZE:?Cloud MPI world size is missing}"
[[ $OMPI_COMM_WORLD_SIZE == 2 ]] || {
  echo "expected two Cloud MPI ranks, got $OMPI_COMM_WORLD_SIZE" >&2
  exit 2
}

export RANK=$OMPI_COMM_WORLD_RANK
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

echo "LAUNCH=mpi-native rank=$RANK local_rank=$LOCAL_RANK world_size=$WORLD_SIZE"
exec python scripts/nccl_smoke.py
