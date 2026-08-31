#!/usr/bin/env bash
set -u

blocks=${1:?usage: cloud_monarch_bench_suite.sh BLOCKS ARM [ARM ...]}
shift
(( $# > 0 )) || { echo "provide at least one arm" >&2; exit 2; }
root=$(cd "$(dirname "$0")/.." && pwd)
world_size=${OMPI_COMM_WORLD_SIZE:?missing OMPI_COMM_WORLD_SIZE}

case "$world_size" in
  1) hmoe_parallel=ddp; dense_parallel=ddp ;;
  2|4) hmoe_parallel=ep; dense_parallel=pp ;;
  *) echo "WORLD_SIZE must be 1, 2, or 4" >&2; exit 2 ;;
esac

status=0
for arm in "$@"; do
  "$root/scripts/cloud_monarch_pretrain.sh" hmoe "$arm" "$blocks" "$hmoe_parallel" bench
  code=$?
  echo "BENCH_EXIT=$code model=hmoe arm=$arm blocks=$blocks world_size=$world_size parallelism=$hmoe_parallel"
  (( code == 0 )) || status=$code

  "$root/scripts/cloud_monarch_pretrain.sh" dense "$arm" "$blocks" "$dense_parallel" bench
  code=$?
  echo "BENCH_EXIT=$code model=dense arm=$arm blocks=$blocks world_size=$world_size parallelism=$dense_parallel"
  (( code == 0 )) || status=$code
done

echo "EXIT=$status"
exit "$status"
