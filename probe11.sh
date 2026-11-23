#!/bin/bash
for spec in "031 lm5r50ec" "036 lm5r50ec" "036 lmlinrevr50"; do
  set -- $spec; V=$1; A=$2
  LOG=$(ls -t /workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v$V/arms/$A/logs/*/train-*.log 2>/dev/null | head -1)
  echo "### V$V ARM=$A"
  echo "LOG=$LOG"
  echo "--- args"
  grep -aE "micro-batch-size|global-batch-size|data_parallel_size|world_size" "$LOG" 2>/dev/null | head -4
  echo "--- iterations"
  grep -aE "iteration +[0-9]+/ +17242" "$LOG" 2>/dev/null | head -40
done
echo "### done"
