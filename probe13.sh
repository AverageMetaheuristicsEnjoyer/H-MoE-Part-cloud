#!/bin/bash
for A in lm5r50ec lmlinrevr50; do
  LOG=$(ls -t /workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v036/arms/$A/logs/*/train-*.log 2>/dev/null | head -1)
  echo "### $A $LOG"
  grep -aE "micro-batch-size|global-batch-size" "$LOG" 2>/dev/null | head -2
  grep -aE "iteration +[0-9]+/ +17242" "$LOG" 2>/dev/null | tail -3
done
echo "### done"
