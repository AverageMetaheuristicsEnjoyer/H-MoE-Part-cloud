#!/bin/bash
R=/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v036
echo "== df"; df -h /workspace-SR006.nfs3 | tail -1
echo "== root"; ls -la $R 2>&1 | head -12
for a in lm5r50ec lmlinrevr50; do
  echo "== $a"
  cat $R/arms/$a/continue-progress.json 2>/dev/null | head -c 300; echo
  LOG=$(ls -t $R/arms/$a/logs/*/train-*.log 2>/dev/null | head -1)
  echo "LOG=$LOG"
  grep -aE "iteration +[0-9]+/" "$LOG" 2>/dev/null | tail -2
done
echo "== payload-progress"; tail -3 $R/payload-progress.jsonl 2>/dev/null
echo "== done"
