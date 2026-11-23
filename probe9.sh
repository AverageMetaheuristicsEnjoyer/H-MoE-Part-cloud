#!/bin/bash
for V in 031 035; do
  R=/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v$V
  echo "== v$V root"; ls -la $R 2>&1 | head -12
  echo "== v$V progress"; tail -3 $R/payload-progress.jsonl 2>/dev/null
  LOG=$(ls -t $R/arms/*/logs/*/train-*.log 2>/dev/null | head -1)
  echo "== v$V trainlog $LOG"; grep -aE "iteration +[0-9]+/" "$LOG" 2>/dev/null | tail -2
done
echo "== df"; df -h /workspace-SR006.nfs3 | tail -1
echo "== done"
