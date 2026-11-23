#!/bin/bash
R=/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v031
echo "== df"; df -h /workspace-SR006.nfs3 | tail -1
echo "== root"; ls -la $R 2>&1 | head -20
echo "== progress"; cat $R/payload-progress.jsonl 2>/dev/null | tail -5
echo "== continue-progress"; cat $R/arms/lm5r50ec/continue-progress.json 2>/dev/null | head -c 1200
echo
echo "== newest train log"; LOG=$(ls -t $R/arms/lm5r50ec/logs/*/train-*.log 2>/dev/null | head -1); echo "LOG=$LOG"; tail -c 3000 "$LOG" 2>/dev/null | tail -8
echo "== last iteration lines"; grep -aE "iteration +[0-9]+/" "$LOG" 2>/dev/null | tail -3
echo "== done"
