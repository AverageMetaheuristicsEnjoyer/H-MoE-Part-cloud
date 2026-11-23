#!/bin/bash
LOG=$(ls -t /workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v031/arms/lm5r50ec/logs/*/train-*.log 2>/dev/null | head -1)
echo "LOG=$LOG"
grep -aE "iteration +[0-9]+/ +17242" "$LOG" 2>/dev/null | head -40
echo "=== last"
grep -aE "iteration +[0-9]+/ +17242" "$LOG" 2>/dev/null | tail -5
