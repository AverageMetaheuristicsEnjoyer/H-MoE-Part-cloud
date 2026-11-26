#!/bin/bash
R=/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v041
L=$(ls -t $R/arms/lm3block/logs/*/train-*.log 2>/dev/null | head -1); echo "LOG=$L"
echo "== iterations"; grep -aE "iteration +[0-9]+/" "$L" 2>/dev/null | tail -2
echo "== tail 40"; tail -40 "$L" 2>/dev/null
echo "== payload stderr tail"; tail -25 $R/payload.stderr.log 2>&1
echo "== payload stdout tail"; tail -20 $R/payload.stdout.log 2>&1
echo "== done"
