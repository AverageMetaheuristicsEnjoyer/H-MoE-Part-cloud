#!/bin/bash
R=/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v041/arms/lm3block
L=$(ls -t $R/logs/*/train-*.log 2>/dev/null | head -1); echo "LOG=$L"
echo "== last iterations"; grep -aE "iteration +[0-9]+/" "$L" | tail -3
echo "== last 60 lines"; tail -60 "$L"
echo "== abort markers"; grep -anE "terminate called|what\(\)|CUDA error|Assertion|assert|OOM|out of memory|SIGABRT|Aborted" "$L" | tail -10
echo "== done"
