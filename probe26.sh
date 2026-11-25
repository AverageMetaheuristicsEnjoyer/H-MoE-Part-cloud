#!/bin/bash
R=/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v041/arms/lm3block
echo "== segment stderr"; tail -30 $R/segment-seg0001452-0002178.stderr.log 2>&1
echo "== segment stdout tail"; tail -20 $R/segment-seg0001452-0002178.stdout.log 2>&1
echo "== train log errors"
L=$(ls -t $R/logs/*/train-*.log 2>/dev/null | head -1); echo "LOG=$L"
grep -aE "Error|error|Traceback|out of memory|CUDA|NCCL|Assertion" "$L" 2>/dev/null | tail -20
echo "== done"
