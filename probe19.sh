#!/bin/bash
L=$(ls -t /workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v038/arms/lmlinrevr50/logs/*/train-*.log 2>/dev/null | head -1)
echo "LOG=$L"
echo "== error lines"
grep -anE "Error|error|Traceback|out of memory|CUDA|NCCL|Assertion|Exception" "$L" 2>/dev/null | tail -25
echo "== context 120 lines before end"
tail -120 "$L" 2>/dev/null | head -90
echo "== done"
