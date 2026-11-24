#!/bin/bash
R=/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v038/arms/lmlinrevr50
echo "== segment stderr"; tail -40 $R/segment-seg0001452-0002178.stderr.log 2>&1
echo "== segment stdout tail"; tail -30 $R/segment-seg0001452-0002178.stdout.log 2>&1
echo "== train log tail"; L=$(ls -t $R/logs/*/train-*.log 2>/dev/null | head -1); echo "LOG=$L"; tail -25 "$L" 2>&1
echo "== done"
