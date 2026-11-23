#!/bin/bash
R=/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v036
echo "== df"; df -h /workspace-SR006.nfs3 | tail -1
echo "== root"; ls -la $R 2>&1 | head -15
echo "== stderr tail"; tail -60 $R/payload.stderr.log 2>&1
echo "== stdout tail"; tail -25 $R/payload.stdout.log 2>&1
echo "== receipts"; ls -la $R/arms/*/ 2>&1 | head -20
echo "== done"
