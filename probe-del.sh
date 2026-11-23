#!/bin/bash
set -x
for p in /workspace-SR006.nfs3/xandi281/sae-parallel-wave-20260922-v1          /workspace-SR006.nfs3/xandi281/sae-parallel-wave-20260922-no-limit-v2          /workspace-SR006.nfs3/xandi281/sae-residual-pilot-20260922-v1          /workspace-SR006.nfs3/xandi281/dense-replication-qwen-20260916-v1          /workspace-SR006.nfs3/hmoe-cloud/coat-20261002; do
  if [ -d "$p" ]; then echo "REMOVING $p"; rm -rf -- "$p"; else echo "MISSING $p"; fi
done
df -h /workspace-SR006.nfs3 /workspace-SR006.nfs2 | tail -2
echo done
