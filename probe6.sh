#!/bin/bash
set -x
mkdir -p /workspace-SR006.nfs2/relocated-20261123
mv /workspace-SR006.nfs3/xandi281/hmoe-legacy-runtime-20260924 /workspace-SR006.nfs2/relocated-20261123/ 2>&1
sync
df -h /workspace-SR006.nfs3 /workspace-SR006.nfs2 2>&1 | tail -3
du -sh /workspace-SR006.nfs2/relocated-20261123/* 2>&1
echo done
