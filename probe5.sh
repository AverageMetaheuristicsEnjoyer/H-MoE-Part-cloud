#!/bin/bash
set -x
mkdir -p /workspace-SR006.nfs2/relocated-20261123
mv /workspace-SR006.nfs3/hmoe-lionmoe-20261116 /workspace-SR006.nfs2/relocated-20261123/ 2>&1
df -h /workspace-SR006.nfs3 /workspace-SR006.nfs2 2>&1 | tail -3
du -sh /workspace-SR006.nfs2/relocated-20261123/hmoe-lionmoe-20261116 2>&1
echo done
