#!/bin/bash
echo "== df"; df -h /workspace-SR006.nfs3 /workspace-SR006.nfs2 /home/jovyan /tmp 2>&1
echo "== df -i"; df -i /workspace-SR006.nfs3 /workspace-SR006.nfs2 /home/jovyan 2>&1
echo "== root"; ls -la /workspace-SR006.nfs3 2>&1 | head -20
echo "== our"; ls -la /workspace-SR006.nfs3/hmoe-lionmoe-20261123 2>&1 | head -20
echo "== done"
