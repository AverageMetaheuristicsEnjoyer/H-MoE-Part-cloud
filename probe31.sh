#!/bin/bash
echo "== df"; df -h /workspace-SR006.nfs3 /workspace-SR006.nfs2 /home/jovyan /tmp 2>&1 | tail -4
echo "== inodes"; df -i /workspace-SR006.nfs3 | tail -1
echo "== top-level"; timeout 200 du -shx /workspace-SR006.nfs3/* 2>/dev/null | sort -h | tail -8
echo "== our campaign"; timeout 120 du -shx /workspace-SR006.nfs3/hmoe-lionmoe-20261123/* 2>/dev/null | sort -h | tail -10
echo "== done"
