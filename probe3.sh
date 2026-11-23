#!/bin/bash
echo "== df"; df -h /workspace-SR006.nfs3 /workspace-SR006.nfs2 /home/jovyan 2>&1
echo "== top-level du (nfs3, maxdepth1)"
timeout 420 du -shx /workspace-SR006.nfs3/* 2>/dev/null | sort -h | tail -25
echo "== campaign dir"
for d in /workspace-SR006.nfs3/hmoe-lionmoe-20261123/*; do timeout 120 du -sh "$d" 2>/dev/null; done
echo "== campaign subdirs of fullrun-016-v031"
timeout 180 du -sh /workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v031/* 2>/dev/null
echo "== logs dir"
timeout 120 du -sh /workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v031/arms/*/logs 2>/dev/null
echo "== done"
