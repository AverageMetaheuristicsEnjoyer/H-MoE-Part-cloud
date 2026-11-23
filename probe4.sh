#!/bin/bash
echo "== xandi281 (32G)"; timeout 200 du -shx /workspace-SR006.nfs3/xandi281/* 2>/dev/null | sort -h | tail -12
echo "== rebuttal-20261123"; timeout 120 du -shx /workspace-SR006.nfs3/rebuttal-20261123/* 2>/dev/null | sort -h | tail -8
echo "== hmoe-cloud"; timeout 120 du -shx /workspace-SR006.nfs3/hmoe-cloud/* 2>/dev/null | sort -h | tail -8
echo "== hmoe-lionmoe-20261116"; timeout 120 du -shx /workspace-SR006.nfs3/hmoe-lionmoe-20261116/* 2>/dev/null | sort -h | tail -8
echo "== mlsub-logs count"; ls /workspace-SR006.nfs3/mlsub-logs 2>/dev/null | wc -l
echo "== done"
