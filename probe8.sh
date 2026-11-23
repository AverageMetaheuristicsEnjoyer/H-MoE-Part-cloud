#!/bin/bash
X=/workspace-SR006.nfs3/xandi281
echo "== xandi281 top"; ls -la $X 2>&1 | head -25
for d in dense-replication-qwen-20260916-v1 training-free-looped-moe kimi-recsys-corrected sae-parallel-wave-20260922-v1; do
  echo "== $d"; ls -la $X/$d 2>&1 | head -8
done
H=/workspace-SR006.nfs3/hmoe-cloud
echo "== hmoe-cloud top"; ls -la $H 2>&1 | head -20
C=$H/coat-20261002
echo "== coat-20261002"; ls -la $C 2>&1 | head -20
echo "== coat-20261002 sizes"; timeout 120 du -sh $C/* 2>/dev/null | sort -h | tail -10
echo "== coat-20261002 file types"; find $C -maxdepth 2 -type f 2>/dev/null | head -20
echo "== done"
