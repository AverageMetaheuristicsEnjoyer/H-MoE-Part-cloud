#!/bin/bash
echo "== host"; hostname; nvidia-smi -L 2>/dev/null | head -3
echo "== mounts"; mount 2>/dev/null | grep -Ei 'workspace|nfs|jovyan' | head -20
echo "== df"; df -h 2>/dev/null | grep -Ei 'workspace|nfs|jovyan|/tmp|/$'
echo "== df -i"; df -i 2>/dev/null | grep -Ei 'workspace|nfs|jovyan|/tmp'
echo "== root"; ls -la / | head -30
for base in /workspace-SR006.nfs3 /workspace-SR006.nfs2 /workspace-SR006.nfs1 /home/jovyan; do
  echo "== $base"; ls -la "$base" 2>&1 | head -20
done
echo "== our subtree"; du -sh /workspace-SR006.nfs3/hmoe-lionmoe-20261123 2>&1
echo "== per-run"; du -sh /workspace-SR006.nfs3/hmoe-lionmoe-20261123/* 2>&1 | sort -h | tail -20
echo "== root listing"; ls -la /workspace-SR006.nfs3/hmoe-lionmoe-20261123 2>&1 | head -30
echo "== v031 listing"; ls -la /workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v031 2>&1 | head -40
echo "== v031 arms"; du -sh /workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v031/arms/* 2>&1
echo "== top big"; timeout 150 du -shx /workspace-SR006.nfs3/* 2>/dev/null | sort -h | tail -10
echo "== tmp"; df -h /tmp; ls -la /tmp | head -15
echo "== done"
