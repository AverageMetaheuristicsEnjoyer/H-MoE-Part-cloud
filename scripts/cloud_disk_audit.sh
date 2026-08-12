#!/usr/bin/env bash
# What is filling the persistent volume? Read-only.
set -u
echo "=== df ==="
df -h /home/jovyan | tail -1
echo
echo "=== top level ==="
du -sh /home/jovyan/* /home/jovyan/.[!.]* 2>/dev/null | sort -rh | head -25
echo
echo "=== largest 25 files ==="
find /home/jovyan -type f -size +100M -printf '%10s %p\n' 2>/dev/null | sort -rn | head -25
echo
echo "=== stage3 artifacts ==="
du -sh /home/jovyan/hmoe-cloud 2>/dev/null
find /home/jovyan/hmoe-cloud -name results.jsonl 2>/dev/null | wc -l
echo "EXIT=0"
exit 0
