#!/usr/bin/env bash
# Remove the throwaway smoke checkpoints; they filled the volume. Trunk/branch
# directories are left alone.
set -u
root=${STAGE3_MOE_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3}
echo "=== before ==="; df -h /workspace-SR006.nfs3 | tail -1
du -sh "$root"/* 2>/dev/null
echo "=== removing $root/smoke ==="
rm -rf "$root/smoke"
echo "=== after ==="; df -h /workspace-SR006.nfs3 | tail -1
du -sh "$root"/* 2>/dev/null
echo "EXIT=0"
exit 0
