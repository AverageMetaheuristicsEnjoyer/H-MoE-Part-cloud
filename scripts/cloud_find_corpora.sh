#!/usr/bin/env bash
# Find every Megatron-indexed corpus the job can reach, without assuming where it is.
#
#   mlsub run ... --entry scripts/cloud_find_corpora.sh --gpus cpu --no-pip
#
# Earlier passes missed things twice: they searched three hard-coded roots, and one
# used -xdev, which stops at every mount boundary -- and /home/jovyan/shares/* are
# separate mounts of the other volumes. So this one enumerates the mount table first
# and searches what it finds. Read-only.
set -u

echo "=== every mount, unfiltered ==="
df -hT 2>&1
echo
echo "--- mount table ---"
mount 2>/dev/null | grep -vE " (proc|sysfs|devpts|cgroup|mqueue|devtmpfs|securityfs|bpf|tracefs|debugfs|configfs|fusectl|pstore|hugetlbfs|binfmt_misc) " | head -40

echo
echo "=== shares/ ==="
ls -la /home/jovyan/shares 2>&1 | head -15

echo
echo "=== candidate roots ==="
# Every non-pseudo mountpoint, plus the usual suspects, deduplicated by device+inode
# so a volume mounted twice is searched once.
roots=$(df -P 2>/dev/null | awk 'NR>1 && $6 !~ /^\/(proc|sys|dev|run|etc\/|var\/lib\/kubelet)/ {print $6}' | sort -u)
echo "$roots"

echo
echo "=== every .idx over 1 MB and every .bin over 500 MB ==="
seen=""
for root in $roots /home/user /opt /srv /mnt; do
  [ -d "$root" ] || continue
  key=$(stat -c '%d:%i' "$root" 2>/dev/null) || continue
  case " $seen " in *" $key "*) echo "  (skip $root: already searched as another mountpoint)"; continue ;; esac
  seen="$seen $key"
  echo "--- $root"
  timeout 300 find "$root" -name '*.idx' -size +1M -printf '%10s %p\n' 2>/dev/null |
    grep -v '/\.git/' || true
  timeout 300 find "$root" -name '*.bin' -size +500M -printf '%10s %p\n' 2>/dev/null || true
done

echo
echo "=== anything called like the corpus, directories only ==="
for root in $roots; do
  timeout 180 find "$root" -maxdepth 8 -type d \
    \( -iname '*megatron*' -o -iname '*fineweb*' \) -printf '%p\n' 2>/dev/null |
    grep -viE '/\.git|site-packages|third_party|/src/' | head -20
done
echo "EXIT=0"
exit 0
