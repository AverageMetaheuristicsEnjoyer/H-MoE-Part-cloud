#!/usr/bin/env bash
set -euo pipefail

roots=(/home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3)

echo "=== capacity ==="
df -h "${roots[@]}" 2>&1
df -i "${roots[@]}" 2>&1

for root in "${roots[@]}"; do
  echo "=== usage depth 2: $root ==="
  du -x -h --max-depth=2 "$root" 2>/dev/null | sort -h | tail -60 || true

  echo "=== files over 1 GiB: $root ==="
  find "$root" -xdev -type f -size +1G \
    -printf '%s %u %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -nr | head -60 || true

  echo "=== checkpoint and upload markers: $root ==="
  find "$root" -xdev -maxdepth 7 -type f \
    \( -name 'latest_checkpointed_iteration.txt' -o -name '*manifest*.json' \
       -o -name '*.upload*' -o -name '*tracker*' \) \
    -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort | tail -100 || true
done

detail_roots=(
  /home/jovyan/rl_muon
  /home/jovyan/hmoe-checkpoints
  /home/jovyan/hmoe-cloud
  /workspace-SR006.nfs2/hmoe-checkpoints
  /workspace-SR006.nfs2/dimativator
  /workspace-SR006.nfs3/hmoe-checkpoints
  /workspace-SR006.nfs3/tucker-late-growth-20260827
)

for root in "${detail_roots[@]}"; do
  [[ -d $root ]] || continue
  echo "=== detailed usage depth 4: $root ==="
  du -x -h --max-depth=4 "$root" 2>/dev/null | sort -h | tail -100 || true

  echo "=== large-file inode/link detail: $root ==="
  find "$root" -xdev -type f -size +500M \
    -printf '%i %n %s %u %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null \
    | sort -k3,3nr | head -100 || true
done
