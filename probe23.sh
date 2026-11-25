#!/bin/bash
for V in 039 040; do
  for L in $(ls -t /workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v$V/arms/lm5r50ec/logs/*/train-*.log 2>/dev/null); do
    if grep -qa 'LIONMUON_STEP' "$L"; then
      echo "LOG=$L"
      grep -a 'LIONMUON_STEP' "$L" | head -40
      echo "--- count: $(grep -ac 'LIONMUON_STEP' "$L")"
      exit 0
    fi
  done
done
echo "none"
