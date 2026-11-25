#!/bin/bash
for A in lm5r50ec lmlinrevr50; do
  echo "### ARM $A"
  found=0
  for V in 035 036 037 038 039 040; do
    for L in $(ls -t /workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v$V/arms/$A/logs/*/train-*.log 2>/dev/null); do
      if grep -qa 'LIONMUON_STEP' "$L"; then
        echo "LOG=$L"
        grep -a -m2 'GPUS=' "$L"
        echo "--- first 30"
        grep -a 'LIONMUON_STEP' "$L" | head -30
        echo "--- last 15"
        grep -a 'LIONMUON_STEP' "$L" | tail -15
        echo "--- count: $(grep -ac 'LIONMUON_STEP' "$L")"
        found=1
        break
      fi
    done
    [ "$found" = "1" ] && break
  done
done
echo "### done"
