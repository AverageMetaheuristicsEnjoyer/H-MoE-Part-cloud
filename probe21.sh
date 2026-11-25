#!/bin/bash
python3 - <<'EOF'
import json,os
p='/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v039/partial-error.json'
if os.path.exists(p):
    d=json.load(open(p))
    for k in ('error_type','error','last_good_checkpoint','mode','arm','completed_stages','finished_at'):
        print(k,'=',d.get(k))
    print('--- traceback tail'); print((d.get('traceback') or '')[-1500:])
else: print('no partial-error.json')
EOF
R=/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v039/arms/lmlinrevr50
echo "== arm files"; ls -la $R 2>&1 | tail -12
echo "== continue-summary?"; ls -la /workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v039/continue-summary.json 2>&1
echo "== done"
