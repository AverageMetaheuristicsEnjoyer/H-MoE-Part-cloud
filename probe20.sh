#!/bin/bash
python3 - <<'EOF'
import json,os
p='/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v039/partial-error.json'
if os.path.exists(p):
    d=json.load(open(p))
    for k in ('error_type','error','last_good_checkpoint','mode','arm','completed_stages','finished_at'):
        print(k,'=',d.get(k))
    print('--- traceback tail'); print((d.get('traceback') or '')[-1800:])
else: print('no partial-error.json')
EOF
echo "== receipts"; ls -la /workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v039/arms/lmlinrevr50/hf-receipt* 2>&1 | tail -4
echo "== df"; df -h /workspace-SR006.nfs3 | tail -1
echo "== done"
