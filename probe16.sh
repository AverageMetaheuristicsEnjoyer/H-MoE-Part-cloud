#!/bin/bash
R=/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v037
python3 - <<'EOF'
import json,glob,os
p='/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v037/partial-error.json'
if os.path.exists(p):
    d=json.load(open(p))
    for k in ('error_type','error','last_good_checkpoint','mode','arm','completed_stages','finished_at'):
        print(k,'=',d.get(k))
    print('--- traceback tail'); print((d.get('traceback') or '')[-1200:])
else:
    print('no partial-error.json')
EOF
echo "== receipts"; ls -la $R/arms/*/hf-receipt* 2>&1 | tail -5
echo "== df"; df -h /workspace-SR006.nfs3 | tail -1
echo "== done"
