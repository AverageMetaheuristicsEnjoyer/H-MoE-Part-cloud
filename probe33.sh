#!/bin/bash
python3 - <<'EOF'
import json,os
p='/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v042/partial-error.json'
if os.path.exists(p):
    d=json.load(open(p))
    for k in ('error_type','error','last_good_checkpoint','completed_stages','finished_at'):
        print(k,'=',d.get(k))
    print('--- traceback tail'); print((d.get('traceback') or '')[-1400:])
else: print('no partial-error.json')
EOF
echo "== done"
