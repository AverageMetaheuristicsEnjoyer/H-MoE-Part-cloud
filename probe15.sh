#!/bin/bash
R=/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v036
echo "== partial-error"
python3 - <<'EOF'
import json
p='/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v036/partial-error.json'
d=json.load(open(p))
for k in ('error_type','error','last_good_checkpoint','mode','arm','completed_stages','finished_at'):
    print(k, '=', d.get(k))
print('--- traceback (tail 1500)')
print((d.get('traceback') or '')[-1500:])
tails=d.get('segment_log_tails') or {}
for name,txt in list(tails.items())[:2]:
    print('--- segment', name, '(tail 800)')
    print((txt or '')[-800:])
EOF
echo "== arms"; ls -la $R/arms/* 2>&1 | head -24
echo "== hf receipts"; ls -la $R/arms/*/hf-receipt* 2>&1 | head -8
echo "== done"
