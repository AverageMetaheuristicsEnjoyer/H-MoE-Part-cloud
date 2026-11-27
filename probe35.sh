#!/bin/bash
set -eu
export PYTHONUSERBASE=/home/jovyan/.local-te4
export PATH=$PYTHONUSERBASE/bin:$PATH
V=${VERSION:?}
R=/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v$V
OUT=/tmp/pe-$V.txt
python3 - "$R" "$OUT" <<'EOF'
import json, sys, os
root, out = sys.argv[1], sys.argv[2]
p = os.path.join(root, 'partial-error.json')
with open(out, 'w') as fh:
    if not os.path.exists(p):
        fh.write('no partial-error.json\n'); sys.exit(0)
    d = json.load(open(p))
    for k in ('error_type','error','last_good_checkpoint','completed_stages','finished_at'):
        fh.write(f'{k} = {d.get(k)}\n')
    fh.write('--- traceback tail\n' + (d.get('traceback') or '')[-1800:])
EOF
wc -c "$OUT"
python3 - "$V" "$OUT" <<'EOF'
import os, sys
from huggingface_hub import HfApi
v, out = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_file(path_or_fileobj=out, path_in_repo=f'diag/partial-error-v{v}.txt',
                repo_id='AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints', repo_type='model',
                commit_message=f'partial error v{v}')
print('UPLOADED')
EOF
