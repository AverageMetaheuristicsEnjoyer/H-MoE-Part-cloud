#!/bin/bash
set -eu
export PYTHONUSERBASE=/home/jovyan/.local-te4
export PATH=$PYTHONUSERBASE/bin:$PATH
P=${PATH_TO_READ:?}
L=${LABEL:?}
OUT=/tmp/relay-$L.txt
{ echo "PATH=$P"; ls -la "$P" 2>&1 | head -5; echo '--- tail'; tail -120 "$P" 2>&1; } > "$OUT"
wc -c "$OUT"
python3 - "$L" "$OUT" <<'EOF'
import os, sys
from huggingface_hub import HfApi
label, out = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_file(path_or_fileobj=out, path_in_repo=f'diag/{label}.txt',
                repo_id='AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints', repo_type='model',
                commit_message=f'relay {label}')
print('UPLOADED')
EOF
