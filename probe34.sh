#!/bin/bash
set -eu
JOB=${JOB_ID:?}
D=/home/jovyan/mlsub-logs/$JOB
OUT=/tmp/diag-$JOB.txt
{
  echo "JOB=$JOB"
  ls -la "$D" 2>&1
  for f in "$D"/*.log; do
    [ -f "$f" ] || continue
    echo "=== FILE $f"
    tail -250 "$f"
  done
} > "$OUT" 2>&1
wc -c "$OUT"
python3 - <<'EOF'
import os
from huggingface_hub import HfApi
job = os.environ['JOB_ID']
api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_file(path_or_fileobj=f'/tmp/diag-{job}.txt', path_in_repo=f'diag/job-{job}.txt',
                repo_id='AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints', repo_type='model',
                commit_message=f'diag log for {job}')
print('UPLOADED')
EOF
