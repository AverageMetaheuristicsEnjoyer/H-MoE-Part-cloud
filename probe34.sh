#!/bin/bash
set -eu
export PYTHONUSERBASE=/home/jovyan/.local-te4
export PATH=$PYTHONUSERBASE/bin:$PATH
JOB=${JOB_ID:?}
D=/home/jovyan/mlsub-logs/$JOB
OUT=/tmp/diag-$JOB.txt
{
  echo "JOB=$JOB"
  find "$D" -maxdepth 4 -type f 2>&1 | head -20
  for f in $(find "$D" -maxdepth 4 -type f \( -name 'stdout' -o -name 'stderr' -o -name '*.log' \) 2>/dev/null | head -6); do
    echo "=== FILE $f"
    tail -120 "$f"
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
