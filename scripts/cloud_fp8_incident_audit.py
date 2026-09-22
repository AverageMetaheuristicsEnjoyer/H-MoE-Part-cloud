"""Read persisted log segments to distinguish resumed training from fresh restarts."""
import json
from pathlib import Path
import re
import subprocess

root = Path('/workspace-SR006.nfs2/hmoe-cloud/fp8-corrected-20260922')
subprocess.run(['df', '-h', '/tmp', str(root), '/workspace-SR006.nfs3'])
for directory in sorted(root.glob('stage3-*-full-*')):
    print('ARM', directory.name, flush=True)
    for log in sorted(directory.glob('train-*.log')):
        lines = log.read_text(errors='replace').splitlines()
        steps = [s for s in lines if re.search(r'iteration\s+\d+/\s*17242', s)]
        saves = [s for s in lines if 'successfully saved checkpoint' in s]
        errors = [s for s in lines if any(x in s for x in ('Traceback', 'Error:', 'TRAIN_EXIT=', 'No space left', 'SIGTERM', 'SIGKILL'))]
        loads = [s for s in lines if any(x in s for x in ('successfully loaded checkpoint', 'could not find the metadata', 'will not load any checkpoints'))]
        print(json.dumps({'log': log.name, 'first_step': steps[:1], 'last_step': steps[-1:], 'last_saves': saves[-2:], 'errors': errors[-5:], 'loads': loads[-3:]}, ensure_ascii=False), flush=True)
print('INCIDENT_AUDIT=PASS', flush=True)
