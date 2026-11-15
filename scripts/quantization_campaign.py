import importlib.util
import os
import json
import re
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
mode, *args = sys.argv[1:]
if not (root / '.git').exists():
    # The existing result writer records a real Git snapshot of the executed source.
    git_env = dict(os.environ, GIT_AUTHOR_DATE='2026-10-01T15:00:00+00:00',
                   GIT_COMMITTER_DATE='2026-10-01T15:00:00+00:00')
    for command in (['git', 'init', '-q'], ['git', 'add', '.'],
                    ['git', '-c', 'user.name=Quantization campaign', '-c',
                     'user.email=campaign@localhost', 'commit', '-qm', 'Executed encrypted source snapshot']):
        subprocess.run(command, cwd=root, env=git_env, check=True)
os.environ['PYTHONPATH'] = ':'.join(str(root / p) for p in ('third_party/Megatron-LM', 'third_party/emerging-optimizers', '.'))
os.environ.pop('PYTHONNOUSERSITE', None)
spec = importlib.util.find_spec('nvidia')
libs = [p for directory in (spec.submodule_search_locations or []) for p in Path(directory).glob('*/lib')] if spec else []
if libs:
    os.environ['LD_LIBRARY_PATH'] = ':'.join(map(str, libs)) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    for p in libs:
        for package, variable in [('cudnn', 'CUDNN_HOME'), ('curand', 'CURAND_HOME'), ('cuda_nvrtc', 'NVRTC_HOME')]:
            if p.parent.name == package:
                os.environ.setdefault(variable, str(p.parent))
os.environ['NVTE_FP8_BLOCK_SCALING_FP32_SCALES'] = '1'
if mode == 'test':
    subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.total', '--format=csv,noheader'], check=True)
    command = [sys.executable, '-m', 'pytest', '-v', '-rs', 'tests/stage3_moe/test_state_codec_selection.py', 'tests/stage3_moe/test_quantization_diagnostics.py', 'tests/stage3_moe/test_quantization_replay.py']
elif mode == 'coldstart':
    command = [sys.executable, str(root / 'scripts/coldstart_campaign.py'), *args]
elif mode == 'probe':
    command = ['bash', str(root / 'scripts/cloud_quantization_probe.sh'), *args]
elif mode == 'dense-diagnostics':
    command = [sys.executable, str(root / 'tools/dense_quantization_diagnostics.py'), *args]
elif mode == 'timing':
    command = [sys.executable, str(root / 'scripts/quantization_timing.py'), *args]
elif mode == 'rescore':
    command = [sys.executable, str(root / 'scripts/quantization_rescore.py'), *args]
elif mode == 'split-pending':
    command = [sys.executable, str(root / 'scripts/quantization_split.py')]
elif mode == 'collect':
    import base64
    import hashlib
    import io
    import tarfile
    target = Path('/workspace-SR006.nfs3/hmoe-cloud/coldstart-20261002')
    if args:
        target = (target / args[0]).resolve()
        if not target.is_relative_to('/workspace-SR006.nfs3/hmoe-cloud/coldstart-20261002'):
            raise ValueError('Collection path must remain inside the campaign')
    patterns = ('manifest-*.json', 'summary-*.jsonl', '*snapshot.jsonl', 'diagnostics.jsonl', 'replay.jsonl', '*-source.json',
                'results.jsonl', 'results-q1001*.json', 'timing-*.json', '*receipt*.json')
    if args:
        patterns += ('train-*.log', 'routing_telemetry.jsonl')
    files = sorted({p for pattern in patterns for p in target.rglob(pattern) if p.is_file()})
    stream = io.BytesIO()
    receipts = []
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for path in files:
            data = path.read_bytes()
            name = str(path.relative_to(target))
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
            receipts.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        data = json.dumps(receipts).encode()
        info = tarfile.TarInfo('collection-manifest.json')
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    payload = stream.getvalue()
    encoded = base64.b64encode(payload).decode()
    chunks = [encoded[i:i+12000] for i in range(0, len(encoded), 12000)]
    if len(chunks) > 150:
        raise RuntimeError('Collection exceeds bounded log transport; use a narrower snapshot')
    for index, chunk in enumerate(chunks):
        print(f'COLLECT_CHUNK {index} {chunk}', flush=True)
    print('COLLECT_RECEIPT ' + json.dumps({'files': len(files), 'chunks': len(chunks),
          'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}), flush=True)
    raise SystemExit(0)
elif mode == 'inspect':
    target = Path('/workspace-SR006.nfs3/hmoe-cloud/coldstart-20261002')
    subprocess.run(['df', '-h', '/tmp', '/workspace-SR006.nfs3'], check=False)
    for p in sorted(target.rglob('train-*.log')):
        lines = p.read_text(errors='replace').splitlines()
        step_lines = [line for line in lines if re.search(r'iteration\s+\d+/', line)]
        errors = [line for line in lines if any(k in line for k in ('Traceback (most recent call last)', 'Error:', 'AssertionError'))]
        print('PROGRESS ' + json.dumps({'log': str(p), 'last_step': step_lines[-1] if step_lines else None,
            'errors': errors[-3:], 'tail': lines[-2:]}), flush=True)
    for p in sorted(target.rglob('*receipt*.json')):
        print('INIT_RECEIPT ' + p.read_text(), flush=True)
    for p in sorted(target.rglob('summary-*.jsonl')):
        for line in p.read_text(errors='replace').splitlines():
            row = json.loads(line)
            row['step_count'] = len(row.pop('steps', []))
            print('SUMMARY ' + json.dumps(row), flush=True)
    for p in sorted(target.rglob('*snapshot.jsonl')):
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        print('DIAGNOSTIC ' + json.dumps({'file': str(p), 'rows': len(rows), 'last': rows[-1] if rows else None}), flush=True)
    for p in sorted(target.rglob('diagnostics.jsonl')):
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        print('DIAGNOSTIC ' + json.dumps({'file': str(p), 'rows': len(rows), 'completed_blocks': [r for r in rows if r['kind']=='live_block_complete']}), flush=True)
    for p in sorted(target.rglob('replay.jsonl')):
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        print('REPLAY ' + json.dumps({'file': str(p), 'rows': len(rows), 'last_step': max((r.get('local_step', 0) for r in rows), default=0), 'completed': [r for r in rows if r['kind'] in ('replay_complete','replay_block_complete')]}), flush=True)
    for p in sorted(target.rglob('results-q1001*.json')):
        print('RESCORE ' + json.dumps({'file': str(p), 'results': json.loads(p.read_text())}), flush=True)
    raise SystemExit(0)
else:
    raise SystemExit('Unknown campaign mode: ' + mode)
result = subprocess.run(command, cwd=root)
print('CAMPAIGN_EXIT=' + str(result.returncode), flush=True)
raise SystemExit(result.returncode)
