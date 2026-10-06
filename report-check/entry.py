"""Cloud.ru check entry for the Stage 3 MoE reproduction branch (private repository Huawei-stage2, branch stage3_moe).

The branch is private, so its tree travels as a Fernet-encrypted payload (report-check/payload.fernet, key in the
BUNDLE_KEY job environment). mlsub starts one MPI rank per GPU; rank 0 decrypts the tree and runs one command of it on
all GPUs of the pod. Arguments are KEY=VALUE tokens (mlsub allows only letters, digits and . _ : / = + , [ ] - in --args):

  payload=SHA8 mode=figure ids=9a,12a [smoke=1] [gpus=4] [revision=main]   scripts/run_report_figure.py
  payload=SHA8 mode=pairs [smoke=1] [only=adamw-coatopt,...]                scripts/run_report_figure.py --downstream-pairs
  payload=SHA8 mode=replot ids=9,16                                         scripts/report_figures.py --runs-dir _replot
  payload=SHA8 mode=plan ids=9,10                                          scripts/run_report_figure.py --plan
  payload=SHA8 mode=pytest [k=EXPR]                                        python -m pytest tests/stage3_moe
  payload=SHA8 mode=shell cmd=NAME                                         one of the fixed commands in SHELL below
  mode=inventory                                                           checkpoint files on the shared volumes
  mode=hfupload list=NAME                                                  upload the files of uploads/NAME.txt (HF_TOKEN env)
  mode=hfdelete list=NAME [confirm=1]                                      verify each file of uploads/NAME.txt on HF (sha256 +
                                                                           size of the LFS object on main); with confirm=1 delete
                                                                           only the verified local files

Results come back through the job log: a gzip+base64 tarball of the output directory (plots, result.json files,
provenance, run logs' tails) as RCHK lines plus an RRCP receipt; decode with decode_rchk.py.
"""
import re
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

if os.environ.get('OMPI_COMM_WORLD_RANK', '0') != '0':
    print('CHECK_IDLE_RANK=' + os.environ['OMPI_COMM_WORLD_RANK'], flush=True)
    raise SystemExit(0)

DATA_CANDIDATES = ['/home/jovyan/data/fineweb-edu-gpt2-megatron',
                   '/workspace-SR006.nfs2/hmoe-data/fineweb-edu-gpt2-megatron',
                   '/workspace-SR006.nfs3/hmoe-data/fineweb-edu-gpt2-megatron']
VOLUMES = ['/home/jovyan', '/workspace-SR006.nfs2', '/workspace-SR006.nfs3']
SHELL = {
    'tesource': 'python - <<"PY"\nimport importlib.util, pathlib\nroot = pathlib.Path(importlib.util.find_spec("transformer_engine").origin).parent\n'
                'for rel in ("pytorch/fp8.py", "common/recipe/__init__.py"):\n    p = root / rel\n    print("=== " + str(rel))\n'
                '    print(p.read_text() if p.exists() else "MISSING")\nPY',
    'env': 'python -c "import torch, transformer_engine, triton; print(torch.__version__, torch.version.cuda, '
           'transformer_engine.__version__, triton.__version__)"; nvidia-smi; df -h /tmp',
    'setup-check': 'bash setup.sh --check-only',
}


def sh(command, cwd=None, env=None, check=True):
    print(f'CHECK_RUN {command}', flush=True)
    result = subprocess.run(command, shell=True, cwd=cwd, env=env)
    print(f'CHECK_EXIT {result.returncode} {command[:120]}', flush=True)
    if check and result.returncode:
        raise SystemExit(result.returncode)
    return result.returncode


def decrypt(destination, expected):
    here = Path(__file__).resolve().parent
    candidate = here / 'payloads' / f'{expected}.fernet'
    payload = (candidate if candidate.exists() else here / 'payload.fernet').read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if not digest.startswith(expected):
        raise SystemExit(f'payload {digest[:8]} is not the requested {expected}')
    deps = destination.parent / 'deps'
    sh(f'{sys.executable} -m pip install -q --disable-pip-version-check --no-cache-dir --only-binary=:all: '
       f'--target {deps} cryptography==46.0.5')
    sys.path.insert(0, str(deps))
    from cryptography.fernet import Fernet
    plaintext = Fernet(os.environ.pop('BUNDLE_KEY').encode()).decrypt(payload)
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(plaintext), mode='r:gz') as archive:
        archive.extractall(destination, filter='data')
    print('CHECK_PAYLOAD ' + (destination / 'payload_manifest.json').read_text()[:300].replace('\n', ' '), flush=True)
    # result_writer records `git rev-parse HEAD`: give the decrypted tree a local commit.
    sh(f'git -C {destination} init -q && git -C {destination} add -A && '
       f'git -C {destination} -c user.name=check -c user.email=check@localhost commit -qm payload-{expected}')


def hfupload(name):
    """Upload the checkpoint files listed in uploads/NAME.txt (lines: LOCAL_PATH REMOTE_PATH) and verify sha256 + size."""
    here = Path(__file__).resolve().parent
    sh('python -m pip install --user -q --disable-pip-version-check huggingface_hub hf_transfer 2>&1 | tail -1', check=False)
    import site
    sys.path.insert(0, site.getusersitepackages())
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ['HF_TOKEN'])
    repo = 'AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints'
    for line in (here / 'uploads' / f'{name}.txt').read_text().split('\n'):
        if not line.strip():
            continue
        local, remote = line.split()
        digest = hashlib.sha256()
        with open(local, 'rb') as handle:
            for block in iter(lambda: handle.read(1 << 24), b''):
                digest.update(block)
        sha, size = digest.hexdigest(), Path(local).stat().st_size
        for attempt in range(5):
            try:
                api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo, commit_message=f'legacy: {remote}')
                entry = next(e for e in api.list_repo_tree(repo, path_in_repo=str(Path(remote).parent), expand=True)
                             if e.path == remote)
                ok = entry.lfs is not None and entry.lfs.sha256 == sha and entry.size == size
                print(f'HFUP {remote} bytes={size} sha256={sha} verified={ok}', flush=True)
                break
            except Exception as error:
                print(f'HFUP_RETRY {remote} {type(error).__name__}', flush=True)
                time.sleep(120)


def hfdelete(name, confirm):
    sh('python -m pip install --user -q --disable-pip-version-check huggingface_hub 2>&1 | tail -1', check=False)
    import site
    sys.path.insert(0, site.getusersitepackages())
    from huggingface_hub import HfApi
    api = HfApi()
    repo = 'AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints'
    here = Path(__file__).resolve().parent
    for line in (here / 'uploads' / f'{name}.txt').read_text().split('\n'):
        if not line.strip():
            continue
        local, remote = line.split()
        if not Path(local).is_file():
            print(f'HFDEL_ABSENT {local}', flush=True)
            continue
        digest = hashlib.sha256()
        with open(local, 'rb') as handle:
            for block in iter(lambda: handle.read(1 << 24), b''):
                digest.update(block)
        sha, size = digest.hexdigest(), Path(local).stat().st_size
        entries = [e for e in api.list_repo_tree(repo, path_in_repo=str(Path(remote).parent), expand=True) if e.path == remote]
        ok = bool(entries) and entries[0].lfs is not None and entries[0].lfs.sha256 == sha and entries[0].size == size
        print(f'HFDEL_CHECK {remote} bytes={size} sha256={sha} on_hf={ok}', flush=True)
        if ok and confirm:
            Path(local).unlink()
            print(f'HFDEL_REMOVED {local}', flush=True)


def emit(directory):
    """Print a tarball of the small outputs as base64 chunks."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for path in sorted(Path(directory).rglob('*')):
            if not path.is_file():
                continue
            if path.suffix in ('.pt', '.bin', '.idx', '.safetensors') or path.stat().st_size > 20_000_000:
                continue
            if path.suffix in ('.log', '.txt') and path.stat().st_size > 2_000_000:
                tail = path.read_bytes()[-2_000_000:]
                info = tarfile.TarInfo(str(path.relative_to(directory)) + '.tail')
                info.size = len(tail)
                archive.addfile(info, io.BytesIO(tail))
                continue
            archive.add(path, arcname=str(path.relative_to(directory)))
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode()
    chunks = [encoded[i:i + 12000] for i in range(0, len(encoded), 12000)]
    for index, chunk in enumerate(chunks):
        print(f'RCHK {index} {chunk}', flush=True)
    print('RRCP ' + json.dumps({'chunks': len(chunks), 'bytes': len(payload),
                                'sha256': hashlib.sha256(payload).hexdigest()}), flush=True)


def main():
    args = dict(token.split('=', 1) for token in sys.argv[1:] if '=' in token)
    mode = args.get('mode', 'figure')
    print('CHECK_ARGS ' + json.dumps(args), flush=True)
    sh('nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader', check=False)
    if mode == 'hfupload':
        hfupload(args['list'])
        return
    if mode == 'hfdelete':
        hfdelete(args['list'], args.get('confirm') == '1')
        return
    if mode == 'logtail':
        job = re.sub(r'[^a-z0-9-]', '', args['id'])
        for base in ('/home/jovyan/shares/SR006.nfs2/mlsub-logs', '/workspace-SR006.nfs2/mlsub-logs'):
            sh(f"date -u; find {base} -path '*{job}*' -type f -exec ls -la --time-style=full-iso {{}} + 2>&1 | head",
               check=False)
            sh(f"find {base} -path '*{job}*' -type f -name stdout 2>/dev/null | head -2 | while read f; do "
               f"tail -c 400000 \"$f\" | grep -av RCHK | tail -40 | cut -c1-300; "
               f"grep -av RCHK \"$f\" | grep -aE 'CHECK_|Traceback|Error|FAILED|failed|status' | tail -80 | cut -c1-300; done",
               check=False)
        return
    if mode == 'inventory':
        for volume in VOLUMES:
            sh(f"find {volume} -xdev \\( -name '*.pt' -o -name 'model_optim_rng.pt' -o -name 'latest_checkpointed_iteration.txt' \\) "
               f"-size +100M -printf 'INV %s %TY-%Tm-%Td %p\\n' 2>/dev/null | sort -k4 | head -2000", check=False)
            sh(f"df -h {volume}", check=False)
        return
    work = Path('/tmp/report-check')
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    repo = work / 'repo'
    decrypt(repo, args['payload'])
    env = dict(os.environ)
    env.pop('PYTHONNOUSERSITE', None)
    env['HF_HUB_ENABLE_HF_TRANSFER'] = env.get('HF_HUB_ENABLE_HF_TRANSFER', '1')
    env['HF_HOME'] = str(work / 'hf-home')
    # The te4 image carries torch / TE / triton; the rest is installed into the user site of this disposable pod.
    sh('python -m pip install --user -q --disable-pip-version-check "torch==2.8.0" lm-eval==0.4.11 '
       'nvidia-cuda-cccl-cu12==12.9.27 matplotlib hf_transfer 2>&1 | tail -3', env=env, check=False)
    data = next((d for d in DATA_CANDIDATES if Path(d, 'data', 'train.bin').is_file()), None)
    print(f'CHECK_DATA {data}', flush=True)
    out = work / 'out'
    common = f'--data-root {data} --checkpoint-dir {work}/ckpt --output-dir {out}'
    if args.get('revision'):
        common += f' --checkpoint-revision {args["revision"]}'
    started = time.time()
    code = 0
    try:
        if mode == 'figure':
            ids = ' '.join(args['ids'].split(','))
            extra = (' --smoke' if args.get('smoke') == '1' else '') + (f' --gpus {args["gpus"]}' if args.get('gpus') else '')
            code = sh(f'python scripts/run_report_figure.py {ids} {common}{extra}', cwd=repo, env=env, check=False)
        elif mode == 'pairs':
            endpoints = ' '.join(args.get('only', '').split(','))
            extra = ' --smoke' if args.get('smoke') == '1' else ''
            code = sh(f'python scripts/run_report_figure.py --downstream-pairs {endpoints} {common}{extra}', cwd=repo, env=env, check=False)
        elif mode == 'plan':
            ids = ' '.join(args['ids'].split(','))
            code = sh(f'python scripts/run_report_figure.py {ids} --plan {common}', cwd=repo, env=env, check=False)
        elif mode == 'pytest':
            sh('python -m pip install --user -q pytest 2>&1 | tail -1', env=env, check=False)
            k = f' -k "{args["k"]}"' if args.get('k') else ''
            out.mkdir(parents=True, exist_ok=True)
            code = sh(f'python -m pytest -q tests/stage3_moe{k} 2>&1 | tee {out}/pytest.log | tail -60', cwd=repo, env=env, check=False)
        elif mode == 'replot':
            ids = ' '.join(args['ids'].split(','))
            runs = '' if args.get('archived') == '1' else ' --runs-dir _replot'
            code = sh(f'python scripts/report_figures.py --figures {ids}{runs} --output-dir {out}',
                      cwd=repo, env=env, check=False)
        elif mode == 'shell':
            code = sh(SHELL[args['cmd']], cwd=repo, env=env, check=False)
    finally:
        print(f'CHECK_DONE mode={mode} exit={code} seconds={int(time.time() - started)}', flush=True)
        if out.exists():
            emit(out)


main()
