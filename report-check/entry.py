"""Cloud.ru check entry for the public Stage 3 MoE reproduction branch.

mlsub starts one MPI rank per GPU; rank 0 clones github.com/AverageMetaheuristicsEnjoyer/Huawei-stage2 at REF and runs
one command of that public code on all GPUs of the pod. Arguments are KEY=VALUE tokens (mlsub allows only
letters, digits and . _ : / = + , [ ] - in --args):

  ref=SHA mode=figure ids=9a,12a [smoke=1] [gpus=4] [revision=main]   scripts/run_report_figure.py
  ref=SHA mode=pairs [smoke=1] [only=adamw-coatopt]                   scripts/run_report_figure.py --downstream-pairs
  ref=SHA mode=plan ids=9,10                                          scripts/run_report_figure.py --plan
  ref=SHA mode=pytest [k=EXPR]                                        python -m pytest tests/stage3_moe
  ref=SHA mode=shell cmd=NAME                                         one of the fixed commands in SHELL below
  mode=inventory                                                      checkpoint files on the shared volumes

Results come back through the job log: a gzip+base64 tarball of the output directory (plots, result.json files,
provenance, run logs' tails) as RCHK lines plus an RRCP receipt; decode with decode_rchk.py.
"""
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

REPO = 'https://github.com/AverageMetaheuristicsEnjoyer/Huawei-stage2'
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
    sh(f'git clone -q {REPO} -b stage3_moe {repo}')
    sh(f'git -C {repo} checkout -q {args["ref"]}')
    sh(f'git -C {repo} log --oneline -1')
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
            extra = (' --smoke' if args.get('smoke') == '1' else '') + (f' --only {args["only"]}' if args.get('only') else '')
            code = sh(f'python scripts/run_report_figure.py --downstream-pairs {common}{extra}', cwd=repo, env=env, check=False)
        elif mode == 'plan':
            ids = ' '.join(args['ids'].split(','))
            code = sh(f'python scripts/run_report_figure.py {ids} --plan {common}', cwd=repo, env=env, check=False)
        elif mode == 'pytest':
            sh('python -m pip install --user -q pytest 2>&1 | tail -1', env=env, check=False)
            k = f' -k "{args["k"]}"' if args.get('k') else ''
            out.mkdir(parents=True, exist_ok=True)
            code = sh(f'python -m pytest -q tests/stage3_moe{k} 2>&1 | tee {out}/pytest.log | tail -60', cwd=repo, env=env, check=False)
        elif mode == 'shell':
            code = sh(SHELL[args['cmd']], cwd=repo, env=env, check=False)
    finally:
        print(f'CHECK_DONE mode={mode} exit={code} seconds={int(time.time() - started)}', flush=True)
        if out.exists():
            emit(out)


main()
