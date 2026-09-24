"""Verified rolling checkpoint storage for the bounded SlimAdam Cloud experiment."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

REPO = 'AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints'


def archive(checkpoint, iteration, prefix, api):
    source = checkpoint / f'iter_{iteration:07d}'
    staging = checkpoint.parent / f'.{checkpoint.name}-archive-{iteration}'
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(source, staging, copy_function=os.link)
    files = {}
    for path in staging.rglob('*.pt'):
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        files[str(path.relative_to(staging))] = [path.stat().st_size, digest]
    if not files or any(size == 0 for size, _ in files.values()):
        raise RuntimeError('Empty checkpoint')
    remote = f'{prefix}/{source.name}'
    api.upload_folder(repo_id=REPO, repo_type='model', folder_path=str(staging),
                      path_in_repo=remote, allow_patterns=['*.pt'])
    entries = {entry.path.removeprefix(remote + '/'): entry for entry in api.list_repo_tree(
        REPO, path_in_repo=remote, repo_type='model', recursive=True
    ) if getattr(entry, 'size', None) is not None}
    for name, (size, digest) in files.items():
        entry = entries.get(name)
        lfs = getattr(entry, 'lfs', None)
        remote_hash = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
        if entry is None or entry.size != size or remote_hash != digest:
            raise RuntimeError(f'Checkpoint verification failed: {remote}/{name}')
    pointer = {'iteration': iteration, 'files': files}
    api.upload_file(repo_id=REPO, repo_type='model', path_in_repo=f'{prefix}/resume.json',
                    path_or_fileobj=json.dumps(pointer).encode())
    shutil.rmtree(staging)
    print(f'HF_VERIFIED iteration={iteration} path={remote}', flush=True)
    return pointer


def restore(checkpoint, run_dir, prefix, api):
    from huggingface_hub import hf_hub_download
    if not api.file_exists(REPO, f'{prefix}/resume.json', repo_type='model'):
        if any(run_dir.glob('train-*.log')):
            raise RuntimeError('Prior training has no archived checkpoint; refusing fresh restart')
        return None
    metadata = checkpoint.parent / 'restore-metadata'
    pointer_path = hf_hub_download(REPO, f'{prefix}/resume.json', repo_type='model',
                                   local_dir=metadata, token=api.token, force_download=True)
    pointer = json.loads(Path(pointer_path).read_text())
    for name, (size, digest) in pointer['files'].items():
        relative = f"{prefix}/iter_{pointer['iteration']:07d}/{name}"
        downloaded = Path(hf_hub_download(REPO, relative, repo_type='model',
                                         local_dir=checkpoint.parent / 'restore', token=api.token))
        with downloaded.open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        if downloaded.stat().st_size != size or actual != digest:
            raise RuntimeError('Restored checkpoint failed hash/size verification')
        target = checkpoint / f"iter_{pointer['iteration']:07d}" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded), target)
    (checkpoint / 'latest_checkpointed_iteration.txt').write_text(str(pointer['iteration']))
    print(f"HF_RESTORED iteration={pointer['iteration']}", flush=True)
    return pointer


def train(command, env, cwd, checkpoint, run_dir, prefix):
    from huggingface_hub import HfApi
    api = HfApi(token=env['HF_TOKEN'])
    previous = restore(checkpoint, run_dir, prefix, api)
    verified = previous['iteration'] if previous else 0
    process = subprocess.Popen(command, env=env, cwd=cwd)
    tracker = checkpoint / 'latest_checkpointed_iteration.txt'
    while True:
        code = process.poll()
        value = tracker.read_text().strip() if tracker.exists() else ''
        latest = int(value) if value.isdigit() else 0
        if latest > verified:
            try:
                previous = archive(checkpoint, latest, prefix, api)
                old = verified
                verified = latest
                if old and old != 2254:
                    api.delete_folder(repo_id=REPO, repo_type='model',
                                      path_in_repo=f'{prefix}/iter_{old:07d}')
            except Exception as error:
                print(f'HF_RETRY iteration={latest} error_type={type(error).__name__}', flush=True)
                time.sleep(30)
                continue
        if code is not None:
            if code != 0:
                raise RuntimeError(f'Training exited {code}; archived iteration={verified}')
            if latest != 2254 or verified != 2254:
                raise RuntimeError('Training ended without a verified 2254-step checkpoint')
            return
        time.sleep(30)
