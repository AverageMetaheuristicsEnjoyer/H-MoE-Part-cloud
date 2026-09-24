"""Relocate the old project toolkit with verified content and preserve its original path."""
import hashlib
import json
import os
from pathlib import Path
import shutil

source = Path('/home/jovyan/hmoe-cloud/cuda-12.1.1')
base = Path('/workspace-SR006.nfs3/xandi281/hmoe-legacy-runtime-20260924')
target = base / source.name
backup = source.with_name(source.name + '.relocating-20260924')


def manifest(root):
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            relative = str(path.relative_to(root))
            if path.is_symlink():
                result[relative] = {'symlink': os.readlink(path)}
            elif path.is_file():
                with path.open('rb') as handle:
                    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
                result[relative] = {'size': path.stat().st_size, 'sha256': digest}
    return result


def main():
    if source.is_symlink() and source.resolve() == target:
        print('RELOCATION_RESULT=ALREADY_COMPLETE', flush=True)
        return
    if not source.is_dir() or source.is_symlink() or backup.exists() or target.exists():
        raise RuntimeError('Unexpected source/destination state; refusing to overwrite')
    base.mkdir(parents=True, exist_ok=True)
    original = manifest(source)
    size = sum(item.get('size', 0) for item in original.values())
    if shutil.disk_usage(base).free < size + 2 * 1024**3:
        raise RuntimeError('Destination lacks toolkit size plus 2 GiB headroom')
    print(f'COPY source={source} destination={target} bytes={size}', flush=True)
    shutil.copytree(source, target, symlinks=True)
    if manifest(target) != original or manifest(source) != original:
        raise RuntimeError('Verification failed; source left intact')
    (base / 'cuda-12.1.1-manifest.json').write_text(json.dumps(original, sort_keys=True) + '\n')
    source.rename(backup)
    try:
        source.symlink_to(target, target_is_directory=True)
    except Exception:
        backup.rename(source)
        raise
    if source.resolve() != target or manifest(source) != original:
        raise RuntimeError('Linked path verification failed; original retained at backup')
    shutil.rmtree(backup)
    print(f'RELOCATION_RESULT=PASS files={len(original)} bytes={size}', flush=True)
    print(f'FREE_JOVYAN_BYTES={shutil.disk_usage(source.parent).free}', flush=True)
    print(f'FREE_NFS3_BYTES={shutil.disk_usage(base).free}', flush=True)


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    print('EXIT=1', flush=True)
else:
    print('EXIT=0', flush=True)
