import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

os.umask(0o077)
key = os.environ.pop('BUNDLE_KEY')
# mlsub starts one MPI rank per GPU and each runs this file; the payload drives every GPU itself.
if os.environ.get('OMPI_COMM_WORLD_RANK', '0') != '0':
    print('BUNDLE_IDLE_RANK=' + os.environ['OMPI_COMM_WORLD_RANK'], flush=True)
    raise SystemExit(0)
role = os.environ.get('BUNDLE_ROLE', '')
here = Path(__file__).resolve().parent
repo = here.parent
started = time.time()
expected_carrier_commit = os.environ['EXPECTED_CARRIER_COMMIT']
actual_carrier_commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                       capture_output=True, text=True, check=True).stdout.strip()
if actual_carrier_commit != expected_carrier_commit:
    raise RuntimeError(f'Carrier commit mismatch: {actual_carrier_commit} != {expected_carrier_commit}')
public_manifest = json.loads((here / 'manifest.json').read_text())
if role and role not in public_manifest.get('ready', []):
    print('BUNDLE_ROLE_NOT_READY=' + role, flush=True)
    raise SystemExit(0)
if public_manifest['bundle_id'] != os.environ['EXPECTED_BUNDLE_ID']:
    raise RuntimeError('Public bundle id does not match the expected id')
if public_manifest['payload_sha256'] != os.environ['EXPECTED_PAYLOAD_SHA256']:
    raise RuntimeError('Public payload hash does not match the expected hash')
payload = (here / 'payload.fernet').read_bytes()
actual_payload_sha256 = hashlib.sha256(payload).hexdigest()
if actual_payload_sha256 != public_manifest['payload_sha256']:
    raise RuntimeError('Encrypted payload checksum mismatch')
with tempfile.TemporaryDirectory(prefix='encrypted-job-', dir='/tmp') as directory:
    destination = Path(directory)
    deps = destination / 'dependencies'
    pip_started = time.monotonic()
    pip_result = subprocess.run(['timeout', '-k', '5s', '120s', sys.executable, '-m', 'pip',
                                 'install', '-q', '--disable-pip-version-check', '--no-cache-dir',
                                 '--only-binary=:all:', '--target', str(deps), 'cryptography==46.0.5'])
    pip_seconds = time.monotonic() - pip_started
    print(f'BOOTSTRAP_DEPENDENCY_EXIT={pip_result.returncode} BOOTSTRAP_DEPENDENCY_SECONDS={pip_seconds:.3f} '
          'BOOTSTRAP_DEPENDENCY_TIMEOUT=120 BOOTSTRAP_DEPENDENCY_TARGET=job_tmp', flush=True)
    if pip_result.returncode:
        raise RuntimeError(f'Isolated bootstrap dependency installation failed: {pip_result.returncode}')
    sys.path.insert(0, str(deps))
    from cryptography.fernet import Fernet
    plaintext = Fernet(key.encode()).decrypt(payload)
    del key
    source = destination / 'source'
    source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(plaintext), mode='r:gz') as archive:
        archive.extractall(source, filter='data')
    del plaintext
    manifest = json.loads((source / 'bundle_manifest.json').read_text())
    if manifest['bundle_id'] != public_manifest['bundle_id']:
        raise RuntimeError('Bundle id mismatch')
    if manifest['source_base'] != os.environ['EXPECTED_SOURCE_COMMIT'] or manifest['source_dirty']:
        raise RuntimeError('Source commit mismatch or dirty source manifest')
    archived_files = {path.relative_to(source).as_posix() for path in source.rglob('*') if path.is_file()}
    if archived_files != set(manifest['sha256']) | {'bundle_manifest.json'}:
        raise RuntimeError('Archived file set differs from the source manifest')
    for relative, expected in manifest['sha256'].items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Source file checksum mismatch: {relative}')
    print('BUNDLE_AUTHENTICATED=' + manifest['bundle_id'] + ' ROLE=' + role
          + ' WAITED_SECONDS=' + str(int(time.time() - started)), flush=True)
    print('CARRIER_COMMIT=' + actual_carrier_commit + ' SOURCE_COMMIT=' + manifest['source_base']
          + ' PAYLOAD_SHA256=' + actual_payload_sha256 + ' SOURCE_FILES_VERIFIED=' + str(len(manifest['sha256'])), flush=True)
    if sys.argv[1:] != ['--verify-only']:
        os.environ['BUNDLE_ID'] = manifest['bundle_id']
        raise SystemExit(subprocess.call(manifest['entry'] + sys.argv[1:], cwd=source))
