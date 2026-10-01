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
role = os.environ.get('BUNDLE_ROLE', '')
wait = float(os.environ.get('BUNDLE_WAIT_SECONDS', '0'))
here = Path(__file__).resolve().parent
repo = here.parent
branch = subprocess.run(['git', '-C', str(repo), 'rev-parse', '--abbrev-ref', 'HEAD'],
                        capture_output=True, text=True, check=True).stdout.strip()
started = time.time()
while True:
    # Late binding: always run the newest payload the branch carries, once its role is released.
    fetched = subprocess.run(['git', '-C', str(repo), 'fetch', '-q', '--depth', '1', 'origin', branch],
                             capture_output=True, text=True)
    if fetched.returncode == 0:
        subprocess.run(['git', '-C', str(repo), 'checkout', '-q', 'FETCH_HEAD', '--', here.name], check=True)
    public_manifest = json.loads((here / 'manifest.json').read_text())
    if not role or role in public_manifest.get('ready', []):
        break
    if time.time() - started > wait:
        print('BUNDLE_ROLE_NOT_READY=' + role, flush=True)
        raise SystemExit(0)
    time.sleep(120)
payload = (here / 'payload.fernet').read_bytes()
if hashlib.sha256(payload).hexdigest() != public_manifest['payload_sha256']:
    raise RuntimeError('Encrypted payload checksum mismatch')
with tempfile.TemporaryDirectory(prefix='encrypted-job-', dir='/tmp') as directory:
    destination = Path(directory)
    deps = destination / 'dependencies'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--disable-pip-version-check',
                    '--no-cache-dir', '--only-binary=:all:', '--target', str(deps),
                    'cryptography==46.0.5'], check=True)
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
    print('BUNDLE_AUTHENTICATED=' + manifest['bundle_id'] + ' ROLE=' + role
          + ' WAITED_SECONDS=' + str(int(time.time() - started)), flush=True)
    if sys.argv[1:] != ['--verify-only']:
        os.environ['BUNDLE_ID'] = manifest['bundle_id']
        raise SystemExit(subprocess.call(manifest['entry'] + sys.argv[1:], cwd=source))
