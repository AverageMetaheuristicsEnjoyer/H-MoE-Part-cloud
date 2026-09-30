import hashlib, io, json, os, subprocess, sys, tarfile, tempfile, traceback
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text())
try:
    key = os.environ.pop("BUNDLE_KEY")
    payload = (root / "payload.fernet").read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["payload_sha256"]:
        raise RuntimeError("Encrypted payload checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="broad-downstream-runner-", dir="/tmp") as directory:
        work = Path(directory)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--only-binary=:all:",
                        "--target", str(work / "deps"), "cryptography==46.0.5"], check=True)
        sys.path.insert(0, str(work / "deps"))
        from cryptography.fernet import Fernet
        plaintext = Fernet(key.encode()).decrypt(payload)
        del key
        with tarfile.open(fileobj=io.BytesIO(plaintext), mode="r:gz") as archive:
            archive.extractall(work / "source", filter="data")
        del plaintext
        plan = json.loads((work / "source" / "plan.json").read_text())
        for name, digest in plan["source_files_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("Evaluation source mismatch: " + name)
        if sys.argv[1:] == ["--verify-only"]:
            print("RUNNER_AUTHENTICATED=PASS", flush=True)
        else:
            result = subprocess.run([sys.executable, str(work / "source" / manifest["entry"]), *sys.argv[1:]])
            print("BROAD_DOWNSTREAM_EXIT=" + str(result.returncode), flush=True)
except Exception:
    traceback.print_exc()
    print("BROAD_DOWNSTREAM_EXIT=1", flush=True)
