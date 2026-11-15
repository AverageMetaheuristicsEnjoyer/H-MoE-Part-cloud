"""Stage pinned Dense FP32-state checkpoints one at a time, diagnose, and remove."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "DimaTivator/stage3_dense"
REVISION = "7f32f490b7ec4989056eb8ac90b20a36f3f3b58d"
SOURCES = {
    "adamw": ("adamw_fp8_act/pre_decay/main.pt", 5934381201, "78db70b2b5fd340143b8ade828a72c444c637f8354a5874aff5060cc3d50c848"),
    "muon": ("muon_fp8_act/pre_decay/main.pt", 4471554840, "01e93471dcd8f8a4aa2a54411c9727e87b2c02fb8edbcfa9b09ed991bdc3dfa1"),
    "ademamix": ("ademamix_fp8_act/pre_decay/main.pt", 7912455373, "3c80e1867832fc67e75b5cc8d9cee6bbca3eef648c6de8b915cf45780640d59d"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=tuple(SOURCES), default=list(SOURCES))
    parser.add_argument("--samples-per-stratum", type=int, default=3)
    parser.add_argument("--ademamix-replay-steps", type=int, default=0)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        source, expected_size, expected_sha = SOURCES[arm]
        output = args.output_root / f"{arm}-snapshot.jsonl"
        if output.exists():
            raise FileExistsError(f"existing diagnostic is preserved: {output}; use another output root")
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"dense-quant-{arm}-", dir="/tmp") as staging:
            replay_env = dict(os.environ)
            if args.ademamix_replay_steps:
                dependencies = Path(staging) / "python"
                subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(dependencies), "tiktoken==0.11.0"], check=True)
                replay_env["PYTHONPATH"] = str(dependencies) + os.pathsep + replay_env.get("PYTHONPATH", "")
                subprocess.run([sys.executable, str(ROOT / "tools/dense_ademamix_replay.py"), "--help"], env=replay_env, check=True)
            free = shutil.disk_usage(staging).free
            if free < expected_size + 2 * 1024 ** 3:
                raise RuntimeError(f"insufficient /tmp capacity: {free} free; need {expected_size + 2 * 1024 ** 3}")
            checkpoint = Path(staging) / "main.pt"
            url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{source}"
            digest, size, announced = hashlib.sha256(), 0, 0
            print(f"DENSE_DOWNLOAD arm={arm} revision={REVISION} bytes={expected_size}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, checkpoint.open("wb") as stream:
                while chunk := response.read(8 * 1024 ** 2):
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    if size - announced >= 1024 ** 3:
                        announced = size
                        print(f"DENSE_DOWNLOAD_PROGRESS arm={arm} bytes={size}", flush=True)
            actual_sha = digest.hexdigest()
            if size != expected_size or actual_sha != expected_sha:
                raise RuntimeError(f"checkpoint integrity mismatch for {arm}: bytes={size}, sha256={actual_sha}")
            receipt = {
                "arm": arm,
                "repo": REPO,
                "revision": REVISION,
                "path": source,
                "bytes": size,
                "sha256": actual_sha,
                "integrity": "verified_against_pinned_hf_lfs_metadata",
                "training_compute": "FP8",
                "training_state_precision": "FP32",
                "diagnostic_codec": "current MoE DRE/maxabs implementation on Dense state tensors",
                "limitation": (
                    "real-gradient recurrent replay at frozen Dense weights; fresh data cursor; "
                    "not historical failure reproduction or original Dense codec parity"
                    if args.ademamix_replay_steps else
                    "snapshot only; no Dense gradients, loss, or original Dense codec parity"
                ),
            }
            receipt_path = args.output_root / f"{arm}-source.json"
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            if args.ademamix_replay_steps:
                if arm != 'ademamix':
                    raise ValueError('Dense replay is defined for AdEMAMix only')
                subprocess.run([sys.executable, '-u', str(ROOT/'tools/dense_ademamix_replay.py'), str(checkpoint), '--output', str(args.output_root/'replay.jsonl'), '--steps', str(args.ademamix_replay_steps)], env=replay_env, check=True)
            else:
                subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "quantization_diagnostics.py"), str(checkpoint), "--output", str(output), "--samples-per-stratum", str(args.samples_per_stratum)], check=True)
        print(f"DENSE_ARM_RESULT=PASS arm={arm} temporary_checkpoint_removed=true elapsed_seconds={time.monotonic() - started:.1f}", flush=True)
    print("DENSE_DIAGNOSTICS=PASS", flush=True)


if __name__ == "__main__":
    main()
