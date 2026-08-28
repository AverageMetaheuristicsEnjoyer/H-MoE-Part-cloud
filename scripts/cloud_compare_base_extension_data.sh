#!/usr/bin/env bash
# Read-only comparison of the persisted base and time-match extension artifacts.
set -euo pipefail

base=${STAGE3_MOE_ORIGINAL_DATA_ROOT:-/home/jovyan/data/fineweb-edu-gpt2-megatron}
extension=${STAGE3_MOE_EXTENSION_ROOT:-/workspace-SR006.nfs2/hmoe-data/fineweb-edu-time-match-extension}

python - "$base" "$extension" <<'PY'
import json
import struct
import sys
from pathlib import Path

import numpy as np


def conversion_records(root):
    records = []
    for path in root.rglob("*.json"):
        try:
            value = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or not {"conversion", "source", "tokenizer"} <= value.keys():
            continue
        records.append((path, value))
    return records


def stable_identity(value):
    conversion = value["conversion"]
    tokenizer = value["tokenizer"]
    source = value["source"]
    return {
        "source_repo": source["repo"],
        "source_revision": source["revision"],
        "tokenizer_repo": tokenizer["repo"],
        "tokenizer_revision": tokenizer["revision"],
        "vocab_size": tokenizer["vocab_size"],
        "eod_id": tokenizer["eod_id"],
        "datatrove_commit": conversion["datatrove_commit"],
        "document_order": conversion["document_order"],
        "normalization": conversion["normalization"],
        "implicit_special_tokens": conversion["implicit_special_tokens"],
        "append_eod_id": conversion["append_eod_id"],
        "token_dtype": conversion["token_dtype"],
        "batch_size": conversion["batch_size"],
        "tasks": conversion["tasks"],
    }


def summarize_records(label, root):
    records = conversion_records(root)
    if not records:
        raise RuntimeError(f"no conversion manifests under {root}")
    identities = {json.dumps(stable_identity(value), sort_keys=True) for _, value in records}
    if len(identities) != 1:
        raise RuntimeError(f"{label} has multiple conversion identities")
    environments = {
        json.dumps(value.get("environment", {}), sort_keys=True) for _, value in records
    }
    sources = sorted(value["source"]["path"] for _, value in records)
    print(f"{label}_conversion_manifests={len(records)}")
    print(f"{label}_identity={next(iter(identities))}")
    print(f"{label}_environments={json.dumps(sorted(environments))}")
    print(f"{label}_sources={json.dumps(sources)}")
    return next(iter(identities)), set(sources)


def index_stats(label, prefix):
    idx = Path(f"{prefix}.idx")
    binary = Path(f"{prefix}.bin")
    with idx.open("rb") as stream:
        if stream.read(9) != b"MMIDIDX\x00\x00":
            raise RuntimeError(f"bad index header: {idx}")
        version = struct.unpack("<Q", stream.read(8))[0]
        dtype_code = struct.unpack("<B", stream.read(1))[0]
        sequences = struct.unpack("<Q", stream.read(8))[0]
        documents = struct.unpack("<Q", stream.read(8))[0]
        lengths = np.fromfile(stream, dtype=np.int32, count=sequences)
    if version != 1 or dtype_code != 8 or documents != sequences + 1:
        raise RuntimeError(f"unexpected indexed layout: {idx}")
    tokens = int(lengths.sum(dtype=np.int64))
    if binary.stat().st_size != 2 * tokens:
        raise RuntimeError(f"uint16 payload size mismatch: {binary}")
    quantiles = np.quantile(lengths, [0.01, 0.1, 0.5, 0.9, 0.99]).tolist()
    offsets = np.cumsum(lengths, dtype=np.int64) - 1
    sample = np.linspace(0, sequences - 1, min(10000, sequences), dtype=np.int64)
    payload = np.memmap(binary, dtype=np.uint16, mode="r")
    eod_rate = float(np.mean(payload[offsets[sample]] == 50256))
    print(
        f"{label}_index version={version} dtype_code={dtype_code}"
        f" documents={sequences} tokens={tokens} mean_length={float(lengths.mean()):.6f}"
        f" quantiles={json.dumps(quantiles)} sampled_eod_rate={eod_rate:.6f}"
    )


base = Path(sys.argv[1])
extension = Path(sys.argv[2])
base_identity, base_sources = summarize_records("base", base)
extension_identity, extension_sources = summarize_records("extension", extension)
if base_identity != extension_identity:
    raise RuntimeError("base and extension conversion identities differ")
if base_sources & extension_sources:
    raise RuntimeError("base and extension source paths overlap")
index_stats("base", base / "data/train")
index_stats("extension", extension / "data/train")
print("DATA_PIPELINE_COMPARE=pass")
PY

echo "EXIT=0"
