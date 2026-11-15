#!/usr/bin/env python3

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


MEGATRON_COMMIT = "571370c829ca768fe37244f4e2e7f28d8accc4ab"
EOD_ID = 50_256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def split_records(root, split):
    if split == "train":
        manifest_path = root / "provenance/train-merge-manifest.json"
    else:
        manifest_path = root / f"provenance/{split}/manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    records = manifest["artifacts"]
    bin_records = [
        record for name, record in records.items() if name.endswith(".bin")
    ]
    idx_records = [
        record for name, record in records.items() if name.endswith(".idx")
    ]
    if len(bin_records) != 1 or len(idx_records) != 1:
        raise RuntimeError(f"unexpected artifact records for {split}")
    return bin_records[0], idx_records[0]


def main():
    args = parse_args()
    megatron_root = args.megatron_root.resolve()
    revision = subprocess.run(
        ["git", "-C", str(megatron_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != MEGATRON_COMMIT:
        raise RuntimeError(
            f"Megatron commit is {revision}, expected {MEGATRON_COMMIT}"
        )
    sys.path.insert(0, str(megatron_root))
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    for split in ("train", "development", "final"):
        prefix = args.artifact_root / f"data/{split}"
        bin_path = Path(f"{prefix}.bin")
        idx_path = Path(f"{prefix}.idx")
        bin_record, idx_record = split_records(args.artifact_root, split)
        for path, record in ((bin_path, bin_record), (idx_path, idx_record)):
            if (
                not path.is_file()
                or path.stat().st_size != record["size_bytes"]
                or sha256_file(path) != record["sha256"]
            ):
                raise RuntimeError(f"artifact differs: {path}")

        dataset = IndexedDataset(str(prefix), multimodal=False, mmap=True)
        documents = len(dataset.sequence_lengths)
        tokens = int(np.sum(dataset.sequence_lengths, dtype=np.int64))
        if (
            dataset.index.dtype != np.uint16
            or documents != bin_record["indexed_documents"]
            or tokens != bin_record["indexed_tokens"]
            or bin_path.stat().st_size != 2 * tokens
            or not np.array_equal(
                dataset.document_indices,
                np.arange(documents + 1, dtype=np.int64),
            )
            or int(dataset[0][-1]) != EOD_ID
            or int(dataset[documents - 1][-1]) != EOD_ID
        ):
            raise RuntimeError(f"indexed layout differs for {split}")
        print(
            f"{split}=pass documents={documents} indexed_tokens={tokens}"
            f" bin_sha256={bin_record['sha256']}"
            f" idx_sha256={idx_record['sha256']}"
        )


if __name__ == "__main__":
    main()
