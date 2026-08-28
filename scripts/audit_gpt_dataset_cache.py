#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_root(label, root):
    rows = []
    for description_path in sorted(root.rglob("*-GPTDataset-train-description.txt")):
        description = json.loads(description_path.read_text())
        stem = description_path.name.removesuffix("description.txt")
        paths = {
            name: description_path.with_name(f"{stem}{name}_index.npy")
            for name in ("document", "sample", "shuffle")
        }
        if not all(path.is_file() for path in paths.values()):
            continue
        document = np.load(paths["document"], mmap_mode="r")
        sample = np.load(paths["sample"], mmap_mode="r")
        shuffle = np.load(paths["shuffle"], mmap_mode="r")
        ordered = np.sort(np.asarray(shuffle))
        shuffle_is_permutation = bool(
            np.array_equal(ordered, np.arange(len(shuffle), dtype=shuffle.dtype))
        )
        sample_is_monotonic = bool(
            sample.ndim == 2
            and sample.shape[1] == 2
            and np.all(
                (sample[1:, 0] > sample[:-1, 0])
                | (
                    (sample[1:, 0] == sample[:-1, 0])
                    & (sample[1:, 1] >= sample[:-1, 1])
                )
            )
        )
        row = {
            "label": label,
            "description_path": str(description_path),
            "description_hash": stem.split("-", 1)[0],
            "dataset_path": description.get("dataset_path"),
            "num_samples": description.get("num_samples"),
            "random_seed": description.get("random_seed"),
            "sequence_length": description.get("sequence_length"),
            "document_index": {
                "shape": list(document.shape),
                "dtype": str(document.dtype),
                "minimum": int(document.min()),
                "maximum": int(document.max()),
                "sha256": sha256_file(paths["document"]),
            },
            "sample_index": {
                "shape": list(sample.shape),
                "dtype": str(sample.dtype),
                "monotonic": sample_is_monotonic,
                "sha256": sha256_file(paths["sample"]),
            },
            "shuffle_index": {
                "shape": list(shuffle.shape),
                "dtype": str(shuffle.dtype),
                "is_permutation": shuffle_is_permutation,
                "sha256": sha256_file(paths["shuffle"]),
            },
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no complete GPTDataset train caches under {root}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--extension-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "base": audit_root("base", args.base_root),
        "extension": audit_root("extension", args.extension_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"REPORT={args.output}")
    print("GPT_CACHE_AUDIT=pass")


if __name__ == "__main__":
    main()
