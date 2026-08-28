#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import numpy as np


def source_shards(root):
    artifact_manifest = json.loads((root / "artifact-manifest.json").read_bytes())
    selected = set(artifact_manifest["selected_source_paths"])
    rows = []
    for path in root.rglob("*.json"):
        try:
            value = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or not {"source", "conversion", "artifacts"} <= value.keys():
            continue
        match = re.fullmatch(r"data/train-(\d{5})-of-00100\.parquet", value["source"]["path"])
        if match is None or value["source"]["path"] not in selected:
            continue
        prefix = value["conversion"]["output_prefix"]
        artifact = value["artifacts"][f"{Path(prefix).name}.bin"]
        rows.append(
            {
                "shard": int(match.group(1)),
                "documents": int(artifact["indexed_documents"]),
                "tokens": int(artifact["indexed_tokens"]),
            }
        )
    rows.sort(key=lambda row: row["shard"])
    if not rows:
        raise RuntimeError(f"no source conversion manifests under {root}")
    return rows


def cache_arrays(root):
    descriptions = sorted((root / "data/train").rglob("*-GPTDataset-train-description.txt"))
    for description in descriptions:
        stem = description.name.removesuffix("description.txt")
        paths = {
            name: description.with_name(f"{stem}{name}_index.npy")
            for name in ("document", "sample", "shuffle")
        }
        if all(path.is_file() for path in paths.values()):
            return {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    raise RuntimeError(f"no complete GPTDataset cache under {root}")


def window_report(label, start, stop, shard_ids, expected):
    observed = np.bincount(shard_ids[start:stop], minlength=len(expected))
    fractions = observed / observed.sum()
    adjacent = shard_ids[start:stop]
    adjacent_same = float(np.mean(adjacent[1:] == adjacent[:-1]))
    expected_adjacent_same = float(np.sum(expected * expected))
    return {
        "label": label,
        "sample_start": start,
        "sample_stop": stop,
        "samples": stop - start,
        "observed_fraction": fractions.tolist(),
        "expected_token_fraction": expected.tolist(),
        "total_variation": float(0.5 * np.abs(fractions - expected).sum()),
        "maximum_absolute_difference": float(np.abs(fractions - expected).max()),
        "adjacent_same_shard_rate": adjacent_same,
        "expected_adjacent_same_shard_rate": expected_adjacent_same,
        "adjacent_same_shard_excess": adjacent_same - expected_adjacent_same,
    }


def audit_dataset(root, windows):
    shards = source_shards(root)
    arrays = cache_arrays(root)
    document = arrays["document"]
    sample = arrays["sample"]
    shuffle = arrays["shuffle"]
    maximum = max(stop for _, _, stop in windows)
    if maximum > len(shuffle):
        raise RuntimeError(f"requested {maximum} samples but cache has {len(shuffle)}")
    boundaries = np.cumsum([row["documents"] for row in shards], dtype=np.int64)
    if int(boundaries[-1]) != len(document):
        raise RuntimeError("conversion manifests and GPTDataset document index disagree")
    unshuffled_samples = np.asarray(shuffle[:maximum], dtype=np.int64)
    document_positions = np.asarray(sample[unshuffled_samples, 0], dtype=np.int64)
    document_ids = np.asarray(document[document_positions], dtype=np.int64)
    shard_ids = np.searchsorted(boundaries, document_ids, side="right")
    tokens = np.asarray([row["tokens"] for row in shards], dtype=np.float64)
    expected = tokens / tokens.sum()
    return {
        "root": str(root),
        "shards": shards,
        "windows": [
            window_report(label, start, stop, shard_ids, expected)
            for label, start, stop in windows
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--extension-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    global_batch = 208
    report = {
        "schema_version": 1,
        "method": "source shard of the starting document for every shuffled GPTDataset sample",
        "base": audit_dataset(
            args.base_root,
            [
                ("base_0_13794", 0, 13_794 * global_batch),
                ("base_13794_17242", 13_794 * global_batch, 17_242 * global_batch),
            ],
        ),
        "extension": audit_dataset(
            args.extension_root,
            [("extension_0_2328", 0, 2_328 * global_batch)],
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for dataset in (report["base"], report["extension"]):
        for window in dataset["windows"]:
            print(
                f"SHARD_MIX label={window['label']} samples={window['samples']}"
                f" tv={window['total_variation']:.8f}"
                f" max_abs={window['maximum_absolute_difference']:.8f}"
                f" adjacent_excess={window['adjacent_same_shard_excess']:.8f}"
            )
    print(f"REPORT={args.output}")
    print("GPT_SHARD_MIX_AUDIT=pass")


if __name__ == "__main__":
    main()
