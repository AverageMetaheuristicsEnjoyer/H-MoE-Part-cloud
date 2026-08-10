#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


MEGATRON_COMMIT = "571370c829ca768fe37244f4e2e7f28d8accc4ab"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-root", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    from megatron.core.datasets import indexed_dataset

    module_path = Path(indexed_dataset.__file__).resolve()
    if not module_path.is_relative_to(megatron_root):
        raise RuntimeError(f"loaded Megatron from {module_path}, not {megatron_root}")

    selection_bytes = args.selection.read_bytes()
    selection = json.loads(selection_bytes)
    if selection["schema_version"] != 1:
        raise RuntimeError("unsupported train-selection schema")
    input_prefixes = [
        args.selection.parent / record["output_prefix"]
        for record in selection["selected_shards"]
    ]
    if not input_prefixes:
        raise RuntimeError("selection contains no input prefixes")

    input_documents = 0
    input_tokens = 0
    input_records = []
    for prefix, record in zip(input_prefixes, selection["selected_shards"]):
        manifest_path = args.selection.parent / record["manifest"]
        manifest_bytes = manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != record["manifest_sha256"]:
            raise RuntimeError(f"input manifest hash mismatch: {manifest_path}")
        manifest = json.loads(manifest_bytes)
        if manifest["source"]["path"] != record["source_path"]:
            raise RuntimeError(f"input source mismatch: {manifest_path}")
        manifest_prefix = manifest_path.parent / manifest["conversion"]["output_prefix"]
        if manifest_prefix != prefix:
            raise RuntimeError(f"input prefix mismatch: {manifest_path}")
        if not Path(f"{prefix}.bin").is_file() or not Path(f"{prefix}.idx").is_file():
            raise RuntimeError(f"input prefix is incomplete: {prefix}")
        for suffix in ("bin", "idx"):
            path = Path(f"{prefix}.{suffix}")
            artifact = manifest["artifacts"][path.name]
            if path.stat().st_size != artifact["size_bytes"]:
                raise RuntimeError(f"input artifact size mismatch: {path}")
            if sha256_file(path) != artifact["sha256"]:
                raise RuntimeError(f"input artifact hash mismatch: {path}")
        input_records.append(
            {
                "artifacts": {
                    Path(f"{prefix}.bin").name: manifest["artifacts"][
                        Path(f"{prefix}.bin").name
                    ],
                    Path(f"{prefix}.idx").name: manifest["artifacts"][
                        Path(f"{prefix}.idx").name
                    ],
                },
                "indexed_documents": record["indexed_documents"],
                "indexed_tokens": record["indexed_tokens"],
                "manifest_sha256": record["manifest_sha256"],
                "source_path": record["source_path"],
            }
        )
        dataset = indexed_dataset.IndexedDataset(
            str(prefix), multimodal=False, mmap=True
        )
        if dataset.index.dtype != np.uint16:
            raise RuntimeError(f"input prefix is not uint16: {prefix}")
        if not np.array_equal(
            dataset.document_indices,
            np.arange(len(dataset.sequence_lengths) + 1, dtype=np.int64),
        ):
            raise RuntimeError(f"input has multiple sequences per document: {prefix}")
        input_documents += len(dataset.sequence_lengths)
        input_tokens += int(np.sum(dataset.sequence_lengths, dtype=np.int64))
        if (
            len(dataset.sequence_lengths) != record["indexed_documents"]
            or int(np.sum(dataset.sequence_lengths, dtype=np.int64))
            != record["indexed_tokens"]
        ):
            raise RuntimeError(f"input and selection shard totals differ: {prefix}")
    if input_documents != selection["indexed_documents"]:
        raise RuntimeError("input and selection document totals differ")
    if input_tokens != selection["indexed_tokens"]:
        raise RuntimeError("input and selection token totals differ")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_bin = Path(f"{args.output_prefix}.bin")
    output_idx = Path(f"{args.output_prefix}.idx")
    partial_prefix = args.output_prefix.with_name(args.output_prefix.name + ".partial")
    partial_bin = Path(f"{partial_prefix}.bin")
    partial_idx = Path(f"{partial_prefix}.idx")
    manifest_path = args.output_prefix.with_name(
        args.output_prefix.name + ".manifest.json"
    )
    if any(
        path.exists()
        for path in (
            output_bin,
            output_idx,
            partial_bin,
            partial_idx,
            manifest_path,
        )
    ):
        raise RuntimeError("merge output or partial output already exists")

    builder = indexed_dataset.IndexedDatasetBuilder(
        str(partial_bin), dtype=np.uint16
    )
    for prefix in input_prefixes:
        builder.add_index(str(prefix))
    builder.finalize(str(partial_idx))

    merged = indexed_dataset.IndexedDataset(
        str(partial_prefix), multimodal=False, mmap=True
    )
    merged_documents = len(merged.sequence_lengths)
    merged_tokens = int(np.sum(merged.sequence_lengths, dtype=np.int64))
    if merged_documents != input_documents or merged_tokens != input_tokens:
        raise RuntimeError("merged totals differ from inputs")
    if not np.array_equal(
        merged.document_indices,
        np.arange(merged_documents + 1, dtype=np.int64),
    ):
        raise RuntimeError("merged output has multiple sequences per document")
    if partial_bin.stat().st_size != merged_tokens * np.dtype(np.uint16).itemsize:
        raise RuntimeError("merged binary size does not match uint16 token count")
    del merged

    os.replace(partial_bin, output_bin)
    os.replace(partial_idx, output_idx)
    manifest = {
        "artifacts": {
            output_bin.name: {
                "indexed_documents": merged_documents,
                "indexed_tokens": merged_tokens,
                "sha256": sha256_file(output_bin),
                "size_bytes": output_bin.stat().st_size,
            },
            output_idx.name: {
                "sha256": sha256_file(output_idx),
                "size_bytes": output_idx.stat().st_size,
            },
        },
        "inputs": input_records,
        "megatron_commit": MEGATRON_COMMIT,
        "schema_version": 1,
        "selection": {
            "artifact_path": args.selection.name,
            "sha256": hashlib.sha256(selection_bytes).hexdigest(),
        },
        "token_dtype": "uint16",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"output_prefix={args.output_prefix}")
    print(f"indexed_documents={merged_documents}")
    print(f"indexed_tokens={merged_tokens}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
