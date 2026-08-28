#!/usr/bin/env python3

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np


def index_lengths(prefix):
    idx = Path(f"{prefix}.idx")
    binary = Path(f"{prefix}.bin")
    with idx.open("rb") as stream:
        if stream.read(9) != b"MMIDIDX\x00\x00":
            raise RuntimeError(f"bad index header: {idx}")
        version = struct.unpack("<Q", stream.read(8))[0]
        dtype_code = struct.unpack("<B", stream.read(1))[0]
        sequences = struct.unpack("<Q", stream.read(8))[0]
        document_indices = struct.unpack("<Q", stream.read(8))[0]
    if version != 1 or dtype_code != 8 or document_indices != sequences + 1:
        raise RuntimeError(f"unexpected indexed layout: {idx}")
    lengths = np.memmap(idx, dtype=np.int32, mode="r", offset=34, shape=(sequences,))
    if binary.stat().st_size != 2 * int(lengths.sum(dtype=np.int64)):
        raise RuntimeError(f"payload size mismatch: {binary}")
    return lengths


def digest_index(prefix, progress_every=250_000):
    lengths = index_lengths(prefix)
    payload = np.memmap(f"{prefix}.bin", dtype=np.uint16, mode="r")
    digests = {}
    offset = 0
    started = time.time()
    for index, length_value in enumerate(lengths):
        length = int(length_value)
        end = offset + length
        digest = hashlib.sha256(payload[offset:end]).digest()
        count, tokens = digests.get(digest, (0, 0))
        digests[digest] = (count + 1, tokens + length)
        offset = end
        if progress_every and (index + 1) % progress_every == 0:
            print(
                f"holdout_progress prefix={prefix} documents={index + 1}"
                f" elapsed_seconds={time.time() - started:.1f}",
                flush=True,
            )
    return lengths, digests


def scan_training(label, prefix, heldouts, base_hashes=None, windows=32):
    lengths = index_lengths(prefix)
    payload = np.memmap(f"{prefix}.bin", dtype=np.uint16, mode="r")
    seen = set()
    duplicate_occurrences = 0
    duplicate_tokens = 0
    matched_heldout_hashes = {name: set() for name in heldouts}
    train_matches = {name: {"documents": 0, "indexed_tokens": 0} for name in heldouts}
    base_matches = {"documents": 0, "indexed_tokens": 0, "unique_hashes": set()}
    window_rows = [
        {
            "documents": 0,
            "indexed_tokens": 0,
            "duplicate_occurrences": 0,
            **{f"{name}_matches": 0 for name in heldouts},
        }
        for _ in range(windows)
    ]
    window_seen = [set() for _ in range(windows)]
    offset = 0
    started = time.time()
    for index, length_value in enumerate(lengths):
        length = int(length_value)
        end = offset + length
        digest = hashlib.sha256(payload[offset:end]).digest()
        window = min(windows - 1, index * windows // len(lengths))
        row = window_rows[window]
        row["documents"] += 1
        row["indexed_tokens"] += length
        if digest in seen:
            duplicate_occurrences += 1
            duplicate_tokens += length
        else:
            seen.add(digest)
        if digest in window_seen[window]:
            row["duplicate_occurrences"] += 1
        else:
            window_seen[window].add(digest)
        for name, hashes in heldouts.items():
            if digest in hashes:
                matched_heldout_hashes[name].add(digest)
                train_matches[name]["documents"] += 1
                train_matches[name]["indexed_tokens"] += length
                row[f"{name}_matches"] += 1
        if base_hashes is not None and digest in base_hashes:
            base_matches["documents"] += 1
            base_matches["indexed_tokens"] += length
            base_matches["unique_hashes"].add(digest)
        offset = end
        if (index + 1) % 250_000 == 0:
            print(
                f"training_progress label={label} documents={index + 1}"
                f" unique={len(seen)} elapsed_seconds={time.time() - started:.1f}",
                flush=True,
            )

    heldout_overlap = {}
    for name, hashes in heldouts.items():
        matched = matched_heldout_hashes[name]
        matched_documents = sum(hashes[digest][0] for digest in matched)
        matched_tokens = sum(hashes[digest][1] for digest in matched)
        total_documents = sum(value[0] for value in hashes.values())
        total_tokens = sum(value[1] for value in hashes.values())
        heldout_overlap[name] = {
            "matched_documents": matched_documents,
            "matched_document_rate": matched_documents / total_documents,
            "matched_indexed_tokens": matched_tokens,
            "matched_indexed_token_rate": matched_tokens / total_tokens,
            "matched_unique_hashes": len(matched),
            "training_match_occurrences": train_matches[name],
        }

    for row in window_rows:
        row["mean_length"] = row["indexed_tokens"] / row["documents"]
        row["duplicate_occurrence_rate"] = row["duplicate_occurrences"] / row["documents"]
        for name in heldouts:
            row[f"{name}_match_rate"] = row[f"{name}_matches"] / row["documents"]

    adjacent_length_correlation = float(
        np.corrcoef(lengths[:-1], lengths[1:])[0, 1]
    )
    report = {
        "prefix": str(prefix),
        "documents": len(lengths),
        "indexed_tokens": int(lengths.sum(dtype=np.int64)),
        "unique_exact_documents": len(seen),
        "duplicate_occurrences": duplicate_occurrences,
        "duplicate_occurrence_rate": duplicate_occurrences / len(lengths),
        "duplicate_occurrence_tokens": duplicate_tokens,
        "adjacent_length_correlation": adjacent_length_correlation,
        "heldout_overlap": heldout_overlap,
        "windows": window_rows,
        "elapsed_seconds": time.time() - started,
    }
    if base_hashes is not None:
        report["base_overlap"] = {
            "matched_documents": base_matches["documents"],
            "matched_document_rate": base_matches["documents"] / len(lengths),
            "matched_indexed_tokens": base_matches["indexed_tokens"],
            "matched_unique_hashes": len(base_matches["unique_hashes"]),
        }
    return report, seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--extension-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    heldouts = {}
    heldout_summary = {}
    for name in ("development", "final"):
        lengths, hashes = digest_index(args.base_root / "data" / name)
        heldouts[name] = hashes
        heldout_summary[name] = {
            "documents": len(lengths),
            "indexed_tokens": int(lengths.sum(dtype=np.int64)),
            "unique_exact_documents": len(hashes),
        }

    base, base_hashes = scan_training(
        "base", args.base_root / "data" / "train", heldouts
    )
    extension, _ = scan_training(
        "extension",
        args.extension_root / "data" / "train",
        heldouts,
        base_hashes=base_hashes,
    )
    report = {
        "schema_version": 1,
        "method": "SHA-256 of each complete uint16 indexed document including appended EOD",
        "heldouts": heldout_summary,
        "base": base,
        "extension": extension,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    print(f"REPORT={args.output}")
    print("INDEXED_DUPLICATE_AUDIT=pass")


if __name__ == "__main__":
    main()
