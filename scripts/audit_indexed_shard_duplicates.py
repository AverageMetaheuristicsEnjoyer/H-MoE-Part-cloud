#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_indexed_data_duplicates import index_lengths


def load_shards(selection_path):
    selection = json.loads(selection_path.read_text())
    return [
        {
            "shard": record["shard"],
            "documents": record["indexed_documents"],
        }
        for record in selection["selected_shards"]
    ]


def scan(label, prefix, shards, group_size=4):
    lengths = index_lengths(prefix)
    if sum(shard["documents"] for shard in shards) != len(lengths):
        raise RuntimeError(f"{label} shard document counts do not match merged index")
    payload = np.memmap(f"{prefix}.bin", dtype=np.uint16, mode="r")
    group_sets = []
    group_rows = []
    shard_rows = []
    document = 0
    offset = 0
    for shard_position, shard in enumerate(shards):
        if shard_position % group_size == 0:
            group_sets.append(set())
            group_rows.append(
                {
                    "shards": [],
                    "documents": 0,
                    "duplicate_occurrences": 0,
                }
            )
        group_seen = group_sets[-1]
        group = group_rows[-1]
        group["shards"].append(shard["shard"])
        shard_seen = set()
        shard_duplicates = 0
        for _ in range(shard["documents"]):
            length = int(lengths[document])
            end = offset + length
            digest = hashlib.sha256(payload[offset:end]).digest()
            if digest in shard_seen:
                shard_duplicates += 1
            else:
                shard_seen.add(digest)
            if digest in group_seen:
                group["duplicate_occurrences"] += 1
            else:
                group_seen.add(digest)
            document += 1
            offset = end
        group["documents"] += shard["documents"]
        shard_rows.append(
            {
                "shard": shard["shard"],
                "documents": shard["documents"],
                "duplicate_occurrences": shard_duplicates,
                "duplicate_occurrence_rate": shard_duplicates / shard["documents"],
            }
        )
        print(
            f"SHARD_DUPLICATES label={label} shard={shard['shard']}"
            f" documents={shard['documents']} duplicates={shard_duplicates}"
            f" rate={shard_duplicates / shard['documents']:.8f}",
            flush=True,
        )
    for group in group_rows:
        group["duplicate_occurrence_rate"] = (
            group["duplicate_occurrences"] / group["documents"]
        )
        print(
            f"GROUP_DUPLICATES label={label} shards={group['shards']}"
            f" documents={group['documents']}"
            f" duplicates={group['duplicate_occurrences']}"
            f" rate={group['duplicate_occurrence_rate']:.8f}",
            flush=True,
        )
    return {"shards": shard_rows, "groups": group_rows}, group_sets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--extension-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    base, base_sets = scan(
        "base",
        args.base_root / "data" / "train",
        load_shards(args.base_root / "provenance" / "train-selection.json"),
    )
    extension, extension_sets = scan(
        "extension",
        args.extension_root / "data" / "train",
        load_shards(args.extension_root / "shards" / "train-selection.json"),
    )
    cross = []
    for base_index, base_hashes in enumerate(base_sets):
        for extension_index, extension_hashes in enumerate(extension_sets):
            overlap = len(base_hashes & extension_hashes)
            cross.append(
                {
                    "base_group": base_index,
                    "extension_group": extension_index,
                    "unique_exact_hashes": overlap,
                }
            )
            print(
                f"GROUP_CROSS_OVERLAP base_group={base_index}"
                f" extension_group={extension_index} unique_exact_hashes={overlap}"
            )
    report = {
        "schema_version": 1,
        "method": "SHA-256 of each complete uint16 indexed document including appended EOD",
        "base": base,
        "extension": extension,
        "cross_group_overlap": cross,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"REPORT={args.output}")
    print("INDEXED_SHARD_DUPLICATE_AUDIT=pass")


if __name__ == "__main__":
    main()
