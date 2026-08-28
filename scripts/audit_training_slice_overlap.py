#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_indexed_data_duplicates import index_lengths


GLOBAL_BATCH_SIZE = 208
SOURCE_ITERATION = 13_794
ORIGINAL_END = 17_242
PLATEAU_STEPS = 2_328
EXTENSION_PHASE_STEPS = 5_776


def holdout_index(prefix):
    lengths = index_lengths(prefix)
    payload = np.memmap(f"{prefix}.bin", dtype=np.uint16, mode="r")
    records = {}
    offset = 0
    for length_value in lengths:
        length = int(length_value)
        end = offset + length
        digest = hashlib.sha256(payload[offset:end]).digest()
        count, tokens = records.get(digest, (0, 0))
        records[digest] = (count + 1, tokens + length)
        offset = end
    digest_to_id = {digest: index for index, digest in enumerate(records)}
    return {
        "digest_to_id": digest_to_id,
        "document_counts": np.array([value[0] for value in records.values()], dtype=np.int32),
        "token_counts": np.array([value[1] for value in records.values()], dtype=np.int64),
    }


def training_match_ids(prefix, heldouts):
    lengths = index_lengths(prefix)
    payload = np.memmap(f"{prefix}.bin", dtype=np.uint16, mode="r")
    matches = {
        name: np.full(len(lengths), -1, dtype=np.int32) for name in heldouts
    }
    offset = 0
    for document, length_value in enumerate(lengths):
        length = int(length_value)
        end = offset + length
        digest = hashlib.sha256(payload[offset:end]).digest()
        for name, heldout in heldouts.items():
            match_id = heldout["digest_to_id"].get(digest)
            if match_id is not None:
                matches[name][document] = match_id
        offset = end
        if (document + 1) % 500_000 == 0:
            print(
                f"match_index_progress prefix={prefix} documents={document + 1}",
                flush=True,
            )
    return matches


def find_cache(root, num_samples):
    for description_path in sorted(root.rglob("*-GPTDataset-train-description.txt")):
        text = description_path.read_text()
        if not text.strip():
            continue
        description = json.loads(text)
        if description.get("num_samples") != num_samples:
            continue
        stem = description_path.name.removesuffix("description.txt")
        paths = {
            name: description_path.with_name(f"{stem}{name}_index.npy")
            for name in ("document", "sample", "shuffle")
        }
        if all(path.is_file() for path in paths.values()):
            return {
                "description_hash": stem.split("-", 1)[0],
                **{
                    name: np.load(path, mmap_mode="r")
                    for name, path in paths.items()
                },
            }
    raise RuntimeError(f"no complete train cache for num_samples={num_samples} under {root}")


def sample_slice(name, cache, start, end, match_ids):
    if not 0 <= start < end <= len(cache["shuffle"]):
        raise RuntimeError(f"invalid sample slice {name}: {start}:{end}")
    matched_ids = {split: set() for split in match_ids}
    samples_with_match = {split: 0 for split in match_ids}
    matching_document_touches = {split: 0 for split in match_ids}
    shuffle = cache["shuffle"][start:end]
    sample = cache["sample"]
    document = cache["document"]
    for ordinal, shuffled_index_value in enumerate(shuffle):
        shuffled_index = int(shuffled_index_value)
        begin_position = int(sample[shuffled_index, 0])
        end_position = int(sample[shuffled_index + 1, 0])
        end_offset = int(sample[shuffled_index + 1, 1])
        stop = end_position + (1 if end_offset else 0)
        document_ids = document[begin_position:stop]
        for split, ids_by_document in match_ids.items():
            ids = ids_by_document[document_ids]
            ids = ids[ids >= 0]
            if len(ids):
                samples_with_match[split] += 1
                matching_document_touches[split] += len(ids)
                matched_ids[split].update(map(int, ids))
        if (ordinal + 1) % 500_000 == 0:
            print(
                f"slice_progress name={name} samples={ordinal + 1}",
                flush=True,
            )
    return {
        "name": name,
        "start_sample": start,
        "end_sample": end,
        "samples": end - start,
        "samples_with_match": samples_with_match,
        "matching_document_touches": matching_document_touches,
        "matched_ids": matched_ids,
    }


def coverage(heldouts, matched_ids):
    result = {}
    for split, ids in matched_ids.items():
        heldout = heldouts[split]
        selected = np.fromiter(ids, dtype=np.int64, count=len(ids))
        matched_documents = int(heldout["document_counts"][selected].sum()) if len(selected) else 0
        matched_tokens = int(heldout["token_counts"][selected].sum()) if len(selected) else 0
        total_documents = int(heldout["document_counts"].sum())
        total_tokens = int(heldout["token_counts"].sum())
        result[split] = {
            "matched_documents": matched_documents,
            "matched_document_rate": matched_documents / total_documents,
            "matched_indexed_tokens": matched_tokens,
            "matched_indexed_token_rate": matched_tokens / total_tokens,
            "matched_unique_hashes": len(ids),
        }
    return result


def union_slices(slices):
    return {
        split: set().union(*(item["matched_ids"][split] for item in slices))
        for split in slices[0]["matched_ids"]
    }


def public_slice(item, heldouts):
    return {
        key: value for key, value in item.items() if key != "matched_ids"
    } | {"coverage": coverage(heldouts, item["matched_ids"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--extension-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    heldouts = {
        name: holdout_index(args.base_root / "data" / name)
        for name in ("development", "final")
    }
    base_matches = training_match_ids(args.base_root / "data" / "train", heldouts)
    extension_matches = training_match_ids(
        args.extension_root / "data" / "train", heldouts
    )
    base_cache = find_cache(
        args.base_root / "data" / "train", ORIGINAL_END * GLOBAL_BATCH_SIZE
    )
    extension_cache = find_cache(
        args.extension_root / "data" / "train",
        EXTENSION_PHASE_STEPS * GLOBAL_BATCH_SIZE,
    )

    source_samples = SOURCE_ITERATION * GLOBAL_BATCH_SIZE
    original_end_samples = ORIGINAL_END * GLOBAL_BATCH_SIZE
    plateau_samples = PLATEAU_STEPS * GLOBAL_BATCH_SIZE
    extension_end_samples = EXTENSION_PHASE_STEPS * GLOBAL_BATCH_SIZE
    slices = {
        "source_base": sample_slice(
            "source_base", base_cache, 0, source_samples, base_matches
        ),
        "original_base_tail": sample_slice(
            "original_base_tail",
            base_cache,
            source_samples,
            original_end_samples,
            base_matches,
        ),
        "extension_plateau": sample_slice(
            "extension_plateau", extension_cache, 0, plateau_samples, extension_matches
        ),
        "extension_decay": sample_slice(
            "extension_decay",
            extension_cache,
            plateau_samples,
            extension_end_samples,
            extension_matches,
        ),
    }

    trajectories = {}
    for name, members in {
        "source_13794": ["source_base"],
        "original_17242": ["source_base", "original_base_tail"],
        "old_time_match_decay_start": ["source_base", "extension_plateau"],
        "old_time_match_19570": [
            "source_base",
            "extension_plateau",
            "extension_decay",
        ],
        "corrected_19570": [
            "source_base",
            "original_base_tail",
            "extension_plateau",
        ],
    }.items():
        trajectories[name] = {
            "members": members,
            "coverage": coverage(
                heldouts, union_slices([slices[member] for member in members])
            ),
        }

    report = {
        "schema_version": 1,
        "method": "exact heldout-document exposure through the cached GPTDataset sample and shuffle indices",
        "cache": {
            "base_description_hash": base_cache["description_hash"],
            "extension_description_hash": extension_cache["description_hash"],
        },
        "slices": {
            name: public_slice(item, heldouts) for name, item in slices.items()
        },
        "trajectories": trajectories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for name, value in trajectories.items():
        for split, result in value["coverage"].items():
            print(
                f"TRAJECTORY_OVERLAP name={name} split={split}"
                f" matched_documents={result['matched_documents']}"
                f" matched_document_rate={result['matched_document_rate']:.8f}"
                f" matched_indexed_tokens={result['matched_indexed_tokens']}"
                f" matched_indexed_token_rate={result['matched_indexed_token_rate']:.8f}"
            )
    print(f"REPORT={args.output}")
    print("TRAINING_SLICE_OVERLAP_AUDIT=pass")


if __name__ == "__main__":
    main()
