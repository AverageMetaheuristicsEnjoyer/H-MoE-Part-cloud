#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def record(path, relative_to):
    return {
        "path": str(path.relative_to(relative_to)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()

    output_root = args.output_root
    selection_path = output_root / "shards/train-selection.json"
    merge_path = output_root / "data/train.manifest.json"
    smoke_path = output_root / "gpt-dataset-smoke.log"
    train_bin = output_root / "data/train.bin"
    train_idx = output_root / "data/train.idx"
    for path in (selection_path, merge_path, smoke_path, train_bin, train_idx):
        if not path.is_file():
            raise RuntimeError(f"missing extension artifact: {path}")

    plan = json.loads(args.plan.read_bytes())
    selection = json.loads(selection_path.read_bytes())
    merge = json.loads(merge_path.read_bytes())
    training = plan["training"]
    target = training["extension_target_indexed_tokens"]
    shard_start = training["shard_start_inclusive"]
    shards = [record["shard"] for record in selection["selected_shards"]]
    if shards != list(range(shard_start, shard_start + len(shards))):
        raise RuntimeError(
            f"extension source shards are not consecutive from shard {shard_start}"
        )
    if set(plan["previous_training"]) & {
        record["source_path"] for record in selection["selected_shards"]
    }:
        raise RuntimeError("extension source paths overlap the base source paths")
    if selection["indexed_tokens"] < target:
        raise RuntimeError("extension is smaller than the Muon time-match phase")
    merged_tokens = merge["artifacts"]["train.bin"]["indexed_tokens"]
    if merged_tokens != selection["indexed_tokens"]:
        raise RuntimeError("merged and selected token totals differ")
    if merge["artifacts"]["train.bin"]["sha256"] != sha256_file(train_bin):
        raise RuntimeError("train.bin hash differs from merge manifest")
    if merge["artifacts"]["train.idx"]["sha256"] != sha256_file(train_idx):
        raise RuntimeError("train.idx hash differs from merge manifest")
    if "gpt_dataset=pass" not in smoke_path.read_text(encoding="utf-8"):
        raise RuntimeError("GPTDataset smoke did not pass")

    data_phase = plan.get(
        "data_phase",
        {
            "global_batch_sequences": 208,
            "max_target_iteration": 22208,
            "phase_transition_iteration": 13794,
            "sequence_length": 2048,
        },
    )
    data_phase["minimum_indexed_tokens"] = target
    manifest = {
        "data_phase": data_phase,
        "files": [
            record(train_bin, output_root),
            record(train_idx, output_root),
            record(smoke_path, output_root),
            record(selection_path, output_root),
            record(merge_path, output_root),
        ],
        "indexed_documents": selection["indexed_documents"],
        "indexed_tokens": selection["indexed_tokens"],
        "schema_version": 1,
        "selected_source_paths": [
            item["source_path"] for item in selection["selected_shards"]
        ],
        "source_plan": {
            "path": args.plan.name,
            "sha256": sha256_file(args.plan),
        },
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path = output_root / "artifact-manifest.json"
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise RuntimeError("existing artifact manifest differs")
    manifest_path.write_bytes(manifest_bytes)
    print(
        f"time_match_data=pass shards={shards}"
        f" indexed_documents={selection['indexed_documents']}"
        f" indexed_tokens={selection['indexed_tokens']}"
        f" manifest_sha256={hashlib.sha256(manifest_bytes).hexdigest()}"
    )


if __name__ == "__main__":
    main()
