#!/usr/bin/env python3

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from prepare_fineweb_edu_datatrove import (
    DATATROVE_COMMIT,
    GPT2_EOD,
    GPT2_VOCAB_SIZE,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    sha256_file,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--datatrove-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def load_completed_shard(
    shard_dir,
    expected_dataset,
    expected_source,
    expected_save_filename,
    verify_hashes,
):
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"incomplete shard directory: {shard_dir}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["schema_version"] != 1:
        raise RuntimeError(f"unexpected manifest schema: {manifest_path}")
    source = manifest["source"]
    if (
        source["repo"] != expected_dataset["repo"]
        or source["revision"] != expected_dataset["revision"]
        or source["path"] != expected_source
    ):
        raise RuntimeError(f"unexpected source in {manifest_path}")
    conversion = manifest["conversion"]
    expected_output_prefix = str(
        Path("tokens") / f"{expected_save_filename}_00000_tokens"
    )
    if conversion != {
        "append_eod_id": GPT2_EOD,
        "batch_size": 10_000,
        "datatrove_commit": DATATROVE_COMMIT,
        "document_order": "physical Parquet row order",
        "implicit_special_tokens": False,
        "limit_documents": -1,
        "normalization": None,
        "output_prefix": expected_output_prefix,
        "tasks": 1,
        "token_dtype": "uint16",
    }:
        raise RuntimeError(f"unexpected conversion contract in {manifest_path}")
    tokenizer = manifest["tokenizer"]
    if (
        tokenizer["repo"] != TOKENIZER_REPO
        or tokenizer["revision"] != TOKENIZER_REVISION
        or tokenizer["vocab_size"] != GPT2_VOCAB_SIZE
        or tokenizer["eod_id"] != GPT2_EOD
    ):
        raise RuntimeError(f"unexpected tokenizer in {manifest_path}")
    prefix = shard_dir / manifest["conversion"]["output_prefix"]
    bin_path = Path(f"{prefix}.bin")
    idx_path = Path(f"{prefix}.idx")
    bin_record = manifest["artifacts"][bin_path.name]
    idx_record = manifest["artifacts"][idx_path.name]
    for path, record in ((bin_path, bin_record), (idx_path, idx_record)):
        if not path.is_file() or path.stat().st_size != record["size_bytes"]:
            raise RuntimeError(f"artifact size mismatch: {path}")
        if verify_hashes and sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {path}")
    return {
        "indexed_documents": bin_record["indexed_documents"],
        "indexed_tokens": bin_record["indexed_tokens"],
        "conversion_identity": {
            "environment": manifest["environment"],
            "tokenizer": tokenizer,
        },
        "manifest": str(manifest_path.relative_to(shard_dir.parent)),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "output_prefix": str(prefix.relative_to(shard_dir.parent)),
        "source_path": expected_source,
    }


def main():
    args = parse_args()
    plan_bytes = args.plan.read_bytes()
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes)
    training = plan["training"]
    if plan["schema_version"] != 2:
        raise RuntimeError("unsupported source-plan schema")
    if training["selection"] != "smallest whole-file prefix covering the target":
        raise RuntimeError("unsupported training selection policy")
    target = training.get(
        "extension_target_indexed_tokens", training.get("base_target_indexed_tokens")
    )
    if target is None:
        raise RuntimeError("training target is missing")
    args.output_root.mkdir(parents=True, exist_ok=True)
    converter = Path(__file__).with_name("prepare_fineweb_edu_datatrove.py")

    selected = []
    indexed_documents = 0
    indexed_tokens = 0
    conversion_identity = None
    for shard in range(
        training["shard_start_inclusive"],
        training["shard_end_exclusive"],
    ):
        source_path = training["path_template"].format(shard=shard)
        shard_dir = args.output_root / f"train-shard-{shard:05d}"
        existed = shard_dir.exists()
        if not existed:
            subprocess.run(
                [
                    sys.executable,
                    str(converter),
                    "--datatrove-root",
                    str(args.datatrove_root),
                    "--output-dir",
                    str(shard_dir),
                    "--source-path",
                    source_path,
                    "--limit",
                    "-1",
                    "--save-filename",
                    f"fineweb_edu_train_{shard:05d}",
                ],
                check=True,
            )
        record = load_completed_shard(
            shard_dir,
            expected_dataset=plan["dataset"],
            expected_source=source_path,
            expected_save_filename=f"fineweb_edu_train_{shard:05d}",
            verify_hashes=existed,
        )
        if conversion_identity is None:
            conversion_identity = record.pop("conversion_identity")
        elif record.pop("conversion_identity") != conversion_identity:
            raise RuntimeError("selected shards use different conversion environments")
        record["shard"] = shard
        selected.append(record)
        indexed_documents += record["indexed_documents"]
        indexed_tokens += record["indexed_tokens"]
        print(
            f"selected_shard={shard:05d}"
            f" cumulative_documents={indexed_documents}"
            f" cumulative_indexed_tokens={indexed_tokens}",
            flush=True,
        )
        if indexed_tokens >= target:
            break
    if indexed_tokens < target:
        raise RuntimeError("training candidates ended below the requested capacity")
    previous_indexed_tokens = indexed_tokens - selected[-1]["indexed_tokens"]
    if not previous_indexed_tokens < target <= indexed_tokens:
        raise RuntimeError("selected source files are not the smallest covering prefix")

    selection = {
        "dataset": plan["dataset"],
        "indexed_documents": indexed_documents,
        "indexed_tokens": indexed_tokens,
        "overshoot_indexed_tokens": indexed_tokens - target,
        "previous_indexed_tokens": previous_indexed_tokens,
        "schema_version": 1,
        "selected_shards": selected,
        "source_plan": {
            "artifact_path": args.plan.name,
            "sha256": plan_sha256,
        },
        "target_indexed_tokens": target,
    }
    selection_bytes = (
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    selection_path = args.output_root / "train-selection.json"
    if selection_path.exists() and selection_path.read_bytes() != selection_bytes:
        raise RuntimeError(f"existing selection differs: {selection_path}")
    selection_path.write_bytes(selection_bytes)
    print(f"selection={selection_path}")


if __name__ == "__main__":
    main()
