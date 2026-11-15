#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from prepare_fineweb_edu_datatrove import (
    DATATROVE_COMMIT,
    FINEWEB_EDU_REPO,
    FINEWEB_EDU_REVISION,
    GPT2_EOD,
    GPT2_VOCAB_SIZE,
    TOKENIZER_JSON_SHA256,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    sha256_file,
)

LINES_TO_BUFFER = 1_000_000
SIGNATURE_RECORD_BYTES = struct.calcsize("<QHI")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datatrove-root", required=True, type=Path)
    parser.add_argument("--train-selection", required=True, type=Path)
    parser.add_argument("--development-manifest", required=True, type=Path)
    parser.add_argument("--final-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def document_text(document) -> str:
    return document.text


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def require_sha256(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"invalid SHA-256 for {label}")


def resolve_below(base, relative, label):
    base = base.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise RuntimeError(f"{label} escapes {base}")
    return path


def validate_tokenizer(manifest, label):
    tokenizer = manifest["tokenizer"]
    if (
        tokenizer["repo"] != TOKENIZER_REPO
        or tokenizer["revision"] != TOKENIZER_REVISION
        or tokenizer["vocab_size"] != GPT2_VOCAB_SIZE
        or tokenizer["eod_id"] != GPT2_EOD
    ):
        raise RuntimeError(f"unexpected tokenizer in {label}")
    tokenizer_json = [
        record
        for record in tokenizer["files"]
        if record["path"] == "tokenizer.json"
    ]
    if (
        len(tokenizer_json) != 1
        or tokenizer_json[0]["sha256"] != TOKENIZER_JSON_SHA256
    ):
        raise RuntimeError(f"unexpected tokenizer.json in {label}")
    return tokenizer


def validate_source(manifest, expected_path, label):
    source = manifest["source"]
    if (
        source["repo"] != FINEWEB_EDU_REPO
        or source["revision"] != FINEWEB_EDU_REVISION
        or source["path"] != expected_path
    ):
        raise RuntimeError(f"unexpected source in {label}")
    require_sha256(source["sha256"], f"{label} source")
    if source["size_bytes"] <= 0:
        raise RuntimeError(f"invalid source size in {label}")
    return source


def validate_conversion(manifest, label, minimum_indexed_tokens=None):
    conversion = manifest["conversion"]
    expected = {
        "append_eod_id": GPT2_EOD,
        "batch_size": 10_000,
        "datatrove_commit": DATATROVE_COMMIT,
        "document_order": "physical Parquet row order",
        "implicit_special_tokens": False,
        "limit_documents": -1,
        "normalization": None,
        "output_prefix": conversion["output_prefix"],
        "tasks": 1,
        "token_dtype": "uint16",
    }
    if minimum_indexed_tokens is not None:
        expected["minimum_indexed_tokens"] = minimum_indexed_tokens
    if conversion != expected:
        raise RuntimeError(f"unexpected conversion contract in {label}")
    output_prefix = Path(conversion["output_prefix"])
    if output_prefix.is_absolute() or output_prefix.parent != Path("tokens"):
        raise RuntimeError(f"unexpected output prefix in {label}")
    return output_prefix


def load_index_lengths(path):
    with path.open("rb") as stream:
        if stream.read(9) != b"MMIDIDX\x00\x00":
            raise RuntimeError(f"unexpected Megatron index header: {path}")
        version = struct.unpack("<Q", stream.read(8))[0]
        dtype_code = struct.unpack("<B", stream.read(1))[0]
        sequences = struct.unpack("<Q", stream.read(8))[0]
        documents = struct.unpack("<Q", stream.read(8))[0]
        lengths = np.frombuffer(
            stream.read(4 * sequences), dtype="<i4"
        ).copy()
        pointers = np.frombuffer(
            stream.read(8 * sequences), dtype="<i8"
        ).copy()
        document_indices = np.frombuffer(
            stream.read(8 * documents), dtype="<i8"
        ).copy()
        if stream.read(1):
            raise RuntimeError(f"unexpected trailing bytes in {path}")
    if version != 1 or dtype_code != 8:
        raise RuntimeError(f"unexpected Megatron version/dtype in {path}")
    if documents != sequences + 1 or np.any(lengths <= 0):
        raise RuntimeError(f"unexpected Megatron document layout in {path}")
    if not np.array_equal(
        document_indices, np.arange(documents, dtype=np.int64)
    ):
        raise RuntimeError(f"noncontiguous document boundaries in {path}")
    expected_pointers = np.zeros(sequences, dtype=np.int64)
    if sequences > 1:
        expected_pointers[1:] = (
            np.cumsum(lengths[:-1], dtype=np.int64)
            * np.dtype(np.uint16).itemsize
        )
    if not np.array_equal(pointers, expected_pointers):
        raise RuntimeError(f"unexpected sequence pointers in {path}")
    return lengths


def artifact_record(manifest, prefix, suffix, label):
    name = f"{prefix.name}.{suffix}"
    try:
        record = manifest["artifacts"][name]
    except KeyError as error:
        raise RuntimeError(f"missing {suffix} artifact in {label}") from error
    require_sha256(record["sha256"], f"{label} {suffix}")
    return record


def load_training(selection_path):
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    if selection["schema_version"] != 1:
        raise RuntimeError("unsupported train-selection schema")
    if selection["dataset"] != {
        "repo": FINEWEB_EDU_REPO,
        "revision": FINEWEB_EDU_REVISION,
    }:
        raise RuntimeError("unexpected training dataset")
    records = selection["selected_shards"]
    if not records:
        raise RuntimeError("train-selection contains no shards")

    sources = []
    manifests = []
    documents = 0
    tokens = 0
    environment = None
    tokenizer = None
    seen_paths = set()
    for record in records:
        source_path = record["source_path"]
        if (
            Path(source_path).parent != Path("data")
            or source_path in seen_paths
        ):
            raise RuntimeError(f"invalid or repeated training source: {source_path}")
        seen_paths.add(source_path)
        manifest_path = resolve_below(
            selection_path.parent,
            record["manifest"],
            "training manifest",
        )
        manifest_bytes = manifest_path.read_bytes()
        if sha256_bytes(manifest_bytes) != record["manifest_sha256"]:
            raise RuntimeError(f"training manifest hash mismatch: {manifest_path}")
        manifest = json.loads(manifest_bytes)
        if manifest["schema_version"] != 1:
            raise RuntimeError(f"unexpected manifest schema: {manifest_path}")
        source = validate_source(manifest, source_path, manifest_path)
        output_prefix = validate_conversion(manifest, manifest_path)
        current_tokenizer = validate_tokenizer(manifest, manifest_path)
        if environment is None:
            environment = manifest["environment"]
            tokenizer = current_tokenizer
        elif (
            manifest["environment"] != environment
            or current_tokenizer != tokenizer
        ):
            raise RuntimeError("training conversion identities differ")

        local_prefix = resolve_below(
            manifest_path.parent,
            output_prefix,
            "training output prefix",
        )
        expected_selection_prefix = resolve_below(
            selection_path.parent,
            record["output_prefix"],
            "train-selection output prefix",
        )
        if local_prefix != expected_selection_prefix:
            raise RuntimeError(f"training output prefix mismatch: {manifest_path}")
        bin_record = artifact_record(
            manifest, output_prefix, "bin", manifest_path
        )
        idx_record = artifact_record(
            manifest, output_prefix, "idx", manifest_path
        )
        if (
            bin_record["indexed_documents"] != record["indexed_documents"]
            or bin_record["indexed_tokens"] != record["indexed_tokens"]
        ):
            raise RuntimeError(f"training counts differ: {manifest_path}")
        for suffix, artifact in (("bin", bin_record), ("idx", idx_record)):
            path = Path(f"{local_prefix}.{suffix}")
            if not path.is_file() or path.stat().st_size != artifact["size_bytes"]:
                raise RuntimeError(f"training artifact size mismatch: {path}")
        lengths = load_index_lengths(Path(f"{local_prefix}.idx"))
        if (
            len(lengths) != record["indexed_documents"]
            or int(np.sum(lengths, dtype=np.int64))
            != record["indexed_tokens"]
            or Path(f"{local_prefix}.bin").stat().st_size
            != record["indexed_tokens"] * np.dtype(np.uint16).itemsize
        ):
            raise RuntimeError(f"training indexed counts differ: {manifest_path}")

        documents += record["indexed_documents"]
        tokens += record["indexed_tokens"]
        sources.append(
            {
                "indexed_documents": record["indexed_documents"],
                "indexed_tokens": record["indexed_tokens"],
                "path": source_path,
            }
        )
        manifests.append(
            {
                "artifact_path": record["manifest"],
                "sha256": record["manifest_sha256"],
                "source": {
                    "path": source_path,
                    "sha256": source["sha256"],
                    "size_bytes": source["size_bytes"],
                },
            }
        )

    if (
        documents != selection["indexed_documents"]
        or tokens != selection["indexed_tokens"]
    ):
        raise RuntimeError("train-selection totals differ from shard manifests")
    if (
        selection["previous_indexed_tokens"]
        + records[-1]["indexed_tokens"]
        != tokens
        or not selection["previous_indexed_tokens"]
        < selection["target_indexed_tokens"]
        <= tokens
        or selection["overshoot_indexed_tokens"]
        != tokens - selection["target_indexed_tokens"]
    ):
        raise RuntimeError("train-selection is not the declared covering prefix")
    return {
        "documents": documents,
        "environment": environment,
        "input": {
            "artifact_path": selection_path.name,
            "manifests": manifests,
            "sha256": sha256_bytes(selection_bytes),
        },
        "source_paths": seen_paths,
        "sources": sources,
        "tokenizer": tokenizer,
        "tokens": tokens,
    }


def load_heldout(manifest_path, split):
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["schema_version"] != 2:
        raise RuntimeError(f"unexpected {split} manifest schema")
    selection = manifest["selection"]
    minimum = selection["minimum_indexed_tokens"]
    if (
        Path(manifest["source"]["path"]).parent != Path("data")
        or selection["source_text_sequence_hash"]["algorithm"]
        != "sha256(concat(length_uint64_le || text_utf8))"
    ):
        raise RuntimeError(f"unexpected {split} source-selection contract")
    output_prefix = validate_conversion(
        manifest,
        manifest_path,
        minimum_indexed_tokens=minimum,
    )
    source = validate_source(
        manifest, manifest["source"]["path"], manifest_path
    )
    tokenizer = validate_tokenizer(manifest, manifest_path)
    if (
        selection["indexed_tokens"] - minimum
        != selection["overshoot_indexed_tokens"]
        or not selection["previous_indexed_tokens"]
        < minimum
        <= selection["indexed_tokens"]
    ):
        raise RuntimeError(f"{split} is not the smallest covering prefix")

    prefix = resolve_below(
        manifest_path.parent, output_prefix, f"{split} output prefix"
    )
    bin_path = Path(f"{prefix}.bin")
    idx_path = Path(f"{prefix}.idx")
    bin_record = artifact_record(manifest, output_prefix, "bin", manifest_path)
    idx_record = artifact_record(manifest, output_prefix, "idx", manifest_path)
    for path, record in ((bin_path, bin_record), (idx_path, idx_record)):
        if not path.is_file() or path.stat().st_size != record["size_bytes"]:
            raise RuntimeError(f"{split} artifact size mismatch: {path}")
    if sha256_file(idx_path) != idx_record["sha256"]:
        raise RuntimeError(f"{split} index hash mismatch: {idx_path}")

    lengths = load_index_lengths(idx_path)
    indexed_tokens = int(np.sum(lengths, dtype=np.int64))
    if (
        len(lengths) != selection["indexed_documents"]
        or indexed_tokens != selection["indexed_tokens"]
        or bin_record["indexed_documents"] != len(lengths)
        or bin_record["indexed_tokens"] != indexed_tokens
        or bin_path.stat().st_size
        != indexed_tokens * np.dtype(np.uint16).itemsize
    ):
        raise RuntimeError(f"{split} indexed counts differ")

    selection_path = manifest_path.parent / "selection.json"
    selection_record = manifest["artifacts"].get("selection.json")
    if selection_record is None or not selection_path.is_file():
        raise RuntimeError(f"{split} selection artifact is missing")
    require_sha256(selection_record["sha256"], f"{split} selection")
    if (
        selection_path.stat().st_size != selection_record["size_bytes"]
        or sha256_file(selection_path) != selection_record["sha256"]
        or json.loads(selection_path.read_bytes()) != selection
    ):
        raise RuntimeError(f"{split} selection artifact differs from manifest")

    return {
        "environment": manifest["environment"],
        "input": {
            "artifact_path": manifest_path.name,
            "index": {
                "artifact_path": str(output_prefix.with_suffix(".idx")),
                "sha256": idx_record["sha256"],
                "size_bytes": idx_record["size_bytes"],
            },
            "selection": {
                "artifact_path": "selection.json",
                "sha256": selection_record["sha256"],
            },
            "sha256": sha256_bytes(manifest_bytes),
            "source": {
                "path": source["path"],
                "sha256": source["sha256"],
                "size_bytes": source["size_bytes"],
            },
        },
        "lengths": lengths,
        "selection": selection,
        "source_path": source["path"],
        "tokenizer": tokenizer,
    }


def parse_candidate_ordinals(folder, document_count):
    ordinals = []
    files = []
    for path in sorted(folder.rglob("*.c4_dup")):
        data = path.read_bytes()
        if len(data) % np.dtype("<u4").itemsize:
            raise RuntimeError(f"truncated duplicate file: {path}")
        values = np.frombuffer(data, dtype="<u4")
        ordinals.extend(int(value) for value in values)
        files.append(
            {
                "artifact_path": str(path.relative_to(folder.parent.parent)),
                "sha256": sha256_bytes(data),
                "size_bytes": len(data),
            }
        )
    if len(ordinals) != len(set(ordinals)):
        raise RuntimeError("duplicate candidate ordinal emitted more than once")
    if any(ordinal < 0 or ordinal >= document_count for ordinal in ordinals):
        raise RuntimeError("candidate ordinal is outside the held-out prefix")
    return sorted(ordinals), files


def validate_signature(path, expected_documents, output_dir):
    if not path.is_file():
        raise RuntimeError(f"missing signature file: {path}")
    if path.stat().st_size != expected_documents * SIGNATURE_RECORD_BYTES:
        raise RuntimeError(f"signature document count differs: {path}")
    return {
        "artifact_path": str(path.relative_to(output_dir)),
        "documents": expected_documents,
        "size_bytes": path.stat().st_size,
    }


def replay_heldout(reader, heldout, ordinals, hash_fc, split):
    selected = set(ordinals)
    candidates = []
    sequence_digest = hashlib.sha256()
    source_utf8_bytes = 0
    documents = 0
    for ordinal, document in enumerate(reader.run()):
        raw = document.text.encode("utf-8")
        source_utf8_bytes += len(raw)
        sequence_digest.update(struct.pack("<Q", len(raw)))
        sequence_digest.update(raw)
        if ordinal in selected:
            candidates.append(
                {
                    "confirmed": False,
                    "hash64": hash_fc(document.text),
                    "indexed_tokens": int(heldout["lengths"][ordinal]),
                    "ordinal": ordinal,
                    "raw": raw,
                    "sha256": sha256_bytes(raw),
                    "utf8_bytes": len(raw),
                }
            )
        documents += 1
    selection = heldout["selection"]
    if (
        documents != selection["indexed_documents"]
        or source_utf8_bytes != selection["source_utf8_bytes"]
        or sequence_digest.hexdigest()
        != selection["source_text_sequence_hash"]["digest"]
    ):
        raise RuntimeError(f"{split} source replay differs from selection")
    if [candidate["ordinal"] for candidate in candidates] != ordinals:
        raise RuntimeError(f"{split} candidate replay is incomplete")
    return candidates


def split_report(heldout, candidates, duplicate_files):
    matched = [candidate for candidate in candidates if candidate["confirmed"]]
    documents = heldout["selection"]["indexed_documents"]
    indexed_tokens = heldout["selection"]["indexed_tokens"]
    source_utf8_bytes = heldout["selection"]["source_utf8_bytes"]
    matched_indexed_tokens = sum(
        candidate["indexed_tokens"] for candidate in matched
    )
    matched_utf8_bytes = sum(
        candidate["utf8_bytes"] for candidate in matched
    )
    return {
        "candidate_documents_64bit": len(candidates),
        "candidate_indexed_tokens_64bit": sum(
            candidate["indexed_tokens"] for candidate in candidates
        ),
        "candidate_ordinals_64bit": [
            candidate["ordinal"] for candidate in candidates
        ],
        "candidate_utf8_bytes_64bit": sum(
            candidate["utf8_bytes"] for candidate in candidates
        ),
        "confirmed_match_ordinals": [
            candidate["ordinal"] for candidate in matched
        ],
        "duplicate_stage_artifacts": duplicate_files,
        "indexed_documents": documents,
        "indexed_tokens": indexed_tokens,
        "matched_documents": len(matched),
        "matched_document_rate": len(matched) / documents,
        "matched_indexed_token_rate": matched_indexed_tokens / indexed_tokens,
        "matched_indexed_tokens": matched_indexed_tokens,
        "matched_utf8_byte_rate": matched_utf8_bytes / source_utf8_bytes,
        "matched_utf8_bytes": matched_utf8_bytes,
        "sha1_64_collision_candidates": len(candidates) - len(matched),
        "source_utf8_bytes": source_utf8_bytes,
        "unique_matched_texts": len({candidate["raw"] for candidate in matched}),
    }


def main():
    args = parse_args()
    datatrove_root = args.datatrove_root.resolve()
    revision = subprocess.run(
        ["git", "-C", str(datatrove_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != DATATROVE_COMMIT:
        raise RuntimeError(
            f"DataTrove commit is {revision}, expected {DATATROVE_COMMIT}"
        )

    training = load_training(args.train_selection.resolve())
    heldout = {
        "development": load_heldout(
            args.development_manifest.resolve(), "development"
        ),
        "final": load_heldout(args.final_manifest.resolve(), "final"),
    }
    heldout_paths = {
        item["source_path"] for item in heldout.values()
    }
    if (
        len(heldout_paths) != 2
        or training["source_paths"].intersection(heldout_paths)
    ):
        raise RuntimeError("training/development/final source paths overlap")
    for split, item in heldout.items():
        if (
            item["environment"] != training["environment"]
            or item["tokenizer"] != training["tokenizer"]
        ):
            raise RuntimeError(f"{split} conversion identity differs")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(datatrove_root / "src"))
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.dedup.exact_dedup import (
        ExactDedupBuildIndex,
        ExactDedupConfig,
        ExactDedupSignature,
        ExactFindDedups,
    )
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.utils.hashing import HashConfig, create_hash_func

    module_path = Path(
        sys.modules[ExactDedupSignature.__module__].__file__
    ).resolve()
    if not module_path.is_relative_to(datatrove_root):
        raise RuntimeError(
            f"loaded DataTrove from {module_path}, not {datatrove_root}"
        )

    source_folder = (
        f"hf://datasets/{FINEWEB_EDU_REPO}"
        f"@{FINEWEB_EDU_REVISION}/data"
    )
    hash_config = HashConfig(hash_fc="sha1", precision=64)
    dedup_config = ExactDedupConfig(
        content_getter=document_text,
        hash_config=hash_config,
        only_dedup_in_index=True,
    )
    paths_file = args.output_dir / "train-paths.txt"
    paths_file.write_text(
        "".join(
            f"{Path(source['path']).name}\n"
            for source in training["sources"]
        ),
        encoding="utf-8",
    )

    stages = args.output_dir / "stages"
    logs = args.output_dir / "logs"
    train_signatures = stages / "train-signatures"
    train_index = stages / "train-index"
    LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                data_folder=source_folder,
                paths_file=str(paths_file),
                read_metadata=False,
                recursive=False,
                shuffle_files=False,
            ),
            ExactDedupSignature(
                output_folder=str(train_signatures),
                config=dedup_config,
                finder_workers=1,
            ),
        ],
        tasks=len(training["sources"]),
        workers=1,
        logging_dir=str(logs / "train-signatures"),
        skip_completed=False,
    ).run()
    train_signature_artifacts = [
        validate_signature(
            train_signatures / "0000" / f"{rank:05d}.c4_sig",
            source["indexed_documents"],
            args.output_dir,
        )
        for rank, source in enumerate(training["sources"])
    ]
    LocalPipelineExecutor(
        pipeline=[
            ExactDedupBuildIndex(
                data_folder=str(train_signatures),
                output_folder=str(train_index),
                index_name="train",
                config=dedup_config,
                lines_to_buffer=LINES_TO_BUFFER,
            )
        ],
        tasks=1,
        workers=1,
        logging_dir=str(logs / "train-index"),
        skip_completed=False,
    ).run()
    train_index_path = train_index / "train.c4_index"
    if not train_index_path.is_file():
        raise RuntimeError("DataTrove did not create the training exact index")
    if (
        not train_index_path.stat().st_size
        or train_index_path.stat().st_size % np.dtype("<u8").itemsize
        or train_index_path.stat().st_size
        > training["documents"] * np.dtype("<u8").itemsize
    ):
        raise RuntimeError("training exact index size is invalid")

    candidate_ordinals = {}
    duplicate_artifacts = {}
    heldout_signature_artifacts = {}
    for split, item in heldout.items():
        source_filename = Path(item["source_path"]).name
        signatures = stages / f"{split}-signatures"
        duplicates = stages / f"{split}-duplicates"
        LocalPipelineExecutor(
            pipeline=[
                ParquetReader(
                    data_folder=source_folder,
                    glob_pattern=source_filename,
                    limit=item["selection"]["indexed_documents"],
                    read_metadata=False,
                    recursive=False,
                    shuffle_files=False,
                ),
                ExactDedupSignature(
                    output_folder=str(signatures),
                    config=dedup_config,
                    finder_workers=1,
                ),
            ],
            tasks=1,
            workers=1,
            logging_dir=str(logs / f"{split}-signatures"),
            skip_completed=False,
        ).run()
        heldout_signature_artifacts[split] = validate_signature(
            signatures / "0000" / "00000.c4_sig",
            item["selection"]["indexed_documents"],
            args.output_dir,
        )
        LocalPipelineExecutor(
            pipeline=[
                ExactFindDedups(
                    data_folder=str(signatures),
                    output_folder=str(duplicates),
                    index_folder=str(train_index),
                    config=dedup_config,
                    lines_to_buffer=LINES_TO_BUFFER,
                )
            ],
            tasks=1,
            workers=1,
            logging_dir=str(logs / f"{split}-find"),
            skip_completed=False,
        ).run()
        (
            candidate_ordinals[split],
            duplicate_artifacts[split],
        ) = parse_candidate_ordinals(
            duplicates, item["selection"]["indexed_documents"]
        )

    hash_fc = create_hash_func(hash_config, str)
    candidates = []
    split_candidates = {}
    for split, item in heldout.items():
        reader = ParquetReader(
            data_folder=source_folder,
            glob_pattern=Path(item["source_path"]).name,
            limit=item["selection"]["indexed_documents"],
            read_metadata=False,
            recursive=False,
            shuffle_files=False,
        )
        split_candidates[split] = replay_heldout(
            reader,
            item,
            candidate_ordinals[split],
            hash_fc,
            split,
        )
        candidates.extend(split_candidates[split])

    candidates_by_hash = {}
    for candidate in candidates:
        candidates_by_hash.setdefault(candidate["hash64"], []).append(candidate)
    replayed_training_documents = 0
    if candidates:
        for source in training["sources"]:
            reader = ParquetReader(
                data_folder=source_folder,
                glob_pattern=Path(source["path"]).name,
                limit=-1,
                read_metadata=False,
                recursive=False,
                shuffle_files=False,
            )
            source_documents = 0
            for document in reader.run():
                possible = candidates_by_hash.get(hash_fc(document.text))
                if possible:
                    raw = document.text.encode("utf-8")
                    digest = sha256_bytes(raw)
                    for candidate in possible:
                        if candidate["sha256"] == digest and candidate["raw"] == raw:
                            candidate["confirmed"] = True
                source_documents += 1
            if source_documents != source["indexed_documents"]:
                raise RuntimeError(
                    f"training source replay count differs: {source['path']}"
                )
            replayed_training_documents += source_documents
        if replayed_training_documents != training["documents"]:
            raise RuntimeError("training replay total differs")

    config = {
        "confirmation": [
            "sha256(raw UTF-8)",
            "raw UTF-8 byte equality",
        ],
        "content": "released text without normalization",
        "dataset": {
            "repo": FINEWEB_EDU_REPO,
            "revision": FINEWEB_EDU_REVISION,
        },
        "datatrove_commit": DATATROVE_COMMIT,
        "hash_config": {"hash_fc": "sha1", "precision": 64},
        "lines_to_buffer": LINES_TO_BUFFER,
        "only_dedup_in_index": True,
        "scope": (
            "full selected training source files, including indexed-capacity "
            "overshoot beyond planned consumption"
        ),
        "training_source_paths": [
            source["path"] for source in training["sources"]
        ],
        "development": {
            "indexed_documents": heldout["development"]["selection"][
                "indexed_documents"
            ],
            "source_path": heldout["development"]["source_path"],
        },
        "final": {
            "indexed_documents": heldout["final"]["selection"][
                "indexed_documents"
            ],
            "source_path": heldout["final"]["source_path"],
        },
    }
    config_bytes = json.dumps(
        config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    report = {
        "config": config,
        "config_sha256": sha256_bytes(config_bytes),
        "inputs": {
            "development": heldout["development"]["input"],
            "final": heldout["final"]["input"],
            "training": training["input"],
        },
        "data_mutated": False,
        "heldout_signature_artifacts": heldout_signature_artifacts,
        "measurement_only": True,
        "schema_version": 1,
        "splits": {
            split: split_report(
                heldout[split],
                split_candidates[split],
                duplicate_artifacts[split],
            )
            for split in ("development", "final")
        },
        "training": {
            "confirmation_scan_performed": bool(candidates),
            "indexed_documents": training["documents"],
            "indexed_tokens": training["tokens"],
            "replayed_documents": replayed_training_documents,
            "signature_artifacts": train_signature_artifacts,
        },
        "training_index": {
            "artifact_path": str(train_index_path.relative_to(args.output_dir)),
            "sha256": sha256_file(train_index_path),
            "size_bytes": train_index_path.stat().st_size,
        },
    }
    report_bytes = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    partial_path = args.output_dir / "exact-audit.json.partial"
    report_path = args.output_dir / "exact-audit.json"
    with partial_path.open("xb") as stream:
        stream.write(report_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial_path, report_path)
    print(f"report={report_path}")
    for split in ("development", "final"):
        result = report["splits"][split]
        print(
            f"{split}_matched_documents={result['matched_documents']}"
            f" {split}_matched_indexed_tokens="
            f"{result['matched_indexed_tokens']}"
            f" {split}_matched_utf8_bytes={result['matched_utf8_bytes']}"
        )


if __name__ == "__main__":
    main()
