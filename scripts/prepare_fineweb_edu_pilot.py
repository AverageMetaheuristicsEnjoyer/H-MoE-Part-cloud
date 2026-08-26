#!/usr/bin/env python3

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from numbers import Integral
from pathlib import Path

import huggingface_hub
import numpy as np
import pyarrow
import pyarrow.parquet as pq
import torch
import transformers
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers import GPT2TokenizerFast


FINEWEB_EDU_REPO = "HuggingFaceFW/fineweb_edu_100BT-shuffled"
FINEWEB_EDU_PATH = "data"
FINEWEB_EDU_REVISION = "be6b2a50d3a9c60d330c45384e80c7863cd3a25d"
TOKENIZER_REPO = "openai-community/gpt2"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
MEGATRON_COMMIT = "571370c829ca768fe37244f4e2e7f28d8accc4ab"
SEQUENCE_LENGTH = 2_048
PILOT_SAMPLES = 4_883
TARGET_LOSS_TOKENS = PILOT_SAMPLES * SEQUENCE_LENGTH
MINIMUM_INDEXED_TOKENS = TARGET_LOSS_TOKENS + 1
VALIDATION_MODULUS = 1_000
GPT2_VOCAB_SIZE = 50_257
GPT2_EOD = 50_256
TOKENIZER_FILES = (
    "added_tokens.json",
    "config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_line(stream, value):
    stream.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    )


def load_megatron(megatron_root):
    megatron_root = megatron_root.resolve()
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

    sys.modules["transformer_engine"] = None
    sys.modules["transformer_engine_torch"] = None
    sys.path.insert(0, str(megatron_root))
    from megatron.core.datasets import indexed_dataset
    from megatron.core.tokenizers import MegatronTokenizer

    module_path = Path(indexed_dataset.__file__).resolve()
    if not module_path.is_relative_to(megatron_root):
        raise RuntimeError(f"loaded Megatron from {module_path}, not {megatron_root}")
    return indexed_dataset, MegatronTokenizer


def prepare_tokenizer(api, MegatronTokenizer):
    resolved_revision = api.model_info(
        TOKENIZER_REPO, revision=TOKENIZER_REVISION
    ).sha
    if resolved_revision != TOKENIZER_REVISION:
        raise RuntimeError(
            f"tokenizer resolved to {resolved_revision}, expected {TOKENIZER_REVISION}"
        )

    snapshot = Path(
        snapshot_download(
            repo_id=TOKENIZER_REPO,
            revision=TOKENIZER_REVISION,
            allow_patterns=list(TOKENIZER_FILES),
        )
    )
    for required_file in ("vocab.json", "merges.txt"):
        if not (snapshot / required_file).is_file():
            raise RuntimeError(f"tokenizer snapshot lacks {required_file}")

    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path=str(snapshot),
        metadata_path={"library": "huggingface"},
        additional_special_tokens=[],
        include_special_tokens=False,
        trust_remote_code=False,
        use_fast=True,
    )
    backend = tokenizer._tokenizer.tokenizer
    if not isinstance(backend, GPT2TokenizerFast):
        raise RuntimeError(f"expected GPT2TokenizerFast, got {type(backend).__name__}")
    if tokenizer.vocab_size != GPT2_VOCAB_SIZE or len(backend) != GPT2_VOCAB_SIZE:
        raise RuntimeError(
            f"expected GPT-2 vocabulary {GPT2_VOCAB_SIZE}, got "
            f"{tokenizer.vocab_size}/{len(backend)}"
        )
    if tokenizer.eod != GPT2_EOD:
        raise RuntimeError(f"expected GPT-2 EOD {GPT2_EOD}, got {tokenizer.eod}")

    files = []
    for filename in TOKENIZER_FILES:
        path = snapshot / filename
        if path.is_file():
            files.append(
                {
                    "path": filename,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return tokenizer, files


def verify_index(indexed_dataset, prefix, entries):
    dataset = indexed_dataset.IndexedDataset(
        str(prefix), multimodal=False, mmap=True
    )
    if dataset.index.dtype != np.uint16:
        raise RuntimeError(f"{prefix.name} uses {dataset.index.dtype}, not uint16")
    if len(dataset) != len(entries):
        raise RuntimeError(
            f"{prefix.name} has {len(dataset)} sequences for {len(entries)} documents"
        )
    expected_boundaries = np.arange(len(entries) + 1, dtype=np.int64)
    if not np.array_equal(dataset.document_indices, expected_boundaries):
        raise RuntimeError(f"{prefix.name} does not contain one sequence per document")

    indexed_tokens = 0
    for sequence_index, entry in enumerate(entries):
        sequence = dataset[sequence_index]
        if len(sequence) != entry["indexed_token_count"]:
            raise RuntimeError(
                f"{prefix.name} sequence {sequence_index} has the wrong length"
            )
        if int(sequence[-1]) != GPT2_EOD:
            raise RuntimeError(
                f"{prefix.name} sequence {sequence_index} lacks the appended EOD"
            )
        sequence_hash = hashlib.sha256(
            np.asarray(sequence, dtype=np.uint16).tobytes(order="C")
        ).hexdigest()
        if sequence_hash != entry["indexed_ids_sha256"]:
            raise RuntimeError(
                f"{prefix.name} sequence {sequence_index} differs from its manifest"
            )
        indexed_tokens += len(sequence)

    del sequence
    del dataset
    expected_bytes = indexed_tokens * np.dtype(np.uint16).itemsize
    actual_bytes = Path(f"{prefix}.bin").stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"{prefix.name}.bin has {actual_bytes} bytes, expected {expected_bytes}"
        )
    return indexed_tokens


def main():
    args = parse_args()
    start_time = time.monotonic()

    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise RuntimeError(f"output path is not a directory: {args.output_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    indexed_dataset, MegatronTokenizer = load_megatron(args.megatron_root)
    api = HfApi()

    fineweb_edu_revision = api.dataset_info(
        FINEWEB_EDU_REPO, revision=FINEWEB_EDU_REVISION
    ).sha
    if fineweb_edu_revision != FINEWEB_EDU_REVISION:
        raise RuntimeError(
            f"FineWeb-Edu resolved to {fineweb_edu_revision}, "
            f"expected {FINEWEB_EDU_REVISION}"
        )

    tokenizer, tokenizer_files = prepare_tokenizer(api, MegatronTokenizer)
    token_dtype = indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size)
    if token_dtype != np.uint16:
        raise RuntimeError(f"expected uint16 token payload, got {token_dtype}")

    parquet_files = sorted(
        entry.path
        for entry in api.list_repo_tree(
            FINEWEB_EDU_REPO,
            path_in_repo=FINEWEB_EDU_PATH,
            recursive=True,
            repo_type="dataset",
            revision=FINEWEB_EDU_REVISION,
        )
        if getattr(entry, "path", "").endswith(".parquet")
    )
    if not parquet_files:
        raise RuntimeError(f"no Parquet files found under {FINEWEB_EDU_PATH}")

    train_prefix = args.output_dir / "train_text_document"
    valid_prefix = args.output_dir / "valid_text_document"
    train_builder = indexed_dataset.IndexedDatasetBuilder(
        f"{train_prefix}.bin", dtype=token_dtype
    )
    valid_builder = indexed_dataset.IndexedDatasetBuilder(
        f"{valid_prefix}.bin", dtype=token_dtype
    )

    document_manifest_path = args.output_dir / "documents.jsonl"
    entries = []
    seen_ids = set()
    source_files = []
    train_documents = 0
    valid_documents = 0
    train_indexed_tokens = 0
    valid_indexed_tokens = 0
    train_source_bytes = 0
    valid_source_bytes = 0
    last_source_file = None
    last_source_row = None
    target_reached = False

    with document_manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
        for source_file in parquet_files:
            local_path = Path(
                hf_hub_download(
                    repo_id=FINEWEB_EDU_REPO,
                    filename=source_file,
                    repo_type="dataset",
                    revision=FINEWEB_EDU_REVISION,
                )
            )
            parquet = pq.ParquetFile(local_path)
            source_record = {
                "path": source_file,
                "rows": parquet.metadata.num_rows,
                "rows_scanned": 0,
                "sha256": sha256_file(local_path),
                "size_bytes": local_path.stat().st_size,
            }
            source_files.append(source_record)
            row_base = 0

            for batch in parquet.iter_batches(
                batch_size=1_024, columns=["id", "text", "token_count"]
            ):
                columns = batch.to_pydict()
                for offset in range(batch.num_rows):
                    source_row = row_base + offset
                    fineweb_edu_id = columns["id"][offset]
                    text = columns["text"][offset]
                    fineweb_edu_token_count = columns["token_count"][offset]
                    if not isinstance(fineweb_edu_id, str) or not fineweb_edu_id:
                        raise RuntimeError(
                            f"invalid FineWeb-Edu id at {source_file}:{source_row}"
                        )
                    if fineweb_edu_id in seen_ids:
                        raise RuntimeError(
                            f"duplicate FineWeb-Edu id: {fineweb_edu_id}"
                        )
                    seen_ids.add(fineweb_edu_id)
                    if not isinstance(text, str) or not text:
                        raise RuntimeError(
                            f"invalid text at {source_file}:{source_row}"
                        )
                    if isinstance(fineweb_edu_token_count, bool) or not isinstance(
                        fineweb_edu_token_count, Integral
                    ):
                        raise RuntimeError(
                            f"invalid token_count at {source_file}:{source_row}"
                        )

                    text_bytes = text.encode("utf-8")
                    is_validation = (
                        int.from_bytes(hashlib.sha256(text_bytes).digest(), "big")
                        % VALIDATION_MODULUS
                        == 0
                    )
                    token_ids = tokenizer.tokenize(text)
                    if (
                        not token_ids
                        or min(token_ids) < 0
                        or max(token_ids) >= GPT2_VOCAB_SIZE
                    ):
                        raise RuntimeError(
                            f"invalid GPT-2 IDs at {source_file}:{source_row}"
                        )

                    indexed_ids = token_ids + [GPT2_EOD]
                    indexed_array = np.asarray(indexed_ids, dtype=np.uint16)

                    if is_validation:
                        split = "valid"
                        train_ordinal = None
                        valid_ordinal = valid_documents
                        valid_builder.add_document(
                            torch.tensor(indexed_ids, dtype=torch.int32),
                            [len(indexed_ids)],
                        )
                        valid_documents += 1
                        valid_indexed_tokens += len(indexed_ids)
                        valid_source_bytes += len(text_bytes)
                    else:
                        split = "train"
                        train_ordinal = train_documents
                        valid_ordinal = None
                        train_builder.add_document(
                            torch.tensor(indexed_ids, dtype=torch.int32),
                            [len(indexed_ids)],
                        )
                        train_documents += 1
                        train_indexed_tokens += len(indexed_ids)
                        train_source_bytes += len(text_bytes)

                    entry = {
                        "fineweb_edu_id": fineweb_edu_id,
                        "fineweb_edu_minus_tokenizer_token_count": (
                            int(fineweb_edu_token_count) - len(token_ids)
                        ),
                        "fineweb_edu_token_count": int(fineweb_edu_token_count),
                        "indexed_ids_sha256": hashlib.sha256(
                            indexed_array.tobytes(order="C")
                        ).hexdigest(),
                        "indexed_token_count": len(indexed_ids),
                        "source_file": source_file,
                        "source_row": source_row,
                        "source_utf8_bytes": len(text_bytes),
                        "split": split,
                        "text_sha256": hashlib.sha256(text_bytes).hexdigest(),
                        "tokenizer_token_count": len(token_ids),
                        "train_ordinal": train_ordinal,
                        "valid_ordinal": valid_ordinal,
                    }
                    entries.append(entry)
                    write_json_line(manifest, entry)
                    source_record["rows_scanned"] += 1
                    last_source_file = source_file
                    last_source_row = source_row

                    if train_indexed_tokens >= MINIMUM_INDEXED_TOKENS:
                        target_reached = True
                        break

                if target_reached:
                    break
                row_base += batch.num_rows
            if target_reached:
                break

    train_builder.finalize(f"{train_prefix}.idx")
    valid_builder.finalize(f"{valid_prefix}.idx")

    if not target_reached:
        raise RuntimeError("FineWeb-Edu source ended before the pilot target")
    if valid_documents == 0:
        raise RuntimeError("the pilot scan selected no validation documents")

    train_entries = [entry for entry in entries if entry["split"] == "train"]
    valid_entries = [entry for entry in entries if entry["split"] == "valid"]
    fineweb_edu_token_count = sum(
        entry["fineweb_edu_token_count"] for entry in entries
    )
    tokenizer_token_count = sum(entry["tokenizer_token_count"] for entry in entries)
    token_count_mismatch_documents = sum(
        entry["fineweb_edu_token_count"] != entry["tokenizer_token_count"]
        for entry in entries
    )
    token_count_absolute_drift = sum(
        abs(entry["fineweb_edu_minus_tokenizer_token_count"]) for entry in entries
    )
    if [entry["train_ordinal"] for entry in train_entries] != list(
        range(train_documents)
    ):
        raise RuntimeError("training ordinals are not contiguous")
    if [entry["valid_ordinal"] for entry in valid_entries] != list(
        range(valid_documents)
    ):
        raise RuntimeError("validation ordinals are not contiguous")
    if sum(entry["source_utf8_bytes"] for entry in train_entries) != train_source_bytes:
        raise RuntimeError("training byte total differs from the manifest")
    if sum(entry["source_utf8_bytes"] for entry in valid_entries) != valid_source_bytes:
        raise RuntimeError("validation byte total differs from the manifest")
    verified_train_tokens = verify_index(
        indexed_dataset, train_prefix, train_entries
    )
    verified_valid_tokens = verify_index(
        indexed_dataset, valid_prefix, valid_entries
    )
    if verified_train_tokens != train_indexed_tokens:
        raise RuntimeError("training token total differs from the manifest")
    if verified_valid_tokens != valid_indexed_tokens:
        raise RuntimeError("validation token total differs from the manifest")
    if verified_train_tokens < MINIMUM_INDEXED_TOKENS:
        raise RuntimeError("training data cannot supply the requested pilot samples")

    artifact_paths = (
        document_manifest_path,
        Path(f"{train_prefix}.bin"),
        Path(f"{train_prefix}.idx"),
        Path(f"{valid_prefix}.bin"),
        Path(f"{valid_prefix}.idx"),
    )
    artifacts = {
        path.name: {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in artifact_paths
    }
    summary = {
        "artifacts": artifacts,
        "counts": {
            "fineweb_edu_minus_tokenizer_tokens": (
                fineweb_edu_token_count - tokenizer_token_count
            ),
            "fineweb_edu_token_count": fineweb_edu_token_count,
            "scanned_documents": len(entries),
            "token_count_absolute_drift": token_count_absolute_drift,
            "token_count_mismatch_documents": token_count_mismatch_documents,
            "tokenizer_token_count": tokenizer_token_count,
            "train_documents": train_documents,
            "train_indexed_tokens": train_indexed_tokens,
            "train_source_utf8_bytes": train_source_bytes,
            "valid_documents": valid_documents,
            "valid_indexed_tokens": valid_indexed_tokens,
            "valid_source_utf8_bytes": valid_source_bytes,
        },
        "environment": {
            "huggingface_hub": huggingface_hub.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "fineweb_edu": {
            "available_parquet_files": len(parquet_files),
            "path": FINEWEB_EDU_PATH,
            "repo": FINEWEB_EDU_REPO,
            "revision": FINEWEB_EDU_REVISION,
            "source_files": source_files,
            "token_count_semantics": (
                "advisory inherited metadata; selection uses recomputed tokenizer IDs"
            ),
        },
        "format": {
            "append_eod": GPT2_EOD,
            "one_indexed_sequence_per_document": True,
            "token_dtype": "uint16",
        },
        "megatron": {
            "commit": MEGATRON_COMMIT,
        },
        "selection": {
            "last_scanned_source_file": last_source_file,
            "last_scanned_source_row": last_source_row,
            "minimum_indexed_tokens": MINIMUM_INDEXED_TOKENS,
            "next_train_ordinal": train_documents,
            "pilot_samples": PILOT_SAMPLES,
            "sequence_length": SEQUENCE_LENGTH,
            "target_loss_tokens": TARGET_LOSS_TOKENS,
            "train_ordinal_range": [0, train_documents],
            "validation_rule": (
                "int.from_bytes(sha256(text_utf8), 'big') % 1000 == 0"
            ),
        },
        "tokenizer": {
            "eod": GPT2_EOD,
            "files": tokenizer_files,
            "repo": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "vocab_size": GPT2_VOCAB_SIZE,
        },
    }
    summary_path = args.output_dir / "manifest.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    elapsed = time.monotonic() - start_time
    print(f"wrote {train_indexed_tokens:,} training token IDs")
    print(f"wrote {valid_indexed_tokens:,} validation token IDs")
    print(f"elapsed seconds: {elapsed:.3f}")
    print(f"manifest: {summary_path}")


if __name__ == "__main__":
    main()
