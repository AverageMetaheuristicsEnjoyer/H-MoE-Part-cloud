#!/usr/bin/env python3

import argparse
import hashlib
import json
import platform
import struct
import subprocess
import sys
from functools import partial
from pathlib import Path

import huggingface_hub
import numpy as np
import pyarrow
import tokenizers
from huggingface_hub import HfApi, snapshot_download


FINEWEB_EDU_REPO = "HuggingFaceFW/fineweb_edu_100BT-shuffled"
FINEWEB_EDU_REVISION = "be6b2a50d3a9c60d330c45384e80c7863cd3a25d"
TOKENIZER_REPO = "openai-community/gpt2"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
TOKENIZER_JSON_SHA256 = (
    "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"
)
DATATROVE_COMMIT = "87f7bad5c4a56ec648265fbf0b91d7d226bad428"
GPT2_VOCAB_SIZE = 50_257
GPT2_EOD = 50_256
GPT2_EOD_TOKEN = "<|endoftext|>"
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
    parser.add_argument("--datatrove-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--source-path",
        default="data/train-00000-of-00100.parquet",
    )
    parser.add_argument("--limit", default=-1, type=int)
    parser.add_argument("--minimum-indexed-tokens", type=int)
    parser.add_argument("--save-filename", default="fineweb_edu")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path):
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repo_file_info(api, path):
    entries = api.get_paths_info(
        FINEWEB_EDU_REPO,
        paths=[path],
        repo_type="dataset",
        revision=FINEWEB_EDU_REVISION,
        expand=True,
    )
    if len(entries) != 1 or entries[0].path != path:
        raise RuntimeError(f"source path did not resolve uniquely: {path}")
    entry = entries[0]
    lfs = entry.lfs
    lfs_sha256 = lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256
    lfs_size = lfs["size"] if isinstance(lfs, dict) else lfs.size
    if entry.size != lfs_size:
        raise RuntimeError(f"repository and LFS sizes differ for {path}")
    return {"sha256": lfs_sha256, "size_bytes": entry.size}


def index_summary(path):
    with path.open("rb") as stream:
        if stream.read(9) != b"MMIDIDX\x00\x00":
            raise RuntimeError("unexpected Megatron index header")
        version = struct.unpack("<Q", stream.read(8))[0]
        dtype_code = struct.unpack("<B", stream.read(1))[0]
        sequences = struct.unpack("<Q", stream.read(8))[0]
        documents = struct.unpack("<Q", stream.read(8))[0]
        lengths = np.frombuffer(stream.read(4 * sequences), dtype=np.int32)
        pointers = np.frombuffer(stream.read(8 * sequences), dtype=np.int64)
        document_indices = np.frombuffer(stream.read(8 * documents), dtype=np.int64)
        if stream.read(1):
            raise RuntimeError("unexpected trailing bytes in Megatron index")
    if version != 1 or dtype_code != 8:
        raise RuntimeError(
            f"unexpected Megatron index version/dtype: {version}/{dtype_code}"
        )
    if documents != sequences + 1:
        raise RuntimeError("Megatron index does not contain one sequence per document")
    if not np.array_equal(document_indices, np.arange(documents, dtype=np.int64)):
        raise RuntimeError("Megatron document boundaries are not contiguous")
    expected_pointers = np.zeros(sequences, dtype=np.int64)
    if sequences > 1:
        expected_pointers[1:] = np.cumsum(
            lengths[:-1], dtype=np.int64
        ) * np.dtype(np.uint16).itemsize
    if not np.array_equal(pointers, expected_pointers):
        raise RuntimeError("Megatron sequence pointers do not match lengths")
    return {
        "indexed_documents": int(sequences),
        "indexed_tokens": int(np.sum(lengths, dtype=np.int64)),
    }


def select_indexed_prefix(
    data,
    rank,
    world_size,
    *,
    minimum_indexed_tokens,
    selection_path,
):
    del rank, world_size
    documents = 0
    indexed_tokens = 0
    previous_indexed_tokens = 0
    source_utf8_bytes = 0
    source_text_digest = hashlib.sha256()
    for document in data:
        token_count = document.metadata["token_count"]
        text_bytes = document.text.encode("utf-8")
        previous_indexed_tokens = indexed_tokens
        indexed_tokens += token_count
        source_utf8_bytes += len(text_bytes)
        source_text_digest.update(struct.pack("<Q", len(text_bytes)))
        source_text_digest.update(text_bytes)
        documents += 1
        yield document
        if indexed_tokens >= minimum_indexed_tokens:
            break
    if indexed_tokens < minimum_indexed_tokens:
        raise RuntimeError(
            f"source ended at {indexed_tokens} indexed tokens, "
            f"below target {minimum_indexed_tokens}"
        )
    selection = {
        "indexed_documents": documents,
        "indexed_tokens": indexed_tokens,
        "minimum_indexed_tokens": minimum_indexed_tokens,
        "overshoot_indexed_tokens": indexed_tokens - minimum_indexed_tokens,
        "previous_indexed_tokens": previous_indexed_tokens,
        "source_text_sequence_hash": {
            "algorithm": "sha256(concat(length_uint64_le || text_utf8))",
            "digest": source_text_digest.hexdigest(),
        },
        "source_utf8_bytes": source_utf8_bytes,
    }
    Path(selection_path).write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if args.limit == 0 or args.limit < -1:
        raise ValueError("--limit must be -1 or a positive integer")
    if (
        args.minimum_indexed_tokens is not None
        and args.minimum_indexed_tokens <= 0
    ):
        raise ValueError("--minimum-indexed-tokens must be positive")
    if args.minimum_indexed_tokens is not None and args.limit != -1:
        raise ValueError("--minimum-indexed-tokens requires --limit -1")
    if "/" in args.save_filename or not args.save_filename:
        raise ValueError("--save-filename must be one nonempty path component")
    if Path(args.source_path).parent.as_posix() != "data":
        raise ValueError("--source-path must name a file directly under data/")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datatrove_root = args.datatrove_root.resolve()
    revision = git_revision(datatrove_root)
    if revision != DATATROVE_COMMIT:
        raise RuntimeError(
            f"DataTrove commit is {revision}, expected {DATATROVE_COMMIT}"
        )
    sys.path.insert(0, str(datatrove_root / "src"))

    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.tokens import MegatronDocumentTokenizer, TokensCounter

    api = HfApi()
    dataset_revision = api.dataset_info(
        FINEWEB_EDU_REPO, revision=FINEWEB_EDU_REVISION
    ).sha
    if dataset_revision != FINEWEB_EDU_REVISION:
        raise RuntimeError(
            f"dataset resolved to {dataset_revision}, expected {FINEWEB_EDU_REVISION}"
        )
    tokenizer_revision = api.model_info(
        TOKENIZER_REPO, revision=TOKENIZER_REVISION
    ).sha
    if tokenizer_revision != TOKENIZER_REVISION:
        raise RuntimeError(
            f"tokenizer resolved to {tokenizer_revision}, expected {TOKENIZER_REVISION}"
        )
    source = repo_file_info(api, args.source_path)

    tokenizer_snapshot = Path(
        snapshot_download(
            repo_id=TOKENIZER_REPO,
            revision=TOKENIZER_REVISION,
            allow_patterns=list(TOKENIZER_FILES),
        )
    )
    tokenizer_json = tokenizer_snapshot / "tokenizer.json"
    if sha256_file(tokenizer_json) != TOKENIZER_JSON_SHA256:
        raise RuntimeError("pinned tokenizer.json has an unexpected SHA-256")
    raw_tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_json))
    if raw_tokenizer.get_vocab_size() != GPT2_VOCAB_SIZE:
        raise RuntimeError(
            f"expected GPT-2 vocabulary {GPT2_VOCAB_SIZE}, "
            f"got {raw_tokenizer.get_vocab_size()}"
        )
    if raw_tokenizer.token_to_id(GPT2_EOD_TOKEN) != GPT2_EOD:
        raise RuntimeError("pinned tokenizer has an unexpected EOD ID")
    if raw_tokenizer.normalizer is not None:
        raise RuntimeError("pinned GPT-2 tokenizer unexpectedly has a normalizer")

    tokens_dir = args.output_dir / "tokens"
    logs_dir = args.output_dir / "logs"
    source_folder = (
        f"hf://datasets/{FINEWEB_EDU_REPO}"
        f"@{FINEWEB_EDU_REVISION}/data"
    )
    source_filename = Path(args.source_path).name
    pipeline = [
        ParquetReader(
            data_folder=source_folder,
            glob_pattern=source_filename,
            limit=args.limit,
            read_metadata=False,
            recursive=False,
            shuffle_files=False,
        )
    ]
    selection_path = args.output_dir / "selection.json"
    if args.minimum_indexed_tokens is not None:
        pipeline.extend(
            [
                TokensCounter(
                    tokenizer_name_or_path=str(tokenizer_json),
                    count_eos_token=True,
                    batch_size=10_000,
                ),
                partial(
                    select_indexed_prefix,
                    minimum_indexed_tokens=args.minimum_indexed_tokens,
                    selection_path=selection_path,
                ),
            ]
        )
    pipeline.append(
        MegatronDocumentTokenizer(
            output_folder=str(tokens_dir),
            save_filename=args.save_filename,
            tokenizer_name_or_path=str(tokenizer_json),
            eos_token=GPT2_EOD_TOKEN,
            batch_size=10_000,
        )
    )
    LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,
        workers=1,
        logging_dir=str(logs_dir),
        skip_completed=False,
    ).run()

    prefix = tokens_dir / f"{args.save_filename}_00000_tokens"
    bin_path = Path(f"{prefix}.bin")
    idx_path = Path(f"{prefix}.idx")
    if not bin_path.is_file() or not idx_path.is_file():
        raise RuntimeError("DataTrove did not create the expected Megatron pair")
    indexed = index_summary(idx_path)
    expected_bin_size = indexed["indexed_tokens"] * np.dtype(np.uint16).itemsize
    if bin_path.stat().st_size != expected_bin_size:
        raise RuntimeError(
            f"token payload has {bin_path.stat().st_size} bytes, "
            f"expected {expected_bin_size}"
        )

    selection = None
    if args.minimum_indexed_tokens is not None:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection["indexed_documents"] != indexed["indexed_documents"]:
            raise RuntimeError("selected and indexed document counts differ")
        if selection["indexed_tokens"] != indexed["indexed_tokens"]:
            raise RuntimeError("selected and indexed token counts differ")
        if not (
            selection["previous_indexed_tokens"]
            < args.minimum_indexed_tokens
            <= selection["indexed_tokens"]
        ):
            raise RuntimeError("selected prefix is not the smallest one meeting target")

    tokenizer_files = []
    for filename in TOKENIZER_FILES:
        path = tokenizer_snapshot / filename
        if path.is_file():
            tokenizer_files.append(
                {
                    "path": filename,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )

    artifacts = {}
    for path in (bin_path, idx_path):
        artifacts[path.name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    artifacts[bin_path.name].update(indexed)
    if selection is not None:
        artifacts[selection_path.name] = {
            "sha256": sha256_file(selection_path),
            "size_bytes": selection_path.stat().st_size,
        }

    manifest = {
        "artifacts": artifacts,
        "conversion": {
            "append_eod_id": GPT2_EOD,
            "batch_size": 10_000,
            "datatrove_commit": DATATROVE_COMMIT,
            "document_order": "physical Parquet row order",
            "implicit_special_tokens": False,
            "limit_documents": args.limit,
            "normalization": None,
            "output_prefix": str(Path("tokens") / prefix.name),
            "tasks": 1,
            "token_dtype": "uint16",
        },
        "environment": {
            "huggingface_hub": huggingface_hub.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
            "python": platform.python_version(),
            "tokenizers": tokenizers.__version__,
        },
        "schema_version": 2 if selection is not None else 1,
        "source": {
            "path": args.source_path,
            "repo": FINEWEB_EDU_REPO,
            "revision": FINEWEB_EDU_REVISION,
            **source,
        },
        "tokenizer": {
            "eod_id": GPT2_EOD,
            "eod_token": GPT2_EOD_TOKEN,
            "files": tokenizer_files,
            "repo": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "vocab_size": GPT2_VOCAB_SIZE,
        },
    }
    if selection is not None:
        manifest["conversion"]["minimum_indexed_tokens"] = (
            args.minimum_indexed_tokens
        )
        manifest["selection"] = selection
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"prefix={prefix}")
    print(f"indexed_documents={indexed['indexed_documents']}")
    print(f"indexed_tokens={indexed['indexed_tokens']}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
