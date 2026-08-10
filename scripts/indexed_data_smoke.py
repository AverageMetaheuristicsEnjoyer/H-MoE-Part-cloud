#!/usr/bin/env python3

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi

from prepare_fineweb_edu_pilot import (
    GPT2_EOD,
    GPT2_VOCAB_SIZE,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    load_megatron,
    prepare_tokenizer,
    sha256_file,
    verify_index,
)
from prepare_fineweb_edu_datatrove import select_indexed_prefix


DATATROVE_COMMIT = "87f7bad5c4a56ec648265fbf0b91d7d226bad428"
TEXTS = (
    "hello world",
    "A deterministic byte-level BPE smoke.\n",
    "naïve café — Καλημέρα",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--megatron-root",
        default=Path("/workspace/third_party/Megatron-LM"),
        type=Path,
    )
    parser.add_argument(
        "--datatrove-root",
        default=Path("/workspace/third_party/datatrove"),
        type=Path,
    )
    parser.add_argument(
        "--datatrove-site-packages",
        default=Path("/workspace/.venv-data/lib/python3.12/site-packages"),
        type=Path,
    )
    return parser.parse_args()


def build_reference(indexed_dataset, tokenizer, root):
    prefix = root / "text_document_00000_tokens"
    dtype = indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size)
    assert dtype == np.uint16
    builder = indexed_dataset.IndexedDatasetBuilder(f"{prefix}.bin", dtype=dtype)
    entries = []
    for text in TEXTS:
        ids = tokenizer.tokenize(text) + [GPT2_EOD]
        payload = np.asarray(ids, dtype=np.uint16)
        builder.add_document(torch.tensor(ids, dtype=torch.int32), [len(ids)])
        entries.append(
            {
                "indexed_token_count": len(ids),
                "indexed_ids_sha256": hashlib.sha256(payload.tobytes()).hexdigest(),
            }
        )
    builder.finalize(f"{prefix}.idx")
    verify_index(indexed_dataset, prefix, entries)
    return prefix, entries


def build_datatrove(
    MegatronDocumentTokenizer,
    Document,
    tokenizer_json,
    texts,
    root,
    name,
    batch_size,
):
    MegatronDocumentTokenizer(
        output_folder=str(root),
        save_filename=name,
        tokenizer_name_or_path=str(tokenizer_json),
        eos_token="<|endoftext|>",
        batch_size=batch_size,
    ).run(
        (Document(text=text, id=str(index)) for index, text in enumerate(texts)),
        rank=0,
        world_size=1,
    )
    return root / f"{name}_00000_tokens"


def main():
    if not __debug__:
        raise RuntimeError("run smoke tests without Python optimization")

    args = parse_args()
    datatrove_revision = subprocess.run(
        ["git", "-C", str(args.datatrove_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert datatrove_revision == DATATROVE_COMMIT, datatrove_revision
    assert args.datatrove_site_packages.is_dir(), args.datatrove_site_packages
    sys.path.append(str(args.datatrove_site_packages))
    sys.path.insert(0, str(args.datatrove_root / "src"))
    from datatrove.data import Document
    from datatrove.pipeline.tokens import MegatronDocumentTokenizer, TokensCounter

    indexed_dataset, MegatronTokenizer = load_megatron(args.megatron_root)
    tokenizer, tokenizer_files = prepare_tokenizer(HfApi(), MegatronTokenizer)
    assert tokenizer.vocab_size == GPT2_VOCAB_SIZE
    assert tokenizer.eod == GPT2_EOD

    backend = tokenizer._tokenizer.tokenizer
    tokenizer_json = Path(backend.name_or_path) / "tokenizer.json"
    assert tokenizer_json.is_file(), tokenizer_json

    with (
        tempfile.TemporaryDirectory() as reference_dir,
        tempfile.TemporaryDirectory() as first_dir,
        tempfile.TemporaryDirectory() as second_dir,
        tempfile.TemporaryDirectory() as merge_dir,
        tempfile.TemporaryDirectory() as selected_dir,
    ):
        reference_prefix, entries = build_reference(
            indexed_dataset, tokenizer, Path(reference_dir)
        )
        datatrove_prefixes = []
        for output_dir, batch_size in ((first_dir, 2), (second_dir, 10_000)):
            prefix = build_datatrove(
                MegatronDocumentTokenizer,
                Document,
                tokenizer_json,
                TEXTS,
                Path(output_dir),
                "text_document",
                batch_size,
            )
            verify_index(indexed_dataset, prefix, entries)
            datatrove_prefixes.append(prefix)

        reference_hashes = {
            suffix: sha256_file(Path(f"{reference_prefix}.{suffix}"))
            for suffix in ("bin", "idx")
        }
        for prefix in datatrove_prefixes:
            hashes = {
                suffix: sha256_file(Path(f"{prefix}.{suffix}"))
                for suffix in ("bin", "idx")
            }
            assert hashes == reference_hashes, (hashes, reference_hashes)

        merge_root = Path(merge_dir)
        first_part = build_datatrove(
            MegatronDocumentTokenizer,
            Document,
            tokenizer_json,
            TEXTS[:2],
            merge_root / "first_part",
            "part",
            10_000,
        )
        second_part = build_datatrove(
            MegatronDocumentTokenizer,
            Document,
            tokenizer_json,
            TEXTS[2:],
            merge_root / "second_part",
            "part",
            10_000,
        )
        merged_hashes = []
        for name in ("merged_first", "merged_second"):
            prefix = merge_root / name
            builder = indexed_dataset.IndexedDatasetBuilder(
                f"{prefix}.bin", dtype=np.uint16
            )
            builder.add_index(str(first_part))
            builder.add_index(str(second_part))
            builder.finalize(f"{prefix}.idx")
            verify_index(indexed_dataset, prefix, entries)
            merged_hashes.append(
                {
                    suffix: sha256_file(Path(f"{prefix}.{suffix}"))
                    for suffix in ("bin", "idx")
                }
            )
        assert merged_hashes[0] == merged_hashes[1] == reference_hashes

        selection_root = Path(selected_dir)
        selection_path = selection_root / "selection.json"
        selection_target = entries[0]["indexed_token_count"] + 1
        counted_documents = TokensCounter(
            tokenizer_name_or_path=str(tokenizer_json),
            count_eos_token=True,
            batch_size=2,
        ).run(
            (Document(text=text, id=str(index)) for index, text in enumerate(TEXTS)),
            rank=0,
            world_size=1,
        )
        MegatronDocumentTokenizer(
            output_folder=str(selection_root),
            save_filename="selected",
            tokenizer_name_or_path=str(tokenizer_json),
            eos_token="<|endoftext|>",
            batch_size=2,
        ).run(
            select_indexed_prefix(
                counted_documents,
                rank=0,
                world_size=1,
                minimum_indexed_tokens=selection_target,
                selection_path=selection_path,
            ),
            rank=0,
            world_size=1,
        )
        selected_prefix = selection_root / "selected_00000_tokens"
        verify_index(indexed_dataset, selected_prefix, entries[:2])
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        source_text_digest = hashlib.sha256()
        for text in TEXTS[:2]:
            text_bytes = text.encode("utf-8")
            source_text_digest.update(struct.pack("<Q", len(text_bytes)))
            source_text_digest.update(text_bytes)
        assert selection == {
            "indexed_documents": 2,
            "indexed_tokens": sum(
                entry["indexed_token_count"] for entry in entries[:2]
            ),
            "minimum_indexed_tokens": selection_target,
            "overshoot_indexed_tokens": (
                entries[1]["indexed_token_count"] - 1
            ),
            "previous_indexed_tokens": entries[0]["indexed_token_count"],
            "source_text_sequence_hash": {
                "algorithm": "sha256(concat(length_uint64_le || text_utf8))",
                "digest": source_text_digest.hexdigest(),
            },
            "source_utf8_bytes": sum(
                len(text.encode("utf-8")) for text in TEXTS[:2]
            ),
        }

    print(
        f"tokenizer=pass repo={TOKENIZER_REPO} revision={TOKENIZER_REVISION}"
        f" files={len(tokenizer_files)} vocab={tokenizer.vocab_size} eod={tokenizer.eod}"
    )
    print(
        f"datatrove_megatron_parity=pass commit={DATATROVE_COMMIT}"
        f" dtype=uint16 bin_sha256={reference_hashes['bin']}"
        f" idx_sha256={reference_hashes['idx']}"
        " repeats=2 batch_sizes=2,10000"
    )
    print("mcore_add_index_merge=pass source_prefixes=2 repeats=2")
    print(
        "minimum_indexed_prefix=pass"
        f" target={selection_target}"
        f" documents={selection['indexed_documents']}"
        f" indexed_tokens={selection['indexed_tokens']}"
        f" source_utf8_bytes={selection['source_utf8_bytes']}"
    )


if __name__ == "__main__":
    main()
