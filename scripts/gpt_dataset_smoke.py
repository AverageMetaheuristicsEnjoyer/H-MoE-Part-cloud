#!/usr/bin/env python3

import argparse
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi

from prepare_fineweb_edu_pilot import (
    PILOT_SAMPLES,
    SEQUENCE_LENGTH,
    load_megatron,
    prepare_tokenizer,
    sha256_file,
)


RANDOM_SEED = 1_234


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--megatron-root",
        default=Path("/workspace/third_party/Megatron-LM"),
        type=Path,
    )
    parser.add_argument("--data-prefix", required=True, type=Path)
    parser.add_argument("--samples", default=PILOT_SAMPLES, type=int)
    parser.add_argument("--sequence-length", default=SEQUENCE_LENGTH, type=int)
    return parser.parse_args()


def hash_array(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_once(
    GPTDataset,
    GPTDatasetConfig,
    Split,
    tokenizer,
    cache_dir,
    data_prefix,
    samples,
    sequence_length,
):
    config = GPTDatasetConfig(
        random_seed=RANDOM_SEED,
        sequence_length=sequence_length,
        blend=([str(data_prefix)], None),
        split="1,0,0",
        path_to_cache=str(cache_dir),
        mmap_bin_files=True,
        tokenizer=tokenizer,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
    )
    indexed = GPTDataset.build_low_level_dataset(str(data_prefix), config)
    assert indexed.index.dtype == np.uint16, indexed.index.dtype
    document_ids = np.arange(
        GPTDataset.numel_low_level_dataset(indexed), dtype=np.int32
    )
    indexed_tokens = int(np.sum(indexed.sequence_lengths, dtype=np.int64))
    assert indexed_tokens >= samples * sequence_length + 1, indexed_tokens

    dataset = GPTDataset(
        indexed,
        str(data_prefix),
        document_ids,
        samples,
        Split.train,
        config,
    )
    tokens_per_epoch = dataset._get_num_tokens_per_epoch()
    num_epochs = dataset._get_num_epochs(tokens_per_epoch)
    assert num_epochs == 1, num_epochs
    assert tokens_per_epoch == indexed_tokens
    assert dataset.document_index.dtype == np.int32
    assert len(dataset.document_index) == len(document_ids)
    np.testing.assert_array_equal(np.sort(dataset.document_index), document_ids)

    available_samples = dataset.sample_index.shape[0] - 1
    assert available_samples >= samples, available_samples
    assert len(dataset) == available_samples
    assert dataset.sample_index.ndim == 2 and dataset.sample_index.shape[1] == 2
    assert dataset.shuffle_index.shape == (available_samples,)
    np.testing.assert_array_equal(
        np.sort(dataset.shuffle_index),
        np.arange(available_samples, dtype=dataset.shuffle_index.dtype),
    )

    for sample_id in (0, samples - 1):
        sample = dataset[sample_id]
        assert sample["tokens"].shape == (sequence_length,)
        assert sample["labels"].shape == (sequence_length,)
        assert sample["loss_mask"].shape == (sequence_length,)
        assert sample["position_ids"].shape == (sequence_length,)
        assert sample["tokens"].dtype == torch.int64
        assert sample["labels"].dtype == torch.int64
        assert sample["loss_mask"].dtype == torch.float32
        assert sample["position_ids"].dtype == torch.int64

    arrays = {
        "document_index": dataset.document_index,
        "sample_index": dataset.sample_index,
        "shuffle_index": dataset.shuffle_index,
    }
    hashes = {}
    for name, array in arrays.items():
        cache_files = list(cache_dir.glob(f"*-{name}.npy"))
        assert len(cache_files) == 1, (name, cache_files)
        hashes[name] = {
            "array_sha256": hash_array(array),
            "npy_sha256": sha256_file(cache_files[0]),
            "shape": list(array.shape),
            "dtype": array.dtype.str,
        }

    return {
        "available_samples": available_samples,
        "documents": len(document_ids),
        "hashes": hashes,
        "indexed_tokens": indexed_tokens,
        "num_epochs": num_epochs,
    }


def main():
    if not __debug__:
        raise RuntimeError("run smoke tests without Python optimization")

    args = parse_args()
    assert args.samples > 0
    assert args.sequence_length > 0
    assert Path(f"{args.data_prefix}.bin").is_file()
    assert Path(f"{args.data_prefix}.idx").is_file()

    _, MegatronTokenizer = load_megatron(args.megatron_root)
    tokenizer, _ = prepare_tokenizer(HfApi(), MegatronTokenizer)
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.core.datasets.utils import Split, compile_helpers

    compile_helpers()

    with tempfile.TemporaryDirectory(
        prefix="gpt-dataset-cache-1-"
    ) as first_cache, tempfile.TemporaryDirectory(
        prefix="gpt-dataset-cache-2-"
    ) as second_cache:
        first = build_once(
            GPTDataset,
            GPTDatasetConfig,
            Split,
            tokenizer,
            Path(first_cache),
            args.data_prefix,
            args.samples,
            args.sequence_length,
        )
        second = build_once(
            GPTDataset,
            GPTDatasetConfig,
            Split,
            tokenizer,
            Path(second_cache),
            args.data_prefix,
            args.samples,
            args.sequence_length,
        )

    assert first == second, (first, second)
    print(
        f"gpt_dataset=pass prefix={args.data_prefix}"
        f" requested_samples={args.samples}"
        f" available_samples={first['available_samples']}"
        f" documents={first['documents']}"
        f" indexed_tokens={first['indexed_tokens']}"
        f" epochs={first['num_epochs']}"
    )
    for name, values in first["hashes"].items():
        print(
            f"{name}=pass shape={values['shape']} dtype={values['dtype']}"
            f" array_sha256={values['array_sha256']}"
            f" npy_sha256={values['npy_sha256']}"
        )
    print("gpt_dataset_repeat_cache=pass caches=2")
    print(
        f"bin_sha256={sha256_file(Path(f'{args.data_prefix}.bin'))}"
        f" idx_sha256={sha256_file(Path(f'{args.data_prefix}.idx'))}"
    )


if __name__ == "__main__":
    main()
