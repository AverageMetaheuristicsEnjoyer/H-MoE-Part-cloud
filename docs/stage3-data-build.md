# Stage 3 FineWeb-Edu data build

This is the authoritative operational contract for the corrected data build.
It supersedes the hash-based split and exact-sized artifact procedure in
[`stage2-data-prep.md`](stage2-data-prep.md) and `design.md` section 7. Those
files remain historical records and are not the production procedure.

Starting this stage authorizes environment setup and bounded pilots. It does
not authorize the full training-data conversion, public upload, or any model
run.

## Fixed inputs

| Item | Pin |
|---|---|
| Source | `HuggingFaceFW/fineweb_edu_100BT-shuffled` |
| Source revision | `be6b2a50d3a9c60d330c45384e80c7863cd3a25d` |
| Tokenizer | `openai-community/gpt2` |
| Tokenizer revision | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| DataTrove | `v0.9.0`, commit `87f7bad5c4a56ec648265fbf0b91d7d226bad428` |
| Megatron-LM | `core_v0.18.2`, commit `571370c829ca768fe37244f4e2e7f28d8accc4ab` |

DataTrove 0.9 requires `huggingface-hub<1`, while the verified training venv
contains Hub 1.10 and Transformers 5.3. Data preparation therefore uses a
separate container-local `.venv-data`. The existing `.venv` and SIF are not
modified. Both venvs run inside the same pinned SIF; the separation is for
Python dependency isolation, not a second container.

The tokenizer contract is:

- encode the released `text` without external normalization;
- use the pinned fast GPT-2 tokenizer, whose JSON has no normalizer;
- add no BOS or other implicit special token;
- append exactly one `<|endoftext|>` ID 50,256 per nonempty document;
- store the 50,257-entry vocabulary as `uint16`.

FineWeb-Edu's inherited `token_count` is metadata, not the experiment's budget.
All capacities are recomputed from the indexed GPT-2 payload, including EOD.

## Split and extension policy

The source was globally shuffled with seed 42 before it was written into 100
Parquet files. Split assignment therefore uses source files, not a new
document hash:

- training candidates are `data/train-00000-of-00100.parquet` upward;
- development is reserved from `data/train-00098-of-00100.parquet`;
- final evaluation is reserved from
  `data/train-00099-of-00100.parquet`;
- files 98 and 99 can never enter training;
- the base training artifact consumes the smallest whole-file prefix whose
  recomputed indexed capacity covers 7.345B tokens;
- an extension starts with the next unused training candidate and records a
  new explicit source-file list.

The generated artifact manifest freezes the exact files actually used.
Training, development, final, base, and extension path sets must be disjoint.
This guarantees disjoint source rows, not unique text: exact or near duplicate
content may still occur in different source files. Cross-split duplicate
measurement remains a separate gate; global re-deduplication is not enabled by
default.

The machine-readable assignment is
[`configs/fineweb_edu_source_plan.json`](../configs/fineweb_edu_source_plan.json).

Whole-source-file artifacts may contain more tokens than the experimental
budget. MCore sample counts and training steps determine how many tokens are
consumed. An extension is a new declared phase/prefix; it is not silently
added to an existing weighted blend.

## Ownership

DataTrove owns Parquet reading, physical row order, batch tokenization, EOD
insertion, and per-worker Megatron `.bin/.idx` writing. Pinned MCore owns
deterministic prefix merging, indexed loading, sample packing, shuffling, and
consumed-sample accounting.

Project code owns only:

- the pinned source-file assignment;
- source, tokenizer, environment, and output hashes;
- generated artifact manifests;
- DataTrove-versus-MCore parity tests;
- bpb/validation slice accounting.

The bespoke [`prepare_fineweb_edu_pilot.py`](../scripts/prepare_fineweb_edu_pilot.py)
is not a production writer. It is retained temporarily as an independent MCore
oracle until the new parity and loader gates pass.

## Bounded pilot

The first pilot reads the first 15,000 documents from training source file 0.
It does not create validation data and is not uploaded. The document limit is
only a bounded plumbing test; the authoritative capacity is the number of
indexed IDs written.

Run the same command into two empty output directories:

```bash
/workspace/.venv-data/bin/python \
  scripts/prepare_fineweb_edu_datatrove.py \
  --datatrove-root /workspace/third_party/datatrove \
  --output-dir /workspace/data/fineweb-edu-datatrove-pilot-run1 \
  --limit 15000 \
  --save-filename fineweb_edu_pilot
```

The script reads the pinned Parquet directly through Hugging Face's filesystem
and records the repository LFS SHA-256 and size. It creates one pair:

```text
tokens/fineweb_edu_pilot_00000_tokens.bin
tokens/fineweb_edu_pilot_00000_tokens.idx
```

and a deterministic `manifest.json`. DataTrove logs are retained separately
because they contain timings.

## Gates before a full conversion

1. The fixed three-document ASCII/newline/Unicode fixture must produce
   byte-identical `.bin` and `.idx` files through DataTrove and pinned MCore.
   MCore must load it as one `uint16` sequence per document with one final EOD.
2. The 15,000-document pilot must run twice with identical `.bin`, `.idx`, and
   deterministic manifest hashes.
3. Pinned MCore `GPTDataset` must load the pilot twice with independent caches,
   produce identical document/sample/shuffle indices, and supply 4,883 samples
   of length 2,048 in one epoch without document repetition.
4. A multi-source fixture must prove that MCore `add_index()` merging in the
   declared source order preserves every document and is deterministic.
5. The source plan must resolve all selected paths at the pinned revision and
   prove train/development/final path-set disjointness.
6. Live durable disk is checked again before the full build.

The full conversion and Hugging Face upload require a separate approval after
these results and an artifact card are shown.

## Still open before final evaluation data

The development and final pools are fixed, but their evaluated whole-document
slices are not yet frozen. We still need to choose document counts that give
approximately 8M and 100M indexed tokens, then record their exact token and
source-byte capacities. bpb must use the UTF-8 bytes belonging to the evaluated
whole documents; a full one-billion-token source shard cannot be used as the
denominator for a smaller evaluated slice.

Whether to run a DataTrove cross-split exact/near-duplicate audit before
freezing those slices also remains a user decision. If enabled, its method and
removal policy will be approved before it mutates any artifact.
