# Stage 2 data-preparation pilot

> Superseded operationally by
> [`stage3-data-build.md`](stage3-data-build.md). This file preserves the
> earlier hash-split proposal and is not the production data procedure.

This file specifies the CPU portion of Stage 2 and the bounded first action of
Stage 3. It does not authorize the full 7.345B-token preparation or any upload.
The environment, indexed-data, and loader gates passed on node207 on
2026-07-29. The data pilot used ordinary FineWeb because the target corpus had
been misidentified; that result is preserved as superseded evidence in
[`stage2-node207.md`](stage2-node207.md). The corrected FineWeb-Edu pilot has
not yet been run.

## Scope

[`prepare_fineweb_edu_pilot.py`](../scripts/prepare_fineweb_edu_pilot.py)
reads the official
[globally shuffled FineWeb-Edu 100BT artifact](https://huggingface.co/datasets/HuggingFaceFW/fineweb_edu_100BT-shuffled)
and stops after it has enough training IDs for 4,883 samples at sequence length
2,048:

```text
4,883 * 2,048 = 10,000,384 loss tokens
minimum indexed payload = 10,000,385 token IDs
```

Whole documents can make the actual indexed payload slightly larger. Each
nonempty document is encoded without normalization or implicit special tokens,
then one GPT-2 EOD ID 50,256 is appended. The token payload is `uint16`.
The cutoff uses these recomputed IDs, including EOD separators, because they
are the tokens consumed by training.

FineWeb-Edu inherits FineWeb's stored `token_count`. That field is advisory
rather than an equality oracle: FineWeb's published
[DataTrove pipeline](https://github.com/huggingface/datatrove/blob/main/examples/fineweb.py)
runs `TokensCounter()` before `PIIFormatter()`, so email and IP replacement can
change the count of the released text. The old ordinary-FineWeb pilot measured
this effect, but its mismatch statistics do not transfer to FineWeb-Edu. The
new manifest will retain both counts and report signed and absolute drift; the
training cutoff always uses IDs recomputed from released `text`.

Documents are visited in lexicographic Parquet-path order and then physical row
order. The source has already been globally shuffled with seed 42. For this
small pilot only, a document enters validation exactly when the SHA-256 hash of
the released, unnormalized UTF-8 text falls in bucket zero:

```text
int.from_bytes(sha256(text.encode("utf-8")), "big") % 1000 == 0
```

The pilot validation artifact contains only validation documents encountered
before the training pilot reaches its target. It validates the split mechanism;
`1/1000` is not the final validation policy.

## Fixed inputs

The following are already fixed:

- FineWeb-Edu repository:
  `HuggingFaceFW/fineweb_edu_100BT-shuffled`, path `data`.
- FineWeb-Edu revision:
  `be6b2a50d3a9c60d330c45384e80c7863cd3a25d`.
- GPT-2 tokenizer repository: `openai-community/gpt2`.
- GPT-2 tokenizer revision:
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- Megatron-LM commit:
  `571370c829ca768fe37244f4e2e7f28d8accc4ab` (`core_v0.18.2`).
- GPT-2 vocabulary size 50,257 and EOD ID 50,256.

The revisions are constants in the script; neither input can silently follow
`main`. The FineWeb-Edu artifact has 102,063,987 rows in 100 Parquet shards and
was produced in 2026 by sampling FineWeb-Edu and globally shuffling the
documents. Its `100BT` name is an approximate source label, not the token
budget used by this project. We recompute GPT-2 IDs from `text`.

The script downloads the allowlisted tokenizer files, hashes only that
allowlist rather than unrelated files already present in the shared Hugging
Face cache, uses it through MCore's Hugging Face tokenizer wrapper, and
requires a `GPT2TokenizerFast` backend. It does not allow the generic
`GPT2BPETokenizer` path to download mutable files implicitly.

The pinned container must provide Python, PyTorch, NumPy, Transformers,
Hugging Face Hub, and PyArrow. Megatron-LM must be checked out at the commit
above and passed through `--megatron-root`. Exact package versions are recorded
in the output manifest and the node207 environment evidence.

Hugging Face credentials are read by `huggingface_hub` from its normal login or
`HF_TOKEN_PATH`/`HF_TOKEN`. A token must not be passed on the command line or
written into this repository.

## Pilot command

```bash
python scripts/prepare_fineweb_edu_pilot.py \
  --megatron-root <absolute-path-to-pinned-megatron-lm> \
  --output-dir <empty-durable-output-directory>
```

The script refuses a nonempty output directory. It creates:

- `train_text_document.bin` and `.idx`;
- `valid_text_document.bin` and `.idx`;
- deterministic `documents.jsonl`;
- `manifest.json` with source, tokenizer, environment, count, cursor, and
  artifact hashes.

The local pilot is discarded after validation. It is not uploaded, so it does
not duplicate documents in the eventual public 7.345B-token artifact.

## Pass criteria

The script fails unless:

- the dataset, tokenizer, and Megatron revisions match their requested commits;
- the tokenizer has exactly 50,257 entries and EOD ID 50,256;
- every selected document records both its recomputed token count and
  FineWeb-Edu's inherited `token_count`, and their aggregate signed/absolute
  drift agrees with the manifest;
- every document is nonempty, every FineWeb-Edu ID is unique in the scanned range,
  and train/validation membership follows the fixed hash rule;
- MCore selects `numpy.uint16`;
- `.idx` contains one sequence per document and the loaded sequences agree
  with their manifest hashes;
- each loaded sequence ends with the one appended EOD;
- each `.bin` size is exactly two bytes times its indexed-token count;
- all token and UTF-8 byte totals agree;
- the training payload can supply 4,883 non-repeated 2,048-token samples;
- at least one validation document is selected.

Run the pilot twice into two empty directories in the same pinned environment.
`documents.jsonl`, both `.bin` files, both `.idx` files, and the artifact
hashes inside `manifest.json` must be identical. Any difference blocks the
full preparation. Runtime is printed but intentionally omitted from the
manifest so it cannot break deterministic comparison.

After the corrected pilot, rerun the following loader gate:

```bash
python scripts/gpt_dataset_smoke.py
```

[`gpt_dataset_smoke.py`](../scripts/gpt_dataset_smoke.py) will load
`/workspace/data/fineweb-edu-pilot-run1/train_text_document` for exactly 4,883
requested samples. It constructs the pinned MCore `GPTDataset` twice with
independent temporary caches, requires one epoch, verifies that the document
and shuffle indices are permutations without repeats, reads the first and last
requested samples, and requires identical document/sample/shuffle array and
`.npy` hashes across both builds. It builds MCore's `helpers_cpp` through the
pinned checkout's standard `compile_helpers()` entry point.

## Production split recommendation

There is no production rule that validation must be 0.1% of either documents
or tokens. Validation size is chosen for metric variance and desired reporting
slices. [Paloma](https://arxiv.org/html/2312.10523) measured this tradeoff with
a 1.4B model and used at least 1M tokens per source; its sweep extended through
8M tokens. The project's 100M final holdout is therefore a deliberate
low-noise reporting and slicing budget, not a literature requirement.

The proposed final build has three disjoint, whole-document artifacts:

- 7.345B recomputed indexed tokens for training;
- 8M recomputed indexed tokens for periodic development validation;
- 100M recomputed indexed tokens for a frozen final holdout.

Split assignment happens before tokenization. Its key is the SHA-256 digest of
the exact released `text.encode("utf-8")`, with no normalization and no EOD
added. Token IDs are not hashed because changing the tokenizer or special-token
policy must not change membership. The FineWeb-Edu `id` is retained as
provenance but is not the key: identical text can occur under different IDs or
Common Crawl dumps. Fixed recomputed token targets, rounded up to the final
whole document, are authoritative; hash intervals are only a deterministic
way to allocate candidates while scanning. This follows the production pattern
in [Dolma v1.5](https://github.com/allenai/dolma/blob/main/configs/dolma-v1_5/eval-set.md),
whose [sampler hashes released document text bytes](https://github.com/allenai/dolma/blob/main/scripts/hash_sample.py)
before tokenization, while using SHA-256 here instead of MD5.

Exact duplicate documents therefore cannot cross splits. Before freezing the
full artifacts, apply a heldout-contamination audit modeled on
[Paloma's paragraph-level procedure](https://arxiv.org/html/2312.10523):
first remove and backfill final-holdout documents that overlap the development
set, then index qualifying paragraphs from both heldout sets, remove any
training document containing a matching paragraph, and backfill training to
7.345B tokens. Log the match/removal rate and inspect a sample. This is
heldout decontamination, not a new global deduplication recipe for FineWeb-Edu.
Also record heldout composition by Common Crawl `dump` and educational
`int_score`.

The final split algorithm and the 100M holdout remain subject to explicit user
approval before the long build. The 10M pilot uses one validation artifact and
the `1/1000` bucket only to test deterministic plumbing.

## Later extensions

A training artifact records a half-open training-ordinal range and the last
scanned source cursor. A later extension starts after that cursor, so its
source rows cannot overlap the base artifact. The extension must also check its
FineWeb-Edu IDs and text hashes against the base manifest before claiming
document-level non-overlap. The pilot records the required cursor and ordinal
metadata but does not validate an extension. It is rebuilt as part of the
single final base artifact rather than retained as a separate shard.

Do not pass a base prefix and an extension prefix together as an ordinary
Megatron data blend. `BlendedDataset` samples or interleaves its inputs; it
does not concatenate them. Changing that blend on resume changes sample order
and can duplicate or omit data.

An extended run must instead use one training prefix per declared phase. With
fixed global batch size, pinned MCore's `--phase-transition-iterations`
checkpoints and exits at a boundary and resets the phase-local consumed-sample
offset after restart. This path needs a two-prefix toy checkpoint/restart test
before use, and the launcher must select the next prefix explicitly.
