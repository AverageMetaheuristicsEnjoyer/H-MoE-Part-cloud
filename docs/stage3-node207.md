# Stage 3 node207 data-build and audit record

Date: 2026-07-29 to 2026-07-30  
Host: `node207-23`  
Project: `/home/user1/xandi281/H-MoE-Part`

The bounded DataTrove pilot, full selected-prefix conversion, public-directory
staging, exhaustive exact cross-split audit, and full MinHash measurement
passed. No data was removed or mutated. The sanitized artifact was published
to Hugging Face and independently downloaded and verified. No model run was
started.

## Environment

The verified training `.venv` was left unchanged. Data preparation uses
`.venv-data` inside the existing pinned SIF because released DataTrove 0.9
requires `huggingface-hub<1`, while the training environment uses Hub 1.10 and
Transformers 5.3.

DataTrove is a clean checkout at
`87f7bad5c4a56ec648265fbf0b91d7d226bad428` (`v0.9.0`). The resolved
dependencies are pinned in [`requirements-data.txt`](../requirements-data.txt);
the observed freeze is retained remotely as `logs/data-venv-pip-freeze.txt`
(SHA-256
`22a1916736afdec204f597ded22971b0c3214aeb631372fecb7980058c30aa4b`).
Both data and training venvs pass `pip check`.

The data venv occupies 272MB and the DataTrove checkout 37MB. Durable storage
had 400GB free before and after the pilot.

## Source assignment

[`fineweb_edu_source_plan.json`](../configs/fineweb_edu_source_plan.json)
reserves source files 0–97 as ordered training candidates, file 98 for
development, and file 99 for final evaluation. The current schema-v2 plan's
SHA-256 is
`4e872852ad7918b3d673f3ce8ab317ba46bfc31e973def6799933128522df5e9`.
The earlier pilot log records the superseded schema-v1 plan hash
`ca57c29...`; that evidence remains historical rather than current.

The live source-plan gate resolved all 100 paths at dataset revision
`be6b2a50d3a9c60d330c45384e80c7863cd3a25d` and proved the three path sets
disjoint. Evidence: `logs/fineweb-edu-source-plan.log`.

## Writer parity

The fixed ASCII, newline, and Unicode fixture was written through DataTrove
twice with tokenization batch sizes 2 and 10,000, and independently through
pinned MCore. Both formats were byte-identical:

| Artifact | SHA-256 |
|---|---|
| fixture `.bin` | `ca21640a5d83efd310434a33d349ad86f1181a150232c286d89cdab848918046` |
| fixture `.idx` | `68e80ac8a01e5b369a417e190947cccf90240f03162e0aeb6fb870d291854b83` |

MCore loaded one `uint16` sequence per document and verified each exact token
array and final EOD 50,256. A second fixture split across two DataTrove prefixes
was merged twice with MCore `IndexedDatasetBuilder.add_index()`; both merges
were byte-identical to the one-prefix reference. Evidence:
`logs/datatrove-megatron-parity.log`.

## FineWeb-Edu pilot

Both pilot runs read the first 15,000 physical rows from
`data/train-00000-of-00100.parquet` through the pinned Hugging Face filesystem.
The source file's repository LFS identity is:

```text
size:    2,703,170,007 bytes
sha256:  1f390aa780ce0d3fcd8e3826977158aa2c4a3cc62b262612a8404db366f5a843
```

Each run produced 15,801,191 indexed IDs in 15,000 documents. The capacity is
7,715 complete 2,048-token MCore samples, above the requested 4,883-sample
pilot gate. DataTrove reported 5.24s and 5.53s pipeline runtime; these are
functional data-preparation timings, not training throughput measurements.

The two deterministic outputs matched byte for byte:

| Artifact | Size | SHA-256 |
|---|---:|---|
| `.bin` | 31,602,382 bytes | `9ccb74b37b768729a6684bbb3fa81da3a02af9b76d2341747a7ac2055c05c96b` |
| `.idx` | 300,042 bytes | `4c8686fca5cf8ffd0782fc4f9c12e7aad92b55ebcec2146dc5c4bc4f7679e06f` |
| `manifest.json` | 2,328 bytes | `10dddd4e95974e425e32d45723ebb20b25cac266bf0df4fcc5fe761bec84b05c` |

Evidence is retained in:

- `logs/fineweb-edu-datatrove-pilot-run1.log`;
- `logs/fineweb-edu-datatrove-pilot-run2.log`;
- `logs/fineweb-edu-datatrove-pilot-repeat.log`.

Pinned MCore then built `GPTDataset` twice with independent temporary caches.
Both used one epoch, all 15,000 documents exactly once in the document index,
and identical index arrays:

| Index | Shape | Array SHA-256 |
|---|---:|---|
| document | `[15000]` | `79882ba9035edee93542feb2d150309efb2c46824804ca3282b9842a3ea8c1bf` |
| sample | `[7716, 2]` | `74e5fdc05317b1f8cd2eef0b33c95357be235902c0b15fe431dadd7db0985fec` |
| shuffle | `[7715]` | `51f02c1e978f4a318c95fac04c3c0ccc6c4a8b4067a9cd9641bd980c9063d307` |

Evidence: `logs/gpt-dataset-datatrove-pilot.log`.

## Pilot GPU occupancy and cleanup

These gates were CPU data checks and make no tokens/s, MFU, GPU-memory, or
training-speed claim. At preflight only GPU 3 was empty. Before the final
occupancy check, a foreign process from
`/home/user1/xandi281/TT/venv/bin/python` (PID 263191) allocated 63,542MiB on
GPU 3. It was not touched. Therefore there is currently no clean GPU or NVLink
pair for a measured run.

The temporary 100-document API smoke and the duplicate second 31MB pilot
directory were removed after their hashes were recorded. The retained first
pilot is 31MB. Both can be reproduced from the pinned public source.

## Full selected-prefix artifact

The full build selected training source files 0 through 7, development source
file 98, and final source file 99. The staged artifact is
`data/fineweb-edu-public`; the original per-source outputs remain under
`data/fineweb-edu-full`.

| Split | Documents | Available indexed IDs | UTF-8 source bytes |
|---|---:|---:|---:|
| training | 7,290,286 | 7,556,553,510 | 35,173,581,642 |
| development | 7,599 | 8,000,266 | 37,310,247 |
| final | 96,830 | 100,001,217 | 465,855,889 |

Planned training consumption remains 7,344,816,128 IDs. The additional
211,737,382 available training IDs are whole-source-file overshoot and do not
change the training budget.

The staged Megatron data hashes are:

| Artifact | SHA-256 |
|---|---|
| `data/train.bin` | `94c1cd2266a162a615d27b6b11ac1c1cc5887d629742eefad8efff1976b46222` |
| `data/train.idx` | `564815271149b0322bd262b9f681e3feb30a96d7c677162ad5ef9f2ee3458614` |
| `data/development.bin` | `b806f2f1ba34e8d86c3f6517f9aa305fa554d850c44362790bf6b84b88f47d55` |
| `data/development.idx` | `aac74ffcf4d090a3bc96c320227a25805f3b0666f4595e36ac6f98a0c828d713` |
| `data/final.bin` | `9fb1f3d21deabe8b9cc85675b21cebb7434f1221ea8cc8ebb825069729e97a49` |
| `data/final.idx` | `efca13527d24b44423fc521947ce8c630b6739e15f5d8139ad84d9efc56be85d` |

Build evidence includes `logs/fineweb-edu-full-train-build.log`, the per-source
manifests under `data/fineweb-edu-full`, and the copied provenance under
`data/fineweb-edu-public/provenance`.

## Exhaustive exact cross-split audit

The measurement-only exact audit covered all selected training documents and
the complete development and final prefixes. Every SHA-1-64 candidate was
confirmed with SHA-256 and raw UTF-8 equality; there were zero 64-bit collision
candidates.

| Held-out split | Confirmed documents | Document rate | Confirmed indexed IDs | ID rate | Confirmed UTF-8 bytes | Byte rate |
|---|---:|---:|---:|---:|---:|---:|
| development | 375 | 4.9349% | 362,367 | 4.5294% | 1,681,021 | 4.5055% |
| final | 4,909 | 5.0697% | 4,743,920 | 4.7439% | 22,077,649 | 4.7392% |

Report:
`data/fineweb-edu-audits/exact/exact-audit.json`
(SHA-256
`b6b6b8440f111def2908da098a834e6b1242d091f642c30d96eecf514e0acdce`).
The original compact copy was staged at
`data/fineweb-edu-public/audits/exact-cross-split.json`. Before publication,
its free-text scope field was removed because it referred to a private
consumption plan. No configuration value, measurement, or result changed.
The sanitized public copy has SHA-256
`f365f10c91bde84d8f687b544df66fcd31d0fc6969625db54bad6b89f53d7690`.

## Full MinHash measurement

The full candidate-only MinHash run used English spaCy tokenization, 5-grams,
14 buckets, 8 hashes per bucket, SHA-1 at 64-bit precision, seed 1, and four
CPU workers. `CUDA_VISIBLE_DEVICES=0` only satisfied the existing Singularity
wrapper; this was not GPU work.

Coverage was complete:

| Scope | Documents | Indexed IDs | UTF-8 source bytes |
|---|---:|---:|---:|
| all | 7,394,715 | 7,664,554,993 | 35,676,747,778 |
| training | 7,290,286 | 7,556,553,510 | 35,173,581,642 |
| development | 7,599 | 8,000,266 | 37,310,247 |
| final | 96,830 | 100,001,217 | 465,855,889 |

All 7,394,715 documents received a 5-gram signature. The clustering stage
reported 297,949 total candidate clusters containing 616,374 documents,
including within-split clusters. Of those, 8,439 clusters crossed from
training into development or final and contained 19,640 documents:
11,169 training, 598 development, and 7,873 final. Two candidate clusters
touched both held-out splits. The maximum candidate cluster size was 545.
These are MinHash candidates, not confirmed near duplicates.

| Held-out split | Candidate documents | Document rate | Candidate indexed IDs | ID rate | Candidate UTF-8 bytes | Byte rate |
|---|---:|---:|---:|---:|---:|---:|
| development | 598 | 7.8695% | 639,308 | 7.9911% | 2,961,162 | 7.9366% |
| final | 7,873 | 8.1307% | 8,491,487 | 8.4914% | 39,284,333 | 8.4327% |

Every confirmed exact-match held-out document is contained in the MinHash
candidate set. Beyond those confirmed exact matches, MinHash added 223
development and 2,964 final candidate documents. Those additional candidates
were not separately confirmed.

The run completed with exit status 0 in 15,782.531 seconds (4:23:02.531).
The outer durable process measured 4:23:06 wall time. The one-minute scratch
ledger observed a peak of 8,622,130,466 bytes (8.03 GiB), equal to the retained
final audit-tree size. The output had not been cleaned when this was recorded.

The independently rechecked stable tree contains 197 files. Every file size
and SHA-256 matched `artifact-hashes.json`; the recomputed tree SHA-256 is
`d550804eb3869d5c99cf3466383264088cf0a6b9ac92e7377012f1056a49d92b`.
The report SHA-256 is
`eec5a04a626ad7fd4cfbb947224cad61f9e88d0b436742bb98a65a24411dc767`.

Evidence:

- `data/fineweb-edu-public/audits/minhash-full/benchmark-report.json`;
- `data/fineweb-edu-public/audits/minhash-full/artifact-hashes.json`;
- `data/fineweb-edu-public/audits/minhash-full/cross-split-candidate-clusters.jsonl`;
- `logs/fineweb-edu-minhash-full.log`;
- `logs/fineweb-edu-minhash-full.pid`;
- `logs/fineweb-edu-minhash-full-scratch.tsv`.

No filtering step ran. DataTrove's computed removal statistic was not applied,
and no train, development, or final artifact was changed.

## Publication decision

The user chose publication of the measured artifact unchanged with prominent
overlap disclosure. No removal, filtering, or backfill policy was applied.
The artifact card does not describe the held-out splits as contamination-free
and labels MinHash results as candidate clusters rather than confirmed near
duplicates.

The pre-upload artifact card is
`data/fineweb-edu-public/README.md` (SHA-256
`b71b671ee3ac8b0f027a71b8ed89c394f4cd59287cc317057e3dfa449cf28b60`).
The machine-readable
`data/fineweb-edu-public/artifact-manifest.json` has SHA-256
`7a7cf91633081dd080f3cff96407bd61d4c237d88a5e926cffe36fc5b8cbfcc8`.
Its 34 payload-file records were independently checked against every file
size and SHA-256. The deterministic payload tree SHA-256 is
`fe0dc4130d96371f39dadada4356a8f745ac6a1e1ff09ae79d11fa664d24d297`;
the manifest itself is excluded from that self-referential tree.

The public metadata was scanned before this final hash pass. It contains no
credentials, API keys, passwords, usernames, hostnames, IP addresses,
absolute local paths, SSH details, private log paths, or project-design
references. The repository-intention field and private consumption, target,
minimum, overshoot, and previous-count fields were removed. Actual split
counts, selected source shards, revision pins, data hashes, and audit results
were retained. The six `.bin` and `.idx` hashes did not change.

Pinned MCore pre-upload verification passed all three independent prefixes:

| Split | Documents | Indexed IDs |
|---|---:|---:|
| training | 7,290,286 | 7,556,553,510 |
| development | 7,599 | 8,000,266 |
| final | 96,830 | 100,001,217 |

Evidence:
`logs/verify-fineweb-edu-public-sanitized-preupload.log`
(SHA-256
`77dd3ee7b0fcc73c6391d99944264d637176bb2b266118e75f74d23e9590609c`).

The user approved the sanitized card, manifest, and exact 35-file upload set.
Earlier failed or skipped pilot evidence remains in `logs/`; no failure record
was overwritten or deleted.

## Hugging Face publication and independent verification

The public dataset repository is:

`https://huggingface.co/datasets/AverageMetaheuristicsEnjoyer/fineweb-edu-gpt2-megatron`

The immutable verified commit is
`a2b8660e317ea2449c6ef57d9710a6e66c914751`. Anonymous API access returned
`private=false` and the same commit. The repository contains the approved 35
files plus Hub-generated `.gitattributes` (SHA-256
`8cda381e7a5f360e24b960632058f8bcac44f1af5c5ac84c15ba242e15bf9a5a`).
There were no other remote files.

The resilient upload completed successfully with no parameter change.
Evidence:

- `logs/fineweb-edu-hf-upload.log` (SHA-256
  `e0e37a06a7450eeedf5664a71fcca6935632cc61c292ff6a6df31e7fd535cfa7`);
- `logs/fineweb-edu-hf-upload.pid`.

The immutable commit was then downloaded anonymously into the independent
directory `data/fineweb-edu-hf-verify-a2b8660e317ea244`, using a fresh
download/Xet cache. All 36 remote files were fetched. Every one of the 34
manifest payload records matched its size and SHA-256, the downloaded manifest
and card matched their approved hashes, the payload tree matched
`fe0dc4130d96371f39dadada4356a8f745ac6a1e1ff09ae79d11fa664d24d297`,
and none of the payload files was a symlink.

Evidence:

- `logs/fineweb-edu-hf-download-verify.log` (SHA-256
  `977a131ddf6ecbb2d261bbc4af5a5135498e152beb86edcc044506ce555baf8e`);
- `logs/fineweb-edu-hf-downloaded-hash-verify.log` (SHA-256
  `a42cd227959d4955ee8acd4fc311adc2b2ca9a595cc4f1721ce5a11c10ffc310`);
- `logs/verify-fineweb-edu-hf-downloaded-mcore.log` (SHA-256
  `77dd3ee7b0fcc73c6391d99944264d637176bb2b266118e75f74d23e9590609c`);
- `logs/fineweb-edu-hf-download-verify.pid`.

Pinned MCore passed the independently downloaded train, development, and final
prefixes with the same document counts, indexed-ID counts, and binary/index
hashes as the staged artifact.

After this verification, 39,582,675,098 bytes of confirmed project-owned
duplicates and scratch were removed:

- the independent verification download;
- `data/fineweb-edu-full`;
- the retained full MinHash scratch tree;
- upload/download resume caches and temporary publication helper scripts.

The canonical `data/fineweb-edu-public` staging tree, compact audit evidence,
all logs, containers, environments, and the potentially shared `data/hf-cache`
were retained. Free disk increased from 362GB to 383GB.
