# Stage 2 node207 execution record

Date: 2026-07-29  
Host: `node207-23`  
Project: `/home/user1/xandi281/H-MoE-Part`

Stage 2 passed its bounded environment, GPU, communication, indexed-data, and
10M-token pilot gates. This was functional validation, not a throughput
benchmark. No full 7.345B-token build, Hugging Face upload, model training, or
performance claim was made.

## Runtime

Docker could pull the pinned NGC image but could not start even a minimal new
container; each attempt remained in `Created`. The three existing foreign
Postgres, MinIO, and Qdrant containers were left untouched. The working runtime
is therefore Singularity CE 4.5.0 plus a project-local venv:

- NGC source: `nvcr.io/nvidia/pytorch:26.04-py3`
- registry index:
  `sha256:192d749b4d773610ec9e01c0443a9df545d196c412b7b8fd33bfa3da362a49e7`
- Linux/amd64 manifest:
  `sha256:be06a21bd95a46bce1a5cfc0576051a40209f328440edaa2ba5cd35abf85ca1a`
- SIF:
  `containers/pytorch-26.04-be06a21b.sif`
- SIF SHA-256:
  `4c9b4c486068552056b07dfbc6c8d36131cb44ddc151e9a04c5e7ce158be8b5e`

`scripts/node207_env.sh` uses Singularity's clean environment, disables the
problematic automatic home and `/tmp` mounts, binds the project at `/workspace`,
and mounts the Hugging Face credential read-only from outside the project. It
explicitly removes inherited token and `PYTHONOPTIMIZE` variables before
setting `HF_TOKEN_PATH`. Authentication was verified as
`AverageMetaheuristicsEnjoyer`; the credential is absent from the repository,
logs, and command lines.

Main resolved versions:

| Component | Exact value |
|---|---|
| Python | 3.12.3 |
| PyTorch | `2.12.0a0+0291f960b6.nv26.04.48445190` |
| CUDA reported by PyTorch | 13.2 |
| Megatron-Core | `0.18.2+571370c`, commit `571370c829ca768fe37244f4e2e7f28d8accc4ab` |
| TransformerEngine | `2.16.0+b9d690e`, commit `b9d690e042b1c4e455214e7dab65d6d3512c05d6` |
| Emerging-Optimizers | 0.2.0, commit `1effa026ff096b7fa1063ca2fba19d98be6e6cdf` |
| NVIDIA Resiliency Extension | 0.6.0 |
| Transformers / tokenizers | 5.3.0 / 0.22.2 |
| Hugging Face Hub / PyArrow / NumPy | 1.10.1 / 23.0.1 / 2.1.0 |

TransformerEngine was built for SM90 only. `pip check` reports no broken
requirements. The complete observed package set is retained remotely in
`logs/pip-freeze.txt`, and hashes of the SIF, scripts, and final pilot manifest
are in `logs/artifact-hashes.log`. Every assertion-based smoke rejects Python
optimized mode before running its checks.

## GPU and communication gates

GPUs 3 and 4 were clean immediately before the smokes and form a direct NVLink
pair (`NV17`). They were again at 0 MiB and 0% utilization after the tests.
Foreign workloads on GPUs 1, 2, and 6 were not touched.

| Gate | Result | Evidence |
|---|---|---|
| Hopper identity | H100 80GB, SM90, cuBLASLt 13.4 | `logs/stage2-smoke-gpu3.log` |
| FP8 compute | TE E4M3 block-scaling `Linear` forward/backward finite, FP32 scales enabled | `logs/stage2-smoke-gpu3.log` |
| Expert GEMM API | Two-group TE `GroupedLinear` forward/backward finite | `logs/stage2-smoke-gpu3.log` |
| Muon reference | One EO step finite with FP32 parameter and FP32 momentum | `logs/stage2-smoke-gpu3.log` |
| Two-GPU NCCL | Exact all-reduce result on GPUs 3 and 4 | `logs/nccl-smoke-gpu3-4.log` |

These checks establish API and numerical functionality only. They do not
measure MoE dispatch/combine, tokens/s, MFU, peak VRAM, or relative speed.

## FineWeb pilot

The fixed inputs were FineWeb `sample/100BT` at revision
`9bb295ddab0e05d785b879661af7260fed5140fc`, GPT-2 tokenizer revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, and the pinned MCore checkout.
The tokenizer has 50,257 entries, uses EOD 50,256, and produces a `uint16`
Megatron payload.

The first run correctly exposed an invalid gate in the preparation script:
FineWeb's stored `token_count` is computed before its PII formatter, whereas
the released text has already been rewritten. Row 10 therefore stored 184 but
tokenized to 180. The corrected cutoff uses token IDs recomputed from the
released text and keeps FineWeb's count as advisory metadata. In the final
14,785-document scan, 515 counts differed; stored minus recomputed was +1,705
tokens and absolute drift was 2,067 tokens. The failed run is retained in
`logs/fineweb-pilot-failed-prepii-count-gate.log`.

A later manifest audit found that recursively hashing the shared tokenizer
snapshot also captured unused files left by earlier cache activity. Hashing was
restricted to the exact tokenizer allowlist, and both pilot builds plus the
MCore loader gate were rerun. Token payload hashes were unchanged. The final
manifest hash is
`71480ec708b817aed8a5ccc062cf02d7085ed81cc2b6c2c9cfc2b0ad8c385616`.

Final two-build results:

| Quantity | Result |
|---|---:|
| Requested training samples | 4,883 x 2,048 |
| Training indexed IDs | 10,000,897 |
| Training documents | 14,777 |
| Validation indexed IDs / documents | 3,985 / 8 |
| Scanned source documents | 14,785 |
| Source Parquet cursor | `sample/100BT/000_00000.parquet`, row 14,784 |
| Runtime | 28.370 s and 27.934 s |

All five data artifacts and the corrected manifest were byte-identical across
the two builds; the comparisons and both sets of hashes are retained in
`logs/fineweb-pilot-repeat-compare.log`. Key artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `documents.jsonl` | `7e694c58d8e0d56b027f090baa15935ff373da31677e054914c040f0ab6c2b3f` |
| train `.bin` | `799fc41f429cd0e23f1bb30fe4725a61af42f7bb1043ae02fc8d8ac20e2a3311` |
| train `.idx` | `44ee6c3815487a29e08999850bf0a41994ba79008a660b1e8dc5c932497b764c` |
| validation `.bin` | `3748dafaa6f01b6a570355476e9a54878c3d81fa20fcf0a7a2e804d6f1d3574b` |
| validation `.idx` | `778f578170660d08ec845dd436a602e27d2abdef9fb7ad048fba58bd198933b4` |

The pinned `GPTDataset` then produced exactly 4,883 available samples in one
epoch. Its two independent caches had identical index hashes:

| Index | Shape | Array SHA-256 |
|---|---:|---|
| document | `[14777]` | `ae86cb738410b92ae4c31e5c85d071604b487fc73d55c348d75d7fbae5d53e24` |
| sample | `[4884, 2]` | `ad0d95d958cdb0270aaa5ce1bf9238a5905cb18335c907c8a6caa02f14db3999` |
| shuffle | `[4883]` | `fe045730194eba64a3bc2fb9e06b6231bf528498c85180e660dbe399822f6425` |

The first loader attempt failed because MCore's `helpers_cpp` had not yet been
built. The gate now invokes the pinned checkout's standard
`compile_helpers()` entry point; the failure and passing rerun are retained in
`logs/gpt-dataset-smoke-failed-missing-helper.log` and
`logs/gpt-dataset-smoke.log`.

## Cleanup and remaining gates

After validation, both 27MB pilot directories, independent dataset caches,
the 11GB Singularity conversion cache, the 380MB TransformerEngine build tree,
six failed Docker containers, and the Docker-only copies of the NGC and Alpine
images were removed. The verified SIF, venv, source checkouts, compiled MCore
helper, logs, and 2.1GB cached source Parquet shard remain. Reported free disk
increased from 392GB to 400GB. No foreign container, process, image, or GPU
allocation was removed.

Before the full data build:

1. Approve the 7.345B-token preparation and public upload as a separate,
   longer operation.
2. Choose the public Hugging Face dataset repository name.
3. Decide whether this local directory is a new Git repository or provide its
   intended remote origin; it is still not a usable Git worktree.
4. Freeze the approximately 100M-token validation reserve to an exact
   whole-document cutoff and approve the artifact card before upload.
