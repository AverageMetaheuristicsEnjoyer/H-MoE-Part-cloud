# Stage 1: GPU server preflight

## node207

Date: 2026-07-28  
Host: `node207-23`  
Server target: `/home/user1/xandi281/H-MoE-Part`

This was a read-only preflight. It made no server-side changes, pulled no
images, stopped no processes, and started no runs.

## node207 outcome

Stage 1 was executed but has not passed its clean-GPU criterion. The pinned
container is compatible with the host, but every GPU had a foreign compute
process at the occupancy snapshot. The server project directory also does not
exist yet.

## Shared version manifest

| Component | Resolved value |
|---|---|
| Container tag | `nvcr.io/nvidia/pytorch:26.04-py3` |
| Registry index digest | `sha256:192d749b4d773610ec9e01c0443a9df545d196c412b7b8fd33bfa3da362a49e7` |
| Linux/amd64 manifest digest | `sha256:be06a21bd95a46bce1a5cfc0576051a40209f328440edaa2ba5cd35abf85ca1a` |
| Compressed image layers | 78 layers, 10,377,952,971 bytes |
| Megatron-LM/Core | `core_v0.18.2`, commit `571370c829ca768fe37244f4e2e7f28d8accc4ab` |
| TransformerEngine | commit `b9d690e042b1c4e455214e7dab65d6d3512c05d6` |
| Emerging-Optimizers | `v0.2.0`, commit `1effa026ff096b7fa1063ca2fba19d98be6e6cdf` |
| COAT reference | commit `80ec99f47aaa09231b07ace1fd04c30a1e30ec18` |

The image tag and both registry digests resolved successfully, but the image
is not currently present on node207.

## node207 host and accelerator inventory

| Check | Observed result | Status |
|---|---|---|
| OS | Ubuntu 24.04, Linux 6.17.0-40-generic | Pass |
| CPU | AMD EPYC 9554, 224 logical CPUs | Pass |
| RAM | 1.5 TiB total, 1.3 TiB available at the snapshot | Pass |
| GPUs | Seven NVIDIA H100 SXM5 80GB GPUs; 81,559 MiB each, 700 W limit | Hardware pass |
| MIG | Disabled on every GPU | Pass |
| Driver | NVIDIA 595.71.05 | Pass |
| Container runtimes | Docker 29.1.3 and Singularity CE 4.5.0 | Pass |
| NVIDIA Container Toolkit | 1.19.1; Docker exposes the NVIDIA runtime | Pass |

The pinned image contains CUDA 13.2.1. NVIDIA specifies driver 595.58.03 or
newer for CUDA 13.2 Update 1, so the installed 595.71.05 driver is compatible.
The relevant sources are the
[NGC 26.04 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-04.html)
and [CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).

## node207 GPU topology and occupancy

Direct NVLink pairs:

- GPU 0-1: `NV18`, NUMA node 0.
- GPU 3-4: `NV17`, NUMA node 1.
- GPU 5-6: `NV18`, NUMA node 1.

GPU 2 has no direct NVLink peer among the visible devices. No four-GPU set is
fully NVLink-connected. GPU 3-6 is the natural four-GPU candidate because both
NVLink pairs remain on NUMA node 1, but communication between the two pairs
still crosses PCIe host bridges and must be measured.

At the snapshot, every GPU had at least one foreign compute process. GPU memory
use ranged from 3,637 to 44,685 MiB. GPUs 1, 2, 3, and 6 were actively
computing. These observations are not performance measurements; they only show
that no clean two-GPU allocation was available. No process was stopped.

Before any GPU smoke test, rerun the occupancy check and require both GPUs of
one direct-NVLink pair to have no foreign processes or unexplained allocations.

## node207 storage and data

| Check | Observed result |
|---|---|
| Durable filesystem | 3.7 TB XFS, 403 GB free, 90% used |
| `/tmp` | 508 GB free tmpfs; not durable storage |
| Requested source | [FineWeb `sample-100BT`](https://huggingface.co/api/datasets/HuggingFaceFW/fineweb/tree/main/sample%2F100BT?recursive=true&expand=true): 50 Parquet files, 107,378,018,791 bytes |
| Existing `xandi281/fineweb-local` | One 2.1 GB FineWeb-Edu shard; wrong dataset |

The durable disk is sufficient for the compressed image, the raw FineWeb
source, one indexed 7.345B-token subset, and bounded checkpoint retention. It
does not have a safe margin for retaining the raw source, a fully tokenized
100B-token copy, and many full-state checkpoints simultaneously. Stage 3 must
set an explicit source/index/checkpoint retention budget before downloading
the dataset.

## node207 paths, version control, and network

- `/home/user1/xandi281/H-MoE-Part` does not exist. No similarly named checkout
  was found under `/home/user1/xandi281`.
- The local `/home/adam/Programming/H-MoE-Part/.git` directory is empty and
  `git rev-parse` fails. There is currently no project commit or remote origin
  to lock or sync.
- Registry manifest resolution succeeded. GitHub and Hugging Face returned
  HTTP 200. The unauthenticated `nvcr.io/v2/` response was the expected HTTP
  401 registry challenge; Docker then resolved the public manifest.

## Remaining node207 gates

1. Decide whether this is a new Git repository or provide the intended remote
   origin.
2. Create or sync the project only to
   `/home/user1/xandi281/H-MoE-Part`.
3. Recheck occupancy when a direct-NVLink GPU pair is expected to be free.
4. Before Stage 3, approve a disk-retention policy for raw data, indexed data,
   and checkpoints.

## mipt-h200-server

Date: 2026-07-28  
Host: `ubuntu-gpu-pod-vv-h200`  
User home: `/data/users/xandi281`  
Server target: `/data/users/xandi281/H-MoE-Part`

The configured OpenVPN session was already connected, so no second session was
started. This was a read-only preflight. It made no server-side changes,
pulled no images, stopped no processes, and started no runs.

### H200 outcome

The H200 system has the better accelerator topology: all eight H200s are
mutually connected by NVLink. It is not ready for Stage 2, however, because
all GPUs are occupied, durable storage has only 63.6 GB free, and the Docker
environment available to the user has neither the NVIDIA runtime nor a CDI GPU
specification.

Its driver is eligible for CUDA 13.x minor-version compatibility but is older
than the driver paired with CUDA 13.2 Update 1, so the pinned stack also
requires an actual container smoke before it can be accepted.

### H200 host and accelerator inventory

| Check | Observed result | Status |
|---|---|---|
| OS | Ubuntu 24.04, Linux 5.15.0-160-generic | Pass |
| CPU | Intel Xeon Platinum 8568Y+, 192 logical CPUs | Pass |
| RAM | 2.0 TiB total, 1.8 TiB available at the snapshot | Pass |
| GPUs | Eight NVIDIA H200 GPUs; 143,771 MiB each, 700 W limit | Hardware pass |
| MIG | Disabled on every GPU | Pass |
| Driver | NVIDIA 580.82.07 | Conditional for CUDA 13.2.1 |
| Docker | 29.6.2; storage root `/data/docker` | Daemon accessible; GPU injection unavailable |
| NVIDIA container integration | No `nvidia-container-cli`, `nvidia-ctk`, NVIDIA Docker runtime, or CDI spec found | **Fail** |

NVIDIA documents two relevant thresholds. CUDA 13.x minor-version
compatibility starts at driver 580, which this host satisfies. The driver
paired with CUDA 13.2 Update 1 is 595.58.03, which this host does not satisfy.
The pinned image may therefore work through minor-version compatibility, but
features that require a newer driver or PTX JIT are not guaranteed. Do not
change the host driver or accept this environment until the actual
TransformerEngine/CUDA smoke passes. See the
[CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
and [CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).

### H200 GPU topology and occupancy

Every GPU pair is reported as `NV18`; this is the full NVSwitch-connected
topology we would prefer for four-GPU expert-parallel measurements. GPUs 0-3
are NUMA-local to node 0 and GPUs 4-7 to node 1, but GPU-to-GPU links remain
NVLink across the full set.

At the snapshot, every H200 had at least one foreign compute process. Memory
use ranged from 56,859 to 138,131 MiB. GPUs 0-4 and 7 reported 100%
utilization, GPU 6 reported 77%, and GPU 5 reported 0% while still holding
multiple allocations. No GPU qualifies as clean, and no process was stopped.

Before any GPU smoke test, all selected GPUs must have no foreign processes or
unexplained allocations. Occupancy must be checked again during the measured
interval.

### H200 storage and data

| Check | Observed result |
|---|---|
| Durable `/data` | 13.9 TB total, 13.1 TB used, 63.6 GB free |
| User home | 330 GB under `/data/users/xandi281` |
| Container storage | `/data/docker`, on the same nearly full filesystem |
| Docker accounting | 170.6 GB images, 20.77 GB stopped containers, 48.31 GB reported reclaimable image data |
| Root overlay | 115 GB free; not approved as durable project storage |
| Shared memory | 995 GB free tmpfs; not durable project storage |

The 63.6 GB durable free space cannot hold the 107.378 GB FineWeb source, the
pinned image, indexed data, and checkpoints. Docker reports shared reclaimable
objects, but they are not this project's data and must not be deleted without
separate ownership confirmation and approval. The H200 server needs additional
durable storage or an administrator-approved cleanup before data preparation
or image pull.

### H200 paths, access, and network

- `/data/users/xandi281/H-MoE-Part` does not exist, and no matching checkout
  was found under the user home.
- `nvcr.io`, GitHub, and Hugging Face were reachable; the pinned NGC index and
  Linux/amd64 manifest digests matched the shared version manifest above.
- GPU devices and NVSwitch devices are visible in the outer environment, but
  that alone does not provide GPU injection into containers launched by this
  Docker daemon.
- The first `ssh mipt-h200-server` attempt failed locally before connecting
  because OpenSSH rejected
  `/etc/ssh/ssh_config.d/20-systemd-ssh-proxy.conf` for unsafe ownership or
  permissions. Using the exact host, port, user, and pinned key from
  `AGENTS.md` with `ssh -F /dev/null` succeeded. The VPN and key were healthy;
  no SSH configuration was modified.

### Remaining H200 gates

1. Obtain enough durable storage for the image, FineWeb subset preparation,
   and bounded checkpoints.
2. Have the administrator provide NVIDIA Container Toolkit integration, or
   explicitly choose and pin a non-container environment instead.
3. Recheck occupancy when two or four H200s are expected to be free.
4. Pull and smoke-test the immutable image only after the storage/runtime gates
   and a separate write approval.
5. Create or sync the project only to
   `/data/users/xandi281/H-MoE-Part`.

## Current server recommendation

Use node207 for the first Stage 2 environment smoke once a clean NVLink pair is
available. Its driver directly matches CUDA 13.2 Update 1 requirements, its
NVIDIA Docker runtime is configured, and its current disk margin can support
the bounded initial workflow. Keep the H200 server as the preferred later
four-GPU performance target because of its full NVLink topology and larger
memory, but do not begin setup there until its durable-storage and container
integration gates are resolved.
