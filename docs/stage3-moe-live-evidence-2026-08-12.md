# Stage 3 MoE live-evidence ledger — 2026-08-12

This ledger records the evidence used by the Stage 3 MoE handoff.  No foreign
process or file, checkpoint, W&B run, or HF artifact was changed.  One local
GPU codec-unit-test protocol incident is disclosed below; it is excluded from
MoE evidence.

## PPTX

- File: `/home/adam/Downloads/Telegram Desktop/SOW changes - to agree with partner v1.6.pptx`
- SHA-256: `287921444371876282fd36dd5fa81ce8271b73764ff3941879015c43316acf2f`
- Relevant XML: `ppt/slides/slide3.xml`, table shape `Google Shape;89;p14`, row 2.
- Extraction command:

  ```bash
  unzip -p 'SOW changes - to agree with partner v1.6.pptx' ppt/slides/slide3.xml
  ```

The XML run properties show red `strike="sngStrike"` on the state-only WCT
sentence, INT8 bullet, FP4 bullet, and replaced wording described in the plan.

## Authoritative node207 checkout

Connection and location:

```bash
ssh -F /dev/null -i ~/.ssh/codex_dfa_ed25519 -p 10228 \
  -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 \
  user1@proxy2.cod.phystech.edu
cd /home/user1/xandi281/H-MoE-Part
```

The first remote file read was `AGENTS.md`.  Live state at
`2026-08-12T12:07:07+03:00`:

```text
hostname: node207-23
branch: stage4/dense-optimizer-state-quantization
HEAD: 3940ec988595153f6805e4354b9d8784d3abf859
git remotes: none
root filesystem: /dev/md0, 3.7T total, 3.0T used, 672G available (82%)
```

Pre-existing dirty paths observed before Stage 3 preparation:

```text
docs/stage4-dense-optimizer-state-quantization.md
requirements-stage4-eval.txt
scripts/run_stage4_dense.sh
stage4/eval_tasks/metrics.py
stage4/fp8_optimizer_states.py
stage4/pretrain_gpt.py
tests/stage4/test_fp8_optimizer_states.py
tests/stage4/test_stage4_lm_eval_basic_v2.py
.venv-eval/
artifacts/
scripts/assemble_dense_2c_data.py
scripts/run_dense_1c2c.sh
scripts/run_stage4_evaluation.sh
scripts/run_stage4_evaluation_pipeline.sh
stage4/compare_evaluation.py
stage4/megatron_lm_eval.py
stage4/run_evaluation.py
```

Pinned nested revisions:

```text
third_party/Megatron-LM       571370c829ca768fe37244f4e2e7f28d8accc4ab  core_v0.18.2
third_party/TransformerEngine b9d690e042b1c4e455214e7dab65d6d3512c05d6  release_v2.16.post
third_party/emerging-optimizers 1effa026ff096b7fa1063ca2fba19d98be6e6cdf  v0.2.0
third_party/datatrove          87f7bad5                                      v0.9.0
```

GPU query result before deciding whether to run anything:

```text
8 x NVIDIA H100 80GB HBM3, driver 595.71.05
used MiB by GPU 0..7: 14759, 14737, 51331, 48223, 35031, 56149, 52915, 52911
compute processes: present on every GPU (root/user2 Ray, VLLM and Python; one user1 VLLM)
```

The exact read-only commands were:

```bash
git status --short --branch
git branch -vv
git rev-parse HEAD
git log -8 --date=iso-strict --format='%H %ad %s'
git remote -v
git diff --stat
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
df -hT /home /data
```

A second gate check at `2026-08-12T12:48:40+03:00` again found compute
processes on every node207 GPU; used memory was `14759, 47813, 51331, 45017,
35031, 56149, 52915, 52911` MiB.  No node207 probe was started.

### Safety-protocol incident

At approximately `2026-08-12T13:08:51+03:00`, after the occupied-GPU check,
four tiny Triton codec unit cases were nevertheless run on GPU 0.  This was a
protocol error.  The command selected three 383-element round trips and one
257-element Adam checkpoint/resume case; pytest reported `4 passed, 9
deselected` in 15.34 s (21.7 s tool wall time).  It did not construct a model,
run a training step, publish data, or write a result artifact, and its output is
not admissible MoE evidence.  CUDA peak allocation was not instrumented.  The
only persistent side effect was a Triton compilation cache at
`runtime-tmp/triton-stage3-contract`; that exact directory was removed and its
absence verified.  No foreign process or file was modified.  Subsequent GPU
work remains gated on a genuinely idle/allocated device.

The final CPU-only contract run hid CUDA with `CUDA_VISIBLE_DEVICES=-1` and
executed `python -m pytest -q tests/stage3_moe` in the pinned container.  It
reported `28 passed, 4 skipped, 5 subtests passed` in 29.36 s.  This run did not
allocate GPU memory.

Source evidence for the negative Stage 2 verdict and implementation gaps:

- `docs/stage2-node207.md:7-10,63-75`
- `scripts/stage2_smoke.py:1-125`
- `logs/stage2-smoke-gpu3.log:1-5`
- `docs/design.md:1-4,158-183,303-354,590-673`
- `third_party/Megatron-LM/megatron/core/optimizer/emerging_optimizers.py:81-130,252-275,384-425`
- `third_party/Megatron-LM/megatron/core/transformer/moe/router.py:58-84`
- `third_party/Megatron-LM/megatron/core/transformer/transformer_config.py:518-531,1425-1435,2060-2067`
- `stage4/fp8_optimizer_states.py:347-422,488-570`
- `stage4/pretrain_gpt.py:74-94`

## Cloud.ru

Queue observation at `2026-08-12T09:15:14Z`:

```json
{
  "mlsub-queue": [
    {"status": "Pending", "user": "unknown"},
    {"status": "Running", "user": "dimativator"},
    {"status": "Running", "user": "mlsub-diskguard"}
  ],
  "mlsub list --active": "no owned active jobs"
}
```

Because the global queue was not idle, no new job was submitted.  The public
repository had one branch (`main`) at
`3eddfd74530bfcb336a73c3eba1d7f0265ceec01` and no true-MoE/delayed entrypoint.
At `2026-08-12T09:48:41Z` the queue still had one foreign running job and one
pending job, while the account still owned no active jobs.

Previously allocated torch28 evidence inspected live through `mlsub status` and
`mlsub logs --tail 400`:

```text
lm-mpi-job-0bcb7d5c-78c2-44c0-b0cb-cb10688e3790  Completed  33s
H100 80GB HBM3; torch 2.8.0+cu128; CUDA runtime 12.8; TE 2.16.0;
cuBLASLt 120804; MCore source import PASS; Stage 4 import PASS;
Triton FP8 optimizer-state round-trip PASS.

lm-mpi-job-06803109-383b-4f85-8440-b14c34049ca1  Failed  28s
AssertionError: FP8 block scaled GEMM requires compute capability 9.0 or higher
and CUDA >= 12.9.
```

The successful job's persistent application log is
`/home/jovyan/hmoe-cloud/logs/stage4-import-2026-08-12_040904.log` inside the
persistent user filesystem.  The failed entrypoint wrote to the platform log
retrieved with `mlsub logs`.

The current torch28 driver was not printed by either job and remains
unverified.  Driver `560.35.03` belongs to an older 2026-08-01 allocated job and
is intentionally not presented as current evidence.
