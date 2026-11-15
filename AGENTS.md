# H-MoE-Part-cloud — orientation

This is the **public mirror** the Cloud.ru platform clones. Code reaches a GPU job only
through it: `mlsub` does `git clone --depth 1 --branch <branch>` and runs one entry script.
Local files and the gateway's home directory never travel to the cloud.

**Work happens on the `stage3/moe-short-probes` branch, not `main`.**

## Where things stand

Read [`docs/stage3-1c-results.md`](docs/stage3-1c-results.md) first — it is the results
ledger: final losses for all six 1C arms, the memory gate, the wave-1 downstream tables, where
every artifact lives, and the open work. Training is finished; do not relaunch it.

Supporting documents:

- [`docs/stage3-moe-experiment-plan.md`](docs/stage3-moe-experiment-plan.md) — the design and
  the gates.
- [`docs/design.md`](docs/design.md) — the schedule and model configuration.
- [`docs/stage3-data-build.md`](docs/stage3-data-build.md) — the tokenized corpus and its
  split policy.
- [`docs/stage3-moe-live-evidence-2026-08-12.md`](docs/stage3-moe-live-evidence-2026-08-12.md)
  — the evidence ledger for the handoff.

## Running anything

```bash
ssh brain_lab                      # gateway, key auth, no sudo, no VPN
mlsub run --repo https://github.com/AverageMetaheuristicsEnjoyer/H-MoE-Part-cloud \
  --branch stage3/moe-short-probes --entry scripts/<script>.sh --gpus 1 --image torch28 \
  --note <short-note> --env KEY=VALUE --args "..."
mlsub list | status <job> | logs <job> | kill <job>
```

- **`--image torch28` is mandatory** for GPU work; the default image kills the launcher with
  exit 1 and an empty log.
- `mlsub status` returns lowercase `"status"`; `mlsub list` uses capitalised words.
- **`mlsub logs` hangs on a running job.** To see live progress, submit a CPU job running
  `scripts/cloud_verify_pretrain.sh` — it reads the persisted train log, which the platform
  API cannot.
- An environment value **may not contain a space**; use another separator.
- Every entry script exits 0 on purpose so the platform keeps the logs. The real status is in
  `TRAIN_EXIT` / `ARM_EXIT` / `EXIT` lines.
- `/home/jovyan` intermittently refuses new directories while `df` still shows free space.
  Pass `STAGE3_MOE_LOG_ROOT=/workspace-SR006.nfs2/hmoe-cloud/pretrain` when that happens.

## The entry scripts that matter

| script | what it does |
|---|---|
| `scripts/run_stage3_moe_pretrain.sh ARM MODE` | the launcher: `full`, `trunk`, `decay-1p2b`, `smoke`, `bench`, `resume-bench`, `eval-downstream` |
| `scripts/cloud_moe_full.sh` | one 1C arm, idempotent; resubmitting the same job is the resume path |
| `scripts/cloud_moe_eval.sh ARM...` | downstream scoring; both arms of a pair belong in one job |
| `scripts/cloud_finalize.sh FILTER` | paired intervals + pair verdicts, CPU only |
| `scripts/cloud_verify_pretrain.sh SUBSTR` | read-only: what a run left behind, including a live one |
| `scripts/cloud_disk_audit.sh` | the volumes and what fills them |
| `scripts/cloud_hf_offload_retained.sh` | move a retained checkpoint to the HF archive |

## Comparison machinery

`stage3_moe/pair_results.py` turns two `results.jsonl` records into a verdict. It is strict on
purpose: the two records must agree on normalized MCore argv (only the arm id inside path
options is masked), denominators, GPU identity, host, image, revisions and config hash.

The practical consequence: **training runs can never pair** — each arm ran in its own job on
its own card. Verdicts come from *evaluation* records instead, where both arms are scored in
one job from a checkpoint staged at a fixed path. Quote the memory and WCT gates from the
training records and the downstream effect from the evaluation records; never present one
`pair_results` verdict line as if it covered both.
