#!/usr/bin/env bash
# Archive the retained peak-LR checkpoint of the 1C arms to the private HF repo, and
# only once every file is verified against the server's own listing, optionally drop
# the local copy.  Written for nfs2: from iter_0014157 each mb=4 arm holds retained +
# rolling, 2 x 39.8 GB, which does not fit beside another student's 19 GB.
#
#   mlsub run ... --entry scripts/cloud_hf_offload_retained.sh --gpus cpu
#     --args "OWNER/REPO delete 300" --env HF_TOKEN=...
#
#   $1 repo      OWNER/REPO, or a bare name to be resolved against the token's namespace
#   $2 delete    the literal word `delete` removes the verified local copy; anything
#                else keeps it (upload only)
#   $3 minutes   how long to keep waiting for arms that have not written the checkpoint
#                yet, polling every 5 min.  0 = process whatever exists and exit.
#   $4.. arms    optional list of arm names to wait on, one word each. A
#                directory that will never produce the checkpoint would otherwise hold
#                the poll open to the deadline. It goes here rather than in an --env
#                because mlsub rejects an env value containing a space or a comma.
#
# Override the target with STAGE3_MOE_CKPT_ROOT / STAGE3_MOE_RETAIN_ITER /
# STAGE3_MOE_HF_PREFIX to serve the mb=16 wave on nfs3 instead, and STAGE3_MOE_ARMS
# (comma separated -- mlsub rejects an env value containing a space) to restrict
# which arms are waited on -- a directory that will
# never produce the checkpoint would otherwise hold the poll open to the deadline.
set -u

repo=${1:?usage: cloud_hf_offload_retained.sh OWNER/REPO [delete|keep] [wait_minutes]}
delete=${2:-keep}
wait_min=${3:-0}
arms_arg="${*:4}"

root=${STAGE3_MOE_CKPT_ROOT:-/workspace-SR006.nfs2/hmoe-checkpoints/stage3-1c-mb4/1c}
iter=${STAGE3_MOE_RETAIN_ITER:-13794}
prefix=${STAGE3_MOE_HF_PREFIX:-1c-mb4}

echo "HF_OFFLOAD repo=$repo delete=$delete wait=${wait_min}min root=$root iter=$iter"
df -h "$root" | tail -1

: "${HF_TOKEN:?set HF_TOKEN with mlsub run --env}"
unset PYTHONNOUSERSITE
python -c 'import huggingface_hub' 2>/dev/null || pip install --user -q huggingface_hub
# A resumable checkpoint is one ~14 GB file, so file-level parallelism buys nothing and
# the chunked Rust uploader is the only lever on the 3.3 MB/s the endpoint upload saw.
if pip install --user -q hf_transfer 2>/dev/null && python -c 'import hf_transfer' 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  echo "hf_transfer: enabled"
else
  echo "hf_transfer: unavailable, falling back to the python uploader"
fi

HMOE_REPO=$repo HMOE_DELETE=$delete HMOE_WAIT=$wait_min HMOE_ROOT=$root \
HMOE_ITER=$iter HMOE_PREFIX=$prefix HMOE_ARMS=$arms_arg python - <<'PY'
import os, pathlib, shutil, sys, time
from huggingface_hub import HfApi

repo = os.environ["HMOE_REPO"]
do_delete = os.environ["HMOE_DELETE"] == "delete"
deadline = time.time() + float(os.environ["HMOE_WAIT"]) * 60
root = pathlib.Path(os.environ["HMOE_ROOT"])
iteration = int(os.environ["HMOE_ITER"])
prefix = os.environ["HMOE_PREFIX"]
name = f"iter_{iteration:07d}"

api = HfApi(token=os.environ["HF_TOKEN"])
if "/" not in repo:
    # Resolve the namespace from the token rather than guessing it: a wrong owner is a
    # 403 halfway through a multi-hour upload.
    repo = f'{api.whoami()["name"]}/{repo}'
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
print(f"REPO_READY {repo} (private)", flush=True)

arms = os.environ["HMOE_ARMS"].replace(",", " ").split() or sorted(d.name for d in root.glob("*") if d.is_dir())
print(f"ARMS {' '.join(arms)}", flush=True)
done, failed = set(), set()

def tracker_of(arm):
    t = root / arm / "latest_checkpointed_iteration.txt"
    return t.read_text().strip() if t.is_file() else ""

def handle(arm):
    local = root / arm / name
    remote = f"{prefix}/{arm}/{name}"
    files = {str(p.relative_to(local)): p.stat().st_size for p in sorted(local.rglob("*")) if p.is_file()}
    total = sum(files.values())
    print(f"--- UPLOAD {local} -> {remote} ({len(files)} files, {total / 2**30:.2f} GiB)", flush=True)
    t0 = time.time()
    try:
        api.upload_folder(folder_path=str(local), path_in_repo=remote, repo_id=repo, repo_type="model")
    except Exception as exc:                       # noqa: BLE001 - report and carry on
        print(f"UPLOAD_FAILED {remote}: {type(exc).__name__}: {exc}", flush=True)
        return False
    dt = time.time() - t0
    print(f"UPLOADED {remote} in {dt / 60:.1f} min ({total / 2**20 / max(dt, 1):.1f} MB/s)", flush=True)

    # Verify from the server's own listing, not from the upload call's return value.
    remote_files = {}
    for entry in api.list_repo_tree(repo, path_in_repo=remote, repo_type="model", recursive=True):
        size = getattr(entry, "size", None)
        if size is not None:
            remote_files[entry.path[len(remote) + 1:]] = size
    missing = {k: v for k, v in files.items() if remote_files.get(k) != v}
    if missing:
        print(f"VERIFY_FAILED {remote}: {len(missing)} file(s) missing or wrong size, keeping local", flush=True)
        for k in list(missing)[:5]:
            print(f"    {k}: local={files[k]} remote={remote_files.get(k)}", flush=True)
        return False
    print(f"VERIFIED {remote}: {len(remote_files)} files match", flush=True)

    if do_delete:
        # The tracker still names this checkpoint until the next save lands, and a
        # resubmitted job would try to load exactly what the tracker names.
        live = tracker_of(arm)
        if live == str(iteration):
            print(f"DELETE_DEFERRED {arm}: tracker still at {live}, it is the resume point", flush=True)
            return False
        shutil.rmtree(local)
        print(f"REMOVED_LOCAL {local} (tracker at {live})", flush=True)
    return True

while True:
    for arm in arms:
        if arm in done:
            continue
        if not (root / arm / name).is_dir():
            continue
        if handle(arm):
            done.add(arm)
            failed.discard(arm)
        else:
            failed.add(arm)
    pending = [a for a in arms if a not in done]
    if not pending or time.time() >= deadline:
        break
    print(f"WAITING for {len(pending)}: {' '.join(pending)} "
          f"({(deadline - time.time()) / 60:.0f} min left)", flush=True)
    time.sleep(300)

print(f"DONE={len(done)} PENDING={len([a for a in arms if a not in done])} RETRY={len(failed)}")
for arm in arms:
    print(f"  {arm}: {'archived' if arm in done else 'not archived'} tracker={tracker_of(arm)}")
PY
echo "PY_EXIT=$?"
df -h "$root" | tail -1
echo "EXIT=0"
exit 0
