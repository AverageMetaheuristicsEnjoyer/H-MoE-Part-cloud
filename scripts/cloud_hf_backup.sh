#!/usr/bin/env bash
# Archive finished Stage 3 checkpoints to a private Hugging Face repo and, only once every
# file is verified present at the right size, reclaim the local space.  Usage:
#   mlsub run ... --entry scripts/cloud_hf_backup.sh --gpus cpu
#     --args "OWNER/REPO endpoints delete" --env HF_TOKEN=...
#
# Sets:
#   endpoints  the ten weights-only 1.2B deliverables (six mb=4 + four mb=16), ~2 GB each
#   branch     the six resumable trunk branch points at iteration 2254, 7-16 GB each
#   unpin      no upload: drop the duplicate iter_0002254 *hardlink entries* that the decay
#              seeding left under 1p2b/ and stage3-mb16/.  They are extra directory names
#              for blocks that stage3/trunk already holds, so removing them frees nothing
#              by itself -- it is what lets the 1C runs actually reclaim the branch point
#              when they roll past it.
#
# The third argument must be the literal word `delete` for anything to be removed;
# anything else (or nothing) uploads and keeps.
set -u

repo=${1:?usage: cloud_hf_backup.sh OWNER/REPO endpoints|branch|unpin [delete]}
set_name=${2:?usage: cloud_hf_backup.sh OWNER/REPO endpoints|branch|unpin [delete]}
delete=${3:-keep}

nfs3=/workspace-SR006.nfs3/hmoe-checkpoints
echo "HF_BACKUP repo=$repo set=$set_name delete=$delete"
df -h "$nfs3" | tail -1

if [[ $set_name == unpin ]]; then
  freed=0
  for d in "$nfs3"/stage3/1p2b/*/iter_0002254 "$nfs3"/stage3-mb16/trunk/*/iter_0002254 \
           "$nfs3"/stage3-mb16/1p2b/*/iter_0002254; do
    [[ -d $d ]] || continue
    if [[ $delete == delete ]]; then
      echo "UNPIN $d"; rm -rf "$d"; freed=$((freed + 1))
    else
      echo "WOULD UNPIN $d"
    fi
  done
  echo "UNPINNED=$freed"
  df -h "$nfs3" | tail -1
  echo "EXIT=0"; exit 0
fi

: "${HF_TOKEN:?set HF_TOKEN with mlsub run --env}"
unset PYTHONNOUSERSITE
python -c 'import huggingface_hub' 2>/dev/null || pip install --user -q huggingface_hub

HMOE_REPO=$repo HMOE_SET=$set_name HMOE_DELETE=$delete HMOE_NFS3=$nfs3 python - <<'PY'
import os, pathlib, sys
from huggingface_hub import HfApi

repo = os.environ["HMOE_REPO"]
which = os.environ["HMOE_SET"]
do_delete = os.environ["HMOE_DELETE"] == "delete"
nfs3 = pathlib.Path(os.environ["HMOE_NFS3"])

# (local directory, path inside the repo). Only directories that exist are touched.
jobs = []
if which == "endpoints":
    for arm_dir in sorted((nfs3 / "stage3" / "1p2b").glob("*")):
        jobs.append((arm_dir / "iter_0002818", f"1p2b-mb4/{arm_dir.name}/iter_0002818"))
    for arm_dir in sorted((nfs3 / "stage3-mb16" / "1p2b").glob("*")):
        jobs.append((arm_dir / "iter_0002818", f"1p2b-mb16/{arm_dir.name}/iter_0002818"))
elif which == "branch":
    for arm_dir in sorted((nfs3 / "stage3" / "trunk").glob("*")):
        jobs.append((arm_dir / "iter_0002254", f"trunk/{arm_dir.name}/iter_0002254"))
else:
    sys.exit(f"unknown set: {which}")

jobs = [(l, r) for l, r in jobs if l.is_dir()]
print(f"CANDIDATES={len(jobs)}", flush=True)
if not jobs:
    print("nothing to do"); sys.exit(0)

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
print(f"REPO_READY {repo} (private)", flush=True)

uploaded, verified, removed = 0, 0, 0
for local, remote in jobs:
    local_files = {
        str(p.relative_to(local)): p.stat().st_size for p in sorted(local.rglob("*")) if p.is_file()
    }
    total = sum(local_files.values())
    print(f"--- UPLOAD {local} -> {remote} ({len(local_files)} files, {total / 2**30:.2f} GiB)", flush=True)
    try:
        api.upload_folder(folder_path=str(local), path_in_repo=remote, repo_id=repo, repo_type="model")
        uploaded += 1
    except Exception as exc:                      # noqa: BLE001 - report and carry on
        print(f"UPLOAD_FAILED {remote}: {type(exc).__name__}: {exc}", flush=True)
        continue

    # Verify from the server's own listing, not from the upload call's return value.
    remote_files = {}
    for entry in api.list_repo_tree(repo, path_in_repo=remote, repo_type="model", recursive=True):
        size = getattr(entry, "size", None)
        if size is not None:
            remote_files[entry.path[len(remote) + 1:]] = size
    missing = {k: v for k, v in local_files.items() if remote_files.get(k) != v}
    if missing:
        print(f"VERIFY_FAILED {remote}: {len(missing)} file(s) missing or wrong size, keeping local copy", flush=True)
        for k in list(missing)[:5]:
            print(f"    {k}: local={local_files[k]} remote={remote_files.get(k)}")
        continue
    verified += 1
    print(f"VERIFIED {remote}: {len(remote_files)} files match", flush=True)

    if do_delete:
        import shutil
        shutil.rmtree(local)
        removed += 1
        print(f"REMOVED_LOCAL {local}", flush=True)

print(f"UPLOADED={uploaded} VERIFIED={verified} REMOVED={removed}")
PY
echo "PY_EXIT=$?"
df -h "$nfs3" | tail -1
echo "EXIT=0"
exit 0
