#!/usr/bin/env bash
# Run an arm to the full 1C budget (17,242 steps = 7,344,816,128 tokens) as a
# continuation of its trunk, and make the job idempotent so the platform can kill it and
# we just resubmit.  Usage:
#   mlsub run ... --entry scripts/cloud_moe_full.sh --args "ARM [ARM ...]" --gpus 1
#     --env STAGE3_MOE_CKPT_ROOT=/workspace-SR006.nfs2/hmoe-checkpoints/stage3-1c-mb4
#     --env STAGE3_MOE_MICRO_BATCH=4 --env STAGE3_MOE_RUN_SUFFIX=mb4
#     --env WANDB_API_KEY=... --env WANDB_BASE_URL=https://wandb-radfan.ru
# A nonstandard branch-point directory can be supplied for a single arm with
# STAGE3_MOE_BRANCH_CHECKPOINT_DIR. On the same filesystem it is moved atomically into
# the normal 1C destination so checkpoint rotation can reclaim it after the next save.
#
# Seeding: the destination is filled once from the trunk branch point and then owns
# itself.  Hardlinks are used when source and destination sit on the same volume (free)
# and a real copy otherwise (7-16 GB per arm, cross-volume).  A destination that already
# has a tracker is left alone -- that is the resume path, and it is why resubmitting this
# same job continues instead of restarting.
#
# One checkpoint directory belongs to exactly one live job.  Two jobs writing the same
# --save would race on the tracker and delete each other's checkpoints.
# Always exits 0 so the platform keeps the logs; the real status is ARM_EXIT.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
src_root=${STAGE3_MOE_SRC_CKPT_ROOT:-/workspace-SR006.nfs3/hmoe-checkpoints/stage3}
dst_root=${STAGE3_MOE_CKPT_ROOT:?set STAGE3_MOE_CKPT_ROOT to the checkpoint root for this wave}
full_dir_name=${STAGE3_MOE_FULL_DIR:-1c}
branch=2254
branch_dir=$(printf 'iter_%07d' "$branch")
direct_source=${STAGE3_MOE_BRANCH_CHECKPOINT_DIR:-}
retain_interval=13794

if [[ -n $direct_source && $# != 1 ]]; then
  echo "DIRECT_SOURCE_FAIL expected exactly one arm, got $#"
  echo "EXIT=1"
  exit 0
fi

echo "FULL_WAVE root=$dst_root dir=$full_dir_name micro_batch=${STAGE3_MOE_MICRO_BATCH:-4} suffix=${STAGE3_MOE_RUN_SUFFIX:-none}"
df -h "$src_root" "$dst_root" 2>/dev/null | grep -v ^Filesystem

for arm in "$@"; do
  if [[ ! $arm =~ ^[a-z0-9_]+$ ]]; then
    echo "SKIP invalid arm name: $arm"
    continue
  fi
  dst="$dst_root/$full_dir_name/$arm"
  tracker="$dst/latest_checkpointed_iteration.txt"

  if [[ -f $tracker ]]; then
    resume_iteration=$(cat "$tracker")
    if [[ ! $resume_iteration =~ ^[0-9]+$ ]]; then
      echo "SKIP $arm: invalid resume tracker at $tracker"
      continue
    fi
    resume_dir=$(printf '%s/iter_%07d' "$dst" "$resume_iteration")
    if [[ ! -d $resume_dir ]]; then
      echo "SKIP $arm: invalid resume tracker at $tracker"
      continue
    fi
    echo "=== ARM $arm RESUME from $resume_iteration in $dst ==="
  else
    if [[ -n $direct_source ]]; then
      source_tracker="$direct_source/latest_checkpointed_iteration.txt"
      src="$direct_source/$branch_dir"
      if [[ ! -f $source_tracker || $(cat "$source_tracker") != "$branch" ]]; then
        echo "SKIP $arm: direct source tracker is not $branch at $source_tracker"
        continue
      fi
    else
      src="$src_root/trunk/$arm/$branch_dir"
    fi
    if [[ ! -d $src ]]; then
      echo "SKIP $arm: no branch point at $src"
      continue
    fi
    mkdir -p "$(dirname "$dst")"
    if [[ -n $direct_source && $(stat -c %d "$direct_source") == $(stat -c %d "$(dirname "$dst")") ]]; then
      echo "=== ARM $arm SEED move $direct_source -> $dst ==="
      mv -- "$direct_source" "$dst" || { echo "SKIP $arm: atomic seed move failed"; continue; }
    else
      mkdir -p "$dst"
      if [[ $(stat -c %d "$src") == $(stat -c %d "$dst") ]]; then
        echo "=== ARM $arm SEED hardlink $src -> $dst ==="
        cp -al "$src" "$dst/" || { echo "SKIP $arm: hardlink seeding failed"; continue; }
      else
        echo "=== ARM $arm SEED copy $src -> $dst (cross-volume, this takes minutes) ==="
        cp -a "$src" "$dst/.$branch_dir.partial" || { echo "SKIP $arm: copy failed"; continue; }
        mv "$dst/.$branch_dir.partial" "$dst/$branch_dir" || { echo "SKIP $arm: rename failed"; continue; }
      fi
      # Written last: the tracker is what marks the seed complete, so a job killed during
      # the copy leaves no half-seeded directory that a resume would trust.
      echo "$branch" > "$tracker"
    fi
    du -sh "$dst" 2>/dev/null
  fi

  resume_iteration=$(cat "$tracker" 2>/dev/null)
  if [[ ! $resume_iteration =~ ^[0-9]+$ ]]; then
    echo "SKIP $arm: destination has an invalid tracker"
    continue
  fi
  resume_dir=$(printf '%s/iter_%07d' "$dst" "$resume_iteration")
  if [[ ! -d $resume_dir ]]; then
    echo "SKIP $arm: destination is not a complete checkpoint"
    continue
  fi
  checkpoint_kb=$(du -sk "$resume_dir" | awk '{print $1}')
  existing_checkpoints=$(find "$dst" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' | wc -l)
  additional_checkpoints=1
  if (( retain_interval <= 17242 && existing_checkpoints == 1 && resume_iteration <= retain_interval )); then
    additional_checkpoints=2
  fi
  available_kb=$(df -Pk "$dst_root" | awk 'NR == 2 {print $4}')
  required_kb=$((additional_checkpoints * checkpoint_kb + 2 * 1024 * 1024))
  if (( available_kb < required_kb )); then
    echo "SKIP $arm: insufficient rolling-checkpoint space available_kb=$available_kb required_kb=$required_kb checkpoint_kb=$checkpoint_kb"
    continue
  fi
  echo "FULL_PREFLIGHT_PASS arm=$arm checkpoint=$resume_iteration retain_interval=$retain_interval available_kb=$available_kb required_kb=$required_kb"
  if [[ ${STAGE3_MOE_PREFLIGHT_ONLY:-0} == 1 ]]; then
    continue
  fi

  "$root/scripts/run_stage3_moe_pretrain.sh" "$arm" full
  echo "ARM_EXIT=$? arm=$arm"
  ls -d "$dst"/iter_* 2>/dev/null | sort | tail -3
  df -h "$dst_root" | tail -1
done

echo "EXIT=0"
exit 0
