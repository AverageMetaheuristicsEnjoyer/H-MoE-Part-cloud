#!/usr/bin/env bash
set -u

source_root=/home/jovyan/hmoe-cloud/pretrain
destination_root=/workspace-SR006.nfs3/hmoe-cloud/pretrain
mkdir -p "$destination_root"

for source in "$source_root"/*-eval-downstream-broad-v2-1c; do
  [[ -d $source ]] || continue
  name=${source##*/}
  temporary="$destination_root/.$name.partial"
  destination="$destination_root/$name"
  [[ ! -e $temporary && ! -e $destination ]] || { echo "destination exists: $name"; exit 1; }
  cp -a "$source" "$temporary" || exit 1
  diff -qr "$source" "$temporary" || exit 1
  mv "$temporary" "$destination" || exit 1
  rm -rf "$source"
  echo "MOVED $name"
done

df -h /home/jovyan /workspace-SR006.nfs3
echo EXIT=0
