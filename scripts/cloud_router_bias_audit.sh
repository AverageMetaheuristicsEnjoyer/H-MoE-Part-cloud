#!/usr/bin/env bash
# Read-only inspection of router expert-bias tensors in Stage 3 checkpoints.
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"

arm=adamw_fp8gemm_state_fp32
paths=(
  "source-original:/workspace-SR006.nfs3/hmoe-checkpoints/stage3/trunk/$arm/iter_0013794"
  "source-resume:/workspace-SR006.nfs3/hmoe-checkpoints/stage3-time-match-source/$arm/iter_0013794"
  "original-17242:/workspace-SR006.nfs3/hmoe-checkpoints/stage3/trunk/$arm/iter_0017242"
  "time-match-19570-nfs2:/workspace-SR006.nfs2/hmoe-checkpoints/stage3-time-match/time-match/$arm/iter_0019570"
  "time-match-19570-home:/home/jovyan/hmoe-checkpoints/stage3-time-match/time-match/$arm/iter_0019570"
  "extension-control-17242:/workspace-SR006.nfs3/hmoe-checkpoints/stage3-extension-decay-control/extension-decay-control/$arm/iter_0017242"
  "stretched-19570:/workspace-SR006.nfs3/hmoe-checkpoints/stage3-time-match-stretched-v1/time-match-stretched/$arm/iter_0019570"
)

files=()
labels=()
for item in "${paths[@]}"; do
  label=${item%%:*}
  path=${item#*:}
  file=$(find "$path" -maxdepth 2 -type f -name 'model_optim_rng.pt' -print -quit 2>/dev/null)
  if [[ -n $file ]]; then
    labels+=("$label")
    files+=("$file")
    stat -c "CHECKPOINT_FILE label=$label size=%s inode=%i path=%n" "$file"
  else
    echo "CHECKPOINT_MISSING label=$label path=$path"
  fi
done

python - "${labels[*]}" "${files[@]}" <<'PY'
import hashlib
import sys

import torch

labels = sys.argv[1].split()
files = sys.argv[2:]
for label, path in zip(labels, files):
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    model = checkpoint["model"]
    keys = sorted(k for k in model if k.endswith("expert_bias"))
    tensors = [model[k].detach().float().reshape(-1) for k in keys]
    flat = torch.cat(tensors) if tensors else torch.empty(0)
    digest = hashlib.sha256(flat.numpy().tobytes()).hexdigest() if tensors else "none"
    args = checkpoint.get("args")
    consumed = getattr(args, "consumed_train_samples", None)
    print(
        f"BIAS_SUMMARY label={label} iteration={checkpoint.get('iteration')} "
        f"consumed_train_samples={consumed} keys={len(keys)} values={flat.numel()} "
        f"nonzero={torch.count_nonzero(flat).item()} min={flat.min().item():.9g} "
        f"max={flat.max().item():.9g} mean={flat.mean().item():.9g} "
        f"std={flat.std(unbiased=False).item():.9g} sha256={digest} "
        f"optimizer={'optimizer' in checkpoint} "
        f"scheduler={'opt_param_scheduler' in checkpoint or 'lr_scheduler' in checkpoint} "
        f"rng={'rng_state' in checkpoint}"
    )
    for key, tensor in zip(keys, tensors):
        key_digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()[:16]
        print(
            f"BIAS_LAYER label={label} key={key} min={tensor.min().item():.9g} "
            f"max={tensor.max().item():.9g} std={tensor.std(unbiased=False).item():.9g} "
            f"sha256={key_digest}"
        )
    del checkpoint, model, tensors, flat
PY

for root in /workspace-SR006.nfs2/hmoe-cloud/pretrain /workspace-SR006.nfs3/hmoe-cloud/pretrain; do
  [[ -d $root ]] || continue
  find "$root" -type f -name 'train-*.log' \( -path '*time-match*' -o -path '*extension-decay-control*' \) -print0 2>/dev/null |
    xargs -0 grep -HiaE 'load_return:|missing key|unexpected key|successfully loaded checkpoint|loading checkpoint.*iteration 13794' 2>/dev/null |
    tail -80
done

echo EXIT=0
