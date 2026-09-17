#!/usr/bin/env bash
# Is an expert's Adam state harder to quantize than a dense layer's?
#
#   mlsub run --entry scripts/cloud_expert_state_codec_error.sh --gpus 1 --image te4
#
# FP8 optimizer state costs AdamW +0.70 % of validation loss in this MoE against +0.035 %
# in the dense code base, and switching exp_avg_sq from E5M2 to E4M3 -- the dense setting --
# recovers only a fraction of it. The standing hypothesis for the rest is MoE-specific and
# is the one docs/stage3-moe-experiment-plan.md section 7 already names: routed experts see
# ~13.85 % of tokens each (top-8 of 64), so an expert's second moment is estimated from far
# fewer samples than a dense layer's, making it noisier and harder to represent.
#
# This measures that directly and without training: load a checkpoint whose Adam state is
# FP32, push every state tensor through the project's own codec and back, and compare the
# round-trip error of expert tensors against dense ones. No training step, no optimizer, a
# few minutes.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
stage_root=${STAGE3_MOE_PROBE_STAGE:-/tmp/hmoe-expert-codec}
base_arm=${STAGE3_MOE_PROBE_BASE_ARM:-adamw_bf16_state_fp32}
base_iter=${STAGE3_MOE_PROBE_BASE_ITER:-13794}
hf_repo=${STAGE3_MOE_PROBE_HF_REPO:-AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints}
hf_prefix=${STAGE3_MOE_PROBE_HF_PREFIX:-1c-mb4}
export HF_HOME=${STAGE3_MOE_HF_HOME:-/tmp/hmoe-expert-codec/hf}
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"

unset PYTHONNOUSERSITE
nvidia_lib_path=$(find /home/user/conda/lib/python*/site-packages/nvidia \
  -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: - || true)
[[ -n $nvidia_lib_path ]] && export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

echo "IMAGE=${MLSUB_IMAGE:-unset} ARM=$base_arm ITER=$base_iter"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /tmp | tail -1

iter_dir=$(printf '%s/%s/iter_%07d' "$stage_root" "$base_arm" "$base_iter")
if [[ ! -f "$iter_dir/mp_rank_00/model_optim_rng.pt" ]]; then
  echo "=== staging $hf_prefix/$base_arm/iter_$base_iter ==="
  mkdir -p "$stage_root/$base_arm"
  pip install --user --no-cache-dir hf_transfer 2>&1 | tail -1
  python -c 'import hf_transfer' >/dev/null 2>&1 && export HF_HUB_ENABLE_HF_TRANSFER=1
  python - "$hf_repo" "$hf_prefix/$base_arm" "$stage_root/$base_arm" "$(printf 'iter_%07d' "$base_iter")" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo, remote, dest, name = sys.argv[1:5]
snapshot_download(repo_id=repo, allow_patterns=[f"{remote}/{name}/**"], local_dir=dest, max_workers=8)
staged, final = os.path.join(dest, remote, name), os.path.join(dest, name)
if os.path.isdir(staged) and not os.path.isdir(final):
    os.renames(staged, final)
print("STAGED", final)
PY
fi

python - "$iter_dir/mp_rank_00/model_optim_rng.pt" <<'PY'
import sys, torch, collections
from stage3_moe.optimizer_states import (
    StateSpec, init_fp8_state, quantize_fp8_state_, dequantize_fp8_state,
)

path = sys.argv[1]
blob = torch.load(path, map_location="cpu", weights_only=False)
print("TOP_KEYS", sorted(k for k in blob if not k.startswith("_")))

# MCore nests the torch-format optimizer state one or two levels down depending on the
# optimizer wrapper; find the first dict that looks like one rather than assuming a shape.
def find_state(node, depth=0):
    if depth > 4 or not isinstance(node, dict):
        return None
    if "state" in node and "param_groups" in node:
        return node
    for value in node.values():
        found = find_state(value, depth + 1)
        if found is not None:
            return found
    return None

opt = find_state(blob.get("optimizer", blob))
if opt is None:
    print("NO_OPTIMIZER_STATE_FOUND"); raise SystemExit(0)
state = opt["state"]
print("STATE_ENTRIES", len(state))

# The model is hidden=1024, dense ffn=2816, expert ffn=256, swiglu, 64 experts over 17 MoE
# layers plus one dense layer. So an expert's two matrices are the only ones with these
# element counts, which classifies without needing the name->index mapping.
EXPERT_NUMEL = {2 * 256 * 1024, 1024 * 256}
DENSE_FFN_NUMEL = {2 * 2816 * 1024, 1024 * 2816}

def classify(numel, shape):
    if numel in EXPERT_NUMEL:
        return "expert"
    if numel in DENSE_FFN_NUMEL:
        return "dense_ffn"
    # Everything else -- attention, embeddings, router, norms -- is reported by its own
    # shape instead of being lumped together or dropped. Attention is the honest
    # comparison group for an expert; a first pass compared 2210 expert tensors against
    # the two dense-FFN ones, which is no comparison at all.
    return "other" + str(tuple(shape))

def roundtrip(value, dtype, recipe="dre", signed=True):
    spec = StateSpec("probe", signed, dtype, recipe)
    holder = {}
    value = value.cuda().float().contiguous()
    init_fp8_state(holder, spec, value)
    quantize_fp8_state_(holder, spec, value)
    back = dequantize_fp8_state(holder, spec)
    # In float64, and scaled first. exp_avg_sq holds squared gradients, so its elements
    # are small enough that a float32 dot product underflows: the first pass reported
    # cosine 0.071 alongside a relative error of 0.005, which cannot both be true.
    reference = value.double()
    restored = back.double()
    denom = reference.norm()
    if denom == 0:
        return float("nan"), float("nan")
    reference = reference / denom
    restored = restored / denom
    rel = (restored - reference).norm().item()
    cos = torch.dot(restored.view(-1), reference.view(-1)).item() / max(
        restored.norm().item(), 1e-300
    )
    return rel, cos

counts = collections.Counter()
acc = collections.defaultdict(list)
for entry in state.values():
    if not isinstance(entry, dict):
        continue
    for name, signed in (("exp_avg", True), ("exp_avg_sq", False)):
        value = entry.get(name)
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            continue
        kind = classify(value.numel(), value.shape)
        counts[(kind, name)] += 1
        for label, dtype in (("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)):
            rel, cos = roundtrip(value, dtype, signed=signed)
            acc[(kind, name, label)].append((rel, cos))

print("TENSOR_COUNTS", dict(counts))
print()
print(f"{'class':10s} {'state':11s} {'fmt':5s} {'n':>5s} {'rel_L2 mean':>12s} {'median':>10s} {'p95':>10s} {'cos mean':>10s}")
keep = {k for k in {c for c, _, _ in acc} if sum(len(acc[j]) for j in acc if j[0] == k) >= 4}
for key in sorted(k for k in acc if k[0] in keep):
    kind, name, label = key
    rows = acc[key]
    rel = sorted(r for r, _ in rows)
    cos = [c for _, c in rows]
    mean = sum(rel) / len(rel)
    print(f"{kind:10s} {name:11s} {label:5s} {len(rel):5d} {mean:12.6f} "
          f"{rel[len(rel)//2]:10.6f} {rel[int(0.95*(len(rel)-1))]:10.6f} {sum(cos)/len(cos):10.6f}")
print("EXPERT_CODEC_AUDIT=DONE")
PY
echo "PY_EXIT=$?"
exit 0
