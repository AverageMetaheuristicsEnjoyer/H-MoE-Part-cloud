#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
source "$root/configs/dense-1p028b.sh"

[[ ${MLSUB_IMAGE:-} == torch28 ]] || { echo "requires --image torch28" >&2; exit 2; }

tools_root=${STAGE3_MOE_DATA_TOOLS_ROOT:-/workspace-SR006.nfs2/hmoe-data/tools}
megatron="$tools_root/Megatron-LM"
expected_commit=571370c829ca768fe37244f4e2e7f28d8accc4ab
[[ -d $megatron/.git ]] || { echo "missing data-tools Megatron checkout: $megatron" >&2; exit 2; }
[[ $(git -C "$megatron" rev-parse HEAD) == "$expected_commit" ]] || {
  echo "unexpected Megatron commit in $megatron" >&2
  exit 2
}

base_data=${MONARCH_BASE_DATA:-/home/jovyan/data/fineweb-edu-gpt2-megatron/data}
old_extension=${MONARCH_OLD_EXTENSION:-/workspace-SR006.nfs2/hmoe-data/fineweb-edu-time-match-extension/data/train}
new_extension=${MONARCH_DENSE_EXTENSION:-/workspace-SR006.nfs2/hmoe-data/fineweb-edu-dense-1c-extension/data/train}
train_data=("$base_data/train" "$old_extension" "$new_extension")
for prefix in "${train_data[@]}" "$base_data/development" "$base_data/final"; do
  [[ -s ${prefix}.bin && -s ${prefix}.idx ]] || { echo "missing indexed dataset: $prefix" >&2; exit 2; }
done
[[ -s ${new_extension%/data/train}/artifact-manifest.json ]] || {
  echo "missing dense extension manifest" >&2
  exit 2
}

cache=${MONARCH_DATA_CACHE_PATH:-/workspace-SR006.nfs2/monarch-pretrain/data-cache/dense-1c}
log_root=${MONARCH_LOG_ROOT:-/home/jovyan/hmoe-cloud/monarch-pretrain}
log="$log_root/dense-1c-cache-$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$cache" "$log_root"

unset PYTHONNOUSERSITE
nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
  -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: - || true)
export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH="$megatron:$root"
make -s -C "$megatron/megatron/core/datasets"

echo "DENSE_CACHE target_samples=10039120 cache=$cache log=$log"
set +e
python "$megatron/tools/prepare_cache.py" \
  "${DENSE_1P028B_MODEL_ARGS[@]}" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --micro-batch-size 4 \
  --global-batch-size 208 \
  --train-iters 48265 \
  --eval-interval 250 \
  --eval-iters 32 \
  --tokenizer-type NullTokenizer \
  --vocab-size 50257 \
  --null-tokenizer-eod-id 50256 \
  --null-tokenizer-pad-id -1 \
  --train-data-path "${train_data[@]}" \
  --valid-data-path "$base_data/development" \
  --test-data-path "$base_data/final" \
  --data-cache-path "$cache" \
  --dataloader-type single \
  --num-workers 0 \
  --no-create-attention-mask-in-dataloader \
  --seed 1234 \
  --prepare-cache-world-size 1 \
  2>&1 | tee "$log"
cache_exit=${PIPESTATUS[0]}
set -e
echo "CACHE_EXIT=$cache_exit"
(( cache_exit == 0 )) || exit "$cache_exit"
grep -Eq 'train dataset length:[[:space:]]+10039120$' "$log" || {
  echo "dense cache did not prove the full 1C sample count" >&2
  exit 1
}
grep -q '> finished preparing dataset cache' "$log"
du -sh "$cache"
echo "EXIT=0"
