#!/usr/bin/env bash
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
  -mindepth 2 -maxdepth 2 -type d -name lib -print 2>/dev/null | paste -sd: - || true)
export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDNN_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cudnn
export CURAND_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/curand
export NVRTC_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cuda_nvrtc

python - <<'PY'
import torch

from megatron.core.optimizer.emerging_optimizers import _EMERGING_OPTIMIZERS
from stage3_moe.muon import (
    MuonFP8ShadowDiagnostic,
    _shadow_category,
    _shadow_seed,
    _stochastic_maxabs_roundtrip,
    _tensor_error_metrics,
    install_muon_contract,
)

assert _shadow_category("model.decoder.layers.0.self_attention.linear_qkv.weight") == "attention_qkv"
assert _shadow_category("model.decoder.layers.0.mlp.linear_fc1.weight") == "dense_mlp_fc1"
assert _shadow_category("model.decoder.layers.1.mlp.experts.linear_fc1.weight0") == "routed_expert_fc1"
assert _shadow_category("model.decoder.layers.1.mlp.router.weight") is None

reference = torch.tensor([1.0, -2.0, 3.0])
metrics = _tensor_error_metrics(reference, reference)
assert abs(metrics["cosine"] - 1.0) < 1e-6
assert metrics["relative_l2"] == 0.0
assert metrics["norm_ratio"] == 1.0

exact = torch.tensor([448.0, 1.0, 1.125, -0.015625])
restored, payload = _stochastic_maxabs_roundtrip(exact, group_size=4, seed=7)
assert torch.equal(restored, exact)
assert torch.equal(payload, exact)
value = torch.full((128,), 1.0625)
value[0] = 448.0
first, first_payload = _stochastic_maxabs_roundtrip(value, group_size=128, seed=11)
replay, replay_payload = _stochastic_maxabs_roundtrip(value, group_size=128, seed=11)
other, _ = _stochastic_maxabs_roundtrip(value, group_size=128, seed=12)
assert torch.equal(first, replay)
assert torch.equal(first_payload, replay_payload)
assert not torch.equal(first, other)
assert torch.equal(first_payload.to(torch.float8_e4m3fn).float(), first_payload)
assert _shadow_seed("parameter", 3) == _shadow_seed("parameter", 3)
assert _shadow_seed("parameter", 3) != _shadow_seed("parameter", 4)

entry = _EMERGING_OPTIMIZERS["muon"]
install_muon_contract(fp8_states=False, shadow_states=True)
assert entry.optimizer_cls is MuonFP8ShadowDiagnostic
try:
    install_muon_contract(fp8_states=True, shadow_states=True)
except ValueError:
    pass
else:
    raise AssertionError("FP8 shadow diagnostic accepted an FP8-state primary arm")
print("MUON_SHADOW_CONTRACT=pass")
PY
code=$?
echo "TEST_EXIT=$code"
exit 0
