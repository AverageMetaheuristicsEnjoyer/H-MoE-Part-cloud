#!/usr/bin/env bash
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"

python - <<'PY'
import torch

from megatron.core.optimizer.emerging_optimizers import _EMERGING_OPTIMIZERS
from stage3_moe.muon import (
    MuonFP8ShadowDiagnostic,
    _shadow_category,
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
