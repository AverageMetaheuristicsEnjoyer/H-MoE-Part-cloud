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
import copy

import torch

from megatron.core.optimizer.emerging_optimizers import _EMERGING_OPTIMIZERS
from stage3_moe.muon import (
    DRE2StateSplitSwiGLUTensorParallelMuon,
    MuonFP8ShadowDiagnostic,
    SplitSwiGLUTensorParallelMuon,
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

parameter = torch.nn.Parameter(torch.randn(256, 256, device="cuda"))
optimizer = DRE2StateSplitSwiGLUTensorParallelMuon([parameter], lr=1e-3)
optimizer.group_size = 256
parameter.grad = torch.randn_like(parameter)
optimizer.step()
state = optimizer.state[parameter]
assert state["momentum_buffer"].dtype == torch.float8_e4m3fn
assert state["momentum_buffer_residual"].dtype == torch.float8_e4m3fn

resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
resumed = DRE2StateSplitSwiGLUTensorParallelMuon([resumed_parameter], lr=1e-3)
resumed.group_size = 256
resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
resumed_state = resumed.state[resumed_parameter]
assert all(
    torch.equal(value, resumed_state[key])
    for key, value in state.items()
    if torch.is_tensor(value)
)
grad = torch.randn_like(parameter)
parameter.grad = grad.clone()
resumed_parameter.grad = grad.clone()
optimizer.step()
resumed.step()
assert torch.equal(parameter, resumed_parameter)
assert all(
    torch.equal(value, resumed.state[resumed_parameter][key])
    for key, value in optimizer.state[parameter].items()
    if torch.is_tensor(value)
)

torch.manual_seed(1234)
reference_parameter = torch.nn.Parameter(torch.randn(256, 256, device="cuda"))
dre2_parameter = torch.nn.Parameter(reference_parameter.detach().clone())
reference_optimizer = SplitSwiGLUTensorParallelMuon(
    [reference_parameter], lr=1.0, weight_decay=0.0
)
dre2_optimizer = DRE2StateSplitSwiGLUTensorParallelMuon(
    [dre2_parameter], lr=1.0, weight_decay=0.0
)
dre2_optimizer.group_size = 256
generator = torch.Generator(device="cuda")
generator.manual_seed(5678)
max_state_rel = 0.0
max_update_rel = 0.0
min_update_cos = 1.0
for _ in range(1000):
    grad = torch.randn(
        reference_parameter.shape, device="cuda", generator=generator
    )
    reference_before = reference_parameter.detach().clone()
    dre2_before = dre2_parameter.detach().clone()
    reference_parameter.grad = grad.clone()
    dre2_parameter.grad = grad.clone()
    reference_optimizer.step()
    dre2_optimizer.step()
    reference_update = reference_before - reference_parameter
    dre2_update = dre2_before - dre2_parameter
    update_metrics = _tensor_error_metrics(reference_update, dre2_update)
    max_update_rel = max(max_update_rel, update_metrics["relative_l2"])
    min_update_cos = min(min_update_cos, update_metrics["cosine"])
    reference_state = reference_optimizer.state[reference_parameter]["momentum_buffer"]
    dre2_state = dre2_optimizer._restore_state(
        dre2_optimizer.state[dre2_parameter], dre2_optimizer.state_specs[0]
    )
    max_state_rel = max(
        max_state_rel,
        _tensor_error_metrics(reference_state, dre2_state)["relative_l2"],
    )
assert max_state_rel <= 0.03
assert max_update_rel <= 0.02
assert min_update_cos >= 0.999
print(
    "MUON_DRE2_LONG_HORIZON=pass"
    f" max_state_rel={max_state_rel:.6g}"
    f" max_update_rel={max_update_rel:.6g}"
    f" min_update_cos={min_update_cos:.6g}"
)

entry = _EMERGING_OPTIMIZERS["muon"]
install_muon_contract(fp8_states=True, state_recipe="dre2", group_size=256)
assert entry.optimizer_cls is DRE2StateSplitSwiGLUTensorParallelMuon
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
