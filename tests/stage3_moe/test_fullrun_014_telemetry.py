import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from stage3_moe.result_writer import TOKEN_TALLY, install_router_tallies


def config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=64,
        moe_router_topk=8,
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="none",
        moe_router_score_function="sigmoid",
        moe_router_pre_softmax=True,
        moe_router_dtype="fp32",
        bf16=True,
    )


def wrapped_router():
    torch.cuda.set_device(0)
    groups = ProcessGroupCollection()
    for name in ("tp", "pp", "cp", "tp_cp", "tp_dp_cp"):
        setattr(groups, name, None)
    router = TopKRouter(config(), pg_collection=groups)
    wrapped = Float16Module(config(), router)
    router._maintain_float32_expert_bias()
    return wrapped, router


def routing_map(valid_tokens=1, padded_tokens=0):
    rows = []
    for index in range(valid_tokens + padded_tokens):
        row = torch.zeros(64, dtype=torch.bool, device="cuda")
        row[torch.arange(index % 8, index % 8 + 8, device="cuda")] = True
        rows.append(row)
    return torch.stack(rows)


def assert_invariants(counts, valid_tokens, calls):
    assert counts.dtype == torch.int64
    assert counts.device.type == "cuda"
    total = int(counts.sum().item())
    assert total == 8 * valid_tokens * calls
    assert int(counts.max().item()) <= valid_tokens * calls
    live = int((counts > 0).sum().item())
    if valid_tokens * calls:
        assert live >= 8
        mean = counts.to(torch.float64).mean()
        cv = counts.to(torch.float64).std(unbiased=False) / mean
        assert float(cv) <= 7**0.5 + 1e-12
    else:
        assert live == 0
    return {"assignments": total, "live_experts": live, "max_expert_count": int(counts.max().item())}


def saturation_and_semantics():
    wrapped, router = wrapped_router()
    assert router.local_tokens_per_expert.dtype == torch.bfloat16
    assert router.expert_bias.dtype == torch.float32
    install_router_tallies([wrapped])
    tally = getattr(router, TOKEN_TALLY)
    assert tally.dtype == torch.int64
    router.local_tokens_per_expert.fill_(8_388_608)
    one = routing_map()
    before = router.local_tokens_per_expert.detach().clone()
    for _ in range(4):
        router._apply_expert_bias(one)
    assert torch.equal(router.local_tokens_per_expert, before)
    exact = assert_invariants(tally, 1, 4)
    assert exact["assignments"] == 32 and exact["live_experts"] == 8

    tally.zero_()
    router.local_tokens_per_expert.zero_()
    padded = routing_map(valid_tokens=2, padded_tokens=2)
    mask = torch.tensor([False, False, True, True], device="cuda")
    router._apply_expert_bias(padded, padding_mask=mask)
    padded_result = assert_invariants(tally, 2, 1)
    before_no_grad = tally.clone()
    with torch.no_grad():
        router._apply_expert_bias(one)
    assert torch.equal(tally, before_no_grad)
    all_padding = torch.ones(4, dtype=torch.bool, device="cuda")
    before_all_padding = tally.clone()
    router._apply_expert_bias(padded, padding_mask=all_padding)
    assert torch.equal(tally, before_all_padding)
    return {
        "production_count_dtype_after_float16module": str(router.local_tokens_per_expert.dtype),
        "expert_bias_dtype_after_maintenance": str(router.expert_bias.dtype),
        "observer_dtype": str(tally.dtype),
        "bf16_counter_unchanged_at_8388608_after_four_increments": True,
        "exact_counter_assignments_after_four_increments": exact["assignments"],
        "padding": padded_result,
        "no_grad_unchanged": True,
        "all_padding_unchanged": True,
        "top8_invariants_passed": True,
    }


def optimizer_state_copy(optimizer):
    result = {}
    for parameter, state in optimizer.state.items():
        result[parameter] = {key: value.detach().clone() if torch.is_tensor(value) else value for key, value in state.items()}
    return result


def onoff_equivalence():
    off_wrap, off = wrapped_router()
    on_wrap, on = wrapped_router()
    on.load_state_dict(off.state_dict())
    install_router_tallies([on_wrap])
    off_adam = torch.optim.Adam(off.parameters(), lr=0.0)
    on_adam = torch.optim.Adam(on.parameters(), lr=0.0)
    off_adam.zero_grad(); on_adam.zero_grad()
    for parameter in off.parameters():
        parameter.grad = torch.ones_like(parameter) * 0.01
    for parameter in on.parameters():
        parameter.grad = torch.ones_like(parameter) * 0.01
    off.local_tokens_per_expert.fill_(8_388_608)
    on.local_tokens_per_expert.fill_(8_388_608)
    mapping = routing_map(valid_tokens=3, padded_tokens=1)
    mask = torch.tensor([False, False, False, True], device="cuda")
    off._apply_expert_bias(mapping, padding_mask=mask)
    on._apply_expert_bias(mapping, padding_mask=mask)
    off_adam.step(); on_adam.step()
    assert torch.equal(off.local_tokens_per_expert, on.local_tokens_per_expert)
    assert torch.equal(off.expert_bias, on.expert_bias)
    for left, right in zip(off.parameters(), on.parameters()):
        assert torch.equal(left, right)
        assert torch.equal(left.grad, right.grad)
    off_state = optimizer_state_copy(off_adam)
    on_state = optimizer_state_copy(on_adam)
    assert len(off_state) == len(on_state)
    for (left_parameter, left), (right_parameter, right) in zip(off_state.items(), on_state.items()):
        assert left.keys() == right.keys()
        for key in left:
            if torch.is_tensor(left[key]):
                assert torch.equal(left[key], right[key])
            else:
                assert left[key] == right[key]
    return {
        "parameters_equal": True,
        "gradients_equal": True,
        "production_counts_equal": True,
        "expert_bias_equal": True,
        "adam_state_equal": True,
        "controller_math_untouched": True,
    }


def main():
    result = {
        "schema": "hmoe-fullrun014-telemetry-v1",
        "saturation": saturation_and_semantics(),
        "onoff": onoff_equivalence(),
        "full_model_constructed": False,
        "optimizer_or_router_math_changed": False,
        "passed": True,
    }
    output = Path(os.environ["HMOE_CONTRACT_ROOT"]) / "telemetry-contracts.json"
    with output.open("w") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("FULLRUN014_TELEMETRY " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
