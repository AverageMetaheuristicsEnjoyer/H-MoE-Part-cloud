import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from stage3_moe.routing_audit import RoutingAudit


ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "Megatron-LM"))


class FakeRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_moe_experts=4, moe_router_bias_update_rate=1e-3)
        self.topk = 2
        self.register_buffer("expert_bias", torch.tensor([0.30, 0.0, 0.0, -0.30]))

    def routing(self, logits, padding_mask=None):
        scores = torch.sigmoid(logits.reshape(-1, 4)) + self.expert_bias
        indices = torch.topk(scores, self.topk, dim=-1).indices
        routing_map = torch.zeros_like(scores, dtype=torch.bool)
        routing_map.scatter_(1, indices, True)
        return scores, routing_map


class FakeChunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = FakeRouter()


def test_audit_observes_bias_effect_without_changing_routing(tmp_path, monkeypatch):
    import megatron.training

    args = SimpleNamespace(
        eval_global_batch_size=1,
        eval_micro_batch_size=1,
        data_parallel_size=1,
    )
    monkeypatch.setattr(megatron.training, "get_args", lambda: args)
    chunk = FakeChunk()
    audit = RoutingAudit([chunk], tmp_path / "audit.json")
    logits = torch.tensor(
        [
            [[0.0, 0.2, 0.1, 2.0]],
            [[0.1, 0.4, 0.3, 1.5]],
        ]
    )

    _, routing_map = chunk.router.routing(logits)
    state = audit.layers[0]

    assert state["tokens"].item() == 2
    assert state["actual"].sum().item() == 4
    assert state["routing_mismatches"].item() == 0
    assert state["bias_changes"].item() > 0
    assert state["bias_updates"].item() == 1
    expected = torch.topk(
        torch.sigmoid(logits.reshape(-1, 4)) + chunk.router.expert_bias, 2, dim=-1
    ).indices
    expected_map = torch.zeros_like(routing_map)
    expected_map.scatter_(1, expected, True)
    assert torch.equal(routing_map, expected_map)

    audit.reset()
    assert state["tokens"].item() == 0
    assert state["actual"].sum().item() == 0
