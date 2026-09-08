import torch

from stage3_moe.result_writer import routing_balance_summary


def test_routing_balance_summary_matches_deepseek_maxvio_definition():
    summary = routing_balance_summary(
        torch.tensor([[10, 10, 10, 10], [0, 10, 20, 10]])
    )

    assert summary["maxvio_per_layer"] == [0.0, 1.0]
    assert summary["maxvio_mean"] == 0.5
    assert summary["minimum_to_mean_min"] == 0.0
    assert summary["maximum_to_mean_max"] == 2.0
    assert summary["zero_experts_max"] == 1


def test_routing_balance_summary_aggregates_a_rolling_window_before_scoring():
    first = torch.tensor([[0, 20], [10, 10]])
    second = torch.tensor([[20, 0], [10, 10]])

    summary = routing_balance_summary(torch.stack((first, second)).sum(dim=0))

    assert summary["maxvio_max"] == 0.0
    assert summary["minimum_to_mean_min"] == 1.0
    assert summary["coefficient_of_variation_max"] == 0.0
