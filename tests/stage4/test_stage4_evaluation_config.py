import json
from pathlib import Path


CONFIG = Path(__file__).parents[2] / "configs" / "stage4-evaluation.json"


def test_stage4_evaluation_config_is_pinned_and_complete():
    config = json.loads(CONFIG.read_text())

    assert config["primary"]["provider"] == "lm-eval"
    assert config["primary"]["version"] == "0.4.11"
    assert config["primary"]["include_path"] == "stage4/eval_tasks"
    assert config["primary"]["task_group"] == "basic_v2"
    assert config["primary"]["tasks"] == [
        "basic_v2_hellaswag",
        "basic_v2_arc_easy",
        "basic_v2_arc_challenge",
        "basic_v2_piqa",
        "basic_v2_gsm8k_gold_bpb_5shot",
    ]
    assert config["primary"]["log_samples"]

    secondary = config["secondary"]
    assert secondary["provider"] == "lm-eval"
    assert secondary["version"] == "0.4.11"
    assert [task["name"] for task in secondary["tasks"]] == [
        "wikitext",
        "c4",
        "winogrande",
        "openbookqa",
        "mmlu",
    ]
    assert secondary["log_samples"]
    assert secondary["preserve_per_example_outputs"]


def test_stage4_evaluation_config_has_paired_comparison_rules():
    comparison = json.loads(CONFIG.read_text())["comparison"]

    assert comparison == {
        "pair_key": "task_and_doc_id",
        "bootstrap_confidence": 0.95,
        "accuracy_test": "mcnemar_secondary",
        "continuous_metrics": "paired_bootstrap",
    }
