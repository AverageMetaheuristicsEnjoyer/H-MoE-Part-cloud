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
    assert secondary["analysis_scope"] == "exploratory_posthoc_wave2"
    assert secondary["task_group"] == "stage3_wave2"
    assert [task["name"] for task in secondary["tasks"]] == [
        "wikitext",
        "c4",
        "winogrande",
        "openbookqa",
        "mmlu",
    ]
    assert [task["metrics"] for task in secondary["tasks"]] == [
        ["bits_per_byte"],
        ["bits_per_byte"],
        ["acc"],
        ["acc", "acc_norm"],
        ["acc"],
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


def test_broad_screen_is_frozen_before_scoring():
    broad = json.loads(CONFIG.read_text())["broad_screen"]

    assert broad["analysis_scope"] == "exploratory_posthoc_broad_v1"
    assert broad["task_group"] == "stage3_broad_v1"
    assert [task["name"] for task in broad["tasks"]] == [
        "blimp", "swag", "mnli", "mnli_mismatch", "qnli", "qqp", "prost",
        "toxigen", "moral_stories", "boolq", "race",
        "lambada_openai", "pile_10k", "leaderboard_mmlu_pro",
    ]
    assert broad["log_samples"]
    assert broad["preserve_per_example_outputs"]
