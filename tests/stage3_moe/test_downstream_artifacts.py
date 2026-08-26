import json
import tempfile
import unittest
from pathlib import Path

from stage3_moe.downstream_artifacts import collect_downstream


class DownstreamArtifactsTest(unittest.TestCase):
    def test_collects_builtin_metrics_and_pools_mmlu_as_one_task(self):
        class Scalar:
            def item(self):
                return 3

        output = {
            "results": {
                "wikitext": {
                    "alias": "wikitext",
                    "bits_per_byte,none": 1.25,
                    "bits_per_byte_stderr,none": "N/A",
                },
                "mmlu_a": {"acc,none": 0.5, "acc_stderr,none": 0.1},
                "mmlu_b": {"acc,none": 1.0, "acc_stderr,none": 0.0},
            },
            "groups": {
                "mmlu": {"acc,none": 0.75, "acc_stderr,none": 0.1},
                "mmlu_stem": {"acc,none": 0.75, "acc_stderr,none": 0.1},
            },
            "group_subtasks": {
                "stage3_wave2": ["wikitext", "mmlu"],
                "mmlu": ["mmlu_stem"],
                "mmlu_stem": ["mmlu_a", "mmlu_b"],
                "wikitext": [],
                "mmlu_a": [],
                "mmlu_b": [],
            },
            "higher_is_better": {
                "wikitext": {"bits_per_byte": False},
                "mmlu": {"acc": True},
                "mmlu_stem": {"acc": True},
                "mmlu_a": {"acc": True},
                "mmlu_b": {"acc": True},
            },
            "samples": {
                "wikitext": [
                    {"doc_id": 0, "bits_per_byte": (-10.0, 12), "target": Scalar()}
                ],
                "mmlu_a": [{"doc_id": 0, "acc": 0.0}],
                "mmlu_b": [{"doc_id": 0, "acc": 1.0}],
            },
        }
        root = Path(tempfile.mkdtemp())

        downstream = collect_downstream(output, root)

        self.assertEqual(
            [(item["task"], item["metric"]) for item in downstream],
            [("mmlu", "acc"), ("wikitext", "bits_per_byte")],
        )
        mmlu_rows = [json.loads(line) for line in (root / "mmlu.jsonl").read_text().splitlines()]
        self.assertEqual([row["doc_id"] for row in mmlu_rows], ["mmlu_a:0", "mmlu_b:0"])
        wiki_row = json.loads((root / "wikitext.jsonl").read_text())
        self.assertEqual(wiki_row["target"], 3)
        self.assertEqual(wiki_row["metrics"]["bits_per_byte"], [-10.0, 12])


if __name__ == "__main__":
    unittest.main()
