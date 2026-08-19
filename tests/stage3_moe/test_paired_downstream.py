import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from stage3_moe.paired_inference import (
    _load_per_example,
    _paired_bootstrap,
    downstream_intervals,
    mcnemar,
)


def write_artifact(run_dir, task, rows):
    artifact = run_dir / "downstream" / f"{task}.jsonl"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open("w") as handle:
        for doc_id, metrics in rows.items():
            handle.write(json.dumps({"doc_id": doc_id, "metrics": metrics}, sort_keys=True) + "\n")
    return hashlib.sha256(artifact.read_bytes()).hexdigest()


def make_records(baseline_values, treatment_values, task="basic_v2_piqa", metric="acc_v2"):
    """A (baseline, treatment) replicate whose per-example artifacts sit beside it."""
    root = Path(tempfile.mkdtemp())
    paths, records = [], []
    for values in (baseline_values, treatment_values):
        run_dir = root / f"run{len(paths)}"
        run_dir.mkdir()
        digest = write_artifact(
            run_dir, task, {index: {metric: value} for index, value in enumerate(values)}
        )
        paths.append(run_dir / "results.jsonl")
        records.append(
            {
                "measurement": {
                    "downstream": [
                        {
                            "task": task,
                            "metric": metric,
                            "higher_is_better": True,
                            "value": sum(values) / len(values),
                            "per_example_artifact_sha256": digest,
                        }
                    ]
                }
            }
        )
    return [(paths[0], paths[1], records[0], records[1])]


class McNemarTest(unittest.TestCase):
    def test_all_discordance_in_one_direction_is_significant(self):
        baseline = [1.0] * 10 + [1.0] * 10
        treatment = [0.0] * 10 + [1.0] * 10
        probability, only_baseline, only_treatment = mcnemar(baseline, treatment)
        self.assertEqual((only_baseline, only_treatment), (10, 0))
        self.assertLess(probability, 0.01)

    def test_balanced_discordance_is_not_significant(self):
        baseline = [1.0, 0.0] * 10
        treatment = [0.0, 1.0] * 10
        probability, only_baseline, only_treatment = mcnemar(baseline, treatment)
        self.assertEqual((only_baseline, only_treatment), (10, 10))
        self.assertEqual(probability, 1.0)

    def test_no_discordant_pairs(self):
        self.assertEqual(mcnemar([1.0, 0.0], [1.0, 0.0]), (1.0, 0, 0))


class BootstrapTest(unittest.TestCase):
    def test_identical_arms_bracket_zero(self):
        values = [1.0, 0.0] * 50
        interval = _paired_bootstrap(values, values, True)
        self.assertEqual(interval, [0.0, 0.0])

    def test_a_worse_treatment_shows_positive_degradation(self):
        baseline = [1.0] * 100
        treatment = [1.0] * 80 + [0.0] * 20
        low, high = _paired_bootstrap(baseline, treatment, True)
        self.assertGreater(low, 0.0)
        self.assertLess(high, 0.45)

    def test_lower_is_better_flips_the_sign(self):
        baseline = [2.0] * 100
        treatment = [1.0] * 100
        low, high = _paired_bootstrap(baseline, treatment, False)
        self.assertAlmostEqual(low, -0.5, places=6)
        self.assertAlmostEqual(high, -0.5, places=6)

    def test_too_few_documents(self):
        self.assertIsNone(_paired_bootstrap([1.0], [1.0], True))


class DownstreamIntervalsTest(unittest.TestCase):
    def test_matching_artifacts_produce_one_interval_per_metric(self):
        replicates = make_records([1.0, 0.0] * 50, [1.0, 0.0] * 50)
        block, secondary = downstream_intervals(replicates)
        self.assertEqual(len(block), 1)
        self.assertEqual(block[0]["task"], "basic_v2_piqa")
        self.assertEqual(block[0]["metric"], "acc_v2")
        self.assertEqual(len(block[0]["degradation_ci95"]), 2)
        # 0/1 outcomes also get the secondary paired test.
        self.assertIn(("basic_v2_piqa", "acc_v2"), secondary)

    def test_a_tampered_artifact_is_refused(self):
        replicates = make_records([1.0, 0.0] * 50, [1.0, 0.0] * 50)
        baseline_path = replicates[0][0]
        artifact = baseline_path.parent / "downstream" / "basic_v2_piqa.jsonl"
        artifact.write_text(artifact.read_text() + '{"doc_id": 999, "metrics": {"acc_v2": 1.0}}\n')
        self.assertEqual(downstream_intervals(replicates)[0], [])

    def test_missing_artifact_leaves_the_gate_inconclusive(self):
        replicates = make_records([1.0, 0.0] * 50, [1.0, 0.0] * 50)
        baseline_path = replicates[0][0]
        (baseline_path.parent / "downstream" / "basic_v2_piqa.jsonl").unlink()
        self.assertEqual(downstream_intervals(replicates)[0], [])

    def test_hash_check_reads_back_what_was_written(self):
        replicates = make_records([1.0, 0.0], [1.0, 0.0])
        path = replicates[0][0]
        digest = replicates[0][2]["measurement"]["downstream"][0]["per_example_artifact_sha256"]
        rows = _load_per_example(path, "basic_v2_piqa", digest)
        self.assertEqual(rows, {0: {"acc_v2": 1.0}, 1: {"acc_v2": 0.0}})


if __name__ == "__main__":
    unittest.main()
