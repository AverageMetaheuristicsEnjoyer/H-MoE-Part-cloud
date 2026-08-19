import json
import unittest
from pathlib import Path

try:
    import jsonschema
except ModuleNotFoundError:
    jsonschema = None

from stage3_moe.pair_results import compare_runs, load_run


SHA = "a" * 64
ROOT = Path(__file__).resolve().parents[2]


def make_run(arm, role, axis, optimizer, gemm, state, *, protocol="formal_timing"):
    return {
        "schema_version": 1,
        "record_type": "run",
        "run_id": arm,
        "arm_id": arm,
        "status": "completed",
        "comparison": {
            "optimizer": optimizer,
            "gemm_mode": gemm,
            "optimizer_state_mode": state,
            "match_key_sha256": SHA,
        },
        "denominators": {
            "micro_batch_sequences_per_gpu": 1,
            "global_batch_sequences": 1,
            "sequence_length": 2048,
            "loss_tokens_per_step": 2048,
            "gpu_count": 1,
            "total_parameters": 1028926976,
            "active_parameters_per_token": 280243712,
            "dp": 1,
            "tp": 1,
            "pp": 1,
            "cp": 1,
            "ep": 1,
            "etp": 1,
        },
        "provenance": {
            "git_commit": "b" * 40,
            "git_dirty": False,
            "git_diff_sha256": None,
            "config_sha256": SHA,
            "data_manifest_sha256": SHA,
            "argv_scope": "effective_mcore_argv",
            "argv": ["probe"],
        },
        "environment": {
            "site": "test",
            "host": "host",
            "scheduler_job_id": "1",
            "image": "torch28",
            "python": "3.12",
            "torch": "2.8",
            "cuda_runtime": "12.8",
            "driver": "test",
            "cublaslt": "test",
            "triton": "test",
            "transformer_engine": "2.16",
            "megatron_core": "0.18.2",
            "emerging_optimizers": "0.2.0",
            "nccl": "test",
            "gpus": [{"uuid": "GPU-test", "name": "H100"}],
            "gpu_clean": {"before": True, "during": True, "after": True},
        },
        "parameter_groups": [
            {
                "role": "adamw_all" if optimizer == "adamw" else "muon_matrix",
                "parameters": 1,
                "active_parameters_per_token": 1,
                "named_parameter_manifest_sha256": SHA,
            }
        ],
        "optimizer_state": {
            "persistent_data_bytes": 100,
            "metadata_bytes": 0,
            "persistent_total_bytes": 100,
            "master_parameter_bytes": 400,
            "tensors": [],
        },
        "measurement": {
            "protocol": {
                "kind": protocol,
                "warmup_steps": 20 if protocol != "smoke" else 3,
                "measured_steps": 100 if protocol != "smoke" else 5,
                "e2e_train_steps": 120 if protocol != "smoke" else 8,
            },
            "memory": {"max_allocated_bytes": 100, "max_reserved_bytes": 120},
            "timing": {
                "tokens_per_second": 100,
                "optimizer_step_seconds": 0.1,
                "full_step_seconds": 1.0,
                "e2e_wct_seconds": 110,
                "e2e_wct_scope": (
                    "process_start_to_result_write"
                    if protocol == "smoke"
                    else "launcher_start_to_process_exit"
                ),
                "optimizer_step_samples_seconds": [0.1] * (100 if protocol != "smoke" else 5),
                "full_step_samples_seconds": [1.0] * (100 if protocol != "smoke" else 5),
            },
            "loss": {"training": 2.0, "validation": 2.0},
            "routing": {
                "scope": "global_unpadded",
                "tokens_per_expert_artifact_sha256": SHA,
                "minimum_to_mean": 0.5,
                "maximum_to_mean": 1.5,
                "coefficient_of_variation": 0.1,
                "dropped_tokens": 0,
            },
            "downstream": [
                {
                    "task": "piqa",
                    "metric": "acc",
                    "higher_is_better": True,
                    "value": 1.0,
                    "per_example_artifact_sha256": SHA,
                }
            ],
            "inference": None,
        },
    }


def state_pair(protocol="formal_timing", optimizer="adamw"):
    prefix = optimizer
    baseline = make_run(
        f"{prefix}_bf16_state_fp32", "baseline", "optimizer_state", optimizer, "bf16", "fp32", protocol=protocol
    )
    treatment = make_run(
        f"{prefix}_bf16_state_fp8", "treatment", "optimizer_state", optimizer, "bf16", "fp8_hybrid", protocol=protocol
    )
    if protocol != "smoke":
        treatment["measurement"]["inference"] = {
            "baseline_run_id": baseline["run_id"],
            "method": "paired bootstrap",
            "confidence_level": 0.95,
            "sidedness": "one-sided",
            "memory_allocated_ratio_ci95": [1.10, 1.12],
            "e2e_wct_ratio_ci95": None,
            "validation_loss_degradation_ci95": [0.0, 0.009],
            "downstream": [
                {"task": "piqa", "metric": "acc", "degradation_ci95": [0.0, 0.009]}
            ],
        }
    return baseline, treatment


def compute_pair(protocol="formal_timing", optimizer="muon"):
    prefix = optimizer
    baseline = make_run(
        f"{prefix}_bf16_state_fp32", "baseline", "fp8_gemm", optimizer, "bf16", "fp32", protocol=protocol
    )
    treatment = make_run(
        f"{prefix}_fp8gemm_state_fp32",
        "treatment",
        "fp8_gemm",
        optimizer,
        "fp8_delayed_hybrid",
        "fp32",
        protocol=protocol,
    )
    treatment["provenance"]["argv"] = [
        "probe",
        "--fp8-format",
        "hybrid",
        "--fp8-recipe",
        "delayed",
    ]
    if protocol != "smoke":
        treatment["measurement"]["inference"] = {
            "baseline_run_id": baseline["run_id"],
            "method": "paired bootstrap",
            "confidence_level": 0.95,
            "sidedness": "one-sided",
            "memory_allocated_ratio_ci95": None,
            "e2e_wct_ratio_ci95": [1.10, 1.12],
            "validation_loss_degradation_ci95": [0.0, 0.009],
            "downstream": [
                {"task": "piqa", "metric": "acc", "degradation_ci95": [0.0, 0.009]}
            ],
        }
    return baseline, treatment


class PairResultsTest(unittest.TestCase):
    @unittest.skipUnless(jsonschema, "jsonschema is not installed")
    def test_json_schema_accepts_all_run_and_pair_fixtures(self):
        schema = json.loads((ROOT / "stage3_moe" / "result.schema.json").read_text())
        validator_class = jsonschema.validators.validator_for(schema)
        validator_class.check_schema(schema)
        validator = validator_class(schema)
        pairs = []
        for optimizer in ("adamw", "muon"):
            pairs.extend(
                (
                    state_pair(optimizer=optimizer),
                    compute_pair(optimizer=optimizer),
                )
            )
        for baseline, treatment in pairs:
            validator.validate(baseline)
            validator.validate(treatment)
            validator.validate(compare_runs(baseline, treatment))
        self.assertEqual(len(pairs) * 2, 8)
        self.assertEqual(len(pairs), 4)

    def test_all_four_exact_pairs_are_supported(self):
        for optimizer in ("adamw", "muon"):
            with self.subTest(axis="optimizer_state", optimizer=optimizer):
                self.assertEqual(compare_runs(*state_pair(optimizer=optimizer))["axis"], "optimizer_state")
            with self.subTest(axis="fp8_gemm", optimizer=optimizer):
                self.assertEqual(compare_runs(*compute_pair(optimizer=optimizer))["axis"], "fp8_gemm")

    def test_state_memory_gate_uses_literal_1_10_ratio(self):
        baseline, treatment = state_pair()
        baseline["measurement"]["memory"]["max_allocated_bytes"] = 110
        treatment["measurement"]["memory"]["max_allocated_bytes"] = 100
        self.assertEqual(compare_runs(baseline, treatment)["gates"]["memory"]["status"], "pass")

        treatment["measurement"]["inference"]["memory_allocated_ratio_ci95"] = [1.08, 1.09]
        self.assertEqual(compare_runs(baseline, treatment)["gates"]["memory"]["status"], "fail")

        treatment["measurement"]["inference"]["memory_allocated_ratio_ci95"] = [1.09, 1.11]
        self.assertEqual(compare_runs(baseline, treatment)["gates"]["memory"]["status"], "inconclusive")

    def test_compute_wct_gate_and_overall_pass(self):
        baseline, treatment = compute_pair()
        baseline["measurement"]["timing"]["e2e_wct_seconds"] = 110
        treatment["measurement"]["timing"]["e2e_wct_seconds"] = 100
        result = compare_runs(baseline, treatment)
        self.assertEqual(result["gates"]["wct"]["status"], "pass")
        self.assertAlmostEqual(result["effects"]["e2e_wct_seconds"]["ratio"], 1.1)
        self.assertEqual(result["verdict"], "pass")

        treatment["measurement"]["inference"]["e2e_wct_ratio_ci95"] = [1.09, 1.11]
        self.assertEqual(compare_runs(baseline, treatment)["gates"]["wct"]["status"], "inconclusive")

        treatment["measurement"]["inference"]["e2e_wct_ratio_ci95"] = [1.07, 1.09]
        self.assertEqual(compare_runs(baseline, treatment)["gates"]["wct"]["status"], "fail")

    def test_exact_pair_invariants_reject_mismatch(self):
        baseline, treatment = state_pair()
        treatment["comparison"]["optimizer"] = "muon"
        with self.assertRaisesRegex(ValueError, "same optimizer"):
            compare_runs(baseline, treatment)

        baseline, treatment = state_pair()
        treatment["comparison"]["gemm_mode"] = "fp8_delayed_hybrid"
        with self.assertRaisesRegex(ValueError, "changes more than"):
            compare_runs(baseline, treatment)

        baseline, treatment = state_pair()
        treatment["denominators"]["global_batch_sequences"] = 2
        with self.assertRaisesRegex(ValueError, "denominators differ"):
            compare_runs(baseline, treatment)

        baseline, treatment = state_pair()
        treatment["provenance"]["config_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "config hashes differ"):
            compare_runs(baseline, treatment)

        baseline, treatment = state_pair()
        treatment["provenance"]["argv"].extend(["--micro-batch-size", "2"])
        with self.assertRaisesRegex(ValueError, "normalized effective MCore argv differ"):
            compare_runs(baseline, treatment)

        baseline, treatment = compute_pair()
        treatment["provenance"]["argv"].extend(["--hidden-size", "2048"])
        with self.assertRaisesRegex(ValueError, "normalized effective MCore argv differ"):
            compare_runs(baseline, treatment)

    def test_downstream_degradation_is_strictly_less_than_one_percent(self):
        baseline, treatment = state_pair()
        baseline["measurement"]["memory"]["max_allocated_bytes"] = 110
        treatment["measurement"]["memory"]["max_allocated_bytes"] = 100
        treatment["measurement"]["downstream"][0]["value"] = 0.991
        self.assertEqual(compare_runs(baseline, treatment)["gates"]["downstream"]["status"], "pass")

        treatment["measurement"]["inference"]["downstream"][0]["degradation_ci95"] = [0.01, 0.012]
        self.assertEqual(compare_runs(baseline, treatment)["gates"]["downstream"]["status"], "fail")

        treatment["measurement"]["inference"]["downstream"][0]["degradation_ci95"] = [0.009, 0.011]
        self.assertEqual(compare_runs(baseline, treatment)["gates"]["downstream"]["status"], "inconclusive")

    def test_validation_uses_paired_upper_confidence_bound(self):
        baseline, treatment = state_pair()
        self.assertEqual(compare_runs(baseline, treatment)["gates"]["validation"]["status"], "pass")

        treatment["measurement"]["inference"]["validation_loss_degradation_ci95"] = [0.01, 0.012]
        self.assertEqual(compare_runs(baseline, treatment)["gates"]["validation"]["status"], "fail")

        treatment["measurement"]["inference"]["validation_loss_degradation_ci95"] = [0.009, 0.011]
        self.assertEqual(compare_runs(baseline, treatment)["gates"]["validation"]["status"], "inconclusive")

    def test_point_estimates_without_paired_confidence_bounds_are_inconclusive(self):
        for baseline, treatment, gate in (
            (*state_pair(), "memory"),
            (*compute_pair(), "wct"),
        ):
            treatment["measurement"]["inference"] = None
            result = compare_runs(baseline, treatment)
            self.assertEqual(result["gates"][gate]["status"], "inconclusive")
            self.assertEqual(result["verdict"], "inconclusive")

    def test_inference_method_uses_schema_case_sensitivity(self):
        baseline, treatment = state_pair()
        treatment["measurement"]["inference"]["method"] = "Paired bootstrap"
        with self.assertRaisesRegex(ValueError, "lowercase paired"):
            compare_runs(baseline, treatment)

    def test_formal_wct_requires_outer_launcher_scope(self):
        baseline, treatment = compute_pair()
        for run in (baseline, treatment):
            run["measurement"]["timing"]["e2e_wct_scope"] = "process_start_to_result_write"
        result = compare_runs(baseline, treatment)
        self.assertEqual(result["gates"]["wct"]["status"], "inconclusive")
        self.assertEqual(result["verdict"], "inconclusive")

    def test_short_smoke_is_inconclusive_and_jsonl_loads_one_run(self):
        baseline, treatment = compute_pair(protocol="smoke")
        with self.subTest("jsonl loader"):
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                path = f"{directory}/results.jsonl"
                with open(path, "w") as handle:
                    handle.write(json.dumps(baseline) + "\n")
                self.assertEqual(load_run(path), baseline)
        self.assertEqual(compare_runs(baseline, treatment)["verdict"], "inconclusive")


if __name__ == "__main__":
    unittest.main()


class PerArmPathTest(unittest.TestCase):
    """A pair may differ in where each arm reads and writes, and nowhere else."""

    def pair(self, baseline_extra, treatment_extra):
        baseline = make_run(
            "adamw_bf16_state_fp32", "baseline", "fp8_gemm", "adamw", "bf16", "fp32"
        )
        treatment = make_run(
            "adamw_fp8gemm_state_fp32", "treatment", "fp8_gemm", "adamw",
            "fp8_delayed_hybrid", "fp32",
        )
        baseline["provenance"]["argv"] = list(baseline["provenance"]["argv"]) + baseline_extra
        treatment["provenance"]["argv"] = (
            list(treatment["provenance"]["argv"]) + treatment_extra + ["--fp8-format", "hybrid", "--fp8-recipe", "delayed"]
        )
        return baseline, treatment

    def test_paths_differing_only_by_the_arm_id_are_accepted(self):
        baseline, treatment = self.pair(
            ["--load", "/ckpt/trunk/adamw_bf16_state_fp32",
             "--wandb-exp-name", "stage3-adamw_bf16_state_fp32-decay"],
            ["--load", "/ckpt/trunk/adamw_fp8gemm_state_fp32",
             "--wandb-exp-name", "stage3-adamw_fp8gemm_state_fp32-decay"],
        )
        self.assertEqual(compare_runs(baseline, treatment)["axis"], "fp8_gemm")

    def test_a_different_checkpoint_root_is_still_refused(self):
        baseline, treatment = self.pair(
            ["--load", "/ckpt/trunk/adamw_bf16_state_fp32"],
            ["--load", "/somewhere-else/trunk/adamw_fp8gemm_state_fp32"],
        )
        with self.assertRaises(ValueError):
            compare_runs(baseline, treatment)

    def test_a_computational_argument_is_still_refused(self):
        baseline, treatment = self.pair(["--lr", "1.63e-3"], ["--lr", "1.0e-3"])
        with self.assertRaises(ValueError):
            compare_runs(baseline, treatment)
