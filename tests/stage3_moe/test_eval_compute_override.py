import unittest

from stage3_moe.pretrain_gpt import take_stage3_args, validate_axis


def mcore_argv(*compute_args):
    return [
        "pretrain_gpt.py",
        "--optimizer",
        "adam",
        "--train-iters",
        "17242",
        *compute_args,
    ]


class EvalComputeOverrideTest(unittest.TestCase):
    def test_fp8_arm_allows_bf16_only_for_explicit_evaluation(self):
        validate_axis(
            "adamw_bf16_state_fp32", "fp32", mcore_argv(), 0, 1
        )
        validate_axis(
            "adamw_bf16_state_fp32", "fp32", mcore_argv(), 0, 1, "bf16"
        )
        validate_axis(
            "adamw_fp8gemm_state_fp32",
            "fp32",
            mcore_argv(),
            0,
            1,
            "bf16",
        )
        with self.assertRaises(ValueError):
            validate_axis(
                "adamw_fp8gemm_state_fp32", "fp32", mcore_argv(), 0, 1
            )

    def test_evaluation_mode_must_match_effective_mcore_argv(self):
        fp8_argv = mcore_argv("--fp8-format", "hybrid", "--fp8-recipe", "delayed")
        validate_axis(
            "adamw_fp8gemm_state_fp32", "fp32", fp8_argv, 0, 1
        )
        validate_axis(
            "adamw_fp8gemm_state_fp32",
            "fp32",
            fp8_argv,
            0,
            1,
            "fp8_delayed_hybrid",
        )
        with self.assertRaises(ValueError):
            validate_axis(
                "adamw_fp8gemm_state_fp32", "fp32", fp8_argv, 0, 1, "bf16"
            )

    def test_compute_mode_is_accepted_only_with_downstream_tasks(self):
        argv = [
            "pretrain_gpt.py",
            "--stage3-arm",
            "adamw_fp8gemm_state_fp32",
            "--stage3-result-path",
            "results.jsonl",
            "--stage3-warmup-steps",
            "0",
            "--stage3-measure-steps",
            "1",
            "--optimizer-state-precision",
            "fp32",
            "--stage3-eval-compute-mode",
            "bf16",
        ]
        with self.assertRaises(ValueError):
            take_stage3_args(argv)

        args, remaining = take_stage3_args(
            [*argv, "--stage3-eval-downstream", "basic_v2_piqa"]
        )
        self.assertEqual(args.stage3_eval_compute_mode, "bf16")
        self.assertNotIn("--stage3-eval-compute-mode", remaining)


if __name__ == "__main__":
    unittest.main()
