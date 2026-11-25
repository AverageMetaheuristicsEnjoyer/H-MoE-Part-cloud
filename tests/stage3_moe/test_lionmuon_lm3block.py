import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "Megatron-LM"))
sys.path.insert(0, str(ROOT / "third_party" / "emerging-optimizers"))

from stage3_moe import lionmuon
from stage3_moe.lionmuon import (
    LIONMUON_BLOCK_FACTORS,
    LIONMUON_BLOCK_KEY,
    LionMuonMixin,
    lionmuon_block_boundaries,
    lionmuon_block_draw,
    lionmuon_block_for_layer,
    lionmuon_block_for_name,
    lionmuon_block_layers_label,
    lionmuon_block_probability,
    lionmuon_block_telemetry,
    parse_lionmuon_block_factors,
    parse_schedule,
    piecewise_linear,
)

SEED = 1234
FULL_ITERS = 17242
BLOCK_LAYERS = ("0-5", "6-11", "12-17")
_MODULE_COUNTER = 0


def load_fullrun(mode="arm-lm3block", arm=None):
    """Import the real runner without creating its cloud-only contract directory."""
    global _MODULE_COUNTER
    _MODULE_COUNTER += 1
    env = {
        "HMOE_FULLRUN016_MODE": mode,
        "HMOE_FULLRUN016_VERSION": "031",
        "HMOE_JOB_NOTE": "hmoe-lionmoe-20261123-fullrun-016-v031",
        "HMOE_CONTRACT_ROOT": "/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v031",
        "EXPECTED_BUNDLE_ID": "hmoe-lionmoe-20261123-v031",
        "EXPECTED_SOURCE_COMMIT": "0" * 40,
        "EXPECTED_CARRIER_COMMIT": "0" * 40,
        "EXPECTED_PAYLOAD_SHA256": "0" * 64,
    }
    if arm is not None:
        env["HMOE_FULLRUN016_ARM"] = arm
    name = f"_test_lionmoe_fullrun016_{_MODULE_COUNTER}"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "lionmoe_fullrun_016.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
            Path, "mkdir", new=lambda self, *args, **kwargs: None):
        spec.loader.exec_module(module)
    module.npar = lambda: 1
    return module


def independent_lm3block_telemetry(start, end):
    """Fixture built directly from the colleague's spec, not via production telemetry helpers."""
    lines = []
    muon_steps = [0, 0, 0]
    for t in range(start, end + 1):
        base_p = 0.35 + (0.05 - 0.35) * (t / FULL_ITERS)
        for block, factor in enumerate((1.5, 0.75, 0.75)):
            probability = min(1.0, max(0.0, factor * base_p))
            draw = random.Random(f"lionmuon-{SEED}-{t}-block-{block}").random()
            muon = int(draw < probability)
            muon_steps[block] += muon
            optimizer_step = t - start + 1
            lines.append(f"LIONMUON_STEP_BLOCK b={block} t={t} muon={muon} muon_steps={muon_steps[block]} "
                         f"optimizer_steps={optimizer_step} p={probability:.6g} rule=prob factor={factor:.6g} "
                         f"layers={BLOCK_LAYERS[block]} power=0")
    return "\n".join(lines) + "\n"


def replace_once(test, text, old, new):
    test.assertIn(old, text)
    return text.replace(old, new, 1)


class ToyHost(torch.optim.Optimizer):
    def __init__(self, blocks):
        groups = []
        self.param_blocks = {}
        self.lr_calls = []
        self.spectral_calls = 0
        for block in blocks:
            params = [torch.nn.Parameter(torch.ones(2, 2)) for _ in range(2)]
            group = {"params": params, "lr": 0.01, "weight_decay": 0.0}
            if block is not None:
                group[LIONMUON_BLOCK_KEY] = block
            groups.append(group)
            for parameter in params:
                self.param_blocks[parameter] = block
        super().__init__(groups, {"lr": 0.01, "weight_decay": 0.0})
        self.fp32_matmul_prec = "highest"
        self._soft_scale_mode = "spectral"
        self._soft_extra_scale = 0.2
        for group in self.param_groups:
            self._init_group(group)

    def _init_group(self, group):
        for parameter in group["params"]:
            if "momentum_buffer" not in self.state[parameter]:
                self.state[parameter]["momentum_buffer"] = torch.zeros_like(parameter)

    def orthogonalize(self, parameter, direction, **kwargs):
        self.spectral_calls += 1
        return direction

    def _apply_weight_decay_inplace(self, parameter, grad, lr, weight_decay):
        self.lr_calls.append((self.param_blocks[parameter], lr))

    def pre_weight_update_fn_inplace(self, parameter, update):
        pass

    def post_weight_update_fn_inplace(self, parameter):
        pass


def make_toy(block_mode):
    cls = type("ToyLionMuon", (LionMuonMixin, ToyHost), {})
    optimizer = cls((0, 1, 2) if block_mode else (None,))
    optimizer.lion_period = [(0.0, 5)]
    optimizer.lion_prob = parse_schedule("0:0.35,1:0.05", float)
    optimizer.lion_block_factors = LIONMUON_BLOCK_FACTORS if block_mode else None
    optimizer.lion_num_layers = 18
    optimizer.lion_power = None
    optimizer.lion_total = FULL_ITERS
    optimizer.lion_seed = SEED
    optimizer.lion_base_iter = 0
    optimizer.lion_beta1 = 0.9
    optimizer.lion_beta2 = 0.99
    optimizer.lion_lr_ratio = 50
    optimizer.lion_period_experts = None
    optimizer.lion_energycal = False
    optimizer.lion_energycal_rho_ref = 0.8
    optimizer.lion_energycal_exponent = 1.0
    optimizer.lion_energycal_qmin = 0.0
    optimizer.lion_fixed_routed_alpha = 1.0
    optimizer.lion_energycal_diagnostics = False
    optimizer.lion_step_telemetry = True
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.full_like(parameter, 0.1)
    return optimizer


class ScheduleMathTests(unittest.TestCase):
    def test_linear_schedule_endpoints_and_midpoint(self):
        points = parse_schedule("0:0.35,1:0.05", float)
        self.assertAlmostEqual(piecewise_linear(points, 0.0), 0.35, places=15)
        self.assertAlmostEqual(piecewise_linear(points, 0.5), 0.20, places=15)
        self.assertAlmostEqual(piecewise_linear(points, 1.0), 0.05, places=15)
        self.assertAlmostEqual(piecewise_linear(points, 8621 / FULL_ITERS), 0.20, places=15)

    def test_uneven_block_boundaries_n_1_through_7(self):
        expected = {
            1: ((0, 1), (1, 1), (1, 1)),
            2: ((0, 1), (1, 2), (2, 2)),
            3: ((0, 1), (1, 2), (2, 3)),
            4: ((0, 2), (2, 3), (3, 4)),
            5: ((0, 2), (2, 4), (4, 5)),
            6: ((0, 2), (2, 4), (4, 6)),
            7: ((0, 3), (3, 5), (5, 7)),
        }
        for count, boundaries in expected.items():
            with self.subTest(count=count):
                self.assertEqual(lionmuon_block_boundaries(count), boundaries)
                assignments = [lionmuon_block_for_layer(layer, count) for layer in range(count)]
                for block, (start, end) in enumerate(boundaries):
                    self.assertEqual(assignments[start:end], [block] * (end - start))
                self.assertEqual(len(assignments), count)
        self.assertEqual(lionmuon_block_boundaries(18), ((0, 6), (6, 12), (12, 18)))
        self.assertEqual([lionmuon_block_layers_label(block, 18) for block in range(3)],
                         list(BLOCK_LAYERS))

    def test_stable_megatron_names_map_to_depth_blocks(self):
        for layer in range(18):
            name = f"model_chunk0.decoder.layers.{layer}.mlp.experts.linear_fc2.weight0"
            self.assertEqual(lionmuon_block_for_name(name, 18), layer // 6)
        self.assertIsNone(lionmuon_block_for_name("model_chunk0.embedding.word_embeddings.weight", 18))
        self.assertIsNone(lionmuon_block_for_name("model_chunk0.output_layer.weight", 18))

    def test_mean_factor_conservation_and_parser(self):
        self.assertEqual(sum(LIONMUON_BLOCK_FACTORS), 3.0)
        self.assertEqual(sum(LIONMUON_BLOCK_FACTORS) / 3, 1.0)
        self.assertEqual(parse_lionmuon_block_factors("1.5_0.75_0.75"), LIONMUON_BLOCK_FACTORS)
        with self.assertRaisesRegex(ValueError, "mean one"):
            parse_lionmuon_block_factors("1.25,0.75,0.75")

    def test_probability_clamping(self):
        self.assertEqual(lionmuon_block_probability(0.8, 1.5), 1.0)
        self.assertEqual(lionmuon_block_probability(0.8, 0.75), 0.6)
        self.assertEqual(lionmuon_block_probability(0.2, -0.5), 0.0)
        self.assertEqual(lionmuon_block_probability(-0.1, 1.5), 0.0)

    def test_draw_determinism_and_block_independence(self):
        global_state = random.getstate()
        for block in range(3):
            probability = 0.4
            key = f"lionmuon-{SEED}-17-block-{block}"
            expected = random.Random(key).random() < probability
            self.assertEqual(lionmuon_block_draw(SEED, 17, block, probability), expected)
            self.assertEqual(lionmuon_block_draw(SEED, 17, block, probability), expected)
        self.assertEqual(random.getstate(), global_state)
        sequences = {
            block: tuple(lionmuon_block_draw(SEED, t, block, 0.5) for t in range(1, 65))
            for block in range(3)
        }
        self.assertNotEqual(sequences[0], sequences[1])
        self.assertNotEqual(sequences[0], sequences[2])
        self.assertNotEqual(sequences[1], sequences[2])
        other_seed_expected = random.Random(f"lionmuon-{SEED + 1}-17-block-0").random() < 0.4
        self.assertEqual(lionmuon_block_draw(SEED + 1, 17, 0, 0.4), other_seed_expected)

    def test_telemetry_format(self):
        self.assertEqual(
            lionmuon_block_telemetry(0, 1, True, 0.525, 1.5, 18, 1, 1),
            "LIONMUON_STEP_BLOCK b=0 t=1 muon=1 muon_steps=1 optimizer_steps=1 p=0.525 "
            "rule=prob factor=1.5 layers=0-5 power=0",
        )
        self.assertEqual(
            lionmuon_block_telemetry(2, 17242, False, 0.0375, 0.75, 18, 0, 17242),
            "LIONMUON_STEP_BLOCK b=2 t=17242 muon=0 muon_steps=0 optimizer_steps=17242 p=0.0375 "
            "rule=prob factor=0.75 layers=12-17 power=0",
        )
        self.assertEqual(lionmuon_block_telemetry(2, 1, False, 0.1, 0.75, 1, 0, 1),
                         "LIONMUON_STEP_BLOCK b=2 t=1 muon=0 muon_steps=0 optimizer_steps=1 p=0.1 "
                         "rule=prob factor=0.75 layers=empty power=0")


class OptimizerBehaviorTests(unittest.TestCase):
    def test_block_step_shares_draw_and_emits_three_lines(self):
        optimizer = make_toy(block_mode=True)
        output = io.StringIO()
        with mock.patch.object(lionmuon, "megatron_iteration", lambda: 0), \
                mock.patch.object(lionmuon, "_telemetry_rank0", lambda: True), \
                contextlib.redirect_stdout(output):
            optimizer.step()
        base_p = 0.35 + (0.05 - 0.35) / FULL_ITERS
        expected_lines = []
        expected_muon_total = 0
        for block, factor in enumerate(LIONMUON_BLOCK_FACTORS):
            probability = min(1.0, factor * base_p)
            expected = random.Random(f"lionmuon-{SEED}-1-block-{block}").random() < probability
            expected_muon_total += int(expected)
            expected_lr = 0.01 if expected else 0.01 / 50
            calls = [lr for called_block, lr in optimizer.lr_calls if called_block == block]
            self.assertEqual(calls, [expected_lr, expected_lr])
            expected_lines.append(lionmuon_block_telemetry(block, 1, expected, probability, factor, 18,
                                                            int(expected), 1))
        lines = output.getvalue().splitlines()
        self.assertEqual([line for line in lines if line.startswith("LIONMUON_STEP_BLOCK")], expected_lines)
        self.assertFalse(any(line.startswith("LIONMUON_STEP t=") for line in lines))
        self.assertEqual(optimizer.spectral_calls, 2 * expected_muon_total)
        self.assertEqual(optimizer._lionmuon_spectral_updates, 2 * expected_muon_total)
        self.assertEqual(optimizer._lionmuon_lion_updates, 2 * (3 - expected_muon_total))

    def test_legacy_global_probability_step_and_telemetry_unchanged(self):
        optimizer = make_toy(block_mode=False)
        output = io.StringIO()
        with mock.patch.object(lionmuon, "megatron_iteration", lambda: 0), \
                mock.patch.object(lionmuon, "_telemetry_rank0", lambda: True), \
                contextlib.redirect_stdout(output):
            optimizer.step()
        probability = 0.35 + (0.05 - 0.35) / FULL_ITERS
        expected = random.Random(f"lionmuon-{SEED}-1").random() < probability
        expected_lr = 0.01 if expected else 0.01 / 50
        self.assertEqual([lr for _block, lr in optimizer.lr_calls], [expected_lr, expected_lr])
        expected_line = (f"LIONMUON_STEP t=1 muon={int(expected)} muon_steps={int(expected)} optimizer_steps=1 "
                         f"rule=prob value={probability:.6g} power=0")
        self.assertIn(expected_line, output.getvalue().splitlines())
        self.assertEqual(optimizer._lionmuon_spectral_updates, 2 * int(expected))
        self.assertEqual(optimizer._lionmuon_lion_updates, 2 * (1 - int(expected)))


class FullrunCheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fullrun = load_fullrun()

    def test_arm_environment_and_existing_arm_isolation(self):
        with mock.patch.dict(os.environ, {"STAGE3_MOE_LIONMUON_ENERGYCAL": "1"}, clear=False):
            env, _removed = self.fullrun.base_arm_env("unit-lm3block")
        self.assertEqual(self.fullrun.ARM, "lm3block")
        self.assertEqual(env["STAGE3_MOE_LIONMUON_PROB"], "0:0.35,1:0.05")
        self.assertEqual(env["STAGE3_MOE_LIONMUON_BLOCK_FACTORS"], "1.5,0.75,0.75")
        self.assertEqual(env["STAGE3_MOE_LIONMUON_NUM_LAYERS"], "18")
        self.assertEqual(env["STAGE3_MOE_LIONMUON_LR_RATIO"], "50")
        self.assertEqual(env["STAGE3_MOE_LIONMUON_BETA1"], "0.9")
        self.assertEqual(env["STAGE3_MOE_LIONMUON_BETA2"], "0.99")
        self.assertEqual(env["STAGE3_MOE_EP"], "1")
        self.assertNotIn("STAGE3_MOE_LIONMUON_ENERGYCAL", env)

        linear = load_fullrun(mode="arm-lmlinrev")
        linear_env, _ = linear.base_arm_env("unit-linear")
        self.assertEqual(linear_env["STAGE3_MOE_LIONMUON_ENERGYCAL"], "0")
        self.assertEqual(linear_env["STAGE3_MOE_LIONMUON_PROB"], "0:0.35,1:0.05")
        self.assertNotIn("STAGE3_MOE_LIONMUON_BLOCK_FACTORS", linear_env)
        control = load_fullrun(mode="gate-save", arm="lm5r50ec")
        control_env, _ = control.base_arm_env("unit-control")
        self.assertEqual(control_env["STAGE3_MOE_LIONMUON_PERIOD"], "5")
        self.assertEqual(control_env["STAGE3_MOE_LIONMUON_ENERGYCAL"], "1")
        self.assertNotIn("STAGE3_MOE_LIONMUON_BLOCK_FACTORS", control_env)

    def test_checker_accepts_gate_segment(self):
        result = self.fullrun.check_realized_schedule(independent_lm3block_telemetry(1, 50), "lm3block", 1, 50)
        self.assertEqual(result["telemetry_steps"], 50)
        self.assertEqual(result["telemetry_lines"], 150)
        self.assertEqual(result["t_range"], [1, 50])
        self.assertEqual(result["block_layers"], list(BLOCK_LAYERS))
        self.assertFalse(result["full_schedule_rate_gate"])
        self.assertGreater(result["rate_target"], 0.34)

    def test_checker_accepts_full_schedule_and_enforces_rate_gate(self):
        log_text = independent_lm3block_telemetry(1, FULL_ITERS)
        result = self.fullrun.check_realized_schedule(log_text, "lm3block", 1, FULL_ITERS)
        self.assertTrue(result["full_schedule_rate_gate"])
        self.assertEqual(result["rate_target"], 0.2)
        self.assertEqual(result["rate_tolerance"], 0.02)
        self.assertLessEqual(abs(result["realized_overall_muon_rate"] - 0.2), 0.02)
        print("LM3BLOCK_FULL_SCHEDULE " + json.dumps(result, sort_keys=True), flush=True)
        with mock.patch.object(self.fullrun, "LM3BLOCK_RATE_TARGET", 0.9):
            with self.assertRaisesRegex(RuntimeError, "overall Muon rate"):
                self.fullrun.check_realized_schedule(log_text, "lm3block", 1, FULL_ITERS)

    def test_checker_rejects_wrong_factor_probability_draw_and_layers(self):
        valid = independent_lm3block_telemetry(1, 4)
        wrong_probability = re.sub(r"p=[0-9.eE+-]+", "p=0.123", valid, count=1)
        self.assertNotEqual(wrong_probability, valid)
        cases = {
            "wrong factor": (replace_once(self, valid, "factor=1.5", "factor=1.25"), "block factor"),
            "wrong probability": (wrong_probability, "probability"),
            "wrong layers": (replace_once(self, valid, "layers=0-5", "layers=0-4"), "layer split"),
        }
        first_muon = re.search(r"muon=(\d)", valid).group(1)
        cases["wrong draw"] = (replace_once(self, valid, f"muon={first_muon}",
                                            f"muon={1 - int(first_muon)}"), "deterministic draw")
        for name, (log_text, message) in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, message):
                    self.fullrun.check_realized_schedule(log_text, "lm3block", 1, 4)

    def test_checker_rejects_missing_duplicate_and_noncontiguous_lines(self):
        valid = independent_lm3block_telemetry(1, 4)
        lines = valid.splitlines()
        missing_block = "\n".join(line for line in lines if not ("t=1 " in line and "b=1 " in line)) + "\n"
        with self.assertRaisesRegex(RuntimeError, "cover exactly"):
            self.fullrun.check_realized_schedule(missing_block, "lm3block", 1, 4)
        missing_step = "\n".join(line for line in lines if "t=2 " not in line) + "\n"
        with self.assertRaisesRegex(RuntimeError, "cover exactly"):
            self.fullrun.check_realized_schedule(missing_step, "lm3block", 1, 4)
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            self.fullrun.check_realized_schedule(lines[0] + "\n" + valid, "lm3block", 1, 4)
        with self.assertRaisesRegex(RuntimeError, "No LIONMUON_STEP_BLOCK"):
            self.fullrun.check_realized_schedule("", "lm3block", 1, 4)

    def test_checker_accepts_resume_boundaries(self):
        for start, end in ((51, 60), (2179, 2180)):
            with self.subTest(start=start, end=end):
                result = self.fullrun.check_realized_schedule(
                    independent_lm3block_telemetry(start, end), "lm3block", start, end)
                self.assertEqual(result["t_range"], [start, end])
                self.assertEqual(result["telemetry_steps"], end - start + 1)

    def test_checker_rejects_wrong_counters_power_and_unexpected_range(self):
        valid = independent_lm3block_telemetry(1, 4)
        wrong_msteps = re.sub(r"muon_steps=\d+", "muon_steps=99", valid, count=1)
        wrong_osteps = replace_once(self, valid, "optimizer_steps=1", "optimizer_steps=2")
        wrong_power = replace_once(self, valid, "power=0", "power=1")
        for name, log_text, message in (
                ("muon counter", wrong_msteps, "muon_steps increment"),
                ("optimizer counter", wrong_osteps, "optimizer_steps/global-t"),
                ("power", wrong_power, "POWER")):
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, message):
                    self.fullrun.check_realized_schedule(log_text, "lm3block", 1, 4)
        with self.assertRaisesRegex(RuntimeError, "mismatch|cover exactly|row count"):
            self.fullrun.check_realized_schedule(valid, "lm3block", 2, 4)

    def test_legacy_checker_rejects_wrong_counters_power_and_range(self):
        lines = []
        muon_steps = 0
        for t in (1, 2):
            probability = 0.35 - 0.30 * t / FULL_ITERS
            muon = int(random.Random(f"lionmuon-{SEED}-{t}").random() < probability)
            muon_steps += muon
            lines.append(f"LIONMUON_STEP t={t} muon={muon} muon_steps={muon_steps} optimizer_steps={t} "
                         f"rule=prob value={probability:.6g} power=0")
        valid = "\n".join(lines) + "\n"
        cases = (
            (re.sub(r"muon_steps=\d+", "muon_steps=99", valid, count=1), "muon_steps increment"),
            (replace_once(self, valid, "optimizer_steps=1", "optimizer_steps=2"), "optimizer_steps/global-t"),
            (replace_once(self, valid, "power=0", "power=1"), "POWER"),
        )
        for log_text, message in cases:
            with self.assertRaisesRegex(RuntimeError, message):
                self.fullrun.check_realized_schedule(log_text, "lmlinrevr50", 1, 2)
        with self.assertRaisesRegex(RuntimeError, "row count|mismatch|cover exactly"):
            self.fullrun.check_realized_schedule(valid, "lmlinrevr50", 2, 2)

    def test_print_independent_golden_oracle_values(self):
        goldens = {}
        for t in (1, 51, 2179, 17242):
            probability = 0.35 - 0.30 * t / FULL_ITERS
            global_muon = int(random.Random(f"lionmuon-{SEED}-{t}").random() < probability)
            blocks = []
            for block, factor in enumerate((1.5, 0.75, 0.75)):
                block_p = min(1.0, max(0.0, factor * probability))
                block_muon = int(random.Random(f"lionmuon-{SEED}-{t}-block-{block}").random() < block_p)
                blocks.append((block_p, block_muon))
            goldens[t] = {"p": probability, "global_muon": global_muon, "blocks": blocks}
        print("LM3BLOCK_INDEPENDENT_GOLDENS " + json.dumps(goldens, sort_keys=True), flush=True)

    def test_legacy_checker_paths_still_accept(self):
        period_lines = []
        probability_lines = []
        points = parse_schedule("0:0.35,1:0.05", float)
        period_count = 0
        probability_count = 0
        for t in range(1, 6):
            period_muon = int((t - 1) % 5 == 0)
            period_count += period_muon
            period_lines.append(f"LIONMUON_STEP t={t} muon={period_muon} muon_steps={period_count} "
                                f"optimizer_steps={t} rule=period value=5 power=0")
            probability = piecewise_linear(points, t / FULL_ITERS)
            probability_muon = int(random.Random(f"lionmuon-{SEED}-{t}").random() < probability)
            probability_count += probability_muon
            probability_lines.append(f"LIONMUON_STEP t={t} muon={probability_muon} "
                                     f"muon_steps={probability_count} optimizer_steps={t} "
                                     f"rule=prob value={probability:.6g} power=0")
        period = self.fullrun.check_realized_schedule("\n".join(period_lines) + "\n", "lm5r50ec", 1, 5)
        linear = self.fullrun.check_realized_schedule("\n".join(probability_lines) + "\n", "lmlinrevr50", 1, 5)
        self.assertEqual(period["rule"], "period5")
        self.assertEqual(linear["rule"], "prob_0.35_to_0.05")


if __name__ == "__main__":
    unittest.main(verbosity=2)
