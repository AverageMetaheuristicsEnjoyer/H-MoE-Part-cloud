#!/usr/bin/env python3

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "Megatron-LM"))

from megatron.training import training


args = SimpleNamespace(
    train_samples=None,
    train_iters=22_208,
    global_batch_size=208,
    full_validation=False,
    skip_train=False,
    eval_interval=250,
    start_eval_at_iter=None,
    eval_iters=32,
    phase_transition_iterations=[13_794],
    iteration=13_794,
)
with patch.object(training, "get_args", return_value=args):
    train_samples, _, _ = training.get_train_valid_test_num_samples()

expected = (22_208 - 13_794) * 208
if train_samples != expected:
    raise RuntimeError(f"phase samples are {train_samples}, expected {expected}")
print(f"phase_transition=pass local_consumed_samples=0 phase_samples={train_samples}")
