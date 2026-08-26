#!/usr/bin/env python3

import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]
training_path = ROOT / "third_party/Megatron-LM/megatron/training/training.py"
tree = ast.parse(training_path.read_text(encoding="utf-8"))
function = next(
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "get_train_valid_test_num_samples"
)


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
namespace = {"get_args": lambda: args}
module = ast.Module(body=[function], type_ignores=[])
exec(compile(module, str(training_path), "exec"), namespace)
train_samples, _, _ = namespace["get_train_valid_test_num_samples"]()

expected = (22_208 - 13_794) * 208
if train_samples != expected:
    raise RuntimeError(f"phase samples are {train_samples}, expected {expected}")
print(f"phase_transition=pass local_consumed_samples=0 phase_samples={train_samples}")
