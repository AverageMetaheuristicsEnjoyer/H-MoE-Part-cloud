"""Passive full-development evaluation capture (baseline-eval-030).

The stock MCore evaluator prints only the aggregate loss; the exact valid-token denominator
and the per-iteration losses stay inside evaluate(). This module wraps the forward-backward
function factory that evaluate() uses and records the [loss_sum, num_tokens] reports it
already returns -- record-only: no math, dtype, return-value or control-flow change, and it
is installed only when STAGE3_MOE_FULL_DEV_EVAL=1 (the eval-full-dev launcher mode).

With --full-validation the sampler yields one sample per batch and the protocol pins
--eval-global-batch-size 1 / --eval-micro-batch-size 1, so each captured record is exactly
one evaluation iteration / one development sample (DP=TP=PP=1). The runner sums the records
into the evaluator's own numerator/denominator and cross-checks the ratio against the printed
aggregate. A capture failure is logged and never breaks the evaluation itself.
"""
import atexit
import json
import time
from pathlib import Path

import torch


def install(capture_path):
    import megatron.training.training as training

    path = Path(capture_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original_factory = training.get_forward_backward_func
    state = {"records": 0, "installed_at": time.perf_counter(), "path": str(path), "errors": 0}
    stream = path.open("w")
    atexit.register(stream.close)

    def factory():
        forward_backward = original_factory()

        def capturing(*args, **kwargs):
            result = forward_backward(*args, **kwargs)
            try:
                if isinstance(result, list):
                    for index, loss_dict in enumerate(result):
                        if not isinstance(loss_dict, dict):
                            continue
                        for key, value in loss_dict.items():
                            if torch.is_tensor(value) and value.numel() == 2:
                                pair = value.detach().view(-1).to(device="cpu", dtype=torch.float64)
                                stream.write(json.dumps({"record": state["records"], "microbatch": index,
                                                         "key": key, "loss_sum": float(pair[0]),
                                                         "num_tokens": float(pair[1])}, sort_keys=True) + "\n")
                                state["records"] += 1
                    stream.flush()
            except Exception as error:
                state["errors"] += 1
                print(f"FULL_DEV_CAPTURE_ERROR {type(error).__name__}: {error}", flush=True)
            return result

        return capturing

    training.get_forward_backward_func = factory
    print(f"FULL_DEV_CAPTURE installed path={path}", flush=True)
    return state
