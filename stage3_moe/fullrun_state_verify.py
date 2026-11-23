"""Full-state save/load verification for fullrun-016 (passive, gated; no optimizer math change).

Two hook points, both installed only for fullrun-016 segments:
  * save-time dump (STAGE3_MOE_FULLRUN_STATE_DUMP=1): wraps training.save_checkpoint_and_time and,
    AFTER the save completes, hashes the in-memory state that was just saved (BF16 model params,
    FP32 masters, every raw optimizer-state tensor incl. Muon momentum and Adam moments), records
    scheduler state, RNG states, iteration/consumed samples, param-group structure and the next
    LionMuon P5/PROB phase. The manifest is written next to the checkpoint so the runner uploads it
    with the checkpoint files.
  * load-time verify (STAGE3_MOE_FULLRUN_VERIFY_MANIFEST=<path>): wraps setup_model_and_optimizer
    (MCore loads the checkpoint inside it, training.py:2045) and recomputes the same manifest from
    the LOADED state, comparing it to the fetched save-time manifest BEFORE any new update. Any
    mismatch raises -- the gate cannot pass silently. It also writes loaded-state-manifest.json.

Hashing is raw-bytes SHA256 per tensor (dtype/shape recorded); it reads tensors only and never
mutates state, dtypes, RNG or control flow. Param-group collapse is checked structurally (group
count + per-group param counts + identifier keys); MCore's _filter_and_reorder_param_groups already
keeps same-key groups in checkpoint order without aliasing (the Muon fix), verified statically.
"""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


def _tensor_record(name, tensor):
    detached = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()), "sha256": hashlib.sha256(detached).hexdigest()}


def _raw_optimizers(optimizer):
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained:
        return [wrapped.optimizer if hasattr(wrapped, "optimizer") else wrapped for wrapped in chained]
    return [optimizer]


def _mixed_wrappers(optimizer):
    chained = getattr(optimizer, "chained_optimizers", None)
    return list(chained) if chained else [optimizer]


def collect_state(model, optimizer, scheduler, iteration_override=None, flops_override=None):
    from megatron.training import get_args
    args = get_args()
    model_records = []
    for chunk_index, chunk in enumerate(model):
        for name, parameter in chunk.named_parameters():
            model_records.append(_tensor_record(f"model_chunk{chunk_index}.{name}", parameter))
        # Buffers (e.g. router expert_bias) are part of the saved model state; hash them too
        # so pre-update verify covers router bias, not only parameters (skeptic v031 M3).
        for name, buffer in chunk.named_buffers():
            model_records.append(_tensor_record(f"model_chunk{chunk_index}.buffer.{name}", buffer))
    master_records = []
    state_records = []
    group_structure = []
    for wrapper in _mixed_wrappers(optimizer):
        for group_index, group in enumerate(getattr(wrapper, "fp32_from_float16_groups", []) or []):
            for param_index, master in enumerate(group):
                master_records.append(_tensor_record(f"master.w{group_index}.p{param_index}", master))
        raw = wrapper.optimizer if hasattr(wrapper, "optimizer") else wrapper
        # Key state tensors by stable (group, param) position, NOT dict enumeration order: the
        # insertion order of raw.state can differ between save and load, which would otherwise
        # cause a spurious gate mismatch. Param-group order is preserved across load (the Muon
        # _filter_and_reorder_param_groups fix), so this naming is stable.
        raw_state = getattr(raw, "state", {})
        for group_index, group in enumerate(getattr(raw, "param_groups", []) or []):
            for param_index, parameter in enumerate(group.get("params", [])):
                for key, value in raw_state.get(parameter, {}).items():
                    if torch.is_tensor(value):
                        state_records.append(_tensor_record(f"optstate.g{group_index}.p{param_index}.{key}", value))
        for group_index, group in enumerate(getattr(raw, "param_groups", []) or []):
            group_structure.append({"group": group_index, "num_params": len(group.get("params", [])),
                                    "lr": group.get("lr"), "weight_decay": group.get("weight_decay"),
                                    "wd_mult": group.get("wd_mult"), "lr_mult": group.get("lr_mult"),
                                    "is_expert_parallel": group.get("is_expert_parallel"),
                                    "is_decoupled_lr": group.get("is_decoupled_lr"),
                                    "stage3_routed_expert": group.get("stage3_routed_expert"),
                                    "stage3_split_swiglu_fc1": group.get("stage3_split_swiglu_fc1"),
                                    "lionmuon_energycal": group.get("lionmuon_energycal")})
    scheduler_state = scheduler.state_dict() if scheduler is not None else None
    rng = {"torch_rng_state_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
           "numpy_rng_state_sha256": hashlib.sha256(json.dumps(np.random.get_state(), default=str).encode()).hexdigest(),
           "python_rng_state_sha256": hashlib.sha256(json.dumps(random.getstate(), default=str).encode()).hexdigest()}
    try:
        from megatron.core import tensor_parallel
        tracker_states = tensor_parallel.get_cuda_rng_tracker().get_states()
        rng["cuda_rng_tracker"] = {name: hashlib.sha256(state.numpy().tobytes()).hexdigest()
                                   for name, state in tracker_states.items()}
    except Exception as error:
        rng["cuda_rng_tracker_error"] = str(error)
    # At SAVE time args.iteration / args.num_floating_point_operations_so_far are STALE (MCore's
    # train() loop tracks local iteration/num_flops and passes them to save_checkpoint_and_time as
    # arguments; args.* are only set at load/setup). The save-dump must capture the SAME values MCore
    # writes to the checkpoint, so saving_with_dump passes them explicitly. At LOAD time args.* are
    # restored from the checkpoint and the overrides are None. (gate-restore diagnosed 3 mismatches:
    # iteration 0vs50, num_flops 0vs3.39e16, next_step_phase next_t 1vs51 — all stale-args artifacts;
    # model/masters/optimizer/scheduler/RNG/consumed hashes matched exactly.)
    iteration = int(iteration_override) if iteration_override is not None else int(getattr(args, "iteration", 0))
    next_t = iteration + 1
    lion = next((raw for raw in _raw_optimizers(optimizer) if hasattr(raw, "lion_period")), None)
    phase = {"next_t": next_t}
    if lion is not None:
        if getattr(lion, "lion_prob", None) is not None:
            from stage3_moe.lionmuon import piecewise_linear
            p = piecewise_linear(lion.lion_prob, next_t / lion.lion_total)
            phase.update({"rule": "prob", "p": p,
                          "expected_muon": random.Random(f"lionmuon-{lion.lion_seed}-{next_t}").random() < p})
        else:
            from stage3_moe.lionmuon import piecewise_constant
            period = piecewise_constant(lion.lion_period, next_t / lion.lion_total)
            phase.update({"rule": "period", "period": period,
                          "expected_muon": (next_t - 1 - lion.lion_base_iter) % period == 0})
    return {"iteration": iteration, "consumed_train_samples": int(getattr(args, "consumed_train_samples", 0)),
            "seed": int(getattr(args, "seed", 0)),
            "num_floating_point_operations_so_far": int(flops_override) if flops_override is not None else int(getattr(args, "num_floating_point_operations_so_far", 0)),
            "scheduler_state": scheduler_state, "rng": rng, "next_step_phase": phase,
            "param_groups": group_structure,
            "model_param_hashes": model_records, "fp32_master_hashes": master_records,
            "optimizer_state_hashes": state_records,
            "counts": {"model_params": len(model_records), "masters": len(master_records),
                       "optimizer_state_tensors": len(state_records)}}


def _compare(expected, actual):
    mismatches = []
    for key in ("iteration", "consumed_train_samples", "seed", "num_floating_point_operations_so_far",
                "scheduler_state", "rng", "next_step_phase", "param_groups", "counts"):
        if expected.get(key) != actual.get(key):
            mismatches.append({"field": key, "expected": expected.get(key), "actual": actual.get(key)})
    for key in ("model_param_hashes", "fp32_master_hashes", "optimizer_state_hashes"):
        exp = {item["name"]: item["sha256"] for item in expected.get(key, [])}
        act = {item["name"]: item["sha256"] for item in actual.get(key, [])}
        if exp.keys() != act.keys():
            mismatches.append({"field": key + ".names", "missing": sorted(exp.keys() - act.keys())[:20],
                               "extra": sorted(act.keys() - exp.keys())[:20]})
        for name in sorted(exp.keys() & act.keys()):
            if exp[name] != act[name]:
                mismatches.append({"field": key, "name": name, "expected_sha256": exp[name], "actual_sha256": act[name]})
    return mismatches


def install():
    import megatron.training.training as training

    dump_enabled = os.environ.get("STAGE3_MOE_FULLRUN_STATE_DUMP") == "1"
    verify_path = os.environ.get("STAGE3_MOE_FULLRUN_VERIFY_MANIFEST")
    dump_dir = os.environ.get("STAGE3_MOE_FULLRUN_DUMP_DIR")
    loaded_path = os.environ.get("STAGE3_MOE_FULLRUN_LOADED_MANIFEST_PATH")

    if dump_enabled and dump_dir:
        original_save = training.save_checkpoint_and_time

        def saving_with_dump(iteration, model, optimizer, scheduler, *args, **kwargs):
            # save_checkpoint_and_time(iteration, model, optimizer, scheduler,
            #   num_floating_point_operations_so_far, checkpointing_context, ...): capture the
            # exact iteration/num_flops MCore saves (args.* are stale during the train loop).
            flops = args[0] if args else kwargs.get("num_floating_point_operations_so_far")
            result = original_save(iteration, model, optimizer, scheduler, *args, **kwargs)
            try:
                manifest = collect_state(model, optimizer, scheduler,
                                         iteration_override=iteration, flops_override=flops)
                target = Path(dump_dir) / f"iter_{int(iteration):07d}" / "state-manifest.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".tmp")
                with tmp.open("w") as stream:
                    stream.write(json.dumps(manifest, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(tmp, target)
                print(f"FULLRUN_STATE_DUMP iteration={iteration} path={target} "
                      f"counts={manifest['counts']}", flush=True)
            except Exception as error:
                print(f"FULLRUN_STATE_DUMP_ERROR {type(error).__name__}: {error}", flush=True)
                raise
            return result

        training.save_checkpoint_and_time = saving_with_dump

    if verify_path or loaded_path:
        original_setup = training.setup_model_and_optimizer

        def setup_with_verify(*args, **kwargs):
            model, optimizer, scheduler = original_setup(*args, **kwargs)
            actual = collect_state(model, optimizer, scheduler)
            if loaded_path:
                Path(loaded_path).parent.mkdir(parents=True, exist_ok=True)
                with Path(loaded_path).open("w") as stream:
                    stream.write(json.dumps(actual, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            if verify_path:
                expected = json.loads(Path(verify_path).read_text())
                mismatches = _compare(expected, actual)
                report = {"verified": not mismatches, "mismatches": mismatches[:200],
                          "mismatch_count": len(mismatches),
                          "expected_iteration": expected.get("iteration"), "actual_iteration": actual.get("iteration")}
                report_path = Path(verify_path).parent / "gate-restore-verification.json"
                with report_path.open("w") as stream:
                    stream.write(json.dumps(report, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                if mismatches:
                    raise RuntimeError(f"FULLRUN gate-restore state mismatch BEFORE updates: {len(mismatches)} fields; "
                                       f"see {report_path}")
                print(f"FULLRUN_GATE_RESTORE_VERIFIED iteration={actual['iteration']} "
                      f"consumed={actual['consumed_train_samples']} phase={actual['next_step_phase']}", flush=True)
            return model, optimizer, scheduler

        training.setup_model_and_optimizer = setup_with_verify
