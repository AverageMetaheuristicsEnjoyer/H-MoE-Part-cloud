"""Find what a checkpoint save leaves behind in the FP8 arms. Off unless asked for.

The FP8 states are the only tensors a step replaces instead of updating in place,
so anything that outlives the save while still holding them pins a second copy.
This records whether that happens, whether a collection releases it (a cycle) and,
if it survives even that, what still refers to the pinned tensors.
"""

import gc
import os
import types
import weakref

import torch


ENABLED = os.environ.get("STAGE3_MOE_MEM_AUDIT", "0") == "1"
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
STEPS_WATCHED = 3

_watched = []
_steps_left = 0


def _is_fp8(obj):
    # gc.get_objects() hands back the whole heap, including objects whose
    # attribute access raises.
    try:
        return isinstance(obj, torch.Tensor) and obj.dtype in FP8_DTYPES and obj.is_cuda
    except Exception:
        return False


def _live_fp8_tensors():
    return [obj for obj in gc.get_objects() if _is_fp8(obj)]


def _distinct_bytes(tensors):
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values()), len(storages)


def _report(label):
    live_bytes, live_count = _distinct_bytes(_live_fp8_tensors())
    print(
        f"MEMAUDIT {label}"
        f" allocated={torch.cuda.memory_allocated()}"
        f" max_allocated={torch.cuda.max_memory_allocated()}"
        f" fp8_bytes={live_bytes} fp8_storages={live_count}",
        flush=True,
    )


def _describe(referrer):
    detail = type(referrer).__name__
    if isinstance(referrer, dict):
        keys = [key for key in referrer if isinstance(key, str)][:8]
        return f"{detail} len={len(referrer)} str_keys={keys}"
    if isinstance(referrer, (list, tuple, set)):
        return f"{detail} len={len(referrer)}"
    if isinstance(referrer, types.FrameType):
        code = referrer.f_code
        return f"{detail} {code.co_filename}:{referrer.f_lineno} in {code.co_name}"
    return detail


def _name_holders(tensor, ours):
    referrers = [r for r in gc.get_referrers(tensor) if id(r) not in ours]
    print(f"MEMAUDIT holders count={len(referrers)}", flush=True)
    for index, referrer in enumerate(referrers[:6]):
        print(f"MEMAUDIT holder[{index}] {_describe(referrer)}", flush=True)
        for outer in gc.get_referrers(referrer)[:3]:
            if id(outer) in ours:
                continue
            print(f"MEMAUDIT holder[{index}].referrer {_describe(outer)}", flush=True)


def _tick():
    global _steps_left
    if _steps_left <= 0:
        return
    _steps_left -= 1
    alive = [reference() for reference in _watched]
    alive = [tensor for tensor in alive if tensor is not None]
    pinned_bytes, pinned_count = _distinct_bytes(alive)
    print(
        f"MEMAUDIT after_save_step steps_left={_steps_left}"
        f" pinned_tensors={len(alive)} pinned_storages={pinned_count}"
        f" pinned_bytes={pinned_bytes}"
        f" allocated={torch.cuda.memory_allocated()}"
        f" max_allocated={torch.cuda.max_memory_allocated()}",
        flush=True,
    )
    if alive and _steps_left == 0:
        _name_holders(alive[0], {id(alive), id(_watched)})


def install():
    """Wrap the save and the training step so the audit brackets both."""
    if not ENABLED:
        return
    import megatron.training.training as training

    original_save = training.save_checkpoint
    original_train_step = training.train_step

    def leader():
        # Distributed is not up yet when the audit is installed.
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    def audited_save(*args, **kwargs):
        global _watched, _steps_left
        if not leader():
            return original_save(*args, **kwargs)
        iteration = args[0]
        _report(f"save iteration={iteration} phase=before")
        before = _live_fp8_tensors()
        result = original_save(*args, **kwargs)
        _report(f"save iteration={iteration} phase=after")
        collected = gc.collect()
        _report(f"save iteration={iteration} phase=after_gc collected={collected}")
        # Every one of these is replaced by the next step, so any that outlive it
        # are pinned by something the save left behind.
        _watched = [weakref.ref(tensor) for tensor in before]
        _steps_left = STEPS_WATCHED
        del before
        return result

    def audited_train_step(*args, **kwargs):
        result = original_train_step(*args, **kwargs)
        if leader():
            _tick()
        return result

    training.save_checkpoint = audited_save
    training.train_step = audited_train_step
    print("MEMAUDIT installed", flush=True)
