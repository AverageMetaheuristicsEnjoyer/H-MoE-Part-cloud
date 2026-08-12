import atexit
import ctypes
import hashlib
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from importlib import metadata
from pathlib import Path

import torch

from stage3_moe import (
    ACTIVE_PARAMETERS,
    ADAMW_FALLBACK_PARAMETERS,
    MUON_MATRIX_PARAMETERS,
    TOTAL_PARAMETERS,
)


FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}
METADATA_PREFIXES = ("scale_", "expand_", "sqrt_minmax_")


def _raw_optimizers(optimizer):
    for wrapped in getattr(optimizer, "chained_optimizers", [optimizer]):
        raw = getattr(wrapped, "optimizer", wrapped)
        if raw is not None:
            yield raw


def assert_fp8_adam_bootstrap(optimizer):
    adam_optimizers = [raw for raw in _raw_optimizers(optimizer) if _role(raw) != "muon_matrix"]
    if not adam_optimizers or any(type(raw).__name__ != "FP8StateAdamW" for raw in adam_optimizers):
        classes = [type(raw).__name__ for raw in adam_optimizers]
        raise AssertionError(f"FP8 Adam bootstrap failed; raw classes={classes}")


def _role(raw):
    state_specs = getattr(raw, "state_specs", ())
    keys = {spec.name if hasattr(spec, "name") else spec[0] for spec in state_specs}
    if "momentum_buffer" in keys or "Muon" in type(raw).__name__:
        return "muon_matrix"
    return "adamw_fallback"


def _storage_key(tensor):
    storage = tensor.untyped_storage()
    return (tensor.device.type, tensor.device.index, storage.data_ptr(), storage.nbytes())


def optimizer_state_ledger(optimizer, arm):
    rows = defaultdict(lambda: {"numel": 0, "bytes": 0, "quantized": False})
    data_bytes = 0
    metadata_bytes = 0
    seen = set()
    raw_optimizers = list(_raw_optimizers(optimizer))
    adam_only = arm.startswith("adamw_")

    for raw in raw_optimizers:
        role = "adamw_all" if adam_only else _role(raw)
        for state in raw.state.values():
            for key, value in state.items():
                if not torch.is_tensor(value):
                    continue
                storage_key = _storage_key(value)
                if storage_key in seen:
                    continue
                seen.add(storage_key)
                bytes_ = value.untyped_storage().nbytes()
                state_key = (
                    "metadata"
                    if key.startswith(METADATA_PREFIXES)
                    else key if key in {"exp_avg", "exp_avg_sq", "momentum_buffer", "step"}
                    else "metadata"
                )
                row = rows[(role, state_key, str(value.dtype))]
                row["numel"] += value.numel()
                row["bytes"] += bytes_
                row["quantized"] |= value.dtype in FP8_DTYPES
                if state_key == "metadata" or state_key == "step":
                    metadata_bytes += bytes_
                else:
                    data_bytes += bytes_

    state_fp8 = arm.endswith("_state_fp8")
    expected_adam = (
        (torch.float8_e4m3fn, torch.float8_e5m2)
        if state_fp8
        else (torch.float32, torch.float32)
    )
    saw_adam = False
    saw_muon = False
    for raw in raw_optimizers:
        role = "adamw_all" if adam_only else _role(raw)
        for state in raw.state.values():
            if role in {"adamw_all", "adamw_fallback"} and "exp_avg" in state:
                saw_adam = True
                if (
                    state["exp_avg"].dtype, state["exp_avg_sq"].dtype
                ) != expected_adam:
                    raise AssertionError(f"{role} Adam state precision contract failed")
            if role == "muon_matrix" and "momentum_buffer" in state:
                saw_muon = True
                expected = torch.float8_e4m3fn if state_fp8 else torch.float32
                if state["momentum_buffer"].dtype != expected:
                    raise AssertionError("Muon momentum precision contract failed")
                if state_fp8 and any(key.startswith(("expand_", "sqrt_minmax_")) for key in state):
                    raise AssertionError("Muon momentum must use maxabs, not DRE")
    if not saw_adam:
        raise AssertionError("no initialized Adam state found")
    if not adam_only and not saw_muon:
        raise AssertionError("no initialized Muon momentum found")

    tensors = [
        {
            "group_role": role,
            "state_key": key,
            "dtype": dtype,
            "quantized": value["quantized"],
            "numel": value["numel"],
            "bytes": value["bytes"],
        }
        for (role, key, dtype), value in sorted(rows.items())
    ]
    master_seen = set()
    master_bytes = 0
    for raw in raw_optimizers:
        for group in raw.param_groups:
            for parameter in group["params"]:
                key = _storage_key(parameter)
                if key not in master_seen:
                    master_seen.add(key)
                    master_bytes += parameter.untyped_storage().nbytes()
    return {
        "persistent_data_bytes": data_bytes,
        "metadata_bytes": metadata_bytes,
        "persistent_total_bytes": data_bytes + metadata_bytes,
        "master_parameter_bytes": master_bytes,
        "tensors": tensors,
    }


def parameter_group_ledger(optimizer, arm, parameter_names):
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise NotImplementedError(
            "Stage 3 parameter ledger currently supports the one-GPU EP=1 short probes only"
        )
    counts = defaultdict(int)
    names = defaultdict(list)
    active = defaultdict(int)
    fc1_names = set()
    split_fc1_names = set()
    router_names = set()
    fallback_router_names = set()
    adam_only = arm.startswith("adamw_")
    for raw in _raw_optimizers(optimizer):
        role = "adamw_all" if adam_only else _role(raw)
        for group in raw.param_groups:
            for parameter in group["params"]:
                name = parameter_names.get(id(parameter))
                if name is None:
                    raise AssertionError("optimizer parameter has no stable model name")
                counts[role] += parameter.numel()
                names[role].append(f"{name}:{tuple(parameter.shape)}:{parameter.numel()}")
                if role == "muon_matrix" and name.endswith(".router.weight"):
                    raise AssertionError("router weight reached Muon")
                if name.endswith(".router.weight"):
                    router_names.add(name)
                    if role == "adamw_fallback":
                        fallback_router_names.add(name)
                if role == "muon_matrix" and ".linear_fc1.weight" in name:
                    fc1_names.add(name)
                    if group.get("stage3_split_swiglu_fc1"):
                        split_fc1_names.add(name)
    if not adam_only and fc1_names != split_fc1_names:
        missing = sorted(fc1_names - split_fc1_names)
        raise AssertionError(f"Muon SwiGLU FC1 split flag missing for {missing[:3]}")
    if not adam_only and len(fc1_names) != 1 + 17 * 65:
        raise AssertionError(f"expected 1106 Muon SwiGLU FC1 weights, found {len(fc1_names)}")
    if not adam_only and len(router_names) != 17:
        raise AssertionError(f"expected 17 MoE router weights, found {len(router_names)}")
    if not adam_only and router_names != fallback_router_names:
        raise AssertionError("not every router weight is in the AdamW fallback group")
    if sum(counts.values()) != TOTAL_PARAMETERS:
        raise AssertionError(
            f"total parameter mismatch: {sum(counts.values())} != {TOTAL_PARAMETERS}"
        )
    if not adam_only and dict(counts) != {
        "muon_matrix": MUON_MATRIX_PARAMETERS,
        "adamw_fallback": ADAMW_FALLBACK_PARAMETERS,
    }:
        raise AssertionError(f"Muon parameter groups do not match the design: {dict(counts)}")
    active.update(
        {"adamw_all": ACTIVE_PARAMETERS}
        if adam_only
        else {"muon_matrix": 176_160_768, "adamw_fallback": 104_082_944}
    )
    return [
        {
            "role": role,
            "parameters": count,
            "active_parameters_per_token": active[role],
            "named_parameter_manifest_sha256": hashlib.sha256(
                "\n".join(sorted(names[role])).encode()
            ).hexdigest(),
        }
        for role, count in sorted(counts.items())
    ]


def _normalized_match_argv(argv):
    normalized = []
    skip = False
    for item in argv:
        if skip:
            skip = False
            continue
        if item in {"--fp8-format", "--fp8-recipe"}:
            skip = True
            continue
        normalized.append(item)
    return normalized


def _comparison(arm, argv):
    computed_match_key = hashlib.sha256(
        "\0".join(_normalized_match_argv(argv)).encode()
    ).hexdigest()
    return {
        "optimizer": "muon" if arm.startswith("muon_") else "adamw",
        "gemm_mode": "fp8_delayed_hybrid" if "_fp8gemm_" in arm else "bf16",
        "optimizer_state_mode": "fp8_hybrid" if arm.endswith("_state_fp8") else "fp32",
        "match_key_sha256": os.environ.get(
            "STAGE3_MOE_MATCH_KEY_SHA256", computed_match_key
        ),
    }


def _version(distribution):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "source-checkout"


def _cublaslt_version():
    if "STAGE3_MOE_CUBLASLT" in os.environ:
        return os.environ["STAGE3_MOE_CUBLASLT"]
    library = ctypes.CDLL("libcublasLt.so")
    library.cublasLtGetVersion.restype = ctypes.c_size_t
    return str(library.cublasLtGetVersion())


def _environment():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    gpu_uuid = os.environ.get("STAGE3_MOE_GPU_UUID", str(getattr(props, "uuid", device)))
    mcore_commit = os.environ.get("STAGE3_MOE_MCORE_COMMIT", "source-checkout")
    eo_commit = os.environ.get("STAGE3_MOE_EO_COMMIT")
    return {
        "site": os.environ.get("STAGE3_MOE_SITE", "node207"),
        "host": socket.gethostname(),
        "scheduler_job_id": os.environ.get("STAGE3_MOE_SCHEDULER_JOB_ID"),
        "image": os.environ.get("MLSUB_IMAGE", os.environ.get("STAGE3_MOE_IMAGE", "node207_env")),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "driver": os.environ.get("STAGE3_MOE_DRIVER", "not-reported"),
        "cublaslt": _cublaslt_version(),
        "triton": _version("triton"),
        "transformer_engine": _version("transformer-engine"),
        "megatron_core": f"source@{mcore_commit}",
        "emerging_optimizers": (
            f"source@{eo_commit}" if eo_commit else _version("emerging-optimizers")
        ),
        "nccl": ".".join(str(item) for item in torch.cuda.nccl.version()),
        "gpus": [
            {
                "uuid": gpu_uuid,
                "name": torch.cuda.get_device_name(device),
            }
        ],
        "gpu_clean": {
            key: os.environ.get(f"STAGE3_MOE_GPU_CLEAN_{key.upper()}", "0") == "1"
            for key in ("before", "during", "after")
        },
    }


def _provenance(argv):
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    )
    tracked_diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"], cwd=root
    )
    diff = tracked_diff + b"\0STATUS\0" + status.encode()
    if "--mock-data" in argv:
        data_manifest_sha256 = hashlib.sha256("mcore-mock-data".encode()).hexdigest()
    else:
        data_manifest_sha256 = os.environ["STAGE3_MOE_DATA_MANIFEST_SHA256"]
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest() if status else None,
        "config_sha256": os.environ.get(
            "STAGE3_MOE_CONFIG_SHA256", hashlib.sha256("\0".join(argv).encode()).hexdigest()
        ),
        "data_manifest_sha256": data_manifest_sha256,
        "argv": argv,
        "argv_scope": "effective_mcore_argv",
    }


class Probe:
    def __init__(self, *, arm, result_path, warmup_steps, measured_steps, program_start, argv):
        self.arm = arm
        self.result_path = Path(result_path)
        self.warmup_steps = warmup_steps
        self.measured_steps = measured_steps
        self.program_start = program_start
        self.argv = argv
        self.step = 0
        self.full_step_seconds = []
        self.optimizer_step_seconds = []
        self.losses = []
        self.optimizer = None
        self.parameter_names = {}
        self.written = False

    def reset(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def after_step(self, elapsed, result):
        self.step += 1
        if self.step > self.warmup_steps:
            self.full_step_seconds.append(elapsed)
            loss_dict = result[0]
            if "lm loss" in loss_dict:
                self.losses.append(float(loss_dict["lm loss"]))

    def write(self, status="completed"):
        if self.written or self.optimizer is None:
            return
        torch.cuda.synchronize()
        self.written = True
        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from megatron.training import get_args

            args = get_args()
            full_step = statistics.median(self.full_step_seconds) if self.full_step_seconds else None
            tokens_per_second = (
                args.global_batch_size * args.seq_length / full_step if full_step else None
            )
            record = {
                "schema_version": 1,
                "record_type": "run",
                "run_id": os.environ["STAGE3_MOE_RUN_ID"],
                "arm_id": self.arm,
                "status": status,
                "comparison": _comparison(self.arm, self.argv),
                "denominators": {
                    "micro_batch_sequences_per_gpu": args.micro_batch_size,
                    "global_batch_sequences": args.global_batch_size,
                    "sequence_length": args.seq_length,
                    "loss_tokens_per_step": args.global_batch_size * args.seq_length,
                    "gpu_count": args.world_size,
                    "total_parameters": TOTAL_PARAMETERS,
                    "active_parameters_per_token": ACTIVE_PARAMETERS,
                    "dp": args.data_parallel_size,
                    "tp": args.tensor_model_parallel_size,
                    "pp": args.pipeline_model_parallel_size,
                    "cp": args.context_parallel_size,
                    "ep": args.expert_model_parallel_size,
                    "etp": args.expert_tensor_parallel_size,
                },
                "provenance": _provenance(self.argv),
                "environment": _environment(),
                "parameter_groups": parameter_group_ledger(
                    self.optimizer, self.arm, self.parameter_names
                ),
                "optimizer_state": optimizer_state_ledger(self.optimizer, self.arm),
                "measurement": {
                    "protocol": {
                        "kind": (
                            "formal_timing"
                            if self.warmup_steps >= 20 and self.measured_steps >= 100
                            else "smoke"
                        ),
                        "warmup_steps": self.warmup_steps,
                        "measured_steps": len(self.full_step_seconds),
                        "e2e_train_steps": self.step,
                    },
                    "memory": {
                        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
                    },
                    "timing": {
                        "tokens_per_second": tokens_per_second,
                        "optimizer_step_seconds": (
                            statistics.median(self.optimizer_step_seconds)
                            if self.optimizer_step_seconds
                            else None
                        ),
                        "full_step_seconds": full_step,
                        "e2e_wct_seconds": time.perf_counter() - self.program_start,
                        "e2e_wct_scope": "process_start_to_result_write",
                        "optimizer_step_samples_seconds": self.optimizer_step_seconds,
                        "full_step_samples_seconds": self.full_step_seconds,
                    },
                    "loss": {
                        "training": statistics.mean(self.losses) if self.losses else None,
                        "validation": None,
                    },
                    "routing": {
                        "scope": "global_unpadded",
                        "tokens_per_expert_artifact_sha256": None,
                        "minimum_to_mean": None,
                        "maximum_to_mean": None,
                        "coefficient_of_variation": None,
                        "dropped_tokens": None,
                    },
                    "downstream": [],
                    "inference": None,
                },
            }
            with self.result_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as error:
            print(f"stage3 result write failed: {error}", file=sys.stderr, flush=True)
            raise


def install_probe(*, arm, result_path, warmup_steps, measured_steps, program_start, argv):
    import megatron.training.training as training

    probe = Probe(
        arm=arm,
        result_path=result_path,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        program_start=program_start,
        argv=argv,
    )
    original = training.train_step
    original_setup = training.setup_model_and_optimizer

    def named_setup(*args, **kwargs):
        model, optimizer, scheduler = original_setup(*args, **kwargs)
        if probe.arm.endswith("_state_fp8"):
            assert_fp8_adam_bootstrap(optimizer)
        for chunk_index, chunk in enumerate(model):
            for name, parameter in chunk.named_parameters():
                stable_name = f"model_chunk{chunk_index}.{name}"
                probe.parameter_names[id(parameter)] = stable_name
                if hasattr(parameter, "main_param"):
                    probe.parameter_names[id(parameter.main_param)] = stable_name
        return model, optimizer, scheduler

    training.setup_model_and_optimizer = named_setup

    timed_optimizer_ids = set()

    def measured_train_step(*args, **kwargs):
        optimizer = args[3]
        probe.optimizer = optimizer
        if id(optimizer) not in timed_optimizer_ids:
            original_optimizer_step = optimizer.step

            def timed_optimizer_step(*step_args, **step_kwargs):
                torch.cuda.synchronize()
                optimizer_start = time.perf_counter()
                value = original_optimizer_step(*step_args, **step_kwargs)
                torch.cuda.synchronize()
                if probe.step >= probe.warmup_steps:
                    probe.optimizer_step_seconds.append(time.perf_counter() - optimizer_start)
                return value

            optimizer.step = timed_optimizer_step
            timed_optimizer_ids.add(id(optimizer))
        if probe.step == 0 and probe.warmup_steps == 0:
            probe.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        probe.after_step(time.perf_counter() - start, result)
        if probe.step == probe.warmup_steps:
            probe.reset()
        if probe.step == probe.warmup_steps + probe.measured_steps:
            probe.write()
        return result

    training.train_step = measured_train_step

    def write_unfinished():
        expected = probe.warmup_steps + probe.measured_steps
        if probe.step < expected:
            probe.write(status="failed")

    atexit.register(write_unfinished)
