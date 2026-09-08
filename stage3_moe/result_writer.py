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
from collections import defaultdict, deque
from importlib import import_module, metadata
from pathlib import Path

import torch

from stage3_moe.pair_results import mask_arm_in_paths
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
    if hasattr(raw, "optimizer_role"):
        return raw.optimizer_role
    state_specs = getattr(raw, "state_specs", ())
    keys = {spec.name if hasattr(spec, "name") else spec[0] for spec in state_specs}
    if "momentum_buffer" in keys or "Muon" in type(raw).__name__:
        return "muon_matrix"
    return "adamw_fallback"


TOKEN_TALLY = "stage3_tokens_per_expert"


def routing_balance_summary(counts):
    counts = counts.detach().to(device="cpu", dtype=torch.float64)
    if counts.ndim != 2 or bool((counts.sum(dim=1) <= 0).any()):
        raise ValueError("routing counts must be a non-empty layer-by-expert matrix")
    means = counts.mean(dim=1)
    minimum_to_mean = counts.min(dim=1).values / means
    maximum_to_mean = counts.max(dim=1).values / means
    maxvio = maximum_to_mean - 1.0
    coefficient = counts.std(dim=1, unbiased=False) / means
    return {
        "tokens_per_expert": counts.to(torch.int64).tolist(),
        "maxvio_per_layer": maxvio.tolist(),
        "maxvio_mean": float(maxvio.mean()),
        "maxvio_p95": float(torch.quantile(maxvio, 0.95)),
        "maxvio_max": float(maxvio.max()),
        "minimum_to_mean_per_layer": minimum_to_mean.tolist(),
        "minimum_to_mean_min": float(minimum_to_mean.min()),
        "maximum_to_mean_per_layer": maximum_to_mean.tolist(),
        "maximum_to_mean_max": float(maximum_to_mean.max()),
        "coefficient_of_variation_per_layer": coefficient.tolist(),
        "coefficient_of_variation_max": float(coefficient.max()),
        "zero_experts_per_layer": (counts == 0).sum(dim=1).tolist(),
        "zero_experts_max": int((counts == 0).sum(dim=1).max()),
    }


def _moe_routers(model_chunks):
    """MCore registers `local_tokens_per_expert` on every router when expert bias is on."""
    for chunk_index, chunk in enumerate(model_chunks):
        for name, module in chunk.named_modules():
            counts = getattr(module, "local_tokens_per_expert", None)
            if torch.is_tensor(counts):
                yield f"model_chunk{chunk_index}.{name}", module


def install_router_tallies(model_chunks):
    """Accumulate per-expert token counts that MCore itself discards every step.

    `reset_model_temporary_tensors` zeroes `local_tokens_per_expert` inside each
    training step once the expert bias has been updated, so the buffer is empty by
    the time results are written. Wrapping `_apply_expert_bias` and taking the delta
    it produces keeps MCore's own counting - including the padding mask and the
    grad-enabled guard - while surviving the reset.
    """
    for _, module in _moe_routers(model_chunks):
        if hasattr(module, TOKEN_TALLY):
            continue
        setattr(module, TOKEN_TALLY, torch.zeros_like(module.local_tokens_per_expert))
        original = module._apply_expert_bias

        def tally(routing_map, padding_mask=None, _module=module, _original=original):
            before = _module.local_tokens_per_expert.detach().clone()
            result = _original(routing_map, padding_mask=padding_mask)
            tallied = getattr(_module, TOKEN_TALLY)
            tallied += _module.local_tokens_per_expert.detach() - before
            return result

        module._apply_expert_bias = tally


def _storage_key(tensor):
    storage = tensor.untyped_storage()
    return (tensor.device.type, tensor.device.index, storage.data_ptr(), storage.nbytes())


def optimizer_state_ledger(optimizer, arm):
    rows = defaultdict(lambda: {"numel": 0, "bytes": 0, "quantized": False})
    data_bytes = 0
    metadata_bytes = 0
    seen = set()
    raw_optimizers = list(_raw_optimizers(optimizer))
    optimizer_name = arm.split("_", 1)[0]
    adam_only = optimizer_name == "adamw"

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
    saw_frugal = False
    saw_slimadam = False
    for raw in raw_optimizers:
        role = "adamw_all" if adam_only else _role(raw)
        for parameter, state in raw.state.items():
            if role in {"adamw_all", "adamw_fallback"} and "exp_avg" in state:
                saw_adam = True
                if (
                    state["exp_avg"].dtype, state["exp_avg_sq"].dtype
                ) != expected_adam:
                    raise AssertionError(f"{role} Adam state precision contract failed")
            if role in {"frugal_matrix", "slimadam_all"} and "exp_avg" in state:
                if (
                    state["exp_avg"].dtype != torch.float32
                    or state["exp_avg_sq"].dtype != torch.float32
                ):
                    raise AssertionError(f"{role} state precision contract failed")
                if role == "frugal_matrix":
                    from stage3_moe.frugal import FRUGAL_DENSITY

                    saw_frugal = True
                    expected_shape = (
                        parameter.shape[0],
                        int(parameter.shape[1] * FRUGAL_DENSITY),
                    )
                    if state["exp_avg"].shape != expected_shape:
                        raise AssertionError("FRUGAL coordinate first moment shape contract failed")
                    if state["exp_avg_sq"].shape != expected_shape:
                        raise AssertionError("FRUGAL coordinate second moment shape contract failed")
                    if state["coord_indices"].shape != (expected_shape[1],):
                        raise AssertionError("FRUGAL coordinate index shape contract failed")
                else:
                    saw_slimadam = True
                    if state["exp_avg"].shape != parameter.shape:
                        raise AssertionError("SlimAdam first moment shape contract failed")
                    dims = next(
                        group["slim_compress_dims"]
                        for group in raw.param_groups
                        if any(candidate is parameter for candidate in group["params"])
                    )
                    expected_shape = list(parameter.shape)
                    if dims is not None:
                        for dim in dims:
                            expected_shape[dim] = 1
                    if tuple(state["exp_avg_sq"].shape) != tuple(expected_shape):
                        raise AssertionError(
                            "SlimAdam second moment shape contract failed"
                        )
            if role == "muon_matrix" and "momentum_buffer" in state:
                saw_muon = True
                expected = torch.float8_e4m3fn if state_fp8 else torch.float32
                if state["momentum_buffer"].dtype != expected:
                    raise AssertionError("Muon momentum precision contract failed")
                if state_fp8 and any(
                    key.startswith(("expand_", "sqrt_minmax_")) for key in state
                ):
                    raise AssertionError("Muon momentum must use maxabs, not DRE")
    if optimizer_name in {"adamw", "muon", "frugal"} and not saw_adam:
        raise AssertionError("no initialized Adam state found")
    if optimizer_name == "muon" and not saw_muon:
        raise AssertionError("no initialized Muon momentum found")
    if optimizer_name == "frugal" and not saw_frugal:
        raise AssertionError("no initialized FRUGAL state found")
    if optimizer_name == "slimadam" and not saw_slimadam:
        raise AssertionError("no initialized SlimAdam state found")

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


ROLES = (
    "adamw_all",
    "adamw_fallback",
    "muon_matrix",
    "frugal_matrix",
    "slimadam_all",
)


def _is_routed_expert(name):
    """Routed experts are the only expert-parallel tensors.

    Names read `...mlp.experts.linear_fc1.weight0` for routed experts and
    `...mlp.shared_experts.linear_fc1.weight` for the shared one, so `.experts.`
    matches routed experts and never the shared expert. This is decided by name
    rather than MCore's `allreduce` flag because the optimizer holds FP32 master
    copies, which do not necessarily carry the flag.
    """
    return ".experts." in name


def _sum_over_expert_ranks(values):
    """Sum per-rank quantities across the expert-parallel group; identity at EP=1."""
    if not torch.distributed.is_initialized():
        return list(values)
    from megatron.core import parallel_state

    group = parallel_state.get_expert_model_parallel_group()
    if torch.distributed.get_world_size(group=group) == 1:
        return list(values)
    buffer = torch.tensor(list(values), dtype=torch.float64, device=torch.cuda.current_device())
    torch.distributed.all_reduce(buffer, op=torch.distributed.ReduceOp.SUM, group=group)
    return [int(round(value)) for value in buffer.tolist()]


# An evaluation run builds the optimizer but never steps it, so no state tensor is ever
# materialized and `optimizer_state_ledger` has nothing to inspect. Null says that; zero
# would claim the arm holds no state, which is a different and false statement.
UNSTEPPED_OPTIMIZER_STATE = {
    "master_parameter_bytes": None,
    "metadata_bytes": None,
    "persistent_data_bytes": None,
    "persistent_total_bytes": None,
    "tensors": [],
}


def parameter_group_ledger(optimizer, arm, parameter_names):
    counts = defaultdict(int)
    expert_counts = defaultdict(int)
    names = defaultdict(list)
    active = defaultdict(int)
    fc1_names = set()
    expert_fc1_names = set()
    split_fc1_names = set()
    router_names = set()
    fallback_router_names = set()
    optimizer_name = arm.split("_", 1)[0]
    adam_only = optimizer_name == "adamw"
    for raw in _raw_optimizers(optimizer):
        role = "adamw_all" if adam_only else _role(raw)
        for group in raw.param_groups:
            for parameter in group["params"]:
                name = parameter_names.get(id(parameter))
                if name is None:
                    raise AssertionError("optimizer parameter has no stable model name")
                counts[role] += parameter.numel()
                if _is_routed_expert(name):
                    expert_counts[role] += parameter.numel()
                names[role].append(f"{name}:{tuple(parameter.shape)}:{parameter.numel()}")
                if role == "muon_matrix" and name.endswith(".router.weight"):
                    raise AssertionError("router weight reached Muon")
                if role == "frugal_matrix" and name.endswith(".router.weight"):
                    raise AssertionError("router weight reached FRUGAL")
                if name.endswith(".router.weight"):
                    router_names.add(name)
                    if role == "adamw_fallback":
                        fallback_router_names.add(name)
                if role == "muon_matrix" and ".linear_fc1.weight" in name:
                    fc1_names.add(name)
                    if _is_routed_expert(name):
                        expert_fc1_names.add(name)
                    if group.get("stage3_split_swiglu_fc1"):
                        split_fc1_names.add(name)
                if role == "slimadam_all":
                    from stage3_moe.slim_adam import (
                        SLIM_COMPRESS_DIMS,
                        slim_compression_dims,
                    )

                    expected_dims = slim_compression_dims(parameter, name)
                    if group.get(SLIM_COMPRESS_DIMS) != expected_dims:
                        raise AssertionError(
                            f"SlimAdam compression rule mismatch for {name}"
                        )

    # Replicated tensors are identical on every expert rank, so count them once and
    # add only the sharded expert tensors summed across the expert-parallel group.
    reduced = _sum_over_expert_ranks(
        [expert_counts[role] for role in ROLES] + [len(expert_fc1_names)]
    )
    global_counts = {
        role: counts[role] - expert_counts[role] + reduced[index]
        for index, role in enumerate(ROLES)
        if role in counts
    }
    global_fc1 = len(fc1_names) - len(expert_fc1_names) + reduced[-1]

    if optimizer_name == "muon" and fc1_names != split_fc1_names:
        missing = sorted(fc1_names - split_fc1_names)
        raise AssertionError(f"Muon SwiGLU FC1 split flag missing for {missing[:3]}")
    if optimizer_name == "muon" and global_fc1 != 1 + 17 * 65:
        raise AssertionError(f"expected 1106 Muon SwiGLU FC1 weights, found {global_fc1}")
    if optimizer_name in {"muon", "frugal"} and len(router_names) != 17:
        raise AssertionError(f"expected 17 MoE router weights, found {len(router_names)}")
    if optimizer_name in {"muon", "frugal"} and router_names != fallback_router_names:
        raise AssertionError("not every router weight is in the AdamW fallback group")
    if sum(global_counts.values()) != TOTAL_PARAMETERS:
        raise AssertionError(
            f"total parameter mismatch: {sum(global_counts.values())} != {TOTAL_PARAMETERS}"
        )
    expected_counts = {
        "adamw": {"adamw_all": TOTAL_PARAMETERS},
        "muon": {
            "muon_matrix": MUON_MATRIX_PARAMETERS,
            "adamw_fallback": ADAMW_FALLBACK_PARAMETERS,
        },
        "frugal": {
            "frugal_matrix": MUON_MATRIX_PARAMETERS,
            "adamw_fallback": ADAMW_FALLBACK_PARAMETERS,
        },
        "slimadam": {"slimadam_all": TOTAL_PARAMETERS},
    }[optimizer_name]
    if global_counts != expected_counts:
        raise AssertionError(
            f"{optimizer_name} parameter groups do not match the design: {global_counts}"
        )
    active.update(
        {
            "adamw": {"adamw_all": ACTIVE_PARAMETERS},
            "muon": {"muon_matrix": 176_160_768, "adamw_fallback": 104_082_944},
            "frugal": {"frugal_matrix": 176_160_768, "adamw_fallback": 104_082_944},
            "slimadam": {"slimadam_all": ACTIVE_PARAMETERS},
        }[optimizer_name]
    )
    return [
        {
            "role": role,
            "parameters": count,
            "parameters_global": global_counts[role],
            "active_parameters_per_token": active[role],
            # manifest covers this rank's shard only
            "named_parameter_manifest_sha256": hashlib.sha256(
                "\n".join(sorted(names[role])).encode()
            ).hexdigest(),
        }
        for role, count in sorted(counts.items())
    ]


def _normalized_match_argv(argv, arm=None):
    # The match key is what proves two runs controlled the same factors, so it has to
    # ignore exactly what the pair comparison ignores: where this arm reads and writes.
    argv = mask_arm_in_paths(argv, arm) if arm is not None else list(argv)
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
        "\0".join(_normalized_match_argv(argv, arm)).encode()
    ).hexdigest()
    return {
        "optimizer": arm.split("_", 1)[0],
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
        # Filled by an evaluation run; a training run leaves them as they are.
        self.downstream = []
        self.validation_loss = None
        self.protocol_kind = None
        self.optimizer_step_seconds = []
        self.losses = []
        self.optimizer = None
        self.parameter_names = {}
        self.written = False
        self.model_chunks = []
        self.router_names = []
        self.routing_window = deque(maxlen=100)
        self.pending_routing = None
        self.previous_load_distribution = None
        self.previous_bias_update_sign = None

    def capture_router_update(self, counts, expert_bias, updated_expert_bias):
        counts = counts.detach().to(device="cpu", dtype=torch.float64)
        expert_bias = expert_bias.detach().to(device="cpu", dtype=torch.float64)
        updated_expert_bias = updated_expert_bias.detach().to(
            device="cpu", dtype=torch.float64
        )
        if self.router_names and counts.shape[0] != len(self.router_names):
            raise AssertionError(
                f"router telemetry layer mismatch: {counts.shape[0]} != {len(self.router_names)}"
            )

        distribution = counts / counts.sum(dim=1, keepdim=True)
        if self.previous_load_distribution is None:
            load_tv = None
        else:
            load_tv = 0.5 * (
                distribution - self.previous_load_distribution
            ).abs().sum(dim=1)
        bias_delta = updated_expert_bias - expert_bias
        bias_update_sign = torch.sign(bias_delta)
        if self.previous_bias_update_sign is None:
            bias_flip_fraction = None
        else:
            comparable = (bias_update_sign != 0) & (self.previous_bias_update_sign != 0)
            bias_flip_fraction = float(
                ((bias_update_sign != self.previous_bias_update_sign) & comparable)
                .to(torch.float64)
                .mean()
            )

        self.routing_window.append(counts)
        self.pending_routing = {
            "counts": counts,
            "expert_bias": updated_expert_bias,
            "bias_delta_abs_max": float(bias_delta.abs().max()),
            "bias_update_flip_fraction": bias_flip_fraction,
            "load_tv": load_tv,
        }
        self.previous_load_distribution = distribution
        self.previous_bias_update_sign = bias_update_sign

    def write_routing_telemetry(self, iteration):
        if self.pending_routing is None:
            return
        from megatron.training import get_args, get_tensorboard_writer, get_wandb_writer

        args = get_args()
        interval = int(os.environ.get("STAGE3_MOE_ROUTING_TELEMETRY_INTERVAL", 10))
        emit = iteration == 1 or iteration % interval == 0 or iteration == args.train_iters
        if not emit:
            self.pending_routing = None
            return

        batch = routing_balance_summary(self.pending_routing["counts"])
        rolling = routing_balance_summary(torch.stack(tuple(self.routing_window)).sum(dim=0))
        bias = self.pending_routing["expert_bias"]
        bias_range = bias.max(dim=1).values - bias.min(dim=1).values
        load_tv = self.pending_routing["load_tv"]
        record = {
            "schema_version": 1,
            "iteration": iteration,
            "scope": "global_batch_unpadded",
            "layers": self.router_names,
            "batch": batch,
            "rolling_100": {
                "window_steps": len(self.routing_window),
                **rolling,
            },
            "expert_bias": {
                "values": bias.tolist(),
                "minimum_per_layer": bias.min(dim=1).values.tolist(),
                "maximum_per_layer": bias.max(dim=1).values.tolist(),
                "mean_per_layer": bias.mean(dim=1).tolist(),
                "std_per_layer": bias.std(dim=1, unbiased=False).tolist(),
                "range_per_layer": bias_range.tolist(),
                "absolute_max": float(bias.abs().max()),
                "range_max": float(bias_range.max()),
                "delta_absolute_max": self.pending_routing["bias_delta_abs_max"],
                "update_flip_fraction": self.pending_routing[
                    "bias_update_flip_fraction"
                ],
            },
            "drift": {
                "load_total_variation_from_previous_mean": (
                    None if load_tv is None else float(load_tv.mean())
                ),
                "load_total_variation_from_previous_max": (
                    None if load_tv is None else float(load_tv.max())
                ),
            },
            "dropped_tokens": 0 if args.moe_expert_capacity_factor is None else None,
        }

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            artifact = self.result_path.parent / "routing_telemetry.jsonl"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            with artifact.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

        scalars = {
            "routing/batch_maxvio_mean": batch["maxvio_mean"],
            "routing/batch_maxvio_max": batch["maxvio_max"],
            "routing/batch_minimum_to_mean_min": batch["minimum_to_mean_min"],
            "routing/batch_cv_max": batch["coefficient_of_variation_max"],
            "routing/rolling100_maxvio_mean": rolling["maxvio_mean"],
            "routing/rolling100_maxvio_max": rolling["maxvio_max"],
            "routing/rolling100_minimum_to_mean_min": rolling[
                "minimum_to_mean_min"
            ],
            "routing/rolling100_cv_max": rolling["coefficient_of_variation_max"],
            "routing/expert_bias_absolute_max": record["expert_bias"]["absolute_max"],
            "routing/expert_bias_range_max": record["expert_bias"]["range_max"],
        }
        if load_tv is not None:
            scalars["routing/load_tv_previous_mean"] = float(load_tv.mean())
            scalars["routing/load_tv_previous_max"] = float(load_tv.max())
        if self.pending_routing["bias_update_flip_fraction"] is not None:
            scalars["routing/bias_update_flip_fraction"] = self.pending_routing[
                "bias_update_flip_fraction"
            ]
        for index, (batch_maxvio, rolling_maxvio) in enumerate(
            zip(batch["maxvio_per_layer"], rolling["maxvio_per_layer"])
        ):
            scalars[f"routing/layer_{index:02d}_batch_maxvio"] = batch_maxvio
            scalars[f"routing/layer_{index:02d}_rolling100_maxvio"] = rolling_maxvio

        writer = get_tensorboard_writer()
        if writer:
            for name, value in scalars.items():
                writer.add_scalar(name, value, iteration)
        wandb_writer = get_wandb_writer()
        if wandb_writer:
            wandb_writer.log(scalars, iteration)
        self.pending_routing = None

    def reset(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        # Drop the warm-up tally so routing describes the measured window only.
        for _, module in _moe_routers(self.model_chunks):
            tallied = getattr(module, TOKEN_TALLY, None)
            if tallied is not None:
                tallied.zero_()

    def routing_metrics(self):
        empty = {
            "scope": "global_unpadded",
            "tokens_per_expert_artifact_sha256": None,
            "minimum_to_mean": None,
            "maximum_to_mean": None,
            "coefficient_of_variation": None,
            "dropped_tokens": None,
        }
        names, rows = [], []
        for name, module in _moe_routers(self.model_chunks):
            tallied = getattr(module, TOKEN_TALLY, None)
            if tallied is None:
                continue
            names.append(name)
            rows.append(tallied.detach().to(torch.float64))
        if not rows:
            return empty
        matrix = torch.stack(rows)
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            # Each rank tallies its own tokens across all experts; these probes pin TP=CP=1,
            # so summing over ranks yields the global count.
            torch.distributed.all_reduce(matrix, op=torch.distributed.ReduceOp.SUM)
        matrix = matrix.cpu()
        if float(matrix.sum()) <= 0:
            return empty

        from megatron.training import get_args

        capacity_factor = get_args().moe_expert_capacity_factor
        artifact = self.result_path.parent / "tokens_per_expert.json"
        artifact.write_text(
            json.dumps(
                {
                    "scope": "global_unpadded",
                    "window": "measured_steps_only",
                    "layers": names,
                    "tokens_per_expert": matrix.tolist(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        # design.md requires every layer to satisfy the thresholds, so report the worst layer.
        means = matrix.mean(dim=1)
        live = means > 0
        if not bool(live.any()):
            return empty
        minimum_to_mean = float((matrix.min(dim=1).values[live] / means[live]).min())
        maximum_to_mean = float((matrix.max(dim=1).values[live] / means[live]).max())
        coefficient = float((matrix.std(dim=1, unbiased=False)[live] / means[live]).max())
        return {
            "scope": "global_unpadded",
            "tokens_per_expert_artifact_sha256": hashlib.sha256(
                artifact.read_bytes()
            ).hexdigest(),
            "minimum_to_mean": minimum_to_mean,
            "maximum_to_mean": maximum_to_mean,
            "coefficient_of_variation": coefficient,
            # dropless MoE: no capacity factor means no token can be dropped.
            "dropped_tokens": 0 if capacity_factor is None else None,
        }

    def after_step(self, elapsed, result, iteration=None):
        self.step += 1
        if self.step > self.warmup_steps:
            self.full_step_seconds.append(elapsed)
            loss_dict = result[0]
            if "lm loss" in loss_dict:
                self.losses.append(float(loss_dict["lm loss"]))
        self.write_routing_telemetry(
            self.step if iteration is None else int(iteration) + 1
        )

    def write(self, status="completed"):
        if self.written or self.optimizer is None:
            return
        torch.cuda.synchronize()
        self.written = True
        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        # Both ledgers reduce across the expert group, so every rank must reach them;
        # only rank 0 then writes, otherwise the ranks would clobber one file.
        parameter_groups = parameter_group_ledger(
            self.optimizer, self.arm, self.parameter_names
        )
        routing = self.routing_metrics()
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
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
                "parameter_groups": parameter_groups,
                "optimizer_state": (
                    UNSTEPPED_OPTIMIZER_STATE
                    if self.protocol_kind == "evaluation"
                    else optimizer_state_ledger(self.optimizer, self.arm)
                ),
                "measurement": {
                    "protocol": {
                        "kind": self.protocol_kind or (
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
                        "validation": self.validation_loss,
                    },
                    "routing": routing,
                    "downstream": self.downstream,
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

    finalize_model_grads = import_module(
        "megatron.core.distributed.finalize_model_grads"
    )

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
    original_bias_update = finalize_model_grads.get_updated_expert_bias

    def captured_bias_update(tokens_per_expert, expert_bias, *args, **kwargs):
        updated = original_bias_update(tokens_per_expert, expert_bias, *args, **kwargs)
        probe.capture_router_update(tokens_per_expert, expert_bias, updated)
        return updated

    finalize_model_grads.get_updated_expert_bias = captured_bias_update

    def named_setup(*args, **kwargs):
        model, optimizer, scheduler = original_setup(*args, **kwargs)
        if probe.arm.endswith("_state_fp8") and "--skip-train" not in argv:
            assert_fp8_adam_bootstrap(optimizer)
        probe.optimizer = optimizer
        probe.model_chunks = list(model)
        install_router_tallies(probe.model_chunks)
        probe.router_names = [name for name, _ in _moe_routers(probe.model_chunks)]
        for chunk_index, chunk in enumerate(model):
            for name, parameter in chunk.named_parameters():
                stable_name = f"model_chunk{chunk_index}.{name}"
                probe.parameter_names[id(parameter)] = stable_name
                if hasattr(parameter, "main_param"):
                    probe.parameter_names[id(parameter.main_param)] = stable_name
        return model, optimizer, scheduler

    training.setup_model_and_optimizer = named_setup

    # `evaluate` returns the loss dict but not which split it came from, and
    # `evaluate_and_print_results` knows the split but not the value; between them the
    # validation number the quality gate reads can be recovered.
    last_evaluation = {}
    original_evaluate = training.evaluate
    original_report = training.evaluate_and_print_results

    def capturing_evaluate(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        last_evaluation["losses"] = result[0]
        return result

    def capturing_report(prefix, *args, **kwargs):
        value = original_report(prefix, *args, **kwargs)
        loss = (last_evaluation.get("losses") or {}).get("lm loss")
        if loss is not None and "validation set" in str(prefix):
            probe.validation_loss = float(loss)
        return value

    training.evaluate = capturing_evaluate
    training.evaluate_and_print_results = capturing_report

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
        probe.after_step(
            time.perf_counter() - start, result, iteration=kwargs.get("iteration")
        )
        if probe.step == probe.warmup_steps:
            probe.reset()
        if probe.step == probe.warmup_steps + probe.measured_steps:
            probe.write()
        return result

    training.train_step = measured_train_step

    def write_unfinished():
        expected = probe.warmup_steps + probe.measured_steps
        if probe.step < expected:
            # An evaluation run scores a checkpoint and never trains, so falling short of
            # the measured window is what it is supposed to do.
            probe.write(status="completed" if probe.protocol_kind == "evaluation" else "failed")

    atexit.register(write_unfinished)
    return probe
