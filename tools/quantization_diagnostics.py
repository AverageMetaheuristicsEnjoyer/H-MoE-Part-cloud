"""Read-only diagnostics using the production MoE state codecs.

Checkpoint mode measures snapshot distortion only. The optional live hook also
measures the first real-gradient update; neither mode estimates training loss.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "Megatron-LM"))
sys.path.insert(0, str(ROOT / "third_party" / "emerging-optimizers"))

import torch

from stage3_moe.optimizer_states import (
    StateSpec,
    dequantize_fp8_state,
    init_fp8_state,
    quantize_fp8_state_,
)

SIGNED = {
    "exp_avg": True,
    "exp_avg_sq": False,
    "momentum_buffer": True,
    "momentum": True,
    "variance": False,
    "exp_avg_fast": True,
    "exp_avg_slow": True,
}
QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
# (codec, second-moment format) candidates; "coat" is the upstream COAT codec (both moments E4M3).
CODECS = (("maxabs", torch.float8_e4m3fn), ("dre", torch.float8_e4m3fn), ("dre", torch.float8_e5m2), ("coat", torch.float8_e4m3fn))
# How FP8 states found in a checkpoint were encoded: "auto" = dre if expand_ metadata exists else maxabs;
# COAT checkpoints carry the same metadata names, so they need STORED_CODEC = "coat".
STORED_CODEC = "auto"


def emit(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def metrics(reference, restored):
    reference, restored = reference.float().flatten(), restored.float().flatten()
    if not torch.isfinite(reference).all() or not torch.isfinite(restored).all():
        return {
            "reference_nonfinite": int((~torch.isfinite(reference)).sum()),
            "restored_nonfinite": int((~torch.isfinite(restored)).sum()),
        }
    delta = restored - reference
    # Small second moments can square to zero in FP32; reduction precision matters.
    norm = torch.linalg.vector_norm(reference, dtype=torch.float64)
    restored_norm = torch.linalg.vector_norm(restored, dtype=torch.float64)
    delta_norm = torch.linalg.vector_norm(delta, dtype=torch.float64)
    stride = max(1, (reference.numel() + 65535) // 65536)
    sample = delta.abs()[::stride].double()
    sample_reference = reference[::stride].abs().double()
    nonzero = sample_reference != 0
    sample_relative = sample[nonzero] / sample_reference[nonzero]
    rms = norm / reference.numel() ** 0.5
    thresholds = (0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
    norm_product = norm * restored_norm
    quantiles = torch.tensor(QUANTILES, device=sample.device, dtype=torch.float64)
    return {
        "relative_frobenius": float(delta_norm / norm) if norm > 0 else (0.0 if delta_norm == 0 else None),
        "reference_norm": float(norm),
        "error_norm": float(delta_norm),
        "cosine": float(torch.sum(reference.double() * restored, dtype=torch.float64) / norm_product) if norm_product > 0 else None,
        "bias_over_rms": float(delta.sum(dtype=torch.float64) / reference.numel() / rms) if rms > 0 else None,
        "new_zero_fraction": float(((reference != 0) & (restored == 0)).float().mean()),
        "zero_fraction_before": float((reference == 0).float().mean()),
        "zero_fraction_after": float((restored == 0).float().mean()),
        "sample_stride": stride,
        "sample_size": sample.numel(),
        "absolute_error_quantiles": dict(zip(map(str, QUANTILES), torch.quantile(sample, quantiles).tolist())),
        "element_relative_error_nonzero_sample_size": sample_relative.numel(),
        "element_relative_error_quantiles": dict(zip(map(str, QUANTILES), torch.quantile(sample_relative, quantiles).tolist())) if sample_relative.numel() else None,
        "error_over_source_rms_cdf": {str(x): float((sample / rms <= x).double().mean()) for x in thresholds} if rms > 0 else None,
    }


def roundtrip(value, name, recipe, dtype=torch.float8_e4m3fn):
    spec = StateSpec(name, SIGNED[name], dtype, recipe)
    state = {}
    init_fp8_state(state, spec, value, group_size=128)
    quantize_fp8_state_(state, spec, value, group_size=128)
    restored = dequantize_fp8_state(state, spec, group_size=128)
    payload = state[name].float()
    storage = sum(t.numel() * t.element_size() for t in state.values())
    info = {
        "payload_bytes": state[name].numel(),
        "metadata_bytes": storage - state[name].numel(),
        "metadata_dtype": "float32",
        # Maxabs maps the group maximum to a rail by construction. This is not clipping.
        "payload_rail_fraction": float((payload.abs() == torch.finfo(dtype).max).float().mean()),
        "payload_nonfinite": int((~torch.isfinite(payload)).sum()),
    }
    return restored, info


def materialize(state, name):
    value = state[name]
    if value.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return value.detach().to(device="cuda", dtype=torch.float32)
    recipe = STORED_CODEC if STORED_CODEC != "auto" else ("dre" if f"expand_{name}" in state else "maxabs")
    metadata = {k: v.cuda() for k, v in state.items() if torch.is_tensor(v) and (k == name or k.endswith("_" + name))}
    spec = StateSpec(name, SIGNED[name], value.dtype, recipe)
    return dequantize_fp8_state(metadata, spec, group_size=128)


def selected_parameters(groups, states, samples):
    strata = defaultdict(list)
    for group_index, group in enumerate(groups):
        for position, parameter in enumerate(group["params"]):
            state = states.get(parameter, {})
            keys = tuple(k for k in SIGNED if k in state and torch.is_tensor(state[k]))
            if not keys:
                continue
            shape = tuple(state[keys[0]].shape)
            strata[(group_index, shape, keys)].append((position, parameter))
    for (group_index, shape, keys), entries in strata.items():
        count = min(samples, len(entries))
        indices = {round(i * (len(entries) - 1) / max(count - 1, 1)) for i in range(count)}
        for index in sorted(indices):
            position, parameter = entries[index]
            yield group_index, position, parameter, keys


def state_rows(state, label, output):
    values = {name: materialize(state, name) for name in SIGNED if name in state and torch.is_tensor(state[name])}
    for name, value in values.items():
        for recipe, dtype in CODECS:
            if dtype == torch.float8_e5m2 and SIGNED[name]:
                continue
            restored, info = roundtrip(value, name, recipe, dtype)
            emit(output, {**label, "kind": "state_snapshot", "state": name, "source_dtype": str(state[name].dtype), "shape": list(value.shape), "codec": recipe, "format": str(dtype), **info, **metrics(value, restored)})
    return values


def adam_direction(values, group, step):
    mkey, vkey = ("exp_avg", "exp_avg_sq") if "exp_avg_sq" in values else ("momentum", "variance")
    m, v = values[mkey], values[vkey]
    beta1, beta2 = group.get("betas", (group.get("beta1", 0.9), group.get("beta2", 0.95)))
    correction = group.get("bias_correction", True)
    first = 1 - beta1 ** step if correction else 1.0
    second = 1 - beta2 ** step if correction else 1.0
    return (m / first) / ((v / second).sqrt() + group.get("eps", group.get("epsilon", 1e-8)))


def adam_rows(values, state, group, label, output, gradient=None):
    if "exp_avg_sq" not in values and "variance" not in values:
        return
    if "exp_avg_slow" in values or "exp_avg_fast" in values:
        return  # AdEMAMix is not AdamW; snapshot metrics remain valid.
    if not (("betas" in group or ("beta1" in group and "beta2" in group)) and ("eps" in group or "epsilon" in group)):
        emit(output, {**label, "kind": "adam_direction_skipped", "reason": "checkpoint lacks explicit beta or epsilon hyperparameters"})
        return
    mkey, vkey = ("exp_avg", "exp_avg_sq") if "exp_avg_sq" in values else ("momentum", "variance")
    step = int(state.get("step", group.get("step", 0)))
    if step < 1:
        emit(output, {**label, "kind": "adam_direction_skipped", "reason": "checkpoint has no positive optimizer step"})
        return
    beta1, beta2 = group.get("betas", (group.get("beta1", 0.9), group.get("beta2", 0.95)))
    def advance(current):
        if gradient is None:
            return current
        return {mkey: current[mkey].lerp(gradient, 1 - beta1), vkey: current[vkey] * beta2 + gradient.square() * (1 - beta2)}
    reference = adam_direction(advance(values), group, step + int(gradient is not None))
    for recipe, vtype in CODECS:
        changed = {mkey: roundtrip(values[mkey], mkey, recipe)[0], vkey: roundtrip(values[vkey], vkey, recipe, vtype)[0]}
        direction = adam_direction(advance(changed), group, step + int(gradient is not None))
        emit(output, {**label, "kind": "adam_next_gradient_direction" if gradient is not None else "adam_snapshot_direction", "excludes": "learning rate and common decoupled weight decay; no loss evaluation", "codec": recipe, "second_format": str(vtype), "step": step, "betas": [beta1, beta2], "epsilon": group.get("eps", group.get("epsilon")), "bias_correction": group.get("bias_correction", True), **metrics(reference, direction)})


def raw_optimizers(optimizer):
    if hasattr(optimizer, "chained_optimizers"):
        for child in optimizer.chained_optimizers:
            yield from raw_optimizers(child)
    elif hasattr(optimizer, "optimizer"):
        yield optimizer.optimizer
    else:
        yield optimizer


def master_parameter_names(optimizer, names):
    if hasattr(optimizer, "chained_optimizers"):
        for child in optimizer.chained_optimizers:
            master_parameter_names(child, names)
    if hasattr(optimizer, "float16_groups"):
        for originals, masters in zip(optimizer.float16_groups, optimizer.fp32_from_float16_groups):
            for original, master in zip(originals, masters):
                names[id(master)] = names.get(id(original))


def ema_rows(values, group, gradient, label, output):
    betas = group.get("betas", (group.get("beta1", 0.9), group.get("beta2", 0.95)))
    beta1, beta2 = betas[:2]
    for name, value in values.items():
        if name in ("exp_avg", "momentum"):
            next_value = value.lerp(gradient, 1 - beta1)
        elif name in ("exp_avg_sq", "variance"):
            next_value = value * beta2 + gradient.square() * (1 - beta2)
        elif name == "momentum_buffer":
            next_value = value.lerp(gradient, 1 - group["momentum"])
        elif name == "exp_avg_slow":
            next_value = value.lerp(gradient, 1 - betas[2])
        else:
            continue
        changed = next_value != value
        for recipe in ("maxabs", "dre", "coat"):
            before = roundtrip(value, name, recipe)[0]
            after = roundtrip(next_value, name, recipe)[0]
            emit(output, {**label, "kind": "one_step_ema_resolution", "state": name, "codec": recipe, "reference_changed_count": int(changed.sum()), "lost_change_fraction": float(((after == before) & changed).sum() / changed.sum().clamp_min(1)), "meaning": "identical decoded values despite a nonzero FP32 EMA change; not payload-code equality", **metrics(next_value - value, after - before)})


def snr(value, dims):
    """SlimAdam's SNR_K (Kalra et al. 2025, eq. 3): mean over the other dimensions of mean_K^2 / var_K."""
    mean = value.mean(dim=dims, keepdim=True)
    var = value.var(dim=dims, keepdim=True, unbiased=False)
    return float((mean.square() / var.clamp_min(1e-30)).mean())


def snr_rows(current, names, block, output):
    """SNR of every matrix second moment along fan_in (dim 1) and fan_out (dim 0): which dimension SlimAdam may
    average over at this point of training, for every layer of this model rather than the paper's GPT."""
    for gi, group in enumerate(current.param_groups):
        for position, parameter in enumerate(group["params"]):
            state = current.state.get(parameter, {})
            if "exp_avg_sq" not in state or not torch.is_tensor(state["exp_avg_sq"]) or parameter.dim() != 2:
                continue
            value = materialize(state, "exp_avg_sq")
            emit(output, {"kind": "second_moment_snr", "block": block, "group": gi, "position": position, "name": names.get(id(parameter)),
                          "shape": list(value.shape), "source_dtype": str(state["exp_avg_sq"].dtype),
                          "snr_fan_in": snr(value, (1,)), "snr_fan_out": snr(value, (0,)), "snr_all": snr(value, (0, 1))})
            del value


def install_live_diagnostics(optimizer, models, output_path, samples=3):
    """Attach one read-only diagnostic to each raw optimizer's first real step."""
    global STORED_CODEC
    if os.environ.get("STAGE3_MOE_ADAM_STATE_CODEC", "").strip().lower() == "coat":
        STORED_CODEC = "coat"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = {}
    for module in models:
        while hasattr(module, "module"):
            module = module.module
        names.update({id(p): name for name, p in module.named_parameters()})
    master_parameter_names(optimizer, names)
    emit(output, {"kind": "protocol", "mode": "live_first_gradient", "group_size": 128, "selection": "first/middle/last per optimizer group, shape and state keys", "samples_per_stratum": samples, "warning": "separate diagnostic run; not valid peak-memory or throughput measurement"})

    for block, raw in enumerate(raw_optimizers(optimizer)):
        fired = [False]
        def inspect_step(current, args, kwargs, block=block, fired=fired):
            if fired[0]:
                return
            fired[0] = True
            with torch.no_grad():
                snr_rows(current, names, block, output)
                for gi, position, parameter, keys in selected_parameters(current.param_groups, current.state, samples):
                    if parameter.grad is None:
                        continue
                    group = current.param_groups[gi]
                    label = {"block": block, "group": gi, "position": position, "name": names.get(id(parameter)), "optimizer": type(current).__name__}
                    state = current.state[parameter]
                    values = state_rows(state, label, output)
                    gradient = parameter.grad.detach().float()
                    ema_rows(values, group, gradient, label, output)
                    if "momentum_buffer" in values and hasattr(current, "orthogonalize"):
                        momentum = group["momentum"]
                        def direction(buffer):
                            updated = buffer.lerp(gradient, 1 - momentum)
                            incoming = gradient.lerp(updated, momentum) if current.nesterov else updated
                            group_kwargs = {k: v for k, v in group.items() if k != "params"}
                            return current.orthogonalize(parameter, incoming, **group_kwargs).float()
                        previous_precision = torch.get_float32_matmul_precision()
                        torch.set_float32_matmul_precision(current.fp32_matmul_prec)
                        try:
                            reference = direction(values["momentum_buffer"])
                            for recipe in ("maxabs", "dre", "coat"):
                                restored = roundtrip(values["momentum_buffer"], "momentum_buffer", recipe)[0]
                                actual = direction(restored)
                                emit(output, {**label, "kind": "muon_next_gradient_orthogonalized_direction", "codec": recipe, "nesterov": current.nesterov, "excludes": "learning rate and common decoupled weight decay; no loss evaluation", **metrics(reference, actual)})
                        finally:
                            torch.set_float32_matmul_precision(previous_precision)
                    else:
                        adam_rows(values, state, group, label, output, gradient)
                    del values
                torch.cuda.synchronize()
                emit(output, {"kind": "live_block_complete", "block": block})
        raw.register_step_pre_hook(inspect_step)


def install_m2_trace(optimizer, models, output_dir, snapshot_steps, samples=3, sample_size=1 << 20):
    """Step-by-step record of AdEMAMix's slow EMA m2 (m1 as the control) on the live run.

    For first/middle/last parameters of every (group, shape) stratum, a fixed sample of elements is decoded
    after every optimizer step: `changed_m2[i]` is the fraction of sampled elements whose decoded value
    changed in step i + 1 (step 1 against the zero initial state). At the snapshot steps the full-tensor norms
    and the sampled values are kept, and the record is saved to OUTPUT_DIR/m2trace-<mode>-b<block>-<job>.pt,
    so an FP8 run is compared with the FP32-state run of the same probe (scripts/coat_campaign.py).
    """
    global STORED_CODEC
    if os.environ.get("STAGE3_MOE_ADAM_STATE_CODEC", "").strip().lower() == "coat":
        STORED_CODEC = "coat"
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = {}
    for module in models:
        while hasattr(module, "module"):
            module = module.module
        names.update({id(p): name for name, p in module.named_parameters()})
    master_parameter_names(optimizer, names)
    job = (os.environ.get("MLSUB_JOB_NAME") or os.environ.get("HOSTNAME", "job"))[-12:]

    for block, raw in enumerate(raw_optimizers(optimizer)):
        if "AdEMAMix" not in type(raw).__name__:
            continue
        trace = {"step": 0}

        def after_step(current, args, kwargs, block=block, trace=trace):
            trace["step"] += 1
            step = trace["step"]
            with torch.no_grad():
                if step == 1:
                    spec = next((s for s in getattr(current, "state_specs", ()) if s.name == "exp_avg_slow"), None)
                    # "fp32": plain AdEMAMix (all states FP32), the reference; "m2fp32": FP8 m1/v with an FP32 m2.
                    mode = "fp32" if spec is None else "m2fp32" if spec.dtype == torch.float32 else ("e4m3-sr" if spec.stochastic else "e4m3-rtn")
                    trace["tracked"] = []
                    for gi, position, parameter, keys in selected_parameters(current.param_groups, current.state, samples):
                        if "exp_avg_slow" in keys:
                            index = torch.arange(0, parameter.numel(), max(1, parameter.numel() // sample_size), device=parameter.device)
                            trace["tracked"].append((names.get(id(parameter)) or f"{gi}:{position}", parameter, index))
                    trace["previous"] = {name: (0.0, 0.0) for name, _, _ in trace["tracked"]}
                    trace["label"] = f"{mode}-b{block}-{job}"
                    trace["data"] = {"label": trace["label"], "mode": mode, "beta3": current.param_groups[0]["betas"][2],
                                     "params": {name: {"shape": list(p.shape), "changed_m2": [], "changed_m1": [], "snapshots": {}}
                                                for name, p, _ in trace["tracked"]}}
                for name, parameter, index in trace["tracked"]:
                    state = current.state[parameter]
                    slow, fast = materialize(state, "exp_avg_slow"), materialize(state, "exp_avg")
                    slow_sample, fast_sample = slow.view(-1)[index], fast.view(-1)[index]
                    record = trace["data"]["params"][name]
                    previous_slow, previous_fast = trace["previous"][name]
                    record["changed_m2"].append(float((slow_sample != previous_slow).float().mean()))
                    record["changed_m1"].append(float((fast_sample != previous_fast).float().mean()))
                    trace["previous"][name] = (slow_sample, fast_sample)
                    if step in snapshot_steps:
                        record["snapshots"][step] = {
                            "norm_m2": float(torch.linalg.vector_norm(slow, dtype=torch.float64)),
                            "norm_m1": float(torch.linalg.vector_norm(fast, dtype=torch.float64)),
                            "m2": slow_sample.cpu(), "m1": fast_sample.cpu()}
                    del slow, fast
                if step in snapshot_steps:
                    torch.save(trace["data"], output / f"m2trace-{trace['label']}.pt")

        raw.register_step_post_hook(after_step)


def checkpoint_blocks(value):
    if isinstance(value, (list, tuple)):
        for block in value:
            yield from checkpoint_blocks(block)
    elif "state" in value and "param_groups" in value:
        yield value
    elif "optimizer" in value:
        yield from checkpoint_blocks(value["optimizer"])
    else:
        raise ValueError(f"unsupported optimizer checkpoint keys: {list(value)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-stratum", type=int, default=3)
    parser.add_argument("--stored-codec", choices=("auto", "maxabs", "dre", "coat"), default="auto")
    options = parser.parse_args()
    global STORED_CODEC
    STORED_CODEC = options.stored_codec
    if options.samples_per_stratum < 1:
        parser.error("samples-per-stratum must be positive")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    if options.output.exists():
        parser.error("output already exists; use a new path to preserve provenance")
    start = time.monotonic()
    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    emit(options.output, {"kind": "protocol", "mode": "checkpoint_snapshot", "checkpoint": str(options.checkpoint), "file_bytes": options.checkpoint.stat().st_size, "iteration": checkpoint.get("iteration", checkpoint.get("itr")), "group_size": 128, "codec_implementation": "stage3_moe.optimizer_states", "selection": "first/middle/last per optimizer group, shape and state keys", "samples_per_stratum": options.samples_per_stratum, "warning": "no gradients: snapshot distortion and Adam snapshot direction only; not the actual next update or loss"})
    count = 0
    for block_id, block in enumerate(checkpoint_blocks(checkpoint["optimizer"])):
        for gi, position, parameter, keys in selected_parameters(block["param_groups"], block["state"], options.samples_per_stratum):
            state, group = block["state"][parameter], block["param_groups"][gi]
            label = {"block": block_id, "group": gi, "position": position, "parameter_id": str(parameter)}
            values = state_rows(state, label, options.output)
            adam_rows(values, state, group, label, options.output)
            count += 1
            del values
    torch.cuda.synchronize()
    if count == 0:
        raise RuntimeError("checkpoint contains no recognized optimizer tensors")
    emit(options.output, {"kind": "complete", "parameters_sampled": count, "elapsed_seconds": time.monotonic() - start})
    print(f"QUANTIZATION_DIAGNOSTICS=PASS parameters={count} output={options.output}", flush=True)


if __name__ == "__main__":
    main()
