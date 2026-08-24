import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from emerging_optimizers import utils
from megatron.core.optimizer.emerging_optimizers import (
    TensorParallelMuon,
    _EMERGING_OPTIMIZERS,
)
from megatron.core.optimizer.optimizer_config import ParamKey, ParamWithNamePredicate

from stage3_moe.optimizer_states import (
    FP8StateOptimizerMixin,
    GROUP_SIZE,
    MUON_STATE_SPECS,
    StateSpec,
    dequantize_fp8_state,
    init_fp8_state,
    quantize_fp8_state_,
)


SPLIT_SWIGLU_FC1 = "stage3_split_swiglu_fc1"


def is_router_weight(param: torch.Tensor, name: str) -> bool:
    return len(param.shape) == 2 and name.endswith(".router.weight")


def is_swiglu_fc1_weight(param: torch.Tensor, name: str) -> bool:
    return len(param.shape) == 2 and ".linear_fc1.weight" in name


class SplitSwiGLUTensorParallelMuon(TensorParallelMuon):
    def orthogonalize(
        self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        split_fc1 = kwargs.pop(SPLIT_SWIGLU_FC1, False)
        if not split_fc1:
            return super().orthogonalize(p, grad, **kwargs)
        if grad.ndim != 2 or grad.shape[0] % 2:
            raise ValueError(
                f"SwiGLU FC1 must be an even 2D matrix, got {tuple(grad.shape)}"
            )
        gate, up = grad.chunk(2, dim=0)
        parent = super()
        return torch.cat(
            (
                parent.orthogonalize(p, gate, **kwargs),
                parent.orthogonalize(p, up, **kwargs),
            ),
            dim=0,
        )


class FP8StateSplitSwiGLUTensorParallelMuon(
    FP8StateOptimizerMixin, SplitSwiGLUTensorParallelMuon
):
    state_specs = MUON_STATE_SPECS


def _shadow_category(name: str) -> str | None:
    if ".self_attention.linear_qkv.weight" in name:
        return "attention_qkv"
    if ".decoder.layers.0.mlp.linear_fc1.weight" in name:
        return "dense_mlp_fc1"
    if ".mlp.experts.linear_fc1.weight0" in name:
        return "routed_expert_fc1"
    return None


def _tensor_error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.float().reshape(-1)
    candidate = candidate.float().reshape(-1)
    reference_norm = torch.linalg.vector_norm(reference)
    candidate_norm = torch.linalg.vector_norm(candidate)
    denominator = reference_norm.clamp_min(1e-30)
    cosine_denominator = (reference_norm * candidate_norm).clamp_min(1e-30)
    values = torch.stack(
        (
            torch.dot(reference, candidate) / cosine_denominator,
            torch.linalg.vector_norm(candidate - reference) / denominator,
            candidate_norm / denominator,
        )
    ).cpu()
    cosine, relative_l2, norm_ratio = (float(value) for value in values)
    return {
        "cosine": cosine,
        "relative_l2": relative_l2,
        "norm_ratio": norm_ratio,
    }


def _shadow_seed(name: str, step: int) -> int:
    digest = hashlib.sha256(f"{name}:{step}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _stochastic_maxabs_roundtrip(
    value: torch.Tensor, *, group_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = value.float().contiguous().view(-1)
    padding = (-flat.numel()) % group_size
    if padding:
        flat = torch.cat((flat, flat.new_zeros(padding)))
    groups = flat.view(-1, group_size)
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    scale = (groups.abs().amax(dim=1, keepdim=True) / fp8_max).clamp_min(1e-30)
    normalized = groups / scale
    magnitude = normalized.abs().clamp_max(fp8_max)
    normal_min = 2.0**-6
    spacing = torch.where(
        magnitude < normal_min,
        torch.full_like(magnitude, 2.0**-9),
        torch.exp2(torch.floor(torch.log2(magnitude.clamp_min(normal_min))) - 3),
    )
    lower = torch.floor(magnitude / spacing) * spacing
    upper = torch.minimum(lower + spacing, torch.full_like(lower, fp8_max))
    probability = torch.where(upper > lower, (magnitude - lower) / (upper - lower), 0.0)
    generator = torch.Generator(device=value.device)
    generator.manual_seed(seed)
    draw = torch.rand(
        probability.shape,
        dtype=probability.dtype,
        device=probability.device,
        generator=generator,
    )
    payload = torch.where(draw < probability, upper, lower) * normalized.sign()
    restored = (payload * scale).view(-1)[: value.numel()].view_as(value)
    return restored, payload.view(-1)[: value.numel()].view_as(value)


class MuonFP8ShadowDiagnostic(SplitSwiGLUTensorParallelMuon):
    def configure_shadow(self, parameter_names: dict[int, str], path: Path) -> None:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise ValueError("Muon FP8 shadow diagnostic requires one process")

        selected = {}
        candidates = sorted(
            (
                name,
                parameter,
                _shadow_category(name),
            )
            for group in self.param_groups
            for parameter in group["params"]
            if (name := parameter_names.get(id(parameter))) is not None
        )
        for name, parameter, category in candidates:
            if category is not None and category not in selected:
                selected[category] = (name, parameter)
        expected = {"attention_qkv", "dense_mlp_fc1", "routed_expert_fc1"}
        if set(selected) != expected:
            raise AssertionError(f"Muon FP8 shadow parameters missing: {sorted(expected - set(selected))}")

        spec = MUON_STATE_SPECS[0]
        self._shadow_quantizer = os.environ.get(
            "STAGE3_MOE_MUON_SHADOW_QUANTIZER", "deterministic"
        )
        if self._shadow_quantizer not in {
            "deterministic",
            "dre",
            "dre2",
            "stochastic",
        }:
            raise ValueError(f"unknown Muon shadow quantizer: {self._shadow_quantizer}")
        self._shadow_highest_ns = os.environ.get(
            "STAGE3_MOE_MUON_SHADOW_HIGHEST_NS", "0"
        ) == "1"
        self._shadow_group_size = int(
            os.environ.get("STAGE3_MOE_MUON_SHADOW_GROUP_SIZE", str(GROUP_SIZE))
        )
        if self._shadow_group_size not in {32, GROUP_SIZE, 256}:
            raise ValueError(
                f"unknown Muon shadow group size: {self._shadow_group_size}"
            )
        self._shadow_spec = spec
        if self._shadow_quantizer in {"dre", "dre2"}:
            self._shadow_spec = StateSpec(
                spec.name, spec.signed, spec.dtype, "dre"
            )
        self._shadow_parameters = {}
        for category, (name, parameter) in selected.items():
            reference = self.state[parameter].get(spec.name)
            if reference is None or reference.dtype != torch.float32:
                raise AssertionError(f"Muon FP32 checkpoint state missing for {name}")
            info = {
                "category": category,
                "name": name,
            }
            if self._shadow_quantizer != "stochastic":
                shadow_state = {}
                init_fp8_state(
                    shadow_state,
                    self._shadow_spec,
                    reference,
                    group_size=self._shadow_group_size,
                )
                quantize_fp8_state_(
                    shadow_state,
                    self._shadow_spec,
                    reference,
                    group_size=self._shadow_group_size,
                )
                info["state"] = shadow_state
                if self._shadow_quantizer == "dre2":
                    primary = dequantize_fp8_state(
                        shadow_state,
                        self._shadow_spec,
                        group_size=self._shadow_group_size,
                    )
                    residual_state = {}
                    init_fp8_state(
                        residual_state,
                        self._shadow_spec,
                        reference,
                        group_size=self._shadow_group_size,
                    )
                    quantize_fp8_state_(
                        residual_state,
                        self._shadow_spec,
                        reference - primary,
                        group_size=self._shadow_group_size,
                    )
                    info["residual_state"] = residual_state
            else:
                info["persistent"], _ = _stochastic_maxabs_roundtrip(
                    reference,
                    group_size=self._shadow_group_size,
                    seed=_shadow_seed(name, 0),
                )
            self._shadow_parameters[parameter] = info

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        self._shadow_path = path
        self._shadow_step = 0
        self._shadow_pending = {}

    def _prepare_shadow_step(self) -> None:
        if not hasattr(self, "_shadow_parameters"):
            return
        self._shadow_step += 1
        spec = self._shadow_spec
        fp8_max = float(torch.finfo(spec.dtype).max)
        for group in self.param_groups:
            beta = group["momentum"]
            for parameter in group["params"]:
                info = self._shadow_parameters.get(parameter)
                if info is None or parameter.grad is None:
                    continue
                grad = parameter.grad.detach()
                reference_previous = self.state[parameter][spec.name]
                if self._shadow_quantizer != "stochastic":
                    shadow_previous = dequantize_fp8_state(
                        info["state"], spec, group_size=self._shadow_group_size
                    )
                    if self._shadow_quantizer == "dre2":
                        shadow_previous += dequantize_fp8_state(
                            info["residual_state"],
                            spec,
                            group_size=self._shadow_group_size,
                        )
                else:
                    shadow_previous = info["persistent"]
                reference_new = torch.lerp(reference_previous, grad, 1 - beta)
                shadow_new = torch.lerp(shadow_previous, grad, 1 - beta)
                if self._shadow_quantizer != "stochastic":
                    quantize_fp8_state_(
                        info["state"],
                        spec,
                        shadow_new,
                        group_size=self._shadow_group_size,
                    )
                    persisted_shadow = dequantize_fp8_state(
                        info["state"], spec, group_size=self._shadow_group_size
                    )
                    saturation_fractions = [
                        (info["state"][spec.name].float().abs() == fp8_max)
                        .float()
                        .mean()
                    ]
                    if self._shadow_quantizer == "dre2":
                        quantize_fp8_state_(
                            info["residual_state"],
                            spec,
                            shadow_new - persisted_shadow,
                            group_size=self._shadow_group_size,
                        )
                        persisted_shadow += dequantize_fp8_state(
                            info["residual_state"],
                            spec,
                            group_size=self._shadow_group_size,
                        )
                        saturation_fractions.append(
                            (
                                info["residual_state"][spec.name].float().abs()
                                == fp8_max
                            )
                            .float()
                            .mean()
                        )
                    saturation_fraction = float(
                        torch.stack(saturation_fractions).mean()
                    )
                else:
                    persisted_shadow, payload = _stochastic_maxabs_roundtrip(
                        shadow_new,
                        group_size=self._shadow_group_size,
                        seed=_shadow_seed(info["name"], self._shadow_step),
                    )
                    info["persistent"] = persisted_shadow
                    saturation_fraction = float(
                        (payload.abs() == fp8_max).float().mean()
                    )
                reference_input = torch.lerp(grad, reference_new, beta)
                shadow_input = torch.lerp(grad, shadow_new, beta)
                state_metrics = _tensor_error_metrics(reference_new, persisted_shadow)
                reference_nonzero = reference_new != 0
                state_metrics["underflow_fraction"] = float(
                    (reference_nonzero & (persisted_shadow == 0)).sum()
                    / reference_nonzero.sum().clamp_min(1)
                )
                state_metrics["saturation_fraction"] = saturation_fraction
                self._shadow_pending[parameter] = {
                    "info": info,
                    "reference_input": reference_input,
                    "shadow_input": shadow_input,
                    "state": state_metrics,
                }

    def orthogonalize(
        self, parameter: torch.Tensor, grad: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        reference_update = super().orthogonalize(parameter, grad, **kwargs)
        pending = getattr(self, "_shadow_pending", {}).pop(parameter, None)
        if pending is None:
            return reference_update

        shadow_update = super().orthogonalize(
            parameter, pending["shadow_input"], **kwargs
        )
        reference_replay = _tensor_error_metrics(grad, pending["reference_input"])
        pre_ns = _tensor_error_metrics(grad, pending["shadow_input"])
        post_ns = _tensor_error_metrics(reference_update, shadow_update)
        record = {
            "schema_version": 1,
            "step": self._shadow_step,
            "category": pending["info"]["category"],
            "parameter_name": pending["info"]["name"],
            "shape": list(parameter.shape),
            "quantizer": self._shadow_quantizer,
            "group_size": self._shadow_group_size,
            "state_components": 2 if self._shadow_quantizer == "dre2" else 1,
            "state": pending["state"],
            "reference_replay": reference_replay,
            "pre_newton_schulz": pre_ns,
            "post_newton_schulz": post_ns,
            "ns_relative_error_amplification": post_ns["relative_l2"]
            / max(pre_ns["relative_l2"], 1e-30),
        }
        if self._shadow_highest_ns:
            with utils.fp32_matmul_precision("highest"):
                reference_highest = super().orthogonalize(parameter, grad, **kwargs)
                shadow_highest = super().orthogonalize(
                    parameter, pending["shadow_input"], **kwargs
                )
            post_ns_highest = _tensor_error_metrics(reference_highest, shadow_highest)
            record.update(
                {
                    "post_newton_schulz_highest": post_ns_highest,
                    "highest_ns_relative_error_amplification": post_ns_highest[
                        "relative_l2"
                    ]
                    / max(pre_ns["relative_l2"], 1e-30),
                    "reference_medium_vs_highest": _tensor_error_metrics(
                        reference_highest, reference_update
                    ),
                    "shadow_medium_vs_highest": _tensor_error_metrics(
                        shadow_highest, shadow_update
                    ),
                }
            )
        with self._shadow_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return reference_update

    @torch.no_grad()
    def step(self, closure=None):
        self._prepare_shadow_step()
        return super().step(closure)


def install_muon_shadow_probe(probe, path: Path) -> None:
    import megatron.training.training as training

    previous = training.setup_model_and_optimizer

    def setup_with_shadow(*args, **kwargs):
        model, optimizer, scheduler = previous(*args, **kwargs)
        from stage3_moe.result_writer import _raw_optimizers

        diagnostics = [
            raw
            for raw in _raw_optimizers(optimizer)
            if isinstance(raw, MuonFP8ShadowDiagnostic)
        ]
        if len(diagnostics) != 1:
            raise AssertionError(f"expected one Muon FP8 shadow optimizer, found {len(diagnostics)}")
        diagnostics[0].configure_shadow(probe.parameter_names, path)
        return model, optimizer, scheduler

    training.setup_model_and_optimizer = setup_with_shadow


def install_muon_contract(*, fp8_states: bool, shadow_states: bool = False) -> None:
    if fp8_states and shadow_states:
        raise ValueError("Muon FP8 shadow diagnostic requires FP32 optimizer state")
    entry = _EMERGING_OPTIMIZERS["muon"]
    overrides = dict(entry.default_param_overrides)
    overrides[
        ParamKey(
            with_name_predicate=ParamWithNamePredicate(
                name="stage3_router_adam_fallback", fn=is_router_weight
            )
        )
    ] = {"optimizer": "adam"}
    overrides[
        ParamKey(
            with_name_predicate=ParamWithNamePredicate(
                name="stage3_split_swiglu_fc1", fn=is_swiglu_fc1_weight
            )
        )
    ] = {SPLIT_SWIGLU_FC1: True}
    entry.default_param_overrides = overrides
    if fp8_states:
        entry.optimizer_cls = FP8StateSplitSwiGLUTensorParallelMuon
    elif shadow_states:
        entry.optimizer_cls = MuonFP8ShadowDiagnostic
    else:
        entry.optimizer_cls = SplitSwiGLUTensorParallelMuon
