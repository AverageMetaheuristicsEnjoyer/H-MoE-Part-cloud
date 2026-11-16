import json
import math
import os
from pathlib import Path
import re
import time

import torch

from stage3_moe.lionmuon import LionMuonMixin


class EnergyCalObserver:
    def __init__(self, parameter_names, output_dir):
        self.parameter_names = parameter_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "energycal_diagnostics.jsonl"
        self.blocks_path = self.output_dir / "energycal_blocks.jsonl"
        self.summary_path = self.output_dir / "energycal_summary.json"
        self.window_start = int(os.environ.get("STAGE3_MOE_ENERGYCAL_DIAGNOSTICS_START", "1"))
        self.window_end = int(os.environ.get("STAGE3_MOE_ENERGYCAL_DIAGNOSTICS_END", "20"))
        if self.window_start < 1 or self.window_end < self.window_start:
            raise ValueError("Invalid EnergyCal observer window")
        self.current = {}
        self.current_step = None
        self.records = []
        self.started = None
        self.callback_seconds = 0.0
        self.grad_none = 0
        self.materialized_zero = 0
        self.total_grad_params = 0
        self.expected_blocks_per_spectral_step = 31
        self.coverage = {
            "moe_layers": [1, 9, 17], "routed_experts": [0, 31, 63],
            "routed_blocks_per_spectral_step": 27,
            "controls": ["dense_fc1_gate", "dense_fc1_up", "shared_fc2", "attention_qkv_combined"],
            "full_tensor_dumps": False,
        }

    def enabled(self, iteration):
        return self.window_start <= iteration <= self.window_end

    def start_step(self, optimizer, iteration):
        self.current_step = iteration
        self.current = {}
        self.step_record_start = len(self.records)
        self.started = time.perf_counter()
        self.callback_seconds = 0.0
        self.grad_none = 0
        self.materialized_zero = 0
        self.total_grad_params = 0
        if (iteration - 1) % 5:
            return
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                self.total_grad_params += 1
                if parameter.grad is None:
                    self.grad_none += 1
                elif not torch.count_nonzero(parameter.grad):
                    self.materialized_zero += 1
        self.callback_seconds += time.perf_counter() - self.started

    def selected(self, name):
        layer = re.search(r"\.layers\.(\d+)\.", name)
        if not layer:
            return None
        layer_index = int(layer.group(1))
        if ".mlp.experts." in name:
            expert = re.search(r"\.weight(\d+)$", name)
            if layer_index not in (1, 9, 17) or not expert or int(expert.group(1)) not in (0, 31, 63):
                return None
            if ".linear_fc1." in name:
                return layer_index, int(expert.group(1)), "fc1"
            if ".linear_fc2." in name:
                return layer_index, int(expert.group(1)), "fc2"
            return None
        if layer_index == 0 and ".mlp.linear_fc1." in name:
            return layer_index, None, "dense_control"
        if layer_index == 1 and ".mlp.shared_experts.linear_fc2." in name:
            return layer_index, None, "shared_control"
        if layer_index == 1 and ".self_attention.linear_qkv." in name:
            return layer_index, None, "attention_control"
        return None

    def before_calibration(self, optimizer, parameter, update, group_kwargs, iteration):
        if (iteration - 1) % 5:
            return
        name = self.parameter_names.get(id(parameter), f"unnamed:{id(parameter)}")
        selected = self.selected(name)
        if selected is None:
            return
        split = group_kwargs.get("stage3_split_swiglu_fc1", False)
        blocks = update.chunk(2, dim=0) if split else (update,)
        gradients = parameter.grad.chunk(2, dim=0) if split else (parameter.grad,)
        momentum = optimizer.state[parameter]["momentum_buffer"]
        direction = momentum * optimizer.lion_beta1 + parameter.grad * (1 - optimizer.lion_beta1)
        directions = direction.chunk(2, dim=0) if split else (direction,)
        momenta = momentum.chunk(2, dim=0) if split else (momentum,)
        rows = []
        callback_started = time.perf_counter()
        for block, grad, old_direction, old_momentum in zip(blocks, gradients, directions, momenta):
            scale = 0.2 * math.sqrt(max(block.shape))
            rho = block.float().square().sum() / (scale * scale * min(block.shape))
            q = (rho / optimizer.lion_energycal_rho_ref).pow(optimizer.lion_energycal_exponent).clamp(
                optimizer.lion_energycal_qmin, 1)
            rows.append({"rho": float(rho), "q": float(q), "pre_norm": float(block.norm()),
                         "grad_norm": float(grad.norm()), "direction_norm": float(old_direction.norm()),
                         "old_momentum_norm": float(old_momentum.norm())})
        self.current[id(parameter)] = rows
        self.callback_seconds += time.perf_counter() - callback_started

    def after_calibration(self, optimizer, parameter, update, group_kwargs, iteration):
        if (iteration - 1) % 5 or id(parameter) not in self.current:
            return
        name = self.parameter_names.get(id(parameter), f"unnamed:{id(parameter)}")
        layer, expert, fc_type = self.selected(name)
        split = group_kwargs.get("stage3_split_swiglu_fc1", False)
        blocks = update.chunk(2, dim=0) if split else (update,)
        gradients = parameter.grad.chunk(2, dim=0) if split else (parameter.grad,)
        rows = self.current.pop(id(parameter))
        callback_started = time.perf_counter()
        for index, (block, grad, row) in enumerate(zip(blocks, gradients, rows)):
            post_norm = float(block.norm())
            actual_ratio = post_norm / row["pre_norm"] if row["pre_norm"] else (0.0 if post_norm == 0 else None)
            denominator = post_norm * row["grad_norm"]
            alignment = float((block.float() * grad.float()).sum()) / denominator if denominator else None
            routed = ".mlp.experts." in name
            applied = bool(optimizer.lion_energycal and routed)
            record = {
                "iteration": iteration, "layer": layer, "expert": expert,
                "fc_type": ({"fc1": ("fc1_gate", "fc1_up"), "dense_control": ("dense_fc1_gate", "dense_fc1_up")}
                            .get(fc_type, (fc_type, fc_type)))[index],
                "routed": routed, "applied": applied,
                "rho": None if fc_type == "attention_control" else row["rho"],
                "hypothetical_q": None if fc_type == "attention_control" else row["q"],
                "actual_q": actual_ratio if applied else None,
                "energy_scope": "combined_qkv_update_not_per_ns_block" if fc_type == "attention_control" else "logical_ns_block",
                "pre_update_norm": row["pre_norm"], "post_update_norm": post_norm,
                "grad_norm": row["grad_norm"], "direction_norm": row["direction_norm"],
                "old_momentum_norm": row["old_momentum_norm"],
                "update_over_grad_alignment": alignment,
            }
            record["nonfinite"] = any(value is not None and not math.isfinite(value)
                                      for value in record.values() if isinstance(value, float))
            for key, value in list(record.items()):
                if isinstance(value, float) and not math.isfinite(value):
                    record[key] = None
            self.records.append(record)
        self.callback_seconds += time.perf_counter() - callback_started

    def end_step(self, optimizer, iteration):
        if self.started is None:
            return
        observer_window_seconds = time.perf_counter() - self.started
        spectral = (iteration - 1) % 5 == 0
        expected = self.expected_blocks_per_spectral_step if spectral else 0
        selected_this_step = len(self.records) - self.step_record_start
        new_records = self.records[self.step_record_start:]
        with self.blocks_path.open("a") as stream:
            for item in new_records:
                stream.write(json.dumps(item, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        record = {"iteration": iteration, "spectral": spectral,
                  "grad_parameter_count": self.total_grad_params,
                  "grad_none": self.grad_none, "materialized_zero": self.materialized_zero,
                  "selected_records": selected_this_step, "missing_selected": max(0, expected - selected_this_step),
                  "observer_window_seconds": observer_window_seconds,
                  "observer_callback_seconds": self.callback_seconds,
                  "timing_scope": "window includes normal optimizer work; callbacks are not a clean incremental-overhead measurement",
                  "diagnostics_scope": "sampled_blocks_only; no full tensors retained"}
        with self.path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.current.clear()
        self.started = None
        if iteration == self.window_end:
            self.write_summary()
            self.records.clear()

    @staticmethod
    def quantile(values, fraction):
        values = sorted(values)
        position = fraction * (len(values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    def distribution(self, records, field):
        values = [record[field] for record in records if record.get(field) is not None and math.isfinite(record[field])]
        if not values:
            return None
        return {"min": values and min(values), "p10": self.quantile(values, 0.1),
                "p50": self.quantile(values, 0.5), "p90": self.quantile(values, 0.9),
                "max": max(values)}

    def summary_group(self, records):
        q = [record["hypothetical_q"] for record in records
             if record.get("hypothetical_q") is not None and math.isfinite(record["hypothetical_q"])]
        actual = [record["actual_q"] for record in records if record.get("actual_q") is not None and math.isfinite(record["actual_q"])]
        return {"samples": len(records), "rho": self.distribution(records, "rho"),
                "hypothetical_q": self.distribution(records, "hypothetical_q"),
                "actual_q": self.distribution(records, "actual_q"),
                "fraction_hypothetical_q_lt_0_99": sum(value < 0.99 for value in q) / len(q) if q else None,
                "fraction_hypothetical_q_lt_0_9": sum(value < 0.9 for value in q) / len(q) if q else None,
                "fraction_hypothetical_q_lt_0_5": sum(value < 0.5 for value in q) / len(q) if q else None,
                "fraction_hypothetical_q_zero": sum(value == 0 for value in q) / len(q) if q else None,
                "actual_q_samples": len(actual),
                "nonfinite": sum(record["nonfinite"] for record in records)}

    def write_summary(self):
        groups = {}
        for record in self.records:
            key = f"layer_{record['layer']:02d}/{record['fc_type']}"
            groups.setdefault(key, []).append(record)
            groups.setdefault("all_routed" if record["routed"] else "all_controls", []).append(record)
        summary = {"schema": "hmoe-lionmoe-energycal-measurement-summary-v2",
                   "window": [self.window_start, self.window_end],
                   "spectral_steps": sorted({record["iteration"] for record in self.records}),
                   "coverage": self.coverage, "groups": {key: self.summary_group(value) for key, value in sorted(groups.items())}}
        with self.summary_path.open("w") as stream:
            stream.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def install_measurement_observer(optimizer, parameter_names, output_dir):
    if os.environ.get("STAGE3_MOE_MEASUREMENT_ENERGYCAL_DIAGNOSTICS", "0") != "1":
        return None
    from stage3_moe.lionmuon import LionMuonMixin
    raw_optimizers = [wrapped.optimizer for wrapped in getattr(optimizer, "chained_optimizers", [])]
    lion = [raw for raw in raw_optimizers if isinstance(raw, LionMuonMixin)]
    if not lion:
        raise RuntimeError("No actual LionMuon optimizer found for measurement observer")
    observer = EnergyCalObserver(parameter_names, output_dir)
    for raw in lion:
        raw._lionmuon_measurement_observer = observer
    observer.metadata = {"raw_classes": [type(raw).__name__ for raw in raw_optimizers],
                         "lion_classes": [type(raw).__name__ for raw in lion]}
    return observer
