import hashlib
import json
import os
from pathlib import Path

import torch


class RoutingAudit:
    def __init__(self, model, output_path):
        from megatron.training import get_args

        args = get_args()
        self.output_path = Path(output_path)
        self.microbatches_per_global_batch = args.eval_global_batch_size // (
            args.eval_micro_batch_size * args.data_parallel_size
        )
        self.layers = []
        self.evaluations = []
        for chunk_index, chunk in enumerate(model):
            for name, module in chunk.named_modules():
                if torch.is_tensor(getattr(module, "expert_bias", None)):
                    self._install_layer(f"model_chunk{chunk_index}.{name}", module)

    def _install_layer(self, name, module):
        experts = module.config.num_moe_experts
        device = module.expert_bias.device
        state = {
            "name": name,
            "module": module,
            "tokens": torch.zeros((), dtype=torch.int64, device=device),
            "actual": torch.zeros(experts, dtype=torch.int64, device=device),
            "frozen": torch.zeros(experts, dtype=torch.int64, device=device),
            "unbiased": torch.zeros(experts, dtype=torch.int64, device=device),
            "adaptive_pending": torch.zeros(experts, dtype=torch.int64, device=device),
            "raw_sum": torch.zeros(experts, dtype=torch.float64, device=device),
            "raw_square_sum": torch.zeros(experts, dtype=torch.float64, device=device),
            "bias_changes": torch.zeros((), dtype=torch.int64, device=device),
            "adaptive_changes": torch.zeros((), dtype=torch.int64, device=device),
            "routing_mismatches": torch.zeros((), dtype=torch.int64, device=device),
            "bias_updates": torch.zeros((), dtype=torch.int64, device=device),
            "margin_sum": torch.zeros((), dtype=torch.float64, device=device),
            "margin_min": torch.full((), float("inf"), dtype=torch.float64, device=device),
            "margin_below_1e3": torch.zeros((), dtype=torch.int64, device=device),
            "margin_below_1e2": torch.zeros((), dtype=torch.int64, device=device),
            "checkpoint_bias": module.expert_bias.detach().clone(),
            "microbatch_calls": 0,
        }
        self.layers.append(state)
        original = module.routing

        def audited_routing(logits, padding_mask=None, _state=state, _module=module):
            probs, routing_map = original(logits, padding_mask=padding_mask)
            with torch.no_grad():
                flat_logits = logits.detach().reshape(-1, experts).float()
                flat_actual = routing_map.detach().reshape(-1, experts).bool()
                if padding_mask is not None:
                    active = ~padding_mask.detach().reshape(-1).bool()
                    flat_logits = flat_logits[active]
                    flat_actual = flat_actual[active]
                raw_scores = torch.sigmoid(flat_logits)
                biased_scores = raw_scores + _module.expert_bias.detach().float()
                frozen_scores = raw_scores + _state["checkpoint_bias"]
                unbiased_indices = torch.topk(raw_scores, _module.topk, dim=-1).indices
                biased_topk = torch.topk(biased_scores, _module.topk + 1, dim=-1)
                biased_indices = biased_topk.indices[:, : _module.topk]
                frozen_indices = torch.topk(frozen_scores, _module.topk, dim=-1).indices

                unbiased_map = torch.zeros_like(flat_actual)
                unbiased_map.scatter_(1, unbiased_indices, True)
                frozen_map = torch.zeros_like(flat_actual)
                frozen_map.scatter_(1, frozen_indices, True)
                computed_map = torch.zeros_like(flat_actual)
                computed_map.scatter_(1, biased_indices, True)
                margins = biased_topk.values[:, _module.topk - 1] - biased_topk.values[:, _module.topk]

                _state["tokens"] += flat_logits.shape[0]
                _state["actual"] += flat_actual.sum(dim=0)
                _state["frozen"] += frozen_map.sum(dim=0)
                _state["unbiased"] += unbiased_map.sum(dim=0)
                _state["adaptive_pending"] += flat_actual.sum(dim=0)
                _state["raw_sum"] += raw_scores.sum(dim=0, dtype=torch.float64)
                _state["raw_square_sum"] += (raw_scores * raw_scores).sum(
                    dim=0, dtype=torch.float64
                )
                _state["bias_changes"] += (flat_actual ^ unbiased_map).sum() // 2
                _state["adaptive_changes"] += (flat_actual ^ frozen_map).sum() // 2
                _state["routing_mismatches"] += (flat_actual ^ computed_map).sum() // 2
                _state["margin_sum"] += margins.sum(dtype=torch.float64)
                _state["margin_min"] = torch.minimum(
                    _state["margin_min"], margins.min().to(torch.float64)
                )
                _state["margin_below_1e3"] += (margins <= 1e-3).sum()
                _state["margin_below_1e2"] += (margins <= 1e-2).sum()
                _state["microbatch_calls"] += 1
                if _state["microbatch_calls"] % self.microbatches_per_global_batch == 0:
                    counts = _state["adaptive_pending"]
                    offset = counts.to(torch.float32).mean() - counts
                    _module.expert_bias.add_(
                        torch.sign(offset) * _module.config.moe_router_bias_update_rate
                    )
                    counts.zero_()
                    _state["bias_updates"] += 1
            return probs, routing_map

        module.routing = audited_routing

    def reset(self):
        for state in self.layers:
            state["module"].expert_bias.copy_(state["checkpoint_bias"])
            state["microbatch_calls"] = 0
            for key, value in state.items():
                if not torch.is_tensor(value):
                    continue
                if key == "checkpoint_bias":
                    continue
                if key == "margin_min":
                    value.fill_(float("inf"))
                else:
                    value.zero_()

    @staticmethod
    def _balance(counts):
        counts = counts.to(torch.float64)
        mean = counts.mean()
        return {
            "minimum_to_mean": float(counts.min() / mean),
            "maximum_to_mean": float(counts.max() / mean),
            "coefficient_of_variation": float(counts.std(unbiased=False) / mean),
        }

    def record(self, split, loss):
        rows = []
        for state in self.layers:
            reduced = {}
            for key, value in state.items():
                if not torch.is_tensor(value):
                    continue
                if key == "checkpoint_bias":
                    continue
                reduced[key] = value.detach().clone()
                if torch.distributed.is_initialized():
                    op = (
                        torch.distributed.ReduceOp.MIN
                        if key == "margin_min"
                        else torch.distributed.ReduceOp.SUM
                    )
                    torch.distributed.all_reduce(reduced[key], op=op)

            tokens = int(reduced["tokens"].item())
            topk = state["module"].topk
            raw_mean = reduced["raw_sum"] / tokens
            raw_variance = reduced["raw_square_sum"] / tokens - raw_mean.square()
            checkpoint_bias = state["checkpoint_bias"].float().cpu()
            final_bias = state["module"].expert_bias.detach().float().cpu()
            actual = reduced["actual"].cpu()
            frozen = reduced["frozen"].cpu()
            unbiased = reduced["unbiased"].cpu()
            rows.append(
                {
                    "layer": state["name"],
                    "tokens": tokens,
                    "checkpoint_expert_bias": checkpoint_bias.tolist(),
                    "checkpoint_expert_bias_sha256": hashlib.sha256(
                        checkpoint_bias.numpy().tobytes()
                    ).hexdigest(),
                    "final_adaptive_expert_bias": final_bias.tolist(),
                    "maximum_absolute_bias_change": float(
                        (final_bias - checkpoint_bias).abs().max()
                    ),
                    "bias_updates": int(reduced["bias_updates"].item()),
                    "raw_score_mean": raw_mean.cpu().tolist(),
                    "raw_score_std": raw_variance.clamp_min(0).sqrt().cpu().tolist(),
                    "actual_counts": actual.tolist(),
                    "frozen_counts": frozen.tolist(),
                    "unbiased_counts": unbiased.tolist(),
                    "actual_balance": self._balance(actual),
                    "frozen_balance": self._balance(frozen),
                    "unbiased_balance": self._balance(unbiased),
                    "assignment_fraction_changed_by_bias": float(
                        reduced["bias_changes"] / (tokens * topk)
                    ),
                    "assignment_fraction_changed_by_adaptation": float(
                        reduced["adaptive_changes"] / (tokens * topk)
                    ),
                    "computed_routing_mismatch_fraction": float(
                        reduced["routing_mismatches"] / (tokens * topk)
                    ),
                    "biased_topk_margin_mean": float(reduced["margin_sum"] / tokens),
                    "biased_topk_margin_min": float(reduced["margin_min"]),
                    "biased_topk_margin_fraction_below_1e3": float(
                        reduced["margin_below_1e3"] / tokens
                    ),
                    "biased_topk_margin_fraction_below_1e2": float(
                        reduced["margin_below_1e2"] / tokens
                    ),
                }
            )

        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        mismatch = max(row["computed_routing_mismatch_fraction"] for row in rows)
        if mismatch != 0:
            raise AssertionError(f"routing reconstruction mismatch: {mismatch}")
        from megatron.training import get_args

        args = get_args()
        self.evaluations.append(
            {
                "split": split,
                "loss": loss,
                "layers": rows,
                "worst_actual_cv": max(
                    row["actual_balance"]["coefficient_of_variation"] for row in rows
                ),
                "worst_frozen_cv": max(
                    row["frozen_balance"]["coefficient_of_variation"] for row in rows
                ),
                "worst_unbiased_cv": max(
                    row["unbiased_balance"]["coefficient_of_variation"] for row in rows
                ),
                "worst_actual_minimum_to_mean": min(
                    row["actual_balance"]["minimum_to_mean"] for row in rows
                ),
            }
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "checkpoint_label": os.environ["STAGE3_ROUTING_CHECKPOINT_LABEL"],
                    "checkpoint_iteration": args.iteration,
                    "seed": args.seed,
                    "eval_iters": args.eval_iters,
                    "eval_global_batch_size": args.eval_global_batch_size,
                    "eval_micro_batch_size": args.eval_micro_batch_size,
                    "routing_protocol": "adaptive expert-bias replay with frozen weights",
                    "valid_data_path": args.valid_data_path,
                    "test_data_path": args.test_data_path,
                    "evaluations": self.evaluations,
                },
                sort_keys=True,
            )
            + "\n"
        )


def install(output_path):
    import megatron.training.training as training

    audit = {}
    last_loss = {}
    original_setup = training.setup_model_and_optimizer

    def setup(*args, **kwargs):
        model, optimizer, scheduler = original_setup(*args, **kwargs)
        audit["collector"] = RoutingAudit(model, output_path)
        return model, optimizer, scheduler

    training.setup_model_and_optimizer = setup
    original_evaluate = training.evaluate

    def evaluate(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        loss = (result[0] or {}).get("lm loss")
        last_loss["lm loss"] = None if loss is None else float(loss)
        return result

    training.evaluate = evaluate
    original_report = training.evaluate_and_print_results

    def report(prefix, *args, **kwargs):
        collector = audit["collector"]
        collector.reset()
        value = original_report(prefix, *args, **kwargs)
        split = "test" if "test set" in str(prefix) else "validation"
        collector.record(split, last_loss.get("lm loss"))
        return value

    training.evaluate_and_print_results = report
