import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import traceback

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "Megatron-LM"))
sys.path.insert(0, str(ROOT / "third_party" / "emerging-optimizers"))

from stage3_moe import lionmuon
from stage3_moe.lionmuon import LionMuonMixin
from stage3_moe.muon import SplitSwiGLUTensorParallelMuon, install_muon_contract

SPEC = importlib.util.spec_from_file_location("lionmuon_c673aa8", ROOT / "tests/stage3_moe/lionmuon_c673aa8.py")
ORIGINAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORIGINAL)

DEVICE = "cuda"
CLOCK = {"iteration": None}
lionmuon.megatron_iteration = lambda: CLOCK["iteration"]
ORIGINAL.megatron_iteration = lambda: CLOCK["iteration"]
PG = None
MCORE_INSTALLED = False
NS_CAPTURE = []
NS_OWNER = None
FP32_RECONSTRUCTION_RTOL = 1e-6
FP32_RECONSTRUCTION_ATOL = 1e-8


def real_optimizer(mixin, specs, enabled=True):
    cls = type("RealLionMuon", (mixin, SplitSwiGLUTensorParallelMuon), {})
    groups = [{"params": [p], "stage3_split_swiglu_fc1": split, "stage3_routed_expert": routed,
               "label": label} for label, p, split, routed in specs]
    optimizer = cls(groups, lr=0.01, momentum=0.95, nesterov=True, weight_decay=0.1,
                    use_decoupled_weight_decay=True, split_qkv=False, is_qkv_fn=lambda _: False,
                    qkv_split_shapes=None, fp32_matmul_prec="medium", coefficient_type="quintic",
                    num_ns_steps=5, scale_mode="spectral", extra_scale_factor=0.2,
                    pg_collection=PG, tp_mode="duplicated")
    optimizer._soft_scale_mode = "spectral"
    optimizer._soft_extra_scale = 0.2
    optimizer.lion_period = [(0.0, 5)]
    optimizer.lion_prob = None
    optimizer.lion_power = None
    optimizer.lion_period_experts = None
    optimizer.lion_base_iter = 0
    optimizer.lion_total = 100
    optimizer.lion_beta1 = 0.9
    optimizer.lion_beta2 = 0.99
    optimizer.lion_lr_ratio = 50
    if hasattr(optimizer, "lion_energycal"):
        optimizer.lion_energycal = enabled
        optimizer.lion_energycal_rho_ref = 0.8
        optimizer.lion_energycal_exponent = 1.0
        optimizer.lion_energycal_qmin = 0.0
        optimizer.lion_energycal_diagnostics = True
    optimizer.wd_calls = []
    original_wd = optimizer._apply_weight_decay_inplace

    def record_wd(parameter, grad, lr, weight_decay):
        optimizer.wd_calls.append((lr, weight_decay))
        original_wd(parameter, grad, lr, weight_decay)

    optimizer._apply_weight_decay_inplace = record_wd
    return optimizer


def real_matrix(rows, cols, rank_one=False):
    value = torch.eye(rows, cols, dtype=torch.float32, device="cuda")
    if rank_one:
        value.zero_()
        value[0, 0] = 1.0
    return torch.nn.Parameter(value)


def real_specs(rank_one=False, include_nonrouted=False):
    fc1 = real_matrix(512, 1024, rank_one)
    fc2 = real_matrix(1024, 256, rank_one)
    specs = [("routed_fc1", fc1, True, True), ("routed_fc2", fc2, False, True)]
    if include_nonrouted:
        specs += [("dense", real_matrix(512, 1024), False, False),
                  ("shared", real_matrix(1024, 256), False, False)]
    return specs


def real_grads(specs, iteration):
    for _, parameter, _, _ in specs:
        parameter.grad = torch.eye(parameter.shape[0], parameter.shape[1], device=DEVICE) * (0.01 * (iteration + 1))


def rank_one_grads(specs):
    for _, parameter, _, _ in specs:
        parameter.grad = torch.zeros_like(parameter)
        parameter.grad[0, 0] = 1.0


def test_real_ns_pinned_original_cycles():
    global NS_OWNER
    reference_specs = real_specs(include_nonrouted=True)
    candidate_specs = real_specs(include_nonrouted=True)
    reference = real_optimizer(ORIGINAL.LionMuonMixin, reference_specs)
    candidate = real_optimizer(LionMuonMixin, candidate_specs)
    NS_CAPTURE.clear()
    counts = {"original": [], "candidate": []}
    try:
        for iteration in range(10):
            CLOCK["iteration"] = iteration
            for owner, optimizer, specs in (("original", reference, reference_specs),
                                            ("candidate", candidate, candidate_specs)):
                NS_OWNER = owner
                before = len([item for item in NS_CAPTURE if item["owner"] == owner])
                real_grads(specs, iteration)
                optimizer.step()
                after = len([item for item in NS_CAPTURE if item["owner"] == owner])
                counts[owner].append(after - before)
            for (_, left, _, _), (_, right, _, _) in zip(reference_specs, candidate_specs):
                assert torch.equal(left, right)
                assert torch.equal(reference.state[left]["momentum_buffer"],
                                   candidate.state[right]["momentum_buffer"])
            assert candidate._lion_energycal_last or iteration % 5
    finally:
        NS_OWNER = None
        CLOCK["iteration"] = None
    assert counts["original"] == counts["candidate"] == [4, 0, 0, 0, 0, 4, 0, 0, 0, 0]
    assert reference.wd_calls == candidate.wd_calls
    assert all(item["raw_dtype"] == "torch.float32" for item in NS_CAPTURE)
    return {"cycles": 2, "ns_calls_per_path": len(NS_CAPTURE) // 2,
            "raw_dtypes": sorted(set(item["raw_dtype"] for item in NS_CAPTURE))}


def test_real_ns_energy_precision_and_split():
    global NS_OWNER
    specs = real_specs(rank_one=True)
    candidate = real_optimizer(LionMuonMixin, specs)
    NS_CAPTURE.clear()
    scaled_inputs = []
    original_apply = candidate._lion_energycal_apply

    def capture_scaled(update, split):
        scaled_inputs.append(update.detach().clone())
        return original_apply(update, split)

    candidate._lion_energycal_apply = capture_scaled
    try:
        NS_OWNER = "candidate"
        rank_one_grads(specs)
        candidate.step()
    finally:
        NS_OWNER = None
        CLOCK["iteration"] = None
    assert len(NS_CAPTURE) == 3
    assert len(scaled_inputs) == 2
    diagnostics = candidate._lion_energycal_last
    assert len(diagnostics) == 3
    scaled_blocks = list(scaled_inputs[0].chunk(2, dim=0)) + [scaled_inputs[1]]
    errors = []
    for index, (capture, scaled, diagnostic) in enumerate(zip(NS_CAPTURE, scaled_blocks, diagnostics)):
        raw = capture["raw"]
        scale = 0.2 * (max(raw.shape) ** 0.5)
        raw_rho = raw.float().square().sum() / min(raw.shape)
        reconstructed = scaled.float().square().sum() / (scale * scale * min(raw.shape))
        torch.testing.assert_close(reconstructed, raw_rho, rtol=FP32_RECONSTRUCTION_RTOL,
                                   atol=FP32_RECONSTRUCTION_ATOL)
        expected_q = (raw_rho / 0.8).clamp(0, 1)
        torch.testing.assert_close(torch.tensor(diagnostic["q"], device=DEVICE), expected_q,
                                   rtol=FP32_RECONSTRUCTION_RTOL, atol=FP32_RECONSTRUCTION_ATOL)
        assert diagnostic["q"] < 1.0
        errors.append({"block": index, "shape": list(raw.shape), "raw_dtype": str(raw.dtype),
                       "ns_kwargs": capture["kwargs"],
                       "raw_rho": float(raw_rho), "reconstructed_rho": float(reconstructed),
                       "relative_error": float(abs(reconstructed - raw_rho) / raw_rho.clamp_min(1e-30)),
                       "q": diagnostic["q"], "rtol": FP32_RECONSTRUCTION_RTOL,
                       "atol": FP32_RECONSTRUCTION_ATOL})
    assert all(torch.count_nonzero(p) for _, p, _, _ in specs)
    return {"precision_contract": "captured FP32 raw NS output; BF16 NS rounding is inside that reference",
            "blocks": errors}


class TinyNamedModule(torch.nn.Module):
    def __init__(self, shape, weight_name):
        super().__init__()
        setattr(self, weight_name, torch.nn.Parameter(torch.eye(shape[0], shape[1], dtype=torch.bfloat16, device="cuda")))


def tiny_mcore_model():
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.module import MegatronModule

    config = TransformerConfig(num_layers=1, hidden_size=1024, num_attention_heads=8,
                               num_query_groups=2, kv_channels=128, bf16=True)

    class Model(MegatronModule):
        def __init__(self):
            super().__init__(config)
            layer0 = torch.nn.ModuleDict({"mlp": torch.nn.ModuleDict({
                "linear_fc1": TinyNamedModule((512, 1024), "weight")})})
            layer1 = torch.nn.ModuleDict({"mlp": torch.nn.ModuleDict({
                "experts": torch.nn.ModuleDict({
                    "linear_fc1": TinyNamedModule((512, 1024), "weight0"),
                    "linear_fc2": TinyNamedModule((1024, 256), "weight0"),
                }),
                "shared_experts": torch.nn.ModuleDict({
                    "linear_fc2": TinyNamedModule((1024, 256), "weight")}),
                "router": TinyNamedModule((64, 1024), "weight"),
            })})
            self.decoder = torch.nn.ModuleDict({"layers": torch.nn.ModuleList([layer0, layer1])})

    return Model()


def mcore_candidate():
    global MCORE_INSTALLED
    from megatron.core.optimizer import get_megatron_optimizer
    from megatron.core.optimizer.optimizer_config import OptimizerConfig
    if not MCORE_INSTALLED:
        install_muon_contract(fp8_states=False)
        MCORE_INSTALLED = True
    model = tiny_mcore_model()
    config = OptimizerConfig(optimizer="muon", lr=0.01, min_lr=0.0002, weight_decay=0.1,
                             bf16=True, params_dtype=torch.bfloat16, clip_grad=1,
                             use_distributed_optimizer=False, muon_momentum=0.95,
                             muon_nesterov=True, muon_num_ns_steps=5, muon_scale_mode="spectral",
                             muon_extra_scale_factor=0.2, muon_tp_mode="duplicated",
                             muon_split_qkv=False, adam_beta1=0.9, adam_beta2=0.95)
    optimizer = get_megatron_optimizer(config=config, model_chunks=[model], pg_collection=PG)
    return model, optimizer


def raw_parts(chained):
    return [wrapped.optimizer for wrapped in chained.chained_optimizers]


def group_assignment(model, chained):
    assignments = {}
    for name, parameter in model.named_parameters():
        main = getattr(parameter, "main_param", parameter)
        found = []
        for raw in raw_parts(chained):
            for group in raw.param_groups:
                if any(main is candidate for candidate in group["params"]):
                    found.append((raw, group))
        assert len(found) == 1, name
        raw, group = found[0]
        assignments[name] = {"class": type(raw).__name__, "mro": [cls.__name__ for cls in type(raw).__mro__[:4]],
                             "routed": group.get("stage3_routed_expert", False),
                             "split": group.get("stage3_split_swiglu_fc1", False),
                             "shape": list(parameter.shape), "model_dtype": str(parameter.dtype),
                             "main_dtype": str(main.dtype)}
    return assignments


def test_actual_mcore_grouping():
    model, chained = mcore_candidate()
    assignments = group_assignment(model, chained)
    assert len(chained.chained_optimizers) == 2
    for suffix in ("experts.linear_fc1.weight0", "experts.linear_fc2.weight0"):
        item = next(value for name, value in assignments.items() if suffix in name)
        assert item["routed"] and "Lionmuon" in item["mro"][0]
    assert next(value for name, value in assignments.items() if "experts.linear_fc1" in name)["split"]
    for suffix in ("shared_experts.linear_fc2.weight", "layers.0.mlp.linear_fc1.weight"):
        item = next(value for name, value in assignments.items() if suffix in name)
        assert not item["routed"] and "Lionmuon" in item["mro"][0]
    router = next(value for name, value in assignments.items() if name.endswith("router.weight"))
    assert not router["routed"] and "Lionmuon" not in router["mro"][0]
    return {"class": type(chained).__name__, "wrapped_classes": [type(item).__name__ for item in chained.chained_optimizers],
            "assignments": assignments}


def set_mcore_grads(model, iteration):
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.01 * (iteration + 1))


def optimizer_tensors(chained):
    tensors = []
    for raw in raw_parts(chained):
        for state in raw.state.values():
            for key, value in sorted(state.items()):
                if torch.is_tensor(value):
                    tensors.append(value)
    return tensors


def assert_optimizer_equal(left, right):
    left_tensors = optimizer_tensors(left)
    right_tensors = optimizer_tensors(right)
    assert len(left_tensors) == len(right_tensors)
    for a, b in zip(left_tensors, right_tensors):
        assert torch.equal(a, b)


def remove_calibration_metadata(value):
    if isinstance(value, list):
        for item in value:
            remove_calibration_metadata(item)
    elif isinstance(value, dict):
        if "param_groups" in value:
            for group in value["param_groups"]:
                group.pop("lionmuon_energycal", None)
        for child in value.values():
            remove_calibration_metadata(child)


def test_mcore_wrapper_resume_and_config():
    output = Path(os.environ["HMOE_CONTRACT_ROOT"]) / "mcore-tiny-checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for offset in range(5):
        model, reference = mcore_candidate()
        for iteration in range(offset):
            CLOCK["iteration"] = iteration
            set_mcore_grads(model, iteration)
            reference.step()
        for raw in raw_parts(reference):
            if hasattr(raw, "_lion_energycal_group"):
                for group in raw.param_groups:
                    raw._lion_energycal_group(group)
        checkpoint = {"model": model.state_dict(), "optimizer": reference.state_dict(), "iteration": offset}
        path = output / f"offset-{offset}.pt"
        torch.save(checkpoint, path)
        CLOCK["iteration"] = offset
        set_mcore_grads(model, offset)
        reference.step()
        restored_model, restored = mcore_candidate()
        restored_model.load_state_dict(torch.load(path, weights_only=False)["model"])
        restored.load_state_dict(torch.load(path, weights_only=False)["optimizer"])
        set_mcore_grads(restored_model, offset)
        restored.step()
        for (_, left), (_, right) in zip(model.state_dict().items(), restored_model.state_dict().items()):
            assert torch.equal(left, right)
        assert_optimizer_equal(reference, restored)
        reference_momentum = optimizer_tensors(reference)
        restored_momentum = optimizer_tensors(restored)
        assert reference_momentum
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(reference_momentum, restored_momentum))
        receipts.append({"offset": offset, "path": str(path), "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    saved = torch.load(output / "offset-0.pt", weights_only=False)
    mismatch_model, mismatch = mcore_candidate()
    for raw in raw_parts(mismatch):
        if hasattr(raw, "lion_energycal_rho_ref"):
            raw.lion_energycal_rho_ref = 0.7
    try:
        mismatch.load_state_dict(copy.deepcopy(saved["optimizer"]))
        raise AssertionError("MCore wrapper accepted EnergyCal mismatch")
    except ValueError as error:
        assert "mismatch" in str(error)
    missing = copy.deepcopy(saved["optimizer"])
    remove_calibration_metadata(missing)
    missing_model, missing_candidate = mcore_candidate()
    try:
        missing_candidate.load_state_dict(missing)
        raise AssertionError("MCore wrapper accepted missing EnergyCal metadata")
    except ValueError as error:
        assert "explicit switch" in str(error)
    disabled_model, disabled = mcore_candidate()
    for raw in raw_parts(disabled):
        if hasattr(raw, "lion_energycal"):
            raw.lion_energycal = False
    CLOCK["iteration"] = 0
    set_mcore_grads(disabled_model, 0)
    disabled.step()
    disabled_saved = copy.deepcopy({"model": disabled_model.state_dict(), "optimizer": disabled.state_dict()})
    remove_calibration_metadata(disabled_saved["optimizer"])
    restored_disabled_model, restored_disabled = mcore_candidate()
    for raw in raw_parts(restored_disabled):
        if hasattr(raw, "lion_energycal"):
            raw.lion_energycal = False
    restored_disabled_model.load_state_dict(disabled_saved["model"])
    restored_disabled.load_state_dict(disabled_saved["optimizer"])
    CLOCK["iteration"] = 1
    set_mcore_grads(disabled_model, 1)
    set_mcore_grads(restored_disabled_model, 1)
    disabled.step()
    restored_disabled.step()
    assert_optimizer_equal(disabled, restored_disabled)
    CLOCK["iteration"] = None
    return {"offsets": receipts, "wrapper": "actual ChainedOptimizer with BF16 Float16OptimizerWithFloat16Params",
            "mismatch_rejected": True, "missing_metadata_rejected": True,
            "disabled_baseline_compatible": True}


TESTS = [test_real_ns_pinned_original_cycles, test_real_ns_energy_precision_and_split,
         test_actual_mcore_grouping, test_mcore_wrapper_resume_and_config]


def main():
    global PG
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Production contracts require exactly one visible GPU")
    output = Path(os.environ["HMOE_CONTRACT_ROOT"])
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hmoe-contract004-rendezvous-") as directory:
        torch.distributed.init_process_group("nccl", init_method=f"file://{directory}/store", rank=0, world_size=1)
        group = torch.distributed.new_group([0])
        from megatron.core.process_groups_config import ProcessGroupCollection
        PG = ProcessGroupCollection(tp=group, expt_tp=group, mp=group, tp_ep_pp=group)
        import megatron.core.optimizer.emerging_optimizers as production
        original_ns = production.newton_schulz_tp

        def capture_ns(*args, **kwargs):
            raw = original_ns(*args, **kwargs)
            NS_CAPTURE.append({"owner": NS_OWNER, "raw": raw.detach().clone(), "raw_dtype": str(raw.dtype),
                               "shape": list(raw.shape), "kwargs": {key: repr(value) for key, value in kwargs.items()}})
            return raw

        production.newton_schulz_tp = capture_ns
        results = []
        try:
            for test in TESTS:
                CLOCK["iteration"] = None
                try:
                    value = test()
                    torch.cuda.synchronize()
                    result = {"name": test.__name__, "status": "passed"}
                    if value is not None:
                        result["evidence"] = value
                except Exception as error:
                    result = {"name": test.__name__, "status": "failed", "error": repr(error),
                              "traceback": traceback.format_exc()}
                finally:
                    CLOCK["iteration"] = None
                results.append(result)
                print("PRODUCTION_CONTRACT_TEST " + json.dumps(result, sort_keys=True), flush=True)
        finally:
            production.newton_schulz_tp = original_ns
            torch.distributed.destroy_process_group()
    report = {"schema": "hmoe-lionmoe-energycal-production-contract-v1", "training_invoked": False,
              "full_model_constructed": False, "real_checkpoint_downloaded": False,
              "test_count": len(results), "passed": sum(item["status"] == "passed" for item in results),
              "results": results}
    with (output / "production-results.json").open("w") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    passed = all(item["status"] == "passed" for item in results)
    print("PRODUCTION_CONTRACT_RESULT " + json.dumps({"passed": passed, "test_count": len(results),
                                                       "report": str(output / "production-results.json")}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
