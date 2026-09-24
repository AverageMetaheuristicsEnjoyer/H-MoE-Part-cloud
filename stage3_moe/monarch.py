import math
import os
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn.functional as F


def _butterfly(x, w1, w2):
    batch_shape = x.shape[:-1]
    batch = math.prod(batch_shape)
    blocks, q, p = w1.shape
    out_blocks, s, r = w2.shape
    if out_blocks * r != blocks * q:
        raise ValueError("invalid Monarch factor shapes")
    x1 = x.reshape(batch, blocks, p).transpose(0, 1)
    y1 = torch.bmm(x1, w1.transpose(-1, -2))
    y1 = y1.transpose(0, 1).reshape(batch, r, out_blocks).permute(2, 0, 1)
    y2 = torch.bmm(y1, w2.transpose(-1, -2))
    return y2.permute(1, 2, 0).reshape(*batch_shape, s * out_blocks)


class _Permutation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, index, inverse):
        ctx.save_for_backward(inverse)
        return x.index_select(0, index)

    @staticmethod
    def backward(ctx, grad_output):
        (inverse,) = ctx.saved_tensors
        return grad_output.index_select(0, inverse), None, None


def _packed_layout(counts, blocks):
    active_experts = torch.nonzero(counts, as_tuple=False).flatten()
    active_counts = counts.index_select(0, active_experts)
    active_starts = active_counts.cumsum(0) - active_counts
    token_experts = torch.repeat_interleave(
        torch.arange(active_counts.numel(), device=counts.device), active_counts
    )
    positions = torch.arange(token_experts.numel(), device=counts.device) - torch.repeat_interleave(
        active_starts, active_counts
    )
    block_counts = active_counts.repeat_interleave(blocks)
    block_starts = (block_counts.cumsum(0) - block_counts).reshape(-1, blocks)
    packed_index = (
        block_starts.index_select(0, token_experts) + positions[:, None]
    ).reshape(-1)
    inverse_index = torch.empty_like(packed_index).scatter_(
        0,
        packed_index,
        torch.arange(packed_index.numel(), device=counts.device),
    )
    return active_experts, block_counts.cumsum(0).to(torch.int32), packed_index, inverse_index


def _init_factors(parameters, expert):
    """Uniform init with bound 1/sqrt(fan_in) per factor, tagged for the batched Muon step."""
    rng = nullcontext()
    if expert:
        from megatron.core.tensor_parallel.random import (
            get_cuda_rng_tracker,
            get_expert_parallel_rng_tracker_name,
        )

        rng = get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name())
    with rng, torch.no_grad():
        for parameter in parameters:
            # monarch_factor: a stack of matrices that Muon orthogonalises one by one
            parameter.monarch_factor = True
            parameter.allreduce = not expert
            fan_in = parameter.shape[-1]
            bound = math.sqrt(3.0) / math.sqrt(fan_in) * math.sqrt(2.0 / 6.0)
            parameter.uniform_(-bound, bound)


class MonarchFactors(torch.nn.Module):
    def __init__(
        self, in_features, out_features, blocks, groups, dtype, device, expert=False,
        share1=False, share2=False,
    ):
        """share1/share2 tie that factor across all groups (one copy, groups=1)."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.blocks = blocks
        self.groups = groups
        in_block = math.ceil(in_features / blocks)
        out_block = math.ceil(out_features / blocks)
        self.in_extended = in_block * blocks
        self.out_extended = out_block * blocks
        if self.in_extended < self.out_extended:
            shape1 = (groups, blocks, in_block, in_block)
            shape2 = (groups, blocks, out_block, in_block)
        else:
            shape1 = (groups, blocks, out_block, in_block)
            shape2 = (groups, blocks, out_block, out_block)
        if share1:
            shape1 = (1, *shape1[1:])
        if share2:
            shape2 = (1, *shape2[1:])
        self.blkdiag1 = torch.nn.Parameter(torch.empty(shape1, dtype=dtype, device=device))
        self.blkdiag2 = torch.nn.Parameter(torch.empty(shape2, dtype=dtype, device=device))
        _init_factors((self.blkdiag1, self.blkdiag2), expert)

    def forward(self, x, group=0):
        x = x.to(self.blkdiag1.dtype)
        if x.shape[-1] < self.in_extended:
            x = F.pad(x, (0, self.in_extended - x.shape[-1]))
        output = _butterfly(x, self.blkdiag1[group], self.blkdiag2[group])
        return output[..., : self.out_features]

    def forward_grouped(self, x):
        x = x.to(self.blkdiag1.dtype)
        if x.shape[-1] < self.in_extended:
            x = F.pad(x, (0, self.in_extended - x.shape[-1]))
        groups, batch, _ = x.shape
        blocks, q, p = self.blkdiag1.shape[1:]
        out_blocks, s, r = self.blkdiag2.shape[1:]
        x1 = x.reshape(groups, batch, blocks, p).permute(0, 2, 1, 3)
        y1 = x1 @ self.blkdiag1.transpose(-1, -2)
        y1 = y1.permute(0, 2, 1, 3).reshape(groups, batch, r, out_blocks)
        y1 = y1.permute(0, 3, 1, 2)
        y2 = y1 @ self.blkdiag2.transpose(-1, -2)
        return y2.permute(0, 2, 3, 1).reshape(groups, batch, s * out_blocks)[
            ..., : self.out_features
        ]

    def _packed_stage(self, rows, weight, layout, batch, blocks):
        """rows: (batch * blocks, k), token-major, tokens grouped by expert."""
        if weight.shape[0] == 1 and self.groups > 1:
            # one factor for every expert: a plain bmm, no packing needed
            k = rows.shape[-1]
            planes = rows.reshape(batch, blocks, k).transpose(0, 1)
            out = torch.bmm(planes, weight[0].transpose(-1, -2))
            return out.transpose(0, 1).reshape(batch * blocks, -1)
        active_experts, offsets, packed_index, inverse_index = layout
        rows = _Permutation.apply(rows, inverse_index, packed_index)
        if active_experts.numel() != weight.shape[0]:
            weight = weight.index_select(0, active_experts)
        out = torch._grouped_mm(rows, weight.flatten(0, 1).transpose(-1, -2), offsets)
        return _Permutation.apply(out, packed_index, inverse_index)

    def forward_packed(self, x, layout):
        x = x.to(self.blkdiag1.dtype)
        if x.shape[-1] < self.in_extended:
            x = F.pad(x, (0, self.in_extended - x.shape[-1]))
        batch = x.shape[0]
        blocks, q, p = self.blkdiag1.shape[1:]
        out_blocks, s, r = self.blkdiag2.shape[1:]

        y1 = self._packed_stage(x.reshape(-1, p), self.blkdiag1, layout, batch, blocks)
        y1 = y1.reshape(batch, r, out_blocks).transpose(1, 2).reshape(-1, r)
        y2 = self._packed_stage(y1, self.blkdiag2, layout, batch, out_blocks)
        y2 = y2.reshape(batch, out_blocks, s)
        return y2.transpose(1, 2).reshape(batch, s * out_blocks)[..., : self.out_features]


def _active(weight, active_experts):
    if active_experts.numel() != weight.shape[0]:
        return weight.index_select(0, active_experts)
    return weight


class DenseExperts(torch.nn.Module):
    """A dense matrix per expert: one grouped GEMM over the expert-sorted tokens."""

    blocks = 1

    def __init__(self, in_features, out_features, groups, dtype, device, expert=False):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(groups, out_features, in_features, dtype=dtype, device=device)
        )
        _init_factors((self.weight,), expert)

    def forward_packed(self, x, layout):
        active_experts, offsets = layout[:2]
        weight = _active(self.weight, active_experts)
        return torch._grouped_mm(x.to(weight.dtype), weight.transpose(-1, -2), offsets)


class LowRankExperts(torch.nn.Module):
    """up @ down per expert, rank chosen by the caller: two grouped GEMMs."""

    blocks = 1

    def __init__(self, in_features, out_features, rank, groups, dtype, device, expert=False):
        super().__init__()
        self.down = torch.nn.Parameter(
            torch.empty(groups, rank, in_features, dtype=dtype, device=device)
        )
        self.up = torch.nn.Parameter(
            torch.empty(groups, out_features, rank, dtype=dtype, device=device)
        )
        _init_factors((self.down, self.up), expert)

    def forward_packed(self, x, layout):
        active_experts, offsets = layout[:2]
        down = _active(self.down, active_experts)
        up = _active(self.up, active_experts)
        hidden = torch._grouped_mm(x.to(down.dtype), down.transpose(-1, -2), offsets)
        return torch._grouped_mm(hidden, up.transpose(-1, -2), offsets)


class _MonarchParallelLinear(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        *,
        config,
        bias,
        skip_bias_add,
        is_expert,
        tp_group,
        name=None,
        **kwargs,
    ):
        super().__init__()
        if torch.distributed.get_world_size(tp_group) != 1:
            raise ValueError("Monarch prototype supports tensor parallel size 1")
        blocks = int(os.environ["STAGE3_MONARCH_BLOCKS"])
        self.skip_bias_add = skip_bias_add
        self.factors = MonarchFactors(
            input_size,
            output_size,
            blocks,
            1,
            config.params_dtype,
            torch.cuda.current_device(),
            expert=is_expert and config.expert_model_parallel_size > 1,
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(output_size, dtype=config.params_dtype, device=torch.cuda.current_device())
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x, **kwargs):
        output = self.factors(x)
        if self.bias is None:
            return output, None
        if self.skip_bias_add:
            return output, self.bias
        return output + self.bias, None

    def backward_dw(self):
        return None


class MonarchColumnParallelLinear(_MonarchParallelLinear):
    pass


class MonarchRowParallelLinear(_MonarchParallelLinear):
    pass


class MonarchGroupedMLP(torch.nn.Module):
    def __init__(self, num_local_experts, config, submodules=None, pg_collection=None, name=None):
        super().__init__()
        blocks = int(os.environ["STAGE3_MONARCH_BLOCKS"])
        device = torch.cuda.current_device()
        hidden = config.hidden_size
        expert_hidden = config.moe_ffn_hidden_size
        expert_parallel = config.expert_model_parallel_size > 1
        # "hidden": every expert of the layer shares the factor that touches the
        # model dimension (fc1's input side, fc2's output side)
        share_hidden = os.environ.get("STAGE3_MONARCH_SHARE", "none") == "hidden"
        if share_hidden and expert_parallel:
            raise ValueError("a factor shared across experts needs expert parallel size 1")
        # the routed-expert bank: Monarch factors, Monarch with a dense down
        # projection (fc2), or low rank at the Monarch parameter count
        kind = os.environ.get("STAGE3_MONARCH_EXPERTS", "monarch")
        if share_hidden and kind != "monarch":
            raise ValueError("--monarch-share needs --monarch-experts monarch")
        self.config = config
        self.num_local_experts = num_local_experts
        args = (num_local_experts, config.params_dtype, device)
        if kind == "lowrank":
            # rank min(in, out) / blocks gives r * (in + out) = the Monarch count
            self.fc1 = LowRankExperts(hidden, 2 * expert_hidden,
                                      min(hidden, 2 * expert_hidden) // blocks, *args,
                                      expert=expert_parallel)
            self.fc2 = LowRankExperts(expert_hidden, hidden,
                                      min(expert_hidden, hidden) // blocks, *args,
                                      expert=expert_parallel)
            return
        self.fc1 = MonarchFactors(hidden, 2 * expert_hidden, blocks, *args,
                                  expert=expert_parallel, share1=share_hidden)
        if kind == "monarch_dense_down":
            self.fc2 = DenseExperts(expert_hidden, hidden, *args, expert=expert_parallel)
        else:
            self.fc2 = MonarchFactors(expert_hidden, hidden, blocks, *args,
                                      expert=expert_parallel, share2=share_hidden)

    def forward(self, hidden_states, tokens_per_expert, permuted_probs):
        counts = tokens_per_expert.to(device=hidden_states.device, dtype=torch.long)
        layouts = {}
        for bank in (self.fc1, self.fc2):
            if bank.blocks not in layouts:
                layouts[bank.blocks] = _packed_layout(counts, bank.blocks)
        intermediate = self.fc1.forward_packed(hidden_states, layouts[self.fc1.blocks])
        gate, value = intermediate.chunk(2, dim=-1)
        output = self.fc2.forward_packed(
            F.silu(gate) * value * permuted_probs.reshape(-1, 1), layouts[self.fc2.blocks]
        )
        return output, None

    def backward_dw(self):
        return None


class _TorchQKRMSNorm(torch.nn.RMSNorm):
    def __init__(self, config, hidden_size, eps=1e-5, **kwargs):
        super().__init__(
            hidden_size,
            eps=eps,
            dtype=config.params_dtype,
            device=torch.cuda.current_device(),
        )


def install_monarch_model(blocks, share="none", experts="monarch"):
    import gpt_builders
    import megatron.training.training as training
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

    os.environ["STAGE3_MONARCH_BLOCKS"] = str(blocks)
    os.environ["STAGE3_MONARCH_SHARE"] = share
    os.environ["STAGE3_MONARCH_EXPERTS"] = experts
    original_block_spec = gpt_builders.get_gpt_decoder_block_spec
    original_dense_spec = gpt_builders.get_gpt_layer_with_transformer_engine_spec
    original_setup = training.setup_model_and_optimizer
    norm = TESpecProvider().layer_norm(has_residual=True)
    linears = MLPSubmodules(
        linear_fc1=MonarchColumnParallelLinear,
        linear_fc2=MonarchRowParallelLinear,
    )

    class MonarchMLP(MLP):
        def __init__(self, *args, pg_collection=None, **kwargs):
            kwargs.pop("is_mtp_layer", None)
            kwargs.pop("layer_number", None)
            super().__init__(*args, tp_group=pg_collection.tp, **kwargs)

    def patch_layer(layer_spec):
        layer = layer_spec.submodules
        layer.input_layernorm = norm
        layer.pre_mlp_layernorm = norm
        layer.self_attention.submodules.linear_qkv = MonarchColumnParallelLinear
        layer.self_attention.submodules.linear_proj = MonarchRowParallelLinear
        layer.self_attention.submodules.q_layernorm = _TorchQKRMSNorm
        layer.self_attention.submodules.k_layernorm = _TorchQKRMSNorm
        layer.sharded_state_dict_keys_map = {}
        mlp = layer.mlp
        if isinstance(mlp, partial) and mlp.func is MoELayer:
            layer.mlp = partial(
                MoELayer,
                submodules=MoESubmodules(
                    experts=MonarchGroupedMLP,
                    shared_experts=partial(SharedExpertMLP, submodules=linears),
                ),
            )
        else:
            layer.mlp = partial(MonarchMLP, submodules=linears)
        return layer_spec

    def monarch_block_spec(config, *args, **kwargs):
        block = original_block_spec(config, *args, **kwargs)
        for layer_spec in block.layer_specs:
            patch_layer(layer_spec)
        return block

    def monarch_dense_spec(*args, **kwargs):
        return patch_layer(original_dense_spec(*args, **kwargs))

    def setup_and_check(*args, **kwargs):
        result = original_setup(*args, **kwargs)
        models = result[0]
        modules = sum(
            isinstance(module, MonarchFactors)
            for model in models
            for module in model.modules()
        )
        factor_parameters = sum(
            parameter.numel()
            for model in models
            for parameter in model.parameters()
            if getattr(parameter, "monarch_factor", False)
        )
        if modules == 0:
            raise RuntimeError("Monarch model hook did not replace any linear layers")
        print(
            f"MONARCH_MODEL_CHECK rank={torch.distributed.get_rank()} "
            f"modules={modules} factor_parameters={factor_parameters}",
            flush=True,
        )
        return result

    gpt_builders.get_gpt_decoder_block_spec = monarch_block_spec
    gpt_builders.get_gpt_layer_with_transformer_engine_spec = monarch_dense_spec
    training.setup_model_and_optimizer = setup_and_check


def install_monarch_muon_contract():
    from emerging_optimizers.orthogonalized_optimizers import muon_utils
    from megatron.core.optimizer.emerging_optimizers import _EMERGING_OPTIMIZERS
    from megatron.core.optimizer.optimizer_config import ParamKey, ParamPredicate

    entry = _EMERGING_OPTIMIZERS["muon"]
    optimizer_cls = entry.optimizer_cls
    original_ns_step = muon_utils.newton_schulz_step

    def monarch_ns_step(x, a, b, c, tp_group=None):
        if x.ndim == 2:
            return original_ns_step(x, a, b, c, tp_group)
        shape = x.shape
        x = x.reshape(-1, shape[-2], shape[-1])
        aa = x @ x.mT
        if tp_group is not None:
            torch.distributed.all_reduce(aa, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        bb = torch.baddbmm(aa, aa, aa, beta=b, alpha=c)
        return torch.baddbmm(x, bb, x, beta=a).reshape(shape)

    class MonarchFactorMuon(optimizer_cls):
        def orthogonalize(self, parameter, grad, **kwargs):
            if not getattr(parameter, "monarch_factor", False):
                return super().orthogonalize(parameter, grad, **kwargs)
            shape = grad.shape
            grad = grad.reshape(-1, shape[-2], shape[-1])
            return super().orthogonalize(parameter, grad, **kwargs).reshape(shape)

    muon_utils.newton_schulz_step = monarch_ns_step
    entry.optimizer_cls = MonarchFactorMuon
    for key in list(entry.default_param_overrides):
        predicate = key.predicate
        if isinstance(predicate, ParamPredicate) and predicate.name == "nonlinear_or_embedding":
            del entry.default_param_overrides[key]
    entry.default_param_overrides[
        ParamKey(
            predicate=ParamPredicate(
                name="nonlinear_or_embedding_except_monarch",
                fn=lambda parameter: (
                    getattr(parameter, "is_embedding_or_output_parameter", False)
                    or (parameter.ndim != 2 and not getattr(parameter, "monarch_factor", False))
                ),
            )
        )
    ] = {"optimizer": "adam"}
