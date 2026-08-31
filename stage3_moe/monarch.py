import math
import os
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


class MonarchFactors(torch.nn.Module):
    def __init__(self, in_features, out_features, blocks, groups, dtype, device, expert=False):
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
        self.blkdiag1 = torch.nn.Parameter(torch.empty(shape1, dtype=dtype, device=device))
        self.blkdiag2 = torch.nn.Parameter(torch.empty(shape2, dtype=dtype, device=device))
        for parameter in (self.blkdiag1, self.blkdiag2):
            parameter.monarch_factor = True
            parameter.allreduce = not expert
            fan_in = parameter.shape[-1]
            bound = math.sqrt(3.0) / math.sqrt(fan_in) * math.sqrt(2.0 / 6.0)
            with torch.no_grad():
                parameter.uniform_(-bound, bound)

    def forward(self, x, group=0):
        if x.shape[-1] < self.in_extended:
            x = F.pad(x, (0, self.in_extended - x.shape[-1]))
        output = _butterfly(x, self.blkdiag1[group], self.blkdiag2[group])
        return output[..., : self.out_features]


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
        if config.expert_model_parallel_size != 1:
            raise ValueError("Monarch expert prototype supports expert parallel size 1")
        blocks = int(os.environ["STAGE3_MONARCH_BLOCKS"])
        device = torch.cuda.current_device()
        hidden = config.hidden_size
        expert_hidden = config.moe_ffn_hidden_size
        self.config = config
        self.num_local_experts = num_local_experts
        self.fc1 = MonarchFactors(
            hidden,
            2 * expert_hidden,
            blocks,
            num_local_experts,
            config.params_dtype,
            device,
            expert=False,
        )
        self.fc2 = MonarchFactors(
            expert_hidden,
            hidden,
            blocks,
            num_local_experts,
            config.params_dtype,
            device,
            expert=False,
        )

    def forward(self, hidden_states, tokens_per_expert, permuted_probs):
        counts = tokens_per_expert.tolist()
        hidden_chunks = hidden_states.split(counts)
        prob_chunks = permuted_probs.reshape(-1, 1).split(counts)
        outputs = []
        for expert, (hidden, probability) in enumerate(zip(hidden_chunks, prob_chunks)):
            intermediate = self.fc1(hidden, expert)
            gate, value = intermediate.chunk(2, dim=-1)
            intermediate = F.silu(gate) * value
            intermediate = intermediate * probability
            outputs.append(self.fc2(intermediate, expert))
        return torch.cat(outputs), None

    def backward_dw(self):
        return None


def install_monarch_model(blocks):
    import gpt_builders
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

    os.environ["STAGE3_MONARCH_BLOCKS"] = str(blocks)
    original = gpt_builders.get_gpt_decoder_block_spec
    norm = TESpecProvider().layer_norm(has_residual=True)
    linears = MLPSubmodules(
        linear_fc1=MonarchColumnParallelLinear,
        linear_fc2=MonarchRowParallelLinear,
    )

    def monarch_spec(config, *args, **kwargs):
        block = original(config, *args, **kwargs)
        for layer_spec in block.layer_specs:
            layer = layer_spec.submodules
            layer.input_layernorm = norm
            layer.pre_mlp_layernorm = norm
            layer.self_attention.submodules.linear_qkv = MonarchColumnParallelLinear
            layer.self_attention.submodules.linear_proj = MonarchRowParallelLinear
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
                layer.mlp = partial(MLP, submodules=linears)
        return block

    gpt_builders.get_gpt_decoder_block_spec = monarch_spec


def install_monarch_muon_contract():
    from megatron.core.optimizer.emerging_optimizers import _EMERGING_OPTIMIZERS
    from megatron.core.optimizer.optimizer_config import ParamKey, ParamPredicate

    entry = _EMERGING_OPTIMIZERS["muon"]
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
