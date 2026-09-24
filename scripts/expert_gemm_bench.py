"""Expert-bank GEMM efficiency: dense vs Monarch vs shared-factor Monarch.

Question: Monarch pays off on large matrices, but MoE experts are narrow
(width 256 at the Stage 3 1C geometry). Factorising a narrow expert makes the
blocks smaller still, so the grouped GEMM may lose more to arithmetic intensity
than the parameterisation saves. This measures that directly, on the real
expert-bank shapes, without Megatron or TransformerEngine.

The packed butterfly path is copied from H-MoE-Part
`stage3_moe/monarch.py` (MonarchFactors.forward_packed / _packed_layout) so the
numbers describe the implementation that actually trains, not a re-derivation.

Variants
    dense           per-expert dense weights, one grouped GEMM per projection
    monarch         per-expert Monarch factors (what ships today)
    shared          the hidden-side factor is shared across experts, which
                    turns that stage from a grouped GEMM into a plain bmm
    bmm_only        both factors shared: not a usable model, but it prices the
                    butterfly with the grouped GEMM and every permutation gone,
                    so it separates "the blocks are too small" from "the packing
                    and the riffle cost more than the GEMM"

All three Monarch variants do the same per-token arithmetic: sharing removes
stored parameters, not FLOPs. Time differences between them are layout alone.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# packed layout + permutation (verbatim from stage3_moe/monarch.py)
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# factor shapes
# --------------------------------------------------------------------------

def monarch_shapes(in_features, out_features, blocks):
    """(shape1, shape2) per expert, following MonarchFactors.__init__."""
    in_block = math.ceil(in_features / blocks)
    out_block = math.ceil(out_features / blocks)
    if in_block * blocks < out_block * blocks:
        return (blocks, in_block, in_block), (blocks, out_block, in_block)
    return (blocks, out_block, in_block), (blocks, out_block, out_block)


def _empty(shape, groups, dtype, device):
    fan_in = shape[-1]
    bound = math.sqrt(3.0) / math.sqrt(fan_in) * math.sqrt(2.0 / 6.0)
    tensor = torch.empty((groups, *shape), dtype=dtype, device=device)
    with torch.no_grad():
        tensor.uniform_(-bound, bound)
    return tensor


class MonarchBank(torch.nn.Module):
    """One projection of the expert bank, Monarch-factorised.

    `sharing` is "none" (a factor pair per expert), "hidden" (the factor
    touching the model dimension is shared across experts) or "all" (both are,
    which is not a usable model but bounds what the butterfly costs with the
    grouped GEMM and its permutations removed entirely).
    """

    def __init__(self, in_features, out_features, blocks, experts, dtype, device,
                 sharing="none"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.blocks = blocks
        shape1, shape2 = monarch_shapes(in_features, out_features, blocks)
        self.in_extended = shape1[-1] * blocks
        # factor 1 consumes the input, factor 2 produces the output: the one on
        # the model-dimension side is the one to share.
        hidden_side_first = in_features > out_features
        self.share1 = sharing == "all" or (sharing == "hidden" and hidden_side_first)
        self.share2 = sharing == "all" or (sharing == "hidden" and not hidden_side_first)
        self.blkdiag1 = torch.nn.Parameter(
            _empty(shape1, 1 if self.share1 else experts, dtype, device))
        self.blkdiag2 = torch.nn.Parameter(
            _empty(shape2, 1 if self.share2 else experts, dtype, device))

    def _stage(self, rows, weight, shared, layout, batch, blocks_dim):
        """rows: (batch * blocks_dim, k) laid out token-major."""
        if shared:
            k = rows.shape[-1]
            planes = rows.reshape(batch, blocks_dim, k).transpose(0, 1)
            out = torch.bmm(planes, weight[0].transpose(-1, -2))
            return out.transpose(0, 1).reshape(batch * blocks_dim, -1)
        active_experts, offsets, packed_index, inverse_index = layout
        packed = _Permutation.apply(rows, inverse_index, packed_index)
        if active_experts.numel() != weight.shape[0]:
            weight = weight.index_select(0, active_experts)
        out = torch._grouped_mm(packed, weight.flatten(0, 1).transpose(-1, -2), offsets)
        return _Permutation.apply(out, packed_index, inverse_index)

    def forward(self, x, layout):
        if x.shape[-1] < self.in_extended:
            x = F.pad(x, (0, self.in_extended - x.shape[-1]))
        batch = x.shape[0]
        blocks, q, p = self.blkdiag1.shape[1:]
        out_blocks, s, r = self.blkdiag2.shape[1:]

        y1 = self._stage(x.reshape(-1, p), self.blkdiag1, self.share1, layout, batch, blocks)
        y1 = y1.reshape(batch, blocks, q)
        y1 = y1.reshape(batch, r, out_blocks).transpose(1, 2).reshape(-1, r)

        y2 = self._stage(y1, self.blkdiag2, self.share2, layout, batch, out_blocks)
        y2 = y2.reshape(batch, out_blocks, s).transpose(1, 2)
        return y2.reshape(batch, s * out_blocks)[..., : self.out_features]


class DenseBank(torch.nn.Module):
    def __init__(self, in_features, out_features, experts, dtype, device):
        super().__init__()
        bound = math.sqrt(3.0) / math.sqrt(in_features)
        weight = torch.empty((experts, out_features, in_features), dtype=dtype, device=device)
        with torch.no_grad():
            weight.uniform_(-bound, bound)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x, layout):
        _, offsets, _, _ = layout
        return torch._grouped_mm(x, self.weight.transpose(-1, -2), offsets)


# --------------------------------------------------------------------------
# the expert MLP under test
# --------------------------------------------------------------------------

class ExpertMLP(torch.nn.Module):
    def __init__(self, variant, hidden, expert_hidden, experts, blocks, dtype, device):
        super().__init__()
        if variant == "dense":
            self.fc1 = DenseBank(hidden, 2 * expert_hidden, experts, dtype, device)
            self.fc2 = DenseBank(expert_hidden, hidden, experts, dtype, device)
        else:
            sharing = {"monarch": "none", "shared": "hidden", "bmm_only": "all"}[variant]
            self.fc1 = MonarchBank(hidden, 2 * expert_hidden, blocks, experts, dtype, device,
                                   sharing=sharing)
            self.fc2 = MonarchBank(expert_hidden, hidden, blocks, experts, dtype, device,
                                   sharing=sharing)
        self.variant = variant

    def forward(self, x, layout, dense_layout):
        chosen = dense_layout if self.variant == "dense" else layout
        gate, value = self.fc1(x, chosen).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * value, chosen)


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_variant(variant, args, device, dtype):
    experts, blocks = args.experts, args.blocks
    routed = experts * args.tokens_per_expert
    counts = torch.full((experts,), args.tokens_per_expert, dtype=torch.long, device=device)

    model = ExpertMLP(variant, args.hidden, args.expert_hidden, experts, blocks, dtype, device)
    layout = _packed_layout(counts, blocks)
    dense_layout = _packed_layout(counts, 1)
    x = torch.randn(routed, args.hidden, dtype=dtype, device=device, requires_grad=True)

    runnable = torch.compile(model) if args.compile else model

    def forward():
        return runnable(x, layout, dense_layout)

    # torch._grouped_mm rejects a grad with zero strides, which is what
    # out.sum().backward() hands it, so seed the backward with a real tensor.
    with torch.no_grad():
        grad_seed = torch.ones_like(model(x, layout, dense_layout))

    def forward_backward():
        forward().backward(grad_seed)

    forward_ms = timed(lambda: forward(), args.warmup, args.iters)
    total_ms = timed(forward_backward, args.warmup, args.iters)

    params = sum(p.numel() for p in model.parameters())
    if variant == "dense":
        per_token_flops = 2 * params / experts
    else:
        # sharing changes storage, not arithmetic, so the per-token cost comes
        # from the factor shapes rather than from the stored parameter count.
        per_token_flops = sum(
            2 * shape[0] * shape[1] * shape[2]
            for bank in (model.fc1, model.fc2)
            for shape in monarch_shapes(bank.in_features, bank.out_features, blocks)
        )
    flops = per_token_flops * routed
    return {
        "variant": variant,
        "blocks": blocks if variant != "dense" else 0,
        "tokens_per_expert": args.tokens_per_expert,
        "routed_tokens": routed,
        "params": params,
        "forward_ms": forward_ms,
        "fwd_bwd_ms": total_ms,
        "forward_tflops": flops / forward_ms * 1e-9,
        "fwd_bwd_tflops": 3 * flops / total_ms * 1e-9,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--expert-hidden", type=int, default=256)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--tokens-per-expert", type=int, default=1024)
    parser.add_argument("--sweep", type=str, default="")
    parser.add_argument("--variants", type=str, default="dense,bmm_only,shared,monarch")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(f"torch {torch.__version__}  gpu {torch.cuda.get_device_name(0)}", flush=True)
    print(f"hidden {args.hidden}  expert_hidden {args.expert_hidden}  "
          f"experts {args.experts}  blocks {args.blocks}  compile {args.compile}",
          flush=True)

    sweep = [int(v) for v in args.sweep.split(",")] if args.sweep else [args.tokens_per_expert]
    columns = ("variant", "blocks", "tokens_per_expert", "routed_tokens", "params",
               "forward_ms", "fwd_bwd_ms", "forward_tflops", "fwd_bwd_tflops", "peak_gb")
    print("PT\t" + "\t".join(columns), flush=True)

    records = []
    for tokens in sweep:
        args.tokens_per_expert = tokens
        for variant in args.variants.split(","):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            record = run_variant(variant, args, device, dtype)
            records.append(record)
            print("PT\t" + "\t".join(
                f"{record[c]:.3f}" if isinstance(record[c], float) else str(record[c])
                for c in columns), flush=True)

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(records, handle, indent=2)
        print(f"wrote {args.json_out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
