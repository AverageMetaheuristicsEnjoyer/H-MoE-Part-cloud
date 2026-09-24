"""How much of a layer's expert bank can one shared hidden-side factor hold?

--monarch-share hidden makes every routed expert of a layer read the residual
stream through one common factor (fc1's input side) and write into it through
another (fc2's output side). With blocks=2 at the Stage 3 geometry that is a
512-dim common input subspace and a 256-dim common output subspace, each split
as a direct sum over the Monarch blocks (2 x 256 on contiguous input halves,
2 x 128 on interleaved output halves).

On a trained dense MoE checkpoint this reports the fraction of the experts'
summed weight energy that the best such subspace captures:

    shared_free     best common subspace of that dimension, any structure
    shared_monarch  best common subspace with the Monarch block structure
    own_monarch     each expert's own best block-structured subspace, averaged
                    (what unshared Monarch experts get)
    random          shared_monarch for Gaussian weights of the same shapes

1.0 means the experts already live in one such subspace; random is the floor
of no common structure. Energy is a proxy for function and dense-trained
weights did not co-adapt to the constraint, so treat it as a prior, not a
loss prediction.
"""

import argparse
import pickle
import re
import sys
import types
from collections import defaultdict

import torch


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    # args carry Megatron enums and the optimizer TE types; neither is needed
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (_Stub,), {})


_pickle = types.SimpleNamespace(Unpickler=_Unpickler, load=pickle.load, __name__="pickle")


def captured(gram, dim, blocks, interleaved):
    """Top-`dim` energy of `gram`, optionally as a direct sum over coordinate blocks."""
    total = gram.diagonal().sum()
    if blocks == 1:
        return (torch.linalg.eigvalsh(gram)[-dim:].sum() / total).item()
    n = gram.shape[0]
    energy = 0.0
    for block in range(blocks):
        index = (torch.arange(block, n, blocks) if interleaved
                 else torch.arange(block * n // blocks, (block + 1) * n // blocks))
        sub = gram.index_select(0, index.to(gram.device)).index_select(1, index.to(gram.device))
        energy += torch.linalg.eigvalsh(sub)[-dim // blocks:].sum()
    return (energy / total).item()


def audit(weights, side, blocks, dim, device):
    """weights: per-expert matrices; side 'out' uses W W^T, 'in' uses W^T W."""
    grams = []
    for w in weights:
        w = w.to(device=device, dtype=torch.float64)
        grams.append(w @ w.T if side == "out" else w.T @ w)
    interleaved = side == "out"  # fc2's output is laid out block-interleaved
    summed = torch.stack(grams).sum(0)
    return {
        "shared_free": captured(summed, dim, 1, interleaved),
        "shared_monarch": captured(summed, dim, blocks, interleaved),
        "own_monarch": sum(captured(g, dim, blocks, interleaved) for g in grams) / len(grams),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--blocks", type=int, default=2)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False,
                       pickle_module=_pickle)
    model = state["model"]
    fc1, fc2 = defaultdict(dict), defaultdict(dict)
    pattern = re.compile(r"decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc([12])\.weight(\d+)$")
    for key, value in model.items():
        match = pattern.search(key)
        if match:
            layer, which, expert = (int(g) for g in match.groups())
            (fc1 if which == 1 else fc2)[layer][expert] = value
    if not fc1:
        print("no TEGroupedMLP expert weights found; expert keys:", flush=True)
        for key in model:
            if ".experts." in key:
                print("  ", key, tuple(model[key].shape) if torch.is_tensor(model[key]) else "")
        return 1

    layers = sorted(fc1)
    sample1, sample2 = fc1[layers[0]][0], fc2[layers[0]][0]
    hidden = sample1.shape[1]
    print(f"iteration {state.get('iteration')}  layers {layers[0]}..{layers[-1]}  "
          f"experts {len(fc1[layers[0]])}  fc1 {tuple(sample1.shape)}  fc2 {tuple(sample2.shape)}",
          flush=True)
    # the shared factor's dimension: fc1 keeps 2*expert_hidden of the input,
    # fc2 writes expert_hidden directions (Monarch shapes at in != out)
    in_dim, out_dim = sample1.shape[0], sample2.shape[1]
    assert in_dim <= hidden and out_dim <= hidden

    generator = torch.Generator().manual_seed(0)
    random_out = audit(
        [torch.randn(sample2.shape, generator=generator) for _ in fc2[layers[0]]],
        "out", args.blocks, out_dim, device)
    random_in = audit(
        [torch.randn(sample1.shape, generator=generator) for _ in fc1[layers[0]]],
        "in", args.blocks, in_dim, device)

    print("AUDIT\tlayer\tside\tdim\tshared_free\tshared_monarch\town_monarch\trandom", flush=True)
    rows = []
    for layer in layers:
        for side, bank, dim, floor in (("fc1_in", fc1, in_dim, random_in),
                                       ("fc2_out", fc2, out_dim, random_out)):
            weights = [bank[layer][e] for e in sorted(bank[layer])]
            result = audit(weights, "in" if side == "fc1_in" else "out", args.blocks, dim, device)
            rows.append((side, result))
            print(f"AUDIT\t{layer}\t{side}\t{dim}\t{result['shared_free']:.4f}\t"
                  f"{result['shared_monarch']:.4f}\t{result['own_monarch']:.4f}\t"
                  f"{floor['shared_monarch']:.4f}", flush=True)
    for side in ("fc1_in", "fc2_out"):
        values = [r for s, r in rows if s == side]
        print(f"MEAN\t{side}\t" + "\t".join(
            f"{k}={sum(v[k] for v in values) / len(values):.4f}"
            for k in ("shared_free", "shared_monarch", "own_monarch")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
