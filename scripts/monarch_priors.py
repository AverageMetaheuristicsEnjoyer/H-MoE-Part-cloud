"""Priors for where Monarch factors fit, read off a trained dense MoE checkpoint.

projection  For every kind of weight matrix: the share of its energy (squared
            Frobenius norm) the best Monarch approximation keeps, next to the
            best plain low-rank approximation with the same parameter count
            and to Gaussian matrices of the same shape.
tying       The share an expert bank keeps when one Monarch factor is tied
            across a group of matrices: across the 64 experts of a layer
            (--monarch-share hidden), across groups of 4 experts, or along a
            path of 4 consecutive layers (path-tied). The checkpoint's routers
            are independent, so experts are matched between layers by weight
            similarity; the match is also run on Gaussian weights, to price
            the selection bias of matching itself.

Both rest on one exact characterisation. With nb blocks, the matrices
MonarchFactors can represent are those whose nb x nb sub-blocks (rows taken
every nb-th, columns in contiguous chunks) each have rank <= min(in, out) /
nb**2, and each sub-block owns a disjoint slice of both factors: the right
factor R (blkdiag1) holds its row space, the left factor L (blkdiag2) its
column space. Hence
  - the best Monarch approximation is a truncated SVD per sub-block;
  - tying R over a group makes every sub-block's row space common to the
    group, and the best fit is PCA of the group's summed row Grams (tying L:
    column Grams), each member keeping its own other factor.
The script checks this against MonarchFactors before reading the checkpoint.
Energy is a proxy for function, and these weights were trained dense: treat
the numbers as priors, not loss predictions.
"""

import argparse
import math
import pickle
import re
import sys
import types
from collections import defaultdict

import torch
from scipy.optimize import linear_sum_assignment

from stage3_moe.monarch import MonarchFactors


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


def subblocks(w, nb):
    """(..., out, in) -> (..., j, a, out/nb, in/nb) in MonarchFactors' layout."""
    out_features, in_features = w.shape[-2:]
    lead = w.ndim - 2
    w = w.reshape(*w.shape[:-2], out_features // nb, nb, nb, in_features // nb)
    return w.permute(*range(lead), lead + 1, lead + 2, lead, lead + 3)


def rank_per_block(w, nb):
    return min(w.shape[-2:]) // nb ** 2


def own_energy(w, nb):
    """Energy the best Monarch approximation keeps, per matrix of a batch."""
    k = rank_per_block(w, nb)
    s = torch.linalg.svdvals(subblocks(w, nb))
    return (s[..., :k] ** 2).sum((-1, -2, -3))


def lowrank_energy(w, nb):
    """Best rank-r approximation with the Monarch parameter count, r = min/nb."""
    r = min(w.shape[-2:]) // nb
    return (torch.linalg.svdvals(w)[..., :r] ** 2).sum(-1)


def tied_energy(group, nb, side):
    """Energy kept when factor `side` ("R" or "L") is shared by the group (M, out, in)."""
    k = rank_per_block(group, nb)
    blocks = subblocks(group, nb)
    if side == "R":
        gram = blocks.transpose(-1, -2) @ blocks
    else:
        gram = blocks @ blocks.transpose(-1, -2)
    return torch.linalg.eigvalsh(gram.sum(0))[..., -k:].sum()


def materialize(factors, group):
    eye = torch.eye(factors.in_features, dtype=factors.blkdiag1.dtype)
    return factors.forward(eye, group=group).T


def selfcheck():
    torch.manual_seed(0)
    for in_features, out_features in ((1024, 512), (256, 1024), (512, 512), (1024, 1536)):
        for nb in (2, 4):
            factors = MonarchFactors(in_features, out_features, nb, 4, torch.float64, "cpu")
            params = (factors.blkdiag1.numel() + factors.blkdiag2.numel()) // 4
            assert params == min(in_features, out_features) * (in_features + out_features) // nb
            m = torch.stack([materialize(factors, g) for g in range(4)])
            kept = own_energy(m, nb) / (m ** 2).sum((-1, -2))
            assert torch.allclose(kept, torch.ones_like(kept), atol=1e-9), kept
            dense = torch.randn(out_features, in_features, dtype=torch.float64)
            assert own_energy(dense, nb) / (dense ** 2).sum() < 0.99
            with torch.no_grad():
                factors.blkdiag1.copy_(factors.blkdiag1[:1].expand_as(factors.blkdiag1))
            m = torch.stack([materialize(factors, g) for g in range(4)])
            total = (m ** 2).sum()
            assert abs(tied_energy(m, nb, "R") / total - 1) < 1e-9
            assert tied_energy(m, nb, "L") / total < 0.99
    print("SELFCHECK ok: sub-block ranks, own and tied fits match MonarchFactors", flush=True)


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------

def report_projection(name, weights, device):
    w = weights.to(device=device, dtype=torch.float32)
    total = (w.double() ** 2).sum()
    out_features, in_features = w.shape[-2:]
    for nb in (2, 4):
        params = min(in_features, out_features) * (in_features + out_features) / nb
        ratio = params / (in_features * out_features)
        own = own_energy(w, nb).double().sum() / total
        lowrank = lowrank_energy(w, nb).double().sum() / total
        noise = torch.randn(4, out_features, in_features, device=device)
        random_own = (own_energy(noise, nb).double().sum() / (noise.double() ** 2).sum())
        print(f"PROJ\t{name}\t{nb}\t{w.shape[0]}\t{out_features}x{in_features}\t"
              f"{ratio:.3f}\t{own:.4f}\t{lowrank:.4f}\t{random_own:.4f}", flush=True)


# --------------------------------------------------------------------------
# tying
# --------------------------------------------------------------------------

def centred_similarity(a, b):
    """Cosine between expert Grams, each layer centred on its own mean Gram."""
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    a = torch.nn.functional.normalize(a.flatten(1), dim=1)
    b = torch.nn.functional.normalize(b.flatten(1), dim=1)
    return a @ b.T


def grams(fc1, fc2):
    # what an expert reads (fc1 input side) and writes (fc2 output side)
    return fc1.transpose(-1, -2) @ fc1, fc2 @ fc2.transpose(-1, -2)


def chain_orders(fc1_layers, fc2_layers):
    """Per layer of a path block, the expert order that aligns it to the previous layer."""
    orders = [torch.arange(fc1_layers[0].shape[0])]
    matched, background = [], []
    previous = grams(fc1_layers[0], fc2_layers[0])
    for fc1, fc2 in zip(fc1_layers[1:], fc2_layers[1:]):
        current = grams(fc1, fc2)
        prev_in = previous[0][orders[-1]]
        prev_out = previous[1][orders[-1]]
        similarity = (centred_similarity(prev_in, current[0])
                      + centred_similarity(prev_out, current[1])).cpu()
        rows, cols = linear_sum_assignment(similarity.numpy(), maximize=True)
        orders.append(torch.as_tensor(cols))
        matched.append(similarity[rows, cols].mean().item())
        background.append(similarity.mean().item())
        previous = current
    return orders, sum(matched) / len(matched), sum(background) / len(background)


def tie_groups(bank_layers, orders):
    """Stack matrix e of each layer in its aligned order: (experts, layers, out, in)."""
    return torch.stack([layer[order] for layer, order in zip(bank_layers, orders)], dim=1)


def report_tying(label, fc1, fc2, blocks, nb, device, generator=None):
    """fc1/fc2: {layer: (experts, out, in)}; blocks: lists of consecutive layers."""
    sides = (("fc1", fc1, "R"), ("fc1", fc1, "L"), ("fc2", fc2, "R"), ("fc2", fc2, "L"))
    experts = fc1[blocks[0][0]].shape[0]
    shuffle = torch.randperm(experts, generator=generator)
    groups_of_four = shuffle.reshape(-1, 4)

    kept = defaultdict(float)
    total = defaultdict(float)
    own = defaultdict(float)
    for layer in [l for block in blocks for l in block]:
        for name, bank, side in sides:
            w = bank[layer].to(device=device, dtype=torch.float64)
            energy = (w ** 2).sum().item()
            key = (name, side)
            kept["experts64", key] += tied_energy(w, nb, side).item()
            total["experts64", key] += energy
            for group in groups_of_four:
                kept["experts4", key] += tied_energy(w[group.to(device)], nb, side).item()
            total["experts4", key] += energy
            own[key] += own_energy(w, nb).sum().item()
            total["own", key] += energy

    matching = []
    for block in blocks:
        fc1_layers = [fc1[l].to(device=device, dtype=torch.float32) for l in block]
        fc2_layers = [fc2[l].to(device=device, dtype=torch.float32) for l in block]
        orders, matched, background = chain_orders(fc1_layers, fc2_layers)
        matching.append((matched, background))
        identity = [torch.arange(experts)] * len(block)
        for name, bank_layers, side in (("fc1", fc1_layers, "R"), ("fc1", fc1_layers, "L"),
                                        ("fc2", fc2_layers, "R"), ("fc2", fc2_layers, "L")):
            key = (name, side)
            for mode, chosen in (("path4_matched", orders), ("path4_index", identity)):
                stacked = tie_groups(bank_layers, [o.to(device) for o in chosen]).double()
                for group in stacked:
                    kept[mode, key] += tied_energy(group, nb, side).item()
                total[mode, key] += (stacked ** 2).sum().item()

    for name, _, side in sides:
        key = (name, side)
        base = own[key] / total["own", key]
        cells = [f"own={base:.4f}"]
        for mode in ("experts64", "experts4", "path4_matched", "path4_index"):
            value = kept[mode, key] / total[mode, key]
            cells.append(f"{mode}={value:.4f} ({value / base:.3f} of own)")
        print(f"TIE\t{label}\t{name}.{side}\t" + "\t".join(cells), flush=True)
    matched = sum(m for m, _ in matching) / len(matching)
    background = sum(b for _, b in matching) / len(matching)
    print(f"MATCH\t{label}\tmatched_similarity={matched:.4f}\tall_pairs={background:.4f}",
          flush=True)


# --------------------------------------------------------------------------

PATTERNS = {
    "routed": re.compile(r"decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc([12])\.weight(\d+)$"),
    "shared_expert": re.compile(r"decoder\.layers\.(\d+)\.mlp\.shared_experts\.linear_fc([12])\.weight$"),
    "dense_mlp": re.compile(r"decoder\.layers\.(\d+)\.mlp\.linear_fc([12])\.weight$"),
    "qkv": re.compile(r"decoder\.layers\.(\d+)\.self_attention\.linear_qkv\.weight$"),
    "proj": re.compile(r"decoder\.layers\.(\d+)\.self_attention\.linear_proj\.weight$"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--path-block", type=int, default=4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selfcheck()

    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False,
                       pickle_module=_pickle)
    model = state["model"]
    found = defaultdict(dict)
    for key, value in model.items():
        if not torch.is_tensor(value) or value.ndim != 2:
            continue
        for kind, pattern in PATTERNS.items():
            match = pattern.search(key)
            if match:
                groups = [int(g) for g in match.groups()]
                found[kind][tuple(groups)] = value
    missing = [kind for kind in PATTERNS if not found[kind]]
    if missing:
        print(f"no weights for {missing}; 2-D decoder keys:", flush=True)
        for key, value in model.items():
            if torch.is_tensor(value) and value.ndim == 2:
                print("  ", key, tuple(value.shape))
        return 1

    fc1 = defaultdict(list)
    fc2 = defaultdict(list)
    for (layer, which, expert), value in sorted(found["routed"].items()):
        (fc1 if which == 1 else fc2)[layer].append(value)
    fc1 = {layer: torch.stack(v) for layer, v in fc1.items()}
    fc2 = {layer: torch.stack(v) for layer, v in fc2.items()}
    layers = sorted(fc1)
    half = fc1[layers[0]].shape[1] // 2
    print(f"iteration {state.get('iteration')}  moe layers {layers[0]}..{layers[-1]}  "
          f"experts {fc1[layers[0]].shape[0]}", flush=True)

    print("PROJ\tmatrix\tnb\tcount\tshape\tparams/dense\tmonarch\tlowrank_same_params\t"
          "monarch_on_gaussian", flush=True)
    routed1 = torch.cat([fc1[l] for l in layers])
    report_projection("routed fc1 (gate+up)", routed1, device)
    report_projection("routed gate", routed1[:, :half], device)
    report_projection("routed up", routed1[:, half:], device)
    report_projection("routed fc2 (down)", torch.cat([fc2[l] for l in layers]), device)
    for kind in ("shared_expert", "dense_mlp"):
        for which in (1, 2):
            stacked = torch.stack([v for k, v in sorted(found[kind].items()) if k[1] == which])
            report_projection(f"{kind} fc{which}", stacked, device)
    for kind in ("qkv", "proj"):
        report_projection(kind, torch.stack([v for _, v in sorted(found[kind].items())]), device)
    del routed1

    blocks = [layers[i:i + args.path_block]
              for i in range(0, len(layers) - args.path_block + 1, args.path_block)]
    print(f"path blocks {blocks}", flush=True)
    report_tying("checkpoint", fc1, fc2, blocks, 2, device, torch.Generator().manual_seed(0))

    generator = torch.Generator().manual_seed(1)
    noise1 = {l: torch.randn(fc1[l].shape, generator=generator) for l in blocks[0]}
    noise2 = {l: torch.randn(fc2[l].shape, generator=generator) for l in blocks[0]}
    report_tying("gaussian", noise1, noise2, blocks[:1], 2, device, torch.Generator().manual_seed(0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
