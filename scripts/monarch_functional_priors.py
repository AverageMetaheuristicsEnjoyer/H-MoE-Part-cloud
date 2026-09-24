"""Activation-weighted Monarch priors: the questions of monarch_priors.py,
measured as output error on real inputs, E||(W - What) x||^2 / E||W x||^2.

A pure-torch forward of the Stage 3 dense MoE checkpoint over the start of the
development split collects the second moment E[x x^T] of every matrix's input
(per expert for routed experts, whose second matrix sees the probability-
weighted SwiGLU output as in TEGroupedMLP). The forward is validated by its LM
loss against the run's reported validation loss before anything is fitted.

    own      best Monarch fit per matrix: block-coordinate reduced-rank
             regression over the input chunks, exact at every step
    lowrank  best rank-r fit with the Monarch parameter count (closed form)
    tied     R or L shared across a group, every member weighted by its own
             inputs, fitted from the weight-energy optimum by alternating
             updates over the input chunks that never increase the error: the
             member factors and a shared L in closed form, a shared R by
             preconditioned conjugate gradients (it solves a sum of Sylvester
             terms). Groups: the 64 experts of a layer, 4 random experts of a
             layer, or 4 consecutive layers with experts matched by what they
             read (input moments: who sees the same tokens), by their weights,
             or by index. Tying one member constrains nothing, so a group of
             one must reproduce the exact own fit - checked before and after.

Sub-block (j, a) of a Monarch matrix (rows o with o % nb == j, input chunk a)
is L_ja R_ja with rank k = min(in, out) / nb**2 and owns disjoint slices of
blkdiag2 (L) and blkdiag1 (R); monarch_priors.py checks that against
MonarchFactors. Tying blkdiag1 across a group shares every R_ja, tying
blkdiag2 every L_ja, so the fits below work on sub-blocks directly.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
import monarch_priors as priors  # noqa: E402

HIDDEN, HEADS, GROUPS, HEAD_DIM, TOPK, SCALE, EPS = 1024, 8, 2, 128, 8, 2.5, 1e-5
BIN_DTYPES = {1: numpy.uint8, 2: numpy.int8, 3: numpy.int16, 4: numpy.int32,
              5: numpy.int64, 6: numpy.float64, 7: numpy.float32, 8: numpy.uint16}


# --------------------------------------------------------------------------
# fits under input moments S = E[x x^T]
# --------------------------------------------------------------------------

def row_groups(W, nb):
    """(..., out, in) -> (..., j, s, in): row group j holds output rows o with o % nb == j."""
    out_features, in_features = W.shape[-2:]
    return W.reshape(*W.shape[:-2], out_features // nb, nb, in_features).transpose(-3, -2)


def solve_right(A, B):
    """B A^-1 for symmetric A."""
    return torch.linalg.solve(A, B.transpose(-1, -2)).transpose(-1, -2)


def ridge(A):
    scale = 1e-6 * A.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None]
    return A + scale * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)


def pcg(op, rhs, x, precondition, iters=40):
    """Batched preconditioned CG; every trailing (k, c) matrix is its own system."""
    dot = lambda a, b: (a * b).sum((-1, -2), keepdim=True)
    r = rhs - op(x)
    z = precondition(r)
    p = z
    rz = dot(r, z)
    for _ in range(iters):
        Ap = op(p)
        alpha = rz / dot(p, Ap).clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        z = precondition(r)
        rz_next = dot(r, z)
        p = z + (rz_next / rz.clamp_min(1e-30)) * p
        rz = rz_next
    return x


def fit_tied(W, S, nb, side, max_sweeps=60, trace=False, tol=1e-6):
    """W (..., M, out, in), S (..., M, in, in): R or L shared over M. Returns kept (init, end)."""
    out_features, in_features = W.shape[-2:]
    k = min(out_features, in_features) // nb ** 2
    c = in_features // nb
    Wj = row_groups(W, nb)                                    # (..., M, j, s, in)
    Sx = S.unsqueeze(-3)                                      # (..., M, 1, in, in)
    total = output_energy(W, S)
    L, R = {}, {}
    for a in range(nb):                                       # weight-energy optimum
        B = Wj[..., a * c:(a + 1) * c]                        # (..., M, j, s, c)
        if side == "R":
            V = torch.linalg.eigh((B.transpose(-1, -2) @ B).sum(-4, keepdim=True))[1][..., -k:]
            R[a], L[a] = V.transpose(-1, -2), B @ V
        else:
            U = torch.linalg.eigh((B @ B.transpose(-1, -2)).sum(-4, keepdim=True))[1][..., -k:]
            L[a], R[a] = U, U.transpose(-1, -2) @ B
    Wh = torch.cat([(L[a] @ R[a]).expand_as(Wj[..., :c]) for a in range(nb)], dim=-1)
    kept = lambda: 1 - (output_energy((Wj - Wh).transpose(-3, -2).reshape(W.shape), S)
                        / total).item()
    history = [kept()]
    for _ in range(max_sweeps):
        for a in range(nb):
            cols = slice(a * c, (a + 1) * c)
            D = Wj - Wh
            D[..., cols] = Wj[..., cols]
            C = D @ Sx[..., :, cols]                          # E[residual x_a^T]
            Saa = ridge(Sx[..., cols, cols])
            if side == "R":
                # members' L given the shared R, then the shared R given the L's
                L[a] = solve_right(ridge(R[a] @ Saa @ R[a].transpose(-1, -2)),
                                   C @ R[a].transpose(-1, -2))
                A = ridge(L[a].transpose(-1, -2) @ L[a])
                rhs = (L[a].transpose(-1, -2) @ C).sum(-4, keepdim=True)
                A_mean = A.mean(-4, keepdim=True)
                S_mean = Saa.mean(-4, keepdim=True)
                R[a] = pcg(lambda X: (A @ X @ Saa).sum(-4, keepdim=True), rhs, R[a],
                           lambda Y: solve_right(S_mean, torch.linalg.solve(A_mean, Y)))
            else:
                # members' R given the shared L, then the shared L given the R's
                R[a] = solve_right(Saa, torch.linalg.solve(
                    ridge(L[a].transpose(-1, -2) @ L[a]), L[a].transpose(-1, -2) @ C))
                L[a] = solve_right(
                    ridge((R[a] @ Saa @ R[a].transpose(-1, -2)).sum(-4, keepdim=True)),
                    (C @ R[a].transpose(-1, -2)).sum(-4, keepdim=True))
            Wh[..., cols] = L[a] @ R[a]
        history.append(kept())
        if history[-1] - history[-2] < tol:
            break
    if trace:
        print("TRACE\t" + " ".join(f"{h:.5f}" for h in history), flush=True)
    return history[0], history[-1]


def output_energy(W, S):
    return torch.einsum("...oi,...ij,...oj->", W, S, W)


def kept_by(W, S, Wh):
    E = W - Wh
    return 1 - (output_energy(E, S) / output_energy(W, S)).item()


def own_rrr(W, S, nb, max_sweeps=60, tol=1e-6):
    """Best Monarch fit per matrix under S: block-coordinate reduced-rank regression,
    exact per chunk, swept until the kept energy stops moving (correlated chunks
    converge slowly)."""
    out_features, in_features = W.shape[-2:]
    k = min(out_features, in_features) // nb ** 2
    c = in_features // nb
    lead = W.shape[:-2]
    # row group j holds output rows o with o % nb == j
    Wj = W.reshape(*lead, out_features // nb, nb, in_features).transpose(-3, -2)
    Wh = torch.zeros_like(Wj)
    Sx = S.unsqueeze(-3)
    total = output_energy(W, S)
    previous = 0.0
    for _ in range(max_sweeps):
        for a in range(nb):
            cols = slice(a * c, (a + 1) * c)
            D = Wj - Wh
            D[..., cols] = Wj[..., cols]
            C = D @ Sx[..., :, cols]                                     # cov(y, x_a)
            Saa = Sx[..., cols, cols]
            ridge = 1e-6 * Saa.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None]
            Saa = Saa + ridge * torch.eye(c, device=W.device, dtype=W.dtype)
            B = torch.linalg.solve(Saa, C.transpose(-1, -2)).transpose(-1, -2)  # C Saa^-1
            U = torch.linalg.eigh(C @ B.transpose(-1, -2))[1][..., -k:]
            Wh[..., cols] = U @ (U.transpose(-1, -2) @ B)
        current = 1 - (output_energy((Wj - Wh).transpose(-3, -2).reshape(W.shape), S) / total).item()
        if current - previous < tol:
            break
        previous = current
    return Wh.transpose(-3, -2).reshape(W.shape)


def lowrank_kept(W, S, nb):
    r = min(W.shape[-2:]) // nb
    evals, evecs = torch.linalg.eigh(S)
    root = (evecs * evals.clamp_min(0).sqrt()[..., None, :]) @ evecs.transpose(-1, -2)
    sv = torch.linalg.svdvals(W @ root)
    return ((sv[..., :r] ** 2).sum() / (sv ** 2).sum()).item()


# --------------------------------------------------------------------------
# forward pass collecting input moments
# --------------------------------------------------------------------------

def rms(x, weight):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS) * weight.float()).to(x.dtype)


def rope_tables(seq, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM))
    freqs = torch.outer(torch.arange(seq, device=device).float(), inv_freq)
    emb = torch.cat([freqs, freqs], -1)
    return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]


def rope(x, cos, sin):
    x32 = x.float()
    x1, x2 = x32.chunk(2, -1)
    return (x32 * cos + torch.cat([-x2, x1], -1) * sin).to(x.dtype)


class Moments:
    def __init__(self, device):
        self.sums = {}
        self.counts = defaultdict(int)
        self.device = device

    def add(self, key, x):
        x = x.reshape(-1, x.shape[-1]).float()
        if x.shape[0] == 0:
            return
        if key not in self.sums:
            self.sums[key] = torch.zeros(x.shape[-1], x.shape[-1], dtype=torch.float64,
                                         device=self.device)
        self.sums[key] += (x.T @ x).double()
        self.counts[key] += x.shape[0]

    def get(self, key):
        return (self.sums[key] / self.counts[key]).float()


def swiglu(x):
    gate, up = x.chunk(2, -1)
    return F.silu(gate) * up


def forward(model, layers, tokens, moments, cos, sin):
    p = lambda key: model[key]
    h = F.embedding(tokens, p("embedding.word_embeddings.weight"))
    batch, seq = tokens.shape
    for layer in range(layers):
        pre = f"decoder.layers.{layer}."
        x = rms(h, p(pre + "self_attention.linear_qkv.layer_norm_weight"))
        moments.add(("qkv", layer), x)
        qkv = (x @ p(pre + "self_attention.linear_qkv.weight").T).view(
            batch, seq, GROUPS, (HEADS // GROUPS + 2) * HEAD_DIM)
        q, k, v = qkv.split([HEADS // GROUPS * HEAD_DIM, HEAD_DIM, HEAD_DIM], dim=-1)
        q = rms(q.reshape(batch, seq, HEADS, HEAD_DIM), p(pre + "self_attention.q_layernorm.weight"))
        k = rms(k, p(pre + "self_attention.k_layernorm.weight"))
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True,
            enable_gqa=True).transpose(1, 2).reshape(batch, seq, HEADS * HEAD_DIM)
        moments.add(("proj", layer), attn)
        h = h + attn @ p(pre + "self_attention.linear_proj.weight").T

        if pre + "mlp.linear_fc1.weight" in model:          # the dense layer
            x = rms(h, p(pre + "mlp.linear_fc1.layer_norm_weight"))
            moments.add(("dense_fc1", layer), x)
            mid = swiglu(x @ p(pre + "mlp.linear_fc1.weight").T)
            moments.add(("dense_fc2", layer), mid)
            h = h + mid @ p(pre + "mlp.linear_fc2.weight").T
            continue

        x = rms(h, p(pre + "pre_mlp_layernorm.weight")).reshape(-1, HIDDEN)
        scores = torch.sigmoid(x.float() @ p(pre + "mlp.router.weight").float().T)
        top = torch.topk(scores + p(pre + "mlp.router.expert_bias").float(), TOPK, dim=-1).indices
        probs = torch.gather(scores, 1, top)
        probs = probs / (probs.sum(-1, keepdim=True) + 1e-20) * SCALE
        out = torch.zeros_like(x)
        for expert in range(p(pre + "mlp.router.weight").shape[0]):
            token, slot = (top == expert).nonzero(as_tuple=True)
            xe = x[token]
            moments.add(("routed_fc1", layer, expert), xe)
            mid = (swiglu(xe @ p(pre + f"mlp.experts.linear_fc1.weight{expert}").T).float()
                   * probs[token, slot, None]).to(x.dtype)
            moments.add(("routed_fc2", layer, expert), mid)
            out.index_add_(0, token, mid @ p(pre + f"mlp.experts.linear_fc2.weight{expert}").T)
        moments.add(("shared_fc1", layer), x)
        mid = swiglu(x @ p(pre + "mlp.shared_experts.linear_fc1.weight").T)
        moments.add(("shared_fc2", layer), mid)
        out = out + mid @ p(pre + "mlp.shared_experts.linear_fc2.weight").T
        h = h + out.view(batch, seq, HIDDEN)
    h = rms(h, p("decoder.final_layernorm.weight"))
    return h @ p("output_layer.weight").T


def read_tokens(prefix, count):
    with open(prefix + ".idx", "rb") as stream:
        header = stream.read(9)
        assert header == b"MMIDIDX\x00\x00", header
        stream.read(8)
        dtype = BIN_DTYPES[stream.read(1)[0]]
    return numpy.memmap(prefix + ".bin", dtype=dtype, mode="r")[:count].astype(numpy.int64)


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------

def participation(S):
    return ((S.diagonal(dim1=-2, dim2=-1).sum(-1) ** 2) / (S * S).sum((-1, -2))).mean().item()


def report_own(name, W, S, nb, device):
    W, S = W.to(device).float(), S.to(device).float()
    own = kept_by(W, S, own_rrr(W, S, nb))
    weight_own = (priors.own_energy(W, nb).sum() / (W ** 2).sum()).item()
    params = min(W.shape[-2:]) * sum(W.shape[-2:]) / nb / (W.shape[-2] * W.shape[-1])
    noise = torch.randn_like(W)
    noise_own = kept_by(noise, S, own_rrr(noise, S, nb))
    print(f"FPROJ\t{name}\t{nb}\t{W.shape[0]}\t{params:.3f}\t{own:.4f}\t"
          f"{lowrank_kept(W, S, nb):.4f}\t{noise_own:.4f}\t{weight_own:.4f}\t"
          f"{participation(S):.1f}", flush=True)


def match_chain(features):
    """features: per layer (experts, d) rows; chain Hungarian matching on centred cosine."""
    orders = [torch.arange(features[0].shape[0])]
    scores = []
    for previous, current in zip(features[:-1], features[1:]):
        a = F.normalize(previous[orders[-1]] - previous.mean(0), dim=1)
        b = F.normalize(current - current.mean(0), dim=1)
        similarity = (a @ b.T).cpu()
        rows, cols = linear_sum_assignment(similarity.numpy(), maximize=True)
        orders.append(torch.as_tensor(cols))
        scores.append((similarity[rows, cols].mean().item(), similarity.mean().item()))
    return orders, scores


def report_tying(label, W1, W2, S1, S2, blocks, nb, device, generator, max_sweeps):
    """W*/S*: {layer: (experts, ...)} on the CPU; returns nothing, prints TIE lines."""
    layers = [l for block in blocks for l in block]
    experts = W1[layers[0]].shape[0]
    groups_of_four = torch.randperm(experts, generator=generator).reshape(-1, 4)
    kept = defaultdict(lambda: [0.0, 0.0, 0.0])            # start, end, weight
    matching = {}
    solver_check = None

    def add(mode, key, W, S, side, trace=False):
        start, end = fit_tied(W, S, nb, side, max_sweeps, trace)
        energy = output_energy(W, S).item()
        kept[mode, key][0] += start * energy
        kept[mode, key][1] += end * energy
        kept[mode, key][2] += energy

    for name, Wd, Sd in (("fc1", W1, S1), ("fc2", W2, S2)):
        for layer in layers:
            W = Wd[layer].to(device).float()
            S = Sd[layer].to(device).float()
            energy = output_energy(W, S).item()
            own = kept_by(W, S, own_rrr(W, S, nb))
            for side in ("R", "L"):
                key = (name, side)
                kept["own", key][1] += own * energy
                kept["own", key][2] += energy
                add("experts64", key, W.unsqueeze(0), S.unsqueeze(0), side,
                    trace=solver_check is None)
                index = groups_of_four.to(device)
                add("experts4", key, W[index], S[index], side)
            if solver_check is None:
                # tying across a group of one constrains nothing: must match own
                single = [fit_tied(W[:8].unsqueeze(1), S[:8].unsqueeze(1), nb, side, max_sweeps)[1]
                          for side in ("R", "L")]
                solver_check = (kept_by(W[:8], S[:8], own_rrr(W[:8], S[:8], nb)), *single)
        for block in blocks:
            stack_W = [Wd[l].to(device).float() for l in block]
            stack_S = [Sd[l].to(device).float() for l in block]
            reads = [S1[l].to(device).float().flatten(1) for l in block]
            weights = [(W1[l].to(device).float().transpose(-1, -2) @ W1[l].to(device).float()
                        ).flatten(1) for l in block]
            for mode, features in (("path4_tokens", reads), ("path4_weights", weights),
                                   ("path4_index", None)):
                if features is None:
                    orders = [torch.arange(experts)] * len(block)
                else:
                    orders, scores = match_chain(features)
                    matching.setdefault(mode, []).extend(scores)
                W = torch.stack([w[o.to(device)] for w, o in zip(stack_W, orders)], dim=1)
                S = torch.stack([s[o.to(device)] for s, o in zip(stack_S, orders)], dim=1)
                for side in ("R", "L"):
                    add(mode, (name, side), W, S, side)
            del stack_W, stack_S, reads, weights

    for name in ("fc1", "fc2"):
        for side in ("R", "L"):
            key = (name, side)
            base = kept["own", key][1] / kept["own", key][2]
            cells = [f"own={base:.4f}"]
            for mode in ("experts64", "experts4", "path4_tokens", "path4_weights", "path4_index"):
                start, end, energy = kept[mode, key]
                cells.append(f"{mode}={end / energy:.4f} ({end / energy / base:.3f} of own, "
                             f"init {start / energy:.4f})")
            print(f"FTIE\t{label}\t{name}.{side}\t" + "\t".join(cells), flush=True)
    for mode, scores in matching.items():
        matched = sum(m for m, _ in scores) / len(scores)
        background = sum(b for _, b in scores) / len(scores)
        print(f"FMATCH\t{label}\t{mode}\tmatched={matched:.4f}\tall_pairs={background:.4f}",
              flush=True)
    print(f"SOLVER\t{label}\town exact={solver_check[0]:.5f}\ttied R, one member="
          f"{solver_check[1]:.5f}\ttied L, one member={solver_check[2]:.5f}", flush=True)


def selfcheck():
    torch.manual_seed(0)
    for (in_features, out_features), nb in (((1024, 512), 2), ((256, 1024), 2), ((1024, 512), 4)):
        W = torch.randn(3, out_features, in_features, dtype=torch.float64)
        eye = torch.eye(in_features, dtype=torch.float64).expand(3, -1, -1)
        energy = (W ** 2).sum()
        exact = (priors.own_energy(W, nb).sum() / energy).item()
        assert abs(kept_by(W, eye, own_rrr(W, eye, nb)) - exact) < 1e-8
        assert abs(lowrank_kept(W, eye, nb) - (priors.lowrank_energy(W, nb).sum() / energy).item()) < 1e-8
        for side in ("R", "L"):
            # unweighted: the start is the closed-form optimum and nothing may move it
            start, end = fit_tied(W, eye, nb, side, max_sweeps=3)
            closed = (priors.tied_energy(W, nb, side) / energy).item()
            assert abs(start - closed) < 1e-8 and abs(end - closed) < 1e-6, (side, start, end, closed)
        # anisotropic inputs, spectrum over five decades, chunks correlated
        small_in, small_out = in_features // 4, out_features // 4
        w = W[:, :small_out, :small_in].contiguous()
        Q = torch.linalg.qr(torch.randn(small_in, small_in, dtype=torch.float64))[0]
        S = ((Q * torch.logspace(-3, 2, small_in, dtype=torch.float64)) @ Q.T).expand(3, -1, -1)
        own = kept_by(w, S, own_rrr(w, S, nb, max_sweeps=400, tol=1e-10))
        for side in ("R", "L"):
            start, end = fit_tied(w.unsqueeze(1), S.unsqueeze(1), nb, side, max_sweeps=400, tol=1e-10)
            assert end >= start - 1e-9 and abs(end - own) < 2e-3, (side, start, end, own)
    print("SELFCHECK ok: exact fits reproduce the closed forms; ties over one member match own",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--data", default="/home/jovyan/data/fineweb-edu-gpt2-megatron/data/development")
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--reported-val-loss", type=float, default=2.650184)
    parser.add_argument("--max-sweeps", type=int, default=60)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    selfcheck()

    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False,
                       pickle_module=priors._pickle)
    model = {k: v.to(device) for k, v in state["model"].items()
             if torch.is_tensor(v) and "_extra_state" not in k}
    layers = 1 + max(int(k.split(".")[2]) for k in model if k.startswith("decoder.layers."))
    print(f"iteration {state.get('iteration')}  layers {layers}  tensors {len(model)}", flush=True)

    tokens = read_tokens(args.data, args.batches * args.batch_size * (args.seq + 1))
    tokens = torch.from_numpy(tokens).view(args.batches, args.batch_size, args.seq + 1)
    cos, sin = rope_tables(args.seq, device)
    moments = Moments(device)
    loss_sum, loss_count = 0.0, 0
    with torch.no_grad():
        for batch in tokens:
            batch = batch.to(device)
            logits = forward(model, layers, batch[:, :-1], moments, cos, sin)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   batch[:, 1:].reshape(-1), reduction="sum")
            loss_sum += loss.item()
            loss_count += batch[:, 1:].numel()
    lm_loss = loss_sum / loss_count
    print(f"FORWARD lm_loss={lm_loss:.4f} reported_val={args.reported_val_loss:.4f} "
          f"tokens={loss_count}", flush=True)
    if abs(lm_loss - args.reported_val_loss) > 0.15:
        print("FORWARD does not reproduce the checkpoint; stopping", flush=True)
        return 2

    moe_layers = sorted({key[1] for key in moments.sums if key[0] == "routed_fc1"})
    experts = 1 + max(key[2] for key in moments.sums if key[0] == "routed_fc1")
    counts = torch.tensor([[moments.counts["routed_fc1", l, e] for e in range(experts)]
                           for l in moe_layers], dtype=torch.float64)
    cv = (counts.std(1, unbiased=False) / counts.mean(1))
    print(f"ROUTING tokens/expert mean={counts.mean():.0f} min={counts.min():.0f} "
          f"cv median={cv.median():.4f} max={cv.max():.4f}", flush=True)

    W1 = {l: torch.stack([model[f"decoder.layers.{l}.mlp.experts.linear_fc1.weight{e}"]
                          for e in range(experts)]).float().cpu() for l in moe_layers}
    W2 = {l: torch.stack([model[f"decoder.layers.{l}.mlp.experts.linear_fc2.weight{e}"]
                          for e in range(experts)]).float().cpu() for l in moe_layers}
    S1 = {l: torch.stack([moments.get(("routed_fc1", l, e)) for e in range(experts)]).cpu()
          for l in moe_layers}
    S2 = {l: torch.stack([moments.get(("routed_fc2", l, e)) for e in range(experts)]).cpu()
          for l in moe_layers}
    moments.sums = {k: v for k, v in moments.sums.items() if k[0] not in ("routed_fc1", "routed_fc2")}
    torch.cuda.empty_cache()

    print("FPROJ\tmatrix\tnb\tcount\tparams/dense\tmonarch\tlowrank_same_params\t"
          "monarch_on_gaussian\tmonarch_weight_energy\tinput_participation", flush=True)
    half = W1[moe_layers[0]].shape[1] // 2
    routed1 = torch.cat([W1[l] for l in moe_layers])
    routed_s1 = torch.cat([S1[l] for l in moe_layers])
    for nb in (2, 4):
        report_own("routed fc1 (gate+up)", routed1, routed_s1, nb, device)
        report_own("routed gate", routed1[:, :half], routed_s1, nb, device)
        report_own("routed up", routed1[:, half:], routed_s1, nb, device)
        report_own("routed fc2 (down)", torch.cat([W2[l] for l in moe_layers]),
                   torch.cat([S2[l] for l in moe_layers]), nb, device)
        for kind, weight, which in (("shared_fc1", "mlp.shared_experts.linear_fc1.weight", moe_layers),
                                    ("shared_fc2", "mlp.shared_experts.linear_fc2.weight", moe_layers),
                                    ("dense_fc1", "mlp.linear_fc1.weight", [0]),
                                    ("dense_fc2", "mlp.linear_fc2.weight", [0]),
                                    ("qkv", "self_attention.linear_qkv.weight", range(layers)),
                                    ("proj", "self_attention.linear_proj.weight", range(layers))):
            W = torch.stack([model[f"decoder.layers.{l}.{weight}"] for l in which]).float()
            S = torch.stack([moments.get((kind, l)) for l in which])
            report_own(kind, W, S, nb, device)
    del routed1, routed_s1

    blocks = [moe_layers[i:i + 4] for i in range(0, len(moe_layers) - 3, 4)]
    print(f"path blocks {blocks}", flush=True)
    report_tying("checkpoint", W1, W2, S1, S2, blocks, 2, device, torch.Generator().manual_seed(0),
                 args.max_sweeps)
    generator = torch.Generator().manual_seed(1)
    noise1 = {l: torch.randn(W1[l].shape, generator=generator) for l in blocks[0]}
    noise2 = {l: torch.randn(W2[l].shape, generator=generator) for l in blocks[0]}
    report_tying("gaussian", noise1, noise2, S1, S2, blocks[:1], 2, device,
                 torch.Generator().manual_seed(0), args.max_sweeps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
