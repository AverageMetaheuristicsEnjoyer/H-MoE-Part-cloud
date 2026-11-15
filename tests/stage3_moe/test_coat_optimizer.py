"""GPU checks of the COAT optimizer port. Lines starting COAT_REPORT carry measurements."""
import io
import json
import math
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from stage3_moe import coat_kernels  # noqa: E402
from stage3_moe.coat_adamw import MCoreCoatAdamW  # noqa: E402

FLT_MAX = torch.finfo(torch.float32).max
SHAPES = [(1,), (127,), (128,), (129,), (1000,), (1024,), (64, 1024), (512, 1024), (1024, 256), (3, 333)]


def report(name, **values):
    print("COAT_REPORT " + json.dumps({"check": name, **values}), flush=True)


def _shuffle_min(v):
    """The kernel's skip-zero min: shfl_down tree over the last dimension (32 lanes)."""
    lanes = v.shape[-1]
    i = lanes // 2
    while i > 0:
        shifted = torch.cat([v[..., i:], v[..., lanes - i:]], dim=-1)  # out-of-range lanes read their own value
        shifted = torch.where(torch.arange(lanes, device=v.device) + i < lanes, shifted, v)
        r = shifted.abs()
        v = torch.where(r > 0, torch.minimum(v, r), v)
        i //= 2
    return v[..., 0]


def _group_minmax(x, numel):
    """Upstream block reduction for groups of 128 = 4 warps of 32."""
    groups = x.numel() // 128
    a = x.abs().view(groups, 4, 32)
    valid = (torch.arange(groups * 128, device=x.device) < numel).view(groups, 4, 32)
    warp_max = a.amax(-1)
    warp_min = _shuffle_min(torch.where(valid, a, torch.full_like(a, FLT_MAX)))
    block = torch.full((groups, 32), 1e9, device=x.device)
    block[:, :4] = warp_min
    block_max = torch.zeros((groups, 32), device=x.device)
    block_max[:, :4] = warp_max
    return block_max.amax(-1), _shuffle_min(block)


def _quantize_ref(values, numel, expand_min=16):
    gmax, gmin = _group_minmax(values, numel)
    mx, mn = gmax + 1e-30, gmin + 1e-30
    ratio = mx / mn
    sqrt_minmax = torch.sqrt(mx) * torch.sqrt(mn)
    upper = torch.tensor(448.0 * 448.0 / 2, device=values.device)
    expand = torch.floor((torch.log2(upper) / torch.log2(ratio)) * expand_min) / expand_min
    g = values.view(-1, 128)
    sign = 1 - 2 * torch.signbit(g).float()
    scaled = sign * torch.pow(g.abs() / sqrt_minmax[:, None], expand[:, None])
    scale = torch.pow(mx / sqrt_minmax, expand) / 448
    # __nv_fp8_e4m3(float) saturates to the finite range; torch's cast would give NaN.
    codes = (scaled / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return codes.view(-1)[:numel], scale, expand, sqrt_minmax


def _dequantize_ref(codes, scale, expand, sqrt_minmax, numel):
    padded = torch.zeros(scale.numel() * 128, device=codes.device)
    padded[:numel] = codes.float()
    x = padded.view(-1, 128) * scale[:, None]
    sign = 1 - 2 * torch.signbit(x).float()
    out = sign * torch.pow(x.abs(), 1 / expand[:, None]) * sqrt_minmax[:, None]
    return out.view(-1)[:numel]


def _fresh(shape, seed, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(shape, device="cuda", generator=g) * scale


def _heavy(shape, seed):
    """Gradient-like values: heavy tails, exact zeros and a dead row."""
    x = _fresh(shape, seed) * torch.exp(2 * _fresh(shape, seed + 1))
    x.view(-1)[:: 7] = 0.0
    if len(shape) == 2:
        x[0] = 0.0
    return x


def _make(shape, seed, cls=MCoreCoatAdamW, **kw):
    p = torch.nn.Parameter(_fresh(shape, seed, 0.02))
    opt = cls([p], lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8, **kw)
    return p, opt


def test_kernel_compiles_for_this_device():
    k = coat_kernels.kernels(torch.device("cuda"))
    report("compile", device=torch.cuda.get_device_name(), kernels=sorted(k.functions))


@pytest.mark.parametrize("shape", SHAPES)
def test_step_matches_upstream_algorithm(shape):
    """One fused step from non-trivial quantized states against a float32 re-statement of the kernel."""
    p, opt = _make(shape, 1)
    for step in range(3):  # populate states through the kernel itself
        p.grad = _heavy(shape, 10 + step)
        opt.step()
    st = opt.state[p]
    numel = p.numel()
    before = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    param_before = p.detach().clone()
    grad = _heavy(shape, 99)
    p.grad = grad
    opt.step()

    beta1, beta2, lr, wd, eps = 0.9, 0.95, 1e-3, 0.1, 1e-8
    step = int(before["step"].item()) + 1
    m = _dequantize_ref(before["exp_avg"].view(-1), before["scale_exp_avg"], before["expand_exp_avg"], before["sqrt_minmax_exp_avg"], numel)
    v = _dequantize_ref(before["exp_avg_sq"].view(-1), before["scale_exp_avg_sq"], before["expand_exp_avg_sq"], before["sqrt_minmax_exp_avg_sq"], numel)
    g = grad.view(-1)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    c1 = 1.0 - torch.pow(torch.tensor(beta1, device="cuda"), step)
    c2 = torch.sqrt(1.0 - torch.pow(torch.tensor(beta2, device="cuda"), step))
    denom = (torch.sqrt(v) / c2 + eps) * c1
    pref = param_before.view(-1) - lr * (m / denom + wd * param_before.view(-1))
    groups = st["scale_exp_avg"].numel()
    mp = torch.zeros(groups * 128, device="cuda"); mp[:numel] = m
    vp = torch.zeros(groups * 128, device="cuda"); vp[:numel] = v
    qm = _quantize_ref(mp, numel)
    qv = _quantize_ref(vp, numel)

    du, dref = p.detach().view(-1) - param_before.view(-1), pref - param_before.view(-1)
    param_err = ((du - dref).norm() / dref.norm()).item()
    param_err_max = ((du - dref).abs().max() / dref.abs().max()).item()
    code_match_m = (st["exp_avg"].view(-1).view(torch.uint8) == qm[0].view(torch.uint8)).float().mean().item()
    code_match_v = (st["exp_avg_sq"].view(-1).view(torch.uint8) == qv[0].view(torch.uint8)).float().mean().item()
    meta = {name: torch.equal(st[key], ref) for name, key, ref in (
        ("expand_m", "expand_exp_avg", qm[2]), ("expand_v", "expand_exp_avg_sq", qv[2]))}
    # Degenerate groups (all zero, or one element) carry expand=inf and scale 1/448; isclose treats
    # those exactly, which a relative error cannot.
    scale_err = 1.0 - min(torch.isclose(st["scale_exp_avg"], qm[1], rtol=2e-5, atol=0, equal_nan=True).float().mean().item(),
                          torch.isclose(st["scale_exp_avg_sq"], qv[1], rtol=2e-5, atol=0, equal_nan=True).float().mean().item())
    report("upstream_parity", shape=list(shape), update_rel_l2=param_err, update_rel_max=param_err_max, code_match_exp_avg=code_match_m,
           code_match_exp_avg_sq=code_match_v, scale_rel_err=scale_err, **meta)
    assert param_err < 1e-4
    assert code_match_m > 0.995 and code_match_v > 0.995
    # scale_rel_err is reported, not asserted: single-value and all-zero groups have
    # expand=inf, where powf(1+ulp, inf) differs between libraries while codes still agree.
    assert torch.isfinite(p).all()


def test_consecutive_updates_track_fp32_adamw():
    """Quantized-state drift against FP32 AdamW over 200 updates (not bitwise; quantified)."""
    shape = (512, 1024)
    p, opt = _make(shape, 2)
    ref = torch.nn.Parameter(p.detach().clone())
    ropt = torch.optim.AdamW([ref], lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    worst_cos, cosines = 1.0, []
    for step in range(200):
        g = _heavy(shape, 1000 + step) * 1e-3
        p.grad, ref.grad = g.clone(), g.clone()
        before, rbefore = p.detach().clone(), ref.detach().clone()
        opt.step(); ropt.step()
        du, dr = (p.detach() - before).view(-1), (ref.detach() - rbefore).view(-1)
        cos = torch.nn.functional.cosine_similarity(du, dr, dim=0).item()
        cosines.append(round(cos, 4))
        worst_cos = min(worst_cos, cos)
    drift = ((p.detach() - ref.detach()).norm() / (ref.detach() - _fresh(shape, 2, 0.02)).norm()).item()
    report("fp32_drift_200", worst_update_cosine=worst_cos, param_drift_over_total_movement=drift,
           cosines_first=cosines[:12], cosines_every20=cosines[::20], argmin=cosines.index(min(cosines)))
    # Reported, not asserted: on heavy-tailed gradients with exact zeros COAT's own algorithm
    # departs from FP32 (test_kernel_tracks_emulated_coat_over_many_steps isolates the port).
    assert torch.isfinite(p).all()


def _qdq(x):
    numel = x.numel()
    padded = torch.zeros(((numel + 127) // 128) * 128, device=x.device)
    padded[:numel] = x.reshape(-1)
    codes, scale, expand, sqrt_minmax = _quantize_ref(padded, numel)
    return _dequantize_ref(codes, scale, expand, sqrt_minmax, numel).view_as(x)


@pytest.mark.parametrize("kind", ["gaussian", "heavy"])
def test_kernel_tracks_emulated_coat_over_many_steps(kind):
    """Separates port fidelity (kernel vs float32 emulation of COAT) from algorithmic error (emulation vs FP32)."""
    shape = (512, 1024)
    p, opt = _make(shape, 20)
    emu = p.detach().clone()
    ref = p.detach().clone()
    m_e = torch.zeros_like(emu); v_e = torch.zeros_like(emu)
    m_r = torch.zeros_like(ref); v_r = torch.zeros_like(ref)
    b1, b2, lr, wd, eps = 0.9, 0.95, 1e-3, 0.1, 1e-8
    cos_kernel_emu, cos_emu_fp32 = [], []
    for step in range(1, 101):
        g = (_fresh(shape, 3000 + step) if kind == "gaussian" else _heavy(shape, 3000 + step)) * 1e-3
        before = p.detach().clone()
        p.grad = g.clone(); opt.step()
        c1 = 1 - torch.pow(torch.tensor(b1, device="cuda"), step)
        c2 = torch.sqrt(1 - torch.pow(torch.tensor(b2, device="cuda"), step))
        outs = []
        for w, m, v, quantized in ((emu, m_e, v_e, True), (ref, m_r, v_r, False)):
            m.mul_(b1).add_((1 - b1) * g); v.mul_(b2).add_((1 - b2) * g * g)
            old = w.clone()
            w.sub_(lr * (m / ((torch.sqrt(v) / c2 + eps) * c1) + wd * w))
            if quantized:
                m.copy_(_qdq(m)); v.copy_(_qdq(v))
            outs.append((w - old).view(-1))
        du = (p.detach() - before).view(-1)
        cos_kernel_emu.append(torch.nn.functional.cosine_similarity(du, outs[0], dim=0).item())
        cos_emu_fp32.append(torch.nn.functional.cosine_similarity(outs[0], outs[1], dim=0).item())
    report("kernel_vs_emulation", kind=kind, min_cos_kernel_vs_emulated=min(cos_kernel_emu),
           min_cos_emulated_vs_fp32=min(cos_emu_fp32), mean_cos_emulated_vs_fp32=sum(cos_emu_fp32) / len(cos_emu_fp32),
           param_rel_diff_kernel_vs_emulated=((p.detach() - emu).norm() / (emu - _fresh(shape, 20, 0.02)).norm()).item())
    assert min(cos_kernel_emu) > 0.99
    if kind == "gaussian":
        assert min(cos_emu_fp32) > 0.99


def test_zero_gradients_and_dead_groups_stay_finite():
    p, opt = _make((1000,), 3)
    for step in range(5):
        p.grad = torch.zeros_like(p)
        opt.step()
    for step in range(5):
        p.grad = _heavy((1000,), 50 + step)
        p.grad[:256] = 0
        opt.step()
    st = opt.state[p]
    assert torch.isfinite(p).all()
    report("zeros", expand_values=sorted(set(st["expand_exp_avg"].tolist()))[:5], finite=True)


def test_checkpoint_round_trip_is_bitwise():
    shape = (64, 1024)
    p, opt = _make(shape, 4)
    for step in range(5):
        p.grad = _heavy(shape, 200 + step); opt.step()
    buffer = io.BytesIO()
    torch.save({"param": p.detach().clone(), "optim": opt.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=False)
    q = torch.nn.Parameter(saved["param"].clone())
    opt2 = MCoreCoatAdamW([q], lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    opt2.load_state_dict(saved["optim"])
    for key, value in opt.state[p].items():
        other = opt2.state[q][key]
        assert value.dtype == other.dtype and torch.equal(value.cpu().view(torch.uint8) if value.dtype.itemsize == 1 else value.cpu(),
                                                          other.cpu().view(torch.uint8) if other.dtype.itemsize == 1 else other.cpu()), key
    for step in range(5):
        g = _heavy(shape, 300 + step)
        p.grad, q.grad = g.clone(), g.clone()
        opt.step(); opt2.step()
    assert torch.equal(p.detach(), q.detach())
    report("checkpoint_roundtrip", bitwise_after_5_steps=True)


def test_fp32_checkpoint_import():
    """A TE FusedAdam style state (FP32 moments, per-group step) is quantized with COAT's own code."""
    shape = (512, 1024)
    p = torch.nn.Parameter(_fresh(shape, 5, 0.02))
    m = _heavy(shape, 6) * 1e-3
    v = (_heavy(shape, 7) * 1e-3) ** 2
    sd = {"state": {0: {"exp_avg": m.clone(), "exp_avg_sq": v.clone()}},
          "param_groups": [{"params": [0], "lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1, "step": 13794}]}
    opt = MCoreCoatAdamW([p], lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    opt.load_state_dict(sd)
    st = opt.state[p]
    assert st["exp_avg"].dtype == torch.float8_e4m3fn and float(st["step"]) == 13794.0
    numel = p.numel()
    qm = _quantize_ref(m.view(-1), numel)
    match = (st["exp_avg"].view(-1).view(torch.uint8) == qm[0].view(torch.uint8)).float().mean().item()
    back = _dequantize_ref(st["exp_avg"].view(-1), st["scale_exp_avg"], st["expand_exp_avg"], st["sqrt_minmax_exp_avg"], numel)
    rel = ((back - m.view(-1)).norm() / m.norm()).item()
    report("fp32_import", code_match=match, exp_avg_qdq_rel_l2=rel)
    assert match > 0.995 and rel < 0.1
    p.grad = _heavy(shape, 8) * 1e-3
    opt.step()
    assert torch.isfinite(p).all() and float(opt.state[p]["step"]) == 13795.0


def test_momentum_codec_matches_upstream_quantizer():
    """Muon momentum: our first-order kernels against the upstream tail (via the import kernel)."""
    for shape in [(129,), (512, 1024), (1024, 256)]:
        x = _heavy(shape, 9) * 1e-2
        numel = x.numel(); groups = (numel + 127) // 128
        a = [torch.empty(shape, device="cuda", dtype=torch.float8_e4m3fn)] + [torch.empty(groups, device="cuda") for _ in range(3)]
        b = [torch.empty(shape, device="cuda", dtype=torch.float8_e4m3fn)] + [torch.empty(groups, device="cuda") for _ in range(3)]
        dummy = [torch.empty(shape, device="cuda", dtype=torch.float8_e4m3fn)] + [torch.empty(groups, device="cuda") for _ in range(3)]
        coat_kernels.momentum_quantize(x.view(-1), *a)
        coat_kernels.import_fp32_moments(x.contiguous(), (x * x).contiguous(), *b, *dummy, 16)
        assert torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8))
        for u, w in zip(a[1:], b[1:]):
            assert torch.equal(u, w)
        out = torch.empty(shape, device="cuda")
        coat_kernels.momentum_dequantize(a[0], a[1], a[2], a[3], out)
        ref = _dequantize_ref(a[0].view(-1), a[1], a[2], a[3], numel)
        assert torch.allclose(out.view(-1), ref, rtol=1e-6, atol=0)
        rel = ((out - x).norm() / x.norm()).item()
        report("momentum_codec", shape=list(shape), bitwise_vs_upstream_tail=True, qdq_rel_l2=rel)


def test_no_fp32_state_copies():
    """The fused step allocates nothing proportional to the parameter."""
    shape = (4096, 4096)
    p, opt = _make(shape, 10)
    p.grad = _heavy(shape, 11) * 1e-3
    opt.step()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    p.grad = _heavy(shape, 12) * 1e-3
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    opt.step()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base
    state_bytes = sum(v.numel() * v.element_size() for v in opt.state[p].values() if torch.is_tensor(v) and v.is_cuda)
    report("fused_step_memory", numel=p.numel(), peak_extra_bytes=extra, state_bytes=state_bytes,
           fp32_state_bytes_would_be=8 * p.numel())
    assert extra < 1 << 20


def _model_shapes():
    shapes = [(50257, 1024), (50257, 1024), (1024,)]
    dense = [(1536, 1024), (1024, 1024), (1024,), (1024,), (128,), (128,), (5632, 1024), (1024, 2816)]
    shapes += dense
    for _ in range(17):
        shapes += [(1536, 1024), (1024, 1024), (1024,), (1024,), (128,), (128,), (64, 1024), (512, 1024), (1024, 256)]
        shapes += [(512, 1024)] * 64 + [(1024, 256)] * 64
    return shapes


def test_optimizer_step_speed_and_memory_on_model_shapes():
    """A/B/C on the model's parameter shapes, sequentially on this GPU."""
    shapes = _model_shapes()
    total = sum(math.prod(s) for s in shapes)
    results = {}
    try:
        from transformer_engine.pytorch.optimizers import FusedAdam
    except ImportError:
        FusedAdam = None
    candidates = [("C_coat_fused", lambda ps: MCoreCoatAdamW(ps, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8))]
    if FusedAdam is not None:
        os.environ["STAGE3_MOE_ADAM_STATE_CODEC"] = "dre"
        os.environ["STAGE3_MOE_FP8_STATE_DTYPES"] = "e4m3:e4m3"
        from stage3_moe.optimizer_states import make_fp8_adamw
        B = make_fp8_adamw(FusedAdam)
        candidates = [("A_te_fused_fp32", lambda ps: FusedAdam(ps, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8, adam_w_mode=True)),
                      ("B_dre_adapter", lambda ps: B(ps, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8, adam_w_mode=True))] + candidates
    for name, factory in candidates:
        params = [torch.nn.Parameter(torch.randn(s, device="cuda") * 0.02) for s in shapes]
        for q in params:
            q.grad = torch.randn_like(q) * 1e-3
        opt = factory(params)
        for _ in range(3):
            opt.step()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(10):
            start.record(); opt.step(); end.record(); torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        state = sum(v.numel() * v.element_size() for st in opt.state.values() for v in st.values() if torch.is_tensor(v) and v.is_cuda)
        results[name] = {"median_ms": sorted(times)[len(times) // 2], "state_bytes": state,
                         "step_peak_extra_bytes": torch.cuda.max_memory_allocated() - base}
        del opt, params
        torch.cuda.empty_cache()
    report("optimizer_step_bench", parameters=total, tensors=len(shapes), results=results)
