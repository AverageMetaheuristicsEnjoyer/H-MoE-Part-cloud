"""Every model config must reproduce the parameter ledger the probe asserts against.

`result_writer.parameter_group_ledger` refuses a run whose optimizer groups do not
sum to the declared counts. That check fires deep inside a GPU job, minutes in;
this one fires on a laptop, and it fires for the same reason: a geometry flag was
changed without the ledger following it.

The counts are rebuilt here from the config's own flags, not copied from
`stage3_moe.MODEL_SHAPES`, so the two are independent statements of the same model.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
CONFIGS = sorted((ROOT / "configs").glob("stage3-moe-*.sh"))


def flag(text, name, cast=int):
    match = re.search(rf"^\s*{re.escape(name)}\s+(\S+)\s*$", text, re.MULTILINE)
    assert match, f"{name} not found"
    return cast(match.group(1))


def variable(text, name):
    match = re.search(rf"^{re.escape(name)}=(\d+)\s*$", text, re.MULTILINE)
    assert match, f"{name} not found"
    return int(match.group(1))


def layer_split(text):
    """`--moe-layer-freq "[0]+[1]*17"` is 1 dense layer followed by 17 MoE layers."""
    match = re.search(r'--moe-layer-freq\s+"\[0\]\+\[1\]\*(\d+)"', text)
    assert match, "--moe-layer-freq not found"
    return 1, int(match.group(1))


def ledger(text):
    layers = flag(text, "--num-layers")
    hidden = flag(text, "--hidden-size")
    ffn = flag(text, "--ffn-hidden-size")
    expert_ffn = flag(text, "--moe-ffn-hidden-size")
    shared_ffn = flag(text, "--moe-shared-expert-intermediate-size")
    heads = flag(text, "--num-attention-heads")
    kv_groups = flag(text, "--num-query-groups")
    head_dim = flag(text, "--kv-channels")
    routed = flag(text, "--num-experts")
    topk = flag(text, "--moe-router-topk")
    vocab = flag(text, "--vocab-size")
    dense_layers, moe_layers = layer_split(text)

    assert dense_layers + moe_layers == layers, "layer frequency does not cover every layer"
    assert shared_ffn == expert_ffn, "the shared expert is one expert wide by design"
    assert 2 * hidden / expert_ffn == 8, "Ling granularity G = 2*d_model/d_expert must stay 8"

    embedding = 2 * vocab * hidden  # untied input and output matrices
    attention = layers * hidden * head_dim * (2 * heads + 2 * kv_groups)  # no biases
    dense_ffn = dense_layers * 3 * hidden * ffn  # SwiGLU: gate, up, down
    per_expert = 3 * hidden * expert_ffn
    experts_total = moe_layers * (routed + 1) * per_expert
    experts_active = moe_layers * (topk + 1) * per_expert
    router = moe_layers * hidden * routed
    # Two RMSNorms per layer, QKNorm on q and k, one final norm.
    norms = layers * 2 * hidden + layers * 2 * head_dim + hidden

    return {
        "total": embedding + attention + dense_ffn + experts_total + router + norms,
        "active": embedding + attention + dense_ffn + experts_active + router + norms,
        "muon_matrix": attention + dense_ffn + experts_total,
        "adamw_fallback": embedding + router + norms,
        "swiglu_fc1_weights": dense_layers + moe_layers * (routed + 1),
        "moe_layers": moe_layers,
    }


@pytest.mark.parametrize("config", CONFIGS, ids=lambda path: path.stem)
def test_config_matches_declared_shape(config):
    from stage3_moe import MODEL_SHAPES

    text = config.read_text()
    computed = ledger(text)
    name = config.stem.removeprefix("stage3-moe-")
    assert name in MODEL_SHAPES, f"{config.name} has no entry in MODEL_SHAPES"
    shape = MODEL_SHAPES[name]

    assert computed["total"] == shape.total
    assert computed["active"] == shape.active
    assert computed["muon_matrix"] == shape.muon_matrix
    assert computed["adamw_fallback"] == shape.adamw_fallback
    assert computed["swiglu_fc1_weights"] == shape.swiglu_fc1_weights
    assert computed["moe_layers"] == shape.moe_layers
    # The two roles partition the model, which is what the probe's assertion means.
    assert shape.muon_matrix + shape.adamw_fallback == shape.total
    assert shape.muon_matrix_active + shape.adamw_fallback == shape.active


@pytest.mark.parametrize("config", CONFIGS, ids=lambda path: path.stem)
def test_config_declares_its_own_counts(config):
    text = config.read_text()
    computed = ledger(text)
    assert variable(text, "STAGE3_MOE_TOTAL_PARAMETERS") == computed["total"]
    assert variable(text, "STAGE3_MOE_ACTIVE_PARAMETERS") == computed["active"]
    assert variable(text, "STAGE3_MOE_ROUTED_EXPERTS") == flag(text, "--num-experts")


def test_shape_table_has_no_unbacked_entry():
    from stage3_moe import MODEL_SHAPES

    on_disk = {path.stem.removeprefix("stage3-moe-") for path in CONFIGS}
    assert set(MODEL_SHAPES) == on_disk


def test_second_shape_moves_state_without_moving_compute():
    """The reason the 2.094B shape exists, stated as a test rather than a comment."""
    from stage3_moe import MODEL_SHAPES

    small = MODEL_SHAPES["1p029b"]
    large = MODEL_SHAPES["2p094b"]
    assert large.total / small.total > 2.0
    assert large.active / small.active < 1.1
    # Sparser than the 1.029B arm, and inside the 4.7-10.9% band Ling validated.
    for shape, bank in ((small, 9 / 65), (large, 9 / 129)):
        assert shape.routed_experts + shape.shared_experts == round(9 / bank)
    assert 0.047 <= 9 / 129 <= 0.109
