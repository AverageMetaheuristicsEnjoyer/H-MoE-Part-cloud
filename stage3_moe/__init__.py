"""Arm list and the frozen parameter ledger of every model shape the code runs.

The counts are not measured at runtime: `result_writer.parameter_group_ledger`
asserts against them, so a config change that silently alters the model is caught
where it happens rather than in a table months later.
"""

import os
from dataclasses import dataclass

ARMS = (
    "adamw_bf16_state_fp32",
    "adamw_bf16_state_fp8",
    "muon_bf16_state_fp32",
    "muon_bf16_state_fp8",
    "adamw_fp8gemm_state_fp32",
    "muon_fp8gemm_state_fp32",
)


@dataclass(frozen=True)
class ModelShape:
    """One MoE geometry and the parameter counts it must reproduce exactly."""

    name: str
    layers: int
    dense_layers: int
    routed_experts: int
    shared_experts: int
    total: int
    active: int
    muon_matrix: int
    adamw_fallback: int

    @property
    def moe_layers(self) -> int:
        return self.layers - self.dense_layers

    @property
    def swiglu_fc1_weights(self) -> int:
        """One per dense FFN plus one per expert: what Muon must split."""
        return self.dense_layers + self.moe_layers * (self.routed_experts + self.shared_experts)

    @property
    def muon_matrix_active(self) -> int:
        return self.active - self.adamw_fallback


MODEL_SHAPES = {
    shape.name: shape
    for shape in (
        # docs/design.md:303 -- 18 layers, hidden 1024, 64 routed + 1 shared expert of
        # width 256, top-8. The six 1C arms were trained at this shape.
        ModelShape(
            name="1p029b",
            layers=18,
            dense_layers=1,
            routed_experts=64,
            shared_experts=1,
            total=1_028_926_976,
            active=280_243_712,
            muon_matrix=924_844_032,
            adamw_fallback=104_082_944,
        ),
        # docs/membench.md -- the same expert width and top-k at 128 routed experts and
        # 20 layers: twice the optimizer state at almost unchanged active compute.
        ModelShape(
            name="2p094b",
            layers=20,
            dense_layers=1,
            routed_experts=128,
            shared_experts=1,
            total=2_094_088_192,
            active=301_023_232,
            muon_matrix=1_988_624_384,
            adamw_fallback=105_463_808,
        ),
    )
}

MODEL_NAME = os.environ.get("STAGE3_MOE_MODEL", "1p029b")
if MODEL_NAME not in MODEL_SHAPES:
    raise ValueError(f"unknown STAGE3_MOE_MODEL {MODEL_NAME!r}: {sorted(MODEL_SHAPES)}")
SHAPE = MODEL_SHAPES[MODEL_NAME]

TOTAL_PARAMETERS = SHAPE.total
ACTIVE_PARAMETERS = SHAPE.active
MUON_MATRIX_PARAMETERS = SHAPE.muon_matrix
ADAMW_FALLBACK_PARAMETERS = SHAPE.adamw_fallback
