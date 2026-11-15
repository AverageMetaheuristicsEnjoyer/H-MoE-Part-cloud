ARMS = (
    "adamw_bf16_state_fp32",
    "adamw_bf16_state_fp8",
    "muon_bf16_state_fp32",
    "muon_bf16_state_fp8",
    "adamw_fp8gemm_state_fp32",
    "muon_fp8gemm_state_fp32",
    "adamw_fp8gemm_state_fp8",
    "muon_fp8gemm_state_fp8",
    "frugal_coord_fp8gemm_state_fp8",
    "frugal_coord_bf16_state_fp32",
    "frugal_coord_fp8gemm_state_fp32",
    "frugal_coord_bf16_state_fp8",
    "slimadam_bf16_state_fp32",
    # COAT: FP8 activation flow (stage3_moe.coat_activation) with COAT optimizer states.
    "adamw_coat_state_fp8",
    "muon_coat_state_fp8",
    "adamw_coat_state_fp32",
    "muon_coat_state_fp32",
    # Optimizer-switching targets (SOW Stage 3 item 3): AdEMAMix and SlimAdam on the COAT flows.
    "slimadam_coat_state_fp32",
    "slimadam_fp8gemm_state_fp32",
    "ademamix_bf16_state_fp32",
    "ademamix_bf16_state_fp8",
    "ademamix_fp8gemm_state_fp32",
    "ademamix_coat_state_fp32",
    "ademamix_coat_state_fp8",
)

TOTAL_PARAMETERS = 1_028_926_976
ACTIVE_PARAMETERS = 280_243_712
MUON_MATRIX_PARAMETERS = 924_844_032
ADAMW_FALLBACK_PARAMETERS = 104_082_944
