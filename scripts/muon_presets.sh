# Named Muon-variant presets (SOW Stage 4). Sourced by cloud_quantization_probe.sh (variant suffix +NAME) and
# run_coat_arm.sh (STAGE3_MOE_MUON_PRESET=NAME). They live in the encrypted bundle because mlsub --args rejects '@'
# and --env rejects ',', and so the schedules do not show in job notes. muon_preset NAME prints K=V words.
muon_preset() {
  case "$1" in
    # LionMuon (arXiv 2605.19811): fixed period P; SignMuon is beta1 = beta2.
    lm2) echo STAGE3_MOE_MUON_VARIANT=lionmuon STAGE3_MOE_LIONMUON_PERIOD=2 ;;
    lm5) echo STAGE3_MOE_MUON_VARIANT=lionmuon STAGE3_MOE_LIONMUON_PERIOD=5 ;;
    sm2) echo STAGE3_MOE_MUON_VARIANT=lionmuon STAGE3_MOE_LIONMUON_PERIOD=2 STAGE3_MOE_LIONMUON_BETA1=0.9 STAGE3_MOE_LIONMUON_BETA2=0.9 ;;
    # A dense colleague's probability schedules of a Muon step (2026-10-19), over the training fraction, mean p = 0.2.
    lmlin) echo STAGE3_MOE_MUON_VARIANT=lionmuon STAGE3_MOE_LIONMUON_PROB=0:0.05,1:0.35 ;;
    lmlinrev) echo STAGE3_MOE_MUON_VARIANT=lionmuon STAGE3_MOE_LIONMUON_PROB=0:0.35,1:0.05 ;;
    lmtail0) echo STAGE3_MOE_MUON_VARIANT=lionmuon STAGE3_MOE_LIONMUON_PROB=0:0.2222,0.9:0.2222,0.9:0 ;;
    lmtail8) echo STAGE3_MOE_MUON_VARIANT=lionmuon STAGE3_MOE_LIONMUON_PROB=0:0.1333,0.9:0.1333,0.9:0.8 ;;
    # Power steps (2026-10-19): a constant p = 0.2 of spectral steps, the power a rising from 0 (LionMuon) to 1 over all
    # of training or over the last 10 %.
    fklin) echo STAGE3_MOE_MUON_VARIANT=lionmuon STAGE3_MOE_LIONMUON_PROB=0.2 STAGE3_MOE_LIONMUON_POWER=0:0,1:1 ;;
    fktail) echo STAGE3_MOE_MUON_VARIANT=lionmuon STAGE3_MOE_LIONMUON_PROB=0.2 STAGE3_MOE_LIONMUON_POWER=0:0,0.9:0,1:1 ;;
    # Stage 4 schedule study (2026-10-22), deterministic periods over the training fraction (the 1C plateau ends at 0.8):
    # more Lion early (P 10 -> 5 -> 2) or late (2 -> 5 -> 10); routed experts at P 10, the rest at 2; Lion LR ratio 50 / 200;
    # LionMuon with Muon^2-F's spectral step and 3 Newton-Schulz steps.
    lmph) echo STAGE3_MOE_MUON_VARIANT=lionmuon STAGE3_MOE_LIONMUON_PERIOD=0:10,0.4:5,0.8:2 ;;
    lmphrev) echo STAGE3_MOE_MUON_VARIANT=lionmuon STAGE3_MOE_LIONMUON_PERIOD=0:2,0.4:5,0.8:10 ;;
    lmexp10) echo STAGE3_MOE_MUON_VARIANT=lionmuon STAGE3_MOE_LIONMUON_PERIOD=2 STAGE3_MOE_LIONMUON_PERIOD_EXPERTS=10 ;;
    lm5r50) echo STAGE3_MOE_MUON_VARIANT=lionmuon STAGE3_MOE_LIONMUON_PERIOD=5 STAGE3_MOE_LIONMUON_LR_RATIO=50 ;;
    lm5r50ec) echo STAGE3_MOE_MUON_VARIANT=lionmuon STAGE3_MOE_LIONMUON_PERIOD=5 STAGE3_MOE_LIONMUON_LR_RATIO=50 STAGE3_MOE_LIONMUON_ENERGYCAL=1 ;;
    lm5r50fixed) echo STAGE3_MOE_MUON_VARIANT=lionmuon STAGE3_MOE_LIONMUON_PERIOD=5 STAGE3_MOE_LIONMUON_LR_RATIO=50 ;;
    lm5r200) echo STAGE3_MOE_MUON_VARIANT=lionmuon STAGE3_MOE_LIONMUON_PERIOD=5 STAGE3_MOE_LIONMUON_LR_RATIO=200 ;;
    lm5m2f3) echo STAGE3_MOE_MUON_VARIANT=lionmuon2f STAGE3_MOE_LIONMUON_PERIOD=5 STAGE3_MOE_MUON_NS_STEPS=3 ;;
    # NorMuon (arXiv 2510.05491), Muon^2 / Muon^2-F (arXiv 2604.09967), and plain Muon with 3 Newton-Schulz steps.
    nm) echo STAGE3_MOE_MUON_VARIANT=normuon ;;
    # NorMuon diagnosis (2026-10-22): only on routed experts / everywhere but them; a slower second moment.
    nmexp) echo STAGE3_MOE_MUON_VARIANT=normuon STAGE3_MOE_NORMUON_SCOPE=experts ;;
    nmdense) echo STAGE3_MOE_MUON_VARIANT=normuon STAGE3_MOE_NORMUON_SCOPE=dense ;;
    nmb999) echo STAGE3_MOE_MUON_VARIANT=normuon STAGE3_MOE_NORMUON_BETA2=0.999 ;;
    m2) echo STAGE3_MOE_MUON_VARIANT=muon2 ;;
    m2f) echo STAGE3_MOE_MUON_VARIANT=muon2f ;;
    m2f3) echo STAGE3_MOE_MUON_VARIANT=muon2f STAGE3_MOE_MUON_NS_STEPS=3 ;;
    ns3) echo STAGE3_MOE_MUON_NS_STEPS=3 ;;
    *) return 1 ;;
  esac
}
