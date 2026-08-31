import argparse
import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MCORE_ROOT = ROOT / "third_party" / "Megatron-LM"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MCORE_ROOT))


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--monarch-blocks", type=int, choices=(2, 4), required=True)
    args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]

    from stage3_moe.monarch import install_monarch_model

    install_monarch_model(args.monarch_blocks)
    if "--optimizer" in remaining and remaining[remaining.index("--optimizer") + 1] == "muon":
        from stage3_moe.muon import install_muon_contract
        from stage3_moe.monarch import install_monarch_muon_contract

        install_muon_contract(fp8_states=False)
        install_monarch_muon_contract()

    print(
        f"HMOE_MONARCH blocks={args.monarch_blocks} "
        f"rank={os.environ.get('RANK', '0')} local_rank={os.environ.get('LOCAL_RANK', '0')} "
        f"pid={os.getpid()}",
        flush=True,
    )
    runpy.run_path(str(MCORE_ROOT / "pretrain_gpt.py"), run_name="__main__")


if __name__ == "__main__":
    main()
