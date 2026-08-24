#!/usr/bin/env bash
set -u

python -m pytest -q tests/stage3_moe/test_optimizer_contract.py \
  -k "shadow_diagnostic or shadow_optimizer"
code=$?
echo "TEST_EXIT=$code"
exit 0
