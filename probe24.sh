#!/bin/bash
set -e
cd /tmp
rm -rf repo
git clone -q --depth 1 --branch kimi/blockp-tests-v2 https://github.com/AverageMetaheuristicsEnjoyer/H-MoE-Part-cloud repo
cd repo
git rev-parse HEAD
export PYTHONPATH="$PWD/third_party/Megatron-LM:$PWD/third_party/emerging-optimizers:$PWD"
python -m unittest tests.stage3_moe.test_lionmuon_lm3block -v 2>&1 | tail -30
