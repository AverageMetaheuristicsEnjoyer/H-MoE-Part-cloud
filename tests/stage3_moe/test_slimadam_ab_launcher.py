import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def test_slimadam_ab_only_changes_fc1_flag_and_output_paths(tmp_path):
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    python = binaries / 'python'
    python.write_text('#!/usr/bin/env python3\nimport json,sys\nif sys.argv[1:3] == ["-m", "torch.distributed.run"]: print("CAPTURE="+json.dumps(sys.argv[1:]))\n')
    python.chmod(0o755)
    gpu = binaries / 'nvidia-smi'
    gpu.write_text('#!/bin/sh\necho 0\n')
    gpu.chmod(0o755)
    commands = []
    for split in (0, 1):
        env = os.environ.copy()
        env.update(PATH=str(binaries)+':'+env['PATH'],
                   STAGE3_MOE_CKPT_ROOT=str(tmp_path / 'checkpoints'),
                   STAGE3_MOE_LOG_ROOT=str(tmp_path / 'logs'),
                   STAGE3_MOE_RUN_SUFFIX=str(split),
                   STAGE3_MOE_SLIM_SPLIT_FC1=str(split),
                   STAGE3_MOE_PROPAGATE_EXIT='1')
        result = subprocess.run(['bash', str(ROOT / 'scripts/run_stage3_moe_pretrain.sh'),
                                 'slimadam_bf16_state_fp32', 'slim-ab'],
                                cwd=ROOT, env=env, text=True, capture_output=True, check=True)
        command = json.loads(next(line.removeprefix('CAPTURE=') for line in result.stdout.splitlines() if line.startswith('CAPTURE=')))
        for flag, value in (('--train-iters','2254'),('--lr-warmup-iters','173'),
                            ('--lr-decay-iters','17242'),('--lr-wsd-decay-iters','3448'),
                            ('--micro-batch-size','4'),('--global-batch-size','208'),
                            ('--seed','1234'),('--moe-router-bias-update-rate','1e-3'),
                            ('--save-interval','587'),('--save-retain-interval','2254')):
            assert command[command.index(flag)+1] == value
        assert ('--stage3-slim-split-fc1' in command) == bool(split)
        for flag in ('--save','--load','--stage3-result-path','--tensorboard-dir','--wandb-save-dir','--wandb-exp-name'):
            if flag in command:
                command[command.index(flag)+1] = '<arm-path>'
        commands.append([value for value in command if value != '--stage3-slim-split-fc1'])
    assert commands[0] == commands[1]
