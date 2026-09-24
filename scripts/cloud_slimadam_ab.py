"""Run the paired SlimAdam FC1 compression experiment on allocated Cloud.ru workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['preflight', 'smoke', 'train'])
parser.add_argument('variant', choices=['baseline', 'split'])
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
experiment = 'slimadam-fc1-ab-20260924-v1'
base = Path(os.environ.get('SLIM_AB_ROOT', '/home/jovyan/hmoe-cloud')) / experiment
checkpoint_root = base / 'checkpoints'
log_root = base / 'logs'
suffix = f'{experiment}-{args.mode}-{args.variant}'
arm = 'slimadam_bf16_state_fp32'
run_dir = log_root / f'stage3-{arm}-slim-ab-{suffix}'
checkpoint_dir = checkpoint_root / 'slim-ab' / f'{arm}-{suffix}'


def run():
    subprocess.run(['df', '-h', '/home/jovyan', '/workspace-SR006.nfs2',
                    '/workspace-SR006.nfs3'], check=False)
    base.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(base).free
    print(f'DISK path={base} free_bytes={free}', flush=True)
    data = Path('/home/jovyan/data/fineweb-edu-gpt2-megatron/data')
    for split in ('train', 'development', 'final'):
        for extension in ('bin', 'idx'):
            path = data / f'{split}.{extension}'
            if not path.is_file():
                raise RuntimeError(f'Missing dataset: {path}')
    # Both arms need space for an old and new ~10 GiB checkpoint at once.
    if free < 48 * 1024**3:
        raise RuntimeError('Need 48 GiB free for the paired checkpoint saves')
    if args.mode == 'preflight':
        print('PREFLIGHT_RESULT=PASS', flush=True)
        return
    if os.environ.get('MLSUB_IMAGE') != 'torch28':
        raise RuntimeError('This experiment requires --image torch28')
    subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.total', '--format=csv'], check=True)
    env = os.environ.copy()
    libraries = Path('/home/user/conda/lib/python3.12/site-packages/nvidia')
    env['LD_LIBRARY_PATH'] = ':'.join(str(path) for path in sorted(libraries.glob('*/lib')))
    env['PYTHONPATH'] = ':'.join(str(root / path) for path in ('third_party/Megatron-LM', 'third_party/emerging-optimizers', '.'))
    subprocess.run([sys.executable, '-c',
                    'import torch, transformer_engine as te; '
                    'assert torch.cuda.device_count() == 1; '
                    'print(f"RUNTIME torch={torch.__version__} cuda={torch.version.cuda} te={te.__version__}")'],
                   env=env, check=True)
    subprocess.run([sys.executable, '-c',
                    'import runpy; '
                    'tests=runpy.run_path("tests/stage3_moe/test_memory_efficient_optimizers.py"); '
                    '[fn() for name, fn in tests.items() if name.startswith("test_")]; '
                    'print("CPU_CONTRACT=PASS")'], cwd=root, env=env, check=True)
    env.update(
        STAGE3_MOE_SLIM_SPLIT_FC1=str(int(args.variant == 'split')),
        STAGE3_MOE_RUN_SUFFIX=suffix,
        STAGE3_MOE_CKPT_ROOT=str(checkpoint_root),
        STAGE3_MOE_LOG_ROOT=str(log_root),
        STAGE3_MOE_DATA_CACHE_PATH=str(base / 'data-cache' / args.variant),
        STAGE3_MOE_MICRO_BATCH='4', STAGE3_MOE_EP='1',
        STAGE3_MOE_LR='1.63e-3', STAGE3_MOE_MIN_LR='1.63e-4',
        STAGE3_MOE_ADAM_BETA2='0.95', STAGE3_MOE_WGRAD_FUSION='0',
        STAGE3_MOE_PROPAGATE_EXIT='1',
        STAGE3_MOE_EVAL_INTERVAL='587',
        STAGE3_MOE_EVAL_ITERS='2' if args.mode == 'smoke' else '32',
        STAGE3_MOE_ROUTING_TELEMETRY_INTERVAL='10',
        STAGE3_MOE_LOG_INTERVAL='1' if args.mode == 'smoke' else '10',
        STAGE3_MOE_WANDB_PROJECT='hmoe-slimadam-fc1-ab', WANDB_MODE='offline',
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'experiment': experiment, 'variant': args.variant, 'mode': args.mode,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'source_sha256': hashlib.sha256((root / 'stage3_moe/slim_adam.py').read_bytes()).hexdigest(),
        'checkpoint_dir': str(checkpoint_dir),
        'controls': {'seed': 1234, 'lr': 0.00163, 'warmup': 173, 'schedule_iters': 17242,
                     'wsd_decay_iters': 3448, 'micro_batch': 4, 'global_batch': 208,
                     'bias_rate': 0.001, 'score_function': 'sigmoid', 'beta2': 0.95},
    }
    (run_dir / 'experiment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    tracker = checkpoint_dir / 'latest_checkpointed_iteration.txt'
    if tracker.exists() and int(tracker.read_text()) >= (14 if args.mode == 'smoke' else 2254):
        print('ALREADY_COMPLETE', flush=True)
        return
    if args.mode == 'train':
        smoke = log_root / f'stage3-{arm}-slim-ab-{experiment}-smoke-{args.variant}' / 'smoke-pass.json'
        if not smoke.exists():
            raise RuntimeError('The variant must pass the save/resume smoke first')
        if json.loads(smoke.read_text())['source_sha256'] != manifest['source_sha256']:
            raise RuntimeError('Smoke evidence belongs to a different optimizer source')
    for target in ([12, 14] if args.mode == 'smoke' else [2254]):
        if tracker.exists() and int(tracker.read_text()) >= target:
            continue
        env['STAGE3_MOE_SLIM_AB_STEPS'] = str(target)
        subprocess.run(['bash', 'scripts/run_stage3_moe_pretrain.sh', arm, 'slim-ab'],
                       cwd=root, env=env, check=True)
        if not tracker.exists() or int(tracker.read_text()) != target:
            raise RuntimeError(f'Checkpoint tracker did not reach {target}')
        compression = json.loads((run_dir / 'slim_compression_manifest.json').read_text())
        fc1 = [row for row in compression if '.linear_fc1.weight' in row['name']]
        if len(fc1) != 1106 or any(row['split_fc1'] != (args.variant == 'split') for row in fc1):
            raise RuntimeError('FC1 assignment contract failed')
        if any(row['split_fc1'] for row in compression if '.linear_fc1.weight' not in row['name']):
            raise RuntimeError('Split enabled outside FC1')
    if args.mode == 'smoke':
        logs = '\n'.join(path.read_text() for path in run_dir.glob('train-*.log'))
        import re
        if not re.search(r'successfully loaded checkpoint[^\n]*iteration\s+12', logs):
            raise RuntimeError('No evidence of loading checkpoint 12')
        (run_dir / 'smoke-pass.json').write_text(json.dumps(manifest, indent=2) + '\n')
        # Only this disposable smoke directory is removed, after successful reload.
        shutil.rmtree(checkpoint_dir)
        print('SMOKE_RESULT=PASS', flush=True)
    else:
        rows = [json.loads(line) for line in (run_dir / 'routing_telemetry.jsonl').read_text().splitlines()]
        endpoint = rows[-1]
        if endpoint['iteration'] != 2254 or endpoint['rolling_100']['window_steps'] != 100:
            raise RuntimeError('Final routing window is incomplete')
        routing = endpoint['rolling_100']
        passed = (routing['minimum_to_mean_min'] >= 0.10
                  and routing['coefficient_of_variation_max'] < 0.20
                  and endpoint['dropped_tokens'] == 0)
        (run_dir / 'endpoint.json').write_text(json.dumps({
            **manifest, 'routing_gate_pass': passed, 'routing': endpoint,
        }, indent=2) + '\n')
        print(f'ROUTING_GATE={"PASS" if passed else "FAIL"}', flush=True)
        print('CALIBRATION_RESULT=COMPLETE', flush=True)
    print(f'ARTIFACTS={run_dir}', flush=True)


try:
    run()
except Exception:
    import traceback
    traceback.print_exc()
    print('EXIT=1', flush=True)
else:
    print('EXIT=0', flush=True)
