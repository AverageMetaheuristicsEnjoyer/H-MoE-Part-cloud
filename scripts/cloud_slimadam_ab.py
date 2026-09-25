"""Run the paired SlimAdam FC1 compression experiment on allocated Cloud.ru workers."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['preflight', 'smoke', 'train', 'pipeline', 'archive-check', 'full'])
parser.add_argument('variant', choices=['baseline', 'split'])
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
experiment = 'slimadam-fc1-ab-20260924-v1'
full = args.mode == 'full'
seed_experiment = experiment
if full:
    if args.variant != 'split':
        raise RuntimeError('The approved full continuation is the split variant')
    experiment = os.environ.get('SLIM_FULL_EXPERIMENT', 'slimadam-split-full-20260925-v1')
final_step = 17242 if full else 2254
# Data parallelism and micro-batch only change the summation order: gb=208 and the
# sampler keep the per-step data stream identical, and the router bias update all-reduces.
gpus = int(os.environ.get('SLIM_AB_GPUS', '1'))
micro_batch = int(os.environ.get('SLIM_AB_MICRO_BATCH', '4'))
default_root = '/workspace-SR006.nfs2/hmoe-cloud' if args.variant == 'baseline' else '/home/jovyan/hmoe-cloud'
base = Path(os.environ.get('SLIM_AB_ROOT', default_root)) / experiment
checkpoint_root = base / 'checkpoints'
archived = os.environ.get('SLIM_AB_ARCHIVE_HF') == '1' and args.mode in ('preflight', 'train', 'full')
if archived:
    checkpoint_root = Path('/tmp') / experiment / args.variant / 'checkpoints'
log_root = base / 'logs'
suffix = f'{experiment}-{args.mode}-{args.variant}'
seed_arm = 'slimadam_bf16_state_fp32'
arm = os.environ.get('SLIM_AB_ARM', seed_arm)
if arm not in (seed_arm, 'slimadam_bf16_state_fp8', 'slimadam_fp8gemm_state_fp32', 'slimadam_fp8gemm_state_fp8'):
    raise RuntimeError(f'Unknown SlimAdam arm: {arm}')
if arm != seed_arm and not full:
    raise RuntimeError('FP8 SlimAdam arms run only as full continuations of the split seed')
launch_mode = 'full' if full else 'slim-ab'
run_dir = log_root / f'stage3-{arm}-{launch_mode}-{suffix}'
checkpoint_dir = checkpoint_root / 'slim-ab' / f'{arm}-{suffix}'
if full:
    checkpoint_dir = checkpoint_dir / arm


def run():
    if full and not archived:
        raise RuntimeError('Full continuation requires SLIM_AB_ARCHIVE_HF=1')
    if args.mode == 'archive-check':
        import tempfile
        import uuid
        from huggingface_hub import HfApi
        sys.path.insert(0, str(root))
        from stage3_moe.slim_checkpoint_archive import REPO, archive, restore
        api = HfApi(token=os.environ['HF_TOKEN'])
        prefix = f'{experiment}/archive-selftest/{uuid.uuid4().hex}'
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'checkpoint'
            path = checkpoint / 'iter_0000322/mp_rank_00/model_optim_rng.pt'
            path.parent.mkdir(parents=True)
            payload = os.urandom(4096)
            path.write_bytes(payload)
            try:
                archive(checkpoint, 322, prefix, api)
                shutil.rmtree(checkpoint)
                restore(checkpoint, Path(temporary) / 'logs', prefix, api)
                assert path.read_bytes() == payload
                print('ARCHIVE_ROUNDTRIP=PASS', flush=True)
            finally:
                api.delete_folder(repo_id=REPO, repo_type='model', path_in_repo=prefix)
        return
    if args.mode == 'pipeline':
        for stage, artifact in [('smoke', 'smoke-pass.json'), ('train', 'endpoint.json')]:
            subprocess.run([sys.executable, __file__, stage, args.variant], check=True)
            evidence = log_root / f'stage3-{arm}-slim-ab-{experiment}-{stage}-{args.variant}' / artifact
            if not evidence.exists():
                raise RuntimeError(f'{stage} did not produce success evidence')
        print('PIPELINE_RESULT=COMPLETE', flush=True)
        return
    subprocess.run(['df', '-h', '/home/jovyan', '/workspace-SR006.nfs2',
                    '/workspace-SR006.nfs3'], check=False)
    subprocess.run(['df', '-i', '/home/jovyan', '/workspace-SR006.nfs2',
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
    # Separate volumes: two measured 9.8 GiB checkpoints plus >2 GiB headroom per arm.
    required_gib = 1 if archived else 22
    if free < required_gib * 1024**3:
        raise RuntimeError(f'Need {required_gib} GiB free on the persistent volume')
    if archived:
        local_free = shutil.disk_usage('/tmp').free
        print(f'LOCAL_DISK free_bytes={local_free}', flush=True)
        if local_free < (60 if full else 40) * 1024**3:
            raise RuntimeError('Insufficient local disk for checkpoints and restore')
        if args.mode in ('train', 'full') and not os.environ.get('HF_TOKEN'):
            raise RuntimeError('HF_TOKEN is required for verified checkpoint archival')
    if args.mode == 'preflight':
        print('PREFLIGHT_RESULT=PASS', flush=True)
        return
    if os.environ.get('MLSUB_IMAGE') != 'te4':
        raise RuntimeError('This experiment requires --image te4')
    subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.total', '--format=csv'], check=True)
    env = os.environ.copy()
    spec = importlib.util.find_spec('nvidia')
    roots = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
    libraries = [path for location in roots for path in sorted(Path(location).glob('*/lib')) if path.is_dir()]
    env['LD_LIBRARY_PATH'] = ':'.join([str(path) for path in libraries] + [env.get('LD_LIBRARY_PATH', '')])
    for library in libraries:
        variable = {'cudnn': 'CUDNN_HOME', 'curand': 'CURAND_HOME', 'cuda_nvrtc': 'NVRTC_HOME'}.get(library.parent.name)
        if variable:
            env.setdefault(variable, str(library.parent))
    env['PYTHONPATH'] = ':'.join(str(root / path) for path in ('third_party/Megatron-LM', 'third_party/emerging-optimizers', '.'))
    run_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-c',
                    'import json, pathlib, sys, torch, transformer_engine as te; '
                    'assert torch.cuda.device_count() == int(sys.argv[2]); '
                    'report=dict(torch=torch.__version__, cuda=torch.version.cuda, '
                    'te=te.__version__, te_path=te.__file__, python=sys.version); '
                    'print("RUNTIME " + json.dumps(report)); '
                    'pathlib.Path(sys.argv[1]).write_text(json.dumps(report, indent=2))',
                    str(run_dir / 'runtime.json'), str(gpus)],
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
        STAGE3_MOE_MICRO_BATCH=str(micro_batch), STAGE3_MOE_EP='1',
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
    # The same FP8 settings as the fp8-combined-20260924 wave on stage3/frugal-fp8.
    if '_fp8gemm_' in arm:
        env.update(STAGE3_MOE_FP8_COMPUTE_ARGS='--fp8-format e4m3 --fp8-recipe blockwise',
                   NVTE_FP8_BLOCK_SCALING_FP32_SCALES='1')
    if arm.endswith('_state_fp8'):
        env.update(STAGE3_MOE_FP8_STATE_DTYPES='e4m3:e4m3', STAGE3_MOE_FP8_DEQUANT_CHUNK='0')
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'experiment': experiment, 'variant': args.variant, 'mode': args.mode, 'arm': arm,
        'fp8_compute': env.get('STAGE3_MOE_FP8_COMPUTE_ARGS'),
        'fp8_state_dtypes': env.get('STAGE3_MOE_FP8_STATE_DTYPES'),
        'image': env['MLSUB_IMAGE'],
        'checkpoint_archive': f'AverageMetaheuristicsEnjoyer/hmoe-stage3-checkpoints/{experiment}/{args.variant}' if archived else None,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'source_sha256': hashlib.sha256((root / 'stage3_moe/slim_adam.py').read_bytes()).hexdigest(),
        'checkpoint_dir': str(checkpoint_dir),
        'target_iteration': final_step,
        'seed_archive': f'{seed_experiment}/split' if full else None,
        'controls': {'seed': 1234, 'lr': 0.00163, 'warmup': 173, 'schedule_iters': 17242,
                     'wsd_decay_iters': 3448, 'micro_batch': micro_batch, 'global_batch': 208,
                     'data_parallel': gpus,
                     'bias_rate': 0.001, 'score_function': 'sigmoid', 'beta2': 0.95},
    }
    (run_dir / 'experiment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    tracker = checkpoint_dir / 'latest_checkpointed_iteration.txt'
    if tracker.exists() and int(tracker.read_text()) >= (14 if args.mode == 'smoke' else final_step):
        print('ALREADY_COMPLETE', flush=True)
        return
    if args.mode in ('train', 'full'):
        smoke_root = Path(default_root) / seed_experiment / 'logs' if full else log_root
        smoke = smoke_root / f'stage3-{seed_arm}-slim-ab-{seed_experiment}-smoke-{args.variant}' / 'smoke-pass.json'
        if not smoke.exists():
            raise RuntimeError('The variant must pass the save/resume smoke first')
        if json.loads(smoke.read_text())['source_sha256'] != manifest['source_sha256']:
            raise RuntimeError('Smoke evidence belongs to a different optimizer source')
    for target in ([12, 14] if args.mode == 'smoke' else [final_step]):
        if tracker.exists() and int(tracker.read_text()) >= target:
            continue
        env['STAGE3_MOE_SLIM_AB_STEPS'] = str(target)
        launch_mode = 'full' if full else 'slim-ab'
        if full:
            env['STAGE3_MOE_FULL_DIR'] = 'slim-ab/' + arm + '-' + suffix
        command = ['bash', 'scripts/run_stage3_moe_pretrain.sh', arm, launch_mode]
        if archived:
            sys.path.insert(0, str(root))
            from stage3_moe.slim_checkpoint_archive import train
            train(command, env, root, checkpoint_dir, run_dir, f'{experiment}/{args.variant}', target=final_step,
                  seed_prefix=f'{seed_experiment}/split' if full else None)
        else:
            subprocess.run(command, cwd=root, env=env, check=True)
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
        if endpoint['iteration'] != final_step or endpoint['rolling_100']['window_steps'] != 100:
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
