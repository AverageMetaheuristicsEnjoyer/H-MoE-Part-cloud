import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from tools.coldstart_log_audit import clean_nonfinite, parse_log

optimizer, precision, seed_text = sys.argv[1:]
assert optimizer in ('adamw', 'muon') and precision in ('bf16', 'blockwise')
seed = int(seed_text)
root = Path(__file__).resolve().parents[1]
output = Path(os.environ['STAGE3_MOE_LOG_ROOT'])
output.mkdir(parents=True, exist_ok=True)
(output / '.coldstart-reserved').touch(exist_ok=False)
steps = int(os.environ.get('STAGE3_MOE_COLDSTART_ITERS', '1000'))
eval_iters = int(os.environ.get('STAGE3_MOE_EVAL_ITERS', '32'))
arm = f'{optimizer}_{"bf16" if precision == "bf16" else "fp8gemm"}_state_fp32'
tag = os.environ['STAGE3_MOE_PROBE_TAG']
run_dir = output / f'stage3-{arm}-coldstart-{tag}'
receipt = output / 'initialization-receipt.json'
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total', '--format=csv,noheader'], text=True)
assert len(gpu.strip().splitlines()) == 1, 'One visible GPU required'
data_root = next(p for p in map(Path, (
    '/home/jovyan/data/fineweb-edu-gpt2-megatron/data',
    '/workspace-SR006.nfs2/hmoe-data/fineweb-edu-gpt2-megatron/data',
    '/workspace-SR006.nfs3/hmoe-data/fineweb-edu-gpt2-megatron/data',
)) if (p / 'train.bin').is_file())
subprocess.run(['df', '-h', '/tmp', str(output)], check=True)
assert shutil.disk_usage(output).free > 1024**3, 'Less than 1 GiB free on output filesystem'
manifest = {
    'optimizer': optimizer, 'precision': precision, 'seed': seed, 'steps': steps,
    'checkpoint': None, 'state_precision': 'fp32', 'gpu': gpu.strip(),
    'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformer_engine', 'triton')},
    'data_files': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest() if p.suffix == '.idx' else None}
                   for split in ('train', 'development', 'final') for p in (data_root / f'{split}.bin', data_root / f'{split}.idx')},
    'source_manifest_sha256': hashlib.sha256((root / 'bundle_manifest.json').read_bytes()).hexdigest(),
    'protocol': {'global_batch': 208, 'micro_batch': 16, 'sequence_length': 2048, 'warmup_iters': 173,
                 'lr_decay_iters': 17242, 'wsd_decay_iters': 3448, 'eval_iters': eval_iters,
                 'routing_interval': 1, 'router_dtype': 'fp32', 'bias_update_rate': 0.001,
                 'interpretation': 'Short cold-start learning; native-recipe evaluation; no full-pretrain equivalence claim'},
}
(output / 'manifest-coldstart.json').write_text(json.dumps(manifest, indent=2) + '\n')
with tempfile.TemporaryDirectory(prefix='hmoe-coldstart-', dir='/tmp') as temporary:
    env = dict(os.environ, STAGE3_MOE_SEED=str(seed), STAGE3_MOE_MICRO_BATCH='16',
               STAGE3_MOE_COLDSTART_ITERS=str(steps), STAGE3_MOE_EP='1',
               STAGE3_MOE_DATA_ROOT=str(data_root), STAGE3_MOE_DATA_CACHE_PATH=temporary + '/data-cache',
               STAGE3_MOE_CKPT_ROOT=temporary + '/no-checkpoints', STAGE3_MOE_RUN_SUFFIX=tag,
               STAGE3_MOE_LOG_INTERVAL='1', STAGE3_MOE_ROUTING_TELEMETRY_INTERVAL='1',
               STAGE3_MOE_EVAL_INTERVAL='250', STAGE3_MOE_PROPAGATE_EXIT='1',
               STAGE3_MOE_COLDSTART_RECEIPT=str(receipt), STAGE3_MOE_IMAGE='te4', MLSUB_IMAGE='te4',
               STAGE3_MOE_FP8_COMPUTE_ARGS='--fp8-format e4m3 --fp8-recipe blockwise',
               WANDB_MODE='offline', WANDB_RUN_ID=run_dir.name)
    started = time.monotonic()
    result = subprocess.run(['bash', str(root / 'scripts/run_stage3_moe_pretrain.sh'), arm, 'coldstart'], cwd=root, env=env)
    logs = sorted(run_dir.glob('train-*.log'))
    audit = parse_log(logs[-1], 0, steps, eval_iters > 0) if logs else {}
    valid = result.returncode == 0 and audit.get('numerical_and_endpoint_checks_pass', False) and receipt.is_file()
    row = {'branch_validated': valid, 'trainer_exit': result.returncode,
           'seconds': time.monotonic() - started, 'initialization': json.loads(receipt.read_text()) if receipt.is_file() else None,
           **audit}
    (output / 'summary-coldstart.jsonl').write_text(json.dumps(clean_nonfinite(row), allow_nan=False) + '\n')
    print('COLDSTART_VALIDATED=' + str(valid).lower(), flush=True)
    print('COLDSTART_EXIT=' + str(result.returncode), flush=True)
    raise SystemExit(0 if valid else 1)
