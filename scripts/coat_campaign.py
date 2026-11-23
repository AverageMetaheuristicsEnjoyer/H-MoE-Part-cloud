"""Entry point of the encrypted COAT series payload.

  worker ARM_ID...   claim arms (four GPUs each) atomically on NFS and run them concurrently
  test               GPU tests of the COAT ports
  inspect            read-only progress of every arm of the series
  preflight          bounded environment/storage/artifact checks; no training
  contract003        tiny EnergyCal optimizer fixtures and artifact checks; no training
  contract004        pinned-original/real-NS/MCore tiny contracts; no training
  contract005        reviewed real-NS/MCore tiny contracts; no training
  contract006        zero-direction/precision-evidence tiny contracts; no training
  contract007        lazy-state/zero-uphalf tiny contracts; no training
  measurement008     checkpoint-free matched 90-step pilot; no weights saved
  measurement008fix  analysis-only phase-summary correction; no training
  analysis009        saved JSON block/timing analysis; no model/CUDA/training
  analysis010        recovered saved JSON analysis with full inventory; no training
  analysis011        autonomous saved JSON analysis; no model/CUDA/training
  postwarmup012      two checkpoint-free 270-step arms; no save/load/eval
  overnight013       checkpoint-free600-step development-validation stages; no test
  fullrun014         storage/telemetry/data admission audit; no training/checkpoint save
  screenb015         fresh audit + matched600-step development screen B (lm5r50 vs lmlinrevr50); no test/checkpoints
  baselineeval030    HF fetch + eval-only full-development re-evaluation of the October lm5r50 endpoint; dev only
"""
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
campaign = Path(os.environ.get('COAT_CAMPAIGN_ROOT', '/workspace-SR006.nfs3/hmoe-cloud/coat-20261002'))
# Round r1 (2026-10-02 00:46 UTC+3) ran every arm of an 8-GPU job on GPUs 0-3; its claims are void.
ROUND = os.environ.get('COAT_ROUND', 'r2')
GPUS_PER_ARM = 4
# Claim order for jobs submitted with BUNDLE_ROLE=coat; the payload decides, not the job's args,
# so queued jobs pick up a new priority without being resubmitted. User priority 2026-10-02 00:45:
# COAT FP8 activations + FP32 states, then BF16 GEMM + COAT FP8 states, each for AdamW and Muon;
# Full COAT (both) is not queued: the user asked for exactly these four processes (00:55).
COAT_PRIORITY = ['adamw-coatact', 'muon-coatact', 'adamw-coatopt', 'muon-coatopt']


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare_source():
    if not (root / '.git').exists():
        # The result writer records a real Git snapshot of the executed source.
        env = dict(os.environ, GIT_AUTHOR_DATE='2026-10-02T00:00:00+00:00',
                   GIT_COMMITTER_DATE='2026-10-02T00:00:00+00:00')
        for command in (['git', 'init', '-q'], ['git', 'add', '.'],
                        ['git', '-c', 'user.name=COAT series', '-c', 'user.email=coat@localhost',
                         'commit', '-qm', 'Executed encrypted source snapshot']):
            subprocess.run(command, cwd=root, env=env, check=True)
    os.environ['PYTHONPATH'] = ':'.join(str(root / p) for p in ('third_party/Megatron-LM', 'third_party/emerging-optimizers', '.'))
    os.environ.pop('PYTHONNOUSERSITE', None)


def worker(arm_ids):
    gpus = subprocess.run(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total', '--format=csv,noheader'],
                          capture_output=True, text=True, check=True).stdout.strip().splitlines()
    print('WORKER_GPUS ' + json.dumps(gpus), flush=True)
    subprocess.run(['df', '-h', '/tmp', str(campaign.parent)], check=False)
    claims = campaign / f'claims-{ROUND}'
    claims.mkdir(parents=True, exist_ok=True)
    slots = len(gpus) // GPUS_PER_ARM
    job = os.environ.get('MLSUB_JOB_NAME') or os.environ.get('HOSTNAME', 'unknown')
    running = []
    for arm_id in arm_ids:
        if len(running) == slots:
            break
        try:
            (claims / arm_id).mkdir()
        except FileExistsError:
            print(f'WORKER_SKIP_CLAIMED arm_id={arm_id} by={(claims / arm_id / "owner.json").read_text() if (claims / arm_id / "owner.json").exists() else "?"}', flush=True)
            continue
        devices = ','.join(str(i) for i in range(len(running) * GPUS_PER_ARM, (len(running) + 1) * GPUS_PER_ARM))
        (claims / arm_id / 'owner.json').write_text(json.dumps({
            'job': job, 'bundle': os.environ.get('BUNDLE_ID'), 'devices': devices, 'gpus': gpus, 'utc': utc()}) + '\n')
        log_dir = campaign / f'wrappers-{ROUND}'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f'{arm_id}-{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}.log'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices, STAGE3_MOE_NPROC=str(GPUS_PER_ARM), COAT_CAMPAIGN_ROOT=str(campaign), COAT_ROUND=ROUND)
        handle = open(log_path, 'w')
        process = subprocess.Popen(['bash', str(root / 'scripts/run_coat_arm.sh'), arm_id], cwd=root, env=env,
                                   stdout=handle, stderr=subprocess.STDOUT)
        running.append((arm_id, process, log_path, handle))
        print(f'WORKER_CLAIM arm_id={arm_id} devices={devices} log={log_path}', flush=True)
        # Staggered starts keep two arms from building their data caches at the same moment.
        time.sleep(30)
    if not running:
        print('WORKER_NOTHING_TO_DO', flush=True)
        return 0
    while running:
        for item in list(running):
            arm_id, process, log_path, handle = item
            if process.poll() is not None:
                handle.close()
                lines = log_path.read_text(errors='replace').splitlines()
                print(f'WORKER_ARM_EXIT arm_id={arm_id} code={process.returncode}', flush=True)
                for line in lines:
                    if line.startswith(('COAT_ARM_', 'COAT_ARCHIVE', 'TRAIN_EXIT', 'TRAIN_LOG')):
                        print(line, flush=True)
                running.remove(item)
        time.sleep(60)
    return 0


def inspect():
    subprocess.run(['df', '-h', str(campaign.parent)], check=False)
    for path in sorted(campaign.glob(f'claims-{ROUND}/*/owner.json')):
        print('CLAIM ' + path.parent.name + ' ' + path.read_text().strip(), flush=True)
    for path in sorted(p for pattern in (f'runs-{ROUND}/*/*/train-*.log', f'smoke/runs-{ROUND}/*/*/train-*.log',
                                         f'bench/runs-{ROUND}/*/*/train-*.log', 'coldstart/*/*/train-*.log') for p in campaign.glob(pattern)):
        lines = path.read_text(errors='replace').splitlines()
        steps = [line for line in lines if re.search(r'iteration\s+\d+/', line)]
        evals = [line for line in lines if 'loss at iteration' in line]
        errors = [line for line in lines if any(k in line for k in ('Traceback (most recent call last)', 'Error:', 'AssertionError', 'nan iterations: [1-9]'))]
        print('PROGRESS ' + json.dumps({'log': str(path), 'mtime': path.stat().st_mtime, 'last_step': steps[-1] if steps else None,
                                        'evals': evals[-4:], 'errors': errors[-3:], 'tail': lines[-2:]}), flush=True)
    for path in sorted(p for pattern in ('wrappers-r*/*.log', f'*/wrappers-{ROUND}/*.log') for p in campaign.glob(pattern)):
        for line in path.read_text(errors='replace').splitlines():
            if line.startswith(('COAT_ARM_', 'COAT_ARCHIVE', 'TRAIN_EXIT')):
                print('WRAPPER ' + path.name + ' ' + line, flush=True)
    return 0


def bounded_payload(script_name, root_variable, marker, timeout_seconds=600):
    output = Path(os.environ[root_variable])
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / 'payload-progress.jsonl'
    started = time.monotonic()

    def progress(state, **fields):
        record = {'state': state, 'utc': utc(), 'elapsed_seconds': round(time.monotonic() - started, 3), **fields}
        with progress_path.open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    progress('started', hard_timeout_seconds=timeout_seconds, kill_after_seconds=5)
    stdout_path = output / 'payload.stdout.log'
    stderr_path = output / 'payload.stderr.log'
    code = 125
    try:
        with stdout_path.open('w') as stdout, stderr_path.open('w') as stderr:
            try:
                result = subprocess.run(['timeout', '-k', '5s', f'{timeout_seconds}s', sys.executable,
                                         str(root / 'scripts' / script_name)],
                                        cwd=root, stdout=stdout, stderr=stderr)
                code = result.returncode
            except Exception as error:
                stderr.write(f'PREFLIGHT_PARENT_ERROR {error!r}\n')
                stderr.flush()
                os.fsync(stderr.fileno())
    finally:
        elapsed = time.monotonic() - started
        timeout = code in (124, 137)
        progress('finished', exit=code, timeout=timeout,
                 timeout_class='payload_hard_timeout_600s' if timeout else None)
        exit_record = {'exit': code, 'elapsed_seconds': round(elapsed, 3),
                       'timeout': timeout, 'hard_timeout_seconds': timeout_seconds, 'kill_after_seconds': 5}
        with (output / 'payload-exit.json').open('w') as stream:
            stream.write(json.dumps(exit_record, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    print(f'{marker}_PAYLOAD_EXIT={code} {marker}_PAYLOAD_SECONDS={elapsed:.3f} TIMEOUT={code in (124, 137)}', flush=True)
    for label, path in (('STDOUT', stdout_path), ('STDERR', stderr_path)):
        print(f'{marker}_{label}_PATH={path}', flush=True)
        print(f'{marker}_{label}_BEGIN', flush=True)
        print(path.read_text(errors='replace') if path.exists() else 'missing', flush=True)
        print(f'{marker}_{label}_END', flush=True)
    return code


def preflight():
    return bounded_payload('lionmoe_preflight.py', 'HMOE_PREFLIGHT_ROOT', 'PREFLIGHT')


def contract003():
    return bounded_payload('lionmoe_contract_003.py', 'HMOE_CONTRACT_ROOT', 'CONTRACT003')


def contract004():
    return bounded_payload('lionmoe_contract_004.py', 'HMOE_CONTRACT_ROOT', 'CONTRACT004')


def contract005():
    return bounded_payload('lionmoe_contract_005.py', 'HMOE_CONTRACT_ROOT', 'CONTRACT005')


def contract006():
    return bounded_payload('lionmoe_contract_006.py', 'HMOE_CONTRACT_ROOT', 'CONTRACT006')


def contract007():
    return bounded_payload('lionmoe_contract_007.py', 'HMOE_CONTRACT_ROOT', 'CONTRACT007')


def measurement008():
    return bounded_payload('lionmoe_measurement_008.py', 'HMOE_CONTRACT_ROOT', 'MEASUREMENT008', 3600)


def measurement008fix():
    return bounded_payload('lionmoe_measurement_008_fix.py', 'HMOE_CONTRACT_ROOT', 'MEASUREMENT008FIX', 3600)


def analysis009():
    return bounded_payload('lionmoe_analysis_009.py', 'HMOE_CONTRACT_ROOT', 'ANALYSIS009', 300)


def analysis010():
    return bounded_payload('lionmoe_analysis_010.py', 'HMOE_CONTRACT_ROOT', 'ANALYSIS010', 300)


def analysis011():
    return bounded_payload('lionmoe_analysis_011.py', 'HMOE_CONTRACT_ROOT', 'ANALYSIS011', 300)


def postwarmup012():
    return bounded_payload('lionmoe_postwarmup_012.py', 'HMOE_CONTRACT_ROOT', 'POSTWARMUP012', 6600)


def overnight013():
    deadline = datetime.datetime(2026, 11, 17, 5, 50, tzinfo=datetime.timezone.utc)
    remaining = int((deadline - datetime.datetime.now(datetime.timezone.utc)).total_seconds())
    if remaining <= 0:
        raise RuntimeError('Overnight training deadline has passed')
    return bounded_payload('lionmoe_overnight_013.py', 'HMOE_CONTRACT_ROOT', 'OVERNIGHT013', min(23400, remaining))


def fullrun014():
    return bounded_payload('lionmoe_fullrun_014.py', 'HMOE_CONTRACT_ROOT', 'FULLRUN014', 600)


def screenb015():
    # Hard payload cap is configurable; the leader sets ~16200s (2 arms x ~6500s + audit + reserve).
    cap = int(os.environ.get('HMOE_SCREENB_PAYLOAD_CAP_SECONDS', '16200'))
    return bounded_payload('lionmoe_screenb_015.py', 'HMOE_CONTRACT_ROOT', 'SCREENB015', cap)


def baselineeval030():
    # Short dev-only job: HF preflight + 2GB download + full-development evaluation; cap configurable.
    cap = int(os.environ.get('HMOE_BASELINE_EVAL_CAP_SECONDS', '3600'))
    return bounded_payload('lionmoe_baseline_eval_030.py', 'HMOE_CONTRACT_ROOT', 'BASELINE_EVAL_V030', cap)


STEP_RE = re.compile(r"iteration\s+(\d+)/\s*\d+ \|.*?elapsed time per iteration \(ms\): ([0-9.]+).*?lm loss: ([0-9.eE+-]+).*?grad norm: ([0-9.eE+-]+)"
                     r".*?skipped iterations:\s+(\d+).*?nan iterations:\s+(\d+)")
EVAL_RE = re.compile(r"loss at iteration\s+(\d+)(?: on (validation|test) set)?\s*\| lm loss value: ([0-9.eE+-]+)")
MEM_RE = re.compile(r"\(after (\d+) iterations\) memory \(MB\) \| allocated: ([0-9.]+) \| max allocated: ([0-9.]+) \| reserved: ([0-9.]+) \| max reserved: ([0-9.]+)")


def parse_log(text):
    steps, evals, memory = [], [], []
    for line in text.splitlines():
        if m := STEP_RE.search(line):
            steps.append([int(m[1]), float(m[3]), float(m[4]), float(m[2]), int(m[5]), int(m[6])])
        elif m := EVAL_RE.search(line):
            evals.append([int(m[1]), m[2] or "validation", float(m[3])])
        elif m := MEM_RE.search(line):
            memory.append([int(m[1])] + [float(m[i]) for i in range(2, 6)])
    return {"steps": steps, "evals": evals, "memory": memory}


def collect(prefixes=()):
    """Compact metrics of every run, as gzip+base64 chunks in the job log.

    `mlsub logs` returns only the last ~200 lines, so once the campaign outgrows ~190 chunks the caller
    restricts the collection to paths starting with one of `prefixes` (e.g. `collect quantprobe/`)."""
    import base64
    import gzip
    import hashlib
    runs = {}
    # Every round: a run restarted after its first claim lives under the next round's directory.
    for path in sorted(p for pattern in ('runs-*/*/*/train-*.log', f'bench/runs-{ROUND}/*/*/train-*.log',
                                         'coldstart/*/*/train-*.log', 'quantprobe/*/*/train-*.log') for p in campaign.glob(pattern)):
        runs[str(path.relative_to(campaign))] = parse_log(path.read_text(errors='replace'))
    for path in sorted(p for pattern in ('**/results.jsonl', 'coldstart/*/summary-coldstart.jsonl', 'coldstart/*/initialization-receipt.json')
                       for p in campaign.glob(pattern)):
        runs[str(path.relative_to(campaign))] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for path in sorted(campaign.glob('official-qoptim/*/*.json')):
        runs[str(path.relative_to(campaign))] = json.loads(path.read_text())
    for path in sorted(campaign.glob('evalfetch/*/*/downstream/downstream.json')):
        runs[str(path.relative_to(campaign))] = json.loads(path.read_text())
    # Routing balance of every full run, every 50th iteration: [iteration, batch maxvio mean, maxvio max,
    # min/mean min, CV max, rolling-100 maxvio mean, rolling-100 CV max, expert-bias range max, load drift mean, max].
    for path in sorted(p for pattern in ('runs-*/*/*/routing_telemetry.jsonl', 'quantprobe/*/*/routing_telemetry.jsonl') for p in campaign.glob(pattern)):
        rows = []
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record['iteration'] % 50 == 0 or record['iteration'] == 1:
                batch, rolling, drift = record['batch'], record['rolling_100'], record['drift']
                rows.append([record['iteration'], batch['maxvio_mean'], batch['maxvio_max'], batch['minimum_to_mean_min'],
                             batch['coefficient_of_variation_max'], rolling['maxvio_mean'], rolling['coefficient_of_variation_max'],
                             record['expert_bias']['range_max'], drift['load_total_variation_from_previous_mean'],
                             drift['load_total_variation_from_previous_max']])
        runs[str(path.relative_to(campaign))] = rows
    for path in sorted(campaign.glob('official-qoptim/*/build.log')):
        runs[str(path.relative_to(campaign))] = path.read_text(errors='replace').splitlines()[-60:]
    if prefixes:
        runs = {k: v for k, v in runs.items() if k.startswith(tuple(prefixes))}
    payload = gzip.compress(json.dumps(runs).encode())
    encoded = base64.b64encode(payload).decode()
    chunks = [encoded[i:i + 12000] for i in range(0, len(encoded), 12000)]
    for index, chunk in enumerate(chunks):
        print(f'COLLECT_CHUNK {index} {chunk}', flush=True)
    print('COLLECT_RECEIPT ' + json.dumps({'runs': len(runs), 'chunks': len(chunks), 'bytes': len(payload),
                                           'sha256': hashlib.sha256(payload).hexdigest()}), flush=True)
    return 0


def cpu_snapshot():
    """The CPU this container may use and how much of its demand was throttled so far."""
    out = {'time': time.time(), 'cpu_count': os.cpu_count(), 'affinity': len(os.sched_getaffinity(0)),
           'loadavg': Path('/proc/loadavg').read_text().split()[:3]}
    for name in ('cpu.max', 'cpu.stat', 'cpu/cpu.cfs_quota_us', 'cpu/cpu.cfs_period_us', 'cpu/cpu.stat', 'cpuacct/cpuacct.usage'):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            out[name] = path.read_text().split()
    return out


def speeddiag(specs):
    """Step time of several arms in ONE allocation, forward then reverse order, with the GPU
    utilization and the container's CPU use and throttling recorded around every run.
    Spec: ARM_ID[+KEY=VALUE...]; e.g. adamw-coatact+STAGE3_MOE_COAT_GROUPED_GEMM=1 (mlsub arguments take no '@')."""
    import statistics
    visible = len(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, check=True).stdout.strip().splitlines())
    stamp = f'{datetime.datetime.now(datetime.timezone.utc):%d%H%M}'
    print('SPEEDDIAG_HOST ' + json.dumps({'gpus': visible, 'cpu': cpu_snapshot(), 'omp': os.environ.get('OMP_NUM_THREADS'),
          'cpu_model': next((l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')), None),
          'nvidia_smi': subprocess.run(['nvidia-smi', '--query-gpu=index,name,uuid,pci.bus_id,utilization.gpu,memory.used', '--format=csv,noheader'],
                                       capture_output=True, text=True).stdout.strip().splitlines()}), flush=True)
    for order, sequence in (('fwd', specs), ('rev', list(reversed(specs)))):
        for index, spec in enumerate(sequence):
            arm_id, *pairs = spec.split('+')
            extra = dict(pair.split('=', 1) for pair in pairs)
            tag = f'sd{stamp}-{order}{index}'
            env = dict(os.environ, COAT_ARM_MODE='bench', COAT_BENCH_TAG=tag, STAGE3_MOE_NPROC=str(visible),
                       COAT_BENCH_WARMUP=os.environ.get('COAT_BENCH_WARMUP', '8'), COAT_BENCH_MEASURE=os.environ.get('COAT_BENCH_MEASURE', '16'),
                       COAT_CAMPAIGN_ROOT=str(campaign / 'speeddiag'), **extra)
            samples = Path(f'/tmp/speeddiag-{tag}.csv')
            sampler = subprocess.Popen(['nvidia-smi', '--query-gpu=utilization.gpu,power.draw', '--format=csv,noheader,nounits', '-lms', '500'],
                                       stdout=open(samples, 'w'), stderr=subprocess.DEVNULL)
            before = cpu_snapshot()
            result = subprocess.run(['bash', str(root / 'scripts/run_coat_arm.sh'), arm_id], cwd=root, env=env, capture_output=True, text=True)
            after = cpu_snapshot()
            sampler.terminate()
            record = {'spec': spec, 'order': order, 'wall_s': round(after['time'] - before['time'], 1), 'cpu_before': before, 'cpu_after': after}
            for line in result.stdout.splitlines():
                if line.startswith('COAT_BENCH '):
                    record['bench'] = json.loads(line[len('COAT_BENCH '):])
                elif 'Traceback' in line or 'Error' in line:
                    record.setdefault('errors', []).append(line[:300])
            for path in (campaign / 'speeddiag').glob(f'runs-{ROUND}/{arm_id}/*{tag}/results.jsonl'):
                timing = json.loads(path.read_text().splitlines()[-1])['measurement']['timing']
                record['step_samples_s'] = [round(v, 3) for v in timing.get('full_step_samples_seconds') or []]
                record['optimizer_samples_s'] = [round(v, 3) for v in timing.get('optimizer_step_samples_seconds') or []]
            rows = [line.split(',') for line in samples.read_text().splitlines() if line.count(',') == 1]
            # The measured window is the end of the run; the first part is start-up and compilation.
            tail = rows[len(rows) * 6 // 10:len(rows) * 95 // 100]
            if tail:
                util = [float(r[0]) for r in tail]
                record['gpu_util_pct'] = {'mean': round(statistics.fmean(util), 1), 'median': statistics.median(util),
                                          'below_50_share': round(sum(u < 50 for u in util) / len(util), 3), 'samples': len(util)}
                record['gpu_power_w_mean'] = round(statistics.fmean(float(r[1]) for r in tail), 1)
            print('COAT_SPEEDDIAG ' + json.dumps(record), flush=True)
    return 0


def probe_record(optimizer, row, order):
    """One summary row of cloud_quantization_probe.sh with its timing and memory from results.jsonl."""
    import statistics
    record = {'optimizer': optimizer, 'variant': row['variant'], 'order': order, 'valid': row['branch_validated'],
              'seconds': row['seconds'], 'validation': row['validation'][-1:], 'test': row['test'][-1:],
              'steps': len(row['steps']), 'nans': row['nans'], 'skipped': row['skipped'], 'lr_first': row.get('lr_first'),
              'lr_last': next((step['lr'] for step in reversed(row['steps']) if 'lr' in step), None)}
    if row['steps']:
        losses = [step['loss'] for step in row['steps']]
        norms = [step['grad_norm'] for step in row['steps'] if step['grad_norm'] is not None]
        record.update(train_loss_last100=round(statistics.fmean(losses[-100:]), 6), grad_norm_max=max(norms, default=None),
                      grad_norm_median=statistics.median(norms) if norms else None)
    results = Path(row['run_dir']) / 'results.jsonl'
    if results.is_file():
        measurement = json.loads(results.read_text().splitlines()[-1])['measurement']
        timing, memory = measurement['timing'], measurement['memory']
        record.update(step_s=timing.get('full_step_seconds'), optimizer_s=timing.get('optimizer_step_seconds'),
                      tokens_per_s=timing.get('tokens_per_second'), measured_steps=measurement['protocol'].get('measured_steps'),
                      max_allocated_gb=round(memory['max_allocated_bytes'] / 2**30, 3),
                      max_reserved_gb=round(memory['max_reserved_bytes'] / 2**30, 3))
    return record


def abba(legs):
    """Step time of optimizers and variants on one allocation (SOW Stage 4 throughput): abba LEG... with
    LEG = OPTIMIZER:VARIANT[+PRESET] (cloud_quantization_probe.sh variants, scripts/muon_presets.sh presets). The legs run forward and then
    reversed, COAT_ABBA_ROUNDS times, each a no-save, no-eval continuation of STAGE3_MOE_PROBE_WARMUP (20) +
    STAGE3_MOE_PROBE_MEASURE (100) steps from that optimizer's checkpoint STAGE3_MOE_PROBE_BASE_ITER (COAT_FETCH_CKPT_<OPTIMIZER>
    = NAME:SHA256 from the archive, else the HF 1c-mb4 checkpoint). Ratios are against the first leg; the micro-batch is
    STAGE3_MOE_MICRO_BATCH and the data-parallel size the job's GPU count."""
    import statistics
    warmup, measure = int(os.environ.get('STAGE3_MOE_PROBE_WARMUP', '20')), int(os.environ.get('STAGE3_MOE_PROBE_MEASURE', '100'))
    stamp = f'{datetime.datetime.now(datetime.timezone.utc):%d%H%M}{os.urandom(2).hex()}'
    job = (os.environ.get('MLSUB_JOB_NAME') or os.environ.get('HOSTNAME', 'job'))[-12:]
    output = campaign / 'abba' / f'{stamp}-{job}'
    gpus = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip().splitlines()
    micro_batch = os.environ.get('STAGE3_MOE_MICRO_BATCH', '16' if len(gpus) == 1 else '13')
    print('COAT_ABBA_HOST ' + json.dumps({'gpus': gpus, 'micro_batch': micro_batch, 'legs': legs, 'output': str(output), 'cpu': cpu_snapshot()}), flush=True)
    timings = {leg: [] for leg in legs}
    for round_index in range(int(os.environ.get('COAT_ABBA_ROUNDS', '1'))):
        for order, sequence in (('fwd', legs), ('rev', list(reversed(legs)))):
            for position, leg in enumerate(sequence):
                optimizer, variant = leg.split(':', 1)
                tag = f'ab{stamp}-r{round_index}-{order}{position}'
                env = dict(os.environ, STAGE3_MOE_LOG_ROOT=str(output), STAGE3_MOE_PROBE_TAG=tag, STAGE3_MOE_PROBE_GPUS=str(len(gpus)),
                           STAGE3_MOE_MICRO_BATCH=micro_batch, STAGE3_MOE_PROBE_ITERS=str(warmup + measure),
                           STAGE3_MOE_PROBE_WARMUP=str(warmup), STAGE3_MOE_PROBE_MEASURE=str(measure),
                           STAGE3_MOE_EVAL_ITERS='0', STAGE3_MOE_PROBE_HASH='0')
                env.pop('COAT_FETCH_CKPT', None)
                if os.environ.get(f'COAT_FETCH_CKPT_{optimizer.upper()}'):
                    env['COAT_FETCH_CKPT'] = os.environ[f'COAT_FETCH_CKPT_{optimizer.upper()}']
                result = subprocess.run(['bash', str(root / 'scripts/cloud_quantization_probe.sh'), optimizer, variant],
                                        cwd=root, env=env, capture_output=True, text=True)
                for line in (result.stdout + result.stderr).splitlines():
                    if line.startswith(('PROBE_ABORT', 'VARIANT_EXIT', 'QUANTIZATION_PROBE_EXIT', 'COAT_FETCH', 'LIONMUON', 'NORMUON', 'MUON2')) or 'stage3 muon' in line or 'Traceback' in line or 'Error:' in line:
                        print('COAT_ABBA_LOG ' + line[:400], flush=True)
                summary = output / f'summary-{optimizer}-{tag}.jsonl'
                for row in (json.loads(line) for line in summary.read_text().splitlines()) if summary.is_file() else ():
                    record = dict(probe_record(optimizer, row, order), leg=leg, round=round_index, micro_batch=int(micro_batch), dp=len(gpus))
                    print('COAT_ABBA ' + json.dumps(record), flush=True)
                    if record['valid'] and record.get('step_s'):
                        timings[leg].append(record)
    reference = statistics.median(r['step_s'] for r in timings[legs[0]]) if timings[legs[0]] else None
    for leg in legs:
        rows = timings[leg]
        if not rows:
            print('COAT_ABBA_SUMMARY ' + json.dumps({'leg': leg, 'valid_runs': 0}), flush=True)
            continue
        step = statistics.median(r['step_s'] for r in rows)
        optimizer_s = [r['optimizer_s'] for r in rows if r.get('optimizer_s') is not None]
        print('COAT_ABBA_SUMMARY ' + json.dumps({
            'leg': leg, 'valid_runs': len(rows), 'step_s_median': step, 'step_s_runs': [r['step_s'] for r in rows],
            'optimizer_s_median': statistics.median(optimizer_s) if optimizer_s else None,
            'tokens_per_s_median': statistics.median(r['tokens_per_s'] for r in rows if r.get('tokens_per_s')),
            'max_allocated_gb': max(r.get('max_allocated_gb') or 0 for r in rows),
            'speedup_vs_first_leg': reference / step if reference else None, 'first_leg': legs[0],
            'micro_batch': int(micro_batch), 'dp': len(gpus)}), flush=True)
    return 0


def quantprobe(args):
    """Matched continuations from the BF16 1C checkpoint 13794 (scripts/cloud_quantization_probe.sh):
    quantprobe OPTIMIZER VARIANT... Every variant resumes the same weights and optimizer state, so
    losses compare across jobs. Step time and memory compare only inside one job; COAT_PROBE_ORDERS=fwd-rev
    runs the list forward and then reversed so drift does not pass for an effect."""
    optimizer, *variants = args
    # Every pod is "mpimaster-0", so two probes of one optimizer started in the same minute used to share this
    # directory (and the summary/manifest files, and the run dirs of equal variants); the random suffix keeps them apart.
    stamp = f'{datetime.datetime.now(datetime.timezone.utc):%d%H%M}{os.urandom(2).hex()}'
    job = (os.environ.get('MLSUB_JOB_NAME') or os.environ.get('HOSTNAME', 'job'))[-12:]
    output = campaign / 'quantprobe' / f'{optimizer}-{stamp}-{job}'
    gpus = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip().splitlines()
    print('COAT_PROBE_HOST ' + json.dumps({'gpus': gpus, 'output': str(output), 'cpu': cpu_snapshot()}), flush=True)
    for order in os.environ.get('COAT_PROBE_ORDERS', 'fwd').split('-'):
        sequence = variants if order == 'fwd' else list(reversed(variants))
        tag = f'qp{stamp}-{order}'
        env = dict(os.environ, STAGE3_MOE_LOG_ROOT=str(output), STAGE3_MOE_PROBE_TAG=tag, STAGE3_MOE_PROBE_GPUS=str(len(gpus)),
                   STAGE3_MOE_PROBE_KEEP_GOING='1')
        result = subprocess.run(['bash', str(root / 'scripts/cloud_quantization_probe.sh'), optimizer, *sequence],
                                cwd=root, env=env, capture_output=True, text=True)
        for line in (result.stdout + result.stderr).splitlines():
            if line.startswith(('PROBE_ABORT', 'VARIANT_EXIT', 'QUANTIZATION_PROBE_EXIT', 'CHECKPOINT_HF_REVISION', 'COAT_FETCH', 'COAT_ARCHIVE', 'ARCHIVE_EXIT', 'COAT_EVALLM', 'COAT_HESSIAN')) or 'stage3 cold switch' in line or 'stage3 lr scale' in line or 'stage3 muon variant' in line or line.startswith(('STAGE3_DEQUANT_ON_LOAD', 'LIONMUON', 'SOFTMUON', 'NORMUON', 'MUON2')) or 'Traceback' in line or 'Error:' in line:
                print('COAT_PROBE_LOG ' + (line if line.startswith('COAT_HESSIAN {') else line[:400]), flush=True)
        summary = output / f'summary-{optimizer}-{tag}.jsonl'
        for row in (json.loads(line) for line in summary.read_text().splitlines()) if summary.is_file() else ():
            print('COAT_PROBE ' + json.dumps(probe_record(optimizer, row, order)), flush=True)
        diagnostics = os.environ.get('STAGE3_MOE_DIAGNOSTIC_PATH')
        if diagnostics and Path(diagnostics).exists():
            print('COAT_STATEDIAG_LIVE ' + json.dumps({'variants': sequence, 'rows': summarize_diagnostics(diagnostics)}), flush=True)
            print('COAT_SNR ' + json.dumps({'variants': sequence, 'base': os.environ.get('COAT_FETCH_CKPT', 'hf-13794').split(':')[0],
                                            'layers': summarize_snr(diagnostics)}), flush=True)
        m2_trace = os.environ.get('STAGE3_MOE_M2_TRACE_PATH')
        if m2_trace and Path(m2_trace).is_dir():
            print('COAT_M2TRACE ' + json.dumps({'variants': sequence, 'traces': summarize_m2trace(m2_trace)}), flush=True)
        activation_stats = os.environ.get('STAGE3_MOE_ACTIVATION_STATS_PATH')
        if activation_stats and Path(activation_stats).exists():
            print('COAT_ACTSTATS ' + json.dumps({'variants': sequence, 'base': os.environ.get('COAT_FETCH_CKPT', 'hf-13794').split(':')[0],
                                                 'summary': summarize_activation_stats(activation_stats)}), flush=True)
    return 0



def evalfetch(specs):
    """Downstream scores of archived endpoints: evalfetch ARM_ID=NAME:SHA256 ... Each endpoint is fetched
    from the gateway (weights only) and scored by scripts/cloud_moe_eval.sh under the BF16 launcher arm of
    its optimizer, so every arm is scored with the same BF16 compute; the series arm id tags the run."""
    import re
    stamp = f'{datetime.datetime.now(datetime.timezone.utc):%d%H%M}'
    job = (os.environ.get('MLSUB_JOB_NAME') or os.environ.get('HOSTNAME', 'job'))[-12:]
    output = campaign / 'evalfetch' / f'{stamp}-{job}'
    stage = Path('/tmp/coat-eval')
    scored = {}
    for spec in specs:
        arm_id, rest = spec.split('=', 1)
        name, sha = rest.split(':')
        launcher_arm = f"{arm_id.split('-')[0]}_bf16_state_fp32"
        iteration = int(re.search(r'iter(\d{7})', name)[1])
        target = stage / arm_id / launcher_arm / f'iter_{iteration:07d}' / 'mp_rank_00' / 'model_optim_rng.pt'
        target.parent.mkdir(parents=True, exist_ok=True)
        fetched = subprocess.run([sys.executable, str(root / 'scripts/coat_archive.py'), 'fetch', name, sha, str(target)],
                                 capture_output=True, text=True)
        print(f'COAT_EVALFETCH_FETCH arm_id={arm_id} exit={fetched.returncode} ' + (fetched.stdout + fetched.stderr)[-400:].strip(), flush=True)
        if fetched.returncode:
            continue
        (target.parents[2] / 'latest_checkpointed_iteration.txt').write_text(f'{iteration}\n')
        env = dict(os.environ, STAGE3_MOE_EVAL_ROOTS=str(target.parents[3]),
                   STAGE3_MOE_EVAL_BUDGET='1c' if iteration == 17242 else f'iter{iteration}',
                   STAGE3_MOE_EVAL_STAGE=str(stage / f'staged-{arm_id}'), STAGE3_MOE_LOG_ROOT=str(output),
                   STAGE3_MOE_RUN_SUFFIX=arm_id, STAGE3_MOE_PROPAGATE_EXIT='1')
        result = subprocess.run(['bash', str(root / 'scripts/cloud_moe_eval.sh'), launcher_arm], cwd=root, env=env,
                                capture_output=True, text=True)
        lines = (result.stdout + result.stderr).splitlines()
        for index, line in enumerate(lines):
            if line.startswith(('=== ARM', 'ARM_EXIT', '--- DOWNSTREAM_JSON', '--- NO DOWNSTREAM', 'SKIP', 'INSTALL_EXIT')) or 'Traceback' in line or 'Error' in line:
                print(f'COAT_EVALFETCH arm_id={arm_id} ' + line[:600], flush=True)
        for path in output.glob(f'*{arm_id}*/downstream/downstream.json'):
            print(f'COAT_EVALFETCH_JSON arm_id={arm_id} ' + json.dumps(json.loads(path.read_text()))[:8000], flush=True)
            scored[arm_id] = path
        shutil.rmtree(stage / arm_id, ignore_errors=True)
    # Endpoints scored in one job on the same examples: the first spec is the baseline of every later one.
    first = specs[0].split('=', 1)[0]
    for arm_id, path in scored.items():
        if arm_id != first and first in scored:
            print(f'COAT_EVALPAIR baseline={first} treatment={arm_id} '
                  + json.dumps(paired_downstream(scored[first], path)), flush=True)
    return 0


def paired_downstream(baseline_json, treatment_json, iterations=2000, seed=1234):
    """Per (task, metric): treatment - baseline on the shared documents, with a two-sided 95 % paired bootstrap
    interval (documents resampled jointly, as in stage3_moe.paired_inference) and exact McNemar for 0/1 metrics.
    `pair_results` cannot pair these records: both sides run under the same launcher arm."""
    import numpy as np
    sys.path.insert(0, str(root))  # run as a script from scripts/: the repository root is not on the path
    from stage3_moe.paired_inference import _bootstrap_aggregates, _load_per_example, mcnemar
    baseline_items = {(i['task'], i['metric']): i for i in json.loads(Path(baseline_json).read_text())}
    out = []
    for item in json.loads(Path(treatment_json).read_text()):
        key = (item['task'], item['metric'])
        if key not in baseline_items:
            continue
        task, metric = key
        rows = [_load_per_example(Path(p).parent.parent / 'results.jsonl', task, i['per_example_artifact_sha256'])
                for p, i in ((baseline_json, baseline_items[key]), (treatment_json, item))]
        record = {'task': task, 'metric': metric, 'baseline': baseline_items[key]['value'], 'treatment': item['value'],
                  'gap': item['value'] - baseline_items[key]['value'], 'higher_is_better': item['higher_is_better']}
        if rows[0] and rows[1]:
            try:  # per-example formats come from lm-eval task code; one odd metric must not lose the whole line
                shared = sorted(d for d in set(rows[0]) & set(rows[1]) if metric in rows[0][d] and metric in rows[1][d])
                base = [rows[0][d][metric] for d in shared]
                treat = [rows[1][d][metric] for d in shared]
                index = np.random.default_rng(seed).integers(0, len(shared), size=(iterations, len(shared)))
                full = np.arange(len(shared))[None, :]
                base_full, treat_full = _bootstrap_aggregates(base, full, metric), _bootstrap_aggregates(treat, full, metric)
                base_boot, treat_boot = _bootstrap_aggregates(base, index, metric), _bootstrap_aggregates(treat, index, metric)
                if len(shared) > 1 and base_boot is not None and treat_boot is not None:
                    difference = treat_boot - base_boot
                    record.update(n=len(shared), gap_resampled=float(treat_full[0] - base_full[0]),
                                  ci95=[float(np.percentile(difference, 2.5)), float(np.percentile(difference, 97.5))])
                    if all(v in (0.0, 1.0) for v in base + treat):
                        p, only_base, only_treat = mcnemar(base, treat)
                        record.update(mcnemar_p=p, only_baseline=only_base, only_treatment=only_treat)
            except (TypeError, ValueError) as error:
                record['ci_error'] = str(error)[:200]
        out.append(record)
    return out

def paramdrift(specs):
    """Parameter drift between archived checkpoints at one iteration (SOW Stage 3 item 2):
    paramdrift REF_ID=NAME:SHA256 ARM_ID=NAME:SHA256 ... The first is the reference (the BF16 control);
    for every other one: relative L2 distance, cosine and max |delta| per parameter group and the ten
    tensors that moved most. CPU only; checkpoints are memory-mapped, optimizer states are not read."""
    import torch
    for extra in (root, root / 'third_party' / 'Megatron-LM'):  # the checkpoint pickles Megatron argument classes
        sys.path.insert(0, str(extra))
    stage = Path('/tmp/coat-drift')
    groups = (('embedding', 'embedding'), ('output_layer', 'output_layer'), ('router', 'router'),
              ('experts', 'experts'), ('attention', 'self_attention'), ('norm', 'norm'))

    def fetch(spec):
        arm_id, rest = spec.split('=', 1)
        name, sha = rest.split(':')
        target = stage / arm_id / name
        target.parent.mkdir(parents=True, exist_ok=True)
        fetched = subprocess.run([sys.executable, str(root / 'scripts/coat_archive.py'), 'fetch', name, sha, str(target)],
                                 capture_output=True, text=True)
        print(f'COAT_PARAMDRIFT_FETCH arm_id={arm_id} exit={fetched.returncode} ' + (fetched.stdout + fetched.stderr)[-300:].strip(), flush=True)
        if fetched.returncode:
            return arm_id, None
        model = torch.load(target, map_location='cpu', mmap=True, weights_only=False)['model']
        tensors = {k: v for k, v in model.items() if isinstance(v, torch.Tensor) and v.is_floating_point()}
        swiglu_alignment(arm_id, tensors)
        return arm_id, tensors

    def swiglu_alignment(arm_id, tensors):
        """Fishman et al. (ICLR 2025) precursor of FP8 divergence: SwiGLU's gate and up rows aligning,
        so that silu(gate x) * (up x) grows like x^2. Per layer, over experts and rows of linear_fc1
        (first half gate, second half up): mean, p99 and max of |cos(gate_i, up_i)|."""
        by_layer = {}
        for name, w in tensors.items():
            m = re.match(r'decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc1\.weight\d+$', name)
            if not m or w.dim() != 2:
                continue
            gate, up = w.float().chunk(2, 0)
            cos = (gate * up).sum(1) / (gate.norm(dim=1) * up.norm(dim=1)).clamp_min(1e-12)
            by_layer.setdefault(int(m[1]), []).append(cos.abs())
        summary = {}
        for layer, parts in sorted(by_layer.items()):
            cos = torch.cat(parts)
            summary[layer] = [round(cos.mean().item(), 5), round(cos.quantile(0.99).item(), 5), round(cos.max().item(), 5)]
        if summary:
            everything = torch.cat([torch.cat(p) for p in by_layer.values()])
            print(f'COAT_SWIGLU_ALIGN arm={arm_id} ' + json.dumps({'all_mean_p99_max': [round(everything.mean().item(), 5),
                  round(everything.quantile(0.99).item(), 5), round(everything.max().item(), 5)], 'by_layer': summary}), flush=True)

    stage.mkdir(parents=True, exist_ok=True)
    print('COAT_PARAMDRIFT_HOST ' + json.dumps({'df_tmp': shutil.disk_usage(stage)._asdict(),
          'meminfo': dict(line.split(':') for line in Path('/proc/meminfo').read_text().splitlines()[:3]),
          'cgroup_mem_max': Path('/sys/fs/cgroup/memory.max').read_text().strip() if Path('/sys/fs/cgroup/memory.max').exists() else None}), flush=True)
    try:
        ref_id, ref = fetch(specs[0])
    except Exception as error:  # the platform drops the log of a failed job, so report and exit 0
        print(f'COAT_PARAMDRIFT_ERROR spec={specs[0]} {type(error).__name__}: {str(error)[:300]}', flush=True)
        return 0
    if ref is None:
        return 0
    for spec in specs[1:]:
        try:
            arm_id, arm = fetch(spec)
        except Exception as error:
            print(f'COAT_PARAMDRIFT_ERROR spec={spec} {type(error).__name__}: {str(error)[:300]}', flush=True)
            continue
        if arm is None:
            continue
        acc = {g: [0.0, 0.0, 0.0, 0.0, 0.0] for g, _ in groups}
        acc['other'] = [0.0, 0.0, 0.0, 0.0, 0.0]
        tensors = []
        for name, a in ref.items():
            if name not in arm or arm[name].shape != a.shape:
                continue
            x, y = a.float(), arm[name].float()
            d = y - x
            dd, xx, xy, yy, dmax = (d * d).sum().item(), (x * x).sum().item(), (x * y).sum().item(), (y * y).sum().item(), d.abs().max().item()
            group = next((g for g, pat in groups if pat in name), 'other')
            s = acc[group]
            s[0] += dd; s[1] += xx; s[2] += xy; s[3] += yy; s[4] = max(s[4], dmax)
            tensors.append((name, (dd / xx) ** 0.5 if xx else 0.0, dmax, x.numel()))
        summary = {g: {'rel_l2': round((s[0] / s[1]) ** 0.5, 6), 'cosine': round(s[2] / (s[1] * s[3]) ** 0.5, 8),
                       'max_abs_delta': round(s[4], 6)} for g, s in acc.items() if s[1]}
        total = [sum(s[i] for s in acc.values()) for i in range(4)]
        summary['all'] = {'rel_l2': round((total[0] / total[1]) ** 0.5, 6), 'cosine': round(total[2] / (total[1] * total[3]) ** 0.5, 8),
                          'tensors': len(tensors)}
        top = sorted(tensors, key=lambda t: -t[1])[:10]
        print(f'COAT_PARAMDRIFT ref={ref_id} arm={arm_id} ' + json.dumps({'groups': summary,
              'top_rel_l2': [[n, round(r, 6), round(m, 6), k] for n, r, m, k in top]}), flush=True)
        shutil.rmtree(stage / arm_id, ignore_errors=True)
    return 0


def wsmmerge(args):
    """WSM (Warmup-Stable-Merge, arXiv 2507.17634 §3): merge the constant-LR checkpoints of one base and score
    the merges. wsmmerge BASE VARIANT... fetches the weights-only saves of the constant-LR run from the gateway
    (WSM_PREFIX = archive name before -iterNNNNNNN, WSM_SHA_<iteration>=SHA256 for each save, every 500
    iterations after 13,794 plus 17,242); theta_0 is the base checkpoint 13,794 (WSM_BASE_CKPT=NAME:SHA) and
    WSM_DECAY_CKPT=NAME:SHA the same arm's decay endpoint, scored alongside as the WSD reference. Merge weights
    follow Theorem 3.1: c_0 = 1 - w_1, c_j = w_j - w_(j+1), c_k = w_k, where w_i is the emulated LR (relative to
    the peak) of the updates between checkpoints i-1 and i. Only floating-point model tensors are averaged (the
    router expert bias included); TE extra state and token counters come from 17,242. Every model is scored by
    eval-lm-fixed under each VARIANT (launcher arms of WSM_OPTIMIZER, default adamw), then the merges go to the
    gateway and the routing audit runs under the BF16 forward."""
    import torch
    sys.path.insert(0, str(root / 'scripts'))
    import coat_archive
    for extra in (root, root / 'third_party' / 'Megatron-LM'):  # the checkpoint pickles Megatron argument classes
        sys.path.insert(0, str(extra))
    base, *variants = args
    start, end, decay = 13794, int(os.environ.get('WSM_END', '17242')), 3448  # WSM_END: a smoke test's last save
    stage = Path('/tmp/coat-wsm')
    output = stage / 'eval'
    output.mkdir(parents=True, exist_ok=True)

    def checkpoint(directory, iteration):
        return directory / f'iter_{iteration:07d}' / 'mp_rank_00' / 'model_optim_rng.pt'

    def fetch(name, sha, label, iteration):
        target = checkpoint(stage / label, iteration)
        target.parent.mkdir(parents=True, exist_ok=True)
        coat_archive.download(name, sha, target)
        (target.parents[2] / 'latest_checkpointed_iteration.txt').write_text(f'{iteration}\n')
        return target

    saved = sorted(int(key[8:]) for key in os.environ if key.startswith('WSM_SHA_'))
    paths = [fetch(*os.environ['WSM_BASE_CKPT'].split(':'), 'base', start)]
    paths += [fetch(f"{os.environ['WSM_PREFIX']}-iter{i:07d}-model.pt", os.environ[f'WSM_SHA_{i}'], f'ckpt-{i}', i) for i in saved]
    iters = [start] + saved
    print('WSM_INPUT ' + json.dumps({'base': base, 'iterations': iters, 'df_tmp': shutil.disk_usage(stage)._asdict()}), flush=True)
    if saved[-1] != end:
        raise RuntimeError(f'the constant-LR run did not reach {end}: {saved}')

    def theorem(w):
        return [1 - w[0]] + [w[i] - w[i + 1] for i in range(len(w) - 1)] + [w[-1]]

    # lr(iteration) / peak of the update that produces iteration t is f((t - 1 - start) / decay).
    schedules = {'sqrt': lambda r: 1 - r ** 0.5,  # 1-sqrt to zero, the paper's decay (App. C)
                 'exp': lambda r: 0.1 + 0.9 * (2 * 0.5 ** r - 1)}  # our WSD: MCore exponential, min_lr = peak/10
    block = lambda f, a, b: sum(f((t - 1 - start) / decay) for t in range(a + 1, b + 1)) / (b - a)
    k = len(iters) - 1
    recipes = {'mean8': [0.0] + [1 / k] * k, 'mean4': [0.0] * (k + 1 - min(4, k)) + [1 / min(4, k)] * min(4, k),
               'sqrt': theorem([block(schedules['sqrt'], iters[i - 1], iters[i]) for i in range(1, k + 1)]),
               'exp': theorem([block(schedules['exp'], iters[i - 1], iters[i]) for i in range(1, k + 1)]),
               'expend': theorem([schedules['exp']((iters[i] - start) / decay) for i in range(1, k + 1)])}
    template = torch.load(paths[-1], map_location='cpu', weights_only=False)
    keys = [name for name, value in template['model'].items() if torch.is_tensor(value) and value.is_floating_point()
            and not name.endswith('_extra_state') and 'tokens_per_expert' not in name]
    models = {'decay': (fetch(*os.environ['WSM_DECAY_CKPT'].split(':'), 'decay', 17242).parents[2], 17242), 'last': (paths[-1].parents[2], end)}
    for recipe, weights in recipes.items():
        assert min(weights) >= -1e-12 and abs(sum(weights) - 1) < 1e-9, (recipe, weights)
        merged = {name: torch.zeros(template['model'][name].shape, dtype=torch.float32) for name in keys}
        for path, c in zip(paths, weights):
            if c <= 0:
                continue
            model = torch.load(path, map_location='cpu', mmap=True, weights_only=False)['model']
            for name in keys:
                merged[name].add_(model[name].float(), alpha=c)
            del model
        moved = sum((merged[name] - template['model'][name].float()).square().sum().item() for name in keys)
        norm = sum(template['model'][name].float().square().sum().item() for name in keys)
        state = dict(template, model=dict(template['model'], **{name: merged[name].to(template['model'][name].dtype) for name in keys}))
        target = checkpoint(stage / 'merged' / recipe, end)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, target)
        (target.parents[2] / 'latest_checkpointed_iteration.txt').write_text(f'{end}\n')
        models[recipe] = (target.parents[2], end)
        del merged, state
        print('WSM_MERGE ' + json.dumps({'base': base, 'recipe': recipe, 'weights': dict(zip(iters, [round(c, 6) for c in weights])),
                                         'tensors': len(keys), 'rel_l2_to_17242': round((moved / norm) ** 0.5, 6)}), flush=True)
    del template

    gpus = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], capture_output=True, text=True).stdout.split()
    results = {}

    def evaluate(mode, label, model, variant):
        directory, iteration = model
        env = dict(os.environ, STAGE3_MOE_PROBE_MODE=mode, STAGE3_MOE_PROBE_LOAD=str(directory), STAGE3_MOE_PROBE_BASE_ITER=str(iteration),
                   STAGE3_MOE_PROBE_ITERS='0', STAGE3_MOE_MICRO_BATCH='16', STAGE3_MOE_PROBE_GPUS=str(len(gpus)),
                   STAGE3_MOE_LOG_ROOT=str(output), STAGE3_MOE_PROBE_TAG=f'wsm-{base}-{label}', STAGE3_MOE_PROBE_KEEP_GOING='1',
                   COAT_FETCH_CKPT=f'wsm-{base}-{label}:local', STAGE3_ROUTING_CHECKPOINT_LABEL=f'wsm-{base}-{label}',
                   STAGE3_MOE_ROUTING_AUDIT_PATH=str(output / f'routing-{label}.json'))
        result = subprocess.run(['bash', str(root / 'scripts/cloud_quantization_probe.sh'), os.environ.get('WSM_OPTIMIZER', 'adamw'), variant], cwd=root, env=env,
                                capture_output=True, text=True)
        for line in (result.stdout + result.stderr).splitlines():
            m = re.search(r'loss at iteration\s+\d+ on (validation|test) set\s*\|\s*lm loss value:\s*([0-9.eE+-]+)', line)
            if line.startswith('COAT_EVALLM') and m:
                results.setdefault(f'{label}/{variant}/{mode}', {})[m[1]] = float(m[2])
            elif line.startswith(('PROBE_ABORT', 'VARIANT_EXIT')) or 'Traceback' in line or 'Error:' in line:
                print(f'WSM_LOG {label} {variant} {mode} ' + line[:400], flush=True)
        print('WSM_EVAL ' + json.dumps({'base': base, 'model': label, 'variant': variant, 'mode': mode,
                                        **results.get(f'{label}/{variant}/{mode}', {})}), flush=True)

    for variant in variants:
        for label, model in models.items():
            evaluate('eval-lm-fixed', label, model, variant)
    print('WSM_RESULT ' + json.dumps({'base': base, 'results': results}), flush=True)
    for recipe in recipes if os.environ.get('WSM_ARCHIVE', '1') == '1' else ():
        name = f'wsm-{base}-{recipe}-iter{end:07d}-model.pt'
        try:
            receipt = coat_archive.upload(checkpoint(models[recipe][0], end), name)
            print('COAT_ARCHIVE ' + json.dumps({'arm_id': f'wsm-{base}-{recipe}', 'iteration': end, 'kind': 'model', **receipt}), flush=True)
        except Exception as error:  # the merge stays on nfs3
            print('COAT_ARCHIVE_FAILED ' + json.dumps({'name': name, 'error': repr(error)[:400]}), flush=True)
    # Paper Table 5: per layer max_violation = (max(L) - mean) / mean, min_violation = (mean - min(L)) / mean, averaged over layers.
    for label, model in models.items():
        evaluate('eval-routing-fixed', label, model, 'bf16')
        audit = output / f'routing-{label}.json'
        if not audit.is_file():
            continue
        for evaluation in json.loads(audit.read_text())['evaluations']:
            rows = [[c / (sum(row['frozen_counts']) / len(row['frozen_counts'])) for c in row['frozen_counts']] for row in evaluation['layers']]
            print('WSM_ROUTING ' + json.dumps({'base': base, 'model': label, 'split': evaluation['split'], 'layers': len(rows),
                  'mean_max_violation': round(sum(max(r) - 1 for r in rows) / len(rows), 4),
                  'mean_min_violation': round(sum(1 - min(r) for r in rows) / len(rows), 4),
                  'mean_cv_frozen': round(sum(row['frozen_balance']['coefficient_of_variation'] for row in evaluation['layers']) / len(rows), 4)}), flush=True)
    shutil.rmtree(stage, ignore_errors=True)
    return 0


def summarize_diagnostics(path):
    """Median per (kind, state, codec, format) of the diagnostics tool's JSONL rows: relative Frobenius
    distortion, element relative-error p99, and the lost-change fraction of the one-step EMA update."""
    import statistics
    groups = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if 'codec' not in row:
            continue
        key = (row['kind'], row.get('state', '-'), row['codec'], row.get('format') or row.get('second_format') or '-')
        g = groups.setdefault(key, {'rel_frob': [], 'p99': [], 'lost': [], 'n': 0})
        g['n'] += 1
        if row.get('relative_frobenius') is not None:
            g['rel_frob'].append(row['relative_frobenius'])
        q = row.get('element_relative_error_quantiles')
        if q and q.get('0.99') is not None:
            g['p99'].append(q['0.99'])
        if row.get('lost_change_fraction') is not None:
            g['lost'].append(row['lost_change_fraction'])
    out = []
    for key, g in sorted(groups.items()):
        out.append({'kind': key[0], 'state': key[1], 'codec': key[2], 'format': key[3], 'n': g['n'],
                    'rel_frob_median': round(statistics.median(g['rel_frob']), 6) if g['rel_frob'] else None,
                    'rel_err_p99_median': round(statistics.median(g['p99']), 5) if g['p99'] else None,
                    'lost_change_median': round(statistics.median(g['lost']), 5) if g['lost'] else None})
    return out


def summarize_m2trace(directory):
    """AdEMAMix m2 traces (tools/quantization_diagnostics.install_m2_trace), medians over the traced parameters:
    per step, the fraction of sampled elements whose decoded m2 (and m1) changed; per snapshot step, the cosine
    of m2 to the first snapshot, the changed fraction since the previous snapshot and, against an FP32-state
    trace of the same probe in the same directory, the norm ratio ||m2|| / ||m2_fp32|| and cos(m2, m2_fp32)."""
    import statistics
    import torch
    traces = [torch.load(p, weights_only=False) for p in sorted(Path(directory).glob('m2trace-*.pt'))]
    reference = next((t for t in traces if t['mode'] == 'fp32'), None)
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=0))
    median = lambda values: float(f'{statistics.median(values):.6g}') if values else None
    out = []
    for trace in traces:
        params = trace['params']
        row = {'label': trace['label'], 'mode': trace['mode'], 'beta3': trace['beta3'], 'params': len(params), 'changed_per_step': {}, 'snapshots': {}}
        for step in (1, 2, 3, 5, 10, 20, 30, 50, 100, 150, 200):
            m2 = [p['changed_m2'][step - 1] for p in params.values() if len(p['changed_m2']) >= step]
            m1 = [p['changed_m1'][step - 1] for p in params.values() if len(p['changed_m1']) >= step]
            if m2:
                row['changed_per_step'][step] = {'m2': median(m2), 'm1': median(m1)}
        steps = sorted({s for p in params.values() for s in p['snapshots']})
        for previous, step in zip([None] + steps[:-1], steps):
            snap = {}
            first = [cos(p['snapshots'][step]['m2'], p['snapshots'][steps[0]]['m2']) for p in params.values() if step in p['snapshots']]
            snap['cos_m2_to_first'] = median(first)
            if previous is not None:
                snap['changed_m2_since_previous'] = median([float((p['snapshots'][step]['m2'] != p['snapshots'][previous]['m2']).float().mean())
                                                            for p in params.values() if step in p['snapshots'] and previous in p['snapshots']])
            snap['norm_m2_median'] = median([p['snapshots'][step]['norm_m2'] for p in params.values() if step in p['snapshots']])
            if reference is not None and trace is not reference:
                pairs = [(p, reference['params'][name]) for name, p in params.items()
                         if name in reference['params'] and step in p['snapshots'] and step in reference['params'][name]['snapshots']]
                snap['norm_ratio_to_fp32'] = median([p['snapshots'][step]['norm_m2'] / r['snapshots'][step]['norm_m2']
                                                     for p, r in pairs if r['snapshots'][step]['norm_m2'] > 0])
                snap['cos_to_fp32'] = median([cos(p['snapshots'][step]['m2'], r['snapshots'][step]['m2']) for p, r in pairs])
                snap['norm_ratio_m1_to_fp32'] = median([p['snapshots'][step]['norm_m1'] / r['snapshots'][step]['norm_m1']
                                                        for p, r in pairs if r['snapshots'][step]['norm_m1'] > 0])
                # m1 forgets in ~10 steps: its cosine is the two runs' gradient divergence plus m1's own codec error.
                snap['cos_m1_to_fp32'] = median([cos(p['snapshots'][step]['m1'], r['snapshots'][step]['m1']) for p, r in pairs])
            row['snapshots'][step] = snap
        out.append(row)
    return out


def summarize_snr(path):
    """Second-moment SNR per layer type (SlimAdam, eq. 3): median and min over the matrices of each type along
    fan_in / fan_out, the fraction whose best dimension clears the paper's cutoff of 1, and the dimension chosen
    by the majority -- the rule SlimAdam would derive for this model at this checkpoint."""
    import statistics
    kinds = (('word_embeddings', 'embedding'), ('output_layer', 'lm_head'), ('linear_qkv', 'attn_qkv'), ('linear_proj', 'attn_proj'),
             ('router', 'router'), ('shared_experts.linear_fc1', 'shared_fc1'), ('shared_experts.linear_fc2', 'shared_fc2'),
             ('linear_fc1', 'expert_fc1'), ('linear_fc2', 'expert_fc2'))
    groups = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get('kind') != 'second_moment_snr':
            continue
        name = row.get('name') or ''
        kind = next((label for needle, label in kinds if needle in name), 'other')
        groups.setdefault(kind, []).append(row)
    out = []
    for kind, rows in sorted(groups.items()):
        fan_in = [r['snr_fan_in'] for r in rows]
        fan_out = [r['snr_fan_out'] for r in rows]
        best = ['fan_in' if a >= b else 'fan_out' for a, b in zip(fan_in, fan_out)]
        out.append({'layer': kind, 'n': len(rows), 'shape': rows[0]['shape'],
                    'fan_in_median': round(statistics.median(fan_in), 3), 'fan_in_min': round(min(fan_in), 3),
                    'fan_out_median': round(statistics.median(fan_out), 3), 'fan_out_min': round(min(fan_out), 3),
                    'all_median': round(statistics.median(r['snr_all'] for r in rows), 3),
                    'best_dim': max(set(best), key=best.count), 'best_dim_share': round(best.count(max(set(best), key=best.count)) / len(best), 2),
                    'frac_best_gt1': round(sum(max(a, b) > 1 for a, b in zip(fan_in, fan_out)) / len(rows), 2)})
    return out


def summarize_activation_stats(path):
    """Per hook kind: median over layers and passes of each metric, and the layer with the largest outlier ratio."""
    import statistics
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    out = {}
    for kind in ('fc1_input', 'fc2_input', 'proj_input', 'fc1_output'):
        group = [r for r in rows if r.get('kind') == kind]
        if not group:
            continue
        metrics = [k for k in group[0] if isinstance(group[0][k], (int, float)) and k != 'rows']
        summary = {m: round(statistics.median(r[m] for r in group), 6) for m in metrics}
        key = 'product_outlier_ratio' if kind == 'fc1_output' else 'outlier_ratio'
        worst = max(group, key=lambda r: r[key])
        summary['worst'] = {'module': worst['module'], key: round(worst[key], 3)}
        summary['n'] = len(group)
        out[kind] = summary
    return out


def fetch_archived(spec, stage):
    """ARM_ID=NAME:SHA256 -> (arm_id, local path); the gateway copy or, since 2026-10-06, HF (coat_archive.py)."""
    arm_id, rest = spec.split('=', 1)
    name, sha = rest.split(':')
    target = stage / arm_id / name
    target.parent.mkdir(parents=True, exist_ok=True)
    fetched = subprocess.run([sys.executable, str(root / 'scripts/coat_archive.py'), 'fetch', name, sha, str(target)],
                             capture_output=True, text=True)
    print(f'COAT_FETCH_RESULT arm_id={arm_id} exit={fetched.returncode} ' + (fetched.stdout + fetched.stderr)[-400:].strip().replace(chr(10), ' | '), flush=True)
    return arm_id, (target if fetched.returncode == 0 else None)


def stateoutliers(specs):
    """COAT Fig. 2-style state statistics of archived full checkpoints: stateoutliers ARM_ID=NAME:SHA256 ...
    Output JSON under <campaign>/stateoutliers/<arm_id>.json and COAT_OUTLIERS lines (tools/state_outliers.py)."""
    stage = Path('/tmp/coat-stateoutliers')
    for spec in specs:
        arm_id, path = fetch_archived(spec, stage)
        if path is None:
            continue
        output = campaign / 'stateoutliers' / f'{arm_id}.json'
        result = subprocess.run([sys.executable, str(root / 'tools/state_outliers.py'), str(path), '--output', str(output), '--tag', arm_id,
                                 '--stored-codec', os.environ.get('COAT_STORED_CODEC', 'coat')], cwd=root, capture_output=True, text=True)
        for line in (result.stdout + result.stderr).splitlines():
            if line.startswith(('COAT_OUTLIERS', 'STATE_OUTLIERS')) or 'Traceback' in line or 'Error' in line:
                print(line[:20000], flush=True)
        print(f'STATEOUTLIERS_EXIT arm_id={arm_id} code={result.returncode}', flush=True)
        shutil.rmtree(stage / arm_id, ignore_errors=True)
    return 0


def drebench(specs):
    """Cost of dynamic range expansion on real moments: drebench ADAM_ARM=NAME:SHA256 [MUON_ARM=NAME:SHA256]
    (tools/dre_cost_bench.py, one GPU)."""
    stage = Path('/tmp/coat-drebench')
    adam_id, adam_path = fetch_archived(specs[0], stage)
    if adam_path is None:
        return 0
    command = [sys.executable, str(root / 'tools/dre_cost_bench.py'), str(adam_path), '--repeats', os.environ.get('COAT_BENCH_REPEATS', '20')]
    if len(specs) > 1:
        muon_id, muon_path = fetch_archived(specs[1], stage)
        if muon_path is not None:
            command += ['--muon', str(muon_path)]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    for line in (result.stdout + result.stderr).splitlines():
        if line.startswith(('COAT_DREBENCH', 'DRE_COST_BENCH')) or 'Traceback' in line or 'Error' in line:
            print(line[:4000], flush=True)
    print(f'DREBENCH_EXIT code={result.returncode}', flush=True)
    shutil.rmtree(stage, ignore_errors=True)
    return 0


def statediag(specs):
    """Snapshot codec metrics on archived checkpoints (SOW Stage 3 item 2): statediag ARM_ID=NAME:SHA256 ...
    Runs tools/quantization_diagnostics.py on each (GPU); FP8-state checkpoints (arm id containing 'coat')
    are decoded with the COAT codec first. Prints COAT_STATEDIAG arm=... with the medians per codec."""
    stage = Path('/tmp/coat-statediag')
    for spec in specs:
        arm_id, rest = spec.split('=', 1)
        name, sha = rest.split(':')
        target = stage / arm_id / name
        target.parent.mkdir(parents=True, exist_ok=True)
        fetched = subprocess.run([sys.executable, str(root / 'scripts/coat_archive.py'), 'fetch', name, sha, str(target)],
                                 capture_output=True, text=True)
        print(f'COAT_STATEDIAG_FETCH arm_id={arm_id} exit={fetched.returncode} ' + (fetched.stdout + fetched.stderr)[-300:].strip(), flush=True)
        if fetched.returncode:
            continue
        output = campaign / 'statediag' / f'{arm_id}.jsonl'
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        stored = 'coat' if 'coat' in arm_id else 'auto'
        result = subprocess.run([sys.executable, str(root / 'tools/quantization_diagnostics.py'), str(target), '--output', str(output),
                                 '--stored-codec', stored], cwd=root, capture_output=True, text=True)
        tail = (result.stdout + result.stderr).strip().splitlines()[-3:]
        print(f'COAT_STATEDIAG_RUN arm_id={arm_id} exit={result.returncode} ' + ' | '.join(line[:200] for line in tail), flush=True)
        if output.exists():
            print(f'COAT_STATEDIAG arm={arm_id} ' + json.dumps(summarize_diagnostics(output)), flush=True)
        shutil.rmtree(stage / arm_id, ignore_errors=True)
    return 0


def loggrep(patterns):
    """loggrep PATTERN... -- lines matching any pattern (regex, '.' for a space) in the live train logs of the
    campaign: first 5 matches per log, with the log's last iteration line. Reads the NFS logs the platform API
    cannot show for a running job."""
    regex = re.compile('|'.join(patterns))
    # COAT_LOGGREP_ROUND narrows the search to runs-<round> (mlsub keeps only the last ~200 lines of a job's output,
    # and --env values may not contain '*').
    round_name = os.environ.get('COAT_LOGGREP_ROUND')
    globs = (f'runs-{round_name}/*/*/train-*.log',) if round_name else ('quantprobe/*/*/train-*.log', 'runs-*/*/*/train-*.log')
    for path in sorted(p for pattern in globs for p in campaign.glob(pattern)):
        lines = path.read_text(errors='replace').splitlines()
        hits = [line[:300] for line in lines if regex.search(line)][:5]
        last = next((line[:300] for line in reversed(lines) if re.search(r'iteration\s+\d+/', line)), None)
        if hits:
            print('COAT_LOGGREP ' + json.dumps({'log': str(path.relative_to(campaign)), 'hits': hits, 'last': last}), flush=True)
    return 0


def main():
    mode, *args = sys.argv[1:]
    prepare_source()
    for name in ('COAT_ARCHIVE_KEY_B64', 'COAT_ARCHIVE_HOSTS_B64'):  # mlsub caps the length of one value
        parts = []
        while f'{name}_{len(parts):02d}' in os.environ:
            parts.append(os.environ.pop(f'{name}_{len(parts):02d}'))
        if parts:
            os.environ[name] = ''.join(parts)
    if mode == 'speeddiag':
        return speeddiag(args)
    if mode == 'quantprobe':
        return quantprobe(args)
    if mode == 'abba':
        return abba(args)
    if mode == 'evalfetch':
        return evalfetch(args)
    if mode == 'paramdrift':
        return paramdrift(args)
    if mode == 'wsmmerge':
        return wsmmerge(args)
    if mode == 'loggrep':
        return loggrep(args)
    if mode == 'statediag':
        return statediag(args)
    if mode == 'stateoutliers':
        return stateoutliers(args)
    if mode == 'drebench':
        return drebench(args)
    if mode == 'moebench':
        result = subprocess.run([sys.executable, str(root / 'tools/moe_layer_bench.py'), *args], cwd=root, capture_output=True, text=True)
        for line in (result.stdout + result.stderr).splitlines():
            if 'MOE_LAYER' in line or 'Error' in line:
                print(line[line.find('MOE_LAYER') if 'MOE_LAYER' in line else 0:][:6000], flush=True)
        print('CAMPAIGN_EXIT=' + str(result.returncode), flush=True)
        return 0
    if mode == 'scalebench':
        # The wrapper reproduces the launcher's library environment: on torch28 TE cannot find cudart without it.
        result = subprocess.run(['bash', str(root / 'scripts/run_scale_bench.sh'), *args], cwd=root, capture_output=True, text=True)
        for line in (result.stdout + result.stderr).splitlines():
            if 'MOE_SCALE' in line or 'SCALEBENCH_ENV' in line or 'Error' in line:
                print(line[line.find('MOE_SCALE') if 'MOE_SCALE' in line else 0:][:3000], flush=True)
        print((result.stdout + result.stderr)[-3000:] if result.returncode else '', flush=True)
        print('CAMPAIGN_EXIT=' + str(result.returncode), flush=True)
        return 0
    if mode == 'archive-probe':
        result = subprocess.run([sys.executable, str(root / 'scripts/coat_archive.py'), 'probe'], capture_output=True, text=True)
        print((result.stdout + result.stderr)[-1500:], flush=True)
        return 0
    if mode == 'preflight':
        return preflight()
    if mode == 'contract003':
        return contract003()
    if mode == 'contract004':
        return contract004()
    if mode == 'contract005':
        return contract005()
    if mode == 'contract006':
        return contract006()
    if mode == 'contract007':
        return contract007()
    if mode == 'measurement008':
        return measurement008()
    if mode == 'measurement008fix':
        return measurement008fix()
    if mode == 'analysis009':
        return analysis009()
    if mode == 'analysis010':
        return analysis010()
    if mode == 'analysis011':
        return analysis011()
    if mode == 'postwarmup012':
        return postwarmup012()
    if mode == 'overnight013':
        return overnight013()
    if mode == 'fullrun014':
        return fullrun014()
    if mode == 'screenb015':
        return screenb015()
    if mode == 'baselineeval030':
        return baselineeval030()
    if mode == 'disk':
        # Free space on every persistent volume and what the series keeps there.
        subprocess.run(['df', '-h', '/home/jovyan', '/workspace-SR006.nfs2', '/workspace-SR006.nfs3', '/tmp'], check=False)
        subprocess.run('du -sh /workspace-SR006.nfs3/hmoe-cloud/* /workspace-SR006.nfs2/hmoe-cloud/* /home/jovyan/hmoe-cloud/* 2>/dev/null | sort -h | tail -25',
                       shell=True, check=False)
        # Top level of every volume (shares/ re-mounts the same volumes, so -x stops there), then inside our own hmoe-* trees.
        subprocess.run('for v in /workspace-SR006.nfs3 /workspace-SR006.nfs2 /home/jovyan; do du -xhd1 $v 2>/dev/null | sort -h | tail -15 | sed "s|^|TOP |"; done; '
                       'du -xhd2 /workspace-SR006.nfs3/hmoe-* /workspace-SR006.nfs2/hmoe-* 2>/dev/null | sort -h | tail -40 | sed "s|^|OURS |"',
                       shell=True, check=False)
        if args:
            # disk SINCE (e.g. 2026-10-04T08:00): bytes of files modified after SINCE on nfs3, by top-level and second-level directory.
            subprocess.run(f"find /workspace-SR006.nfs3 -xdev -type f -newermt '{args[0]}' -printf '%s %h\\n' 2>/dev/null | "
                           "awk '{split($2,p,\"/\"); a[p[3]]+=$1; b[p[3]\"/\"p[4]]+=$1} END {for (k in a) printf \"GROWTH1 %.2f GB %s\\n\", a[k]/1e9, k; "
                           "for (k in b) if (b[k]>5e8) printf \"GROWTH2 %.2f GB %s\\n\", b[k]/1e9, k}' | sort -k2 -n -r | head -40",
                           shell=True, check=False)
        token = Path('/home/jovyan/.cache/huggingface/token')
        print('HF_TOKEN_FILE=' + str(token.is_file()), flush=True)
        return 0
    if mode == 'archivefailed':
        # archivefailed [upload]: endpoints the trainer could not send to the gateway (coat_archive.py keeps them under
        # archive-failed/ on nfs3): name, bytes, sha256; with "upload", send each to the gateway and delete it once received.
        sys.path.insert(0, str(root / 'scripts'))
        import coat_archive
        subprocess.run(['df', '-h', str(campaign)], check=False)
        for path in sorted((campaign / 'archive-failed').glob('*.pt')):
            print('COAT_ARCHIVE_FAILED_COPY ' + json.dumps({'name': path.name, 'bytes': path.stat().st_size, 'sha256': coat_archive.file_sha256(path)}), flush=True)
            if args[:1] == ['hfupload']:
                # HF_TOKEN comes only through the job environment; the file is deleted once HF reports the same sha256.
                from huggingface_hub import HfApi
                api = HfApi(token=os.environ['HF_TOKEN'])
                remote = f'{coat_archive.HF_PREFIX}/{path.name}'
                try:
                    api.upload_file(path_or_fileobj=str(path), path_in_repo=remote, repo_id=coat_archive.HF_REPO, commit_message=f'coat-20261002: {path.name}')
                    info = api.get_paths_info(coat_archive.HF_REPO, [remote], expand=True)[0]
                    remote_sha = info.lfs.sha256 if info.lfs else None
                    print('COAT_HF_UPLOAD ' + json.dumps({'name': path.name, 'sha256': remote_sha, 'bytes': info.size}), flush=True)
                    if remote_sha == coat_archive.file_sha256(path):
                        path.unlink()
                        print('COAT_HF_UPLOAD_VERIFIED ' + path.name, flush=True)
                except Exception as error:
                    print('COAT_HF_UPLOAD_FAILED ' + json.dumps({'name': path.name, 'error': repr(error)[:300]}), flush=True)
            if args[:1] == ['upload']:
                try:
                    receipt = coat_archive.upload(path, path.name)
                    print('COAT_ARCHIVE ' + json.dumps({'name': path.name, **receipt}), flush=True)
                    path.unlink()
                except Exception as error:
                    print('COAT_ARCHIVE_RETRY_FAILED ' + json.dumps({'name': path.name, 'error': repr(error)[:300]}), flush=True)
        return 0
    if mode == 'tesource':
        # Source of the installed TransformerEngine modules behind Megatron's FP8 GEMMs (what they save for
        # backward), as gzip+base64 COLLECT_CHUNK lines; found on sys.path without importing (CPU pods).
        import base64
        import gzip
        import hashlib
        import importlib.metadata
        files = {}
        for name in ('transformer_engine', 'transformer-engine', 'transformer_engine_torch'):
            try:
                files['VERSION ' + name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        for base in map(Path, sys.path):
            module = base / 'transformer_engine' / 'pytorch' / 'module'
            if module.is_dir():
                for rel in ('linear.py', 'grouped_linear.py', 'layernorm_linear.py', 'base.py'):
                    if (module / rel).is_file():
                        files[str(module / rel)] = (module / rel).read_text(errors='replace')
                break
        payload = gzip.compress(json.dumps(files).encode())
        encoded = base64.b64encode(payload).decode()
        chunks = [encoded[i:i + 12000] for i in range(0, len(encoded), 12000)]
        for index, chunk in enumerate(chunks):
            print(f'COLLECT_CHUNK {index} {chunk}', flush=True)
        print('COLLECT_RECEIPT ' + json.dumps({'runs': len(files), 'chunks': len(chunks), 'bytes': len(payload),
                                               'sha256': hashlib.sha256(payload).hexdigest()}), flush=True)
        return 0
    if mode == 'reclaim':
        # User decision 2026-10-05 07:40: free nfs3 by deleting exactly these two series directories, nothing else.
        targets = [campaign / 'm2trace', campaign / 'quantprobe']
        subprocess.run(['du', '-sh', *map(str, targets)], check=False)
        for target in targets:
            shutil.rmtree(target, ignore_errors=True)
            print(f'RECLAIM {target} exists_after={target.exists()}', flush=True)
        subprocess.run(['df', '-h', str(campaign)], check=False)
        return 0
    if mode == 'stageprofile':
        result = subprocess.run([sys.executable, str(root / 'tools/coat_stage_profile.py'), *args], cwd=root, capture_output=True, text=True)
        for line in (result.stdout + result.stderr).splitlines():
            if 'COAT_STAGE' in line or 'Error' in line:
                print(line[line.find('COAT_STAGE') if 'COAT_STAGE' in line else 0:][:6000], flush=True)
        print('CAMPAIGN_EXIT=' + str(result.returncode), flush=True)
        return 0
    if mode == 'worker':
        return worker(COAT_PRIORITY if os.environ.get('BUNDLE_ROLE') == 'coat' else args)
    if mode == 'inspect':
        return inspect()
    if mode == 'collect':
        return collect(args)
    if mode == 'timers':
        # MCore's per-iteration timers (--timing-log-level 1): where each arm's step time goes.
        for path in sorted(campaign.glob(f'runs-{ROUND}/*/*/train-*.log')):
            lines = path.read_text(errors='replace').splitlines()
            timer_lines = [i for i, line in enumerate(lines) if '(min, max) time across ranks' in line]
            if timer_lines:
                start = timer_lines[-1]
                block = [line.strip() for line in lines[start:start + 40] if '....' in line or 'time across' in line]
                print('TIMERS ' + json.dumps({'run': path.parent.name, 'block': block[:30]}), flush=True)
        return 0
    if mode == 'arm':
        # One from-scratch arm on every visible GPU of this job (1 GPU -> DP1 mb 16, 4 GPUs -> DP4 mb 13; gb 208 either way),
        # without the worker's claim bookkeeping: the seed replicates of 2026-10-08 on single-GPU pods.
        visible = len(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, check=True).stdout.strip().splitlines())
        for arm_id in args:
            env = dict(os.environ, STAGE3_MOE_NPROC=str(visible), STAGE3_MOE_MICRO_BATCH=os.environ.get('STAGE3_MOE_MICRO_BATCH', '16' if visible == 1 else '13'),
                       COAT_CAMPAIGN_ROOT=str(campaign), COAT_ROUND=ROUND)
            result = subprocess.run(['bash', str(root / 'scripts/run_coat_arm.sh'), arm_id], cwd=root, env=env, capture_output=True, text=True)
            for line in (result.stdout + result.stderr).splitlines():
                if line.startswith(('COAT_ARM', 'TRAIN_', 'ARM=', 'SCHEDULE', 'OPTIMIZER_HPARAMS', 'LIONMUON', 'NORMUON', 'MUON2')) or 'Traceback' in line or 'Error' in line:
                    print(line[:400], flush=True)
            print(f'ARM_EXIT arm_id={arm_id} code={result.returncode}', flush=True)
        return 0
    if mode == 'smoke':
        visible = len(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, check=True).stdout.strip().splitlines())
        for arm_id in args:
            env = dict(os.environ, COAT_ARM_MODE='smoke', STAGE3_MOE_NPROC=str(visible), COAT_CAMPAIGN_ROOT=str(campaign / 'smoke'))
            subprocess.run(['bash', str(root / 'scripts/run_coat_arm.sh'), arm_id], cwd=root, env=env, check=False)
        return 0
    if mode == 'coldstart':
        # Start only once every pretraining process holds its GPUs (user, 2026-10-02 01:05).
        claims = campaign / f'claims-{ROUND}'
        deadline = time.time() + float(os.environ.get('COAT_DIAG_WAIT_SECONDS', '21600'))
        while not all((claims / arm_id).is_dir() for arm_id in COAT_PRIORITY):
            if time.time() > deadline:
                print('COLDSTART_GATE_TIMEOUT', flush=True)
                return 0
            time.sleep(120)
        optimizer, precision, seed = args
        # A retry after an infrastructure failure needs a fresh directory: the first attempt reserved its own.
        tag = f'c1002-cs-{optimizer}-{precision}-s{seed}' + os.environ.get('COAT_DIAG_SUFFIX', '')
        output = campaign / 'coldstart' / tag
        env = dict(os.environ, STAGE3_MOE_LOG_ROOT=str(output), STAGE3_MOE_PROBE_TAG=tag,
                   STAGE3_MOE_COLDSTART_ITERS='1000', STAGE3_MOE_EVAL_ITERS='32')
        result = subprocess.run([sys.executable, str(root / 'scripts/coat_coldstart.py'), optimizer, precision, seed], cwd=root, env=env)
        print('CAMPAIGN_EXIT=' + str(result.returncode), flush=True)
        return 0
    if mode == 'profile':
        result = subprocess.run([sys.executable, str(root / 'tools/coat_profile.py'), *args], cwd=root, capture_output=True, text=True)
        for line in (result.stdout + result.stderr).splitlines():
            if 'COAT_PROFILE' in line or 'Error' in line or 'error' in line:
                print(line[line.find('COAT_PROFILE') if 'COAT_PROFILE' in line else 0:][:4000], flush=True)
        print('CAMPAIGN_EXIT=' + str(result.returncode), flush=True)
        return 0
    if mode == 'muonchunk':
        # Optimizer step time of Muon with COAT momentum at several dequantization chunk sizes.
        for arm_id, chunk in (('muon-ctrl', None), *(('muon-coatopt', c) for c in args)):
            env = dict(os.environ, COAT_ARM_MODE='bench', COAT_BENCH_TAG=f'chunk{chunk}', COAT_BENCH_WARMUP='3', COAT_BENCH_MEASURE='6',
                       STAGE3_MOE_NPROC='1', COAT_CAMPAIGN_ROOT=str(campaign / 'muonchunk'))
            if chunk is not None:
                env['COAT_MUON_CHUNK'] = chunk
            print(f'MUONCHUNK_BEGIN arm_id={arm_id} chunk={chunk}', flush=True)
            subprocess.run(['bash', str(root / 'scripts/run_coat_arm.sh'), arm_id], cwd=root, env=env, check=False)
        return 0
    if mode == 'bench':
        # Same allocation, forward then reverse order, so drift does not masquerade as an effect.
        visible = len(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, check=True).stdout.strip().splitlines())
        for tag, order in (('fwd', args), ('rev', list(reversed(args)))):
            for arm_id in order:
                env = dict(os.environ, COAT_ARM_MODE='bench', COAT_BENCH_TAG=tag, STAGE3_MOE_NPROC=str(visible),
                           COAT_CAMPAIGN_ROOT=str(campaign / 'bench'))
                subprocess.run(['bash', str(root / 'scripts/run_coat_arm.sh'), arm_id], cwd=root, env=env, check=False)
        return 0
    if mode in ('official-build', 'parity'):
        # Upstream qoptim_cuda built by nvcc vs our NVRTC kernels: build on CPU, compare on a GPU.
        # Streamed, so a job killed mid-build (OOM on the 8 GB CPU nodes) still leaves its diagnostics.
        process = subprocess.Popen([sys.executable, '-u', str(root / 'tools/coat_official_parity.py'),
                                    'build' if mode == 'official-build' else 'compare'],
                                   cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            if 'PARITY' in line or 'Error' in line or 'error' in line:
                print(line[line.find('PARITY') if 'PARITY' in line else 0:].rstrip()[:6000], flush=True)
        print('CAMPAIGN_EXIT=' + str(process.wait()), flush=True)
        return 0
    if mode == 'test':
        subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv,noheader'], check=True)
        if not list(campaign.glob('official-qoptim/*/parity-*.json')):
            # Queued test jobs also run the official-build parity until one has been recorded.
            parity = subprocess.run([sys.executable, str(root / 'tools/coat_official_parity.py'), 'compare'], cwd=root, capture_output=True, text=True)
            for line in (parity.stdout + parity.stderr).splitlines():
                if 'PARITY' in line or 'Error' in line:
                    print(line[line.find('PARITY') if 'PARITY' in line else 0:][:6000], flush=True)
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-rfE', '--tb=line', '-s', '-p', 'no:cacheprovider',
                                 *(args or ['tests/stage3_moe/test_coat_optimizer.py', 'tests/stage3_moe/test_coat_activation.py'])], cwd=root, capture_output=True, text=True)
        # mlsub keeps only the tail of a log: print the measurements and verdicts, nothing else.
        for line in (result.stdout + result.stderr).splitlines():
            if 'COAT_REPORT' in line:
                print(line[line.index('COAT_REPORT'):][:3000], flush=True)
            elif line.startswith(('FAILED', 'ERROR', 'E ', '/')) or ' passed' in line or ' failed' in line:
                print(line[:2000], flush=True)
        print('CAMPAIGN_EXIT=' + str(result.returncode), flush=True)
        return 0
    raise SystemExit('unknown mode: ' + mode)


if __name__ == '__main__':
    raise SystemExit(main())
