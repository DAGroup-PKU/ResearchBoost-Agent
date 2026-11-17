"""Prepare machine-specific records before the agent starts; never runs during grading.

Runs as root on the four allocated GPUs, following the judge's acceptance procedure: the reference recorded twice in
deterministic mode (the two records must be identical, else this host cannot judge bitwise and preparation stops),
its record frozen as bench/records/ref.json, the frozen input fingerprints, the reference's own quick check (PASS,
bitwise) and full check (`check --final`, as root: UNAVAILABLE, no benefit; its two reference timings are kept as
task/timing_ref.json, the frozen reference timing of `time` and of gate 1 of the agent's `check`), and one acceptance control: a tree whose rollout sampling seeds are shifted by
one step must be INCORRECT. Any failure stops the container.

RESEARCHBOOST_SMOKE=1 (integration tests) keeps the two records and the quick check; the full check and the control
are skipped, so only quick checks can be graded.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

WS = Path('/opt/researchboost/ws')
ASSETS = Path(os.environ.get('ASSETS', '/opt/researchboost/assets'))
CANDIDATE = '/submission/r3'
PY = sys.executable
SCRATCH = Path('/tmp/r3-prep')
MIN_GPU_MEMORY_GIB = 130
# One acceptance control: the generation seed of every rank shifted by one step (a different rollout stream).
SEED_LINE = 'gen_seed = (dist.get_rank() + 1) * training_args.total_steps + curr_step'


def log(message):
    print('[bootstrap %s] %s' % (datetime.now().strftime('%H:%M:%S'), message), flush=True)


def run(*args):
    log('run: %s' % ' '.join(str(a) for a in args))
    subprocess.run([PY, *args], cwd=WS, check=True)


def failure_detail(output):
    data = json.loads((WS / output / 'check.json').read_text())
    correctness = data.get('correctness') or {}
    num = correctness.get('numerics') or {}
    keep = {'note': data.get('note') or correctness.get('note'), 'numerics': {k: num.get(k) for k in
            ('status', 'reason', 'problems')}, 'performance_reason': (data.get('performance') or {}).get('reason')}
    lines = ['check.json: ' + json.dumps(keep, default=str)[:3000]]
    logs = sorted((WS / output).rglob('*.log'), key=lambda f: f.stat().st_mtime)[-3:]
    for f in logs:
        tail = f.read_text(errors='replace').splitlines()[-20:]
        lines += ['--- %s (last %d lines)' % (f.relative_to(WS / output), len(tail)), *tail]
    return '\n'.join(lines)


def check(tree, output, expected, *options, verdict=None):
    log('check %s -> %s %s (expect %s%s)' % (tree, output, ' '.join(options), expected,
                                             ', %s' % verdict if verdict else ''))
    result = subprocess.run([PY, 'bench/bench.py', 'check', '--tree', tree, '--out', output, *options], cwd=WS)
    data = json.loads((WS / output / 'check.json').read_text())
    got = (data.get('status'), (data.get('correctness') or {}).get('verdict'))
    if got[0] != expected or result.returncode != (0 if expected == 'PASS' else 1) or \
            (verdict is not None and got[1] != verdict):
        log(failure_detail(output))
        raise RuntimeError('Preparation control failed: %s: got status/verdict %s, expected %s'
                           % (output, got, (expected, verdict)))
    log('check %s: %s %s' % (output, *got))
    return data


def gpus(config):
    import torch
    world = config['world_size']
    if torch.cuda.device_count() != world:
        raise RuntimeError('This task requires exactly %d allocated GPU(s), found %d' % (world, torch.cuda.device_count()))
    arch = int(os.environ['RESEARCHBOOST_GPU_ARCH'])
    props = [torch.cuda.get_device_properties(i) for i in range(world)]
    if any(p.major != arch for p in props):
        raise RuntimeError('This image requires GPUs of compute capability %d.x, found %s'
                           % (arch, sorted({(p.major, p.minor) for p in props})))
    small = [p.name for p in props if p.total_memory / 2**30 < MIN_GPU_MEMORY_GIB]
    if small:
        raise RuntimeError('This task needs at least %d GiB per GPU, found %s' % (MIN_GPU_MEMORY_GIB, small))
    return [p.name for p in props]


def ready(devices, mode, started, **extra):
    (WS / 'task/ready.json').write_text(json.dumps(dict(
        status='ready', mode=mode, gpus=devices, world_size=len(devices),
        preparation={'mode': mode, 'seconds': round(time.time() - started)},
        created=datetime.now().isoformat(timespec='seconds'), **extra), indent=1) + '\n')
    log('ready (%s preparation, %.0f min)' % (mode, (time.time() - started) / 60))


def frozen_inputs():
    """Everything the judge's numbers depend on besides the tree under test: its own code, records and data copies,
    the weights, and the reward service (its code, the detector config it reads)."""
    return ['bench', str(ASSETS / 'BAGEL-7B-MoT'), str(ASSETS / 'mask2former'), str(ASSETS / 'hf'),
            '/opt/reward-server/app_geneval.py', '/opt/reward-server/reward_server',
            '/opt/mmdetection/configs/mask2former', '/opt/mmdetection/configs/_base_', '/opt/mmdetection/mmdet']


def main():
    started = time.time()
    config = json.loads((WS / 'bench/bench.json').read_text())
    devices = gpus(config)
    log('GPUs: %s' % devices)
    lock = Path(config['gpu_lock'])
    lock.touch()
    lock.chmod(0o666)
    smoke = os.environ.get('RESEARCHBOOST_SMOKE') == '1'
    (WS / 'task').mkdir(exist_ok=True)
    (WS / 'bench/records').mkdir(exist_ok=True)
    (WS / 'task/formulation.json').write_text(json.dumps(
        {'task': 'R3 Tree-RL of BAGEL-7B-MoT (sen-ye/R3 @ 761b0b0), GenEval reward',
         'reference': {'path': config['reference'], 'commit': config['reference_commit']}, 'inputs': {}}) + '\n')
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    # The reference recorded twice: this host must reproduce it bitwise, or no bitwise verdict is possible here.
    ref = config['reference']
    for label in ('a', 'b'):
        run('bench/bench.py', 'record', '--tree', ref, '--out', f'runs/bootstrap/record_{label}', '--label', 'ref')
    run('bench/bench.py', 'compare', 'runs/bootstrap/record_a/record.json', 'runs/bootstrap/record_b/record.json')
    record = json.loads((WS / 'runs/bootstrap/record_a/record.json').read_text())
    log('reference record: %d optimizer steps, %d tensors, %d reward requests, reproduced bitwise'
        % (record['steps'], len(record['digests']), record['reward_requests']))
    shutil.copyfile(WS / 'runs/bootstrap/record_a/record.json', WS / config['record']['reference'])
    run('bench/bench.py', 'freeze-inputs', *frozen_inputs())
    # The agent and the final check run as the unprivileged user: every frozen input must be world-readable.
    inputs = json.loads((WS / 'task/formulation.json').read_text())['inputs']
    unreadable = [p for p in inputs if not os.stat(p if os.path.isabs(p) else WS / p).st_mode & 0o004]
    if unreadable:
        raise RuntimeError('%d frozen inputs are not readable by the agent user, e.g. %s' % (len(unreadable),
                                                                                            unreadable[:3]))
    quick = check(CANDIDATE, 'runs/bootstrap/quick', 'PASS', '--quick')
    if not (quick.get('correctness') or {}).get('numerics', {}).get('bitwise'):
        raise RuntimeError("The unchanged tree's quick check is not bitwise identical to the reference record")
    if smoke:
        shutil.rmtree(SCRATCH, ignore_errors=True)
        ready(devices, 'smoke', started)
        return
    # As the final evaluation runs it (root: final_allowed): reference -> candidate -> reference, the reference timings
    # task/timing_ref.json is made from.
    full = check(CANDIDATE, 'runs/bootstrap/full', 'UNAVAILABLE', '--final', verdict='no_benefit')
    perf = full['performance']
    windows = [min(a, b) for a, b in zip(perf['reference_a_s_per_step'], perf['reference_b_s_per_step'])]
    (WS / 'task/timing_ref.json').write_text(json.dumps(
        {'source': 'runs/bootstrap/full (reference A and B of the unchanged tree\'s full check)',
         'window_s_per_step': windows, 's_per_step': round(sum(windows) / len(windows), 4),
         'spread': perf.get('reference_spread'), 'host': perf.get('host')}, indent=1) + '\n')
    log('reference timing: %s s/step per window (spread %s)' % (windows, perf.get('reference_spread')))
    control = SCRATCH / 'accept' / 'seed_shift'
    shutil.copytree(WS / ref, control, ignore=shutil.ignore_patterns('.git'))
    src = (control / 'train/online_rl.py').read_text()
    if src.count(SEED_LINE) != 1:
        raise RuntimeError('acceptance control: the seed line of train/online_rl.py was not found once')
    (control / 'train/online_rl.py').write_text(src.replace(SEED_LINE, SEED_LINE + ' + 1'))
    check(str(control), 'runs/bootstrap/control_seed_shift', 'INCORRECT', '--quick', verdict='not_bitwise')
    shutil.rmtree(SCRATCH, ignore_errors=True)
    ready(devices, 'full', started)


if __name__ == '__main__':
    main()
