"""Prepare machine-specific records before the agent starts; never runs during grading.

Runs as root on the allocated GPU (bench.json world_size), following the judge's acceptance procedure:
  1. the reference's quick record (record.steps steps, seed 0) made twice: the exact items (camera sequence, learning
     rates, densification steps, dtypes) must be identical, else preparation stops; frozen as bench/records/ref.json;
  2. the reference's full runs for the seeds of bench.json trajectory.reference_seeds, each scored by the judge's
     evaluation, their exact items (bench/traj/records/) and the frozen seed sigma of every endpoint (bench/traj/frozen.json);
  3. the frozen input fingerprints (the judge's code and records, the scene, the LPIPS weights, the reference's built
     extensions), every one readable by the agent user;
  4. the unchanged tree's quick check (PASS, exact items) and full check (UNAVAILABLE: no benefit; its two reference
     timings are kept as task/timing_ref.json, the informational baseline of `time`);
  5. acceptance controls (RESEARCHBOOST_GS_CONTROLS, default 1): a tree that always renders the first remaining
     camera must be INCORRECT (exact items); a tree whose D-SSIM weight is halved must be INCORRECT (statistical
     gate); a tree that computes SSIM with upstream's PyTorch implementation instead of fused-ssim (the same formula,
     another kernel) must PASS at N1.
Any failure stops the container.

With a preparation pack ($ASSETS/pack, made from an earlier full preparation on the same GPU model: the quick record,
the reference runs' exact items and evaluations, the frozen sigma), steps 1-2 restore the pack and the reference's
quick record is made once: its exact items must equal the pack's, else full preparation. The quick check follows and
the reference is timed twice (task/timing_ref.json); the full check and the controls are skipped.

RESEARCHBOOST_SMOKE=1 (integration tests) keeps the two records and the quick check; the reference runs, the full
check and the controls are skipped, so only quick checks can be graded.
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
CANDIDATE = '/submission/gaussian-splatting'
PY = sys.executable
SCRATCH = Path('/tmp/gs-prep')
MIN_GPU_MEMORY_GIB = 70
PACK = ASSETS / 'pack'
PACK_DIRS = ['bench/records', 'bench/traj']
CAMERA_LINE = '        rand_idx = randint(0, len(viewpoint_indices) - 1)'
LOSS_LINE = '        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)'
SSIM_LINE = '        if FUSED_SSIM_AVAILABLE:'


def log(message):
    print('[bootstrap %s] %s' % (datetime.now().strftime('%H:%M:%S'), message), flush=True)


def run(*args, cwd=WS, env=None):
    log('run: %s' % ' '.join(str(a) for a in args))
    subprocess.run([PY, *args], cwd=cwd, check=True, env=env)


def failure_detail(output):
    data = json.loads((WS / output / 'check.json').read_text())
    correctness = data.get('correctness') or {}
    num = correctness.get('numerics') or {}
    keep = {'note': data.get('note') or correctness.get('note'),
            'numerics': {k: num.get(k) for k in ('status', 'reason', 'problems')},
            'trajectory': {k: (correctness.get('trajectory') or {}).get(k) for k in ('verdict', 'seeds', 'endpoints')},
            'performance_reason': (data.get('performance') or {}).get('reason')}
    lines = ['check.json: ' + json.dumps(keep, default=str)[:4000]]
    logs = sorted((WS / output).rglob('*.log'), key=lambda f: f.stat().st_mtime)[-3:]
    for f in logs:
        tail = f.read_text(errors='replace').splitlines()[-20:]
        lines += ['--- %s (last %d lines)' % (f.relative_to(WS / output), len(tail)), *tail]
    return '\n'.join(lines)


def check(tree, output, expected, *options, verdict=None, tier=None):
    options = options if '--quick' in options else (*options, '--final')   # gate 1 as the final evaluation runs it
    log('check %s -> %s %s (expect %s%s)' % (tree, output, ' '.join(options), expected,
                                             ', %s' % verdict if verdict else ''))
    result = subprocess.run([PY, 'bench/bench.py', 'check', '--tree', tree, '--out', output, *options], cwd=WS)
    data = json.loads((WS / output / 'check.json').read_text())
    corr = data.get('correctness') or {}
    got = (data.get('status'), corr.get('verdict'), corr.get('measured_tier'))
    if got[0] != expected or result.returncode != (0 if expected == 'PASS' else 1) or \
            (verdict is not None and got[1] != verdict) or (tier is not None and got[2] != tier):
        log(failure_detail(output))
        raise RuntimeError('Preparation control failed: %s: got status/verdict/tier %s, expected %s'
                           % (output, got, (expected, verdict, tier)))
    log('check %s: %s %s %s' % (output, *got))
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


def frozen_inputs(config):
    """Everything the judge's numbers depend on besides the tree under test: its own code and records, the scene,
    the LPIPS / VGG weights and the reference's built extensions."""
    return ['bench', str(ASSETS / 'tandt' / 'truck'), config['torch_home'], config['reference_ext']]


def record_reference(config, labels):
    for label in labels:
        run('bench/bench.py', 'record', '--tree', config['reference'], '--out', f'runs/bootstrap/record_{label}',
            '--label', 'ref')
    return json.loads((WS / f'runs/bootstrap/record_{labels[0]}/record.json').read_text())


def control_tree(config, name, old, new, relpath):
    tree = SCRATCH / 'accept' / name
    shutil.copytree(WS / config['reference'], tree, ignore=shutil.ignore_patterns('.git'))
    src = (tree / relpath).read_text()
    if src.count(old) != 1:
        raise RuntimeError('acceptance control %s: the line to change in %s was not found once' % (name, relpath))
    (tree / relpath).write_text(src.replace(old, new))
    return str(tree)


def main():
    started = time.time()
    config = json.loads((WS / 'bench/bench.json').read_text())
    devices = gpus(config)
    log('GPUs: %s' % devices)
    lock = Path(config['gpu_lock'])
    lock.touch()
    lock.chmod(0o666)
    Path(config['ext_cache']).mkdir(parents=True, exist_ok=True)
    Path(config['ext_cache']).chmod(0o1777)
    smoke = os.environ.get('RESEARCHBOOST_SMOKE') == '1'
    controls = os.environ.get('RESEARCHBOOST_GS_CONTROLS', '1') == '1'
    pack = (PACK / 'pack.json').is_file() and not smoke
    for d in ('task', 'bench/records', 'bench/traj/records'):
        (WS / d).mkdir(parents=True, exist_ok=True)
    head = subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(WS / config['reference']), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, check=True).stdout.strip()
    (WS / 'task/reference_commit').write_text(head + '\n')
    (WS / 'task/formulation.json').write_text(json.dumps(
        {'task': 'gaussian_splatting', 'reference': {'path': config['reference'], 'commit': head},
         'inputs': {}}) + '\n')
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    labels = ['a', 'b']
    if pack:
        meta = json.loads((PACK / 'pack.json').read_text())
        log('preparation pack: %s' % json.dumps(meta))
        if meta.get('reference_commit') != head:
            log('the pack is for reference commit %s, this image has %s: full preparation'
                % (meta.get('reference_commit'), head))
            pack = False
    if pack:
        for d in PACK_DIRS:
            shutil.copytree(PACK / d, WS / d, dirs_exist_ok=True)
        record = record_reference(config, ['a'])
        same = subprocess.run([PY, 'bench/bench.py', 'compare', config['record']['reference'],
                               'runs/bootstrap/record_a/record.json'], cwd=WS, stdout=subprocess.DEVNULL)
        if same.returncode == 0:
            log('reference record: exact items identical to the preparation pack')
        else:
            log('the reference record of this host differs from the preparation pack: full preparation')
            for d in PACK_DIRS:
                shutil.rmtree(WS / d)
                (WS / d).mkdir(parents=True)
            (WS / 'bench/traj/records').mkdir(parents=True, exist_ok=True)
            pack, labels = False, ['b']
    if not pack:
        record = record_reference(config, labels)
        run('bench/bench.py', 'compare', 'runs/bootstrap/record_a/record.json', 'runs/bootstrap/record_b/record.json')
        shutil.copyfile(WS / 'runs/bootstrap/record_a/record.json', WS / config['record']['reference'])
    log('reference record: %d steps, %d renders, %d changes of the number of Gaussians, final %s; exact items '
        'reproduced' % (record['steps'], len(record['cameras']), len(record['count_change_steps']),
                        record.get('final_count')))
    if not pack and not smoke:
        run('bench/bench.py', 'traj-ref', '--out', 'runs/bootstrap/traj')
        frozen = json.loads((WS / config['trajectory']['reference']).read_text())
        log('frozen statistical bounds: %s' % {k: (round(v['reference_mean'], 5), round(v['delta'], 5))
                                               for k, v in frozen['endpoints'].items()})
    run('bench/bench.py', 'freeze-inputs', *frozen_inputs(config))
    # The agent and the final check run as the unprivileged user: every frozen input must be world-readable.
    inputs = json.loads((WS / 'task/formulation.json').read_text())['inputs']
    unreadable = [q for q in inputs if not os.stat(q if os.path.isabs(q) else WS / q).st_mode & 0o004]
    if unreadable:
        raise RuntimeError('%d frozen inputs are not readable by the agent user, e.g. %s' % (len(unreadable),
                                                                                            unreadable[:3]))
    check(CANDIDATE, 'runs/bootstrap/quick', 'PASS', '--quick', verdict='exact_items_pass')
    if smoke:
        shutil.rmtree(SCRATCH, ignore_errors=True)
        ready(devices, 'smoke', started)
        return
    if pack:
        run('bench/bench.py', 'baseline', '--out', 'runs/bootstrap/baseline', '--labels', 'ref_a,ref_b')
        base = json.loads((WS / 'runs/bootstrap/baseline/baseline.json').read_text())
        total, spread, host = base['total_s'], base.get('spread'), None
        source = 'runs/bootstrap/baseline (the reference timed twice)'
    else:
        full = check(CANDIDATE, 'runs/bootstrap/full', 'UNAVAILABLE', verdict='no_benefit')
        perf = full['performance']
        total, spread, host = min(perf['reference_a_s'], perf['reference_b_s']), perf.get('reference_spread'), \
            perf.get('host')
        source = "runs/bootstrap/full (reference A and B of the unchanged tree's full check)"
    (WS / 'task/timing_ref.json').write_text(json.dumps(
        {'source': source, 'total_s': total, 'spread': spread, 'host': host,
         'marks': config['timing']['marks']}, indent=1) + '\n')
    log('reference timing: %.1f s for steps %s (spread %s)' % (total, config['timing']['marks'], spread))
    if controls and not pack:
        tree = control_tree(config, 'first_camera', CAMERA_LINE, '        rand_idx = 0', 'train.py')
        check(tree, 'runs/bootstrap/control_first_camera', 'INCORRECT', '--quick', verdict='exact_items')
        tree = control_tree(config, 'dssim_half', LOSS_LINE,
                            LOSS_LINE.replace('opt.lambda_dssim', '(0.5 * opt.lambda_dssim)'), 'train.py')
        check(tree, 'runs/bootstrap/control_dssim_half', 'INCORRECT', '--skip-timing', verdict='different')
        tree = control_tree(config, 'pytorch_ssim', SSIM_LINE, '        if False:  # acceptance control: '
                            'upstream\'s PyTorch SSIM', 'train.py')
        check(tree, 'runs/bootstrap/control_pytorch_ssim', 'PASS', '--skip-timing', verdict='equivalent', tier='N1')
    shutil.rmtree(SCRATCH, ignore_errors=True)
    ready(devices, 'pack' if pack else 'full', started, controls=bool(controls and not pack))


if __name__ == '__main__':
    main()
