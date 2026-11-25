"""Prepare machine-specific records before the agent starts; never runs during grading.

Runs as root on the allocated GPUs (bench.json world_size), following the judge's acceptance procedure:
  1. the frozen probe batch of the statistical gate (bench/data/probe.pt, from the reference's validation loader);
  2. the reference recorded twice in deterministic mode (the two records must be identical, else this host cannot
     judge bitwise and preparation stops; FlashAttention-3 must be in use), its record frozen as
     bench/records/ref.json;
  3. the reference's trajectories for seeds 1-6 and the frozen seed sigma of every endpoint (bench/traj/);
  4. the frozen input fingerprints, every one readable by the agent user;
  5. the reference's own quick check (PASS, bitwise) and full check (`check --final`, as root: UNAVAILABLE, no
     benefit; its two reference timings are kept as task/timing_ref.json, the frozen reference timing of `time` and
     of gate 1 of the agent's `check`);
  6. acceptance controls (RESEARCHBOOST_NANOCHAT_CONTROLS, default 1): a tree with the logit soft-cap halved must be
     INCORRECT (statistical gate); a tree that runs 8 sequences per micro-batch and 8 accumulation steps (the same
     rows and the same update, summed in another order, FP8 scales per smaller micro-batch) must PASS at N1.
Any failure stops the container.

With a preparation pack ($ASSETS/pack, made from an earlier full preparation on the same GPU model: records, probe
batch, trajectories), steps 1-3 restore the pack and the reference is recorded once: it must be bitwise identical to
the pack's record, else preparation stops. The quick check follows, and the reference timed twice
(task/timing_ref.json); the full check and the controls are skipped.

RESEARCHBOOST_SMOKE=1 (integration tests) keeps the two records and the quick check; trajectories, the full check and
the controls are skipped, so only quick checks can be graded.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

WS = Path('/opt/researchboost/ws')
ASSETS = Path(os.environ.get('ASSETS', '/opt/researchboost/assets'))
CANDIDATE = '/submission/nanochat'
PY = sys.executable
SCRATCH = Path('/tmp/nanochat-prep')
MIN_GPU_MEMORY_GIB = 130
PACK = ASSETS / 'pack'
PACK_FILES = ['bench/records/ref.json', 'bench/data/probe.pt', 'bench/traj/ref_runs.json', 'bench/traj/frozen.json']
SOFTCAP_LINE = '        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]'
MICRO_LINE = 'user_config = vars(args).copy()  # for logging'


def log(message):
    print('[bootstrap %s] %s' % (datetime.now().strftime('%H:%M:%S'), message), flush=True)


def run(*args, cwd=WS, env=None):
    log('run: %s' % ' '.join(str(a) for a in args))
    subprocess.run([PY, *args], cwd=cwd, check=True, env=env)


def failure_detail(output):
    data = json.loads((WS / output / 'check.json').read_text())
    correctness = data.get('correctness') or {}
    num = correctness.get('numerics') or {}
    keep = {'note': data.get('note') or correctness.get('note'), 'numerics': {k: num.get(k) for k in
            ('status', 'reason', 'problems', 'differing')}, 'trajectory': {k: (correctness.get('trajectory') or {})
            .get(k) for k in ('verdict', 'seeds', 'endpoints')},
            'performance_reason': (data.get('performance') or {}).get('reason')}
    lines = ['check.json: ' + json.dumps(keep, default=str)[:4000]]
    logs = sorted((WS / output).rglob('*.log'), key=lambda f: f.stat().st_mtime)[-3:]
    for f in logs:
        tail = f.read_text(errors='replace').splitlines()[-20:]
        lines += ['--- %s (last %d lines)' % (f.relative_to(WS / output), len(tail)), *tail]
    return '\n'.join(lines)


def check(tree, output, expected, *options, verdict=None, tier=None):
    # Every non-quick check as the final evaluation runs it (root: final_allowed); for the full check this is the
    # reference -> candidate -> reference timing that task/timing_ref.json is made from.
    options = options if '--quick' in options else (*options, '--final')
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
    """Everything the judge's numbers depend on besides the tree under test: its own code, records, probe batch and
    trajectories, the data (ClimbMix shards, tokenizer) and the FlashAttention-3 kernel cache."""
    return ['bench', str(ASSETS / 'data'), config['hf_home']]


def make_probe(config):
    ref = WS / config['reference']
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
        base = Path(tmp) / 'nanochat'
        base.mkdir()
        for name in ('base_data_climbmix', 'tokenizer'):
            (base / name).symlink_to(ASSETS / 'data' / name)
        env = dict(os.environ, NANOCHAT_BASE_DIR=str(base), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        run(str(WS / 'bench/make_probe.py'), str(WS / config['probe']), '--world', str(config['world_size']), cwd=ref,
            env=env)


def record_reference(config, labels):
    ref = config['reference']
    for label in labels:
        run('bench/bench.py', 'record', '--tree', ref, '--out', f'runs/bootstrap/record_{label}', '--label', 'ref')
        rec = json.loads((WS / f'runs/bootstrap/record_{label}/record.json').read_text())
        if rec.get('fa3') != ['True']:
            raise RuntimeError('the reference did not run FlashAttention-3 on every rank (%s): the kernel cache in %s '
                               'is not usable here' % (rec.get('fa3'), config['hf_home']))
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
    smoke = os.environ.get('RESEARCHBOOST_SMOKE') == '1'
    controls = os.environ.get('RESEARCHBOOST_NANOCHAT_CONTROLS', '1') == '1'
    pack = (PACK / 'pack.json').is_file() and not smoke
    # A pack records one world size (packs without one were made on 8 GPUs); another one's records cannot match.
    if pack and json.loads((PACK / 'pack.json').read_text()).get('world_size', 8) != config['world_size']:
        log('preparation pack at %s is for another world size: full preparation' % PACK)
        pack = False
    # Gate 3 runs the production settings (11-13): a pack whose trajectories were recorded in deterministic mode
    # (no "trajectories": "production" in its pack.json) cannot serve as its reference.
    if pack and json.loads((PACK / 'pack.json').read_text()).get('trajectories') != 'production':
        log('preparation pack at %s has deterministic-mode trajectories: full preparation' % PACK)
        pack = False
    for d in ('task', 'bench/records', 'bench/data', 'bench/traj'):
        (WS / d).mkdir(parents=True, exist_ok=True)
    (WS / 'task/formulation.json').write_text(json.dumps(
        {'task': config['task'], 'reference': {'path': config['reference'], 'commit': config['reference_commit']},
         'inputs': {}}) + '\n')
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    labels = ['a', 'b']
    if pack:
        meta = json.loads((PACK / 'pack.json').read_text())
        log('preparation pack: %s' % json.dumps(meta))
        for rel in PACK_FILES:
            shutil.copyfile(PACK / rel, WS / rel)
        record = record_reference(config, ['a'])
        same = subprocess.run([PY, 'bench/bench.py', 'compare', config['record']['reference'],
                               'runs/bootstrap/record_a/record.json'], cwd=WS, stdout=subprocess.DEVNULL)
        if same.returncode == 0:
            log('reference record: bitwise identical to the preparation pack')
        else:
            log('the reference record of this host differs from the preparation pack: full preparation')
            for rel in PACK_FILES:
                (WS / rel).unlink()
            pack, labels = False, ['b']
    if not pack:
        make_probe(config)
        # The reference recorded twice: this host must reproduce it bitwise, or no bitwise verdict is possible here.
        record = record_reference(config, labels)
        run('bench/bench.py', 'compare', 'runs/bootstrap/record_a/record.json', 'runs/bootstrap/record_b/record.json')
        shutil.copyfile(WS / 'runs/bootstrap/record_a/record.json', WS / config['record']['reference'])
    log('reference record: %d updates, %d parameters, %d rows per rank, deterministic %s, reproduced bitwise'
        % (record['steps'], len(record['parameters']), len(record['rows'][0]), record['deterministic']))
    if not pack and not smoke:
        run('bench/bench.py', 'traj-ref', '--out', 'runs/bootstrap/traj')
        frozen = json.loads((WS / config['trajectory']['frozen']).read_text())
        log('frozen trajectory bounds: %s' % {k: round(v['delta'], 5) for k, v in frozen['endpoints'].items()})
    run('bench/bench.py', 'freeze-inputs', *frozen_inputs(config))
    # The agent and the final check run as the unprivileged user: every frozen input must be world-readable.
    inputs = json.loads((WS / 'task/formulation.json').read_text())['inputs']
    unreadable = [q for q in inputs if not os.stat(q if os.path.isabs(q) else WS / q).st_mode & 0o004]
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
    if pack:
        run('bench/bench.py', 'baseline', '--out', 'runs/bootstrap/baseline')
        base = json.loads((WS / 'runs/bootstrap/baseline/baseline.json').read_text())
        windows, spread, host = base['window_s_per_update'], base.get('spread'), None
        source = 'runs/bootstrap/baseline (the reference timed twice)'
    else:
        full = check(CANDIDATE, 'runs/bootstrap/full', 'UNAVAILABLE', verdict='no_benefit')
        perf = full['performance']
        windows = [min(a, b) for a, b in zip(perf['reference_a_s_per_update'], perf['reference_b_s_per_update'])]
        spread, host = perf.get('reference_spread'), perf.get('host')
        source = "runs/bootstrap/full (reference A and B of the unchanged tree's full check)"
    (WS / 'task/timing_ref.json').write_text(json.dumps(
        {'source': source, 'window_s_per_update': windows, 's_per_update': round(sum(windows) / len(windows), 4),
         'spread': spread, 'host': host}, indent=1) + '\n')
    log('reference timing: %s s/update per window (spread %s)' % (windows, spread))
    if controls and not pack:
        tree = control_tree(config, 'softcap_half', SOFTCAP_LINE, SOFTCAP_LINE.replace('= 15', '= 7.5'),
                            'nanochat/gpt.py')
        check(tree, 'runs/bootstrap/control_softcap_half', 'INCORRECT', '--skip-timing', verdict='different')
        tree = control_tree(config, 'micro8', MICRO_LINE,
                            MICRO_LINE + '\nargs.device_batch_size = 8  # acceptance control: 8 accumulation steps',
                            'scripts/base_train.py')
        check(tree, 'runs/bootstrap/control_micro8', 'PASS', '--skip-timing', verdict='equivalent', tier='N1')
    shutil.rmtree(SCRATCH, ignore_errors=True)
    ready(devices, 'pack' if pack else 'full', started, controls=bool(controls and not pack))


if __name__ == '__main__':
    main()
