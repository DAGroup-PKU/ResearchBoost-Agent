"""Prepare machine-specific records before the agent starts; never runs during grading.

Runs as root on the allocated GPUs, following the judge's own formulation and acceptance procedure: the reference's
bitwise records (twice) and its non-deterministic variant, the diagnostic thresholds, the loader digest, six reference
trajectory seeds (seed 1 recorded twice, identical) and their frozen delta, the reference timed three times (the
per-update random-number stream, task/timing_ref.json), the frozen input fingerprints, the unchanged tree's quick and
full checks, the numerics self-test and the acceptance controls. Any failure stops the container.

RESEARCHBOOST_SMOKE=1 (integration tests) keeps the records, the loader digest, one timing run of the real entrypoint
and the unchanged tree's quick check; trajectories, the timing baseline, the full check, the self-test and the
controls are skipped, so only quick checks can be graded.

A matching preparation pack (prep_pack.py; tasks/<task>/prep-pack, from a full preparation on the same GPU type and
world size with the same judge workspace) restores the records, thresholds, reference trajectories with their frozen
delta, the self-test and the controls' evidence; this machine still makes the frozen input fingerprints, the timing
baseline and the reference's quick and full checks, and the pack stands only when that quick check is bitwise
identical to the pack's records (else the full preparation runs).
"""
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import prep_pack

WS = Path('/opt/researchboost/ws')
PREP = Path('/opt/researchboost/prep')
ASSETS = Path(os.environ.get('ASSETS', '/opt/researchboost/assets'))
CANDIDATE = '/submission/yoco'
PY = sys.executable
SCRATCH = Path('/tmp/yoco-prep')
TRAJ_SEEDS = '1,2,3,4,5,6'
# Bump by hand whenever this file changes what preparation produces: preparation packs (prep_pack.py) recorded with
# another value are not used.
PREP_VERSION = 2
# Training-environment files the judge's numbers depend on (frozen like the formulated judge's inputs).
ENV_FILES = ('fairseq/trainer.py', 'fairseq/checkpoint_utils.py', 'fairseq/criterions/cross_entropy.py',
             'fairseq/data/iterators.py', 'fairseq/distributed/utils.py', 'fairseq/logging', 'fairseq/optim',
             'fairseq_cli/train.py', 'fairscale/optim/oss.py', 'apex/optimizers/fused_adam.py',
             'flash_attn/flash_attn_interface.py', 'infinibatch/iterators.py')
# (name, expected status, mode, expected verdict or None): the formulated judge's acceptance (task/prep/accept.sh)
CONTROLS = [('setup_entry', 'ERROR', '--quick', None), ('wd0', 'INCORRECT', '--quick', 'structure'),
            ('freeze_norm', 'INCORRECT', '--quick', 'structure'), ('lr102', 'INCORRECT', '--skip-timing', None),
            ('rms_reorder', 'PASS', '--skip-timing', 'equivalent'),
            ('detach_half', 'INCORRECT', '--skip-timing', None)]


def log(message):
    print('[bootstrap %s] %s' % (datetime.now().strftime('%H:%M:%S'), message), flush=True)


def run(*args):
    log('run: %s' % ' '.join(str(a) for a in args))
    subprocess.run([PY, *args], cwd=WS, check=True)


def failure_detail(output, data):
    """What a failed control's check.json says, and the end of the newest logs the check wrote: the workspace goes
    with the container, so this is all that is left to tell why the control failed."""
    correctness = data.get('correctness') or {}
    num = correctness.get('numerics') or {}
    keep = {'reason': data.get('reason') or correctness.get('reason'), 'numerics_status': num.get('status'),
            'numerics_reason': num.get('reason'), 'errors': (num.get('errors') or [])[:3],
            'exact_failures': (num.get('exact_failures') or [])[:3], 'problems': (correctness.get('problems') or [])[:3]}
    lines = ['check.json: ' + json.dumps({k: v for k, v in keep.items() if v}, default=str)[:3000]]
    logs = sorted((WS / output).rglob('*.log'), key=lambda f: f.stat().st_mtime)[-3:]
    for f in logs:
        tail = f.read_text(errors='replace').splitlines()[-15:]
        lines += ['--- %s (last %d lines)' % (f.relative_to(WS / output), len(tail)), *tail]
    return '\n'.join(lines)


def check(tree, output, expected, *options, verdict=None):
    options = options if '--quick' in options else (*options, '--final')   # gate 1 as the final evaluation runs it
    log('check %s -> %s %s (expect %s%s)' % (tree, output, ' '.join(options), expected,
                                             ', %s' % verdict if verdict else ''))
    result = subprocess.run([PY, 'bench/bench.py', 'check', '--tree', tree, '--out', output, *options], cwd=WS)
    data = json.loads((WS / output / 'check.json').read_text())
    correctness = data.get('correctness') or {}
    got = (data.get('status'), correctness.get('verdict'), correctness.get('stage'))
    if got[0] != expected or result.returncode != (0 if expected == 'PASS' else 1) or \
            (verdict is not None and got[1] != verdict):
        log(failure_detail(output, data))
        raise RuntimeError('Preparation control failed: %s: got status/verdict/stage %s, expected %s'
                           % (output, got, (expected, verdict)))
    log('check %s: %s' % (output, got))
    return data


def check_scratch(directory='/tmp'):
    """DataLoader workers pass file descriptors over Unix sockets created under TMPDIR, which the judge puts under
    /tmp (the host's <scratch root>/<task>). Some network and FUSE filesystems refuse Unix sockets (EOPNOTSUPP), and
    every run with loader workers then hangs until its timeout; stop at once with the reason instead."""
    import socket
    import tempfile
    holder = tempfile.mkdtemp(prefix='rb-sock-', dir=directory)
    sock = socket.socket(socket.AF_UNIX)
    try:
        sock.bind(os.path.join(holder, 's'))
    except OSError as exc:
        raise RuntimeError('The scratch directory %s (<scratch root>/<task> on the host) does not support Unix sockets '
                           '(%s); put RESEARCHBOOST_SCRATCH_ROOT on a local filesystem' % (directory, exc)) from exc
    finally:
        sock.close()
        shutil.rmtree(holder, ignore_errors=True)


def test_world_size(config):
    """RESEARCHBOOST_TEST_WORLD_SIZE (test-only): this task's own world size is already 2, the test size of
    the other tasks; 2 (or unset) runs the task as it is, anything else is refused."""
    value = os.environ.get('RESEARCHBOOST_TEST_WORLD_SIZE', '').strip()
    if value not in ('', str(config['world_size'])):
        raise RuntimeError('RESEARCHBOOST_TEST_WORLD_SIZE=%s: this task runs at world size %d only'
                           % (value, config['world_size']))


def gpus(config):
    import torch
    world = config['world_size']
    if torch.cuda.device_count() != world:
        raise RuntimeError('This task requires exactly %d allocated GPU(s), found %d' % (world, torch.cuda.device_count()))
    # Docker cannot select a GPU type, so the image's architecture (compute capability major) is enforced here.
    arch = int(os.environ['RESEARCHBOOST_GPU_ARCH'])
    props = [torch.cuda.get_device_properties(i) for i in range(world)]
    if any(p.major != arch for p in props):
        raise RuntimeError('This image requires GPUs of compute capability %d.x, found %s'
                           % (arch, sorted({(p.major, p.minor) for p in props})))
    return [p.name for p in props]


def trajectory_repeat(ref, again, world):
    """Seed 1 recorded twice. Gate 3 runs the production settings, which are not bitwise repeatable: the repeat's
    largest per-step difference of the per-token loss (nats) is logged and kept in task/traj_repeat.json (the run-to-run noise against the
    seed-to-seed sigma), not required to be zero."""
    worst = {}
    for rank in range(world):
        a = json.loads((WS / ref / ('traj_s1.r%d.json' % rank)).read_text())['steps']
        b = json.loads((again / ('traj_s1.r%d.json' % rank)).read_text())['steps']
        if len(a) != len(b):
            raise RuntimeError('Reference trajectory seed 1 repeat has %d steps on rank %d, expected %d'
                               % (len(b), rank, len(a)))
        diffs = [abs(x['losses']['token'] - y['losses']['token']) for x, y in zip(a, b)
                 if 'token' in (x.get('losses') or {}) and 'token' in (y.get('losses') or {})]
        worst[rank] = max(diffs) if diffs else 0.0
    (WS / 'task/traj_repeat.json').write_text(json.dumps({'max_abs_step_diff_per_rank': worst}, indent=1) + '\n')
    log('reference trajectory seed 1 repeat (production): largest per-step difference %s' % worst)


def ready(devices, mode, started, preparation=None, **extra):
    (WS / 'task/ready.json').write_text(json.dumps(dict(
        status='ready', mode=mode, gpus=devices, world_size=len(devices), test_world_size=None,
        preparation={'mode': preparation or mode, 'seconds': round(time.time() - started)},
        created=datetime.now().isoformat(timespec='seconds'), **extra), indent=1) + '\n')
    log('ready (%s preparation, %.0f min)' % (preparation or mode, (time.time() - started) / 60))


def formulation_skeleton(config):
    (WS / 'task/formulation.json').write_text(json.dumps(
        {'task': 'YOCO-3B pretraining speed (microsoft/unilm YOCO @ f4f7b27a)',
         'reference': {'path': config['reference'], 'commit': config['reference_commit']}, 'inputs': {}}) + '\n')


def loader_digest(config, ref):
    """task/loader_digest.json: the digest of the data the reference's loader yields on every shard."""
    digest = subprocess.run([PY, 'bench/bench.py', 'loader', '--tree', ref], cwd=WS, stdout=subprocess.PIPE,
                            text=True, check=True).stdout.strip().splitlines()[-1]
    loader = json.loads(digest)
    if sorted(loader) != [str(r) for r in range(config['world_size'])]:
        raise RuntimeError('Loader digest does not cover every shard: %s' % sorted(loader))
    (WS / 'task/loader_digest.json').write_text(json.dumps(loader, indent=1) + '\n')


def frozen_inputs():
    site = Path(subprocess.run([PY, '-c', 'import sysconfig; print(sysconfig.get_paths()["purelib"])'],
                               stdout=subprocess.PIPE, text=True, check=True).stdout.strip())
    return ['bench', 'numerics', 'task/loader_digest.json', str(ASSETS / 'data/Data'),
            str(ASSETS / 'tiktoken_cache')] + [str(site / f) for f in ENV_FILES]


def drop_pack(files, shipped):
    """Remove what a pack restored (it only adds preparation output to the shipped workspace), before the full
    preparation takes over."""
    overlap = sorted(rel for rel in files if rel in shipped)
    if overlap:
        raise RuntimeError('the pack replaced shipped workspace files %s; cannot fall back' % overlap[:5])
    for rel in files:
        (WS / rel).unlink(missing_ok=True)


def pack_preparation(root, meta, shipped, config, devices, started):
    """Preparation from the matching pack at root, with its pack.json meta (module docstring). True when the pack
    stands and preparation is done; False after removing the pack's files, when the unchanged tree's quick check here
    is not bitwise identical to them."""
    files = prep_pack.restore(root, WS)
    log('restored the preparation pack %s (%s, world size %d, recorded %s)'
        % (root, meta.get('gpu'), meta['world_size'], meta.get('created')))
    formulation_skeleton(config)
    loader_digest(config, config['reference'])
    run('bench/bench.py', 'freeze-inputs', *frozen_inputs())
    run('bench/bench.py', 'baseline', '--out', 'runs/bootstrap/baseline')
    run(str(PREP / 'mkref.py'), 'task/timing_ref.json', 'runs/bootstrap/baseline/baseline.json')
    try:
        quick = check(CANDIDATE, 'runs/bootstrap/quick', 'PASS', '--quick')
    except RuntimeError as exc:
        quick = {'error': str(exc)}
    if (quick.get('correctness') or {}).get('bitwise') is not True:
        log("the unchanged tree's quick check here is not bitwise identical to the pack's records (%s): full "
            'preparation' % (quick.get('error') or (quick.get('correctness') or {}).get('measured_tier')))
        drop_pack(files, shipped)
        return False
    log("the unchanged tree's quick check here is bitwise identical to the pack's records: the pack stands")
    check(CANDIDATE, 'runs/bootstrap/full', 'UNAVAILABLE', verdict='no_benefit')
    shutil.rmtree(SCRATCH, ignore_errors=True)
    shutil.rmtree(PREP)
    ready(devices, 'full', started, preparation='pack-match',
          pack={k: meta.get(k) for k in ('gpu', 'world_size', 'created', 'packages_sha256', 'note')})
    return True


def main():
    started = time.time()
    shipped = prep_pack.manifest(WS)               # the judge workspace as the image ships it
    check_scratch()
    config_path = WS / 'bench/bench.json'
    config = json.loads(config_path.read_text())
    test_world_size(config)
    devices = gpus(config)
    log('GPUs: %s' % devices)
    # The agent (unprivileged) and the verifier run the bench after this root preparation; the GPU lock must stay
    # openable for them.
    lock = Path(config['gpu_lock'])
    lock.touch()
    lock.chmod(0o666)
    smoke = os.environ.get('RESEARCHBOOST_SMOKE') == '1'
    # (pack directory, pack.json), or (None, why there is no usable pack)
    pack, found = (None, 'smoke preparation') if smoke else prep_pack.find(
        'yoco', config['world_size'], prep_pack.workspace_hash(shipped), PREP_VERSION)
    config['cpu_affinity'] = ','.join(str(cpu) for cpu in sorted(os.sched_getaffinity(0)))
    config_path.write_text(json.dumps(config, indent=1) + '\n')
    # task/ holds formulation.json, the loader digest, the timing reference and ready.json; the image has no such
    # directory (git keeps no empty directories).
    (WS / 'task').mkdir(exist_ok=True)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    ref = config['reference']
    if pack and pack_preparation(pack, found, shipped, config, devices, started):
        return
    if not pack:
        log('full preparation: %s' % found)
    formulation_skeleton(config)
    run('bench/bench.py', 'record', '--tree', ref, '--out', 'bench/records/ref')
    run('bench/bench.py', 'record', '--tree', ref, '--out', 'bench/records/ref_b')
    run('bench/bench.py', 'record', '--tree', ref, '--out', 'bench/records/var_nondet', '--variant', 'nondeterministic')
    run('-m', 'numerics', 'calibrate', '--ref', 'bench/records/ref', '--variants', 'bench/records/ref_b',
        'bench/records/var_nondet', '--out', 'bench/thresholds.json')
    run('-m', 'numerics', 'compare', '--ref', 'bench/records/ref', '--cand', 'bench/records/ref_b',
        '--thresholds', 'bench/thresholds.json', '--out', str(SCRATCH / 'repeat.json'))
    repeat = json.loads((SCRATCH / 'repeat.json').read_text())
    if not repeat.get('checked') or repeat['checked'] != repeat['bitwise_identical']:
        raise RuntimeError('Reference recordings are not bitwise identical: %s/%s'
                           % (repeat.get('bitwise_identical'), repeat.get('checked')))
    log('reference records: repeat %s/%s tensors bitwise identical' % (repeat['bitwise_identical'], repeat['checked']))
    loader_digest(config, ref)
    frozen = frozen_inputs()
    if smoke:
        # Integration smoke test: the records above, one real training run and the quick check. Trajectories, the
        # timing baseline, the full check, the self-test and the controls are skipped, so only quick checks are
        # possible and no speedup can be graded.
        run('bench/bench.py', 'freeze-inputs', *frozen)
        run('bench/bench.py', 'time', '--tree', ref, '--out', 'runs/bootstrap/smoke_time')
        timing = json.loads((WS / 'runs/bootstrap/smoke_time/time.json').read_text())
        if timing.get('status') != 'ok':
            raise RuntimeError('Smoke training run failed: %s %s' % (timing.get('status'), timing.get('reason')))
        check(CANDIDATE, 'runs/bootstrap/quick', 'PASS', '--quick')
        shutil.rmtree(SCRATCH, ignore_errors=True)
        shutil.rmtree(PREP)
        ready(devices, 'smoke', started)
        return
    run('bench/bench.py', 'record', '--tree', ref, '--out', 'bench/traj/ref', '--traj-seeds', TRAJ_SEEDS)
    run('bench/bench.py', 'record', '--tree', ref, '--out', str(SCRATCH / 'traj_s1b'), '--traj-seeds', '1')
    trajectory_repeat('bench/traj/ref', SCRATCH / 'traj_s1b', config['world_size'])
    run('-m', 'numerics', 'traj-freeze', '--ref', 'bench/traj/ref', '--spec', 'bench/traj/spec.json', '--k', '2',
        '--out', 'bench/traj/frozen.json')
    frozen_delta = json.loads((WS / 'bench/traj/frozen.json').read_text())
    log('trajectory endpoints (mean, sigma, delta): %s' % {
        n: (round(e['reference_mean'], 5), round(e['sigma_seed'], 6), round(e['delta'], 6))
        for n, e in frozen_delta['endpoints'].items()})
    run('bench/bench.py', 'freeze-inputs', *frozen)
    run('bench/bench.py', 'baseline', '--out', 'runs/bootstrap/baseline')
    run(str(PREP / 'mkref.py'), 'task/timing_ref.json', 'runs/bootstrap/baseline/baseline.json')
    check(CANDIDATE, 'runs/bootstrap/quick', 'PASS', '--quick')
    quick = json.loads((WS / 'runs/bootstrap/quick/check.json').read_text())
    if (quick.get('correctness') or {}).get('bitwise') is not True:
        raise RuntimeError('The unchanged tree\'s quick check is not bitwise identical to the reference records')
    run('-m', 'numerics', 'selftest', '--ref', 'bench/records/ref', '--variant', 'bench/records/ref_b',
        '--thresholds', 'bench/thresholds.json', '--out', 'runs/bootstrap/selftest.json')
    run(str(PREP / 'make_accept.py'), ref, str(SCRATCH / 'accept_trees'))
    for name, expected, mode, verdict in CONTROLS:
        check(str(SCRATCH / 'accept_trees' / name), 'runs/bootstrap/' + name, expected, mode, verdict=verdict)
    check(CANDIDATE, 'runs/bootstrap/full', 'UNAVAILABLE', verdict='no_benefit')
    # Remove control implementations and preparation helpers from the agent's view.
    shutil.rmtree(SCRATCH, ignore_errors=True)
    shutil.rmtree(PREP)
    ready(devices, 'full', started)


if __name__ == '__main__':
    main()
