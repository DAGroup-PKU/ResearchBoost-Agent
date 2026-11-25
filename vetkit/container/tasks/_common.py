"""Steps shared by the tasks whose bench has `record` + `compare` (exact record identity) and `time` (one timing run
with a frozen-reference field) on the command line: nanochat, r3, bagel_sft and tasks added through the generic module (tasks/_generic.py)."""
import json
from pathlib import Path
import shutil


def simple_prepare(ctx, dirs, test_world_size=False, check_assets=False):
    """What every bootstrap of these tasks does first: GPU checks (bootstrap.gpus(config), or bootstrap.gpu() of a
    1-GPU task), the GPU lock openable for the agent user, the workspace directories, a fresh scratch directory (when
    the bootstrap has one)."""
    b = ctx.b
    if test_world_size:
        b.test_world_size()
    config = json.loads((ctx.ws / 'bench/bench.json').read_text())
    ctx.config = config
    if hasattr(b, 'gpus'):
        ctx.devices = b.gpus(config)
    elif hasattr(b, 'gpu'):       # a 1-GPU task's bootstrap: gpu() checks the GPU, returns its name
        ctx.devices = [b.gpu()]
    else:
        raise SystemExit("the task's bootstrap.py has neither gpus(config) nor gpu()")
    lock = Path(config['gpu_lock'])
    lock.touch()
    lock.chmod(0o666)
    if check_assets:
        b.check_assets(config)
    for d in dirs:
        (ctx.ws / d).mkdir(parents=True, exist_ok=True)
    scratch = getattr(b, 'SCRATCH', None)     # not every bootstrap has a scratch directory of its own
    if scratch is not None:
        shutil.rmtree(str(scratch), ignore_errors=True)
        scratch.mkdir(parents=True)
    ctx.step('prepare', gpus=ctx.devices)


def record_and_compare(ctx, extra_check=None):
    """bench.py record of the reference once, then bench.py compare against the pack's bench record (exit 0 = the
    judge's identity: exact items and every digest)."""
    from svcpack import Mismatch
    out = ctx.out / 'record'
    shutil.rmtree(str(out), ignore_errors=True)
    r = ctx.run('bench/bench.py', 'record', '--tree', ctx.config['reference'], '--out', str(out), '--label', 'ref',
                check=False)
    if r.returncode != 0:
        raise Mismatch('the reference record failed here (exit %d)' % r.returncode)
    if extra_check:
        extra_check(json.loads((out / 'record.json').read_text()))
    ref = ctx.config['record']['reference']
    same = ctx.run('bench/bench.py', 'compare', ref, str(out / 'record.json'), check=False)
    ctx.step('record', identical=same.returncode == 0, compared=ref)
    if same.returncode != 0:
        raise Mismatch("the reference's record here differs from the pack's (%s)" % ref)


def time_once(ctx, field, frozen_value, band, tree=None):
    """bench.py time of the reference once (one retry when INVALID/TIMEOUT: machine interference), then the band.
    tree: what --tree gets (default bench.json's reference)."""
    from svcpack import band_check, Mismatch
    for attempt in range(2):
        out = ctx.out / ('time%d' % attempt)
        shutil.rmtree(str(out), ignore_errors=True)
        ctx.run('bench/bench.py', 'time', '--tree', str(tree or ctx.config['reference']), '--out', str(out), check=False)
        try:
            res = json.loads((out / 'time.json').read_text())
        except (OSError, ValueError):
            res = {'status': 'ERROR', 'reason': 'no time.json'}
        if res.get('status') == 'ok':
            return band_check(ctx, field, res[field], frozen_value, band)
        if res.get('status') not in ('INVALID', 'TIMEOUT') or attempt:
            raise Mismatch('the reference timing run failed here: %s %s' % (res.get('status'), res.get('reason')))
        ctx.step('timing_retry', status=res.get('status'), reason=res.get('reason'))


def simple_finish(ctx):
    from svcpack import write_ready
    if getattr(ctx.b, 'SCRATCH', None) is not None:
        shutil.rmtree(str(ctx.b.SCRATCH), ignore_errors=True)
    write_ready(ctx, ctx.devices)
