"""Prepare machine-specific records before the agent starts; never runs during grading.

The judge's formulation sequence, run through the bench's own code paths: reference records twice (bitwise), a
legitimate variant for the diagnostic thresholds,
reference trajectories and the frozen delta, frozen inputs, the timing reference, the reference's own quick and full
checks, the numerics self-test and the acceptance controls. RESEARCHBOOST_SMOKE=1 keeps only the records, the frozen
inputs, one run of the real entrypoint and the reference quick check.

LLaVA's `bench.py check` exits 0 for every verdict it writes, so verdicts are read from check.json, not exit codes.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_pack  # noqa: E402  (next to this file in the image)

WS = Path('/opt/researchboost/ws')
PREP = Path('/opt/researchboost/prep')
SCRATCH = Path('/tmp/prep')
TREES = Path('/tmp/accept_trees')
PY = sys.executable
REFERENCE = 'targets/reference'
UPSTREAM_COMMIT = 'c121f0432da27facab705978f83c4ada465e46fd'
# First logged loss of the reference run the judge was formulated on (before any optimizer step, so independent of
# the machine's speed): a cross-check of assets, fixture and forward on this machine. Measured at this configuration
# (2 GPUs x 16, global 32) by the first preparation, 2026-10-27 on 2 x H200: 1.3839 in all three timing runs.
VALIDATED_FIRST_LOSS = 1.3839
TRAJ_SEEDS = [0, 1, 2, 3, 4, 5]
# Bump by hand whenever this file changes what preparation produces: preparation packs (prep_pack.py) recorded with
# another value are not used.
PREP_VERSION = 2
# The task's own GPU count: upstream's recipe (8 GPUs x 16 per device = global 128) with the per-device batch kept
# and the global batch scaled to the GPUs, 2 x 16 = 32 (bench.json). No other world size is supported.
DEFAULT_WORLD_SIZE = 2
STARTED = time.time()


def log(message):
    print('[bootstrap %s +%.0fs] %s' % (time.strftime('%H:%M:%S'), time.time() - STARTED, message), flush=True)


def run(*args):
    log('run: ' + ' '.join(str(a) for a in args))
    subprocess.run([PY, *[str(a) for a in args]], cwd=WS, check=True)


def bench():
    """The bench module itself, imported after bench.json is final (it reads the configuration at import)."""
    if 'researchboost_bench' not in sys.modules:
        sys.path.insert(0, str(WS))
        spec = importlib.util.spec_from_file_location('researchboost_bench', WS / 'bench/bench.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules['researchboost_bench'] = module
        spec.loader.exec_module(module)
    return sys.modules['researchboost_bench']


def verdict_problem(result, expected, stage=None, verdict=None, bitwise=None):
    correctness = result.get('correctness') or {}
    got = (result.get('status'), correctness.get('stage'), correctness.get('verdict'), correctness.get('bitwise'))
    if (result.get('status') != expected or (stage is not None and correctness.get('stage') != stage)
            or (verdict is not None and correctness.get('verdict') != verdict)
            or (bitwise is not None and correctness.get('bitwise') is not bitwise)):
        return 'got status/stage/verdict/bitwise %s, expected %s' % (got, (expected, stage, verdict, bitwise))
    return None


def check(tree, output, expected, *options, stage=None, verdict=None, bitwise=None, retry_invalid=False):
    """One bench check; returns check.json. An INVALID run (machine interference) is retried once when allowed."""
    options = options if '--quick' in options else (*options, '--final')   # gate 3 as the final evaluation runs it
    for attempt in range(2 if retry_invalid else 1):
        out = WS / output
        shutil.rmtree(out, ignore_errors=True)
        log('check %s -> %s %s (expect %s)' % (tree, output, ' '.join(options), expected))
        subprocess.run([PY, 'bench/bench.py', 'check', '--tree', str(tree), '--out', output, *options], cwd=WS,
                       check=True)
        result = json.loads((out / 'check.json').read_text())
        if result.get('status') == 'INVALID' and retry_invalid and attempt == 0:
            log('INVALID (%s); retrying once' % (result.get('correctness') or {}).get('note'))
            continue
        problem = verdict_problem(result, expected, stage, verdict, bitwise)
        if problem:
            raise RuntimeError('Preparation control failed: %s: %s' % (output, problem))
        log('check %s: %s' % (output, result.get('status')))
        return result


def gpus():
    code = ('import json, torch\n'
            'print(json.dumps([[torch.cuda.get_device_name(i), list(torch.cuda.get_device_capability(i))]'
            ' for i in range(torch.cuda.device_count())]))')
    return json.loads(subprocess.run([PY, '-c', code], check=True, capture_output=True, text=True).stdout)


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


def test_world_size(ws=WS):
    """The task runs on its own 2 GPUs; there is no other test world size. RESEARCHBOOST_TEST_WORLD_SIZE (set by
    a test-only override) may only name the task's own: returns None, the workspace unchanged."""
    value = os.environ.get('RESEARCHBOOST_TEST_WORLD_SIZE', '').strip()
    if value in ('', str(DEFAULT_WORLD_SIZE)):
        return None
    raise RuntimeError('RESEARCHBOOST_TEST_WORLD_SIZE=%s: this task runs on %d GPUs only'
                       % (value, DEFAULT_WORLD_SIZE))


def finalize_config(world, assets):
    """Absolute asset paths (the numerics adapter and the data probe read bench.json without expanding $ASSETS)."""
    path = WS / 'bench/bench.json'
    config = json.loads(path.read_text())
    config['assets'] = {k: v.replace('$ASSETS', assets) for k, v in config['assets'].items()}
    config['trajectory']['probe'] = config['trajectory']['probe'].replace('$ASSETS', assets)
    # Nsight Systems is not installed in this image; `bench.py profile --detail` (PyTorch profiler) works without it.
    config['nsys'] = shutil.which('nsys')
    config['numerics']['visible_gpus'] = world
    text = json.dumps(config, indent=1) + '\n'
    if '$' in text:
        raise RuntimeError('bench.json still names an unexpanded variable')
    path.write_text(text)
    for key, value in config['assets'].items():
        if not Path(value).exists():
            raise RuntimeError('Missing asset %s: %s' % (key, value))
    return config


def formulation(config, mode):
    manifest = json.loads((WS / 'task/fixture_manifest.json').read_text())
    freeze = '--freeze_backbone' in config['train_args']
    accum = config['grad_accum']
    data = {
        'task': config['task'],
        'mode': mode,
        'workload': {
            'upstream_recipe': 'haotian-liu/LLaVA@%s scripts/v1_5/finetune.sh: 7B instead of 13B, %d GPU(s) x '
                               'per-device %d x gradient accumulation %d = global %d, --report_to none, local paths%s'
                               % (UPSTREAM_COMMIT, config['world_size'], config['per_rank_batch'], accum,
                                  config['world_size'] * config['per_rank_batch'] * accum,
                                  ', LLM backbone frozen (--freeze_backbone True: projector and lm_head train)'
                                  if freeze else ''),
            'per_gpu_load': '16 samples per micro-batch, model_max_length 2048, bf16 + tf32, DeepSpeed ZeRO-3 '
                            '(scripts/zero3.json), flash-attention 2, gradient checkpointing',
            'sampler_note': 'LLaVATrainer groups megabatches by world_size * gradient_accumulation_steps = %d micro-'
                            'batches of %d = %d samples per optimizer step (upstream: 8 x 16 = 128 on 8 GPUs)'
                            % (config['world_size'] * accum, config['per_rank_batch'],
                               config['world_size'] * config['per_rank_batch'] * accum),
            'optimizer': 'AdamW lr 2e-5, cosine, warmup_ratio 0.03, weight decay 0, no gradient clipping',
        },
        'fixture': {
            'source': 'liuhaotian/LLaVA-Instruct-150K@9d451dc7629cfe0469f6ae4432b765cd603d5fcb '
                      'llava_v1_5_mix665k.json',
            'source_sha256': manifest['source_sha256'], 'seed': manifest['seed'], 'method': manifest['method'],
            'sets': {k: {x: v[x] for x in ('n', 'sha256', 'text_only')} for k, v in manifest['sets'].items()},
            'images': manifest['images'],
            'note': 'rebuilt by the asset service from the frozen source indices in task/fixture_manifest.json; '
                    'images read from pinned public zips by range requests (member CRC32 and directory digest checked)',
        },
        'frozen_inputs': {},
    }
    (WS / 'task/formulation.json').write_text(json.dumps(data, indent=1) + '\n')


def freeze_inputs(config, timing=False, trajectories=True):
    """task/formulation.json frozen_inputs with the bench's own content_digest; absolute asset paths."""
    m = bench()
    assets = dict(config['assets'])
    items = {'vicuna': assets['vicuna'], 'clip': assets['clip'], 'projector': str(Path(assets['projector']).parent),
             'images': assets['images'], 'fixture': assets['fixture'], 'thresholds': 'bench/thresholds.json'}
    if trajectories:
        items['traj_frozen'] = 'bench/traj/frozen.json'
    items['records_ref'] = 'bench/records/ref'
    if timing:
        items['timing_reference'] = 'task/timing/reference.json'
    items.update({'cases': 'bench/cases.json', 'traj_spec': 'bench/traj/spec.json',
                  'loader_reference': 'task/loader_reference.json',
                  'partition_sensitivity': 'task/partition_sensitivity.json'})
    path = WS / 'task/formulation.json'
    data = json.loads(path.read_text())
    data['frozen_inputs'] = {}
    for name, rel in items.items():
        target = Path(rel) if Path(rel).is_absolute() else WS / rel
        data['frozen_inputs'][name] = {'path': rel, **m.content_digest(target)}
    path.write_text(json.dumps(data, indent=1) + '\n')
    log('froze %d inputs' % len(items))


def loader_reference(cross):
    """The reference's own data probe on this image; compared with the probe of the formulation's reference run."""
    out = subprocess.run([PY, 'bench/loader_probe.py', '--tree', REFERENCE, '--batches',
                          str(json.loads((WS / 'bench/bench.json').read_text())['coverage']['probe_batches'])],
                         cwd=WS, check=True, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=str(WS))).stdout
    got = json.loads(out.strip().splitlines()[-1])
    (WS / 'task/loader_reference.json').write_text(out.strip().splitlines()[-1] + '\n')
    want = json.loads((PREP / 'loader_reference.validated.json').read_text())
    config = json.loads((WS / 'bench/bench.json').read_text())
    per_step = config['world_size'] * config['grad_accum']
    validated = len(want['batches']) // max(1, len({b['step'] for b in want['batches']}))
    if validated != per_step:
        # The validated probe belongs to another recipe configuration (micro-batches per optimizer step): the first
        # preparation at this one records its probe (task/loader_reference.json), which then becomes the validated one.
        cross['loader_vs_validated'] = ('not validated: the validated probe has %d micro-batches per optimizer step, '
                                        'this configuration %d' % (validated, per_step))
        cross['warnings'].append(cross['loader_vs_validated'])
        log('loader reference: %s' % cross['loader_vs_validated'])
        return
    problem = bench().loader_problem(want, got)
    if problem and set(problem[1]) - {'samples'}:
        raise RuntimeError('The data pipeline differs from the validated probe beyond the sample digests '
                           '(wrong fixture or tokenizer?): %s %s' % problem)
    cross['loader_vs_validated'] = 'identical' if not problem else 'sample digests differ: %s' % problem[0]
    log('loader reference vs validated probe: %s' % cross['loader_vs_validated'])


def records(cross):
    m = bench()
    logs = WS / 'runs/bootstrap/records'
    logs.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(WS / 'bench/records/ref', ignore_errors=True)
    # Absolute output paths: the numerics adapter runs the entrypoint from inside the tree.
    for rec, label, variant in ((str(WS / 'bench/records/ref'), 'ref', None), (str(SCRATCH / 'rec_dup'), 'dup', None),
                                (str(SCRATCH / 'rec_sdpa'), 'variant', 'sdpa')):
        log('record %s' % label)
        r = m.record_cases(REFERENCE, logs, {}, rec, label=label, variant=variant)
        if r['status'] != 'ok':
            raise RuntimeError('Reference record %s failed: %s' % (label, r.get('reason')))
    run(PREP / 'cmp_records.py', 'bench/records/ref', SCRATCH / 'rec_dup')
    run('-m', 'numerics', 'calibrate', '--ref', 'bench/records/ref', '--variants', SCRATCH / 'rec_sdpa',
        '--out', 'bench/thresholds.json')
    subprocess.run([PY, '-m', 'numerics', 'compare', '--ref', 'bench/records/ref', '--cand', str(SCRATCH / 'rec_dup'),
                    '--thresholds', 'bench/thresholds.json', '--out', 'task/repeat-check.json'], cwd=WS)
    repeat = json.loads((WS / 'task/repeat-check.json').read_text())
    if not repeat.get('checked') or repeat['checked'] != repeat['bitwise_identical']:
        raise RuntimeError('Reference recordings are not bitwise identical: %s' % (
            {k: repeat.get(k) for k in ('status', 'checked', 'bitwise_identical')},))
    cross['repeat_records'] = 'bitwise identical (%d tensors)' % repeat['checked']
    trainable_check(json.loads((WS / 'bench/bench.json').read_text()))


def trainable_check(config):
    """The recorded gradients are exactly bench.json's trainable parameters, all nonzero in the image-text case."""
    expected = {name for name, _ in config['numerics']['trainable']}
    for manifest in sorted((WS / 'bench/records/ref').glob('*.r*.json')):
        data = json.loads(manifest.read_text())
        for step in data['steps']:
            grads = {n: d for n, d in step['tensors'].items() if n.startswith('grad/')}
            names = {n.split('/', 2)[2].split('.', 1)[1] for n in grads}
            if names != expected:
                raise RuntimeError('%s: recorded gradients %d, trainable %d, e.g. %s' % (
                    manifest.name, len(names), len(expected), sorted(names ^ expected)[:5]))
            if data['case']['name'] == 'mm':
                zero = [n for n, d in grads.items() if not d['finite'] or not d['norm']]
                if zero:
                    raise RuntimeError('%s: zero or non-finite reference gradients %s' % (manifest.name, zero[:5]))


def trajectories(seeds=TRAJ_SEEDS, frozen='bench/traj/frozen.json'):
    m = bench()
    logs = WS / 'runs/bootstrap/traj'
    logs.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(WS / 'bench/traj/ref', ignore_errors=True)
    env = m.base_env({'PYTHONPATH': str(WS)})
    for seed in seeds:
        log('reference trajectory seed %d' % seed)
        r = m.trajectory_seed(REFERENCE, WS / 'bench/traj/ref', seed, env, logs / ('traj_s%d.log' % seed))
        if r['status'] != 'ok' or r.get('returncode'):
            raise RuntimeError('Reference trajectory seed %d failed: %s' % (seed, r.get('reason') or r))
    run('-m', 'numerics', 'traj-freeze', '--ref', 'bench/traj/ref', '--spec', 'bench/traj/spec.json', '--k', '2',
        '--out', frozen)
    return json.loads((WS / frozen).read_text())


def compare_records(ref, cand, out):
    """The judge's comparison of two record sets; bitwise when every tensor is identical."""
    subprocess.run([PY, '-m', 'numerics', 'compare', '--ref', str(ref), '--cand', str(cand),
                    '--thresholds', 'bench/thresholds.json', '--out', str(out)], cwd=WS)
    report = json.loads((WS / out).read_text())
    bitwise = bool(report.get('checked')) and report['checked'] == report.get('bitwise_identical')
    return {'bitwise': bitwise, 'summary': '%s/%s tensors bitwise identical'
            % (report.get('bitwise_identical'), report.get('checked'))}


def restore_pack(root, meta):
    """Restore a matching preparation pack (prep_pack.py) into the workspace; returns its preparation's cross-checks."""
    prep_pack.restore(root, WS)
    log('restored the preparation pack %s (%s, world size %d, recorded %s)'
        % (root, meta.get('gpu'), meta['world_size'], meta.get('created')))
    return json.loads((WS / 'task/ready.json').read_text()).get('cross_checks')


def trust_pack():
    """RESEARCHBOOST_PREP_TRUST_PACK=1 (test-only): with a matching pack, skip
    recording the reference twice and comparing it with the pack ('pack-trusted' preparation)."""
    return os.environ.get('RESEARCHBOOST_PREP_TRUST_PACK') == '1'


def pack_preparation(pack, cross):
    """Startup with a restored preparation pack: record this machine's reference twice. When the records are bitwise identical to the pack's, its thresholds, trajectories, delta and control
    evidence stand ('pack-match'); otherwise this machine's records, thresholds and reference trajectories (the
    candidate seed schedule) are recorded and the pack's delta is kept ('pack-local-trajectories'). The timing
    baseline, the quick and full checks and, for pack-local-trajectories, the controls follow in main()."""
    root, meta, pack_cross = pack
    cross['pack'] = {'gpu': meta.get('gpu'), 'world_size': meta['world_size'], 'created': meta.get('created'),
                     'packages_sha256': meta.get('packages_sha256'), 'cross_checks': pack_cross}
    if trust_pack():
        # Test-only (RESEARCHBOOST_PREP_TRUST_PACK=1, a machine type already seen at pack-match): the records are not
        # made twice and compared; the reference's quick check in main() must still be bitwise identical to the
        # pack's records, or preparation fails.
        log('records not re-recorded (RESEARCHBOOST_PREP_TRUST_PACK=1); the quick check must match the pack bitwise')
        cross['pack']['records_vs_pack'] = 'not re-recorded (RESEARCHBOOST_PREP_TRUST_PACK=1)'
        return 'pack-trusted'
    m = bench()
    logs = WS / 'runs/bootstrap/records'
    logs.mkdir(parents=True, exist_ok=True)
    for name in ('local_ref', 'local_dup'):
        log('record %s' % name)
        r = m.record_cases(REFERENCE, logs, {}, str(SCRATCH / name), label=name)
        if r['status'] != 'ok':
            raise RuntimeError('Reference record %s failed: %s' % (name, r.get('reason')))
    repeat = compare_records(SCRATCH / 'local_ref', SCRATCH / 'local_dup', SCRATCH / 'local-repeat.json')
    if not repeat['bitwise']:
        raise RuntimeError('Reference recordings are not bitwise identical: %s' % repeat['summary'])
    against = compare_records('bench/records/ref', SCRATCH / 'local_ref', SCRATCH / 'records-vs-pack.json')
    log('local reference records: repeat %s; against the pack: %s' % (repeat['summary'], against['summary']))
    cross['repeat_records'] = 'bitwise identical (%s)' % repeat['summary']
    cross['pack']['records_vs_pack'] = against['summary']
    if against['bitwise']:
        return 'pack-match'
    records(cross)
    pack_frozen = json.loads((WS / 'bench/traj/frozen.json').read_text())
    # The candidate seeds of gate 3 (trajectory.seeds) need local reference values; traj-freeze needs at least three.
    n = max(3, json.loads((WS / 'bench/bench.json').read_text())['trajectory']['seeds'])
    local = trajectories(pack_frozen['reference_seeds'][:n], str(SCRATCH / 'local-frozen.json'))
    frozen = prep_pack.rebase_frozen(pack_frozen, local)
    (WS / 'bench/traj/frozen.json').write_text(json.dumps(frozen, indent=1) + '\n')
    cross['trajectory_delta_source'] = 'preparation pack (%d reference seeds); reference values local (%d seeds)' % (
        len(pack_frozen['reference_seeds']), len(local['reference_seeds']))
    return 'pack-local-trajectories'


def run_controls():
    run(PREP / 'make_accept.py', REFERENCE, TREES)
    check(TREES / 'setup_train', 'runs/bootstrap/setup_train', 'ERROR', '--quick', stage='setup_guard')
    check(TREES / 'sampler_no_accum', 'runs/bootstrap/sampler_no_accum', 'INCORRECT', '--quick',
          stage='loader_check')
    detached = check(TREES / 'projector0_detached', 'runs/bootstrap/projector0_detached', 'INCORRECT', '--quick',
                     stage='structure')
    dead = {p for f in detached['correctness']['numerics']['exact_failures'] if f.get('check') == 'dead_gradient'
            for p in f['parameters']}
    want = {'grad/adamw#0/llava.model.mm_projector.0.weight', 'grad/adamw#0/llava.model.mm_projector.0.bias'}
    if dead != want:
        raise RuntimeError('projector0_detached: dead gradients %s, expected %s' % (sorted(dead), sorted(want)))
    check(TREES / 'rmsnorm_div_sqrt', 'runs/bootstrap/rmsnorm_div_sqrt', 'PASS', '--skip-timing',
          stage='statistical', verdict='equivalent')
    check(TREES / 'soft_labels', 'runs/bootstrap/soft_labels', 'INCORRECT', '--skip-timing', stage='statistical',
          verdict='different')
    # Remove control implementations from the agent's view.
    shutil.rmtree(TREES)


def timing(label, cross, freeze=False):
    for attempt in range(2):
        run('bench/bench.py', 'time', '--tree', REFERENCE, '--out', 'task/timing', '--label', label,
            *(['--freeze'] if freeze else []))
        result = json.loads((WS / ('task/timing/time_%s.json' % label)).read_text())
        if result['status'] == 'INVALID' and attempt == 0:
            log('timing %s INVALID (%s); retrying once' % (label, result.get('problems') or result.get('reason')))
            continue
        if result['status'] != 'ok':
            raise RuntimeError('Reference timing %s: %s %s' % (label, result['status'],
                                                                result.get('problems') or result.get('reason')))
        break
    first = result['logged_loss'].get('1')
    if VALIDATED_FIRST_LOSS is None:
        cross['first_logged_loss_%s' % label] = {'value': first, 'validated': None}
        log('timing %s: %.1f ms/step, peak %.1f GiB, first logged loss %s (no validated value yet: recorded)' % (
            label, result['ms_per_step'], result['peak_memory_allocated_gib'], first))
        return result
    deviation = abs(first - VALIDATED_FIRST_LOSS) / VALIDATED_FIRST_LOSS
    cross['first_logged_loss_%s' % label] = {'value': first, 'validated': VALIDATED_FIRST_LOSS,
                                             'rel_diff': round(deviation, 5)}
    log('timing %s: %.1f ms/step, peak %.1f GiB, first logged loss %s (validated %s, rel diff %.4f)' % (
        label, result['ms_per_step'], result['peak_memory_allocated_gib'], first, VALIDATED_FIRST_LOSS, deviation))
    if deviation > 0.05:
        raise RuntimeError('The first logged loss is %.1f%% off the validated run: assets or forward differ'
                           % (100 * deviation))
    if deviation > 0.01:
        cross['warnings'].append('first logged loss %s is %.2f%% off the validated %s'
                                 % (first, 100 * deviation, VALIDATED_FIRST_LOSS))
    return result


def ready(devices, mode, cross, test_world=None, preparation=None):
    (WS / 'task/ready.json').write_text(json.dumps(dict(
        status='ready', mode=mode, gpus=[name for name, _ in devices], world_size=len(devices),
        test_world_size=test_world, cross_checks=cross, minutes=round((time.time() - STARTED) / 60, 1),
        preparation={'mode': preparation or mode, 'seconds': round(time.time() - STARTED)}), indent=1) + '\n')
    log('ready (%s)' % mode)


def main():
    shipped = prep_pack.manifest(WS)               # the judge workspace as the image ships it
    test_world = test_world_size()
    check_scratch()
    config = json.loads((WS / 'bench/bench.json').read_text())
    world = config['world_size']
    if test_world:
        log('test configuration: world size %d, gradient accumulation %d' % (world, config['grad_accum']))
    devices = gpus()
    if len(devices) != world:
        raise RuntimeError('This task requires exactly %d allocated GPU(s), found %d' % (world, len(devices)))
    # Docker cannot select a GPU type, so the image's architecture (compute capability major) is enforced here.
    arch = int(os.environ['RESEARCHBOOST_GPU_ARCH'])
    if any(cc[0] != arch for _, cc in devices):
        raise RuntimeError('This image requires GPUs of compute capability %d.x, found %s' % (arch, devices))
    log('GPUs: %s' % devices)
    # The agent (unprivileged) and the verifier run the bench after this root preparation; the bench's GPU lock
    # (a fixed path in bench.py) must stay openable for them.
    lock = Path('/tmp/llava_gpu.lock')
    lock.touch()
    lock.chmod(0o666)
    smoke = os.environ.get('RESEARCHBOOST_SMOKE') == '1'
    mode = 'smoke' if smoke else 'full'
    head = subprocess.check_output(['git', '-C', str(WS / REFERENCE), 'rev-parse', 'HEAD'], text=True).strip()
    if head != config['reference_commit'] or head != UPSTREAM_COMMIT:
        raise RuntimeError('Reference checkout at %s, expected %s' % (head, UPSTREAM_COMMIT))
    config = finalize_config(world, os.environ['ASSETS'])
    pack, why = (None, 'smoke preparation') if smoke else prep_pack.find(
        'llava', world, prep_pack.workspace_hash(shipped), PREP_VERSION)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cross = {'warnings': []}
    preparation = mode
    pack_cross = None
    if pack:
        pack_cross = restore_pack(pack, why)
    else:
        log('full preparation: %s' % why)
    formulation(config, mode)
    loader_reference(cross)
    if pack:
        preparation = pack_preparation((pack, why, pack_cross), cross)
    else:
        records(cross)
    if smoke:
        # Integration smoke test: the records above, frozen inputs, one run of the real entrypoint and the quick
        # check. Trajectories, the timing baseline, the full check, the self-test and the controls are skipped, so
        # only quick checks are possible and no speedup can be graded.
        freeze_inputs(config, trajectories=False)
        timing('ref_a', cross, freeze=True)
        check(REFERENCE, 'runs/bootstrap/quick', 'PASS', '--quick', bitwise=True)
        shutil.rmtree(SCRATCH)
        shutil.rmtree(PREP)
        ready(devices, mode, cross, test_world)
        return
    if not pack:
        trajectories()
    freeze_inputs(config)
    # This machine's timing baseline: timing.reference_runs runs of the reference (the first frozen provisionally, so
    # that the others are checked against it), then their mean and spread frozen; then the reference's own quick and
    # full checks, in every mode.
    labels = ['ref_%s' % chr(ord('a') + i) for i in range(config['timing'].get('reference_runs', 2))]
    timing(labels[0], cross, freeze=True)
    for label in labels[1:]:
        timing(label, cross)
    run('bench/bench.py', 'freeze-timing', '--out', 'task/timing', '--labels', *labels)
    frozen = json.loads((WS / 'task/timing/reference.json').read_text())
    cross['reference_timing'] = {k: frozen.get(k) for k in ('ms_per_step', 'spread', 'sigma_rel')}
    freeze_inputs(config, timing=True)
    check(REFERENCE, 'runs/bootstrap/quick', 'PASS', '--quick', bitwise=True)
    full = check(REFERENCE, 'runs/bootstrap/full', 'UNAVAILABLE', verdict='no_benefit', retry_invalid=True)
    cross['reference_full_check'] = {k: (full.get('performance') or {}).get('benefit', {}).get(k)
                                     for k in ('speedup', 'speedup_bar')}
    if preparation not in ('pack-match', 'pack-trusted'):
        # full, pack-local-trajectories: the self-test and the acceptance controls on this machine; with a matching
        # pack its own self-test and control evidence stand.
        run('-m', 'numerics', 'selftest', '--ref', 'bench/records/ref', '--variant', SCRATCH / 'rec_sdpa',
            '--thresholds', 'bench/thresholds.json', '--out', 'task/selftest.json')
        run_controls()
    # Remove preparation helpers from the agent's view.
    shutil.rmtree(SCRATCH)
    shutil.rmtree(PREP)
    ready(devices, mode, cross, test_world, preparation)


if __name__ == '__main__':
    main()
