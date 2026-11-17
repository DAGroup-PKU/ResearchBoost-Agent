"""Prepare machine-specific references before the agent starts; never runs during grading.

Runs as root on the allocated GPUs (4), following the judge's acceptance procedure (bench/bench.py, formulator
commands):
  1. the rollout fixture: one reference rollout-only run with natural lengths (fixture.num_rollouts rollouts, the
     teacher online) -> /opt/researchboost/prep/fixture (rollout dumps with the teacher's answers: the train and
     teacher sides' frozen inputs; every sample's length: the forced lengths of rollout and full-loop runs) and the
     reference's student engine and teacher server layouts (task/layout_ref.json);
  2. the reference rollout with forced lengths (rollout timing run A), scored through the stock student engine (the
     token forcing baseline and tau), its teacher log-probabilities against the stock teacher replayed on the same
     tokens (the rollout's teacher baseline), and the train-inference mismatch of the reference Megatron on it
     (task/rollout_ref.json);
  3. the stock teacher replayed on fixture rollouts 0..3: twice at full concurrency (the frozen values and the
     teacher timing) and once per re-batching of teacher_rules.rebatchings (requests one at a time, chunked prefill
     8192, no overlap scheduling, no radix cache; pooled: the stock teacher's own spread) ->
     task/teacher_ref.json, task/teacher_ref.npz;
  4. the reference train records: deterministic twice, once non-deterministic and twice more (the run-to-run noise),
     and one record on each other fixture set (the data-set sigma) -> task/train_ref.json;
  5. the frozen input fingerprints (judge code, fixture, student and teacher checkpoints, Megatron checkpoint, data),
     every one readable by the agent user;
  6. the unchanged tree's quick check (all three sides, no timing: PASS);
  7. the reference side timings (rollout: run A of step 2 and one more; teacher: the two runs of step 3; train: two
     runs; a pair more than 3% apart is measured again, up to 3 pairs, the first pair within 3% frozen, else the
     last with timing_spread_flag in task/ready.json) and the reference's own full check (`check --final`, as root: UNAVAILABLE, no benefit), whose full-loop
     windows give the step fractions (the teacher's: its share on the critical path) -> task/timing_ref.json;
  8. acceptance controls (RESEARCHBOOST_SLIME_OPD_CONTROLS, default 1): a changed entry point -> ERROR; a logit bias
     in sglang (student and teacher) -> INCORRECT (student token forcing); teacher log-probabilities shifted by +0.05
     in the post-processing -> INCORRECT (teacher bias rule); teacher log-probabilities read one token off ->
     INCORRECT (teacher rate rule); the teacher with chunked prefill 8192 (another batching of the same arithmetic) ->
     PASS N1; the teacher at TP2 -> INCORRECT (layout); student engines of 1 GPU -> INCORRECT (layout); a flipped
     loss sign -> INCORRECT (statistical gate); a reduction in reversed order -> PASS N1; the reference with lr x 1.02
     -> INCORRECT (statistical gate alone); `profile --side train` and `--side teacher` on the reference write traces.
Any failure stops the container.

With a preparation pack ($ASSETS/pack, made by `bootstrap.py --make-pack DEST` from an earlier full preparation on
the same GPU model with the same judge: the fixture, task/layout_ref.json, rollout_ref.json, teacher_ref.json,
teacher_ref.npz, train_ref.json, train_ref_sketch.npy and the controls' evidence, a sha256 manifest), steps 1-4 restore
the pack instead (root-owned) when its key matches (task, world size, GPU model, judge workspace hash, PREP_VERSION,
installed-packages hash) and every file matches the manifest; the quick check (step 6) must then PASS against the
pack's references, else the pack is discarded and the full preparation runs. The timing references (step 7: fresh
rollout, teacher and train pairs, and the reference's own full check) are always measured on this machine; the
controls are skipped (their evidence is in the pack). There is no compile cache to restore: every judge run starts
with empty caches (a timed full loop: its own priming run's, bench.json timing.full.prime).

RESEARCHBOOST_SMOKE=1 (integration tests): a fixture of fixture.smoke_num_rollouts rollouts, steps 2, 3, 4 (two
deterministic records only), 5 and the quick check; no timing, no statistical gate, no controls: only quick checks
can be graded.
"""
from datetime import datetime
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

WS = Path('/opt/researchboost/ws')
PREP = Path('/opt/researchboost/prep')
ASSETS = Path(os.environ.get('ASSETS', '/opt/researchboost/assets'))
CANDIDATE = '/submission/slime_opd'
PY = sys.executable
SCRATCH = Path('/tmp/slime_opd-prep')
PACK = ASSETS / 'pack'
PREP_VERSION = 1
TIMING_SPREAD_MAX = 0.03   # a pair of reference timing runs further apart is measured again
TIMING_PAIRS_MAX = 3
TIMING_SIDES = ('rollout', 'teacher', 'train')
TIMING_CRASH_RETRIES = 2   # a pair whose reference run crashed is measured again, at most this often
PACK_TASK_FILES = ('layout_ref.json', 'rollout_ref.json', 'teacher_ref.json', 'teacher_ref.npz', 'train_ref.json',
                   'train_ref_sketch.npy')
MIN_GPU_MEMORY_GIB = 130
# Host-local cache of the measured reference timings (task/timing_ref.json). The timings depend on this machine and
# this judge, not on the candidate, so a fresh entry is reused instead of measured again; 'off' disables it.
TIMING_CACHE = Path(os.environ.get('RBB_TIMING_CACHE_DIR', '/opt/researchboost/timing-cache'))
TIMING_CACHE_MAX_AGE_H = float(os.environ.get('RBB_TIMING_CACHE_MAX_AGE_H', '48'))


def log(message):
    print('[bootstrap %s] %s' % (datetime.now().strftime('%H:%M:%S'), message), flush=True)


PHASES = {}


def run(*args, phase=None):
    log('run: %s' % ' '.join(str(a) for a in args))
    t0 = time.time()
    subprocess.run([PY, 'bench/bench.py', *args], cwd=WS, check=True)
    if phase:
        PHASES[phase] = round(time.time() - t0)
        log('%s: %.0f min' % (phase, PHASES[phase] / 60))


def failure_detail(output):
    data = json.loads((WS / output / 'check.json').read_text())
    keep = {'status': data.get('status'), 'note': data.get('note'),
            'correctness': {k: v for k, v in (data.get('correctness') or {}).items()
                            if k not in ('rollout', 'teacher', 'train')},
            'performance_reason': (data.get('performance') or {}).get('reason')}
    for s in ('rollout', 'teacher', 'train'):
        r = (data.get('correctness') or {}).get(s)
        if r:
            keep[s] = {k: r.get(k) for k in ('status', 'stage', 'problems', 'reason', 'exact')}
    return 'check.json: ' + json.dumps(keep, default=str)[:6000]


def check(tree, output, expected, *options, verdict=None, tier=None, stage=None):
    # Every non-quick check as the final evaluation runs it (root: final_allowed).
    options = options if ('--quick' in options or '--skip-timing' in options) else (*options, '--final')
    log('check %s -> %s %s (expect %s%s%s)' % (tree, output, ' '.join(options), expected,
                                                ', %s' % verdict if verdict else '', ', stage %s' % stage if stage else ''))
    t0 = time.time()
    result = subprocess.run([PY, 'bench/bench.py', 'check', '--tree', tree, '--out', output, *options], cwd=WS)
    data = json.loads((WS / output / 'check.json').read_text())
    corr = data.get('correctness') or {}
    got = (data.get('status'), corr.get('verdict'), corr.get('measured_tier'), corr.get('stage'))
    sub = corr.get(corr.get('stage')) if isinstance(corr.get(corr.get('stage')), dict) else {}
    substage = (sub or {}).get('stage')
    if got[0] != expected or result.returncode != (0 if expected == 'PASS' else 1) or \
            (verdict is not None and got[1] != verdict) or (tier is not None and got[2] != tier) or \
            (stage is not None and stage not in (got[3], substage)):
        log(failure_detail(output))
        raise RuntimeError('Preparation control failed: %s: got status/verdict/tier/stage %s (%s), expected %s'
                           % (output, got, substage, (expected, verdict, tier, stage)))
    PHASES[output] = round(time.time() - t0)
    log('check %s: %s %s %s %s/%s (%.0f min)' % (output, *got, substage, PHASES[output] / 60))
    return data


def reference_final(out, tries=3):
    """The reference's own --final (must be UNAVAILABLE/no_benefit). INVALID (the two reference runs moved apart) and
    ERROR (a reference run crashed) say this measurement is unusable, not that the host cannot judge: measure again, up
    to `tries` times. Each failed attempt is kept whole as <out>/full_attempt_<k>, run logs included."""
    for k in range(1, tries + 1):
        try:
            return check(CANDIDATE, out + '/full', 'UNAVAILABLE', verdict='no_benefit')
        except RuntimeError:
            status = json.loads((WS / out / 'full/check.json').read_text()).get('status')
            if status not in ('INVALID', 'ERROR') or k == tries:
                raise
            shutil.move(str(WS / out / 'full'), str(WS / out / ('full_attempt_%d' % k)))
            log('reference --final attempt %d: %s, measuring again' % (k, status))


def gpus(config):
    import torch
    world = config['world_size']
    if torch.cuda.device_count() != world:
        raise RuntimeError('This task requires exactly %d allocated GPUs, found %d' % (world, torch.cuda.device_count()))
    props = [torch.cuda.get_device_properties(i) for i in range(world)]
    if any(p.major != 9 for p in props):
        raise RuntimeError('This image requires Hopper GPUs (compute capability 9.x), found %s'
                           % sorted({(p.major, p.minor) for p in props}))
    small = [p.name for p in props if p.total_memory / 2**30 < MIN_GPU_MEMORY_GIB]
    if small:
        raise RuntimeError('This task needs at least %d GiB per GPU, found %s' % (MIN_GPU_MEMORY_GIB, small))
    return [p.name for p in props]


def ready(devices, mode, started, **extra):
    (WS / 'task/ready.json').write_text(json.dumps(dict(
        status='ready', mode=mode, gpus=devices, world_size=len(devices),
        preparation={'mode': mode, 'seconds': round(time.time() - started), 'phases_s': PHASES},
        created=datetime.now().isoformat(timespec='seconds'), **extra), indent=1) + '\n')
    log('ready (%s preparation, %.0f min)' % (mode, (time.time() - started) / 60))


def control_tree(name, edits):
    """A copy of the unchanged candidate tree (three repositories, git, branch opt) with `edits`
    [(repo dir, relative path, old, new)] committed."""
    tree = SCRATCH / 'accept' / name
    shutil.copytree(CANDIDATE, tree, symlinks=True)
    for repo, rel, old, new in edits:
        p = tree / repo / rel
        src = p.read_text()
        if src.count(old) != 1:
            raise RuntimeError('acceptance control %s: the text to change in %s/%s was not found once' % (name, repo, rel))
        p.write_text(src.replace(old, new))
        subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(tree / repo), '-c', 'user.name=acceptance', '-c',
                        'user.email=acceptance@localhost', 'commit', '-qam', 'acceptance control ' + name], check=True)
    return str(tree)


# The controls' edits (exact text of the reference trees).
ENTRY = ('slime', 'train.py', '    main(args, restore_plan)\n', '    main(args, restore_plan)\n    pass\n')
LOGIT_BIAS = ('sglang', 'python/sglang/srt/layers/logits_processor.py',
              '                    logits / self.final_logit_softcapping\n                )\n\n        return logits\n',
              '                    logits / self.final_logit_softcapping\n                )\n\n'
              '        logits[..., ::7] += 2.0  # acceptance control: logit bias\n        return logits\n')
# The recipe's policy loss is PPO's (pg loss type ppo: no score centering), slime/utils/ppo_utils.py; the REINFORCE line
# of loss.py (slime_grpo's control) is not on this recipe's path.
LOSS_SIGN = ('slime', 'slime/utils/ppo_utils.py',
             '    pg_losses1 = -ratio * advantages\n    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages\n',
             '    pg_losses1 = ratio * advantages\n    pg_losses2 = ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages\n')
EQUIV_ORDER = ('slime', 'slime/backends/megatron_utils/cp_utils.py', '                    (x_i * chunked_loss_mask).sum()\n',
               '                    (x_i * chunked_loss_mask).flip(0).sum()\n')


OPD = 'slime/rollout/on_policy_distillation.py'
TEACHER_SHIFT = ('slime', OPD, '        t_log_prob[-response_length:]\n',
                 '        t_log_prob[-response_length:] + 0.05  # acceptance control: shifted teacher log-probs\n')
# One token later (the last value repeated): [:-1] would read the first token's None and crash (ERROR, not INCORRECT).
TEACHER_OFF_BY_ONE = ('slime', OPD, 'reward["meta_info"]["input_token_logprobs"][1:]',
                      '(reward["meta_info"]["input_token_logprobs"][2:] + reward["meta_info"]["input_token_logprobs"][-1:])')
TEACHER_TOUCH = ('slime', OPD, 'def post_process_rewards(', '# acceptance control: a teacher-only change\n'
                 'def post_process_rewards(')


def switches(name, environment):
    """A switches file for a control."""
    p = SCRATCH / ('%s.switches.json' % name)
    p.write_text(json.dumps({'environment': environment}) + '\n')
    return str(p)


LAYOUT = ('slime', 'slime/utils/arguments.py', None, None)


def layout_edit():
    rel = 'slime/utils/arguments.py'
    src = (WS / 'targets/reference' / rel).read_text()
    marker = '\ndef parse_args('
    if src.count(marker) != 1:
        raise RuntimeError('layout control: %s has no single parse_args' % rel)
    new = src.replace(marker, '\ndef _parse_args_orig(') + '''

def parse_args(*args, **kwargs):  # acceptance control: 4 sglang engines x TP1 instead of 2 x TP2
    out = _parse_args_orig(*args, **kwargs)
    a = out[0] if isinstance(out, tuple) else out
    a.rollout_num_gpus_per_engine = 1
    return out
'''
    return ('slime', rel, src, new)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def judge_hash():
    """The judge workspace: every file of bench/ (code, configuration)."""
    h = hashlib.sha256()
    for p in sorted(q for q in (WS / 'bench').rglob('*') if q.is_file() and '__pycache__' not in q.parts):
        h.update(str(p.relative_to(WS)).encode() + b'\0' + sha256_file(p).encode() + b'\n')
    return h.hexdigest()


def pack_key(config, devices):
    return {'task': 'slime_opd', 'world_size': config['world_size'], 'gpu': devices[0],
            'judge_workspace_sha256': judge_hash(), 'prep_version': PREP_VERSION,
            'installed_packages_sha256': sha256_file('/opt/researchboost/installed-packages.txt')}


def manifest(root):
    files = sorted(str(p.relative_to(root)) for p in Path(root).rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256')
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        return dict(zip(files, ex.map(lambda f: sha256_file(Path(root) / f), files)))


def pack_usable(key):
    """The pack's key and every file of its manifest; reasons it cannot be used (empty: usable)."""
    meta = json.loads((PACK / 'pack.json').read_text())
    problems = ['%s: pack %r, here %r' % (k, meta.get(k), v) for k, v in key.items() if meta.get(k) != v]
    if problems:
        return problems
    want = {}
    for line in (PACK / 'MANIFEST.sha256').read_text().splitlines():
        h, rel = line.split('  ', 1)
        want[rel] = h
    need = ['fixture/lengths.json'] + ['task/' + f for f in PACK_TASK_FILES]
    missing = [f for f in need if f not in want]
    if missing:
        return ['manifest lacks %s' % missing]
    got = manifest(PACK)
    bad = sorted(f for f in set(want) | set(got) if f != 'pack.json' and want.get(f) != got.get(f))
    return ['files differ from the manifest: %s' % bad[:5]] if bad else []


def restore_pack():
    if (PREP / 'fixture').exists():
        shutil.rmtree(PREP / 'fixture')
    shutil.copytree(PACK / 'fixture', PREP / 'fixture')
    for f in PACK_TASK_FILES:
        shutil.copyfile(PACK / 'task' / f, WS / 'task' / f)
    subprocess.run(['chown', '-R', 'root:root', str(PREP / 'fixture'), str(WS / 'task')], check=True)
    subprocess.run(['chmod', '-R', 'a+rX,go-w', str(PREP / 'fixture'), str(WS / 'task')], check=True)


def discard_pack():
    shutil.rmtree(PREP / 'fixture', ignore_errors=True)
    for f in PACK_TASK_FILES:
        (WS / 'task' / f).unlink(missing_ok=True)


def make_pack(dest):
    """Formulator: a preparation pack from this workspace's full preparation (run as root in the task image, with the
    workspace's task/, runs/bootstrap/ and /opt/researchboost/prep of that preparation)."""
    config = json.loads((WS / 'bench/bench.json').read_text())
    dest = Path(dest)
    if dest.exists():
        raise SystemExit('%s exists' % dest)
    (dest / 'task').mkdir(parents=True)
    shutil.copytree(PREP / 'fixture', dest / 'fixture')
    for f in PACK_TASK_FILES:
        shutil.copyfile(WS / 'task' / f, dest / 'task' / f)
    ev = dest / 'controls'
    for d in sorted((WS / 'runs/bootstrap').glob('control_*')) + [WS / 'runs/bootstrap/quick', WS / 'runs/bootstrap/full']:
        for name in ('check.json', 'profile.json'):
            if (d / name).is_file():
                (ev / d.name).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(d / name, ev / d.name / name)
    key = pack_key(config, gpus(config))
    (dest / 'pack.json').write_text(json.dumps(dict(key, created=datetime.now().isoformat(timespec='seconds'),
                                                    source='full preparation, controls as expected'), indent=1) + '\n')
    files = manifest(dest)
    (dest / 'MANIFEST.sha256').write_text(''.join('%s  %s\n' % (h, f) for f, h in files.items()))
    print(json.dumps({'pack': str(dest), 'files': len(files), 'key': key}))


def prepare_references(config, smoke, out):
    fx = config['fixture']
    run('fixture', '--out', out, '--rollouts', str(fx['smoke_num_rollouts'] if smoke else fx['num_rollouts']),
        phase='fixture')
    run('freeze-rollout', '--out', out + '/rollout_ref', phase='rollout_ref')
    run('freeze-teacher', '--out', out + '/teacher_ref', phase='teacher_ref')
    run('freeze-train', '--out', out + '/train_ref', *(['--smoke'] if smoke else []), phase='train_ref')


def freeze_inputs():
    run('freeze-inputs', 'bench', str(PREP / 'fixture'), str(ASSETS / 'Qwen3.5-9B'), str(ASSETS / 'Qwen3.5-27B'),
        str(ASSETS / 'Qwen3.5-9B_torch_dist'), str(ASSETS / 'data'), phase='inputs')
    subprocess.run(['chmod', '-R', 'a+rX', str(PREP), str(WS)], check=True)
    inputs = json.loads((WS / 'task/formulation.json').read_text())['inputs']
    unreadable = [q for q in inputs if not os.stat(q if os.path.isabs(q) else WS / q).st_mode & 0o004]
    if unreadable:
        raise RuntimeError('%d frozen inputs are not readable by the agent user, e.g. %s' % (len(unreadable),
                                                                                            unreadable[:3]))


def timing_references(out, pack):
    """The reference timing of each side, as pairs of runs: a pair whose two runs differ by more than
    TIMING_SPREAD_MAX in a window is measured again (fresh runs), up to TIMING_PAIRS_MAX pairs; the first pair within
    it is frozen, else the last one, with a flag. Returns {side: spread} of the sides flagged."""
    flags = {}
    for side in TIMING_SIDES:
        for k in range(1, TIMING_PAIRS_MAX + 1):
            # the full preparation's first rollout and teacher pairs reuse freeze-rollout's and freeze-teacher's runs
            fresh = pack or k > 1
            for attempt in range(1, TIMING_CRASH_RETRIES + 2):
                # a reference timing run that crashed (e.g. an NCCL error) says nothing about the host's timing:
                # measure that pair again (fresh), not counted as one of the pairs
                try:
                    run('freeze-timing', '--out', '%s/timing_%s_%d' % (out, side, k), '--sides', side,
                        *(['--fresh'] if (fresh or attempt > 1) and side in ('rollout', 'teacher') else []),
                        phase='timing_%s_%d' % (side, k))
                    break
                except subprocess.CalledProcessError:
                    if attempt > TIMING_CRASH_RETRIES:
                        raise
                    crashed = WS / ('%s/timing_%s_%d' % (out, side, k))
                    if crashed.exists():
                        shutil.move(str(crashed), str(crashed) + '_crash_%d' % attempt)
                    log('%s timing pair %d: a reference run failed (attempt %d), measuring the pair again'
                        % (side, k, attempt))
            spread = json.loads((WS / 'task/timing_ref.json').read_text())[side]['spread']
            log('%s timing pair %d: spread %.2f%%' % (side, k, 100 * spread))
            if spread <= TIMING_SPREAD_MAX:
                break
        else:
            flags[side] = spread
            log('timing_spread_flag: %s spread %.2f%% after %d pairs (> %.0f%%); the last pair is frozen'
                % (side, 100 * spread, TIMING_PAIRS_MAX, 100 * TIMING_SPREAD_MAX))
    return flags


def timing_cache_key(config, devices):
    """Pack key + this host and its GPUs: the timings are valid for exactly that combination."""
    uuids = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                           capture_output=True, text=True)
    key = dict(pack_key(config, devices),
               gpu_uuids=sorted(filter(None, (u.strip() for u in uuids.stdout.splitlines()))))
    return key, hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def timing_cache_load(config, devices):
    """The cached task/timing_ref.json for this machine and judge, when one is fresh enough; else None."""
    if os.environ.get('RBB_TIMING_CACHE_DIR') == 'off':
        return None
    key, digest = timing_cache_key(config, devices)
    entry = TIMING_CACHE / digest / 'entry.json'
    if not entry.is_file():
        return None
    try:
        data = json.loads(entry.read_text())
    except ValueError:
        return None
    age_h = (time.time() - data.get('created_epoch', 0)) / 3600
    if age_h > TIMING_CACHE_MAX_AGE_H:
        log('timing cache: entry %s is %.1f h old (> %.0f h), measuring again' % (digest, age_h, TIMING_CACHE_MAX_AGE_H))
        return None
    if data.get('key') != key:
        return None
    data['age_h'] = age_h
    data['digest'] = digest
    return data


def shared_timing_ref(config, devices):
    """One reference for every lane (owner, 11-16): RBB_SHARED_TIMING_REF names a timing-cache entry recorded once on a
    quiet machine with this judge; it replaces this lane's own measurement. A missing file or an entry of another judge
    is an error, never a silent re-measurement."""
    path = os.environ.get('RBB_SHARED_TIMING_REF')
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    key, _ = timing_cache_key(config, devices)
    want = {k: v for k, v in key.items() if k != 'gpu_uuids'}
    got = {k: v for k, v in (data.get('key') or {}).items() if k != 'gpu_uuids'}
    if want != got:
        raise RuntimeError('shared timing reference %s was recorded for %s; this judge is %s' % (path, got, want))
    data['age_h'] = (time.time() - data.get('created_epoch', 0)) / 3600
    data['digest'] = 'shared %s (recorded on GPUs %s)' % (path, ','.join(u[-6:] for u in data['key'].get('gpu_uuids', [])))
    return data


def timing_cache_save(config, devices, spread_flags):
    if os.environ.get('RBB_TIMING_CACHE_DIR') == 'off':
        return
    key, digest = timing_cache_key(config, devices)
    out = TIMING_CACHE / digest
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'entry.json').write_text(json.dumps(
            {'key': key, 'created': datetime.now().isoformat(timespec='seconds'), 'created_epoch': time.time(),
             'spread_flags': spread_flags, 'timing_ref': (WS / 'task/timing_ref.json').read_text()}, indent=1) + '\n')
        os.chmod(out / 'entry.json', 0o666)
        log('timing cache: wrote %s' % (out / 'entry.json'))
    except OSError as exc:
        log('timing cache: could not write %s (%s)' % (out, exc))


def main():
    started = time.time()
    config = json.loads((WS / 'bench/bench.json').read_text())
    devices = gpus(config)
    log('GPUs: %s' % devices)
    lock = Path(config['gpu_lock'])
    lock.touch()
    lock.chmod(0o666)
    smoke = os.environ.get('RESEARCHBOOST_SMOKE') == '1'
    controls = os.environ.get('RESEARCHBOOST_SLIME_OPD_CONTROLS', '1') == '1'
    (WS / 'task').mkdir(exist_ok=True)
    (WS / 'task/formulation.json').write_text(json.dumps(
        {'task': config['task'], 'reference': {'slime': config['reference_commit']}, 'inputs': {}}) + '\n')
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    PREP.mkdir(parents=True, exist_ok=True)
    out = 'runs/bootstrap'
    pack = False
    if not smoke and (PACK / 'pack.json').is_file():
        t0 = time.time()
        problems = pack_usable(pack_key(config, devices))
        if problems:
            log('preparation pack at %s not usable (%s): full preparation' % (PACK, '; '.join(problems)))
        else:
            log('preparation pack: %s' % (PACK / 'pack.json').read_text().replace('\n', ' '))
            restore_pack()
            pack = True
        PHASES['pack_restore'] = round(time.time() - t0)
    if not pack:
        prepare_references(config, smoke, out)
    freeze_inputs()
    train_ref = json.loads((WS / 'task/train_ref.json').read_text())
    log('reference records: repeatable bitwise: %s (%d differing tensors)'
        % (train_ref['repeatable'], train_ref['repeat_differing']))
    if smoke:
        expected = 'PASS' if train_ref['repeatable'] else 'NOT_BITWISE'
        check(CANDIDATE, out + '/quick', expected, '--quick')
        shutil.rmtree(SCRATCH, ignore_errors=True)
        ready(devices, 'smoke', started)
        return
    if pack:
        try:
            check(CANDIDATE, out + '/quick', 'PASS', '--quick')
        except RuntimeError as exc:
            log('the unchanged tree does not pass against the pack\'s references here (%s): full preparation' % exc)
            discard_pack()
            pack = False
            prepare_references(config, smoke, out)
            freeze_inputs()
    if not pack:
        check(CANDIDATE, out + '/quick', 'PASS', '--quick')
    cached = shared_timing_ref(config, devices) or timing_cache_load(config, devices)
    if cached:
        t0 = time.time()
        (WS / 'task/timing_ref.json').write_text(cached['timing_ref'])
        spread_flags = cached.get('spread_flags') or {}
        timing_source = 'cache %s measured %s (%.1f h ago)' % (cached['digest'], cached['created'], cached['age_h'])
        PHASES['timing_cache_restore'] = round(time.time() - t0)
        log('timing cache hit: %s; the reference timing pairs and the reference full check are skipped'
            % timing_source)
    else:
        timing_source = 'measured'
        spread_flags = timing_references(out, pack)
        full = reference_final(out)
        run('freeze-timing', '--out', out + '/timing', '--sides', '--full', str(WS / out / 'full/check.json'),
            phase='fractions')
        timing_cache_save(config, devices, spread_flags)
    log('reference timing: %s' % (WS / 'task/timing_ref.json').read_text())
    if controls and not pack:
        check(control_tree('b_entry', [ENTRY]), out + '/control_entry', 'ERROR', '--skip-timing', stage='structure')
        check(control_tree('c_logit_bias', [LOGIT_BIAS]), out + '/control_logit_bias', 'INCORRECT', '--skip-timing',
              stage='token_forcing')
        check(control_tree('c_teacher_shift', [TEACHER_SHIFT]), out + '/control_teacher_shift', 'INCORRECT',
              '--skip-timing', stage='teacher_logprobs')
        check(control_tree('c_teacher_off_by_one', [TEACHER_OFF_BY_ONE]), out + '/control_teacher_off_by_one',
              'INCORRECT', '--skip-timing', stage='teacher_logprobs')
        touch = control_tree('d_teacher_touch', [TEACHER_TOUCH])
        check(touch, out + '/control_teacher_chunked', 'PASS', '--skip-timing', '--config',
              switches('teacher_chunked', {'OPD_TEACHER_SERVER_ARGS': '--chunked-prefill-size 8192'}),
              verdict='equivalent', tier='N1')
        check(touch, out + '/control_teacher_layout', 'INCORRECT', '--skip-timing', '--config',
              switches('teacher_tp2', {'OPD_TEACHER_SERVER_ARGS': '--tp 2'}), stage='layout')
        check(control_tree('c_layout', [layout_edit()]), out + '/control_layout', 'INCORRECT', '--skip-timing',
              stage='layout')
        check(control_tree('c_loss_sign', [LOSS_SIGN]), out + '/control_loss_sign', 'INCORRECT', '--skip-timing',
              stage='statistical')
        check(control_tree('d_equiv_order', [EQUIV_ORDER]), out + '/control_equiv_order', 'PASS', '--skip-timing',
              verdict='equivalent', tier='N1')
        t0 = time.time()
        arm = subprocess.run([PY, 'bench/bench.py', 'control-arm', '--out', out + '/control_lr102'], cwd=WS)
        if arm.returncode != 0:
            raise RuntimeError('Preparation control failed: lr x 1.02 not resolved by the statistical gate')
        PHASES['control_lr102'] = round(time.time() - t0)
        for side in ('train', 'teacher'):
            t0 = time.time()
            prof = subprocess.run([PY, 'bench/bench.py', 'profile', '--tree', CANDIDATE, '--side', side, '--out',
                                   out + '/control_profile_' + side], cwd=WS)
            data = json.loads((WS / out / ('control_profile_%s/profile.json' % side)).read_text())
            if prof.returncode != 0 or not data.get('traces'):
                raise RuntimeError('Preparation control failed: profile --side %s wrote no trace: %s'
                                   % (side, json.dumps({k: data.get(k) for k in ('status', 'reason')})))
            PHASES['control_profile_' + side] = round(time.time() - t0)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    timing = json.loads((WS / 'task/timing_ref.json').read_text())
    ready(devices, 'pack' if pack else 'full', started, controls=bool(controls and not pack),
          timing_source=timing_source,
          timing_spread={s: timing[s]['spread'] for s in (*TIMING_SIDES, 'full') if s in timing},
          **({'timing_spread_flag': spread_flags} if spread_flags else {}),
          **({'pack': json.loads((PACK / 'pack.json').read_text())} if pack else {}))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--make-pack':
        make_pack(sys.argv[2])
    else:
        main()
