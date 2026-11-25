#!/usr/bin/env python3
"""refpack: reference packs for one host (or a few hosts sharing a directory). Prepare a task's judge once, then
verify it cheaply anywhere and judge submissions from it in clean containers.

    refpack.py build  --task T --image IMG --gpu-ids 0,1 [--cpus auto] ...   full preparation once, publish the pack
    refpack.py verify --task T --image IMG --gpu-ids 4,5 ...                  restore the pack on these GPUs, verify
    refpack.py judge  --task T --image IMG --gpu-ids 0,1 --artifacts <trial dir>/artifacts --out DIR ...
                                                                              final judge of a finished trial
    refpack.py show   [--task T]                                              packs in the store
Common options: --tasks-dir tasks (holds <task>/task.toml), --env-file researchboost.env, --store packs.

What a pack is. The task's own startup preparation (environment/bootstrap.py: reference records, trajectories,
frozen sigmas, the frozen reference timing, the controls' evidence ...) run once in a clean container with no network;
everything it wrote under /opt/researchboost/{ws,prep} is collected with a sha256 manifest (container/svcpack.py) and
stored as <store>/<task>/<key>/. Published once (temporary name, then rename), never modified; a pack you no longer
trust is moved away by hand, never silently replaced.

The key: sha256 of {task, judge_hash, image_digest, gpu_model, world_size, driver, prep_version} (container/svckey.py).
judge_hash is the sha256 of the judge workspace as the image ships it, so any judge change is a new key.

verify and judge start from the pack of their key and only verify on this machine: one reference record compared
bitwise with the pack's, one reference timing inside the band of the pack's frozen value (the band checks the
MACHINE, it is not the candidate's bar). No pack, or a pack that does not verify, stops with exit 82 / 81: never a
silent fresh preparation.

judge then puts the submission (Harbor's artifacts/ layout of a finished trial: the task tree, switches.json and
/opt/candidate-env) back in place as the agent user, and runs the task's unchanged final judge (tests/test.sh ->
verify.py -> bench.py check --final) under a guard that kills escaped candidate processes. Output: <out>/check.json,
reward.json, judge.json (pack key, image, judge hash, GPUs, timings), verifier/ (the judge's own files), svc/.

Machine use: build on an otherwise idle WHOLE host (its frozen reference timing becomes the bar of every agent check);
verify and judge on an otherwise idle NUMA half at least (the half holding --gpu-ids, with their cores: --cpus auto).

Task options: --python (the image's training interpreter; default from container/tasks/<task>.py PYTHON), --adapter
(default: container/tasks/<task>.py when it exists, else generic), and for the generic adapter --timing-file,
--timing-field, [--band], [--dirs], [--time-tree-wrap]. Build quality gate: --spread-max S (the frozen reference
spread must be <= S, read from --spread-file / --spread-key in the pack; exit 75 when noisy).
Exit codes: 0 ok; 2 usage; 75 noisy build (retry); 81 the pack does not verify here; 82 no pack for the key; 84 the
submission was refused; 1 infrastructure failure (see the work directory).
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
CONTAINER = KIT / 'container'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CONTAINER))
import svckey  # noqa: E402
from _hostlib import dumps, gpu_facts, load_toml, parse_env_file, resolve_cpus  # noqa: E402

EXIT_INFRA, EXIT_NOISY, EXIT_MISMATCH, EXIT_NOPACK, EXIT_REFUSED = 1, 75, 81, 82, 84


def log(msg):
    print('[refpack %s] %s' % (datetime.now().strftime('%H:%M:%S'), msg), file=sys.stderr, flush=True)


def die(msg, code=2):
    log('ERROR: ' + msg)
    sys.exit(code)


def run(argv, check=True, env=None, timeout=None):
    r = subprocess.run([str(a) for a in argv], capture_output=True, text=True, env=env, timeout=timeout)
    if check and r.returncode:
        die('%s failed (%d): %s' % (' '.join(map(str, argv[:4])), r.returncode, (r.stderr or r.stdout)[-1500:]),
            EXIT_INFRA)
    return r


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


# ------------------------------------------------------------------ the task

class Task:
    def __init__(self, a):
        self.name = a.task
        self.dir = (Path(a.tasks_dir) / a.task).resolve()
        if not (self.dir / 'task.toml').is_file():
            die('no %s/task.toml' % self.dir)
        self.toml = load_toml(self.dir / 'task.toml')
        self.gpus = int((self.toml.get('environment') or {}).get('gpus', 0) or 0)
        self.artifacts = list(self.toml.get('artifacts') or [])
        module = CONTAINER / 'tasks' / ('%s.py' % a.task)
        self.adapter = a.adapter or (a.task if module.is_file() else 'generic')
        if self.adapter != 'generic' and not (CONTAINER / 'tasks' / ('%s.py' % self.adapter)).is_file():
            die('no pack module container/tasks/%s.py (or use --adapter generic)' % self.adapter)
        self.python = a.python
        if not self.python and module.is_file():
            m = re.search(r"^PYTHON = '([^']+)'", module.read_text(), re.M)
            self.python = m.group(1) if m else None
        if not self.python:
            die('give --python: the training interpreter of the task image (the one entrypoint.sh runs bootstrap.py with)')
        self.compat = getattr(a, 'compat', None) or 'never'
        self.config = None
        if self.adapter == 'generic':
            if not (a.timing_file and a.timing_field):
                die('the generic adapter needs --timing-file and --timing-field (the frozen reference timing)')
            self.config = {'adapter': 'generic', 'generic': {
                'dirs': a.dirs.split(',') if a.dirs else ['task', 'bench/records'],
                'timing_file': a.timing_file, 'timing_field': a.timing_field, 'band': a.band,
                'time_tree_wrap': a.time_tree_wrap}}

    def env(self):
        out = {'RBB_SVC_TASK': self.name, 'RBB_SVC_PYTHON': self.python, 'RBB_SVC_COMPAT': self.compat}
        config = self.config or ({'adapter': self.adapter} if self.adapter != self.name else None)
        if config:
            out['RBB_SVC_TASK_CONFIG'] = json.dumps(config, separators=(',', ':'))
        return out


def image_info(ref):
    r = run(['docker', 'image', 'inspect', ref, '--format', '{{json .}}'])
    d = json.loads(r.stdout)
    # The image ID (its config digest) is the same whether the image was built, pulled or `docker load`ed; a registry
    # manifest digest is not. The key uses the ID; the registry digests are recorded for reference.
    return {'ref': ref, 'id': d['Id'], 'digest': svckey.image_digest_of(image_id=d['Id']),
            'repo_digests': d.get('RepoDigests') or []}


def keyinfo(task, image):
    """judge_hash, world_size, prep_version from the image (no network, no GPU)."""
    r = run(['docker', 'run', '--rm', '--network', 'none', '-v', '%s:/opt/rbb-svc:ro' % CONTAINER,
             '--entrypoint', task.python, '-e', 'RBB_SVC_TASK_CONFIG=%s' % task.env().get('RBB_SVC_TASK_CONFIG', ''),
             image['id'], '/opt/rbb-svc/svcpack.py', 'keyinfo', '--task', task.name], timeout=3600)
    return json.loads(r.stdout.strip().splitlines()[-1])


def compute_key(task, image, gpu_ids):
    ki = keyinfo(task, image)
    if int(ki['world_size']) != task.gpus:
        die('bench.json world_size %s != task.toml gpus %s' % (ki['world_size'], task.gpus))
    g = gpu_facts(gpu_ids)
    fields = {'task': task.name, 'judge_hash': ki['judge_hash'], 'image_digest': image['digest'],
              'gpu_model': g['gpu_model'], 'world_size': int(ki['world_size']), 'driver': g['driver'],
              'prep_version': ki['prep_version']}
    return svckey.pack_key(**fields), fields, g


# ------------------------------------------------------------------ the container

def stage_code(dest):
    """A copy of container/ for one run (the code cannot change under a running judge), with the empty mount points
    the role's mounts need inside the read-only /opt/rbb-svc (out/, pack/)."""
    shutil.copytree(str(CONTAINER), str(dest), ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for d in ('out', 'pack'):
        (dest / d).mkdir(exist_ok=True)
    for p in dest.rglob('*'):
        p.chmod(0o755 if p.is_dir() else 0o644)
    return dest


def make_compose(a, task, image, role, work, gpu_ids, cpuset, mounts, env, hide_task_packs=False):
    """The task's own Compose service `main` (shm, ulimits, environment, assets and scratch mounts) as `docker compose
    config` resolves it with the env file, then: only `main` (no other services, no depends_on, no build), the image
    by ID, no network, entry.sh as the entrypoint, the task's CPU and memory limits, these GPUs and CPUs, a fresh /tmp,
    the kit's container code read-only and the role's mounts."""
    base = work / 'compose.base.json'
    base.write_text(json.dumps({'services': {'main': {'image': image['id']}}}) + '\n')
    files = ([task.dir / 'environment' / 'docker-compose.yaml']
             if (task.dir / 'environment' / 'docker-compose.yaml').is_file() else []) + [base]
    files += [Path(f) for f in a.extra_docker_compose]
    argv = ['docker', 'compose', '--env-file', a.env_file, '-p', 'refpackcfg', '--project-directory',
            task.dir / 'environment']
    for f in files:
        argv += ['-f', f]
    cfg_env = dict(os.environ, RESEARCHBOOST_SMOKE='', RESEARCHBOOST_TEST_WORLD_SIZE='')
    main = json.loads(run(argv + ['config', '--format', 'json'], env=cfg_env).stdout)['services']['main']
    for k in ('depends_on', 'build', 'healthcheck', 'container_name', 'networks', 'ports', 'command'):
        main.pop(k, None)
    vols = []
    for v in main.get('volumes', []):
        tgt = v.get('target') if isinstance(v, dict) else str(v).split(':')[1]
        if tgt == '/tmp':
            continue
        if hide_task_packs and tgt == '/opt/researchboost/prep-pack':
            continue
        if role != 'build' and tgt == '/opt/researchboost/prep-pack':
            continue                                   # pack mode: the reference pack is the only reference
        vols.append(v)
    scratch = Path(a.scratch) if a.scratch else work / 'tmp'
    scratch.mkdir(parents=True, exist_ok=True)
    vols.append({'type': 'bind', 'source': str(scratch), 'target': '/tmp'})
    code = stage_code(work / 'svc-code')
    vols.append({'type': 'bind', 'source': str(code), 'target': '/opt/rbb-svc', 'read_only': True})
    for src, tgt, ro in mounts:
        vols.append({'type': 'bind', 'source': str(src), 'target': tgt, 'read_only': ro})
    main['volumes'] = vols
    main['image'] = image['id']
    main['pull_policy'] = 'never'
    main['entrypoint'] = ['/bin/bash', '/opt/rbb-svc/entry.sh', role]
    if role == 'trial':
        main['command'] = ['true']                     # verify only: exit as soon as the pack verified
    main['network_mode'] = 'none'
    menv = main.get('environment') or {}
    if isinstance(menv, list):
        menv = dict(x.split('=', 1) for x in menv)
    menv.update(dict(x.split('=', 1) for x in a.judge_env or []))
    menv.update(task.env())
    menv.update(env)
    envcfg = task.toml.get('environment') or {}
    if envcfg.get('cpus'):
        main['cpus'] = float(envcfg['cpus'])
    if envcfg.get('memory_mb'):
        main['mem_limit'] = '%dm' % envcfg['memory_mb']
    if task.gpus == 0:                                 # CPU-only: no GPU device; the host passes the key's GPU fields
        main.pop('deploy', None)
        g = gpu_facts(gpu_ids)
        menv.update({'RBB_SVC_HOST_GPU_MODEL': g['gpu_model'], 'RBB_SVC_HOST_DRIVER': g['driver']})
    else:
        main['deploy'] = {'resources': {'reservations': {'devices': [
            {'driver': 'nvidia', 'device_ids': list(gpu_ids), 'capabilities': ['gpu']}]}}}
    main['environment'] = menv
    if cpuset:
        main['cpuset'] = cpuset
    labels = main.get('labels') or {}
    labels.update({'vetkit.refpack.role': role, 'vetkit.refpack.task': task.name})
    main['labels'] = labels
    project = 'refpack-%s-%s-%s' % (role, task.name.replace('_', '-'), work.name.rsplit('-', 1)[-1])
    path = work / 'compose.json'
    path.write_text(dumps({'name': project, 'services': {'main': main}}) + '\n')
    return path, project


def run_container(compose, project, timeout_s, logfile):
    """docker compose up (attached) until main exits; then down. Returns main's exit code (None: timeout)."""
    base = ['docker', 'compose', '-p', project, '-f', str(compose)]
    code = None
    try:
        with open(logfile, 'w') as lf:
            p = subprocess.Popen(base + ['up', '--no-build', '--abort-on-container-exit', '--exit-code-from', 'main'],
                                 stdout=lf, stderr=subprocess.STDOUT)
            try:
                code = p.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                log('container timeout after %d s: stopping' % timeout_s)
                subprocess.run(base + ['stop', '-t', '60'], capture_output=True)
                p.wait()
    finally:
        subprocess.run(base + ['down', '--volumes', '--remove-orphans', '-t', '30'], capture_output=True)
    return code


def work_dir(a, role, task):
    d = Path(a.work or 'refpack-work') / ('%s-%s-%s-%s' % (role, task.name, datetime.now().strftime('%m%d-%H%M%S'),
                                                        os.urandom(3).hex()))
    d.mkdir(parents=True)
    return d.resolve()


def require_root(a):
    """The judge's output directories and the packs must be root-owned: a candidate runs as the agent user (often
    uid 1000), and a host user with the same uid would let it plant files that the root verifier then writes through,
    or read the pack's judge-only files."""
    if os.geteuid() != 0 and not a.test_non_root:
        die('run as root (the packs and the judge outputs must be root-owned); --test-non-root only for the tests')


def common(a):
    require_root(a)
    task = Task(a)
    gpu_ids = [g.strip() for g in a.gpu_ids.split(',')] if a.gpu_ids else []
    if not gpu_ids:
        die('--gpu-ids is required (for a CPU-only task: the GPUs of the NUMA half whose cores it uses)')
    if task.gpus and len(gpu_ids) != task.gpus:
        die('--gpu-ids names %d GPUs, the task runs on %d' % (len(gpu_ids), task.gpus))
    if not Path(a.env_file).is_file():
        die('no env file %s' % a.env_file)
    a.env_file = str(Path(a.env_file).resolve())
    image = image_info(a.image)
    key, fields, g = compute_key(task, image, gpu_ids)
    cpuset = resolve_cpus(a.cpus, gpu_ids)
    return task, gpu_ids, image, key, fields, g, cpuset


def store_dir(a, task, key):
    return Path(a.store).resolve() / task.name / key


def check_pack_dir(d):
    meta = json.loads((d / 'pack.json').read_text())
    if hashlib.sha256((d / 'manifest.json').read_bytes()).hexdigest() != meta['content_sha256']:
        die('pack %s: manifest.json does not match pack.json' % d, EXIT_MISMATCH)
    return meta


# ------------------------------------------------------------------ commands

def cmd_build(a):
    task, gpu_ids, image, key, fields, g, cpuset = common(a)
    dest = store_dir(a, task, key)
    if (dest / 'pack.json').is_file():
        print(dumps({'result': 'exists', 'key': key, 'store': str(dest)}))
        return 0
    work = work_dir(a, 'build', task)
    out = work / 'out'
    out.mkdir(mode=0o755)
    mounts = [(out, '/opt/rbb-svc/out', False)]
    if a.full:                                         # hide the task's own packs: a truly full preparation
        empty = work / 'empty'
        empty.mkdir()
        mounts.append((empty, '/opt/researchboost/prep-pack', True))
        assets = parse_env_file(a.env_file).get('RESEARCHBOOST_ASSETS_ROOT')
        if assets and (Path(assets) / task.name / 'pack').is_dir():
            mounts.append((empty, '/opt/researchboost/assets/pack', True))   # e.g. nanochat's own $ASSETS/pack
    compose, project = make_compose(a, task, image, 'build', work, gpu_ids, cpuset, mounts,
                                    {'RBB_SVC_OUT': '/opt/rbb-svc/out'}, hide_task_packs=a.full)
    hc = (task.toml.get('environment') or {}).get('healthcheck') or {}
    timeout = int(hc.get('start_period_sec', 28800)) + 1800
    started = now()
    log('pack build %s key %s on GPUs %s (CPUs %s), work %s' % (task.name, key[:16], ','.join(gpu_ids),
                                                                cpuset or 'any', work))
    code = run_container(compose, project, timeout, work / 'container.log')
    if code != 0 or not (out / 'pack' / 'manifest.json').is_file():
        die('pack build failed (exit %s); see %s (container.log, out/prepare.log)' % (code, work), EXIT_INFRA)
    collect = json.loads((out / 'pack' / 'collect.json').read_text())
    ready = collect.get('ready') or {}
    if ready.get('mode') == 'smoke':
        die('the build prepared in smoke mode (RESEARCHBOOST_SMOKE set in the env file?)', EXIT_INFRA)
    spread = None
    if a.spread_max is not None:
        f = out / 'pack' / 'files' / 'opt/researchboost/ws' / a.spread_file
        try:
            v = json.loads(f.read_text())
            for k in a.spread_key.split('.'):
                v = v[k]
            spread = float(v)
        except (OSError, ValueError, KeyError, TypeError):
            die('quality gate: no %s in %s of the pack' % (a.spread_key, a.spread_file), EXIT_INFRA)
        if spread > a.spread_max:
            die('the reference timing of this build is noisy (spread %.3f > %.3f): not published; rebuild on a '
                'quiet host' % (spread, a.spread_max), EXIT_NOISY)
    mb = (out / 'pack' / 'manifest.json').read_bytes()
    meta = {'schema': 'vetkit-pack/1', 'key': key, 'key_fields': fields, 'content_sha256': hashlib.sha256(mb).hexdigest(),
            'reference_spread': spread, 'files': collect['files'], 'bytes': collect['bytes'], 'roots': collect['roots'],
            'collect_ready': ready,
            'made_by': {'gpus': gpu_ids, 'cpus': cpuset, 'gpu_uuids': g['uuids'], 'image': image,
                        'full': bool(a.full), 'started': started, 'ended': now(), 'work': str(work)},
            'created': now()}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / ('.tmp-%s-%d' % (key[:16], os.getpid()))
    shutil.copytree(str(out / 'pack'), str(tmp), symlinks=True)
    (tmp / 'pack.json').write_text(dumps(meta) + '\n')
    check_pack_dir(tmp)
    os.chmod(str(tmp), 0o700)          # judge-only content (held-out data, frozen references): root only
    try:
        os.rename(str(tmp), str(dest))
    except OSError:
        shutil.rmtree(str(tmp), ignore_errors=True)
        if not (dest / 'pack.json').is_file():
            raise
        log('another build published key %s first; kept that one' % key[:16])
    if not a.keep:
        shutil.rmtree(str(work / 'tmp'), ignore_errors=True)
    print(dumps({'result': 'published', 'key': key, 'store': str(dest), 'files': collect['files'],
                 'bytes': collect['bytes'], 'reference_spread': spread}))
    return 0


def pack_for(a, task, key, fields):
    d = store_dir(a, task, key)
    if not (d / 'pack.json').is_file():
        log('no pack for key %s (%s); run `refpack.py build` first' % (key[:16], json.dumps(fields)))
        sys.exit(EXIT_NOPACK)
    if d.stat().st_mode & 0o077:       # mounted into a container where candidate code runs: never readable by others
        log('pack %s was readable by other users: made it 0700' % d)
        os.chmod(str(d), 0o700)
    return d, check_pack_dir(d)


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def cmd_verify(a):
    task, gpu_ids, image, key, fields, g, cpuset = common(a)
    pack, meta = pack_for(a, task, key, fields)
    work = work_dir(a, 'verify', task)
    out = work / 'svc'
    out.mkdir()
    env = {'RBB_SVC_OUT': '/opt/rbb-svc/out', 'RBB_SVC_IMAGE_DIGEST': image['digest'], 'RBB_SVC_KEY': key}
    compose, project = make_compose(a, task, image, 'trial', work, gpu_ids, cpuset,
                                    [(pack, '/opt/rbb-svc/pack', True), (out, '/opt/rbb-svc/out', False)], env)
    hc = (task.toml.get('environment') or {}).get('healthcheck') or {}
    code = run_container(compose, project, int(hc.get('start_period_sec', 28800)) + 1800, work / 'container.log')
    report = load(out / 'svc-pack.json') or {}
    result = {0: 'verified', EXIT_MISMATCH: 'pack-mismatch', EXIT_NOPACK: 'no-pack'}.get(code, 'infra-failure')
    print(dumps({'result': result, 'container_exit': code, 'key': key, 'reason': report.get('reason'),
                 'record': (report.get('steps') or {}).get('record'), 'timing': (report.get('steps') or {}).get('timing'),
                 'work': str(work)}))
    return {'verified': 0, 'pack-mismatch': EXIT_MISMATCH, 'no-pack': EXIT_NOPACK}.get(result, EXIT_INFRA)


def cmd_judge(a):
    task, gpu_ids, image, key, fields, g, cpuset = common(a)
    pack, meta = pack_for(a, task, key, fields)
    arts = Path(a.artifacts).resolve()
    if not (arts / 'manifest.json').is_file():
        die('%s is not a Harbor artifacts/ directory (no manifest.json)' % arts)
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        die('output directory %s is not empty' % out)
    (out / 'verifier').mkdir(parents=True, exist_ok=True)
    (out / 'svc').mkdir(exist_ok=True)
    for d in (out, out / 'verifier', out / 'svc'):
        if os.geteuid() == 0:
            os.chown(str(d), 0, 0)
        d.chmod(0o755)
    work = work_dir(a, 'judge', task)
    mounts = [(pack, '/opt/rbb-svc/pack', True), (arts, '/logs/artifacts', True),
              (out / 'verifier', '/logs/verifier', False), (out / 'svc', '/opt/rbb-svc/out', False),
              (task.dir / 'tests', '/tests', True)]
    env = {'RBB_SVC_OUT': '/opt/rbb-svc/out', 'RBB_SVC_IMAGE_DIGEST': image['digest'], 'RBB_SVC_KEY': key,
           'RBB_SVC_ARTIFACTS': ' '.join(task.artifacts), 'RBB_SVC_GUARD': '0' if a.no_guard else '1'}
    compose, project = make_compose(a, task, image, 'judge', work, gpu_ids, cpuset, mounts, env)
    timeout = int((task.toml.get('verifier') or {}).get('timeout_sec', 14400)) * 2 + 7200
    started = now()
    log('judge %s: artifacts %s, pack %s, GPUs %s (CPUs %s), out %s' % (task.name, arts, key[:16], ','.join(gpu_ids),
                                                                         cpuset or 'any', out))
    code = run_container(compose, project, timeout, out / 'container.log')
    check, reward = load(out / 'verifier' / 'check.json'), load(out / 'verifier' / 'reward.json')
    svcpack = load(out / 'svc' / 'svc-pack.json') or {}
    guard = load(out / 'svc' / 'guard.json')
    if code == 0 and reward is not None:
        result, exit_code = 'judged', 0
    elif code == EXIT_MISMATCH:
        result, exit_code = 'pack-mismatch', EXIT_MISMATCH
    elif code == EXIT_NOPACK:
        result, exit_code = 'no-pack', EXIT_NOPACK
    elif code == EXIT_REFUSED:
        result, exit_code = 'bundle-refused', EXIT_REFUSED
    else:
        result, exit_code = 'infra-failure', EXIT_INFRA
    # check.json is the judge's only when verify.py certified it with a reward; otherwise it may be the candidate's own
    certified = result == 'judged'
    if check is not None and certified:
        shutil.copyfile(str(out / 'verifier' / 'check.json'), str(out / 'check.json'))
    if reward is not None:
        shutil.copyfile(str(out / 'verifier' / 'reward.json'), str(out / 'reward.json'))
    if not certified:
        check = None
    perf = (check or {}).get('performance') or {}
    corr = (check or {}).get('correctness') or {}
    record = {
        'schema': 'vetkit-judge/1', 'task': task.name, 'result': result, 'container_exit': code, 'reward': reward,
        'status': (check or {}).get('status'), 'speedup': perf.get('speedup'), 'verdict': corr.get('verdict'),
        'tier': corr.get('measured_tier'), 'artifacts': str(arts), 'pack_key': key, 'pack_key_fields': fields,
        'pack_content_sha256': meta['content_sha256'], 'pack_made_by': meta.get('made_by'),
        'pack_verify': {'status': svcpack.get('status'), 'reason': svcpack.get('reason'),
                        'record': (svcpack.get('steps') or {}).get('record'),
                        'timing': (svcpack.get('steps') or {}).get('timing')},
        'image': image, 'judge_hash': fields['judge_hash'], 'evaluator_sha256': (check or {}).get('evaluator_sha256'),
        'gpu_model': fields['gpu_model'], 'driver': fields['driver'], 'gpus': gpu_ids, 'gpu_uuids': g['uuids'],
        'cpus': cpuset, 'guard': ({'killed': guard.get('killed'), 'kills': (guard.get('kills') or [])[:20]} if guard
                                  else ('off (--no-guard)' if a.no_guard else None)),
        'started': started, 'ended': now(), 'work': str(work), 'out': str(out)}
    (out / 'judge.json').write_text(dumps(record) + '\n')
    if not a.keep:
        shutil.rmtree(str(work), ignore_errors=True)
    print(dumps({k: record[k] for k in ('result', 'reward', 'status', 'speedup', 'verdict', 'tier', 'pack_key', 'out')}))
    return exit_code


def cmd_show(a):
    root = Path(a.store)
    rows = []
    for p in sorted(root.glob('%s/*/pack.json' % (a.task or '*'))):
        m = json.loads(p.read_text())
        rows.append({'task': m['key_fields']['task'], 'key': m['key'], 'gpu_model': m['key_fields']['gpu_model'],
                     'world_size': m['key_fields']['world_size'], 'files': m.get('files'),
                     'reference_spread': m.get('reference_spread'), 'created': m.get('created')})
    print(dumps(rows))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('build', 'verify', 'judge'):
        p = sub.add_parser(name)
        p.add_argument('--task', required=True)
        p.add_argument('--tasks-dir', default='tasks')
        p.add_argument('--image', required=True, help='the task image (tag or ID), built from tasks/<task>/environment')
        p.add_argument('--env-file', required=True, help='researchboost.env (asset and scratch roots)')
        p.add_argument('--gpu-ids', required=True, help='host GPUs, e.g. 0,1')
        p.add_argument('--cpus', default='none', help='none, auto (the GPUs\' NUMA share) or a CPU list')
        p.add_argument('--store', default='packs', help='pack store directory (default %(default)s)')
        p.add_argument('--work', help='work directory root (default ./refpack-work)')
        p.add_argument('--scratch', help="the container's /tmp on the host (default <work>/tmp; the judge may need "
                                         "200 GiB: put it on a large local disk)")
        p.add_argument('--extra-docker-compose', action='append', default=[], help='extra Compose override')
        p.add_argument('--judge-env', action='append', help='K=V for the container (repeatable)')
        p.add_argument('--python', help="the task image's training interpreter")
        p.add_argument('--adapter', help='pack module: generic or container/tasks/<name>.py')
        p.add_argument('--timing-file', help='generic adapter: frozen reference timing, relative to the workspace')
        p.add_argument('--timing-field', help='generic adapter: its field (a number)')
        p.add_argument('--band', type=float, help='generic adapter: band of the timing check (default: bench.json '
                                                  'timing.spread_band)')
        p.add_argument('--dirs', help='generic adapter: workspace directories the preparation writes')
        p.add_argument('--time-tree-wrap', help='generic adapter: `time --tree` takes a directory holding the tree')
        p.add_argument('--keep', action='store_true', help='keep the work directory')
        p.add_argument('--compat', choices=('never', 'auto', 'always'), default='never',
                       help="CUDA forward-compat libraries of the image: never (default: as the task's own "
                            "entrypoint, unless it sets them up), auto (only when newer than the host driver), always")
        p.add_argument('--test-non-root', action='store_true', help='tests only: run without root')
    b = sub.choices['build']
    b.add_argument('--full', action='store_true', help="hide the task's own preparation packs")
    b.add_argument('--spread-max', type=float, help='quality gate: largest reference spread to publish')
    b.add_argument('--spread-file', default='task/timing_ref.json')
    b.add_argument('--spread-key', default='spread')
    j = sub.choices['judge']
    j.add_argument('--artifacts', required=True, help="a finished Harbor trial's artifacts/ directory")
    j.add_argument('--out', required=True, help='empty output directory')
    j.add_argument('--no-guard', action='store_true', help='adversarial tests only: no orphan-killing guard')
    s = sub.add_parser('show')
    s.add_argument('--store', default='packs')
    s.add_argument('--task')
    a = ap.parse_args(argv)
    return {'build': cmd_build, 'verify': cmd_verify, 'judge': cmd_judge, 'show': cmd_show}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
