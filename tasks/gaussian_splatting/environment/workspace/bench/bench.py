"""3D Gaussian Splatting training on 1 GPU: one command per question.

Run from the workspace root (/opt/researchboost/ws) with the training environment's Python (bench.json python). GPU
runs take /tmp/gs_gpu.lock themselves, one at a time; do NOT wrap these commands in flock (that would deadlock).

  python bench/bench.py check   --tree <dir> --out <dir> [--config <file>] [--quick] [--skip-timing] [--n1]
        the verdict (gates 0-3 below) -> <out>/check.json; --quick: gates 0 and 2 only (no timing, no trajectories);
        --skip-timing: no gate 1; --n1: gates 0 and 3 only. Every check runs gate 3 as the final evaluation does; only
        gate 1 differs: the candidate timed once against the frozen reference timing of this host. --final (reference
        -> candidate -> reference) is the final evaluation's: root (the preparation) or the verifier's marker only
  python bench/bench.py time    --tree <dir> --out <dir> [--config <file>] [--paired]
        one timing run of the tree (the full training, timing.steps Gaussian-optimizer steps) -> <out>/time.json:
        seconds per window and in total, peak memory, the speed relative to the frozen reference timing of this
        host (task/timing_ref.json, what gate 1 of the agent's check uses). --paired also times the reference right
        before it, fresh.
  python bench/bench.py profile --tree <dir> --out <dir> [--config <file>] [--from K] [--iters M]
        torch.profiler over Gaussian-optimizer steps K..K+M-1 (default 7000, 20) -> <out>/timing_profile.profile.txt
        (operators by CUDA time) and a Chrome trace
  python bench/bench.py build   --tree <dir>
        builds the tree's CUDA extensions the way every run does (bench.json extensions) and prints the cache directory
  formulator only: record, traj-ref, freeze-inputs, baseline (see --help of each)

Every run: the frozen recipe (bench/recipe.py) through the judge's entry (bench/gs_entry.py), which imports the tree's
train.py and calls its training() on the GPU, with the CUDA extensions built from THIS tree: the three extension
directories of bench.json (the rasterizer, simple-knn, fused-ssim) are copied out of the tree and built with the
interpreter of the run (pip install --no-deps --no-build-isolation --target, TORCH_CUDA_ARCH_LIST 9.0) into a cache
directory named after their source digest; the entry puts that directory first on sys.path. A build failure is an
ERROR of the check. The reference's extensions were built the same way when the image was made.

Timing: the agent's check times the candidate once against the reference timing frozen on this host during
preparation (task/timing_ref.json); the final evaluation (check --final) measures reference -> candidate -> reference
fresh, on the host and in the session where it runs.

Interpreters: the tree under test runs with bench.json candidate_python (the agent's environment, which sees the
training environment's packages unless it installs its own versions); reference runs, the judge's evaluation
(bench/gs_eval.py) and formulator commands run with bench.json python (read-only training environment).

Every GPU run holds the GPU lock, stops processes an earlier bench run left behind, waits until the GPU holds no
foreign memory, runs a fresh copy of the tree in its own process group with a timeout, a fresh TMPDIR and fresh compile
caches, and kills what it started afterwards. Every command checks the frozen inputs declared in
task/formulation.json before (mismatch: nothing is measured) and after (mismatch: ERROR), and the source digest of the
tree copy before and after.

Candidate switches: a JSON file {"environment": {"NAME": "value"}} given with --config; the bench sets those
variables for the candidate's training processes (and its extension builds). Variables the bench manages
(bench.json reserved_env) and GS_BENCH_* are refused.

Verdict (check); each gate stops the evaluation when it fails:
  0. structure  train.py with training(), gaussian_renderer/__init__.py, scene/gaussian_model.py, arguments/ and the
                three extension directories must exist, and the extensions must build (else ERROR).
  1. benefit    fresh runs of the whole training (timing.steps Gaussian-optimizer steps), production settings.
                Time = from the end of step timing.marks[0] to the end of the last step (CUDA synchronized at those
                steps only). The agent's check: the candidate once; speedup = frozen reference time / candidate,
                spread = the frozen reference's. check --final: reference A, candidate, reference B; speedup =
                mean(A, B) / candidate, spread = |A - B| / min(A, B), INVALID above timing.spread_band (the reference
                itself moved). speedup <= max(benefit.min_speedup, 1 + 2 * spread) -> UNAVAILABLE (no_benefit),
                rejected without a precision evaluation.
  2. record     record.steps steps of the recipe with seed 0, observed. Exact items, else INCORRECT: one
                GaussianModel, one Gaussian optimizer, the camera of every training render (the sequence of image
                names) identical to the reference, the learning rate of every parameter group of the Gaussian
                optimizer and the active SH degree at every mark identical, the steps
                at which the number of Gaussians changes (densification and pruning) identical, the parameters'
                dtypes and per-point shapes and the optimizer-state dtypes those of the reference (float32).
                The trajectories of gate 3 check the same items over the whole run, per seed.
  3. statistical one full training run on the first frozen reference seed (bench/traj.py; seed S seeds random,
                numpy and torch), the trained model scored by the judge's own evaluation (bench/gs_eval.py:
                reference renderer, test views, PSNR, SSIM, LPIPS) plus log(number of Gaussians). Each endpoint's
                bias (candidate - reference on that seed) within trajectory.sigma_k x the reference's seed sigma ->
                PASS, tier N1; any beyond -> INCORRECT. The same rule in every check.
There is no tier N0: the rasterizer's backward accumulates with atomics, so even the reference is not bitwise
reproducible from run to run.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import inspect  # noqa: F401  (kept for formulator use)
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent                       # workspace


def _expand(value):
    if isinstance(value, str) and '$' in value:
        out = os.path.expandvars(value)
        if '$' in out:
            raise SystemExit(f'{value}: unset variable')
        return out
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


CFG = _expand(json.loads((BENCH / 'bench.json').read_text()))
sys.path.insert(0, str(BENCH))
import recipe  # noqa: E402
import traj as trajectory  # noqa: E402

LOCK = Path(CFG['gpu_lock'])
PY = Path(CFG['python'])
CAND_PY = Path(CFG['candidate_python'])
SCRATCH = Path(CFG.get('scratch', '/tmp'))
TIMEOUT = CFG['timeouts_s']
RESERVED = set(CFG['reserved_env'])
SIGNATURES = [(re.compile(x['regex']), x['reason']) for x in CFG.get('invalid_run_signatures', [])]
COST = {'gpu_seconds': 0.0, 'gpu_runs': 0}
TREE_SKIP = {'.git', '__pycache__', '.venv', 'output', 'build'}
MARK = 'GS_BENCH_RUN'
STRUCTURE = ('train.py', 'gaussian_renderer/__init__.py', 'scene/gaussian_model.py', 'scene/__init__.py',
             'arguments/__init__.py')


def path(key):
    p = Path(_expand(key))
    return p if p.is_absolute() else (ROOT / p).resolve()


def log(msg):
    print(f'[bench {datetime.now():%H:%M:%S}] {msg}', flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def evaluator_sha256():
    """Joint hash of the frozen evaluator: bench/ code, config, records and trajectories; task/*.json."""
    files = sorted(p for p in BENCH.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files += sorted(p for p in (ROOT / 'task').glob('*.json'))
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode() + b'\0' + sha256_file(p).encode() + b'\n')
    return h.hexdigest()


# ---------------------------------------------------------------- GPU runs

class GpuLock:
    def __init__(self, what, timeout=8 * 3600):
        self.what, self.timeout = what, timeout

    def __enter__(self):
        self.f = open(LOCK, 'a+')
        t0, said = time.time(), False
        while True:
            try:
                fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not said:
                    log(f'waiting for {LOCK} ({self.what}); if this command is wrapped in flock, unwrap it')
                    said = True
                if time.time() - t0 > self.timeout:
                    raise SystemExit(f'could not get {LOCK} within {self.timeout}s for {self.what}')
                time.sleep(5)
        self.t0 = time.time()
        log(f'GPU lock held: {self.what}')
        return self

    def __exit__(self, *_):
        COST['gpu_seconds'] += (time.time() - self.t0) * CFG['world_size']
        COST['gpu_runs'] += 1
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def cost():
    return {'gpu_hours': round(COST['gpu_seconds'] / 3600, 3), 'gpu_runs': COST['gpu_runs'],
            'method': f'GPU lock hold time x {CFG["world_size"]} card (startup and failed runs included)'}


def scratch_dir(prefix):
    free = shutil.disk_usage(SCRATCH).free / 2**30
    if free < CFG.get('min_scratch_free_gib', 20):
        raise SystemExit(f'{SCRATCH} has only {free:.1f} GiB free (< {CFG.get("min_scratch_free_gib", 20)}); '
                         f'clean up before measuring')
    return tempfile.TemporaryDirectory(prefix=prefix, dir=SCRATCH)


def gpu_state():
    q = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu,driver_version',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=60)
    out = []
    for line in q.stdout.strip().splitlines():
        idx, name, mem, util, drv = [x.strip() for x in line.split(',')]
        out.append({'index': int(idx), 'name': name, 'memory_mib': int(mem), 'util': int(util), 'driver': drv})
    return out


def wait_idle(what, patience=600):
    t0 = time.time()
    limit = CFG.get('idle_memory_mib', 2048)
    while True:
        state = gpu_state()
        if len(state) != CFG['world_size']:
            return state, f'{len(state)} GPUs visible, expected {CFG["world_size"]}'
        busy = [g for g in state if g['memory_mib'] > limit]
        if not busy:
            return state, None
        if time.time() - t0 > patience:
            return state, f'GPU busy for {patience}s before {what}: {busy}'
        time.sleep(15)


def _marked(value=None):
    """Processes of this user that carry the bench's mark (any run, or the run `value`)."""
    me, pids = os.getuid(), []
    mark = f'{MARK}='.encode() + (value.encode() if value else b'')
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            env = (d / 'environ').read_bytes().split(b'\0')
            if d.stat().st_uid == me and any((e == mark) if value else e.startswith(mark) for e in env):
                pids.append(int(d.name))
        except OSError:
            continue
    return pids


def _kill_marked(value=None):
    """With value None (called with the GPU lock held, when no other bench run can be live): processes left by an
    earlier bench run that was killed before its cleanup. Otherwise the processes of run `value`."""
    found = _marked(value)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pids = _marked(value)
        if not pids:
            break
        for p in pids:
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                pass
        time.sleep(5)
    if found and value is None:
        log(f'stopped {len(found)} processes left by an earlier bench run')
    return len(_marked(value))


def interpreter_path(candidate, inherited):
    rest = [d for d in inherited.split(':') if d and d not in (str(PY.parent), str(CAND_PY.parent))]
    cuda = str(Path(CFG['cuda_home']) / 'bin')
    rest = [d for d in rest if d != cuda]
    return ':'.join(([str(CAND_PY.parent)] if candidate else []) + [str(PY.parent), cuda] + rest)


def under_test(tree):
    return Path(tree).resolve() != (ROOT / CFG['reference']).resolve()


def base_env(tmp, cand_env=None, candidate=False, ext=None):
    """Environment of every run: the caller's environment minus reserved variables, then the candidate's switches,
    then what the bench manages (scratch, caches, assets, toolkit, PATH, the extension directory)."""
    env = {k: v for k, v in os.environ.items() if k not in RESERVED and not k.startswith('GS_BENCH')}
    env.update(cand_env or {})
    caches = Path(tmp) / 'caches'
    env.update({
        'TMPDIR': str(tmp), 'TMP': str(tmp), 'TEMP': str(tmp),
        'TORCHINDUCTOR_CACHE_DIR': str(caches / 'inductor'), 'TRITON_CACHE_DIR': str(caches / 'triton'),
        'CUDA_CACHE_PATH': str(caches / 'cuda'), 'XDG_CACHE_HOME': str(caches / 'xdg'),
        'MPLCONFIGDIR': str(caches / 'mpl'), 'TORCH_HOME': CFG['torch_home'], 'GS_BENCH_ASSETS': CFG['assets'],
        'PYTHONNOUSERSITE': '1', 'CUDA_HOME': CFG['cuda_home'], 'TRITON_LIBCUDA_PATH': '/usr/lib/x86_64-linux-gnu',
        'PATH': interpreter_path(candidate, env.get('PATH', '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin')),
    })
    if ext:
        env['GS_BENCH_EXT'] = str(ext)
    env.pop('PYTHONPATH', None)
    for v in ('inductor', 'triton', 'cuda', 'xdg', 'mpl'):
        (caches / v).mkdir(parents=True, exist_ok=True)
    return env


def run_process(cmd, cwd, env, log_file, what, timeout):
    """One run under the GPU lock: stale bench processes stopped, idle check, then the command in its own process
    group; everything it started is killed afterwards."""
    tag = uuid.uuid4().hex[:12]
    with GpuLock(what):
        _kill_marked()
        before, problem = wait_idle(what)
        if problem:
            return {'status': 'INVALID', 'reason': problem, 'returncode': None, 'gpus_before': before}
        rc, status, t0 = None, None, time.time()
        try:
            env = dict(env, **{MARK: tag})
            with open(log_file, 'w') as lf:
                proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                try:
                    rc = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    status = 'TIMEOUT'
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            wall = time.time() - t0
            left = _kill_marked(tag)
        after, _ = wait_idle(f'{what} (after)', patience=120)
    res = {'status': status or 'ok', 'returncode': rc, 'wall_s': round(wall, 1), 'gpus_before': before,
           'gpus_after': after, 'log': str(log_file)}
    if status == 'TIMEOUT':
        res['reason'] = f'{what} exceeded {timeout}s; process group killed; see {Path(log_file).name}'
    if left:
        res['warning'] = f'{left} processes of the run survived SIGKILL'
    return res


def run_job(tree, mode, steps, out_prefix, tmp, cand_env, candidate, timeout, what, ext, seed=0, marks=(),
            profile=None):
    extra = ['--profile-from', str(profile[0]), '--profile-iters', str(profile[1])] if profile else []
    cmd = [str(CAND_PY if candidate else PY), str(BENCH / 'gs_entry.py'), '--mode', mode, '--out', str(out_prefix),
           '--steps', str(steps), '--seed', str(seed), '--marks', ','.join(str(m) for m in marks), *extra, '--',
           *recipe.argv()]
    env = base_env(tmp, cand_env, candidate=candidate, ext=ext)
    return run_process(cmd, tree, env, Path(f'{out_prefix}.log'), what, timeout)


def signatures(log_file):
    hits, text = [], Path(log_file).read_text(errors='replace') if Path(log_file).exists() else ''
    for pattern, reason in SIGNATURES:
        m = pattern.search(text)
        if m:
            line = text[text.rfind('\n', 0, m.start()) + 1:text.find('\n', m.end())]
            hits.append({'reason': reason, 'line': line.strip()[:300]})
    return hits


def run_file(prefix):
    f = Path(f'{prefix}.json')
    return json.loads(f.read_text()) if f.exists() else None


def run_error(res, rc, log_file, steps):
    if rc == 0 and res and res.get('status') == 'ok' and res.get('steps') == steps:
        return None
    tail = Path(log_file).read_text(errors='replace')[-1500:] if Path(log_file).exists() else ''
    err = (res or {}).get('error') or 'no error recorded by the entry'
    return f'the run exited {rc} before Gaussian-optimizer step {steps}; {err[:800]}; log tail: {tail}'


# ---------------------------------------------------------------- inputs and sources

def tree_digest(tree, only=None):
    h = hashlib.sha256()
    tree = Path(tree)
    roots = [tree / d for d in only] if only else [tree]
    for root in roots:
        for f in sorted(root.rglob('*')):
            rel = f.relative_to(tree)
            if any(part in TREE_SKIP for part in rel.parts) or f.is_symlink() or not f.is_file():
                continue
            h.update(str(rel).encode() + b'\0' + sha256_file(f).encode() + b'\n')
    return h.hexdigest()


def verify_inputs(when):
    spec = json.loads((ROOT / 'task' / 'formulation.json').read_text())['inputs']
    t0, bad = time.time(), {}

    def one(item):
        rel, want = item
        p = path(rel)
        return rel, want['value'], (sha256_file(p) if p.exists() else None)
    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        for rel, want, got in ex.map(one, spec.items()):
            if got != want:
                bad[rel] = {'want': want, 'got': got}
    log(f'inputs {when}: {len(spec) - len(bad)}/{len(spec)} match the formulation ({time.time() - t0:.0f}s)')
    return {'when': when, 'ok': not bad, 'mismatches': bad, 'checked': len(spec), 'seconds': round(time.time() - t0, 1)}


def freeze_inputs(paths):
    out = {}
    for root in paths:
        root = Path(root)
        files = [root] if root.is_file() else sorted(f for f in root.rglob('*') if f.is_file()
                                                      and not {'__pycache__', '.cache', '.locks'} & set(f.parts))
        for f in files:
            out[str(f)] = {'kind': 'sha256', 'value': None}
    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        for f, h in zip(list(out), ex.map(sha256_file, list(out))):
            out[f]['value'] = h
    return out


def check_reference():
    ref = path(CFG['reference'])
    git = ['git', '-c', 'safe.directory=*', '-C', str(ref)]
    dirty = subprocess.run([*git, 'status', '--porcelain', '--untracked-files=no'], capture_output=True,
                           text=True).stdout.strip()
    head = subprocess.run([*git, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    want = (ROOT / 'task' / 'reference_commit').read_text().strip() if (ROOT / 'task' / 'reference_commit').exists() \
        else None
    if dirty or (want and head != want):
        raise SystemExit(f'reference tree is not the frozen reference commit (head {head}, dirty: {bool(dirty)})')
    return ref


def materialize(src, dest):
    def ignore(directory, names):
        return [n for n in names if n in TREE_SKIP or n.endswith('.pyc') or n.endswith('.so')]
    shutil.copytree(src, dest, ignore=ignore, symlinks=False)
    for d in [Path(dest), *(p for p in Path(dest).rglob('*') if p.is_dir())]:
        d.chmod(d.stat().st_mode | 0o700)
    return Path(dest)


def candidate_environment(config_file):
    if not config_file:
        return {}
    data = json.loads(Path(config_file).read_text())
    extra = sorted(set(data) - {'environment'})
    if extra:
        raise SystemExit(f'{config_file}: only an "environment" mapping is accepted, found {extra}')
    env = {str(k): str(v) for k, v in (data.get('environment') or {}).items()}
    bad = sorted(k for k in env if k in RESERVED or k.startswith('GS_BENCH'))
    if bad:
        raise SystemExit(f'{config_file}: {bad} are managed by the bench and cannot be candidate switches')
    return env


VERSIONS = """import importlib.metadata, json, sys
out = {'python': sys.version.split()[0]}
for pkg in sys.argv[1:]:
    try:
        out[pkg] = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        out[pkg] = None
print(json.dumps(out))
"""
_VERSIONS = {}


def versions(python):
    if python not in _VERSIONS:
        r = subprocess.run([str(python), '-c', VERSIONS, 'torch', 'torchvision', 'numpy', 'plyfile', 'opencv-python-headless'],
                           capture_output=True, text=True, timeout=300, env=dict(os.environ, PYTHONNOUSERSITE='1'))
        _VERSIONS[python] = (json.loads(r.stdout.strip().splitlines()[-1]) if not r.returncode else
                             {'error': f'{python} exited {r.returncode}: {r.stderr[-500:]}'})
    return _VERSIONS[python]


# ---------------------------------------------------------------- the CUDA extensions

def build_extensions(tree, cand_env=None, candidate=True):
    """The tree's CUDA extensions, built with the run's interpreter into a cache directory named after their source
    digest (and the interpreter and switches). Returns (directory, None) or (None, failure reason)."""
    if not candidate:
        return Path(CFG['reference_ext']), None
    exts = CFG['extensions']
    missing = [e for e in exts if not (Path(tree) / e / 'setup.py').is_file()]
    if missing:
        return None, f'extension directories without setup.py: {missing}'
    key = hashlib.sha256((tree_digest(tree, exts) + str(CAND_PY) + json.dumps(cand_env or {}, sort_keys=True))
                         .encode()).hexdigest()[:24]
    cache = Path(CFG['ext_cache'])
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / key
    if (target / '.complete').exists():
        return target, None
    log_file = cache / f'{key}.log'
    with tempfile.TemporaryDirectory(prefix='gs-build-', dir=SCRATCH) as tmp:
        staging = Path(tmp) / 'site'
        env = base_env(tmp, cand_env, candidate=True)
        env.update({'TORCH_CUDA_ARCH_LIST': '9.0', 'MAX_JOBS': str(min(16, os.cpu_count() or 8))})
        with open(log_file, 'w') as lf:
            for e in exts:
                src = shutil.copytree(Path(tree) / e, Path(tmp) / 'src' / Path(e).name)
                cmd = [str(CAND_PY), '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '--no-cache-dir',
                       '--disable-pip-version-check', '--target', str(staging), str(src)]
                lf.write(f'$ {" ".join(cmd)}\n')
                lf.flush()
                try:
                    r = subprocess.run(cmd, cwd=tmp, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                       timeout=TIMEOUT['build'])
                except subprocess.TimeoutExpired:
                    return None, f'building {e} exceeded {TIMEOUT["build"]}s; see {log_file}'
                if r.returncode:
                    tail = log_file.read_text(errors='replace')[-1500:]
                    return None, f'building {e} failed (exit {r.returncode}); see {log_file}; tail: {tail}'
        (staging / '.complete').write_text(json.dumps({'key': key, 'built': datetime.now().isoformat()}) + '\n')
        try:
            staging.rename(target)
        except OSError:
            if not (target / '.complete').exists():
                raise
    return target, None


# ---------------------------------------------------------------- gate 1: timing

def windows(stamps):
    marks = CFG['timing']['marks']
    return [stamps[str(b)]['t'] - stamps[str(a)]['t'] for a, b in zip(marks, marks[1:])]


def time_tree(tree, out, label, cand_env=None, candidate=True, profile=None):
    # a profiled run traces steps K..K+M-1 and ends after step K+M-1 (gs_entry.py)
    steps = CFG['timing']['steps'] if not profile else profile[0] + profile[1] - 1
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f'timing_{label}'
    ext, problem = build_extensions(tree, cand_env, candidate)
    if problem:
        return {'status': 'ERROR', 'stage': 'build', 'reason': problem}
    with scratch_dir('gs-t-') as tmp:
        r = run_job(tree, 'time', steps, prefix, tmp, cand_env, candidate, TIMEOUT['timing'],
                    f'timing {label}' + (' (profile)' if profile else ''), ext,
                    marks=[] if profile else CFG['timing']['marks'], profile=profile)
    fingerprint = {'argv': ['gs_entry.py --mode time', *recipe.argv()],
                   'candidate_environment': cand_env or {},
                   'gpus': [(g['index'], g['name'], g['driver']) for g in r.get('gpus_before') or []],
                   'instrumentation': 'torch.profiler' if profile else 'none',
                   'timing_plan': [steps, CFG['timing']['marks']],
                   'versions': versions(CAND_PY if candidate else PY)}
    run = {'run_status': r['status'], 'wall_s': r.get('wall_s'), 'fingerprint': fingerprint, 'log': r.get('log'),
           'extensions': str(ext), **({'warning': r['warning']} if r.get('warning') else {})}
    if r['status'] != 'ok':
        return {'status': r['status'], 'reason': r['reason'], **run}
    res = run_file(prefix)
    err = run_error(res, r['returncode'], r['log'], steps)
    if err:
        return {'status': 'ERROR', 'reason': err, **run}
    hits = signatures(r['log'])
    if hits:
        return {'status': 'INVALID', 'reason': 'invalid-run signature in the log: ' + hits[0]['reason'],
                'signatures': hits, **run}
    out_res = {'status': 'ok', 'steps': steps,
               'peak_memory_allocated_gib': round(res['peak_memory_allocated_bytes'] / 2**30, 2),
               'peak_memory_reserved_gib': round(res['peak_memory_reserved_bytes'] / 2**30, 2),
               'startup_to_first_step_s': round(res['startup_to_first_step_s'], 1),
               'other_optimizer_steps': res.get('other_optimizer_steps'), **run}
    if not profile:
        w = windows(res['marks'])
        out_res.update(window_s=[round(x, 3) for x in w], total_s=round(sum(w), 3),
                       steps_per_s=round((CFG['timing']['marks'][-1] - CFG['timing']['marks'][0]) / sum(w), 2))
    return out_res


def frozen_reference_timing():
    f = ROOT / 'task' / 'timing_ref.json'
    return json.loads(f.read_text()) if f.exists() else None


def frozen_speed(res):
    ref = frozen_reference_timing()
    if res.get('status') == 'ok' and res.get('total_s') and ref:
        res['speedup_vs_frozen_reference'] = round(ref['total_s'] / res['total_s'], 4)
        res['frozen_reference_note'] = ('the reference measured on this host during preparation, what gate 1 of the '
                                        'agent\'s check compares with; the final evaluation times reference -> '
                                        'candidate -> reference')
    return res


FINAL_MARKER = Path('/run/researchboost/final-evaluation')


def final_allowed():
    """`check --final` (reference -> candidate -> reference timing) is the final evaluation's: allowed for root (the
    preparation) and for the verifier, which writes FINAL_MARKER as root in a root-owned directory before it runs the
    judge as the agent user. The agent cannot create it."""
    if os.geteuid() == 0:
        return True
    try:
        st, parent = FINAL_MARKER.stat(), FINAL_MARKER.parent.stat()
    except OSError:
        return False
    return st.st_uid == 0 and parent.st_uid == 0 and not parent.st_mode & 0o022


def benefit_frozen(tree, out, cand_env):
    """Gate 1 of the agent's check: one candidate run against the frozen reference timing of the preparation
    (task/timing_ref.json: the faster of its two reference runs, and their spread). The bar is the final check's
    (min_speedup, or the frozen reference's spread doubled, over one: the larger). The reference -> candidate ->
    reference comparison is the final evaluation's (check --final)."""
    frozen = frozen_reference_timing()
    if not frozen or not frozen.get('total_s'):
        return 'ERROR', {'speedup': None, 'runs': {}, 'reason': 'no frozen reference timing (task/timing_ref.json)'}
    cand = frozen_speed(time_tree(tree, out, 'cand', cand_env=cand_env))
    runs = {'cand': cand}
    if cand['status'] != 'ok':
        status = 'INVALID' if cand['status'] in ('INVALID', 'TIMEOUT') else 'ERROR'
        return status, {'speedup': None, 'runs': runs, 'reason': cand.get('reason') or 'the timing run failed'}
    spread = frozen.get('spread') or 0.0
    speedup = cand['speedup_vs_frozen_reference']
    perf = {'marks': CFG['timing']['marks'], 'reference_s': frozen['total_s'], 'candidate_s': cand['total_s'],
            'candidate_window_s': cand['window_s'], 'speedup': speedup, 'reference_spread': spread,
            'reference': frozen.get('source') or 'task/timing_ref.json',
            'comparison': 'candidate run against the frozen reference timing (the final evaluation times '
                          'reference -> candidate -> reference)', 'runs': runs,
            'library_versions': {'candidate': cand['fingerprint']['versions']}, 'host': socket.gethostname()}
    bar = max(CFG['benefit']['min_speedup'], 1 + 2 * spread)
    perf['benefit'] = {'speedup': speedup, 'speedup_bar': round(bar, 4),
                       'peak_memory_allocated_gib': cand['peak_memory_allocated_gib'], 'passed': speedup > bar}
    return ('ok' if perf['benefit']['passed'] else 'UNAVAILABLE'), perf


def benefit(tree, out, cand_env, final=False):
    """Gate 1. The agent's check: benefit_frozen. The final evaluation (check --final, also the preparation's full
    check of the unchanged tree): reference A -> candidate -> reference B, fresh."""
    if not final:
        return benefit_frozen(tree, out, cand_env)
    t = CFG['timing']
    ref_src = check_reference()
    with scratch_dir('gs-ref-') as tmp:
        ref_tree = materialize(ref_src, Path(tmp) / 'reference')
        ref_a = time_tree(ref_tree, out, 'ref_a', candidate=False)
        cand = time_tree(tree, out, 'cand', cand_env=cand_env) if ref_a['status'] == 'ok' else None
        ref_b = time_tree(ref_tree, out, 'ref_b', candidate=False) if cand and cand['status'] == 'ok' else None
    runs = {'ref_a': ref_a, 'cand': cand, 'ref_b': ref_b}
    bad = {k: v for k, v in runs.items() if v is None or v['status'] != 'ok'}
    if bad:
        status = 'INVALID' if any(v and v['status'] in ('INVALID', 'TIMEOUT') for v in bad.values()) else 'ERROR'
        reason = next((v.get('reason') for v in bad.values() if v and v.get('reason')), 'a timing run failed')
        return status, {'speedup': None, 'runs': runs, 'reason': reason}
    fp = ('argv', 'gpus', 'instrumentation', 'timing_plan')
    diffs = {k: [x for x in fp if ref_a['fingerprint'].get(x) != v['fingerprint'].get(x)]
             for k, v in runs.items() if k != 'ref_a'}
    versions_seen = {'reference': ref_a['fingerprint']['versions'], 'candidate': cand['fingerprint']['versions']}
    versions_seen['differ'] = versions_seen['reference'] != versions_seen['candidate']
    badfp = {k: d for k, d in diffs.items() if d}
    if badfp:
        return 'ERROR', {'speedup': None, 'runs': runs, 'fingerprint_diffs': diffs, 'library_versions': versions_seen,
                         'reason': f'timing runs not comparable: fingerprint differs from ref_a in {badfp}'}
    A, B, C = ref_a['total_s'], ref_b['total_s'], cand['total_s']
    spread = abs(A - B) / min(A, B)
    perf = {'marks': t['marks'], 'reference_a_s': A, 'reference_b_s': B, 'candidate_s': C,
            'reference_a_window_s': ref_a['window_s'], 'reference_b_window_s': ref_b['window_s'],
            'candidate_window_s': cand['window_s'],
            'window_ratios': [round((a + b) / 2 / c, 4) for a, b, c in
                              zip(ref_a['window_s'], ref_b['window_s'], cand['window_s'])],
            'speedup': round((A + B) / 2 / C, 4), 'reference_spread': round(spread, 4), 'runs': runs,
            'fingerprint_diffs': diffs, 'library_versions': versions_seen, 'host': socket.gethostname(),
            'reference_s': round((A + B) / 2, 3)}
    if spread > t['spread_band']:
        perf['problems'] = [f'the two reference runs differ by {spread:.1%} (> {t["spread_band"]:.0%}): the host '
                            f'moved between them; measure again later']
        return 'INVALID', perf
    bar = max(CFG['benefit']['min_speedup'], 1 + 2 * spread)
    perf['benefit'] = {'speedup': perf['speedup'], 'speedup_bar': round(bar, 4),
                       'peak_memory_allocated_gib': cand['peak_memory_allocated_gib'],
                       'reference_peak_memory_allocated_gib': max(ref_a['peak_memory_allocated_gib'],
                                                                  ref_b['peak_memory_allocated_gib']),
                       'passed': perf['speedup'] > bar}
    return ('ok' if perf['benefit']['passed'] else 'UNAVAILABLE'), perf


# ---------------------------------------------------------------- gate 2: exact items

def observed_run(tree, out, label, mode, steps, marks, seed, cand_env=None, candidate=True, timeout=None):
    """One observed run (record or traj) -> the entry's result dict (status ok) or a failure dict."""
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f'{mode}_{label}_s{seed}'
    ext, problem = build_extensions(tree, cand_env, candidate)
    if problem:
        return {'status': 'ERROR', 'stage': 'build', 'reason': problem}
    with scratch_dir('gs-r-') as tmp:
        r = run_job(tree, mode, steps, prefix, tmp, cand_env, candidate, timeout or TIMEOUT[mode],
                    f'{mode} {label} seed {seed}', ext, seed=seed, marks=marks)
    if r['status'] != 'ok':
        return {'status': r['status'], 'reason': r['reason'], 'log': r.get('log')}
    res = run_file(prefix)
    err = run_error(res, r['returncode'], r['log'], steps)
    if err:
        return {'status': 'ERROR', 'reason': err, 'log': r.get('log')}
    res.update(wall_s=r.get('wall_s'), log=r.get('log'), prefix=str(prefix))
    return res


def items(run):
    """The exact items of one observed run (what is compared with the reference's)."""
    return {'steps': run['steps'], 'models': run['models'], 'gaussian_optimizers': run['gaussian_optimizers'],
            'other_optimizer_steps': run['other_optimizer_steps'], 'cameras': run['cameras'],
            'marks': run['marks'], 'count_change_steps': [k for k, _ in run['counts']],
            'dtypes': run['dtypes']}


def compare_items(ref, cand):
    problems = []
    if cand['steps'] != ref['steps']:
        problems.append(f'{cand["steps"]} Gaussian-optimizer steps, the reference {ref["steps"]}')
    if cand['models'] != 1:
        problems.append(f'{cand["models"]} GaussianModel instances created (exactly one expected)')
    if cand['gaussian_optimizers'] != 1:
        problems.append(f'{cand["gaussian_optimizers"]} Gaussian optimizers stepped (exactly one expected)')
    rc, cc = ref['cameras'], cand['cameras']
    if len(cc) != len(rc):
        problems.append(f'{len(cc)} training renders through gaussian_renderer.render, the reference {len(rc)} '
                        f'(one per iteration)')
    else:
        first = next((i for i, (a, b) in enumerate(zip(rc, cc)) if a != b), None)
        if first is not None:
            problems.append(f'the camera of iteration {first + 1} is {cc[first]}, the reference\'s {rc[first]}: the '
                            f'camera sequence differs')
    for m, want in ref['marks'].items():
        got = {k: v for k, v in (cand['marks'].get(m) or {}).items() if k in ('lr', 'active_sh_degree')}
        want = {k: v for k, v in want.items() if k in ('lr', 'active_sh_degree')}
        if got != want:
            problems.append(f'at step {m}: learning rates / SH degree {got}, the reference {want}')
            break
    if cand['count_change_steps'] != ref['count_change_steps']:
        a, b = ref['count_change_steps'], cand['count_change_steps']
        problems.append(f'the number of Gaussians changes at different steps (densification / pruning): reference '
                        f'{a[:6]}... ({len(a)} changes), candidate {b[:6]}... ({len(b)} changes)')
    if cand['dtypes'] != ref['dtypes']:
        problems.append(f'parameter / optimizer-state dtypes or per-point shapes {cand["dtypes"]}, the reference '
                        f'{ref["dtypes"]}')
    return problems


def numerics(tree, out, cand_env):
    """Gate 2 (quick record): record.steps steps with seed 0 against the frozen reference record."""
    rc = CFG['record']
    ref = json.loads(path(rc['reference']).read_text())
    cand = observed_run(tree, out, 'cand', 'record', rc['steps'], rc['marks'], 0, cand_env=cand_env)
    if cand.get('status') != 'ok':
        return {'status': cand.get('status', 'ERROR'), 'reason': cand.get('reason'), 'log': cand.get('log'),
                'stage': cand.get('stage', 'record')}
    problems = compare_items(ref, items(cand))
    res = {'status': 'INCORRECT' if problems else 'PASS', 'exact_items_ok': not problems, 'problems': problems,
           'steps': rc['steps'], 'run': {k: cand.get(k) for k in ('wall_s', 'final_count', 'log')},
           'measured_tier': None}
    return res


# ---------------------------------------------------------------- gate 3: trajectories

def evaluate_gaussians(files, out, label):
    """The judge's evaluation (bench/gs_eval.py, reference interpreter, reference tree and extensions)."""
    ref_src = check_reference()
    with scratch_dir('gs-e-') as tmp:
        tree = materialize(ref_src, Path(tmp) / 'reference')
        dest = out / f'eval_{label}.json'
        cmd = [str(PY), str(BENCH / 'gs_eval.py'), '--gaussians', *[str(f) for f in files], '--out', str(dest),
               '--model-path', str(Path(tmp) / 'model'), '--', *recipe.argv()]
        env = base_env(tmp, None, candidate=False, ext=Path(CFG['reference_ext']))
        r = run_process(cmd, tree, env, out / f'eval_{label}.log', f'evaluation {label}', TIMEOUT['eval'])
    if r['status'] != 'ok' or r['returncode'] != 0 or not dest.exists():
        tail = Path(r.get('log') or '').read_text(errors='replace')[-1500:] if r.get('log') else ''
        return {'status': r['status'] if r['status'] != 'ok' else 'ERROR',
                'reason': r.get('reason') or f'the evaluation exited {r["returncode"]}: {tail}'}
    return json.loads(dest.read_text())


def traj_run(tree, out, label, seed, cand_env=None, candidate=True):
    tj = CFG['trajectory']
    run = observed_run(tree, out, label, 'traj', tj['steps'], tj['marks'], seed, cand_env=cand_env,
                       candidate=candidate)
    if run.get('status') != 'ok':
        return run
    ev = evaluate_gaussians([f'{run["prefix"]}.gaussians.pt'], out, f'{label}_s{seed}')
    if ev.get('status') != 'ok':
        return {'status': ev.get('status', 'ERROR'), 'reason': ev.get('reason'), 'stage': 'evaluation'}
    Path(f'{run["prefix"]}.gaussians.pt').unlink(missing_ok=True)
    return {'status': 'ok', 'items': items(run), 'result': ev['results'][0], 'wall_s': run.get('wall_s'),
            'log': run.get('log')}


def reference_items(seed):
    f = path(CFG['trajectory']['reference_records']) / f'ref_s{seed}.json'
    return json.loads(f.read_text())


def trajectories(tree, out, cand_env):
    """Gate 3 (and gate 2 over the whole run, per seed): the first trajectory.seeds frozen reference seeds (one), each
    endpoint's bias within trajectory.sigma_k x sigma_seed (traj.bound). The same in every check."""
    tj = CFG['trajectory']
    frozen = json.loads(path(tj['reference']).read_text())
    results = {}
    for seed in sorted(int(s) for s in frozen['reference_values'])[:tj['seeds']]:
        run = traj_run(tree, out, 'cand', seed, cand_env=cand_env)
        if run['status'] != 'ok':
            return {'verdict': 'error', 'status': run['status'], 'reason': run.get('reason'), 'seed': seed,
                    'log': run.get('log')}
        problems = compare_items(reference_items(seed), run['items'])
        if problems:
            return {'verdict': 'exact_items', 'seed': seed, 'problems': problems}
        results[seed] = run['result']
    report = trajectory.bound(frozen, results, k=tj['sigma_k'])
    (out / 'traj_report.json').write_text(json.dumps(report, indent=1) + '\n')
    report['results'] = {str(s): v for s, v in results.items()}
    return report


# ---------------------------------------------------------------- verdict

def coverage(tree, cand_env):
    missing = [f for f in STRUCTURE if not (Path(tree) / f).is_file()]
    missing += [e for e in CFG['extensions'] if not (Path(tree) / e / 'setup.py').is_file()]
    if missing:
        return 'ERROR', {'passed': False, 'stage': 'structure', 'note': f'missing: {missing}'}
    if not re.search(r'^def training\(', (Path(tree) / 'train.py').read_text(errors='replace'), re.M):
        return 'ERROR', {'passed': False, 'stage': 'structure', 'note': 'train.py has no top-level training()'}
    _, problem = build_extensions(tree, cand_env, True)
    if problem:
        return 'ERROR', {'passed': False, 'stage': 'build', 'note': problem}
    return None


def statistical(tree, out, cand_env, correctness, performance):
    """Gate 3 and its verdict (every check, check --n1 included)."""
    traj = trajectories(tree, out, cand_env)
    if traj.get('verdict') == 'exact_items':
        return 'INCORRECT', dict(correctness, passed=False, verdict='exact_items', stage='statistical',
                                 note=f'seed {traj["seed"]}: ' + '; '.join(traj['problems'])), performance
    correctness = dict(correctness, stage='statistical', trajectory=traj)
    if traj['verdict'] == 'equivalent':
        return 'PASS', dict(correctness, passed=True, verdict='equivalent', measured_tier='N1'), performance
    if traj['verdict'] == 'error':
        return ('INVALID' if traj.get('status') in ('INVALID', 'TIMEOUT') else 'ERROR'), \
            dict(correctness, passed=False, note=traj.get('reason') or traj.get('error') or 'a trajectory run failed'), \
            performance
    return 'INCORRECT', dict(correctness, passed=False, verdict='different',
                             note=f'an endpoint\'s bias on seed {traj["seeds"]} is beyond '
                                  f'{CFG["trajectory"]["sigma_k"]} x the reference\'s seed sigma'), performance


def evaluate(tree, out, skip_timing=False, cand_env=None, quick=False, final=False, n1=False):
    """Gates 0 -> 1 -> 2 -> 3 (n1: gates 0 and 3 only). Returns (status, correctness, performance)."""
    cand_env = cand_env or {}
    failed = coverage(tree, cand_env)
    if failed:
        return failed[0], failed[1], None
    if n1:
        return statistical(tree, out, cand_env, {'measured_tier': None, 'gates_run': [0, 3],
                                                 'note': 'check --n1: gates 0 and 3 only; gates 1 and 2 not run (the '
                                                         'final evaluation runs every gate)'}, None)
    performance = None
    if not (quick or skip_timing):
        status, performance = benefit(tree, out, cand_env, final=final)
        if status in ('ERROR', 'INVALID'):
            return status, {'passed': False, 'stage': 'benefit',
                            'note': (performance or {}).get('reason') or '; '.join((performance or {}).get('problems')
                                                                                 or []) or 'timing failed'}, performance
        if status == 'UNAVAILABLE':
            performance['measured_speedup'] = performance.pop('speedup')
            return 'UNAVAILABLE', {'passed': None, 'stage': 'benefit', 'verdict': 'no_benefit',
                                   'note': 'not faster than the reference: rejected without a precision evaluation'}, \
                performance
    num = numerics(tree, out, cand_env)
    correctness = {'stage': 'record', 'measured_tier': None, 'numerics': num}
    if num.get('status') not in ('PASS', 'INCORRECT'):
        st = 'INVALID' if num.get('status') in ('INVALID', 'TIMEOUT') else 'ERROR'
        return st, dict(correctness, passed=False, note=num.get('reason') or 'the record run failed'), performance
    if num['status'] == 'INCORRECT':
        return 'INCORRECT', dict(correctness, passed=False, verdict='exact_items', note='; '.join(num['problems'])), \
            performance
    if quick:
        return 'PASS', dict(correctness, passed=None, verdict='exact_items_pass', stage='record (quick)',
                            note='the exact items of the first record.steps steps match the reference; the statistical '
                                 'gate (a full check or --skip-timing) decides'), performance
    return statistical(tree, out, cand_env, correctness, performance)


def guarded(out, fn, tree_src=None):
    out.mkdir(parents=True, exist_ok=True)
    start = verify_inputs('start')
    if not start['ok']:
        return {'status': 'ERROR', 'stage': 'inputs', 'inputs': [start],
                'note': 'a frozen input differs from task/formulation.json; nothing was measured'}
    if tree_src is None:
        try:
            res = fn(None)
        except SystemExit as exc:
            res = {'status': 'ERROR', 'note': f'the evaluation stopped: {exc}'}
    else:
        with scratch_dir('gs-src-') as tmp:
            tree = materialize(tree_src, Path(tmp) / 'tree')
            digest = tree_digest(tree)
            try:
                res = fn(tree)
            except SystemExit as exc:
                res = {'status': 'ERROR', 'note': f'the evaluation stopped: {exc}'}
            res['source_digest'] = digest
            if tree_digest(tree) != digest:
                res['status'] = 'ERROR'
                res['note'] = 'the tree changed while it was being measured (the code wrote into its own tree)'
    end = verify_inputs('end')
    res['inputs'] = [start, end]
    if not end['ok']:
        res['status'] = 'ERROR'
        res['note'] = 'a frozen input changed during the run'
    return res


def finish(out, name, res):
    res['cost'] = cost()
    res['evaluator_sha256'] = evaluator_sha256()
    res['host'] = {'hostname': socket.gethostname(), 'gpus': [(g['index'], g['name']) for g in gpu_state()]}
    res['finished'] = datetime.now().isoformat(timespec='seconds')
    (out / name).write_text(json.dumps(res, indent=1, default=str) + '\n')
    log(f'wrote {out / name}: status {res.get("status")}')
    summary = {'status': res.get('status')}
    perf = res.get('performance')
    if isinstance(perf, dict):
        summary['speedup'] = perf.get('speedup', perf.get('measured_speedup'))
    corr = res.get('correctness')
    if isinstance(corr, dict) and corr.get('measured_tier'):
        summary['tier'] = corr['measured_tier']
    for k in ('speedup_vs_frozen_reference', 'speedup_vs_paired_reference', 'total_s'):
        if k in res:
            summary[k] = res[k]
    print(json.dumps(summary))
    return 0 if res.get('status') in ('PASS', 'ok') else 1


def cmd_check(a):
    if a.final and not final_allowed():
        raise SystemExit('check --final is the final evaluation\'s (reference -> candidate -> reference); it runs after '
                         'your working time. Your check runs every gate with the same rules; only gate 1 times your '
                         'tree once against the frozen reference timing.')
    if a.n1 and (a.final or a.quick or a.skip_timing):
        raise SystemExit('check --n1 runs gates 0 and 3 only; it does not combine with --final, --quick or '
                         '--skip-timing')
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)

    def run(tree):
        status, correctness, performance = evaluate(tree, out, skip_timing=a.skip_timing, cand_env=cand_env,
                                                    quick=a.quick, final=a.final, n1=a.n1)
        return {'status': status, 'tree': str(Path(a.tree).resolve()), 'config': a.config, 'environment': cand_env,
                'quick': a.quick, 'skip_timing': a.skip_timing, 'final': a.final, 'n1': a.n1,
                'correctness': correctness, 'performance': performance}
    return finish(out, 'check.json', guarded(out, run, Path(a.tree).resolve()))


def cmd_time(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)

    def run(tree):
        ref = None
        if a.paired:
            with scratch_dir('gs-ref-') as tmp:
                ref = time_tree(materialize(check_reference(), Path(tmp) / 'reference'), out, 'reference',
                                candidate=False)
        res = frozen_speed(time_tree(tree, out, 'tree', cand_env=cand_env, candidate=under_test(a.tree)))
        if ref is not None:
            res['paired_reference'] = ref
            if res['status'] == 'ok' and ref['status'] == 'ok':
                res['speedup_vs_paired_reference'] = round(ref['total_s'] / res['total_s'], 4)
                res['note'] = ('screening only: one reference run just before, same host and session; the verdict is '
                               'the full check (reference -> candidate -> reference)')
        return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env,
                    host=socket.gethostname())
    return finish(out, 'time.json', guarded(out, run, Path(a.tree).resolve()))


def cmd_profile(a):
    if a.start < 2 or a.iters < 1:
        raise SystemExit('profile: --from must be at least 2 (the trace starts after step K-1) and --iters at least 1')
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)

    def run(tree):
        res = time_tree(tree, out, 'profile', cand_env=cand_env, candidate=under_test(a.tree),
                        profile=(a.start, a.iters))
        if res['status'] == 'ok':
            prof = out / 'timing_profile.profile.txt'
            res['profile_table'] = str(prof) if prof.exists() else None
            res['trace'] = str(out / 'timing_profile.trace.json')
            res['note'] = (f'Gaussian-optimizer steps {a.start}..{a.start + a.iters - 1} under torch.profiler; '
                           f'instrumented: not comparable with `time`')
        return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env)
    return finish(out, 'profile.json', guarded(out, run, Path(a.tree).resolve()))


def cmd_build(a):
    with scratch_dir('gs-src-') as tmp:
        tree = materialize(Path(a.tree).resolve(), Path(tmp) / 'tree')
        ext, problem = build_extensions(tree, candidate_environment(a.config), under_test(a.tree))
    print(json.dumps({'status': 'ok' if not problem else 'ERROR', 'extensions': str(ext) if ext else None,
                      'reason': problem}))
    return 0 if not problem else 1


def cmd_baseline(a):
    out = Path(a.out).resolve()

    def run(_):
        ref_src = check_reference()
        with scratch_dir('gs-ref-') as tmp:
            tree = materialize(ref_src, Path(tmp) / 'reference')
            runs = {k: time_tree(tree, out, k, candidate=False) for k in a.labels.split(',')}
        ok = [v for v in runs.values() if v['status'] == 'ok']
        res = {'status': 'ok' if len(ok) == len(runs) else 'ERROR', 'runs': runs}
        if len(ok) == len(runs):
            tot = [v['total_s'] for v in runs.values()]
            res['total_s'] = round(min(tot), 3)
            res['spread'] = round((max(tot) - min(tot)) / min(tot), 4)
            res['totals'] = tot
        return res
    return finish(out, 'baseline.json', guarded(out, run))


def cmd_record(a):
    """Formulator: the reference's quick record (record.steps steps, seed 0) -> <out>/record.json."""
    out = Path(a.out).resolve()
    rc = CFG['record']
    with scratch_dir('gs-src-') as tmp:
        tree = materialize(Path(a.tree).resolve(), Path(tmp) / 'tree')
        run = observed_run(tree, out, a.label, 'record', rc['steps'], rc['marks'], 0, candidate=under_test(a.tree))
    rec = dict(items(run), status='ok', final_count=run.get('final_count')) if run.get('status') == 'ok' else run
    (out / 'record.json').write_text(json.dumps(rec) + '\n')
    print(json.dumps({'status': rec.get('status'), 'out': str(out / 'record.json'), 'reason': rec.get('reason'),
                      'cost': cost()}))
    return 0 if rec.get('status') == 'ok' else 1


def cmd_compare(a):
    """Formulator: compare the exact items of two record.json files."""
    problems = compare_items(json.loads(Path(a.ref).read_text()), json.loads(Path(a.cand).read_text()))
    print(json.dumps({'problems': problems}, indent=1))
    return 0 if not problems else 1


def cmd_traj_ref(a):
    """Formulator: the reference's full runs for seeds 1..N -> their exact items (trajectory.reference_records) and
    evaluations, and delta frozen from their spread -> trajectory.reference."""
    tj = CFG['trajectory']
    out = Path(a.out).resolve()
    seeds = tj['reference_seeds'][:a.seeds] if a.seeds else tj['reference_seeds']
    recs = path(tj['reference_records'])
    recs.mkdir(parents=True, exist_ok=True)
    results = {}
    with scratch_dir('gs-src-') as tmp:
        tree = materialize(check_reference(), Path(tmp) / 'tree')
        for seed in seeds:
            run = traj_run(tree, out, 'ref', seed, candidate=False)
            if run['status'] != 'ok':
                print(json.dumps({'status': run['status'], 'seed': seed, 'reason': run.get('reason')}))
                return 1
            (recs / f'ref_s{seed}.json').write_text(json.dumps(run['items']) + '\n')
            results[seed] = run['result']
            log(f'reference seed {seed}: PSNR {run["result"]["psnr"]:.3f} SSIM {run["result"]["ssim"]:.4f} '
                f'LPIPS {run["result"]["lpips"]:.4f} Gaussians {run["result"]["count"]} ({run["wall_s"]:.0f}s)')
    frozen = trajectory.freeze(results, tj['endpoints'], k=tj['sigma_k'])
    frozen['results'] = {str(s): r for s, r in results.items()}
    path(tj['reference']).write_text(json.dumps(frozen, indent=1) + '\n')
    print(json.dumps({'status': 'ok', 'endpoints': frozen['endpoints'], 'cost': cost()}))
    return 0


def cmd_freeze_inputs(a):
    inputs = freeze_inputs(a.paths)
    f = ROOT / 'task' / 'formulation.json'
    data = json.loads(f.read_text()) if f.exists() else {}
    data['inputs'] = inputs
    f.write_text(json.dumps(data, indent=1) + '\n')
    print(json.dumps({'files': len(inputs)}))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    k = sub.add_parser('check')
    k.add_argument('--tree', required=True)
    k.add_argument('--out', required=True)
    k.add_argument('--config')
    k.add_argument('--quick', action='store_true')
    k.add_argument('--skip-timing', action='store_true')
    k.add_argument('--final', action='store_true', help=argparse.SUPPRESS)   # the final evaluation's (root / verifier)
    k.add_argument('--n1', action='store_true', help='gates 0 and 3 only (no timing, no record)')
    t = sub.add_parser('time')
    t.add_argument('--tree', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--config')
    t.add_argument('--paired', action='store_true', help='also time the reference right before (same host, fresh)')
    f = sub.add_parser('profile')
    f.add_argument('--tree', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--config')
    f.add_argument('--from', dest='start', type=int, default=7000)
    f.add_argument('--iters', type=int, default=20)
    bd = sub.add_parser('build', help='build the tree\'s CUDA extensions as every run does')
    bd.add_argument('--tree', required=True)
    bd.add_argument('--config')
    b = sub.add_parser('baseline')
    b.add_argument('--out', required=True)
    b.add_argument('--labels', default='ref_a,ref_b,ref_c')
    r = sub.add_parser('record', help="formulator: the reference's quick record")
    r.add_argument('--tree', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--label', default='ref')
    c = sub.add_parser('compare', help='formulator: compare the exact items of two record.json files')
    c.add_argument('ref')
    c.add_argument('cand')
    j = sub.add_parser('traj-ref', help="formulator: the reference's full runs and the frozen delta")
    j.add_argument('--out', required=True)
    j.add_argument('--seeds', type=int, default=0)
    fi = sub.add_parser('freeze-inputs', help='formulator: write the frozen input fingerprints')
    fi.add_argument('paths', nargs='+')
    a = p.parse_args()
    return {'freeze-inputs': cmd_freeze_inputs, 'check': cmd_check, 'time': cmd_time, 'profile': cmd_profile,
            'build': cmd_build, 'baseline': cmd_baseline, 'record': cmd_record, 'compare': cmd_compare,
            'traj-ref': cmd_traj_ref}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
