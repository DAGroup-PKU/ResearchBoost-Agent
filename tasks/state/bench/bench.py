"""State training bench on one A800: one command per question.

  python bench/bench.py check   --tree <dir> --out <dir> [--config <file>] [--quick] [--skip-timing]
  python bench/bench.py time    --tree <dir> --out <dir> [--config <file>]
  python bench/bench.py profile --tree <dir> --out <dir> [--config <file>] [--nsys] [--iters N]
  python bench/bench.py baseline --out <dir>          (formulator) the reference timed three times

Verdict (check); gates run in order and stop at the first failure:
  0. structure  the CLI and the training-setup layer are byte-identical to the reference (hash list in
                bench/frozen_files.txt), and the frozen batch fixture still hashes the same. A candidate works on
                the model, the loss and the kernels; the launch path and the recipe do not move.
  1. benefit    reference, candidate, reference: fresh `state tx train` runs of the real entry point with cold
                compile caches. speedup = mean reference window time / candidate window time. INVALID (measure
                again later) when a reference is off the frozen reference, when the two references differ, or
                when a window ratio drifts. speedup <= max(1.02, 1 + 2 x reference spread) -> UNAVAILABLE.
  2. bitwise    deterministic 3-step cases through the numerics adapter. Exact items (phase, RNG fingerprints,
                call counters, learning rate, step counter, shapes, finiteness) must hold, else INCORRECT. Every
                recorded tensor bitwise identical -> PASS (N0).
  3. statistical only when gate 2 is not bitwise: paired short trajectories, per-endpoint bias within 3 sigma.

Tiers: N0 bitwise with the same dtypes and math modes; N2 a dtype or math mode changed; N1 otherwise. Above
bench.json agent_max_tier -> INCORRECT.

The numerics record fixes the SDPA backends to the math implementation (see bench/state_adapter.py): flash
attention's backward uses nondeterministic atomics, so with it on the baseline cannot reproduce itself bit for
bit. Gate 1 measures speed with the recipe untouched.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent


def _expand(value):
    if isinstance(value, str) and '$' in value:
        out = os.path.expandvars(value)
        if '$' in out:
            raise SystemExit(f'{value}: set ASSETS (and any other variable it names) before running the bench')
        return out
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


CFG = _expand(json.loads((BENCH / 'bench.json').read_text()))
sys.path.insert(0, str(BENCH))
import recipe  # noqa: E402

LOCK = Path(CFG['gpu_lock'])
PY = Path(CFG['python'])
TIERS = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5']
SCRATCH = Path(CFG.get('scratch', '/tmp'))
TIMEOUT = CFG['timeouts_s']
RESERVED = set(CFG['reserved_env'])
SIGNATURES = [(re.compile(x['regex']), x['reason']) for x in CFG.get('invalid_run_signatures', [])]
COST = {'gpu_seconds': 0.0, 'gpu_runs': 0}
TREE_SKIP = {'.git', '__pycache__', 'wandb', 'outputs', 'figures'}


def path(key):
    p = Path(_expand(key))
    return p if p.is_absolute() else (ROOT / p).resolve()


def log(msg):
    print(f'[bench {datetime.now():%H:%M:%S}] {msg}', flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
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
                    log(f'waiting for {LOCK} ({self.what})')
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
            'method': f'GPU lock hold time x {CFG["world_size"]} card(s), startup and failed runs included'}


def scratch_dir(prefix):
    free = shutil.disk_usage(SCRATCH).free / 2**30
    if free < CFG.get('min_scratch_free_gib', 20):
        raise SystemExit(f'{SCRATCH} has only {free:.1f} GiB free; clean up before measuring')
    return tempfile.TemporaryDirectory(prefix=prefix, dir=SCRATCH)


def gpu_state():
    q = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=60)
    out = []
    for line in q.stdout.strip().splitlines():
        idx, uid, mem, util = [x.strip() for x in line.split(',')]
        out.append({'index': int(idx), 'uuid': uid, 'memory_mib': int(mem), 'util': int(util)})
    return out


def our_gpus():
    """The GPUs this benchmark may use, from CUDA_VISIBLE_DEVICES (indices into nvidia-smi)."""
    want = [x for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x.strip()]
    return [int(x) for x in want] if want else None


def wait_idle(what, patience=600):
    allowed = our_gpus()
    t0 = time.time()
    while True:
        state = [g for g in gpu_state() if allowed is None or g['index'] in allowed]
        busy = [g for g in state if g['memory_mib'] > 1024 or g['util'] > 5]
        if not busy:
            return state, None
        if time.time() - t0 > patience:
            return state, f'GPUs busy for {patience}s before {what}: {busy}'
        time.sleep(15)


def _tagged(tag):
    me, pids = os.getuid(), []
    mark = f'STATE_BENCH_RUN={tag}'.encode()
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            if d.stat().st_uid == me and mark in (d / 'environ').read_bytes().split(b'\0'):
                pids.append(int(d.name))
        except OSError:
            continue
    return pids


def _kill_tagged(tag):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pids = _tagged(tag)
        if not pids:
            return 0
        for p in pids:
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                pass
        time.sleep(5)
    return len(_tagged(tag))


def base_env(tmp, cand_env=None, pythonpath=None):
    env = {k: v for k, v in os.environ.items() if k not in RESERVED and not k.startswith('STATE_BENCH')}
    env.update(cand_env or {})
    caches = Path(tmp) / 'caches'
    env.update({
        'TMPDIR': str(tmp), 'TMP': str(tmp), 'TEMP': str(tmp),
        'TORCHINDUCTOR_CACHE_DIR': str(caches / 'inductor'), 'TRITON_CACHE_DIR': str(caches / 'triton'),
        'CUDA_CACHE_PATH': str(caches / 'cuda'), 'XDG_CACHE_HOME': str(caches / 'xdg'),
        'MPLCONFIGDIR': str(caches / 'mpl'),
        'HF_HOME': CFG['hf_home'], 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
        'WANDB_MODE': 'disabled', 'PYTHONNOUSERSITE': '1',
        'PATH': f'{PY.parent}:{env.get("PATH", "/usr/local/bin:/usr/bin:/bin")}',
    })
    env.pop('PYTHONPATH', None)
    if pythonpath:
        env['PYTHONPATH'] = ':'.join(str(p) for p in pythonpath)
    for v in ('inductor', 'triton', 'cuda', 'xdg', 'mpl'):
        (caches / v).mkdir(parents=True, exist_ok=True)
    return env


def run_gpu(cmd, cwd, env, log_file, what, timeout):
    tag = uuid.uuid4().hex[:12]
    with GpuLock(what):
        before, problem = wait_idle(what)
        if problem:
            return {'status': 'INVALID', 'reason': problem, 'returncode': None, 'gpus_before': before}
        env = dict(env, STATE_BENCH_RUN=tag)
        t0 = time.time()
        status = None
        with open(log_file, 'w') as lf:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                rc, status = None, 'TIMEOUT'
        wall = time.time() - t0
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        left = _kill_tagged(tag)
        after, _ = wait_idle(f'{what} (after)', patience=60)
    res = {'status': status or 'ok', 'returncode': rc, 'wall_s': round(wall, 1),
           'gpus_before': before, 'gpus_after': after}
    if status == 'TIMEOUT':
        res['reason'] = f'{what} exceeded {timeout}s; process group killed; see {Path(log_file).name}'
    if left:
        res['warning'] = f'{left} processes of the run survived SIGKILL'
    return res


def signatures(log_file):
    hits, text = [], Path(log_file).read_text(errors='replace') if Path(log_file).exists() else ''
    for pattern, reason in SIGNATURES:
        m = pattern.search(text)
        if m:
            line = text[text.rfind('\n', 0, m.start()) + 1:text.find('\n', m.end())]
            hits.append({'reason': reason, 'line': line.strip()[:300]})
    return hits


# ---------------------------------------------------------------- inputs and sources

def tree_digest(tree):
    h = hashlib.sha256()
    tree = Path(tree)
    for f in sorted(tree.rglob('*')):
        rel = f.relative_to(tree)
        if rel.parts and rel.parts[0] in TREE_SKIP and len(rel.parts) == 1:
            continue
        if '__pycache__' in rel.parts or f.is_symlink() or not f.is_file():
            continue
        h.update(str(rel).encode() + b'\0' + sha256_file(f).encode() + b'\n')
    return h.hexdigest()


def check_reference():
    ref = path(CFG['reference'])
    git = ['git', '-c', 'safe.directory=*', '-C', str(ref)]
    dirty = subprocess.run([*git, 'status', '--porcelain', '--untracked-files=no'], capture_output=True,
                           text=True).stdout.strip()
    head = subprocess.run([*git, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    if dirty or head != CFG['reference_commit']:
        raise SystemExit(f'reference tree is not the frozen upstream commit (head {head}, dirty: {bool(dirty)})')
    return ref


def materialize(src, dest):
    """Copy the tree. Only run-output directories at the tree ROOT are skipped (a `wandb` directory deeper in,
    such as src/state/configs/wandb/, is a Hydra config group and must come along)."""
    src = Path(src).resolve()

    def ignore(directory, names):
        d = Path(directory).resolve()
        out = [n for n in names if n.endswith('.pyc') or n == '__pycache__']
        if d == src:
            out += [n for n in names if n in TREE_SKIP]
        return out
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
    bad = sorted(k for k in env if k in RESERVED or k.startswith('STATE_BENCH'))
    if bad:
        raise SystemExit(f'{config_file}: {bad} are managed by the bench and cannot be candidate switches')
    return env


def versions():
    out = {'python': sys.version.split()[0]}
    for pkg in ('torch', 'triton', 'transformers', 'lightning', 'cell_load', 'anndata', 'geomloss'):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


FINGERPRINT_KEYS = ('argv', 'math_mode', 'gpus', 'compile_cache', 'instrumentation', 'versions', 'timing_plan')


def fingerprint_diff(a, b):
    if a is None or b is None:
        return ['missing fingerprint']
    return [k for k in FINGERPRINT_KEYS if a.get(k) != b.get(k)]


def file_fingerprint(p, kind):
    if kind == 'sha256':
        return sha256_file(p)
    st = Path(p).stat()
    return f'{st.st_size}:{st.st_mtime_ns}'


def verify_inputs(when):
    spec = json.loads((ROOT / 'task' / 'formulation.json').read_text())['inputs']
    bad = {}
    for rel, want in spec.items():
        p = path(rel)
        got = file_fingerprint(p, want['kind']) if p.exists() else None
        if got != want['value']:
            bad[rel] = {'want': want['value'], 'got': got}
    log(f'inputs {when}: {len(spec) - len(bad)}/{len(spec)} match the formulation')
    return {'when': when, 'ok': not bad, 'mismatches': bad, 'checked': len(spec)}


def freeze_inputs(paths, big=400 << 20):
    out = {}
    for root in paths:
        root = Path(root)
        files = [root] if root.is_file() else sorted(f for f in root.rglob('*') if f.is_file())
        for f in files:
            kind = 'stat' if f.stat().st_size > big else 'sha256'
            out[str(f)] = {'kind': kind, 'value': file_fingerprint(f, kind)}
    return out


# ---------------------------------------------------------------- gate 0: structure

def frozen_files():
    return [ln.strip() for ln in (BENCH / 'frozen_files.txt').read_text().splitlines() if ln.strip()
            and not ln.startswith('#')]


def coverage(tree):
    """Gate 0 (CPU): the launch path and the recipe are byte-identical, and the fixture did not move."""
    ref_src = check_reference()
    problems = []
    for rel in frozen_files():
        a, b = Path(ref_src) / rel, Path(tree) / rel
        if not b.is_file():
            problems.append(f'{rel}: missing')
        elif sha256_file(a) != sha256_file(b):
            problems.append(f'{rel}: differs from the reference')
    fx = json.loads((BENCH / 'fixture_manifest.json').read_text())
    for rel, want in fx.items():
        p = path(rel)
        got = sha256_file(p) if p.exists() else None
        if got != want:
            problems.append(f'frozen fixture {rel} changed: the batches the judge replays are not the frozen ones')
    if problems:
        return 'ERROR', {'passed': False, 'stage': 'structure', 'problems': problems,
                         'note': 'the CLI and the training-setup layer are frozen; put the change in the model, '
                                 'the loss or the kernels that the trainer calls, behind an off-by-default switch'}
    return None


# ---------------------------------------------------------------- gate 2: bitwise records

def adapter_kwargs(tree, scratch, steps=None):
    return {'repo': str(tree), 'toml': str(path(CFG['fixture']['toml'])),
            'dataset_dir': _expand(str(CFG['fixture']['dataset_dir'])),
            'fixture': str(path('bench/fixture.pt')),
            'steps': int(steps or CFG['numerics'].get('case_steps', 3)),
            'batch_size': recipe.BATCH_SIZE, 'cell_set_len': recipe.CELL_SET_LEN}


def record_cases(tree, rec, cases, logs_dir, cand_env, label, losses_only=False, sketch=True):
    n = CFG['numerics']
    for case in cases:
        with scratch_dir('state-r-') as tmp:
            cases_file = Path(tmp) / 'case.json'
            cases_file.write_text(json.dumps([case]))
            env = base_env(tmp, cand_env, pythonpath=[ROOT, Path(tree) / recipe.SRC])
            cmd = [str(PY), '-m', 'numerics', 'record', '--adapter', n['adapter'],
                   '--adapter-kwargs', json.dumps(adapter_kwargs(tree, Path(tmp) / 'train', case.get('steps'))),
                   '--cases', str(cases_file),
                   *(['--losses-only'] if losses_only else
                     (['--sketch-dim', str(n['sketch_dim'])] if sketch else [])),
                   '--math-mode', CFG['math_mode'], '--deterministic', '--out', str(rec)]
            log_file = Path(logs_dir) / f'{label}_{case["name"]}.log'
            r = run_gpu(cmd, ROOT, env, log_file, f'{label}: case {case["name"]}',
                        TIMEOUT['trajectory' if losses_only else 'numerics'])
        if r['status'] != 'ok' or r['returncode']:
            return {'status': 'INVALID' if r['status'] == 'INVALID' else 'ERROR',
                    'reason': r.get('reason') or f'record exited {r["returncode"]}; see {log_file.name}'}
    return None


def numerics(tree, out, cand_env):
    n = CFG['numerics']
    rec = out / 'records'
    cases = json.loads(path(n['cases']).read_text())
    failed = record_cases(tree, rec, cases, out, cand_env, 'num')
    if failed:
        return failed
    env = base_env(out, pythonpath=[ROOT])
    rep_file = out / 'numerics.json'
    subprocess.run([str(PY), '-m', 'numerics', 'compare', '--ref', str(path(n['ref'])), '--cand', str(rec),
                    '--thresholds', str(path(n['thresholds'])), '--out', str(rep_file),
                    *sum((['--exclude', e] for e in n.get('exclude') or []), [])],
                   cwd=ROOT, env=env, capture_output=True, text=True)
    if not rep_file.exists():
        return {'status': 'ERROR', 'reason': 'numerics compare produced no report'}
    rep = json.loads(rep_file.read_text())
    bits = rep.get('bitwise_identical', 0)
    checked = rep.get('checked', 0)
    rep['bitwise'] = rep['status'] == 'PASS' and checked > 0 and bits == checked and rep.get('bitwise_trusted', True)
    rep['report'] = str(rep_file)
    return rep


def resolved_toml(dest_dir):
    """cell_load reads the dataset TOML itself and does not expand environment variables; write a resolved copy
    so the dataset directory can be named through ${ASSETS} in bench.json."""
    text = path(CFG['fixture']['toml']).read_text()
    text = re.sub(r'\$\{(\w+)\}', lambda m: os.environ.get(m.group(1), m.group(0)), text)
    if '${' in text:
        raise SystemExit(f'bench/dataset.toml still names an unset variable: {text}')
    out = Path(dest_dir) / 'dataset.toml'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


# ---------------------------------------------------------------- gate 1: timing

def time_tree(tree, out, label, cand_env=None, profile=None, profile_iters=5, against_frozen=True):
    t = CFG['timing']
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f'timing_{label}'
    with scratch_dir('state-t-') as tmp:
        train_args = recipe.hydra_overrides(str(resolved_toml(tmp)), Path(tmp) / 'train', label,
                                            CFG['timing'].get('max_steps', 10_000_000))
        hook_args = ['--tree', str(tree), '--out', str(prefix),
                     '--warmup', str(t['warmup']), '--window', str(t['window']), '--repeats', str(t['repeats'])]
        # The fingerprint must compare the *plan*, not this run's paths: two runs differ only by where they write.
        stable_argv = ['python bench/timing.py', '--tree', '<tree>', '--out', '<out>',
                       '--warmup', str(t['warmup']), '--window', str(t['window']), '--repeats', str(t['repeats']),
                       *[x.replace(str(tmp), '<scratch>').replace(str(tree), '<tree>')
                         for x in train_args
                         if not x.startswith('output_dir=') and not x.startswith('name=')]]
        if profile:
            hook_args += ['--profile', profile, '--profile-iters', str(profile_iters)]
        argv = [str(PY), str(BENCH / 'timing.py'), *hook_args, *train_args]
        env = base_env(tmp, cand_env, pythonpath=[Path(tree) / recipe.SRC])
        log_file = out / f'timing_{label}.log'
        what = f'timing {label}{f" ({profile})" if profile else ""}'
        r = run_gpu(argv, tree, env, log_file, what, TIMEOUT['profile' if profile else 'timing'])
    plan = [t['warmup'], t['window'], t['repeats']] if not profile else [t['warmup'], profile_iters, 1]
    fingerprint = {'argv': stable_argv, 'candidate_environment': cand_env or {},
                   'math_mode': CFG['math_mode'], 'gpus': [(g['index'], g['uuid']) for g in r.get('gpus_before') or []],
                   'compile_cache': 'cold (fresh per run)', 'instrumentation': profile or 'none',
                   'timing_plan': plan, 'versions': versions()}
    run = {'run_status': r['status'], 'wall_s': r.get('wall_s'), 'fingerprint': fingerprint, 'log': str(log_file)}
    if r['status'] != 'ok':
        return {'status': r['status'], 'reason': r['reason'], **run}
    rank_file = Path(f'{prefix}.r0.json')
    if r['returncode'] or not rank_file.exists():
        return {'status': 'ERROR', 'reason': f'timing exited {r["returncode"]}; see {log_file.name}', **run}
    rank = json.loads(rank_file.read_text())
    if rank['status'] != 'ok':
        return {'status': 'ERROR', 'reason': (rank.get('error') or 'timing run failed')[:800], **run}
    hits = signatures(log_file)
    if hits:
        return {'status': 'INVALID', 'reason': 'invalid-run signature: ' + hits[0]['reason'], 'signatures': hits, **run}
    return {**summarize_timing(rank, r['wall_s'], profile, against_frozen), **run}


def summarize_timing(rank, wall, profile=None, against_frozen=True):
    t = CFG['timing']
    problems = []
    ws = rank['windows']
    if len(ws) != (1 if profile else t['repeats']):
        problems.append(f'{len(ws)} of {1 if profile else t["repeats"]} windows recorded')
    for w in ws:
        if w.get('compiles_in_window'):
            problems.append(f'window {w["k"]}: torch.compile compiled {w["compiles_in_window"]} graphs inside the '
                            f'timing window (still compiling in steady state)')
    secs = [w['seconds'] for w in ws]
    total = sum(secs) if secs else None
    frozen, rng_problem = None, None
    ref_file = ROOT / 'task' / 'timing_ref.json'
    if against_frozen and not profile and ref_file.exists():
        frozen = json.loads(ref_file.read_text())
        n_check = t['warmup'] + t['window'] * t['repeats']
        want = frozen['rng_fingerprints'][:n_check]
        if rank['rng_fingerprints'][:n_check] != want:
            k = next((i for i, (a, b) in enumerate(zip(rank['rng_fingerprints'], want)) if a != b), None)
            rng_problem = (f'the random-number stream differs from the reference from iteration {k} on (random '
                           f'numbers drawn in a different amount or order change the work of every later iteration)')
    if rng_problem:
        problems.append(rng_problem)
    per_window = 1 if profile else t['window']
    n_iter = per_window * len(secs)
    res = {'status': 'ok' if not problems and secs else 'ERROR', 'problems': problems,
           'window_s': [round(s, 3) for s in secs], 'total_s': round(total, 3) if total else None,
           'window_iterations': per_window, 'iterations_timed': n_iter or None,
           's_per_iteration': round(total / n_iter, 4) if total and n_iter else None,
           'peak_memory_allocated_gib': round(rank['peak_memory_allocated_bytes'] / 2**30, 2),
           'startup_to_first_iteration_s': rank['startup_to_first_iteration_s'], 'process_wall_s': round(wall, 1)}
    if len(secs) >= 2:
        # Drift detector: the first timed window against the mean of the later ones. A window disturbed by
        # someone else's job shows up here even when the run total looks normal.
        res['window_ratio'] = round(secs[0] / statistics.fmean(secs[1:]), 4)
        if frozen and frozen.get('window_ratio'):
            res['window_ratio_vs_frozen'] = round(res['window_ratio'] / frozen['window_ratio'] - 1, 4)
    return res


def strip(run):
    return {k: v for k, v in run.items() if k != 'rank'} if run else run


def benefit(tree, out, cand_env):
    t = CFG['timing']
    frozen = json.loads((ROOT / 'task' / 'timing_ref.json').read_text())
    ref_src = check_reference()
    with scratch_dir('state-ref-') as tmp:
        ref_tree = materialize(ref_src, Path(tmp) / 'reference')
        ref_a = time_tree(ref_tree, out, 'ref_a')
        cand = time_tree(tree, out, 'cand', cand_env=cand_env) if ref_a['status'] == 'ok' else None
        ref_b = time_tree(ref_tree, out, 'ref_b') if cand and cand['status'] == 'ok' else None
    runs = {'ref_a': strip(ref_a), 'cand': strip(cand), 'ref_b': strip(ref_b)}
    bad = {k: v for k, v in runs.items() if v is None or v['status'] != 'ok'}
    if bad:
        status = 'INVALID' if any(v and v['status'] in ('INVALID', 'TIMEOUT') for v in bad.values()) else 'ERROR'
        return status, {'speedup': None, 'runs': runs}
    diffs = {k: fingerprint_diff(ref_a.get('fingerprint'), v.get('fingerprint')) for k, v in runs.items() if k != 'ref_a'}
    for k in diffs:
        diffs[k] = [d for d in diffs[k] if d != 'candidate_environment']
    badfp = {k: d for k, d in diffs.items() if d}
    if badfp:
        return 'ERROR', {'speedup': None, 'runs': runs, 'fingerprint_diffs': diffs,
                         'reason': f'timing runs not comparable: fingerprint differs from ref_a in {badfp}'}
    ra, rb, c = ref_a['total_s'], ref_b['total_s'], cand['total_s']
    ref = (ra + rb) / 2
    perf = {'reference_s_per_iteration': round(ref / ref_a['iterations_timed'], 4),
            'candidate_s_per_iteration': cand['s_per_iteration'], 'speedup': round(ref / c, 4),
            'reference_spread': round(abs(ra - rb) / ref, 4),
            'frozen_reference_s_per_iteration': frozen['s_per_iteration'],
            'runs': runs, 'fingerprint_diffs': diffs}
    problems = []
    for name, run in (('ref_a', ref_a), ('ref_b', ref_b)):
        off = run['total_s'] / frozen['total_s'] - 1
        if abs(off) > t['frozen_band']:
            problems.append(f'{name} took {run["total_s"]:.1f}s, {off:+.1%} off the frozen reference '
                            f'{frozen["total_s"]:.1f}s (> {t["frozen_band"]:.0%})')
    pair = [abs(a - b) / ((a + b) / 2) for a, b in zip(ref_a['window_s'], ref_b['window_s'])]
    perf['reference_pair_window_spread'] = [round(x, 4) for x in pair]
    # Per-window differences are the noisiest statistic on a shared host; judge the two reference runs on their
    # window MEANS, and report the per-window spread as a diagnostic.
    mean_a, mean_b = statistics.fmean(ref_a['window_s']), statistics.fmean(ref_b['window_s'])
    perf['reference_mean_gap'] = round(abs(mean_a - mean_b) / ((mean_a + mean_b) / 2), 4)
    if perf['reference_mean_gap'] > t['pair_band']:
        problems.append(f'the two reference runs differ by {perf["reference_mean_gap"]:.1%} in their window mean '
                        f'(> {t["pair_band"]:.0%})')
    for name, run in (('ref_a', ref_a), ('cand', cand), ('ref_b', ref_b)):
        band = t['window_band'] if name != 'cand' else t['window_band_candidate']
        dev = run.get('window_ratio_vs_frozen')
        if dev is not None and abs(dev) > band:
            problems.append(f'{name}: window ratio {run["window_ratio"]} is {dev:+.1%} off the frozen ratio '
                            f'{frozen["window_ratio"]} (> {band:.0%}): one window was disturbed')
    perf['problems'] = problems
    if problems:
        return 'INVALID', perf
    bar = max(CFG['benefit']['min_speedup'], 1 + 2 * perf['reference_spread'])
    perf['benefit'] = {'speedup': perf['speedup'], 'speedup_bar': round(bar, 4),
                       'peak_memory_allocated_gib': cand['peak_memory_allocated_gib'],
                       'passed': perf['speedup'] > bar}
    return ('ok' if perf['benefit']['passed'] else 'UNAVAILABLE'), perf


# ---------------------------------------------------------------- verdict

def evaluate(tree, out, skip_timing=False, cand_env=None, quick=False):
    failed = coverage(tree)
    if failed:
        return failed[0], failed[1], None
    if quick:
        num = numerics(tree, out, cand_env)
        if num.get('exact_failures'):
            return 'INCORRECT', {'passed': False, 'stage': 'structure', 'verdict': 'structure', 'numerics': num}, None
        if num.get('errors') or num.get('status') not in ('PASS', 'INCORRECT'):
            return 'ERROR', {'passed': False, 'stage': 'structure', 'numerics': num}, None
        return 'PASS', {'passed': True, 'stage': 'structure (quick: gates 1 and 3 not run)',
                        'bitwise': num['bitwise'], 'measured_tier': num.get('measured_tier'), 'numerics': num}, None
    performance = None
    if not skip_timing:
        status, performance = benefit(tree, out, cand_env)
        if status in ('ERROR', 'INVALID'):
            return status, {'passed': False, 'stage': 'benefit',
                            'note': (performance or {}).get('reason') or '; '.join((performance or {}).get('problems') or [])
                            or 'timing failed'}, performance
        if status == 'UNAVAILABLE':
            performance['measured_speedup'] = performance.pop('speedup')
            return 'UNAVAILABLE', {'passed': None, 'stage': 'benefit', 'verdict': 'no_benefit',
                                   'note': 'not faster than the reference: rejected without a precision evaluation'}, performance
    num = numerics(tree, out, cand_env)
    tier = num.get('measured_tier')
    correctness = {'stage': 'bitwise', 'measured_tier': tier, 'numerics': num}
    if num.get('exact_failures'):
        return 'INCORRECT', dict(correctness, passed=False, verdict='structure',
                                 note='exact checks failed: the candidate does different work'), performance
    if num.get('errors') or num.get('status') not in ('PASS', 'INCORRECT'):
        return 'ERROR', dict(correctness, passed=False, note='numerics could not compare the records'), performance
    if tier not in TIERS or TIERS.index(tier) > TIERS.index(CFG['agent_max_tier']):
        return 'INCORRECT', dict(correctness, passed=False, verdict='needs_signoff',
                                 note=f'measured tier {tier} exceeds {CFG["agent_max_tier"]}'), performance
    if num['bitwise']:
        return 'PASS', dict(correctness, passed=True, verdict='bitwise'), performance
    traj = trajectories(tree, out, cand_env)
    correctness.update(stage='statistical', trajectory=traj)
    if traj['verdict'] == 'equivalent':
        return 'PASS', dict(correctness, passed=True, verdict='equivalent'), performance
    note = ('an endpoint deviates from the reference beyond 3 x its seed spread' if traj['verdict'] == 'different'
            else 'no verdict from the trajectory')
    return 'INCORRECT', dict(correctness, passed=False, verdict=traj['verdict'], note=note), performance


def trajectories(tree, out, cand_env):
    tj = CFG['trajectory']
    spec = json.loads(path(tj['spec']).read_text())
    frozen_path = path(tj['frozen'])
    frozen = json.loads(frozen_path.read_text())
    seeds = frozen['reference_seeds']
    rec = out / 'trajectory'
    failed = record_cases(tree, rec, [dict(spec['case'], name=f'traj_s{s}', seed=s) for s in seeds],
                          out, cand_env, 'traj', losses_only=True, sketch=False)
    if failed:
        return {'verdict': 'error', **failed}
    env = base_env(out, pythonpath=[ROOT])
    rep_file = out / 'traj_report.json'
    subprocess.run([str(PY), '-m', 'numerics', 'traj-compare', '--frozen', str(frozen_path), '--cand', str(rec),
                    '--sigma-k', '3', '--out', str(rep_file)], cwd=ROOT, env=env, capture_output=True, text=True)
    if not rep_file.exists():
        return {'verdict': 'error', 'reason': 'traj-compare produced no report'}
    rep = json.loads(rep_file.read_text())
    rep['report'] = str(rep_file)
    return rep


def guarded(out, fn, tree_src=None):
    out.mkdir(parents=True, exist_ok=True)
    start = verify_inputs('start')
    if not start['ok']:
        return {'status': 'ERROR', 'stage': 'inputs', 'inputs': [start],
                'note': 'a frozen input differs from task/formulation.json; nothing was measured'}
    if tree_src is None:
        res, digest = fn(None), None
    else:
        with scratch_dir('state-src-') as tmp:
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
    res['finished'] = datetime.now().isoformat(timespec='seconds')
    (out / name).write_text(json.dumps(res, indent=1, default=str) + '\n')
    log(f'wrote {out / name}: status {res.get("status")}')
    summary = {'status': res.get('status')}
    perf = res.get('performance')
    if isinstance(perf, dict):
        summary['speedup'] = perf.get('speedup', perf.get('measured_speedup'))
    print(json.dumps(summary))
    return 0 if res.get('status') in ('PASS', 'ok') else 1


def evaluator_sha256():
    h = hashlib.sha256()
    for f in sorted(BENCH.glob('*')):
        if f.is_file() and f.suffix in ('.py', '.json', '.toml', '.txt'):
            h.update(f.name.encode() + b'\0' + sha256_file(f).encode() + b'\n')
    return h.hexdigest()


# ---------------------------------------------------------------- commands

def cmd_check(a):
    out = Path(a.out).resolve()
    # `--final` is the harness's own full check (goal/clawo/judge.sh). Its marker lives outside this tree, so a
    # candidate cannot ask for -- and cannot get -- a final verdict of its own.
    if a.final and not (ROOT / 'task' / 'final-evaluation').exists():
        print("check: --final belongs to the harness; task/final-evaluation is not present", file=sys.stderr)
        return 2
    if a.final:
        a.quick, a.skip_timing = False, False
    cand_env = candidate_environment(a.config)

    def run(tree):
        status, correctness, performance = evaluate(tree, out, skip_timing=a.skip_timing, cand_env=cand_env,
                                                    quick=a.quick)
        return {'status': status, 'tree': str(Path(a.tree).resolve()), 'config': a.config, 'environment': cand_env,
                'correctness': correctness, 'performance': performance,
                'final': bool(a.final), 'n1': bool(a.n1), 'quick': bool(a.quick),
                'skip_timing': bool(a.skip_timing)}
    return finish(out, 'check.json', guarded(out, run, Path(a.tree).resolve()))


def cmd_time(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)

    def run(tree):
        r = time_tree(tree, out, 'cand', cand_env=cand_env)
        frozen = json.loads((ROOT / 'task' / 'timing_ref.json').read_text())
        if r.get('total_s'):
            r['speedup_vs_frozen_reference'] = round(frozen['total_s'] / r['total_s'], 4)
        return r
    return finish(out, 'time.json', guarded(out, run, Path(a.tree).resolve()))


def cmd_profile(a):
    out = Path(a.out).resolve()

    def run(tree):
        return time_tree(tree, out, 'profile', profile='torch', profile_iters=a.iters)
    return finish(out, 'profile.json', guarded(out, run, Path(a.tree).resolve()))


def cmd_baseline(a):
    """Formulator: the reference timed three times, to see the noise floor before freezing anything."""
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref_src = check_reference()
    runs = []
    with scratch_dir('state-base-') as tmp:
        ref_tree = materialize(ref_src, Path(tmp) / 'reference')
        for i in range(3):
            runs.append(strip(time_tree(ref_tree, out, f'base{i}')))
    tot = [r.get('total_s') for r in runs]
            # noqa
    spread = (max(tot) - min(tot)) / (sum(tot) / len(tot)) if all(tot) else None
    res = {'status': 'ok' if all(r.get('status') == 'ok' for r in runs) else 'ERROR', 'runs': runs,
           'total_s': tot, 'relative_spread': round(spread, 4) if spread is not None else None}
    (out / 'baseline.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(json.dumps({'status': res['status'], 'total_s': tot, 'spread': res['relative_spread']}))
    return 0


def cmd_freeze_timing(a):
    """Formulator: freeze the reference timing that gate 1 compares against."""
    ref_src = check_reference()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with scratch_dir('state-freeze-') as tmp:
        ref_tree = materialize(ref_src, Path(tmp) / 'reference')
        r = time_tree(ref_tree, out, 'freeze', against_frozen=False)
    if r.get('status') != 'ok':
        raise SystemExit(f'reference timing failed: {r.get("reason") or r.get("problems")}')
    rank = json.loads((out / 'timing_freeze.r0.json').read_text())
    (ROOT / 'task' / 'timing_ref.json').write_text(json.dumps({
        'total_s': r['total_s'], 'window_s': r['window_s'], 'window_ratio': r['window_ratio'],
        's_per_iteration': r['s_per_iteration'], 'iterations_timed': r['iterations_timed'],
        'rng_fingerprints': rank['rng_fingerprints'], 'fingerprint': r['fingerprint'],
    }, indent=1) + '\n')
    print(json.dumps({'total_s': r['total_s'], 's_per_iteration': r['s_per_iteration']}))
    return 0


def cmd_freeze_inputs(a):
    spec = freeze_inputs([path(p) for p in a.paths])
    (ROOT / 'task' / 'formulation.json').write_text(json.dumps({'inputs': spec}, indent=1) + '\n')
    print(json.dumps({'frozen': len(spec)}))
    return 0


def cmd_record(a):
    """Formulator: record the reference cases (run twice; the two recordings must be bitwise identical)."""
    n = CFG['numerics']
    tree = check_reference()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads(path(n['cases']).read_text())
    failed = record_cases(tree, out, cases, out, {}, 'ref')
    if failed:
        raise SystemExit(f'record failed: {failed.get("reason")}')
    print(json.dumps({'recorded': [c['name'] for c in cases]}))
    return 0


def main():
    p = argparse.ArgumentParser(prog='bench.py')
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check')
    c.add_argument('--tree', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--config')
    c.add_argument('--quick', action='store_true')
    c.add_argument('--skip-timing', action='store_true')
    c.add_argument('--final', action='store_true',
                   help="the harness's full check; refused unless task/final-evaluation exists")
    c.add_argument('--n1', action='store_true')
    t = sub.add_parser('time')
    t.add_argument('--tree', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--config')
    pr = sub.add_parser('profile')
    pr.add_argument('--tree', required=True)
    pr.add_argument('--out', required=True)
    pr.add_argument('--config')
    pr.add_argument('--nsys', action='store_true')
    pr.add_argument('--iters', type=int, default=5)
    b = sub.add_parser('baseline')
    b.add_argument('--out', required=True)
    fr = sub.add_parser('freeze-timing')
    fr.add_argument('--out', required=True)
    fi = sub.add_parser('freeze-inputs')
    fi.add_argument('paths', nargs='+')
    rc = sub.add_parser('record')
    rc.add_argument('--out', required=True)
    a = p.parse_args()
    return {'check': cmd_check, 'time': cmd_time, 'profile': cmd_profile, 'baseline': cmd_baseline,
            'freeze-timing': cmd_freeze_timing, 'freeze-inputs': cmd_freeze_inputs,
            'record': cmd_record}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
