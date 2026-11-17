"""BAGEL-7B-MoT supervised fine-tuning on 4 GPUs: one command per question.

Run from the workspace root (/opt/researchboost/ws) with the training environment's Python (bench.json python). GPU
runs take /tmp/bagel_gpu.lock themselves, one at a time; do NOT wrap these commands in flock (that would deadlock).

  python bench/bench.py check   --tree <dir> --out <dir> [--config <file>] [--quick | --skip-timing | --n1]
        the verdict (gates 0-3 below) -> <out>/check.json; --quick: gates 0 and 2 only (no timing, no trajectories);
        --skip-timing: no gate 1; --n1: gates 0 and 3 only, after a check whose gate 2 was not bitwise identical.
        --final is the final evaluation's (root or the verifier's marker, final_allowed): gate 1 as reference ->
        candidate -> reference, and the timing noise rules make a result INVALID
  python bench/bench.py time    --tree <dir> --out <dir> [--config <file>] [--paired]
        one timing run of the tree (5 warmup steps, then 2 windows of 10 optimizer steps) -> <out>/time.json:
        s/step, peak memory, and the speedup over the frozen reference timing (the mean of the preparation's 3
        reference runs: frozen speed / this run's) with that reference's own run-to-run noise; without a frozen
        reference timing it times the reference once first and reads the speedup against that run (cached in the
        scratch directory for the next time on the same GPUs; its noise is then not measured). --paired also times
        the reference right before it, fresh, and reports that ratio too (screening numbers, not a verdict).
  python bench/bench.py profile --tree <dir> --out <dir> [--config <file>] [--from K] [--iters M]
        torch.profiler over optimizer steps K..K+M-1 (default 8, 2) on rank 0 -> <out>/profile.txt (operators by
        CUDA time), a Chrome trace and the step times of the run
  formulator only: record, compare, traj-ref, freeze-inputs, baseline, freeze-timing (see --help of each)

Every run: the frozen recipe (bench/recipe.py) through the judge's entry (bench/bagel_entry.py), which runs the tree's
train/pretrain_unified_navit.py as __main__ on CFG world_size GPUs (torchrun, one node), with the released BAGEL-7B-MoT
checkpoint and upstream's bagel_example data from the asset mount and a fresh results directory. A run ends after the
optimizer step it measures up to; no further step starts. Each rank is pinned to its share of the container's CPUs
(bench/cpu_affinity.py), the same for every tree; every GPU run reports the CPUs the container kept busy.

THE SPEEDUP OF THE FINAL VERDICT IS MEASURED FRESH: the final evaluation's benefit gate (--final) times reference ->
candidate -> reference on the host and in the session where it runs. The frozen reference timing
(task/timing/reference.json, from the preparation's timing.reference_runs reference runs on this machine) is what
`time` and the agent's check (gate 1 without --final: one candidate run) read the speedup against, and the final
evaluation rejects (INVALID) a check whose reference runs are more than timing.max_reference_drift off it.

Interpreters: the tree under test runs with bench.json candidate_python (the agent's environment, which sees the
training environment's packages unless it installs its own versions); reference runs and formulator commands run
with bench.json python (read-only training environment).

Every GPU run holds the GPU lock, stops processes an earlier bench run left behind, waits until the GPUs hold no
foreign memory, runs a fresh copy of the tree in its own process group with a timeout and a fresh TMPDIR, and kills
what it started afterwards. Every command checks the frozen inputs declared in task/formulation.json before
(mismatch: nothing is measured) and after (mismatch: ERROR), and the source digest of the tree copy before and after.

Candidate switches: a JSON file {"environment": {"NAME": "value"}} given with --config; the bench sets those
variables for the candidate's training processes. Variables the bench manages (bench.json reserved_env) and
BAGEL_BENCH_* are refused.

Verdict (check); each gate stops the evaluation when it fails:
  0. structure  train/pretrain_unified_navit.py, modeling/bagel/bagel.py, data/dataset_base.py, data/dataset_info.py
                and train/fsdp_utils.py must exist (else ERROR).
  1. benefit    fresh runs, timing.warmup + timing.repeats x timing.window steps each, production settings. Each
                window's seconds per step = (end of its last step - end of the step before it) / its steps, the
                slowest rank deciding (every rank synchronizes CUDA at each step end); a run's s/step = the mean of
                its windows. The agent's check: one candidate run against the frozen reference timing; speedup =
                frozen s/step / candidate, reference spread = the frozen runs' (max - min) / mean; the noise rules
                below are reported, never a rejection. The final evaluation (--final): reference A, candidate,
                reference B; speedup = mean(A, B) / candidate, reference spread = |A - B| / mean(A, B); fingerprints
                must agree with A (else ERROR). In the final evaluation only, a run whose window range
                ((max - min) / mean) exceeds timing.max_window_spread, or a reference run off the frozen reference
                timing by more than timing.max_reference_drift, makes the result INVALID (rerun later); elsewhere both
                are reported.
                No speedup above max(benefit.min_speedup, 1 + twice the reference spread) -> UNAVAILABLE
                (no_benefit), rejected without a precision evaluation.
  2. record     deterministic mode, record.steps optimizer steps from the released checkpoint. Exact items, else
                INCORRECT: the number of steps, no steps of another optimizer, exactly one trained FSDP model and one
                EMA, the hyper-parameters of every parameter group at every step, the batches every rank moved to the
                GPU (a SHA-256 of every field, in order), the trained parameters' names, shapes and dtypes (model and
                EMA), no optimizer-state dtype the reference does not have. Then the SHA-256 of every local shard of
                the model's flat parameters, their gradients and both AdamW moments, and of the EMA, after every
                step: all identical to the frozen reference record (bench/records/ref.json) -> PASS, tier N0.
  3. statistical for every candidate, bitwise ones too: a paired trajectory (bench/traj.py; the production settings, as timed,
                trajectory.steps steps from the released checkpoint perturbed per seed) on the first trajectory.seeds
                (1) of the frozen reference seeds; 14 endpoints (CE, MSE and the log gradient norm over steps 50-100
                and 100-150; the 10-step parameter displacement of the ViT, the language model, the other parameters
                and the EMA over steps 50-100 and 100-140). Each endpoint's bias, candidate - reference, against
                trajectory.sigma_k (3) x the reference's seed-to-seed sigma (frozen before any candidate): all within
                -> PASS, tier N1; any beyond -> INCORRECT. The same in every check, the final evaluation's included.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
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
import conform  # noqa: E402
import recipe  # noqa: E402
import traj as trajectory  # noqa: E402

LOCK = Path(CFG['gpu_lock'])
PY = Path(CFG['python'])
CAND_PY = Path(CFG['candidate_python'])
TORCHRUN = PY.parent / 'torchrun'
SCRATCH = Path(CFG.get('scratch', '/tmp'))
TIMEOUT = CFG['timeouts_s']
RESERVED = set(CFG['reserved_env'])
SIGNATURES = [(re.compile(x['regex']), x['reason']) for x in CFG.get('invalid_run_signatures', [])]
COST = {'gpu_seconds': 0.0, 'gpu_runs': 0}
TREE_SKIP = {'.git', '__pycache__', '.venv', 'wandb', 'results', 'checkpoints'}
MARK = 'BAGEL_BENCH_RUN'
SERIES = ('ce', 'mse', 'grad_norm')
DISP_SERIES = ('disp_vit', 'disp_lm', 'disp_other', 'disp_ema')


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
            'method': f'GPU lock hold time x {CFG["world_size"]} cards (startup and failed runs included)'}


def scratch_dir(prefix):
    free = shutil.disk_usage(SCRATCH).free / 2**30
    if free < CFG.get('min_scratch_free_gib', 50):
        raise SystemExit(f'{SCRATCH} has only {free:.1f} GiB free (< {CFG.get("min_scratch_free_gib", 50)}); '
                         f'clean up before measuring')
    return tempfile.TemporaryDirectory(prefix=prefix, dir=SCRATCH)


def gpu_state():
    q = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu,driver_version,uuid',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=60)
    out = []
    for line in q.stdout.strip().splitlines():
        idx, name, mem, util, drv, uid = [x.strip() for x in line.split(',')]
        out.append({'index': int(idx), 'name': name, 'memory_mib': int(mem), 'util': int(util), 'driver': drv,
                    'uuid': uid})
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
            return state, f'GPUs busy for {patience}s before {what}: {busy}'
        time.sleep(15)


def cgroup_cpu_seconds():
    """CPU time this container has used so far, in seconds (cgroup v2 cpu.stat usage_usec, else v1 cpuacct.usage);
    None where neither is readable."""
    try:
        for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines():
            if line.startswith('usage_usec '):
                return int(line.split()[1]) / 1e6
    except (OSError, ValueError):
        pass
    try:
        return int(Path('/sys/fs/cgroup/cpuacct/cpuacct.usage').read_text()) / 1e9
    except (OSError, ValueError):
        return None


class CpuSampler:
    """The container's CPU time once a second while a GPU run holds the lock (nothing else of the bench runs then):
    how many CPUs the run kept busy, over the whole run and over any interval of it (a timing window)."""

    def __init__(self, period=1.0):
        self.period, self.samples, self.stop = period, [], threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while True:
            used = cgroup_cpu_seconds()
            if used is not None:
                self.samples.append((time.time(), used))
            if self.stop.wait(self.period):
                break

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        used = cgroup_cpu_seconds()
        if used is not None:
            self.samples.append((time.time(), used))

    def busy_cpus(self, start=None, end=None):
        """Mean number of busy CPUs (hardware threads) between two wall times (default: the whole run)."""
        s = [x for x in self.samples if (start is None or x[0] >= start) and (end is None or x[0] <= end)]
        if len(s) < 2 or s[-1][0] <= s[0][0]:
            return None
        return round((s[-1][1] - s[0][1]) / (s[-1][0] - s[0][0]), 2)


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


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def interpreter_path(candidate, inherited):
    rest = [d for d in inherited.split(':') if d and d not in (str(PY.parent), str(CAND_PY.parent))]
    cuda = str(Path(CFG['cuda_home']) / 'bin')
    rest = [d for d in rest if d != cuda]
    return ':'.join(([str(CAND_PY.parent)] if candidate else []) + [str(PY.parent), cuda] + rest)


def under_test(tree):
    return Path(tree).resolve() != (ROOT / CFG['reference']).resolve()


def base_env(tmp, cand_env=None, candidate=False):
    """Environment of every training process: the caller's environment minus reserved variables, then the
    candidate's switches, then what the bench manages (scratch, caches, data, offline switches, toolkit, PATH)."""
    env = {k: v for k, v in os.environ.items() if k not in RESERVED and not k.startswith('BAGEL_BENCH')}
    env.update(cand_env or {})
    caches = Path(tmp) / 'caches'
    env.update({
        'TMPDIR': str(tmp), 'TMP': str(tmp), 'TEMP': str(tmp),
        'TORCHINDUCTOR_CACHE_DIR': str(caches / 'inductor'), 'TRITON_CACHE_DIR': str(caches / 'triton'),
        'CUDA_CACHE_PATH': str(caches / 'cuda'), 'XDG_CACHE_HOME': str(caches / 'xdg'),
        'MPLCONFIGDIR': str(caches / 'mpl'), 'BAGEL_BENCH_DATA': str(path(CFG['data'])),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
        'WANDB_MODE': 'offline', 'WANDB_DIR': str(Path(tmp) / 'wandb'), 'WANDB_SILENT': 'true',
        'PYTHONNOUSERSITE': '1', 'CUDA_HOME': CFG['cuda_home'], 'TRITON_LIBCUDA_PATH': '/usr/lib/x86_64-linux-gnu',
        'PATH': interpreter_path(candidate, env.get('PATH', '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin')),
    })
    env.pop('PYTHONPATH', None)
    for v in ('inductor', 'triton', 'cuda', 'xdg', 'mpl'):
        (caches / v).mkdir(parents=True, exist_ok=True)
    (Path(tmp) / 'wandb').mkdir(parents=True, exist_ok=True)
    return env


def run_job(tree, mode, steps, out_prefix, tmp, cand_env, candidate, timeout, what, seed=None, profile=None,
            data_config=None, judge_env=None, production=False, data_seed=None):
    """One run under the GPU lock: stale bench processes stopped, idle check, then torchrun of the judge's entry on the
    tree; everything started is killed afterwards. Reports the CPUs the container kept busy."""
    tag = uuid.uuid4().hex[:12]
    log_file = Path(f'{out_prefix}.log')
    with GpuLock(what), CpuSampler() as cpu:
        _kill_marked()
        before, problem = wait_idle(what)
        if problem:
            return {'status': 'INVALID', 'reason': problem, 'returncode': None, 'gpus_before': before}
        rc, status, t0 = None, None, time.time()
        try:
            extra = ['--profile-from', str(profile[0]), '--profile-iters', str(profile[1])] if profile else []
            if seed is not None:
                extra += ['--seed', str(seed)]
            if production:
                extra += ['--production']
            if data_seed is not None:
                extra += ['--data-seed', str(data_seed)]
            results = Path(tmp) / 'results'
            runner = [str(CAND_PY), '-m', 'torch.distributed.run'] if candidate else [str(TORCHRUN)]
            cmd = [*runner, '--nnodes=1', f'--nproc-per-node={CFG["world_size"]}', '--master-addr', '127.0.0.1',
                   '--master-port', str(free_port()), str(BENCH / 'bagel_entry.py'), '--mode', mode, '--out',
                   str(out_prefix), '--steps', str(steps), *extra, '--',
                   *recipe.argv(str(path(CFG['model'])), str(path(data_config or CFG['data_config'])), str(results))]
            env = base_env(tmp, cand_env, candidate=candidate)
            if mode in ('record', 'traj', 'conform') and not production:
                env.update(CFG['record']['environment'])
            env.update(judge_env or {})
            env[MARK] = tag
            with open(log_file, 'w') as lf:
                proc = subprocess.Popen(cmd, cwd=tree, env=env, stdout=lf, stderr=subprocess.STDOUT,
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
           'gpus_after': after, 'log': str(log_file), '_cpu': cpu,
           'cpu': {'busy_cpus': cpu.busy_cpus(), 'allowed_cpus': len(os.sched_getaffinity(0)),
                   'method': 'container cgroup CPU time / wall time over the run, sampled every second'}}
    if status == 'TIMEOUT':
        res['reason'] = f'{what} exceeded {timeout}s; process group killed; see {log_file.name}'
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


def rank_files(prefix):
    out = []
    for rank in range(CFG['world_size']):
        f = Path(f'{prefix}.r{rank}.json')
        out.append(json.loads(f.read_text()) if f.exists() else None)
    return out


def rank_errors(ranks, rc, log_file, steps):
    if rc == 0 and all(x and x.get('status') == 'ok' and x.get('steps') == steps for x in ranks):
        return None
    errors = [x['error'] for x in ranks if x and x.get('error')]
    tail = Path(log_file).read_text(errors='replace')[-1500:] if Path(log_file).exists() else ''
    return (f'the run exited {rc} before optimizer step {steps} on every rank; '
            f'{errors[:1] or "no rank error recorded"}; log tail: {tail}')


def clear(out, stem):
    for f in out.glob(f'{stem}.r*.json'):
        f.unlink()


# ---------------------------------------------------------------- inputs and sources

def tree_digest(tree):
    h = hashlib.sha256()
    tree = Path(tree)
    for f in sorted(tree.rglob('*')):
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
    if dirty or head != CFG['reference_commit']:
        raise SystemExit(f'reference tree is not the frozen upstream commit (head {head}, dirty: {bool(dirty)})')
    return ref


def materialize(src, dest):
    def ignore(directory, names):
        return [n for n in names if n in TREE_SKIP or n.endswith('.pyc')]
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
    bad = sorted(k for k in env if k in RESERVED or k.startswith('BAGEL_BENCH'))
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
        r = subprocess.run([str(python), '-c', VERSIONS, 'torch', 'torchvision', 'triton', 'flash_attn', 'transformers',
                            'numpy', 'safetensors'], capture_output=True, text=True, timeout=300,
                           env=dict(os.environ, PYTHONNOUSERSITE='1'))
        _VERSIONS[python] = (json.loads(r.stdout.strip().splitlines()[-1]) if not r.returncode else
                             {'error': f'{python} exited {r.returncode}: {r.stderr[-500:]}'})
    return _VERSIONS[python]


FINGERPRINT_KEYS = ('argv', 'gpus', 'compile_cache', 'instrumentation', 'timing_plan')


def fingerprint_diff(a, b):
    if a is None or b is None:
        return ['missing fingerprint']
    return [k for k in FINGERPRINT_KEYS if a.get(k) != b.get(k)]


# ---------------------------------------------------------------- gate 1: timing

def timing_plan():
    t = CFG['timing']
    return t['warmup'], t['window'], t['repeats']


def windows():
    """[(first step, last step)] of each timing window (0-based optimizer steps)."""
    warmup, window, repeats = timing_plan()
    return [(warmup + k * window, warmup + (k + 1) * window - 1) for k in range(repeats)]


def window_seconds(ranks):
    """Seconds per step of each window, the slowest rank deciding; and each window's wall span on rank 0."""
    out, spans = [], []
    for first, last in windows():
        per_rank = []
        for x in ranks:
            e = x['step_end_s']
            per_rank.append((e[last] - e[first - 1]) / (last - first + 1))
        out.append(max(per_rank))
        spans.append((ranks[0]['step_end_s'][first - 1], ranks[0]['step_end_s'][last]))
    return out, spans


def time_tree(tree, out, label, cand_env=None, candidate=True, profile=None, noise_gate=False):
    """One timing run. noise_gate (the final evaluation's reference -> candidate -> reference only): a run whose window
    range exceeds timing.max_window_spread is INVALID; elsewhere it is reported and the run stands."""
    warmup, window, repeats = timing_plan()
    steps = warmup + window * repeats if not profile else profile[0] + profile[1]
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f'timing_{label}'
    clear(out, f'timing_{label}')
    with scratch_dir('bg-t-') as tmp:
        r = run_job(tree, 'time', steps, prefix, tmp, cand_env, candidate, TIMEOUT['timing'],
                    f'timing {label}' + (' (profile)' if profile else ''), profile=profile,
                    data_config=CFG['timing']['data_config'],
                    data_seed=CFG['timing'].get('final_data_seed') if noise_gate else None)
    cpu = r.pop('_cpu', None)
    fingerprint = {'argv': ['bagel_entry.py --mode time', *recipe.argv('<model>', '<data config>', '<results>')],
                   'candidate_environment': cand_env or {},
                   'gpus': [(g['index'], g['name'], g['driver']) for g in r.get('gpus_before') or []],
                   'compile_cache': 'cold (fresh per run)', 'instrumentation': 'torch.profiler' if profile else 'none',
                   'timing_plan': [warmup, window, repeats] if not profile else ['profile', *profile],
                   'versions': versions(CAND_PY if candidate else PY)}
    run = {'run_status': r['status'], 'wall_s': r.get('wall_s'), 'fingerprint': fingerprint, 'log': r.get('log'),
           'cpu': r.get('cpu'), **({'warning': r['warning']} if r.get('warning') else {})}
    if r['status'] != 'ok':
        return {'status': r['status'], 'reason': r['reason'], **run}
    ranks = rank_files(prefix)
    err = rank_errors(ranks, r['returncode'], r['log'], steps)
    if err:
        return {'status': 'ERROR', 'reason': err, **run}
    hits = signatures(r['log'])
    if hits:
        return {'status': 'INVALID', 'reason': 'invalid-run signature in the log: ' + hits[0]['reason'],
                'signatures': hits, **run}
    e0 = ranks[0]['step_end_s']
    res = {'status': 'ok', 'steps': steps,
           'step_s_rank0': [round(e0[i] - e0[i - 1], 4) for i in range(1, len(e0))],
           'peak_memory_allocated_gib': round(max(x['peak_memory_allocated_bytes'] for x in ranks) / 2**30, 2),
           'peak_memory_reserved_gib': round(max(x['peak_memory_reserved_bytes'] for x in ranks) / 2**30, 2),
           'startup_to_first_step_s': round(max(x['startup_to_first_step_s'] for x in ranks), 1),
           'cpu_binding': [x.get('cpu_binding') for x in ranks],
           'other_optimizer_steps': sorted({x.get('other_optimizer_steps') for x in ranks}), **run, '_ranks': ranks}
    if profile:
        return res
    w, spans = window_seconds(ranks)
    mean = statistics.fmean(w)
    res.update(window_s_per_step=[round(x, 4) for x in w], s_per_step=round(mean, 4),
               window_range=round((max(w) - min(w)) / mean, 4),
               tokens_per_s=round(recipe.EXPECTED_TOKENS * CFG['world_size'] / mean))
    if cpu is not None and res.get('cpu'):
        res['cpu']['busy_cpus_in_windows'] = cpu.busy_cpus(spans[0][0], spans[-1][1])
    spread = CFG['timing']['max_window_spread']
    if res['window_range'] > spread:
        noisy = f'window range {res["window_range"]} (windows {res["window_s_per_step"]}) > {spread}: interference during the run'
        if noise_gate:                  # the final evaluation's A -> candidate -> B only
            res['status'] = 'INVALID'
            res['problems'] = [noisy + '; rerun later']
        else:
            res['notes'] = [noisy + ' (reported; only the final evaluation rejects a noisy run)']
    ref = frozen_reference_timing()
    if ref and res['status'] == 'ok':
        res['speedup_vs_frozen_reference'] = round(ref['s_per_step'] / res['s_per_step'], 4)
        res.update(reference_noise(ref, res['speedup_vs_frozen_reference']))
    return res


def strip(run):
    return {k: v for k, v in run.items() if k != '_ranks'} if run else run


# `time` without the preparation's frozen reference timing: one run of the reference stands in for it, kept here for
# the next `time` on the same GPUs (ensure_reference_timing).
REFERENCE_ONCE = SCRATCH / 'bagel_reference_timing_once.json'
_reference_once = None


def prepared_reference_timing():
    """This machine's frozen reference timing (the mean of the preparation's reference runs), or None."""
    f = path(CFG['timing']['frozen'])
    return json.loads(f.read_text()) if f.exists() else None


def frozen_reference_timing():
    """The preparation's frozen reference timing, else the one run of the reference `time` made (or None)."""
    return prepared_reference_timing() or _reference_once


def reference_once_key():
    """What a cached one-run reference timing must match: these GPUs, the timing plan and the reference."""
    return {'gpus': sorted(g['uuid'] for g in gpu_state()), 'world_size': CFG['world_size'],
            'timing_plan': list(timing_plan()), 'reference_commit': CFG['reference_commit']}


def ensure_reference_timing(out, tree):
    """`time` reads its speedup against the frozen reference timing of the preparation. Without one (this machine
    was not prepared, or the file is missing), the reference is timed once here and that run stands in for it, cached
    in REFERENCE_ONCE for the next `time` on the same GPUs, timing plan and reference; its run-to-run noise is not
    measured. Nothing to do when the frozen timing exists or when the tree timed is the reference itself. Returns
    (how the reference timing was obtained, the reference run timed now or None)."""
    global _reference_once
    if prepared_reference_timing():
        return 'frozen', None
    if not under_test(tree):
        return None, None
    key = reference_once_key()
    try:
        cached = json.loads(REFERENCE_ONCE.read_text())
    except (OSError, ValueError):
        cached = None
    if cached and cached.get('key') == key:
        _reference_once = cached
        log(f'no frozen reference timing: using the run of the reference cached in {REFERENCE_ONCE}')
        return 'one run of the reference, cached', None
    log('no frozen reference timing: timing the reference once first')
    with scratch_dir('bg-ref-') as tmp:
        res = strip(time_tree(materialize(check_reference(), Path(tmp) / 'reference'), out, 'reference_once',
                              candidate=False))
    if res['status'] != 'ok':
        log(f'the reference run failed ({res["status"]}: {res.get("problems") or res.get("reason")}): no speedup')
        return 'one run of the reference, failed', res
    _reference_once = dict(freeze_reference_timing([res]), spread=None, sigma_rel=None, key=key,
                           source='one run of the reference by time: no frozen reference timing was found')
    REFERENCE_ONCE.write_text(json.dumps(_reference_once, indent=1) + '\n')
    return 'one run of the reference, timed now', res


def reference_noise(ref, speedup):
    """The frozen reference's own run-to-run spread and whether a speedup over it is beyond that noise: the same rule
    as gate 1's bar, 1 + 2 x spread."""
    spread = ref.get('spread')
    if spread is None:
        return {'reference_noise': {'runs': len(ref.get('runs') or []), 'spread': None, 'band': None,
                                    'note': 'one run of the reference: its run-to-run noise is not measured'},
                'beyond_reference_noise': None}
    band = round(2 * spread, 4)
    return {'reference_noise': {'runs': len(ref.get('runs') or []), 'spread': spread, 'band': band},
            'beyond_reference_noise': speedup - 1 > band}


def freeze_reference_timing(results):
    """The frozen reference timing from reference runs: the mean s/step and per-window seconds per step, their spread
    ((max - min) / mean, the noise the agent's `time` is read against)."""
    s = [r['s_per_step'] for r in results]
    mean = statistics.fmean(s)
    return {'s_per_step': round(mean, 5),
            'window_s_per_step': [round(statistics.fmean(w), 5) for w in zip(*(r['window_s_per_step'] for r in results))],
            'peak_memory_allocated_gib': max(r['peak_memory_allocated_gib'] for r in results),
            'spread': round((max(s) - min(s)) / mean, 4),
            'sigma_rel': round(statistics.stdev(s) / mean, 4) if len(s) > 1 else None,
            'timing_plan': list(timing_plan()),
            'runs': [{'s_per_step': r['s_per_step'], 'window_s_per_step': r['window_s_per_step']} for r in results]}


def benefit_frozen(tree, out, cand_env):
    """Gate 1 of the agent's check: one candidate run against the frozen reference timing of the preparation (the mean
    of its timing.reference_runs reference runs, each warmup + window x repeats; task/timing/reference.json). The bar
    is the final evaluation's (min_speedup, or the frozen reference's spread doubled, over one: the larger); the noise rules are reported,
    never a rejection. The reference -> candidate -> reference comparison is the final evaluation's (check --final)."""
    source, ref_once = ensure_reference_timing(out, tree)
    frozen = frozen_reference_timing()
    if not frozen:
        return 'ERROR', {'speedup': None, 'runs': {'reference_once': ref_once},
                         'reason': 'no reference timing (the reference run failed)'}
    cand = strip(time_tree(tree, out, 'cand', cand_env=cand_env))
    runs = {'cand': cand, **({'reference_once': ref_once} if ref_once else {})}
    if cand['status'] != 'ok':
        status = 'INVALID' if cand['status'] in ('INVALID', 'TIMEOUT') else 'ERROR'
        return status, {'speedup': None, 'runs': runs,
                        'reason': (cand.get('reason') or '; '.join(cand.get('problems') or [])
                                   or 'the timing run failed')}
    spread = frozen.get('spread') or 0.0
    speedup = cand['speedup_vs_frozen_reference']
    perf = {'reference_s_per_step': frozen['s_per_step'], 'candidate_s_per_step': cand['s_per_step'],
            'speedup': speedup, 'reference_spread': spread, 'reference': source or 'frozen',
            'comparison': 'candidate run against the frozen reference timing (the final evaluation times '
                          'reference -> candidate -> reference)',
            'windows': windows(), 'runs': runs, 'library_versions': {'candidate': cand['fingerprint']['versions']},
            'host': socket.gethostname(), **({'notes': cand['notes']} if cand.get('notes') else {})}
    if frozen.get('window_s_per_step'):
        perf['diagnostics'] = {'window_ratios': [round(a / x, 4) for a, x in
                                                 zip(frozen['window_s_per_step'], cand['window_s_per_step'])]}
    bar = max(CFG['benefit']['min_speedup'], 1 + 2 * spread)
    perf['benefit'] = {'speedup': speedup, 'speedup_bar': round(bar, 4),
                       'peak_memory_allocated_gib': cand['peak_memory_allocated_gib'],
                       'reference_peak_memory_allocated_gib': frozen.get('peak_memory_allocated_gib'),
                       'passed': speedup > bar}
    return ('ok' if perf['benefit']['passed'] else 'UNAVAILABLE'), perf


def benefit(tree, out, cand_env, final=False):
    """Gate 1. The agent's check (not final): benefit_frozen. The final evaluation (final): reference -> candidate ->
    reference; the noise rules (a run's window range, a reference off the frozen timing) make the result INVALID
    there."""
    if not final:
        return benefit_frozen(tree, out, cand_env)
    t = CFG['timing']
    ref_src = check_reference()
    with scratch_dir('bg-ref-') as tmp:
        ref_tree = materialize(ref_src, Path(tmp) / 'reference')
        ref_a = time_tree(ref_tree, out, 'ref_a', candidate=False, noise_gate=final)
        cand = time_tree(tree, out, 'cand', cand_env=cand_env, noise_gate=final) if ref_a['status'] == 'ok' else None
        ref_b = (time_tree(ref_tree, out, 'ref_b', candidate=False, noise_gate=final)
                 if cand and cand['status'] == 'ok' else None)
    runs = {'ref_a': strip(ref_a), 'cand': strip(cand), 'ref_b': strip(ref_b)}
    bad = {k: v for k, v in runs.items() if v is None or v['status'] != 'ok'}
    if bad:
        status = 'INVALID' if any(v and v['status'] in ('INVALID', 'TIMEOUT') for v in bad.values()) else 'ERROR'
        reason = next((v.get('reason') or '; '.join(v.get('problems') or []) for v in bad.values()
                       if v and (v.get('reason') or v.get('problems'))), 'a timing run failed')
        return status, {'speedup': None, 'runs': runs, 'reason': reason}
    diffs = {k: fingerprint_diff(ref_a.get('fingerprint'), v.get('fingerprint')) for k, v in runs.items() if k != 'ref_a'}
    versions_seen = {'reference': ref_a['fingerprint']['versions'], 'candidate': cand['fingerprint']['versions']}
    versions_seen['differ'] = versions_seen['reference'] != versions_seen['candidate']
    badfp = {k: d for k, d in diffs.items() if d}
    if badfp:
        return 'ERROR', {'speedup': None, 'runs': runs, 'fingerprint_diffs': diffs, 'library_versions': versions_seen,
                         'reason': f'timing runs not comparable: fingerprint differs from ref_a in {badfp}'}
    ra, rb, c = ref_a['s_per_step'], ref_b['s_per_step'], cand['s_per_step']
    ref = (ra + rb) / 2
    spread = abs(ra - rb) / ref
    perf = {'reference_s_per_step': round(ref, 4), 'candidate_s_per_step': c, 'speedup': round(ref / c, 4),
            'reference_spread': round(spread, 4), 'windows': windows(), 'runs': runs, 'fingerprint_diffs': diffs,
            'library_versions': versions_seen, 'host': socket.gethostname(),
            'diagnostics': {'window_ratios': [round(statistics.fmean([a, b]) / x, 4) for a, b, x in
                                              zip(ref_a['window_s_per_step'], ref_b['window_s_per_step'],
                                                  cand['window_s_per_step'])]}}
    frozen, notes = prepared_reference_timing(), []
    for k in ('ref_a', 'ref_b'):
        if frozen is None:
            notes.append('no frozen reference timing: the reference drift is not checked')
            break
        dev = abs(runs[k]['s_per_step'] / frozen['s_per_step'] - 1)
        if dev > t['max_reference_drift']:
            notes.append(f'{k} is {dev:.1%} off the frozen reference timing (> {t["max_reference_drift"]:.0%}): the '
                         f'host is not in the state it was prepared in')
    if notes:
        if final and frozen is not None:
            perf['problems'] = [n + '; measure again later' for n in notes]
            return 'INVALID', perf
        perf['notes'] = [n + ' (reported; only the final evaluation rejects it)' for n in notes]
    bar = max(CFG['benefit']['min_speedup'], 1 + 2 * spread)
    perf['benefit'] = {'speedup': perf['speedup'], 'speedup_bar': round(bar, 4),
                       'peak_memory_allocated_gib': cand['peak_memory_allocated_gib'],
                       'reference_peak_memory_allocated_gib': max(ref_a['peak_memory_allocated_gib'],
                                                                  ref_b['peak_memory_allocated_gib']),
                       'passed': perf['speedup'] > bar}
    return ('ok' if perf['benefit']['passed'] else 'UNAVAILABLE'), perf


# ---------------------------------------------------------------- gate 2: deterministic record

def record_tree(tree, out, label, cand_env=None, candidate=True):
    """One deterministic record run -> the record dict (or a failure dict with 'status')."""
    steps = CFG['record']['steps']
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f'record_{label}'
    clear(out, f'record_{label}')
    t0 = time.time()
    with scratch_dir('bg-r-') as tmp:
        r = run_job(tree, 'record', steps, prefix, tmp, cand_env, candidate, TIMEOUT['record'], f'record {label}')
    r.pop('_cpu', None)
    if r['status'] != 'ok':
        return {'status': r['status'], 'reason': r['reason'], 'log': r.get('log')}
    ranks = rank_files(prefix)
    err = rank_errors(ranks, r['returncode'], r['log'], steps)
    if err:
        return {'status': 'ERROR', 'reason': err, 'log': r.get('log')}
    return dict(aggregate_record(ranks, steps), wall_s=r.get('wall_s'), log=r.get('log'), cpu=r.get('cpu'),
                record_seconds=round(time.time() - t0, 1))


def aggregate_record(ranks, steps):
    """The record of one run from its rank files: digests keyed by rank."""
    r0 = ranks[0]
    return {'status': 'ok', 'steps': r0['steps'],
            'digests': [{f'r{x["rank"]}.{e[0]}': e[1:] for x in ranks for e in x['digests'][k]} for k in range(steps)],
            'parameters': r0['parameters'], 'ema_parameters': r0['ema_parameters'],
            'parameters_all_ranks_equal': all(x['parameters'] == r0['parameters'] for x in ranks),
            'hparams': r0['hparams'],
            'hparams_all_ranks_equal': all(x['hparams'] == r0['hparams'] for x in ranks),
            'state_dtypes': sorted({tuple(d) for x in ranks for d in x['state_dtypes']}),
            'batches': [x['batches'][:steps] for x in ranks],
            'batches_seen': [len(x['batches']) for x in ranks],
            'other_optimizer_steps': max(x.get('other_optimizer_steps') or 0 for x in ranks),
            'deterministic': {'patched': r0.get('deterministic_patched'),
                              'algorithms': r0.get('deterministic_algorithms')},
            'dataset_info': r0.get('dataset_info'),
            'optimizer': r0.get('optimizer'),
            'peak_memory_allocated_gib': round(max(x['peak_memory_allocated_bytes'] for x in ranks) / 2**30, 2)}


def compare_records(ref, cand):
    """Exact items first (any difference: INCORRECT), then the bitwise comparison of every shard after every step.
    Returns the numerics section of check.json."""
    exact = []
    if cand['steps'] != ref['steps']:
        exact.append(f'{cand["steps"]} optimizer steps recorded, the reference {ref["steps"]}')
    if cand.get('other_optimizer_steps'):
        exact.append(f'{cand["other_optimizer_steps"]} steps of a second optimizer')
    if not cand.get('hparams_all_ranks_equal', True) or not cand.get('parameters_all_ranks_equal', True):
        exact.append('the hyper-parameters or the trained parameters differ between ranks')
    for k, (c, r) in enumerate(zip(cand['hparams'], ref['hparams'])):
        if c != r:
            exact.append(f'step {k}: parameter groups stepped with {c}, the reference with {r} '
                         f'([lr, betas, eps, weight_decay, amsgrad, maximize, elements] per group)')
            break
    for rank, (c, r, seen) in enumerate(zip(cand['batches'], ref['batches'], cand['batches_seen'])):
        if seen < ref['steps']:
            exact.append(f'rank {rank}: {seen} batches moved to the GPU through SimpleCustomBatch.cuda in '
                         f'{ref["steps"]} steps')
            break
        if c != r:
            k = next(i for i, (a, b) in enumerate(zip(c, r)) if a != b)
            exact.append(f'rank {rank} trained on a different batch than the reference at step {k}')
            break
    for what in ('parameters', 'ema_parameters'):
        pc = {p[0]: p[1:] for p in cand[what]}
        pr = {p[0]: p[1:] for p in ref[what]}
        if set(pc) != set(pr):
            exact.append(f'{what}: names differ: missing {sorted(set(pr) - set(pc))[:5]}, new '
                         f'{sorted(set(pc) - set(pr))[:5]}')
            continue
        shapes = [n for n in pr if pc[n][0] != pr[n][0]]
        dtypes = [n for n in pr if pc[n][1] != pr[n][1]]
        trained = [n for n in pr if what == 'parameters' and pc[n][2] != pr[n][2]]
        if shapes:
            exact.append(f'{what}: shapes differ: {shapes[:5]}')
        if dtypes:
            exact.append(f'precision change (not accepted): {what} dtypes differ: '
                         f'{[(n, pc[n][1], pr[n][1]) for n in dtypes[:5]]}')
        if trained:
            exact.append(f'the parameters that are trained differ: {trained[:5]}')
    new_state = sorted({tuple(x) for x in cand['state_dtypes']} - {tuple(x) for x in ref['state_dtypes']})
    if new_state:
        exact.append(f'precision change (not accepted): optimizer state tensors of a dtype the reference '
                     f'does not have: {new_state}')
    res = {'exact_items_ok': not exact, 'problems': exact,
           'compared': {'steps': ref['steps'], 'parameters': len(ref['parameters']),
                        'shards_per_step': len(ref['digests'][0]) if ref['digests'] else 0}}
    if exact:
        return dict(res, status='INCORRECT', bitwise=False, measured_tier=None)
    differing = []
    for k, (c, r) in enumerate(zip(cand['digests'], ref['digests'])):
        if c != r:
            keys = sorted(set(c) | set(r))
            bad = [x for x in keys if c.get(x) != r.get(x)]
            differing.append({'step': k, 'shards': len(bad), 'first': bad[:3],
                              'layout_differs': set(c) != set(r)})
    if not differing:
        return dict(res, status='PASS', bitwise=True, measured_tier='N0')
    return dict(res, status='NOT_BITWISE', bitwise=False, measured_tier=None, differing=differing[:3],
                first_differing_step=differing[0]['step'])


def numerics(tree, out, cand_env):
    ref = json.loads(path(CFG['record']['reference']).read_text())
    cand = record_tree(tree, out, 'cand', cand_env=cand_env)
    if cand.get('status') != 'ok':
        return {'status': cand.get('status', 'ERROR'), 'reason': cand.get('reason'), 'log': cand.get('log')}
    t0 = time.time()
    res = compare_records(ref, cand)
    res['compare_seconds'] = round(time.time() - t0, 2)
    res['run'] = {k: cand.get(k) for k in ('wall_s', 'record_seconds', 'peak_memory_allocated_gib', 'deterministic',
                                           'optimizer', 'batches_seen', 'cpu', 'log')}
    return res


# ---------------------------------------------------------------- gate 2b: interleaved-layout conformance

def conform_run(tree, out, label, cand_env=None, candidate=True, pools=False, production=False, data_seed=None):
    """One deterministic run of conformance.steps steps on the held-out data configuration -> {'status': 'ok',
    'ranks': [...]} or a failure dict (bench/conform.py, bagel_entry.py --mode conform)."""
    cf = CFG['conformance']
    steps = cf['steps']
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f'conform_{label}'
    clear(out, prefix.name)
    with scratch_dir('bg-c-') as tmp:
        r = run_job(tree, 'conform', steps, prefix, tmp, cand_env, candidate, TIMEOUT['conformance'],
                    f'conformance {label}', data_config=cf['data_config'],
                    judge_env={'BAGEL_BENCH_CONFORM_POOLS': '1'} if pools else None, production=production,
                    data_seed=data_seed)
    r.pop('_cpu', None)
    if r['status'] != 'ok':
        return {'status': r['status'], 'reason': r['reason'], 'log': r.get('log')}
    ranks = rank_files(prefix)
    err = rank_errors(ranks, r['returncode'], r['log'], steps)
    if err:
        return {'status': 'ERROR', 'reason': err, 'log': r.get('log')}
    keep = []
    for x in ranks:
        forwards = x['conform'][:steps]
        if pools:
            for f, flags in zip(forwards, x['pools']):
                f['pool'] = flags
        keep.append({'rank': x['rank'], 'forwards': forwards, 'forward_calls': x['forwards'],
                     'batches': x['batches'][:steps], 'batches_seen': len(x['batches']),
                     'vae_draws': x['vae_draws']})
    return {'status': 'ok', 'steps': steps, 'ranks': keep, 'wall_s': r.get('wall_s'), 'cpu': r.get('cpu'),
            'log': r.get('log')}


def compare_conform(ref, cand):
    """Gate 2b's verdict from the frozen reference run and the candidate's: exact items (the same interleaved batches,
    one Bagel.forward per step, the same MSE tokens, the judge's noise reached both judged calls), then the statistic
    (bench/conform.py). Returns the conformance section of check.json."""
    cf = CFG['conformance']
    exact, unpinned = [], []
    steps = ref['steps']
    for r, c in zip(ref['ranks'], cand['ranks']):
        rank = r['rank']
        if c['forward_calls'] != steps:
            exact.append(f'rank {rank}: Bagel.forward called {c["forward_calls"]} times in {steps} steps')
        if c['batches'] != r['batches']:
            k = next((i for i, (a, b) in enumerate(zip(c['batches'], r['batches'])) if a != b), len(c['batches']))
            exact.append(f'rank {rank} trained on a different interleaved batch than the reference at step {k}')
        for k, (fr, fc) in enumerate(zip(r['forwards'], c['forwards'])):
            if fc['mse'] is None or len(fc['mse']) != len(fr['mse']):
                exact.append(f'rank {rank} step {k}: {None if fc["mse"] is None else len(fc["mse"])} per-token MSE '
                             f'values, the reference {len(fr["mse"])}')
            if fc['latent_draws'] != fr['latent_draws']:
                unpinned.append(f'rank {rank} step {k}: latent noise drawn as {fc["latent_draws"]} inside '
                                f'Bagel.forward, the reference {fr["latent_draws"]}')
        rv, cv = r['vae_draws'], c['vae_draws']
        if [v['draws'] for v in cv[:len(rv)]] != [v['draws'] for v in rv]:
            unpinned.append(f'rank {rank}: VAE posterior samples drawn as {[v["draws"] for v in cv[:len(rv)]]} in '
                            f'AutoEncoder.encode, the reference {[v["draws"] for v in rv]}')
    res = {'steps': steps, 'data_config': cf['data_config'], 'ratio_max': cf['ratio_max'], 'floor': cf['floor']}
    if exact:
        return dict(res, status='INCORRECT', verdict='exact_items', problems=exact[:8])
    if unpinned:
        return dict(res, status='ERROR', verdict='noise_not_pinned', problems=unpinned[:8],
                    note='the judge could not give the candidate its noise (hook points: the VAE posterior sample '
                         'drawn in AutoEncoder.encode and the latent noise in Bagel.forward, each with '
                         'torch.randn_like, torch.randn or Tensor.normal_)')
    stat = conform.verdict(conform.statistic([r['forwards'] for r in ref['ranks']],
                                             [c['forwards'] for c in cand['ranks']]), cf['ratio_max'], cf['floor'])
    batches = stat.pop('batches')
    stat['worst_batches'] = sorted(batches, key=lambda b: -max(b['r_n'] or 0, b['r_j'] or 0))[:4]
    return dict(res, status='PASS' if stat['verdict'] == 'conforming' else 'INCORRECT', **stat)


FINAL_EVALUATION = False   # set by `check --final`: gates 1 and 2b run on timing/conformance.final_data_seed


def conformance(tree, out, cand_env):
    """Gate 2b: the candidate on the held-out interleaved data configuration against the frozen reference run, in
    deterministic mode and then in production mode (the code path timing runs, FlashAttention 3 included). The final
    evaluation does both on another data seed."""
    cf = CFG['conformance']
    data_seed = cf.get('final_data_seed') if FINAL_EVALUATION else None
    ref_key, prod_key = ('final_reference', 'final_production_reference') if FINAL_EVALUATION else \
        ('reference', 'production_reference')
    ref = json.loads(path(cf[ref_key]).read_text())
    t0 = time.time()
    cand = conform_run(tree, out, 'cand', cand_env=cand_env, data_seed=data_seed)
    if cand.get('status') != 'ok':
        return {'status': cand.get('status', 'ERROR'), 'verdict': 'error', 'reason': cand.get('reason'),
                'log': cand.get('log')}
    res = compare_conform(ref, cand)
    res['run'] = {'wall_s': cand.get('wall_s'), 'cpu': cand.get('cpu'), 'log': cand.get('log'),
                  'gate_seconds': round(time.time() - t0, 1)}
    (out / 'conform_report.json').write_text(json.dumps(res, indent=1) + '\n')
    if res.get('status') == 'PASS':
        cand_p = conform_run(tree, out, 'cand_prod', cand_env=cand_env, production=True, data_seed=data_seed)
        if cand_p.get('status') != 'ok':
            return {'status': cand_p.get('status', 'ERROR'), 'verdict': 'error', 'reason': cand_p.get('reason'),
                    'log': cand_p.get('log'), 'stage': 'production'}
        prod = compare_conform(json.loads(path(cf[prod_key]).read_text()), cand_p)
        (out / 'conform_report_production.json').write_text(json.dumps(prod, indent=1) + '\n')
        res['production'] = prod
        if prod.get('status') != 'PASS':
            res.update(verdict=prod.get('verdict'), status=prod.get('status'), stage='production',
                       problems=prod.get('problems'))
    return res


# ---------------------------------------------------------------- gate 3: trajectories

def traj_run(tree, out, label, seed, cand_env=None, candidate=True):
    """One deterministic trajectory -> {'status': 'ok', 'series': {name: [values]}} or a failure dict."""
    steps = CFG['trajectory']['steps']
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f'traj_{label}_s{seed}'
    clear(out, prefix.name)
    with scratch_dir('bg-j-') as tmp:
        r = run_job(tree, 'traj', steps, prefix, tmp, cand_env, candidate, TIMEOUT['trajectory'],
                    f'trajectory {label} seed {seed}', seed=seed, data_config=CFG['trajectory']['data_config'],
                    production=CFG['trajectory'].get('production', False))
    r.pop('_cpu', None)
    if r['status'] != 'ok':
        return {'status': r['status'], 'reason': r['reason'], 'log': r.get('log')}
    ranks = rank_files(prefix)
    err = rank_errors(ranks, r['returncode'], r['log'], steps)
    if err:
        return {'status': 'ERROR', 'reason': err, 'log': r.get('log')}
    if any((x.get('perturbed') or {}).get('seed') != seed for x in ranks):
        return {'status': 'ERROR', 'reason': 'the seed perturbation was not applied on every rank', 'log': r.get('log')}
    series = {k: ranks[0]['series'][k] for k in SERIES}
    disp = ranks[0]['displacement']
    for i, name in enumerate(DISP_SERIES):
        series[name] = [d[i] for d in disp]
    return {'status': 'ok', 'series': series, 'wall_s': r.get('wall_s'), 'cpu': r.get('cpu'), 'log': r.get('log')}


def trajectories(tree, out, cand_env, seeds=None, report_name='traj_report.json'):
    """Gate 3, the same in every check (the agent's, `check --n1`, the final evaluation's): the candidate's trajectory
    on `seeds`, by default the first trajectory.seeds (1) of the frozen reference seeds, each endpoint's bias against
    trajectory.sigma_k (3) x the reference's seed sigma (bench/traj.py bound). Every seed is its own process."""
    tj = CFG['trajectory']
    frozen = json.loads(path(tj['frozen']).read_text())
    runs = {}
    for seed in seeds or frozen['reference_seeds'][:tj['seeds']]:
        run = traj_run(tree, out, 'cand', seed, cand_env=cand_env)
        if run['status'] != 'ok':
            return {'verdict': 'error', 'status': run['status'], 'reason': run.get('reason'), 'seed': seed,
                    'log': run.get('log')}
        runs[seed] = run['series']
    report = trajectory.bound(frozen, runs, k=tj['sigma_k'])
    (out / report_name).write_text(json.dumps(report, indent=1) + '\n')
    report['series'] = {str(s): v for s, v in runs.items()}
    return report


FINAL_MARKER = Path('/run/researchboost/final-evaluation')


def final_allowed():
    """`check --final` (reference -> candidate -> reference timing, interference rules enforced) is the final
    evaluation's: allowed for root (the preparation) and for the verifier, which writes FINAL_MARKER as root in a
    root-owned directory before it runs the judge as the agent user. The agent cannot create it."""
    if os.geteuid() == 0:
        return True
    try:
        st, parent = FINAL_MARKER.stat(), FINAL_MARKER.parent.stat()
    except OSError:
        return False
    return st.st_uid == 0 and parent.st_uid == 0 and not parent.st_mode & 0o022


# ---------------------------------------------------------------- verdict

def coverage(tree):
    missing = [f for f in ('train/pretrain_unified_navit.py', 'modeling/bagel/bagel.py', 'data/dataset_base.py',
                           'data/dataset_info.py', 'train/fsdp_utils.py') if not (Path(tree) / f).is_file()]
    if missing:
        return 'ERROR', {'passed': False, 'stage': 'structure', 'note': f'missing: {missing}'}
    return None


def statistical(tree, out, cand_env, correctness, performance):
    """Gate 3 and its verdict (of every check and of check --n1). A candidate with an endpoint beyond the bound on the
    first seed is measured once more on the next frozen seed (trajectory.retest_seeds) and is INCORRECT only when that
    seed is beyond too (owner, 11-18): one production trajectory on one seed rejected an equivalent control, the
    cross-entropy reorder, with its ViT displacement at 3.6 sigma."""
    tj = CFG['trajectory']
    traj = trajectories(tree, out, cand_env)
    if traj['verdict'] == 'different' and tj.get('retest_seeds', 0):
        frozen = json.loads(path(tj['frozen']).read_text())
        again = frozen['reference_seeds'][tj['seeds']:tj['seeds'] + tj['retest_seeds']]
        if again:
            log(f'gate 3: an endpoint on seed {traj["seeds"]} is beyond the bound; measuring once more on seed {again}')
            first = {k: traj.get(k) for k in ('seeds', 'verdict', 'endpoints', 'error')}
            traj = dict(trajectories(tree, out, cand_env, seeds=again, report_name='traj_report_retest.json'),
                        first_seed=first)
    correctness = dict(correctness, stage='statistical', trajectory=traj)
    if traj['verdict'] == 'equivalent':
        if 'first_seed' in traj:
            correctness['note'] = (f'seed {traj["first_seed"]["seeds"]} was beyond the bound, seed {traj["seeds"]} '
                                   'within: gate 3 passes on the second seed')
        return 'PASS', dict(correctness, passed=True, verdict='equivalent', measured_tier='N1'), performance
    if traj['verdict'] == 'error':
        return ('INVALID' if traj.get('status') in ('INVALID', 'TIMEOUT') else 'ERROR'), \
            dict(correctness, passed=False,
                 note=traj.get('reason') or traj.get('error') or 'a trajectory run failed'), performance
    seeds = (f'seed {traj["first_seed"]["seeds"]} and again on seed {traj["seeds"]}' if 'first_seed' in traj
             else f'seed {traj["seeds"]}')
    return 'INCORRECT', dict(correctness, passed=False, verdict='different',
                             note=traj.get('error') or f'an endpoint\'s bias on {seeds} is beyond '
                                  f'{CFG["trajectory"]["sigma_k"]} x the reference\'s seed sigma'), performance


def evaluate(tree, out, skip_timing=False, cand_env=None, quick=False, final=True, n1=False):
    """Gates 0 -> 1 -> 2 -> 2b -> 3 (quick: 0 and 2; n1: 0, 2b and 3). Returns (status, correctness, performance)."""
    cand_env = cand_env or {}
    failed = coverage(tree)
    if failed:
        return failed[0], failed[1], None
    if n1:
        # check --n1: after a check whose gate 2 was not bitwise identical, gates 2b and 3 (no timing, no records).
        correctness = {'gates_run': [0, '2b', 3], 'note': 'check --n1: gates 2b and 3 only; gates 1 and 2 not run '
                                                          '(the final evaluation runs every gate)'}
        failed = conformance_gate(tree, out, cand_env, correctness, None)
        if failed:
            return failed
        return statistical(tree, out, cand_env, correctness, None)
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
    correctness = {'stage': 'record', 'measured_tier': num.get('measured_tier'), 'numerics': num}
    if num.get('status') not in ('PASS', 'INCORRECT', 'NOT_BITWISE'):
        st = 'INVALID' if num.get('status') in ('INVALID', 'TIMEOUT') else 'ERROR'
        return st, dict(correctness, passed=False, note=num.get('reason') or 'the record run failed'), performance
    if num['status'] == 'INCORRECT':
        return 'INCORRECT', dict(correctness, passed=False, verdict='exact_items',
                                 note='; '.join(num['problems'])), performance
    if quick:
        if num['bitwise']:
            return 'PASS', dict(correctness, passed=True, verdict='bitwise',
                                stage='record (quick: gates 1, 2b and 3 not run)'), performance
        return 'NOT_BITWISE', dict(correctness, passed=None, verdict='not_bitwise',
                                   note='exact items pass, not bitwise identical (first differing step '
                                        f'{num.get("first_differing_step")}); the statistical gate (a full check or '
                                        '--skip-timing) decides'), performance
    failed = conformance_gate(tree, out, cand_env, correctness, performance)
    if failed:
        return failed
    status, corr, perf = statistical(tree, out, cand_env, correctness, performance)
    if num['bitwise'] and status == 'PASS':
        # gate 2 judged the deterministic path bitwise; gate 3 ran anyway on the production (timed) path
        return 'PASS', dict(corr, verdict='bitwise', measured_tier='N0'), perf
    return status, corr, perf


def conformance_gate(tree, out, cand_env, correctness, performance):
    """Gate 2b in a check (also for a tree bitwise identical on gate 2: a mask that is wrong only on interleaved
    layouts is bitwise identical on the recipe's data). Adds its section to correctness; returns the check's result
    when it stops the evaluation, else None."""
    conf = conformance(tree, out, cand_env)
    correctness['conformance'] = conf
    if conf['status'] == 'PASS':
        return None
    if conf['status'] == 'INCORRECT':
        note = ('; '.join(conf.get('problems') or []) if conf.get('verdict') == 'exact_items' else
                f'interleaved layouts: R {conf.get("r")} against the bound {conf.get("ratio_max")} x max(R_B, '
                f'floor {conf.get("floor")}) = {conf.get("bound")}')
        return 'INCORRECT', dict(correctness, passed=False, stage='conformance', measured_tier=None,
                                 verdict='nonconforming', note=note), performance
    st = 'INVALID' if conf['status'] in ('INVALID', 'TIMEOUT') else 'ERROR'
    return st, dict(correctness, passed=False, stage='conformance', measured_tier=None,
                    note=conf.get('note') or '; '.join(conf.get('problems') or []) or conf.get('reason')
                    or 'the conformance run failed'), performance


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
        with scratch_dir('bg-src-') as tmp:
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
    for k in ('speedup_vs_frozen_reference', 'beyond_reference_noise', 'reference_noise', 'speedup_vs_paired_reference',
              's_per_step', 'reference_timing', 'notes'):
        if k in res:
            summary[k] = res[k]
    print(json.dumps(summary))
    return 0 if res.get('status') in ('PASS', 'ok') else 1


def cmd_check(a):
    out = Path(a.out).resolve()
    if a.final and not final_allowed():
        raise SystemExit('check --final is the final evaluation\'s (run by the harness after your time is up) and is '
                         'not available here. `check` runs every gate the final evaluation runs; its timing reads one '
                         'run against the frozen reference timing instead of reference -> candidate -> reference.')
    global FINAL_EVALUATION
    FINAL_EVALUATION = bool(a.final)
    if a.n1 and (a.final or a.quick or a.skip_timing):
        raise SystemExit('check --n1 runs gates 0 and 3 only; it takes no --final, --quick or --skip-timing')
    cand_env = candidate_environment(a.config)

    def run(tree):
        status, correctness, performance = evaluate(tree, out, skip_timing=a.skip_timing, cand_env=cand_env,
                                                    quick=a.quick, final=a.final, n1=a.n1)
        return {'status': status, 'tree': str(Path(a.tree).resolve()), 'config': a.config, 'environment': cand_env,
                'quick': a.quick, 'skip_timing': a.skip_timing, 'n1': a.n1, 'final': a.final,
                'correctness': correctness, 'performance': performance}
    return finish(out, 'check.json', guarded(out, run, Path(a.tree).resolve()))


def cmd_time(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)

    def run(tree):
        reference, ref = ensure_reference_timing(out, a.tree)
        if a.paired and ref is None:      # the one run of the reference just made serves as the paired one
            with scratch_dir('bg-ref-') as tmp:
                ref = strip(time_tree(materialize(check_reference(), Path(tmp) / 'reference'), out, 'reference',
                                      candidate=False))
        res = strip(time_tree(tree, out, 'tree', cand_env=cand_env, candidate=under_test(a.tree)))
        if reference:
            res['reference_timing'] = reference
        if a.paired and ref is not None:
            res['paired_reference'] = ref
            if res['status'] == 'ok' and ref['status'] == 'ok':
                res['speedup_vs_paired_reference'] = round(ref['s_per_step'] / res['s_per_step'], 4)
                res['note'] = ('screening only: one reference run just before, same host and session; the final '
                               'evaluation times reference -> candidate -> reference')
        return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env,
                    host=socket.gethostname())
    return finish(out, 'time.json', guarded(out, run, Path(a.tree).resolve()))


def cmd_profile(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)

    def run(tree):
        res = strip(time_tree(tree, out, 'profile', cand_env=cand_env, candidate=under_test(a.tree),
                              profile=(a.start, a.iters)))
        if res['status'] == 'ok':
            prof = out / 'timing_profile.profile.txt'
            res['profile_table'] = str(prof) if prof.exists() else None
            res['trace'] = str(out / 'timing_profile.trace.json')
            res['note'] = (f'optimizer steps {a.start}..{a.start + a.iters - 1} under torch.profiler on rank 0; '
                           f'instrumented: not comparable with `time`')
        return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env)
    return finish(out, 'profile.json', guarded(out, run, Path(a.tree).resolve()))


def cmd_baseline(a):
    """Formulator: the reference timed timing.reference_runs times; their mean and spread frozen (freeze-timing)."""
    out = Path(a.out).resolve()

    def run(_):
        labels = ['ref_%s' % chr(ord('a') + i) for i in range(CFG['timing'].get('reference_runs', 3))]
        ref_src = check_reference()
        with scratch_dir('bg-ref-') as tmp:
            tree = materialize(ref_src, Path(tmp) / 'reference')
            runs = {k: strip(time_tree(tree, out, k, candidate=False)) for k in labels}
        for k, v in runs.items():
            (out / f'time_{k}.json').write_text(json.dumps(v, indent=1, default=str) + '\n')
        ok = [v for v in runs.values() if v['status'] == 'ok']
        res = {'status': 'ok' if len(ok) == len(runs) else 'ERROR', 'runs': runs, 'labels': labels}
        if len(ok) == len(runs):
            res['frozen'] = freeze_reference_timing(ok)
        return res
    return finish(out, 'baseline.json', guarded(out, run))


def cmd_freeze_timing(a):
    """Formulator: the frozen reference timing from <out>/baseline.json -> timing.frozen."""
    data = json.loads((Path(a.out).resolve() / 'baseline.json').read_text())
    if data.get('status') != 'ok':
        raise SystemExit(f'baseline.json: {data.get("status")}')
    target = path(CFG['timing']['frozen'])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data['frozen'], indent=1) + '\n')
    print(json.dumps({k: data['frozen'][k] for k in ('s_per_step', 'spread', 'sigma_rel')}))
    return 0


def cmd_record(a):
    """Formulator: one deterministic record run of a tree -> <out>/record.json."""
    out = Path(a.out).resolve()
    with scratch_dir('bg-src-') as tmp:
        tree = materialize(Path(a.tree).resolve(), Path(tmp) / 'tree')
        rec = record_tree(tree, out, a.label, cand_env=candidate_environment(a.config), candidate=False)
    (out / 'record.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({'status': rec.get('status'), 'out': str(out / 'record.json'), 'reason': rec.get('reason'),
                      'cost': cost()}))
    return 0 if rec.get('status') == 'ok' else 1


def cmd_compare(a):
    """Formulator: compare two record.json files (exact items, then bitwise)."""
    res = compare_records(json.loads(Path(a.ref).read_text()), json.loads(Path(a.cand).read_text()))
    print(json.dumps(res, indent=1))
    return 0 if res.get('bitwise') else 1


def cmd_traj_ref(a):
    """Formulator: the reference's trajectories for seeds 1..N -> bench.json trajectory.reference, and delta frozen
    from their spread -> trajectory.frozen; --repeat S runs seed S again, which must reproduce it exactly."""
    tj = CFG['trajectory']
    out = Path(a.out).resolve()
    seeds = list(range(1, a.seeds + 1))
    with scratch_dir('bg-src-') as tmp:
        tree = materialize(check_reference(), Path(tmp) / 'tree')
        runs = {}
        for seed in seeds:
            run = traj_run(tree, out, 'ref', seed, candidate=False)
            if run['status'] != 'ok':
                print(json.dumps({'status': run['status'], 'seed': seed, 'reason': run.get('reason')}))
                return 1
            runs[seed] = run['series']
            log(f'reference trajectory seed {seed}: ce {run["series"]["ce"][0]:.4f} -> {run["series"]["ce"][-1]:.4f}, '
                f'mse {run["series"]["mse"][0]:.4f} -> {run["series"]["mse"][-1]:.4f} ({run["wall_s"]:.0f}s, '
                f'busy CPUs {(run.get("cpu") or {}).get("busy_cpus")})')
        repeat = None
        if a.repeat:
            again = traj_run(tree, out, 'ref_repeat', a.repeat, candidate=False)
            if again['status'] != 'ok':
                print(json.dumps({'status': again['status'], 'seed': a.repeat, 'reason': again.get('reason')}))
                return 1
            repeat = {'seed': a.repeat, 'identical': again['series'] == runs[a.repeat]}
            log(f'repeat of seed {a.repeat}: identical {repeat["identical"]}')
            if not repeat['identical'] and not tj.get('production'):
                print(json.dumps({'status': 'ERROR', 'reason': f'seed {a.repeat} did not reproduce exactly'}))
                return 1
    spec = {'steps': tj['steps'], 'endpoints': tj['endpoints'], 'init_perturbation': '2^-7', 'deterministic': not tj.get('production'),
            'displacement_every': 10}
    frozen = trajectory.freeze(runs, spec, k=tj['k'])
    frozen['repeat'] = repeat
    path(tj['reference']).parent.mkdir(parents=True, exist_ok=True)
    path(tj['reference']).write_text(json.dumps({str(s): v for s, v in runs.items()}) + '\n')
    path(tj['frozen']).write_text(json.dumps(frozen, indent=1) + '\n')
    print(json.dumps({'status': 'ok', 'endpoints': frozen['endpoints'], 'cost': cost()}))
    return 0


def cmd_conform_ref(a):
    """Formulator: the reference's gate-2b run on the held-out data configuration, with each MSE token's pool from
    its own masks -> conformance.reference; --repeat runs it again, which must reproduce it exactly."""
    out = Path(a.out).resolve()
    with scratch_dir('bg-src-') as tmp:
        tree = materialize(check_reference(), Path(tmp) / 'tree')
        seed = CFG['conformance']['final_data_seed'] if a.final_seed else None
        run = conform_run(tree, out, 'ref', candidate=False, pools=True, production=a.production, data_seed=seed)
        if run['status'] != 'ok':
            print(json.dumps({'status': run['status'], 'reason': run.get('reason')}))
            return 1
        repeat = None
        if a.repeat:
            again = conform_run(tree, out, 'ref_repeat', candidate=False, pools=True, production=a.production,
                                data_seed=seed)
            if again['status'] != 'ok':
                print(json.dumps({'status': again['status'], 'reason': again.get('reason')}))
                return 1
            strip_ = lambda x: [{k: v for k, v in r.items()} for r in x['ranks']]
            repeat = {'identical': strip_(again) == strip_(run)}
            if not repeat['identical'] and not a.production:   # production mode: not bitwise reproducible by construction
                print(json.dumps({'status': 'ERROR', 'reason': 'the reference conformance run did not reproduce '
                                  'exactly'}))
                return 1
    frozen = {'steps': run['steps'], 'data_config': CFG['conformance']['data_config'], 'ranks': run['ranks'],
              'repeat': repeat, 'wall_s': run.get('wall_s')}
    stat = conform.statistic([r['forwards'] for r in run['ranks']], [r['forwards'] for r in run['ranks']])
    frozen['tokens'] = stat['tokens']
    missing = [p for p in conform.POOLS if not stat['tokens'][p]]
    if missing:
        print(json.dumps({'status': 'ERROR', 'reason': f'the reference run has no MSE tokens in the pools {missing}: '
                          'the held-out data does not test them (raise conformance.steps)'}))
        return 1
    target = path(CFG['conformance'][('final_' if a.final_seed else '') + ('production_reference' if a.production else 'reference')])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(frozen) + '\n')
    print(json.dumps({'status': 'ok', 'tokens': frozen['tokens'], 'repeat': repeat, 'wall_s': run.get('wall_s'),
                      'cost': cost()}))
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
    k.add_argument('--final', action='store_true', help=argparse.SUPPRESS)   # the final evaluation only (final_allowed)
    k.add_argument('--n1', action='store_true',
                   help='gates 0 and 3 only (the statistical gate), after a check whose gate 2 was not bitwise '
                        'identical: no timing, no records')
    t = sub.add_parser('time')
    t.add_argument('--tree', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--config')
    t.add_argument('--paired', action='store_true', help='also time the reference right before (same host, fresh)')
    f = sub.add_parser('profile')
    f.add_argument('--tree', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--config')
    f.add_argument('--from', dest='start', type=int, default=8)
    f.add_argument('--iters', type=int, default=2)
    b = sub.add_parser('baseline', help='formulator: the reference timed timing.reference_runs times')
    b.add_argument('--out', required=True)
    z = sub.add_parser('freeze-timing', help='formulator: freeze the baseline as the reference timing')
    z.add_argument('--out', required=True)
    r = sub.add_parser('record', help='formulator: one deterministic record run of a tree')
    r.add_argument('--tree', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--label', default='ref')
    r.add_argument('--config')
    c = sub.add_parser('compare', help='formulator: compare two record.json files')
    c.add_argument('ref')
    c.add_argument('cand')
    j = sub.add_parser('traj-ref', help="formulator: the reference's trajectories and the frozen delta")
    j.add_argument('--out', required=True)
    j.add_argument('--seeds', type=int, default=6)
    j.add_argument('--repeat', type=int, default=1, help='run this seed again; it must reproduce exactly (0: none)')
    cr = sub.add_parser('conform-ref', help="formulator: the reference's gate-2b run, frozen")
    cr.add_argument('--production', action='store_true')
    cr.add_argument('--final-seed', action='store_true')
    cr.add_argument('--out', required=True)
    cr.add_argument('--repeat', type=int, default=1, help='run it again; it must reproduce exactly (0: no)')
    fi = sub.add_parser('freeze-inputs', help='formulator: write the frozen input fingerprints')
    fi.add_argument('paths', nargs='+')
    a = p.parse_args()
    return {'freeze-inputs': cmd_freeze_inputs, 'check': cmd_check, 'time': cmd_time, 'profile': cmd_profile,
            'baseline': cmd_baseline, 'freeze-timing': cmd_freeze_timing, 'record': cmd_record,
            'compare': cmd_compare, 'traj-ref': cmd_traj_ref, 'conform-ref': cmd_conform_ref}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
