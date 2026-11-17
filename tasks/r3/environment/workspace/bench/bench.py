"""R3 Tree-RL of BAGEL-7B-MoT on 8 GPUs: one command per question.

Run from the workspace root (/opt/researchboost/ws) with the training environment's Python (bench.json python). GPU
runs take /tmp/r3_gpu.lock themselves, one at a time; do NOT wrap these commands in flock (that would deadlock).

  python bench/bench.py check   --tree <dir> --out <dir> [--config <file>] [--quick] [--skip-timing]
        the verdict (gates 0-2 below) -> <out>/check.json; --quick: gates 0 and 2 only (no timing);
        --skip-timing: no gate 1. --final is the final evaluation's (root or the verifier's marker,
        final_allowed): gate 1 as reference -> candidate -> reference.
  python bench/bench.py time    --tree <dir> --out <dir> [--config <file>] [--paired]
        one timing run of the tree (timing.steps optimizer steps; the windows of bench.json timing.windows) ->
        <out>/time.json: s/step per window, peak memory, and the speed relative to the frozen reference timing of
        this host (task/timing_ref.json). --paired also times the reference right before it, fresh.
  python bench/bench.py profile --tree <dir> --out <dir> [--config <file>] [--from K] [--iters M]
        torch.profiler over optimizer steps K..K+M-1 (default 11, 2) on rank 0 -> <out>/profile.txt (operators by
        CUDA time), a Chrome trace and the step times of the run
  python bench/bench.py baseline --out <dir>          (formulator) the reference timed twice
  formulator only: record, freeze-inputs (see --help of each)

Every run: the frozen recipe (bench/recipe.py) through the judge's entry (bench/r3_entry.py), which imports the tree's
train.online_rl and runs its main() on CFG world_size GPUs (torchrun, one node), against the judge's GenEval reward
service (upstream yifan123/reward-server, started fresh for each run on the same GPUs, eight workers, bench/
reward_app.py), with the judge's frozen copies of the training prompts and of the evaluation fixture (the first eight
GenEval test prompts). A run ends after the optimizer step it measures up to; no further step starts.

Timing: the agent's check times the candidate once against the frozen reference timing of this host
(task/timing_ref.json, the reference runs of the preparation); the final evaluation (check --final) measures reference
-> candidate -> reference fresh, on the host and in the session where it runs.

Interpreters: the tree under test runs with bench.json candidate_python (the agent's environment, which sees the
training environment's packages unless it installs its own versions); reference runs and formulator commands run
with bench.json python (read-only training environment). The reward service always runs with the reward environment.

Every GPU run holds the GPU lock, waits until the GPUs hold no foreign memory, runs a fresh copy of the tree in its
own process group with a timeout, a fresh TMPDIR and fresh compile caches (cold compile), and kills what it started
afterwards. Every command checks the frozen inputs declared in task/formulation.json before (mismatch: nothing is
measured) and after (mismatch: ERROR), and the source digest of the tree copy before and after.

Candidate switches: a JSON file {"environment": {"NAME": "value"}} given with --config; the bench sets those
variables for the candidate's training processes only (never for the reward service). Variables the bench manages
(bench.json reserved_env) are refused.

Verdict (check); each gate stops the evaluation when it fails:
  0. structure  the tree's train/online_rl.py must import and its main() must run the frozen recipe through the
                optimizer steps the gates measure (else ERROR).
  1. benefit    fresh runs, timing.steps optimizer steps each, production settings. For each window of bench.json
                timing.windows (steps 10-11, 12-13, 14-15; each holds a generate and a refine step): seconds per
                step = (end of its last step - end of the step before it) / its number of steps, the slowest rank
                deciding (every rank synchronizes CUDA at each step end). The agent's check: one candidate run; ratio
                = frozen reference / candidate per window (task/timing_ref.json: min(A, B) per window of the
                preparation's two reference runs), reference spread = the frozen one; a frozen spread beyond
                timing.spread_band is reported, not a rejection. The final evaluation (--final): reference A,
                candidate, reference B; ratio = min(A, B) / candidate per window; reference spread = the largest
                |A - B| / min(A, B) over the windows; INVALID when the spread exceeds timing.spread_band (the
                reference itself moved), ERROR when a fingerprint differs from A's. In both, speedup = the MEAN of the
                window ratios; not above the bar (benefit.min_speedup, or the spread doubled, over one: the larger) ->
                UNAVAILABLE (no_benefit), rejected without a precision evaluation.
  2. bitwise    deterministic mode (torch deterministic algorithms, warn only; cuDNN deterministic; flash-attn
                deterministic backward; CUBLAS_WORKSPACE_CONFIG), record.steps optimizer steps (steps 0-13: seven
                generate and seven refine steps, with the step-0 evaluation). Must equal the frozen reference record
                (bench/records/ref.json) EXACTLY: the learning rates and optimizer hyper-parameters of every step, the
                number of optimizer steps, the SHA-256 of the full tensor of every parameter the optimizer holds and of
                every tensor of its state after the last step, and every request the reward service received (the
                SHA-256 of each image, of the prompts' metadata, of the response). Anything else -> INCORRECT. Only
                tier N0 (bitwise identical) is accepted in this task; there is no statistical gate.
"""
import argparse
import collections
import concurrent.futures
import fcntl
import hashlib
import json
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

LOCK = Path(CFG['gpu_lock'])
PY = Path(CFG['python'])
CAND_PY = Path(CFG['candidate_python'])
REWARD_BIN = Path(CFG['reward']['python']).parent
TORCHRUN = PY.parent / 'torchrun'
SCRATCH = Path(CFG.get('scratch', '/tmp'))
TIMEOUT = CFG['timeouts_s']
RESERVED = set(CFG['reserved_env'])
SIGNATURES = [(re.compile(x['regex']), x['reason']) for x in CFG.get('invalid_run_signatures', [])]
COST = {'gpu_seconds': 0.0, 'gpu_runs': 0}
TREE_SKIP = {'.git', '__pycache__', 'output', 'outputs', 'ckpt', 'wandb'}
MARK = 'R3_BENCH_RUN'


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
    """Joint hash of the frozen evaluator: bench/ code, config, data copies and records; task/*.json."""
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
    if free < CFG.get('min_scratch_free_gib', 100):
        raise SystemExit(f'{SCRATCH} has only {free:.1f} GiB free (< {CFG.get("min_scratch_free_gib", 100)}); '
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
            return state, f'GPUs busy for {patience}s before {what}: {busy}'
        time.sleep(15)


def _tagged(tag):
    me, pids = os.getuid(), []
    mark = f'{MARK}={tag}'.encode()
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            if d.stat().st_uid == me and mark in (d / 'environ').read_bytes().split(b'\0'):
                pids.append(int(d.name))
        except OSError:
            continue
    return pids


def _orphans():
    """Processes of this user left by an earlier bench run (they carry the bench's mark): a run killed before its
    cleanup leaves its reward service or training ranks behind. Called with the GPU lock held, when no other bench
    run can be live."""
    me, pids = os.getuid(), []
    mark = f'{MARK}='.encode()
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            if d.stat().st_uid == me and any(e.startswith(mark) for e in (d / 'environ').read_bytes().split(b'\0')):
                pids.append(int(d.name))
        except OSError:
            continue
    return pids


def _kill_orphans():
    found = _orphans()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in _orphans():
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                pass
        if _orphans():
            time.sleep(10)
    if found:
        log(f'stopped {len(found)} processes left by an earlier bench run')
    return len(found)


def _die_with_parent():
    """preexec_fn of the reward service: SIGTERM to it when the bench process that started it dies."""
    import ctypes
    ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


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


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def interpreter_path(candidate, inherited):
    rest = [d for d in inherited.split(':') if d and d not in (str(PY.parent), str(CAND_PY.parent), str(REWARD_BIN))]
    cuda = str(Path(CFG['cuda_home']) / 'bin')
    rest = [d for d in rest if d != cuda]
    return ':'.join(([str(CAND_PY.parent)] if candidate else []) + [str(PY.parent), cuda] + rest)


def under_test(tree):
    return Path(tree).resolve() != (ROOT / CFG['reference']).resolve()


def base_env(tmp, cand_env=None, candidate=False):
    """Environment of every training process: the caller's environment minus reserved variables, then the
    candidate's switches, then what the bench manages (scratch, caches, offline switches, toolkit, PATH)."""
    env = {k: v for k, v in os.environ.items() if k not in RESERVED and not k.startswith('R3_BENCH')}
    env.update(cand_env or {})
    caches = Path(tmp) / 'caches'
    env.update({
        'TMPDIR': str(tmp), 'TMP': str(tmp), 'TEMP': str(tmp),
        'TORCHINDUCTOR_CACHE_DIR': str(caches / 'inductor'), 'TRITON_CACHE_DIR': str(caches / 'triton'),
        'CUDA_CACHE_PATH': str(caches / 'cuda'), 'XDG_CACHE_HOME': str(caches / 'xdg'),
        'MPLCONFIGDIR': str(caches / 'mpl'),
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


def reward_env(tmp, port, log_prefix=None):
    r = CFG['reward']
    env = {k: v for k, v in os.environ.items() if k not in RESERVED and not k.startswith('R3_')}
    env.update({
        'TMPDIR': str(tmp), 'PYTHONNOUSERSITE': '1', 'PYTHONPATH': f'{BENCH}:{r["server_dir"]}',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_CACHE': str(path(r['hf_cache'])),
        'R3_REWARD_DETECTOR_CONFIG': r['detector_config'], 'R3_REWARD_DETECTOR_DIR': str(path(r['detector_dir'])),
        'R3_REWARD_BIND': f'127.0.0.1:{port}', 'R3_REWARD_GPUS': str(CFG['world_size']),
        'PATH': ':'.join([str(REWARD_BIN), '/usr/local/bin', '/usr/bin', '/bin']),
    })
    if log_prefix:
        env['R3_REWARD_LOG'] = str(log_prefix)
    return env


def start_reward(tmp, out_log, log_prefix, tag):
    """The judge's GenEval service, fresh for this run; returns (process, port) once every worker has loaded its
    models, or (None, reason)."""
    port = free_port()
    env = dict(reward_env(tmp, port, log_prefix), **{MARK: tag})
    lf = open(out_log, 'w')
    proc = subprocess.Popen([str(REWARD_BIN / 'gunicorn'), '-c', str(BENCH / 'gunicorn_conf.py'),
                             'reward_app:create_app()'], cwd=CFG['reward']['server_dir'], env=env, stdout=lf,
                            stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=_die_with_parent)
    t0 = time.time()
    while time.time() - t0 < TIMEOUT['reward_start']:
        time.sleep(5)
        if proc.poll() is not None:
            return None, f'reward service exited {proc.returncode}; see {Path(out_log).name}'
        loaded = Path(out_log).read_text(errors='replace').count("Function 'load_models' executed")
        if loaded >= CFG['world_size']:
            return (proc, port), None
    return None, f'reward service not ready within {TIMEOUT["reward_start"]}s; see {Path(out_log).name}'


def stop_reward(handle):
    if handle and handle[0].poll() is None:
        try:
            os.killpg(handle[0].pid, signal.SIGTERM)
            handle[0].wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(handle[0].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_job(tree, mode, steps, out_prefix, tmp, cand_env, candidate, timeout, what, reward_log=None, profile=None):
    """One run under the GPU lock: idle check, the reward service, then torchrun of the judge's entry on the tree;
    everything started is killed afterwards."""
    tag = uuid.uuid4().hex[:12]
    log_file = Path(f'{out_prefix}.log')
    with GpuLock(what):
        _kill_orphans()
        before, problem = wait_idle(what)
        if problem:
            return {'status': 'INVALID', 'reason': problem, 'returncode': None, 'gpus_before': before}
        handle, why = start_reward(tmp, f'{out_prefix}.reward.log', reward_log, tag)
        if handle is None:
            _kill_tagged(tag)
            return {'status': 'ERROR', 'reason': why, 'returncode': None, 'gpus_before': before}
        rc, status, t0 = None, None, time.time()
        try:
            port = handle[1]
            train_args = recipe.argv(path(CFG['model']), Path(tmp) / 'train-log', f'http://127.0.0.1:{port}',
                                     path(CFG['train_data']), path(CFG['val_data']))
            prof = ['--profile-from', str(profile[0]), '--profile-iters', str(profile[1])] if profile else []
            runner = [str(CAND_PY), '-m', 'torch.distributed.run'] if candidate else [str(TORCHRUN)]
            cmd = [*runner, '--nnodes=1', f'--nproc-per-node={CFG["world_size"]}', '--master-addr', '127.0.0.1',
                   '--master-port', str(free_port()), str(BENCH / 'r3_entry.py'), '--mode', mode, '--out',
                   str(out_prefix), '--steps', str(steps), *prof, '--', *train_args]
            env = base_env(tmp, cand_env, candidate=candidate)
            if mode == 'record':
                env.update(CFG['record']['environment'])
            env[MARK] = tag
            if CFG.get('cpu_affinity'):
                cmd = ['taskset', '-c', CFG['cpu_affinity'], *cmd]
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
            stop_reward(handle)
            left = _kill_tagged(tag)
        after, _ = wait_idle(f'{what} (after)', patience=120)
    res = {'status': status or 'ok', 'returncode': rc, 'wall_s': round(wall, 1), 'gpus_before': before,
           'gpus_after': after, 'log': str(log_file)}
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
        # .cache: huggingface_hub's download metadata in a local_dir, not read by any run
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
    bad = sorted(k for k in env if k in RESERVED or k.startswith('R3_'))
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
        r = subprocess.run([str(python), '-c', VERSIONS, 'torch', 'triton', 'flash_attn', 'transformers',
                            'accelerate', 'numpy'], capture_output=True, text=True, timeout=300,
                           env=dict(os.environ, PYTHONNOUSERSITE='1'))
        _VERSIONS[python] = (json.loads(r.stdout.strip().splitlines()[-1]) if not r.returncode else
                             {'error': f'{python} exited {r.returncode}: {r.stderr[-500:]}'})
    return _VERSIONS[python]


FINGERPRINT_KEYS = ('argv', 'gpus', 'compile_cache', 'instrumentation', 'timing_plan', 'reward_service')


def fingerprint_diff(a, b):
    if a is None or b is None:
        return ['missing fingerprint']
    return [k for k in FINGERPRINT_KEYS if a.get(k) != b.get(k)]


# ---------------------------------------------------------------- gate 1: timing

def window_seconds(ranks):
    """Seconds per step of each window, the slowest rank deciding."""
    out = []
    for first, last in CFG['timing']['windows']:
        per_rank = []
        for x in ranks:
            e = x['step_end_s']
            per_rank.append((e[last] - e[first - 1]) / (last - first + 1))
        out.append(max(per_rank))
    return out


def time_tree(tree, out, label, cand_env=None, candidate=True, profile=None):
    steps = CFG['timing']['steps'] if not profile else profile[0] + profile[1]
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f'timing_{label}'
    for f in out.glob(f'timing_{label}.r*.json'):
        f.unlink()
    with scratch_dir('r3-t-') as tmp:
        r = run_job(tree, 'time', steps, prefix, tmp, cand_env, candidate, TIMEOUT['timing'],
                    f'timing {label}' + (' (profile)' if profile else ''), profile=profile)
    fingerprint = {'argv': ['r3_entry.py --mode time', *recipe.argv('<model>', '<log>', '<reward>', '<train>',
                                                                     '<val>')],
                   'candidate_environment': cand_env or {},
                   'gpus': [(g['index'], g['name'], g['driver']) for g in r.get('gpus_before') or []],
                   'compile_cache': 'cold (fresh per run)', 'instrumentation': 'torch.profiler' if profile else 'none',
                   'timing_plan': [steps, CFG['timing']['windows']],
                   'reward_service': 'reward_app.py + gunicorn_conf.py (bench/), fresh per run',
                   'versions': versions(CAND_PY if candidate else PY)}
    run = {'run_status': r['status'], 'wall_s': r.get('wall_s'), 'fingerprint': fingerprint, 'log': r.get('log'),
           **({'warning': r['warning']} if r.get('warning') else {})}
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
           'step_s_rank0': [round(e0[i] - e0[i - 1], 3) for i in range(1, len(e0))],
           'peak_memory_allocated_gib': round(max(x['peak_memory_allocated_bytes'] for x in ranks) / 2**30, 2),
           'peak_memory_reserved_gib': round(max(x['peak_memory_reserved_bytes'] for x in ranks) / 2**30, 2),
           'startup_to_first_step_s': round(max(x['startup_to_first_step_s'] for x in ranks), 1),
           'other_optimizer_steps': sorted({x.get('other_optimizer_steps') for x in ranks}), **run, '_ranks': ranks}
    if not profile:
        w = window_seconds(ranks)
        res.update(window_s_per_step=[round(x, 4) for x in w], s_per_step=round(sum(w) / len(w), 4))
    return res


def strip(run):
    return {k: v for k, v in run.items() if k != '_ranks'} if run else run


FROZEN_TIMING = ROOT / 'task' / 'timing_ref.json'


def frozen_reference_timing():
    """The reference timing frozen by the preparation (window_s_per_step = min(A, B) per window of its two reference
    runs, s_per_step, spread), or None."""
    return json.loads(FROZEN_TIMING.read_text()) if FROZEN_TIMING.exists() else None


def frozen_speed(res):
    ref = frozen_reference_timing()
    if res.get('status') == 'ok' and res.get('window_s_per_step') and ref:
        res['speedup_vs_frozen_reference'] = round(ref['s_per_step'] / res['s_per_step'], 4)
        res['frozen_reference_note'] = ('the reference measured on this host during preparation (mean over the '
                                        'windows); gate 1 of your check uses the mean of the per-window ratios')
    return res


def benefit_frozen(tree, out, cand_env):
    """Gate 1 of the agent's check: one candidate run against the frozen reference timing of the preparation
    (task/timing_ref.json). Per window ratio = frozen / candidate, speedup = the mean; the bar is the final
    evaluation's (min_speedup, or the frozen reference's spread doubled, over one: the larger). The noise rule (a
    reference spread beyond timing.spread_band) is reported, never a rejection. Reference -> candidate -> reference is the final evaluation's
    (check --final)."""
    t = CFG['timing']
    frozen = frozen_reference_timing()
    if not frozen:
        return 'ERROR', {'speedup': None, 'reason': f'no frozen reference timing ({FROZEN_TIMING}): the preparation '
                                                    f'did not time the reference'}
    cand = time_tree(tree, out, 'cand', cand_env=cand_env)
    runs = {'cand': strip(cand)}
    if cand['status'] != 'ok':
        status = 'INVALID' if cand['status'] in ('INVALID', 'TIMEOUT') else 'ERROR'
        return status, {'speedup': None, 'runs': runs, 'reason': cand.get('reason') or 'the timing run failed'}
    F, C = frozen['window_s_per_step'], cand['window_s_per_step']
    ratios = [f / c for f, c in zip(F, C)]
    spread = frozen.get('spread') or 0.0
    perf = {'windows': t['windows'], 'reference_s_per_step_windows': F, 'candidate_s_per_step': C,
            'window_ratios': [round(x, 4) for x in ratios], 'speedup': round(sum(ratios) / len(ratios), 4),
            'reference_spread': spread, 'reference': f'frozen ({FROZEN_TIMING}; {frozen.get("source")})',
            'comparison': 'one candidate run against the frozen reference timing (the final evaluation times '
                          'reference -> candidate -> reference)',
            'runs': runs, 'library_versions': {'candidate': cand['fingerprint']['versions']},
            'host': socket.gethostname(), 'reference_s_per_step': frozen['s_per_step'],
            'candidate_mean_s_per_step': round(sum(C) / len(C), 4)}
    if spread > t['spread_band']:
        perf['notes'] = [f'the frozen reference runs differ by {spread:.1%} in a window (> {t["spread_band"]:.0%}); '
                         f'reported only: in the final evaluation two such reference runs make the result INVALID']
    bar = max(CFG['benefit']['min_speedup'], 1 + 2 * spread)
    perf['benefit'] = {'speedup': perf['speedup'], 'speedup_bar': round(bar, 4),
                       'peak_memory_allocated_gib': cand['peak_memory_allocated_gib'],
                       'passed': perf['speedup'] > bar}
    return ('ok' if perf['benefit']['passed'] else 'UNAVAILABLE'), perf


def benefit(tree, out, cand_env, final=False):
    """Gate 1. The agent's check (not final): benefit_frozen. The final evaluation (final): reference -> candidate ->
    reference, fresh; a fingerprint that differs from reference A's is ERROR, a reference spread beyond
    timing.spread_band INVALID."""
    if not final:
        return benefit_frozen(tree, out, cand_env)
    t = CFG['timing']
    ref_src = check_reference()
    with scratch_dir('r3-ref-') as tmp:
        ref_tree = materialize(ref_src, Path(tmp) / 'reference')
        ref_a = time_tree(ref_tree, out, 'ref_a', candidate=False)
        cand = time_tree(tree, out, 'cand', cand_env=cand_env) if ref_a['status'] == 'ok' else None
        ref_b = time_tree(ref_tree, out, 'ref_b', candidate=False) if cand and cand['status'] == 'ok' else None
    runs = {'ref_a': strip(ref_a), 'cand': strip(cand), 'ref_b': strip(ref_b)}
    bad = {k: v for k, v in runs.items() if v is None or v['status'] != 'ok'}
    if bad:
        status = 'INVALID' if any(v and v['status'] in ('INVALID', 'TIMEOUT') for v in bad.values()) else 'ERROR'
        reason = next((v.get('reason') for v in bad.values() if v and v.get('reason')), 'a timing run failed')
        return status, {'speedup': None, 'runs': runs, 'reason': reason}
    diffs = {k: fingerprint_diff(ref_a.get('fingerprint'), v.get('fingerprint')) for k, v in runs.items() if k != 'ref_a'}
    versions_seen = {'reference': ref_a['fingerprint']['versions'], 'candidate': cand['fingerprint']['versions']}
    versions_seen['differ'] = versions_seen['reference'] != versions_seen['candidate']
    badfp = {k: d for k, d in diffs.items() if d}
    if badfp:
        return 'ERROR', {'speedup': None, 'runs': runs, 'fingerprint_diffs': diffs, 'library_versions': versions_seen,
                         'reason': f'timing runs not comparable: fingerprint differs from ref_a in {badfp}'}
    A, B, C = ref_a['window_s_per_step'], ref_b['window_s_per_step'], cand['window_s_per_step']
    ratios = [min(a, b) / c for a, b, c in zip(A, B, C)]
    spread = max(abs(a - b) / min(a, b) for a, b in zip(A, B))
    perf = {'windows': t['windows'], 'reference_a_s_per_step': A, 'reference_b_s_per_step': B,
            'candidate_s_per_step': C, 'window_ratios': [round(x, 4) for x in ratios],
            'speedup': round(sum(ratios) / len(ratios), 4), 'reference_spread': round(spread, 4), 'runs': runs,
            'fingerprint_diffs': diffs, 'library_versions': versions_seen, 'host': socket.gethostname(),
            'reference_s_per_step': round(sum(min(a, b) for a, b in zip(A, B)) / len(A), 4),
            'candidate_mean_s_per_step': round(sum(C) / len(C), 4)}
    if spread > t['spread_band']:
        perf['problems'] = [f'the two reference runs differ by {spread:.1%} in a window (> {t["spread_band"]:.0%}): '
                            f'the host moved between them; measure again later']
        return 'INVALID', perf
    bar = max(CFG['benefit']['min_speedup'], 1 + 2 * spread)
    perf['benefit'] = {'speedup': perf['speedup'], 'speedup_bar': round(bar, 4),
                       'peak_memory_allocated_gib': cand['peak_memory_allocated_gib'],
                       'reference_peak_memory_allocated_gib': max(ref_a['peak_memory_allocated_gib'],
                                                                  ref_b['peak_memory_allocated_gib']),
                       'passed': perf['speedup'] > bar}
    return ('ok' if perf['benefit']['passed'] else 'UNAVAILABLE'), perf


# ---------------------------------------------------------------- gate 2: bitwise record

def reward_stream(prefix):
    entries = []
    for f in sorted(Path(prefix).parent.glob(Path(prefix).name + '.*.jsonl')):
        for line in f.read_text().splitlines():
            if line.strip():
                entries.append(json.loads(line))
    return entries


def record_tree(tree, out, label, cand_env=None, candidate=True):
    """One deterministic record run -> the record dict (or a failure dict with 'status')."""
    steps = CFG['record']['steps']
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f'record_{label}'
    for f in list(out.glob(f'record_{label}.r*.json')) + list(out.glob(f'record_{label}.rewards.*.jsonl')):
        f.unlink()
    with scratch_dir('r3-r-') as tmp:
        r = run_job(tree, 'record', steps, prefix, tmp, cand_env, candidate, TIMEOUT['record'], f'record {label}',
                    reward_log=out / f'record_{label}.rewards')
    if r['status'] != 'ok':
        return {'status': r['status'], 'reason': r['reason'], 'log': r.get('log')}
    ranks = rank_files(prefix)
    err = rank_errors(ranks, r['returncode'], r['log'], steps)
    if err:
        return {'status': 'ERROR', 'reason': err, 'log': r.get('log')}
    stream = reward_stream(out / f'record_{label}.rewards')
    r0 = ranks[0]
    rec = {'status': 'ok', 'steps': r0['steps'], 'lrs': r0['lrs'], 'hparams': r0['hparams'],
           'n_params': r0['n_params'], 'digests': r0['digests'],
           'other_optimizer_steps': max(x.get('other_optimizer_steps') or 0 for x in ranks),
           'lrs_all_ranks_equal': all(x['lrs'] == r0['lrs'] for x in ranks),
           'rewards': sorted(([e.get('images'), e.get('meta'), e.get('response_sha256'), e.get('error')]
                              for e in stream), key=json.dumps),
           'reward_prompts': sorted({p for e in stream for p in e.get('prompts') or []}),
           'reward_requests': len(stream),
           'deterministic': {'patched': r0.get('deterministic_patched'), 'algorithms': r0.get('deterministic_algorithms')},
           'optimizer': r0.get('optimizer'), 'digest_s': r0.get('digest_s'), 'wall_s': r.get('wall_s'),
           'peak_memory_allocated_gib': round(max(x['peak_memory_allocated_bytes'] for x in ranks) / 2**30, 2),
           'log': r.get('log')}
    return rec


def compare_records(ref, cand):
    """Exact comparison: every listed item must be identical. Returns the numerics section of check.json."""
    problems = []
    if cand['steps'] != ref['steps']:
        problems.append(f'{cand["steps"]} optimizer steps recorded, the reference {ref["steps"]}')
    if cand.get('other_optimizer_steps'):
        problems.append(f'{cand["other_optimizer_steps"]} steps of a second optimizer')
    if not cand.get('lrs_all_ranks_equal', True):
        problems.append('the learning rates differ between ranks')
    if cand['lrs'] != ref['lrs']:
        k = next((i for i, (a, b) in enumerate(zip(cand['lrs'], ref['lrs'])) if a != b), None)
        problems.append(f'learning rates differ from the reference (first at step {k})')
    if cand['hparams'] != ref['hparams']:
        problems.append(f'optimizer hyper-parameters differ: {cand["hparams"]} vs the reference {ref["hparams"]}')
    if cand['n_params'] != ref['n_params']:
        problems.append(f'the optimizer holds {cand["n_params"]} parameters, the reference {ref["n_params"]}')
    key = lambda e: (e[1], tuple(e[2]), e[3], e[4])  # noqa: E731 (kind, shape, dtype, sha256), order-free
    cd, rd = collections.Counter(map(key, cand['digests'])), collections.Counter(map(key, ref['digests']))
    differing = {}
    if cd != rd:
        missing = rd - cd
        for kind, shape, dtype, _ in missing.elements():
            differing[kind] = differing.get(kind, 0) + 1
        aligned = len(cand['digests']) == len(ref['digests'])
        first = None
        if aligned:
            first = next(([r[0], r[1], r[2]] for c, r in zip(cand['digests'], ref['digests']) if key(c) != key(r)),
                         None)
        problems.append(f'final parameter / optimizer-state tensors differ from the reference: {sum(missing.values())} '
                        f'of {len(ref["digests"])} ({differing})' + (f'; first: {first}' if first else ''))
    cr, rr = collections.Counter(map(json.dumps, cand['rewards'])), collections.Counter(map(json.dumps, ref['rewards']))
    if cr != rr or cand['reward_requests'] != ref['reward_requests']:
        new = sorted({p for p in cand['reward_prompts']} - set(ref['reward_prompts']))[:3]
        problems.append(f'the reward service received different requests: {sum((cr - rr).values())} of '
                        f'{cand["reward_requests"]} requests not in the reference ({ref["reward_requests"]} requests)'
                        + (f'; prompts not in the reference: {new}' if new else ''))
    bitwise = not problems
    return {'status': 'PASS' if bitwise else 'INCORRECT', 'bitwise': bitwise,
            'measured_tier': 'N0' if bitwise else None, 'problems': problems,
            'differing_tensor_kinds': differing,
            'compared': {'optimizer_steps': ref['steps'], 'tensors': len(ref['digests']),
                         'reward_requests': ref['reward_requests']}}


def numerics(tree, out, cand_env):
    ref = json.loads(path(CFG['record']['reference']).read_text())
    cand = record_tree(tree, out, 'cand', cand_env=cand_env)
    if cand.get('status') != 'ok':
        return {'status': cand.get('status', 'ERROR'), 'reason': cand.get('reason'), 'log': cand.get('log')}
    res = compare_records(ref, cand)
    res['run'] = {k: cand.get(k) for k in ('wall_s', 'digest_s', 'peak_memory_allocated_gib', 'deterministic',
                                           'optimizer', 'log')}
    return res


# ---------------------------------------------------------------- verdict

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


def coverage(tree):
    if not (Path(tree) / 'train' / 'online_rl.py').is_file():
        return 'ERROR', {'passed': False, 'stage': 'structure', 'note': 'train/online_rl.py is missing'}
    return None


def evaluate(tree, out, skip_timing=False, cand_env=None, quick=False, final=False):
    """Gates 0 -> 1 -> 2 (final: gate 1 as reference -> candidate -> reference). Returns (status, correctness,
    performance)."""
    cand_env = cand_env or {}
    failed = coverage(tree)
    if failed:
        return failed[0], failed[1], None
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
    correctness = {'stage': 'bitwise', 'measured_tier': num.get('measured_tier'), 'numerics': num}
    if num.get('status') not in ('PASS', 'INCORRECT'):
        st = 'INVALID' if num.get('status') in ('INVALID', 'TIMEOUT') else 'ERROR'
        return st, dict(correctness, passed=False, note=num.get('reason') or 'the record run failed'), performance
    if not num['bitwise']:
        return 'INCORRECT', dict(correctness, passed=False, verdict='not_bitwise',
                                 note='not bitwise identical to the reference; this task accepts N0 only'), performance
    stage = 'bitwise (quick: gate 1 not run)' if quick else 'bitwise'
    return 'PASS', dict(correctness, passed=True, verdict='bitwise', stage=stage), performance


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
        with scratch_dir('r3-src-') as tmp:
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
    for k in ('speedup_vs_frozen_reference', 'speedup_vs_paired_reference', 's_per_step'):
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
    cand_env = candidate_environment(a.config)

    def run(tree):
        status, correctness, performance = evaluate(tree, out, skip_timing=a.skip_timing, cand_env=cand_env,
                                                    quick=a.quick, final=a.final)
        return {'status': status, 'tree': str(Path(a.tree).resolve()), 'config': a.config, 'environment': cand_env,
                'quick': a.quick, 'skip_timing': a.skip_timing, 'final': a.final,
                'correctness': correctness, 'performance': performance}
    return finish(out, 'check.json', guarded(out, run, Path(a.tree).resolve()))


def cmd_time(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)

    def run(tree):
        ref = None
        if a.paired:
            with scratch_dir('r3-ref-') as tmp:
                ref = strip(time_tree(materialize(check_reference(), Path(tmp) / 'reference'), out, 'reference',
                                      candidate=False))
        res = frozen_speed(strip(time_tree(tree, out, 'tree', cand_env=cand_env, candidate=under_test(a.tree))))
        if ref is not None:
            res['paired_reference'] = ref
            if res['status'] == 'ok' and ref['status'] == 'ok':
                res['speedup_vs_paired_reference'] = round(min(r / c for r, c in zip(ref['window_s_per_step'],
                                                                                      res['window_s_per_step'])), 4)
                res['note'] = ('screening only: one reference run just before, same host and session; the score is '
                               'the final evaluation\'s full check (reference -> candidate -> reference)')
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
    out = Path(a.out).resolve()

    def run(_):
        ref_src = check_reference()
        with scratch_dir('r3-ref-') as tmp:
            tree = materialize(ref_src, Path(tmp) / 'reference')
            runs = {k: time_tree(tree, out, k, candidate=False) for k in ('ref_a', 'ref_b')}
        ok = [v for v in runs.values() if v['status'] == 'ok']
        res = {'status': 'ok' if len(ok) == len(runs) else 'ERROR', 'runs': {k: strip(v) for k, v in runs.items()}}
        if len(ok) == len(runs):
            A, B = runs['ref_a']['window_s_per_step'], runs['ref_b']['window_s_per_step']
            res['window_s_per_step'] = [round(min(x, y), 4) for x, y in zip(A, B)]
            res['s_per_step'] = round(sum(res['window_s_per_step']) / len(A), 4)
            res['spread'] = round(max(abs(x - y) / min(x, y) for x, y in zip(A, B)), 4)
        return res
    return finish(out, 'baseline.json', guarded(out, run))


def cmd_record(a):
    """Formulator: one deterministic record run of a tree -> <out>/record.json."""
    out = Path(a.out).resolve()
    tree = Path(a.tree).resolve()
    with scratch_dir('r3-src-') as tmp:
        tree = materialize(tree, Path(tmp) / 'tree')
        rec = record_tree(tree, out, a.label, cand_env=candidate_environment(a.config), candidate=False)
    (out / 'record.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({'status': rec.get('status'), 'out': str(out / 'record.json'), 'reason': rec.get('reason'),
                      'cost': cost()}))
    return 0 if rec.get('status') == 'ok' else 1


def cmd_compare(a):
    """Formulator: compare two record.json files exactly."""
    res = compare_records(json.loads(Path(a.ref).read_text()), json.loads(Path(a.cand).read_text()))
    print(json.dumps(res, indent=1))
    return 0 if res['bitwise'] else 1


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
    k.add_argument('--quick', action='store_true', help='gates 0 and 2 only (no timing)')
    k.add_argument('--skip-timing', action='store_true', help='no gate 1')
    k.add_argument('--final', action='store_true', help=argparse.SUPPRESS)   # the final evaluation only (final_allowed)
    t = sub.add_parser('time')
    t.add_argument('--tree', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--config')
    t.add_argument('--paired', action='store_true', help='also time the reference right before (same host, fresh)')
    f = sub.add_parser('profile')
    f.add_argument('--tree', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--config')
    f.add_argument('--from', dest='start', type=int, default=11)
    f.add_argument('--iters', type=int, default=2)
    b = sub.add_parser('baseline')
    b.add_argument('--out', required=True)
    r = sub.add_parser('record', help='formulator: one deterministic record run of a tree')
    r.add_argument('--tree', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--label', default='ref')
    r.add_argument('--config')
    c = sub.add_parser('compare', help='formulator: compare two record.json files exactly')
    c.add_argument('ref')
    c.add_argument('cand')
    fi = sub.add_parser('freeze-inputs', help='formulator: write the frozen input fingerprints')
    fi.add_argument('paths', nargs='+')
    a = p.parse_args()
    return {'freeze-inputs': cmd_freeze_inputs, 'check': cmd_check, 'time': cmd_time, 'profile': cmd_profile,
            'baseline': cmd_baseline, 'record': cmd_record, 'compare': cmd_compare}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
