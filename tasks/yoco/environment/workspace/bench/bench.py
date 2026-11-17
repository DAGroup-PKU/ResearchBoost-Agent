"""YOCO-3B pretraining bench on 2 GPUs: one command per question.

Run from the workspace root (/opt/researchboost/ws) with the training environment's Python (bench.json python). GPU runs take /tmp/yoco_gpu.lock themselves, one at
a time; do NOT wrap these commands in flock (that would deadlock).

  python bench/bench.py check   --tree <dir> --out <dir> [--config <file>] [--quick | --skip-timing | --n1]
        the verdict (gates 0-3 below) -> <out>/check.json; --quick: gates 0 and 2 (exact checks + bitwise) only, no
        timing, no trajectories; --skip-timing: no gate 1; --n1: gates 0 and 3 only, after a check whose gate 2 was
        not bitwise identical. --final is the final evaluation's (root or the verifier's marker, final_allowed):
        gate 1 as reference -> candidate -> reference
  python bench/bench.py time    --tree <dir> --out <dir> [--config <file>] [--paired]
        one timing run of the tree (warmup, then timing.repeats windows of timing.window updates) -> <out>/time.json:
        s/update, tokens/s, peak memory, and the speedup over the frozen reference timing (the mean of the
        preparation's 3 baseline runs of the reference) with that reference's own run-to-run noise; without a frozen
        reference timing it times the reference once first and reads the speedup against that run (cached in the
        scratch directory for the next time on the same GPUs; its noise is then not measured). --paired also times
        the reference right before it, fresh, on the same host, and reports that ratio too (screening numbers, not a
        verdict).
  python bench/bench.py profile --tree <dir> --out <dir> [--config <file>] [--nsys] [--iters N]
        torch.profiler over N updates (default 5) after the warmup: GPU busy fraction, fairseq's phase ranges
        (forward, backward, reduce-grads, multiply-grads, clip-grads, optimizer, OSS step/broadcast), time per model
        block, kernel categories and top kernels -> <out>/profile.json, <out>/profile.md, a Chrome trace;
        --nsys: an Nsight Systems trace of the same span instead
  python bench/bench.py baseline --out <dir>          (formulator) the reference timed three times: noise floor
  formulator only: record, loader, setup, freeze-inputs (see --help of each)

THE SPEEDUP OF THE FINAL VERDICT IS MEASURED FRESH. Identical hosts can differ by several percent in update time with
bitwise-identical losses, and one host can drift between sessions. The final evaluation's benefit gate (--final)
therefore measures reference -> candidate -> reference fresh, on the host and in the session where it runs.
task/timing_ref.json keeps the per-update random-number stream (checked) and the frozen reference timing of this
machine (the mean of the preparation's timing.reference_runs baseline runs and their spread): `time` and the agent's
check (gate 1 without --final: one candidate run) read the speedup against it, and the final evaluation only uses it
to reject a check whose reference runs are more than timing.max_reference_drift off it (INVALID).

Interpreters: the tree under test (--tree of check, time and profile) runs with bench.json candidate_python, the
agent's environment, which sees the training environment's packages unless it installs its own versions; that also
holds for the gate-0 probes (setup fingerprint and loader digest), which import the tree's modules. The reference runs
(also a time or profile run of the reference checkout itself) and the formulator commands (baseline, record, loader,
setup) run with bench.json python, the read-only training environment, without user site-packages. Timing fingerprints
record the library versions of each run's interpreter; a difference between reference and candidate versions is
recorded, not an error.

Every GPU run holds the GPU lock, waits until the visible GPUs hold no foreign memory (more than bench.json
idle_memory_mib per GPU; utilization is not checked), runs the code from a fresh copy
of the tree in its own process group with a timeout, a fresh TMPDIR and fresh compile caches (Triton, Inductor, CUDA;
cold compile), and kills what it started afterwards. Timing runs carry a fingerprint (argv, switches, math mode, GPU
names, cache policy, instrumentation, library versions -- the last recorded, not compared), are scanned for invalid-run signatures, fail if
torch.compile compiled anything inside a timing window, and fail if the per-update random-number stream differs from
the frozen reference. Every command checks the frozen inputs declared in task/formulation.json before (mismatch:
nothing is measured) and after (mismatch: ERROR), and the source digest of the tree copy before and after.

Candidate switches: a JSON file {"environment": {"NAME": "value"}} given with --config; the bench sets those
variables for the candidate's processes only. Variables the bench manages (bench.json reserved_env) are refused.

Verdict (check); each gate stops the evaluation when it fails:
  0. structure  the entry yoco/train.py, the live fairseq_cli.train cli_main() and main(), and train() outside the body
                of its per-update loop `for i, samples in enumerate(progress):` must equal the reference AST (taken
                after importing the tree's modules as the entry does; see bench/yoco_setup.py) -> else ERROR; the
                tree's data loader (task.get_batch_iterator, as the trainer calls it) must yield bitwise the same first
                batches as the reference on every shard (one per rank; task/loader_digest.json) -> else INCORRECT.
  1. benefit    the agent's check: one fresh `torchrun` run of the candidate against the frozen reference timing
                (the preparation's timing.reference_runs reference runs, task/timing_ref.json); speedup = the frozen
                mean update time / the candidate's; reference spread = the frozen runs' (max - min) / mean; the
                noise rules below are reported, never a rejection.
                The final evaluation (--final): reference, candidate, reference: fresh `torchrun` runs of the real
                entry point with the frozen recipe and cold compile caches; fingerprints must agree with the first
                reference run (else ERROR), except for the library versions, which are recorded for both.
                speedup = the mean update time of the two reference runs / the candidate's (each run's total of
                its windows / its updates); reference spread = the references' relative gap. The median update
                times are reported as diagnostics. In the final evaluation only: INVALID (measure again later)
                when a run's window range (max - min) / mean exceeds timing.max_window_spread or a reference run is
                more than timing.max_reference_drift off the frozen reference timing; elsewhere both are reported.
                speedup <= max(1.02, 1 + 2 x reference spread) -> UNAVAILABLE (no_benefit), rejected without a
                precision evaluation.
  2. bitwise    the case(s) in bench/cases.json in deterministic mode, one process per case: updates 0 and 1
                (warmup, LR 0 and 3e-6) and, after the tree's own loop body has run the updates in between,
                updates 60 and 61 (after warmup, LR ~1.5e-4). Exact checks (phase, RNG fingerprints before/after,
                forward call counts, invariants incl. scheduled LR, Adam hyper-parameters and step, master/state
                dtypes, per-rank batch digests, sample sizes, update counter; tensor names and shapes; finiteness;
                cross-rank pattern) must hold, else INCORRECT. Every recorded tensor (loss/, grad/, delta/, optim/,
                buffer/) bitwise identical -> PASS (N0). Tensors larger than numerics.sketch_dim are stored as
                CountSketches with exact norms and hashes.
  3. statistical  for every candidate, bitwise ones too: a paired trajectory (losses only, production settings, the
                recipe from scratch; the seed is the model-initialization seed, the data stream is the recipe's) on
                the first trajectory.seeds (1) of the frozen reference seeds, one process per seed. Each endpoint's
                bias, candidate - reference, against trajectory.sigma_k (3) x the reference's seed-to-seed sigma
                (frozen from 6 reference seeds before any candidate): all within -> PASS (N1), any beyond ->
                INCORRECT. The same in every check, the final evaluation's included.
  Tiers: N0 everything bitwise with the same dtypes and math modes; otherwise N1 (a changed dtype or math mode is
  allowed and reported as precision_changed). Per-tensor tolerance gates (bench/thresholds.json) are diagnostics.
"""
import argparse
import concurrent.futures
import fcntl
import functools
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
PY = Path(CFG['python'])                      # training environment: the judge, reference runs, formulator commands
CAND_PY = Path(CFG['candidate_python'])       # the agent's environment: every run of the tree under test
TORCHRUN = PY.parent / 'torchrun'
SCRATCH = Path(CFG.get('scratch', '/tmp'))
TIMEOUT = CFG['timeouts_s']
RESERVED = set(CFG['reserved_env'])
SIGNATURES = [(re.compile(x['regex']), x['reason']) for x in CFG.get('invalid_run_signatures', [])]
COST = {'gpu_seconds': 0.0, 'gpu_runs': 0}
TREE_SKIP = {'.git', '__pycache__', 'imgs', 'outputs', 'checkpoints'}
MARK = 'YOCO_BENCH_RUN'


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
    """Joint hash of the frozen evaluator: numerics/, bench/ code and config, frozen records' manifests, task/ files."""
    files = sorted(p for p in (ROOT / 'numerics').rglob('*.py'))
    # bench code and config, every frozen record / trajectory manifest (they carry the sha256 of every recorded
    # tensor), the trajectory spec and frozen delta; then task/*.json (formulation inputs, loader digest, RNG stream)
    files += sorted(p for p in BENCH.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                    and p.suffix in ('.py', '.json', '.yaml', '.sh'))
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
            'method': f'GPU lock hold time x {CFG["world_size"]} cards (startup, warmup and failed runs included)'}


def scratch_dir(prefix):
    free = shutil.disk_usage(SCRATCH).free / 2**30
    if free < CFG.get('min_scratch_free_gib', 50):
        raise SystemExit(f'{SCRATCH} has only {free:.1f} GiB free (< {CFG.get("min_scratch_free_gib", 50)}); '
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
    """No foreign memory on the visible GPUs (the host keepalive's ~0.65 GB per GPU is allowed; its utilization is
    ignored, the host pauses it while a GPU job of this task runs)."""
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
    """PATH of a launched process: the candidate environment's bin first for the tree under test, the training
    environment's bin first (and the candidate's removed) for everything else; then the CUDA toolkit."""
    rest = [d for d in inherited.split(':') if d and d not in (str(PY.parent), str(CAND_PY.parent))]
    cuda = str(Path(CFG['cuda_home']) / 'bin')
    rest = [d for d in rest if d != cuda]
    return ':'.join(([str(CAND_PY.parent)] if candidate else []) + [str(PY.parent), cuda] + rest)


def under_test(tree):
    """A tree named on the command line runs as a candidate unless it is the reference checkout itself."""
    return Path(tree).resolve() != (ROOT / CFG['reference']).resolve()


def base_env(tmp, cand_env=None, pythonpath=None, candidate=False):
    """Environment of every bench-launched process: the caller's environment minus reserved variables, then the
    candidate's switches, then what the bench manages (scratch, caches, offline switches, toolkit, PATH)."""
    env = {k: v for k, v in os.environ.items() if k not in RESERVED and not k.startswith('YOCO_BENCH')}
    env.update(cand_env or {})
    caches = Path(tmp) / 'caches'
    env.update({
        'TMPDIR': str(tmp), 'TMP': str(tmp), 'TEMP': str(tmp),
        'TORCHINDUCTOR_CACHE_DIR': str(caches / 'inductor'), 'TRITON_CACHE_DIR': str(caches / 'triton'),
        'CUDA_CACHE_PATH': str(caches / 'cuda'), 'XDG_CACHE_HOME': str(caches / 'xdg'),
        'MPLCONFIGDIR': str(caches / 'mpl'),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
        'WANDB_MODE': 'disabled', 'WANDB_DISABLED': 'true', 'PYTHONNOUSERSITE': '1',
        'TIKTOKEN_CACHE_DIR': str(path(CFG['tiktoken_cache'])),
        'CUDA_HOME': CFG['cuda_home'], 'TRITON_LIBCUDA_PATH': '/usr/lib/x86_64-linux-gnu',
        'PATH': interpreter_path(candidate, env.get('PATH', '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin')),
    })
    env.pop('PYTHONPATH', None)
    if pythonpath:
        env['PYTHONPATH'] = ':'.join(str(p) for p in pythonpath)
    for v in ('inductor', 'triton', 'cuda', 'xdg', 'mpl'):
        (caches / v).mkdir(parents=True, exist_ok=True)
    return env


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
        import threading
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


def cpu_summary(sampler, start=None, end=None):
    return {'busy_cpus': sampler.busy_cpus(start, end), 'allowed_cpus': len(os.sched_getaffinity(0)),
            'method': 'container cgroup CPU time / wall time over the run, sampled every second'}


PHASE_MARK = re.compile(r'\[numerics-phase\] rank=0 case=(\S+) phase=(\S+) t=([0-9.]+)')


def record_phases(log_file):
    """Rank 0's phase marks of a record run (numerics.record.phase_mark): seconds from process start at each
    case/phase."""
    try:
        text = Path(log_file).read_text(errors='replace')
    except OSError:
        return None
    return {f'{m.group(1)}/{m.group(2)}': float(m.group(3)) for m in PHASE_MARK.finditer(text)} or None


def run_gpu(cmd, cwd, env, log_file, what, timeout):
    """One GPU job under the lock: idle check, own process group, timeout, leftovers killed."""
    tag = uuid.uuid4().hex[:12]
    with GpuLock(what), CpuSampler() as cpu:
        before, problem = wait_idle(what)
        if problem:
            return {'status': 'INVALID', 'reason': problem, 'returncode': None, 'gpus_before': before}
        env = dict(env, **{MARK: tag})
        if CFG.get('cpu_affinity'):        # the CPUs this container was given, for every run alike
            cmd = ['taskset', '-c', CFG['cpu_affinity'], *cmd]
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
    res = {'status': status or 'ok', 'returncode': rc, 'wall_s': round(wall, 1), 'gpus_before': before,
           'gpus_after': after, 'cpu': cpu_summary(cpu, t0, t0 + wall), 'cpu_sampler': cpu}
    if status == 'TIMEOUT':
        res['reason'] = f'{what} exceeded {timeout}s; process group killed; see {Path(log_file).name}'
    if left:
        res['warning'] = f'{left} processes of the run survived SIGKILL'
    if any(g['memory_mib'] > CFG.get('idle_memory_mib', 2048) for g in after or []):
        res['warning'] = (res.get('warning', '') + ' GPU memory still in use 60s after the run').strip()
    return res


def signatures(log_file):
    hits, text = [], Path(log_file).read_text(errors='replace') if Path(log_file).exists() else ''
    for pattern, reason in SIGNATURES:
        m = pattern.search(text)
        if m:
            line = text[text.rfind('\n', 0, m.start()) + 1:text.find('\n', m.end())]
            hits.append({'reason': reason, 'line': line.strip()[:300]})
    return hits


def launch(argv, port=None, candidate=False):
    """`torchrun` on one node, CFG['world_size'] processes, 127.0.0.1 and a fresh port (no --standalone: the
    container may have no network, and its hostname may not resolve). The tree under test runs torchrun's module
    with the candidate interpreter, whose workers then use that interpreter too."""
    runner = [str(CAND_PY), '-m', 'torch.distributed.run'] if candidate else [str(TORCHRUN)]
    return [*runner, '--nnodes=1', f'--nproc-per-node={CFG["world_size"]}', '--master-addr', '127.0.0.1',
            '--master-port', str(port or free_port()), *argv]


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
    """Every frozen input file of task/formulation.json, by sha256 (host-independent)."""
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
    """Formulator: sha256 of every file under the given paths."""
    out = {}
    for root in paths:
        root = Path(root)
        files = [root] if root.is_file() else sorted(f for f in root.rglob('*') if f.is_file()
                                                      and '__pycache__' not in f.parts)
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
    bad = sorted(k for k in env if k in RESERVED or k.startswith('YOCO_BENCH'))
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


@functools.lru_cache(maxsize=None)
def versions(python):
    """Library versions as the given interpreter sees them (the run's own interpreter, not the judge's)."""
    r = subprocess.run([str(python), '-c', VERSIONS, 'torch', 'triton', 'flash_attn', 'apex', 'fairseq', 'fairscale',
                        'infinibatch', 'numpy', 'tiktoken'], capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONNOUSERSITE='1'))
    if r.returncode:
        return {'error': f'{python} exited {r.returncode}: {r.stderr[-500:]}'}
    return json.loads(r.stdout.strip().splitlines()[-1])


# 'versions' is recorded in every fingerprint but not compared: the candidate may run other library versions in
# its own environment (performance.library_versions lists both).
FINGERPRINT_KEYS = ('argv', 'math_mode', 'gpus', 'compile_cache', 'instrumentation', 'timing_plan')


def fingerprint_diff(a, b):
    if a is None or b is None:
        return ['missing fingerprint']
    return [k for k in FINGERPRINT_KEYS if a.get(k) != b.get(k)]


# ---------------------------------------------------------------- gate 0: structure

SETUP_PROBE = r'''
import json, sys
script_dir, ws = sys.argv[1], sys.argv[2]
sys.path.append(ws)
from bench import yoco_setup
fp, files = yoco_setup.fingerprint(script_dir)
print("SETUP_FINGERPRINT " + json.dumps({"fingerprint": fp, "files": files}))
'''


def setup_fingerprint(tree, candidate=True):
    tree = Path(tree)
    with scratch_dir('yoco-sp-') as tmp:
        env = base_env(tmp, candidate=candidate)
        r = subprocess.run([str(CAND_PY if candidate else PY), '-c', SETUP_PROBE, str(tree / recipe.ENTRY_DIR), str(ROOT)],
                           cwd=tree / recipe.ENTRY_DIR, env=env, capture_output=True, text=True,
                           timeout=TIMEOUT['probe'])
    for line in r.stdout.splitlines():
        if line.startswith('SETUP_FINGERPRINT '):
            return json.loads(line[len('SETUP_FINGERPRINT '):]), None
    return None, (r.stderr or r.stdout)[-3000:]


def setup_guard(ref, tree):
    want, err = setup_fingerprint(ref, candidate=False)
    if want is None:
        raise SystemExit(f'reference setup probe failed: {err}')
    got, err = setup_fingerprint(tree)
    if got is None:
        return [f'setup probe failed (entry not importable?): {err}'], None
    problems = []
    if got['fingerprint'].get('train.update_loops') != 1:
        problems.append(f'expected exactly one `for i, samples in enumerate(progress):` loop in train(), found '
                        f'{got["fingerprint"].get("train.update_loops")}')
    for name in ('entry', 'cli_main', 'main', 'train'):
        if got['fingerprint'].get(name) != want['fingerprint'].get(name):
            problems.append(f'{name} differs from the reference' +
                            (' outside the per-update loop body' if name == 'train' else ''))
    return problems, got['files']


LOADER_PROBE = r'''
import hashlib, json, os, runpy, sys
script_dir, shard, k, argv = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), json.loads(sys.argv[4])
sys.path.insert(0, script_dir)
os.chdir(script_dir)
runpy.run_path("train.py", run_name="__loader_probe__")
import torch
from fairseq import options, utils
import fairseq.tasks as ftasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
parser = options.get_training_parser()
args = options.parse_args_and_arch(parser, input_args=argv)
cfg = convert_namespace_to_omegaconf(args)
utils.import_user_module(cfg.common)
task = ftasks.setup_task(cfg.task)
task.load_dataset(cfg.dataset.train_subset, epoch=1, combine=True)
itr = task.get_batch_iterator(
    dataset=task.dataset(cfg.dataset.train_subset), max_tokens=cfg.dataset.max_tokens,
    max_sentences=cfg.dataset.batch_size, max_positions=None, ignore_invalid_inputs=True,
    required_batch_size_multiple=cfg.dataset.required_batch_size_multiple, seed=cfg.common.seed,
    num_shards=int(os.environ["YOCO_PROBE_SHARDS"]), shard_id=shard, num_workers=cfg.dataset.num_workers, epoch=1,
    data_buffer_size=cfg.dataset.data_buffer_size, disable_iterator_cache=True)
def dig(x, h):
    if isinstance(x, dict):
        for key in sorted(x, key=str):
            h.update(str(key).encode()); dig(x[key], h)
    elif isinstance(x, (list, tuple)):
        h.update(b"["); [dig(v, h) for v in x]; h.update(b"]")
    elif isinstance(x, torch.Tensor):
        t = x.contiguous(); h.update(f"{t.dtype}{tuple(t.shape)}".encode()); h.update(t.view(-1).view(torch.uint8).numpy().tobytes())
    else:
        h.update(repr(x).encode())
it = iter(itr.next_epoch_itr(shuffle=True))
out = []
for _ in range(k):
    h = hashlib.sha256(); dig(next(it), h); out.append(h.hexdigest()[:32])
print("LOADER_DIGEST " + json.dumps(out), flush=True)
# no itr.close(): joining the infinibatch prefetch process can hang forever when it was forked while torch
# inductor's compile-worker pool existed; the bench kills this process group instead
os._exit(0)
'''


def loader_digest(tree, candidate=True):
    """First `loader.batches` batches of every shard (rank) from the tree's task.get_batch_iterator, as the trainer
    calls it (one CPU process per shard, in parallel)."""
    tree = Path(tree)
    ld = CFG['loader']
    with scratch_dir('yoco-ld-') as tmp:
        env = base_env(tmp, candidate=candidate)
        env['YOCO_PROBE_SHARDS'] = str(ld['shards'])
        argv = recipe.argv(Path(tmp) / 'save', 1000)
        procs = []
        for shard in range(ld['shards']):
            lf = open(Path(tmp) / f'probe{shard}.log', 'w+')
            p = subprocess.Popen([str(CAND_PY if candidate else PY), '-c', LOADER_PROBE, str(tree / recipe.ENTRY_DIR), str(shard),
                                  str(ld['batches']), json.dumps(argv)], cwd=tree / recipe.ENTRY_DIR, env=env,
                                 stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            procs.append((p, lf))
        out, err = {}, None
        for shard, (p, lf) in enumerate(procs):
            try:
                p.wait(timeout=TIMEOUT['probe'])
            except subprocess.TimeoutExpired:
                err = f'loader probe shard {shard} timed out'
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            lf.seek(0)
            text = lf.read()
            lf.close()
            line = next((x for x in text.splitlines() if x.startswith('LOADER_DIGEST ')), None)
            if line is None:
                err = err or f'shard {shard}: {text[-2000:]}'
            else:
                out[str(shard)] = json.loads(line[len('LOADER_DIGEST '):])
    return (out, None) if err is None else (None, err)


def coverage(tree):
    """Gate 0 (CPU): setup guard and loader digests. Returns (status, correctness) or None when it holds."""
    ref_src = check_reference()
    problems, files = setup_guard(ref_src, tree)
    if problems:
        return 'ERROR', {'passed': False, 'stage': 'setup_guard', 'problems': problems, 'files': files,
                         'note': 'the numerics adapter re-creates this setup and runs only the per-update loop body of '
                                 'the tree; move the change into code the loop body calls (Trainer.train_step, the '
                                 'task, criterion, model, optimizer) or into the loop body itself'}
    frozen = json.loads(path(CFG['loader']['frozen']).read_text())
    got, err = loader_digest(tree)
    if got != frozen:
        return 'INCORRECT', {'passed': False, 'stage': 'loader_check', 'error': err,
                             'differing_shards': sorted(k for k in frozen if (got or {}).get(k) != frozen[k]),
                             'note': 'the data loader yields different first batches than the reference'}
    return None


# ---------------------------------------------------------------- gate 2: bitwise records

def adapter_kwargs(tree, scratch, variant=None):
    kw = {'repo': str(tree), 'scratch': str(scratch)}
    if variant:
        kw['variant'] = variant
    return kw


def record_cases(tree, rec, cases, logs_dir, cand_env, label, variant=None, losses_only=False, sketch=True,
                 candidate=True, run_log=None, deterministic=True):
    """One `numerics record` process (all ranks) per case. Returns None on success, else an error dict. run_log, a
    list: each case's wall time, busy CPUs and record phases are appended to it."""
    n = CFG['numerics']
    runs = []
    for case in cases:
        with scratch_dir('yoco-r-') as tmp:
            cases_file = Path(tmp) / 'case.json'
            cases_file.write_text(json.dumps([case]))
            env = base_env(tmp, cand_env, pythonpath=[ROOT], candidate=candidate)
            cmd = launch(['-m', 'numerics', 'record', '--adapter', n['adapter'], '--adapter-kwargs',
                          json.dumps(adapter_kwargs(tree, Path(tmp) / 'train', variant)), '--cases', str(cases_file),
                          *(['--losses-only'] if losses_only else
                            (['--sketch-dim', str(n['sketch_dim'])] if sketch else [])),
                          # no --dtype-trace: under its dispatch mode compiled regions fall back to eager
                          '--math-mode', CFG['math_mode'],
                          # gate 2 deterministic; gate 3 (trajectories) the production settings that are timed
                          *(['--deterministic'] if deterministic else []), '--out', str(rec)],
                         candidate=candidate)
            log_file = Path(logs_dir) / f'{label}_{case["name"]}.log'
            r = run_gpu(cmd, ROOT, env, log_file, f'{label}: case {case["name"]}',
                        TIMEOUT['trajectory' if losses_only else 'numerics'])
        runs.append({'case': case['name'], 'wall_s': r.get('wall_s'), 'cpu': r.get('cpu'),
                     'phases_s': None if losses_only else record_phases(log_file)})
        if r['status'] != 'ok' or r['returncode']:
            return {'status': 'INVALID' if r['status'] == 'INVALID' else 'ERROR', 'runs': runs,
                    'reason': r.get('reason') or f'record exited {r["returncode"]}; see {log_file.name}'}
    if run_log is not None:
        run_log.extend(runs)
    return None


def missing_tensor_failures(report):
    """A step whose tensor set lacks reference tensors and adds none of its own did not do that work (e.g. no
    gradient / no optimizer state for parameters the reference updates): an exact failure, not a layout error."""
    out = []
    for err in report.get('errors') or []:
        if isinstance(err, dict) and err.get('missing') and not err.get('unexpected'):
            out.append({'where': err['where'], 'check': 'missing_tensors',
                        'note': 'the candidate produced none of these tensors: it did not do that work',
                        'missing': err['missing'][:20]})
    return out


def diagnostic_only(data):
    """Per-tensor tolerances are diagnostic only (gate 2 in the task statement): a comparison beyond them without an
    exact failure is not bitwise identical and gate 3 decides, so it reads NOT_BITWISE, not INCORRECT. Returns
    whether the report changed."""
    if data.get('status') != 'INCORRECT' or data.get('exact_failures') or data.get('errors'):
        return False
    data['status'] = 'NOT_BITWISE'
    data['note'] = ('beyond the per-tensor tolerances, which are diagnostic only (%d unresolved); no exact failure: '
                    'gate 3 decides' % len(data.get('unresolved') or []))
    return True


def numerics(tree, out, cand_env):
    n = CFG['numerics']
    rec = out / 'records'
    shutil.rmtree(rec, ignore_errors=True)
    cases = json.loads(path(n['cases']).read_text())
    record_runs = []
    failed = record_cases(tree, rec, cases, out, cand_env, 'record', run_log=record_runs)
    if failed:
        return failed
    report = out / 'compare.json'
    env = base_env(out, pythonpath=[ROOT])
    compare_start = time.time()
    r = subprocess.run([str(PY), '-m', 'numerics', 'compare', '--ref', str(path(n['ref'])), '--cand', str(rec),
                        '--thresholds', str(path(n['thresholds'])),
                        *[a for x in n.get('exclude', []) for a in ('--exclude', f'{x["regex"]}={x["reason"]}')],
                        '--out', str(report)], cwd=ROOT, env=env, capture_output=True, text=True)
    (out / 'compare.log').write_text(r.stdout + r.stderr)
    compare_s = round(time.time() - compare_start, 1)
    shutil.rmtree(out / 'caches', ignore_errors=True)
    if not report.exists():
        return {'status': 'ERROR', 'reason': f'compare exited {r.returncode}; see compare.log'}
    data = json.loads(report.read_text())
    if not n.get('keep_records'):
        shutil.rmtree(rec, ignore_errors=True)
    extra = missing_tensor_failures(data)
    data['exact_failures'] = (data.get('exact_failures') or []) + extra
    if extra:
        data['errors'] = [e for e in data.get('errors') or []
                          if not (isinstance(e, dict) and e.get('missing') and not e.get('unexpected'))]
    if diagnostic_only(data):
        report.write_text(json.dumps(data, indent=1) + '\n')
    summary = {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in data.items()
               if k not in ('status', 'measured_tier')}
    bitwise = (data.get('checked') == data.get('bitwise_identical') and not data.get('dtype_changes')
               and not data.get('precision_mode_changes') and data.get('bitwise_trusted', False))
    return {'status': data.get('status'), 'measured_tier': data.get('measured_tier'),
            'precision_changed': bool(data.get('dtype_changes') or data.get('precision_mode_changes')),
            'report': str(report),
            'errors': len(data.get('errors') or []), 'exact_failures': data.get('exact_failures') or [],
            'bitwise': bitwise, 'summary': summary,
            'cost': {'record_runs': record_runs, 'compare_s': compare_s}}


# ---------------------------------------------------------------- gate 3: trajectories

def traj_case(spec, seed):
    return dict(spec['case'], name=f'traj_s{seed}', seed=seed, trajectory=True)


def trajectories(tree, out, cand_env):
    """Gate 3, the same in every check (the agent's, `check --n1`, the final evaluation's): the candidate's trajectory
    on the first trajectory.seeds (1) of the frozen reference seeds, each endpoint's bias against trajectory.sigma_k
    (3) x the reference's seed sigma (numerics traj-compare --sigma-k). Every seed is its own process."""
    tj = CFG['trajectory']
    spec = json.loads(path(tj['spec']).read_text())
    frozen_path = path(tj['frozen'])
    seeds = json.loads(frozen_path.read_text())['reference_seeds'][:tj['seeds']]
    rec, runs = out / 'trajectory', []
    failed = record_cases(tree, rec, [traj_case(spec, s) for s in seeds], out, cand_env, 'traj', losses_only=True,
                          run_log=runs, deterministic=False)
    if failed:
        return {'verdict': 'error', **failed}
    report_path = out / 'traj_report.json'
    env = base_env(out, pythonpath=[ROOT])
    subprocess.run([str(PY), '-m', 'numerics', 'traj-compare', '--frozen', str(frozen_path), '--cand', str(rec),
                    '--sigma-k', str(tj['sigma_k']), '--out', str(report_path)],
                   cwd=ROOT, env=env, capture_output=True, text=True)
    shutil.rmtree(out / 'caches', ignore_errors=True)
    if not report_path.exists():
        return {'verdict': 'error', 'reason': 'traj-compare produced no report'}
    report = json.loads(report_path.read_text())
    report.update(report=str(report_path), runs=runs)
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


# ---------------------------------------------------------------- gate 1: timing

def timing_plan():
    t = CFG['timing']
    return t['warmup'], t['window'], t['repeats']


def time_tree(tree, out, label, cand_env=None, profile=None, profile_iters=5, candidate=True, noise_gate=False):
    """One timing run. noise_gate (the final evaluation's reference -> candidate -> reference only): a run whose window
    range exceeds timing.max_window_spread is INVALID; elsewhere it is reported and the run stands."""
    warmup, window, repeats = timing_plan()
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f'timing_{label}'
    for f in out.glob(f'timing_{label}.r*.json'):
        f.unlink()
    last = warmup + (profile_iters if profile else window * repeats)
    with scratch_dir('yoco-t-') as tmp:
        train_args = recipe.argv(Path(tmp) / 'save', last + 20)
        hook_args = ['--warmup', str(warmup), '--window', str(window), '--repeats', str(repeats),
                     *(['--profile', profile, '--profile-iters', str(profile_iters)] if profile else [])]
        argv = launch([str(BENCH / 'timing.py'), '--out', str(prefix), *hook_args, '--', *train_args],
                      candidate=candidate)
        stable_argv = [f'torchrun --nproc-per-node={CFG["world_size"]}', 'timing.py', *hook_args, '--',
                       *[x.replace(str(tmp), '<scratch>') for x in train_args]]
        env = base_env(tmp, cand_env, candidate=candidate)
        if profile == 'nsys':
            nsys = shutil.which('nsys', path=env['PATH'])
            if not nsys:
                raise SystemExit('nsys not found')
            argv = [nsys, 'profile', '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                    '--trace=cuda,nvtx,osrt,cublas', '--sample=none', '--cuda-memory-usage=false',
                    '--force-overwrite=true', '-o', str(out / f'nsys_{label}'), *argv]
        log_file = out / f'timing_{label}.log'
        what = f'timing {label}{f" ({profile})" if profile else ""}'
        r = run_gpu(argv, Path(tree) / recipe.ENTRY_DIR, env, log_file, what,
                    TIMEOUT['profile' if profile else 'timing'])
    plan = [warmup, window, repeats] if not profile else [warmup, profile_iters, 1]
    fingerprint = {'argv': stable_argv, 'candidate_environment': cand_env or {},
                   'math_mode': CFG['math_mode'],
                   'gpus': [(g['index'], g['name'], g['driver']) for g in r.get('gpus_before') or []],
                   'compile_cache': 'cold (fresh per run)', 'instrumentation': profile or 'none',
                   'timing_plan': plan, 'versions': versions(CAND_PY if candidate else PY)}
    run = {'run_status': r['status'], 'wall_s': r.get('wall_s'), 'fingerprint': fingerprint, 'log': str(log_file),
           'cpu': r.get('cpu'), **({'warning': r['warning']} if r.get('warning') else {})}
    if r['status'] != 'ok':
        return {'status': r['status'], 'reason': r['reason'], **run}
    ranks = []
    for rank in range(CFG['world_size']):
        f = Path(f'{prefix}.r{rank}.json')
        ranks.append(json.loads(f.read_text()) if f.exists() else None)
    if r['returncode'] or any(x is None or x['status'] != 'ok' for x in ranks):
        errors = [x['error'] for x in ranks if x and x.get('error')]
        return {'status': 'ERROR', 'reason': f'timing exited {r["returncode"]}; {errors[:1]}; see {log_file.name}',
                **run}
    hits = signatures(log_file)
    if hits:
        return {'status': 'INVALID', 'reason': 'invalid-run signature in the log: ' + hits[0]['reason'],
                'signatures': hits, **run}
    windows = [w for x in ranks for w in x.get('windows') or [] if 'wall_start' in w]
    if windows and r.get('cpu_sampler') and run['cpu']:
        run['cpu']['busy_cpus_in_windows'] = r['cpu_sampler'].busy_cpus(min(w['wall_start'] for w in windows),
                                                                         max(w['wall_end'] for w in windows))
    run['cpu_binding'] = [x.get('cpu_binding') for x in ranks]
    return {**summarize_timing(ranks, r['wall_s'], profile, profile_iters, noise_gate=noise_gate), **run,
            '_ranks': ranks}


def window_shape(secs):
    m = statistics.fmean(secs)
    return [round(s / m, 4) for s in secs]


def summarize_timing(ranks, wall, profile=None, profile_iters=5, noise_gate=False):
    warmup, window, repeats = timing_plan()
    repeats = 1 if profile else repeats
    size = profile_iters if profile else window
    per_window = {}
    for x in ranks:
        for w in x['windows']:
            prev = per_window.get(w['k'])
            if prev is None or w['seconds'] > prev['seconds']:
                per_window[w['k']] = w
    problems = []
    ws = [per_window[k] for k in range(repeats) if k in per_window]
    if len(ws) != repeats:
        problems.append(f'{len(ws)} of {repeats} windows')
    for w in ws:
        if w.get('compiles_in_window'):
            problems.append(f'window {w["k"]}: torch.compile compiled {w["compiles_in_window"]} graphs inside the '
                            f'timing window (still compiling in steady state)')
    secs = [w['seconds'] for w in ws]
    total = sum(secs) if secs else None
    first, last = warmup, warmup + size * repeats
    medians, slow = [], None
    for x in ranks:
        e = x['iteration_end_s']
        d = [e[i] - e[i - 1] for i in range(first, min(last, len(e)))]
        if d:
            medians.append(statistics.median(d))
            if x['rank'] == 0:
                med0 = statistics.median(d)
                slow = round(sum(v > 1.2 * med0 for v in d) / len(d), 4)
    frozen = None
    ref_file = ROOT / 'task' / 'timing_ref.json'
    if not profile and ref_file.exists():
        frozen = json.loads(ref_file.read_text())
        n_check = warmup + window * repeats
        for x in ranks:
            want = frozen['rng_fingerprints'][str(x['rank'])][:n_check]
            if x['rng_fingerprints'][:n_check] != want:
                k = next((i for i, (a, b) in enumerate(zip(x['rng_fingerprints'], want)) if a != b), None)
                problems.append(f'rank {x["rank"]}: the random-number stream differs from the reference from update '
                                f'{k} on (random numbers drawn in a different amount or order)')
                break
    other = sorted({x.get('other_optimizer_steps') for x in ranks})
    res = {
        'status': 'ok' if not problems and secs else 'ERROR', 'problems': problems,
        'window_s': [round(s, 3) for s in secs], 'total_s': round(total, 3) if total else None,
        'iterations_timed': size * repeats,
        's_per_update': round(total / (size * len(secs)), 4) if secs else None,
        'median_update_s': round(max(medians), 5) if medians else None,
        'slow_update_fraction_rank0': slow,
        'window_range': round((max(secs) - min(secs)) / statistics.fmean(secs), 4) if secs else None,
        'tokens_per_s': round(size * len(secs) * CFG['world_size'] * 2048 / total, 1) if secs else None,
        'peak_memory_allocated_gib': round(max(x['peak_memory_allocated_bytes'] for x in ranks) / 2**30, 2),
        'peak_memory_reserved_gib': round(max(x['peak_memory_reserved_bytes'] for x in ranks) / 2**30, 2),
        'startup_to_first_update_s': round(max(x['startup_to_first_iteration_s'] or 0 for x in ranks), 1),
        'other_optimizer_steps': other,
        'process_wall_s': round(wall, 1),
    }
    if len(secs) > 1:
        res['window_shape'] = window_shape(secs)
    spread = CFG['timing']['max_window_spread']
    if res['status'] == 'ok' and not profile and res['window_range'] > spread:
        noisy = f'window range {res["window_range"]} (windows {res["window_s"]}) > {spread}: interference during the run'
        if noise_gate:                  # the final evaluation's A -> candidate -> B only
            res['status'] = 'INVALID'
            res['problems'].append(noisy + '; rerun later')
        else:
            res['notes'] = [noisy + ' (reported; only the final evaluation rejects a noisy run)']
    ref = frozen_reference_timing()
    if ref and res.get('s_per_update') and not profile:
        res['speedup_vs_frozen_reference'] = round(ref['s_per_update'] / res['s_per_update'], 4)
        res.update(reference_noise(ref, res['speedup_vs_frozen_reference']))
    return res


# `time` without the preparation's frozen reference timing: one run of the reference stands in for it, kept here for
# the next `time` on the same GPUs (ensure_reference_timing).
REFERENCE_ONCE = SCRATCH / 'yoco_reference_timing_once.json'
_reference_once = None


def prepared_reference_timing():
    """This machine's frozen reference timing (task/timing_ref.json 'frozen', written at preparation from the
    baseline runs), or None."""
    f = ROOT / 'task' / 'timing_ref.json'
    return json.loads(f.read_text()).get('frozen') if f.exists() else None


def frozen_reference_timing():
    """The preparation's frozen reference timing, else the one run of the reference `time` made (or None)."""
    return prepared_reference_timing() or _reference_once


def reference_once_key():
    """What a cached one-run reference timing must match: these GPUs, the timing plan and the reference."""
    q = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], capture_output=True, text=True)
    return {'gpus': sorted(q.stdout.split()), 'world_size': CFG['world_size'], 'timing_plan': list(timing_plan()),
            'reference_commit': CFG.get('reference_commit')}


def ensure_reference_timing(out, tree):
    """`time` reads its speedup against the frozen reference timing of the preparation. Without one (this machine
    was not prepared, or the file has no frozen section), the reference is timed once here and that run stands in for
    it, cached in REFERENCE_ONCE for the next `time` on the same GPUs, timing plan and reference; its run-to-run
    noise is not measured. Nothing to do when the frozen timing exists or when the tree timed is the reference
    itself. Returns (how the reference timing was obtained, the reference run timed now or None)."""
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
    with scratch_dir('yoco-ref-') as tmp:
        res = strip(time_tree(materialize(check_reference(), Path(tmp) / 'reference'), out, 'reference_once',
                              candidate=False))
    if res['status'] != 'ok':
        log(f'the reference run failed ({res["status"]}: {res.get("problems") or res.get("reason")}): no speedup')
        return 'one run of the reference, failed', res
    _reference_once = {'s_per_update': res['s_per_update'], 'window_s': res['window_s'], 'spread': None,
                       'sigma_rel': None, 'runs': [{'s_per_update': res['s_per_update'], 'window_s': res['window_s']}],
                       'key': key, 'source': 'one run of the reference by time: no frozen reference timing was found'}
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


def strip(run):
    return {k: v for k, v in run.items() if k != '_ranks'} if run else run


def benefit_frozen(tree, out, cand_env):
    """Gate 1 of the agent's check: one candidate run against the frozen reference timing of the preparation (the mean
    of its timing.reference_runs reference runs, each warmup + window x repeats; task/timing_ref.json). The bar is the
    final evaluation's (min_speedup, or the frozen reference's spread doubled, over one: the larger); the noise rules are reported, never a
    rejection. The reference -> candidate -> reference comparison is the final evaluation's (check --final)."""
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
                        'reason': f'timing run: {cand["status"]} {cand.get("problems") or cand.get("reason")}'}
    spread = frozen.get('spread') or 0.0
    speedup = cand['speedup_vs_frozen_reference']
    ref_file = ROOT / 'task' / 'timing_ref.json'
    info = (json.loads(ref_file.read_text()).get('informational') or {}) if ref_file.exists() else {}
    perf = {'reference_s_per_update': frozen['s_per_update'], 'candidate_s_per_update': cand['s_per_update'],
            'speedup': speedup, 'reference_spread': spread, 'reference': source or 'frozen',
            'comparison': 'candidate run against the frozen reference timing (the final evaluation times '
                          'reference -> candidate -> reference)',
            'runs': runs, 'library_versions': {'candidate': cand['fingerprint']['versions']},
            'host': socket.gethostname(), **({'notes': cand['notes']} if cand.get('notes') else {})}
    bar = max(CFG['benefit']['min_speedup'], 1 + 2 * spread)
    perf['benefit'] = {'speedup': speedup, 'speedup_bar': round(bar, 4),
                       'peak_memory_allocated_gib': cand['peak_memory_allocated_gib'],
                       'reference_peak_memory_allocated_gib': info.get('peak_memory_allocated_gib'),
                       'passed': speedup > bar}
    return ('ok' if perf['benefit']['passed'] else 'UNAVAILABLE'), perf


def benefit(tree, out, cand_env, final=False):
    """Gate 1. The agent's check (not final): benefit_frozen. The final evaluation (final): reference -> candidate ->
    reference, measured now on this host; the noise rules (a run's window range, a reference off the frozen timing)
    make the result INVALID there."""
    if not final:
        return benefit_frozen(tree, out, cand_env)
    t = CFG['timing']
    ref_src = check_reference()
    with scratch_dir('yoco-ref-') as tmp:
        ref_tree = materialize(ref_src, Path(tmp) / 'reference')
        ref_a = time_tree(ref_tree, out, 'ref_a', candidate=False, noise_gate=final)
        cand = time_tree(tree, out, 'cand', cand_env=cand_env, noise_gate=final) if ref_a['status'] == 'ok' else None
        ref_b = (time_tree(ref_tree, out, 'ref_b', candidate=False, noise_gate=final)
                 if cand and cand['status'] == 'ok' else None)
    runs = {'ref_a': strip(ref_a), 'cand': strip(cand), 'ref_b': strip(ref_b)}
    bad = {k: v for k, v in runs.items() if v is None or v['status'] != 'ok'}
    if bad:
        status = 'INVALID' if any(v and v['status'] in ('INVALID', 'TIMEOUT') for v in bad.values()) else 'ERROR'
        return status, {'speedup': None, 'runs': runs}
    diffs = {k: fingerprint_diff(ref_a.get('fingerprint'), v.get('fingerprint')) for k, v in runs.items() if k != 'ref_a'}
    versions_seen = {'reference': ref_a['fingerprint']['versions'], 'candidate': cand['fingerprint']['versions']}
    versions_seen['differ'] = versions_seen['reference'] != versions_seen['candidate']
    badfp = {k: d for k, d in diffs.items() if d}
    if badfp:
        return 'ERROR', {'speedup': None, 'runs': runs, 'fingerprint_diffs': diffs, 'library_versions': versions_seen,
                         'reason': f'timing runs not comparable: fingerprint differs from ref_a in {badfp}'}
    ra, rb, c = ref_a['s_per_update'], ref_b['s_per_update'], cand['s_per_update']
    ma, mb, cm = ref_a['median_update_s'], ref_b['median_update_s'], cand['median_update_s']
    ref, refm = (ra + rb) / 2, (ma + mb) / 2
    perf = {'reference_s_per_update': round(ref, 4), 'candidate_s_per_update': c,
            'speedup': round(ref / c, 4), 'reference_spread': round(abs(ra - rb) / ref, 4),
            'diagnostics': {'reference_median_update_s': round(refm, 5), 'candidate_median_update_s': cm,
                            'speedup_by_median': round(refm / cm, 4), 'reference_gap_median': round(abs(ma - mb) / refm, 4)},
            'runs': runs, 'fingerprint_diffs': diffs, 'library_versions': versions_seen, 'host': socket.gethostname()}
    frozen, notes = frozen_reference_timing(), []
    for k, v in (('ref_a', ra), ('ref_b', rb)):
        dev = abs(v / frozen['s_per_update'] - 1) if frozen else 0.0
        if dev > t['max_reference_drift']:
            off = (f'{k} is {dev:.1%} off the frozen reference timing (> {t["max_reference_drift"]:.0%}): the machine was '
                   f'not in its usual state')
            if final:
                perf['problems'] = [off + '; rerun later']
                return 'INVALID', perf
            notes.append(off + ' (reported; only the final evaluation rejects it)')
    if notes:
        perf['notes'] = notes
    bar = max(CFG['benefit']['min_speedup'], 1 + 2 * perf['reference_spread'])
    perf['benefit'] = {'speedup': perf['speedup'], 'speedup_bar': round(bar, 4),
                       'peak_memory_allocated_gib': cand['peak_memory_allocated_gib'],
                       'reference_peak_memory_allocated_gib': round((ref_a['peak_memory_allocated_gib'] +
                                                                     ref_b['peak_memory_allocated_gib']) / 2, 2),
                       'passed': perf['speedup'] > bar}
    return ('ok' if perf['benefit']['passed'] else 'UNAVAILABLE'), perf


# ---------------------------------------------------------------- verdict

def statistical(tree, out, cand_env, correctness, performance):
    """Gate 3 and its verdict (of every check and of check --n1)."""
    traj = trajectories(tree, out, cand_env)
    correctness = dict(correctness, stage='statistical', trajectory=traj)
    if traj['verdict'] == 'equivalent':
        return 'PASS', dict(correctness, passed=True, verdict='equivalent'), performance
    if traj['verdict'] == 'error':
        return ('INVALID' if traj.get('status') == 'INVALID' else 'ERROR'), \
            dict(correctness, passed=False, verdict='error', note=traj.get('reason') or traj.get('error')), performance
    return 'INCORRECT', dict(correctness, passed=False, verdict='different',
                             note=f'an endpoint\'s bias on seed {traj["seeds"]} is beyond '
                                  f'{CFG["trajectory"]["sigma_k"]} x the reference\'s seed sigma'), performance


def evaluate(tree, out, skip_timing=False, cand_env=None, quick=False, final=True, n1=False):
    """Gates 0 -> 1 -> 2 -> 3 (n1: gates 0 and 3 only). Returns (status, correctness, performance)."""
    cand_env = cand_env or {}
    failed = coverage(tree)
    if failed:
        return failed[0], failed[1], None
    if n1:
        # check --n1: after a check whose gate 2 was not bitwise identical, gate 3 alone (no timing, no records).
        return statistical(tree, out, cand_env, {'gates_run': [0, 3], 'note': 'check --n1: gate 3 only; gates 1 and '
                                                 '2 not run (the final evaluation runs every gate)'}, None)
    if quick:
        num = numerics(tree, out, cand_env)
        if num.get('exact_failures'):
            return 'INCORRECT', {'passed': False, 'stage': 'structure', 'verdict': 'structure', 'numerics': num}, None
        if num.get('errors') or num.get('status') not in ('PASS', 'NOT_BITWISE', 'INCORRECT'):
            st = 'INVALID' if num.get('status') == 'INVALID' else 'ERROR'
            return st, {'passed': False, 'stage': 'structure', 'numerics': num}, None
        return 'PASS', {'passed': True, 'stage': 'structure (quick: gates 1 and 3 not run)',
                        'bitwise': num['bitwise'], 'measured_tier': num.get('measured_tier'), 'numerics': num}, None
    performance = None
    if not skip_timing:
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
    tier = num.get('measured_tier')
    correctness = {'stage': 'bitwise', 'measured_tier': tier, 'numerics': num}
    if num.get('exact_failures'):
        return 'INCORRECT', dict(correctness, passed=False, verdict='structure',
                                 note='exact checks failed: the candidate does different work'), performance
    if num.get('errors') or num.get('status') not in ('PASS', 'NOT_BITWISE', 'INCORRECT'):
        st = 'INVALID' if num.get('status') == 'INVALID' else 'ERROR'
        return st, dict(correctness, passed=False, note='numerics could not compare the records'), performance
    status, corr, perf = statistical(tree, out, cand_env, correctness, performance)
    if num['bitwise'] and status == 'PASS':
        # gate 2 judged the deterministic path bitwise; gate 3 ran anyway on the production (timed) path
        return 'PASS', dict(corr, verdict='bitwise', measured_tier=tier), perf
    return status, corr, perf


def guarded(out, fn, tree_src=None):
    """Inputs checked before (nothing measured on a mismatch) and after; the tree is copied first and its digest
    checked again at the end (the code must not write into its own tree)."""
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
        with scratch_dir('yoco-src-') as tmp:
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
    for k in ('speedup_vs_frozen_reference', 'beyond_reference_noise', 'reference_noise',
              'speedup_vs_paired_reference', 's_per_update', 'notes'):
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
            with scratch_dir('yoco-ref-') as tmp:
                ref = strip(time_tree(materialize(check_reference(), Path(tmp) / 'reference'), out, 'reference',
                                      candidate=False))
        res = strip(time_tree(tree, out, 'tree', cand_env=cand_env, candidate=under_test(a.tree)))
        if reference:
            res['reference_timing'] = reference
        if a.paired and ref is not None:
            res['paired_reference'] = ref
            if res['status'] == 'ok' and ref['status'] == 'ok':
                res['speedup_vs_paired_reference'] = round(ref['s_per_update'] / res['s_per_update'], 4)
                res['note'] = ('screening only: one reference run just before, same host and session; the final '
                               'evaluation times reference -> candidate -> reference')
        return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env,
                    host=socket.gethostname())
    return finish(out, 'time.json', guarded(out, run, Path(a.tree).resolve()))


CATEGORIES = [
    ('nccl', r'nccl'),
    ('flash-attention', r'flash|fmha'),
    ('gemm', r'gemm|cutlass|sm90_xmma|sm80_xmma|cublas|ampere_|hopper_|s16816|nvjet|xmma'),
    ('optimizer (multi_tensor)', r'multi_tensor|adam|l2norm'),
    ('triton (rotary / recurrence / inductor)', r'^triton_|rotary|recurrence'),
    ('jiterator (swiglu)', r'jiterator|swiglu'),
    ('loss (log_softmax / nll)', r'softmax|nll'),
    ('memcpy/cast', r'copy|memcpy|cast|convert|_to_copy|memset'),
    ('reduce', r'reduce'),
    ('elementwise', r'elementwise|vectorized|unrolled|index|cat|mul|add|exp|fill'),
]


def categorize(kernels):
    cats = {}
    for k in kernels:
        n = k['name'].lower()
        cat = next((c for c, rx in CATEGORIES if re.search(rx, n)), 'other')
        cats[cat] = cats.get(cat, 0.0) + k['total_ms']
    return {k: round(v, 1) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}


def cmd_profile(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)
    mode = 'nsys' if a.nsys else 'torch'

    def run(tree):
        res = time_tree(tree, out, 'profile', cand_env=cand_env, profile=mode, profile_iters=a.iters,
                        candidate=under_test(a.tree))
        ranks = res.pop('_ranks', None)
        if res['status'] != 'ok':
            return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env)
        warmup = CFG['timing']['warmup']
        res['note'] = (f'{a.iters} updates after a warmup of {warmup}; instrumented: the time is not comparable with '
                       f'`time`')
        r0 = ranks[0]
        ends = r0['iteration_end_s']
        res['update_s_host'] = [round(ends[i] - ends[i - 1], 3) for i in range(warmup, min(len(ends), warmup + a.iters))]
        if mode == 'torch':
            summ = json.loads((out / 'timing_profile.torchprof.json').read_text())
            summ['categories_ms'] = categorize(summ['kernels'])
            res['torch_profile'] = summ
            res['trace'] = str(out / 'timing_profile.trace.json')
            md = ['# profile', '', f'tree {a.tree}, config {a.config}', '', res['note'], '',
                  f'window {summ["span_s"]:.2f} s for {summ["iterations"]} updates; GPU busy '
                  f'{summ["gpu_busy_s"]:.2f} s ({summ["gpu_busy_fraction"]:.0%}); kernel time '
                  f'{summ["kernel_time_s"]:.2f} s (rank 0)', '',
                  'host time per update (s): ' + ', '.join(map(str, res['update_s_host'])), '',
                  '## kernel time by category (ms, rank 0)', '', '| category | ms |', '|---|---|']
            md += [f'| {k} | {v} |' for k, v in summ['categories_ms'].items()]
            md += ['', '## ranges (fairseq phases, model blocks; GPU ms of kernels launched inside, ranges nest)', '',
                   '| range | GPU ms | calls | CPU ms |', '|---|---|---|---|']
            md += [f'| {k} | {v["cuda_ms"]} | {v["calls"]} | {v["cpu_ms"]} |' for k, v in summ['annotations'].items()]
            md += ['', '## top kernels', '', '| ms | calls | share | kernel |', '|---|---|---|---|']
            md += [f'| {k["total_ms"]} | {k["calls"]} | {k["share_of_kernel_time"]} | `{k["name"][:140]}` |'
                   for k in summ['kernels'][:50]]
            md += ['', '## top CPU ops (self time)', '', '| ms | calls | op |', '|---|---|---|']
            md += [f'| {o["self_cpu_ms"]} | {o["calls"]} | {o["name"]} |' for o in summ['cpu_ops_self_time'][:25]]
            (out / 'profile.md').write_text('\n'.join(md) + '\n')
        else:
            res['trace'] = str(out / 'nsys_profile.nsys-rep')
        return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env)
    return finish(out, 'profile.json', guarded(out, run, Path(a.tree).resolve()))


def cmd_baseline(a):
    out = Path(a.out).resolve()

    def run(_):
        ref_src = check_reference()
        with scratch_dir('yoco-ref-') as tmp:
            tree = materialize(ref_src, Path(tmp) / 'reference')
            runs = {k: time_tree(tree, out, k, candidate=False)
                    for k in ['ref_%s' % chr(ord('a') + i) for i in range(CFG['timing'].get('reference_runs', 3))]}
        ok = [v for v in runs.values() if v['status'] == 'ok']
        tot = [v['total_s'] for v in ok]
        res = {'status': 'ok' if len(ok) == len(runs) else 'ERROR', 'runs': {k: strip(v) for k, v in runs.items()},
               'total_s': tot, 'spread': round((max(tot) - min(tot)) / statistics.mean(tot), 4) if tot else None}
        if ok:
            r = ok[0]['_ranks']
            res['rng_fingerprints'] = {str(x['rank']): x['rng_fingerprints'] for x in r}
            res['rng_identical_across_runs'] = all(
                [x['rng_fingerprints'] for x in v['_ranks']] == [x['rng_fingerprints'] for x in r] for v in ok)
            res['update_end_s'] = {k: v['_ranks'][0]['iteration_end_s'] for k, v in runs.items() if v['status'] == 'ok'}
        return res
    return finish(out, 'baseline.json', guarded(out, run))


# ---------------------------------------------------------------- formulator commands

def cmd_record(a):
    """Formulator: record cases (default bench cases) of a tree into a directory (reference records, variants)."""
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(a.cases).read_text()) if a.cases else json.loads(path(CFG['numerics']['cases']).read_text())
    tree = Path(a.tree).resolve()
    if a.traj_seeds:
        spec = json.loads(path(CFG['trajectory']['spec']).read_text())
        cases = [traj_case(spec, int(s)) for s in a.traj_seeds.split(',')]
    with scratch_dir('yoco-src-') as tmp:
        tree = materialize(tree, Path(tmp) / 'tree')
        failed = record_cases(tree, out, cases, out, candidate_environment(a.config), 'formulate', variant=a.variant,
                              losses_only=bool(a.traj_seeds), candidate=False, deterministic=not a.traj_seeds)
    print(json.dumps({'status': 'ok' if not failed else failed, 'out': str(out), 'cost': cost()}))
    return 1 if failed else 0


def cmd_freeze_inputs(a):
    inputs = freeze_inputs(a.paths)
    f = ROOT / 'task' / 'formulation.json'
    data = json.loads(f.read_text()) if f.exists() else {}
    data['inputs'] = inputs
    f.write_text(json.dumps(data, indent=1) + '\n')
    print(json.dumps({'files': len(inputs)}))
    return 0


def cmd_loader(a):
    got, err = loader_digest(Path(a.tree).resolve(), candidate=False)
    print(json.dumps(got) if got else err)
    return 0 if got else 1


def cmd_setup(a):
    got, err = setup_fingerprint(Path(a.tree).resolve(), candidate=False)
    print(json.dumps({'files': got['files'], 'loops': got['fingerprint'].get('train.update_loops'),
                      'sha': {k: hashlib.sha256(str(v).encode()).hexdigest()[:16]
                              for k, v in got['fingerprint'].items()}}) if got else err)
    return 0 if got else 1


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
    f.add_argument('--nsys', action='store_true')
    f.add_argument('--iters', type=int, default=5)
    b = sub.add_parser('baseline')
    b.add_argument('--out', required=True)
    r = sub.add_parser('record', help='formulator: record numerics cases or reference trajectories')
    r.add_argument('--tree', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--cases')
    r.add_argument('--variant')
    r.add_argument('--config')
    r.add_argument('--traj-seeds')
    ld = sub.add_parser('loader', help='formulator: print the loader digest of a tree')
    ld.add_argument('--tree', required=True)
    sp = sub.add_parser('setup', help='formulator: print the setup fingerprint of a tree')
    sp.add_argument('--tree', required=True)
    fi = sub.add_parser('freeze-inputs', help='formulator: write the frozen input fingerprints')
    fi.add_argument('paths', nargs='+')
    a = p.parse_args()
    return {'freeze-inputs': cmd_freeze_inputs, 'check': cmd_check, 'time': cmd_time, 'profile': cmd_profile,
            'baseline': cmd_baseline, 'record': cmd_record, 'loader': cmd_loader, 'setup': cmd_setup}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
