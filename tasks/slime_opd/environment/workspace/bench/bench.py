"""slime_opd: on-policy distillation in slime on 4 GPUs (student Qwen3.5-9B: sglang rollout + Megatron training;
teacher Qwen3.5-27B: an sglang server that scores the student's samples). One command per question.

Run from the workspace root (/opt/researchboost/ws) with the training environment's Python (bench.json python). GPU
runs take /tmp/slime_opd_gpu.lock themselves, one at a time; do NOT wrap these commands in flock (that would
deadlock). --tree is a directory holding the three repositories under test: slime/, sglang/, Megatron-LM/.

  python bench/bench.py check   --tree <dir> --out <dir> [--config <file>] [--side rollout|teacher|train] [--quick | --skip-timing | --n1]
        the verdict -> <out>/check.json. The sides judged are the ones the tree's diff against the reference touches
        (bench.json sides and sides_exclusive; a teacher switch in --config judges the teacher; a changed file outside
        every side, or no change at all, judges all three), plus --side. --quick / --skip-timing: no gate 1 (same
        correctness gates). --n1: no gate 1 and no bitwise attempt: the statistical gates only. --final is the final
        evaluation's (root or the verifier's marker): the full RL loop timed reference -> candidate -> reference, and
        all three sides judged.
  python bench/bench.py time    --tree <dir> --out <dir> [--config <file>] [--side rollout|teacher|train]
        one timing run per side (default: the sides the diff touches) -> <out>/time.json: seconds per window against
        the frozen reference timing of this host (task/timing_ref.json), and the projected step speedup.
  python bench/bench.py profile --tree <dir> --out <dir> --side rollout|teacher|train [--config <file>]
        rollout: torch profiler on every student sglang worker for 200 decode steps of rollout 1; teacher: the
        teacher server's profiler over 64 forward steps of replayed rollout 1; train: torch.profiler over a replayed train
        step (dump 2, after its first use, as the timed steps) on every rank -> traces under <out>/trace and <out>/kernel_report.md of the largest trace.
  formulator only: fixture, freeze-rollout, freeze-teacher, freeze-train, freeze-timing, control-arm, freeze-inputs

Every run is one bench/launch.sh process of a fresh copy of the tree (the frozen recipe, colocated on the 4 GPUs; the
teacher's sglang server first in rollout, full and teacher modes), in its own process group, with a fresh TMPDIR and
fresh compile caches (cold), a timeout, and the judge's instruments (bench/rollout_hook.py, bench/train_hook.py,
bench/force_len.py, bench/teacher_replay.py: clocks, the parallel layout that ran, the rollout dump, the generation
ends, the parameter records, the teacher's replayed log-probabilities). Every GPU run holds the GPU lock, stops
processes an earlier bench run left behind, waits until the GPUs hold no foreign memory, and kills what it started
afterwards. Every command checks the frozen inputs declared in task/formulation.json before (mismatch: nothing is
measured) and after (mismatch: ERROR), and the source digest of the tree copy before and after.

Interpreters: the tree under test runs with bench.json candidate_python (the agent's environment, which sees the
training environment's packages unless it installs its own versions); reference runs, the scorer and formulator
commands use bench.json python.

Candidate switches: a JSON file {"environment": {"NAME": "value"}} given with --config; the bench sets those
variables for the candidate's processes. Variables the bench manages (bench.json reserved_env) and SLIME_OPD_BENCH_*
are refused. OPD_TEACHER_SERVER_ARGS adds arguments to the teacher's sglang.launch_server (bench/launch.sh).

Clocks (the judge's own wall clock, bench.json timing): rollout mode, 4 rollouts with every sample's length forced to
the reference fixture's (max_new_tokens = recorded length, ignore_eos), the teacher online; seconds per rollout = end
of rollout k-1 (after the judge's dump) -> end of rollout k, windows k = 1..3; teacher mode, the samples of fixture
rollouts 0..3 replayed through the tree's reward path against the teacher alone, seconds per rollout = first request
-> post-processing done, windows k = 1..3; train mode, 8 train steps on the frozen rollout dumps in the order 4, 1, 2, 3, 1, 2,
3, 4 (bench.json timing.train.dump_order), seconds per step = before step k -> before step k+1 (slowest rank), windows
k = 4..6 (dumps 1..3; the warm-up runs dumps 1..4 once, 4 for step 7's forward passes inside window 6); full loop (final evaluation), 5 RL steps with forced lengths,
seconds per step = end of rollout k-1 -> end of rollout k, windows k = 2..4, after an untimed priming run (train mode
on dumps 0..4, same tree and switches) whose compile caches the timed run starts from. Either way the shape-dependent
first uses (Triton autotuning and compilation: TRITON_CACHE_AUTOTUNING=1 keeps the autotune results in the cache)
happen before the clock; train and full timings report `first_use`, the seconds of Triton autotuning and compilation
inside each timed window (slowest process), which should be about zero.

Verdict (check); each gate stops the evaluation when it fails:
  0. structure  the required files exist and slime/train.py is the reference's (AST) -> else ERROR.
  1. benefit    agent's check: per judged side one run against the frozen reference windows of this host; side
                speedup = the mean of the per-window ratios; projected step speedup = 1 / (1 - sum f (1 - 1 / s)) with
                the frozen step fractions f of the sides (the teacher's f is its share on the critical path: the end of
                a rollout minus its last generation end; when the rollout side is timed too, its windows already hold
                the teacher's share and the teacher's term is dropped); must exceed the bar (benefit.min_speedup or
                one plus twice the frozen reference spread, the larger) -> else UNAVAILABLE (no_benefit). Final
                evaluation: full loop reference A, candidate, reference B; speedup = min over windows of min(A, B) /
                candidate; a reference spread beyond timing.spread_band is INVALID.
  L. layout     every run's parallel layout as it ran (Megatron parallel state of every rank; every student sglang
                engine's server arguments; the teacher server's) equals the reference's (TP2 x CP2 = DP1 actor,
                2 engines x TP2, teacher TP4) -> else INCORRECT.
                No engine and not the teacher runs speculative decoding (task rule; server arguments
                speculative_algorithm / speculative_draft_model_path) -> else INCORRECT.
  R. rollout    exact items of the rollout dumps (samples, prompts, forced lengths, finite log-probs, the teacher's
                answer with a log-probability for every token) -> else INCORRECT; student token forcing through the
                stock student engine (mean |d| <= 2x the reference's own, large gaps not significantly more frequent,
                sampler paired t <= 4); the teacher log-probabilities the rollout carries against the stock teacher
                replayed on the same tokens (rate, events, bias, relative to the reference rollout's own); and the
                train-inference mismatch of the reference Megatron on the candidate's rollout <= 2x the reference's.
  S. teacher    the candidate's teacher replay of fixture rollouts 0..3 against the frozen stock replay of the same
                tokens: exact items (samples, one finite log-probability per response token) -> else INCORRECT;
                mean |d| <= 2x the stock teacher's own spread (the unchanged teacher re-batched four ways, pooled:
                teacher_rules.rebatchings; floor teacher_rules.mean_abs_floor),
                tokens with |d| > tau not significantly more frequent, no per-sample bias (paired t <= 4 and beyond
                the floor) -> else INCORRECT.
  T. train      deterministic record (3 train steps on the frozen dumps, whose teacher answers the tree's own
                post-processing turns into teacher log-probabilities): exact items -> else INCORRECT; every bf16
                parameter, fp32 master and Adam moment bitwise identical after every update -> PASS N0; else the
                statistical gate (endpoints within noise_k x the reference's run-to-run spread AND sigma_k x the
                spread over 6 data sets; update-direction cosine >= the frozen minimum) -> PASS N1, else INCORRECT;
                and the mismatch of the candidate's Megatron on the frozen rollout <= 2x the reference's.
  F. final      the full loop's mismatch (candidate vs both reference runs) <= 2x, and its layout.
"""
import argparse
import ast
import concurrent.futures
import fcntl
import hashlib
import json
import math
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
ROOT = BENCH.parent


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
LOCK = Path(CFG['gpu_lock'])
PY = Path(CFG['python'])
CAND_PY = Path(CFG['candidate_python'])
SCRATCH = Path(CFG.get('scratch', '/tmp'))
TIMEOUT = CFG['timeouts_s']
RESERVED = set(CFG['reserved_env'])
SIGNATURES = [(re.compile(x['regex']), x['reason']) for x in CFG.get('invalid_run_signatures', [])]
COST = {'gpu_seconds': 0.0, 'gpu_runs': 0}
TREE_SKIP = {'.git', '__pycache__', '.pytest_cache', 'wandb'}
MARK = 'SLIME_OPD_BENCH_RUN'
PREFIX = 'SLIME_OPD_BENCH'
TASK = ROOT / 'task'
REPOS = ('slime', 'sglang', 'megatron')
FIX = CFG['fixture']


def log(msg):
    print(f'[bench {datetime.now():%H:%M:%S}] {msg}', flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def evaluator_sha256():
    """Joint hash of the frozen evaluator: bench/ code and config, task/ frozen references."""
    files = sorted(p for p in BENCH.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files += sorted(p for p in TASK.glob('*') if p.is_file() and p.suffix in ('.json', '.npy', '.npz'))
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode() + b'\0' + sha256_file(p).encode() + b'\n')
    return h.hexdigest()


def gpu_uuids():
    """The GPUs this run sees, by UUID: the frozen reference of a lane is valid for exactly these."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=60).stdout
        return sorted(u.strip() for u in out.splitlines() if u.strip())
    except Exception:
        return []


def frozen(name):
    f = TASK / name
    return json.loads(f.read_text()) if f.exists() else None


# ---------------------------------------------------------------- trees

def trees_of(root):
    """{'slime': .., 'sglang': .., 'megatron': ..} of a --tree directory."""
    return {k: (Path(root) / CFG['candidate'][k]).resolve() for k in REPOS}


def reference_trees():
    return {k: (ROOT / CFG['reference'][k]).resolve() for k in REPOS}


def git(repo, *a):
    return subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(repo), *a], capture_output=True,
                          text=True).stdout


def check_reference():
    ref = reference_trees()
    for k, d in ref.items():
        if git(d, 'status', '--porcelain', '--untracked-files=no').strip():
            raise SystemExit(f'reference tree {d} has local changes')
    head = git(ref['slime'], 'rev-parse', 'HEAD').strip()
    if head != CFG['reference_commit']:
        raise SystemExit(f'reference slime is not the frozen upstream commit (head {head})')
    return ref


def changed_files(cand):
    """Paths (prefixed by the repository directory) where the candidate differs from the reference's commit,
    untracked files included."""
    ref = reference_trees()
    out = []
    for k in REPOS:
        head = git(ref[k], 'rev-parse', 'HEAD').strip()
        names = set(git(cand[k], 'diff', '--name-only', head).split())
        for line in git(cand[k], 'status', '--porcelain', '--untracked-files=all').splitlines():
            names.add(line[3:].strip())
        out += [CFG['candidate'][k] + '/' + n for n in sorted(names) if n]
    return out


ALL_SIDES = ('rollout', 'teacher', 'train')


def sides_of(files, cand_env=None):
    """Sides a diff touches: bench.json sides_exclusive (a path prefix that belongs to the listed sides only) first,
    then every side whose prefixes match; a teacher switch judges the teacher; a file outside every side, or no change
    at all, judges all three."""
    sides, other = set(), []
    for f in files:
        if f.endswith(('.md', '.txt', '.rst')):
            continue
        excl = [v for k, v in CFG.get('sides_exclusive', {}).items() if f.startswith(k)]
        if excl:
            sides.update(excl[0])
            continue
        hit = [s for s, pre in CFG['sides'].items() if any(f.startswith(p) for p in pre)]
        if hit:
            sides.update(hit)
        else:
            other.append(f)
    if any(k in (cand_env or {}) for k in CFG.get('teacher_switches', [])):
        sides.add('teacher')
    if other or not sides:
        sides.update(ALL_SIDES)
    return sides, other


def materialize(src, dest):
    """A fresh copy of the three repositories (no .git): {'slime': dest/slime, ...}."""
    def ignore(directory, names):
        return [n for n in names if n in TREE_SKIP or n.endswith('.pyc')]
    out = {}
    for k in REPOS:
        d = Path(dest) / CFG['candidate'][k]
        shutil.copytree(src[k], d, ignore=ignore, symlinks=True)
        out[k] = d
    for d in Path(dest).rglob('*'):
        if d.is_dir() and not d.is_symlink():
            d.chmod(d.stat().st_mode | 0o700)
    return out


def tree_digest(tr):
    h = hashlib.sha256()
    for k in REPOS:
        base = Path(tr[k])
        for f in sorted(base.rglob('*')):
            rel = f.relative_to(base)
            if any(part in TREE_SKIP for part in rel.parts) or f.is_symlink() or not f.is_file() or \
                    f.suffix == '.pyc':
                continue
            h.update(f'{k}/{rel}'.encode() + b'\0' + sha256_file(f).encode() + b'\n')
    return h.hexdigest()


def structure(cand):
    missing = [f for f in CFG['required_files'] if not (Path(cand['slime']).parent / f).is_file()]
    if missing:
        return [f'missing: {missing}']
    ref = reference_trees()
    a = ast.dump(ast.parse((Path(cand['slime']) / 'train.py').read_text()))
    b = ast.dump(ast.parse((ref['slime'] / 'train.py').read_text()))
    return [] if a == b else ['slime/train.py differs from the reference (the judged entry point must not change)']


# ---------------------------------------------------------------- GPU runs

class GpuLock:
    def __init__(self, what, timeout=10 * 3600):
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


def _marked(value=None):
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


def base_env(tmp, cand_env=None, candidate=False):
    """Environment of every run: the caller's environment minus reserved variables, then the candidate's switches,
    then what the bench manages (scratch, caches, offline switches, toolkit, PATH)."""
    env = {k: v for k, v in os.environ.items() if k not in RESERVED and not k.startswith(PREFIX)}
    env.update(cand_env or {})
    caches = Path(tmp) / 'caches'
    env.update({
        'TMPDIR': str(tmp), 'TMP': str(tmp), 'TEMP': str(tmp),
        'TORCHINDUCTOR_CACHE_DIR': str(caches / 'inductor'), 'TRITON_CACHE_DIR': str(caches / 'triton'),
        'TORCH_EXTENSIONS_DIR': str(caches / 'torch_ext'), 'CUDA_CACHE_PATH': str(caches / 'cuda'),
        'XDG_CACHE_HOME': str(caches / 'xdg'), 'FLASHINFER_WORKSPACE_BASE': str(caches / 'flashinfer'),
        'RAY_TMPDIR': str(Path(tmp) / 'ray'), 'MPLCONFIGDIR': str(caches / 'mpl'),
        'ASSETS': os.environ.get('ASSETS', '/opt/researchboost/assets'),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
        'HF_HOME': str(caches / 'hf'), 'WANDB_MODE': 'disabled', 'PYTHONNOUSERSITE': '1',
        'CUDA_HOME': CFG['cuda_home'],
        'PATH': interpreter_path(candidate, env.get('PATH', '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin')),
    })
    env.pop('PYTHONPATH', None)
    for v in ('inductor', 'triton', 'torch_ext', 'cuda', 'xdg', 'flashinfer', 'mpl', 'hf'):
        (caches / v).mkdir(parents=True, exist_ok=True)
    (Path(tmp) / 'ray').mkdir(parents=True, exist_ok=True)
    return env


def run_slime(tr, mode, out, num_rollout, what, candidate, cand_env=None, deterministic=False, timeout=None,
              caches_from=None, keep_caches=None, first_use=False, **launch):
    """One train.py run of trees `tr` via bench/launch.sh under the GPU lock. Returns a dict with status
    ('ok', 'ERROR', 'TIMEOUT', 'INVALID'), the log, the judge's record directory and the wall time."""
    out = Path(out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    jd = out / 'judge'
    tag = uuid.uuid4().hex[:12]
    with scratch_dir('sg-run-') as tmp:
        if caches_from:
            # the compile caches of this tree's priming run (time_side): the run starts warm
            shutil.copytree(caches_from, Path(tmp) / 'caches', symlinks=True)
        env = base_env(tmp, cand_env, candidate=candidate)
        env.update(SLIME=str(tr['slime']), SGLANG=str(tr['sglang']), MEGATRON=str(tr['megatron']), MODE=mode,
                   OUT=str(out), NUM_ROLLOUT=str(num_rollout), SLIME_OPD_BENCH_DIR=str(jd))
        if deterministic:
            env.update(CFG['train_record']['environment'])
        if first_use:
            # timing runs: Triton keeps autotune results in its cache (a priming run's carry over to the timed run)
            # and the judge's hook logs the autotuning and compilation that still happen (bench/train_hook.py)
            env.update({'TRITON_CACHE_AUTOTUNING': '1', 'SLIME_OPD_BENCH_FIRST_USE': '1'})
            if os.environ.get('RBB_FIRST_USE_ALL') == '1':
                # operator's opt-in: measure the first uses of every process, the sglang engines included
                env['SLIME_OPD_BENCH_FIRST_USE_ALL'] = '1'
        env.update({k: str(v) for k, v in launch.items() if v is not None})
        env[MARK] = tag
        with GpuLock(what):
            _kill_marked()
            before, problem = wait_idle(what)
            if problem:
                return {'status': 'INVALID', 'reason': problem, 'out': str(out), 'judge': str(jd)}
            rc, status, t0 = None, None, time.time()
            log(f'run {what}: {mode} x{num_rollout} -> {out}')
            try:
                with open(out / 'launcher.log', 'w') as lf:
                    proc = subprocess.Popen(['bash', str(BENCH / 'launch.sh')], cwd=str(out), env=env, stdout=lf,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        rc = proc.wait(timeout=timeout or TIMEOUT[mode])
                    except subprocess.TimeoutExpired:
                        status = 'TIMEOUT'
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            finally:
                wall = time.time() - t0
                left = _kill_marked(tag)
            after, _ = wait_idle(f'{what} (after)', patience=180)
        if keep_caches and rc == 0 and status is None:
            shutil.copytree(Path(tmp) / 'caches', keep_caches, symlinks=True)
    res = {'status': status or ('ok' if rc == 0 else 'ERROR'), 'returncode': rc, 'wall_s': round(wall, 1),
           'out': str(out), 'judge': str(jd), 'log': str(out / 'train.log'),
           'gpus': [(g['index'], g['name'], g['driver']) for g in before]}
    if status == 'TIMEOUT':
        res['reason'] = f'{what} exceeded {timeout or TIMEOUT[mode]}s; process group killed'
    elif rc != 0:
        res['reason'] = f'{what}: train.py exited {rc}; log tail: {log_tail(out / "train.log")}'
    if left:
        res['warning'] = f'{left} processes of the run survived SIGKILL'
    hits = signatures(out / 'train.log')
    if hits and res['status'] == 'ok':
        res.update(status='INVALID', reason='invalid-run signature in the log: ' + hits[0]['reason'], signatures=hits)
    log(f'  {what}: {res["status"]} rc={rc} wall={wall:.0f}s')
    return res


def log_tail(p, n=1500):
    p = Path(p)
    if not p.exists():
        return ''
    text = p.read_text(errors='replace')
    # the first Python traceback is the most useful part of a failed slime run
    i = text.find('Traceback (most recent call last)')
    return (text[i:i + n] if i >= 0 else text[-n:]).replace('\n', ' | ')


def signatures(log_file):
    hits, text = [], Path(log_file).read_text(errors='replace') if Path(log_file).exists() else ''
    for pattern, reason in SIGNATURES:
        m = pattern.search(text)
        if m:
            line = text[text.rfind('\n', 0, m.start()) + 1:text.find('\n', m.end())]
            hits.append({'reason': reason, 'line': line.strip()[:300]})
    return hits


ANSI = re.compile(r'\x1b\[[0-9;]*m')
STEP_LINE = re.compile(r'model\.py:\d+ - step (\d+): (\{.*\})')


def _lit(s):
    s = re.sub(r'\b(nan|inf|-inf)\b', 'None', s)
    try:
        return ast.literal_eval(s)
    except Exception:  # noqa: BLE001
        return None


def train_metrics(log_file):
    """slime's per-step training metrics (model.py 'step N: {...}' lines)."""
    res = {}
    if not Path(log_file).exists():
        return res
    for line in Path(log_file).read_text(errors='replace').splitlines():
        m = STEP_LINE.search(ANSI.sub('', line))
        if m:
            d = _lit(m.group(2))
            if d is not None:
                res[int(m.group(1))] = d
    return res


# ---------------------------------------------------------------- the judge's clocks and layout

def _jsonl(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def windows(mode, jd, num_rollout):
    """Seconds per rollout (rollout), per train step (train) or per RL step (full) of the windows of bench.json
    timing[mode], from the judge's clocks; (list, problem)."""
    warm = CFG['timing'][mode]['warmup']
    if mode == 'teacher':
        rows = {r['rollout_id']: r for r in _jsonl(Path(jd) / 'teacher_clock.jsonl')}
        if sorted(rows) != list(range(num_rollout)):
            return None, f'the teacher replay covered rollouts {sorted(rows)}, expected 0..{num_rollout - 1}'
        return [rows[k]['end'] - rows[k]['start'] for k in range(warm, num_rollout)], None
    if mode in ('rollout', 'full'):
        rows = {r['rollout_id']: r for r in _jsonl(Path(jd) / 'rollout_clock.jsonl')}
        if sorted(rows) != list(range(num_rollout)):
            return None, f'the judge saw rollouts {sorted(rows)}, expected 0..{num_rollout - 1}'
        return [rows[k]['entry'] - rows[k - 1]['exit'] for k in range(max(warm, 1), num_rollout)], None
    per_rank = {}
    for f in sorted(Path(jd).glob('train_clock.rank*.jsonl')):
        rows = _jsonl(f)
        per_rank[f.name] = {}
        for r in rows:
            per_rank[f.name].setdefault(r['rollout_id'], r['t'])
    if len(per_rank) != CFG['world_size']:
        return None, f'train clocks of {len(per_rank)} ranks, expected {CFG["world_size"]}'
    for name, t in per_rank.items():
        if sorted(t) != list(range(num_rollout)):
            return None, f'{name}: train steps {sorted(t)}, expected 0..{num_rollout - 1}'
    return [max(t[k + 1] - t[k] for t in per_rank.values()) for k in range(warm, num_rollout - 1)], None


def window_bounds(mode, jd, num_rollout):
    """(start, end) wall clock of each timed window of a rollout, train or full run (as windows())."""
    warm = CFG['timing'][mode]['warmup']
    if mode in ('rollout', 'full'):
        rows = {r['rollout_id']: r for r in _jsonl(Path(jd) / 'rollout_clock.jsonl')}
        return [(rows[k - 1]['exit'], rows[k]['entry']) for k in range(max(warm, 1), num_rollout)]
    ts = []
    for f in sorted(Path(jd).glob('train_clock.rank*.jsonl')):
        t = {}
        for r in _jsonl(f):
            t.setdefault(r['rollout_id'], r['t'])
        ts.append(t)
    return [(min(t[k] for t in ts), max(t[k + 1] for t in ts)) for k in range(warm, num_rollout - 1)]


def first_use_windows(mode, jd, num_rollout):
    """Seconds of Triton autotuning and compilation inside each timed window (the judge's hook, first_use.<pid>.jsonl
    of every Megatron process): per window the largest sum of one process; None when the hook logged nothing."""
    files = sorted(Path(jd).glob('first_use.*.jsonl'))
    if not files:
        return None
    try:
        bounds = window_bounds(mode, jd, num_rollout)
    except (KeyError, ValueError):
        return None
    events = [_jsonl(f) for f in files]
    per, names = [], {}
    for a, b in bounds:
        worst = 0.0
        for ev in events:
            tot = 0.0
            for e in ev:
                o = min(e['t1'], b) - max(e['t0'], a)
                if o > 0:
                    tot += o
                    names[e['name']] = names.get(e['name'], 0.0) + o
            worst = max(worst, tot)
        per.append(round(worst, 2))
    top = sorted(names.items(), key=lambda x: -x[1])[:6]
    return {'seconds': per, 'processes': len(files),
            'kernels_s': {n: round(v, 1) for n, v in top},
            'note': 'Triton autotuning and compilation inside each timed window, slowest process; kernels_s sums all '
                    'processes'}


def teacher_tails(jd, num_rollout):
    """Per rollout k >= 1 of a forced-length run: the end of the rollout minus the last generation end of its samples
    (the teacher's share of the rollout on the critical path)."""
    per_roll = FIX['samples_per_rollout']  # sample indices of rollout k: [k * per_roll, (k + 1) * per_roll)
    ends = {}
    for r in _jsonl(Path(jd) / 'gen_end.jsonl'):
        ends[r['index']] = r['t']
    clock = {r['rollout_id']: r for r in _jsonl(Path(jd) / 'rollout_clock.jsonl')}
    out = []
    for k in range(1, num_rollout):
        idx = [i for i in ends if k * per_roll <= i < (k + 1) * per_roll]
        if k in clock and idx:
            out.append(round(clock[k]['entry'] - max(ends[i] for i in idx), 3))
    return out


def layout_of(jd, mode):
    lay = {}
    if mode in ('train', 'full'):
        lay['megatron'] = [json.loads(f.read_text()) for f in sorted(Path(jd).glob('megatron_layout.rank*.json'))]
    if mode in ('rollout', 'full'):
        f = Path(jd) / 'sglang_layout.json'
        lay['sglang'] = json.loads(f.read_text()) if f.exists() else None
    if mode in ('rollout', 'full', 'teacher'):
        f = Path(jd) / 'teacher_layout.json'
        try:
            lay['teacher'] = json.loads(f.read_text()) if f.exists() else None
        except ValueError:
            lay['teacher'] = {'error': 'unreadable server info'}
    return lay


SGLANG_KEYS = ('tp_size', 'dp_size', 'pp_size', 'ep_size', 'nnodes', 'enable_dp_attention')
SPECULATIVE_KEYS = ('speculative_algorithm', 'speculative_draft_model_path')


def speculative_of(info):
    """The speculative-decoding settings a server reports (task rule: none); {} when it runs none."""
    return {k: info.get(k) for k in SPECULATIVE_KEYS if info.get(k) not in (None, '', 'None', 'none', 'NONE', False)}


def layout_problems(lay):
    """The layout that ran against bench.json layout (Megatron) and the reference's frozen engine layout."""
    probs = []
    want = CFG['layout']
    if 'megatron' in lay:
        m = lay['megatron']
        if len(m) != want['megatron']['world']:
            probs.append(f'{len(m)} Megatron ranks reported their layout, expected {want["megatron"]["world"]}')
        for r in m:
            bad = {k: r.get(k) for k, v in want['megatron'].items() if r.get(k) != v}
            if bad:
                probs.append(f'Megatron rank {r.get("rank")}: {bad}, expected {want["megatron"]}')
                break
    if 'sglang' in lay:
        s = lay['sglang']
        ref = (frozen('layout_ref.json') or {}).get('sglang_engine')
        if not s or s.get('error'):
            probs.append(f'the sglang engines\' layout could not be read: {(s or {}).get("error")}')
        else:
            eng = s.get('engines') or []
            if len(eng) != want['sglang']['engines'] or not all(e.get('answered') for e in eng):
                probs.append(f'{len(eng)} sglang engines answered ({[e.get("url") for e in eng]}), expected '
                             f'{want["sglang"]["engines"]}')
            for e in eng:
                got = {k: e.get(k) for k in SGLANG_KEYS}
                if got.get('tp_size') != want['sglang']['tp_size'] or (ref is not None and got != ref):
                    probs.append(f'sglang engine {e.get("url")}: {got}, expected {ref or want["sglang"]}')
                    break
            for e in eng:
                spec = speculative_of(e)
                if spec:
                    probs.append(f'sglang engine {e.get("url")} runs speculative decoding {spec}: not allowed (task '
                                 f'rule: no speculative decoding)')
                    break
    if 'teacher' in lay:
        t = lay['teacher']
        ref = (frozen('layout_ref.json') or {}).get('teacher')
        if not t or t.get('error'):
            probs.append(f'the teacher server\'s layout could not be read: {(t or {}).get("error")}')
        else:
            got = {k: t.get(k) for k in SGLANG_KEYS}
            if any(got.get(k) != v for k, v in want['teacher'].items()) or (ref is not None and got != ref):
                probs.append(f'teacher server: {got}, expected {ref or want["teacher"]}')
            spec = speculative_of(t)
            if spec:
                probs.append(f'teacher server runs speculative decoding {spec}: not allowed (task rule: no '
                             f'speculative decoding)')
    return probs


# ---------------------------------------------------------------- timing

def replay_dir(dest, order):
    """A dump directory whose rollout i is fixture dump order[i] (symbolic links)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for i, k in enumerate(order):
        (dest / f'{i}.pt').unlink(missing_ok=True)
        (dest / f'{i}.pt').symlink_to(Path(FIX['dumps']) / f'{k}.pt')
    return str(dest)


def prime_caches(tr, out, label, candidate, cand_env, dest, side='full'):
    """The untimed priming run(s) before a timed run (bench.json timing.<side>.prime: one stage or a list of them,
    each a mode and a number of rollouts). Stage 1 starts cold, every later stage from the caches of the one before,
    so the stages add up: train mode primes the training path on the fixture dumps, rollout mode the generation path
    on the same forced lengths. The caches (Triton's kernels and autotune results, Inductor's) end up in `dest`/last.
    Returns (the last run, that directory or None)."""
    stages = CFG['timing'][side]['prime']
    stages = [stages] if isinstance(stages, dict) else stages
    r, src = None, None
    for i, p in enumerate(stages):
        kept = Path(dest) / f'stage{i + 1}'
        launch = {'LOAD_DUMPS': FIX['dumps']} if p['mode'] == 'train' else {'FORCE_LENGTHS': FIX['lengths']}
        r = run_slime(tr, p['mode'], Path(out) / f'prime{i + 1}_{label}', p['num_rollout'],
                      f'prime {p["mode"]} {label}', candidate, cand_env, caches_from=src,
                      keep_caches=str(kept), first_use=True, **launch)
        if r['status'] != 'ok':
            return r, None
        src = str(kept)
    return r, src


def time_side(side, tr, out, label, candidate, cand_env=None, dump=None, primed=None):
    """One timing run of `side` (rollout | train | full). train: the warm-up replays the timed steps' dumps
    (timing.train.dump_order); full: starts from the compile caches of a priming run (`primed`, else its own)."""
    with scratch_dir('sg-time-') as tmp:
        return _time_side(side, tr, out, label, candidate, cand_env, dump, primed, Path(tmp))


def _time_side(side, tr, out, label, candidate, cand_env, dump, primed, tmp):
    t = CFG['timing'][side]
    n = t['num_rollout']
    launch, prime = {}, None
    if side in ('rollout', 'full'):
        launch['FORCE_LENGTHS'] = FIX['lengths']
    if side in ('train', 'teacher'):
        launch['LOAD_DUMPS'] = replay_dir(tmp / 'dumps', t['dump_order']) if t.get('dump_order') \
            else FIX['dumps']
    if side == 'rollout' and dump is not False:
        launch['SLIME_OPD_BENCH_DUMP'] = '1'
    if t.get('prime') and primed is None:
        prime, primed = prime_caches(tr, out, label, candidate, cand_env, tmp / 'prime_caches', side)
        if primed is None:
            return dict(prime, reason=f'the untimed priming run failed: {prime.get("reason")}', prime=strip(prime))
    r = run_slime(tr, side, Path(out) / f'{side}_{label}', n, f'{side} {label}', candidate, cand_env,
                  caches_from=primed, first_use=True, **launch)
    if prime:
        r['prime'] = {k: prime.get(k) for k in ('status', 'wall_s', 'out')}
    if r['status'] != 'ok':
        return r
    w, problem = windows(side, r['judge'], n)
    if problem:
        return dict(r, status='ERROR', reason=problem)
    fu = first_use_windows(side, r['judge'], n)
    if fu:
        r['first_use'] = fu
        if max(fu['seconds']) > CFG['timing']['first_use_flag_s']:
            r['first_use_warning'] = (
                f'{max(fu["seconds"]):.1f} s of Triton autotuning/compilation inside a timed window '
                f'(kernels: {", ".join(fu["kernels_s"])}). The warm-up runs the timed steps\' data first (train) and '
                f'the full loop starts from a priming run\'s caches, so this is a first use they do not reach (shapes '
                f'that differ from run to run, or an autotuner whose results Triton cannot cache)')
    lay = layout_of(r['judge'], side)
    r.update(windows=[round(x, 3) for x in w], mean=round(statistics.fmean(w), 3), layout=lay,
             layout_problems=layout_problems(lay))
    m = train_metrics(r['log'])
    r['mismatch_steps'] = {i: d.get('train/train_rollout_logprob_abs_diff') for i, d in sorted(m.items())}
    if side == 'rollout':
        clock = _jsonl(Path(r['judge']) / 'rollout_clock.jsonl')
        r['dump_s'] = [round(c.get('dump_s', 0.0), 2) for c in clock]
    if side in ('rollout', 'full'):
        r['teacher_tail_s'] = teacher_tails(r['judge'], n)
    return r


def strip(run):
    return {k: v for k, v in run.items() if k not in ('layout',)} if run else run


def side_speedup(side, run):
    ref = (frozen('timing_ref.json') or {}).get(side)
    if not ref:
        return None
    ratios = [f / c for f, c in zip(ref['windows'], run['windows'])]
    return {'reference_windows': ref['windows'], 'candidate_windows': run['windows'],
            'window_ratios': [round(x, 4) for x in ratios], 'speedup': round(sum(ratios) / len(ratios), 4),
            'reference_spread': ref['spread']}


def projected(speedups):
    """Projected step speedup from side speedups and the frozen step fractions of the sides."""
    fr = (frozen('timing_ref.json') or {}).get('fractions')
    if not fr:
        return None
    sp = dict(speedups)
    if 'rollout' in sp:
        sp.pop('teacher', None)  # the rollout windows already hold the teacher's share
    if any(s not in fr for s in sp):
        return None
    return round(1.0 / (1.0 - sum(fr[s] * (1 - 1 / v) for s, v in sp.items())), 4)


def benefit_agent(side_runs):
    """Gate 1 of the agent's check from its side runs (one per judged side) against the frozen reference windows."""
    ft = frozen('timing_ref.json')
    if not ft:
        return 'ERROR', {'speedup': None, 'reason': 'no frozen reference timing (task/timing_ref.json)'}
    perf = {'sides': {}, 'comparison': 'one candidate run per judged side against the frozen reference timing of '
                                       'this host; the final evaluation times the full loop reference -> '
                                       'candidate -> reference', 'host': socket.gethostname()}
    for s, run in side_runs.items():
        perf['sides'][s] = side_speedup(s, run)
    sp = {s: v['speedup'] for s, v in perf['sides'].items()}
    perf['side_speedup'] = sp
    perf['speedup'] = projected(sp)
    perf['fractions'] = ft.get('fractions')
    spread = max(ft[s]['spread'] for s in side_runs)
    bar = max(CFG['benefit']['min_speedup'], 1 + 2 * spread)
    perf['benefit'] = {'speedup': perf['speedup'], 'speedup_bar': round(bar, 4),
                       'passed': perf['speedup'] is not None and perf['speedup'] > bar}
    return ('ok' if perf['benefit']['passed'] else 'UNAVAILABLE'), perf


def benefit_frozen(cand, out, cand_env, ft):
    """Gate 1 against this lane's frozen reference: only the candidate's full loop is timed.

    The frozen reference (task/timing_ref.json 'full') is the mean of the three runs of the reference tree that
    the preparation measured on these GPUs, with this judge workspace and these installed packages (its key, and so
    the host-local cache's key, covers all of that). Per window the ratio is that reference over the candidate and
    the score is the mean of the windows, exactly as with a fresh reference pair; the bar still rises with the reference's
    own spread. What a repeated reference used to catch -- a machine that moved -- is reported instead: the
    candidate's own windows have to agree, and a run that starts on a busy machine is INVALID before it begins
    (run_slime wait_idle).
    """
    c = time_side('full', cand, out, 'cand', candidate=True, cand_env=cand_env)
    runs = {'cand': strip(c)}
    if c['status'] != 'ok':
        status = 'ERROR' if c['status'] in ('ERROR', 'TIMEOUT') else c['status']
        return status, {'speedup': None, 'runs': runs, 'reference_source': 'frozen',
                        'reason': c.get('reason') or 'the candidate timing run failed'}, c
    R, C = ft['windows'], c['windows']
    if len(R) != len(C):
        return 'INVALID', {'speedup': None, 'runs': runs, 'reference_source': 'frozen',
                           'reason': f'the frozen reference has {len(R)} windows, this run {len(C)}'}, c
    ratios = [r / x for r, x in zip(R, C)]
    spread = float(ft.get('spread') or 0.0)
    own = max((max(C) - min(C)) / min(C), 0.0)
    perf = {'reference_a_s_per_step': R, 'reference_b_s_per_step': None, 'candidate_s_per_step': C,
            'window_ratios': [round(x, 4) for x in ratios], 'speedup': round(sum(ratios) / len(ratios), 4),
            'reference_spread': round(spread, 4), 'runs': runs, 'host': socket.gethostname(),
            'reference_source': 'frozen', 'reference_s_per_step': round(statistics.fmean(R), 3),
            'candidate_mean_s_per_step': round(statistics.fmean(C), 3),
            'candidate_window_spread': round(own, 4),
            'reference_pack': {k: ft.get(k) for k in ('measured', 'spread', 'pair_spread', 'runs', 'gpu_uuids',
                                                      'source')},
            'first_use_s': {'cand': (c.get('first_use') or {}).get('seconds')}}
    warn = []
    if max(perf['first_use_s']['cand'] or [0]) > CFG['timing']['first_use_flag_s']:
        warn.append(f'cand: Triton autotuning/compilation inside a timed window '
                    f'({perf["first_use_s"]["cand"]} s): the priming run did not reach it')
    if own > CFG['timing']['spread_band']:
        warn.append(f"the candidate's own windows differ by {own:.1%} (> {CFG['timing']['spread_band']:.0%}): "
                    f'something else was on the machine while it was timed, and that window pulls the mean down; '
                    f'time it again on a quiet machine before adopting it')
    if gpu_uuids() and ft.get('gpu_uuids') and gpu_uuids() != ft['gpu_uuids']:
        if os.environ.get('RBB_FROZEN_ANY_GPU') != '1':
            return 'INVALID', dict(perf, reason='the frozen reference was measured on other GPUs'), c
        perf['reference_other_gpus'] = True   # one reference per task for every lane (owner, 11-16)
    if warn:
        perf['warnings'] = warn
    bar = max(CFG['benefit']['min_speedup'], 1 + 2 * spread)
    perf['benefit'] = {'speedup': perf['speedup'], 'speedup_bar': round(bar, 4), 'passed': perf['speedup'] > bar}
    cm = [v for v in c['mismatch_steps'].values() if v is not None]
    perf['mismatch_full_loop'] = mm_verdict(statistics.fmean(cm) if cm else None, ft.get('mismatch_reference'))
    return ('ok' if perf['benefit']['passed'] else 'UNAVAILABLE'), perf, c


def benefit_final(cand, out, cand_env):
    """Gate 1 of the final evaluation: the candidate's full loop against this lane's frozen reference
    (benefit_frozen), or, when there is none -- the formulator measuring a lane for the first time -- the fresh
    reference A -> candidate -> reference B that freezes it."""
    ft = (frozen('timing_ref.json') or {}).get('full') or {}
    if ft.get('windows') and os.environ.get('RBB_FROZEN_REFERENCE') != 'off':
        log(f'frozen reference of this lane: {ft["windows"]} s/step, spread {float(ft.get("spread") or 0):.2%}, '
            f'measured {ft.get("measured")}; timing the candidate only')
        return benefit_frozen(cand, out, cand_env, ft)
    ref_src = check_reference()
    with scratch_dir('sg-ref-') as tmp:
        ref_tree = materialize(ref_src, Path(tmp) / 'reference')
        # one untimed priming run of the reference: reference A and B start from copies of its caches
        prime, primed = prime_caches(ref_tree, out, 'ref', False, None, Path(tmp) / 'prime_caches')
        if primed is None:
            ref_a = dict(prime, reason=f'the reference\'s priming run failed: {prime.get("reason")}')
        else:
            ref_a = time_side('full', ref_tree, out, 'ref_a', candidate=False, primed=primed)
            ref_a['prime'] = {k: prime.get(k) for k in ('status', 'wall_s', 'out')}
        c = time_side('full', cand, out, 'cand', candidate=True, cand_env=cand_env) \
            if ref_a['status'] == 'ok' else None
        ref_b = time_side('full', ref_tree, out, 'ref_b', candidate=False, primed=primed) \
            if c and c['status'] == 'ok' else None
    runs = {'ref_a': strip(ref_a), 'cand': strip(c), 'ref_b': strip(ref_b)}
    bad = {k: v for k, v in runs.items() if v is None or v['status'] != 'ok'}
    if bad:
        status = 'INVALID' if any(v and v['status'] in ('INVALID', 'TIMEOUT') and k != 'cand'
                                  for k, v in bad.items()) else 'ERROR'
        if (runs.get('cand') or {}).get('status') == 'TIMEOUT':
            status = 'ERROR'
        reason = next((v.get('reason') for v in bad.values() if v and v.get('reason')), 'a timing run failed')
        return status, {'speedup': None, 'runs': runs, 'reason': reason}, c
    A, B, C = ref_a['windows'], ref_b['windows'], c['windows']
    ratios = [min(a, b) / x for a, b, x in zip(A, B, C)]
    spread = max(abs(a - b) / min(a, b) for a, b in zip(A, B))
    perf = {'reference_a_s_per_step': A, 'reference_b_s_per_step': B, 'candidate_s_per_step': C,
            'window_ratios': [round(x, 4) for x in ratios], 'speedup': round(sum(ratios) / len(ratios), 4),
            'reference_spread': round(spread, 4), 'runs': runs, 'host': socket.gethostname(),
            'reference_s_per_step': round(statistics.fmean(min(a, b) for a, b in zip(A, B)), 3),
            'candidate_mean_s_per_step': round(statistics.fmean(C), 3),
            'first_use_s': {k: (v.get('first_use') or {}).get('seconds') for k, v in
                            (('ref_a', ref_a), ('cand', c), ('ref_b', ref_b))}}
    late = [k for k in ('ref_a', 'ref_b') if max(perf['first_use_s'][k] or [0]) > CFG['timing']['first_use_flag_s']]
    if late:
        perf['warnings'] = [f'{k}: Triton autotuning/compilation inside a timed window ({perf["first_use_s"][k]} s): '
                            f'the priming run did not reach it' for k in late]
    if spread > CFG['timing']['spread_band']:
        perf['problems'] = [f'the two reference runs differ by {spread:.1%} in a window (> '
                            f'{CFG["timing"]["spread_band"]:.0%}): the host moved between them; measure again later']
        return 'INVALID', perf, c
    bar = max(CFG['benefit']['min_speedup'], 1 + 2 * spread)
    perf['benefit'] = {'speedup': perf['speedup'], 'speedup_bar': round(bar, 4), 'passed': perf['speedup'] > bar}
    # the full loop's own train-inference mismatch, candidate against both reference runs
    cm = [v for v in c['mismatch_steps'].values() if v is not None]
    rm = [v for x in (ref_a, ref_b) for v in x['mismatch_steps'].values() if v is not None]
    perf['mismatch_full_loop'] = mm_verdict(statistics.fmean(cm) if cm else None, statistics.fmean(rm) if rm else None)
    return ('ok' if perf['benefit']['passed'] else 'UNAVAILABLE'), perf, c


# ---------------------------------------------------------------- rollout side

def load_dumps(d, ids):
    import torch
    return {i: torch.load(Path(d) / f'{i}.pt', weights_only=False)['samples'] for i in ids}


def rollout_exact(cand_dir, ref_dir, ids):
    """Exact items of the candidate's rollout dumps against the frozen reference rollout (forced lengths), the
    teacher's answer included."""
    import numpy as np
    lengths = {int(k): v for k, v in json.loads(Path(FIX['lengths']).read_text()).items()}
    missing = [i for i in ids if not (Path(cand_dir) / f'{i}.pt').exists()]
    if missing:
        return [f'no dump of rollouts {missing}']
    c, r = load_dumps(cand_dir, ids), load_dumps(ref_dir, ids)
    bad = []
    for i in ids:
        if len(c[i]) != len(r[i]):
            bad.append(f'rollout {i}: {len(c[i])} samples vs {len(r[i])}')
            continue
        for a, b in zip(sorted(c[i], key=lambda s: s['index']), sorted(r[i], key=lambda s: s['index'])):
            for key in ('index', 'group_index', 'prompt', 'label'):
                if a.get(key) != b.get(key):
                    bad.append(f'rollout {i} sample {b["index"]}: {key} differs')
            n = a.get('response_length')
            if n != lengths.get(a.get('index')):
                bad.append(f'rollout {i} sample {a.get("index")}: length {n} != forced {lengths.get(a.get("index"))}')
                continue
            lp = a.get('rollout_log_probs') or []
            if len(lp) != n or not all(math.isfinite(v) for v in lp):
                bad.append(f'rollout {i} sample {a["index"]}: rollout_log_probs missing or not finite')
            if list(a.get('tokens') or [])[:-n or None] != list(b.get('tokens') or [])[:-n or None]:
                bad.append(f'rollout {i} sample {a["index"]}: prompt tokens differ')
            t = teacher_from_answer(a)
            if t is None or len(t) != n or not np.all(np.isfinite(t)):
                bad.append(f'rollout {i} sample {a["index"]}: the teacher\'s answer (sample.reward, sglang /generate '
                           f'meta_info.input_token_logprobs) has no finite log-probability for each of the {n} response '
                           f'tokens')
            if len(bad) > 20:
                return bad
    return bad


def score(dump_dir, ids, out_npz, log_path):
    """Token forcing with the stock (reference) sglang on the judge GPUs, under the GPU lock."""
    ref = reference_trees()
    with scratch_dir('sg-score-') as tmp:
        env = base_env(tmp, None, candidate=False)
        env.update(PYTHONPATH=f'{ref["sglang"]}/python:{ROOT}', SLIME_OPD_BENCH_RUN=uuid.uuid4().hex[:12])
        cmd = [str(PY), '-m', 'bench.score', '--dumps', str(dump_dir), '--ids', *map(str, ids), '--model',
               CFG['assets']['model_hf'], '--out', str(out_npz), '--topk', '0']
        with GpuLock('token forcing'):
            _kill_marked()
            _, problem = wait_idle('token forcing')
            if problem:
                return 'INVALID', problem
            with open(log_path, 'w') as lf:
                p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT,
                                     start_new_session=True)
                try:
                    rc = p.wait(timeout=TIMEOUT['score'])
                except subprocess.TimeoutExpired:
                    rc = 'TIMEOUT'
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            _kill_marked(env[MARK])
    return ('ok', None) if rc == 0 else ('ERROR', f'scorer exited {rc}; see {log_path}')


def tf_stats(npz):
    import numpy as np
    z = np.load(npz)
    d = z['reported'] - z['stock']
    ad = np.abs(d)
    ad = np.where(np.isfinite(ad), ad, np.inf)
    per, o = {}, 0
    for idx, n in zip(z['index'], z['lens']):
        per[int(idx)] = float(np.mean(z['stock'][o:o + n]))
        o += n
    return {'n_tokens': int(len(d)), 'mean_abs': float(ad.mean()), 'abs': ad, 'per_sample_stock_mean': per,
            'quantiles': {str(q): float(np.quantile(ad, q)) for q in (0.5, 0.99, 0.999, 0.9999)},
            'max_abs': float(ad.max())}


def binom_sf(k, n, p):
    """P(X >= k), X ~ Binomial(n, p). Exact: the event counts have a mean of a few (n p ~ 3 at 2e6 tokens), where the
    normal approximation is far off in the tail (k=9, mean 3: 7.5e-4 against the exact 3.8e-3). Normal approximation
    only for n p above 1000, where the exact sum would underflow its first term."""
    if k <= 0:
        return 1.0
    mu = n * p
    if mu > 1000:
        return 0.5 * math.erfc(((k - 0.5 - mu) / math.sqrt(max(mu * (1 - p), 1e-12))) / math.sqrt(2))
    pmf, cdf = math.exp(n * math.log1p(-p)), 0.0
    for i in range(min(k, n + 1)):
        cdf += pmf
        pmf *= (n - i) / (i + 1) * p / (1 - p)
    return max(1.0 - cdf, 0.0)


def rollout_verdict(npz, ref):
    rules = CFG['rollout_rules']
    c = tf_stats(npz)
    rt = ref['token_forcing']
    out = {'candidate': {k: v for k, v in c.items() if k not in ('abs', 'per_sample_stock_mean')},
           'reference': {k: rt[k] for k in ('n_tokens', 'mean_abs', 'quantiles', 'max_abs', 'tau', 'event_rate')}}
    fails = []
    ratio = c['mean_abs'] / rt['mean_abs']
    out['mean_abs_ratio'] = round(ratio, 4)
    if not ratio <= rules['mean_abs_ratio_max']:
        fails.append(f'rate rule: mean |dlogp| {c["mean_abs"]:.5f} is {ratio:.2f}x the reference\'s '
                     f'({rt["mean_abs"]:.5f}; limit {rules["mean_abs_ratio_max"]}x)')
    tau, p0 = rt['tau'], max(rt['event_rate'], 1.0 / rt['n_tokens'])
    k = int((c['abs'] > tau).sum())
    pval = binom_sf(k, c['n_tokens'], p0)
    out['events'] = {'tau': tau, 'count': k, 'rate': k / c['n_tokens'], 'reference_rate': p0, 'p_value': pval}
    if pval < rules['event_p'] and k / c['n_tokens'] > rules['event_factor'] * p0:
        fails.append(f'event rule: {k} tokens with |dlogp| > {tau:.3f} (rate {k / c["n_tokens"]:.2e} vs reference '
                     f'{p0:.2e}, p={pval:.1e})')
    ref_per = {int(i): v for i, v in rt['per_sample_stock_mean'].items()}
    pairs = [c['per_sample_stock_mean'][i] - ref_per[i] for i in c['per_sample_stock_mean'] if i in ref_per]
    m, sd = statistics.fmean(pairs), statistics.stdev(pairs)
    t = m / (sd / math.sqrt(len(pairs))) if sd > 0 else (0.0 if m == 0 else math.inf)
    out['sampler'] = {'pairs': len(pairs), 'mean_diff': m, 't': t, 't_max': rules['sampler_t_max']}
    if not abs(t) <= rules['sampler_t_max']:
        fails.append(f'sampler check: stock log-prob of sampled tokens shifted by {m:+.4f} per token (t={t:.1f})')
    out['status'] = 'INCORRECT' if fails else 'PASS'
    out['failures'] = fails
    return out


def mm_verdict(val, ref_val):
    if val is None or ref_val is None:
        return {'status': 'ERROR', 'reason': 'train-inference mismatch not logged', 'candidate': val,
                'reference': ref_val}
    r = val / ref_val
    return {'candidate': val, 'reference': ref_val, 'ratio': round(r, 4), 'max_ratio': CFG['mismatch_ratio_max'],
            'status': 'PASS' if r <= CFG['mismatch_ratio_max'] else 'INCORRECT'}


def mismatch_on(dump_dir, out, label):
    """slime's train_rollout_logprob_abs_diff of the reference Megatron on rollout dumps (lr 0, first step)."""
    ref_src = check_reference()
    with scratch_dir('sg-ref-') as tmp:
        tr = materialize(ref_src, Path(tmp) / 'reference')
        r = run_slime(tr, 'train', Path(out) / f'mismatch_{label}', 1, f'mismatch {label}', False,
                      LOAD_DUMPS=dump_dir, LR='0')
    vals = [d.get('train/train_rollout_logprob_abs_diff') for _, d in sorted(train_metrics(r['log']).items())]
    vals = [v for v in vals if v is not None]
    return dict(strip(r), values=vals, mean=statistics.fmean(vals) if vals else None)


# ---------------------------------------------------------------- the teacher

def teacher_from_answer(sample):
    """The teacher log-probabilities of a dumped sample's response tokens from its reward (the teacher server's
    /generate answer, as the reference's post-processing reads it: input_token_logprobs[1:], last response_length)."""
    import numpy as np
    rew, n = sample.get('reward'), sample.get('response_length')
    try:
        lp = rew['meta_info']['input_token_logprobs'][1:]
        vals = np.array([x[0] if x[0] is not None else np.nan for x in lp], dtype=np.float64)
    except (TypeError, KeyError, IndexError):
        return None
    return vals[-n:] if n and len(vals) >= n else None


def teacher_of_dumps(dump_dir, ids):
    """{(rollout id, sample index): teacher log-probabilities} of the samples of rollout dumps."""
    out = {}
    for i, samples in load_dumps(dump_dir, ids).items():
        for smp in samples:
            out[(i, int(smp['index']))] = teacher_from_answer(smp)
    return out


def teacher_npz(path):
    """{(rollout id, sample index): values} of a teacher replay (bench/teacher_replay.py)."""
    import numpy as np
    z = np.load(path)
    out, o = {}, 0
    for rid, idx, n in zip(z['rollout_ids'], z['index'], z['lens']):
        out[(int(rid), int(idx))] = z['values'][o:o + n]
        o += n
    return out


def teacher_diffs(cand, stock):
    """Per sample: the token differences candidate - stock (same tokens), with exact-item problems."""
    import numpy as np
    bad, per = [], {}
    if set(cand) != set(stock):
        miss = sorted(set(stock) - set(cand))[:5]
        extra = sorted(set(cand) - set(stock))[:5]
        return {}, [f'teacher log-probabilities for samples {sorted(cand)[:3]}...: missing {miss}, unexpected {extra}']
    for key in sorted(stock):
        c, r = cand[key], stock[key]
        if c is None or len(c) != len(r):
            bad.append(f'sample {key}: {None if c is None else len(c)} teacher log-probabilities, expected {len(r)}')
        elif not np.all(np.isfinite(c)):
            bad.append(f'sample {key}: teacher log-probabilities not finite')
        else:
            per[key] = np.asarray(c, dtype=np.float64) - np.asarray(r, dtype=np.float64)
        if len(bad) > 20:
            break
    return per, bad


def teacher_stats(per):
    """Token and per-sample statistics of teacher differences."""
    import numpy as np
    d = np.concatenate(list(per.values())) if per else np.zeros(0)
    ad = np.abs(d)
    return {'n_tokens': int(len(d)), 'mean_abs': float(ad.mean()) if len(d) else 0.0,
            'mean_signed': float(d.mean()) if len(d) else 0.0, 'abs': ad,
            'per_sample_signed': {':'.join(map(str, k)): float(v.mean()) for k, v in per.items() if len(v)},
            'quantiles': {str(q): float(np.quantile(ad, q)) for q in (0.5, 0.99, 0.999, 0.9999)} if len(d) else {},
            'max_abs': float(ad.max()) if len(d) else 0.0}


def teacher_baseline(per):
    """Frozen baseline of a reference comparison: the floors of bench.json teacher_rules applied. `per` is one
    comparison {(rollout id, sample index): d} or several pooled ({(name, rollout id, sample index): d}: every token of
    every comparison counts once in the quantile, the event rate and the mean)."""
    import numpy as np
    rules = CFG['teacher_rules']
    st = teacher_stats(per)
    tau = max(rules['event_factor'] * (float(np.quantile(st['abs'], rules['event_quantile'])) if st['n_tokens'] else 0.0),
              rules['tau_floor'])
    base = {k: st[k] for k in ('n_tokens', 'mean_abs', 'mean_signed', 'quantiles', 'max_abs')}
    events = int((st['abs'] > tau).sum())
    base.update(tau=tau, events=events, event_rate=float(events) / max(st['n_tokens'], 1),
                mean_abs_bound=max(st['mean_abs'], rules['mean_abs_floor']))
    return base


def teacher_verdict(per, base):
    """Rate, event and bias rules of teacher differences against a frozen baseline (teacher_baseline)."""
    rules = CFG['teacher_rules']
    c = teacher_stats(per)
    out = {'candidate': {k: v for k, v in c.items() if k not in ('abs', 'per_sample_signed')},
           'reference': {k: base[k] for k in ('n_tokens', 'mean_abs', 'mean_abs_bound', 'tau', 'event_rate')}}
    fails = []
    ratio = c['mean_abs'] / base['mean_abs_bound']
    out['mean_abs_ratio'] = round(ratio, 4)
    if not ratio <= rules['mean_abs_ratio_max']:
        fails.append(f'teacher rate rule: mean |dlogp| {c["mean_abs"]:.5f} is {ratio:.2f}x the reference\'s '
                     f'{base["mean_abs_bound"]:.5f} (limit {rules["mean_abs_ratio_max"]}x)')
    tau, p0 = base['tau'], max(base['event_rate'], 1.0 / max(base['n_tokens'], 1))
    k = int((c['abs'] > tau).sum())
    pval = binom_sf(k, c['n_tokens'], p0)
    out['events'] = {'tau': tau, 'count': k, 'rate': k / max(c['n_tokens'], 1), 'reference_rate': p0, 'p_value': pval}
    if pval < rules['event_p'] and k / max(c['n_tokens'], 1) > rules['event_factor'] * p0:
        fails.append(f'teacher event rule: {k} tokens with |dlogp| > {tau:.3f} (rate {k / c["n_tokens"]:.2e} vs '
                     f'reference {p0:.2e}, p={pval:.1e})')
    xs = list(c['per_sample_signed'].values())
    if len(xs) >= 2:
        m, sd = statistics.fmean(xs), statistics.stdev(xs)
        t = m / (sd / math.sqrt(len(xs))) if sd > 0 else (0.0 if m == 0 else math.inf)
        out['bias'] = {'samples': len(xs), 'mean_signed': m, 't': t, 't_max': rules['bias_t_max'],
                       'floor': rules['mean_abs_floor']}
        if abs(t) > rules['bias_t_max'] and abs(m) > rules['mean_abs_floor']:
            fails.append(f'teacher bias rule: teacher log-probabilities shifted by {m:+.4f} per token on average '
                         f'(paired t={t:.1f} over {len(xs)} samples)')
    out['status'] = 'INCORRECT' if fails else 'PASS'
    out['failures'] = fails
    return out


def teacher_replay_run(tr, dumps, out, label, candidate, cand_env=None, num=None, concurrency=None, server_args=None):
    """One teacher replay of rollout dumps 0..num-1 (MODE=teacher): clock, layout and the replayed values. For the
    reference's re-batchings (freeze-teacher): `concurrency` requests at a time, `server_args` appended to the teacher
    server's arguments (the same variable as the candidate switch OPD_TEACHER_SERVER_ARGS)."""
    n = num or CFG['timing']['teacher']['num_rollout']
    launch = {'LOAD_DUMPS': str(dumps)}
    if concurrency:
        launch['SLIME_OPD_BENCH_REPLAY_CONCURRENCY'] = str(concurrency)
    if server_args:
        launch['OPD_TEACHER_SERVER_ARGS'] = server_args
    r = run_slime(tr, 'teacher', Path(out) / f'teacher_{label}', n, f'teacher {label}', candidate, cand_env,
                  timeout=TIMEOUT['teacher'], **launch)
    if r['status'] != 'ok':
        return r
    w, problem = windows('teacher', r['judge'], n)
    if problem:
        return dict(r, status='ERROR', reason=problem)
    lay = layout_of(r['judge'], 'teacher')
    npz = Path(r['judge']) / 'teacher_logprobs.npz'
    if not npz.exists():
        return dict(r, status='ERROR', reason='the teacher replay wrote no log-probabilities')
    r.update(windows=[round(x, 3) for x in w], mean=round(statistics.fmean(w), 3), layout=lay,
             layout_problems=layout_problems(lay), values=str(npz))
    return r


def judge_rollout(cand, out, cand_env, run=None):
    """Gate R on one rollout run of the candidate (the timing run of the agent's check when there is one)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)  # the agent's non-quick check passes its timing run; nothing else creates <out>/rollout
    ref = frozen('rollout_ref.json')
    if run is None:
        run = time_side('rollout', cand, out, 'cand', candidate=True, cand_env=cand_env)
    res = {'run': strip(run)}
    if run['status'] != 'ok':
        return dict(res, status=run['status'] if run['status'] in ('INVALID',) else 'ERROR',
                    candidate_failed=run['status'] in ('ERROR', 'TIMEOUT'), reason=run.get('reason'))
    if run['layout_problems']:
        return dict(res, status='INCORRECT', stage='layout', problems=run['layout_problems'])
    dumps = Path(run['judge']) / 'rollout_data'
    ids = list(range(CFG['timing']['rollout']['num_rollout']))
    bad = rollout_exact(dumps, ref['dumps'], ids)
    res['exact'] = bad or 'ok'
    if bad:
        return dict(res, status='INCORRECT', stage='exact_items', problems=bad)
    st, why = score(dumps, ids, out / 'scores.npz', out / 'score.log')
    if st != 'ok':
        return dict(res, status=st, reason=why, candidate_failed=st == 'ERROR')
    res['token_forcing'] = rollout_verdict(out / 'scores.npz', ref)
    if res['token_forcing']['status'] != 'PASS':
        return dict(res, status='INCORRECT', stage='token_forcing', problems=res['token_forcing']['failures'])
    # the teacher log-probabilities the rollout carries, against the stock teacher replayed on the same tokens
    ref_src = check_reference()
    with scratch_dir('sg-ref-') as tmp:
        rt = materialize(ref_src, Path(tmp) / 'reference')
        rep = teacher_replay_run(rt, dumps, out, 'stock_on_cand', False, num=len(ids))
    if rep['status'] != 'ok':
        return dict(res, status='INVALID' if rep['status'] == 'INVALID' else 'ERROR', teacher_run=strip(rep),
                    reason=f'stock teacher replay of the candidate rollout: {rep.get("reason")}')
    per, bad = teacher_diffs(teacher_of_dumps(dumps, ids), teacher_npz(rep['values']))
    if bad:
        return dict(res, status='INCORRECT', stage='rollout_teacher', problems=bad)
    res['rollout_teacher'] = teacher_verdict(per, ref['teacher'])
    if res['rollout_teacher']['status'] != 'PASS':
        return dict(res, status='INCORRECT', stage='rollout_teacher', problems=res['rollout_teacher']['failures'])
    mm = mismatch_on(dumps, out, 'rollout')
    if mm['status'] != 'ok':
        return dict(res, status='ERROR' if mm['status'] != 'INVALID' else 'INVALID', mismatch_run=mm,
                    candidate_failed=mm['status'] == 'ERROR', reason=mm.get('reason'))
    res['mismatch'] = mm_verdict(mm['mean'], ref['mismatch'])
    st = res['mismatch']['status']
    return dict(res, status='PASS' if st == 'PASS' else st, stage='mismatch' if st != 'PASS' else 'rollout',
                candidate_failed=st == 'ERROR')


def judge_teacher(cand, out, cand_env, run=None):
    """Gate S: the candidate's teacher replay of fixture rollouts 0..n-1 against the frozen stock replay."""
    out = Path(out)
    ref = frozen('teacher_ref.json')
    if run is None:
        run = teacher_replay_run(cand, FIX['dumps'], out, 'cand', True, cand_env)
    res = {'run': strip(run)}
    if run['status'] != 'ok':
        return dict(res, status='INVALID' if run['status'] == 'INVALID' else 'ERROR',
                    candidate_failed=run['status'] in ('ERROR', 'TIMEOUT'), reason=run.get('reason'))
    if run['layout_problems']:
        return dict(res, status='INCORRECT', stage='layout', problems=run['layout_problems'])
    per, bad = teacher_diffs(teacher_npz(run['values']), teacher_npz(TASK / 'teacher_ref.npz'))
    res['exact'] = bad or 'ok'
    if bad:
        return dict(res, status='INCORRECT', stage='exact_items', problems=bad)
    res['teacher'] = teacher_verdict(per, ref['baseline'])
    st = res['teacher']['status']
    return dict(res, status=st, stage='teacher' if st == 'PASS' else 'teacher_logprobs',
                problems=res['teacher']['failures'])


# ---------------------------------------------------------------- train side

def record_run(tr, out, label, start, num, candidate, cand_env=None, deterministic=True, lr=None):
    """Train-only run on fixture dumps [start, start + num) with the parameter records (deterministic mode)."""
    r = run_slime(tr, 'train', Path(out) / f'record_{label}', start + num, f'record {label}', candidate, cand_env,
                  deterministic=deterministic, timeout=TIMEOUT['record'], LOAD_DUMPS=FIX['dumps'],
                  START_ROLLOUT=start, SLIME_OPD_BENCH_RECORD='1', LR=lr)
    rec = load_record(r)
    rec['run'] = strip(r)
    if r['status'] == 'ok':
        rec['layout_problems'] = layout_problems(layout_of(r['judge'], 'train'))
    return rec


def load_record(r):
    import numpy as np
    rec = Path(r['judge']) / 'records'
    steps, sketch = {}, {}
    for f in sorted(rec.glob('r*.s*.rank*.json')):
        d = json.loads(f.read_text())
        steps.setdefault(d['rollout_id'], {})[d['rank']] = d
        sk = f.with_name(f.name.replace('.json', '.sketch.npy'))
        if sk.exists():
            sketch.setdefault(d['rollout_id'], {})[d['rank']] = np.load(sk)
    m = train_metrics(r['log'])
    losses = {i: {k: d.get(k) for k in ('train/loss', 'train/grad_norm', 'train/train_rollout_logprob_abs_diff')}
              for i, d in m.items()}
    return {'status': r['status'], 'records': steps, 'sketch': sketch, 'losses': losses}


def endpoints(run, start, num):
    last = start + num - 1
    recs = run['records'].get(last, {})
    upd = math.sqrt(sum(r['update_sq'] for r in recs.values())) if recs else None
    vsq = math.sqrt(sum(r.get('exp_avg_sq_sq', 0.0) for r in recs.values())) if recs else None
    ls = [v['train/loss'] for _, v in sorted(run['losses'].items()) if v.get('train/loss') is not None]
    # the first step's gradient norm (the frozen state, before any update): later steps' gradient norms are
    # heavy-tailed run to run (the unchanged reference once gave 2.09 instead of ~1.0 at step 1; see docs/slime_opd.md)
    gn = [v['train/grad_norm'] for _, v in sorted(run['losses'].items()) if v.get('train/grad_norm') is not None]
    return {'update_norm': upd, 'exp_avg_sq_norm': vsq, 'loss_mean': statistics.fmean(ls) if ls else None,
            'grad_norm_first': gn[0] if gn else None}


def update_sketch(run, start, num):
    import numpy as np
    sk = run['sketch'].get(start + num - 1, {})
    return np.concatenate([sk[r] for r in sorted(sk)]) if sk else None


def cosine(a, b):
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compare_records(cand, ref):
    """Exact items (problems) and the bitwise comparison (differing tensors) of two records."""
    exact, diff = [], []
    if set(cand['records']) != set(ref['records']):
        return [f'steps recorded {sorted(cand["records"])} vs {sorted(ref["records"])}'], diff
    for sid, ranks in ref['records'].items():
        c = cand['records'][sid]
        if set(c) != set(ranks):
            exact.append(f'step {sid}: ranks {sorted(c)} vs {sorted(ranks)}')
            continue
        for rk, r in ranks.items():
            a = c[rk]
            if a['lr'] != r['lr']:
                exact.append(f'step {sid} rank {rk}: lr {a["lr"]} vs {r["lr"]}')
            if a['shape'] != r['shape']:
                exact.append(f'step {sid} rank {rk}: parameter or optimizer-state names/shapes differ')
                continue
            if a['dtype'] != r['dtype']:
                bad = [n for n in r['dtype'] if a['dtype'].get(n) != r['dtype'][n]][:3]
                exact.append(f'step {sid} rank {rk}: precision change (not accepted): dtypes differ for {bad}')
            for n, h in r['sha'].items():
                if a['sha'].get(n) != h:
                    diff.append(f'step {sid} rank {rk} {n}')
    return exact, diff


def judge_train(cand, out, cand_env, timing_run=None, n1=False):
    out = Path(out)
    ref = frozen('train_ref.json')
    tr = CFG['train_record']
    run = record_run(cand, out, 'cand', tr['start'], tr['num_rollout'], True, cand_env)
    res = {'run': run['run']}
    if run['status'] != 'ok':
        return dict(res, status='INVALID' if run['status'] == 'INVALID' else 'ERROR',
                    candidate_failed=run['status'] != 'INVALID', reason=run['run'].get('reason'))
    if run['layout_problems']:
        return dict(res, status='INCORRECT', stage='layout', problems=run['layout_problems'])
    refrun = {'records': {int(k): {int(r): v for r, v in d.items()} for k, d in ref['records'].items()}}
    exact, diff = compare_records(run, refrun)
    res['exact'] = exact or 'ok'
    if exact:
        return dict(res, status='INCORRECT', stage='exact_items', problems=exact)
    if not diff and not n1:
        res.update(bitwise=True, measured_tier='N0', gate='PASS')
    else:
        res.update(bitwise=not diff, differing=len(diff), first_differing=diff[:3],
                   **({'note': 'check --n1: the statistical gate decides (no bitwise verdict)'} if n1 else {}))
        if 'trajectory' not in ref:
            # smoke preparation: no data-set sigma frozen, no statistical gate
            return dict(res, status='NOT_BITWISE', stage='record',
                        problems=['not bitwise identical; no statistical gate in a smoke preparation'])
        ep = endpoints(run, tr['start'], tr['num_rollout'])
        tj = ref['trajectory']
        rows, fails = {}, []
        for k, v in ep.items():
            rv, sig, tol = tj['reference'][k], tj['sigma'][k], tj['tolerance'][k]
            d = None if v is None else v - rv
            ok = v is not None and abs(d) <= tr['sigma_k'] * sig and abs(d) <= tol
            rows[k] = {'candidate': v, 'reference': rv, 'd': d, 'tolerance_paired_noise': tol,
                       'sigma_data_sets': sig, 'within': ok}
            if not ok:
                fails.append(k)
        import numpy as np
        sk = update_sketch(run, tr['start'], tr['num_rollout'])
        cos = cosine(sk, np.load(TASK / 'train_ref_sketch.npy')) if sk is not None else None
        rows['update_direction_cos'] = {'candidate': cos, 'min': ref['direction']['cos_min'],
                                        'reference_noise_cos': ref['direction']['cos_noise'],
                                        'within': cos is not None and cos >= ref['direction']['cos_min']}
        if not rows['update_direction_cos']['within']:
            fails.append('update_direction_cos')
        res.update(statistical=rows, measured_tier='N1' if not fails else None,
                   gate='INCORRECT' if fails else 'PASS')
        if fails:
            return dict(res, status='INCORRECT', stage='statistical',
                        problems=[f'{k} beyond the reference\'s noise' for k in fails])
    # the candidate's Megatron on the frozen rollout, first step (before any update)
    mval = run['losses'].get(tr['start'], {}).get('train/train_rollout_logprob_abs_diff')
    res['mismatch'] = mm_verdict(mval, ref['mismatch'])
    st = res['mismatch']['status']
    return dict(res, status=st, stage='train' if st == 'PASS' else 'mismatch', candidate_failed=st == 'ERROR')


# ---------------------------------------------------------------- inputs

def verify_inputs(when):
    spec = (frozen('formulation.json') or {}).get('inputs', {})
    t0, bad = time.time(), {}

    def one(item):
        rel, want = item
        p = Path(rel) if os.path.isabs(rel) else ROOT / rel
        return rel, want['value'], (sha256_file(p) if p.exists() else None)
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        for rel, want, got in ex.map(one, spec.items()):
            if got != want:
                bad[rel] = {'want': want, 'got': got}
    log(f'inputs {when}: {len(spec) - len(bad)}/{len(spec)} match the formulation ({time.time() - t0:.0f}s)')
    return {'when': when, 'ok': not bad, 'mismatches': dict(list(bad.items())[:10]), 'checked': len(spec),
            'seconds': round(time.time() - t0, 1)}


def freeze_inputs(paths):
    out = {}
    for root in paths:
        root = Path(root) if os.path.isabs(root) else ROOT / root
        files = [root] if root.is_file() else sorted(f for f in root.rglob('*') if f.is_file()
                                                      and not {'__pycache__', '.cache', '.locks'} & set(f.parts))
        for f in files:
            out[str(f)] = {'kind': 'sha256', 'value': None}
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        for f, h in zip(list(out), ex.map(sha256_file, list(out))):
            out[f]['value'] = h
    return out


def candidate_environment(config_file):
    if not config_file:
        return {}
    data = json.loads(Path(config_file).read_text())
    extra = sorted(set(data) - {'environment'})
    if extra:
        raise SystemExit(f'{config_file}: only an "environment" mapping is accepted, found {extra}')
    env = {str(k): str(v) for k, v in (data.get('environment') or {}).items()}
    bad = sorted(k for k in env if k in RESERVED or k.startswith(PREFIX))
    if bad:
        raise SystemExit(f'{config_file}: {bad} are managed by the bench and cannot be candidate switches')
    return env


FINAL_MARKER = Path('/run/researchboost/final-evaluation')


def final_allowed():
    """`check --final` is the final evaluation's: root (the preparation) or the verifier, which writes FINAL_MARKER
    as root in a root-owned directory before it runs the judge as the agent user."""
    if os.geteuid() == 0:
        return True
    try:
        st, parent = FINAL_MARKER.stat(), FINAL_MARKER.parent.stat()
    except OSError:
        return False
    return st.st_uid == 0 and parent.st_uid == 0 and not parent.st_mode & 0o022


# ---------------------------------------------------------------- verdict

def evaluate(cand, src_trees, out, cand_env, sides, quick=False, final=False, n1=False):
    """Returns (status, correctness, performance)."""
    s0 = structure(cand)
    if s0:
        return 'ERROR', {'passed': False, 'stage': 'structure', 'note': '; '.join(s0)}, None
    performance, side_runs, full_cand = None, {}, None
    if final:
        status, performance, full_cand = benefit_final(cand, out, cand_env)
        if status in ('ERROR', 'INVALID'):
            note = performance.get('reason') or '; '.join(performance.get('problems') or []) or 'timing failed'
            return status, {'passed': False, 'stage': 'benefit', 'note': note}, performance
        if full_cand['layout_problems']:
            return 'INCORRECT', {'passed': False, 'stage': 'layout', 'verdict': 'layout',
                                 'note': '; '.join(full_cand['layout_problems'])}, performance
        if status == 'UNAVAILABLE':
            performance['measured_speedup'] = performance.pop('speedup')
            return 'UNAVAILABLE', {'passed': None, 'stage': 'benefit', 'verdict': 'no_benefit',
                                   'note': 'not faster than the reference: rejected without a precision '
                                           'evaluation'}, performance
        if performance['mismatch_full_loop']['status'] != 'PASS':
            st = performance['mismatch_full_loop']['status']
            return st, {'passed': False, 'stage': 'mismatch', 'verdict': 'mismatch',
                        'candidate_failed': st == 'ERROR',
                        'note': f'full-loop train-inference mismatch: {performance["mismatch_full_loop"]}'}, \
                performance
    elif not quick:
        for s in sorted(sides):
            run = teacher_replay_run(cand, FIX['dumps'], out, 'cand', True, cand_env) if s == 'teacher' else \
                time_side(s, cand, out, 'cand', candidate=True, cand_env=cand_env)
            side_runs[s] = run
            if run['status'] != 'ok':
                st = 'INVALID' if run['status'] == 'INVALID' else 'ERROR'
                return st, {'passed': False, 'stage': 'benefit', 'candidate_failed': st == 'ERROR',
                            'note': run.get('reason')}, {'runs': {s: strip(run)}}
            if run['layout_problems']:
                return 'INCORRECT', {'passed': False, 'stage': 'layout', 'verdict': 'layout',
                                     'note': '; '.join(run['layout_problems'])}, {'runs': {s: strip(run)}}
        status, performance = benefit_agent(side_runs)
        performance['runs'] = {s: strip(r) for s, r in side_runs.items()}
        if status == 'ERROR':
            return status, {'passed': False, 'stage': 'benefit', 'note': performance.get('reason')}, performance
        if status == 'UNAVAILABLE':
            performance['measured_speedup'] = performance.pop('speedup')
            return 'UNAVAILABLE', {'passed': None, 'stage': 'benefit', 'verdict': 'no_benefit',
                                   'note': 'not faster than the reference: rejected without a precision '
                                           'evaluation'}, performance
    correctness = {'sides': sorted(sides)}
    tiers = []
    for s in sorted(sides):
        if s == 'rollout':
            r = judge_rollout(cand, out / 'rollout', cand_env, run=side_runs.get('rollout'))
            tiers.append('N1' if r['status'] == 'PASS' else None)
        elif s == 'teacher':
            r = judge_teacher(cand, out / 'teacher', cand_env, run=side_runs.get('teacher'))
            tiers.append('N1' if r['status'] == 'PASS' else None)
        else:
            r = judge_train(cand, out / 'train', cand_env, n1=n1)
            tiers.append(r.get('measured_tier') if r['status'] == 'PASS' else None)
        correctness[s] = r
        if r['status'] != 'PASS':
            verdict = {'INCORRECT': 'different'}.get(r['status'])
            return r['status'], dict(correctness, passed=False if r['status'] != 'INVALID' else None, stage=s,
                                     verdict=verdict, candidate_failed=bool(r.get('candidate_failed')),
                                     note=f'{s}: ' + '; '.join(map(str, r.get('problems') or [r.get('reason')]))), \
                performance
    tier = 'N0' if all(t == 'N0' for t in tiers) else 'N1'
    return 'PASS', dict(correctness, passed=True, measured_tier=tier,
                        verdict='bitwise' if tier == 'N0' else 'equivalent'), performance


def guarded(out, fn, tree_src=None):
    out.mkdir(parents=True, exist_ok=True)
    start = verify_inputs('start')
    if not start['ok']:
        return {'status': 'ERROR', 'stage': 'inputs', 'inputs': [start],
                'note': 'a frozen input differs from task/formulation.json; nothing was measured'}
    if tree_src is None:
        try:
            res = fn(None, None)
        except SystemExit as exc:
            res = {'status': 'ERROR', 'note': f'the evaluation stopped: {exc}'}
    else:
        with scratch_dir('sg-src-') as tmp:
            tr = materialize(tree_src, Path(tmp) / 'tree')
            digest = tree_digest(tr)
            try:
                res = fn(tr, tree_src)
            except SystemExit as exc:
                res = {'status': 'ERROR', 'note': f'the evaluation stopped: {exc}'}
            res['source_digest'] = digest
            if tree_digest(tr) != digest:
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
    print(json.dumps(summary))
    return 0 if res.get('status') in ('PASS', 'ok') else 1


def judged_sides(src, side, cand_env=None):
    files = changed_files(src)
    sides, other = sides_of(files, cand_env)
    if side:
        sides.add(side)
    return sides, files, other


def cmd_check(a):
    out = Path(a.out).resolve()
    if a.final and not final_allowed():
        raise SystemExit('check --final is the final evaluation\'s (run by the harness after your time is up) and is '
                         'not available here. `check` runs the same correctness gates; its timing reads one run per '
                         'side against the frozen reference timing instead of the full loop reference -> candidate '
                         '-> reference.')
    cand_env = candidate_environment(a.config)
    src = trees_of(a.tree)

    def run(tr, _):
        sides, files, other = judged_sides(src, a.side, cand_env)
        if a.final:
            sides = set(ALL_SIDES)
        status, correctness, performance = evaluate(tr, src, out, cand_env, sides,
                                                    quick=a.quick or a.skip_timing or a.n1, final=a.final, n1=a.n1)
        return {'status': status, 'tree': str(Path(a.tree).resolve()), 'config': a.config, 'environment': cand_env,
                'quick': a.quick, 'skip_timing': a.skip_timing, 'n1': a.n1, 'final': a.final,
                'judged_sides': sorted(sides),
                'changed_files': files[:200], 'files_outside_a_side': other[:20],
                'correctness': correctness, 'performance': performance}
    return finish(out, 'check.json', guarded(out, run, src))


def cmd_time(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)
    src = trees_of(a.tree)
    is_ref = all(src[k] == reference_trees()[k] for k in REPOS)

    def run(tr, _):
        sides = {a.side} if a.side else judged_sides(src, None, cand_env)[0]
        runs, sp = {}, {}
        for s in sorted(sides):
            r = teacher_replay_run(tr, FIX['dumps'], out, 'tree', not is_ref, cand_env) if s == 'teacher' else \
                time_side(s, tr, out, 'tree', candidate=not is_ref, cand_env=cand_env, dump=False)
            runs[s] = strip(r)
            if r['status'] == 'ok':
                ss = side_speedup(s, r)
                if ss:
                    runs[s]['vs_frozen_reference'] = ss
                    sp[s] = ss['speedup']
        res = {'status': 'ok' if all(r['status'] == 'ok' for r in runs.values()) else 'ERROR', 'runs': runs,
               'side_speedup': sp, 'tree': str(Path(a.tree).resolve()), 'config': a.config, 'environment': cand_env}
        if sp and len(sp) == len(runs):
            res['projected_step_speedup'] = projected(sp)
            res['note'] = ('screening only: one run per side against the frozen reference timing of this host; the '
                           'score is the final evaluation\'s full-loop timing (reference -> candidate -> reference)')
        bad = [p for r in runs.values() for p in (r.get('layout_problems') or [])]
        if bad:
            res['layout_problems'] = bad
        return res
    return finish(out, 'time.json', guarded(out, run, src))


def cmd_profile(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)
    src = trees_of(a.tree)

    def run(tr, _):
        trace = out / 'trace'
        trace.mkdir(parents=True, exist_ok=True)
        os.chmod(trace, 0o777)
        launch = {'PROFILE_DIR': str(trace)}
        if a.side == 'rollout':
            launch['FORCE_LENGTHS'] = FIX['lengths']
            n = 2
        elif a.side == 'teacher':
            launch['LOAD_DUMPS'] = FIX['dumps']
            n = 2
        else:
            # the timed steps' order (timing.train.dump_order) up to the traced step: step 5 = dump 2, run once before
            order = CFG['timing']['train']['dump_order'][:7]
            launch['LOAD_DUMPS'] = replay_dir(out / 'profile_dumps', order)
            launch['PROFILE_STEP'] = '5'
            n = len(order)
        r = run_slime(tr, a.side, out / f'profile_{a.side}', n, f'profile {a.side}', True, cand_env, **launch)
        traces = sorted((p for p in trace.rglob('*') if p.is_file() and re.search(r'\.json(\.gz)?$', p.name)),
                        key=lambda p: p.stat().st_size, reverse=True)
        res = {'status': r['status'], 'run': strip(r), 'traces': [str(t) for t in traces[:8]]}
        if traces:
            subprocess.run([str(PY), str(BENCH / 'kernel_util.py'), str(traces[0]), '--out',
                            str(out / 'kernel_report.md'), '--json', str(out / 'kernel_report.json')], check=False)
            res['kernel_report'] = str(out / 'kernel_report.md')
        elif r['status'] == 'ok':
            res.update(status='ERROR', reason='the profiler wrote no trace')
        res['note'] = 'instrumented: not comparable with `time`'
        return res
    return finish(out, 'profile.json', guarded(out, run, src))


# ---------------------------------------------------------------- formulator commands

def cmd_fixture(a):
    """Formulator: the frozen rollout fixture from one reference rollout-only run with natural lengths."""
    fx = Path(FIX['dir'])
    ref_src = check_reference()
    with scratch_dir('sg-ref-') as tmp:
        tr = materialize(ref_src, Path(tmp) / 'reference')
        r = run_slime(tr, 'rollout', Path(a.out) / 'fixture_run', a.rollouts, 'fixture', False,
                      SLIME_OPD_BENCH_DUMP='1', timeout=TIMEOUT['rollout'] * 6)
    if r['status'] != 'ok':
        raise SystemExit(f'fixture run failed: {r.get("reason")}')
    lay = layout_of(r['judge'], 'rollout')
    if layout_problems(lay):
        raise SystemExit(f'the reference rollout layout is not the recipe\'s: {layout_problems(lay)}')
    engine = {k: lay['sglang']['engines'][0].get(k) for k in SGLANG_KEYS}
    lr = frozen('layout_ref.json') or {}
    lr['sglang_engine'] = engine
    lr['teacher'] = {k: (lay.get('teacher') or {}).get(k) for k in SGLANG_KEYS}
    (TASK / 'layout_ref.json').write_text(json.dumps(lr, indent=1) + '\n')
    if fx.exists():
        shutil.rmtree(fx)
    shutil.copytree(Path(r['judge']) / 'rollout_data', fx / 'rollout_ref')
    import torch
    lengths, seen = {}, set()
    for rid in range(a.rollouts):
        x = torch.load(fx / 'rollout_ref' / f'{rid}.pt', weights_only=False)
        idx = [s['index'] for s in x['samples']]
        if seen & set(idx):
            raise SystemExit(f'sample indices repeat across rollouts at {rid}')
        seen |= set(idx)
        for s in x['samples']:
            lengths[s['index']] = s['response_length']
    (fx / 'lengths.json').write_text(json.dumps(lengths))
    clock = _jsonl(Path(r['judge']) / 'rollout_clock.jsonl')
    print(json.dumps({'status': 'ok', 'rollouts': a.rollouts, 'samples': len(lengths),
                      'tokens': sum(lengths.values()), 'engine': engine, 'wall_s': r['wall_s'],
                      'dump_s': [round(c.get('dump_s', 0), 1) for c in clock], 'cost': cost()}))
    return 0


def cmd_freeze_rollout(a):
    """Formulator: the reference rollout with forced lengths (timing run A of the rollout side, the frozen forced
    dumps), its student token forcing baseline through the stock student engine, the baseline of the teacher
    log-probabilities it carries against the stock teacher replayed on the same tokens, and the mismatch baseline."""
    out = Path(a.out)
    ref_src = check_reference()
    with scratch_dir('sg-ref-') as tmp:
        tr = materialize(ref_src, Path(tmp) / 'reference')
        r = time_side('rollout', tr, out, 'ref_a', candidate=False)
    if r['status'] != 'ok':
        raise SystemExit(f'reference rollout failed: {r.get("reason")}')
    dumps = Path(FIX['dir']) / 'rollout_ref_forced'
    if dumps.exists():
        shutil.rmtree(dumps)
    shutil.copytree(Path(r['judge']) / 'rollout_data', dumps)
    ids = list(range(CFG['timing']['rollout']['num_rollout']))
    st, why = score(dumps, ids, out / 'scores.npz', out / 'score.log')
    if st != 'ok':
        raise SystemExit(f'scorer failed: {why}')
    import numpy as np
    s = tf_stats(out / 'scores.npz')
    rules = CFG['rollout_rules']
    tau = rules['event_factor'] * float(np.quantile(s['abs'], rules['event_quantile']))
    tf = {k: s[k] for k in ('n_tokens', 'mean_abs', 'quantiles', 'max_abs', 'per_sample_stock_mean')}
    tf.update(tau=tau, event_rate=float((s['abs'] > tau).sum()) / s['n_tokens'])
    with scratch_dir('sg-ref-') as tmp:
        tr = materialize(ref_src, Path(tmp) / 'reference')
        rep = teacher_replay_run(tr, dumps, out, 'stock_on_ref', False, num=len(ids))
    if rep['status'] != 'ok' or rep['layout_problems']:
        raise SystemExit(f'stock teacher replay of the reference rollout failed: {rep.get("reason") or rep["layout_problems"]}')
    per, bad = teacher_diffs(teacher_of_dumps(dumps, ids), teacher_npz(rep['values']))
    if bad:
        raise SystemExit(f'the reference rollout\'s teacher answers: {bad[:3]}')
    teacher = teacher_baseline(per)
    mm = mismatch_on(dumps, out, 'reference')
    if mm['mean'] is None:
        raise SystemExit(f'mismatch baseline failed: {mm.get("reason")}')
    res = {'dumps': str(dumps), 'token_forcing': tf, 'teacher': teacher,
           'mismatch': mm['mean'], 'mismatch_values': mm['values'], 'timing_run': strip(r)}
    (TASK / 'rollout_ref.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(json.dumps({'status': 'ok', 'mean_abs': tf['mean_abs'], 'tau': tau, 'event_rate': tf['event_rate'],
                      'teacher': teacher, 'mismatch': mm['mean'], 'windows': r['windows'],
                      'teacher_tail_s': r.get('teacher_tail_s'), 'cost': cost()}, default=str))
    return 0


def cmd_freeze_teacher(a):
    """Formulator: the stock teacher replayed on fixture rollouts 0..n-1: twice with every sample of a rollout at once
    (the frozen values: run A; timing: runs A and B), then once per re-batching of teacher_rules.rebatchings (the same
    unchanged teacher with requests one at a time, another prefill chunking, no overlap scheduling, no radix cache).
    The baseline pools the four comparisons with run A (the stock teacher's own spread) -> task/teacher_ref.json,
    task/teacher_ref.npz.

    Why re-batchings and not repeats: the teacher (Qwen3.5) is a hybrid model whose linear-attention layers carry a
    recurrent state across the prefill chunks of a request. Repeats of the same batching are bitwise identical (and
    requests 8 at a time give the same chunk boundaries: they follow the first-come-first-served token stream), so they
    would freeze the floors, a tolerance no legitimate change of the teacher's batching or kernels can meet: each of
    the four re-batchings moves every token after a request's first chunk boundary (about 5% of tokens by more than
    0.05), and the unchanged teacher was seen doing so on its own once (a scheduling race)."""
    out = Path(a.out)
    ref_src = check_reference()
    runs = {}
    variants = [('ref_a', {}), ('ref_b', {})] + [
        ('rebatch_' + v['name'], {'concurrency': v.get('concurrency'), 'server_args': v.get('server_args')})
        for v in CFG['teacher_rules']['rebatchings']]
    with scratch_dir('sg-ref-') as tmp:
        tr = materialize(ref_src, Path(tmp) / 'reference')
        for name, kw in variants:
            r = teacher_replay_run(tr, FIX['dumps'], out, name, False, **kw)
            if r['status'] != 'ok' or r['layout_problems']:
                raise SystemExit(f'stock teacher replay {name} failed: {r.get("reason") or r["layout_problems"]}')
            runs[name] = r
    A = teacher_npz(runs['ref_a']['values'])
    per_b, bad = teacher_diffs(teacher_npz(runs['ref_b']['values']), A)
    pooled, single = {}, {}
    for name, _ in variants[2:]:
        per, bad_v = teacher_diffs(teacher_npz(runs[name]['values']), A)
        bad = bad or bad_v
        pooled.update({(name, *k): v for k, v in per.items()})
        single[name] = teacher_baseline(per)
    if bad:
        raise SystemExit(f'stock teacher replays disagree in exact items: {bad[:3]}')
    shutil.copyfile(runs['ref_a']['values'], TASK / 'teacher_ref.npz')
    wa, wb = runs['ref_a']['windows'], runs['ref_b']['windows']
    res = {'baseline': teacher_baseline(pooled), 'rebatchings': single,
           'repeat_same_batching': teacher_baseline(per_b),
           'bitwise_repeat': all(float(abs(v).max()) == 0.0 for v in per_b.values() if len(v)),
           'timing_runs': [wa, wb], 'layout': runs['ref_a']['layout'].get('teacher'),
           'runs': {k: strip(v) for k, v in runs.items()}}
    (TASK / 'teacher_ref.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(json.dumps({'status': 'ok', 'baseline': res['baseline'], 'bitwise_repeat': res['bitwise_repeat'],
                      'rebatchings': {k: {f: v[f] for f in ('mean_abs', 'tau', 'events')} for k, v in single.items()},
                      'windows': [wa, wb], 'cost': cost()}, default=str))
    return 0


def cmd_freeze_timing(a):
    """Formulator: reference side timings (rollout: the run of freeze-rollout + one more; train: two runs) and the step
    fractions from the full-loop reference windows of the reference's own final check -> task/timing_ref.json."""
    out = Path(a.out)
    ft = frozen('timing_ref.json') or {}
    ref_src = check_reference()
    with scratch_dir('sg-ref-') as tmp:
        tr = materialize(ref_src, Path(tmp) / 'reference')
        for side in a.sides:
            runs = []
            if side == 'rollout' and frozen('rollout_ref.json') and not a.fresh:
                runs.append(frozen('rollout_ref.json')['timing_run'])
            if side == 'teacher' and frozen('teacher_ref.json') and not a.fresh:
                runs += [{'windows': w} for w in frozen('teacher_ref.json')['timing_runs']]
            primed = None
            if CFG['timing'][side].get('prime') and len(runs) < 2:
                # one untimed priming run of this side: both runs of the pair start from copies of its caches
                prime, primed = prime_caches(tr, out, f'ref_{side}', False, None,
                                             Path(tmp) / f'prime_{side}', side)
                if primed is None:
                    raise SystemExit(f'reference {side} priming failed: {prime.get("reason")}')
            while len(runs) < 2:
                if side == 'teacher':
                    r = teacher_replay_run(tr, FIX['dumps'], out, f'ref_{len(runs)}', False)
                else:
                    r = time_side(side, tr, out, f'ref_{len(runs)}', candidate=False, dump=side == 'rollout',
                                  primed=primed)
                if r['status'] != 'ok' or r['layout_problems']:
                    raise SystemExit(f'reference {side} timing failed: {r.get("reason") or r["layout_problems"]}')
                runs.append(strip(r))
            A, B = runs[0]['windows'], runs[1]['windows']
            fus = [(r.get('first_use') or {}).get('seconds') for r in runs]
            if any(max(f or [0]) > CFG['timing']['first_use_flag_s'] for f in fus):
                log(f'WARNING {side}: the reference has Triton autotuning/compilation inside a timed window: {fus}')
            ft[side] = {'windows': [round(min(x, y), 3) for x, y in zip(A, B)], 'runs': [A, B], 'first_use_s': fus,
                        'mean': round(statistics.fmean(min(x, y) for x, y in zip(A, B)), 3),
                        'spread': round(max(abs(x - y) / min(x, y) for x, y in zip(A, B)), 4)}
            if side == 'rollout':
                tails = [t for r in runs for t in (r.get('teacher_tail_s') or [])]
                ft['teacher_tail_s'] = round(statistics.fmean(tails), 3) if tails else None
            log(f'{side}: {ft[side]}')
    if a.full:
        full = json.loads(Path(a.full).read_text())['performance']
        # three runs of the reference tree on this machine: A and B of its own final check and the check's own
        # candidate run (the unchanged tree). Per window the mean of the three is the frozen reference of this
        # lane, and their spread sets the benefit bar.
        R = [full['reference_a_s_per_step'], full['reference_b_s_per_step'], full['candidate_s_per_step']]
        W = [round(statistics.fmean(w), 3) for w in zip(*R)]  # mean of the three reference runs (owner, 11-16): the candidate is timed once, a best-of-3 reference would be biased against it
        spread = round(max((max(w) - min(w)) / min(w) for w in zip(*R)), 4)
        ft['full'] = {'windows': W, 'mean': round(statistics.fmean(W), 3), 'spread': spread, 'runs': R,
                      'pair_spread': full['reference_spread'], 'source': a.full,
                      'mismatch_reference': (full.get('mismatch_full_loop') or {}).get('reference'),
                      'measured': datetime.now().isoformat(timespec='seconds'), 'gpu_uuids': gpu_uuids()}
    if 'full' in ft and 'rollout' in ft and 'train' in ft:
        ft['fractions'] = {s: round(ft[s]['mean'] / ft['full']['mean'], 4) for s in ('rollout', 'train')}
        if ft.get('teacher_tail_s') is not None:
            # the teacher's share of a step on the critical path (inside the rollout's share)
            ft['fractions']['teacher'] = round(ft['teacher_tail_s'] / ft['full']['mean'], 4)
    ft['host'] = socket.gethostname()
    (TASK / 'timing_ref.json').write_text(json.dumps(ft, indent=1) + '\n')
    print(json.dumps({k: v for k, v in ft.items()}, default=str))
    return 0


def cmd_freeze_train(a):
    """Formulator: reference records (deterministic twice: repeatability; non-deterministic once and two more
    deterministic: run-to-run noise) on set 0 and one record on each other fixture set (data-set sigma)."""
    out = Path(a.out)
    tr_cfg = CFG['train_record']
    s0, n0 = tr_cfg['start'], tr_cfg['num_rollout']
    plan = [('rec_a', s0, True), ('rec_b', s0, True), ('rec_nondet', s0, False), ('rec_c', s0, True),
            ('rec_d', s0, True)]
    plan += [(f'set{k}', a0, True) for k, (a0, _) in enumerate(tr_cfg['sets'][1:], 1)]
    if a.smoke:
        plan = plan[:2]
    if a.only:
        plan = [p for p in plan if p[0] in a.only]
    runs = {}
    todo = []
    for name, start, det in plan:
        prev = existing_record(out, name, start, n0)
        if prev is not None:
            log(f'{name}: reusing the complete record run in {out}/record_{name}')
            runs[name] = prev
        else:
            todo.append((name, start, det))
    if todo:
        ref_src = check_reference()
        with scratch_dir('sg-ref-') as tmp:
            tr = materialize(ref_src, Path(tmp) / 'reference')
            for name, start, det in todo:
                runs[name] = record_run(tr, out, name, start, n0, False, deterministic=det)
                if runs[name]['status'] != 'ok':
                    raise SystemExit(f'{name} failed: {runs[name]["run"].get("reason")}')
                if runs[name]['layout_problems']:
                    raise SystemExit(f'{name}: layout {runs[name]["layout_problems"]}')
    if a.only:
        print(json.dumps({'status': 'ok', 'runs': sorted(runs), 'cost': cost()}))
        return 0
    import numpy as np
    exact, diff = compare_records(runs['rec_a'], runs['rec_b'])
    if exact:
        raise SystemExit(f'two reference records differ in exact items: {exact[:3]}')
    log(f'reference repeatability: {len(diff)} differing tensors')
    sk1 = update_sketch(runs['rec_a'], s0, n0)
    np.save(TASK / 'train_ref_sketch.npy', sk1)
    res = {'records': runs['rec_a']['records'], 'repeatable': not diff, 'repeat_diff': diff[:20],
           'repeat_differing': len(diff),
           'mismatch': runs['rec_a']['losses'].get(s0, {}).get('train/train_rollout_logprob_abs_diff')}
    if not a.smoke:
        eps = [endpoints(runs['rec_a'], s0, n0)]
        set_sk = []
        for k, (a0, _) in enumerate(tr_cfg['sets'][1:], 1):
            eps.append(endpoints(runs[f'set{k}'], a0, n0))
            set_sk.append(update_sketch(runs[f'set{k}'], a0, n0))
        sigma = {k: statistics.stdev([e[k] for e in eps]) for k in eps[0]}
        reps = [runs[n] for n in ('rec_a', 'rec_b', 'rec_nondet', 'rec_c', 'rec_d')]
        rep_eps = [endpoints(r, s0, n0) for r in reps]
        noise = {k: max(abs(x[k] - y[k]) for i, x in enumerate(rep_eps) for y in rep_eps[i + 1:]) for k in eps[0]}
        tol = {k: max(tr_cfg['noise_k'] * noise[k], 1e-6 * abs(eps[0][k])) for k in eps[0]}
        cos_reps = [cosine(update_sketch(r, s0, n0), sk1) for r in reps[1:]]
        cos_noise = min(cos_reps)
        res['direction'] = {'cos_noise': cos_noise, 'cos_repeats': cos_reps, 'k': tr_cfg['cos_k'],
                            'cos_min': min(1 - tr_cfg['cos_k'] * (1 - cos_noise), 1 - 1e-4),
                            'cos_other_sets': [cosine(x, sk1) for x in set_sk]}
        res['trajectory'] = {'reference': eps[0], 'sets': eps, 'sigma': sigma, 'repeats': rep_eps, 'noise': noise,
                             'noise_k': tr_cfg['noise_k'], 'tolerance': tol}
    (TASK / 'train_ref.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(json.dumps({'status': 'ok', 'repeatable': res['repeatable'], 'differing': len(diff),
                      'direction': res.get('direction'), 'trajectory': {k: res.get('trajectory', {}).get(k) for k in
                                                                        ('reference', 'sigma', 'tolerance')},
                      'mismatch': res['mismatch'], 'cost': cost()}, default=str))
    return 0


def existing_record(out, name, start, num):
    """A complete record run of an earlier (or another lane's) freeze-train in out/record_<name>, or None."""
    d = Path(out) / f'record_{name}'
    rc = d / 'exit_code'
    if not rc.exists() or rc.read_text().strip() != '0':
        return None
    r = {'status': 'ok', 'judge': str(d / 'judge'), 'log': str(d / 'train.log'), 'out': str(d)}
    rec = load_record(r)
    if sorted(rec['records']) != list(range(start, start + num)) or \
            any(len(v) != CFG['world_size'] for v in rec['records'].values()):
        return None
    rec['run'] = r
    rec['layout_problems'] = layout_problems(layout_of(r['judge'], 'train'))
    return rec


def cmd_control_arm(a):
    """Formulator: the reference trees with lr x a.lr_factor, judged by the statistical gate alone (the lr is an exact
    item for any tree; this arm measures whether the statistical gate resolves such an update)."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr_cfg = CFG['train_record']
    ref = frozen('train_ref.json')
    ref_src = check_reference()
    with scratch_dir('sg-ref-') as tmp:
        tr = materialize(ref_src, Path(tmp) / 'reference')
        run = record_run(tr, out, 'arm', tr_cfg['start'], tr_cfg['num_rollout'], False, lr=f'{1e-6 * a.lr_factor:.6g}')
    if run['status'] != 'ok':
        raise SystemExit(f'arm failed: {run["run"].get("reason")}')
    import numpy as np
    ep = endpoints(run, tr_cfg['start'], tr_cfg['num_rollout'])
    tj = ref['trajectory']
    rows = {}
    for k, v in ep.items():
        d = v - tj['reference'][k]
        rows[k] = {'candidate': v, 'reference': tj['reference'][k], 'd': d, 'tolerance': tj['tolerance'][k],
                   'sigma': tj['sigma'][k], 'within': abs(d) <= tr_cfg['sigma_k'] * tj['sigma'][k]
                   and abs(d) <= tj['tolerance'][k]}
    cos = cosine(update_sketch(run, tr_cfg['start'], tr_cfg['num_rollout']), np.load(TASK / 'train_ref_sketch.npy'))
    rows['update_direction_cos'] = {'candidate': cos, 'min': ref['direction']['cos_min'],
                                    'within': cos >= ref['direction']['cos_min']}
    st = 'PASS' if all(r['within'] for r in rows.values()) else 'INCORRECT'
    (out / 'check.json').write_text(json.dumps({'status': st, 'lr_factor': a.lr_factor, 'statistical': rows},
                                               indent=1) + '\n')
    print(json.dumps({'status': st, 'statistical': rows}))
    return 0 if st == 'INCORRECT' else 1


def cmd_freeze_inputs(a):
    inputs = freeze_inputs(a.paths)
    f = TASK / 'formulation.json'
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
    k.add_argument('--side', choices=list(ALL_SIDES), help='judge this side too')
    k.add_argument('--quick', action='store_true', help='no timing (the correctness gates only)')
    k.add_argument('--skip-timing', action='store_true', help='no timing (the same as --quick)')
    k.add_argument('--n1', action='store_true', help='no timing, no bitwise attempt: the statistical gates only')
    k.add_argument('--final', action='store_true', help=argparse.SUPPRESS)
    t = sub.add_parser('time')
    t.add_argument('--tree', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--config')
    t.add_argument('--side', choices=list(ALL_SIDES))
    f = sub.add_parser('profile')
    f.add_argument('--tree', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--config')
    f.add_argument('--side', choices=list(ALL_SIDES), required=True)
    x = sub.add_parser('fixture', help='formulator')
    x.add_argument('--out', required=True)
    x.add_argument('--rollouts', type=int, default=FIX['num_rollouts'])
    x = sub.add_parser('freeze-rollout', help='formulator')
    x.add_argument('--out', required=True)
    x = sub.add_parser('freeze-teacher', help='formulator')
    x.add_argument('--out', required=True)
    x = sub.add_parser('freeze-timing', help='formulator')
    x.add_argument('--out', required=True)
    x.add_argument('--sides', nargs='*', default=list(ALL_SIDES))
    x.add_argument('--full', help='check.json of the reference\'s own final check')
    x.add_argument('--fresh', action='store_true', help='two new rollout runs (do not reuse freeze-rollout\'s)')
    x = sub.add_parser('freeze-train', help='formulator')
    x.add_argument('--out', required=True)
    x.add_argument('--smoke', action='store_true')
    x.add_argument('--only', nargs='+', help='run these record runs only (lanes in parallel), no summary')
    x = sub.add_parser('control-arm', help='formulator')
    x.add_argument('--out', required=True)
    x.add_argument('--lr-factor', type=float, default=1.02)
    x = sub.add_parser('freeze-inputs', help='formulator')
    x.add_argument('paths', nargs='+')
    a = p.parse_args()
    if a.cmd in ('fixture', 'freeze-rollout', 'freeze-teacher', 'freeze-timing', 'freeze-train', 'control-arm',
                 'freeze-inputs') \
            and os.geteuid() != 0:
        raise SystemExit(f'{a.cmd} is the formulator\'s (root, during preparation)')
    if a.cmd == 'check' and a.final and (a.quick or a.skip_timing or a.n1):
        raise SystemExit('check --final takes no --quick, --skip-timing or --n1')
    if a.cmd == 'check' and a.n1 and (a.quick or a.skip_timing):
        raise SystemExit('check --n1 takes no --quick or --skip-timing (it runs no timing already)')
    return {'check': cmd_check, 'time': cmd_time, 'profile': cmd_profile, 'fixture': cmd_fixture,
            'freeze-rollout': cmd_freeze_rollout, 'freeze-teacher': cmd_freeze_teacher,
            'freeze-timing': cmd_freeze_timing,
            'freeze-train': cmd_freeze_train, 'control-arm': cmd_control_arm,
            'freeze-inputs': cmd_freeze_inputs}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
