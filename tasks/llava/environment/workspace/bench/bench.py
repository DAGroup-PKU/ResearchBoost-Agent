"""LLaVA-1.5 7B finetune bench: one command per question.

Run from the workspace root (ws/) with the training environment's Python (bench.json python). GPU phases take
/tmp/llava_gpu.lock themselves, one phase at a time; do NOT wrap these commands in flock (that would deadlock). Other
GPU commands of yours: `flock /tmp/llava_gpu.lock <cmd>`.

  python bench/bench.py check --tree <dir> --out <dir> [--config <file>] [--quick | --skip-timing | --n1]
        the verdict: gates 0 -> 1 -> 2 -> 3 below; writes <out>/check.json. --quick: gate 0 (setup guard, data
        loader) and the structure record (exact checks of gate 2), never a score. --skip-timing: no gate 1. --n1:
        gates 0 and 3 only, after a check whose gate 2 was not bitwise identical. --final is the final
        evaluation's (root or the verifier's marker, final_allowed): gate 1 as reference -> candidate -> reference.
  python bench/bench.py time --tree <dir> --out <dir> [--config <file>] [--label L]
        one timing run of a tree (same windows as gate 1), with its speedup over the frozen reference timing (the
        mean of the preparation's reference runs) and that reference's own run-to-run noise; without a frozen
        reference timing it times the reference once first and reads the speedup against that run (cached in /tmp
        for the next time on the same GPUs; its noise is then not measured)
  python bench/bench.py freeze-timing --out <dir> --labels a b c
        formulation only: freeze the mean of the reference runs time_<label>.json as the reference timing
  python bench/bench.py profile --tree <dir> --out <dir> [--config <file>] [--steps K] [--detail]
        Nsight Systems trace of K steps (default 2) right after the warmup (needs nsys), or with --detail the
        PyTorch profiler over K steps (default 1); writes profile_<label>.json and a short profile_<label>.md
  python bench/bench.py sweep --tree <dir> --config a.json --config b.json ... --out <dir>
        screen: reference, one timing per switch config, reference again; never a score
  python bench/bench.py baseline --out <dir>    reference against itself (noise floor, bitwise reproducibility)

Every GPU run (timing, numerics record, trajectories, probes) goes through run_gpu: it holds the GPU lock, waits until
the visible GPUs are empty and idle (INVALID after 10 minutes otherwise), runs in its own process group with a timeout
(bench.json timeouts_s; TIMEOUT kills the group) with a fresh short TMPDIR, and records the GPU state after. Every
timing run gets fresh compile caches (Inductor, Triton, CUDA) under /tmp, is scanned for the invalid-run signatures
in bench.json, fails if torch.compile compiled anything inside a timing window, and carries a fingerprint (argv,
switches, math mode, GPU UUIDs, cache, library versions). Every command hashes every frozen input declared in
task/formulation.json (weights, fixture, images, thresholds, frozen trajectories, reference records) before
(mismatch: nothing is measured) and after (mismatch: ERROR), and digests the source tree it copied at the start and
again at the end (a tree that wrote into itself: ERROR). Outputs carry cost.gpu_hours (lock hold time x world size).

Candidate switches: a JSON file {"environment": {"NAME": "value", ...}}; the bench sets those variables for the
candidate's processes only (never for the reference). Variables the bench manages itself cannot be switches.

Interpreters: the tree under test runs with bench.json candidate_python, the agent's environment, which sees the
training environment's packages unless it installs its own versions. The reference (and a time or profile run of the
reference checkout itself) runs with bench.json python, the read-only training environment, without user
site-packages. Timing fingerprints record the library versions of each run's interpreter; a difference between
reference and candidate versions is recorded, not an error.

Math mode: reference and candidate both start in the frozen math mode from bench.json (tf32, as upstream's --tf32
True), set by the bench before any training code is imported. A candidate that changes math flags is never N0; gate 3
decides, and the numerics result reports precision_changed.

Verdict flow; each gate stops the evaluation when it fails:
  0. coverage   train() in llava/train/train.py and the ModelArguments / DataArguments / TrainingArguments classes
                must equal the reference AST (the adapter and the data probe re-create that setup), and the tree's
                data pipeline must yield bitwise the same first micro-batches as the reference (bench/loader_probe.py).
  1. benefit    the agent's check: one fresh torchrun of the candidate against the frozen reference timing (the
                preparation's reference runs); the noise rules below are reported, the bar is the same.
                The final evaluation (--final): reference, candidate, reference: fresh torchruns of the real entrypoint on the timing fixture; their
                fingerprints must agree with the first reference run (else ERROR, not comparable; library
                versions are recorded for both and not compared); in the final check (--final) only, a run whose
                windows disagree with each other (relative to the frozen reference windows) by more than
                max_window_spread, or a reference run off the frozen reference by more than max_reference_drift,
                is INVALID (rerun later), elsewhere both are reported; a run whose logged per-step losses are off the frozen reference ones
                trained on other batches: ERROR. A speedup not above the bar max(min_speedup, 1 + twice the
                reference spread) -> rejected as no benefit (status UNAVAILABLE, verdict no_benefit).
  2. bitwise    short cases (bench/cases.json: one optimizer step on an image-text batch, one on a text-only batch,
                each with the run's ranks, accumulation and per-device batch) recorded in deterministic mode. Exact
                checks (phase, call counts, RNG, invariants incl. learning rates and each step's data digests,
                cross-rank pattern) must hold, else INCORRECT. Every tensor bitwise identical -> PASS (N0), adopt.
  3. statistical  run for every candidate, bitwise ones included: a paired training trajectory (losses only,
                compressed schedule, the production settings that are timed, not the deterministic ones) on
                the first frozen reference seed (trajectory.seeds = 1). Each endpoint's bias, candidate - reference, against trajectory.sigma_k
                (3) x the reference's seed-to-seed sigma (frozen before any candidate): all within -> PASS (N1),
                any beyond -> INCORRECT. The same in every check, the final evaluation's included.
  Tiers: N0 every tensor bitwise identical with identical dtypes and math modes; N1 otherwise (a changed dtype or math
  mode is allowed and reported as precision_changed). Per-tensor tolerance gates are reported as diagnostics only.
"""
import argparse
import ast
import fcntl
import functools
import hashlib
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
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent                       # workspace


def _expand(value):
    """bench.json and task/formulation.json name machine paths through ${ASSETS} (the task's read-only assets)."""
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
LOCK = Path('/tmp/llava_gpu.lock')
PY = Path(CFG['python'])                      # training environment: the judge, the reference, the formulation
CAND_PY = Path(CFG['candidate_python'])       # the agent's environment: every run of the tree under test
TORCHRUN = PY.parent / 'torchrun'
SCRATCH = '/tmp'
TIMEOUT = {'timing': 2400, 'numerics': 2400, 'trajectory': 3600, 'probe': 900, **CFG.get('timeouts_s', {})}
RESERVED = {'TMPDIR', 'TMP', 'TEMP', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH',
            'CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'NVIDIA_TF32_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG',
            'HF_HOME', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'MASTER_PORT', 'MASTER_ADDR'}
SIGNATURES = [(re.compile(x['regex']), x['reason']) for x in CFG.get('invalid_run_signatures', [])]
COST = {'gpu_seconds': 0.0, 'gpu_runs': 0}


def path(key):
    value = Path(_expand(key))
    return value if value.is_absolute() else (ROOT / value).resolve()


def log(msg):
    print(f'[bench {datetime.now():%H:%M:%S}] {msg}', flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def evaluator_sha256():
    files = sorted(p for p in (ROOT / 'numerics').rglob('*.py'))
    files += sorted(p for p in BENCH.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                    and 'records' not in p.parts and 'ref' not in p.parts)
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode() + b'\0' + sha256_file(p).encode() + b'\n')
    return h.hexdigest()


class GpuLock:
    def __init__(self, what, timeout=4 * 3600):
        self.what, self.timeout = what, timeout

    def __enter__(self):
        LOCK.touch(exist_ok=True)
        self.fd = os.open(LOCK, os.O_RDONLY)
        start, said = time.time(), False
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.acquired = time.time()
                return self
            except BlockingIOError:
                if not said:
                    log(f'waiting for {LOCK} ({self.what}); if this command is wrapped in flock, unwrap it')
                    said = True
                if time.time() - start > self.timeout:
                    raise SystemExit(f'GPU lock not acquired within {self.timeout}s')
                time.sleep(5)

    def __exit__(self, *_):
        COST['gpu_seconds'] += (time.time() - self.acquired) * CFG['world_size']
        COST['gpu_runs'] += 1
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


def cost():
    return {'gpu_hours': round(COST['gpu_seconds'] / 3600, 4), 'gpu_runs': COST['gpu_runs'],
            'method': f'GPU lock hold time x {CFG["world_size"]} cards: includes startup, warmup and failed runs'}


def scratch_dir(prefix):
    free = shutil.disk_usage(SCRATCH).free / 2**30
    need = CFG.get('min_scratch_free_gib', 50)
    if free < need:
        raise SystemExit(f'{SCRATCH} has {free:.1f} GiB free, the bench needs {need}: free space first')
    return tempfile.TemporaryDirectory(prefix=prefix, dir=SCRATCH)


def gpu_state():
    smi = shutil.which('nvidia-smi')
    if not smi:
        return None
    r = subprocess.run([smi, '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, timeout=60)
    rows = []
    for line in r.stdout.splitlines():
        try:
            idx, uuid, mem, util = [x.strip() for x in line.split(',')]
            rows.append({'index': int(idx), 'uuid': uuid, 'memory_used_mib': float(mem), 'utilization': float(util)})
        except ValueError:
            continue
    return rows


def wait_idle(what, patience=600):
    start = time.time()
    while True:
        state = gpu_state()
        busy = [g for g in state or [] if g['memory_used_mib'] > 1024 or g['utilization'] > 5]
        if not busy:
            return state, None
        if time.time() - start > patience:
            return state, f'GPU {[g["index"] for g in busy]} not idle before {what} for {patience}s (another process?)'
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


def run_gpu(cmd, cwd, env, log_file, what, timeout):
    with GpuLock(what), scratch_dir('lg-') as tmpdir, CpuSampler() as cpu:
        env = dict(env if env is not None else os.environ, TMPDIR=tmpdir, TMP=tmpdir, TEMP=tmpdir)
        before, problem = wait_idle(what)
        if problem:
            return {'status': 'INVALID', 'reason': problem, 'returncode': None, 'gpus_before': before}
        log(what)
        start = time.time()
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc, status = proc.wait(timeout=timeout), None
            except subprocess.TimeoutExpired:
                for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, 30)):
                    try:
                        os.killpg(proc.pid, sig)
                        proc.wait(timeout=grace)
                        break
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        continue
                rc, status = None, 'TIMEOUT'
            for sig in (signal.SIGKILL,):        # leftovers of the group (loader workers, compile workers)
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    pass
        wall = time.time() - start
        after, _ = wait_idle(f'{what} (after)', patience=90)
    out = {'status': status or 'ok', 'returncode': rc, 'wall_s': round(wall, 1), 'gpus_before': before,
           'gpus_after': after, 'cpu': cpu_summary(cpu, start, start + wall), 'cpu_sampler': cpu}
    if status:
        out['reason'] = f'{what} exceeded {timeout}s; process group killed; see {Path(log_file).name}'
    elif any(g['memory_used_mib'] > 1024 for g in after or []):
        out['warning'] = 'GPU memory still in use 90s after the run ended (leaked process?)'
    return out


def signatures(log_file):
    hits, text = [], Path(log_file).read_text(errors='replace') if Path(log_file).exists() else ''
    for pattern, reason in SIGNATURES:
        m = pattern.search(text)
        if m:
            line = text[text.rfind('\n', 0, m.start()) + 1:text.find('\n', m.end())]
            hits.append({'reason': reason, 'line': line.strip()[:300]})
    return hits


def tree_digest(tree):
    h = hashlib.sha256()
    for p in sorted(x for x in Path(tree).rglob('*') if x.is_file() and not x.is_symlink()
                    and '__pycache__' not in x.parts and '.git' not in x.parts):
        h.update(str(p.relative_to(tree)).encode() + b'\0' + sha256_file(p).encode() + b'\n')
    return h.hexdigest()


def content_digest(p):
    p = Path(p)
    if p.is_file():
        return {'sha256': sha256_file(p)}
    h, n = hashlib.sha256(), 0
    for f in sorted(p.rglob('*')):
        if f.is_file():
            h.update(str(f.relative_to(p)).encode() + b'\0' + sha256_file(f).encode() + b'\n')
            n += 1
    return {'sha256': h.hexdigest(), 'files': n}


def frozen_inputs():
    assets = json.loads((ROOT / 'task' / 'formulation.json').read_text())['frozen_inputs']
    return [(name, a['path'], a['sha256']) for name, a in assets.items()]


def verify_inputs(when):
    start, bad = time.time(), []
    items = frozen_inputs()
    for name, rel, want in items:
        target = path(rel)
        got = content_digest(target)['sha256'] if target.exists() else None
        if got != want:
            bad.append({'input': name, 'path': rel, 'declared': want, 'found': got})
    out = {'when': when, 'ok': not bad, 'checked': len(items), 'mismatches': bad, 'seconds': round(time.time() - start, 1)}
    log(f'inputs {when}: {len(items) - len(bad)}/{len(items)} match the formulation')
    return out


def guarded_inputs():
    before = verify_inputs('before')
    if not before['ok']:
        raise SystemExit(f'frozen inputs differ from task/formulation.json, nothing measured: {before["mismatches"]}')
    return before


# 'versions' is recorded in every fingerprint but not compared: the candidate may run other library versions in
# its own environment (performance.library_versions lists both).
FINGERPRINT_KEYS = ('math_mode', 'gpus', 'compile_cache', 'instrumentation', 'timing_plan')


def fingerprint_diff(a, b):
    a, b = a or {}, b or {}
    return [k for k in FINGERPRINT_KEYS if a.get(k) != b.get(k)]


def source_end_check(result, tree, key='source_digest'):
    result[f'{key}_end'] = tree_digest(tree)
    if result[f'{key}_end'] != result[key]:
        result['status'] = 'ERROR'
        result['reason'] = 'the source tree changed while it ran (the code wrote into its own tree)'
    return result


VERSIONS = """import importlib.metadata, json, sys
out = {}
for name in sys.argv[1:]:
    try:
        out[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        out[name] = None
print(json.dumps(out))
"""


@functools.lru_cache(maxsize=None)
def versions(python):
    """Library versions as the given interpreter sees them (the run's own interpreter, not the judge's)."""
    r = subprocess.run([str(python), '-c', VERSIONS, 'torch', 'triton', 'numpy', 'transformers', 'deepspeed',
                        'flash-attn', 'accelerate'], capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONNOUSERSITE='1'))
    if r.returncode:
        return {'error': f'{python} exited {r.returncode}: {r.stderr[-500:]}'}
    return json.loads(r.stdout.strip().splitlines()[-1])


def free_port():
    import socket
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


# Static rendezvous on the loopback: inside the sandbox's own network namespace the container hostname (which
# torchrun --standalone resolves for its c10d store) has no address.
def torchrun_argv(candidate=False):
    """torchrun with the reference's interpreter, or torchrun's module with the candidate interpreter (its workers
    then use that interpreter too)."""
    runner = [str(CAND_PY), '-m', 'torch.distributed.run'] if candidate else [str(TORCHRUN)]
    return [*runner, '--nnodes=1', f'--nproc-per-node={CFG["world_size"]}', '--master-addr=127.0.0.1',
            f'--master-port={free_port()}', '--local-addr=127.0.0.1']


def base_env(extra=None, candidate=False):
    """The caller's environment without the bench's own variables, then `extra`. PATH puts the run's interpreter
    first: the candidate environment for the tree under test; the training environment, without user
    site-packages, for everything else."""
    env = dict(os.environ)
    for k in list(env):
        if k.startswith('LLAVA_BENCH_'):
            del env[k]
    env.update(extra or {})
    rest = [d for d in env.get('PATH', '/usr/local/bin:/usr/bin:/bin').split(':')
            if d and d not in (str(PY.parent), str(CAND_PY.parent))]
    env['PATH'] = ':'.join(([str(CAND_PY.parent)] if candidate else []) + [str(PY.parent)] + rest)
    if not candidate:
        env['PYTHONNOUSERSITE'] = '1'
    return env


def under_test(tree):
    """A tree named on the command line runs as a candidate unless it is the reference checkout itself."""
    return Path(tree).resolve() != (ROOT / CFG['reference']).resolve()


# ---------------------------------------------------------------- source trees

def materialize(src, dest):
    """Copy a source tree (no .git, no caches) to a scratch directory; the copy is what runs."""
    src, dest = Path(src).resolve(), Path(dest)

    def ignore(directory, names):
        return {'.git', '__pycache__', 'checkpoints', 'wandb'} & set(names)
    shutil.copytree(src, dest, symlinks=False, ignore=ignore)
    for d in [dest, *(p for p in dest.rglob('*') if p.is_dir())]:
        d.chmod(d.stat().st_mode | 0o700)
    return dest


def candidate_environment(config_file):
    if config_file is None:
        return {}
    data = json.loads(Path(config_file).read_text())
    unknown = sorted(set(data or {}) - {'environment', 'note'})
    if unknown:
        raise SystemExit(f'{config_file}: unknown keys {unknown}; a switch file is {{"environment": {{...}}}}')
    env = (data or {}).get('environment') or {}
    if not isinstance(env, dict) or not all(isinstance(k, str) for k in env):
        raise SystemExit(f'{config_file}: environment must be a mapping of variable names to values')
    reserved = sorted(k for k in env if k in RESERVED or k.startswith('LLAVA_BENCH_'))
    if reserved:
        raise SystemExit(f'{config_file}: {reserved} are set by the bench and cannot be candidate switches')
    return {k: str(v) for k, v in env.items()}


def check_reference():
    ref = path(CFG['reference'])
    head = subprocess.run(['git', '-C', str(ref), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(ref), 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True).stdout.strip()
    if head != CFG['reference_commit'] or dirty:
        raise SystemExit(f'reference checkout changed (HEAD {head}, dirty={bool(dirty)})')
    return ref


# ---------------------------------------------------------------- 0. coverage

GUARDED = ('train', 'ModelArguments', 'DataArguments', 'TrainingArguments')


def setup_fingerprint(tree):
    module = ast.parse((Path(tree) / 'llava' / 'train' / 'train.py').read_text())
    parts = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in GUARDED:
            parts[node.name] = ast.dump(node, include_attributes=False)
    return parts


def setup_guard(ref, tree):
    want, got = setup_fingerprint(ref), setup_fingerprint(tree)
    return [f'{name} in llava/train/train.py differs from the reference' for name in GUARDED
            if got.get(name) != want.get(name)]


def loader_digest(tree, out, label, cand_env=None, candidate=False):
    """The candidate's switch file goes to the probe too: the probe has to walk the same code path the
    timed and recorded runs walk, or a data change behind an environment switch would be invisible here."""
    log_file = out / f'loader_{label}.log'
    r = subprocess.run([str(CAND_PY if candidate else PY), str(BENCH / 'loader_probe.py'), '--tree', str(tree),
                        '--batches', str(CFG['coverage']['probe_batches'])],
                       cwd=ROOT, env=base_env({'PYTHONPATH': str(ROOT), **(cand_env or {})}, candidate),
                       capture_output=True, text=True, timeout=TIMEOUT['probe'])
    log_file.write_text(r.stdout[-20000:] + '\n' + r.stderr[-20000:])
    if r.returncode:
        return None, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1]), None


def loader_problem(want, got):
    """Gate 0's data rule (the task statement): for every optimizer step the probe covers, the global multiset of samples must
    be bitwise the reference's, and so must the global row / token / label counts and the micro-batch sizes. Which
    samples share a micro-batch and which (rank, micro-step) they go to may differ. Returns (note, detail) or None."""
    if got.get('dataset_len') != want.get('dataset_len'):
        return ('the data pipeline sees a different number of samples',
                {'dataset_len': [want.get('dataset_len'), got.get('dataset_len')]})
    if len(got.get('batches') or []) != len(want['batches']):
        return ('the data pipeline yields a different number of micro-batches',
                {'micro_batches': [len(want['batches']), len(got.get('batches') or [])]})

    def per_step(rows):
        steps = {}
        for r in rows:
            steps.setdefault(r['step'], []).append(r)
        return {k: {'micro_batches': len(v), 'micro_rows': sorted(x['rows'] for x in v),
                    'tokens': sum(x['tokens'] for x in v), 'labels': sum(x['labels'] for x in v),
                    'samples': sorted(d for x in v for d in x['row_shas'])} for k, v in steps.items()}

    w, g = per_step(want['batches']), per_step(got['batches'])
    for step in sorted(w):
        a, b = w[step], g.get(step, {})
        diff = {k: [v, b.get(k)] for k, v in a.items() if b.get(k) != v}
        if diff:
            if 'samples' in diff:
                miss = sorted(set(a['samples']) - set(b.get('samples') or []))
                extra = sorted(set(b.get('samples') or []) - set(a['samples']))
                diff['samples'] = {'missing': miss[:5], 'unexpected': extra[:5],
                                   'counts': [len(miss), len(extra)],
                                   'note': 'a sample repeated or dropped inside the step' if not miss and not extra
                                           else 'this optimizer step trains on other samples'}
            return (f"optimizer step {step} does not train on the reference's samples", diff)
    return None


def reference_partition():
    """The reference's own data probe, frozen at formulation time: what the reference's data pipeline hands the
    Trainer for the first coverage.probe_batches micro-batches. Gate 0 re-probes the reference tree itself; this
    copy is what the cheaper commands compare against."""
    f = path(CFG['coverage']['reference_probe'])
    return json.loads(f.read_text()) if f.exists() else None


def partition_state(tree, out, label, cand_env=None, candidate=False):
    """Does this tree's data pipeline train every optimizer step on the reference's samples, and did it pick its
    own partition of them? The task statement lets a candidate decide how a step's samples are spread over its
    micro-batches, which moves the logged loss (a mean over micro-batches of each one's token mean), so the
    check for "same data" is the samples themselves on the CPU, not the loss."""
    want = reference_partition()
    if want is None:
        return {'checked': False, 'repartitioned': False, 'note': 'no frozen reference probe'}
    got, err = loader_digest(tree, out, f'{label}_partition', cand_env, candidate)
    if got is None:
        return {'checked': False, 'repartitioned': False, 'error': err}
    bad = loader_problem(want, got)
    if bad:
        return {'checked': True, 'ok': False, 'repartitioned': None, 'note': bad[0], 'detail': bad[1]}
    return {'checked': True, 'ok': True,
            'repartitioned': [x['sha'] for x in want['batches']] != [x['sha'] for x in got['batches']],
            'step0': [b['row_shas'] for b in got['batches'] if b['step'] == 0]}


def expected_first_logged_loss(groups):
    """What the first logged loss has to be, for this partition of the first optimizer step and for each loss
    convention the task statement allows. It is recorded before any optimizer step and rows of a batch do not interact,
    so with the reference's per-sample token-mean cross entropy c_i and supervised label count L_i
    (task/first_step_ce.json, every sample measured alone, hence independent of the partition):

        two_level_mean   = (1/n) * sum_mb ( sum L_i c_i ) / ( sum L_i )      the upstream formula
        global_token_mean = ( sum_i L_i c_i ) / ( sum_i L_i )                invariant under the partition

    Calibrated so that the upstream convention on the reference's own partition comes out at the frozen
    reference's number: that divides out whatever bias measuring one sample at a time introduces, and leaves
    the full 0.5% for the candidate's own arithmetic. Returns None when the table cannot answer."""
    f = path(CFG['timing']['first_step_ce'])
    ref = frozen_reference_timing()
    if not groups or not f.exists() or not ref:
        return None
    ce = json.loads(f.read_text())
    table, base = ce['table'], ce.get('predicted_logged_loss_reference_partition')
    parts = []
    for g in groups:
        if any(d not in table for d in g):
            return None
        num = sum(table[d][0] * table[d][1] for d in g)
        den = sum(table[d][0] for d in g)
        if not den:
            return None
        parts.append((num, den))
    first = min(ref['logged_loss'], key=lambda k: float(k))
    scale = (ref['logged_loss'][first] / base) if base else 1.0
    return {'two_level_mean': scale * sum(n / d for n, d in parts) / len(parts),
            'global_token_mean': scale * sum(n for n, _ in parts) / sum(d for _, d in parts),
            'calibration': round(scale, 6)}


def coverage(tree, out, cand_env=None):
    ref_src = check_reference()
    problems = setup_guard(ref_src, tree)
    if problems:
        return 'ERROR', {'passed': False, 'stage': 'setup_guard', 'problems': problems,
                         'note': 'the adapter and the data probe re-create train() setup from the reference; move the '
                                 'change into a module the training code calls (model, trainer, data, entrypoint)'}
    want, err = loader_digest(ref_src, out, 'reference')
    got, err2 = loader_digest(tree, out, 'candidate', cand_env, candidate=True)
    if want is None:
        return 'ERROR', {'passed': False, 'stage': 'loader_check', 'reference_error': err}
    if got is None:
        return 'INCORRECT', {'passed': False, 'stage': 'loader_check', 'error': err2,
                             'note': 'the data probe did not run on the candidate tree'}
    bad = loader_problem(want, got)
    if bad:
        return 'INCORRECT', {'passed': False, 'stage': 'loader_check', 'error': err2,
                             'note': bad[0], 'detail': bad[1]}
    moved = [x['sha'] for x in want['batches']] != [x['sha'] for x in got['batches']]
    who = "its own" if moved else "the reference's"
    log(f"gate 0: every optimizer step trains on the reference's samples; the candidate's partition into "
        f"micro-batches is {who}")
    return None


# ---------------------------------------------------------------- 2. numerics

def record_cmd(cases_file, out_dir, adapter_kwargs, extra=(), candidate=False):
    n = CFG['numerics']
    return [*torchrun_argv(candidate), '-m', 'numerics', 'record', '--adapter', n['adapter'], '--adapter-kwargs', json.dumps(adapter_kwargs),
            '--cases', str(cases_file), '--math-mode', CFG['math_mode'], *extra, '--out', str(out_dir)]


def record_cases(tree, out, cand_env, rec, label='record', variant=None, deterministic=True, candidate=False):
    """Each case in its own torchrun (a DeepSpeed engine is built once per process)."""
    n = CFG['numerics']
    cases = json.loads(path(n['cases']).read_text())
    env = base_env({'PYTHONPATH': str(ROOT), **cand_env}, candidate)
    kwargs = {'repo': str(tree), **({'variant': variant} if variant else {})}
    runs = []
    for case in cases:
        cf = out / f'case_{case["name"]}.json'
        cf.write_text(json.dumps([case]))
        cmd = record_cmd(cf, rec, kwargs, ['--sketch-dim', str(n['sketch_dim']),
                                           *(['--deterministic'] if deterministic else [])], candidate)
        r = run_gpu(cmd, ROOT, env, out / f'{label}_{case["name"]}.log', f'numerics {label} {case["name"]}',
                    TIMEOUT['numerics'])
        runs.append({'case': case['name'], 'wall_s': r.get('wall_s'), 'cpu': r.get('cpu'),
                     'phases_s': record_phases(out / f'{label}_{case["name"]}.log')})
        if r['status'] != 'ok' or r['returncode']:
            return {'status': 'ERROR', 'runs': runs, 'reason': r.get('reason') or
                    f'record exited {r["returncode"]}; see {label}_{case["name"]}.log'}
    return {'status': 'ok', 'runs': runs}


PHASE_MARK = re.compile(r'\[numerics-phase\] rank=0 case=(\S+) phase=(\S+) t=([0-9.]+)')


def record_phases(log_file):
    """Rank 0's phase marks of a record run (numerics.record.phase_mark): seconds from process start at each
    case/phase."""
    try:
        text = Path(log_file).read_text(errors='replace')
    except OSError:
        return None
    return {f'{m.group(1)}/{m.group(2)}': float(m.group(3)) for m in PHASE_MARK.finditer(text)} or None


def dead_gradients(ref_dir, cand_dir):
    """Parameters that the reference trains but the candidate does not: a gradient whose norm is exactly zero (or
    not finite) where the reference's is nonzero. A whole module that silently stops learning changes the training
    far too little in a short trajectory to be caught by the statistical gate (a 33M-parameter projector in a 6.8B
    model), so it is an exact check here. Reference zeros are expected (e.g. the image projector in a text-only
    batch contributes an empty slice) and are compared away."""
    sys.path.insert(0, str(ROOT))
    from numerics.judge import manifests
    ref, cand = manifests(ref_dir), manifests(cand_dir)
    out = []
    for key, rm in ref.items():
        cm = cand.get(key)
        if cm is None:
            continue
        for re_, ce in zip(rm['steps'], cm['steps']):
            def zeros(entry):
                return {n: d for n, d in entry['tensors'].items()
                        if n.startswith('grad/') and (not d['finite'] or (d['norm'] or 0.0) == 0.0)}
            new = sorted(set(zeros(ce)) - set(zeros(re_)))
            if new:
                out.append({'where': f'{key[0]} r{key[1]} s{re_["step"]}', 'check': 'dead_gradient',
                            'parameters': new[:20], 'count': len(new),
                            'note': 'the reference has a nonzero gradient for these parameters, the candidate does '
                                    'not (that part of the model no longer trains)'})
    return out


def record_environment_problem():
    """The gate-2 records carry process-wide RNG fingerprints, and numerics.record's rng_fingerprint()
    hashes one CUDA RNG state per VISIBLE device. Records made where a different number of GPUs is visible
    therefore carry a fingerprint no candidate can reproduce while every tensor is identical -- a failure
    that reads as 'the candidate is wrong' and is not. The formulation pins the count the reference records
    were made with, so the mismatch is named instead of discovered."""
    want = CFG['numerics'].get('visible_gpus')
    state = gpu_state()
    got = len(state) if state else None
    if want and got and got != want:
        return (f'{got} GPUs are visible, but bench/records/ref was recorded with {want}: the recorded RNG '
                f'fingerprints cannot match here (run inside the task sandbox, or re-record the reference)')
    return None


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
    problem = record_environment_problem()
    if problem:
        return {'status': 'ERROR', 'reason': problem}
    n = CFG['numerics']
    rec = out / 'records'
    r = record_cases(tree, out, cand_env, rec, candidate=True)
    if r['status'] != 'ok':
        return r
    record_runs = r['runs']
    report = out / 'compare.json'
    compare_start = time.time()
    r = subprocess.run([str(PY), '-m', 'numerics', 'compare', '--ref', str(path(n['ref'])), '--cand', str(rec),
                        '--thresholds', str(path(n['thresholds'])),
                        *[a for x in n.get('exclude', []) for a in ('--exclude', f'{x["regex"]}={x["reason"]}')],
                        '--out', str(report)], cwd=ROOT, env=base_env({'PYTHONPATH': str(ROOT)}),
                       capture_output=True, text=True)
    (out / 'compare.log').write_text(r.stdout + r.stderr)
    compare_s = round(time.time() - compare_start, 1)
    if not report.exists():
        return {'status': 'ERROR', 'reason': f'compare exited {r.returncode}; see compare.log'}
    data = json.loads(report.read_text())
    dead = dead_gradients(path(n['ref']), rec)
    if dead:
        data.setdefault('exact_failures', []).extend(dead)
        data['status'] = 'INCORRECT'
        report.write_text(json.dumps(data, indent=1) + '\n')
    if not n.get('keep_records'):
        shutil.rmtree(rec, ignore_errors=True)
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


def trajectory_seed(tree, out_dir, seed, env, log_file, variant=None, candidate=False):
    tj = CFG['trajectory']
    spec = json.loads(path(tj['spec']).read_text())
    cf = Path(str(log_file).replace('.log', '_case.json'))
    cf.write_text(json.dumps([dict(spec['case'], name=f'traj_s{seed}', seed=seed)]))
    kwargs = {'repo': str(tree), **({'variant': variant} if variant else {})}
    # production settings (no --deterministic): gate 3 judges the configuration that is timed
    cmd = record_cmd(cf, out_dir, kwargs, ['--losses-only'], candidate)
    return run_gpu(cmd, ROOT, env, log_file, f'trajectory seed {seed}', TIMEOUT['trajectory'])


def trajectories(tree, out, cand_env):
    """Gate 3, the same in every check (the agent's, `check --n1`, the final evaluation's): the candidate's trajectory
    on the first trajectory.seeds (1) of the frozen reference seeds, each endpoint's bias against trajectory.sigma_k
    (3) x the reference's seed sigma (numerics traj-compare --sigma-k). Every seed is its own process."""
    tj = CFG['trajectory']
    frozen_path = path(tj['frozen'])
    seeds = json.loads(frozen_path.read_text())['reference_seeds'][:tj['seeds']]
    rec, runs = out / 'trajectory', []
    env = base_env({'PYTHONPATH': str(ROOT), **cand_env}, candidate=True)
    for seed in seeds:
        r = trajectory_seed(tree, rec, seed, env, out / f'traj_s{seed}.log', candidate=True)
        runs.append({'seed': seed, 'wall_s': r.get('wall_s'), 'cpu': r.get('cpu')})
        if r['status'] != 'ok' or r['returncode']:
            return {'verdict': 'error',
                    'reason': r.get('reason') or f'trajectory seed {seed} exited {r["returncode"]}; see traj_s{seed}.log'}
    report_path = out / 'traj_report.json'
    subprocess.run([str(PY), '-m', 'numerics', 'traj-compare', '--frozen', str(frozen_path), '--cand', str(rec),
                    '--sigma-k', str(tj['sigma_k']), '--out', str(report_path)],
                   cwd=ROOT, env=base_env({'PYTHONPATH': str(ROOT)}), capture_output=True, text=True)
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


# ---------------------------------------------------------------- 1. timing

def _power_sampler(out_file):
    smi = shutil.which('nvidia-smi')
    if not smi:
        return None
    return subprocess.Popen([smi, '--query-gpu=timestamp,index,power.draw,clocks.sm,utilization.gpu',
                             '--format=csv,noheader,nounits', '-lms', '500'],
                            stdout=open(out_file, 'w'), stderr=subprocess.DEVNULL)


def _read_power(f):
    rows = []
    for line in Path(f).read_text().splitlines():
        try:
            ts, idx, pw, clk, util = [x.strip() for x in line.split(',')]
            t = datetime.strptime(ts, '%Y/%m/%d %H:%M:%S.%f').timestamp()
            rows.append((t, int(idx), float(pw), float(clk), float(util)))
        except ValueError:
            continue
    return rows


def timing_argv(tree, out_dir):
    sys.path.insert(0, str(ROOT))
    from bench.llava_adapter import train_argv
    t = CFG['timing']
    return train_argv(t.get('args', {}), out_dir)


def time_tree(tree, out, label, cand_env=None, profile_steps=None, detail=False, candidate=False, noise_gate=False):
    """One timing run. noise_gate (the final check's reference -> candidate -> reference only): a run whose windows
    disagree by more than max_window_spread is INVALID; elsewhere the spread is reported and the run stands."""
    t = CFG['timing']
    out.mkdir(parents=True, exist_ok=True)
    scratch = scratch_dir('lt-')
    results = Path(scratch.name)
    tag = label if not profile_steps else f'{label}_{"detail" if detail else "nsys"}'
    prefix = out / f'timing_{tag}'
    plan = ['--warmup', str(t['warmup']), '--window', str(t['window']), '--repeats', str(t['repeats'])]
    argv = [*torchrun_argv(candidate), str(BENCH / 'timing.py'), '--out', str(prefix), *plan,
            '--math-mode', CFG['math_mode']]
    detail_out = out / f'detail_{label}'
    if profile_steps:
        argv += ['--profile-steps', str(profile_steps)]
        if detail:
            argv += ['--detail-out', str(detail_out)]
    data = (partition_state(tree, out, tag, cand_env, candidate) if not profile_steps
            else {'checked': False, 'repartitioned': False})
    if data.get('ok') is False:
        scratch.cleanup()
        return {'status': 'ERROR', 'data_check': data,
                'reason': f'the data pipeline does not train on the reference samples: {data["note"]} '
                          f'(gate 0 judges this; nothing was timed)'}
    argv += ['--', *timing_argv(tree, results / 'out')]
    caches = {k: str(results / v) for k, v in (('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('TRITON_CACHE_DIR', 'triton'),
                                                ('CUDA_CACHE_PATH', 'cuda'))}
    for v in caches.values():
        Path(v).mkdir()
    env = base_env({'PYTHONPATH': str(tree), **caches, **(cand_env or {})}, candidate)
    if profile_steps and not detail:
        nsys = shutil.which('nsys') or CFG.get('nsys')
        if not nsys or not Path(nsys).exists():
            raise SystemExit('nsys is not installed in this environment; use profile --detail (PyTorch profiler)')
        argv = [nsys, 'profile', '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                '--trace=cuda,nvtx,osrt,cublas', '--sample=none', '--cpuctxsw=none', '--force-overwrite=true',
                '-o', str(out / f'nsys_{label}'), *argv]
    for f in Path(out).glob(f'timing_{tag}.r*.json'):
        f.unlink()
    log_file = out / f'timing_{tag}.log'
    what = f'timing {label}' + (f' ({"detail" if detail else "nsys"} {profile_steps} steps)' if profile_steps else '')
    sampler = None if profile_steps else _power_sampler(out / f'power_{label}.csv')
    try:
        r = run_gpu(argv, tree, env, log_file, what, TIMEOUT['timing'])
    finally:
        if sampler:
            sampler.terminate()
            sampler.wait()
        scratch.cleanup()
    fingerprint = {'argv': [a for a in argv if not a.startswith(str(SCRATCH) + '/lt-')
                            and not a.startswith('--master-port')],
                   'candidate_environment': cand_env or {}, 'math_mode': CFG['math_mode'],
                   'gpus': [(g['index'], g['uuid']) for g in r.get('gpus_before') or []],
                   'compile_cache': 'cold (fresh per run)', 'timing_plan': plan,
                   'instrumentation': ('torch.profiler' if detail else 'nsys') if profile_steps else 'none',
                   'signatures_checked': [reason for _, reason in SIGNATURES],
                   'versions': versions(CAND_PY if candidate else PY)}
    run = {'run_status': r['status'], 'wall_s': r.get('wall_s'), 'fingerprint': fingerprint, 'cpu': r.get('cpu'),
           **({'warning': r['warning']} if r.get('warning') else {})}
    if r['status'] != 'ok':
        return {'status': r['status'], 'reason': r['reason'], **run}
    ranks = []
    for rank in range(CFG['world_size']):
        f = Path(f'{prefix}.r{rank}.json')
        ranks.append(json.loads(f.read_text()) if f.exists() else None)
    if r['returncode'] or any(x is None or x['status'] != 'ok' for x in ranks):
        errors = [x['error'] for x in ranks if x and x.get('error')]
        return {'status': 'ERROR', 'reason': f'timing exited {r["returncode"]}; {errors[:1]}; see {log_file.name}', **run}
    hits = signatures(log_file)
    if hits:
        return {'status': 'INVALID', 'reason': 'invalid-run signature in the log: ' + hits[0]['reason'],
                'signatures': hits, **run}
    if profile_steps:
        ws = [x['profile_window'] for x in ranks]
        problems = []
        if any(w is None for w in ws):
            problems.append('profiled window missing on some rank')
        if any(w and w['compiles_in_window'] for w in ws):
            problems.append('torch.compile compiled graphs inside the profiled window (not steady state)')
        res = {'status': 'INVALID' if problems else 'ok', 'problems': problems,
               'window': {'steps': profile_steps, 'seconds_max_rank': round(max(w['seconds'] for w in ws if w), 3)
                          if any(ws) else None},
               'peak_memory_allocated_gib': round(max(x['peak_memory_allocated_bytes'] for x in ranks) / 2**30, 2),
               'log': log_file.name, **run}
        if detail:
            res['traces'] = sorted(str(p) for p in out.glob(f'{detail_out.name}.r*.json*'))
        else:
            res['trace'] = str(out / f'nsys_{label}.nsys-rep')
        return res
    windows = [w for x in ranks for w in x.get('windows') or []]
    if windows and r.get('cpu_sampler') and run['cpu']:
        run['cpu']['busy_cpus_in_windows'] = r['cpu_sampler'].busy_cpus(min(w['wall_start'] for w in windows),
                                                                         max(w['wall_end'] for w in windows))
    run['cpu_binding'] = [x.get('cpu_binding') for x in ranks]
    return {**summarize_timing(ranks, out / f'power_{label}.csv', r['wall_s'],
                               repartitioned=bool(data.get('repartitioned')),
                               expected_first=expected_first_logged_loss(data.get('step0')), noise_gate=noise_gate),
            'data_check': {k: v for k, v in data.items() if k != 'step0'}, **run}


# `time` without the preparation's frozen reference timing: one run of the reference stands in for it, kept here for
# the next `time` on the same GPUs (ensure_reference_timing).
REFERENCE_ONCE = Path(SCRATCH) / 'llava_reference_timing_once.json'
_reference_once = None


def frozen_reference_timing():
    """The preparation's frozen reference timing, else the one run of the reference `time` made (or None)."""
    f = path(CFG['timing']['frozen'])
    return json.loads(f.read_text()) if f.exists() else _reference_once


def reference_once_key():
    """What a cached one-run reference timing must match: these GPUs, the timing plan and the reference."""
    t = CFG['timing']
    return {'gpus': sorted(g['uuid'] for g in gpu_state() or []), 'world_size': CFG['world_size'],
            'timing_plan': [t['warmup'], t['window'], t['repeats']], 'reference_commit': CFG['reference_commit']}


def ensure_reference_timing(out, tree):
    """`time` reads its speedup against the frozen reference timing of the preparation. Without one (this machine
    was not prepared, or the file is missing), the reference is timed once here and that run stands in for it, cached
    in REFERENCE_ONCE for the next `time` on the same GPUs, timing plan and reference; its run-to-run noise is not
    measured. Nothing to do when the frozen timing exists or when the tree timed is the reference itself. Returns
    how the reference timing was obtained."""
    global _reference_once
    if path(CFG['timing']['frozen']).exists():
        return 'frozen'
    if not under_test(tree):
        return None
    key = reference_once_key()
    try:
        cached = json.loads(REFERENCE_ONCE.read_text())
    except (OSError, ValueError):
        cached = None
    if cached and cached.get('key') == key:
        _reference_once = cached
        log(f'no frozen reference timing: using the run of the reference cached in {REFERENCE_ONCE}')
        return 'one run of the reference, cached'
    log('no frozen reference timing: timing the reference once first')
    with scratch_dir('ltr-') as tmp:
        res = time_tree(materialize(check_reference(), Path(tmp) / 'ref'), out, 'reference_once')
    if res['status'] != 'ok':
        log(f'the reference run failed ({res["status"]}: {res.get("problems") or res.get("reason")}): no speedup')
        return 'one run of the reference, failed'
    _reference_once = dict(freeze_reference_timing([res]), spread=None, key=key,
                           source='one run of the reference by time: no frozen reference timing was found')
    REFERENCE_ONCE.write_text(json.dumps(_reference_once, indent=1) + '\n')
    return 'one run of the reference, timed now'


def reference_noise(ref, speedup):
    """The frozen reference's own run-to-run spread (timing.reference_runs runs at preparation) and whether a speedup
    over it is beyond that noise: the same rule as the final check's bar, 1 + 2 x spread."""
    spread = ref.get('spread')
    if spread is None:
        return {'reference_noise': {'runs': len(ref.get('runs') or []), 'spread': None, 'band': None,
                                    'note': 'one run of the reference: its run-to-run noise is not measured'},
                'beyond_reference_noise': None}
    band = round(2 * spread, 4)
    return {'reference_noise': {'runs': len(ref.get('runs') or []), 'spread': spread, 'band': band},
            'beyond_reference_noise': speedup - 1 > band}


FROZEN_KEYS = ('ms_per_step', 'window_seconds', 'logged_loss', 'samples_per_s', 'peak_memory_allocated_gib',
               'startup_to_first_step_s')


def freeze_reference_timing(results):
    """The frozen reference timing from the preparation's reference runs: mean ms/step and per-window seconds, their
    spread ((max - min) / mean, the noise the agent's `time` is read against), the first run's logged losses."""
    ms = [r['ms_per_step'] for r in results]
    mean = statistics.mean(ms)
    windows = [round(statistics.mean(w), 3) for w in zip(*(r['window_seconds'] for r in results))]
    frozen = {k: results[0][k] for k in FROZEN_KEYS}
    frozen.update(ms_per_step=round(mean, 1), window_seconds=windows,
                  samples_per_s=round(statistics.mean(r['samples_per_s'] for r in results), 2),
                  peak_memory_allocated_gib=max(r['peak_memory_allocated_gib'] for r in results),
                  spread=round((max(ms) - min(ms)) / mean, 4),
                  sigma_rel=round(statistics.stdev(ms) / mean, 4) if len(ms) > 1 else None,
                  runs=[{'ms_per_step': r['ms_per_step'], 'window_seconds': r['window_seconds']} for r in results])
    return frozen


def summarize_timing(ranks, power_csv, wall, repartitioned=False, expected_first=None, noise_gate=False):
    t = CFG['timing']
    per_window = {}
    for x in ranks:
        for w in x['windows']:
            prev = per_window.get(w['k'])
            if prev is None or w['seconds'] > prev['seconds']:
                per_window[w['k']] = w
    problems = []
    ws = [per_window[k] for k in range(t['repeats']) if k in per_window]
    if len(ws) != t['repeats']:
        problems.append(f'{len(ws)} of {t["repeats"]} windows')
        return {'status': 'ERROR', 'problems': problems}
    compiled = [w for w in ws if w.get('compiles_in_window')]
    if compiled:
        problems.append(f'torch.compile compiled {sum(w["compiles_in_window"] for w in compiled)} graphs inside timing '
                        f'windows (still compiling in steady state)')
    window_s = [w['seconds'] for w in ws]
    ms = 1000 * sum(window_s) / (t['repeats'] * t['window'])
    power = _read_power(power_csv) if Path(power_csv).exists() else []
    inside = [p for p in power if any(w['wall_start'] <= p[0] <= w['wall_end'] for w in ws)]
    watts = {}
    for p in inside:
        watts.setdefault(p[1], []).append(p[2])
    out = {'status': 'ok' if not problems else 'ERROR', 'problems': problems,
           'ms_per_step': round(ms, 1), 'window_seconds': [round(s, 3) for s in window_s],
           'samples_per_s': round(CFG['world_size'] * CFG['per_rank_batch'] * CFG['grad_accum'] / (ms / 1000), 2),
           'avg_power_w_all_gpus': round(sum(statistics.mean(v) for v in watts.values()), 1) if watts else None,
           'gpu_util_median': statistics.median([p[4] for p in inside]) if inside else None,
           'peak_memory_allocated_gib': round(max(x['peak_memory_allocated_bytes'] for x in ranks) / 2**30, 2),
           'peak_memory_reserved_gib': round(max(x['peak_memory_reserved_bytes'] for x in ranks) / 2**30, 2),
           'startup_to_first_step_s': round(max(x['startup_to_first_step_s'] or 0 for x in ranks), 1),
           'process_wall_s': round(wall, 1),
           'logged_loss': next((x['logged_loss'] for x in ranks if x['rank'] == 0), {}),
           'step_wall_s_rank0': next((x['step_wall_s'] for x in ranks if x['rank'] == 0), [])}
    ref = frozen_reference_timing()
    if ref and out['status'] == 'ok':
        ratios = [s / r for s, r in zip(window_s, ref['window_seconds'])]
        med = statistics.median(ratios)
        out['window_ratio_to_frozen'] = [round(x, 4) for x in ratios]
        out['window_spread'] = round((max(ratios) - min(ratios)) / med, 4)
        out['speedup_vs_frozen_reference'] = round(ref['ms_per_step'] / ms, 4)
        out.update(reference_noise(ref, out['speedup_vs_frozen_reference']))
        if out['window_spread'] > t['max_window_spread']:
            noisy = (f'window spread {out["window_spread"]} (relative to the frozen reference windows) '
                     f'> {t["max_window_spread"]}: interference during the run')
            if noise_gate:              # the final check's A -> candidate -> B only
                out['status'] = 'INVALID'
                out['problems'].append(noisy + '; rerun later')
            else:
                out.setdefault('notes', []).append(noisy + ' (reported; only the final check rejects a noisy run)')
        # This is a coarse "working run" guard, not a precision gate (gates 2 and 3 judge numerics). Which
        # samples the run trained on is checked exactly, on the CPU, by the data probe (partition_state here,
        # gate 0 in check) -- not by this band:
        #   - with the reference's own partition the first logged step happens before any optimizer step, so it
        #     must still match tightly whatever else a candidate changes;
        #   - a candidate that repartitions the step (allowed by the task statement) moves the logged loss even though it
        #     trains on exactly the reference's samples: the number is a mean over micro-batches of each one's
        #     token mean, so it follows the partition. Measured on the reference tree in
        #     task/partition_sensitivity.json; the band is twice the largest shift seen there.
        #   - later steps are only bounded loosely either way: a legitimate N1 change (a different but
        #     equivalent update or reduction order) perturbs them, while a broken run (loss collapsing to zero,
        #     diverging, NaN) leaves the band by far.
        band1, against = t['max_logged_loss_rel_diff_step1'], 'the frozen reference'
        band = t['max_logged_loss_rel_diff_repartitioned' if repartitioned else 'max_logged_loss_rel_diff']
        if expected_first:
            against = 'the value this partition and loss convention must produce'
        elif repartitioned:
            # The partition moved the number and the table cannot say by how much: fall back to the
            # sensitivity measured on the reference itself (task/partition_sensitivity.json).
            band1 = t['max_logged_loss_rel_diff_step1_repartitioned']
            against = 'the frozen reference (no prediction available for this partition)'
        out['logged_loss_band'] = {'step1': band1, 'later_steps': band, 'repartitioned': bool(repartitioned),
                                   'step1_compared_to': against}
        diffs = {}
        for step, v in out['logged_loss'].items():
            r = ref['logged_loss'].get(step)
            if r is not None:
                diffs[int(step)] = abs(v - r) / max(abs(r), 1e-6)
        first = min(diffs) if diffs else None
        out['logged_loss_max_rel_diff'] = round(max(diffs.values()), 4) if diffs else None
        out['logged_loss_first_step_rel_diff'] = round(diffs[first], 5) if diffs else None
        bad_finite = [s for s, v in out['logged_loss'].items()
                      if not (v == v and abs(v) != float('inf')) or (v == 0.0 and ref['logged_loss'].get(s))]
        if diffs and expected_first:
            got_first = next(v for k, v in out['logged_loss'].items() if int(k) == first)
            rel = {k: abs(got_first - v) / max(abs(v), 1e-6)
                   for k, v in expected_first.items() if k != 'calibration'}
            convention = min(rel, key=rel.get)
            diffs[first] = rel[convention]
            out['first_step_expected'] = {k: round(v, 6) for k, v in expected_first.items()}
            out['first_step_matched_convention'] = convention
            out['first_step_rel_diff_vs_expected'] = {k: round(v, 5) for k, v in rel.items()}
        if not diffs:
            out['status'] = 'ERROR'
            out['problems'].append('no per-step losses logged: cannot tell whether the run trained on the same data')
        elif diffs[first] > band1:
            out['status'] = 'ERROR'
            out['problems'].append(f'the first logged loss (before any optimizer step) differs from {against} '
                                   f'by {round(diffs[first], 5)} (> {band1}): the forward computes something '
                                   f'else'
                                   + ('' if repartitioned else ', or the run did not train on the same batches'))
        elif bad_finite:
            out['status'] = 'ERROR'
            out['problems'].append(f'logged loss is zero or not finite at steps {bad_finite[:5]} where the reference '
                                   f'is not: the run broke')
        elif max(diffs.values()) > band:
            out['status'] = 'ERROR'
            out['problems'].append(f'logged per-step losses leave the sanity band: up to '
                                   f'{out["logged_loss_max_rel_diff"]} (> {band}); the run diverged or computes '
                                   f'something else')
    return out


def compare_timing(ref_a, cand, ref_b):
    ra, rb, c = ref_a['ms_per_step'], ref_b['ms_per_step'], cand['ms_per_step']
    ref = (ra + rb) / 2
    return {'reference_ms_per_step': round(ref, 1), 'candidate_ms_per_step': c, 'speedup': round(ref / c, 4),
            'reference_spread': round(abs(ra - rb) / ref, 4)}


def benefit_frozen(tree, out, cand_env):
    """Gate 1 of the agent's check: one candidate run against the frozen reference timing of the preparation (the mean
    of its timing.reference_runs reference runs, each warmup + window x repeats). The bar is the final check's
    (min_speedup, or the frozen reference's spread doubled, over one: the larger); the noise rules are reported,
    never a rejection. The
    reference -> candidate -> reference comparison is the final evaluation's (check --final)."""
    source = ensure_reference_timing(out, tree)
    cand = time_tree(tree, out, 'cand', cand_env=cand_env, candidate=True)
    frozen = frozen_reference_timing()
    runs = {'cand': cand}
    if cand['status'] != 'ok':
        return cand['status'] if cand['status'] == 'INVALID' else 'ERROR', \
            {'speedup': None, 'runs': runs, 'reason': f'timing run: {cand["status"]} '
                                                      f'{cand.get("problems") or cand.get("reason")}'}
    if not frozen:
        return 'ERROR', {'speedup': None, 'runs': runs, 'reason': 'no reference timing (the reference run failed)'}
    spread = frozen.get('spread') or 0.0
    speedup = cand['speedup_vs_frozen_reference']
    perf = {'reference_ms_per_step': frozen['ms_per_step'], 'candidate_ms_per_step': cand['ms_per_step'],
            'speedup': speedup, 'reference_spread': spread, 'reference': source or 'frozen',
            'comparison': 'candidate run against the frozen reference timing (the final evaluation times '
                          'reference -> candidate -> reference)', 'runs': runs,
            'library_versions': {'candidate': cand['fingerprint']['versions']},
            **({'notes': cand['notes']} if cand.get('notes') else {})}
    bar = max(CFG['benefit']['min_speedup'], 1 + 2 * spread)
    perf['benefit'] = {'speedup': speedup, 'speedup_bar': round(bar, 4), 'passed': speedup > bar,
                       'peak_memory_gib': cand['peak_memory_allocated_gib'],
                       'reference_peak_memory_gib': frozen.get('peak_memory_allocated_gib')}
    return ('ok' if speedup > bar else 'no_benefit'), perf


def benefit(tree, out, cand_env, final=False, paired=False):
    """Gate 1. The agent's check (neither final nor paired): benefit_frozen. The final evaluation (final) and
    `baseline` (paired): reference -> candidate -> reference; the noise rules (a run's window spread, a reference off
    the frozen timing) make it INVALID in the final evaluation only, elsewhere they are reported."""
    if not (final or paired):
        return benefit_frozen(tree, out, cand_env)
    ref_src = check_reference()
    with scratch_dir('lr-') as tmp:
        ref_tree = materialize(ref_src, Path(tmp) / 'ref')
        ref_a = time_tree(ref_tree, out, 'ref_a', noise_gate=final)
        cand = (time_tree(tree, out, 'cand', cand_env=cand_env, candidate=True, noise_gate=final)
                if ref_a['status'] == 'ok' else None)
        ref_b = time_tree(ref_tree, out, 'ref_b', noise_gate=final) if cand and cand['status'] == 'ok' else None
    runs = {'ref_a': ref_a, 'cand': cand, 'ref_b': ref_b}
    if any(v is None or v['status'] != 'ok' for v in runs.values()):
        bad = {k: (v or {}).get('status') for k, v in runs.items()}
        status = 'INVALID' if any(v and v['status'] == 'INVALID' for v in runs.values()) else 'ERROR'
        return status, {'speedup': None, 'runs': runs, 'reason': f'timing runs: {bad}'}
    diffs = {k: fingerprint_diff(ref_a.get('fingerprint'), v.get('fingerprint')) for k, v in runs.items() if k != 'ref_a'}
    versions_seen = {'reference': ref_a['fingerprint']['versions'], 'candidate': cand['fingerprint']['versions']}
    versions_seen['differ'] = versions_seen['reference'] != versions_seen['candidate']
    bad = {k: d for k, d in diffs.items() if d}
    if bad:
        return 'ERROR', {'speedup': None, 'runs': runs, 'fingerprint_diffs': diffs, 'library_versions': versions_seen,
                         'reason': f'timing runs not comparable: fingerprint differs from ref_a in {bad}'}
    drift = CFG['timing']['max_reference_drift']
    frozen = frozen_reference_timing()
    notes = []
    for k in ('ref_a', 'ref_b'):
        dev = abs(runs[k]['ms_per_step'] / frozen['ms_per_step'] - 1)
        if dev > drift:
            off = f'{k} is {dev:.1%} off the frozen reference timing (> {drift:.0%}): the machine was not in its usual state'
            if final:
                return 'INVALID', {'speedup': None, 'runs': runs, 'reason': off + '; rerun later'}
            notes.append(off + ' (reported; only the final check rejects it)')
    perf = {**compare_timing(ref_a, cand, ref_b), 'runs': runs, 'fingerprint_diffs': diffs,
            'library_versions': versions_seen, **({'notes': notes} if notes else {})}
    bar = max(CFG['benefit']['min_speedup'], 1 + 2 * perf['reference_spread'])
    ref_mem = (ref_a['peak_memory_allocated_gib'] + ref_b['peak_memory_allocated_gib']) / 2
    perf['benefit'] = {'speedup': perf['speedup'], 'speedup_bar': round(bar, 4), 'passed': perf['speedup'] > bar,
                       'peak_memory_gib': cand['peak_memory_allocated_gib'],
                       'reference_peak_memory_gib': round(ref_mem, 2)}
    return ('ok' if perf['speedup'] > bar else 'no_benefit'), perf


def statistical(tree, out, cand_env, correctness, performance):
    """Gate 3 and its verdict (statistical() of every check and of check --n1)."""
    traj = trajectories(tree, out, cand_env)
    correctness = dict(correctness, stage='statistical', trajectory=traj)
    if traj['verdict'] == 'equivalent':
        return 'PASS', dict(correctness, passed=True, verdict='equivalent'), performance
    if traj['verdict'] == 'error':
        return 'ERROR', dict(correctness, passed=False, verdict='error', note=traj.get('reason')), performance
    return 'INCORRECT', dict(correctness, passed=False, verdict='different',
                             note=f'an endpoint\'s bias on seed {traj["seeds"]} is beyond {CFG["trajectory"]["sigma_k"]} '
                                  f'x the reference\'s seed sigma'), performance


def evaluate(tree, out, skip_timing=False, cand_env=None, quick=False, final=True, n1=False):
    """Gates 0 -> 1 -> 2 -> 3 (n1: gates 0 and 3 only). Returns (status, correctness, performance)."""
    cand_env = cand_env or {}
    failed = coverage(tree, out, cand_env)
    if failed:
        return failed[0], failed[1], None
    if n1:
        # check --n1: after a check whose gate 2 was not bitwise identical, gate 3 alone (no timing, no records).
        return statistical(tree, out, cand_env, {'gates_run': [0, 3], 'note': 'check --n1: gate 3 only; gates 1 and '
                                                 '2 not run (the final evaluation runs every gate)'}, None)
    if quick:
        num = numerics(tree, out, cand_env)
        if num.get('errors') or num['status'] not in ('PASS', 'NOT_BITWISE', 'INCORRECT'):
            return 'ERROR', {'passed': False, 'stage': 'structure', 'numerics': num}, None
        if num['exact_failures']:
            return 'INCORRECT', {'passed': False, 'stage': 'structure', 'verdict': 'structure', 'numerics': num}, None
        return 'PASS', {'passed': True, 'stage': 'structure (quick self-test: gates 1 and 3 not run)',
                        'bitwise': num['bitwise'], 'measured_tier': num.get('measured_tier'), 'numerics': num}, None
    performance = None
    if not skip_timing:
        state, performance = benefit(tree, out, cand_env, final=final)
        if state in ('ERROR', 'INVALID'):
            return state, {'passed': False, 'stage': 'benefit',
                           'note': (performance or {}).get('reason') or 'timing failed'}, performance
        if state == 'no_benefit':
            measured = performance.pop('speedup')
            performance['measured_speedup'] = measured
            return 'UNAVAILABLE', {'passed': None, 'stage': 'benefit', 'verdict': 'no_benefit',
                                   'note': 'not faster than the reference: rejected without a precision evaluation'}, \
                performance
    num = numerics(tree, out, cand_env)
    tier = num.get('measured_tier')
    correctness = {'stage': 'bitwise', 'measured_tier': tier, 'numerics': num}
    if num.get('errors') or num['status'] not in ('PASS', 'NOT_BITWISE', 'INCORRECT'):
        return 'ERROR', dict(correctness, passed=False, note='numerics could not compare the records'), performance
    if num['exact_failures']:
        return 'INCORRECT', dict(correctness, passed=False, verdict='structure',
                                 note='exact checks failed: the candidate does different work'), performance
    status, corr, perf = statistical(tree, out, cand_env, correctness, performance)
    if num['bitwise'] and status == 'PASS':
        # gate 2 judged the deterministic path bitwise; gate 3 ran anyway on the production (timed) path
        return 'PASS', dict(corr, verdict='bitwise', measured_tier=tier), perf
    return status, corr, perf


# ---------------------------------------------------------------- commands

def cmd_check(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.final and not final_allowed():
        raise SystemExit('check --final is the final evaluation\'s (run by the harness after your time is up) and is '
                         'not available here. `check` runs every gate the final evaluation runs; its timing reads one '
                         'run against the frozen reference timing instead of reference -> candidate -> reference.')
    if args.n1 and (args.final or args.quick or args.skip_timing):
        raise SystemExit('check --n1 runs gates 0 and 3 only; it takes no --final, --quick or --skip-timing')
    evaluator = evaluator_sha256()
    inputs = [guarded_inputs()]
    cand_env = candidate_environment(args.config)
    with scratch_dir('lc-') as tmp:
        tree = materialize(args.tree, Path(tmp) / 'tree')
        digest = tree_digest(tree)
        try:
            status, correctness, performance = evaluate(tree, out, skip_timing=args.skip_timing, cand_env=cand_env,
                                                        quick=args.quick, final=args.final, n1=args.n1)
        except (Exception, SystemExit) as error:
            status, performance = 'ERROR', None
            correctness = {'passed': False, 'stage': 'evaluation_aborted',
                           'note': f'the evaluation stopped before a verdict: {type(error).__name__}: {error}'}
        result = {'tree': str(Path(args.tree).resolve()), 'config': str(Path(args.config).resolve()) if args.config
                  else None, 'source_digest': digest, 'candidate_environment': cand_env, 'status': status,
                  'quick': args.quick, 'skip_timing': args.skip_timing, 'n1': args.n1, 'final': args.final,
                  'correctness': correctness, 'performance': performance or {'speedup': None}}
        source_end_check(result, tree)
    result.update(cost=cost(), inputs=inputs + [verify_inputs('after')], evaluator_sha256=evaluator)
    if not result['inputs'][-1]['ok']:
        result['status'] = 'ERROR'
    if evaluator_sha256() != evaluator:
        result['status'] = 'ERROR'
        result['reason'] = 'the evaluator changed during the evaluation'
    if result['status'] != 'PASS' and result['performance'].get('speedup') is not None:
        result['performance']['measured_speedup'] = result['performance'].pop('speedup')
        result['performance']['speedup'] = None
    (out / 'check.json').write_text(json.dumps(result, indent=1) + '\n')
    log(f'check {result["status"]}; speedup {result["performance"].get("speedup")}; wrote {out / "check.json"}')


def cmd_time(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = [guarded_inputs()]
    reference = None if args.freeze else ensure_reference_timing(out, args.tree)
    with scratch_dir('ltt-') as tmp:
        tree = materialize(args.tree, Path(tmp) / 'tree')
        digest = tree_digest(tree)
        res = time_tree(tree, out, args.label, cand_env=candidate_environment(args.config),
                        candidate=under_test(args.tree))
        res['source_digest'] = digest
        if reference:
            res['reference_timing'] = reference
        source_end_check(res, tree)
    inputs.append(verify_inputs('after'))
    if not inputs[-1]['ok']:
        res['status'] = 'ERROR'
    (out / f'time_{args.label}.json').write_text(json.dumps({**res, 'inputs': inputs, 'cost': cost()}, indent=1) + '\n')
    frozen = path(CFG['timing']['frozen'])
    if args.freeze and res['status'] == 'ok':
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_text(json.dumps({k: res[k] for k in FROZEN_KEYS}, indent=1) + '\n')
        log(f'froze the reference timing in {frozen} (one run; freeze-timing replaces it with the runs\' mean)')
    log(json.dumps({k: res.get(k) for k in ('status', 'ms_per_step', 'speedup_vs_frozen_reference',
                                            'beyond_reference_noise', 'reference_noise', 'window_spread',
                                            'peak_memory_allocated_gib', 'problems', 'notes')}))


def cmd_freeze_timing(args):
    """Formulation only: the frozen reference timing from the reference runs time_<label>.json in --out."""
    out = Path(args.out).resolve()
    results = []
    for label in args.labels:
        res = json.loads((out / f'time_{label}.json').read_text())
        if res['status'] != 'ok':
            raise SystemExit(f'time_{label}.json: {res["status"]} {res.get("problems") or res.get("reason")}')
        results.append(res)
    frozen = freeze_reference_timing(results)
    target = path(CFG['timing']['frozen'])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(frozen, indent=1) + '\n')
    log(f'froze the reference timing from {len(results)} runs in {target}: {frozen["ms_per_step"]} ms/step, '
        f'spread {frozen["spread"]}')


def nsys_summary(rep, out_md):
    """Kernel and CUDA API summaries of an nsys report (nsys stats), for profile_<label>.md."""
    nsys = shutil.which('nsys') or CFG.get('nsys')
    lines = []
    for report in ('cuda_gpu_kern_sum', 'cuda_gpu_mem_time_sum', 'nvtx_sum'):
        r = subprocess.run([nsys, 'stats', '--report', report, '--format', 'csv', '--force-export=true', str(rep)],
                           capture_output=True, text=True, timeout=1800)
        rows = [x for x in r.stdout.splitlines() if x.count(',') >= 3]
        lines.append(f'## {report}\n')
        lines += [f'    {x[:240]}' for x in rows[:41]]
        lines.append('')
    Path(out_md).write_text('\n'.join(lines) + '\n')


def detail_summary(traces, out_md, steps):
    """Top operators and kernels of the rank-0 PyTorch profiler trace."""
    import collections
    trace = sorted(traces)[0]
    events = json.loads(Path(trace).read_text()).get('traceEvents', [])
    kern, ops = collections.Counter(), collections.Counter()
    for e in events:
        if e.get('cat') == 'kernel':
            kern[e.get('name', '')[:160]] += e.get('dur', 0)
        elif e.get('cat') == 'cpu_op':
            ops[e.get('name', '')[:120]] += e.get('dur', 0)
    total = sum(kern.values()) or 1
    lines = [f'# detail profile, rank 0, {steps} step(s)', '', f'GPU kernel time: {total / 1e3 / steps:.0f} ms/step', '',
             '| kernel | ms/step | share |', '|---|---|---|']
    lines += [f'| {n} | {d / 1e3 / steps:.1f} | {d / total:.1%} |' for n, d in kern.most_common(40)]
    lines += ['', '| cpu op (inclusive) | ms/step |', '|---|---|']
    lines += [f'| {n} | {d / 1e3 / steps:.1f} |' for n, d in ops.most_common(40)]
    Path(out_md).write_text('\n'.join(lines) + '\n')


def cmd_profile(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    steps = args.steps or (1 if args.detail else 2)
    results = {'source': str(Path(args.tree).resolve()), 'config': args.config, 'inputs': [guarded_inputs()]}
    env = candidate_environment(args.config)
    with scratch_dir('lp-') as tmp:
        tree = materialize(args.tree, Path(tmp) / 'tree')
        results['source_digest'] = tree_digest(tree)
        res = time_tree(tree, out, args.label, cand_env=env, profile_steps=steps, detail=args.detail,
                        candidate=under_test(args.tree))
        results['run'] = res
        source_end_check(results, tree)
    md = out / f'profile_{args.label}.md'
    try:
        if res['status'] == 'ok' and args.detail:
            detail_summary(res['traces'], md, steps)
        elif res['status'] == 'ok':
            nsys_summary(res['trace'], md)
    except Exception as error:      # the summary is a convenience; the trace is the result
        results['summary_error'] = f'{type(error).__name__}: {error}'
    results['inputs'].append(verify_inputs('after'))
    changed = results.get('source_digest_end') != results['source_digest']
    results['status'] = 'ERROR' if changed or not results['inputs'][-1]['ok'] else res['status']
    results['cost'] = cost()
    (out / f'profile_{args.label}.json').write_text(json.dumps(results, indent=1) + '\n')
    log(f'profile {results["status"]}; {md if md.exists() else "no summary"}')


def cmd_sweep(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    labels = [Path(c).stem for c in args.config]
    if len(set(labels)) != len(labels):
        raise SystemExit(f'config file names must be distinct (they label the runs): {labels}')
    result = {'note': 'screen, one timing run per config between two reference runs: not a score', 'complete': False,
              'started': datetime.now().isoformat(timespec='seconds'), 'reference': {}, 'configs': {}}

    def save():
        result['cost'] = cost()
        (out / 'sweep.json').write_text(json.dumps(result, indent=1) + '\n')

    def brief(res):
        keep = ('status', 'reason', 'ms_per_step', 'speedup_vs_frozen_reference', 'window_spread',
                'peak_memory_allocated_gib', 'process_wall_s', 'problems', 'signatures', 'warning')
        return {**{k: res[k] for k in keep if res.get(k) is not None}, 'fingerprint': res.get('fingerprint')}

    result['inputs'] = [guarded_inputs()]
    with scratch_dir('ls-') as tmp:
        ref_tree = materialize(check_reference(), Path(tmp) / 'ref')
        tree = materialize(args.tree, Path(tmp) / 'tree')
        result['source_digest'] = tree_digest(tree)
        result['reference']['a'] = brief(time_tree(ref_tree, out, 'sweep_ref_a'))
        save()
        for cfg, label in zip(args.config, labels):
            try:
                env = candidate_environment(cfg)
                res = time_tree(tree, out, f'sweep_{label}', cand_env=env, candidate=True)
            except (Exception, SystemExit) as error:
                env, res = None, {'status': 'ERROR', 'reason': f'{type(error).__name__}: {error}'}
            row = {'config': str(Path(cfg).resolve()), 'environment': env, **brief(res)}
            diff = fingerprint_diff(result['reference']['a'].get('fingerprint'), row.get('fingerprint'))
            if diff and row['status'] == 'ok':
                row.update(status='INVALID', reason=f'fingerprint differs from reference a in {diff}')
            result['configs'][label] = row
            log(f'{label}: {res["status"]} {res.get("ms_per_step")} ms')
            save()
        result['reference']['b'] = brief(time_tree(ref_tree, out, 'sweep_ref_b'))
        source_end_check(result, tree)
    refs = [r['ms_per_step'] for r in result['reference'].values() if isinstance(r, dict) and r.get('status') == 'ok']
    if refs:
        ref = statistics.mean(refs)
        result['reference'].update(ms_per_step=round(ref, 1),
                                   spread=round((max(refs) - min(refs)) / ref, 4) if len(refs) == 2 else None)
        for row in result['configs'].values():
            row['speedup_vs_reference'] = round(ref / row['ms_per_step'], 4) if row.get('status') == 'ok' else None
    result['inputs'].append(verify_inputs('after'))
    result.update(complete=True, finished=datetime.now().isoformat(timespec='seconds'))
    save()
    log(f'wrote {out / "sweep.json"}')


def cmd_baseline(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref_src = check_reference()
    inputs = [guarded_inputs()]
    with scratch_dir('lb-') as tmp:
        tree = materialize(ref_src, Path(tmp) / 'tree')
        state, performance = benefit(tree, out, {}, paired=True)
        status, correctness, _ = evaluate(tree, out, skip_timing=True) if not args.timing_only else ('-', None, None)
    result = {'baseline': True, 'note': 'reference evaluated as its own candidate', 'benefit_state': state,
              'status': status, 'correctness': correctness, 'performance': performance, 'cost': cost(),
              'inputs': inputs + [verify_inputs('after')]}
    (out / 'baseline.json').write_text(json.dumps(result, indent=1) + '\n')
    log(f'baseline {status}; wrote {out / "baseline.json"}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    k = sub.add_parser('check')
    k.add_argument('--tree', required=True)
    k.add_argument('--out', required=True)
    k.add_argument('--config', help='switch file {"environment": {...}}')
    k.add_argument('--quick', action='store_true', help='gate 0 and the structure record only (no timing, no stats)')
    k.add_argument('--skip-timing', action='store_true')
    k.add_argument('--final', action='store_true', help=argparse.SUPPRESS)   # the final evaluation only (final_allowed)
    k.add_argument('--n1', action='store_true',
                   help='gates 0 and 3 only (the statistical gate), after a check whose gate 2 was not bitwise '
                        'identical: no timing, no records')
    t = sub.add_parser('time')
    t.add_argument('--tree', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--label', default='tree')
    t.add_argument('--config')
    t.add_argument('--freeze', action='store_true', help='formulation only: write this run as the frozen reference')
    f = sub.add_parser('profile')
    f.add_argument('--tree', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--label', default='tree')
    f.add_argument('--config')
    f.add_argument('--steps', type=int, help='steps to capture after the warmup (default 2; 1 with --detail)')
    f.add_argument('--detail', action='store_true', help='PyTorch profiler instead of nsys')
    w = sub.add_parser('sweep')
    w.add_argument('--tree', required=True)
    w.add_argument('--config', action='append', required=True)
    w.add_argument('--out', required=True)
    z = sub.add_parser('freeze-timing')
    z.add_argument('--out', required=True, help='directory with the reference runs time_<label>.json')
    z.add_argument('--labels', nargs='+', required=True)
    b = sub.add_parser('baseline')
    b.add_argument('--out', required=True)
    b.add_argument('--timing-only', action='store_true')
    args = p.parse_args()
    {'check': cmd_check, 'time': cmd_time, 'profile': cmd_profile, 'sweep': cmd_sweep,
     'baseline': cmd_baseline, 'freeze-timing': cmd_freeze_timing}[args.cmd](args)


if __name__ == '__main__':
    main()
