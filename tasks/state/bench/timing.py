"""Iteration timing of a State training tree's real entry point (`state tx train`), nothing patched.

Run as:  python bench/timing.py --tree <tree> --out <prefix> --warmup W --window N --repeats R -- <hydra overrides>

The entry point is imported from the tree (its `src/` is put first on sys.path) and run with its own argv, so
every part of the candidate's training path is exercised. One iteration of `Trainer.fit` is one optimizer step:
State configures a single Adam optimizer over all parameters and steps it once per `training_step`
(`state/tx/models/base.py: configure_optimizers`), with `accumulate_grad_batches = 1`.

The first `warmup` iterations are skipped (model load, first compile, allocator warmup), then `repeats` windows
of `window` iterations are timed. Window edges are bracketed by torch.cuda.synchronize(); nothing else
synchronizes, so the measurement does not add a host sync per iteration. Each iteration's CUDA and CPU random
generator state is fingerprinted host-side (no device sync): the bench compares that sequence with the frozen
reference, so every timed run must draw the same random numbers, in the same order, as the reference.

After the last window the hook raises StopTiming (a BaseException, so `except Exception` in the training code
cannot swallow it) and the process exits through os._exit after writing <prefix>.r<rank>.json.
"""
import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

T_START = time.perf_counter()
BENCH = Path(__file__).resolve().parent


class StopTiming(BaseException):
    """BaseException, so `except Exception` blocks in the training code cannot swallow it."""


def rng_fingerprint():
    h = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_initialized():
        h.update(torch.cuda.get_rng_state().numpy().tobytes())
    return h.hexdigest()[:16]


def compile_count():
    """Graphs torch.compile has compiled so far in this process (0 if dynamo was never imported)."""
    utils = sys.modules.get('torch._dynamo.utils')
    if utils is None:
        return 0
    return utils.counters['stats']['unique_graphs'] + utils.counters['aot_autograd']['total']


def kill_descendants():
    """End every process this one started (compile workers, data-loader workers) before os._exit; a profiler
    such as nsys waits for the whole process tree."""
    children = {}
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            with open(f'/proc/{d}/stat') as f:
                ppid = int(f.read().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(d))
    stack, found = [os.getpid()], []
    while stack:
        for c in children.get(stack.pop(), []):
            found.append(c)
            stack.append(c)
    for p in found:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--window', type=int, default=20)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--profile', default=None, help='torch | nsys | None')
    ap.add_argument('--profile-iters', type=int, default=5)
    ap.add_argument('overrides', nargs='*', help='Hydra overrides, including the tx train subcommand')
    a = ap.parse_args()

    tree = Path(a.tree).resolve()
    sys.path.insert(0, str(tree / 'src'))
    os.chdir(tree)

    profiled = a.profile is not None
    prof = None
    if a.profile == 'torch':
        from torch.profiler import profile as _profile, ProfilerActivity
        prof = _profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=False, profile_memory=False)
    size = a.profile_iters if profiled else a.window
    repeats = 1 if profiled else a.repeats
    first_timed = a.warmup
    last_step = first_timed + size * repeats          # exclusive

    st = {'k': 0, 't_last': None, 'win_t0': None, 'win_compiles': None,
          'windows': [], 'rng': [], 'samples': [], 'first_step_t': None, 'dtype_traced': False}

    def hook(*_args, **_kwargs):
        i = st['k']
        now = time.perf_counter()
        st['k'] = i + 1
        st['rng'].append(rng_fingerprint())
        if st['first_step_t'] is None:
            st['first_step_t'] = now - T_START
        if i >= first_timed:
            if (i - first_timed) % size == 0:           # window start
                torch.cuda.synchronize()
                st['win_t0'] = time.perf_counter()
                st['win_compiles'] = compile_count()
            if (i - first_timed) % size == size - 1:    # window end
                torch.cuda.synchronize()
                win_t1 = time.perf_counter()
                st['windows'].append({
                    'k': (i - first_timed) // size,
                    'seconds': win_t1 - st['win_t0'],
                    'compiles_in_window': compile_count() - st['win_compiles'],
                })
        if st['k'] >= last_step:
            raise StopTiming()

    register_optimizer_step_post_hook(hook)

    # Hand the tree's own CLI its own argv: `state tx train <overrides>`.
    sys.argv = ['state', *a.overrides]
    status, error = 'ok', None
    try:
        if prof is not None:
            prof.__enter__()
        from state.__main__ import main as state_main
        state_main()
    except StopTiming:
        pass
    except BaseException as exc:                        # noqa: BLE001 - any failure is reported, not swallowed
        import traceback
        status, error = 'error', f'{type(exc).__name__}: {exc}\n{traceback.format_exc()[-4000:]}'
    if prof is not None:
        prof.__exit__(None, None, None)

    if prof is not None:
        try:
            outdir = Path(a.out).parent
            (outdir / f'{Path(a.out).name}.profile.txt').write_text(
                prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=60))
            prof.export_chrome_trace(str(outdir / f'{Path(a.out).name}.trace.json'))
        except Exception as _e:  # profiling output must never fail the run
            print('profile export failed:', _e, file=sys.stderr)
    alloc = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
    reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
    rank = int(os.environ.get('RANK', '0'))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / f'{out.name}.r{rank}.json').write_text(json.dumps({
        'status': status, 'error': error, 'rank': rank, 'world_size': int(os.environ.get('WORLD_SIZE', '1')),
        'windows': st['windows'], 'rng_fingerprints': st['rng'], 'steps_completed': st['k'],
        'first_step_s': round(st['first_step_t'] or 0, 2),
        'startup_to_first_iteration_s': round(st['first_step_t'] or 0, 2),
        'peak_memory_allocated_bytes': alloc, 'peak_memory_reserved_bytes': reserved,
        'torch': torch.__version__, 'compile_count_total': compile_count(),
    }, indent=1) + '\n')
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    kill_descendants()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if status == 'ok' else 1)


if __name__ == '__main__':
    main()
