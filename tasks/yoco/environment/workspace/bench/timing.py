"""Update timing of a source tree's real YOCO entry point (`<tree>/yoco/train.py` run as __main__), nothing patched.

Run under `torchrun` with `<tree>/yoco` as the working directory:

  torchrun --nproc-per-node=2 ... <bench>/timing.py --out <prefix> [options] -- <train.py arguments>

sys.path[0] becomes `<tree>/yoco` (as for `python train.py`), so the tree's `models`, `tasks`, `criterions` and
anything else it ships are what gets imported.

One iteration = one update of fairseq's training loop. Iterations are counted with torch's global optimizer-step
post hook: the first optimizer object that ever steps defines the iteration (in the reference the shard-local
FusedAdam inside fairscale's OSS; one step per update). The hook fires right after that optimizer's step, so the
time between two hooks is exactly one full update cycle (the rest of update k after its Adam step: parameter
broadcast, fp32->bf16 copy, LR update, logging bookkeeping, then data fetch, forward, loss, backward, gradient
all-reduce, clipping and the Adam step of update k+1).

The first `warmup` iterations are skipped (model build, compilation, allocator and loader warmup), then `repeats`
windows of `window` iterations are timed. Window edges are bracketed by torch.cuda.synchronize(); nothing else
synchronizes. Each iteration is an NVTX range ("iter <i>"). After the last window the hook raises StopTiming and the
run ends. At every iteration end the hook also fingerprints the CPU and CUDA random generators (host-side state, no
device sync) and reads torch.cuda.max_memory_allocated() (fairseq resets the peak statistics once per update, after
the optimizer step, so the value is the peak of the update so far).

--profile nsys: cudaProfilerStart at the start of the first window, cudaProfilerStop after `profile_iters`
iterations. --profile torch: the same span under torch.profiler; fairseq's own record_function ranges ("forward",
"backward", "reduce-grads", "multiply-grads", "clip-grads", "optimizer", fairscale's "fairscale::oss::*") plus
ranges this script adds around the top-level model blocks (embedding, self-decoder, cross-decoder, final norm,
output projection; found by attribute names, best effort) are summarized; rank 0 writes a summary and a Chrome trace.

Before the training code starts, each rank is pinned to its share of the CPUs (bench/cpu_affinity.py: the cores on
its GPU's NUMA node, split between the ranks there; torch's intra-op threads = the share's physical cores).

Each rank writes <prefix>.r<rank>.json; the caller merges ranks (max over ranks per window).
"""
import argparse
import gc
import hashlib
import json
import os
import runpy
import signal
import sys
import time
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

T_START = time.perf_counter()
BENCH = Path(__file__).resolve().parent


def compile_count():
    """Graphs torch.compile has compiled so far in this process (0 if dynamo was never imported)."""
    utils = sys.modules.get('torch._dynamo.utils')
    if utils is None:
        return 0
    return utils.counters['stats']['unique_graphs'] + utils.counters['aot_autograd']['total']


def rng_fingerprint():
    h = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_initialized():
        h.update(torch.cuda.get_rng_state().numpy().tobytes())
    return h.hexdigest()[:16]


class StopTiming(BaseException):
    """BaseException, so `except Exception` / `except RuntimeError` blocks in the training code cannot swallow it."""


def kill_descendants():
    """End every process this one started (the infinibatch prefetch process, compile workers) before os._exit."""
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
    for pid in found:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return len(found)


PHASE_RANGES = ('forward', 'backward', 'reduce-grads', 'multiply-grads', 'clip-grads', 'optimizer',
                'fairscale::oss::optim_step', 'fairscale::oss::refresh_trainable')
BLOCKS = (('embed_tokens', 'embedding'), ('self_decoder', 'self_decoder'), ('cross_decoder', 'cross_decoder'),
          ('layer_norm', 'final_norm'), ('output_projection', 'output_projection'))


def _is_gpu_annotation(name):
    return (name.startswith(('## ', 'train_step-', 'nccl:', 'Optimizer.', 'ProfilerStep', 'fairscale::', 'c10d::'))
            or name in PHASE_RANGES)


class TorchProfile:
    def __init__(self, out):
        self.out, self.prof, self.handles = out, None, []

    def _find_core(self):
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.nn.Module) and hasattr(obj, 'self_decoder') and hasattr(obj, 'cross_decoder'):
                    return obj
            except Exception:  # noqa: BLE001 -- objects with odd __getattr__
                continue
        return None

    def start(self):
        core = self._find_core()
        if core is not None:
            for attr, label in BLOCKS:
                mod = getattr(core, attr, None)
                if isinstance(mod, torch.nn.Module):
                    self._wrap(mod, label)
            for i, layer in enumerate(getattr(core.self_decoder, 'layers', [])):
                for attr in ('mixer', 'ffn'):
                    m = getattr(layer, attr, None)
                    if isinstance(m, torch.nn.Module):
                        self._wrap(m, f'self.{attr}')
            for i, layer in enumerate(getattr(core.cross_decoder, 'layers', [])):
                for attr in ('mixer', 'ffn'):
                    m = getattr(layer, attr, None)
                    if isinstance(m, torch.nn.Module):
                        self._wrap(m, f'cross.{attr}')
        from torch.profiler import ProfilerActivity, profile
        self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False,
                            with_stack=False, profile_memory=False)
        self.prof.__enter__()

    def _wrap(self, mod, label):
        ctx = {}

        def pre(_m, _a):
            r = torch.autograd.profiler.record_function(f'## {label}')
            r.__enter__()
            ctx.setdefault('stack', []).append(r)

        def post(_m, _a, _o):
            if ctx.get('stack'):
                ctx['stack'].pop().__exit__(None, None, None)
        self.handles.append(mod.register_forward_pre_hook(pre))
        self.handles.append(mod.register_forward_hook(post))

    def stop(self, span_s, iterations):
        self.prof.__exit__(None, None, None)
        for h in self.handles:
            h.remove()
        rank = int(os.environ.get('RANK', '0'))
        if rank != 0:
            return
        kernels, annotations, intervals, ops = {}, {}, [], {}
        for e in self.prof.events():
            dev = str(getattr(e, 'device_type', ''))
            if dev.endswith('CUDA') and _is_gpu_annotation(e.name):
                continue        # GPU-side copies of record_function ranges: not kernels, they would double count
            if dev.endswith('CUDA'):
                k = kernels.setdefault(e.name, [0.0, 0])
                k[0] += e.time_range.elapsed_us()
                k[1] += 1
                intervals.append((e.time_range.start, e.time_range.end))
                continue
            name = e.name
            label = None
            if name.startswith('## '):
                label = name[3:]
            elif name in PHASE_RANGES:
                label = f'phase:{name}'
            elif name.startswith('train_step-'):
                label = 'phase:train_step'
            if label is not None:
                a = annotations.setdefault(label, [0.0, 0, 0.0])
                a[0] += e.cuda_time_total
                a[1] += 1
                a[2] += e.cpu_time_total
            elif name.startswith('aten::') or name.startswith('c10d::') or name.startswith('nccl'):
                o = ops.setdefault(name, [0.0, 0])
                o[0] += e.self_cpu_time_total
                o[1] += 1
        intervals.sort()
        busy, cur = 0.0, None
        for s, t in intervals:
            if cur is None or s > cur[1]:
                if cur:
                    busy += cur[1] - cur[0]
                cur = [s, t]
            else:
                cur[1] = max(cur[1], t)
        if cur:
            busy += cur[1] - cur[0]
        total_kernel = sum(v[0] for v in kernels.values())
        summary = {
            'span_s': span_s, 'iterations': iterations,
            'gpu_busy_s': busy / 1e6, 'gpu_busy_fraction': (busy / 1e6) / span_s if span_s else None,
            'kernel_time_s': total_kernel / 1e6,
            'kernels': [{'name': n[:300], 'total_ms': round(v[0] / 1000, 3), 'calls': v[1],
                         'share_of_kernel_time': round(v[0] / total_kernel, 4) if total_kernel else None}
                        for n, v in sorted(kernels.items(), key=lambda kv: -kv[1][0])[:120]],
            'annotations': {n: {'cuda_ms': round(v[0] / 1000, 2), 'calls': v[1], 'cpu_ms': round(v[2] / 1000, 2)}
                            for n, v in sorted(annotations.items())},
            'cpu_ops_self_time': [{'name': n, 'self_cpu_ms': round(v[0] / 1000, 2), 'calls': v[1]}
                                  for n, v in sorted(ops.items(), key=lambda kv: -kv[1][0])[:40]],
        }
        Path(f'{self.out}.torchprof.json').write_text(json.dumps(summary, indent=1) + '\n')
        self.prof.export_chrome_trace(f'{self.out}.trace.json')


class IterTimer:
    def __init__(self, warmup, window, repeats, profile=None, profile_iters=5, out=None):
        self.warmup, self.window, self.repeats, self.profile = warmup, window, repeats, profile
        self.marks = {warmup + k * window - 1: k for k in range(repeats + 1)}   # iteration index -> edge k
        self.last = max(self.marks)
        if profile:
            self.marks = {warmup - 1: 0}
            self.last = warmup - 1 + profile_iters
        self.profile_iters = profile_iters
        self.iter_opt = None            # id of the optimizer whose steps define iterations
        self.other_steps = 0            # steps of any other optimizer object (e.g. the OSS wrapper)
        self.it = -1                    # iterations finished, minus one
        self.edges = {}                 # k -> (perf_counter, wall, compiled graphs)
        self.first_iter_s = None
        self.nvtx_open = False
        self.ends = []                  # per finished iteration: perf_counter at its end (no sync, host time)
        self.rng = []                   # per finished iteration: RNG fingerprint at its end
        self.peak = []                  # per finished iteration: max_memory_allocated at its end
        self.torchprof = TorchProfile(out) if profile == 'torch' else None
        self.profile_t0 = None

    def _open(self):
        torch.cuda.nvtx.range_push(f'iter {self.it + 1}')
        self.nvtx_open = True

    def __call__(self, opt, _args, _kwargs):
        if self.iter_opt is None:
            self.iter_opt = id(opt)
        if id(opt) != self.iter_opt:
            self.other_steps += 1
            return
        self.it += 1
        self.ends.append(time.perf_counter())
        self.rng.append(rng_fingerprint())
        self.peak.append(torch.cuda.max_memory_allocated())
        if self.nvtx_open:
            torch.cuda.nvtx.range_pop()
            self.nvtx_open = False
        if self.it == 0:
            torch.cuda.synchronize()
            self.first_iter_s = time.perf_counter() - T_START
        if self.profile:
            if self.it == self.warmup - 1:
                torch.cuda.synchronize()
                self.profile_t0 = time.perf_counter()
                self.edges[0] = (self.profile_t0, time.time(), compile_count())
                if self.profile == 'nsys':
                    torch.cuda.cudart().cudaProfilerStart()
                else:
                    self.torchprof.start()
            elif self.it == self.last:
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                self.edges[1] = (t1, time.time(), compile_count())
                if self.profile == 'nsys':
                    torch.cuda.cudart().cudaProfilerStop()
                else:
                    self.torchprof.stop(t1 - self.profile_t0, self.profile_iters)
                raise StopTiming()
            self._open()
            return
        k = self.marks.get(self.it)
        if k is not None:
            torch.cuda.synchronize()
            self.edges[k] = (time.perf_counter(), time.time(), compile_count())
            if self.it == self.last:
                raise StopTiming()
        self._open()

    def report(self):
        windows = []
        n = 1 if self.profile else self.repeats
        size = self.profile_iters if self.profile else self.window
        for k in range(n):
            if k not in self.edges or k + 1 not in self.edges:
                continue
            a, b = self.edges[k], self.edges[k + 1]
            first = self.warmup + k * size
            windows.append({'k': k, 'iterations': size, 'first_iteration': first, 'seconds': b[0] - a[0],
                            'wall_start': a[1], 'wall_end': b[1], 'compiles_in_window': b[2] - a[2],
                            'peak_memory_allocated_bytes': max(self.peak[first:first + size] or [0])})
        return windows


def main():
    if '--' not in sys.argv:
        raise SystemExit('usage: timing.py [options] -- <train.py arguments>')
    split = sys.argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='output prefix; writes <prefix>.r<rank>.json')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--window', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--entry', default='train.py', help='relative to the working directory (<tree>/yoco)')
    parser.add_argument('--profile', choices=('nsys', 'torch'))
    parser.add_argument('--profile-iters', type=int, default=5)
    args = parser.parse_args(sys.argv[1:split])
    train_argv = sys.argv[split + 1:]
    # bench/cpu_affinity.py by path: the bench's directory stays off sys.path
    import importlib.util
    spec = importlib.util.spec_from_file_location('yoco_bench_cpu_affinity', BENCH / 'cpu_affinity.py')
    cpu_affinity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpu_affinity)
    binding = cpu_affinity.bind_local_rank(int(os.environ.get('LOCAL_RANK', '0')),
                                           int(os.environ.get('LOCAL_WORLD_SIZE', os.environ.get('WORLD_SIZE', '1'))))

    timer = IterTimer(args.warmup, args.window, args.repeats, args.profile, args.profile_iters, args.out)
    handle = register_optimizer_step_post_hook(timer)
    entry = Path(args.entry).resolve()
    script_dir = entry.parent
    # as `python train.py` from <tree>/yoco: the script's directory first on sys.path, the bench not on it
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != BENCH]
    sys.path.insert(0, str(script_dir))
    sys.argv = [str(entry), *train_argv]
    rank = int(os.environ.get('RANK', '0'))
    status, error = 'incomplete', None
    try:
        runpy.run_path(str(entry), run_name='__main__')
    except StopTiming:
        status = 'ok'
    except BaseException as exc:  # report, then exit nonzero
        import traceback
        traceback.print_exc()
        status, error = 'error', f'{type(exc).__name__}: {exc}'
    finally:
        handle.remove()
    result = {
        'status': status, 'error': error, 'rank': rank, 'world_size': int(os.environ.get('WORLD_SIZE', '1')),
        'warmup': args.warmup, 'window': args.window, 'repeats': args.repeats, 'profile': args.profile,
        'iterations_seen': timer.it + 1, 'other_optimizer_steps': timer.other_steps,
        'startup_to_first_iteration_s': timer.first_iter_s,
        'peak_memory_allocated_bytes': max(timer.peak or [0]),
        'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
        'windows': timer.report() if status == 'ok' else [],
        'iteration_end_s': [round(t - T_START, 4) for t in timer.ends],
        'rng_fingerprints': timer.rng,
        'train_argv': train_argv, 'cpu_binding': binding,
        'math_flags_at_end': {'matmul.allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                              'cudnn.allow_tf32': torch.backends.cudnn.allow_tf32,
                              'float32_matmul_precision': torch.get_float32_matmul_precision()},
    }
    Path(f'{args.out}.r{rank}.json').write_text(json.dumps(result, indent=1) + '\n')
    sys.stdout.flush()
    sys.stderr.flush()
    kill_descendants()
    os._exit(0 if status == 'ok' else 1)


if __name__ == '__main__':
    main()
