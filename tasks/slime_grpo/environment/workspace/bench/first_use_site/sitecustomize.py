"""Opt-in (SLIME_GRPO_BENCH_FIRST_USE_ALL=1, which puts this directory on PYTHONPATH in bench/launch.sh): every
Python process of the run -- the Megatron ranks, the rollout manager and the sglang engines -- appends its Triton
autotuning and compilation to $SLIME_GRPO_BENCH_DIR/first_use.<pid>.jsonl, the records bench.py sums per timed
window (first_use_windows). The judge's own hook (bench/train_hook.py) only reaches the Megatron processes, so this
is how the generation path is measured; it is off in ordinary runs and changes no arithmetic when on.

Defensive by construction: a failure anywhere leaves the process exactly as it was, and a sitecustomize further
along sys.path still runs.
"""
import os
import sys

_DONE = [False]


def _chain():
    """Run the next sitecustomize on the path, if the image has one (we shadow it)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in sys.path:
        if not isinstance(d, str) or not d:
            continue
        try:
            if os.path.abspath(d) == here:
                continue
            p = os.path.join(d, 'sitecustomize.py')
            if not os.path.isfile(p):
                continue
        except Exception:
            continue
        import importlib.util
        spec = importlib.util.spec_from_file_location('sitecustomize_chained', p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return


def _patch():
    """Wrap Triton's autotuner and compiler, once Triton is imported. Returns True when installed."""
    import json
    import time
    try:
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import JITFunction
    except Exception:
        return False
    if getattr(Autotuner._bench, '_rbb_first_use', False):
        return True
    out = os.path.join(os.environ['SLIME_GRPO_BENCH_DIR'], 'first_use.%d.jsonl' % os.getpid())
    depth = [0]

    def name_of(obj):
        fn = obj
        for _ in range(8):
            if not hasattr(fn, 'fn'):
                break
            fn = fn.fn
        return getattr(fn, '__name__', type(fn).__name__)

    def emit(kind, obj, t0):
        try:
            with open(out, 'a') as f:
                f.write(json.dumps({'kind': kind, 'name': name_of(obj), 't0': t0, 't1': time.time()}) + '\n')
        except Exception:
            pass

    bench0, compile0 = Autotuner._bench, JITFunction._do_compile

    def _bench(self, *a, **k):
        t0 = time.time()
        depth[0] += 1
        try:
            return bench0(self, *a, **k)
        finally:
            depth[0] -= 1
            emit('autotune', self, t0)

    def _do_compile(self, *a, **k):
        t0 = time.time()
        try:
            return compile0(self, *a, **k)
        finally:
            if depth[0] == 0:
                emit('compile', self, t0)

    _bench._rbb_first_use = True
    Autotuner._bench = _bench
    JITFunction._do_compile = _do_compile
    return True


def _install():
    if os.environ.get('SLIME_GRPO_BENCH_FIRST_USE_ALL') != '1' or not os.environ.get('SLIME_GRPO_BENCH_DIR'):
        return
    if _patch():          # Triton already imported (rare at startup)
        _DONE[0] = True
        return
    import builtins       # else: patch as soon as something imports Triton, then step out of the way
    orig = builtins.__import__

    def _imp(name, *a, **k):
        mod = orig(name, *a, **k)
        if not _DONE[0] and name.split('.')[0] == 'triton':
            try:
                if _patch():
                    _DONE[0] = True
                    builtins.__import__ = orig
            except Exception:
                _DONE[0] = True
                builtins.__import__ = orig
        return mod

    builtins.__import__ = _imp


try:
    _chain()
except Exception:
    pass
try:
    _install()
except Exception:
    pass
