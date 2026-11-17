"""Judge-owned Megatron hook, called before every train step (slime --custom-megatron-before-train-step-hook-path).

Every call (every mode) appends the wall clock to $SLIME_OPD_BENCH_DIR/train_clock.rank<r>.jsonl (the judge's train
step time is before step k -> before step k+1, slowest rank), and the first call writes the parallel layout this
rank runs in (Megatron's parallel state: tp, cp, pp, dp, ep, world) to megatron_layout.rank<r>.json.
With SLIME_OPD_BENCH_RECORD=1 it also records the state (below) to $SLIME_OPD_BENCH_DIR/records.

The state seen before step k is the result of updates 0..k-1, so steps 0, 1, 2 capture the initial state and two
optimizer updates (the second one exercises Adam's moments). Per rank it records:
  - SHA-256 of every bf16 model parameter (names, shapes, dtypes are exact items),
  - SHA-256 of every fp32 master shard and its Adam moments (exp_avg, exp_avg_sq) held by the CPU-offloaded
    HybridDeviceOptimizer (with lr 1e-6 an update is far below one bf16 ulp: only the fp32 masters show it),
  - ||theta_k - theta_0||^2 over the fp32 masters, ||exp_avg_sq||^2, and a CountSketch (dim 65536, fixed hash) of
    theta_k - theta_0 for the direction check.
Digests equal hashlib.sha256 over each tensor's contiguous bytes (as numerics/shapipe.py); CPU tensors are hashed
in place in a thread pool (no copies: the masters and moments are ~27 GB per rank).
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from bench.shapipe import tensor_shas

SKETCH_DIM = 1 << 16
CHUNK = 1 << 26
_INIT = {}


def _masters(optimizer):
    """[(name, fp32 master shard, optimizer state dict)] in a fixed order."""
    out = []
    chain = getattr(optimizer, 'chained_optimizers', None) or [optimizer]
    for ci, o in enumerate(chain):
        inner = getattr(o, 'optimizer', None)
        subs = getattr(inner, 'sub_optimizers', None)
        if subs is not None:
            for si, so in enumerate(subs):
                for pi, p in enumerate(p for g in so.param_groups for p in g['params']):
                    out.append((f'c{ci}.s{si}.p{pi}', p, so.state.get(p, {})))
        else:
            for gi, g in enumerate(inner.param_groups):
                for pi, p in enumerate(g['params']):
                    out.append((f'c{ci}.g{gi}.p{pi}', p, inner.state.get(p, {})))
    return out


def _sha_inplace(t):
    t = t.detach()
    if t.is_cuda:
        t = t.cpu()
    t = t.contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(memoryview(t.numpy())).hexdigest()


def _sketch_add(sk, key, delta):
    """CountSketch of a flat fp32 CPU tensor: bucket and sign from a generator seeded by (key, chunk)."""
    seed0 = int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)
    n = delta.numel()
    for ci, s in enumerate(range(0, n, CHUNK)):
        d = delta[s:s + CHUNK].to('cuda', non_blocking=False).double()
        g = torch.Generator(device='cuda')
        g.manual_seed(seed0 + ci)
        idx = torch.randint(0, SKETCH_DIM, (d.numel(),), device='cuda', generator=g)
        sign = torch.randint(0, 2, (d.numel(),), device='cuda', generator=g).double().mul_(2).sub_(1)
        sk.index_add_(0, idx, d * sign)


_LAYOUT = False
_FIRST_USE = None


def _install_first_use(d):
    """Timing runs only (SLIME_OPD_BENCH_FIRST_USE=1): append every Triton autotune benchmark (Autotuner._bench: one candidate
    config, its compilation included) and every Triton compilation outside one (JITFunction._do_compile: the first use
    of a kernel specialization in this process, a disk-cache load or a real compile) with its wall clock to
    first_use.<pid>.jsonl. The judge sums them per timed window (bench.py first_use_windows); a warmed run's timed
    windows hold about none. Installed at the first train step (before any timed window); changes no arithmetic."""
    global _FIRST_USE
    if _FIRST_USE is not None:
        return
    _FIRST_USE = os.path.join(d, f'first_use.{os.getpid()}.jsonl')
    try:
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import JITFunction
    except Exception:  # no Triton: nothing to watch
        return
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
            with open(_FIRST_USE, 'a') as f:
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

    Autotuner._bench = _bench
    JITFunction._do_compile = _do_compile


def _clock(d, rank, rollout_id, step_id):
    global _LAYOUT
    t = time.time()
    if not _LAYOUT:
        _LAYOUT = True
        from megatron.core import parallel_state as ps
        lay = {'rank': rank, 'world': dist.get_world_size() if dist.is_initialized() else 1,
               'tp': ps.get_tensor_model_parallel_world_size(), 'cp': ps.get_context_parallel_world_size(),
               'pp': ps.get_pipeline_model_parallel_world_size(), 'dp': ps.get_data_parallel_world_size(),
               'ep': ps.get_expert_model_parallel_world_size(), 'device': torch.cuda.current_device(),
               'visible': os.environ.get('CUDA_VISIBLE_DEVICES')}
        with open(os.path.join(d, f'megatron_layout.rank{rank}.json'), 'w') as f:
            json.dump(lay, f)
    with open(os.path.join(d, f'train_clock.rank{rank}.jsonl'), 'a') as f:
        f.write(json.dumps({'rollout_id': rollout_id, 'step_id': step_id, 'rank': rank, 't': t}) + '\n')


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    d = os.environ.get('SLIME_OPD_BENCH_DIR')
    if not d:
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    if os.environ.get('SLIME_OPD_BENCH_FIRST_USE') == '1':
        _install_first_use(d)
    _clock(d, rank, rollout_id, step_id)
    if os.environ.get('SLIME_OPD_BENCH_RECORD') != '1':
        return
    out = os.path.join(d, 'records')
    rec = {'rollout_id': rollout_id, 'step_id': step_id, 'rank': rank,
           'lr': [g['lr'] for g in optimizer.param_groups], 'sha': {}, 'shape': {}, 'dtype': {}}
    with torch.no_grad():
        named = [(f'{ci}.{n}', p.detach()) for ci, chunk in enumerate(model) for n, p in chunk.named_parameters()]
        for (n, p), h in zip(named, tensor_shas([p for _, p in named])):
            rec['sha'][f'model/{n}'] = h
            rec['shape'][f'model/{n}'] = list(p.shape)
            rec['dtype'][f'model/{n}'] = str(p.dtype)
        masters = _masters(optimizer)
        jobs = []
        for n, p, st in masters:
            jobs.append((f'master/{n}', p))
            for k in ('exp_avg', 'exp_avg_sq'):
                if k in st:
                    jobs.append((f'{k}/{n}', st[k]))
        for n, t in jobs:
            rec['shape'][n] = list(t.shape)
            rec['dtype'][n] = str(t.dtype)
        with ThreadPoolExecutor(max_workers=16) as ex:
            for (n, _), h in zip(jobs, ex.map(lambda j: _sha_inplace(j[1]), jobs)):
                rec['sha'][n] = h
        upd, vsq = 0.0, 0.0
        sk = torch.zeros(SKETCH_DIM, dtype=torch.float64, device='cuda')
        for n, p, st in masters:
            flat = p.detach().reshape(-1)
            if n not in _INIT:
                _INIT[n] = flat.clone()
            delta = flat.double() - _INIT[n].double() if flat.numel() < CHUNK else None
            if delta is not None:
                upd += float((delta * delta).sum())
                _sketch_add(sk, n, delta.float())
            else:
                for s in range(0, flat.numel(), CHUNK):
                    d = flat[s:s + CHUNK].double() - _INIT[n][s:s + CHUNK].double()
                    upd += float((d * d).sum())
                _sketch_add(sk, n, flat - _INIT[n])
            if 'exp_avg_sq' in st:
                v = st['exp_avg_sq'].reshape(-1)
                for s in range(0, v.numel(), CHUNK):
                    vsq += float(v[s:s + CHUNK].double().pow(2).sum())
        rec['update_sq'] = upd
        rec['exp_avg_sq_sq'] = vsq
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, f'r{rollout_id}.s{step_id}.rank{rank}')
    np.save(base + '.sketch.npy', sk.cpu().numpy())
    with open(base + '.json', 'w') as f:
        json.dump(rec, f)
