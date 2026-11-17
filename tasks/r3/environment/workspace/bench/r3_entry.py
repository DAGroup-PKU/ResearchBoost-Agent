"""The judge's entry of every R3 run: one process per rank under torchrun, from the root of the tree under test.

    r3_entry.py --mode time|record --out PREFIX --steps N [--profile-from K --profile-iters M] -- <recipe arguments>

It imports the tree's `train.online_rl` and runs its `main()` with the frozen recipe (bench/recipe.py), after
installing a global optimizer step post-hook (torch.optim.optimizer.register_optimizer_step_post_hook). The first
optimizer that steps is the training optimizer; after each of its steps the hook synchronizes CUDA and stamps the
time (time.time(), comparable across ranks of the host), the learning rates and the hyper-parameters of its groups.
After step N the run ends: every rank writes PREFIX.r<rank>.json and the processes exit (no further step starts).

--mode record (deterministic mode): torch's deterministic algorithms (warn only), cuDNN deterministic, and flash-attn's
deterministic backward forced through the module attributes the training code imports at run time (flash_attn.* and
flash_attn.flash_attn_interface.*, patched before the tree is imported). After step N the full tensor of every
parameter the optimizer holds and of every tensor of its state (exp_avg, exp_avg_sq, step) is gathered (DTensor
full_tensor(); the same values whatever the sharding) and hashed with SHA-256; ranks share the hashing.
--mode time: production settings (nothing forced); --profile-from K --profile-iters M records a torch.profiler trace
of steps K..K+M-1 on rank 0."""
import argparse
import hashlib
import json
import os
import sys
import time

T_START = time.time()
cut = sys.argv.index('--')
p = argparse.ArgumentParser()
p.add_argument('--mode', choices=('time', 'record'), required=True)
p.add_argument('--out', required=True)
p.add_argument('--steps', type=int, required=True)
p.add_argument('--profile-from', type=int)
p.add_argument('--profile-iters', type=int, default=2)
A = p.parse_args(sys.argv[1:cut])
TRAIN_ARGV = sys.argv[cut + 1:]
RANK, WORLD = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])

import torch  # noqa: E402

DETERMINISTIC_PATCHED = []
if A.mode == 'record':
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import flash_attn  # noqa: E402
    import flash_attn.flash_attn_interface as fai  # noqa: E402
    for name in ('flash_attn_func', 'flash_attn_varlen_func', 'flash_attn_qkvpacked_func',
                 'flash_attn_varlen_qkvpacked_func', 'flash_attn_kvpacked_func', 'flash_attn_varlen_kvpacked_func'):
        orig = getattr(fai, name, None)
        if orig is None:
            continue

        def forced(*a, _orig=orig, **k):
            k['deterministic'] = True
            return _orig(*a, **k)
        forced.__name__ = forced.__qualname__ = name
        setattr(fai, name, forced)
        if getattr(flash_attn, name, None) is orig:
            setattr(flash_attn, name, forced)
        DETERMINISTIC_PATCHED.append(name)

import torch.distributed as dist  # noqa: E402
from torch.optim.optimizer import register_optimizer_step_post_hook  # noqa: E402

S = {'opt': None, 'n': 0, 'ends': [], 'lrs': [], 'hparams': None, 'others': 0, 'prof': None}


def hparams(opt):
    keep = ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'fused', 'capturable')
    return [{k: (list(g[k]) if isinstance(g.get(k), tuple) else g.get(k)) for k in keep if k in g}
            for g in opt.param_groups]


def write(payload):
    with open(f'{A.out}.r{RANK}.json', 'w') as f:
        json.dump(payload, f)


from shapipe import ShaPipeline, tensor_sha  # noqa: E402,F401  (bench/, the script's directory)


def digests(opt):
    """[index, kind, shape, dtype, sha256] of every parameter the optimizer holds and every tensor of its state, as
    full tensors; collective (every rank walks the same list), rank i hashes the entries with index % WORLD == i."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        from torch.distributed._tensor import DTensor
    out = []
    params = [q for g in opt.param_groups for q in g['params']]
    # The gathers stay in order on every rank (collectives); the owner's digests run in the pipeline meanwhile.
    with ShaPipeline() as pipe:
        for i, q in enumerate(params):
            entries = [('param', q)] + [(k, v) for k, v in sorted(opt.state.get(q, {}).items()) if torch.is_tensor(v)]
            for kind, t in entries:
                full = t.detach().full_tensor() if isinstance(t, DTensor) else t.detach()
                if i % WORLD == RANK:
                    out.append([i, kind, list(full.shape), str(full.dtype).replace('torch.', ''), pipe.submit(full)])
                del full
        out = [[*e[:4], e[4].result()] for e in out]
    return out, len(params)


def finish(opt):
    base = {'rank': RANK, 'world': WORLD, 'mode': A.mode, 'status': 'ok', 'steps': S['n'],
            'step_end_s': S['ends'], 'start_s': T_START, 'startup_to_first_step_s': S['ends'][0] - T_START,
            'lrs': S['lrs'], 'hparams': S['hparams'], 'other_optimizer_steps': S['others'],
            'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved(),
            'deterministic_patched': DETERMINISTIC_PATCHED,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'optimizer': type(opt).__module__ + '.' + type(opt).__name__}
    if A.mode == 'record':
        t0 = time.time()
        mine, n_params = digests(opt)
        allr = [None] * WORLD
        dist.all_gather_object(allr, mine)
        base.update(n_params=n_params, digest_s=round(time.time() - t0, 1))
        if RANK == 0:
            base['digests'] = sorted((e for part in allr for e in part), key=lambda e: (e[0], e[1]))
    write(base)
    sys.stdout.flush()
    sys.stderr.flush()
    dist.barrier()
    os._exit(0)


def post_hook(opt, args, kwargs):
    if S['opt'] is None:
        S['opt'] = opt
        S['hparams'] = hparams(opt)
    if opt is not S['opt']:
        S['others'] += 1
        return
    torch.cuda.synchronize()
    S['ends'].append(time.time())
    S['lrs'].append([g['lr'] for g in opt.param_groups])
    if hparams(opt) != S['hparams'] and len(S['lrs']) > 0:
        # lr moves with the schedule; anything else changing is recorded
        now = hparams(opt)
        for g_old, g_new in zip(S['hparams'], now):
            g_old_wo = {k: v for k, v in g_old.items() if k != 'lr'}
            g_new_wo = {k: v for k, v in g_new.items() if k != 'lr'}
            if g_old_wo != g_new_wo:
                S.setdefault('hparam_changes', []).append([S['n'], g_new_wo])
    S['n'] += 1
    k = S['n']
    if A.profile_from is not None and RANK == 0:
        if k == A.profile_from:
            S['prof'] = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, with_stack=False)
            S['prof'].__enter__()
        elif S['prof'] is not None and k == A.profile_from + A.profile_iters:
            S['prof'].__exit__(None, None, None)
            S['prof'].export_chrome_trace(f'{A.out}.trace.json')
            table = S['prof'].key_averages().table(sort_by='self_cuda_time_total', row_limit=60)
            with open(f'{A.out}.profile.txt', 'w') as f:
                f.write(table)
            S['prof'] = None
    if k == A.steps:
        finish(opt)


register_optimizer_step_post_hook(post_hook)

sys.path.insert(0, os.getcwd())
sys.argv = [os.path.join(os.getcwd(), 'train', 'online_rl.py'), *TRAIN_ARGV]
try:
    import train.online_rl as entry
    entry.main()
except BaseException as exc:
    write({'rank': RANK, 'status': 'error', 'error': f'{type(exc).__name__}: {exc}'[:2000], 'steps': S['n']})
    raise
write({'rank': RANK, 'status': 'error', 'error': f'main() returned after {S["n"]} of {A.steps} optimizer steps',
       'steps': S['n']})
sys.exit(3)
