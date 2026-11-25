"""The judge's entry of every nanochat run: one process per rank under torchrun, from the root of the tree under test.

    nanochat_entry.py --mode time|record|traj --out PREFIX --steps N [--seed S] [--probe FILE]
                      [--profile-from K --profile-iters M] -- <recipe arguments>

It runs the tree's `scripts/base_train.py` as `__main__` (runpy, the way `python -m scripts.base_train` does) with the
frozen recipe (bench/recipe.py), after installing a global optimizer step post-hook
(torch.optim.optimizer.register_optimizer_step_post_hook). The first optimizer that steps is the training optimizer;
after each of its steps the hook synchronizes CUDA and stamps the time (time.time(), comparable across the ranks of
the host). After step N the run ends: every rank writes PREFIX.r<rank>.json and the processes exit (no further step
starts).

--mode time: production settings, nothing forced or observed but the step ends; --profile-from K --profile-iters M
records a torch.profiler trace of steps K..K+M-1 on rank 0.

--mode record (deterministic mode; --mode traj runs the production settings, as --mode time, with the observing
hooks only): torch's deterministic algorithms (warn only; Inductor then compiles
one configuration per kernel, without autotuning), cuDNN deterministic, Inductor's `deterministic` option where this
torch has one, and FlashAttention-3's deterministic backward forced on the module the tree loads through
`kernels.get_kernel` (every function of its flash_attn_interface that takes
`deterministic`). Modules of the tree are observed through post-import hooks (the tree imports them itself, in its
own order):
  - nanochat.gpt.GPT.init_weights registers the model (the instance whose parameters are the trained weights);
  - nanochat.dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit, for split "train", is wrapped:
    every row it yields gets an exact integer digest of its input and target tokens (record);
  - with --seed S (traj), init_weights is followed by multiplying every parameter by (1 + 2^-6 z), z ~ N(0, 1) from a
    CUDA generator seeded with S, the parameters taken in sorted name order.
record: after every step, the SHA-256 of the full tensor of every model parameter (ranks share the hashing; the
optimizer all-gathers the parameters, so every rank holds them whole), the hyper-parameters every parameter was
stepped with, and the dtypes of every optimizer state tensor.
traj: after every step, the model's mean loss on this rank's rows of the frozen probe batch (--probe, eager forward,
no grad): the trajectory the statistical gate compares."""
import argparse
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
import runpy
import sys
import time

T_START = time.time()
cut = sys.argv.index('--')
p = argparse.ArgumentParser()
p.add_argument('--mode', choices=('time', 'record', 'traj'), required=True)
p.add_argument('--out', required=True)
p.add_argument('--steps', type=int, required=True)
p.add_argument('--seed', type=int)
p.add_argument('--probe')
p.add_argument('--profile-from', type=int)
p.add_argument('--profile-iters', type=int, default=2)
A = p.parse_args(sys.argv[1:cut])
TRAIN_ARGV = sys.argv[cut + 1:]
RANK, WORLD = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
OBSERVE = A.mode in ('record', 'traj')
DETERMINISTIC = A.mode == 'record'   # gate 2; gate 3 (traj) runs the production settings that are timed
INIT_PERTURBATION = 2.0 ** -6

import torch  # noqa: E402  (does not initialize CUDA: the tree's base_train sets the allocator configuration first)

S = {'opt': None, 'n': 0, 'ends': [], 'lrs': [], 'others': 0, 'prof': None, 'models': [], 'loaders': [],
     'digests': [], 'hparams': [], 'probe': [], 'patched': [], 'inductor_deterministic': None,
     'inductor_first_configuration': False, 'perturbed': None}

if DETERMINISTIC:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import torch._inductor.config as inductor_config  # noqa: E402
    if hasattr(inductor_config, 'deterministic'):
        inductor_config.deterministic = True
        S['inductor_deterministic'] = True
    else:
        S['inductor_deterministic'] = False
    # Deterministic algorithms keep Inductor (torch 2.9.1) from benchmarking pointwise kernels, but a reduction with a
    # reduction hint still gets two configurations (triton_heuristics._reduction_configs returns them before its
    # deterministic check), and the first launch times both: timing noise picks one per rank and per run (every run
    # has its own Inductor cache), so the reduction's order differs between two runs of the same tree. Here the first
    # configuration is taken without timing, in every run of every tree.
    from torch._inductor.runtime import triton_heuristics  # noqa: E402

    def _first_configuration(self, *args, **kwargs):
        return {launcher: float(i) for i, launcher in enumerate(self.launchers)}
    triton_heuristics.CachingAutotuner.benchmark_all_configs = _first_configuration
    S['inductor_first_configuration'] = True


# ---------------------------------------------------------------- post-import hooks

def _force_deterministic(module):
    """Wrap every flash_attn_* function of a FlashAttention-3 interface module that takes `deterministic`."""
    fai = getattr(module, 'flash_attn_interface', module)
    for name in sorted(dir(fai)):
        fn = getattr(fai, name)
        if not name.startswith('flash_attn') or not callable(fn) or getattr(fn, '_bench_forced', False):
            continue
        try:
            if 'deterministic' not in inspect.signature(fn).parameters:
                continue
        except (TypeError, ValueError):
            continue

        def forced(*a, _orig=fn, **k):
            k['deterministic'] = True
            return _orig(*a, **k)
        forced.__name__ = forced.__qualname__ = name
        forced._bench_forced = True
        setattr(fai, name, forced)
        S['patched'].append(f'{getattr(fai, "__name__", "?")}.{name}')
    return module


def _hook_kernels(mod):
    orig = mod.get_kernel

    def get_kernel(*a, **k):
        return _force_deterministic(orig(*a, **k))
    mod.get_kernel = get_kernel


def _perturb(model, seed):
    named = sorted(model.named_parameters(), key=lambda kv: kv[0])
    g = torch.Generator(device=named[0][1].device)
    g.manual_seed(seed)
    with torch.no_grad():
        for _, q in named:
            z = torch.randn(q.shape, generator=g, device=q.device, dtype=torch.float32)
            q.copy_((q.float() * (1 + INIT_PERTURBATION * z)).to(q.dtype))
    S['perturbed'] = {'seed': seed, 'scale': INIT_PERTURBATION, 'parameters': len(named)}


def _hook_gpt(mod):
    cls = mod.GPT
    orig = cls.init_weights

    def init_weights(self, *a, **k):
        out = orig(self, *a, **k)
        S['models'].append(self)
        if A.mode == 'traj' and A.seed is not None and len(S['models']) == 1:
            _perturb(self, A.seed)
        return out
    cls.init_weights = init_weights


ROW_W = None


def _row_digests(inputs, targets):
    """[B, 4] int64: two random linear forms of each row's inputs and of its targets (exact integer arithmetic)."""
    global ROW_W
    T = inputs.shape[-1]
    if ROW_W is None or ROW_W.shape[0] != T or ROW_W.device != inputs.device:
        g = torch.Generator(device='cpu')
        g.manual_seed(0x5EED)
        ROW_W = torch.randint(1, 2 ** 31 - 1, (T, 2), generator=g, dtype=torch.int64).to(inputs.device)
    a = (inputs.to(torch.int64).unsqueeze(-1) * ROW_W).sum(1)
    b = (targets.to(torch.int64).unsqueeze(-1) * ROW_W).sum(1)
    return torch.cat([a, b], dim=1)


def _hook_dataloader(mod):
    name = 'tokenizing_distributed_data_loader_with_state_bos_bestfit'
    orig = getattr(mod, name)

    def loader(tokenizer, B, T, split, *a, **k):
        gen = orig(tokenizer, B, T, split, *a, **k)
        if split != 'train':
            yield from gen
            return
        rows = []
        S['loaders'].append(rows)
        for item in gen:
            rows.append(_row_digests(item[0], item[1]))
            yield item
    loader.__name__ = loader.__qualname__ = name
    setattr(mod, name, loader)


HOOKS = {'nanochat.gpt': _hook_gpt}
if DETERMINISTIC:
    HOOKS['kernels'] = _hook_kernels   # FlashAttention-3's deterministic backward: gate 2 only
if A.mode == 'record':
    HOOKS['nanochat.dataloader'] = _hook_dataloader


class _PostImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in HOOKS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None or not hasattr(spec.loader, 'exec_module'):
            return spec
        loader, hook = spec.loader, HOOKS[name]

        class Loader(importlib.abc.Loader):
            def create_module(self, s):
                return loader.create_module(s)

            def exec_module(self, module):
                loader.exec_module(module)
                hook(module)
        spec.loader = Loader()
        return spec


if OBSERVE:
    sys.meta_path.insert(0, _PostImport())

import torch.distributed as dist  # noqa: E402
from torch.optim.optimizer import register_optimizer_step_post_hook  # noqa: E402


def cuda():
    """Asked only after the tree has initialized its device (never before base_train sets its allocator options)."""
    return torch.cuda.is_initialized()


# ---------------------------------------------------------------- observation

from shapipe import ShaPipeline, tensor_sha  # noqa: E402,F401  (bench/, the script's directory)


def _names(model):
    return {id(q): n for n, q in model.named_parameters()}


def param_digests(model):
    """[name, shape, dtype, sha256] of the parameters this rank hashes (sorted names, index % WORLD == RANK)."""
    out = []
    with ShaPipeline() as pipe:              # asynchronous pinned copies, hashed in threads
        for i, (name, q) in enumerate(sorted(model.named_parameters(), key=lambda kv: kv[0])):
            if i % WORLD == RANK:
                out.append([name, list(q.shape), str(q.dtype).replace('torch.', ''), pipe.submit(q)])
        out = [[*e[:3], e[3].result()] for e in out]
    return out


HP_KEYS = ('kind', 'lr', 'momentum', 'weight_decay', 'betas', 'eps', 'ns_steps', 'beta2')


def _plain(v):
    if torch.is_tensor(v):
        return v.item() if v.numel() == 1 else v.tolist()
    return list(v) if isinstance(v, tuple) else v


def param_hparams(opt, names):
    """{parameter name: [hyper-parameters of the group that stepped it]}; parameters outside the model by index."""
    out = {}
    for gi, g in enumerate(opt.param_groups):
        hp = [_plain(g.get(k)) for k in HP_KEYS]
        for qi, q in enumerate(g['params']):
            out[names.get(id(q), f'<group {gi} param {qi}>')] = hp
    return out


def state_dtypes(opt):
    seen = set()
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                seen.add((k, str(v.dtype).replace('torch.', '')))
    return sorted(seen)


PROBE = None


def probe_loss(model):
    global PROBE
    if PROBE is None:
        data = torch.load(A.probe, map_location='cpu', weights_only=True)
        dev = next(model.parameters()).device
        PROBE = (data['inputs'][RANK].to(dev), data['targets'][RANK].to(dev))
    with torch.no_grad():
        loss = model(PROBE[0], PROBE[1])
    return float(loss.float().item())


def write(payload):
    with open(f'{A.out}.r{RANK}.json', 'w') as f:
        json.dump(payload, f)


def fa3_in_use():
    m = sys.modules.get('nanochat.flash_attention')
    return None if m is None else bool(getattr(m, 'USE_FA3', False))


def finish(opt):
    base = {'rank': RANK, 'world': WORLD, 'mode': A.mode, 'status': 'ok', 'steps': S['n'],
            'step_end_s': S['ends'], 'start_s': T_START, 'startup_to_first_step_s': S['ends'][0] - T_START,
            'lrs': S['lrs'], 'other_optimizer_steps': S['others'],
            'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated() if cuda() else 0,
            'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved() if cuda() else 0,
            'fa3': fa3_in_use(), 'deterministic_patched': S['patched'],
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'inductor_deterministic': S['inductor_deterministic'],
            'inductor_first_configuration': S['inductor_first_configuration'],
            'optimizer': type(opt).__module__ + '.' + type(opt).__name__, 'models': len(S['models'])}
    if A.mode == 'record':
        rows = [r for rows in S['loaders'] for r in rows]
        rows = torch.cat(rows).cpu().tolist() if rows else []
        base.update(digests=S['digests'], hparams=S['hparams'], state_dtypes=state_dtypes(opt), rows=rows,
                    loaders=len(S['loaders']))
    if A.mode == 'traj':
        base.update(probe=S['probe'], perturbed=S['perturbed'], seed=A.seed)
    write(base)
    sys.stdout.flush()
    sys.stderr.flush()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    os._exit(0)


def fail(message):
    write({'rank': RANK, 'status': 'error', 'error': message[:2000], 'steps': S['n']})
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(4)


def post_hook(opt, args, kwargs):
    if S['opt'] is None:
        S['opt'] = opt
    if opt is not S['opt']:
        S['others'] += 1
        return
    if cuda():
        torch.cuda.synchronize()
    S['ends'].append(time.time())
    S['lrs'].append([_plain(g.get('lr')) for g in opt.param_groups])
    if OBSERVE:
        if len(S['models']) != 1:
            fail(f'{len(S["models"])} models initialized through nanochat.gpt.GPT.init_weights; the judge observes '
                 f'exactly one')
        model = S['models'][0]
        if A.mode == 'record':
            S['digests'].append(param_digests(model))
            S['hparams'].append(param_hparams(opt, _names(model)))
        else:
            S['probe'].append(probe_loss(model))
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
sys.argv = [os.path.join(os.getcwd(), 'scripts', 'base_train.py'), *TRAIN_ARGV]
try:
    runpy.run_module('scripts.base_train', run_name='__main__', alter_sys=True)
except BaseException as exc:
    write({'rank': RANK, 'status': 'error', 'error': f'{type(exc).__name__}: {exc}'[:2000], 'steps': S['n']})
    raise
write({'rank': RANK, 'status': 'error', 'error': f'base_train returned after {S["n"]} of {A.steps} optimizer steps',
       'steps': S['n']})
sys.exit(3)
