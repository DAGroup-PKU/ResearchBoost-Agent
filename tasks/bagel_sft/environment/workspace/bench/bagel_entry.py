"""The judge's entry of every BAGEL-SFT run: one process per rank under torchrun, from the root of the tree under test.

    bagel_entry.py --mode time|record|traj|conform --out PREFIX --steps N [--seed S]
                   [--profile-from K --profile-iters M] -- <recipe arguments>

It runs the tree's `train/pretrain_unified_navit.py` as `__main__` (runpy, the way `python
train/pretrain_unified_navit.py` does) with the frozen recipe (bench/recipe.py), after installing a global optimizer
step post-hook (torch.optim.optimizer.register_optimizer_step_post_hook). The first optimizer that steps is the
training optimizer; after each of its steps the hook synchronizes CUDA and stamps the time (time.time(), comparable
across the ranks of the host). After step N the run ends: every rank writes PREFIX.r<rank>.json and the processes exit
(no further step starts).

Every mode: upstream's data/dataset_info.py names placeholder paths (`your_data_path/bagel_example/...`); after the
tree's module is imported, its DATASET_INFO entries are pointed at the judge's copy of upstream's bagel_example data
(BAGEL_BENCH_DATA), and the editing set's parquet index is rewritten for that location (a copy in TMPDIR; entries and
order unchanged; upstream names it seedxedit_multi_nas.json, the example archive ships it as seedxedit_multi.json).
Nothing else in DATASET_INFO is touched.

--mode time: production settings, nothing forced or observed but the step ends; --profile-from K --profile-iters M
records a torch.profiler trace of steps K..K+M-1 on rank 0.

--mode record, --mode traj and --mode conform (deterministic mode): torch's deterministic algorithms (strict: an operation without a
deterministic implementation raises; with warn_only, torch 2.5 keeps the non-deterministic backward of
scaled_dot_product_attention's memory-efficient kernel, which the reference uses), cuDNN deterministic, and
FlashAttention's deterministic backward forced (every function of flash_attn.flash_attn_interface that takes
`deterministic`, also as re-exported by the flash_attn package). Observed:
  - the two FSDP roots (FullyShardedDataParallel instances that no other one contains): the trained model (its flat
    parameters are the training optimizer's) and the EMA (none of them are);
  - every batch the training loop moves to the GPU through data.dataset_base.SimpleCustomBatch.cuda: a SHA-256 of all
    of its fields (record);
  - modeling.bagel.bagel.Bagel.forward: the per-token `ce` and `mse` it returns (traj);
  - FullyShardedDataParallel.clip_grad_norm_: the total gradient norm before clipping (traj).
conform (gate 2b, bench/conform.py; the bench passes the held-out data configuration in the recipe arguments): the
judge's noise injected into every AutoEncoder.encode (the VAE posterior sample) and every Bagel.forward (the latent
noise), keyed by the ordinal of the call on the rank (bench/conform.py Injector); per Bagel.forward: the per-token MSE
(mean over the latent channels) and CE, and the noise draws of every judged call; the SHA-256 of every batch, as
record does; with BAGEL_BENCH_CONFORM_POOLS=1 (the reference's preparation only) also each batch's pool flags from
its dense masks.
record: after every step, the SHA-256 of every local shard of the model's flat parameters, of their gradients and of
both AdamW moments, and of the EMA's flat parameters (the EMA as it is when the step ends: its update for this step
follows the optimizer step); the hyper-parameters of every parameter group; the dtypes of the optimizer state; the
trained parameters' names, shapes and dtypes.
traj: with --seed S, the global seed becomes BASE_SEED + S and, when the optimizer is created (the weights loaded, both
models wrapped), every trained parameter of the model and its EMA counterpart are multiplied by (1 + 2^-7 z),
z ~ N(0, 1) from a CUDA generator seeded with (S, the parameter's name): the same perturbation whatever the flat
layout. After every step: the global mean CE and MSE of the step (the sums of the per-token values over all ranks over
their counts, as the training script normalizes them) and the total gradient norm; after every DISP_EVERY steps, the
L2 norm of the change since DISP_EVERY steps before of the model's ViT, language-model and other parameters and of the
EMA (summed over the ranks' shards)."""
import argparse
import concurrent.futures
import gc
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import math
import os
import runpy
import sys
import time

T_START = time.time()
cut = sys.argv.index('--')
p = argparse.ArgumentParser()
p.add_argument('--mode', choices=('time', 'record', 'traj', 'conform'), required=True)
p.add_argument('--out', required=True)
p.add_argument('--steps', type=int, required=True)
p.add_argument('--seed', type=int)
p.add_argument('--data-seed', type=int)
p.add_argument('--production', action='store_true')
p.add_argument('--profile-from', type=int)
p.add_argument('--profile-iters', type=int, default=2)
A = p.parse_args(sys.argv[1:cut])
TRAIN_ARGV = sys.argv[cut + 1:]
if A.data_seed is not None:
    TRAIN_ARGV += ['--data_seed', str(A.data_seed)]
RANK, WORLD = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
OBSERVE = A.mode in ('record', 'traj', 'conform')
DETERMINISTIC = OBSERVE and not A.production   # --production: observed like the mode, on the code path timing runs
CONFORM = A.mode == 'conform'
POOLS = CONFORM and os.environ.get('BAGEL_BENCH_CONFORM_POOLS') == '1'
BASE_SEED = 4396                      # upstream's default global seed
INIT_PERTURBATION = 2.0 ** -7
DISP_EVERY = 10
DATA = os.environ['BAGEL_BENCH_DATA']
if A.mode == 'traj' and A.seed is not None:
    TRAIN_ARGV += ['--global_seed', str(BASE_SEED + A.seed)]

import torch  # noqa: E402

# Each rank pinned to its share of the container's CPUs (bench/cpu_affinity.py: the physical cores of its GPU's NUMA
# node, split between the ranks there; torch's intra-op threads = the share's physical cores), the same for every tree.
# Nothing else of the bench's directory stays importable by the training code.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [x for x in sys.path if os.path.abspath(x or '.') != _HERE] + [_HERE]
import cpu_affinity  # noqa: E402
if CONFORM:
    import conform  # noqa: E402
    INJECTOR = conform.Injector(torch, RANK)
    INJECTOR.install()          # before the tree's modules bind torch.randn_like and friends
sys.path.remove(_HERE)
CPU_BINDING = cpu_affinity.bind_local_rank(int(os.environ.get('LOCAL_RANK', '0')),
                                           int(os.environ.get('LOCAL_WORLD_SIZE', os.environ.get('WORLD_SIZE', '1'))))

S = {'conform': [], 'pools': [], 'opt': None, 'n': 0, 'ends': [], 'lrs': [], 'others': 0, 'prof': None, 'roots': None, 'batches': [],
     'digests': [], 'hparams': [], 'patched': [], 'perturbed': None, 'dataset_info': None,
     'loss_sums': None, 'norms': [], 'series': {'ce': [], 'mse': [], 'grad_norm': []}, 'disp': [], 'snap': None,
     'forwards': 0}

if DETERMINISTIC:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------- post-import hooks

def _force_deterministic(module):
    """Wrap every flash_attn* function of flash_attn.flash_attn_interface that takes `deterministic`, in the interface
    module and wherever the flash_attn package re-exports it."""
    fai = sys.modules.get('flash_attn.flash_attn_interface')
    if fai is None:
        return module
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
        if getattr(module, name, None) is fn:
            setattr(module, name, forced)
        S['patched'].append(name)
    return module


def _hook_dataset_info(mod):
    """Point upstream's placeholder paths at the judge's data; the editing index rewritten for that location."""
    info = mod.DATASET_INFO
    try:
        t2i, edit, vlm = info['t2i_pretrain']['t2i'], info['unified_edit']['seedxedit_multi'], info['vlm_sft']['llava_ov']
    except KeyError as exc:
        raise RuntimeError(f'data/dataset_info.py: DATASET_INFO has no entry {exc} (the judge sets the data paths of '
                           f'upstream\'s t2i, seedxedit_multi and llava_ov entries)')
    old = 'your_data_path/bagel_example'
    src = os.path.join(DATA, 'editing/parquet_info/seedxedit_multi.json')
    with open(src) as f:
        index = json.load(f)
    relocated = {k.replace(old, DATA, 1): v for k, v in index.items()}
    dst = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'bench_seedxedit_multi_r{RANK}.json')
    with open(dst, 'w') as f:
        json.dump(relocated, f)
    t2i['data_dir'] = os.path.join(DATA, 't2i')
    edit['data_dir'] = os.path.join(DATA, 'editing/seedxedit_multi')
    edit['parquet_info_path'] = dst
    vlm['data_dir'] = os.path.join(DATA, 'vlm/images')
    vlm['jsonl_path'] = os.path.join(DATA, 'vlm/llava_ov_si.jsonl')
    S['dataset_info'] = {'data': DATA, 'editing_index_entries': len(relocated)}


def _batch_digest(batch):
    h = hashlib.sha256()
    for name in sorted(vars(batch)):
        v = getattr(batch, name)
        h.update(name.encode() + b'\0')
        if torch.is_tensor(v):
            t = v.detach().contiguous().cpu()
            h.update(f'{t.dtype}{tuple(t.shape)}'.encode())
            if t.numel():
                h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()))
        else:
            h.update(json.dumps(v, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _hook_dataset_base(mod):
    cls = mod.SimpleCustomBatch
    orig = cls.cuda

    def cuda(self, *a, **k):
        S['batches'].append(_batch_digest(self))
        return orig(self, *a, **k)
    cls.cuda = cuda


def _hook_bagel(mod):
    cls = mod.Bagel
    orig = cls.forward

    def forward(self, *a, **k):
        out = orig(self, *a, **k)
        S['forwards'] += 1
        if isinstance(out, dict):
            ce, mse = out.get('ce'), out.get('mse')
            dev = next((t.device for t in (ce, mse) if torch.is_tensor(t)), None)
            if dev is not None:
                z = torch.zeros((), device=dev, dtype=torch.float64)
                sums = torch.stack([
                    ce.detach().double().sum() if torch.is_tensor(ce) else z,
                    torch.tensor(float(ce.numel()) if torch.is_tensor(ce) else 0.0, device=dev, dtype=torch.float64),
                    mse.detach().double().mean(dim=-1).sum() if torch.is_tensor(mse) else z,
                    torch.tensor(float(mse.shape[0]) if torch.is_tensor(mse) else 0.0, device=dev,
                                 dtype=torch.float64)])
                S['loss_sums'] = sums if S['loss_sums'] is None else S['loss_sums'] + sums
        return out
    cls.forward = forward


def _hook_dataset_conform(mod):
    cls = mod.SimpleCustomBatch
    orig = cls.cuda

    def cuda(self, *a, **k):
        S['batches'].append(_batch_digest(self))
        if POOLS:
            S['pools'].append(conform.pool_flags(self.nested_attention_masks, self.sample_lens,
                                                 self.mse_loss_indexes.tolist()))
        return orig(self, *a, **k)
    cls.cuda = cuda


def _hook_bagel_conform(mod):
    cls = mod.Bagel
    orig = cls.forward

    def forward(self, *a, **k):
        idx = k.get('packed_vae_token_indexes')
        numel = None
        if torch.is_tensor(idx):
            numel = int(idx.numel()) * int(self.latent_patch_size) ** 2 * int(self.latent_channel)
        out = INJECTOR.call(conform.LATENT, lambda: orig(self, *a, **k), numel=numel)
        S['forwards'] += 1
        mse = out.get('mse') if isinstance(out, dict) else None
        ce = out.get('ce') if isinstance(out, dict) else None
        S['conform'].append({
            'mse': mse.detach().float().mean(dim=-1).cpu().tolist() if torch.is_tensor(mse) else None,
            'ce': ce.detach().float().cpu().tolist() if torch.is_tensor(ce) else None,
            'latent_draws': INJECTOR.last[conform.LATENT]})
        return out
    cls.forward = forward


def _hook_autoencoder(mod):
    cls = mod.AutoEncoder
    orig = cls.encode

    def encode(self, *a, **k):
        return INJECTOR.call(conform.VAE, lambda: orig(self, *a, **k))
    cls.encode = encode


HOOKS = {'data.dataset_info': _hook_dataset_info}
if DETERMINISTIC:
    HOOKS['flash_attn'] = _force_deterministic
    HOOKS['flash_attn.flash_attn_interface'] = _force_deterministic
if A.mode == 'record':
    HOOKS['data.dataset_base'] = _hook_dataset_base
if A.mode == 'traj':
    HOOKS['modeling.bagel.bagel'] = _hook_bagel
if CONFORM:
    HOOKS['data.dataset_base'] = _hook_dataset_conform
    HOOKS['modeling.bagel.bagel'] = _hook_bagel_conform
    HOOKS['modeling.autoencoder'] = _hook_autoencoder


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


sys.meta_path.insert(0, _PostImport())

import torch.distributed as dist  # noqa: E402
from torch.optim.optimizer import register_optimizer_step_post_hook  # noqa: E402

if A.mode == 'traj':
    from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP  # noqa: E402
    _orig_clip = _FSDP.clip_grad_norm_

    def _clip(self, *a, **k):
        norm = _orig_clip(self, *a, **k)
        S['norms'].append(norm.detach().float().clone() if torch.is_tensor(norm) else torch.tensor(float(norm)))
        return norm
    _FSDP.clip_grad_norm_ = _clip


def cuda():
    return torch.cuda.is_initialized()


# ---------------------------------------------------------------- the two FSDP roots and their flat parameters

def fsdp_roots():
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    every = [o for o in gc.get_objects() if isinstance(o, FSDP)]
    inner = set()
    for f in every:
        for m in f.modules():
            if m is not f and isinstance(m, FSDP):
                inner.add(id(m))
    return [f for f in every if id(f) not in inner]


def _clean(name):
    for wrapper in ('_fsdp_wrapped_module.', '_checkpoint_wrapped_module.', '_orig_mod.'):
        name = name.replace(wrapper, '')
    return name


def flat_params(root):
    """[(unit name, flat parameter, [(parameter name, shape, start in the full parameter, offset in the shard,
    elements in the shard)])] of every FSDP unit of a root, in module order."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    out = []
    for unit_name, m in root.named_modules():
        if not isinstance(m, FSDP):
            continue
        handle = getattr(m, '_handle', None)
        fp = getattr(handle, 'flat_param', None) if handle is not None else None
        if fp is None:
            continue
        prefix = _clean(unit_name + '.') if unit_name else ''
        segs = []
        for fqn, shape, info in zip(fp._fqns, fp._shapes, fp._shard_param_infos):
            if info.in_shard:
                segs.append((_clean(prefix + fqn), tuple(shape), info.intra_param_start_idx, info.offset_in_shard,
                             info.numel_in_shard))
        out.append((_clean(unit_name) or '<root>', fp, segs, [(_clean(prefix + q), tuple(s)) for q, s in
                                                               zip(fp._fqns, fp._shapes)]))
    return out


def find_roots(opt):
    """(model root, EMA root): the trained model's parameters are the optimizer's (its flat parameters, or the original
    parameters with use_orig_params), the EMA's are not."""
    if S['roots'] is None:
        owned = {id(q) for g in opt.param_groups for q in g['params']}
        roots = fsdp_roots()
        model = [r for r in roots if any(id(q) in owned for q in r.parameters())]
        ema = [r for r in roots if r not in model]
        if len(model) != 1 or len(ema) != 1:
            fail(f'{len(roots)} FSDP roots ({len(model)} trained by the optimizer, {len(ema)} not): the judge '
                 f'observes exactly one trained model and one EMA')
        S['roots'] = (model[0], ema[0])
    return S['roots']


def group(name):
    if name.startswith('vit_model.'):
        return 0
    if name.startswith('language_model.'):
        return 1
    return 2


# ---------------------------------------------------------------- traj: seed perturbation and displacements

def _name_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:7], 'big')


def perturb(model, ema, seed):
    trained = {}
    with torch.no_grad():
        for _, fp, segs, _ in flat_params(model):
            if not fp.requires_grad:
                continue
            for name, shape, start, off, n in segs:
                trained[name] = (shape, start, n)
        done = 0
        for root in (model, ema):
            for _, fp, segs, _ in flat_params(root):
                for name, shape, start, off, n in segs:
                    if name not in trained:
                        continue
                    g = torch.Generator(device=fp.device)
                    g.manual_seed(_name_seed(seed, name))
                    z = torch.randn(math.prod(shape), generator=g, device=fp.device, dtype=torch.float32)
                    seg = fp.data[off:off + n]
                    seg.copy_((seg.float() * (1 + INIT_PERTURBATION * z[start:start + n])).to(seg.dtype))
                    done += 1
    S['perturbed'] = {'seed': seed, 'scale': INIT_PERTURBATION, 'segments': done, 'parameters': len(trained)}


def _shards(model, ema):
    """[(key, flat parameter, [(group 0-3, offset, elements)])] of both roots: model parameters by group, EMA as 3."""
    out = []
    for tag, root in (('model', model), ('ema', ema)):
        for i, (_, fp, segs, _) in enumerate(flat_params(root)):
            if tag == 'model' and not fp.requires_grad:
                continue
            out.append((f'{tag}{i}', fp, [(3 if tag == 'ema' else group(n), off, k) for n, _, _, off, k in segs]))
    return out


def displacement(model, ema):
    """L2 norms of the change since the last snapshot (ViT, LM, other, EMA), then a new snapshot (host memory)."""
    shards = _shards(model, ema)
    sq = torch.zeros(4, device=shards[0][1].device, dtype=torch.float64)
    snap = S['snap']
    new = {}
    with torch.no_grad():
        for key, fp, segs in shards:
            cur = fp.data
            if snap is not None:
                old = snap[key].to(cur.device, non_blocking=True)
                for gi, off, k in segs:
                    d = cur[off:off + k].double() - old[off:off + k].double()
                    sq[gi] += (d * d).sum()
            new[key] = cur.detach().to('cpu', copy=True)
    S['snap'] = new
    if snap is None:
        return None
    dist.all_reduce(sq)
    return [math.sqrt(x) for x in sq.tolist()]


# ---------------------------------------------------------------- record

# bench/shapipe.py, imported like cpu_affinity above: the bench's directory is importable only for this import
# (it is off sys.path otherwise, so that nothing of the bench shadows the training code's modules).
sys.path.append(_HERE)
from shapipe import tensor_sha, tensor_shas  # noqa: E402,F401
sys.path.remove(_HERE)


def digests(opt, model, ema):
    """[key, shape, dtype, sha256] of every local shard: model flat parameters, gradients, AdamW moments; EMA."""
    items = []
    for i, (_, fp, _, _) in enumerate(flat_params(model)):
        items.append((f'model{i}.param', fp.data))
        if fp.grad is not None:
            items.append((f'model{i}.grad', fp.grad))
        for k, v in sorted(opt.state.get(fp, {}).items()):
            if torch.is_tensor(v):
                items.append((f'model{i}.state.{k}', v))
    for i, (_, fp, _, _) in enumerate(flat_params(ema)):
        items.append((f'ema{i}.param', fp.data))
    # Asynchronous pinned copies, hashed in threads, at most shapipe.INFLIGHT_BYTES on the host at a time (the
    # earlier formulation copied every shard to the host first).
    shas = tensor_shas([t for _, t in items], workers=max(1, min(16, len(os.sched_getaffinity(0)))))
    return [[k, list(t.shape), str(t.dtype).replace('torch.', ''), sha] for (k, t), sha in zip(items, shas)]


HP_KEYS = ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize')


def _plain(v):
    if torch.is_tensor(v):
        return v.item() if v.numel() == 1 else v.tolist()
    return list(v) if isinstance(v, tuple) else v


def hparams(opt):
    return [[_plain(g.get(k)) for k in HP_KEYS] + [sum(q.numel() for q in g['params'])] for g in opt.param_groups]


def state_dtypes(opt):
    seen = set()
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                seen.add((k, str(v.dtype).replace('torch.', '')))
    return sorted(seen)


def parameters(root):
    names = []
    for _, fp, _, full in flat_params(root):
        for name, shape in full:
            names.append([name, list(shape), str(fp.dtype).replace('torch.', ''), bool(fp.requires_grad)])
    return sorted(names)


# ---------------------------------------------------------------- end of a run

def write(payload):
    with open(f'{A.out}.r{RANK}.json', 'w') as f:
        json.dump(payload, f)


def finish(opt):
    base = {'rank': RANK, 'world': WORLD, 'mode': A.mode, 'status': 'ok', 'steps': S['n'],
            'step_end_s': S['ends'], 'start_s': T_START, 'startup_to_first_step_s': S['ends'][0] - T_START,
            'lrs': S['lrs'], 'other_optimizer_steps': S['others'],
            'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated() if cuda() else 0,
            'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved() if cuda() else 0,
            'deterministic_patched': S['patched'],
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'dataset_info': S['dataset_info'], 'cpu_binding': CPU_BINDING,
            'optimizer': type(opt).__module__ + '.' + type(opt).__name__}
    if A.mode == 'record':
        model, ema = find_roots(opt)
        base.update(digests=S['digests'], hparams=S['hparams'], state_dtypes=state_dtypes(opt), batches=S['batches'],
                    parameters=parameters(model), ema_parameters=parameters(ema))
    if CONFORM:
        vae = [{'ordinal': k, 'draws': d} for kind, k, d in INJECTOR.log if kind == conform.VAE]
        base.update(conform=S['conform'], batches=S['batches'], pools=S['pools'] if POOLS else None, vae_draws=vae,
                    forwards=S['forwards'])
    if A.mode == 'traj':
        base.update(series=S['series'], displacement=S['disp'], disp_every=DISP_EVERY, perturbed=S['perturbed'],
                    seed=A.seed, forwards=S['forwards'])
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


# ---------------------------------------------------------------- hooks on the training optimizer

_orig_opt_init = torch.optim.Optimizer.__init__


def _opt_init(self, *a, **k):
    _orig_opt_init(self, *a, **k)
    if A.mode == 'traj' and A.seed is not None and S['perturbed'] is None and S['opt'] is None:
        S['opt'] = self
        model, ema = find_roots(self)
        perturb(model, ema, A.seed)


torch.optim.Optimizer.__init__ = _opt_init


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
    k = S['n']
    if A.mode == 'record':
        model, ema = find_roots(opt)
        S['digests'].append(digests(opt, model, ema))
        S['hparams'].append(hparams(opt))
    elif A.mode == 'traj':
        model, ema = find_roots(opt)
        if S['perturbed'] is None and A.seed is not None:
            fail('the seed perturbation was not applied: the training optimizer was not created through '
                 'torch.optim.Optimizer.__init__')
        sums = S['loss_sums']
        if sums is None:
            fail('modeling.bagel.bagel.Bagel.forward returned no per-token ce/mse in this step')
        dist.all_reduce(sums)
        ce_s, ce_n, mse_s, mse_n = sums.tolist()
        S['series']['ce'].append(ce_s / ce_n if ce_n else None)
        S['series']['mse'].append(mse_s / mse_n if mse_n else None)
        S['loss_sums'] = None
        if len(S['norms']) != 1:
            fail(f'FullyShardedDataParallel.clip_grad_norm_ was called {len(S["norms"])} times in step {k} (once '
                 f'expected): the judge reads the gradient norm there')
        S['series']['grad_norm'].append(float(S['norms'].pop()))
        if k % DISP_EVERY == 0:
            d = displacement(model, ema)
            if d is not None:
                S['disp'].append(d)
    S['n'] += 1
    n = S['n']
    if A.profile_from is not None and RANK == 0:
        if n == A.profile_from:
            S['prof'] = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, with_stack=False)
            S['prof'].__enter__()
        elif S['prof'] is not None and n == A.profile_from + A.profile_iters:
            S['prof'].__exit__(None, None, None)
            S['prof'].export_chrome_trace(f'{A.out}.trace.json')
            table = S['prof'].key_averages().table(sort_by='self_cuda_time_total', row_limit=60)
            with open(f'{A.out}.profile.txt', 'w') as f:
                f.write(table)
            S['prof'] = None
    if n == A.steps:
        finish(opt)


register_optimizer_step_post_hook(post_hook)

sys.path.insert(0, os.getcwd())
sys.argv = [os.path.join(os.getcwd(), 'train', 'pretrain_unified_navit.py'), *TRAIN_ARGV]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except BaseException as exc:
    write({'rank': RANK, 'status': 'error', 'error': f'{type(exc).__name__}: {exc}'[:2000], 'steps': S['n']})
    raise
write({'rank': RANK, 'status': 'error', 'error': f'the training script returned after {S["n"]} of {A.steps} optimizer '
                                                 f'steps', 'steps': S['n']})
sys.exit(3)
