"""numerics adapter for YOCO-3B pretraining (microsoft/unilm YOCO, fairseq fork sunyt32/fairseq@zero_fs).

build() re-creates the real run's setup from the TREE'S OWN code, the way `<tree>/yoco/train.py` does it:
  1. import the entry's modules from the tree (`import models, tasks, criterions`, `from fairseq_cli.train import
     cli_main`; the entry's `__main__` block is not run);
  2. run the statements of the live `cli_main()` with the frozen recipe argv, its `distributed_utils.call_main(cfg,
     main)` calling, in place of `main`, the statements of the live `main()` that come before its epoch loop
     (seeding, task, model, criterion, Trainer, checkpoint/iterator loading, optimizer build via get_lr());
  3. enter `metrics.aggregate("train")` (train()'s decorator) and run the statements of the live `train()` before
     its per-update loop (iterator, GroupedIterator, progress bar, begin_epoch).
step() then runs ONE pass of the body of `for i, samples in enumerate(progress):`, extracted verbatim from the
tree's train() (AST) and executed with the setup's variables -- so the candidate's own loop body, Trainer.train_step
and everything below it is exactly what gets recorded. Gate 0 of the bench requires the entry, cli_main(), main() and
train() outside that loop body to equal the reference AST, so this replay is the real setup.

Cases (bench/cases.json): 'warmup' steps run updates 0 and 1 (LR 0 and 3e-6); before the first 'post' step the
adapter runs the same loop body without recording until num_updates == case['post_start'] (60: past the 50 warmup
updates, LR 1.49995e-4), then records updates 60 and 61. Trajectories (losses only) run updates 0..steps-1.

What the recorder observes (numerics.record):
- loss/sum: this rank's criterion loss (sum over its tokens, as the criterion returns it; captured with a forward
  hook on the criterion module, so the loss must be computed by calling the criterion module); loss/token: that sum
  divided by this rank's sample size (float64);
- grad/, delta/, optim/ on a canonical, partition-independent view of the fp32 master weights ('master'): for every
  trainable parameter, in model.named_parameters() order, this rank keeps the r-th of world-size equal contiguous chunks of
  the flattened tensor (zero padded). Around the real fp32 optimizer step (trainer.optimizer.fp32_optimizer.step,
  called by FP16Optimizer.step after the gradients have been normalized by the sample size and clipped) the view
  is filled with the fp32 master gradient and weight before the step and the weight after it; Adam's exp_avg /
  exp_avg_sq are taken from whichever rank owns the parameter's optimizer state (ZeRO-1: one rank per parameter;
  the owner broadcasts it). The view is a torch optimizer whose step() runs the real step, so the recorder's step
  hooks see the gradient before and the weights after it;
- buffer/bf16.*: the same chunked view of the bf16 model parameters after the update (what the next forward uses);
- forward calls of the model (top-level module hook);
- exact invariants: update counter, scheduled LR of the update and LR after it, Adam hyper-parameters and step count
  of the optimizer holding the state, dtypes of master weights / master gradients / Adam states, the update's batch
  digest and a cumulative digest of every batch consumed so far (per rank), sample size and token count, number of
  criterion calls and of fp32 optimizer steps in the update, a digest of the parsed configuration and of the
  trainable parameter names and shapes.
Contract for candidates (what this adapter reads): trainer.get_model(), trainer.get_criterion() (called as a module
once per micro-batch), trainer.optimizer.fp16_params / fp32_params (list, or fairseq's flat dict) and
trainer.optimizer.fp32_optimizer.step(); Adam state as torch optimizer state ('exp_avg', 'exp_avg_sq') keyed by the
fp32 master parameter, in trainer.optimizer.fp32_optimizer.optimizer or its `.optim`, with 'step' in the param group
(or in the state). A candidate that changes this layout must keep these attributes as views.

Deterministic mode (numerics record --deterministic): torch's deterministic algorithms (warn_only, as numerics sets
them: strict mode refuses the upstream model because torch 2.3 flags CUDA cumsum, used by gate retention over the
256-wide chunk dimension, as nondeterministic; that innermost-dimension scan uses no atomics), cuDNN deterministic, a
fixed cuBLAS workspace, and flash-attn's deterministic backward (flash_attn_interface._flash_attn_backward gets
deterministic=True; upstream does not pass the flag, and its dq accumulation uses atomics otherwise). Same for
reference and candidate. Evidence of determinism: the reference recorded twice is bitwise identical.

Trajectories: everything is the recipe (--seed 1: initialization, data order, per-update seeds), except that right
after main()'s model-building statement (fp32 weights on the CPU, before the Trainer casts them to bf16) every
parameter is multiplied by (1 + eps * z), z ~ N(0, 1) from a private generator seeded with the case seed, eps =
case['init_perturbation'] (bench/traj/spec.json). The seed spread is therefore the trajectory spread under a
perturbation of the initial weights a few bf16 ulps in size -- the scale of the numerical noise a legitimate (N1)
candidate introduces -- with the data stream identical for every seed and for reference and candidate. (A full
re-initialization per seed was measured first: its spread is as large as the effect of a 2% learning-rate change on
these 200 updates, so it cannot tell a wrong candidate from a legitimate one.)

No fp64 golden record: a 3.4B-parameter model under DDP in fp64 is out of reach; per-tensor gates are diagnostics.
"""
import ast
import hashlib
import inspect
import json
import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path

import torch

from numerics.adapter import Adapter, Objects

from bench import recipe, yoco_setup


def deterministic_flash_attn():
    """flash-attn's backward with deterministic=True (no atomics in dq accumulation)."""
    import flash_attn.flash_attn_interface as fi
    for name in ('_flash_attn_backward', '_flash_attn_varlen_backward'):
        f = getattr(fi, name, None)
        if f is None or getattr(f, '_bench_deterministic', False):
            continue
        params = list(inspect.signature(f).parameters)
        if 'deterministic' not in params:
            raise RuntimeError(f'flash_attn_interface.{name} has no deterministic argument')
        idx = params.index('deterministic')

        def wrapped(*args, _f=f, _i=idx, **kwargs):
            if len(args) > _i:
                args = list(args)
                args[_i] = True
            else:
                kwargs['deterministic'] = True
            return _f(*args, **kwargs)
        wrapped._bench_deterministic = True
        setattr(fi, name, wrapped)


def digest_samples(samples):
    """Digest of one update's micro-batches as the loop hands them to trainer.train_step (CPU tensors)."""
    h = hashlib.sha256()

    def walk(x):
        if isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode())
                walk(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(b'[')
            for v in x:
                walk(v)
            h.update(b']')
        elif isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            h.update(f'{t.dtype}{tuple(t.shape)}'.encode())
            h.update(t.view(-1).view(torch.uint8).numpy().tobytes())
        else:
            h.update(repr(x).encode())
    walk(samples)
    return h.hexdigest()[:32]


def _fail_fast():
    """An uncaught exception on one rank ends that process at once (os._exit after the traceback), so torchrun sees
    the failure and stops the other ranks. Without this, a failing rank hangs in interpreter shutdown (NCCL, the
    infinibatch prefetch process) and the whole job sits on the GPUs until the bench's timeout."""
    def hook(etype, value, tb):
        try:
            import traceback
            msg = ''.join(traceback.format_exception(etype, value, tb))
            os.write(2, f'[adapter rank {os.environ.get("RANK")}] uncaught exception, exiting:\n{msg}'.encode())
        finally:
            os._exit(1)
    sys.excepthook = hook


def _isolate_triton_cache_random():
    """Triton 2.3's FileCacheManager.put() draws its temporary file name from Python's GLOBAL `random`
    (triton/runtime/cache.py: rnd_id = random.randint(0, 1000000)). With one cache directory shared by all ranks,
    whether a rank compiles a kernel itself (and calls put) or finds it already written by another rank is a race, so
    the reference consumed the global random stream a different number of times in two identical recordings, and the
    RNG-fingerprint exact item (which includes Python's global random) did not reproduce on most ranks. Giving that
    module its own generator removes the race from the global stream; it changes no computation and applies to the
    reference and every candidate alike. A candidate's own use of the global random stream is still an exact item."""
    import random
    import triton.runtime.cache as tcache
    if not isinstance(getattr(tcache, 'random', None), random.Random):
        tcache.random = random.Random()


class _View(torch.optim.Optimizer):
    """The recorder observes this optimizer. Its step(closure) runs the real fp32 optimizer step (the closure),
    which also refreshes the view's weights and states."""

    def __init__(self, params):
        super().__init__(params, {'lr': 0.0})

    def step(self, closure=None):
        return closure()


def _tree_module(names_shapes, world, kind):
    """A module mirroring the parameter names; 'param' -> fp32 Parameters, 'bf16' -> bf16 buffers; one chunk each."""
    root = torch.nn.Module()
    out = []
    for name, numel, dtype in names_shapes:
        *path, leaf = name.split('.')
        m = root
        for part in path:
            if part not in m._modules:
                m.add_module(part, torch.nn.Module())
            m = m._modules[part]
        per = -(-numel // world)
        if kind == 'param':
            t = torch.nn.Parameter(torch.zeros(per, dtype=torch.float32), requires_grad=True)
            m.register_parameter(leaf, t)
        else:
            t = torch.zeros(per, dtype=dtype)
            m.register_buffer(leaf, t)
        out.append((name, m, leaf))
    return root, out


def _close_prefetch(epoch_itr, timeout=60.0):
    """Close the trainer's iterator (infinibatch prefetch iterators), bounded. When a case ends, the prefetch process
    may still hold a batch in its inter-process queue; once the parent's fetcher thread has stopped, nothing reads
    it, the process cannot flush its queue at exit, and the iterator's _shutdown() joins it forever (seen in the
    reference record on 2-GPU Hopper hosts). After the timeout, the prefetch processes still alive are terminated so
    that close() returns. Runs after the case's last measured step: no recorded value depends on it."""
    import gc
    import threading
    closer = threading.Thread(target=lambda: _quiet(epoch_itr.close), daemon=True)
    closer.start()
    closer.join(timeout)
    if closer.is_alive():
        import multiprocessing
        for obj in gc.get_objects():   # infinibatch's _ForkPrefetchIterator / _ForkPrefetchIteratorExperimental
            module = getattr(type(obj), '__module__', None)   # not a str for some C types
            if not (isinstance(module, str) and module.startswith('infinibatch')):
                continue
            proc = obj.__dict__.get('_prefetch_process') if hasattr(obj, '__dict__') else None
            if isinstance(proc, multiprocessing.Process) and proc.is_alive():
                proc.terminate()
        closer.join(timeout)


def _quiet(fn):
    try:
        fn()
    except Exception:  # noqa: BLE001 -- shutting down the data loader only
        pass


class YocoAdapter(Adapter):
    name = 'yoco-3b-pretrain'

    def __init__(self, repo, scratch, variant=None):
        self.repo = Path(repo).resolve()
        self.scratch = Path(scratch)
        if variant not in (None, 'nondeterministic'):
            raise ValueError(f'unknown variant {variant!r}')
        self.variant = variant
        self.case = {}
        self._record = False
        self._inv = {}

    # ------------------------------------------------------------------ build
    def configure(self, case):
        self.case = dict(case)

    def build(self, seed, golden=False):
        if golden:
            raise NotImplementedError('no fp64 golden record for a 3.4B model under DDP; arbitration is diagnostic')
        for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
            if k not in os.environ:
                raise RuntimeError('numerics record must run under torchrun')
        self.rank, self.world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
        self.local = int(os.environ['LOCAL_RANK'])
        _fail_fast()
        _isolate_triton_cache_random()
        torch.cuda.set_device(self.local)
        # this rank's share of the CPUs, intra-op threads = its physical cores (os.cpu_count() counts the host's)
        from bench.cpu_affinity import bind_local_rank
        self.cpu_binding = bind_local_rank(self.local, int(os.environ.get('LOCAL_WORLD_SIZE', self.world)))
        self.deterministic = torch.are_deterministic_algorithms_enabled() and self.variant != 'nondeterministic'
        if self.deterministic:
            # warn_only as numerics sets it, not strict: torch 2.3 flags its CUDA cumsum as nondeterministic and
            # strict mode stops the upstream model (gate retention's g.float().cumsum(-1) over the 256-wide chunk
            # dimension). That innermost-dimension scan has no atomics; the reference is recorded twice and must be
            # bitwise identical, which is the actual evidence of determinism.
            torch.use_deterministic_algorithms(True, warn_only=True)
            deterministic_flash_attn()
        elif self.variant == 'nondeterministic':
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = False
        self.trajectory = bool(self.case.get('trajectory'))
        # trajectories: the case seed selects a small multiplicative perturbation of the recipe's initial weights
        self.perturb = (int(seed), float(self.case['init_perturbation'])) if self.trajectory else None
        name = self.case.get('name', 'case')
        save_dir = self.scratch / f'{name}_r{self.rank}'
        max_update = int(self.case.get('max_update', 100000))
        script_dir = self.repo / recipe.ENTRY_DIR
        os.chdir(script_dir)
        T = yoco_setup.import_entry(script_dir)
        if not str(Path(T.__file__).resolve()).startswith('/'):
            raise RuntimeError('fairseq_cli.train has no file')
        self._T = T
        argv = recipe.argv(save_dir, max_update)
        holder = {}

        # ---- main(): statements before the epoch loop, wrapped in a function that returns its locals
        main_node, main_globals, _ = yoco_setup.function_ast(T.main)
        pre_main, _while = yoco_setup.split_main(main_node)
        body = []
        built = False
        for stmt in pre_main:
            body.append(stmt)
            if not built and 'build_model' in ast.dump(stmt):
                body.append(ast.parse('__bench_after_build_model__(model)').body[0])
                built = True
        if not built:
            raise RuntimeError('main() has no model-building statement before its epoch loop')
        body.append(ast.parse('return locals()').body[0])
        fn = ast.FunctionDef(name='__bench_main_setup__', args=main_node.args, body=body, decorator_list=[],
                             returns=None, type_comment=None)
        mod = ast.Module(body=[fn], type_ignores=[])
        ast.fix_missing_locations(mod)
        g_main = dict(main_globals)

        def after_build_model(model):
            if self.perturb is None:
                return
            seed, eps = self.perturb
            gen = torch.Generator().manual_seed(seed)      # private generator: the global RNG streams are untouched
            with torch.no_grad():
                for _name, p in model.named_parameters():
                    if p.is_floating_point():
                        p.mul_(1 + eps * torch.randn(p.shape, generator=gen, dtype=p.dtype, device=p.device))
        g_main['__bench_after_build_model__'] = after_build_model
        exec(compile(mod, filename=inspect.getsourcefile(inspect.unwrap(T.main)), mode='exec'), g_main)
        main_setup = g_main['__bench_main_setup__']

        def setup_main(cfg, **kwargs):
            holder['main'] = main_setup(cfg, **kwargs)

        # ---- cli_main(): its statements with `main` bound to setup_main
        cli_node, cli_globals, _ = yoco_setup.function_ast(T.cli_main)
        g_cli = dict(cli_globals)
        g_cli['main'] = setup_main
        fn = ast.FunctionDef(name='__bench_cli__', args=cli_node.args, body=cli_node.body, decorator_list=[],
                             returns=None, type_comment=None)
        mod = ast.Module(body=[fn], type_ignores=[])
        ast.fix_missing_locations(mod)
        exec(compile(mod, filename=inspect.getsourcefile(inspect.unwrap(T.cli_main)), mode='exec'), g_cli)
        saved = sys.argv
        sys.argv = [str(script_dir / yoco_setup.ENTRY), *argv]
        try:
            g_cli['__bench_cli__']()
        finally:
            sys.argv = saved
        ns = holder['main']
        cfg, trainer, task, epoch_itr = ns['cfg'], ns['trainer'], ns['task'], ns['epoch_itr']
        # main()'s while condition and its first-iteration check
        if not (epoch_itr.next_epoch_idx <= ns['max_epoch']):
            raise RuntimeError('main() would not enter its epoch loop')
        if ns['lr'] <= cfg.optimization.stop_min_lr:
            raise RuntimeError('main() would stop before training (lr <= stop_min_lr)')
        self.cfg, self.trainer, self.task, self.epoch_itr = cfg, trainer, task, epoch_itr

        # ---- train(): enter its decorator, run the statements before the per-update loop
        train_node, train_globals, train_file = yoco_setup.function_ast(T.train)
        pre_train, loop, post_train = yoco_setup.split_train(train_node)
        if not (isinstance(train_node.decorator_list, list) and len(train_node.decorator_list) == 1):
            raise RuntimeError('train() is expected to have exactly its @metrics.aggregate("train") decorator')
        self._stack = ExitStack()
        g_train = dict(train_globals)
        deco = eval(compile(ast.Expression(body=train_node.decorator_list[0]), train_file, 'eval'), g_train)
        self._stack.enter_context(deco)
        fn = ast.FunctionDef(name='__bench_train_setup__', args=train_node.args,
                             body=[*pre_train, ast.parse('return locals()').body[0]], decorator_list=[],
                             returns=None, type_comment=None)
        mod = ast.Module(body=[fn], type_ignores=[])
        ast.fix_missing_locations(mod)
        exec(compile(mod, filename=train_file, mode='exec'), g_train)
        tns = g_train['__bench_train_setup__'](cfg, trainer, task, epoch_itr)
        self._ns = dict(train_globals)
        self._ns.update(tns)
        # the loop body, once per exec; `for _ in (0,)` keeps a `break`/`continue` in the body legal
        target_i, target_s = loop.target.elts
        if not (isinstance(target_i, ast.Name) and isinstance(target_s, ast.Name)):
            raise RuntimeError('the update loop target is not `i, samples`')
        self._names = (target_i.id, target_s.id)
        once = ast.For(target=ast.Name(id='__bench_once__', ctx=ast.Store()),
                       iter=ast.Tuple(elts=[ast.Constant(value=0)], ctx=ast.Load()), body=loop.body, orelse=[],
                       type_comment=None)
        mod = ast.Module(body=[once], type_ignores=[])
        ast.fix_missing_locations(mod)
        self._code = compile(mod, filename=train_file, mode='exec')
        self._enum = iter(enumerate(self._ns['progress']))

        # ---- observation
        self._crit = []
        crit = trainer.get_criterion()

        def on_criterion(_m, _a, output):
            loss, sample_size = output[0], output[1]
            self._crit.append((loss.detach().clone(), sample_size))
        self._handles = [crit.register_forward_hook(on_criterion)]
        self._fp32_steps = 0
        opt = trainer.optimizer
        fo = getattr(opt, 'fp32_optimizer', None)
        if fo is None or not hasattr(fo, 'step'):
            raise RuntimeError('trainer.optimizer has no fp32_optimizer.step (see the adapter contract)')
        real_step = fo.step
        adapter = self

        def fp32_step(*args, **kwargs):
            adapter._fp32_steps += 1
            if not adapter._record:
                return real_step(*args, **kwargs)
            adapter._fill('pre')
            adapter.view_opt.param_groups[0]['lr'] = adapter._lr_scheduled
            out = []

            def closure():
                out.append(real_step(*args, **kwargs))
                adapter._fill('post')
                return out[0]
            adapter.view_opt.step(closure)
            return out[0]
        fo.step = fp32_step
        self._fo = fo
        self._cfg_sha = hashlib.sha256(self._cfg_text(cfg).encode()).hexdigest()[:32]
        params = self._trainable()
        self._trainable_sha = hashlib.sha256(json.dumps([(n, list(p.shape), str(p.dtype)) for n, p, _ in params])
                                             .encode()).hexdigest()[:32]
        self._cum = hashlib.sha256()
        self._ff_updates = 0

    @staticmethod
    def _cfg_text(cfg):
        from omegaconf import OmegaConf
        c = OmegaConf.to_container(cfg, resolve=True)
        c.get('checkpoint', {}).pop('save_dir', None)
        for k in ('distributed_rank', 'device_id', 'distributed_port', 'distributed_init_method'):
            c.get('distributed_training', {}).pop(k, None)
        return json.dumps(c, sort_keys=True, default=str)

    # ------------------------------------------------------------------ canonical views
    def _trainable(self):
        """[(name, bf16 param, fp32 master (tensor or (flat, offset)))] in the optimizer's fp16_params order."""
        opt = self.trainer.optimizer
        p16s = getattr(opt, 'fp16_params', None)
        p32s = getattr(opt, 'fp32_params', None)
        if p16s is None or p32s is None:
            raise RuntimeError('trainer.optimizer has no fp16_params / fp32_params (see the adapter contract)')
        names = {}
        for n, p in self.trainer.get_model().named_parameters():
            names.setdefault(id(p), n)
        for n, p in self.trainer.get_criterion().named_parameters():
            names.setdefault(id(p), f'criterion.{n}')
        out = []
        if isinstance(p32s, dict) or torch.is_tensor(p32s):
            flat = p32s if torch.is_tensor(p32s) else next(iter(p32s.values()))
            off = 0
            for p in p16s:
                if not p.requires_grad:
                    continue
                out.append((names.get(id(p), f'param{len(out)}'), p, (flat, off)))
                off += p.numel()
        else:
            p32s = list(p32s)
            if len(p32s) != len(p16s):
                raise RuntimeError('fp16_params and fp32_params differ in length')
            for p, p32 in zip(p16s, p32s):
                out.append((names.get(id(p), f'param{len(out)}'), p, p32))
        return out

    def _chunk(self, flat, dtype=torch.float32):
        n = flat.numel()
        per = -(-n // self.world)
        a, b = self.rank * per, min(n, (self.rank + 1) * per)
        out = torch.zeros(per, dtype=dtype)
        if a < b:
            out[:b - a].copy_(flat[a:b].to(dtype))
        return out

    @staticmethod
    def _master(p32, numel):
        """(weight, grad) of one master parameter, flattened."""
        if isinstance(p32, tuple):
            flat, off = p32
            g = flat.grad
            return flat.data[off:off + numel], (g[off:off + numel] if g is not None else None)
        return p32.data.reshape(-1), (p32.grad.reshape(-1) if p32.grad is not None else None)

    def _holders(self):
        """Torch optimizers that may hold the Adam state of the fp32 masters."""
        out, seen = [], set()
        stack = [getattr(self._fo, 'optimizer', None), getattr(self._fo, '_optimizer', None)]
        while stack:
            o = stack.pop()
            if o is None or id(o) in seen:
                continue
            seen.add(id(o))
            if isinstance(o, torch.optim.Optimizer):
                out.append(o)
            for attr in ('optim', 'optimizer', '_optimizer'):
                inner = getattr(o, attr, None)
                if isinstance(inner, torch.optim.Optimizer):
                    stack.append(inner)
        return out

    def _fill(self, when):
        dtypes = self._dtypes
        if when == 'pre':
            for (name, p16, p32), vp in zip(self._params, self._view_params):
                w, g = self._master(p32, p16.numel())
                if g is None:
                    raise RuntimeError(f'no fp32 master gradient for {name} at the optimizer step')
                dtypes.setdefault('master', set()).add(str(w.dtype))
                dtypes.setdefault('master_grad', set()).add(str(g.dtype))
                vp.data = self._chunk(w)
                vp.grad = self._chunk(g)
            return
        import torch.distributed as dist
        holders = self._holders()
        owner = torch.zeros(len(self._params), dtype=torch.int64, device='cuda')
        local = []
        for j, (name, p16, p32) in enumerate(self._params):
            key = p32[0] if isinstance(p32, tuple) else p32
            st = None
            for h in holders:
                s = h.state.get(key)
                if s and 'exp_avg' in s and 'exp_avg_sq' in s:
                    st = s
                    break
            local.append(st)
            if st is not None:
                owner[j] = self.rank + 1
        dist.all_reduce(owner, op=dist.ReduceOp.MAX)
        owners = owner.tolist()
        for j, ((name, p16, p32), vp) in enumerate(zip(self._params, self._view_params)):
            w, _ = self._master(p32, p16.numel())
            vp.data = self._chunk(w)
            if owners[j] == 0:
                continue                      # no Adam state anywhere: the optim/ tensors are missing
            src = owners[j] - 1
            state = {}
            for key in ('exp_avg', 'exp_avg_sq'):
                if src == self.rank:
                    t = local[j][key]
                    if isinstance(p32, tuple):
                        off = p32[1]
                        t = t.reshape(-1)[off:off + p16.numel()]
                    meta = torch.tensor([{torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}.get(t.dtype, 3)],
                                        device='cuda')
                else:
                    t, meta = None, torch.zeros(1, dtype=torch.int64, device='cuda')
                dist.broadcast(meta, src=src)
                dt = [torch.float32, torch.bfloat16, torch.float16, torch.float64][int(meta.item())]
                buf = t.reshape(-1).contiguous() if t is not None else torch.empty(p16.numel(), dtype=dt,
                                                                                    device='cuda')
                dist.broadcast(buf, src=src)
                dtypes.setdefault(key, set()).add(str(dt))
                state[key] = self._chunk(buf)
                del buf
            self.view_opt.state[vp] = state
        hp = set()
        for h in holders:
            for grp in h.param_groups:
                if not any(len(h.state.get(p, {})) for p in grp['params']):
                    continue
                step = grp.get('step')
                if step is None:
                    p0 = next((p for p in grp['params'] if 'step' in h.state.get(p, {})), None)
                    step = h.state[p0]['step'] if p0 is not None else None
                step = float(step) if step is not None else None
                hp.add(json.dumps({'betas': list(grp.get('betas', ())), 'eps': grp.get('eps'),
                                   'weight_decay': grp.get('weight_decay'),
                                   'bias_correction': grp.get('bias_correction'), 'amsgrad': grp.get('amsgrad'),
                                   'adam_w_mode': getattr(h, 'adam_w_mode', None), 'step': step}, sort_keys=True))
        gathered = [None] * self.world
        dist.all_gather_object(gathered, sorted(hp))
        self._adam = sorted({x for g in gathered for x in g})

    def _fill_bf16(self):
        for (name, p16, _), (_n, m, leaf) in zip(self._params, self._bf16_slots):
            m._buffers[leaf] = self._chunk(p16.detach().reshape(-1), dtype=p16.dtype)

    # ------------------------------------------------------------------ Adapter API
    def objects(self):
        self._record = True
        self._params = self._trainable()
        spec = [(n, p.numel(), p.dtype) for n, p, _ in self._params]
        self.view_module, slots = _tree_module(spec, self.world, 'param')
        self._view_params = [getattr(m, leaf) for _n, m, leaf in slots]
        self.view_opt = _View(self._view_params)
        self.bf16_module, self._bf16_slots = _tree_module(spec, self.world, 'bf16')
        self._dtypes = {}
        self._adam = []
        return Objects(models={'model': self.trainer.get_model(), 'master': self.view_module,
                               'bf16': self.bf16_module},
                       optimizers={'adam': self.view_opt},
                       sync={'model': 'replicated', 'master': 'local', 'bf16': 'local'},
                       count_forwards=True)

    def phases(self):
        return ['warmup', 'post', 'train']

    def _natural(self):
        return 'warmup' if self.trainer.get_num_updates() < recipe.WARMUP_UPDATES else 'post'

    def set_phase(self, phase, step):
        if phase == 'train':
            return
        if phase == 'post':
            target = int(self.case.get('post_start', 60))     # the first 'post' step runs update `target`
            saved, self._record = self._record, False
            try:
                while self.trainer.get_num_updates() < target:
                    self._update()
                    self._ff_updates += 1
            finally:
                self._record = saved
        natural = self._natural()
        if phase != natural:
            raise RuntimeError(f'case step {step} is labelled {phase!r} but update {self.trainer.get_num_updates()} '
                               f'is a {natural!r} update')

    def batch(self, step, rank):
        return None          # the trainer's own (deterministic) iterator; its batches are digested as invariants

    def _update(self):
        """One pass of the tree's per-update loop body."""
        t = self.trainer
        nu = t.get_num_updates()
        self._lr_scheduled = float(t.get_lr())
        self._crit = []
        self._fp32_steps = 0
        i, samples = next(self._enum)
        dig = digest_samples(samples)
        self._cum.update(dig.encode())
        ns = self._ns
        ns[self._names[0]], ns[self._names[1]] = i, samples
        exec(self._code, ns)
        if ns.get('should_stop'):
            raise RuntimeError('the training loop asked to stop (max_update reached?)')
        if t.get_num_updates() != nu + 1:
            raise RuntimeError(f'the loop body moved num_updates from {nu} to {t.get_num_updates()}, expected '
                               f'{nu + 1} (overflow / skipped update?)')
        return dig

    def step(self, step, batch):
        t = self.trainer
        dig = self._update()
        if not self._crit:
            raise RuntimeError('the criterion module was not called during the update')
        loss_sum = torch.stack([c[0].float() for c in self._crit]).sum() if len(self._crit) > 1 else self._crit[0][0]
        sample_size = sum(float(c[1]) for c in self._crit)
        losses = {'sum': loss_sum, 'token': loss_sum.detach().double() / sample_size}
        if self._record:
            self._fill_bf16()
        inv = {'num_updates': int(t.get_num_updates()), 'lr_scheduled': self._lr_scheduled,
               'lr_after': float(t.get_lr()), 'data/batch': dig,
               'data/cumulative': self._cum.hexdigest()[:32], 'sample_size': sample_size,
               'criterion_calls': len(self._crit), 'fp32_optimizer_steps': self._fp32_steps,
               'fast_forward_updates': self._ff_updates, 'cfg': self._cfg_sha, 'trainable': self._trainable_sha,
               'world_size': self.world}
        if self._record:
            inv['adam'] = list(self._adam)
            inv['dtypes'] = {k: sorted(v) for k, v in sorted(self._dtypes.items())}
        self._inv = inv
        if step == int(self.case.get('steps', 1)) - 1:
            self._finish()
        return losses

    def _finish(self):
        for h in getattr(self, '_handles', []):
            h.remove()
        self._handles = []
        try:
            self._stack.close()
        except Exception:  # noqa: BLE001 -- metrics bookkeeping only
            pass
        _close_prefetch(self.epoch_itr)  # the infinibatch prefetch process

    def invariants(self):
        return dict(self._inv)

    def canonical(self, kind, name, tensor):
        return {name: tensor}

    def save_state(self, path):
        raise NotImplementedError('cases start from scratch and fast-forward with the tree\'s own loop body')

    def load_state(self, path):
        raise NotImplementedError('cases start from scratch and fast-forward with the tree\'s own loop body')
