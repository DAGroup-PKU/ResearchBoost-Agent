"""The judge's entry of every Gaussian Splatting run: one process, from the root of the tree under test.

    gs_entry.py --mode time|record|traj --out PREFIX --steps N [--seed S] [--marks K1,K2,...]
                [--profile-from K --profile-iters M] -- <recipe arguments>

It imports the tree's train.py as a module and calls its `training()` with the frozen recipe (bench/recipe.py): the
argument groups of the tree's `arguments` package parse the recipe's command line, the test, save and checkpoint
iteration lists are empty and the network viewer is never started. Before train.py is imported:
  - random, numpy and torch are seeded with S (default 0, as upstream's safe_state);
  - gaussian_renderer.render is wrapped (train.py binds it at import): in record and traj modes every call records
    the image name of its camera; in time mode the wrapper is not installed;
  - scene.gaussian_model.GaussianModel.__init__ is wrapped: every instance is registered.
A global optimizer step post-hook (torch.optim.optimizer.register_optimizer_step_post_hook) observes every step. The
Gaussian optimizer is the one whose parameter groups are named xyz, f_dc, f_rest, opacity, scaling, rotation; any
other optimizer that steps (upstream: the exposure optimizer) is counted separately. After Gaussian-optimizer step N
the run ends (upstream's last iteration takes no optimizer step, so the parameters after step N = iterations - 1 are
the trained ones): the entry writes PREFIX.json and exits.

--mode time: at each mark (Gaussian-optimizer step numbers) CUDA is synchronized and the time stamped; nothing else is
observed. --profile-from K --profile-iters M records a torch.profiler trace of steps K..K+M-1.
--mode record / traj: per step, the number of Gaussians (a host-side shape read, no synchronization); at each mark
the learning rate of every parameter group of both optimizers and the active SH degree; at the end the dtypes of the
parameters and optimizer states and the camera sequence. traj additionally saves the trained parameters
(PREFIX.gaussians.pt: the raw _xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation and the SH degrees)
for the judge's own evaluation (bench/gs_eval.py), which renders them with the reference rasterizer."""
import argparse
import json
import os
import random
import sys
import time

T_START = time.time()
cut = sys.argv.index('--')
p = argparse.ArgumentParser()
p.add_argument('--mode', choices=('time', 'record', 'traj'), required=True)
p.add_argument('--out', required=True)
p.add_argument('--steps', type=int, required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--marks', default='')
p.add_argument('--profile-from', type=int)
p.add_argument('--profile-iters', type=int, default=2)
A = p.parse_args(sys.argv[1:cut])
TRAIN_ARGV = sys.argv[cut + 1:]
OBSERVE = A.mode in ('record', 'traj')
MARKS = sorted({int(x) for x in A.marks.split(',') if x} | {A.steps})
GAUSSIAN_GROUPS = ('xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation')

sys.path.insert(0, os.getcwd())
# the CUDA extensions the bench built from this tree (diff_gaussian_rasterization, simple_knn, fused_ssim) come first
if os.environ.get('GS_BENCH_EXT'):
    sys.path.insert(0, os.environ['GS_BENCH_EXT'])

import numpy as np  # noqa: E402
import torch  # noqa: E402

random.seed(A.seed)
np.random.seed(A.seed)
torch.manual_seed(A.seed)
torch.cuda.set_device(torch.device('cuda:0'))

S = {'status': 'running', 'mode': A.mode, 'seed': A.seed, 'steps': 0, 'other_optimizer_steps': 0,
     'gaussian_optimizers': 0, 'marks': {}, 'counts': [], 'cameras': [], 'models': 0}
STATE = {'gaussian_opt': None, 'model': None, 'last_count': None, 'prof': None, 'done': False}


def write(payload):
    tmp = f'{A.out}.json.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f)
    os.replace(tmp, f'{A.out}.json')


def fail(message):
    S.update(status='error', error=message[:2000])
    write(S)
    sys.stdout.flush()
    os._exit(4)


# ---------------------------------------------------------------- observation of the tree's modules

try:
    import gaussian_renderer
    import scene.gaussian_model as gaussian_model_module
except Exception as e:  # noqa: BLE001
    fail(f'importing gaussian_renderer / scene.gaussian_model failed: {type(e).__name__}: {e}')

if OBSERVE:
    _render = gaussian_renderer.render

    def render(viewpoint_camera, *args, **kwargs):
        if not STATE['done']:
            S['cameras'].append(getattr(viewpoint_camera, 'image_name', None))
        return _render(viewpoint_camera, *args, **kwargs)
    render.__wrapped__ = _render
    gaussian_renderer.render = render

_GM = gaussian_model_module.GaussianModel
_gm_init = _GM.__init__


def _gm_init_observed(self, *args, **kwargs):
    _gm_init(self, *args, **kwargs)
    S['models'] += 1
    STATE['model'] = self


_GM.__init__ = _gm_init_observed

try:
    import train  # noqa: E402  (the tree's train.py; binds gaussian_renderer.render now)
    from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
except Exception as e:  # noqa: BLE001
    fail(f'importing the tree\'s train.py failed: {type(e).__name__}: {e}')


# ---------------------------------------------------------------- the step hook

def group_names(opt):
    return tuple(g.get('name') for g in opt.param_groups)


def lrs(opt):
    return {str(g.get('name')): float(g['lr']) for g in opt.param_groups}


def dtypes():
    out = {'parameters': {}, 'optimizer_state': []}
    opt, model = STATE['gaussian_opt'], STATE['model']
    if model is not None:
        for n in ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation'):
            t = getattr(model, n, None)
            out['parameters'][n] = None if t is None else [str(t.dtype), list(t.shape[1:])]
    if opt is not None:
        seen = set()
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v) and v.dim() > 0:
                    seen.add((k, str(v.dtype)))
        out['optimizer_state'] = sorted(seen)
    return out


def save_gaussians():
    m = STATE['model']
    if m is None:
        fail('no GaussianModel was created')
    blob = {n: getattr(m, '_' + n).detach().float().contiguous().cpu() for n in
            ('xyz', 'features_dc', 'features_rest', 'opacity', 'scaling', 'rotation')}
    blob['active_sh_degree'] = int(m.active_sh_degree)
    blob['max_sh_degree'] = int(m.max_sh_degree)
    torch.save(blob, f'{A.out}.gaussians.pt')


def finish():
    STATE['done'] = True
    torch.cuda.synchronize()
    S.update(status='ok', end_s=time.time(),
             peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
             peak_memory_reserved_bytes=torch.cuda.max_memory_reserved())
    if STATE['prof'] is not None:
        STATE['prof'].stop()
        STATE['prof'].export_chrome_trace(f'{A.out}.trace.json')
        with open(f'{A.out}.profile.txt', 'w') as f:
            f.write(STATE['prof'].key_averages().table(sort_by='cuda_time_total', row_limit=60))
    if OBSERVE:
        S['dtypes'] = dtypes()
        S['final_count'] = int(STATE['model'].get_xyz.shape[0]) if STATE['model'] is not None else None
    if A.mode == 'traj':
        save_gaussians()
    write(S)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def post_hook(opt, args, kwargs):
    if STATE['done']:
        return
    names = group_names(opt)
    if set(names) != set(GAUSSIAN_GROUPS):
        S['other_optimizer_steps'] += 1
        if OBSERVE and S['steps'] + 1 in MARKS:
            S['marks'].setdefault(str(S['steps'] + 1), {})['other_lr'] = lrs(opt)
        return
    if STATE['gaussian_opt'] is not opt:
        S['gaussian_optimizers'] += 1
        STATE['gaussian_opt'] = opt
    S['steps'] += 1
    k = S['steps']
    if k == 1:
        S['startup_to_first_step_s'] = time.time() - T_START
    if OBSERVE:
        n = int(STATE['model'].get_xyz.shape[0]) if STATE['model'] is not None else -1
        if n != STATE['last_count']:
            S['counts'].append([k, n])
            STATE['last_count'] = n
    if k in MARKS:
        if A.mode == 'time':
            torch.cuda.synchronize()
            S['marks'][str(k)] = {'t': time.perf_counter()}
        else:
            entry = S['marks'].setdefault(str(k), {})
            entry['lr'] = lrs(opt)
            entry['active_sh_degree'] = int(STATE['model'].active_sh_degree) if STATE['model'] is not None else None
    if A.profile_from is not None:
        if k == A.profile_from - 1:
            STATE['prof'] = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA])
            STATE['prof'].start()
        elif STATE['prof'] is not None and k == A.profile_from + A.profile_iters - 1:
            finish()
    if k >= A.steps:
        finish()


from torch.optim.optimizer import register_optimizer_step_post_hook  # noqa: E402

register_optimizer_step_post_hook(post_hook)


# ---------------------------------------------------------------- the run

def main():
    parser = argparse.ArgumentParser()
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    args = parser.parse_args(TRAIN_ARGV)
    args.model_path = f'{A.out}.model'
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    if A.steps > opt.iterations - 1:
        fail(f'--steps {A.steps} > iterations - 1 = {opt.iterations - 1} (the last iteration takes no step)')
    train.training(dataset, opt, pipe, [], [], [], None, -1)
    # training() returned without reaching step N: the tree took fewer Gaussian-optimizer steps than the recipe
    fail(f'training() returned after {S["steps"]} Gaussian-optimizer steps, {A.steps} expected')


try:
    main()
except SystemExit as e:
    fail(f'the tree called sys.exit({e.code}) after {S["steps"]} steps')
except BaseException as e:  # noqa: BLE001
    import traceback
    fail(f'{type(e).__name__}: {e} after {S["steps"]} steps\n{traceback.format_exc()[-1500:]}')
