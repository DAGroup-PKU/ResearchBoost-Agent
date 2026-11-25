"""Task adapter: drives one real State training step for the numerics judge.

What the adapter does NOT do: implement the model, the loss, or the step. It builds the tree's own objects
through the tree's own factories (`get_datamodule`, `get_lightning_module`) from the tree's own Hydra config,
and runs the tree's own `training_step`. Everything the judge records - gradients, parameter updates,
optimizer state, buffers, counters, dtypes - is captured by the generic hooks in `numerics.record`.

The optimizer step is reproduced the way PyTorch Lightning's default loop runs it: zero_grad, forward+loss
(`training_step`), backward, gradient clip (`training.gradient_clip_val`, 10 in the frozen recipe), step. The
adapter keeps `model._trainer = None`, so `self.log()` inside `training_step` is skipped with a warning; logging
does not touch the math.

Batches are replayed from a frozen fixture (bench/fixture.pt, built once by bench/build_fixture.py). The fixture
is the same for the reference and every candidate, so a candidate that changes *how* batches are produced is
measured by gate 1 (real entry point, real dataloader) and judged for numerics here on identical inputs.
"""
import json
import os
import sys
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from numerics.adapter import Adapter, Objects


def _compose(tree, overrides):
    """The tree's own Hydra config, composed exactly as `state tx train` composes it."""
    from hydra import compose, initialize_config_dir
    sys.path.insert(0, str(Path(tree) / 'src'))
    cfg_dir = str((Path(tree) / 'src' / 'state' / 'configs').resolve())
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        return compose(config_name='config', overrides=list(overrides))


class StateAdapter(Adapter):
    name = 'state'

    def __init__(self, repo, toml, fixture, steps=1, batch_size=8, cell_set_len=512, model='tahoe_best',
                 lr=1e-4, weight_decay=5e-4, grad_clip=10.0, step_offset=0, dataset_dir=None):
        self.repo = str(Path(repo).resolve())
        self.toml, self.fixture = str(toml), str(fixture)
        self.steps, self.batch_size, self.cell_set_len = int(steps), int(batch_size), int(cell_set_len)
        self.model_name, self.lr, self.weight_decay, self.grad_clip = model, lr, weight_decay, grad_clip
        self.step_offset = int(step_offset)
        self.dataset_dir = dataset_dir
        self._batches, self.losses = None, []
        self._toml_copy = None

    def _bind_tree(self):
        src = str(Path(self.repo) / 'src')
        if src in sys.path:
            sys.path.remove(src)
        sys.path.insert(0, src)
        for name in [m for m in sys.modules if m == 'state' or m.startswith('state.')]:
            del sys.modules[name]
        import state as _state
        if not Path(_state.__file__).resolve().is_relative_to(Path(self.repo).resolve()):
            raise RuntimeError(f'state resolved to {_state.__file__}, not to the tree under test {self.repo}')

    def _resolve_toml(self):
        """cell_load reads the TOML itself and does not expand environment variables; resolve them into a
        private copy so the dataset directory can be named through ${ASSETS} in bench.json."""
        import re, tempfile
        text = Path(self.toml).read_text()
        env = dict(os.environ)
        if self.dataset_dir:
            env['DATASET_DIR'] = self.dataset_dir
        text = re.sub(r'\$\{(\w+)\}', lambda m: env.get(m.group(1), m.group(0)), text)
        if self._toml_copy is None:
            d = tempfile.mkdtemp(prefix='state-toml-')
            p = Path(d) / 'dataset.toml'
            p.write_text(text)
            self._toml_copy = str(p)
        return self._toml_copy

    # ---------------------------------------------------------------- required

    def build(self, seed, golden=False):
        # Deterministic stand-in for the numerics record: flash and memory-efficient SDPA run their backward with
        # nondeterministic atomics, so the baseline cannot reproduce itself bit for bit with them on (measured:
        # 79/79 gradients differ run to run with flash, 0/79 with it off). The record therefore fixes the SDPA
        # backends to the math implementation, identically for the reference and for every candidate, and
        # `math_mode` records that choice per step so a candidate that switches it back is rejected. Precision is
        # unchanged (the math backend uses the same dtypes); gate 1 measures speed with the recipe untouched.
        if torch.are_deterministic_algorithms_enabled():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        # The tree under test must be the `state` package that gets imported: put its src first on sys.path and
        # drop anything already imported from another copy (an editable install of the upstream package is on
        # sys.path in this environment, and `state` imported before this point would silently be that copy).
        self._bind_tree()

        from cell_load.utils.modules import get_datamodule
        from state.tx.utils import get_lightning_module

        overrides = [
            f'data.kwargs.toml_config_path={self._resolve_toml()}',
            'data.kwargs.embed_key=X_hvg',
            'data.kwargs.pert_col=perturbation',
            'data.kwargs.cell_type_key=cell_type',
            'data.kwargs.control_pert=DMSO_TF',
            'data.kwargs.batch_col=gem_group',
            'data.kwargs.num_workers=0',          # no worker processes: the fixture replays the inputs
            f'model={self.model_name}',
            f'training.batch_size={self.batch_size}',
            f'training.lr={self.lr}',
            f'training.weight_decay={self.weight_decay}',
            f'training.gradient_clip_val={self.grad_clip}',
        ]
        cfg = _compose(self.repo, overrides)
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(cfg, resolve=True)

        dm = get_datamodule(cfg['data']['name'], cfg['data']['kwargs'], batch_size=cfg['training']['batch_size'],
                            cell_sentence_len=self.cell_set_len)
        dm.setup(stage='fit')
        var_dims = dm.get_var_dims()
        self._dm, self._var_dims = dm, var_dims

        # The decoder config the trainer injects into the model kwargs (see _cli/_tx/_train.py).
        output_space = cfg['data']['kwargs'].get('output_space', 'gene')
        gene_dim = var_dims['hvg_dim'] if output_space == 'gene' else var_dims['gene_dim']
        cfg['model']['kwargs']['decoder_cfg'] = dict(
            latent_dim=var_dims['output_dim'], gene_dim=gene_dim,
            hidden_dims=cfg['model']['kwargs'].get('decoder_hidden_dims', [1024, 1024, 512]),
            dropout=cfg['model']['kwargs'].get('decoder_dropout', 0.1),
            residual_decoder=cfg['model']['kwargs'].get('residual_decoder', False),
        )

        model = get_lightning_module(cfg['model']['name'], cfg['data']['kwargs'], cfg['model']['kwargs'],
                                     cfg['training'], var_dims)
        model._trainer = None                       # self.log() is skipped; it does not affect the math
        if golden:
            model = model.double()
        self.model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

        # The tree's OWN optimizer, through the tree's own hook: a candidate that changes how the optimizer is
        # built (fused/foreach, a different class, gradient handling) must be visible to the judge, and the
        # reference must use exactly what `trainer.fit` would use.
        opt = self.model.configure_optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        if isinstance(opt, dict):
            opt = opt['optimizer']
        self.opt = opt

        fx = torch.load(self.fixture, map_location='cpu', weights_only=False)
        want = 'golden' if golden else 'ref'
        self._batches = fx[want][: self.steps]
        if len(self._batches) < self.steps:
            raise ValueError(f'fixture has {len(self._batches)} batches, case needs {self.steps}')

    def objects(self):
        return Objects(models={'model': self.model}, optimizers={'adam': self.opt},
                       sync={'model': 'replicated'}, capture_outputs=('model',), count_forwards=True)

    def batch(self, step, rank):
        b = self._batches[step]
        dev = next(self.model.parameters()).device
        out = {}
        for k, v in b.items():
            if torch.is_tensor(v):
                if v.is_sparse:
                    v = v.to_dense()
                if v.is_floating_point():
                    v = v.to(dev, dtype=next(self.model.parameters()).dtype)
                else:
                    v = v.to(dev)
            out[k] = v
        return out

    def step(self, step, batch):
        self.opt.zero_grad(set_to_none=True)
        loss = self.model.training_step(batch, step)
        loss.backward()
        if self.grad_clip:
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.opt.step()
        self._step = step
        return {'train_loss': loss.detach()}

    # ---------------------------------------------------------------- optional

    def invariants(self):
        g = self.opt.param_groups[0]
        return {'lr': float(g['lr']),
                'step': int(self._step + self.step_offset),
                'weight_decay': float(g.get('weight_decay', 0.0)),
                'optimizer': type(self.opt).__name__}

    def save_state(self, path):
        torch.save({'model': self.model.state_dict(), 'opt': self.opt.state_dict()}, path)

    def load_state(self, path):
        st = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(st['model'])
        self.opt.load_state_dict(st['opt'])
