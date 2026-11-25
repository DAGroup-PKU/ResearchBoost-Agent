"""The frozen training recipe: State transition model, Tahoe-100M slice, one GPU.

Everything here is part of the workload. A candidate may not change it: the bench passes these Hydra
overrides on every run, so the tree's own config defaults cannot move them. What a candidate MAY change is
anything the training code does under this recipe: the model modules, the loss, the data pipeline (as long
as the batches are identical), the optimizer implementation, kernels, precision within N2.
"""
from pathlib import Path

ENTRY = 'src/state/__main__.py'      # gate 0 reads this for the AST compare
SRC = 'src'                          # the tree's package root, first on sys.path
MODEL = 'tahoe_best'                 # the preprint's best Tahoe configuration
CELL_SET_LEN = 512                   # model.kwargs.cell_set_len, from model/tahoe_best.yaml
BATCH_SIZE = 8                       # cell sentences per GPU step
DEVICES = 1                          # fixed per-GPU load; the agent must not scale it to fill the machine
UPDATE_RATIO = 1                     # one optimizer step per iteration (single Adam over all parameters)


def hydra_overrides(toml, out_dir, name, max_steps):
    """The Hydra override list of one training run. `out_dir` and `name` are bench-managed (reserved_env)."""
    return [
        'tx', 'train',
        f'data.kwargs.toml_config_path={toml}',
        'data.kwargs.embed_key=X_hvg',
        'data.kwargs.pert_col=perturbation',
        'data.kwargs.cell_type_key=cell_type',
        'data.kwargs.control_pert=DMSO_TF',
        'data.kwargs.batch_col=gem_group',
        f'model={MODEL}',
        f'training.batch_size={BATCH_SIZE}',
        f'training.devices={DEVICES}',
        f'training.max_steps={max_steps}',
        'training.val_freq=100000000',
        'training.ckpt_every_n_steps=100000000',
        'training.strategy=auto',
        'use_wandb=false',
        'overwrite=true',
        f'output_dir={out_dir}',
        f'name={name}',
    ]


def adapter_recipe(toml, batch_size=None):
    """The same recipe as plain keyword arguments, for the numerics adapter (which builds the objects itself)."""
    return {
        'toml_config_path': str(toml),
        'embed_key': 'X_hvg',
        'pert_col': 'perturbation',
        'cell_type_key': 'cell_type',
        'control_pert': 'DMSO_TF',
        'batch_col': 'gem_group',
        'output_space': 'all',
        'batch_size': BATCH_SIZE if batch_size is None else batch_size,
        'cell_set_len': CELL_SET_LEN,
        'model': MODEL,
    }
