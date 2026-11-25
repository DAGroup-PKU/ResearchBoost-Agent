"""Formulator: freeze the batches the numerics judge replays.

Draws `--batches` batches from the tree's own dataloader with a fixed seed and writes them to one .pt file,
in two variants: `ref` (the recipe's precision) and `golden` (float64, for fp64 arbitration). Replaying a
frozen batch keeps gate 2/3 independent of the data pipeline, which gate 1 measures separately on the real
entry point.

usage: python bench/build_fixture.py --tree <tree> --toml <dataset.toml> --out bench/fixture.pt [--batches 10]
"""
import argparse, random, sys
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--tree', required=True)
ap.add_argument('--toml', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--batches', type=int, default=10)
ap.add_argument('--batch-size', type=int, default=8)
ap.add_argument('--cell-set-len', type=int, default=512)
ap.add_argument('--seed', type=int, default=42)
a = ap.parse_args()

import os, re, tempfile
_toml_text = Path(a.toml).read_text()
_toml_text = re.sub(r'\$\{(\w+)\}', lambda m: os.environ.get(m.group(1), m.group(0)), _toml_text)
_tmp_toml = Path(tempfile.mkdtemp(prefix='statefix-')) / 'dataset.toml'
_tmp_toml.write_text(_toml_text)
a.toml = str(_tmp_toml)
sys.path.insert(0, str(Path(a.tree).resolve() / 'src'))
from cell_load.utils.modules import get_datamodule

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
kw = dict(toml_config_path=a.toml, embed_key='X_hvg', pert_col='perturbation', cell_type_key='cell_type',
          control_pert='DMSO_TF', batch_col='gem_group', output_space='all', num_workers=0)
dm = get_datamodule('PerturbationDataModule', kw, batch_size=a.batch_size, cell_sentence_len=a.cell_set_len)
dm.setup(stage='fit')
dl = dm.train_dataloader()

batches = []
for i, b in enumerate(dl):
    batches.append({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in b.items()})
    if len(batches) >= a.batches:
        break
print(f'drew {len(batches)} batches', flush=True)

def pack(b, golden=False):
    """Store big sparse tensors (the raw gene counts, ~2% dense) as COO, and optionally cast to float64."""
    out = {}
    for k, v in b.items():
        if torch.is_tensor(v) and v.is_floating_point():
            v = v.double() if golden else v
            if v.numel() > 1 << 22 and float((v != 0).float().mean()) < 0.25:
                out[k] = v.to_sparse_coo()
                continue
        out[k] = v
    return out


def to_golden(b):
    return pack(b, golden=True)

torch.save({'ref': [pack(b) for b in batches], 'golden': [to_golden(b) for b in batches],
            'meta': {'batch_size': a.batch_size, 'cell_set_len': a.cell_set_len, 'seed': a.seed,
                     'toml': a.toml, 'n': len(batches)}},
           Path(a.out))
print('wrote', a.out, flush=True)
