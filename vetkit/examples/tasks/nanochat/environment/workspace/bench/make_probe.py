"""Formulator: the frozen probe batch of the statistical gate, from the reference tree (run from its root, CPU only).

    python make_probe.py OUT.pt [--world 8] [--rows 4]

For every rank r of the world, the first `rows` rows that the reference's validation loader
(nanochat.dataloader.tokenizing_distributed_data_loader_bos_bestfit, split "val": the last ClimbMix shard of
NANOCHAT_BASE_DIR) yields on rank r -> {'inputs': [world, rows, T], 'targets': [world, rows, T]} (int64)."""
import argparse
import os
import sys

import torch

p = argparse.ArgumentParser()
p.add_argument('out')
p.add_argument('--world', type=int, default=8)
p.add_argument('--rows', type=int, default=4)
p.add_argument('--seq', type=int, default=2048)
a = p.parse_args()
sys.path.insert(0, os.getcwd())
from nanochat.tokenizer import get_tokenizer  # noqa: E402
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit  # noqa: E402

tok = get_tokenizer()
inputs, targets = [], []
for rank in range(a.world):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(a.world))
    x, y = next(tokenizing_distributed_data_loader_bos_bestfit(tok, a.rows, a.seq, split='val', device='cpu'))
    inputs.append(x.clone())
    targets.append(y.clone())
torch.save({'inputs': torch.stack(inputs), 'targets': torch.stack(targets)}, a.out)
print(f'probe batch: {a.world} ranks x {a.rows} rows x {a.seq} tokens -> {a.out}')
