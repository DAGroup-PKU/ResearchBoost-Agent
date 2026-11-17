"""The frozen workload: upstream YOCO/scripts/train.sh (microsoft/unilm @ f4f7b27a), with these deviations: the data
path (the task's corpus under $ASSETS/data/Data), --arch yoco_3b (the script's yoco_base is not registered), 2
processes on one node (upstream's per-GPU load: 2048 tokens per GPU, batch 1), --seed 1 written out, --save-dir in a
scratch directory, --max-update to end a run, --zero-sharding os (without it the reference does not fit in memory on
80 GB GPUs; kept for the same optimizer layout).

Entry: `<tree>/yoco/train.py`, run as a script from `<tree>/yoco` (upstream: `cd YOCO/yoco && torchrun ... train.py`),
so `models`, `tasks` and `criterions` are imported from the tree and sys.path[0] is `<tree>/yoco`.
"""
import os
from pathlib import Path

ENTRY_DIR = 'yoco'                 # relative to the tree root: the working directory of a run
ENTRY = 'yoco/train.py'            # relative to the tree root
DATA = os.path.join(os.environ.get('ASSETS', '/opt/researchboost/assets'), 'data', 'Data')
WORLD_SIZE = 2
WARMUP_UPDATES = 50

# upstream argument list (train.sh order), then the recorded additions; {save_dir} and {max_update} are filled in
UPSTREAM = [
    '--save-interval-updates', '5000',
    '--no-epoch-checkpoints',
    '--arch', 'yoco_3b',
    '--criterion', 'cross_entropy',
    '--task', 'gpt',
    '--tokens-per-sample', '2048',
    '--tokenizer-pad-to-multiple', '8',
    '--pad-to-max-len',
    '--optimizer', 'adam', '--adam-betas', '(0.9, 0.95)',
    '--adam-eps', '1e-06',
    '--clip-norm', '2.0',
    '--lr', '0.00015',
    '--lr-scheduler', 'polynomial_decay',
    '--warmup-updates', str(WARMUP_UPDATES),
    '--weight-decay', '0.05',
    '--batch-size', '1',
    '--model-parallel-size', '1',
    '--update-freq', '1',
    '--batch-read-ahead', '1000',
    '--total-num-update', '300000',
    '--log-format', 'simple', '--log-interval', '10', '--disable-validation',
    '--tiktoken-model', 'cl100k_base',
    '--no-save',
    '--bf16',
    '--zero-sharding', 'os',
    '--seed', '1',
    '--save-dir', '{save_dir}',
    '--max-update', '{max_update}',
]


def argv(save_dir, max_update):
    """train.py arguments (after the script name) for one run. Timing runs, numerics cases and trajectories all use
    this list unchanged; only the scratch save directory and --max-update differ between runs (--max-update only
    ends a run: the LR schedule uses --total-num-update 300000 and --warmup-updates 50)."""
    fill = {'save_dir': str(Path(save_dir)), 'max_update': str(int(max_update))}
    return [DATA, *[a.format(**fill) for a in UPSTREAM]]
