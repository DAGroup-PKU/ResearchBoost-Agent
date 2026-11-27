"""Describe the upstream single-node launcher; do not execute it."""

import json
from pathlib import Path

BENCH = Path(__file__).with_name("bench.json")
LAUNCHER = "scripts/train_self_gradient_forcing_framewise.sh"


def launch_plan(tree, config, logdir):
    world_size = json.loads(BENCH.read_text())["world_size"]
    return {
        "cwd": str(tree.resolve()),
        "env": {"NUM_GPUS": str(world_size), "NNODES": "1", "NODE_RANK": "0",
                "MASTER_ADDR": "127.0.0.1", "ENABLE_WANDB": "0"},
        "argv": ["/bin/bash", LAUNCHER, str(config.resolve()), str(logdir.resolve()),
                 "--no_save", "--no_visualize"],
    }
