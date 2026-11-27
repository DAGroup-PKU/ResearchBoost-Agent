"""Describe the upstream single-node launcher; do not execute it."""

import json
from pathlib import Path

BENCH = Path(__file__).with_name("bench.json")
LAUNCHER = "scripts/train_self_gradient_forcing_framewise.sh"


def launch_plan(tree, config, logdir):
    contract = json.loads(BENCH.read_text())
    baseline = contract["reference_baseline"]
    exact = contract["gate0"]["exact_items"]
    return {
        "cwd": str(tree.resolve()),
        "reference_base_config": baseline["base_config"],
        "frozen_config_overrides": {**exact["config"], **baseline["config"]},
        "reference_baseline": baseline,
        "update_boundary": contract["gate0"]["update_boundary"],
        "env": {"NUM_GPUS": str(contract["world_size"]), "NNODES": str(exact["runtime"]["nnodes"]), "NODE_RANK": "0",
                "MASTER_ADDR": "127.0.0.1", "ENABLE_WANDB": "0"},
        "argv": ["/bin/bash", LAUNCHER, str(config.resolve()), str(logdir.resolve()),
                 "--no_save", "--no_visualize"],
    }
