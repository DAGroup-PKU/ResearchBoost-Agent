"""Validate tasks with the installed Harbor models (python scripts/validate_config.py tasks/llava [...])."""
import argparse
from pathlib import Path
import toml
from harbor.models.task.config import TaskConfig

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("tasks", type=Path, nargs="+")
args = parser.parse_args()
# GPUs per trial, as each task.toml states them (Hopper, at least 130 GiB per GPU).
HARDWARE = {"llava": (2, ["H200"]), "yoco": (2, ["H200"]), "nanochat": (2, ["H200"]), "bagel_sft": (4, ["H200"]),
            "r3": (4, ["H200"]), "gaussian_splatting": (1, ["H200"]), "slime_grpo": (4, ["H200"]), "slime_opd": (4, ["H200"])}
WORKDIR = {"gaussian_splatting": "/submission/gaussian-splatting"}
for task in args.tasks:
    config = TaskConfig.model_validate(toml.loads((task / "task.toml").read_text()))
    assert config.task.name == "researchboost/" + task.name, config.task.name
    # one container: the judge runs in the agent's environment after the agent finishes
    assert config.verifier.environment_mode == "shared"
    assert config.environment.docker_image is None, "images are built on site from environment/Dockerfile"
    gpus, types = HARDWARE[task.name]
    assert config.environment.gpus == gpus and config.environment.gpu_types == types
    # the upstream checkout under /submission (its directory name where upstream uses another, as for 3DGS)
    assert config.environment.workdir == WORKDIR.get(task.name, "/submission/" + task.name)
    # the submission, its switches and the agent's installed packages (a re-judge puts all three back)
    assert config.artifacts == [config.environment.workdir, "/submission/switches.json", "/opt/candidate-env"]
    assert config.agent.user == "agent"
    assert config.agent.network_mode == "public"
    assert config.verifier.network_mode == "no-network"
    print("%s: Harbor configuration validated" % task)
