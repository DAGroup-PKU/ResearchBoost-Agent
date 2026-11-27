"""Prepare the structural workspace without starting training."""

import json
import subprocess
from pathlib import Path

WORKSPACE = Path("/opt/researchboost/ws")
CANDIDATE = Path("/submission/sgf_framewise")
READY = Path("/run/researchboost-ready")


def main():
    READY.unlink(missing_ok=True)
    config = json.loads((WORKSPACE / "bench/bench.json").read_text())
    reference = WORKSPACE / config["reference"]
    commit = subprocess.check_output(
        ["/usr/bin/git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != config["reference_commit"]:
        raise SystemExit("Unexpected reference revision")
    for tree in (reference, CANDIDATE):
        if not (tree / "scripts/train_self_gradient_forcing_framewise.sh").is_file():
            raise SystemExit(f"Missing SGF launcher in {tree}")
    metadata = WORKSPACE / "task/ready.json"
    metadata.parent.mkdir(exist_ok=True)
    metadata.write_text(json.dumps({"status": "ready", "mode": "structure", "reference_commit": commit}) + "\n")
    READY.touch()


if __name__ == "__main__":
    main()
