"""Print a proposed SGF launch plan without allocating GPUs."""

import argparse
import json
from pathlib import Path

from recipe import launch_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tree", "config", "logdir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(launch_plan(args.tree, args.config, args.logdir), indent=2))


if __name__ == "__main__":
    main()
