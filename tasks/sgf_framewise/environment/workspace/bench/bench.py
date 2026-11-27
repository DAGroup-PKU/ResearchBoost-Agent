"""Standard bench interface for the structural draft; no scoring backend."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("record", "time", "check"):
        command = commands.add_parser(name)
        command.add_argument("--tree", required=True, type=Path)
        command.add_argument("--out", required=name != "check", type=Path)
        command.add_argument("--config", type=Path)
        if name == "record":
            command.add_argument("--label", default="ref")
        elif name == "time":
            command.add_argument("--paired", action="store_true")
        else:
            command.add_argument("--quick", action="store_true")
            command.add_argument("--skip-timing", action="store_true")
            command.add_argument("--final", action="store_true")
    compare = commands.add_parser("compare")
    compare.add_argument("ref", type=Path)
    compare.add_argument("cand", type=Path)

    args = parser.parse_args()
    if getattr(args, "final", False) and (args.quick or args.skip_timing):
        parser.error("--final requires a full check")
    result = {"status": "UNAVAILABLE", "command": args.command,
              "reason": "GPU evaluation is not enabled for this structural draft"}
    if args.command == "check":
        result.update(final=args.final, correctness={"verdict": "not_evaluated"}, performance={})
    payload = json.dumps(result) + "\n"
    if getattr(args, "out", None) is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{args.command}.json").write_text(payload)
    print(payload, end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
