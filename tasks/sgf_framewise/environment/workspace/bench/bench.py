"""Standard bench interface for the structural draft; no scoring backend."""

import argparse
import json
import os
import stat
from pathlib import Path

CONTRACT = Path(__file__).with_name("bench.json")
FINAL_MARKER = Path("/run/researchboost/final-evaluation")


def final_allowed():
    if os.geteuid() == 0:
        return True
    try:
        marker = FINAL_MARKER.lstat()
        directory = FINAL_MARKER.parent.lstat()
    except OSError:
        return False
    return (stat.S_ISREG(marker.st_mode) and stat.S_ISDIR(directory.st_mode)
            and all(info.st_uid == 0 and not info.st_mode & 0o022 for info in (marker, directory)))


def unique_object(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("Duplicate switches manifest key")
    return value


def validate_switches(path, contract):
    if path is None:
        return
    try:
        manifest = json.loads(path.read_text(), object_pairs_hook=unique_object)
    except (OSError, ValueError):
        raise ValueError("Cannot read a valid switches manifest") from None
    if not isinstance(manifest, dict) or manifest.keys() - {"environment"}:
        raise ValueError("Expected a switches manifest with an optional environment object")
    environment = manifest.get("environment", {})
    if not isinstance(environment, dict) or not all(isinstance(value, str) for value in environment.values()):
        raise ValueError("Switch environment values must be strings")
    allowed = set(contract["switches"]["allowed_candidate_environment"]) - set(contract["reserved_env"])
    if environment.keys() - allowed:
        raise ValueError("Switches must be explicitly allowed and not managed by the judge")


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
            modes = command.add_mutually_exclusive_group()
            modes.add_argument("--quick", action="store_true")
            modes.add_argument("--skip-timing", action="store_true")
            modes.add_argument("--n1", action="store_true")
            modes.add_argument("--final", action="store_true", help=argparse.SUPPRESS)
    compare = commands.add_parser("compare")
    compare.add_argument("ref", type=Path)
    compare.add_argument("cand", type=Path)

    args = parser.parse_args()
    if getattr(args, "final", False) and not final_allowed():
        parser.error("--final requires the verifier marker")
    result = {"status": "UNAVAILABLE", "command": args.command,
              "reason": "GPU evaluation is not enabled for this structural draft"}
    if args.command == "check":
        result.update(final=args.final, correctness={"verdict": "not_evaluated", "gates_run": []}, performance={})
    try:
        validate_switches(getattr(args, "config", None), json.loads(CONTRACT.read_text()))
    except ValueError as exc:
        result.update(status="INCORRECT", reason=str(exc))
        if args.command == "check":
            result["correctness"].update(verdict="invalid_switches", stage="gate0")
    payload = json.dumps(result) + "\n"
    if getattr(args, "out", None) is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{args.command}.json").write_text(payload)
    print(payload, end="")
    return 2 if result["status"] == "UNAVAILABLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
