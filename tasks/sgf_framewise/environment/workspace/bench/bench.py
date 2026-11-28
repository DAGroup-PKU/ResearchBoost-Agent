"""Four-H200 SGF judge: record, compare, time, and verifier-only check --final."""

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import (EXIT_CODES, SYSTEM_PATH, Assets, Rejected, fingerprint, gpu_devices, gpu_lock, launch,
                     pinned_tree, read_json, require, snapshot, stop_worker, worker_env, worker_identity)
from scoring import digest, gate0, n1, timing, validate_ranks, window_times

CONTRACT = Path(__file__).with_name("bench.json")
WORKSPACE = CONTRACT.parent.parent
SCHEMA = 1

RESOLVE_CONFIG = """
import json, sys
from omegaconf import OmegaConf
config = OmegaConf.merge(OmegaConf.load('configs/default_config.yaml'), OmegaConf.load(sys.argv[1]))
print(json.dumps(OmegaConf.to_container(config, resolve=True), allow_nan=False))
"""

PROBE_RUNTIME = """
import os, sys, torch
from importlib.metadata import version
sys.path.insert(1, os.getcwd())
from trainer.distillation import Trainer
assert version('diffusers') == '0.31.0'
assert torch.cuda.is_available() and torch.cuda.device_count() == int(sys.argv[1])
if sys.argv[2] == 'reference':
    import flash_attn
    assert version('flash_attn').split('.')[0] == '2' and callable(flash_attn.flash_attn_varlen_func)
"""


def validate_switches(path, contract, final):
    require(path is not None or not final, "Final evaluation requires a switches manifest", "INCORRECT")
    if path is None:
        return
    try:
        manifest = read_json(path, 65536)
    except (OSError, ValueError):
        raise Rejected("INCORRECT", "Cannot read a valid switches manifest") from None
    require(isinstance(manifest, dict) and not manifest.keys() - {"environment"},
            "Expected a switches manifest with an optional environment object", "INCORRECT")
    environment = manifest.get("environment", {})
    require(isinstance(environment, dict) and all(isinstance(value, str) for value in environment.values()),
            "Switch environment values must be strings", "INCORRECT")
    allowed = set(contract["switches"]["allowed_candidate_environment"]) - set(contract["reserved_env"])
    require(not environment.keys() - allowed, "Switches must be explicitly allowed and not managed by the judge", "INCORRECT")
    require(not environment, "This judge does not accept candidate environment overrides", "INCORRECT")
    return fingerprint(manifest)


class Judge:
    def __init__(self, args, contract, directory):
        self.args, self.contract, self.directory = args, contract, directory
        self.policy = contract["correctness"]["trajectory"]
        self.id = fingerprint(contract)
        self.path = os.environ.get("PATH", SYSTEM_PATH) if args.worker_user else SYSTEM_PATH
        self.env = worker_env(directory, self.path)
        self.devices = gpu_devices(contract["world_size"], self.env)
        self.env["CUDA_VISIBLE_DEVICES"] = self.devices
        require((args.assets / "SHA256SUMS").is_file(), "Frozen asset SHA256SUMS is missing", "UNAVAILABLE")
        self.reference_python = args.reference_python.absolute()
        self.candidate_python = args.candidate_python.absolute()
        require(os.geteuid() != 0 or args.worker_user, "Root must supply a dedicated unprivileged --worker-user", "UNAVAILABLE")
        self.identity, self.account = worker_identity(args.worker_user)
        stop_worker(self.account)
        self.reference, self.candidate = directory / "reference", directory / "candidate"
        self.reference.mkdir()
        self.candidate.mkdir()
        reference_only = args.command == "record" and args.tree.resolve() == args.reference.resolve()
        try:
            pinned_tree(args.reference.resolve(), contract["reference_commit"], self.reference, self.env)
        except tarfile.TarError as exc:
            raise Rejected("UNAVAILABLE", "Cannot read pinned upstream reference archive") from exc
        self.ref_source = snapshot(self.reference)
        self.cand_source = self.ref_source if reference_only else snapshot(args.tree.resolve(), self.candidate)
        if args.command == "check" and contract["switches"]["implementation_change_required"]:
            require(self.cand_source[1] != self.ref_source[1], "No implementation change from the pinned reference", "INCORRECT")
        self.assets = Assets(args.assets)
        self.assets.overlay(self.reference)
        if not reference_only:
            self.assets.overlay(self.candidate)
        require(self.reference_python.is_file() and (reference_only or self.candidate_python.is_file()),
                "SGF training Python is missing", "UNAVAILABLE")
        self.preflight(self.reference_python, self.reference, "reference")
        if not reference_only:
            self.preflight(self.candidate_python, self.candidate, "candidate")
        baseline = contract["reference_baseline"]
        resolved = self.python(self.reference_python, self.reference, RESOLVE_CONFIG, [baseline["base_config"]])
        require(resolved.returncode == 0, "Cannot resolve pinned upstream configuration", "UNAVAILABLE")
        self.config = json.loads(resolved.stdout)
        self.config.update(baseline["config"])
        self.config.update(contract["gate0"]["exact_items"]["config"])
        self.config.update(no_save=True, no_visualize=True, disable_wandb=True, tf=False, step_time_log_interval=0,
                           config_name="sgf_framewise", logdir="", wandb_save_dir="")

    def python(self, python, tree, script, args):
        try:
            return subprocess.run([str(python), "-I", "-B", "-c", script, *args], cwd=tree, env=self.env,
                                  capture_output=True, text=True, timeout=120, **self.identity)
        finally:
            stop_worker(self.account)

    def preflight(self, python, tree, role):
        try:
            result = self.python(python, tree, PROBE_RUNTIME, [str(self.contract["world_size"]), role])
        except (OSError, subprocess.TimeoutExpired):
            raise Rejected("UNAVAILABLE", f"Cannot start {role} training runtime") from None
        if result.returncode:
            if self.args.out is not None:
                (self.args.out / f"{role}-preflight.log").write_text(result.stdout + result.stderr)
            raise Rejected("UNAVAILABLE", f"{role.capitalize()} CUDA/dependency preflight failed")

    def run(self, label, seed, reference, *, windows=()):
        start = time.monotonic()
        steps = max(high for _, high in windows) if windows else self.policy["completed_steps"]
        config = {**self.config, "seed": seed, "max_steps": steps}
        recipe = fingerprint(config)
        runtime = self.contract["gate0"]["exact_items"]["runtime"]
        spec = {"seed": seed, "steps": steps, "world_size": self.contract["world_size"], "windows": list(windows),
                "reference": reference, "endpoints": not windows, "runtime": runtime, "config": config}
        tree = self.reference if reference else self.candidate
        python = self.reference_python if reference else self.candidate_python
        source = self.ref_source if reference else self.cand_source
        require(snapshot(tree) == source, "Staged training source changed")
        self.assets.verify_unchanged()
        run_dir = self.directory / label
        try:
            ranks = launch(python, tree, run_dir, spec, self.devices, self.identity, self.account,
                           self.contract["run_timeout_sec"], self.contract["invalid_run_signatures"], self.path)
        finally:
            if self.args.out is not None and (run_dir / "train.log").is_file():
                logs = self.args.out / "training"
                logs.mkdir(exist_ok=True)
                shutil.copyfile(run_dir / "train.log", logs / f"{label}.log")
        require(snapshot(tree) == source, "Staged training source changed during evaluation")
        if not reference:
            require(snapshot(self.args.tree.resolve()) == source, "Submitted training source changed during evaluation")
        self.assets.verify_unchanged()
        validate_ranks(ranks, spec, self.contract)
        require(all(rank["config"] == recipe for rank in ranks), "Effective training config changed", "INCORRECT")
        if self.args.out is not None:
            with (self.args.out / "run_times.jsonl").open("a") as stream:
                stream.write(json.dumps({"label": label, "steps": steps, "wall_sec": time.monotonic() - start}) + "\n")
        return {"schema": SCHEMA, "contract": self.id, "seed": seed, "steps": steps, "windows": list(windows),
                "reference": reference, "endpoints": not windows, "recipe": recipe, "assets": self.assets.digest,
                "source": source[0], "ranks": ranks}

    def paired_time(self, final):
        windows = self.contract["timing"]["windows"]
        seed = self.policy["reference_seeds"][0]
        if final:
            offset = secrets.choice(self.contract["timing"]["heldout_window_offsets"])
            windows = [[low + offset, high + offset] for low, high in windows]
            seed = heldout_seeds(self.policy, 1)[0]
        roles = {"reference_a": True, "candidate": False, "reference_b": True}
        runs = {label: self.run(label, seed, roles[label], windows=windows)
                for label in self.contract["timing"]["final_order"]}
        return timing(runs["reference_a"], runs["candidate"], runs["reference_b"], self.contract)


def heldout_seeds(policy, count):
    seeds = set()
    while len(seeds) < count:
        seed = secrets.randbelow(2**31 - 1) + 1
        if seed not in policy["reference_seeds"]:
            seeds.add(seed)
    return sorted(seeds)


def validate_bundle(bundle, reference, contract):
    policy = contract["correctness"]["trajectory"]
    require(bundle["schema"] == SCHEMA and bundle["contract"] == fingerprint(contract), "Incompatible trajectory record")
    runs = bundle["runs"]
    require([run["seed"] for run in runs] == policy["reference_seeds"], "Incomplete trajectory record seeds")
    for run in runs:
        require(run["schema"] == SCHEMA and run["contract"] == bundle["contract"]
                and run["steps"] == policy["completed_steps"] and run["windows"] == [] and run["endpoints"] is True
                and run["reference"] is reference and all(digest(run[key]) for key in ("source", "assets", "recipe")),
                "Invalid trajectory run record")
        validate_ranks(run["ranks"], run, contract)
        require(all(rank["config"] == run["recipe"] for rank in run["ranks"]), "Recorded config differs from recipe")
    require(len({run["assets"] for run in runs}) == len({run["source"] for run in runs}) == 1,
            "Record changed source or assets between seeds")
    return runs


def check(judge, result, args):
    contract, policy = judge.contract, judge.policy
    result["implementation_changed"] = judge.cand_source[1] != judge.ref_source[1]
    heldout = args.final or args.n1
    if heldout:
        seeds = heldout_seeds(policy, policy["heldout_seed_count"])
    else:
        seeds = policy["reference_seeds"][:1]
    pairs = []
    for index, seed in enumerate(seeds):
        ref = judge.run(f"pair-{index}-reference", seed, True)
        cand = judge.run(f"pair-{index}-candidate", seed, False)
        gate0(ref, cand)
        pairs.append((ref, cand))
    gates = result["correctness"]["gates_run"]
    gates.append(0)
    if args.quick:
        return {"status": "OK", "reason": "Gate 0 diagnostic only"}
    if not (args.skip_timing or args.n1):
        result["performance"] = judge.paired_time(args.final)
        gates.append(1)
        perf = result["performance"]
        benefit = contract["benefit"]
        faster = perf["speedup"] > perf["threshold"] if benefit["require_strictly_above_threshold"] else perf["speedup"] >= perf["threshold"]
        require(faster, "Candidate did not clear the live benefit threshold", benefit["below_threshold_status"])
    calibration = []
    for index, seed in enumerate(policy["reference_seeds"]):
        calibration.append(pairs[0][0] if not heldout and seed == seeds[0]
                           else judge.run(f"calibration-{index}", seed, True))
    metrics = n1(calibration, pairs, contract)
    result["correctness"].update(metrics, verdict="equivalent")
    gates.append(3)
    if args.final:
        require(gates == contract["correctness"]["final_required_gates"]
                and metrics["tier"] in contract["correctness"]["accepted_tiers"], "Final gates were not completed")
    return {"status": "PASS" if args.final else "OK"}


def execute(args, contract, result):
    if args.command == "compare":
        refs = validate_bundle(read_json(args.ref), True, contract)
        candidates = validate_bundle(read_json(args.cand), False, contract)
        return {"status": "OK", "correctness": {"verdict": "equivalent", "gates_run": [0, 3],
                **n1(refs, list(zip(refs, candidates, strict=True)), contract)}}
    manifest_digest = validate_switches(args.config, contract, getattr(args, "final", False))
    with gpu_lock(Path(contract["gpu_lock"])), tempfile.TemporaryDirectory(prefix="sgf-judge-", dir="/tmp") as scratch:
        directory = Path(scratch)
        directory.chmod(0o711 if args.worker_user else 0o700)
        judge = Judge(args, contract, directory)
        if args.command == "record":
            reference = args.tree.resolve() == args.reference.resolve()
            runs = [judge.run(f"record-{index}", seed, reference)
                    for index, seed in enumerate(judge.policy["reference_seeds"])]
            bundle = {"schema": SCHEMA, "contract": judge.id, "runs": runs}
            validate_bundle(bundle, reference, contract)
            target = args.out / f"{args.label}.json"
            target.write_text(json.dumps(bundle, allow_nan=False) + "\n")
            outcome = {"status": "OK", "record": str(target)}
        elif args.command == "time":
            if args.paired:
                perf = judge.paired_time(False)
            else:
                windows = contract["timing"]["windows"]
                run = judge.run("candidate", judge.policy["reference_seeds"][0], False, windows=windows)
                perf = {"windows": windows, contract["timing"]["field"]: window_times(run)}
            outcome = {"status": "OK", "performance": perf}
        else:
            outcome = check(judge, result, args)
        require(manifest_digest is None or manifest_digest == fingerprint(read_json(args.config, 65536)),
                "Switches manifest changed during evaluation")
        return outcome


def main(argv=None):
    contract = read_json(CONTRACT)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("record", "time", "check"):
        command = commands.add_parser(name)
        command.add_argument("--tree", required=True, type=Path)
        command.add_argument("--out", required=name != "check", type=Path)
        command.add_argument("--config", type=Path, help="switches.json manifest, not a training YAML")
        command.add_argument("--reference", type=Path, default=WORKSPACE / contract["reference"])
        command.add_argument("--assets", type=Path, default=Path(contract["assets"]))
        command.add_argument("--reference-python", type=Path, default=Path(contract["reference_python"]))
        command.add_argument("--candidate-python", type=Path, default=Path(contract["candidate_python"]))
        command.add_argument("--worker-user", help="dedicated non-root worker account (required for root launches)")
        if name == "record":
            command.add_argument("--label", default="ref")
        elif name == "time":
            command.add_argument("--paired", action="store_true")
        else:
            modes = command.add_mutually_exclusive_group()
            modes.add_argument("--quick", action="store_true")
            modes.add_argument("--skip-timing", action="store_true")
            modes.add_argument("--n1", action="store_true", help="run both held-out N1 trajectories without timing")
            modes.add_argument("--final", action="store_true", help=argparse.SUPPRESS)
    compare = commands.add_parser("compare")
    compare.add_argument("ref", type=Path)
    compare.add_argument("cand", type=Path)

    args = parser.parse_args(argv)
    if getattr(args, "final", False) and os.geteuid() != 0:
        parser.error("--final requires the trusted verifier/controller")
    if args.command == "record" and (not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label) or args.label == "record"):
        parser.error("invalid record label")
    result = {"command": args.command}
    if args.command == "check":
        result.update(final=args.final, correctness={"verdict": "not_evaluated", "gates_run": []}, performance={})
    out = getattr(args, "out", None)
    try:
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            (out / "run_times.jsonl").unlink(missing_ok=True)
        result.update(execute(args, contract, result))
    except Rejected as exc:
        result.update(status=exc.status, reason=str(exc))
    except (OSError, ValueError, KeyError, IndexError, TypeError, ArithmeticError, subprocess.SubprocessError) as exc:
        result.update(status="INVALID", reason=f"Judge input or run failed: {exc}")
    payload = json.dumps(result, allow_nan=False) + "\n"
    if out is not None:
        (out / f"{args.command}.json").write_text(payload)
    print(payload, end="")
    return EXIT_CODES[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
