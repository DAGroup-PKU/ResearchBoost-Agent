"""Harbor shared verifier; only a complete trusted final check can score."""

import json
import math
import os
import pwd
import signal
import subprocess
import time
from pathlib import Path

SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
WORKSPACE = Path("/opt/researchboost/ws")
PYTHON = "/usr/bin/python3"
BENCH = WORKSPACE / "bench/bench.py"
READY = WORKSPACE / "task/ready.json"
LOGS = Path("/logs/verifier")
WORKER = "judge-worker"


def root_only(path):
    pending = Path(path)
    if not pending.is_absolute():
        return False
    try:
        current = Path(pending.anchor)
        components = list(reversed(pending.parts[1:]))
        links = 0
        while components:
            part = components.pop()
            if part == "..":
                current = current.parent
                continue
            current /= part
            info = current.lstat()
            if current.is_symlink():
                links += 1
                if links > 40:
                    return False
                target = current.readlink()
                if target.is_absolute():
                    current = Path(target.anchor)
                    components.extend(reversed(target.parts[1:]))
                else:
                    current = current.parent
                    components.extend(reversed(target.parts))
            elif info.st_uid != 0 or info.st_mode & 0o022:
                return False
        return True
    except (OSError, ValueError):
        return False


def safe_path():
    entries = (os.environ.get("PATH", "") + os.pathsep + SYSTEM_PATH).split(os.pathsep)
    return os.pathsep.join(dict.fromkeys(entry for entry in entries if root_only(entry) and Path(entry).is_dir()))


def live_agent(pid, uid):
    try:
        fields = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines() if ":" in line)
        return int(fields["Uid"].split()[0]) == uid and fields["State"].split()[0] not in {"Z", "X"}
    except (OSError, KeyError, ValueError, IndexError):
        return False


def stop_agent(uid):
    deadline = time.monotonic() + 30
    while True:
        active = [int(path.name) for path in Path("/proc").iterdir()
                  if path.name.isdecimal() and live_agent(path.name, uid)]
        if not active:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("Unprivileged processes survived cleanup")
        for pid in active:
            try:
                handle = os.pidfd_open(pid)
            except ProcessLookupError:
                continue
            try:
                if live_agent(pid, uid):
                    signal.pidfd_send_signal(handle, signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                os.close(handle)
        time.sleep(0.1)


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    if not root_only(LOGS):
        raise RuntimeError("Verifier log directory is not root-only")
    LOGS.chmod(0o700)
    for name in ("reward.json", "reward.txt", "harness.log", "check.json"):
        (LOGS / name).unlink(missing_ok=True)
    (LOGS / "reward.json").write_text(json.dumps({"accepted": 0.0, "speedup": 0.0}) + "\n")
    contract_path = BENCH.with_name("bench.json")
    protected = (PYTHON, READY, BENCH, contract_path, *(BENCH.with_name(name) for name in (
        "runtime.py", "scoring.py", "sgf_entry.py")), Path("/opt/reference-env/bin/python"))
    if not all(root_only(path) for path in protected):
        raise RuntimeError("Missing or non-root-only verifier prerequisites")
    contract = json.loads(contract_path.read_text())
    ready = json.loads(READY.read_text())
    if ready.get("mode") != "gpu" or ready.get("reference_commit") != contract["reference_commit"]:
        raise RuntimeError("Unexpected GPU judge preparation state")
    agent = pwd.getpwnam("agent")
    worker = pwd.getpwnam(WORKER)
    if os.geteuid() != 0 or agent.pw_uid == 0 or worker.pw_uid in (0, agent.pw_uid):
        raise RuntimeError("Verifier, agent, and judge worker must have separate identities")
    stop_agent(agent.pw_uid)
    env = {"PATH": safe_path(), "HOME": "/root", "GIT_CONFIG_GLOBAL": os.devnull}
    command = [PYTHON, "-I", "-B", str(BENCH), "check", "--tree", "/submission/sgf_framewise",
               "--config", "/submission/switches.json", "--out", str(LOGS), "--worker-user", WORKER, "--final"]
    try:
        with (LOGS / "harness.log").open("w") as log:
            completed = subprocess.run(command, cwd=WORKSPACE, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=10680, check=False)
    finally:
        stop_agent(worker.pw_uid)
    expected = {0: "PASS", 1: "INCORRECT", 2: "UNAVAILABLE", 3: "INVALID"}.get(completed.returncode)
    if expected is None:
        raise RuntimeError(f"Unexpected final check exit code: {completed.returncode}")
    payload = (LOGS / "harness.log").read_text()
    result = json.loads(payload)
    if result.get("status") != expected or result.get("command") != "check" or result.get("final") is not True:
        raise RuntimeError("Unexpected final check result")
    (LOGS / "check.json").write_text(payload)
    if expected != "PASS":
        return
    correctness = result["correctness"]
    performance = result["performance"]
    speedup, threshold = performance["speedup"], performance["threshold"]
    if (correctness.get("gates_run") != contract["correctness"]["final_required_gates"]
            or correctness.get("tier") not in contract["correctness"]["accepted_tiers"]
            or correctness.get("verdict") != "equivalent"
            or result.get("implementation_changed") is not True
            or not math.isfinite(speedup) or not math.isfinite(threshold) or not speedup > threshold):
        raise RuntimeError("PASS is missing required gate evidence")
    temporary = LOGS / ".reward.final"
    temporary.unlink(missing_ok=True)
    temporary.write_text(json.dumps({"accepted": 1.0, "speedup": speedup}, allow_nan=False) + "\n")
    temporary.replace(LOGS / "reward.json")


if __name__ == "__main__":
    main()
