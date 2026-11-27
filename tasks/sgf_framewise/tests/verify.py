"""Harbor shared verifier for the no-score structural draft."""

import json
import os
import pwd
import signal
import subprocess
import time
from pathlib import Path

SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
WORKSPACE = Path("/opt/researchboost/ws")
PYTHON = "/usr/bin/python3"
RUNUSER = "/usr/sbin/runuser"
BENCH = WORKSPACE / "bench/bench.py"
READY = WORKSPACE / "task/ready.json"
LOGS = Path("/logs/verifier")


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
            raise RuntimeError("Agent processes survived cleanup")
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
    for name in ("reward.json", "reward.txt", "harness.log"):
        (LOGS / name).unlink(missing_ok=True)
    (LOGS / "reward.json").write_text(json.dumps({"accepted": 0.0, "speedup": 0.0}) + "\n")
    if not all(root_only(path) for path in (PYTHON, RUNUSER, BENCH, READY)):
        raise RuntimeError("Missing or non-root-only verifier prerequisites")
    if json.loads(READY.read_text()).get("mode") != "structure":
        raise RuntimeError("Unexpected preparation mode for structural verifier")
    agent = pwd.getpwnam("agent")
    if agent.pw_uid == 0:
        raise RuntimeError("The agent must not be root")
    stop_agent(agent.pw_uid)
    env = {"PATH": safe_path(), "HOME": agent.pw_dir, "USER": agent.pw_name,
           "LOGNAME": agent.pw_name, "GIT_CONFIG_GLOBAL": os.devnull}
    command = [RUNUSER, "-u", agent.pw_name, "--", PYTHON, "-I", str(BENCH), "check",
               "--tree", "/submission/sgf_framewise", "--config", "/submission/switches.json", "--final"]
    with (LOGS / "harness.log").open("w") as log:
        completed = subprocess.run(command, cwd=WORKSPACE, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=30, check=False)
    if completed.returncode != 2:
        raise RuntimeError(f"Unexpected structural check exit code: {completed.returncode}")


if __name__ == "__main__":
    main()
