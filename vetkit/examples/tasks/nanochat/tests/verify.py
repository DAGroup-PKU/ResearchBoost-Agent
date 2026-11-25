"""Run the nanochat judge; translate its verdict without changing gates. A full-check PASS (tier N0 or N1) is accepted."""
import json
import math
import os
from pathlib import Path
import pwd
import shutil
import signal
import stat
import subprocess
import tempfile
import time

AGENT = "agent"
REJECTED = {"accepted": 0.0, "speedup": 0.0}


def candidate_error(result):
    """ERROR verdicts the candidate caused, scored as a rejection instead of being left unscored: a tree without the
    entry, a candidate timing run that failed after the reference run succeeded, or a candidate record or trajectory
    run that failed (the reference recorded on this host during preparation)."""
    correctness = result.get("correctness") or {}
    runs = (result.get("performance") or {}).get("runs") or {}
    if correctness.get("stage") == "structure":
        return True
    if correctness.get("stage") == "record" and (correctness.get("numerics") or {}).get("status") == "ERROR":
        return True
    if correctness.get("stage") == "statistical" and (correctness.get("trajectory") or {}).get("status") == "ERROR":
        return True
    return (correctness.get("stage") == "benefit" and (runs.get("ref_a") or {}).get("status") == "ok"
            and (runs.get("cand") or {}).get("status") == "ERROR")


FINAL_MARKER = Path("/run/researchboost/final-evaluation")


def allow_final():
    """The judge refuses `check --final` to the agent user unless this root-owned marker exists in a root-owned
    directory the agent cannot write (bench.py final_allowed); written here, after the agent's processes are stopped."""
    FINAL_MARKER.parent.mkdir(parents=True, exist_ok=True)
    os.chown(FINAL_MARKER.parent, 0, 0)
    FINAL_MARKER.parent.chmod(0o755)
    FINAL_MARKER.write_text("%d\n" % int(time.time()))
    os.chown(FINAL_MARKER, 0, 0)
    FINAL_MARKER.chmod(0o644)


def reward(result, returncode):
    if (result.get("quick") is not False or result.get("skip_timing") is not False or result.get("n1") is not False
            or result.get("final") is not True):
        raise ValueError("Only a full check --final can receive a reward")
    status = result.get("status")
    correctness = result.get("correctness") or {}
    performance = result.get("performance") or {}
    if status == "PASS" and returncode == 0:
        speedup = performance.get("speedup")
        if (correctness.get("passed") is not True
                or (correctness.get("verdict"), correctness.get("measured_tier")) not in (("bitwise", "N0"),
                                                                                       ("equivalent", "N1"))
                or performance.get("benefit", {}).get("passed") is not True
                or type(speedup) not in (int, float)
                or not math.isfinite(speedup) or speedup <= 0):
            raise ValueError("Incomplete full-check PASS; no reward written")
        return {"accepted": 1.0, "speedup": speedup}
    if returncode == 1 and (status == "INCORRECT" or (
            status == "UNAVAILABLE" and correctness.get("verdict") == "no_benefit") or (
            status == "ERROR" and candidate_error(result))):
        return dict(REJECTED)
    raise ValueError("Evaluation unavailable or inconsistent: %r (exit %s)" % (status, returncode))


def smoke_reward(result, returncode):
    """Smoke mode: the quick check only (exact items and the bitwise comparison). Not a benchmark score: no timing,
    no trajectories."""
    status = result.get("status")
    if status == "PASS" and returncode == 0:
        bitwise = ((result.get("correctness") or {}).get("numerics") or {}).get("bitwise")
        if type(bitwise) is not bool:
            raise ValueError("Smoke quick-check PASS without a bitwise flag")
        return {"smoke_quick_pass": 1.0, "smoke_bitwise": float(bitwise)}
    if returncode == 1 and status == "NOT_BITWISE":
        return {"smoke_quick_pass": 1.0, "smoke_bitwise": 0.0}
    if returncode == 1 and (status == "INCORRECT" or (status == "ERROR" and candidate_error(result))):
        return {"smoke_quick_pass": 0.0, "smoke_bitwise": 0.0}
    raise ValueError("Smoke quick check unavailable or inconsistent: %r (exit %s)" % (status, returncode))


def switches_problem(path, reserved):
    """The judge's candidate_environment rules: a bad switches file is the candidate's failure, but the judge exits
    before writing a verdict."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        return "unreadable: %s" % error
    if not isinstance(data, dict) or set(data) - {"environment"}:
        return 'only an "environment" mapping is accepted'
    env = data.get("environment") or {}
    if not isinstance(env, dict):
        return '"environment" must be a mapping'
    bad = sorted(str(k) for k in env if str(k) in reserved or str(k).startswith("NANOCHAT_BENCH"))
    return "managed by the bench: %s" % bad if bad else None


def root_only(path):
    """True if `path` exists, and it and every directory on the way to it belong to root and are not writable by group
    or others. Symbolic links are followed one component at a time: a link counts through the directory that holds it,
    then through what it points to (a link in a directory the agent can write is refused wherever it points)."""
    path = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)    # not normalized: ".." is walked
    todo, cur, links = path.split("/")[::-1], "/", 0
    try:
        if os.stat("/").st_uid != 0 or os.stat("/").st_mode & 0o022:
            return False
        while todo:
            part = todo.pop()
            if part in ("", "."):
                continue
            if part == "..":
                cur = os.path.dirname(cur)
                continue
            nxt = os.path.join(cur, part)
            st = os.lstat(nxt)
            if stat.S_ISLNK(st.st_mode):
                links += 1
                if links > 40:
                    return False
                target = os.readlink(nxt)
                cur = "/" if target.startswith("/") else cur
                todo.extend(target.split("/")[::-1])
                continue
            if st.st_uid != 0 or st.st_mode & 0o022:
                return False
            cur = nxt
        return True
    except OSError:
        return False


def safe_path(path):
    """The entries of a PATH that only root can change. The image puts the agent's /opt/candidate-env/bin first; no
    program from there may run as root here, nor inside the judge (which runs as the agent)."""
    keep = []
    for entry in path.split(os.pathsep):
        if entry and os.path.isabs(entry) and entry not in keep and os.path.isdir(entry) and root_only(entry):
            keep.append(entry)
    return os.pathsep.join(keep)


SAFE_PATH = safe_path(os.pathsep.join((os.environ.get("PATH", ""),
                                       "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")))
JUDGE_PYTHON = "/opt/nanochat-env/bin/python"


def tool(name):
    """The absolute path of a program on SAFE_PATH, itself root-only."""
    found = shutil.which(name, path=SAFE_PATH)
    if not found or not root_only(found):
        raise RuntimeError("%s: no root-only program on %s" % (name, SAFE_PATH))
    return found


def agent_env(**extra):
    """The environment of what the verifier runs as the agent: SAFE_PATH, and no global git configuration from the
    agent's HOME (git >= 2.32). The judge puts the candidate environment back on the PATH of the candidate's runs."""
    return dict(os.environ, PATH=SAFE_PATH, GIT_CONFIG_GLOBAL=os.devnull, **extra)


def proc_status(pid):
    """(real uid, state, start time) of a process from /proc, or None once it is gone."""
    try:
        with open("/proc/%d/status" % pid) as stream:
            fields = dict(line.split(":", 1) for line in stream.read().splitlines() if ":" in line)
        with open("/proc/%d/stat" % pid) as stream:
            data = stream.read()
        start = data[data.rindex(")") + 2:].split()[19]              # field 22, after "pid (comm) "
        return int(fields["Uid"].split()[0]), fields["State"].split()[0], start
    except (OSError, ValueError, KeyError, IndexError):
        return None


def proc_cmdline(pid):
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as stream:
            return stream.read(300).replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return "?"


def agent_processes(uid):
    """{pid: start time} of every live process whose real uid is the agent's. Zombies are left out: the container's
    PID 1 may not reap orphans (Harbor's `sleep infinity`), and a zombie no longer runs."""
    procs = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            found = proc_status(int(name))
            if found and found[0] == uid and found[1] not in ("Z", "X"):
                procs[int(name)] = found[2]
    return procs


def kill_process(pid, uid, start):
    """SIGKILL `pid` if it is still the live process of `uid` that started at `start` (no pid reuse)."""
    handle = None
    try:
        handle = os.pidfd_open(pid)
    except (AttributeError, OSError):
        pass
    try:
        again = proc_status(pid)
        if not again or again[0] != uid or again[2] != start or again[1] in ("Z", "X"):
            return
        if handle is not None:
            signal.pidfd_send_signal(handle, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    finally:
        if handle is not None:
            os.close(handle)


def stop_agent_processes(timeout=60):
    """Leftover agent processes would hold the GPU lock or GPU memory (INVALID) or disturb timing windows. Kills every
    live process of the agent user, from /proc (no program by name), and returns those found first."""
    uid = pwd.getpwnam(AGENT).pw_uid
    found = ["%d %s" % (pid, proc_cmdline(pid)) for pid in sorted(agent_processes(uid))]
    deadline = time.time() + timeout
    while True:
        live = agent_processes(uid)
        if not live:
            return found
        if time.time() > deadline:
            raise RuntimeError("Agent processes survive SIGKILL: %s" % ["%d %s" % (pid, proc_cmdline(pid))
                                                                       for pid in sorted(live)[:5]])
        for pid, start in live.items():
            kill_process(pid, uid, start)
        time.sleep(1)


def test_marker(reward, ready):
    """A trial prepared at a test world size (RESEARCHBOOST_TEST_WORLD_SIZE, test-only) is not a bench
    score; its reward carries the world size so that it cannot be mistaken for one."""
    if ready.get("test_world_size"):
        reward = dict(reward, test_world_size=float(ready["test_world_size"]))
    return reward


AGENT_NOTES_MAX_BYTES = 4 << 20


def save_agent_notes(logs, tree, workspace):
    """The agent's work log (/submission/WORK_LOG.md, if it wrote one) and the commit history of its tree since the
    reference commit, with any uncommitted changes, kept as trial artifacts before judging. Not used for scoring;
    never fails the verifier."""
    work_log = Path("/submission/WORK_LOG.md")
    try:
        if work_log.is_file() and not work_log.is_symlink():
            with work_log.open("rb") as stream:
                (logs / "WORK_LOG.md").write_bytes(stream.read(AGENT_NOTES_MAX_BYTES))
    except OSError as exc:
        (logs / "WORK_LOG.md.error").write_text("%s\n" % exc)
    try:
        base = json.loads((Path(workspace) / "bench/bench.json").read_text()).get("reference_commit")
        # As the agent (its repository, its configuration), never as root; root-only programs and PATH.
        git = [tool("runuser"), "-u", AGENT, "--", tool("git"), "-c", "core.fsmonitor=false", "-c", "core.pager=cat",
               "-C", str(tree)]
        span = ["%s..HEAD" % base] if base else ["--max-count=200"]
        out = []
        for args in (["log", "--stat", "--no-color"] + span, ["status", "--porcelain", "--untracked-files=all"]):
            done = subprocess.run(git + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300,
                                  env=agent_env())
            out.append(b"$ git %s\n" % " ".join(args).encode() + done.stdout[:AGENT_NOTES_MAX_BYTES])
        (logs / "git-log-stat.txt").write_bytes(b"\n".join(out))
    except Exception as exc:  # noqa: BLE001 - artifacts only
        (logs / "git-log-stat.txt").write_text("git log failed: %r\n" % exc)


def main():
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    for name in ("reward.json", "reward.txt"):
        (logs / name).unlink(missing_ok=True)
    workspace = Path("/opt/researchboost/ws")
    ready = json.loads((workspace / "task/ready.json").read_text())
    smoke = ready.get("mode") == "smoke"
    # How this trial was prepared (full or from the preparation cache), for the trial's records.
    (logs / "ready.json").write_text(json.dumps(ready, indent=1) + "\n")
    convert = smoke_reward if smoke else reward
    switches = Path("/submission/switches.json")
    reserved = set(json.loads((workspace / "bench/bench.json").read_text())["reserved_env"])
    problem = switches_problem(switches, reserved)
    if problem:
        (logs / "harness.log").write_text("switches.json rejected before the check: %s\n" % problem)
        rejected = {"smoke_quick_pass": 0.0, "smoke_bitwise": 0.0} if smoke else dict(REJECTED)
        (logs / "reward.json").write_text(json.dumps(test_marker(rejected, ready)) + "\n")
        return
    leftover = stop_agent_processes()
    save_agent_notes(logs, "/submission/nanochat", workspace)
    leftover += stop_agent_processes()       # anything the notes' git (the agent's repository configuration) started
    # The judge runs as the agent, like the agent's own checks: same HOME and caches, the tree under test in the
    # agent's candidate environment, and candidate code never runs as root. Its output directory is kept for
    # diagnosis.
    user = pwd.getpwnam(AGENT)
    output = tempfile.mkdtemp(prefix="nanochat-final-", dir=str(logs))
    os.chown(output, user.pw_uid, user.pw_gid)
    # SAFE_PATH, not the image's PATH: the judge's own programs (nvidia-smi, git) are never the agent's.
    env = agent_env(ASSETS="/opt/researchboost/assets", HOME=user.pw_dir, USER=AGENT, LOGNAME=AGENT,
                    PYTHONNOUSERSITE="1")
    if not root_only(JUDGE_PYTHON):
        raise RuntimeError("%s is not root-only" % JUDGE_PYTHON)
    command = [tool("runuser"), "-u", AGENT, "--", JUDGE_PYTHON, str(workspace / "bench/bench.py"),
               "check", "--tree", "/submission/nanochat", "--config", str(switches), "--out", output]
    command += ["--quick"] if smoke else ["--final"]   # gate 1 as reference -> candidate -> reference
    if not smoke:
        allow_final()
    with (logs / "harness.log").open("w") as log:
        log.write("verifier: stopped %d leftover agent process(es) %s; judge runs as %s (uid %d), PATH %s\n"
                  % (len(leftover), leftover[:5], AGENT, user.pw_uid, SAFE_PATH))
        log.flush()
        completed = subprocess.run(command, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT)
    stop_agent_processes()
    check = Path(output) / "check.json"
    if not check.is_file():
        raise RuntimeError("Harness produced no verdict; see harness.log")
    raw = check.read_text()
    (logs / "check.json").write_text(raw)
    converted = convert(json.loads(raw), completed.returncode)
    (logs / "reward.json").write_text(json.dumps(test_marker(converted, ready), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
