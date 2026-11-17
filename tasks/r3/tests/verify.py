"""Run the R3 judge; translate its verdict without changing gates. Only a bitwise (N0) full-check PASS is accepted."""
import json
import math
import os
from pathlib import Path
import pwd
import subprocess
import tempfile
import time

AGENT = "agent"
REJECTED = {"accepted": 0.0, "speedup": 0.0}


def candidate_error(result):
    """ERROR verdicts the candidate caused, scored as a rejection instead of being left unscored: a tree without the
    entry, a candidate timing run that failed after the reference run succeeded, or a candidate record run that
    failed (the reference recorded on this host during preparation)."""
    correctness = result.get("correctness") or {}
    runs = (result.get("performance") or {}).get("runs") or {}
    if correctness.get("stage") == "structure":
        return True
    if correctness.get("stage") == "bitwise" and (correctness.get("numerics") or {}).get("status") == "ERROR":
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
    if result.get("quick") is not False or result.get("skip_timing") is not False or result.get("final") is not True:
        raise ValueError("Only a full check --final can receive a reward")
    status = result.get("status")
    correctness = result.get("correctness") or {}
    performance = result.get("performance") or {}
    if status == "PASS" and returncode == 0:
        speedup = performance.get("speedup")
        if (correctness.get("passed") is not True
                or correctness.get("verdict") != "bitwise"
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
    """Smoke mode: the quick check only (the bitwise gate). Not a benchmark score: no timing."""
    status = result.get("status")
    if status == "PASS" and returncode == 0:
        bitwise = ((result.get("correctness") or {}).get("numerics") or {}).get("bitwise")
        if type(bitwise) is not bool:
            raise ValueError("Smoke quick-check PASS without a bitwise flag")
        return {"smoke_quick_pass": 1.0, "smoke_bitwise": float(bitwise)}
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
    bad = sorted(str(k) for k in env if str(k) in reserved or str(k).startswith("R3_"))
    return "managed by the bench: %s" % bad if bad else None


def stop_agent_processes(timeout=60):
    """Leftover agent processes would hold the GPU lock or GPU memory (INVALID) or disturb timing windows.
    Returns the leftover processes found."""
    found = subprocess.run(["pgrep", "-a", "-u", AGENT], stdout=subprocess.PIPE, text=True).stdout.splitlines()
    deadline = time.time() + timeout
    while subprocess.run(["pgrep", "-u", AGENT], stdout=subprocess.DEVNULL).returncode == 0:
        if time.time() > deadline:
            raise RuntimeError("Agent processes survive SIGKILL")
        subprocess.run(["pkill", "-KILL", "-u", AGENT])
        time.sleep(1)
    return found


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
        # As the agent (its repository, its configuration), never as root.
        git = ["runuser", "-u", AGENT, "--", "git", "-c", "core.fsmonitor=false", "-c", "core.pager=cat", "-C",
               str(tree)]
        span = ["%s..HEAD" % base] if base else ["--max-count=200"]
        out = []
        for args in (["log", "--stat", "--no-color"] + span, ["status", "--porcelain", "--untracked-files=all"]):
            done = subprocess.run(git + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
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
    save_agent_notes(logs, "/submission/r3", workspace)
    # The judge runs as the agent, like the agent's own checks: same HOME and caches, the tree under test in the
    # agent's candidate environment, and candidate code never runs as root. Its output directory is kept for
    # diagnosis.
    user = pwd.getpwnam(AGENT)
    output = tempfile.mkdtemp(prefix="r3-final-", dir=str(logs))
    os.chown(output, user.pw_uid, user.pw_gid)
    env = dict(os.environ, ASSETS="/opt/researchboost/assets", HOME=user.pw_dir, USER=AGENT, LOGNAME=AGENT,
               PYTHONNOUSERSITE="1")
    command = ["runuser", "-u", AGENT, "--", "/opt/r3-env/bin/python", str(workspace / "bench/bench.py"),
               "check", "--tree", "/submission/r3", "--config", str(switches), "--out", output]
    command += ["--quick"] if smoke else ["--final"]   # gate 1 as reference -> candidate -> reference
    if not smoke:
        allow_final()
    with (logs / "harness.log").open("w") as log:
        log.write("verifier: stopped %d leftover agent process(es) %s; judge runs as %s (uid %d)\n"
                  % (len(leftover), leftover[:5], AGENT, user.pw_uid))
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
