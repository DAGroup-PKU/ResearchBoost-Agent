"""guard.py --report F -- CMD...: run the final judge (CMD: bash /tests/test.sh) as the only child of a root-owned
child subreaper that kills, as soon as it appears, every process orphaned inside the judge container that does not run
as root.

Why: the task's judge (verify.py -> bench.py check --final) runs the bench as the agent user, and the bench runs the
candidate's code as that same user. A candidate process that escapes its run (double fork + setsid, a clean
environment so the bench's tag-based kill misses it) could outlive the run and rewrite <out>/check.json in the moment
between the bench writing it and verify.py reading it. Under this guard such a process is an orphan, so the kernel
re-parents it to the guard (PR_SET_CHILD_SUBREAPER), and the guard SIGKILLs it within one tick (10 ms). While the
bench writes its verdict, the only agent-user processes left are therefore the bench itself and its live runs (which the
bench has already stopped by then). Gates, bench.py and verify.py are unchanged; only escapees die earlier than
verify.py's own stop_agent_processes() would kill them.

Reports {kills: [...], child_exit, ticks} to --report (root-only directory). Exit status = CMD's.
"""
import ctypes
import json
import os
import signal
import sys
import time

PR_SET_CHILD_SUBREAPER = 36
TICK = 0.01


def uid_of(pid):
    try:
        with open('/proc/%d/status' % pid) as f:
            for line in f:
                if line.startswith('Uid:'):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def comm_of(pid):
    try:
        with open('/proc/%d/comm' % pid) as f:
            return f.read().strip()
    except OSError:
        return '?'


def children(me):
    out = set()
    try:
        for tid in os.listdir('/proc/%d/task' % me):
            with open('/proc/%d/task/%s/children' % (me, tid)) as f:
                out.update(int(x) for x in f.read().split())
        return out
    except OSError:
        pass
    for d in os.listdir('/proc'):                   # fallback without /proc/<pid>/task/<tid>/children
        if d.isdigit():
            try:
                with open('/proc/%s/stat' % d) as f:
                    if int(f.read().rsplit(')', 1)[1].split()[1]) == me:
                        out.add(int(d))
            except (OSError, ValueError, IndexError):
                pass
    return out


def main():
    argv = sys.argv[1:]
    report = None
    if argv[:1] == ['--report']:
        report, argv = argv[1], argv[2:]
    if argv[:1] == ['--']:
        argv = argv[1:]
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        sys.exit('guard.py: prctl(PR_SET_CHILD_SUBREAPER) failed: errno %d' % ctypes.get_errno())
    me = os.getpid()
    child = os.fork()
    if child == 0:
        os.execvp(argv[0], argv)
    kills, ticks, child_status, signalled = [], 0, None, {}   # signalled: pid -> SIGKILLs sent (a dying process
    # with GPU memory can take seconds to go; it is signalled again every tick, recorded once)
    signal.signal(signal.SIGTERM, lambda *_: os.kill(child, signal.SIGTERM))
    while True:
        ticks += 1
        while True:                                  # reap whatever has exited (the judge, killed orphans)
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                pid = 0
            if not pid:
                break
            if pid == child:
                child_status = status
        others = [p for p in children(me) if p != child]
        for p in others:
            u = uid_of(p)
            if u is None:
                continue
            if u != 0 or child_status is not None:   # escapees; after the judge, everything left
                try:
                    os.kill(p, signal.SIGKILL)
                    if p not in signalled:
                        kills.append({'pid': p, 'uid': u, 'comm': comm_of(p), 't': round(time.time(), 3),
                                      'after_judge': child_status is not None})
                    signalled[p] = signalled.get(p, 0) + 1
                except ProcessLookupError:
                    pass
        if child_status is not None and not others:
            break
        time.sleep(TICK)
    code = os.waitstatus_to_exitcode(child_status) if hasattr(os, 'waitstatus_to_exitcode') else \
        (os.WEXITSTATUS(child_status) if os.WIFEXITED(child_status) else 128 + os.WTERMSIG(child_status))
    if report:
        with open(report, 'w') as f:
            json.dump({'kills': kills[:2000], 'killed': len(kills), 'signals': sum(signalled.values()),
                       'child_exit': code, 'ticks': ticks}, f, indent=1)
        os.chmod(report, 0o644)
    sys.exit(code)


if __name__ == '__main__':
    main()
