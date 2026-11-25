#!/usr/bin/env python3
"""onboard_check: read-only linter for one bench task directory (vetting protocol, stage C wrap-up).

    python3 vetkit/bin/onboard_check.py tasks/<task>                 # exit 0 ok, 2 problems (listed), 1 crash
    python3 vetkit/bin/onboard_check.py tasks/<task> --adapter own   # the task ships its own pack module

Checks what a task needs before nop smoke, pilots and a hand-over to the bench maintainers:
  - layout: task.toml, instruction.md, environment/{Dockerfile, docker-compose.yaml, entrypoint.sh, bootstrap.py,
    no_credentials.py}, tests/{test.sh, verify.py};
  - task.toml against the protocol (artifacts, [agent], [environment], healthcheck, [verifier]); with Harbor installed
    also Harbor's own model (harbor.models.task.config.TaskConfig);
  - the standard bench interface the generic pack module and the judge rely on: bench/bench.py with the record,
    compare, time and check subcommands and `check --final`; bench/bench.json with reference, record.reference,
    gpu_lock, timing.spread_band, world_size (= task.toml gpus), reserved_env, invalid_run_signatures;
  - the statement: no durations (hard rule 2), no vetting data copied into the image;
  - the verifier (tests/): no program a verifier commonly runs is called by bare name (the verifier is root, and the
    image's PATH may start with a directory the agent can write: protocol supplement C10); a warning when the image's
    PATH has a directory the Dockerfile gives to the agent;
  - git: no uncommitted changes under the task directory (the judge hash and the pack key must name a commit).
Nothing is written. Problems block the hand-over; warnings need a look and a sentence in the report.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hostlib import load_toml  # noqa: E402

NAME = re.compile(r'^[a-z][a-z0-9_]{1,40}$')
HEALTH = 'test -f /run/researchboost-ready'
DURATION = re.compile(r'(?i)\b\d+(\.\d+)?\s*(-\s*)?(h|hrs?|hours?|mins?|minutes?)\b|\d+\s*(小时|分钟)')
# Programs a verifier commonly runs. By bare name (the first word of an argv list, the word after `--` for runuser or
# su, os.system / os.popen, a shell line) they are looked up on the inherited PATH (protocol supplement C10).
TOOLS = (r'pgrep|pkill|kill|killall|ps|runuser|su|sudo|setpriv|git|nvidia-smi|python[0-9.]*|bash|sh|env|timeout|'
         r'taskset|numactl|chown|chmod|rm|cp|mv|tar|find|ldconfig|fuser|lsof|cat|head|tail|date|touch|mkdir')
BY_NAME_PY = re.compile(r'(\[\s*|["\']--["\']\s*,\s*|os\.(?:system|popen)\(\s*)["\'](%s)(?=["\'\s])' % TOOLS)
BY_NAME_SH = re.compile(r'^\s*(?:exec\s+|nohup\s+)?(%s)(?=\s|$)' % TOOLS)


def git(path, *args):
    r = subprocess.run(['git', '-C', str(path)] + list(args), capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def check_layout(td, problems, warnings):
    for rel in ('task.toml', 'instruction.md', 'environment/Dockerfile', 'environment/bootstrap.py',
                'tests/test.sh', 'tests/verify.py'):
        if not (td / rel).is_file():
            problems.append('%s missing' % rel)
    for rel in ('environment/docker-compose.yaml', 'environment/entrypoint.sh', 'environment/no_credentials.py',
                'README.md'):
        if not (td / rel).is_file():
            warnings.append('%s missing (every released task has one)' % rel)
    for rel in ('vetting', 'environment/vetting', 'solution/vetting'):
        if (td / rel).exists():
            problems.append('%s inside the task: vetting data stays out of tasks/ (hard rule 1)' % rel)


def check_toml(td, task, cfg, problems, warnings, facts):
    tk, ag, env, ver = (cfg.get(k) or {} for k in ('task', 'agent', 'environment', 'verifier'))
    hc = env.get('healthcheck') or {}
    facts['gpus'] = env.get('gpus')
    if tk.get('name') != 'researchboost/%s' % task:
        warnings.append('[task] name is %r; the bench uses "researchboost/%s"' % (tk.get('name'), task))
    workdir = env.get('workdir')
    if not workdir or not str(workdir).startswith('/submission/'):
        problems.append('[environment] workdir %r: the upstream checkout under /submission/<name>' % workdir)
    elif workdir != '/submission/%s' % task:
        warnings.append('[environment] workdir %r is not /submission/%s (fine when upstream uses another name)'
                        % (workdir, task))
    want = [workdir, '/submission/switches.json', '/opt/candidate-env']
    if cfg.get('artifacts') != want:
        problems.append('artifacts = %r; a re-judge needs exactly %r' % (cfg.get('artifacts'), want))
    if ag.get('user') != 'agent':
        problems.append('[agent] user must be "agent" (the agent never gets root)')
    if ag.get('network_mode') != 'public':
        warnings.append('[agent] network_mode %r (released tasks: "public")' % ag.get('network_mode'))
    if ag.get('timeout_sec') != 36000:
        warnings.append('[agent] timeout_sec %r (the bench budget is 36000 = 10 h)' % ag.get('timeout_sec'))
    if env.get('docker_image'):
        problems.append('[environment] docker_image set: images are built on site from environment/Dockerfile')
    if not isinstance(env.get('gpus'), int):
        problems.append('[environment] gpus missing (0 for a CPU-only task)')
    elif env['gpus'] > 0 and 'H200' not in (env.get('gpu_types') or []):
        warnings.append('[environment] gpu_types %r: the public board runs on H200-class GPUs' % env.get('gpu_types'))
    for k in ('cpus', 'memory_mb', 'build_timeout_sec'):
        if k not in env:
            warnings.append('[environment] %s missing' % k)
    if hc.get('command') != HEALTH:
        problems.append('[environment.healthcheck] command must be %r' % HEALTH)
    if not hc.get('start_period_sec'):
        problems.append('[environment.healthcheck] start_period_sec missing (must cover the whole preparation)')
    if ver.get('environment_mode') != 'shared':
        problems.append('[verifier] environment_mode must be "shared" (the judge runs in the agent\'s container)')
    if ver.get('user') != 'root':
        problems.append('[verifier] user must be "root"')
    if ver.get('network_mode') != 'no-network':
        problems.append('[verifier] network_mode must be "no-network"')
    if not ver.get('timeout_sec'):
        problems.append('[verifier] timeout_sec missing (must cover one full check --final)')
    try:
        from harbor.models.task.config import TaskConfig
    except ImportError:
        warnings.append('Harbor is not installed here: task.toml not validated against its model '
                        '(pip install harbor==0.24.0)')
        return
    try:
        TaskConfig.model_validate(cfg)
        facts['harbor_model'] = 'valid'
    except Exception as e:  # noqa: BLE001 - pydantic's error, reported
        problems.append('Harbor rejects task.toml: %s' % str(e).replace('\n', ' ')[:400])


def check_bench(td, adapter, gpus, problems, warnings, facts):
    benches = sorted(td.rglob('bench/bench.py'))
    jsons = sorted(td.rglob('bench/bench.json'))
    if not benches or not jsons:
        msg = ('no bench/bench.py and bench/bench.json in the task tree (found %d and %d)' % (len(benches), len(jsons)))
        (problems if adapter == 'generic' else warnings).append(
            msg + ('; the generic pack module needs the standard bench interface (or use --adapter own)'
                   if adapter == 'generic' else ''))
        return
    facts['bench'] = str(benches[0].relative_to(td))
    src = benches[0].read_text(errors='replace')
    for sub in ('record', 'compare', 'time', 'check'):
        if not re.search(r"""['"]%s['"]""" % sub, src):
            problems.append('bench.py has no %r subcommand (standard bench interface)' % sub)
    if '--final' not in src:
        problems.append('bench.py has no `check --final` (the verifier-only final evaluation)')
    try:
        bj = json.loads(jsons[0].read_text())
    except ValueError as e:
        problems.append('bench.json does not parse: %s' % e)
        return
    for k in ('reference', 'gpu_lock', 'world_size'):
        if k not in bj:
            problems.append('bench.json has no %r' % k)
    if not (bj.get('record') or {}).get('reference'):
        problems.append('bench.json has no record.reference (the frozen reference record)')
    if (bj.get('timing') or {}).get('spread_band') is None:
        warnings.append('bench.json has no timing.spread_band: the pack timing check needs a band (refpack.py --band)')
    if isinstance(gpus, int) and 'world_size' in bj and int(bj['world_size']) != gpus:
        problems.append('bench.json world_size %s != task.toml gpus %s' % (bj['world_size'], gpus))
    if not bj.get('reserved_env'):
        warnings.append('bench.json has no reserved_env (variables the judge owns and the candidate cannot set)')
    if not bj.get('invalid_run_signatures'):
        warnings.append('bench.json has no invalid_run_signatures (log lines that void a run: recompile limit, '
                        'skipped CUDA graphs, ...)')


def check_statement(td, problems):
    p = td / 'instruction.md'
    if not p.is_file():
        return
    for i, line in enumerate(p.read_text(errors='replace').splitlines(), 1):
        m = DURATION.search(line)
        if m:
            problems.append('instruction.md:%d states a duration %r (hard rule 2: no time in the statement)'
                            % (i, m.group(0)))


def check_dockerfile(td, problems):
    p = td / 'environment/Dockerfile'
    if not p.is_file():
        return
    for i, line in enumerate(p.read_text(errors='replace').splitlines(), 1):
        if re.match(r'(?i)\s*(COPY|ADD)\b', line) and re.search(r'(?i)\bvetting\b|stages\.json|peaks\.json', line):
            problems.append('environment/Dockerfile:%d copies vetting data into the image' % i)


def check_verifier(td, problems, warnings):
    """Protocol supplement C10: the verifier (root) runs programs by absolute path or from a PATH of root-only
    directories, never by bare name from the image's PATH."""
    for p in sorted((td / 'tests').glob('*')):
        if p.suffix not in ('.py', '.sh') or not p.is_file():
            continue
        rx = BY_NAME_PY if p.suffix == '.py' else BY_NAME_SH
        for i, line in enumerate(p.read_text(errors='replace').splitlines(), 1):
            if line.lstrip().startswith('#'):
                continue
            for m in rx.finditer(line):
                problems.append("tests/%s:%d runs %r by bare name: the verifier is root and the image's PATH may start "
                                'with a directory the agent can write; use an absolute path, a PATH of root-only '
                                'directories, or /proc for processes (protocol supplement C10)'
                                % (p.name, i, m.groups()[-1]))
    dockerfile = td / 'environment/Dockerfile'
    if not dockerfile.is_file():
        return
    text = dockerfile.read_text(errors='replace').replace('\\\n', ' ')
    given = set()
    for m in re.finditer(r'\bchown\s+(?:-\S+\s+)*agent(?::\S*)?\s+([^&;|\n]+)', text):
        given.update(t for t in m.group(1).split() if t.startswith('/'))
    for m in re.finditer(r'(?m)^\s*ENV\b.*?\bPATH[= ]"?([^"\s]+)', text):
        for entry in m.group(1).split(':'):
            if any(entry == g or entry.startswith(g.rstrip('/') + '/') for g in given):
                warnings.append("the image's PATH has %s, which the Dockerfile gives to the agent: the verifier must "
                                'not run anything by name from the inherited PATH, nor pass that PATH to what it runs '
                                'as the agent (protocol supplement C10)' % entry)


def check_git(td, allow_dirty, problems, warnings, facts):
    rc, top = git(td, 'rev-parse', '--show-toplevel')
    if rc:
        warnings.append('not in a git repository: the judge hash and pack key cannot name a commit')
        return
    facts['commit'] = git(td, 'rev-parse', 'HEAD')[1]
    facts['branch'] = git(td, 'rev-parse', '--abbrev-ref', 'HEAD')[1]
    st = git(td, 'status', '--porcelain', '--', '.')[1]
    if st:
        (warnings if allow_dirty else problems).append(
            '%d uncommitted or untracked files under the task: commit them (the pack key and the judge hash must '
            'name a commit)' % len(st.splitlines()))


def check(task_dir, adapter='generic', allow_dirty=False):
    td = Path(task_dir).resolve()
    task = td.name
    facts, problems, warnings = {'task': task, 'dir': str(td)}, [], []
    if not NAME.match(task):
        problems.append('task name %r: lower case letters, digits, _ (2-41 chars)' % task)
    if not td.is_dir():
        problems.append('%s does not exist' % td)
        return facts, problems, warnings
    check_layout(td, problems, warnings)
    cfg = {}
    if (td / 'task.toml').is_file():
        try:
            cfg = load_toml(td / 'task.toml')
        except Exception as e:  # noqa: BLE001
            problems.append('task.toml does not parse: %s' % e)
    if cfg:
        check_toml(td, task, cfg, problems, warnings, facts)
    check_bench(td, adapter, facts.get('gpus'), problems, warnings, facts)
    check_statement(td, problems)
    check_dockerfile(td, problems)
    check_verifier(td, problems, warnings)
    check_git(td, allow_dirty, problems, warnings, facts)
    return facts, problems, warnings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('task_dir', help='tasks/<task>')
    ap.add_argument('--adapter', default='generic', choices=('generic', 'own'),
                    help='generic: the task uses the standard bench interface (container/tasks/_generic.py); '
                         'own: the task ships container/tasks/<task>.py')
    ap.add_argument('--allow-dirty', action='store_true', help='uncommitted changes are a warning, not a problem')
    a = ap.parse_args(argv)
    facts, problems, warnings = check(a.task_dir, a.adapter, a.allow_dirty)
    print(json.dumps({'facts': facts, 'problems': problems, 'warnings': warnings}, indent=1, ensure_ascii=False))
    return 2 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
