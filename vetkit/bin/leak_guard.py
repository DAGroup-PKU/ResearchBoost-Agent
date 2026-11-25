#!/usr/bin/env python3
"""Leak guard: nothing from earlier optimization work or from the vetting may reach the agent (hard rule 1).

Scans agent-visible material for banned content: speedup targets and results, references to optimized solutions,
work logs, notes, profiles and hotspot analyses, other hardware, task narratives inside the workspace (the task
statement is instruction.md), plus the task's own deny list.

Agent-visible material of a task (agent_visible_files):
  - instruction.md (the task statement Harbor gives the agent);
  - everything the image recipe copies in (the COPY / ADD sources of environment/Dockerfile: the judge workspace, the
    startup code, the constraints file);
  - the task's preparation packs (prep-pack/), mounted read-only into the container.
With --image, a built image is scanned as well: /opt/researchboost without the upstream reference checkout and
/home/agent, copied out of a created (never started) container, and the candidate tree must be the clean upstream
checkout (/submission/<task> at bench.json's reference_commit, no changes).

    python3 vetkit/bin/leak_guard.py tasks/<task>
    python3 vetkit/bin/leak_guard.py tasks/<task> --deny vetting/<task>/deny.txt
    python3 vetkit/bin/leak_guard.py tasks/<task> --image <task>=<image tag>        # needs docker

--deny FILE: one regular expression per line (# comments), the task's own secrets: earlier results and run names,
switch names from the B-stage ladder, hotspots, names of published fast versions that the statement does NOT name.
Keep that file in vetting/, never in tasks/.

Every hit needs a human look; the grep is a backstop. Allowed by design: the bar 1.02 and the list of forbidden
published fast versions that the statement must name. Read the agent-visible files once yourself as well: switch
names and optimization ideas do not match a regular expression. Exit 0 clean, 1 findings (JSON on stdout).
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

# Applied to every text file.
TOKENS = [
    (r'(?i)\bfaster-[a-z]', 'reference to an optimized solution repository (Faster-*)'),
    (r'WORK_?LOG', 'work log'),
    (r'NOTES\.md', 'notes file'),
    (r'(?i)hot-?spots?', 'hotspot analysis'),
    (r'(?i)\bbottleneck(s|ed)?\b', 'bottleneck analysis'),
    (r'(?i)profil(e|ing) (shows|showed|says)|we found|best (result|speed)', 'analysis or earlier result'),
    (r'baseline\.md', 'baseline profile document'),
    (r'\.codex/skills|\.claude/skills', 'methodology skills from earlier runs'),
    (r'(?i)\bjudge[_ ]changes\b', 'judge change log'),
    (r'(?i)\bvetting/|\bstages\.json\b|\bpeaks\.json\b|\bT_?ref ?/ ?T1\b|\blog[- ]headroom\b',
     'vetting data (bounds, ladder, scorecard)'),
    (r'(?<![A-Za-z0-9])(A100|A800|H100|H800|H20|L[24]0[A-Z]?|B100|B200|B300|GB200|GB300|MI[23]\d\dX?|RTX ?\d{4})\b',
     'other hardware (the statement names only the target GPU class)'),
]
# Applied to narrative and code (not to package lists or recorded run data, where any number can occur).
NUMBERS = [
    (r'(?i)(speed-?up|faster|target)[^\n.(]{0,30}?\b\d+(\.\d+)?\s*[x×](?![\w(])', 'speedup target or result'),
    (r'(?i)\b\d+(\.\d+)?\s*[x×]\s*(speed-?up|faster)', 'speedup target or result'),
    (r'(?i)speed-?up[^\n]{0,20}?(≥|>=|>)\s*\d', 'speedup target'),
]
NARRATIVE_SUFFIXES = {'.md', '.py', '.toml', '.sh', '.yaml', '.yml', '.cfg', '.ini', '.txt', ''}
# Files that must not exist in agent-visible material.
BANNED_NAMES = [
    (r'(^|/)GOAL\.md$', 'task narrative inside the workspace (the task statement is instruction.md)'),
    (r'(^|/)(WORK_?LOG|NOTES|baseline|judge_changes|REPORT)\.md$', 'work log, notes, report or baseline profile'),
    (r'(^|/)(vetting|stages|peaks)\.json$', 'vetting data'),
    (r'(^|/)profile\.(json|md)$|\.torchprof\.json$|\.trace\.json$|\.nsys-rep$|\.qdrep$|\.ncu-rep$', 'profile output'),
]
BINARY_SUFFIXES = {'.pt', '.pth', '.bin', '.safetensors', '.mdb', '.npy', '.npz', '.pkl', '.so', '.pyc', '.png',
                   '.whl', '.jpg', '.jpeg', '.gz', '.tgz', '.zip', '.parquet', '.arrow'}


def _text(path):
    if path.suffix.lower() in BINARY_SUFFIXES:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b'\0' in data[:4096]:
        return None
    return data.decode('utf-8', errors='replace')


def load_deny(path):
    rules = []
    if path:
        for line in Path(path).read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#'):
                rules.append((line.strip(), 'task deny list (%s)' % Path(path).name))
    return rules


def scan_text(label, text, numbers=True, deny=()):
    findings = []
    for pattern, reason in TOKENS + list(deny) + (NUMBERS if numbers else []):
        for match in re.finditer(pattern, text):
            line = text.count('\n', 0, match.start()) + 1
            findings.append({'file': label, 'line': line, 'match': match.group(0)[:80], 'reason': reason})
    return findings


def scan_file(path, label, deny=()):
    findings = []
    for pattern, reason in BANNED_NAMES:
        if re.search(pattern, label):
            findings.append({'file': label, 'line': 0, 'match': Path(label).name, 'reason': reason})
    text = _text(path)
    if text is None:
        return findings
    numbers = path.suffix in NARRATIVE_SUFFIXES or path.name == 'bench.json'
    return findings + scan_text(label, text, numbers, deny)


def dockerfile_sources(environment):
    """COPY/ADD sources of environment/Dockerfile (relative to the build context), shell and JSON (exec) form,
    without --from stages."""
    sources = []
    for line in (environment / 'Dockerfile').read_text().replace('\\\n', ' ').splitlines():
        stripped = line.strip()
        head, _, rest = stripped.partition(' ')
        if head.upper() not in ('COPY', 'ADD'):
            continue
        flags = []
        while rest.startswith('--'):
            flag, _, rest = rest.partition(' ')
            flags.append(flag)
            rest = rest.lstrip()
        if any(f.startswith('--from') for f in flags):
            continue
        if rest.startswith('['):
            args = json.loads(rest)
        else:
            args = shlex.split(rest, comments=True)
        sources += args[:-1]
    return sources


def agent_visible_files(task, unresolved=None):
    """(path, label) of every file of a task that the agent can see. COPY sources that match nothing are appended to
    `unresolved` (a list), so that nothing is skipped silently."""
    task = Path(task)
    out = [(task / 'instruction.md', 'instruction.md')]
    environment = task / 'environment'
    for source in dockerfile_sources(environment):
        if '://' in source:
            if unresolved is not None:
                unresolved.append(source)     # a remote ADD: not scanned here; scan the image (--image)
            continue
        roots = sorted(environment.glob(source)) if any(c in source for c in '*?[') else [environment / source]
        roots = [r for r in roots if r.exists()]
        if not roots and unresolved is not None:
            unresolved.append(source)
        for root in roots:
            files = [root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file())
            for path in files:
                if '__pycache__' in path.parts:
                    continue
                out.append((path, path.relative_to(task).as_posix()))
    packs = task / 'prep-pack'       # mounted read-only into the container (/opt/researchboost/prep-pack)
    if packs.is_dir():
        out += [(path, path.relative_to(task).as_posix()) for path in sorted(packs.rglob('*')) if path.is_file()]
    return out


def scan_task(task, deny=()):
    findings, unresolved = [], []
    for path, label in agent_visible_files(task, unresolved):
        findings += scan_file(path, label, deny)
    findings += [{'file': 'environment/Dockerfile', 'line': 0, 'match': src[:80],
                  'reason': 'COPY/ADD source not found or remote: not scanned (fix the path, or scan the image)'}
                 for src in unresolved]
    return findings


def scan_tree(root, skip=(), deny=()):
    root = Path(root)
    findings = []
    for path in sorted(p for p in root.rglob('*') if p.is_file() and not p.is_symlink()):
        label = path.relative_to(root).as_posix()
        if '__pycache__' in path.parts or any(label == s or label.startswith(s + '/') for s in skip):
            continue
        findings += scan_file(path, label, deny)
    return findings


def scan_image(tag, task, deny=()):
    """Copy the agent-visible directories out of a created (not started) container of the image and scan them;
    check that the candidate tree is the clean upstream checkout."""
    task = Path(task)
    config = json.loads(next(task.rglob('bench/bench.json')).read_text())
    reference = config['reference']                       # relative to the workspace
    workdir = '/submission/%s' % task.name
    cid = subprocess.run(['docker', 'create', tag], check=True, capture_output=True, text=True).stdout.strip()
    findings = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for src in ('/opt/researchboost', '/home/agent', workdir):
                dest = Path(tmp) / src.strip('/').replace('/', '_')
                subprocess.run(['docker', 'cp', '%s:%s' % (cid, src), str(dest)], check=True, capture_output=True)
            findings += [dict(f, file='/opt/researchboost/' + f['file'])
                         for f in scan_tree(Path(tmp) / 'opt_researchboost', skip=('ws/' + reference,), deny=deny)]
            findings += [dict(f, file='/home/agent/' + f['file'])
                         for f in scan_tree(Path(tmp) / 'home_agent', deny=deny)]
            tree = Path(tmp) / workdir.strip('/').replace('/', '_')
            head = subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(tree), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True).stdout.strip()
            dirty = subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(tree), 'status', '--porcelain'],
                                   capture_output=True, text=True).stdout.strip()
            want = config.get('reference_commit')
            if not want or head != want or dirty:
                findings.append({'file': workdir, 'line': 0, 'match': head[:12],
                                 'reason': 'the candidate tree is not the clean upstream checkout at %s%s'
                                           % ((want or '?')[:12], ' (changed files)' if dirty else '')})
    finally:
        subprocess.run(['docker', 'rm', cid], capture_output=True)
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('tasks', nargs='+', type=Path)
    parser.add_argument('--deny', help='file of extra regular expressions for these tasks (keep it in vetting/)')
    parser.add_argument('--image', action='append', default=[], metavar='TASK=TAG',
                        help='also scan a built image of a task (task directory name = image tag)')
    args = parser.parse_args(argv)
    deny = load_deny(args.deny)
    findings = []
    for task in args.tasks:
        findings += [dict(f, task=task.name) for f in scan_task(task, deny)]
    by_name = {task.name: task for task in args.tasks}
    for item in args.image:
        name, _, tag = item.partition('=')
        if name not in by_name or not tag:
            parser.error('--image %s: expected <task>=<tag> for one of the tasks given' % item)
        findings += [dict(f, task=name, image=tag) for f in scan_image(tag, by_name[name], deny)]
    print(json.dumps(findings, indent=1, ensure_ascii=False))
    return 1 if findings else 0


if __name__ == '__main__':
    sys.exit(main())
