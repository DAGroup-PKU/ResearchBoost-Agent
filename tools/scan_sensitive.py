#!/usr/bin/env python3
"""Pre-publish scan: nothing internal may be in the files of this repository.

    python3 tools/scan_sensitive.py                 # every tracked and untracked (not ignored) file of the repo
    python3 tools/scan_sensitive.py vetkit skills   # only these paths
    python3 tools/scan_sensitive.py --list-rules    # print the generic rules and how many private ones are loaded

Two kinds of rules:
  generic   in this file: private and carrier-grade address ranges, local proxy ports, home directories, e-mail
            addresses, API keys / tokens / private keys, credential assignments.
  private   your organization's own terms (company and bucket names, host and cluster names, people, internal tools
            and model code names, unreleased task names, judge internals). They are NOT in this repository: one
            `<category><TAB><regex>` per line in the files named by $SCAN_SENSITIVE_TERMS (colon separated), default
            ~/.config/scan-sensitive/terms.txt. Without any private rule the scan refuses to report "clean" (exit 3)
            unless --generic-only is given.

Run it before every commit that is meant to be shared, and paste its summary line into the pull request.
Exit 0: clean. Exit 1: something needs a human look (file, line and the matched text are printed).

A reviewed false positive goes into tools/scan_allow.txt as `<path glob><TAB><regex of the matched text>`; keep that
file short and say why in a comment line above each entry.
"""
import argparse
import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = {'tools/scan_sensitive.py', 'tools/scan_allow.txt'}
DEFAULT_TERMS = '~/.config/scan-sensitive/terms.txt'

RULES = [
    ('infra', r'\b10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b|\b192\.168\.[0-9]{1,3}\.[0-9]{1,3}\b'),
    ('infra', r'\b172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}\b'),
    ('infra', r'\b100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}\b'),
    ('infra', r'(127\.0\.0\.1|localhost):[0-9]{4,5}\b'),
    ('people', r'/Users/[A-Za-z]|/home/[a-z][a-z0-9_-]*/|-Users-[A-Za-z]'),
    ('people', r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    ('secrets', r'\bsk-[A-Za-z0-9_-]{16,}|\bghp_[A-Za-z0-9]{20,}|\bgho_[A-Za-z0-9]{20,}|\bxox[abp]-|AKIA[0-9A-Z]{16}'),
    ('secrets', r'LTAI[0-9A-Za-z]{12}|BEGIN [A-Z ]*PRIVATE KEY|ntn_[A-Za-z0-9]{20,}|secret_[A-Za-z0-9]{30,}'),
    ('secrets', r'(api[_-]?key|token|password|secret)\s*[:=]\s*["\']?[A-Za-z0-9_\-]{20,}'),
]
BINARY = {'.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.gz', '.tgz', '.pt', '.bin', '.so', '.pyc'}


def private_rules():
    rules, files = [], []
    for p in os.environ.get('SCAN_SENSITIVE_TERMS', DEFAULT_TERMS).split(':'):
        p = Path(os.path.expanduser(p))
        if not p.is_file():
            continue
        files.append(str(p))
        for line in p.read_text().splitlines():
            if line.strip() and not line.startswith('#'):
                cat, _, rx = line.partition('\t')
                rules.append((cat.strip() or 'private', rx.strip()))
    return rules, files


def repo_files(paths):
    out = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '--cached', '--others', '--exclude-standard', '--']
                         + list(paths), capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if p and (ROOT / p).is_file()]


def load_allow():
    allow = []
    f = ROOT / 'tools/scan_allow.txt'
    if f.is_file():
        for line in f.read_text().splitlines():
            if line.strip() and not line.startswith('#'):
                glob, _, rx = line.partition('\t')
                allow.append((glob.strip(), re.compile(rx.strip(), re.I)))
    return allow


def allowed(path, text, allow):
    return any(fnmatch.fnmatch(path, g) and rx.fullmatch(text) for g, rx in allow)


def scan(paths, rules):
    compiled = [(cat, re.compile(rx, re.I)) for cat, rx in rules]
    literals = [w.lower().encode() for cat, rx in rules for w in re.findall(r'[A-Za-z][A-Za-z0-9_-]{4,}', rx)
                if cat != 'secrets']
    allow, hits, n = load_allow(), [], 0
    for rel in repo_files(paths):
        if rel in SELF:
            continue
        n += 1
        for cat, rx in compiled:           # the path itself
            for m in rx.finditer(rel):
                if not allowed(rel, m.group(0), allow):
                    hits.append((cat, rel, 0, m.group(0)))
        if Path(rel).suffix.lower() in BINARY:
            data = (ROOT / rel).read_bytes().lower()
            hits += [('binary', rel, 0, w.decode()) for w in set(literals) if w in data]
            continue
        text = (ROOT / rel).read_text(errors='replace')
        for i, line in enumerate(text.splitlines(), 1):
            for cat, rx in compiled:
                for m in rx.finditer(line):
                    if not allowed(rel, m.group(0), allow):
                        hits.append((cat, rel, i, m.group(0)))
    return n, hits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='*')
    ap.add_argument('--list-rules', action='store_true')
    ap.add_argument('--generic-only', action='store_true', help='scan without private terms (not enough to publish)')
    a = ap.parse_args()
    priv, files = private_rules()
    if a.list_rules:
        for cat, rx in RULES:
            print('%-9s %s' % (cat, rx))
        print('private   %d rules from %s' % (len(priv), ', '.join(files) or '(none found)'))
        return 0
    if not priv and not a.generic_only:
        print('scan_sensitive: no private term list (set SCAN_SENSITIVE_TERMS or create %s); a generic-only scan '
              'is not enough to publish (use --generic-only to run it anyway)' % DEFAULT_TERMS, file=sys.stderr)
        return 3
    n, hits = scan(a.paths, RULES + priv)
    summary = '%d files, %d generic + %d private rules, %d allow-list entries' % (n, len(RULES), len(priv),
                                                                               len(load_allow()))
    if not hits:
        print('scan_sensitive: clean (%s)' % summary)
        return 0
    cats = {}
    for cat, rel, line, text in hits:
        cats.setdefault(cat, []).append('%s:%d: %s' % (rel, line, text))
    for cat, rows in cats.items():
        print('== %s (%d)' % (cat, len(rows)))
        for r in rows:
            print('   ' + r)
    print('scan_sensitive: %d hits (%s)' % (len(hits), summary))
    return 1


if __name__ == '__main__':
    sys.exit(main())
