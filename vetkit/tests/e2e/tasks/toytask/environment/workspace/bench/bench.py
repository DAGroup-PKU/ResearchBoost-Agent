"""Toy judge with the standard bench interface: record, compare, time, check [--final]."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

WS = Path(__file__).resolve().parents[1]
CFG = json.loads((WS / 'bench/bench.json').read_text())
# A real judge's gate 1 is max(1.02, 1 + 2 x the reference's spread). The toy's runs last ~0.1 s, often on a shared
# laptop, so its bar is wide: the faster submission of the end-to-end test is ~1.8x.
BAR = 1.3


def tree_of(t):
    t = Path(t)
    return t if (t / 'train.py').is_file() else t / 'toytask'


def run_tree(tree):
    with tempfile.TemporaryDirectory() as d:
        t0 = time.time()
        subprocess.run([sys.executable, str(tree_of(tree) / 'train.py'), d + '/out.json'], check=True)
        dt = time.time() - t0
        return Path(d + '/out.json').read_bytes(), dt


def record(a):
    out, _ = run_tree(a.tree)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / 'record.json').write_text(json.dumps({'label': a.label, 'digest': hashlib.sha256(out).hexdigest()}))


def compare(a):
    x, y = (json.loads(Path(p).read_text())['digest'] for p in (a.a, a.b))
    print('identical' if x == y else 'different')
    return 0 if x == y else 1


def time_(a):
    _, dt = run_tree(a.tree)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / 'time.json').write_text(json.dumps({'status': 'ok', 's_per_update': dt}))


def check(a):
    ref = WS / CFG['reference']
    cand = Path(a.tree or '/submission/toytask')
    out_r, t_r1 = run_tree(ref)
    out_c, t_c = run_tree(cand)
    _, t_r2 = run_tree(ref)
    t_r = min(t_r1, t_r2)
    same = out_r == out_c
    speedup = t_r / t_c
    status = 'INCORRECT' if not same else ('PASS' if speedup >= BAR else 'UNAVAILABLE')
    res = {'status': status, 'final': bool(a.final), 'performance': {'speedup': speedup},
           'correctness': {'verdict': 'identical' if same else 'different', 'measured_tier': 'N0' if same else None}}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / 'check.json').write_text(json.dumps(res))
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('record'); p.add_argument('--tree'); p.add_argument('--out'); p.add_argument('--label')
    p = sub.add_parser('compare'); p.add_argument('a'); p.add_argument('b')
    p = sub.add_parser('time'); p.add_argument('--tree'); p.add_argument('--out')
    p = sub.add_parser('check'); p.add_argument('--tree'); p.add_argument('--out', default='/tmp/check')
    p.add_argument('--final', action='store_true')
    a = ap.parse_args()
    return {'record': record, 'compare': compare, 'time': time_, 'check': check}[a.cmd](a) or 0


if __name__ == '__main__':
    sys.exit(main())
