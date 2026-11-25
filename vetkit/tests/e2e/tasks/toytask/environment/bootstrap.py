"""Toy startup preparation: freeze the reference record and timing (what a real task's bootstrap does, in small)."""
import json
from pathlib import Path
import subprocess
import sys
import time

WS = Path('/opt/researchboost/ws')
PREP_VERSION = 1


def gpus(config):
    return []          # CPU-only


def main():
    cfg = json.loads((WS / 'bench/bench.json').read_text())
    Path(cfg['gpu_lock']).touch()
    bench = [sys.executable, str(WS / 'bench/bench.py')]
    subprocess.run(bench + ['record', '--tree', str(WS / cfg['reference']), '--out', '/tmp/r', '--label', 'ref'],
                   check=True)
    (WS / 'bench/records').mkdir(parents=True, exist_ok=True)
    (WS / cfg['record']['reference']).write_text(Path('/tmp/r/record.json').read_text())
    ts = []
    for i in range(2):
        subprocess.run(bench + ['time', '--tree', str(WS / cfg['reference']), '--out', '/tmp/t%d' % i], check=True)
        ts.append(json.loads(Path('/tmp/t%d/time.json' % i).read_text())['s_per_update'])
    (WS / 'task').mkdir(exist_ok=True)
    m = sum(ts) / len(ts)
    (WS / 'task/timing_ref.json').write_text(json.dumps({'s_per_update': m, 'spread': abs(ts[0] - ts[1]) / m}))
    (WS / 'task/ready.json').write_text(json.dumps({'status': 'ready', 'mode': 'full', 'created': time.time()}))


if __name__ == '__main__':
    main()
