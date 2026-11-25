"""Toy verifier: the final check as the agent user, then reward.json (PASS -> speedup, else 0). Programs and PATH are
root-only (protocol supplement C10): the image's PATH starts with the agent's /opt/candidate-env/bin."""
import json
import os
from pathlib import Path
import shutil
import subprocess

SYSTEM_PATH = '/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
out = Path('/logs/verifier')
subprocess.run([shutil.which('runuser', path=SYSTEM_PATH), '-u', 'agent', '--', '/usr/local/bin/python3',
                '/opt/researchboost/ws/bench/bench.py', 'check', '--final', '--out', '/tmp/final'],
               check=True, env=dict(os.environ, PATH=SYSTEM_PATH))
check = json.loads(Path('/tmp/final/check.json').read_text())
(out / 'check.json').write_text(json.dumps(check))
ok = check['status'] == 'PASS'
(out / 'reward.json').write_text(json.dumps({'accepted': int(ok), 'speedup': check['performance']['speedup'] if ok else 0}))
