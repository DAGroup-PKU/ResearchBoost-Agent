"""Turn a prepared judge workspace into a preparation pack (tasks/<task>/environment/prep_pack.py).

    python scripts/make_prep_pack.py --task llava --workspace <copy of a prepared /opt/researchboost/ws> \\
        --image <the trial's image id> --out tasks/llava/prep-pack/ws2 --log <its prepare.log>

The workspace is a read-only copy of /opt/researchboost/ws taken after a full startup preparation finished
(task/ready.json with mode full). The image the preparation ran in gives what the workspace looked like before
(read from a created, never started container), the installed packages and the upstream commit. The pack holds the
files preparation changed, a sha256 manifest and pack.json; before it is kept, it is restored into a copy of the
image's workspace and every workspace input task/formulation.json declares is checked against its digest. Needs
Docker on the host; reads the workspace copy, never changes it.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
WS = '/opt/researchboost/ws'
# The tasks whose startup restores a pack from tasks/<task>/prep-pack (the others read <assets root>/<task>/pack; see
# their READMEs).
REFERENCES = {'llava': 'targets/reference', 'yoco': 'targets/reference'}
# Keys a task's bootstrap writes into bench/bench.json for the machine it runs on (the container's CPUs): a pack
# leaves bench.json out when these are its only change, so that the pack fits every machine (startup writes them).
MACHINE_KEYS = {'yoco': ('cpu_affinity',)}


def load_prep_pack(task):
    spec = importlib.util.spec_from_file_location('prep_pack', ROOT / 'tasks' / task / 'environment/prep_pack.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prep_version(task):
    found = re.search(r'^PREP_VERSION = (\d+)$', (ROOT / 'tasks' / task / 'environment/bootstrap.py').read_text(), re.M)
    return int(found.group(1))


def docker(*args):
    return subprocess.run(['docker', *args], check=True, capture_output=True, text=True).stdout.strip()


def content_digest(path):
    """The judges' content digest: sha256 of a file; for a directory, sha256 over (relative path, file sha256)."""
    path = Path(path)
    if path.is_file():
        return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    digest, count = hashlib.sha256(), 0
    for f in sorted(path.rglob('*')):
        if f.is_file():
            digest.update(str(f.relative_to(path)).encode() + b'\0'
                          + hashlib.sha256(f.read_bytes()).hexdigest().encode() + b'\n')
            count += 1
    return {'sha256': digest.hexdigest(), 'files': count}


def workspace_inputs(formulation):
    """(name, workspace path, sha256) of every workspace input formulation.json declares (assets are left out: they
    are checked by the judge in every trial). Two layouts: frozen_inputs / assets {name: {path, sha256}} (llava) and
    inputs {path: {kind: sha256, value}} (yoco; bytecode caches are left out, startup fingerprints its own)."""
    data = json.loads(Path(formulation).read_text())
    found = []
    for section in ('frozen_inputs', 'assets'):
        for name, item in (data.get(section) or {}).items():
            if isinstance(item, dict) and 'path' in item and 'sha256' in item:
                rel = item['path']
                if not rel.startswith(('/', '../')):
                    found.append((name, rel, item['sha256']))
    for rel, item in (data.get('inputs') or {}).items():
        if isinstance(item, dict) and item.get('kind') == 'sha256' and not rel.startswith(('/', '../')) \
                and '__pycache__' not in rel.split('/'):
            found.append((rel, rel, item['value']))
    return found


def machine_only_change(task, prepared, shipped):
    """True when the prepared bench/bench.json differs from the shipped one only in the task's MACHINE_KEYS."""
    keys = MACHINE_KEYS.get(task)
    if not keys:
        return False
    a, b = json.loads(Path(prepared).read_text()), json.loads(Path(shipped).read_text())
    return {k: v for k, v in a.items() if k not in keys} == {k: v for k, v in b.items() if k not in keys}


def check_pack(pp, pack, shipped, prepared, work):
    """Restore the pack into a copy of the shipped workspace (the configuration bootstrap computes itself taken from
    the prepared one) and compare every declared workspace input. Returns the number of inputs checked."""
    restored = work / 'restored'
    subprocess.run(['cp', '-a', str(shipped), str(restored)], check=True)
    for rel in pp.LOCAL_CONFIG:
        if (prepared / rel).is_file():
            shutil.copyfile(prepared / rel, restored / rel)
    pp.restore(pack, restored, owner=(subprocess.os.getuid(), subprocess.os.getgid()))
    inputs = workspace_inputs(restored / 'task/formulation.json')
    bad = [name for name, rel, want in inputs if content_digest(restored / rel)['sha256'] != want]
    if bad or not inputs:
        raise SystemExit('the pack does not reproduce the declared inputs: %s (checked %d)' % (bad, len(inputs)))
    return len(inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True, choices=sorted(REFERENCES))
    parser.add_argument('--workspace', type=Path, required=True, help='a copy of the prepared %s' % WS)
    parser.add_argument('--image', required=True, help='the image the preparation ran in')
    parser.add_argument('--out', type=Path, required=True, help='tasks/<task>/prep-pack/ws<world size>')
    parser.add_argument('--log', type=Path, help="the preparation's prepare.log, kept in the pack")
    parser.add_argument('--note', default='', help='provenance, e.g. host and trial')
    args = parser.parse_args()
    pp = load_prep_pack(args.task)
    ready = json.loads((args.workspace / 'task/ready.json').read_text())
    if ready.get('mode') != 'full':
        sys.exit('the workspace was not prepared in full mode: %s' % ready.get('mode'))
    if (ready.get('preparation') or {}).get('mode', 'full') != 'full':
        sys.exit('the workspace was prepared from a pack (%s); record a pack from a full preparation'
                 % ready['preparation']['mode'])
    work = Path(tempfile.mkdtemp(prefix='prep-pack-'))
    try:
        scratch = docker('create', args.image)
        try:
            docker('cp', '-a', '%s:%s' % (scratch, WS), str(work / 'shipped'))
            docker('cp', '-a', '%s:/opt/researchboost/installed-packages.txt' % scratch, str(work / 'packages.txt'))
        finally:
            docker('rm', scratch)
        commit = subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(work / 'shipped' / REFERENCES[args.task]),
                                 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        info = {'task': args.task, 'world_size': ready['world_size'], 'gpu': ready['gpus'][0],
                'gpu_count': len(ready['gpus']), 'prep_version': prep_version(args.task),
                'packages_sha256': pp.sha256_file(work / 'packages.txt'), 'upstream_commit': commit,
                'image': args.image, 'prepared': ready.get('created'), 'preparation_minutes': ready.get('minutes'),
                'test_world_size': ready.get('test_world_size'), 'note': args.note}
        source = args.workspace
        bench = 'bench/bench.json'
        if machine_only_change(args.task, args.workspace / bench, work / 'shipped' / bench):
            # the pack leaves the machine's bench.json out (MACHINE_KEYS); check_pack still compares the inputs with
            # the prepared workspace's own bench.json
            source = work / 'prepared'
            subprocess.run(['cp', '-a', str(args.workspace), str(source)], check=True)
            shutil.copyfile(work / 'shipped' / bench, source / bench)
            info['machine_local'] = {bench: list(MACHINE_KEYS[args.task])}
        meta = pp.make(source, work / 'shipped', work / 'pack', info, logs=[args.log] if args.log else ())
        checked = check_pack(pp, work / 'pack', work / 'shipped', args.workspace, work)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists():
            shutil.rmtree(args.out)
        shutil.copytree(work / 'pack', args.out)
        pp.verify(args.out)
        print(json.dumps(dict(meta, workspace_inputs_checked=checked, out=str(args.out)), indent=1))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
