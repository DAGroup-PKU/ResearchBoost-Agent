"""svcpack.py: reference packs inside a task container (bin/refpack.py runs it through entry.sh).

Runs as root with the task's own training interpreter (entry.sh picks it), from the image's /opt/researchboost:

  svcpack.py keyinfo  --task T                        the key fields this image determines (judge_hash, world_size,
                                                      prep_version) as JSON on stdout; no GPU needed
  svcpack.py pristine --task T --out F                manifest of the pack roots before preparation (pack build)
  svcpack.py collect  --task T --pristine F --out D   after the task's own preparation: the pack (files/ manifest.json
                                                      collect.json) = every file preparation wrote or changed under
                                                      the pack roots, and what it deleted
  svcpack.py verify   --task T --pack D --report F    pack mode (trial startup and every judge run): check the pack's
                                                      integrity and key, restore it, then ONLY verify on this machine:
                                                      the reference's quick record once (bitwise against the pack's)
                                                      and one reference timing inside the task's band. Never falls
                                                      back to a preparation.

Exit codes of verify: 0 ready; 81 the pack does not verify here (record differs, timing outside the band, key or
integrity mismatch): an infrastructure stop, the pack is not replaced; 82 no pack given; 1 anything else (a crash).
The report (JSON) says which step failed and why.

Per-task logic lives in tasks/<task>.py next to this file; it imports the image's own /opt/researchboost/bootstrap.py
and calls its functions (GPU checks, lock files, configuration the machine computes), so the steps are the task's.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if __name__ == '__main__':
    sys.modules.setdefault('svcpack', sys.modules['__main__'])   # task modules `from svcpack import ...` this module
import svckey  # noqa: E402

WS = Path('/opt/researchboost/ws')
PREP = Path('/opt/researchboost/prep')
DEFAULT_ROOTS = (str(WS), str(PREP))
SKIP_PARTS = ('__pycache__',)
RUNS_FILE_CAP = 64 << 20          # files under ws/runs/ larger than this stay out of packs (logged in collect.json)
EXIT_MISMATCH, EXIT_NOPACK = 81, 82


class Mismatch(Exception):
    """The pack does not verify on this machine (exit 81)."""


def log(msg):
    print('[svcpack %s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def load_task(task):
    # RBB_SVC_TASK_CONFIG (refpack.py, for a task without a module of its own): {"adapter": "generic"} loads
    # tasks/_generic.py, configured by the same JSON; without it, tasks/<task>.py is loaded
    cfg = json.loads(os.environ.get('RBB_SVC_TASK_CONFIG') or '{}')
    name = '_generic' if cfg.get('adapter') == 'generic' else (cfg.get('adapter') or task)
    path = HERE / 'tasks' / ('%s.py' % name)
    if not path.is_file():
        raise SystemExit('svcpack: no task module %s' % path)
    spec = importlib.util.spec_from_file_location('svctask_%s' % task, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def import_bootstrap():
    """The image's bootstrap.py as a module (its main() is not run)."""
    sys.path.insert(0, '/opt/researchboost')
    spec = importlib.util.spec_from_file_location('bootstrap', '/opt/researchboost/bootstrap.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['bootstrap'] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ manifests

def skip_rel(root, rel, is_ws=None):
    parts = Path(rel).parts
    if any(p in SKIP_PARTS for p in parts):
        return True
    is_ws = (root == str(WS)) if is_ws is None else is_ws
    return bool(is_ws and parts and parts[0] == 'targets')   # the upstream checkout: not preparation output


def scan(roots):
    """{absolute path: entry} of every regular file and symlink under the roots, plus {'dirs': [...]}.
    entry: {type: file, sha256, size, mode, uid, gid} or {type: link, target}."""
    files, dirs = {}, []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for d, subdirs, names in os.walk(root):
            rel_d = os.path.relpath(d, root)
            if rel_d != '.' and skip_rel(root, rel_d):
                subdirs[:] = []
                continue
            dirs.append(d)
            for name in names + [s for s in subdirs if os.path.islink(os.path.join(d, s))]:
                p = os.path.join(d, name)
                rel = os.path.relpath(p, root)
                if skip_rel(root, rel):
                    continue
                st = os.lstat(p)
                if stat.S_ISLNK(st.st_mode):
                    files[p] = {'type': 'link', 'target': os.readlink(p)}
                elif stat.S_ISREG(st.st_mode):
                    files[p] = {'type': 'file', 'sha256': svckey.sha256_file(p), 'size': st.st_size,
                                'mode': stat.S_IMODE(st.st_mode), 'uid': st.st_uid, 'gid': st.st_gid}
    return {'files': files, 'dirs': sorted(dirs)}


def judge_hash(ws=WS):
    """sha256 over the judge workspace as shipped: every regular file except targets/ and __pycache__."""
    h = hashlib.sha256()
    entries = []
    for d, subdirs, names in os.walk(str(ws)):
        rel_d = os.path.relpath(d, str(ws))
        if rel_d != '.' and skip_rel(str(ws), rel_d, True):
            subdirs[:] = []
            continue
        for name in names:
            p = os.path.join(d, name)
            rel = os.path.relpath(p, str(ws))
            if skip_rel(str(ws), rel, True) or os.path.islink(p) or not os.path.isfile(p):
                continue
            entries.append((rel, svckey.sha256_file(p)))
    for rel, digest in sorted(entries):
        h.update(rel.encode() + b'\0' + digest.encode() + b'\n')
    return h.hexdigest()


def prep_version(bootstrap_path='/opt/researchboost/bootstrap.py'):
    import re
    text = Path(bootstrap_path).read_text()
    m = re.search(r'^PREP_VERSION = (\d+)\s*$', text, re.M)
    return 'v%s' % m.group(1) if m else 'bootstrap-sha256:%s' % svckey.sha256_file(bootstrap_path)


def host_gpu_facts():
    """A CPU-only task (world size 0): no GPU in this container; the host passes its slot GPUs' model and driver (the
    pack key's host-type fields; refpack.py sets them)."""
    model, driver = os.environ.get('RBB_SVC_HOST_GPU_MODEL'), os.environ.get('RBB_SVC_HOST_DRIVER')
    if not model or not driver:
        raise Mismatch('a CPU-only task without RBB_SVC_HOST_GPU_MODEL / RBB_SVC_HOST_DRIVER from the host')
    seen = sorted(str(p) for p in Path('/dev').glob('nvidia*'))
    if seen:
        raise Mismatch('a CPU-only task sees GPU devices in its container: %s' % seen[:4])
    return {'gpu_model': model, 'driver': driver, 'count': 0, 'uuids': []}


def gpu_facts():
    out = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,uuid', '--format=csv,noheader'],
                         stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
    rows = [[x.strip() for x in line.split(',')] for line in out.strip().splitlines() if line.strip()]
    names, drivers = {r[0] for r in rows}, {r[1] for r in rows}
    if len(names) != 1 or len(drivers) != 1:
        raise Mismatch('mixed GPUs or drivers in this container: %s %s' % (sorted(names), sorted(drivers)))
    return {'gpu_model': rows[0][0], 'driver': rows[0][1], 'count': len(rows), 'uuids': [r[2] for r in rows]}


def bench_world_size():
    return int(json.loads((WS / 'bench/bench.json').read_text())['world_size'])


def keyinfo(task, shipped_hash=None):
    return {'task': task, 'judge_hash': shipped_hash or judge_hash(), 'world_size': bench_world_size(),
            'prep_version': prep_version()}


# ------------------------------------------------------------------ context for the task modules

class Ctx:
    def __init__(self, task, mod, pack=None, meta=None):
        self.task, self.mod, self.pack, self.meta = task, mod, pack, meta or {}
        self.ws = WS
        self.started = time.time()
        self.report = {'task': task, 'steps': {}}
        self.b = None
        self.devices = None
        self.out = WS / 'runs' / 'svc-pack'

    def run(self, *args, check=True, env=None, cwd=None):
        args = [str(a) for a in args]
        log('run: %s' % ' '.join(args))
        return subprocess.run([sys.executable] + args, cwd=str(cwd or WS), check=check,
                              env=dict(os.environ, **(env or {})))

    def json(self, rel):
        return json.loads((WS / rel).read_text())

    def step(self, name, **fields):
        self.report['steps'][name] = dict(fields, t=round(time.time() - self.started, 1))
        log('%s: %s' % (name, json.dumps(fields, default=str)[:600]))


def band_check(ctx, name, value, pack_value, band, lower_is_faster=True):
    """One reference timing against the pack's frozen one: |value / pack - 1| <= band."""
    if not value or not pack_value:
        raise Mismatch('%s: no timing value (here %r, pack %r)' % (name, value, pack_value))
    rel = value / pack_value - 1.0
    ok = abs(rel) <= band
    ctx.step('timing', value=value, pack_value=pack_value, rel=round(rel, 4), band=band, ok=ok, what=name)
    if not ok:
        raise Mismatch('reference timing %s here %.6g vs pack %.6g (%+.1f%%), outside the band +-%.0f%%'
                       % (name, value, pack_value, 100 * rel, 100 * band))
    return rel


# ------------------------------------------------------------------ commands

def cmd_keyinfo(a):
    print(json.dumps(keyinfo(a.task)))


def cmd_pristine(a):
    mod = load_task(a.task)
    roots = list(getattr(mod, 'ROOTS', DEFAULT_ROOTS))
    data = scan(roots)
    data['roots'] = roots
    data['keyinfo'] = keyinfo(a.task)
    Path(a.out).write_text(json.dumps(data) + '\n')
    log('pristine: %d files under %s' % (len(data['files']), roots))


def cmd_collect(a):
    mod = load_task(a.task)
    before = json.loads(Path(a.pristine).read_text())
    roots = before['roots']
    after = scan(roots)
    out = Path(a.out)
    files_dir = out / 'files'
    files_dir.mkdir(parents=True, exist_ok=True)
    manifest, skipped = {}, []
    for p, e in sorted(after['files'].items()):
        old = before['files'].get(p)
        if old == e:
            continue
        if e['type'] == 'file' and p.startswith(str(WS / 'runs') + '/') and e['size'] > RUNS_FILE_CAP:
            skipped.append({'path': p, 'size': e['size']})
            continue
        manifest[p] = e
        dest = files_dir / p.lstrip('/')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if e['type'] == 'link':
            os.symlink(e['target'], str(dest))
        else:
            shutil.copyfile(p, str(dest))
            if svckey.sha256_file(str(dest)) != e['sha256']:
                raise SystemExit('collect: %s changed while copying' % p)
    deleted = sorted(p for p in before['files'] if p not in after['files'])
    deleted_dirs = sorted(d for d in before['dirs'] if d not in set(after['dirs']))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    ready = None
    if (WS / 'task/ready.json').is_file():
        ready = json.loads((WS / 'task/ready.json').read_text())
    collect = {'roots': roots, 'files': len(manifest), 'bytes': sum(e.get('size', 0) for e in manifest.values()),
               'deleted': deleted, 'deleted_dirs': deleted_dirs, 'skipped': skipped, 'ready': ready,
               'keyinfo': before['keyinfo']}
    (out / 'collect.json').write_text(json.dumps(collect, indent=1) + '\n')
    log('collected %d files (%.1f MB), %d deleted, %d skipped' % (len(manifest), collect['bytes'] / 1e6,
                                                                   len(deleted), len(skipped)))


def check_integrity(pack):
    meta = json.loads((pack / 'pack.json').read_text())
    manifest_bytes = (pack / 'manifest.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != meta.get('content_sha256'):
        raise Mismatch('manifest.json does not match pack.json content_sha256')
    manifest = json.loads(manifest_bytes)
    for p, e in manifest.items():
        src = pack / 'files' / p.lstrip('/')
        if e['type'] == 'link':
            if not os.path.islink(str(src)) or os.readlink(str(src)) != e['target']:
                raise Mismatch('pack link %s differs from its manifest' % p)
        elif svckey.sha256_file(str(src)) != e['sha256']:
            raise Mismatch('pack file %s does not match its sha256' % p)
    collect = json.loads((pack / 'collect.json').read_text())
    return meta, manifest, collect


def restore(ctx, manifest, collect):
    machine_local = set(getattr(ctx.mod, 'MACHINE_LOCAL', ()))
    written = 0
    for p, e in sorted(manifest.items()):
        if p in machine_local:
            continue
        dest = Path(p)
        if not any(p.startswith(r.rstrip('/') + '/') for r in collect['roots']):
            raise Mismatch('pack path %s outside its roots' % p)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() or dest.exists():
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(str(dest))
            else:
                dest.unlink()
        if e['type'] == 'link':
            os.symlink(e['target'], p)
        else:
            shutil.copyfile(str(ctx.pack / 'files' / p.lstrip('/')), p)
            os.chmod(p, e['mode'])
            os.chown(p, e['uid'], e['gid'])
            if svckey.sha256_file(p) != e['sha256']:
                raise Mismatch('restored %s differs from the pack' % p)
        written += 1
    removed = 0
    for p in collect.get('deleted', []):
        if os.path.lexists(p):
            os.unlink(p)
            removed += 1
    for d in sorted(collect.get('deleted_dirs', []), key=len, reverse=True):
        if os.path.isdir(d) and not os.path.islink(d):
            shutil.rmtree(d, ignore_errors=True)
    ctx.step('restore', files=written, machine_local=sorted(machine_local & set(manifest)), deleted=removed)


def machine_local_equal(ctx, manifest, rel, ignore_keys=()):
    """A pack file the machine computes itself (MACHINE_LOCAL): the machine's version must equal the pack's, apart
    from ignore_keys of a JSON object."""
    p = str(WS / rel)
    if p not in manifest:
        return
    here = Path(p).read_bytes()
    packed = (ctx.pack / 'files' / p.lstrip('/')).read_bytes()
    if ignore_keys:
        a, b = json.loads(here), json.loads(packed)
        for k in ignore_keys:
            a.pop(k, None)
            b.pop(k, None)
        same = a == b
    else:
        same = here == packed
    ctx.step('machine_local %s' % rel, same=same, ignored=list(ignore_keys))
    if not same:
        raise Mismatch('%s computed here differs from the pack%s' % (
            rel, ' (apart from %s)' % list(ignore_keys) if ignore_keys else ''))


def cmd_verify(a):
    report_path = Path(a.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    mod = load_task(a.task)
    ctx = Ctx(a.task, mod)
    code = 0
    try:
        if not a.pack or not (Path(a.pack) / 'pack.json').is_file():
            ctx.report.update(status='no-pack', reason='no pack mounted at %s' % a.pack)
            code = EXIT_NOPACK
            return
        ctx.pack = Path(a.pack)
        shipped = judge_hash()          # before anything is written into the workspace
        meta, manifest, collect = check_integrity(ctx.pack)
        ctx.meta = meta
        ctx.step('integrity', files=len(manifest), content_sha256=meta['content_sha256'])
        gpu = host_gpu_facts() if bench_world_size() == 0 else gpu_facts()
        fields = dict(keyinfo(a.task, shipped), image_digest=os.environ.get('RBB_SVC_IMAGE_DIGEST', ''),
                      gpu_model=gpu['gpu_model'], driver=gpu['driver'])
        key = svckey.pack_key(**fields)
        ctx.report.update(key=key, key_fields=fields, gpu_uuids=gpu['uuids'])
        want = {f: meta['key_fields'].get(f) for f in svckey.FIELDS}
        differ = {f: (want[f], fields[f]) for f in svckey.FIELDS if str(want[f]) != str(fields[f])}
        if differ or key != meta['key']:
            raise Mismatch('pack key differs (pack, here): %s' % differ)
        expected = os.environ.get('RBB_SVC_KEY')
        if expected and expected != key:
            raise Mismatch('the host expected key %s, this container computes %s' % (expected, key))
        if gpu['count'] != fields['world_size']:
            raise Mismatch('%d GPUs visible, the task runs on %d' % (gpu['count'], fields['world_size']))
        ctx.step('key', key=key)
        if os.environ.get('RESEARCHBOOST_SMOKE') == '1':
            raise Mismatch('RESEARCHBOOST_SMOKE=1: packs are for full trials only')
        ctx.b = import_bootstrap()
        mod.prepare(ctx)
        restore(ctx, manifest, collect)
        if hasattr(mod, 'after_restore'):
            mod.after_restore(ctx, manifest)
        ctx.out.mkdir(parents=True, exist_ok=True)
        mod.verify_record(ctx)
        mod.verify_timing(ctx)
        mod.finish(ctx)
        ctx.report.update(status='ready', seconds=round(time.time() - ctx.started))
    except Mismatch as exc:
        ctx.report.update(status='mismatch', reason=str(exc))
        log('PACK DOES NOT VERIFY: %s' % exc)
        code = EXIT_MISMATCH
    except BaseException as exc:  # noqa: BLE001 - reported, then the container stops
        ctx.report.update(status='error', reason='%s: %s' % (type(exc).__name__, exc),
                          traceback=traceback.format_exc()[-4000:])
        log('svcpack error: %s' % traceback.format_exc())
        code = 1
    finally:
        ctx.report['finished'] = time.strftime('%F %T %Z')
        report_path.write_text(json.dumps(ctx.report, indent=1, default=str) + '\n')
        os.chmod(str(report_path), 0o644)
        sys.exit(code)


def write_ready(ctx, devices, extra=None):
    """task/ready.json in the fields every task's verify.py reads (mode, test_world_size) and the tasks' own
    bookkeeping (gpus, world_size, preparation), plus the svc block."""
    data = {'status': 'ready', 'mode': 'pack', 'gpus': devices, 'world_size': len(devices), 'test_world_size': None,
            'preparation': {'mode': 'svc-pack', 'seconds': round(time.time() - ctx.started)},
            'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'svc': {'key': ctx.report.get('key'), 'pack_made_by': ctx.meta.get('made_by'),
                    'record': ctx.report['steps'].get('record'), 'timing': ctx.report['steps'].get('timing')}}
    if extra:
        data.update(extra)
    (WS / 'task/ready.json').write_text(json.dumps(data, indent=1) + '\n')
    ctx.step('ready', mode='pack')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')
    sub.required = True
    k = sub.add_parser('keyinfo')
    k.add_argument('--task', required=True)
    p = sub.add_parser('pristine')
    p.add_argument('--task', required=True)
    p.add_argument('--out', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--task', required=True)
    c.add_argument('--pristine', required=True)
    c.add_argument('--out', required=True)
    v = sub.add_parser('verify')
    v.add_argument('--task', required=True)
    v.add_argument('--pack', default='/opt/rbb-svc/pack')
    v.add_argument('--report', required=True)
    a = ap.parse_args()
    {'keyinfo': cmd_keyinfo, 'pristine': cmd_pristine, 'collect': cmd_collect, 'verify': cmd_verify}[a.cmd](a)


if __name__ == '__main__':
    main()
