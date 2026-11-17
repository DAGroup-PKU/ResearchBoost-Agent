"""Preparation packs: the variance is shipped, each trial machine measures its own reference values.

A pack is the output of one full startup preparation, recorded by the bench maintainers once per task, world size and
GPU type: reference records, calibration thresholds, reference trajectories with their frozen spread delta,
formulation.json and the acceptance controls' evidence. It is part of the repository, tasks/<task>/prep-pack/ws<world
size>/, which the task's Compose file mounts read-only at /opt/researchboost/prep-pack/:

    pack.json       task, world size, GPU name and count, the judge workspace hash, PREP_VERSION, the image's
                    installed-packages hash, the upstream commit, provenance
    manifest.json   {workspace path: {sha256, size, mode}} of every file under files/
    files/...       what preparation wrote into the judge workspace
    logs/...        the preparation log of the recording

At startup (bootstrap.py), a pack is used when task, world size, judge workspace hash and PREP_VERSION match; its
files are checked against the manifest and restored with root ownership. The machine then always measures its own
reference values (the reference records twice, the timing baseline, the reference's quick and full checks). When its
records are bitwise identical to the pack's, the pack's thresholds, trajectories, delta and control evidence stand
('pack-match'); otherwise the thresholds and the reference trajectories of the candidate seed schedule are recorded
on this machine, delta stays the pack's, and the correctness controls run again ('pack-local-trajectories'). No
matching pack: the full preparation ('full'). scripts/make_prep_pack.py turns a prepared workspace into a pack.

Identical copies live in the environments of the tasks that use packs (tests/test_prep_pack.py keeps them in sync).
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

PACK = 'pack.json'
MANIFEST = 'manifest.json'
PACKS = Path('/opt/researchboost/prep-pack')
# Configuration bootstrap computes itself before a pack is considered; a pack must agree with it, and it is never
# taken from the pack.
LOCAL_CONFIG = ('bench/bench.json', 'bench/traj/spec.json')


def log(message):
    print('[prep-pack %s] %s' % (datetime.now().strftime('%H:%M:%S'), message), flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest(root, skip=()):
    """{relative path: sha256, size, mode} of every regular file under root, without bytecode caches and without the
    top-level entries in skip."""
    root = Path(root)
    out = {}
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if '__pycache__' in rel.parts or rel.parts[0] in skip or path.is_symlink() or not path.is_file():
            continue
        out[rel.as_posix()] = {'sha256': sha256_file(path), 'size': path.stat().st_size,
                               'mode': stat.S_IMODE(path.stat().st_mode)}
    return out


def workspace_hash(files):
    """The judge workspace's identity from its manifest, as the image ships it: every file except the upstream
    checkout under targets/ (it enters packs as its commit)."""
    digest = hashlib.sha256()
    for rel in sorted(files):
        if not rel.startswith('targets/'):
            digest.update(rel.encode() + b'\0' + files[rel]['sha256'].encode() + b'\n')
    return digest.hexdigest()


def pack_dir(world_size, packs=PACKS):
    return Path(packs) / ('ws%d' % world_size)


def find(task, world_size, workspace, prep_version, packs=PACKS):
    """(pack directory, pack.json) when a usable pack exists, else (None, reason)."""
    root = pack_dir(world_size, packs)
    try:
        meta = json.loads((root / PACK).read_text())
    except (OSError, ValueError):
        return None, 'no pack at %s' % root
    want = {'task': task, 'world_size': world_size, 'workspace_hash': workspace, 'prep_version': prep_version}
    differ = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if differ:
        return None, 'pack at %s does not match: %s' % (root, differ)
    return root, meta


def verify(root):
    """The pack's files against its manifest; returns the manifest or raises."""
    root = Path(root)
    files = json.loads((root / MANIFEST).read_text())
    on_disk = {p.relative_to(root / 'files').as_posix() for p in (root / 'files').rglob('*') if p.is_file()}
    if on_disk != set(files):
        raise RuntimeError('pack files and manifest differ: %s' % sorted(on_disk ^ set(files))[:5])
    for rel, want in files.items():
        if sha256_file(root / 'files' / rel) != want['sha256']:
            raise RuntimeError('pack file %s does not match its sha256' % rel)
    return files


def restore(root, workspace, owner=(0, 0)):
    """Check the pack and write its files into the workspace, owned by root with the recorded modes (bootstrap's
    own). The files of LOCAL_CONFIG must equal what this machine computed, or the pack does not fit; they are not
    overwritten. Returns the manifest; raises on any mismatch."""
    files = verify(root)
    if any(rel.startswith('targets/') for rel in files):
        raise RuntimeError('the pack writes into the upstream checkout under targets/')
    workspace = Path(workspace)
    for rel in LOCAL_CONFIG:
        if rel in files and (not (workspace / rel).is_file() or sha256_file(workspace / rel) != files[rel]['sha256']):
            raise RuntimeError('the pack was made with another %s than this machine computes' % rel)
    for rel, want in files.items():
        if rel in LOCAL_CONFIG:
            continue
        dest = workspace / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        shutil.copyfile(root / 'files' / rel, dest)
        os.chmod(dest, want['mode'])
        if os.geteuid() == 0:
            os.chown(dest, *owner)
        if sha256_file(dest) != want['sha256']:
            raise RuntimeError('restored %s differs from the pack' % rel)
    return files


def rebase_frozen(pack_frozen, local_frozen):
    """The statistical gate's frozen reference for a machine whose records differ from the pack's: the pack's spread
    (delta and sigma per endpoint, from its many reference seeds) with this machine's reference trajectories (the
    seeds of the candidate schedule), which the candidate's runs are paired with."""
    if pack_frozen['spec'] != local_frozen['spec']:
        raise ValueError('the local trajectories were recorded with another spec than the pack')
    out = dict(pack_frozen)
    out['reference_seeds'] = local_frozen['reference_seeds']
    out['reference_values'] = local_frozen['reference_values']
    out['delta_from'] = {'source': 'preparation pack', 'reference_seeds': pack_frozen['reference_seeds']}
    out['local_endpoint_stats'] = local_frozen['endpoints']
    return out


def make(workspace, pristine, out, info, logs=()):
    """Write a pack: the files of the prepared workspace that differ from the image's (pristine) workspace."""
    # The upstream checkout under targets/ is not preparation output (git rewrites its index on a stat refresh).
    after, before = manifest(workspace, skip=('targets',)), manifest(pristine)
    changed = sorted(rel for rel, m in after.items() if before.get(rel, {}).get('sha256') != m['sha256'])
    out = Path(out)
    tmp = out.with_name(out.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    for rel in changed:
        dest = tmp / 'files' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(workspace) / rel, dest)
    for source in logs:
        if Path(source).is_file():
            (tmp / 'logs').mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, tmp / 'logs' / Path(source).name)
    (tmp / MANIFEST).write_text(json.dumps({rel: after[rel] for rel in changed}, indent=1) + '\n')
    meta = dict(info, workspace_hash=workspace_hash(before), files=len(changed),
                created=datetime.now().isoformat(timespec='seconds'))
    (tmp / PACK).write_text(json.dumps(meta, indent=1) + '\n')
    verify(tmp)
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return meta
