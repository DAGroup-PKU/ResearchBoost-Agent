"""restore_bundle.py --artifacts DIR --allow PATH... : put a submission bundle (Harbor's artifacts/ layout, as
a finished trial leaves it; mounted read-only) in place inside a fresh judge container, before the unchanged final
judge (tests/test.sh) runs. Run AS THE AGENT USER (entry.sh: runuser -u agent), with /usr/bin/python3 -I: whatever the
bundle holds can only land where the agent itself could write in its own container; root-owned paths (the judge
workspace, /tests, /logs/verifier) are out of reach by construction.

For each entry of artifacts/manifest.json with status ok (or empty) whose source is not under /logs: the recorded copy
at artifacts/<destination> goes over its source path, which must be one of --allow (the task.toml artifacts). A file
replaces the file (a symlink at the target is replaced, never written through). A directory becomes exactly the
bundle's tree, except that symlinks the image has there and the bundle lacks are kept (Harbor's artifact download
drops a virtualenv's interpreter links, e.g. candidate-env/bin/python -> /opt/<env>/bin/python). Symlinks in the
bundle are recreated verbatim (never followed); a directory symlink of the image whose target leaves the directory is
replaced by a real directory before anything is written below it. FIFOs, sockets and devices are refused; setuid,
setgid and sticky bits are dropped.

Prints a JSON report on stdout (entry.sh stores it as root); exit 0 ok, 3 the bundle is refused (not a judge result:
the candidate is rejected by refpack.py judge as 'bundle refused'), 1 crash.
"""
import argparse
import json
import os
import shutil
import stat
import sys


class Refused(Exception):
    pass


def inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root.rstrip('/') + '/')


def copy_file(src, dst, report):
    st = os.lstat(src)
    if not stat.S_ISREG(st.st_mode):
        raise Refused('%s: not a regular file' % src)
    if os.path.islink(dst) or os.path.isfile(dst):
        os.unlink(dst)
    elif os.path.isdir(dst):
        shutil.rmtree(dst)
    with open(src, 'rb') as a, open(dst, 'xb') as b:
        shutil.copyfileobj(a, b, 8 << 20)
    os.chmod(dst, stat.S_IMODE(st.st_mode) & 0o777)
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))
    report['files'] += 1
    report['bytes'] += st.st_size


def ensure_dir(path, root, report):
    if os.path.islink(path):
        if os.path.isdir(path) and inside(path, root):
            return                                  # e.g. lib64 -> lib inside a venv: write through it
        os.unlink(path)
        report['replaced_dir_links'].append(path)
    elif os.path.exists(path) and not os.path.isdir(path):
        os.unlink(path)
    if not os.path.isdir(path):
        os.mkdir(path, 0o755)


def restore_dir(staged, target, report):
    if os.path.islink(target):
        raise Refused('%s: the target directory is a symlink' % target)
    os.makedirs(target, exist_ok=True)
    # 1. what the bundle does not have goes, except the image's symlinks (interpreter links)
    for d, dirs, files in os.walk(target, topdown=False):
        rel = os.path.relpath(d, target)
        for name in files + dirs:
            path = os.path.join(d, name)
            if os.path.islink(path):
                continue
            if os.path.lexists(os.path.join(staged, rel, name)):
                continue
            if os.path.isdir(path):
                try:
                    os.rmdir(path)          # emptied bottom-up; a directory still holding image links stays
                except OSError:
                    continue
            else:
                os.unlink(path)
            report['removed'] += 1
    # 2. the bundle's tree over it
    for d, dirs, files in os.walk(staged):
        rel = os.path.relpath(d, staged)
        tdir = target if rel == '.' else os.path.join(target, rel)
        if rel != '.':
            ensure_dir(tdir, target, report)
        for name in dirs[:] + files:
            src, dst = os.path.join(d, name), os.path.join(tdir, name)
            st = os.lstat(src)
            if stat.S_ISLNK(st.st_mode):
                if name in dirs:
                    dirs.remove(name)
                if os.path.islink(dst) or os.path.isfile(dst):
                    os.unlink(dst)
                elif os.path.isdir(dst):
                    shutil.rmtree(dst)
                link = os.readlink(src)
                os.symlink(link, dst)
                report['links'] += 1
                full = link if os.path.isabs(link) else os.path.normpath(os.path.join(tdir, link))
                if not inside(full, target):
                    report['links_outside'].append([dst, link])
            elif stat.S_ISDIR(st.st_mode):
                continue
            elif stat.S_ISREG(st.st_mode):
                copy_file(src, dst, report)
            else:
                raise Refused('%s: special file (fifo, socket or device)' % src)
        for name in dirs:
            st = os.lstat(os.path.join(d, name))
            ensure_dir(os.path.join(tdir, name), target, report)
            os.chmod(os.path.join(tdir, name), stat.S_IMODE(st.st_mode) & 0o777 | 0o700)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifacts', required=True)
    ap.add_argument('--allow', nargs='+', required=True)
    a = ap.parse_args()
    report = {'entries': [], 'files': 0, 'bytes': 0, 'links': 0, 'removed': 0, 'links_outside': [],
              'replaced_dir_links': [], 'uid': os.getuid()}
    try:
        if os.getuid() == 0:
            raise Refused('restore_bundle.py must run as the agent user, not root')
        root = os.path.realpath(a.artifacts)
        manifest = json.load(open(os.path.join(root, 'manifest.json')))
        allow = {p.rstrip('/') for p in a.allow}
        seen = set()
        for e in manifest:
            src = str(e.get('source', '')).rstrip('/')
            if src.startswith('/logs') or e.get('status') not in ('ok', 'empty'):
                report['entries'].append({'source': src, 'skipped': e.get('status')})
                continue
            if src not in allow:
                raise Refused('artifact %s is not one of the task\'s artifacts %s' % (src, sorted(allow)))
            if src in seen:
                raise Refused('artifact %s listed twice' % src)
            seen.add(src)
            dest = str(e.get('destination', ''))
            rel = dest[len('artifacts/'):] if dest.startswith('artifacts/') else dest
            staged = os.path.normpath(os.path.join(root, rel))
            if not inside(staged, root) or os.path.islink(staged):
                raise Refused('artifact %s: its recorded copy %s leaves the bundle or is a link' % (src, dest))
            if e.get('status') == 'empty':
                os.makedirs(src, exist_ok=True)
                report['entries'].append({'source': src, 'type': 'empty'})
                continue
            if e.get('type') == 'directory':
                if not os.path.isdir(staged):
                    raise Refused('artifact %s: %s is not a directory' % (src, dest))
                restore_dir(staged, src, report)
            else:
                copy_file(staged, src, report)
            report['entries'].append({'source': src, 'type': e.get('type')})
        missing = sorted(allow - seen)
        if missing:
            raise Refused('the bundle lacks artifacts %s' % missing)
        report['status'] = 'ok'
        print(json.dumps(report))
        return 0
    except Refused as exc:
        report.update(status='refused', reason=str(exc))
        print(json.dumps(report))
        return 3


if __name__ == '__main__':
    sys.exit(main())
