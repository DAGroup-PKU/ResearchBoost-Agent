"""safepath.py: which directories and programs only root can change (protocol supplement C10). entry.sh runs it as
root, with an interpreter the agent cannot reach, before any of the submission is put back.

    safepath.py path PATH [PATH...]   the entries of the given PATH values (in order, each once) that root_only()
                                      accepts, joined with ':' on stdout
    safepath.py check FILE...         exit 0 if root_only() accepts every FILE; otherwise name the others, exit 1

root_only(p): p exists, and p and every directory on the way to it belong to root and are not writable by group or
others. Symbolic links are followed one component at a time: a link counts through the directory that holds it (only
whoever can write that directory can change the link), then through what it points to. A link that sits in a
directory the agent can write is therefore refused even when it points into /usr/bin.
Standard library only, Python 3.6 or newer.
"""
import os
import stat
import sys

MAX_LINKS = 40


def _root_only(st):
    return st.st_uid == 0 and not st.st_mode & 0o022


def root_only(path):
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    todo = path.split('/')[::-1]          # a stack: the next component last
    cur, links = '/', 0
    try:
        if not _root_only(os.stat('/')):
            return False
        while todo:
            part = todo.pop()
            if part in ('', '.'):
                continue
            if part == '..':              # cur has no links in it, so its parent is the real one
                cur = os.path.dirname(cur)
                continue
            nxt = os.path.join(cur, part)
            st = os.lstat(nxt)
            if stat.S_ISLNK(st.st_mode):
                links += 1
                if links > MAX_LINKS:
                    return False
                target = os.readlink(nxt)
                if target.startswith('/'):
                    cur = '/'
                todo.extend(target.split('/')[::-1])
                continue
            if not _root_only(st):
                return False
            cur = nxt
        return True
    except OSError:
        return False


def safe_path(*values):
    keep = []
    for value in values:
        for entry in value.split(os.pathsep):
            if entry and os.path.isabs(entry) and entry not in keep and os.path.isdir(entry) and root_only(entry):
                keep.append(entry)
    return os.pathsep.join(keep)


def main(argv):
    if len(argv) >= 2 and argv[0] == 'path':
        print(safe_path(*argv[1:]))
        return 0
    if len(argv) >= 2 and argv[0] == 'check':
        bad = [p for p in argv[1:] if not root_only(p)]
        for p in bad:
            print('safepath.py: %s: not root-only (it, or a directory on the way to it, is missing, not owned by root, '
                  'or writable by group or others)' % p, file=sys.stderr)
        return 1 if bad else 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
