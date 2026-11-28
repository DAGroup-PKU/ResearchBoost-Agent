"""Startup check: the container holds no credentials.

The agent has internet access. What it must not reach is private code (the AI4AI-for-Anything GitHub organization),
and the control for that is that nothing in the container can authenticate anywhere: no tokens or passwords in the
environment, no credential helpers or credentials in git configuration, no credential files (.git-credentials,
.netrc, gh, docker, pip, Hugging Face tokens) and no SSH private keys or agent sockets. The entrypoint runs this
before preparation; any finding stops the container.

    python3 -I no_credentials.py            # exit 1 and list the findings, exit 0 when clean

Identical copies live in every task's environment/ (tests/test_credentials.py keeps them in sync).
"""
import os
from pathlib import Path
import re
import sys

ENV_NAMES = re.compile(r'TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|AUTH_SOCK|ASKPASS',
                       re.I)
HOME_FILES = ('.git-credentials', '.config/git/credentials', '.netrc', '_netrc', '.config/gh/hosts.yml',
              '.docker/config.json', '.pypirc', '.config/pip/pip.conf', '.pip/pip.conf', '.huggingface/token',
              '.cache/huggingface/token', '.config/hub', '.config/glab-cli/config.yml')
GIT_CONFIGS = ('.gitconfig', '.config/git/config')
SSH_PUBLIC = re.compile(r'^(known_hosts(\.old)?|config|authorized_keys|.*\.pub)$')
# a URL with a user (and possibly a token) before the host, e.g. https://x-access-token:...@github.com/...
EMBEDDED = re.compile(r'[a-z][a-z0-9+.-]*://[^/\s@]+@', re.I)
CREDENTIAL_SECTION = re.compile(r'^\s*\[\s*credential\b|^\s*(helper|askpass|sshcommand)\s*=', re.I | re.M)


def _homes(root):
    homes = {root / 'root'}
    if (root / 'home').is_dir():
        homes.update(p for p in (root / 'home').iterdir() if p.is_dir())
    return sorted(homes)


def _git_config_problems(path):
    try:
        text = path.read_text(errors='replace')
    except OSError:
        return []
    out = []
    if CREDENTIAL_SECTION.search(text):
        out.append('%s: credential helper or askpass setting' % path)
    if EMBEDDED.search(text):
        out.append('%s: URL with embedded user or token' % path)
    return out


def scan(root=Path('/'), environ=None):
    """Findings (strings) for the filesystem under `root` and the given environment; [] when clean."""
    root = Path(root)
    environ = os.environ if environ is None else environ
    found = ['environment variable %s' % name for name in sorted(environ) if ENV_NAMES.search(name)]
    found += ['environment variable %s holds a URL with embedded credentials' % name
              for name, value in sorted(environ.items()) if EMBEDDED.search(value or '')]
    configs = [root / 'etc/gitconfig']
    for home in _homes(root):
        configs += [home / name for name in GIT_CONFIGS]
        found += ['credential file %s' % (home / name) for name in HOME_FILES if (home / name).exists()]
        ssh = home / '.ssh'
        if ssh.is_dir():
            found += ['SSH key or agent file %s' % p for p in sorted(ssh.iterdir()) if not SSH_PUBLIC.match(p.name)]
    # git repositories the image ships: their remotes must not carry credentials either
    for base in (root / 'submission', root / 'opt/researchboost/ws/targets'):
        if base.is_dir():
            configs += sorted(base.glob('*/.git/config')) + sorted(base.glob('*/*/.git/config'))
    for config in configs:
        if config.is_file():
            found += _git_config_problems(config)
    hf_home = environ.get('HF_HOME')
    if hf_home:
        token = root / Path(hf_home).relative_to('/') if Path(hf_home).is_absolute() else root / hf_home
        if (token / 'token').exists():
            found.append('credential file %s' % (token / 'token'))
    return found


def main():
    found = scan()
    for line in found:
        print('credential found: %s' % line, file=sys.stderr)
    if found:
        sys.exit('The container must hold no credentials; remove the items above.')
    print('no credentials found')


if __name__ == '__main__':
    main()
