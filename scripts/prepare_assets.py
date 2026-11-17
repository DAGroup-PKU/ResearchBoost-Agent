"""Download and verify a task's data and weights into <assets root>/<task> (optional prefetch; CPU only).

Every task's Compose file runs the same preparation in its `assets` service before a trial starts, so this script
is only needed to fill the cache ahead of time or to check it. It runs the task's own asset script
(tasks/<task>/environment/asset-prep/scripts/), which pins every file by revision and checksum.

    python scripts/prepare_assets.py --task llava --assets-root /data/researchboost/assets
    python scripts/prepare_assets.py --all                       # RESEARCHBOOST_ASSETS_ROOT from the environment
    python scripts/prepare_assets.py --all --offline             # check what is there, no network

Downloads honour HF_ENDPOINT (a Hugging Face mirror) and the usual proxy variables.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    'bagel_sft': 'tasks/bagel_sft/environment/asset-prep/scripts/prepare_bagel_assets.py',
    'gaussian_splatting': 'tasks/gaussian_splatting/environment/asset-prep/scripts/prepare_gs_assets.py',
    'llava': 'tasks/llava/environment/asset-prep/scripts/prepare_assets.py',
    'yoco': 'tasks/yoco/environment/asset-prep/scripts/prepare_yoco_assets.py',
}
# nanochat, slime_grpo and slime_opd have shell scripts in tasks/<task>/environment/asset-prep/; r3 is provisioned by
# hand (tasks/<task>/README.md).


def command(task, assets_root, offline=False, extra=()):
    return [sys.executable, '-I', str(ROOT / SCRIPTS[task]), '--assets-dir', str(Path(assets_root) / task),
            *(['--offline'] if offline else []), *extra]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument('--task', choices=sorted(SCRIPTS))
    which.add_argument('--all', action='store_true')
    parser.add_argument('--assets-root', default=os.environ.get('RESEARCHBOOST_ASSETS_ROOT'),
                        help='default: $RESEARCHBOOST_ASSETS_ROOT')
    parser.add_argument('--offline', action='store_true', help='check existing files without network')
    parser.add_argument('extra', nargs='*', help="after --: arguments for the task's own script (--task only)")
    args = parser.parse_args()
    if not args.assets_root:
        parser.error('give --assets-root or set RESEARCHBOOST_ASSETS_ROOT')
    if args.extra and args.all:
        parser.error('extra arguments need --task')
    for task in sorted(SCRIPTS) if args.all else [args.task]:
        print('== %s -> %s' % (task, Path(args.assets_root) / task), flush=True)
        subprocess.run(command(task, args.assets_root, args.offline, args.extra), check=True)


if __name__ == '__main__':
    main()
