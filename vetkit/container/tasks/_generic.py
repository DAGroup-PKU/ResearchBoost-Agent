"""Any task with the standard bench interface, in pack mode (svcpack.py verify): for tasks that have no module of
their own. Configured by RBB_SVC_TASK_CONFIG (refpack.py --adapter generic sets it):
{"adapter": "generic", "generic": {"dirs": [...], "timing_file": "task/timing_ref.json", "timing_field": "s_per_step",
"band": null}}.

Standard bench interface (the one nanochat, r3 and bagel_sft have): bench/bench.json with `reference`,
`record.reference`, `gpu_lock` and `timing.spread_band`; `bench/bench.py record --tree T --out D --label L`,
`bench/bench.py compare A B` (exit 0 = identical) and `bench/bench.py time --tree T --out D` writing time.json with a
`status` and the timing field; the frozen reference timing in <timing_file> under the workspace.
The pack holds `dirs` (default task and bench/records) and the control evidence.
Verification: one record of the reference compared with the pack's record (exact), one timing of the reference within
the band (generic.band, else bench.json's timing.spread_band) of the frozen value.
"""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

CFG = (json.loads(os.environ.get('RBB_SVC_TASK_CONFIG') or '{}').get('generic') or {})
DIRS = tuple(CFG.get('dirs') or ('task', 'bench/records'))


def prepare(ctx):
    _common.simple_prepare(ctx, DIRS)


def verify_record(ctx):
    _common.record_and_compare(ctx)


def time_tree(ctx):
    """generic.time_tree_wrap: the task's `time --tree` takes a directory that HOLDS the source
    tree under this name (as a submission does), while `record` takes the reference tree itself; the reference is
    then timed through <out>/time-tree/<name> -> the reference (a symlink: bench.py resolves it, nothing is copied)."""
    name = CFG.get('time_tree_wrap')
    if not name:
        return None
    d = ctx.out / 'time-tree'
    shutil.rmtree(str(d), ignore_errors=True)
    d.mkdir(parents=True)
    (d / name).symlink_to((ctx.ws / ctx.config['reference']).resolve())
    return d


def verify_timing(ctx):
    field = CFG['timing_field']
    frozen = json.loads((ctx.ws / CFG['timing_file']).read_text())
    band = CFG.get('band') if CFG.get('band') is not None else ctx.config['timing']['spread_band']
    return _common.time_once(ctx, field, frozen[field], band, tree=time_tree(ctx))


def finish(ctx):
    _common.simple_finish(ctx)
