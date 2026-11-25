"""nanochat in pack mode (svcpack.py verify). ctx.b is the image's bootstrap.py.

The pack holds bench/records/ref.json, bench/data/probe.pt, bench/traj/{ref_runs,frozen}.json, task/formulation.json,
task/timing_ref.json and the control evidence. The final check times reference -> candidate -> reference itself and
reads no frozen timing; task/timing_ref.json is what the agent's own check compares with. Band: timing.spread_band
(0.05), the judge's own reference-vs-reference tolerance.
Verification: bench.py record of the reference once (FlashAttention-3 on every rank, as bootstrap requires) and
bench.py compare against bench/records/ref.json; one bench.py time of the reference within the band of the pack's
s_per_update.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

PYTHON = '/opt/nanochat-env/bin/python'


def prepare(ctx):
    _common.simple_prepare(ctx, ('task', 'bench/records', 'bench/data', 'bench/traj'))


def verify_record(ctx):
    from svcpack import Mismatch

    def fa3(rec):
        if rec.get('fa3') != ['True']:
            raise Mismatch('the reference did not run FlashAttention-3 on every rank here (%s)' % rec.get('fa3'))
    _common.record_and_compare(ctx, fa3)


def verify_timing(ctx):
    frozen = json.loads((ctx.ws / 'task/timing_ref.json').read_text())
    return _common.time_once(ctx, 's_per_update', frozen['s_per_update'], ctx.config['timing']['spread_band'])


def finish(ctx):
    _common.simple_finish(ctx)
