"""Freeze task/timing_ref.json from one or more `bench.py baseline` outputs (python mkref.py <out> <baseline.json>...).

The bench uses `rng_fingerprints` (the per-update random-number stream of every rank; numerics, host-independent) and
`frozen`, this machine's reference timing: the mean update time of the baseline runs, per-window means and the
spread (max - min) / mean of the runs. `time` reports the speedup over it and whether it is beyond that spread; the
final check only rejects (INVALID) a check whose reference runs are more than timing.max_reference_drift off it, its
speedup is measured fresh (reference -> candidate -> reference). `informational` documents the baseline as measured
on the named host."""
import socket
import json
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
runs, fps = [], None
for src in sys.argv[2:]:
    b = json.loads(Path(src).read_text())
    assert b['rng_identical_across_runs'], f'{src}: the reference does not draw the same random numbers in every run'
    if fps is None:
        fps = b['rng_fingerprints']
    assert b['rng_fingerprints'] == fps, f'{src}: RNG stream differs from the first baseline'
    runs += [dict(r, source=src) for r in b['runs'].values() if r['status'] == 'ok']
tot = [r['total_s'] for r in runs]
ws = [r['window_s'] for r in runs]
med = statistics.median(tot)
window_s = [statistics.median(w[k] for w in ws) for k in range(len(ws[0]))]
mean_w = statistics.fmean(window_s)
info = {
    'host': socket.gethostname(), 'note': 'informational only, never used in a verdict',
    'sources': sys.argv[2:], 'runs_total_s': tot, 'total_s': round(med, 3),
    'window_s': [round(x, 3) for x in window_s],
    'window_shape': [round(x / mean_w, 4) for x in window_s],
    'run_window_shapes': [r.get('window_shape') for r in runs],
    's_per_update': round(med / runs[0]['iterations_timed'], 4),
    'iterations_timed': runs[0]['iterations_timed'],
    'spread_max_min_over_mean': round((max(tot) - min(tot)) / statistics.fmean(tot), 4),
    'peak_memory_allocated_gib': max(r['peak_memory_allocated_gib'] for r in runs),
    'startup_to_first_update_s': [r['startup_to_first_update_s'] for r in runs],
    'gpus': runs[0]['fingerprint']['gpus'],
    'versions': runs[0]['fingerprint']['versions'],
}
per = [r['s_per_update'] for r in runs]
mean = statistics.fmean(per)
frozen = {'s_per_update': round(mean, 5),
          'window_s': [round(statistics.fmean(w[k] for w in ws), 4) for k in range(len(ws[0]))],
          'spread': round((max(per) - min(per)) / mean, 4),
          'sigma_rel': round(statistics.stdev(per) / mean, 4) if len(per) > 1 else None,
          'runs': [{'s_per_update': r['s_per_update'], 'window_s': r['window_s']} for r in runs]}
ref = {'timing_plan': runs[0]['fingerprint']['timing_plan'], 'rng_fingerprints': fps, 'frozen': frozen,
       'informational': info}
out.write_text(json.dumps(ref, indent=1) + '\n')
print(json.dumps(info, indent=1))
