"""The statistical gate on short trajectories (standard library only).

A trajectory is one deterministic run of the recipe with seed S: global seed 4396 + S, and the loaded weights of the
model and its EMA multiplied by (1 + 2^-7 z), z ~ N(0, 1) per parameter from a generator seeded with (S, its name); the
data stream is the recipe's (identical for every seed). It records several series (bench/bagel_entry.py): per step the
global mean CE and MSE and the gradient norm before clipping; per block of 10 steps the L2 norm of the change of the
model's ViT, language-model and other parameters and of the EMA. An endpoint is the mean of one series over a window
of its indices (bench.json trajectory.endpoints; the gradient norm through its natural log).

The reference's seed-to-seed standard deviation sigma of every endpoint, measured on its own seeds before any
candidate is seen, is frozen (bench/traj/frozen.json); the bound is delta = k x sigma (k = 2).

A candidate runs the same seeds; per endpoint the paired differences d_i = candidate - reference give mean m and a
paired t-test p-value:
  deviates  |m| > delta and p < alpha       (a real, practically relevant shift)
  within    |m| <= delta
  unclear   |m| > delta but not significant (too few seeds to tell)
Overall: 'different' if any endpoint deviates, 'equivalent' if all are within, otherwise 'inconclusive' (run more
seeds). With final=True an unclear endpoint counts as deviating: the point estimate is beyond delta.

The bench's gate 3 uses bound() instead: one seed, each endpoint's bias |d| against k x sigma (k = 3, bench.json
trajectory.sigma_k), sigma the frozen sigma_seed. compare() is kept for analyses over several seeds."""
import math
import statistics

from stats import t_cdf


def endpoint_values(run, endpoints):
    """run: {series: [value per index]}; returns {endpoint: mean over its window}."""
    out = {}
    for e in endpoints:
        values = run.get(e['series'])
        a, b = e['steps']
        if values is None or len(values) < b:
            raise ValueError(f"endpoint {e['name']}: series {e['series']} has {len(values or [])} values, {b} needed")
        window = values[a:b]
        if any(v is None or not math.isfinite(v) for v in window):
            raise ValueError(f"endpoint {e['name']}: a missing or non-finite value in {e['series']}[{a}:{b}]")
        if e.get('log'):
            if any(v <= 0 for v in window):
                raise ValueError(f"endpoint {e['name']}: a value <= 0 in {e['series']}[{a}:{b}] (log taken)")
            window = [math.log(v) for v in window]
        out[e['name']] = statistics.fmean(window)
    return out


def freeze(runs, spec, k=2.0):
    """runs: {seed: {series: [values]}} of the reference."""
    if len(runs) < 3:
        raise ValueError('need at least three reference seeds to freeze delta')
    values = {seed: endpoint_values(run, spec['endpoints']) for seed, run in sorted(runs.items())}
    endpoints = {}
    for e in spec['endpoints']:
        xs = [v[e['name']] for v in values.values()]
        sigma = statistics.stdev(xs)
        if not sigma > 0:
            raise ValueError(f"endpoint {e['name']}: the reference seeds do not spread (sigma {sigma})")
        endpoints[e['name']] = {'reference_mean': statistics.fmean(xs), 'sigma_seed': sigma, 'delta': k * sigma}
    return {'schema': 'bagel-sft-trajectory-frozen@1', 'spec': spec, 'k': k, 'reference_seeds': sorted(runs),
            'reference_values': {str(s): v for s, v in values.items()}, 'endpoints': endpoints}


def _paired(diffs):
    n = len(diffs)
    mean, sd = statistics.fmean(diffs), statistics.stdev(diffs)
    if sd == 0:
        return mean, sd, (0.0 if mean != 0 else 1.0)
    t = abs(mean) / (sd / math.sqrt(n))
    return mean, sd, 2 * (1 - t_cdf(t, n - 1))


def compare(frozen, cand_runs, alpha=0.05, final=False):
    spec = frozen['spec']
    ref_values = {int(s): v for s, v in frozen['reference_values'].items()}
    seeds = sorted(set(cand_runs) & set(ref_values))
    report = {'seeds': seeds, 'alpha': alpha, 'final': final, 'endpoints': {}, 'verdict': 'inconclusive'}
    if len(seeds) < 2:
        return dict(report, verdict='error', error=f'need at least two paired seeds, have {seeds}')
    try:
        cand_values = {seed: endpoint_values(cand_runs[seed], spec['endpoints']) for seed in seeds}
    except ValueError as exc:
        return dict(report, verdict='different', error=f'candidate trajectory: {exc}')
    states = []
    for e in spec['endpoints']:
        n = e['name']
        delta, sigma = frozen['endpoints'][n]['delta'], frozen['endpoints'][n]['sigma_seed']
        diffs = [cand_values[s][n] - ref_values[s][n] for s in seeds]
        mean, sd, pval = _paired(diffs)
        if abs(mean) <= delta:
            state = 'within'
        elif pval < alpha or final:
            state = 'deviates'
        else:
            state = 'unclear'
        report['endpoints'][n] = {'state': state, 'mean': mean, 'sd': sd, 'p_paired_t': pval, 'delta': delta,
                                  'mean_in_sigma': mean / sigma, 'diffs': diffs}
        states.append(state)
    if any(x == 'deviates' for x in states):
        report['verdict'] = 'different'
    elif all(x == 'within' for x in states):
        report['verdict'] = 'equivalent'
    return report


def bound(frozen, cand_runs, k=3.0):
    """Gate 3 of every check (the agent's, check --n1 and the final evaluation's): the candidate's trajectory on the
    first frozen reference seed. Per endpoint the paired difference d = candidate - reference (the mean over the seeds
    run, by default one) against the bias bound k x sigma_seed, sigma_seed the reference's own seed-to-seed standard
    deviation frozen before any candidate (default k = 3): within |d| <= k x sigma, beyond otherwise. All within:
    'equivalent' (PASS, N1); any beyond, or a missing or non-finite candidate value: 'different' (INCORRECT). No test
    and no further seeds here: bench.py's statistical() measures once more on the next frozen seed when this says
    'different' (bench.json trajectory.retest_seeds)."""
    spec = frozen['spec']
    ref_values = {int(s): v for s, v in frozen['reference_values'].items()}
    seeds = sorted(set(cand_runs) & set(ref_values))
    report = {'seeds': seeds, 'sigma_k': k, 'endpoints': {}, 'verdict': 'error'}
    if not seeds:
        return dict(report, error='no candidate trajectory on a reference seed')
    try:
        cand_values = {seed: endpoint_values(cand_runs[seed], spec['endpoints']) for seed in seeds}
    except ValueError as exc:
        return dict(report, verdict='different', error=f'candidate trajectory: {exc}')
    for e in spec['endpoints']:
        n = e['name']
        sigma = frozen['endpoints'][n]['sigma_seed']
        limit = k * sigma
        diffs = [cand_values[s][n] - ref_values[s][n] for s in seeds]
        mean = statistics.fmean(diffs)
        report['endpoints'][n] = {'state': 'within' if abs(mean) <= limit else 'beyond', 'mean': mean,
                                  'sigma_seed': sigma, 'bound': limit, 'mean_in_sigma': mean / sigma, 'diffs': diffs}
    report['verdict'] = ('equivalent' if all(e['state'] == 'within' for e in report['endpoints'].values())
                         else 'different')
    return report
