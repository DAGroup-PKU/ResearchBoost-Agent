"""The statistical gate on full training runs (standard library only).

A trajectory is one run of the recipe from scratch with seed S (random, numpy and torch seeded with S: the camera
order, the scene's shuffle and the densification's random draws follow from it; the rasterizer's backward adds its
own run-to-run variation). Its endpoints are the judge's evaluation of the trained Gaussians (bench/gs_eval.py): the
mean PSNR, SSIM and LPIPS over the test views, and log(number of Gaussians).

The reference's seed-to-seed standard deviation sigma_seed of every endpoint, measured on its own seeds before any
candidate is seen, is frozen (bench/traj/frozen.json; its delta is sigma_k x sigma_seed).

Gate 3 of every check (the agent's, check --n1 and the final evaluation's), bound(): the candidate's run on the first
frozen reference seed; per endpoint the paired difference d = candidate - reference against k x sigma_seed (k = 3):
within |d| <= k x sigma_seed, beyond otherwise. All within: 'equivalent' (PASS, N1); any beyond: 'different'
(INCORRECT). One seed, no test and no further seeds."""
import math
import statistics


def endpoint_values(result):
    """result: one entry of gs_eval.py's results."""
    return {'psnr': result['psnr'], 'ssim': result['ssim'], 'lpips': result['lpips'],
            'log_count': math.log(result['count'])}


def freeze(results, endpoints, k=3.0):
    """results: {seed: gs_eval result} of the reference."""
    if len(results) < 3:
        raise ValueError('need at least three reference seeds to freeze delta')
    values = {seed: endpoint_values(r) for seed, r in sorted(results.items())}
    out = {}
    for name in endpoints:
        xs = [v[name] for v in values.values()]
        sigma = statistics.stdev(xs)
        if not sigma > 0:
            raise ValueError(f'endpoint {name}: the reference seeds do not spread (sigma {sigma})')
        out[name] = {'reference_mean': statistics.fmean(xs), 'sigma_seed': sigma, 'delta': k * sigma}
    return {'schema': 'gaussian-splatting-trajectory-frozen@1', 'endpoint_names': list(endpoints), 'k': k,
            'reference_seeds': sorted(results), 'reference_values': {str(s): v for s, v in values.items()},
            'endpoints': out}


def bound(frozen, cand_results, k=3.0):
    """Gate 3: cand_results {seed: gs_eval result} on the first frozen reference seed(s); see the module docstring."""
    ref_values = {int(s): v for s, v in frozen['reference_values'].items()}
    seeds = sorted(set(cand_results) & set(ref_values))
    report = {'seeds': seeds, 'sigma_k': k, 'endpoints': {}, 'verdict': 'error'}
    if not seeds:
        return dict(report, error='no candidate run on a reference seed')
    cand_values = {seed: endpoint_values(cand_results[seed]) for seed in seeds}
    for n in frozen['endpoint_names']:
        sigma = frozen['endpoints'][n]['sigma_seed']
        limit = k * sigma
        diffs = [cand_values[s][n] - ref_values[s][n] for s in seeds]
        mean = statistics.fmean(diffs)
        report['endpoints'][n] = {'state': 'within' if abs(mean) <= limit else 'beyond', 'mean': mean,
                                  'sigma_seed': sigma, 'bound': limit, 'mean_in_sigma': mean / sigma,
                                  'diffs': diffs, 'candidate': [cand_values[s][n] for s in seeds]}
    report['verdict'] = ('equivalent' if all(e['state'] == 'within' for e in report['endpoints'].values())
                         else 'different')
    return report
