"""False-reject rate of the statistical gate (gate 3) for a legal rewrite that is not bitwise identical.

Gate 3 as the bench runs it: sigma_hat is the reference's seed-to-seed standard deviation over n reference seeds
(frozen at startup); the candidate runs the first of those seeds, and each endpoint's bias d = candidate - reference
(same seed) must satisfy |d| <= k * sigma_hat. A legal rewrite drifts away from the reference's trajectory: on the same
seed its endpoint correlates with the reference's by rho (1 = identical, 0 = as far apart as two seeds). The model
draws n reference seeds r_0..r_(n-1) ~ N(0, 1), the candidate c = rho r_0 + sqrt(1 - rho^2) e, and counts
|c - r_0| > k * std(r). The compared seed is one of the n (as in the bench); --independent-sigma takes sigma from n
other seeds instead (closed form: d / sigma_hat = sqrt(2 (1 - rho)) t_(n-1)). Endpoints are treated as independent,
the worst case for the overall rate.

  python frr.py --rho 0 0.5 0.8 --endpoints 14           the table of the vetting protocol
  python frr.py --rho 0.5 --endpoints 6 --target 0.01      also the k that keeps the overall rate at 1%
  python frr.py --measured rewrites.json                   rho and the rate from measured legal rewrites

rewrites.json: {"seeds": 6, "k": 3, "runs": {"<rewrite>": [[z per endpoint], ...]}}, where z = d / sigma_hat as the
judge reports it (the bias in units of the frozen sigma), one list per run.
"""
import argparse
import json
import math

import numpy as np

DRAWS = 400_000


def _betacf(a, b, x):
    """Continued fraction of the regularized incomplete beta function (Numerical Recipes 6.4)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / (c if abs(c) > tiny else tiny)
            h *= d * c
        if abs(d * c - 1.0) < 1e-14:
            break
    return h


def t_two_sided(t, nu):
    """P(|T| > t) for Student's t with nu degrees of freedom (regularized incomplete beta)."""
    x = nu / (nu + t * t)
    a, b = nu / 2.0, 0.5
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def ratios(rho, seeds, independent=False, draws=DRAWS, seed=0):
    """|d| / sigma_hat for `draws` simulated gate-3 comparisons (fixed random numbers: monotone in rho and k)."""
    rng = np.random.default_rng(seed)
    ref = rng.standard_normal((draws, seeds))
    sigma = ref.std(axis=1, ddof=1)
    r0 = rng.standard_normal(draws) if independent else ref[:, 0]
    cand = rho * r0 + math.sqrt(max(0.0, 1.0 - rho * rho)) * rng.standard_normal(draws)
    return np.abs(cand - r0) / sigma


def per_endpoint(rho, seeds=6, k=3.0, independent=False):
    if rho >= 1.0:
        return 0.0
    if independent:
        return t_two_sided(k / math.sqrt(2.0 * (1.0 - rho)), seeds - 1)
    return float((ratios(rho, seeds) > k).mean())


def overall(p, endpoints):
    return 1.0 - (1.0 - p) ** endpoints


def k_for_target(rho, seeds, endpoints, target, independent=False):
    per = 1.0 - (1.0 - target) ** (1.0 / endpoints)
    return float(np.quantile(ratios(rho, seeds, independent), 1.0 - per))


def rho_from_second_moment(m2, seeds, independent=False):
    """Invert E[z^2](rho) by bisection (z = d / sigma_hat)."""
    lo, hi = -1.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if float((ratios(mid, seeds, independent, draws=100_000) ** 2).mean()) > m2:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def from_measured(spec, independent=False):
    seeds, k = spec.get("seeds", 6), spec.get("k", 3.0)
    runs = [z for rewrite in spec["runs"].values() for z in rewrite]
    n_end = len(runs[0])
    assert all(len(z) == n_end for z in runs), "every run lists the same endpoints"
    # a legal rewrite has no mean bias: the raw second moment of z per endpoint fixes rho
    rhos = [rho_from_second_moment(sum(run[j] ** 2 for run in runs) / len(runs), seeds, independent)
            for j in range(n_end)]
    ps = [per_endpoint(r, seeds, k, independent) for r in rhos]
    return {"runs": len(runs), "endpoints": n_end, "rho": rhos, "per_endpoint": ps,
            "predicted_overall": 1.0 - math.prod(1.0 - p for p in ps),
            "observed_rejected": sum(1 for z in runs if max(abs(v) for v in z) > k),
            "max_abs_z": max(max(abs(v) for v in z) for z in runs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.5, 0.8])
    ap.add_argument("--seeds", type=int, default=6, help="reference seeds behind sigma (default 6)")
    ap.add_argument("--k", type=float, default=3.0, help="bound in sigma (default 3)")
    ap.add_argument("--endpoints", type=int, default=14)
    ap.add_argument("--target", type=float, help="also print the k that keeps the overall rate at this value")
    ap.add_argument("--independent-sigma", action="store_true",
                    help="sigma from seeds other than the compared one (closed form)")
    ap.add_argument("--measured", help="JSON of measured legal-rewrite z values (see the module docstring)")
    a = ap.parse_args()
    if a.measured:
        r = from_measured(json.load(open(a.measured)), a.independent_sigma)
        print("runs %d, endpoints %d, observed rejected %d, max |z| %.2f" %
              (r["runs"], r["endpoints"], r["observed_rejected"], r["max_abs_z"]))
        print("rho per endpoint: " + ", ".join("%.2f" % x for x in r["rho"]))
        print("predicted overall false-reject rate: %.1f%%" % (100 * r["predicted_overall"]))
        return
    print("seeds %d, k %.2f, endpoints %d%s" % (a.seeds, a.k, a.endpoints,
                                                 ", sigma from other seeds" if a.independent_sigma else ""))
    print("%6s %14s %14s" % ("rho", "per endpoint", "overall"))
    for rho in a.rho:
        p = per_endpoint(rho, a.seeds, a.k, a.independent_sigma)
        line = "%6.2f %13.1f%% %13.1f%%" % (rho, 100 * p, 100 * overall(p, a.endpoints))
        if a.target:
            line += "   k for %.1f%% overall: %.2f" % (
                100 * a.target, k_for_target(rho, a.seeds, a.endpoints, a.target, a.independent_sigma))
        print(line)


if __name__ == "__main__":
    main()
