"""CPU tests of the vetting tools: python -m pytest test_vet.py (peaks.py needs torch; skipped without it)."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bound  # noqa: E402
import frr  # noqa: E402
import scorecard  # noqa: E402


def test_frr_closed_form_matches_student_t():
    # sigma from other seeds: d / sigma_hat = sqrt(2 (1 - rho)) t_5; P(|t_5| > 3) = 0.0301
    assert abs(frr.t_two_sided(3.0, 5) - 0.0301) < 2e-4
    assert abs(frr.per_endpoint(0.5, 6, 3.0, independent=True) - 0.0301) < 2e-4


def test_frr_bench_model_table():
    # the compared seed is one of the six behind sigma (the bench's gate 3)
    expect = {0.0: (0.066, 0.62), 0.5: (0.024, 0.29), 0.8: (0.0044, 0.061)}
    for rho, (p_end, p_all) in expect.items():
        p = frr.per_endpoint(rho, 6, 3.0)
        assert abs(p - p_end) < 0.003, (rho, p)
        assert abs(frr.overall(p, 14) - p_all) < 0.02, (rho, frr.overall(p, 14))


def test_frr_k_for_target_inverts():
    k = frr.k_for_target(0.8, 6, 6, 0.01)
    assert abs(frr.overall(frr.per_endpoint(0.8, 6, k), 6) - 0.01) < 0.002


def test_frr_measured_recovers_rho():
    import numpy as np
    rng = np.random.default_rng(1)
    runs = []
    for _ in range(60):
        z = []
        for _ in range(4):
            r = rng.standard_normal(6)
            c = 0.8 * r[0] + 0.6 * rng.standard_normal()
            z.append(float((c - r[0]) / r.std(ddof=1)))
        runs.append(z)
    out = frr.from_measured({"seeds": 6, "k": 3, "runs": {"x": runs}})
    assert all(abs(r - 0.8) < 0.12 for r in out["rho"]), out["rho"]


def test_bound_example():
    spec = json.load(open(HERE / "example/stages.example.json"))
    peaks = bound.Peaks(json.load(open(HERE / "example/peaks.example.json")))
    ds = bound.DATASHEETS[spec["datasheet"]]
    rows = [bound.stage_bounds(st, ds, peaks, spec["gpus"]) for st in spec["stages"]]
    t0, t1 = sum(r["t0"] for r in rows), sum(r["t1"] for r in rows)
    assert t0 < t1 < spec["t_ref"]
    # linear layers: T0 = FLOPs / 989.5 TFLOP/s exactly
    f = sum(bound.gemm_flops(g) for g in spec["stages"][0]["gemms"])
    assert math.isclose(rows[0]["t0"], f / 989.5e12, rel_tol=1e-9)
    # a stage frozen by a rule takes the frozen time in T1
    st = dict(spec["stages"][0], frozen_time=1.23)
    assert bound.stage_bounds(st, ds, peaks, 2)["t1"] == 1.23


def test_bound_attention_flops():
    a = {"b": 1, "h": 1, "s": 10, "d": 4, "causal": False, "dtype": "bf16", "pass": "fwd"}
    assert bound.attn_flops(a) == 4 * 10 * 10 * 4
    assert bound.attn_flops(dict(a, causal=True, **{"pass": "fwd+bwd"})) == 3.5 * 2 * 10 * 10 * 4


def evaluate(path, stage="E"):
    c, status = scorecard.evaluate(json.load(open(path)), stage)
    return {r[1].split()[0]: r[2] for r in c.rows}, status, scorecard.decision(status, c.rows)


def test_scorecard_example_releases():
    checks, status, dec = evaluate(HERE / "example/vetting.example.json")
    assert dec == "release" and all(v == "PASS" for v in status.values()), (checks, status)


def test_scorecard_flags():
    """Synthetic cards (headroom T_ref / T1 = 4) that trip the flags the released tasks' backtest showed."""
    hr = {"t_ref": 1.0, "t0": 0.2, "t1": 0.25, "t1_source": "measured"}
    pilot = lambda label, sp, **kw: dict({"label": label, "verdict": "PASS", "speedup": sp}, **kw)
    cases = [
        ({"D": {"pilots": [pilot("strong", 3.5)]}}, "D1", "FAIL"),                 # h = 90%: saturated
        ({"D": {"pilots": [pilot("strong", 1.25)]}}, "D2", "WARN"),                # best < 1.3x
        ({"D": {"pilots": [pilot("strong", 2.0), {"label": "weak", "verdict": "INCORRECT"}]}}, "D5", "FAIL"),
        ({"B": dict(hr, full_check_min=75)}, "B6", "WARN"),                          # full check > 60 min
    ]
    for extra, cid, st in cases:
        card = {"task": "t", "B": dict(hr, **extra.get("B", {}))}
        if "D" in extra:
            card["D"] = extra["D"]
        checks, _, _ = evaluate_card(card, "D")
        assert checks[cid] == st, (cid, checks)
    checks, _, _ = evaluate_card({"task": "t", "B": hr, "D": {"pilots": [pilot("strong", 2.5)]}}, "D")
    assert checks["D1"] == "PASS"  # h = 66%


def test_scorecard_protocol_short_names():
    """The protocol's template (short names) reads like the full format."""
    card = {"task": "t", "A": {"upstream_command": True, "public": True, "ambiguities_open": 0, "expert_anchor": None,
                               "bottleneck_class": ["gemm"]},
            "B": {"t_ref": 1.0, "t0": 0.2, "t1": 0.25, "t1_source": "measured", "ladder": 1.1, "aa_spread": 0.01,
                  "full_check_min": 30},
            "C": {"leak_hits": 0, "nop_reward": 0}}
    checks, status, _ = evaluate_card(card, "B")
    assert checks["A1"] == checks["A2"] == checks["B4"] == "PASS" and status["A"] != "incomplete", (status, checks)
    n = scorecard.normalize(card)
    assert n["C"]["leak_scan_hits"] == 0 and n["E"]["nop_reward"] == 0 and n["B"]["ladder"] == {"speedup": 1.1}


def test_scorecard_missing_is_never_pass():
    checks, status, dec = evaluate_card({"task": "t", "B": {"t_ref": 1.0}}, "B")
    assert status["B"] == "incomplete" and checks["B1"] == "-" and dec.startswith("continue A")


def test_scorecard_weak_reference_fails_ladder():
    card = {"task": "t", "B": {"t_ref": 1.0, "t0": 0.2, "t1": 0.25, "t1_source": "measured",
                               "ladder": {"speedup": 1.8}}}
    checks, _, _ = evaluate_card(card, "B")
    assert checks["B4"] == "FAIL"  # ln 1.8 / ln 4 = 42% > 1/3


def test_scorecard_judge_false_reject_blocks():
    card = {"task": "t", "B": {"t_ref": 1.0, "t1": 0.25},
            "D": {"pilots": [{"label": "strong", "verdict": "INCORRECT", "cause": "judge"},
                             {"label": "weak", "verdict": "PASS", "speedup": 1.5}]}}
    checks, _, _ = evaluate_card(card, "D")
    assert checks["D5"] == "FAIL"


def test_scorecard_single_pilot_is_enough():
    card = {"task": "t", "B": {"t_ref": 1.0, "t1": 0.25},
            "D": {"pilots": [{"label": "some-model", "verdict": "PASS", "speedup": 2.0}]}}
    checks, status, _ = evaluate_card(card, "D")
    assert checks["D1"] == "PASS" and checks["D3"] == "SKIP" and checks["D4"] == "SKIP"
    assert status["D"] in ("PASS", "WARN"), (checks, status)


def evaluate_card(card, stage):
    c, status = scorecard.evaluate(card, stage)
    return {r[1].split()[0]: r[2] for r in c.rows}, status, scorecard.decision(status, c.rows)


@pytest.mark.skipif(subprocess.run([sys.executable, "-c", "import torch"], capture_output=True).returncode != 0,
                    reason="torch not installed")
def test_peaks_cpu_plumbing(tmp_path):
    out = tmp_path / "peaks.json"
    subprocess.run([sys.executable, str(HERE / "peaks.py"), "measure", "--stages",
                    str(HERE / "example/stages.example.json"), "--out", str(out), "--device", "cpu",
                    "--settle", "0.05"], check=True, capture_output=True, env=dict(os.environ))
    d = json.load(open(out))
    assert d["gemm"] and d["attn"] and d["hbm_bytes_per_s"] > 0
