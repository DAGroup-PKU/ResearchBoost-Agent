"""Scorecard of a candidate task: every check of the vetting funnel (A desk, B bounds and baseline strength, C judge,
D pilot, E release) against its threshold, and what to do next.

  python scorecard.py vetting/<task>/vetting.json              all stages that have data
  python scorecard.py vetting/<task>/vetting.json --stage B    stop after B (the go/no-go before building a judge)
  python scorecard.py vetting/*/vetting.json                   one summary line per task, then the details
  python scorecard.py ... --markdown                           the same as Markdown, for REPORT.md

A check whose data is missing shows "-" and leaves its stage incomplete; it is never a pass. The thresholds are
initial values calibrated on the eight released tasks (references/guide.md, section 8); change them here, in one place.
Format of vetting.json: example/vetting.example.json.
"""
import argparse
import json
import math
import sys

THRESHOLDS = {
    "headroom_min": 2.0,          # B: T_ref / T1
    "headroom_hard_min": 1.5,     # B: below this the task is rejected outright
    "ladder_share_max": 1 / 3,    # B: ln(ladder speedup) / ln(T_ref / T1)
    "aa_spread_max": 0.03,        # B: |run1 - run2| / mean of two reference timings
    "full_check_min_warn": 60,    # B: minutes of one full check (10 h then allow 10 iterations)
    "full_check_min_fail": 120,
    "scale_shift_max": 0.15,      # B: largest change of a time share (GEMM / memory / comm / host) at full scale
    "accept_kinds_min": 3,        # C: kinds of legal rewrites
    "accept_runs_min": 3,         # C: runs per kind
    "frr_warn": 0.01,             # C: predicted overall false-reject rate of gate 3 (frr.py)
    "frr_fail": 0.05,
    "capture_min": 0.20,          # D: best pilot's share of the log headroom
    "capture_max": 0.80,
    "strong_speedup_min": 1.30,   # D: the best pilot's speedup (a narrower range cannot separate models)
    "rank_corr_min": 0.5,         # D: Spearman of this task's model order against the other tasks' (many models)
}
STAGES = "ABCDE"
CAUSES = {"model", "judge", "infra"}


class Card:
    def __init__(self, d):
        self.d = d
        self.rows = []  # (stage, id, status, value, rule, hint)

    def get(self, path, default=None):
        cur = self.d
        for p in path.split("."):
            if not isinstance(cur, dict) or p not in cur or cur[p] is None:
                return default
            cur = cur[p]
        return cur

    def add(self, stage, cid, status, value, rule, hint=""):
        self.rows.append((stage, cid, status, value, rule, hint))

    def flag(self, stage, cid, path, rule, hint, fail_if_false=True):
        v = self.get(path)
        if v is None:
            self.add(stage, cid, "-", "-", rule)
        else:
            self.add(stage, cid, "PASS" if v else ("FAIL" if fail_if_false else "WARN"), "yes" if v else "no", rule,
                     "" if v else hint)


def fmt(x, nd=2):
    return "-" if x is None else ("%.*f" % (nd, x) if isinstance(x, float) else str(x))


def headroom(c):
    t_ref, t1 = c.get("B.t_ref"), c.get("B.t1")
    return t_ref / t1 if t_ref and t1 else None


def h_of(speedup, hr):
    if speedup is None or hr is None:
        return None
    return math.log(max(speedup, 1.0)) / math.log(hr)


def stage_a(c):
    c.flag("A", "A1 upstream command", "A.realism.upstream_command", "the upstream's own documented command",
           "use the command the upstream documents; do not write a recipe")
    c.flag("A", "A2 public", "A.realism.public", "code, weights and data public, pinned with checksums",
           "replace private inputs with public ones")
    n = c.get("A.ambiguities_open")
    c.add("A", "A3 ambiguities", "-" if n is None else ("PASS" if n == 0 else "FAIL"), fmt(n), "0 open",
          "" if not n else "settle every two-reading rule with the owner and write it into the statement")
    anchor = c.get("A.realism.expert_anchor")
    c.add("A", "A4 expert anchor", "PASS" if anchor else ("WARN" if anchor is not None or "A" in c.d else "-"),
          (anchor[:40] if isinstance(anchor, str) else fmt(anchor)) if anchor else "none",
          "a known fast version or record, measured on the same hardware",
          "" if anchor else "no T_expert: T2 cannot be calibrated, and saturation is judged only against T1")
    cls = c.get("A.realism.bottleneck_class")
    c.add("A", "A5 bottleneck class", "-" if cls is None else ("PASS" if cls else "WARN"),
          ",".join(cls) if cls else fmt(cls), "registered (gemm / memory / comm / host / irregular)",
          "" if cls else "register it; prefer tasks that fill an empty class")


def stage_b(c):
    t_ref, t0, t1 = c.get("B.t_ref"), c.get("B.t0"), c.get("B.t1")
    hr = headroom(c)
    th = THRESHOLDS
    if hr is None:
        c.add("B", "B1 headroom", "-", "-", "T_ref/T1 >= %.1fx" % th["headroom_min"])
    else:
        st = "PASS" if hr >= th["headroom_min"] else ("WARN" if hr >= th["headroom_hard_min"] else "FAIL")
        c.add("B", "B1 headroom", st, "%.2fx" % hr, "T_ref/T1 >= %.1fx" % th["headroom_min"],
              "" if st == "PASS" else "little room above the bar: a larger or harder workload, or reject")
    src = c.get("B.t1_source")
    c.add("B", "B2 T1 measured", "-" if src is None else ("PASS" if src == "measured" else "WARN"), fmt(src),
          "every peak in T1 reached by a microbenchmark",
          "" if src in (None, "measured") else "run peaks.py; an estimated T1 is not a bound")
    if t_ref and t0 and t1:
        ok = t0 <= t1 * 1.0001 and t1 <= t_ref
        t2, tx = c.get("B.t2"), c.get("B.t_expert")
        note = "T0 %.4g <= T1 %.4g <= T_ref %.4g" % (t0, t1, t_ref)
        st, hint = ("PASS", "") if ok else ("FAIL", "a bound above the reference, or T1 below T0: recheck stages.json")
        if ok and t2 is not None and not t1 <= t2 <= (tx if tx is not None else t_ref):
            st, hint = "WARN", "T2 outside [T1, T_expert]: the T2 model is wrong somewhere"
        c.add("B", "B3 consistency", st, note, "T0 <= T1 <= T2 <= T_expert <= T_ref", hint)
    else:
        c.add("B", "B3 consistency", "-", "-", "T0 <= T1 <= T2 <= T_expert <= T_ref")
    lad = c.get("B.ladder.speedup")
    if lad is None or hr is None:
        c.add("B", "B4 switch ladder", "-", fmt(lad), "ln(ladder)/ln(T_ref/T1) <= 1/3")
    else:
        share = math.log(max(lad, 1.0)) / math.log(hr)
        c.add("B", "B4 switch ladder", "PASS" if share <= th["ladder_share_max"] else "FAIL",
              "%.2fx = %.0f%% of log headroom" % (lad, 100 * share), "ln(ladder)/ln(T_ref/T1) <= 1/3",
              "" if share <= th["ladder_share_max"] else
              "the reference is weak: fold these switches into the reference and re-measure, or reject")
    aa = c.get("B.aa_spread")
    c.add("B", "B5 A/A noise", "-" if aa is None else ("PASS" if aa <= th["aa_spread_max"] else "FAIL"),
          "-" if aa is None else "%.1f%%" % (100 * aa), "two reference timings within 3%",
          "" if aa is None or aa <= th["aa_spread_max"] else
          "find the noise first (clocks, CPU pinning, data loading, other work on the host)")
    fc = c.get("B.full_check_min")
    st = "-" if fc is None else ("PASS" if fc <= th["full_check_min_warn"] else
                                 ("WARN" if fc <= th["full_check_min_fail"] else "FAIL"))
    c.add("B", "B6 full check cost", st, "-" if fc is None else "%.0f min" % fc, "<= 60 min",
          "" if st in ("-", "PASS") else "fewer agent iterations in 10 h: shorten windows or trajectories")
    sh = c.get("B.scale_shift")
    c.add("B", "B7 scale representative", "-" if sh is None else ("PASS" if sh <= th["scale_shift_max"] else "WARN"),
          "-" if sh is None else "%.0f pts" % (100 * sh), "time shares within 15 points of full scale",
          "" if sh is None or sh <= th["scale_shift_max"] else "the scaled workload has another bottleneck structure")


def stage_c(c):
    rej = c.get("C.must_reject")
    if rej is None:
        c.add("C", "C1 must-reject", "-", "-", "every mutant rejected by the expected gate")
    else:
        missed = [m["name"] for m in rej if m.get("got") != m.get("expected")]
        names = {m["name"] for m in rej}
        st = "FAIL" if missed else "PASS"
        hint = "missed: " + ", ".join(missed) if missed else ""
        if not missed and c.get("C.accepts_n1", True) and not any("lr" in n for n in names):
            st, hint = "WARN", "add the lr x 1.02 control arm (the smallest effect gate 3 must resolve)"
        c.add("C", "C1 must-reject", st, "%d/%d" % (len(rej) - len(missed), len(rej)),
              "every mutant rejected by the expected gate", hint)
    acc = c.get("C.must_accept")
    if acc is None:
        c.add("C", "C2 must-accept", "-", "-", ">= 3 kinds x 3 runs, 0 rejected")
    else:
        rejected = sum(a.get("rejected", 0) for a in acc)
        runs = sum(a.get("runs", 0) for a in acc)
        thin = len(acc) < THRESHOLDS["accept_kinds_min"] or min(a.get("runs", 0) for a in acc) < \
            THRESHOLDS["accept_runs_min"]
        st = "FAIL" if rejected else ("WARN" if thin else "PASS")
        c.add("C", "C2 must-accept", st, "%d rejected / %d runs, %d kinds" % (rejected, runs, len(acc)),
              ">= 3 kinds x 3 runs, 0 rejected",
              "a legal rewrite was rejected: measure rho, fix the rule (never by loosening it for one candidate)"
              if rejected else ("more kinds or runs" if thin else ""))
    frr = c.get("C.frr_predicted")
    st = "-" if frr is None else ("PASS" if frr <= THRESHOLDS["frr_warn"] else
                                  ("WARN" if frr <= THRESHOLDS["frr_fail"] else "FAIL"))
    c.add("C", "C3 gate-3 false rejects", st, "-" if frr is None else "%.1f%%" % (100 * frr),
          "predicted overall <= 1% (frr.py --measured)",
          "" if st in ("-", "PASS") else "endpoints decorrelate: fewer or later-insensitive endpoints, or calibrate k")
    found, closed = c.get("C.red_team.holes_found"), c.get("C.red_team.holes_closed")
    if found is None:
        c.add("C", "C4 red team", "SKIP", "not run", "optional for now (paused to keep vetting short)")
    else:
        c.add("C", "C4 red team", "PASS" if closed == found else "FAIL", "%s/%s closed" % (closed, found),
              "every hole found closed with a control", "" if closed == found else "close each hole, add a control")
    hits = c.get("C.leak_scan_hits")
    c.add("C", "C5 leak scan", "-" if hits is None else ("PASS" if hits == 0 else "FAIL"), fmt(hits),
          "0 hits in what the agent can see", "" if not hits else "remove targets, results, hotspots, switch names")
    c.flag("C", "C6 rules aligned", "C.rules_aligned", "every judged rule stated, every stated rule judged",
           "align instruction.md and the judge")
    c.flag("C", "C7 startup controls", "C.prep_controls_pass", "startup preparation's controls as expected",
           "fix the judge before any pilot")
    c.flag("C", "C8 held-out timing data", "C.heldout_timing", "final timing window and gate-3 seed unseen by the agent",
           "the agent can tune for the visible window", fail_if_false=False)


def stage_d(c):
    """Any number of pilots (at least one full-budget cell). label = the model; a replicate is "<label>-rep".
    rank (optional) = the model's leaderboard rank, 1 = strongest; labels strong / mid / weak stand for 1 / 2 / 3."""
    pilots = c.get("D.pilots")
    hr = headroom(c)
    th = THRESHOLDS
    if not pilots:
        for cid, rule in (("D1 capture", "best pilot 20-80% of log headroom"),
                          ("D2 range", "best >= %.1fx" % th["strong_speedup_min"]),
                          ("D5 failures attributed", "all attributed, none judge")):
            c.add("D", cid, "-", "-", rule)
        return

    def acc(p):
        return p.get("speedup") if p.get("verdict") == "PASS" else 1.0

    best = max(pilots, key=acc)
    hb = h_of(acc(best), hr)
    if hb is None:
        c.add("D", "D1 capture", "-", "-", "best pilot 20-80% of log headroom")
    else:
        st = "PASS" if th["capture_min"] <= hb <= th["capture_max"] else "FAIL"
        hint = "" if st == "PASS" else ("saturated: a harder workload or retire it" if hb > th["capture_max"] else
                                        "almost nothing captured: check T1 and the rules before blaming the model")
        c.add("D", "D1 capture", st, "%.0f%% (%.3gx)" % (100 * hb, acc(best)), "best pilot 20-80% of log headroom",
              hint)
    s = acc(best)
    c.add("D", "D2 range", "PASS" if s >= th["strong_speedup_min"] else "WARN", "%.3gx" % s,
          "best >= %.1fx" % th["strong_speedup_min"], "" if s >= th["strong_speedup_min"] else
          "near the bar: a narrow range, or a weak model; say which in the report")
    models = [p for p in pilots if not str(p.get("label", "")).endswith("-rep")]
    reps = [p for p in pilots if str(p.get("label", "")).endswith("-rep")]
    by = {p.get("label"): p for p in models}
    if reps and hr:
        rep = reps[0]
        base = by.get(rep["label"][:-4])
        gap_rep = abs(h_of(acc(rep), hr) - h_of(acc(base), hr)) if base else None
        hs = [h_of(acc(p), hr) for p in models]
        gap_model = max(hs) - min(hs) if len(models) >= 2 else None
        if gap_rep is None:
            c.add("D", "D3 replicate", "-", "-", "replicate of a pilot model")
        elif gap_model is None:
            c.add("D", "D3 replicate", "SKIP", "rep gap %.2f" % gap_rep, "one model only: noise shown, nothing to compare")
        else:
            ok = gap_rep < gap_model
            c.add("D", "D3 replicate", "PASS" if ok else "WARN", "rep %.2f vs model %.2f" % (gap_rep, gap_model),
                  "replicate gap < model gap", "" if ok else "noise as large as model differences: more replicates")
    else:
        c.add("D", "D3 replicate", "SKIP", "no replicate", "optional (needs a second cell of one model)")
    ranked = []
    for p in models:
        r = p.get("rank", {"strong": 1, "mid": 2, "weak": 3}.get(p.get("label")))
        if r is not None:
            ranked.append((r, acc(p)))
    if len(ranked) >= 2:
        ranked.sort()
        sp = [x for _, x in ranked]
        ok = all(sp[i] >= sp[i + 1] for i in range(len(sp) - 1))
        c.add("D", "D4 order", "PASS" if ok else "WARN", " >= ".join("%.3g" % x for x in sp),
              "speedups follow the leaderboard order", "" if ok else
              "order differs from the leaderboard: noise or a different skill; replicate")
    else:
        c.add("D", "D4 order", "SKIP", "-", "optional (needs two models with a leaderboard rank)")
    fails = [p for p in pilots if p.get("verdict") != "PASS"] + list(c.get("D.failures", []))
    causes = [p.get("cause") for p in fails]
    unknown = sum(1 for x in causes if x not in CAUSES)
    judge = causes.count("judge")
    infra = causes.count("infra")
    st = "FAIL" if unknown or judge else ("WARN" if infra else "PASS")
    c.add("D", "D5 failures attributed", st, "%d failed: %d unattributed, %d judge, %d infra" %
          (len(fails), unknown, judge, infra), "all attributed, none judge",
          "attribute every INCORRECT/ERROR first" if unknown else
          ("judge false rejects: back to C" if judge else ("fix the infrastructure, rerun" if infra else "")))
    rc = c.get("D.rank_corr")
    if rc is not None:
        c.add("D", "D6 rank correlation", "PASS" if rc >= th["rank_corr_min"] else "WARN", "%.2f" % rc,
              "model order vs other tasks >= 0.5", "" if rc >= th["rank_corr_min"] else
              "this task ranks models differently: replicates tell noise from a different skill")


def stage_e(c):
    c.flag("E", "E1 clean build", "E.clean_build", "image and assets from scratch on a clean host", "fix the build")
    nop = c.get("E.nop_reward")
    c.add("E", "E2 nop", "-" if nop is None else ("PASS" if nop == 0 else "FAIL"), fmt(nop), "nop agent reward 0",
          "" if not nop else "an unchanged tree scored: the bar or the timing is broken")
    c.flag("E", "E3 oracle N/N", "E.oracle_pass", "best pilot tree passes every repeated final check",
           "a flaky accept: find the gate that flips")
    c.flag("E", "E4 versioned", "E.versioned", "judge hash, image digest and upstream commit recorded", "record them")


def normalize(d):
    """Accept the short field names of the protocol's scorecard template (references/protocol.md) next to the full
    format (example/vetting.example.json): A.upstream_command / public / expert_anchor / bottleneck_class ->
    A.realism.*, a number B.ladder -> B.ladder.speedup, C.leak_hits -> C.leak_scan_hits, C.nop_reward ->
    E.nop_reward, E.rejudge_2of2 -> E.oracle_pass. The full format wins where both are given."""
    d = json.loads(json.dumps(d))
    a = d.get("A") or {}
    realism = a.setdefault("realism", {}) if a else {}
    for k in ("upstream_command", "public", "expert_anchor", "bottleneck_class"):
        if k in a and k not in realism:
            realism[k] = a[k]
    b = d.get("B") or {}
    if isinstance(b.get("ladder"), (int, float)):
        b["ladder"] = {"speedup": b["ladder"]}
    cc = d.get("C") or {}
    if "leak_hits" in cc and "leak_scan_hits" not in cc:
        cc["leak_scan_hits"] = cc["leak_hits"]
    e = d.setdefault("E", {}) if ("nop_reward" in cc or d.get("E")) else d.get("E")
    if "nop_reward" in cc and "nop_reward" not in e:
        e["nop_reward"] = cc["nop_reward"]
    if e and "rejudge_2of2" in e and "oracle_pass" not in e:
        e["oracle_pass"] = e["rejudge_2of2"]
    return d


def evaluate(d, upto="E"):
    c = Card(normalize(d))
    for s, fn in zip(STAGES, (stage_a, stage_b, stage_c, stage_d, stage_e)):
        if STAGES.index(s) <= STAGES.index(upto):
            fn(c)
    status = {}
    for s in STAGES[: STAGES.index(upto) + 1]:
        sts = [r[2] for r in c.rows if r[0] == s]
        status[s] = "FAIL" if "FAIL" in sts else ("incomplete" if "-" in sts else ("WARN" if "WARN" in sts else "PASS"))
    return c, status


def decision(status, rows):
    for s, st in status.items():
        if st == "FAIL":
            hard = [r for r in rows if r[0] == s and r[2] == "FAIL"]
            if any(r[1].startswith(("B1", "D1")) for r in hard):
                return "reject or redesign (%s)" % ", ".join(r[1].split()[0] for r in hard)
            return "revise %s (%s)" % (s, ", ".join(r[1].split()[0] for r in hard))
        if st == "incomplete":
            return "continue %s" % s
    last = list(status)[-1]
    return "release" if last == "E" else "go to %s" % STAGES[STAGES.index(last) + 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cards", nargs="+")
    ap.add_argument("--stage", default="E", choices=list(STAGES), help="evaluate up to this stage")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--summary", action="store_true", help="with several cards: the summary table only")
    a = ap.parse_args()
    results = []
    for path in a.cards:
        d = json.load(open(path))
        c, status = evaluate(d, a.stage)
        hr = headroom(c)
        results.append((d.get("task", path), c, status, decision(status, c.rows), hr))
    md = a.markdown
    if len(results) > 1:
        cols = STAGES[: STAGES.index(a.stage) + 1]
        if md:
            print("| task | " + " | ".join(cols) + " | T_ref/T1 | FAIL | WARN | next |")
            print("|" + "---|" * (len(cols) + 5))
        else:
            print("%-19s " % "task" + " ".join("%-10s" % s for s in cols) + " %-8s %-16s %-16s %s" %
                  ("T_ref/T1", "FAIL", "WARN", "next"))
        for task, c, status, dec, hr in results:
            hrs = "%.2fx" % hr if hr else "-"
            fails = ",".join(r[1].split()[0] for r in c.rows if r[2] == "FAIL") or "-"
            warns = ",".join(r[1].split()[0] for r in c.rows if r[2] == "WARN") or "-"
            if md:
                print("| %s | %s | %s | %s | %s | %s |" % (task, " | ".join(status.values()), hrs, fails, warns, dec))
            else:
                print("%-19s " % task + " ".join("%-10s" % v for v in status.values()) +
                      " %-8s %-16s %-16s %s" % (hrs, fails, warns, dec))
        print()
        if a.summary:
            sys.exit(0)
    bad = False
    for task, c, status, dec, hr in results:
        print(("### %s: %s" if md else "== %s: %s") % (task, dec))
        if md:
            print("\n| check | status | value | rule | next |\n|---|---|---|---|---|")
        for stage, cid, st, val, rule, hint in c.rows:
            if md:
                print("| %s | %s | %s | %s | %s |" % (cid, st, val, rule, hint))
            else:
                print("  %-26s %-5s %-34s %s%s" % (cid, st, str(val)[:34], rule, ("  -> " + hint) if hint else ""))
        print()
        bad |= any(v == "FAIL" for v in status.values())
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
