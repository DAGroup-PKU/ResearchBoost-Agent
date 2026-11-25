"""Lower bounds T0 (datasheet) and T1 (measured peaks) on the timed unit of a task, from a stage table.

Each stage is the work between two real synchronization points of one timed unit (one optimizer update, one RL step,
one request batch), per GPU. A stage's time is max(compute, HBM traffic, link traffic); stages add up. Work the rules
let overlap belongs in one stage; communication that cannot hide is a stage of its own.

  T0: compute = FLOPs / datasheet peak of that precision; memory = fused minimum bytes / datasheet HBM bandwidth;
      link = bytes / datasheet NVLink bandwidth per direction. A physical floor: a candidate measured below it goes to
      an audit, it is not a result.
  T1: the same formula with measured peaks (peaks.py): each listed GEMM and attention shape at its own measured
      time, other FLOPs at the best measured GEMM rate of their precision, bytes at the measured copy bandwidth, link
      bytes at the measured bus bandwidth of the collective at that message size. Use this one to vet and to score.
  Rule constraints: a stage with "frozen_time" (for example the GEMMs of a bitwise-only task, which must keep the
      reference's kernel) takes that measured time in T1 instead of the formula.

  python bound.py stages.json                      T0 only
  python bound.py stages.json --peaks peaks.json   T0 and T1, per stage and in total
  python bound.py stages.json --peaks peaks.json --update vetting.json   also write t0/t1 into the scorecard

stages.json (per GPU, per timed unit; see example/stages.example.json):
  {"task": ..., "unit": "s per optimizer update", "t_ref": 0.99, "datasheet": "h200_sxm",
   "stages": [{"name": ..., "t_ref": 0.41,
               "gemms": [{"m": 8192, "n": 4096, "k": 4096, "dtype": "bf16", "count": 72, "batch": 1}],
               "attn":  [{"b": 8, "h": 16, "s": 2048, "d": 128, "causal": true, "dtype": "bf16", "count": 24,
                          "pass": "fwd+bwd"}],
               "flops": {"fp32": 1.2e11},          # FLOPs not covered by gemms/attn
               "bytes": 3.1e10,                     # fused minimum HBM bytes
               "comm": [{"op": "all_reduce", "bytes": 5.2e8, "count": 1}],   # payload per GPU
               "frozen_time": null}]}
"""
import argparse
import json
import math
import sys

# Dense datasheet peaks (FLOP/s, bytes/s). NVLink: per direction, per GPU.
DATASHEETS = {
    "h100_sxm": {"flops": {"fp8": 1979e12, "bf16": 989.5e12, "fp16": 989.5e12, "tf32": 494.7e12, "fp32": 66.9e12,
                           "fp64": 66.9e12},
                 "hbm": 3.35e12, "nvlink": 450e9},
    "h200_sxm": {"flops": {"fp8": 1979e12, "bf16": 989.5e12, "fp16": 989.5e12, "tf32": 494.7e12, "fp32": 66.9e12,
                           "fp64": 66.9e12},
                 "hbm": 4.8e12, "nvlink": 450e9},
}
# nccl-tests bus bandwidth: link bytes per GPU = payload * factor(n)
BUS_FACTOR = {"all_reduce": lambda n: 2.0 * (n - 1) / n, "all_gather": lambda n: (n - 1) / n,
              "reduce_scatter": lambda n: (n - 1) / n, "all_to_all": lambda n: (n - 1) / n,
              "send_recv": lambda n: 1.0, "broadcast": lambda n: 1.0}


def gemm_flops(g):
    return 2.0 * g["m"] * g["n"] * g["k"] * g.get("batch", 1) * g.get("count", 1)


def attn_flops(a):
    fwd = 4.0 * a["b"] * a["h"] * a["s"] * a.get("s_kv", a["s"]) * a["d"] * (0.5 if a.get("causal") else 1.0)
    per = {"fwd": fwd, "bwd": 2.5 * fwd, "fwd+bwd": 3.5 * fwd}[a.get("pass", "fwd+bwd")]
    return per * a.get("count", 1)


def gemm_key(g):
    return (g["m"], g["n"], g["k"], g.get("batch", 1), g["dtype"])


def attn_key(a):
    return (a["b"], a["h"], a["s"], a.get("s_kv", a["s"]), a["d"], bool(a.get("causal")), a["dtype"],
            a.get("pass", "fwd+bwd"))


def comm_link_bytes(c, world):
    return c["bytes"] * BUS_FACTOR[c["op"]](world) * c.get("count", 1)


class Peaks:
    """Measured peaks from peaks.py; lookups fail loudly for a shape that was not measured."""

    def __init__(self, data):
        self.d = data
        self.gemm = {(g["m"], g["n"], g["k"], g.get("batch", 1), g["dtype"]): g["seconds"] for g in data.get("gemm", [])}
        self.attn = {(a["b"], a["h"], a["s"], a.get("s_kv", a["s"]), a["d"], bool(a.get("causal")), a["dtype"],
                      a.get("pass", "fwd+bwd")): a["seconds"] for a in data.get("attn", [])}
        self.best = data.get("gemm_best_flops", {})

    def gemm_time(self, g):
        k = gemm_key(g)
        if k not in self.gemm:
            sys.exit("peaks.json has no measurement for GEMM %s: run peaks.py measure on this stages.json" % (k,))
        return self.gemm[k] * g.get("count", 1)

    def attn_time(self, a):
        k = attn_key(a)
        if k not in self.attn:
            sys.exit("peaks.json has no measurement for attention %s" % (k,))
        return self.attn[k] * a.get("count", 1)

    def flops_rate(self, dtype, datasheet):
        if dtype in self.best:
            return self.best[dtype]
        # no GEMM of this precision measured: fall back to the datasheet, flagged in the output
        return datasheet["flops"][dtype]

    def comm_time(self, c, world, datasheet):
        rows = [r for r in self.d.get("comm", []) if r["op"] == c["op"]]
        if not rows:
            return comm_link_bytes(c, world) / datasheet["nvlink"], False
        rows.sort(key=lambda r: r["bytes"])
        # bus bandwidth at the nearest measured message size not above the payload (smaller messages are slower)
        below = [r for r in rows if r["bytes"] <= c["bytes"]] or rows[:1]
        return comm_link_bytes(c, world) / below[-1]["busbw"], True


def stage_bounds(st, ds, peaks, world):
    flops_by = {}
    for g in st.get("gemms", []):
        flops_by[g["dtype"]] = flops_by.get(g["dtype"], 0.0) + gemm_flops(g)
    for a in st.get("attn", []):
        flops_by[a["dtype"]] = flops_by.get(a["dtype"], 0.0) + attn_flops(a)
    for dt, f in st.get("flops", {}).items():
        flops_by[dt] = flops_by.get(dt, 0.0) + f
    link = sum(comm_link_bytes(c, world) for c in st.get("comm", []))
    t0_parts = {"compute": sum(f / ds["flops"][dt] for dt, f in flops_by.items()),
                "memory": st.get("bytes", 0.0) / ds["hbm"],
                "link": link / ds["nvlink"]}
    out = {"name": st["name"], "flops": flops_by, "t0": max(t0_parts.values()),
           "t0_bound_by": max(t0_parts, key=t0_parts.get), "t_ref": st.get("t_ref"), "notes": []}
    if peaks is None:
        return out
    if st.get("frozen_time") is not None:
        out.update(t1=st["frozen_time"], t1_bound_by="frozen (rule)")
        return out
    compute = sum(peaks.gemm_time(g) for g in st.get("gemms", [])) + \
        sum(peaks.attn_time(a) for a in st.get("attn", []))
    for dt, f in st.get("flops", {}).items():
        if dt not in peaks.best:
            out["notes"].append("%s FLOPs at the datasheet rate (no %s GEMM measured)" % (dt, dt))
        compute += f / peaks.flops_rate(dt, ds)
    comm, measured = 0.0, True
    for c in st.get("comm", []):
        t, m = peaks.comm_time(c, world, ds)
        comm += t
        measured &= m
    if not measured:
        out["notes"].append("collective not measured: datasheet NVLink used")
    hbm = peaks.d.get("hbm_bytes_per_s") or ds["hbm"]
    t1_parts = {"compute": compute, "memory": st.get("bytes", 0.0) / hbm, "link": comm}
    out.update(t1=max(t1_parts.values()), t1_bound_by=max(t1_parts, key=t1_parts.get))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages")
    ap.add_argument("--peaks", help="peaks.json from peaks.py (adds T1)")
    ap.add_argument("--update", help="vetting.json: write B.t0, B.t1 (and B.t1_source) into it")
    a = ap.parse_args()
    spec = json.load(open(a.stages))
    ds = DATASHEETS[spec.get("datasheet", "h200_sxm")]
    peaks = Peaks(json.load(open(a.peaks))) if a.peaks else None
    world = spec.get("gpus", 1)
    rows = [stage_bounds(st, ds, peaks, world) for st in spec["stages"]]
    t_ref = spec.get("t_ref")
    t0 = sum(r["t0"] for r in rows)
    t1 = sum(r["t1"] for r in rows) if peaks else None

    print("%s (%s)" % (spec.get("task", a.stages), spec.get("unit", "s")))
    head = "%-34s %10s %10s %-9s" % ("stage", "t_ref", "T0", "T0 by")
    if peaks:
        head += " %10s %-14s" % ("T1", "T1 by")
    print(head)
    for r in rows:
        line = "%-34s %10s %10.4g %-9s" % (r["name"][:34], "%.4g" % r["t_ref"] if r["t_ref"] else "-", r["t0"],
                                           r["t0_bound_by"])
        if peaks:
            line += " %10.4g %-14s" % (r["t1"], r["t1_bound_by"])
        print(line + ("   " + "; ".join(r["notes"]) if r["notes"] else ""))
    print("%-34s %10s %10.4g %-9s%s" % ("total", "%.4g" % t_ref if t_ref else "-", t0, "",
                                        " %10.4g" % t1 if peaks else ""))
    if t_ref:
        print("T_ref / T0 = %.2fx" % (t_ref / t0) + ("   T_ref / T1 = %.2fx   log headroom ln(T_ref/T1) = %.3f"
                                                     % (t_ref / t1, math.log(t_ref / t1)) if peaks else ""))
    if a.update:
        card = json.load(open(a.update))
        b = card.setdefault("B", {})
        b["t0"] = t0
        if t_ref:
            b.setdefault("t_ref", t_ref)
        if peaks:
            b["t1"], b["t1_source"] = t1, "measured"
        json.dump(card, open(a.update, "w"), indent=1, ensure_ascii=False)
        print("wrote B.t0%s into %s" % (", B.t1" if peaks else "", a.update))


if __name__ == "__main__":
    main()
