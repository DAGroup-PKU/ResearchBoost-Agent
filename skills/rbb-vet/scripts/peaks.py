"""Measured peaks for the T1 bound (bound.py): every number here was reached by a kernel on this GPU.

  shapes   GEMM shapes per timed unit from a torch.profiler trace recorded with record_shapes=True
             python peaks.py shapes --trace trace.json --units 3 > gemms.json
  measure  on one GPU, at settled clocks: each GEMM and attention shape of stages.json, the copy bandwidth of HBM,
           the SM clock and board power while GEMMs run
             python peaks.py measure --stages stages.json --out peaks.json
  comm     the collectives of stages.json at their message sizes, bus bandwidth as nccl-tests defines it; adds the
           "comm" entries to an existing peaks.json
             torchrun --nproc-per-node <gpus> peaks.py comm --stages stages.json --out peaks.json

GEMMs: the faster of torch.matmul (cuBLAS/cuBLASLt heuristic) and torch.compile max-autotune (Triton templates
and cuBLAS candidates timed against each other; tens of seconds per shape, --no-autotune skips it), random normal
data (power draw, hence clocks, depend on the data), after a warm-up long enough for the clock to settle under the
power limit. fp8 uses torch._scaled_mm with tensor-wise scales. Attention: SDPA's flash backend and, when importable,
FlashAttention 3; forward + backward unless the stage says otherwise. Run it on an idle machine: other work on the
host moves every number here.
--device cpu runs the same code on tiny shapes, to test the plumbing without a GPU.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter

import torch

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, "tf32": torch.float32}


# ---------------------------------------------------------------- shapes from a trace
def shapes_from_trace(path, units):
    """Leaf GEMM ops with their input dims; counts divided by the number of timed units in the trace."""
    events = json.load(open(path))
    events = events["traceEvents"] if isinstance(events, dict) else events
    seen = Counter()
    kinds = {"aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm", "aten::_scaled_mm"}
    for e in events:
        if e.get("ph") != "X" or e.get("name") not in kinds:
            continue
        args = e.get("args", {})
        dims, types = args.get("Input Dims"), args.get("Input type", [])
        if not dims:
            continue
        name = e["name"]
        if name in ("aten::mm", "aten::_scaled_mm"):
            (m, k), (_, n) = dims[0], dims[1]
            batch, t = 1, types[0] if types else ""
        elif name == "aten::addmm":
            (m, k), (_, n) = dims[1], dims[2]
            batch, t = 1, types[1] if len(types) > 1 else ""
        else:  # bmm / baddbmm
            mats = dims[-2:]
            (batch, m, k), (_, _, n) = mats[0], mats[1]
            t = types[-2] if len(types) >= 2 else ""
        dtype = "fp8" if name == "aten::_scaled_mm" or "Float8" in t else \
            {"c10::BFloat16": "bf16", "c10::Half": "fp16", "float": "fp32"}.get(t, t or "bf16")
        seen[(m, n, k, batch, dtype)] += 1
    rows = [{"m": m, "n": n, "k": k, "batch": b, "dtype": dt, "count": round(c / units, 3)}
            for (m, n, k, b, dt), c in seen.most_common()]
    return rows


# ---------------------------------------------------------------- timing
class Clocks(threading.Thread):
    """Samples the SM clock and board power of one GPU while the GEMMs run."""

    def __init__(self, index):
        super().__init__(daemon=True)
        self.index, self.samples, self.stop = index, [], threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.read = lambda: (pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                                 pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
        except Exception:
            self.read = self._smi

    def _smi(self):
        out = subprocess.run(["nvidia-smi", "-i", str(self.index), "--query-gpu=clocks.sm,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        sm, pw = out.strip().split(",")
        return float(sm), float(pw)

    def run(self):
        while not self.stop.is_set():
            try:
                self.samples.append(self.read())
            except Exception:
                pass
            time.sleep(0.2)

    def summary(self):
        if not self.samples:
            return {}
        sm = [s for s, _ in self.samples]
        pw = [p for _, p in self.samples]
        return {"sm_mhz_median": statistics.median(sm), "sm_mhz_min": min(sm), "sm_mhz_max": max(sm),
                "power_w_median": statistics.median(pw), "power_w_max": max(pw), "samples": len(sm)}


def bench(fn, device, settle_s, min_s=0.5, repeats=5):
    """Median seconds per call, after `settle_s` of back-to-back calls (clocks settle under the power limit)."""
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    t_end = time.perf_counter() + settle_s
    fn()
    sync()
    while time.perf_counter() < t_end:
        fn()
    sync()
    # calls per repeat from one timed call
    t = time.perf_counter()
    fn()
    sync()
    one = max(time.perf_counter() - t, 1e-6)
    n = max(3, int(min_s / one))
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(n):
                fn()
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b) / 1000.0 / n)
        else:
            t = time.perf_counter()
            for _ in range(n):
                fn()
            times.append((time.perf_counter() - t) / n)
    return statistics.median(times)


# ---------------------------------------------------------------- kernels
def gemm_candidates(g, device, autotune):
    m, n, k, batch, dt = g["m"], g["n"], g["k"], g.get("batch", 1), g["dtype"]
    shape_a = (batch, m, k) if batch > 1 else (m, k)
    shape_b = (batch, k, n) if batch > 1 else (k, n)
    cands = {}
    if dt == "fp8":
        if device.type != "cuda":
            return {}
        a = torch.randn(m, k, device=device).to(torch.float8_e4m3fn)
        b = torch.randn(n, k, device=device).to(torch.float8_e4m3fn).t()  # column-major second operand
        one = torch.ones((), device=device)
        cands["scaled_mm"] = lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        return cands
    torch.backends.cuda.matmul.allow_tf32 = dt == "tf32"
    a = torch.randn(shape_a, device=device, dtype=DTYPES[dt])
    b = torch.randn(shape_b, device=device, dtype=DTYPES[dt])
    cands["cublas"] = lambda: torch.matmul(a, b)
    if autotune and device.type == "cuda":
        try:
            # a fresh compile per shape: one code object recompiled past the limit would silently run eager
            torch._dynamo.reset()
            f = torch.compile(lambda x, y: torch.matmul(x, y), mode="max-autotune-no-cudagraphs", dynamic=False)
            f(a, b)
            cands["inductor-max-autotune"] = lambda: f(a, b)
        except Exception as e:  # autotune is a bonus, not a requirement
            print("  max-autotune unavailable for %s: %s" % (g, str(e).splitlines()[0]), file=sys.stderr)
    return cands


def attn_candidates(at, device):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    b, h, s, skv, d = at["b"], at["h"], at["s"], at.get("s_kv", at["s"]), at["d"]
    dt = DTYPES[at["dtype"]]
    grad = at.get("pass", "fwd+bwd") != "fwd"
    q = torch.randn(b, h, s, d, device=device, dtype=dt, requires_grad=grad)
    kk = torch.randn(b, h, skv, d, device=device, dtype=dt, requires_grad=grad)
    v = torch.randn(b, h, skv, d, device=device, dtype=dt, requires_grad=grad)
    do = torch.randn(b, h, s, d, device=device, dtype=dt)
    causal = bool(at.get("causal"))
    cands = {}

    def run(fwd):
        def call():
            o = fwd()
            if grad:
                torch.autograd.grad(o, (q, kk, v), do)
        return call

    backend = SDPBackend.FLASH_ATTENTION if device.type == "cuda" else SDPBackend.MATH

    def sdpa():
        with sdpa_kernel(backend):
            return torch.nn.functional.scaled_dot_product_attention(q, kk, v, is_causal=causal)
    cands["sdpa-flash" if device.type == "cuda" else "sdpa-math"] = run(sdpa)
    if device.type == "cuda":
        try:
            import flash_attn_interface as fa3  # FlashAttention 3 (Hopper)
            qt, kt, vt = (x.detach().transpose(1, 2).contiguous().requires_grad_(grad) for x in (q, kk, v))
            dot = do.transpose(1, 2).contiguous()

            def fa3_call():
                o = fa3.flash_attn_func(qt, kt, vt, causal=causal)
                o = o[0] if isinstance(o, tuple) else o
                if grad:
                    torch.autograd.grad(o, (qt, kt, vt), dot)
            cands["fa3"] = fa3_call
        except Exception:
            pass
    return cands


def best_of(cands, device, settle_s):
    res = {}
    for name, fn in cands.items():
        try:
            res[name] = bench(fn, device, settle_s)
        except Exception as e:
            print("  %s failed: %s" % (name, str(e).splitlines()[0]), file=sys.stderr)
    if not res:
        return None, None, {}
    name = min(res, key=res.get)
    return name, res[name], res


def hbm_copy(device, gib):
    n = int(gib * 2 ** 30) // 4
    x = torch.empty(n, device=device, dtype=torch.float32).normal_()
    y = torch.empty_like(x)
    t = bench(lambda: y.copy_(x), device, settle_s=1.0)
    return 2.0 * x.numel() * 4 / t  # read + write


# ---------------------------------------------------------------- collect shapes from stages.json
def collect(spec, tiny):
    gemms, attns, comms = {}, {}, {}
    for st in spec["stages"]:
        for g in st.get("gemms", []):
            key = (g["m"], g["n"], g["k"], g.get("batch", 1), g["dtype"])
            gemms[key] = {"m": g["m"], "n": g["n"], "k": g["k"], "batch": g.get("batch", 1), "dtype": g["dtype"]}
        for a in st.get("attn", []):
            key = (a["b"], a["h"], a["s"], a.get("s_kv", a["s"]), a["d"], bool(a.get("causal")), a["dtype"],
                   a.get("pass", "fwd+bwd"))
            attns[key] = {k: a[k] for k in ("b", "h", "s", "d", "dtype") if k in a}
            attns[key].update(s_kv=a.get("s_kv", a["s"]), causal=bool(a.get("causal")), **{"pass": a.get("pass", "fwd+bwd")})
        for c in st.get("comm", []):
            comms[(c["op"], c["bytes"])] = {"op": c["op"], "bytes": c["bytes"]}
    gl, al = list(gemms.values()), list(attns.values())
    if tiny:  # plumbing test: same code paths on shapes a CPU can run
        gl = [dict(g, m=min(g["m"], 64), n=min(g["n"], 64), k=min(g["k"], 64)) for g in gl if g["dtype"] != "fp8"]
        al = [dict(a, b=1, h=2, s=min(a["s"], 64), s_kv=min(a["s_kv"], 64), d=min(a["d"], 32), dtype="fp32")
              for a in al]
    return gl, al, list(comms.values())


def cmd_measure(a):
    spec = json.load(open(a.stages))
    device = torch.device(a.device)
    tiny = device.type != "cuda"
    gemms, attns, _ = collect(spec, tiny)
    out = {"device": torch.cuda.get_device_name(device) if not tiny else "cpu (plumbing test)",
           "torch": torch.__version__, "settle_s": a.settle, "gemm": [], "attn": [], "comm": []}
    if device.type == "cuda":
        out["cuda"], out["cublas_workspace"] = torch.version.cuda, os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    clocks = Clocks(device.index or 0) if device.type == "cuda" else None
    if clocks:
        clocks.start()
    best = {}
    for g in gemms:
        name, sec, allres = best_of(gemm_candidates(g, device, not a.no_autotune), device, a.settle)
        if sec is None:
            continue
        flops = 2.0 * g["m"] * g["n"] * g["k"] * g.get("batch", 1)
        row = dict(g, seconds=sec, flops_per_s=flops / sec, backend=name, candidates=allres)
        out["gemm"].append(row)
        best[g["dtype"]] = max(best.get(g["dtype"], 0.0), flops / sec)
        print("gemm %5d x %5d x %5d (b%d) %-4s %8.1f TFLOP/s  %s" % (g["m"], g["n"], g["k"], g.get("batch", 1),
                                                                    g["dtype"], flops / sec / 1e12, name))
    if clocks:
        clocks.stop.set()
        clocks.join()
        out["clocks_during_gemm"] = clocks.summary()
    out["gemm_best_flops"] = best
    for at in attns:
        name, sec, allres = best_of(attn_candidates(at, device), device, a.settle)
        if sec is None:
            continue
        out["attn"].append(dict(at, seconds=sec, backend=name, candidates=allres))
        print("attn b%d h%d s%d/%d d%d %s %-7s %8.3f ms  %s" % (at["b"], at["h"], at["s"], at["s_kv"], at["d"],
                                                              "causal" if at["causal"] else "full", at["pass"],
                                                              sec * 1e3, name))
    out["hbm_bytes_per_s"] = hbm_copy(device, a.copy_gib if not tiny else 0.01)
    print("HBM copy %.2f TB/s" % (out["hbm_bytes_per_s"] / 1e12))
    if os.path.exists(a.out):  # keep comm entries measured earlier
        out["comm"] = json.load(open(a.out)).get("comm", [])
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote %s" % a.out)


def bench_collective(fn, device, dist, min_s=0.5, repeats=5):
    """Like bench(), but every rank makes the same number of calls (a collective needs all ranks)."""
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(20):
        fn()
    sync()
    t = time.perf_counter()
    for _ in range(5):
        fn()
    sync()
    n = torch.tensor([max(5, int(min_s / max((time.perf_counter() - t) / 5, 1e-6)))], device=device)
    dist.all_reduce(n, op=dist.ReduceOp.MIN)
    n = int(n)
    times = []
    for _ in range(repeats):
        dist.barrier()
        sync()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        sync()
        times.append((time.perf_counter() - t) / n)
    return statistics.median(times)


def cmd_comm(a):
    import torch.distributed as dist
    spec = json.load(open(a.stages))
    _, _, comms = collect(spec, False)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
    factor = {"all_reduce": 2.0 * (world - 1) / world, "all_gather": (world - 1) / world,
              "reduce_scatter": (world - 1) / world, "all_to_all": (world - 1) / world}
    rows = []
    for c in comms:
        if c["op"] not in factor:
            continue
        n = max(world, int(c["bytes"]) // 2 // world * world)  # bf16 elements, divisible by world
        x = torch.randn(n, device=device, dtype=torch.bfloat16 if backend == "nccl" else torch.float32)
        if c["op"] == "all_reduce":
            fn = lambda: dist.all_reduce(x)
        elif c["op"] == "all_gather":
            part, outb = x[: n // world], torch.empty_like(x)
            fn = lambda: dist.all_gather_into_tensor(outb, part)
        elif c["op"] == "reduce_scatter":
            outb = torch.empty(n // world, device=device, dtype=x.dtype)
            fn = lambda: dist.reduce_scatter_tensor(outb, x)
        else:
            outb = torch.empty_like(x)
            fn = lambda: dist.all_to_all_single(outb, x)
        sec = bench_collective(fn, device, dist)
        t = torch.tensor([sec], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # a collective ends when its slowest rank does
        sec = float(t)
        algbw = x.numel() * x.element_size() / sec
        rows.append({"op": c["op"], "bytes": c["bytes"], "world": world, "seconds": sec, "algbw": algbw,
                     "busbw": algbw * factor[c["op"]]})
        if rank == 0:
            print("%-15s %12d B  busbw %7.1f GB/s" % (c["op"], c["bytes"], algbw * factor[c["op"]] / 1e9))
    if rank == 0:
        out = json.load(open(a.out)) if os.path.exists(a.out) else {}
        out["comm"] = rows
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote comm into %s" % a.out)
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shapes", help="GEMM shapes per timed unit from a torch.profiler trace")
    s.add_argument("--trace", required=True)
    s.add_argument("--units", type=float, default=1.0, help="timed units (steps) the trace covers")
    m = sub.add_parser("measure", help="GEMM, attention, HBM copy and clocks on one GPU")
    m.add_argument("--stages", required=True)
    m.add_argument("--out", default="peaks.json")
    m.add_argument("--device", default="cuda:0")
    m.add_argument("--settle", type=float, default=3.0, help="seconds of warm-up per kernel (default 3)")
    m.add_argument("--copy-gib", type=float, default=4.0)
    m.add_argument("--no-autotune", action="store_true", help="skip torch.compile max-autotune")
    c = sub.add_parser("comm", help="collectives at the stage table's message sizes (under torchrun)")
    c.add_argument("--stages", required=True)
    c.add_argument("--out", default="peaks.json")
    a = ap.parse_args()
    if a.cmd == "shapes":
        json.dump(shapes_from_trace(a.trace, a.units), sys.stdout, indent=1)
        print()
    elif a.cmd == "measure":
        cmd_measure(a)
    else:
        cmd_comm(a)


if __name__ == "__main__":
    main()
