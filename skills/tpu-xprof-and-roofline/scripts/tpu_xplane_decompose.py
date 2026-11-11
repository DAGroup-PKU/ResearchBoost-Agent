#!/usr/bin/env python3
"""Zero-dependency TPU xplane.pb step-time & Roofline gap decomposer.

WHY THIS SCRIPT EXISTS
----------------------
Standard TPU profiling workflows require installing heavy TensorBoard / `xprof`
packages or spinning up a browser server just to inspect a `*.xplane.pb` trace.
Even worse, aggregate step times hide whether a regression comes from low MXU
utilization in Linear/MLP fusions, unhidden collective communication on the
synchronous `XLA Ops` stream, an un-checkpointed frozen encoder, or Host/Idle
infeed stalls.

This script provides a 100% standalone, stdlib-only wire-format protobuf parser
for TPU `*.xplane.pb` files:
  1. Extracts `/device:TPU:<N>` planes (`Steps`, `XLA Modules`, `XLA Ops`).
  2. Decodes per-op `flops`, `bytes_accessed`, and the nested binary
     `memory_access_breakdown` protobuf to separate HBM bytes from VMEM bytes.
  3. Attributes every leaf HLO op (skipping `while`/`conditional`/`call`
     containers) into actionable architectural categories:
       - Attention
       - Linear / MLP
       - LM Head & Cross-Entropy
       - Frozen Encoder
       - Optimizer & Grad Accum
       - Exposed Collectives
       - Elementwise / Norm / Other
       - Host / Idle Gaps
  4. Computes per-category achieved MXU TFLOP/s, HBM bandwidth (GB/s),
     theoretical Roofline minimum time `max(flops / PEAK, hbm_bytes / BW)`,
     and ranks categories by **Roofline Gap (ms)** (`actual_ms - roofline_min_ms`).

WHEN TO USE
-----------
- Immediately after capturing a 3-to-5 step profile trace (`*.xplane.pb`) on
  TPU `v5e`, `v5p`, or `v6e`.
- Before attempting kernel or sharding optimizations, to identify the #1
  category by Roofline Gap (ms).

EXAMPLE CLI INVOCATIONS
-----------------------
  # Verify parser and Roofline decomposer in memory (no TPU or trace required):
  python3 scripts/tpu_xplane_decompose.py --self-test

  # Analyze a captured xplane.pb trace for TPU v6e:
  python3 scripts/tpu_xplane_decompose.py --trace /path/to/worker0.xplane.pb --chip v6e

  # Analyze with source directory AST lookup and save JSON + Markdown reports:
  python3 scripts/tpu_xplane_decompose.py \
      --trace /path/to/trace.xplane.pb \
      --chip v5p \
      --src-root /path/to/model_repo \
      --json-out /tmp/xplane_report.json
"""
from __future__ import annotations

import argparse
import ast
import bisect
import collections
import json
import os
import re
import statistics
import struct
import sys
import time
from typing import Any

# Hardware Roofline constants per chip (BF16 peak TFLOP/s and HBM bandwidth bytes/s)
CHIP_SPECS: dict[str, dict[str, float]] = {
    "v5e": {"peak_flops": 197e12, "hbm_bw": 819e9, "hbm_gib": 16.0},
    "v5p": {"peak_flops": 459e12, "hbm_bw": 2765e9, "hbm_gib": 95.74},
    "v6e": {"peak_flops": 918e12, "hbm_bw": 1640e9, "hbm_gib": 31.25},
}

CATEGORIES: tuple[str, ...] = (
    "Attention",
    "Linear / MLP",
    "LM Head & Cross-Entropy",
    "Frozen Encoder",
    "Optimizer & Grad Accum",
    "Exposed Collectives",
    "Elementwise / Norm / Other",
    "Host / Idle Gaps",
)

CONTAINER_OPS = {"while", "conditional", "call"}

COLLECTIVE_RE = re.compile(
    r"all-reduce|all-gather|reduce-scatter|collective-permute|all-to-all|^send|^recv",
    re.IGNORECASE,
)
ENCODER_RE = re.compile(
    r"encoder|vision_tower|vit|vae|tokenizer|quantizer|codebook|patch_embed|_encode_",
    re.IGNORECASE,
)
OPTIM_RE = re.compile(
    r"optax|adam|muon|zero1|grad_accum|apply_updates|clip_by_global_norm|ema_update",
    re.IGNORECASE,
)
HEAD_CE_RE = re.compile(
    r"lm_head|tied_logits|xent|cross_entropy|log_softmax|vocab",
    re.IGNORECASE,
)
ATTN_RE = re.compile(
    r"splash|attention|self_attn|qkv|softmax|flash_attn|grouped_attn|einsum_attn",
    re.IGNORECASE,
)
LINEAR_RE = re.compile(
    r"[qkvo]_proj|gate_proj|up_proj|down_proj|mlp|dense|fc1|fc2|project_",
    re.IGNORECASE,
)
SQUARE_ATTN_RE = re.compile(r"\[(?:\d+,){2,3}(\d+),\1\]")


# =============================================================================
# Minimal Zero-Dependency Protobuf Wire-Format Reader & Encoder
# =============================================================================

def decode_varint(buf: bytes | memoryview, pos: int) -> tuple[int, int]:
    """Decode a base-128 varint from `buf` starting at `pos`."""
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("Truncated varint in protobuf stream")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift >= 70:
            raise ValueError("Varint overflow in protobuf stream")


def to_signed_int64(val: int) -> int:
    """Convert an unsigned 64-bit integer to signed two's complement int64."""
    val &= 0xFFFFFFFFFFFFFFFF
    return val - 0x10000000000000000 if (val & 0x8000000000000000) else val


def iter_proto_fields(buf: bytes | memoryview):
    """Yield `(field_number, wire_type, value)` from a protobuf byte buffer."""
    pos = 0
    n = len(buf)
    mv = memoryview(buf)
    while pos < n:
        tag, pos = decode_varint(mv, pos)
        field_no = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:  # Varint
            val, pos = decode_varint(mv, pos)
            yield field_no, wire_type, val
        elif wire_type == 1:  # 64-bit fixed
            if pos + 8 > n:
                raise ValueError("Truncated 64-bit fixed field")
            val = bytes(mv[pos : pos + 8])
            pos += 8
            yield field_no, wire_type, val
        elif wire_type == 2:  # Length-delimited (string, bytes, submessage)
            length, pos = decode_varint(mv, pos)
            if pos + length > n:
                raise ValueError("Truncated length-delimited field")
            val = mv[pos : pos + length]
            pos += length
            yield field_no, wire_type, val
        elif wire_type == 5:  # 32-bit fixed
            if pos + 4 > n:
                raise ValueError("Truncated 32-bit fixed field")
            val = bytes(mv[pos : pos + 4])
            pos += 4
            yield field_no, wire_type, val
        else:
            raise ValueError(f"Unsupported protobuf wire type: {wire_type}")


def decode_memory_breakdown(raw: bytes | memoryview) -> tuple[int, int]:
    """Decode `memory_access_breakdown` bytes into `(hbm_bytes, vmem_bytes)`.

    In XProf's MemoryAccessBreakdown proto:
      - field 1: repeated MemorySpaceBreakdown
        - field 2: memory_space (1 = HBM, 2/3 = VMEM/SMEM/CMEM)
        - field 3: bytes_accessed (uint64)
    """
    hbm_total = 0
    vmem_total = 0
    try:
        for fno, wtype, sub in iter_proto_fields(raw):
            if fno != 1 or wtype != 2:
                continue
            mem_space = 0
            nbytes = 0
            for efno, ewtype, eval_ in iter_proto_fields(sub):
                if efno == 2 and ewtype == 0:
                    mem_space = int(eval_)
                elif efno == 3 and ewtype == 0:
                    nbytes = int(eval_)
            if mem_space == 1:
                hbm_total += nbytes
            elif mem_space in (2, 3, 4):
                vmem_total += nbytes
    except Exception:
        return 0, 0
    return hbm_total, vmem_total


def parse_xstat(buf: memoryview) -> tuple[int, str, Any]:
    """Parse an `XStat` message into `(metadata_id, kind, value)`."""
    meta_id = 0
    kind = "none"
    val: Any = None
    for fno, wtype, v in iter_proto_fields(buf):
        if fno == 1 and wtype == 0:
            meta_id = to_signed_int64(v)
        elif fno == 2 and wtype == 1:
            kind = "double"
            (val,) = struct.unpack("<d", v)
        elif fno == 3 and wtype == 0:
            kind = "uint64"
            val = int(v)
        elif fno == 4 and wtype == 0:
            kind = "int64"
            val = to_signed_int64(v)
        elif fno == 5 and wtype == 2:
            kind = "str"
            val = bytes(v).decode("utf-8", errors="replace")
        elif fno == 6 and wtype == 2:
            kind = "bytes"
            val = bytes(v)
        elif fno == 7 and wtype == 0:
            kind = "ref"
            val = int(v)
    return meta_id, kind, val


def parse_xevent_metadata(buf: memoryview) -> dict[str, Any]:
    """Parse an `XEventMetadata` message."""
    out: dict[str, Any] = {"id": 0, "name": "", "display_name": "", "stats": []}
    for fno, wtype, v in iter_proto_fields(buf):
        if fno == 1 and wtype == 0:
            out["id"] = to_signed_int64(v)
        elif fno == 2 and wtype == 2:
            out["name"] = bytes(v).decode("utf-8", errors="replace")
        elif fno == 4 and wtype == 2:
            out["display_name"] = bytes(v).decode("utf-8", errors="replace")
        elif fno == 5 and wtype == 2:
            out["stats"].append(parse_xstat(v))
    return out


def parse_xstat_metadata(buf: memoryview) -> tuple[int, str]:
    """Parse an `XStatMetadata` message into `(id, name)`."""
    sid = 0
    name = ""
    for fno, wtype, v in iter_proto_fields(buf):
        if fno == 1 and wtype == 0:
            sid = to_signed_int64(v)
        elif fno == 2 and wtype == 2:
            name = bytes(v).decode("utf-8", errors="replace")
    return sid, name


def parse_xevent(buf: memoryview) -> tuple[int, int, int]:
    """Parse an `XEvent` message into `(offset_ps, duration_ps, metadata_id)`."""
    meta_id = 0
    offset_ps = 0
    duration_ps = 0
    for fno, wtype, v in iter_proto_fields(buf):
        if fno == 1 and wtype == 0:
            meta_id = to_signed_int64(v)
        elif fno == 2 and wtype == 0:
            offset_ps = to_signed_int64(v)
        elif fno == 3 and wtype == 0:
            duration_ps = to_signed_int64(v)
    return offset_ps, duration_ps, meta_id


def parse_xline(buf: memoryview) -> dict[str, Any]:
    """Parse an `XLine` message."""
    name = ""
    timestamp_ns = 0
    raw_events: list[memoryview] = []
    for fno, wtype, v in iter_proto_fields(buf):
        if fno == 2 and wtype == 2:
            name = bytes(v).decode("utf-8", errors="replace")
        elif fno == 3 and wtype == 0:
            timestamp_ns = to_signed_int64(v)
        elif fno == 4 and wtype == 2:
            raw_events.append(v)
    events: list[tuple[int, int, int]] = []
    if name in ("Steps", "XLA Modules", "XLA Ops"):
        base_ps = timestamp_ns * 1000
        for ev_buf in raw_events:
            off_ps, dur_ps, mid = parse_xevent(ev_buf)
            events.append((base_ps + off_ps, dur_ps, mid))
        events.sort(key=lambda x: x[0])
    return {"name": name, "timestamp_ns": timestamp_ns, "events": events}


def parse_xplane(buf: memoryview) -> dict[str, Any]:
    """Parse an `XPlane` message."""
    name = ""
    plane_id = 0
    lines: list[dict[str, Any]] = []
    event_md_raw: dict[int, dict[str, Any]] = {}
    stat_names: dict[int, str] = {}

    for fno, wtype, v in iter_proto_fields(buf):
        if fno == 1 and wtype == 0:
            plane_id = to_signed_int64(v)
        elif fno == 2 and wtype == 2:
            name = bytes(v).decode("utf-8", errors="replace")
        elif fno == 3 and wtype == 2:
            lines.append(parse_xline(v))
        elif fno == 4 and wtype == 2:  # map<int64, XEventMetadata>
            mk = 0
            mv = None
            for efno, ewtype, ev in iter_proto_fields(v):
                if efno == 1 and ewtype == 0:
                    mk = to_signed_int64(ev)
                elif efno == 2 and ewtype == 2:
                    mv = parse_xevent_metadata(ev)
            if mv is not None:
                event_md_raw[mk] = mv
        elif fno == 5 and wtype == 2:  # map<int64, XStatMetadata>
            mk = 0
            mname = ""
            for efno, ewtype, ev in iter_proto_fields(v):
                if efno == 1 and ewtype == 0:
                    mk = to_signed_int64(ev)
                elif efno == 2 and ewtype == 2:
                    _, mname = parse_xstat_metadata(ev)
            stat_names[mk] = mname

    return {
        "id": plane_id,
        "name": name,
        "lines": lines,
        "event_metadata_raw": event_md_raw,
        "stat_names": stat_names,
    }


def parse_xspace_tpu_planes(data: bytes) -> list[dict[str, Any]]:
    """Parse all `/device:TPU:*` planes from an `XSpace` binary protobuf."""
    tpu_planes: list[dict[str, Any]] = []
    mv = memoryview(data)
    for fno, wtype, v in iter_proto_fields(mv):
        if fno == 1 and wtype == 2:
            # Quick peek at plane name before full parse if desired, or parse directly
            plane = parse_xplane(v)
            if re.fullmatch(r"/device:TPU:\d+", plane["name"]):
                tpu_planes.append(plane)
    tpu_planes.sort(key=lambda p: int(p["name"].rsplit(":", 1)[1]))
    return tpu_planes


# =============================================================================
# HLO Op & Source Stack Classification
# =============================================================================

def opcode_of(hlo_text: str) -> str:
    """Extract the HLO opcode from an XEventMetadata name like `%fusion.1 = bf16[...] fusion(...)`."""
    expr = hlo_text.split(" = ", 1)[-1]
    depth = 0
    for i, ch in enumerate(expr):
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        elif ch == " " and depth == 0:
            m = re.match(r"[\w.-]+", expr[i:].strip())
            return m.group(0) if m else "?"
    return expr.strip().split("(", 1)[0] or "?"


class SourceASTIndex:
    """Maps `(filepath, lineno)` from XLA `source_stack` to enclosing Python function names."""

    def __init__(self, roots: list[str]):
        self.roots = [os.path.abspath(r) for r in roots if os.path.isdir(r)]
        self.by_base: dict[str, list[str]] | None = None
        self.file_ranges: dict[str, list[tuple[int, int, str]]] = {}
        self.cache: dict[tuple[str, int], list[str]] = {}

    def _scan_roots(self) -> None:
        self.by_base = collections.defaultdict(list)
        for root in self.roots:
            for dp, dn, fn in os.walk(root):
                dn[:] = [d for d in dn if d not in (".git", "__pycache__", "node_modules", "site-packages")]
                for f in fn:
                    if f.endswith(".py"):
                        self.by_base[f].append(os.path.join(dp, f))

    def locate(self, path: str) -> str | None:
        if os.path.exists(path):
            return path
        if not self.roots:
            return None
        if self.by_base is None:
            self._scan_roots()
        assert self.by_base is not None
        cands = self.by_base.get(os.path.basename(path), [])
        if not cands:
            return None
        tail = "/".join(path.split("/")[-3:])
        cands.sort(key=lambda x: -len(os.path.commonprefix([x[::-1], tail[::-1]])))
        return cands[0]

    def funcs_at(self, path: str, line: int) -> list[str]:
        key = (path, line)
        if key in self.cache:
            return self.cache[key]
        if path not in self.file_ranges:
            real = self.locate(path)
            ranges: list[tuple[int, int, str]] = []
            if real:
                try:
                    with open(real, encoding="utf-8", errors="replace") as fh:
                        tree = ast.parse(fh.read())

                    def walk(node: ast.AST, stack: list[str]) -> None:
                        for ch in ast.iter_child_nodes(node):
                            if isinstance(ch, ast.ClassDef):
                                walk(ch, stack + [ch.name])
                            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                end_ln = getattr(ch, "end_lineno", ch.lineno) or ch.lineno
                                ranges.append((ch.lineno, end_ln, ".".join(stack + [ch.name])))
                                walk(ch, stack + [ch.name])
                            else:
                                walk(ch, stack)

                    walk(tree, [])
                except Exception:
                    pass
            self.file_ranges[path] = ranges
        matches = [(e - s, q) for s, e, q in self.file_ranges[path] if s <= line <= e]
        self.cache[key] = [q for _, q in sorted(matches)]
        return self.cache[key]


def classify_event_meta(
    meta: dict[str, Any],
    src_index: SourceASTIndex | None,
    vocab_size: int,
) -> dict[str, str]:
    """Classify an HLO event metadata entry into an op kind and architectural category."""
    name = meta["name"]
    long_name = meta["long"]
    cat = (meta.get("hlo_category") or "").lower()
    op = meta["op"]
    tf_op = str(meta.get("tf_op", ""))
    stack_txt = str(meta.get("source_stack", "")) or str(meta.get("source", ""))

    is_coll = bool(
        COLLECTIVE_RE.search(op)
        or cat.startswith("host ")
        or COLLECTIVE_RE.search(cat)
        or (op not in ("fusion", "?") and COLLECTIVE_RE.search(name.lower()))
    )
    if is_coll:
        return {"kind": "coll", "category": "Exposed Collectives", "how": "opcode"}

    kind = (
        "mxu"
        if ("convolution" in cat or cat == "dot" or op in ("convolution", "dot"))
        else "kernel"
        if (cat == "custom-call" or op == "custom-call")
        else "ew"
    )

    frames = [(p, int(ln)) for p, ln in re.findall(r"([^\s:]+\.py):(\d+)", stack_txt)]
    func_names: list[str] = []
    if src_index:
        for p, ln in frames:
            func_names.extend(src_index.funcs_at(p, ln))

    haystack = " ".join([tf_op, name, long_name, stack_txt, " ".join(func_names)])

    if ENCODER_RE.search(haystack):
        return {"kind": kind, "category": "Frozen Encoder", "how": "keyword/frame"}
    if OPTIM_RE.search(haystack):
        return {"kind": kind, "category": "Optimizer & Grad Accum", "how": "keyword/frame"}
    if HEAD_CE_RE.search(haystack) or re.search(rf"[\[,]{vocab_size}[\],]", long_name):
        return {"kind": kind, "category": "LM Head & Cross-Entropy", "how": "keyword/shape"}
    if ATTN_RE.search(haystack) or any(int(x) >= 64 for x in SQUARE_ATTN_RE.findall(long_name)):
        return {"kind": kind, "category": "Attention", "how": "keyword/shape"}
    if LINEAR_RE.search(haystack) or kind == "mxu":
        return {"kind": kind, "category": "Linear / MLP", "how": "keyword/mxu"}
    return {"kind": kind, "category": "Elementwise / Norm / Other", "how": "fallback"}


# =============================================================================
# Step Window Detection & Roofline Gap Decomposition
# =============================================================================

def detect_steps(lines_by_name: dict[str, list[tuple[int, int, int]]], fallback_nsteps: int):
    """Identify complete training step intervals `(start_ps, duration_ps, meta_id)`."""
    ops = lines_by_name.get("XLA Ops", [])
    if not ops:
        raise RuntimeError("No 'XLA Ops' events found on the selected TPU plane.")
    lo = min(e[0] for e in ops)
    hi = max(e[0] + e[1] for e in ops)
    tol = 1_000_000_000  # 1 ms in picoseconds

    steps = [
        s for s in lines_by_name.get("Steps", [])
        if s[0] >= lo - tol and (s[0] + s[1]) <= hi + tol
    ]
    if steps:
        return sorted(steps), "Steps line"

    mods = sorted(lines_by_name.get("XLA Modules", []), key=lambda e: -e[1])
    if mods:
        biggest = mods[0][1]
        steps = sorted(
            e for e in mods
            if e[1] > 0.5 * biggest and e[0] >= lo - tol and (e[0] + e[1]) <= hi + tol
        )
        if steps:
            return steps, "XLA Modules (largest)"

    n = max(fallback_nsteps, 1)
    dur = (hi - lo) // n
    return [(lo + i * dur, dur, 0) for i in range(n)], f"window/{n} (estimated)"


def decompose_xplane_bytes(
    raw_bytes: bytes,
    chip: str = "v6e",
    plane_index: int = 0,
    fallback_nsteps: int = 3,
    src_roots: list[str] | None = None,
    vocab_size: int = 128256,
    top_n: int = 5,
) -> dict[str, Any]:
    """Parse an `xplane.pb` binary payload and return the Roofline gap decomposition."""
    if chip not in CHIP_SPECS:
        raise ValueError(f"Unknown chip {chip!r}; choose from {sorted(CHIP_SPECS)}")
    peak_flops = CHIP_SPECS[chip]["peak_flops"]
    hbm_bw = CHIP_SPECS[chip]["hbm_bw"]
    ridge_point = peak_flops / hbm_bw

    t0 = time.perf_counter()
    planes = parse_xspace_tpu_planes(raw_bytes)
    if not planes:
        raise RuntimeError("No `/device:TPU:*` planes found in xplane payload.")

    plane = next((p for p in planes if p["name"].endswith(f":{plane_index}")), planes[0])
    stat_names = plane["stat_names"]
    src_index = SourceASTIndex(src_roots or []) if src_roots else None

    wanted_stats = {
        "hlo_category",
        "tf_op",
        "source_stack",
        "source",
        "flops",
        "bytes_accessed",
        "raw_bytes_accessed",
        "memory_access_breakdown",
    }

    md: dict[int, dict[str, Any]] = {}
    for mid, raw_em in plane["event_metadata_raw"].items():
        disp = raw_em["display_name"] or raw_em["name"]
        entry: dict[str, Any] = {
            "name": disp,
            "long": raw_em["name"],
            "op": opcode_of(raw_em["name"]),
        }
        for sm_id, skind, sval in raw_em["stats"]:
            sname = stat_names.get(sm_id, "")
            if sname in wanted_stats:
                if skind == "ref":
                    entry[sname] = stat_names.get(int(sval), str(sval))
                else:
                    entry[sname] = sval
        md[mid] = entry

    lines_map = {ln["name"]: ln["events"] for ln in plane["lines"]}
    steps, steps_how = detect_steps(lines_map, fallback_nsteps)
    ops = lines_map["XLA Ops"]
    starts = [e[0] for e in ops]

    op_info: dict[int, dict[str, Any]] = {}
    for mid, m in md.items():
        cls_info = classify_event_meta(m, src_index, vocab_size)
        flops = float(m.get("flops", 0) or 0)
        hbm_b = 0
        vmem_b = 0
        if isinstance(m.get("memory_access_breakdown"), (bytes, memoryview)):
            hbm_b, vmem_b = decode_memory_breakdown(m["memory_access_breakdown"])
        if hbm_b == 0:
            hbm_b = int(float(m.get("raw_bytes_accessed", 0) or m.get("bytes_accessed", 0) or 0))
        roof_ps = max(flops / peak_flops, hbm_b / hbm_bw) * 1e12
        cls_info.update(
            flops=flops,
            hbm_bytes=float(hbm_b),
            vmem_bytes=float(vmem_b),
            roof_ps=roof_ps,
        )
        op_info[mid] = cls_info

    acc = {cat: collections.Counter() for cat in CATEGORIES}
    top_ops = collections.defaultdict(collections.Counter)
    per_step_ms: list[float] = []
    total_hlo_flops = 0.0
    total_hbm_bytes = 0.0

    for s0, sd, _ in steps:
        s1 = s0 + sd
        i0 = bisect.bisect_left(starts, s0)
        i1 = bisect.bisect_left(starts, s1)
        leaf_evs = [
            e for e in ops[i0:i1]
            if md.get(e[2], {}).get("op") not in CONTAINER_OPS
        ]
        busy_end = s0
        busy_ps = 0
        for st, du, mid in leaf_evs:
            if mid not in op_info:
                continue
            en = min(st + du, s1)
            if st >= busy_end:
                busy_ps += en - st
                busy_end = en
            elif en > busy_end:
                busy_ps += en - busy_end
                busy_end = en

            info = op_info[mid]
            cat = info["category"]
            c = acc[cat]
            c["ps"] += du
            c["count"] += 1
            c["flops"] += info["flops"]
            c["hbm_bytes"] += info["hbm_bytes"]
            c["vmem_bytes"] += info["vmem_bytes"]
            total_hlo_flops += info["flops"]
            total_hbm_bytes += info["hbm_bytes"]

            if info["kind"] == "mxu":
                c["mxu_ps"] += du
                c["mxu_flops"] += info["flops"]
            elif info["kind"] == "ew":
                c["ew_ps"] += du
                c["ew_bytes"] += info["hbm_bytes"]
            elif info["kind"] == "kernel":
                c["kernel_ps"] += du

            # Collectives and Idle have a 0 ms Roofline lower bound (ideal = hidden/zero)
            if cat == "Exposed Collectives":
                bound_ps = 0.0
            elif info["kind"] == "kernel" and info["flops"] == 0 and info["hbm_bytes"] == 0:
                bound_ps = float(du)
            else:
                bound_ps = min(float(du), info["roof_ps"])
            c["roof_ps"] += bound_ps
            top_ops[cat][mid] += du

        idle_ps = max(0, sd - busy_ps)
        acc["Host / Idle Gaps"]["ps"] += idle_ps
        per_step_ms.append(sd / 1e9)

    n_steps = len(steps)
    mean_step_ms = statistics.mean(per_step_ms)
    rows: list[dict[str, Any]] = []
    for cat in CATEGORIES:
        c = acc[cat]
        actual_ms = c["ps"] / n_steps / 1e9
        roof_ms = c["roof_ps"] / n_steps / 1e9
        gap_ms = max(0.0, actual_ms - roof_ms)
        sec = c["ps"] / n_steps / 1e12
        achieved_tflops = (c["flops"] / n_steps / 1e12 / sec) if sec > 0 and c["flops"] > 0 else 0.0
        achieved_hbm_gbs = (c["hbm_bytes"] / n_steps / 1e9 / sec) if sec > 0 and c["hbm_bytes"] > 0 else 0.0
        top_list = []
        for mid, tot_ps in top_ops[cat].most_common(top_n):
            top_list.append({
                "name": md[mid]["name"],
                "ms_per_step": tot_ps / n_steps / 1e9,
                "gflops": op_info[mid]["flops"] / 1e9,
                "hbm_mb": op_info[mid]["hbm_bytes"] / 1e6,
            })
        rows.append({
            "category": cat,
            "actual_ms": actual_ms,
            "pct_step": 100.0 * actual_ms / max(mean_step_ms, 1e-9),
            "ops_per_step": int(round(c["count"] / n_steps)),
            "tflops_per_step": c["flops"] / n_steps / 1e12,
            "hbm_gb_per_step": c["hbm_bytes"] / n_steps / 1e9,
            "vmem_gb_per_step": c["vmem_bytes"] / n_steps / 1e9,
            "achieved_tflops_s": achieved_tflops,
            "pct_peak_flops": 100.0 * (achieved_tflops * 1e12) / peak_flops,
            "achieved_hbm_gbs": achieved_hbm_gbs,
            "pct_peak_hbm": 100.0 * (achieved_hbm_gbs * 1e9) / hbm_bw,
            "roofline_min_ms": roof_ms,
            "roofline_gap_ms": gap_ms,
            "top_ops": top_list,
        })

    rows_by_gap = sorted(rows, key=lambda r: -r["roofline_gap_ms"])
    step_s = mean_step_ms / 1e3
    hfu_pct = 100.0 * (total_hlo_flops / n_steps) / (peak_flops * step_s) if step_s > 0 else 0.0

    return {
        "chip": chip,
        "plane": plane["name"],
        "host_tpu_planes": len(planes),
        "steps_detected": n_steps,
        "steps_method": steps_how,
        "step_ms_mean": mean_step_ms,
        "step_ms_per_step": per_step_ms,
        "peak_tflops_s": peak_flops / 1e12,
        "peak_hbm_gbs": hbm_bw / 1e9,
        "ridge_flop_per_byte": ridge_point,
        "hlo_flops_per_step": total_hlo_flops / n_steps,
        "hbm_bytes_per_step": total_hbm_bytes / n_steps,
        "achieved_hfu_pct": hfu_pct,
        "categories": rows,
        "ranked_by_gap": rows_by_gap,
        "parse_wall_s": time.perf_counter() - t0,
    }


def render_markdown_report(rep: dict[str, Any]) -> str:
    """Format the decomposition dictionary as a clean GitHub-flavored Markdown report."""
    lines = [
        f"## TPU XPlane Step-Time & Roofline Decomposition (`{rep['chip']}`, `{rep['plane']}`)",
        "",
        f"- **Step Time**: `{rep['step_ms_mean']:.2f} ms` across `{rep['steps_detected']}` steps "
        f"(detected via `{rep['steps_method']}`)",
        f"- **Hardware Spec (`{rep['chip']}`)**: `{rep['peak_tflops_s']:.0f} TFLOP/s` BF16 peak | "
        f"`{rep['peak_hbm_gbs']:.0f} GB/s` HBM | Ridge point `{rep['ridge_flop_per_byte']:.1f} FLOP/byte`",
        f"- **Single-Chip HLO HFU**: `{rep['achieved_hfu_pct']:.1f}%` "
        f"(`{rep['hlo_flops_per_step'] / 1e12:.2f} TFLOP` / `{rep['hbm_bytes_per_step'] / 1e9:.2f} GB` HBM per step)",
        "",
        "### Categories Ranked by Roofline Gap (`actual_ms - roofline_min_ms`)",
        "",
        "| Rank | Category | Actual (ms) | % Step | Roofline Min (ms) | **Roofline Gap (ms)** | Achieved TF/s (%Peak) | Achieved HBM GB/s (%BW) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, r in enumerate(rep["ranked_by_gap"], start=1):
        tf_str = (
            f"{r['achieved_tflops_s']:.1f} ({r['pct_peak_flops']:.0f}%)"
            if r["tflops_per_step"] > 0
            else "n/a"
        )
        bw_str = (
            f"{r['achieved_hbm_gbs']:.0f} ({r['pct_peak_hbm']:.0f}%)"
            if r["hbm_gb_per_step"] > 0
            else "n/a"
        )
        lines.append(
            f"| {idx} | **{r['category']}** | {r['actual_ms']:.2f} | {r['pct_step']:.1f}% | "
            f"{r['roofline_min_ms']:.2f} | **{r['roofline_gap_ms']:.2f}** | {tf_str} | {bw_str} |"
        )
    return "\n".join(lines)


# =============================================================================
# Wire-Format Protobuf Encoder & Self-Test
# =============================================================================

def _enc_varint(val: int) -> bytes:
    val &= 0xFFFFFFFFFFFFFFFF
    out = bytearray()
    while True:
        byte = val & 0x7F
        val >>= 7
        if val:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _enc_field_varint(fno: int, val: int) -> bytes:
    return _enc_varint((fno << 3) | 0) + _enc_varint(val)


def _enc_field_bytes(fno: int, data: bytes | str) -> bytes:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return _enc_varint((fno << 3) | 2) + _enc_varint(len(raw)) + raw


def build_synthetic_xplane_payload() -> bytes:
    """Synthesize a binary `XSpace` wire-format payload in memory for `--self-test`."""
    stat_names = {
        1: "hlo_category",
        2: "tf_op",
        3: "source_stack",
        4: "flops",
        5: "bytes_accessed",
        6: "memory_access_breakdown",
    }

    def make_stat_str(sid: int, text: str) -> bytes:
        return _enc_field_varint(1, sid) + _enc_field_bytes(5, text)

    def make_stat_u64(sid: int, val: int) -> bytes:
        return _enc_field_varint(1, sid) + _enc_field_varint(3, val)

    def make_mem_breakdown(hbm_b: int, vmem_b: int) -> bytes:
        e_hbm = _enc_field_varint(2, 1) + _enc_field_varint(3, hbm_b)
        e_vmem = _enc_field_varint(2, 2) + _enc_field_varint(3, vmem_b)
        raw = _enc_field_bytes(1, e_hbm) + _enc_field_bytes(1, e_vmem)
        return _enc_field_varint(1, 6) + _enc_field_bytes(6, raw)

    # Define synthetic HLO metadata entries:
    # (mid, display_name, hlo_long, hlo_cat, tf_op, flops, hbm_bytes, vmem_bytes, dur_us)
    op_specs = [
        (1, "fusion.mlp", "%fusion.mlp = bf16[8,512,4096] fusion(x)", "convolution fusion",
         "model/layers_0/mlp/gate_proj", int(180e9), int(120e6), int(30e6), 400),
        (2, "fusion.attn", "%fusion.attn = bf16[8,16,512,512] fusion(q,k,v)", "convolution fusion",
         "model/layers_0/self_attn/attention", int(90e9), int(250e6), int(40e6), 350),
        (3, "fusion.head", "%fusion.head = f32[8,512,128256] fusion(h,w)", "convolution fusion",
         "model/lm_head/xent", int(120e9), int(400e6), int(20e6), 500),
        (4, "fusion.enc", "%fusion.enc = f32[8,256,1024] fusion(img)", "convolution fusion",
         "model/vision_tower/patch_embed", int(240e9), int(80e6), int(10e6), 600),
        (5, "all-gather.1", "%all-gather.1 = bf16[8,4096] all-gather(x)", "all-gather",
         "spmd/all_gather", 0, int(50e6), 0, 200),
        (6, "fusion.opt", "%fusion.opt = f32[4096] fusion(p,g)", "loop fusion",
         "optax/adam/apply_updates", int(1e9), int(160e6), 0, 150),
        (7, "while.loop", "%while.loop = (bf16[8]) while(init)", "while",
         "scan_body", 0, 0, 0, 0),
    ]

    plane_buf = bytearray()
    plane_buf += _enc_field_varint(1, 1)
    plane_buf += _enc_field_bytes(2, "/device:TPU:0")

    for sid, sname in stat_names.items():
        sm_msg = _enc_field_varint(1, sid) + _enc_field_bytes(2, sname)
        entry = _enc_field_varint(1, sid) + _enc_field_bytes(2, sm_msg)
        plane_buf += _enc_field_bytes(5, entry)

    for mid, disp, long_hlo, cat, tf_op, fl, hbm_b, vmem_b, _ in op_specs:
        em = bytearray()
        em += _enc_field_varint(1, mid)
        em += _enc_field_bytes(2, long_hlo)
        em += _enc_field_bytes(4, disp)
        em += _enc_field_bytes(5, make_stat_str(1, cat))
        em += _enc_field_bytes(5, make_stat_str(2, tf_op))
        em += _enc_field_bytes(5, make_stat_u64(4, fl))
        em += _enc_field_bytes(5, make_stat_u64(5, hbm_b + vmem_b))
        em += _enc_field_bytes(5, make_mem_breakdown(hbm_b, vmem_b))
        map_entry = _enc_field_varint(1, mid) + _enc_field_bytes(2, bytes(em))
        plane_buf += _enc_field_bytes(4, map_entry)

    ops_line = bytearray()
    ops_line += _enc_field_bytes(2, "XLA Ops")
    ops_line += _enc_field_varint(3, 1000)

    steps_line = bytearray()
    steps_line += _enc_field_bytes(2, "Steps")
    steps_line += _enc_field_varint(3, 1000)

    t_ps = 10_000_000
    for _step in range(2):
        step_start = t_ps
        # Add container while op spanning the active ops (must be ignored by leaf filter)
        w_ev = (
            _enc_field_varint(1, 7)
            + _enc_field_varint(2, step_start)
            + _enc_field_varint(3, 2_200_000_000)
        )
        ops_line += _enc_field_bytes(4, w_ev)

        for mid, _, _, _, _, _, _, _, dur_us in op_specs[:-1]:
            dur_ps = dur_us * 1_000_000
            ev = (
                _enc_field_varint(1, mid)
                + _enc_field_varint(2, t_ps)
                + _enc_field_varint(3, dur_ps)
            )
            ops_line += _enc_field_bytes(4, ev)
            t_ps += dur_ps

        # Add 300 us of host/idle gap at the end of the step
        t_ps += 300_000_000
        step_ev = (
            _enc_field_varint(1, 0)
            + _enc_field_varint(2, step_start)
            + _enc_field_varint(3, t_ps - step_start)
        )
        steps_line += _enc_field_bytes(4, step_ev)
        t_ps += 10_000_000

    plane_buf += _enc_field_bytes(3, bytes(ops_line))
    plane_buf += _enc_field_bytes(3, bytes(steps_line))
    return _enc_field_bytes(1, bytes(plane_buf))


def run_self_test() -> int:
    """Verify the wire-format xplane parser and Roofline gap decomposer in memory."""
    payload = build_synthetic_xplane_payload()
    rep = decompose_xplane_bytes(payload, chip="v6e", plane_index=0)
    by_cat = {r["category"]: r for r in rep["categories"]}

    expected_ms = {
        "Linear / MLP": 0.40,
        "Attention": 0.35,
        "LM Head & Cross-Entropy": 0.50,
        "Frozen Encoder": 0.60,
        "Exposed Collectives": 0.20,
        "Optimizer & Grad Accum": 0.15,
        "Host / Idle Gaps": 0.30,
    }
    for cat, exp_val in expected_ms.items():
        got_val = by_cat[cat]["actual_ms"]
        if abs(got_val - exp_val) > 1e-4:
            raise AssertionError(f"Category {cat}: expected {exp_val} ms, got {got_val} ms")

    # Verify container `while` event (2.2 ms) was excluded and step duration is 2.50 ms
    if abs(rep["step_ms_mean"] - 2.50) > 1e-4:
        raise AssertionError(f"Expected step_ms_mean=2.50, got {rep['step_ms_mean']}")

    # Verify memory_access_breakdown separated HBM from VMEM
    if abs(by_cat["Linear / MLP"]["hbm_gb_per_step"] - 0.12) > 1e-4:
        raise AssertionError(f"Unexpected HBM GB: {by_cat['Linear / MLP']['hbm_gb_per_step']}")
    if abs(by_cat["Linear / MLP"]["vmem_gb_per_step"] - 0.03) > 1e-4:
        raise AssertionError(f"Unexpected VMEM GB: {by_cat['Linear / MLP']['vmem_gb_per_step']}")

    # Verify Exposed Collectives has 0.0 ms Roofline lower bound (100% gap)
    if by_cat["Exposed Collectives"]["roofline_min_ms"] != 0.0:
        raise AssertionError("Exposed Collectives should have 0.0 ms Roofline lower bound")

    print(render_markdown_report(rep))
    print("\n[SELF-TEST PASS] Wire-format xplane.pb parser and Roofline decomposer verified.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Zero-dependency TPU xplane.pb step-time & Roofline gap decomposer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--trace", help="Path to a local *.xplane.pb file.")
    ap.add_argument("--chip", choices=sorted(CHIP_SPECS), default="v6e", help="TPU generation (default: v6e).")
    ap.add_argument("--plane-index", type=int, default=0, help="Device plane index to analyze (default: 0).")
    ap.add_argument("--nsteps", type=int, default=3, help="Fallback step count if Steps line is absent.")
    ap.add_argument("--src-root", action="append", default=[], help="Source root dir for AST stack attribution.")
    ap.add_argument("--vocab-size", type=int, default=128256, help="Vocabulary size for LM Head shape detection.")
    ap.add_argument("--json-out", help="Optional path to save full JSON decomposition report.")
    ap.add_argument("--self-test", action="store_true", help="Run in-memory synthetic xplane.pb self-test.")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()
    if not args.trace:
        ap.error("--trace is required unless --self-test is passed.")

    with open(args.trace, "rb") as fh:
        raw = fh.read()
    rep = decompose_xplane_bytes(
        raw,
        chip=args.chip,
        plane_index=args.plane_index,
        fallback_nsteps=args.nsteps,
        src_roots=args.src_root,
        vocab_size=args.vocab_size,
    )
    print(render_markdown_report(rep))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2)
        print(f"\nWrote JSON report to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
