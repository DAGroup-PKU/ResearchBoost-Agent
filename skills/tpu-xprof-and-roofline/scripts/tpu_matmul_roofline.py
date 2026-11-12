#!/usr/bin/env python3
"""Analytical TPU Roofline & Arithmetic Intensity Calculator for Forward, dX, and dW+FP32 Accum.

WHY THIS SCRIPT EXISTS
----------------------
When migrating models across TPU generations (e.g., from `v5p` at ~166 FLOP/byte
to `v6e` Trillium at ~560 FLOP/byte), engineers often see MXU utilization drop
despite 2x higher peak TFLOPS per chip. The culprit is rarely the forward pass
alone; it is the backward weight gradient with FP32 master gradient accumulation:

    dW = X.T @ dY;  grad_accum_fp32 += dW

Whereas forward (`Y = X @ W`) and activation gradient (`dX = dY @ W.T`) stream
`2 * K * N` bytes of BF16 weights, fused `dW + FP32 grad_accum` must read and
write the FP32 accumulator (`4 + 4 = 8 * K * N` bytes), requiring **4x higher
token batch `M`** to reach the hardware ridge point.

Furthermore, on `v6e` (`ridge = 560 FLOP/byte`), asymptotic arithmetic intensity
as `M -> inf` is bounded by `K * N / (K + N)`. If an unpacked projection has
`K = 1024, N = 1024`, its asymptotic AI is `512 FLOP/byte < 560`, meaning it is
**permanently HBM-bound for any batch size** unless packed with sibling
projections (`qkv_proj` or `gate_up_proj`).

WHAT THIS SCRIPT CALCULATES
---------------------------
1. Per-stage (`fwd`, `dX`, `dW + FP32 accum`, `dW BF16-only`) FLOPs, HBM traffic,
   Arithmetic Intensity (FLOP/byte), Roofline efficiency (% of compute peak),
   critical token count `M_crit` to reach the ridge point, and ideal latency (us).
2. Side-by-side structural optimization comparisons across `v5e`, `v5p`, `v6e`:
   - **Projection Packing**: Unpacked `q/k/v` & `gate/up` vs Packed `qkv` & `gate_up`.
   - **GQA Native Grouped Einsum**: `jnp.repeat(k, G)` + `jnp.repeat(v, G)` vs
     native 5D grouped einsum `[B, T, H_kv, G, D]`.
   - **Supervised Token Compaction (`supervised_compaction`)**: Full `[M, H] @ [H, V]`
     LM Head vs gathering only supervised tokens (`loss_mask == 1`, e.g., 25% of `M`).

EXAMPLE CLI INVOCATIONS
-----------------------
  # Run self-test verifying ridge points, M_crit math, and optimization deltas:
  python3 scripts/tpu_matmul_roofline.py --self-test

  # Full Transformer layer Roofline breakdown on v6e for M=2048 tokens/microbatch:
  python3 scripts/tpu_matmul_roofline.py --chip v6e --M 2048 --hidden 2048 --inter 6144

  # Compare single matmul (M, K, N) across all TPU generations:
  python3 scripts/tpu_matmul_roofline.py --chip all --M 4096 --K 2048 --N 6144
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

# Hardware Roofline specifications per single TPU chip/core
TPU_SPECS: dict[str, dict[str, float]] = {
    "v5e": {"peak_tflops": 197.0, "hbm_gbs": 819.0, "mxu_tile": 128},
    "v5p": {"peak_tflops": 459.0, "hbm_gbs": 2765.0, "mxu_tile": 128},
    "v6e": {"peak_tflops": 918.0, "hbm_gbs": 1640.0, "mxu_tile": 256},
}


def ridge_point(chip: str) -> float:
    """Return the hardware ridge point in FLOP/byte for `chip`."""
    spec = TPU_SPECS[chip]
    return (spec["peak_tflops"] * 1e12) / (spec["hbm_gbs"] * 1e9)


def critical_m(K: int, N: int, ridge: float, weight_bytes_per_elem: float = 2.0) -> float:
    """Return the minimum token dimension `M_crit` needed to become compute-bound.

    For FLOPs = 2*M*K*N and Bytes = 2*M*(K + N) + c*K*N:
      AI(M) >= ridge  <==>  M * (2*K*N - 2*ridge*(K + N)) >= c * ridge * K * N
    Returns `math.inf` if asymptotic AI `K*N / (K + N) <= ridge` (permanently HBM-bound).
    """
    denom = 2.0 * K * N - 2.0 * ridge * (K + N)
    if denom <= 0.0:
        return math.inf
    return (weight_bytes_per_elem * ridge * K * N) / denom


def analyze_matmul_stage(
    name: str,
    M: int,
    K: int,
    N: int,
    chip: str,
    weight_bytes_per_elem: float = 2.0,
    extra_bytes: float = 0.0,
) -> dict[str, Any]:
    """Compute Roofline metrics for one matrix multiplication fusion `(M, K) @ (K, N)`."""
    spec = TPU_SPECS[chip]
    peak_flops = spec["peak_tflops"] * 1e12
    bw_bytes = spec["hbm_gbs"] * 1e9
    ridge = peak_flops / bw_bytes

    flops = 2.0 * M * K * N
    hbm_bytes = 2.0 * M * K + 2.0 * M * N + weight_bytes_per_elem * K * N + extra_bytes
    ai = flops / max(hbm_bytes, 1.0)
    t_compute_us = (flops / peak_flops) * 1e6
    t_mem_us = (hbm_bytes / bw_bytes) * 1e6
    t_roofline_us = max(t_compute_us, t_mem_us)
    m_crit = critical_m(K, N, ridge, weight_bytes_per_elem=weight_bytes_per_elem)

    return {
        "name": name,
        "chip": chip,
        "M": M,
        "K": K,
        "N": N,
        "gflops": flops / 1e9,
        "hbm_mb": hbm_bytes / 1e6,
        "ai_flop_per_byte": ai,
        "ridge_flop_per_byte": ridge,
        "bound": "Compute (MXU)" if ai >= ridge else "HBM Bandwidth",
        "mxu_efficiency_pct": min(100.0, 100.0 * ai / ridge),
        "t_compute_us": t_compute_us,
        "t_mem_us": t_mem_us,
        "t_roofline_us": t_roofline_us,
        "m_crit": m_crit if math.isfinite(m_crit) else None,
        "asymptotic_ai": (K * N) / max(K + N, 1),
    }


def fwd_op(name: str, M: int, K: int, N: int, chip: str, extra_bytes: float = 0.0) -> dict[str, Any]:
    """Forward matmul `Y = X @ W` (BF16 X, W, Y -> 2 bytes per weight element)."""
    return analyze_matmul_stage(name, M, K, N, chip, weight_bytes_per_elem=2.0, extra_bytes=extra_bytes)


def dx_op(name: str, M: int, K: int, N: int, chip: str, extra_bytes: float = 0.0) -> dict[str, Any]:
    """Backward activation gradient `dX = dY @ W^T` (2 bytes per weight element)."""
    return analyze_matmul_stage(name, M, K, N, chip, weight_bytes_per_elem=2.0, extra_bytes=extra_bytes)


def dw_op(name: str, M: int, K: int, N: int, chip: str, fp32_accum: bool = True) -> dict[str, Any]:
    """Backward weight gradient `dW = X^T @ dY`.

    When `fp32_accum=True` (standard mixed-precision training with microbatching or
    FP32 master gradients), XLA reads and writes the FP32 accumulator (`4 + 4 = 8`
    bytes per parameter element). When `fp32_accum=False`, it writes BF16 `dW` (`2` bytes).
    """
    w_bytes = 8.0 if fp32_accum else 2.0
    return analyze_matmul_stage(name, M, K, N, chip, weight_bytes_per_elem=w_bytes)


def compare_transformer_layer_packing(
    chip: str,
    M: int,
    hidden: int = 2048,
    inter: int = 6144,
    q_dim: int = 2048,
    kv_dim: int = 512,
    vocab_size: int = 128256,
    supervised_frac: float = 0.25,
    seq_len: int = 2048,
) -> dict[str, Any]:
    """Compare unpacked vs packed projections, GQA repeat vs grouped einsum, and supervised_compaction."""
    H, I, Q, KV = hidden, inter, q_dim, kv_dim
    G = max(1, Q // max(KV, 1))

    def build_layer_ops(packed: bool) -> list[dict[str, Any]]:
        ops: list[dict[str, Any]] = []
        if packed:
            ops.append(fwd_op("qkv_proj fwd (packed)", M, H, Q + 2 * KV, chip))
            ops.append(fwd_op("o_proj fwd", M, Q, H, chip))
            ops.append(fwd_op("gate_up_proj fwd (packed)", M, H, 2 * I, chip))
            ops.append(fwd_op("down_proj fwd (+SwiGLU read)", M, I, H, chip, extra_bytes=2.0 * M * I))
            # Backward
            ops.append(dx_op("down_proj dX (+SwiGLU bwd)", M, I, H, chip, extra_bytes=6.0 * M * I))
            ops.append(dw_op("down_proj dW+fp32_acc", M, I, H, chip, fp32_accum=True))
            ops.append(dx_op("gate_up_proj dX (packed)", M, H, 2 * I, chip))
            ops.append(dw_op("gate_up_proj dW+fp32_acc (packed)", M, H, 2 * I, chip, fp32_accum=True))
            ops.append(dx_op("o_proj dX", M, Q, H, chip))
            ops.append(dw_op("o_proj dW+fp32_acc", M, Q, H, chip, fp32_accum=True))
            ops.append(dx_op("qkv_proj dX (packed)", M, H, Q + 2 * KV, chip))
            ops.append(dw_op("qkv_proj dW+fp32_acc (packed)", M, H, Q + 2 * KV, chip, fp32_accum=True))
        else:
            ops.extend([
                fwd_op("q_proj fwd", M, H, Q, chip),
                fwd_op("k_proj fwd", M, H, KV, chip),
                fwd_op("v_proj fwd", M, H, KV, chip),
                fwd_op("o_proj fwd", M, Q, H, chip),
                fwd_op("gate_proj fwd", M, H, I, chip),
                fwd_op("up_proj fwd", M, H, I, chip),
                fwd_op("down_proj fwd (+SwiGLU read)", M, I, H, chip, extra_bytes=2.0 * M * I),
                # Backward
                dx_op("down_proj dX (+SwiGLU bwd)", M, I, H, chip, extra_bytes=6.0 * M * I),
                dw_op("down_proj dW+fp32_acc", M, I, H, chip, fp32_accum=True),
                dx_op("gate_proj dX", M, H, I, chip),
                dx_op("up_proj dX", M, H, I, chip),
                dw_op("gate_proj dW+fp32_acc", M, H, I, chip, fp32_accum=True),
                dw_op("up_proj dW+fp32_acc", M, H, I, chip, fp32_accum=True),
                dx_op("o_proj dX", M, Q, H, chip),
                dw_op("o_proj dW+fp32_acc", M, Q, H, chip, fp32_accum=True),
                dx_op("q_proj dX", M, H, Q, chip),
                dx_op("k_proj dX (+residual accum)", M, H, KV, chip, extra_bytes=4.0 * M * H),
                dx_op("v_proj dX (+residual accum)", M, H, KV, chip, extra_bytes=4.0 * M * H),
                dw_op("q_proj dW+fp32_acc", M, H, Q, chip, fp32_accum=True),
                dw_op("k_proj dW+fp32_acc", M, H, KV, chip, fp32_accum=True),
                dw_op("v_proj dW+fp32_acc", M, H, KV, chip, fp32_accum=True),
            ])
        return ops

    unpacked_ops = build_layer_ops(packed=False)
    packed_ops = build_layer_ops(packed=True)
    t_unpacked_us = sum(o["t_roofline_us"] for o in unpacked_ops)
    t_packed_us = sum(o["t_roofline_us"] for o in packed_ops)

    # GQA: jnp.repeat(k, G) + jnp.repeat(v, G) materializes [M, Q] K and V in HBM
    # (fwd write+read + bwd read+reduce = 8 * M * KV * G bytes vs 8 * M * KV bytes)
    bw_bytes = TPU_SPECS[chip]["hbm_gbs"] * 1e9
    gqa_repeat_kv_bytes = 8.0 * M * KV * G
    gqa_native_kv_bytes = 8.0 * M * KV
    gqa_saved_us = ((gqa_repeat_kv_bytes - gqa_native_kv_bytes) / bw_bytes) * 1e6

    # LM Head + Cross-Entropy: full M tokens vs supervised-only M_sup = ceil(M * supervised_frac)
    m_sup = max(128, int(math.ceil(M * supervised_frac / 128.0) * 128))
    head_full = (
        fwd_op("lm_head fwd (full)", M, H, vocab_size, chip)["t_roofline_us"]
        + dx_op("lm_head dX (full)", M, H, vocab_size, chip)["t_roofline_us"]
        + dw_op("lm_head dW+fp32_acc (full)", M, H, vocab_size, chip)["t_roofline_us"]
    )
    head_compacted = (
        fwd_op("lm_head fwd (supervised_compaction)", m_sup, H, vocab_size, chip)["t_roofline_us"]
        + dx_op("lm_head dX (supervised_compaction)", m_sup, H, vocab_size, chip)["t_roofline_us"]
        + dw_op("lm_head dW+fp32_acc (supervised_compaction)", m_sup, H, vocab_size, chip)["t_roofline_us"]
    )

    return {
        "chip": chip,
        "M": M,
        "seq_len": seq_len,
        "ridge_flop_per_byte": ridge_point(chip),
        "unpacked_ops": unpacked_ops,
        "packed_ops": packed_ops,
        "layer_unpacked_us": t_unpacked_us,
        "layer_packed_us": t_packed_us,
        "packing_saved_us_per_layer": t_unpacked_us - t_packed_us,
        "packing_speedup": t_unpacked_us / max(t_packed_us, 1e-9),
        "gqa_ratio": G,
        "gqa_repeat_kv_mb": gqa_repeat_kv_bytes / 1e6,
        "gqa_native_kv_mb": gqa_native_kv_bytes / 1e6,
        "gqa_saved_us_per_layer": gqa_saved_us,
        "lm_head_full_us": head_full,
        "lm_head_trimmed_us": head_compacted,
        "lm_head_saved_us": head_full - head_compacted,
        "lm_head_speedup": head_full / max(head_compacted, 1e-9),
    }


def format_report(comp: dict[str, Any]) -> str:
    """Render a readable Markdown table for the layer Roofline analysis."""
    chip = comp["chip"]
    M = comp["M"]
    ridge = comp["ridge_flop_per_byte"]
    lines = [
        f"## TPU `{chip}` Matmul Roofline Analysis (`M = {M}` tokens, Ridge = `{ridge:.1f} FLOP/B`)",
        "",
        "| Stage / Fusion (Unpacked) | (M, K, N) | GFLOP | HBM (MB) | AI (FLOP/B) | Bound | %Peak MXU | `M_crit` | Time (us) |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for r in comp["unpacked_ops"]:
        m_crit_str = f"{r['m_crit']:.0f}" if r["m_crit"] is not None else "INF (HBM)"
        lines.append(
            f"| {r['name']} | `({r['M']},{r['K']},{r['N']})` | {r['gflops']:.1f} | "
            f"{r['hbm_mb']:.1f} | {r['ai_flop_per_byte']:.0f} | {r['bound']} | "
            f"{r['mxu_efficiency_pct']:.0f}% | {m_crit_str} | {r['t_roofline_us']:.1f} |"
        )
    lines.extend([
        "",
        "### Structural Optimization Summary",
        "",
        f"1. **Projection Packing (`qkv_proj` + `gate_up_proj`)**: "
        f"`{comp['layer_unpacked_us']:.1f} us` -> `{comp['layer_packed_us']:.1f} us` per layer "
        f"(saves **`{comp['packing_saved_us_per_layer']:.1f} us/layer`**, **`{comp['packing_speedup']:.2f}x`**).",
        f"2. **GQA Native Grouped Einsum (`G={comp['gqa_ratio']}` vs `jnp.repeat`)**: "
        f"KV traffic `{comp['gqa_repeat_kv_mb']:.1f} MB` -> `{comp['gqa_native_kv_mb']:.1f} MB` "
        f"(saves **`{comp['gqa_saved_us_per_layer']:.1f} us/layer`**).",
        f"3. **Supervised Token Compaction (`supervised_compaction` on LM Head)**: "
        f"`{comp['lm_head_full_us']:.1f} us` -> `{comp['lm_head_trimmed_us']:.1f} us` "
        f"(saves **`{comp['lm_head_saved_us']:.1f} us/step`**, **`{comp['lm_head_speedup']:.2f}x`**).",
    ])
    return "\n".join(lines)


def run_self_test() -> int:
    """Verify ridge points, dW+FP32 accumulation penalty, and packing speedups."""
    r_v5p = ridge_point("v5p")
    r_v6e = ridge_point("v6e")
    if not (165.0 <= r_v5p <= 167.0):
        raise AssertionError(f"Expected v5p ridge ~166.0, got {r_v5p}")
    if not (559.0 <= r_v6e <= 561.0):
        raise AssertionError(f"Expected v6e ridge ~559.8, got {r_v6e}")

    # Verify that on v6e, K=1024, N=1024 has asymptotic AI = 512 < 559.8 -> M_crit is None (infinite)
    op_small = dw_op("small_dw", M=4096, K=1024, N=1024, chip="v6e", fp32_accum=True)
    if op_small["m_crit"] is not None:
        raise AssertionError("Expected K=1024, N=1024 on v6e to have infinite M_crit")

    # Verify that dW+FP32 accum requires exactly 4x larger M_crit than fwd/dX when K, N are large
    m_fwd = critical_m(4096, 4096, r_v6e, weight_bytes_per_elem=2.0)
    m_dw = critical_m(4096, 4096, r_v6e, weight_bytes_per_elem=8.0)
    if abs(m_dw / m_fwd - 4.0) > 1e-6:
        raise AssertionError(f"Expected dW+acc M_crit to be 4x fwd M_crit, got {m_dw / m_fwd}")

    comp = compare_transformer_layer_packing("v6e", M=2048)
    if comp["layer_packed_us"] >= comp["layer_unpacked_us"]:
        raise AssertionError("Packed projections must be faster than unpacked on v6e")
    if comp["lm_head_trimmed_us"] >= comp["lm_head_full_us"]:
        raise AssertionError("Supervised token compaction must be faster than full LM head")

    print(format_report(comp))
    print("\n[SELF-TEST PASS] Analytical Roofline & Arithmetic Intensity calculator verified.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="TPU Forward / dX / dW+FP32-Accum Roofline & Arithmetic Intensity Calculator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--chip", choices=(*TPU_SPECS.keys(), "all"), default="v6e")
    ap.add_argument("--M", type=int, default=2048, help="Flattened token batch dimension M = B_micro * T.")
    ap.add_argument("--K", type=int, default=0, help="If >0 with --N, analyze a single (M, K, N) matmul.")
    ap.add_argument("--N", type=int, default=0, help="If >0 with --K, analyze a single (M, K, N) matmul.")
    ap.add_argument("--hidden", type=int, default=2048, help="Transformer hidden size H.")
    ap.add_argument("--inter", type=int, default=6144, help="MLP intermediate size I.")
    ap.add_argument("--q-dim", type=int, default=2048, help="Total query projection dim Q.")
    ap.add_argument("--kv-dim", type=int, default=512, help="Total key/value projection dim KV.")
    ap.add_argument("--vocab-size", type=int, default=128256, help="LM Head vocabulary size V.")
    ap.add_argument("--supervised-frac", type=float, default=0.25, help="Fraction of tokens with loss_mask==1.")
    ap.add_argument("--json-out", help="Optional path to write JSON report.")
    ap.add_argument("--self-test", action="store_true", help="Run built-in verification tests.")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()

    chips = list(TPU_SPECS.keys()) if args.chip == "all" else [args.chip]
    results = []
    for c in chips:
        if args.K > 0 and args.N > 0:
            stages = [
                fwd_op("Forward (Y = X @ W)", args.M, args.K, args.N, c),
                dx_op("Backward dX (dY @ W.T)", args.M, args.K, args.N, c),
                dw_op("Backward dW (BF16 out, no accum)", args.M, args.K, args.N, c, fp32_accum=False),
                dw_op("Backward dW + FP32 grad_accum (8B/param)", args.M, args.K, args.N, c, fp32_accum=True),
            ]
            results.append({"chip": c, "M": args.M, "K": args.K, "N": args.N, "stages": stages})
            print(f"\n=== Single Matmul ({args.M}, {args.K}, {args.N}) on TPU {c} (Ridge={ridge_point(c):.1f} FLOP/B) ===")
            for s in stages:
                mc = f"{s['m_crit']:.0f}" if s["m_crit"] is not None else "INF"
                print(
                    f"  {s['name']:42s} | AI={s['ai_flop_per_byte']:6.1f} | {s['bound']:13s} | "
                    f"eff={s['mxu_efficiency_pct']:4.0f}% | M_crit={mc:>6s} | t={s['t_roofline_us']:6.1f} us"
                )
        else:
            comp = compare_transformer_layer_packing(
                c,
                M=args.M,
                hidden=args.hidden,
                inter=args.inter,
                q_dim=args.q_dim,
                kv_dim=args.kv_dim,
                vocab_size=args.vocab_size,
                supervised_frac=args.supervised_frac,
            )
            results.append(comp)
            print(format_report(comp))
            print()

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
