#!/usr/bin/env python3
"""Standalone PyTorch/XLA SPMD RNG consistency probe and HLO/IR RNG auditor.

What problem this solves:
-------------------------
Two subtle PyTorch/XLA random-number bugs on Cloud TPU cause silent training
divergence or false-positive determinism failures:

  1. 2-D Mesh SPMD `randn_like` Backward Sharding Corruption:
     On a 2-D SPMD mesh (e.g. `data=8 x fsdp=8`), if `noise = torch.randn_like(x)`
     is left unannotated and fed into a consumer sharded on `("data", "fsdp", None)`,
     GSPMD propagates the consumer's sharding backward into `rng-bit-generator`.
     The partitioned RNG feeds the forward/backward graph a *different* random
     draw from the tensor returned as a graph output (`step-0 grad norm 0.298`
     vs true `0.2386`), causing flow-matching / diffusion MSE loss to stall at
     2.0 over 120 steps while host inspection of `noise` looks completely normal!
     Fix: `replicated_randn_like(tensor, mesh)` immediately marks the drawn noise
     as replicated `(None, ...)` across all tensor dimensions.

  2. False-Positive Root Seed Advance on Deterministic `torch_xla.sync()`:
     When auditing an inference/rollout decode graph to prove it consumes zero
     device RNG (e.g. when using an explicit CPU generator for multinomial
     sampling), checking `xm.get_rng_state()` before and after `torch_xla.sync()`
     is misleading: `torch_xla` automatically advances the root seed on *every*
     step marker submission via the linear congruential recurrence:
         s' = (1012031 + 7012063 * s) mod 2^64
     even when the compiled HLO contains zero random operations!
     Fix: `audit_hlo_no_rng(tensors)` statically inspects both pre-lowering XLA
     IR (`_get_xla_tensors_text`) and lowered HLO (`_get_xla_tensors_hlo`) for
     `rng-bit-generator`, `philox`, `three_fry`, `randn`, `normal`, `uniform`,
     and `multinomial` (while avoiding false positives on HLO's `exponential`
     opcode used by `softmax`).

When to use it:
---------------
  - Use `replicated_randn_like(tensor, mesh)` whenever drawing Gaussian noise
    inside a multi-axis SPMD model (flow matching, diffusion, VAE, dropout).
  - Use `audit_hlo_no_rng(tensors)` in rollout/decode/packing tests to prove a
    device graph is 100% deterministic.
  - Run `--probe-spmd` across a TPU slice to test whether consumer sharding
    corrupts in-graph RNG values vs. returned host tensors.

Example CLI invocations:
------------------------
  # Verify LCG seed recurrence chain from initial seed 4396:
  python3 scripts/xla_spmd_rng_hlo_audit.py --verify-lcg 4396 --steps 5

  # Run live SPMD RNG consistency probe on a TPU worker:
  python3 scripts/xla_spmd_rng_hlo_audit.py --probe-spmd --mesh 8x8 --spec data,fsdp

  # Run built-in CPU self-tests (LCG formula, HLO/IR scanner fixtures, replicated wrapper):
  python3 scripts/xla_spmd_rng_hlo_audit.py --self-test
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Any, Callable

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


LCG_ADDEND = 1012031
LCG_MULTIPLIER = 7012063
LCG_MODULUS = 1 << 64

COMMON_RNG_PATTERN = (
    r"rng(?:[_-][A-Za-z0-9_-]+)*|"
    r"random(?:[_-][A-Za-z0-9_-]+)*|"
    r"rand[A-Za-z0-9_-]*|"
    r"(?:normal|uniform|bernoulli|(?:native_)?dropout|multinomial|poisson)_?|"
    r"philox[A-Za-z0-9_-]*|"
    r"three_?fry[A-Za-z0-9_-]*"
)

# In HLO text, `exponential` is the elementwise `exp(x)` op (e.g. in softmax),
# NOT `Tensor.exponential_()`. Only flag distribution names in pre-lowering IR.
IR_ONLY_DIST_PATTERN = (
    r"|(?:exponential|geometric|cauchy|log_normal|rrelu_with_noise|"
    r"_standard_gamma|_sample_dirichlet)_?"
)


def advance_xla_root_seed(seed: int, steps: int = 1) -> int:
    """Advance `seed` by `steps` iterations of `torch_xla`'s sync LCG recurrence.

    Every `torch_xla.sync()` submission advances the device root seed via:
        s' = (1012031 + 7012063 * s) mod 2^64
    even when the submitted graph contains zero random operations.
    """
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    curr = int(seed) & (LCG_MODULUS - 1)
    for _ in range(steps):
        curr = (LCG_ADDEND + LCG_MULTIPLIER * curr) % LCG_MODULUS
    return curr


def verify_seed_lcg_chain(
    initial_seed: int, observed_offsets_and_seeds: list[tuple[int, int]]
) -> dict[str, Any]:
    """Verify whether observed `(step_offset, seed)` pairs follow the pure LCG chain."""
    rows = []
    all_match = True
    for offset, observed in observed_offsets_and_seeds:
        expected = advance_xla_root_seed(initial_seed, steps=offset)
        matches = int(observed) == expected
        all_match = all_match and matches
        rows.append(
            {
                "offset": offset,
                "observed": int(observed),
                "expected": expected,
                "matches_lcg": matches,
            }
        )
    return {
        "initial_seed": int(initial_seed),
        "recurrence": "s' = (1012031 + 7012063 * s) mod 2^64",
        "all_match_pure_sync_lcg": all_match,
        "observations": rows,
    }


def classify_random_ops(text: str, kind: str = "hlo") -> list[str]:
    """Scan XLA IR (`kind='ir'`) or HLO (`kind='hlo'`) text for RNG operations."""
    if kind not in ("ir", "hlo"):
        raise ValueError(f"kind must be 'ir' or 'hlo', got {kind!r}")
    extra = IR_ONLY_DIST_PATTERN if kind == "ir" else ""
    pattern = (
        r"(?<![A-Za-z0-9_])(?:"
        + COMMON_RNG_PATTERN
        + extra
        + r")(?![A-Za-z0-9_])"
    )
    return sorted(set(re.findall(pattern, text, flags=re.IGNORECASE)))


def audit_hlo_no_rng(
    tensors: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    graph_reader: Callable[[list[torch.Tensor]], dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Audit pre-lowering XLA IR and lowered HLO of `tensors` to prove zero device RNG.

    Raises `RuntimeError` if any random operation (`rng-bit-generator`,
    `aten::randn`, `aten::multinomial`, etc.) is present in either HLO or IR.
    """
    if torch is not None and isinstance(tensors, torch.Tensor):
        tensor_list: list[Any] = [tensors]
    elif isinstance(tensors, (list, tuple)):
        tensor_list = list(tensors)
    else:
        tensor_list = [tensors]
    if graph_reader is not None:
        graphs = graph_reader(tensor_list)
    else:
        import torch_xla

        graphs = {
            "ir": torch_xla._XLAC._get_xla_tensors_text(tensor_list),
            "hlo": torch_xla._XLAC._get_xla_tensors_hlo(tensor_list),
        }

    if not isinstance(graphs, dict) or not all(
        isinstance(graphs.get(k), str) and graphs[k].strip()
        for k in ("ir", "hlo")
    ):
        raise RuntimeError(
            "audit_hlo_no_rng received empty or invalid {'ir', 'hlo'} graph dump"
        )

    hits: list[str] = []
    # Check HLO first so lowered `rng-bit-generator` is reported before IR ops.
    for kind in ("hlo", "ir"):
        ops = classify_random_ops(graphs[kind], kind=kind)
        hits.extend(f"{kind}:{op}" for op in ops)

    result = {
        "status": "PASS" if not hits else "FAIL",
        "random_operations": hits,
        "ir_lines": len(graphs["ir"].splitlines()),
        "hlo_lines": len(graphs["hlo"].splitlines()),
    }
    if hits:
        raise RuntimeError(
            f"Device graph contains forbidden RNG operation(s): {hits}"
        )
    return result


def replicated_randn_like(
    tensor: torch.Tensor,
    mesh: Any = None,
    mark_sharding_fn: Callable[[torch.Tensor, Any, tuple[Any, ...]], Any] | None = None,
) -> torch.Tensor:
    """Draw `torch.randn_like(tensor)` and immediately pin its SPMD sharding to replicated.

    Prevents 2-D mesh consumer sharding (`data x fsdp`) from propagating backward
    into `rng-bit-generator` and corrupting the in-graph noise draw.
    """
    if torch is not None:
        noise = torch.randn_like(tensor)
    elif hasattr(tensor, "_randn_like"):
        noise = tensor._randn_like()
    else:
        raise ImportError(
            "PyTorch (`pip install tpu-handbook-kit[torch]`) is required for replicated_randn_like."
        )
    if mesh is None and tensor.device.type != "xla":
        return noise

    replicated_spec = tuple(None for _ in range(noise.dim()))
    if mark_sharding_fn is not None:
        mark_sharding_fn(noise, mesh, replicated_spec)
    elif mesh is not None:
        import torch_xla.distributed.spmd as xs

        xs.mark_sharding(noise, mesh, replicated_spec)
    return noise


def probe_spmd_rng_consistency(
    mesh_spec: str = "8x8",
    consumer_spec: str = "data,fsdp",
    rows: int = 512,
    cols: int = 1024,
    force_replicated_noise: bool = False,
) -> dict[str, Any]:
    """Live TPU SPMD probe comparing in-graph random tensor consumers vs host readback.

    Computes `s1 = y.sum(1)`, `s2 = (y * y).sum(1)`, and `g = y[:8] @ y[:8].T`
    inside the SPMD graph where `y = x * 1.0` carries `consumer_spec`, and
    compares against the same statistics computed on CPU from `x.cpu()`.
    """
    import numpy as np
    import torch_xla
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    from torch_xla.core import xla_model as xm

    xr.use_spmd()
    n_dev = xr.global_runtime_device_count()
    if "x" in mesh_spec:
        a, b = (int(v) for v in mesh_spec.split("x"))
        mesh = xs.Mesh(np.arange(n_dev), (a, b), ("data", "fsdp"))
    else:
        mesh = xs.Mesh(np.arange(int(mesh_spec)), (int(mesh_spec),), ("data",))

    spec_tuple = tuple(
        None if s.strip().lower() == "none" else s.strip()
        for s in consumer_spec.split(",")
    )
    spec_tuple = spec_tuple + (None,) * max(0, 2 - len(spec_tuple))

    dev = torch_xla.device()
    torch.manual_seed(0)
    xm.set_rng_state(0)

    base = torch.zeros(rows, cols, device=dev)
    if force_replicated_noise:
        x = replicated_randn_like(base, mesh=mesh)
    else:
        x = torch.randn_like(base)

    y = x * 1.0
    xs.mark_sharding(y, mesh, spec_tuple)
    s1 = y.sum(1)
    s2 = (y * y).sum(1)
    gram = y[:8] @ y[:8].T
    torch_xla.sync()
    xm.wait_device_ops()

    xh = x.cpu().double()
    s1h = s1.cpu().double()
    s2h = s2.cpu().double()
    gh = gram.cpu().double()

    rel_sum = float((s1h - xh.sum(1)).abs().max() / xh.sum(1).abs().max())
    rel_sumsq = float(
        (s2h - (xh * xh).sum(1)).abs().max() / (xh * xh).sum(1).abs().max()
    )
    rel_gram = float(
        (gh - (xh[:8] @ xh[:8].T)).abs().max()
        / (xh[:8] @ xh[:8].T).abs().max()
    )
    consistent = max(rel_sum, rel_sumsq, rel_gram) < 1e-4

    return {
        "mesh": mesh_spec,
        "consumer_spec": consumer_spec,
        "force_replicated_noise": force_replicated_noise,
        "rel_sum": rel_sum,
        "rel_sumsq": rel_sumsq,
        "rel_gram": rel_gram,
        "consistent": consistent,
    }


def run_self_test() -> int:
    """Run CPU self-tests for LCG recurrence, HLO/IR RNG scanner, and replicated wrapper."""
    print("Running xla_spmd_rng_hlo_audit.py --self-test ...")

    # 1. Test LCG seed recurrence at offsets (1, 2, 12, 13, 14, 28, 29, 76) from seed 4396
    init_seed = 4396
    marker_offsets = [1, 2, 12, 13, 14, 28, 29, 76]
    synthetic_observations = [
        (off, advance_xla_root_seed(init_seed, off)) for off in marker_offsets
    ]
    assert synthetic_observations[0][1] == (1012031 + 7012063 * 4396) % (1 << 64)
    lcg_ok = verify_seed_lcg_chain(init_seed, synthetic_observations)
    assert lcg_ok["all_match_pure_sync_lcg"] is True, lcg_ok

    # Mutation check on LCG chain: corrupt offset 14 by +1
    mutated_observations = list(synthetic_observations)
    mutated_observations[4] = (
        mutated_observations[4][0],
        mutated_observations[4][1] + 1,
    )
    lcg_mut = verify_seed_lcg_chain(init_seed, mutated_observations)
    assert lcg_mut["all_match_pure_sync_lcg"] is False, "LCG mutation was not caught"

    # 2. Test HLO/IR RNG static audit on clean softmax HLO vs stochastic HLO/IR fixtures
    clean_ir = """
    IR {
      %0 = f32[4,128]{1,0} xla::device_data()
      %1 = f32[4,128]{1,0} aten::sub(%0, %0)
      %2 = f32[4,128]{1,0} aten::exp(%1)
      %3 = f32[4,2]{1,0} aten::index_copy(%0, %2)
    }
    """
    # Note: HLO uses `exponential` for ordinary `exp(x)` inside softmax!
    # Our classifier must NOT false-positive on `exponential` in HLO mode.
    clean_hlo = """
    HloModule IrToHlo.12
    ENTRY %IrToHlo.12 (p0.1: f32[4,128]) -> (f32[4,128]) {
      %p0.1 = f32[4,128]{1,0} parameter(0)
      %exponential.5 = f32[4,128]{1,0} exponential(f32[4,128]{1,0} %p0.1)
      %add.6 = f32[4,128]{1,0} add(f32[4,128]{1,0} %exponential.5, f32[4,128]{1,0} %p0.1)
    }
    """
    dummy_tensor: Any = torch.zeros(4, 128) if torch is not None else object()
    audit_clean = audit_hlo_no_rng(
        dummy_tensor, graph_reader=lambda _: {"ir": clean_ir, "hlo": clean_hlo}
    )
    assert audit_clean["status"] == "PASS" and not audit_clean["random_operations"], audit_clean

    # Positive control 1: HLO containing `rng-bit-generator` must fail
    bad_hlo = """
    HloModule IrToHlo.99
    ENTRY %IrToHlo.99 (seed.1: u64[2]) -> (f32[4,128]) {
      %seed.1 = u64[2]{0} parameter(0)
      %rng-bit-generator.4 = (u64[2]{0}, u32[4,128]{1,0}) rng-bit-generator(u64[2]{0} %seed.1), algorithm=rng_default
      %bitcast.5 = f32[4,128]{1,0} bitcast(u32[4,128]{1,0} %get-tuple-element(%rng-bit-generator.4), index=1)
    }
    """
    caught_hlo_rng = False
    try:
        audit_hlo_no_rng(
            dummy_tensor, graph_reader=lambda _: {"ir": clean_ir, "hlo": bad_hlo}
        )
    except RuntimeError as exc:
        caught_hlo_rng = "hlo:rng-bit-generator" in str(exc)
    assert caught_hlo_rng, "Failed to catch rng-bit-generator in HLO fixture"

    # Positive control 2: IR containing `aten::randn` or `aten::exponential` must fail
    bad_ir = clean_ir + "\n  %4 = f32[4,128]{1,0} aten::randn()\n  %5 = f32[4,128]{1,0} aten::exponential(%0)\n"
    caught_ir_rng = False
    try:
        audit_hlo_no_rng(
            dummy_tensor, graph_reader=lambda _: {"ir": bad_ir, "hlo": clean_hlo}
        )
    except RuntimeError as exc:
        caught_ir_rng = "ir:randn" in str(exc) and "ir:exponential" in str(exc)
    assert caught_ir_rng, "Failed to catch aten::randn and aten::exponential in IR fixture"

    # 3. Test replicated_randn_like sharding annotation
    recorded_annotations: list[tuple[ tuple[int, ...], tuple[Any, ...] ]] = []

    class MockMesh:
        mesh_shape = (8, 8)

    def mock_mark_sharding(
        t: Any, _mesh: Any, spec: tuple[Any, ...]
    ) -> Any:
        recorded_annotations.append((tuple(t.shape), spec))
        return t

    if torch is not None:
        sample_latents: Any = torch.zeros(8, 1024, 64)
    else:
        class _DummyDevice:
            type = "cpu"

        class _DummyTensor:
            shape = (8, 1024, 64)
            device = _DummyDevice()

            def dim(self) -> int:
                return len(self.shape)

            def _randn_like(self) -> "_DummyTensor":
                return _DummyTensor()

        sample_latents = _DummyTensor()
    noise = replicated_randn_like(
        sample_latents, mesh=MockMesh(), mark_sharding_fn=mock_mark_sharding
    )
    assert noise.shape == (8, 1024, 64)
    assert recorded_annotations == [((8, 1024, 64), (None, None, None))], recorded_annotations

    print(
        "RNG_HLO_AUDIT_SELFTEST "
        + json.dumps(
            {
                "ok": True,
                "lcg_verified_offsets": marker_offsets,
                "clean_hlo_false_positives": 0,
                "caught_hlo_rng_bit_generator": caught_hlo_rng,
                "caught_ir_randn_and_exponential": caught_ir_rng,
                "replicated_sharding_spec": recorded_annotations[0][1],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("PASS: All xla_spmd_rng_hlo_audit self-tests and mutation checks succeeded.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="PyTorch/XLA SPMD RNG consistency probe and static HLO/IR RNG auditor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--verify-lcg",
        type=int,
        default=None,
        metavar="INITIAL_SEED",
        help="Print the XLA sync LCG seed progression starting from INITIAL_SEED.",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Number of LCG sync steps to print with --verify-lcg (default: 8).",
    )
    ap.add_argument(
        "--probe-spmd",
        action="store_true",
        help="Run the live TPU SPMD RNG consistency probe across all workers.",
    )
    ap.add_argument(
        "--mesh",
        default="8x8",
        help="SPMD mesh shape for --probe-spmd, e.g. '64' or '8x8' (default: 8x8).",
    )
    ap.add_argument(
        "--spec",
        default="data,fsdp",
        help="Consumer partition spec for --probe-spmd (default: data,fsdp).",
    )
    ap.add_argument(
        "--replicated-noise",
        action="store_true",
        help="With --probe-spmd: pin drawn noise to replicated (None, None) before consumer.",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in CPU self-tests and exit.",
    )
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if args.verify_lcg is not None:
        chain = [
            {"step": i, "seed": advance_xla_root_seed(args.verify_lcg, i)}
            for i in range(args.steps + 1)
        ]
        print(
            "LCG_CHAIN "
            + json.dumps(
                {
                    "initial_seed": args.verify_lcg,
                    "recurrence": "s' = (1012031 + 7012063 * s) mod 2^64",
                    "chain": chain,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if args.probe_spmd:
        res = probe_spmd_rng_consistency(
            mesh_spec=args.mesh,
            consumer_spec=args.spec,
            force_replicated_noise=args.replicated_noise,
        )
        print("RNGPROBE " + json.dumps(res, sort_keys=True), flush=True)
        return 0 if res["consistent"] else 1

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
