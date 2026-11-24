---
name: tpu-xprof-and-roofline
description: Cloud TPU profiling, Roofline accounting and preflight validation - device completion barriers, narrow-window XProf .xplane.pb decomposition, Torch-XLA StepCounters, forward vs 8KN backward bandwidth floors, and host/libtpu preflight checks. Read before timing or profiling on TPU.
---

# TPU profiling, Roofline and preflight

Use when: formulating, timing or profiling a task on Cloud TPU (`v4`, `v5e`, `v5p`, `v6e`) under PyTorch/XLA or
JAX/Flax/Pallas, and after every accepted round on TPU. This card adapts the CUDA assumptions in
`train-opt-measurement`, `train-opt-tiered-profiling` and `kernel-utilization-analysis` to XLA and TPU hardware.
**Acceptance is always decided by the task's frozen `bench.py check`; the scripts here are for preflight validation,
trace attribution and Roofline pricing, never a substitute for the judge.**

## Preflight and valid-run signatures on TPU

`nvidia-smi` and CUDA UUIDs do not exist on Cloud TPU VMs. Before any baseline or candidate run, verify host health,
`/dev/vfio/*` device availability and `LIBTPU_INIT_ARGS` flag validity on every worker without initializing PJRT:

```bash
python <this card's directory>/scripts/tpu_preflight_gate.py \
    --profile v6e --check-flags "${LIBTPU_INIT_ARGS:-}" --rename-stale-lock

# Optional offline static audit of launch scripts (.sh / .py):
python <this card's directory>/scripts/tpu_preflight_gate.py \
    --audit-script "scripts/*.sh" --strict-audit
```

What the preflight check catches and what the task's valid-run signature list (`train-opt-measurement`) must enforce:

- **Device locks and zombie processes**: another process holding `/dev/vfio/*` or `/dev/accel*`, or an open file
  descriptor on `/tmp/libtpu_lockfile` (verify inode ownership before removing a 0-byte leftover lockfile).
- **Core-dump hazards**: `ulimit -c` must be `0`; an unhandled `libtpu` abort with unlimited core dumps writes tens of
  GB to `/` and leaves processes stuck in `D` state holding `/dev/vfio/*`.
- **Flag routing and validity**: TPU compiler flags (`--xla_tpu_*`, `--xla_enable_async_*`, `--xla_use_shardy`) belong
  in `LIBTPU_INIT_ARGS`, not `XLA_FLAGS`, and must stay byte-identical across baseline and candidate unless the flag
  itself is the declared switch. `tpu_preflight_gate.py` `mmap`-scans `libtpu.so` in under 100 ms to catch invalid
  flags before multi-host coordinator barriers time out.
- **Compilation cache hygiene**: on JAX multi-host runs, set `JAX_COMPILATION_CACHE_DIR` with
  `JAX_COMPILATION_CACHE_MAX_SIZE=-1` (avoid eviction thrashing); record cold vs warm cache in the run fingerprint.
- **Invalid-run counters inside the timing window**:
  - PyTorch/XLA (`torch_xla.debug.metrics.metrics_report()`): `UncachedCompile > 0` inside the steady-state window,
    any `aten::*` fallback op not present in the upstream baseline (for example `aten::_local_scalar_dense`,
    `aten::nonzero`), or unexpected `TransferFromDeviceData` mid-step syncs.
  - JAX (`jax.stages.Lowered` / compile logs): any re-lowering or shape-triggered recompilation after warmup, or a
    custom Pallas path silently falling back to reference HLO ops without `tpu_custom_call`.

## Device completion barriers

Both `torch_xla` and `jax.jit` dispatch asynchronously: Python returns as soon as the HLO execution is queued. Without
a device completion barrier at window boundaries, host wall clock measures Python dispatch (`< 2 ms/step`) and can
report impossible TFLOP/s exceeding physical peak:

- **Inside the step loop**: never insert blocking host readbacks (`.item()`, `float(loss)`, `print(tensor)`, unbuffered
  `jax.debug.callback`). In `torch_xla`, end each step with `torch_xla.sync()` (formerly `xm.mark_step()`) so host
  tracing of step `k + 1` overlaps device execution of step `k`.
- **At timing-window boundaries only**:
  - PyTorch/XLA: call `torch_xla.sync(wait=True); xm.wait_device_ops()` immediately before starting the window timer
    and immediately before stopping it.
  - JAX: assign `out = jax.block_until_ready(out)` on its own line *before* calling `time.perf_counter()` (do not put
    `block_until_ready(out)` and `perf_counter()` in the same tuple expression, where argument evaluation order can
    read the clock before blocking).
- **Warmup and interleaving**: warm up across every phase and bucket shape (`>= 20` steps, covering data-prefetch queue
  depth and compilation), then time steady-state windows interleaved as baseline -> candidate -> baseline on the same
  TPU slice (`train-opt-measurement`). Sanity-check that implied TFLOP/s does not exceed physical chip peak.

## Narrow-window XProf capture and Roofline-gap decomposition

Full-run XProf traces grow to several GB per host and stall the coordinator. Capture **only 3-5 steady-state steps**
(for example steps 15-18) via `jax.profiler.start_trace` / `stop_trace` or `torch_xla.debug.profiler.trace`, then run
the zero-dependency `.xplane.pb` decomposer (the TPU counterpart to `kernel-utilization-analysis`):

```bash
python <this card's directory>/scripts/tpu_xplane_decompose.py \
    --trace /path/to/worker0.xplane.pb --chip v6e --nsteps 3 \
    [--src-root /path/to/repo] [--vocab-size 128256] [--json-out xplane_report.json]
```

What `tpu_xplane_decompose.py` extracts without TensorBoard or `xprof` installed:

1. Parses `/device:TPU:0` (`Steps`, `XLA Modules`, `XLA Ops`) directly from wire-format protobuf.
2. Decodes per-op `flops` and the nested `memory_access_breakdown` payload to separate **HBM bytes** from on-chip
   **VMEM SRAM bytes** (counting VMEM traffic against HBM bandwidth understates Roofline efficiency).
3. Groups leaf HLO ops (excluding `while` / `conditional` / `call` containers) into eight categories: `Attention`,
   `Linear / MLP`, `LM Head & Cross-Entropy`, `Frozen Encoder`, `Optimizer & Grad Accum`, `Exposed Collectives`,
   `Elementwise / Norm / Other`, and `Host / Idle Gaps`.
4. Computes each category's achieved TFLOP/s, HBM GB/s, Roofline floor `max(flops / peak_tflops, hbm_bytes / peak_bw)`,
   and **`roofline_gap_ms = actual_ms - roofline_min_ms`**. Prioritize the category with the largest `roofline_gap_ms`,
   then route to `tpu-xla-action-ladder`.

## Hardware specs and the 8KN backward floor

Per physical chip (`1 jax.Device`; note that GCP names `v4-N` and `v5p-N` by TensorCore count = `2 * chips`, whereas
`v5e-N` and `v6e-N` count physical chips directly):

| Metric (per physical chip) | TPU v4 | TPU v5e | TPU v5p | TPU v6e (Trillium) |
|---|---|---|---|---|
| TensorCores / MXU array | 2 (`4x 128x128`/core) | 1 (`4x 128x128`) | 2 (`4x 128x128`/core) | 1 (`2x 256x256`) |
| Peak BF16 compute | 275 TFLOP/s | 197 TFLOP/s | 459 TFLOP/s | 918 TFLOP/s |
| Peak VPU FP32 compute | ~17 TFLOP/s | ~12 TFLOP/s | ~28 TFLOP/s | ~57 TFLOP/s |
| Usable HBM capacity | 32 GB | 16 GB | 95.7 GiB (96 GB) | 31.25 GiB (32 GB) |
| Peak HBM bandwidth | 1,200 GB/s | 819 GB/s | 2,765 GB/s | 1,640 GB/s |
| BF16 ridge point (`FLOP/B`) | 229 | 240 | 166 | **560** |
| Unidirectional egress ICI | 300 GB/s (3D Torus) | 200 GB/s (2D Torus) | 600 GB/s (3D Torus) | 400 GB/s (2D Torus) |

**Why backward `dW + FP32 grad_accum` dominates high-ridge TPUs (`v6e`):**
Forward (`Y = X @ W`) and activation gradient (`dX = dY @ W.T`) stream `2 K N` bytes of BF16 weights. Backward weight
gradient with master FP32 accumulation (`dW = X.T @ dY; acc_fp32 += dW`) reads and writes the FP32 buffer (`4 + 4 =
8 K N` bytes), cutting weight-side arithmetic intensity by `4x` and requiring `4x` larger flattened token batch `M` to
reach the `560 FLOP/B` ridge on `v6e`. Moreover, as `M -> inf`, asymptotic intensity is bounded by `K N / (K + N)` in
BF16; an unpacked projection with `K = N = 1024` peaks at `512 FLOP/B < 560` and stays HBM-bound at every batch size
until packed (`qkv` or `gate_up`). Price linear layers, GQA and token compaction before editing code:

```bash
python <this card's directory>/scripts/tpu_matmul_roofline.py \
    --chip v6e --M 2048 --hidden 2048 --inter 6144 \
    --vocab-size 128256 --supervised-frac 0.25 [--json-out roofline.json]
```

## Pitfalls

- **`v5p-64` vs `v6e-64` slice comparisons**: `v5p-64` has 32 Megacore chips (`88.5 TB/s` aggregate HBM BW, `95.7 GiB`
  per chip), while `v6e-64` has 64 single-core chips (`104.9 TB/s` aggregate HBM BW, `31.25 GiB` per chip, `4.0x` more
  BF16 FLOP/s). Bandwidth-bound workloads see only ~15-25% gain on `v6e-64` until arithmetic intensity is raised.
- **`8x128` VPU lane and `256x256` MXU tile padding**: VPU vector registers span 8 sublanes (second-minor dim) by 128
  lanes (minor dim). Unaligned hidden, head or vocabulary dimensions trigger on-chip `pad` / `relayout` ops and waste
  systolic cycles.
- **Accidental multi-pass FP32 matmul emulation**: MXU multipliers take BF16 inputs with FP32 accumulation. Passing
  FP32 activations/weights under `jax.lax.Precision.HIGH` or `HIGHEST` decomposes each matmul into `3` or `6` BF16
  passes (`~33%` or `~16.7%` throughput) and breaks Pallas `SplashAttention` lowering. Keep trunk matmuls in explicit
  `bfloat16` with `Precision.DEFAULT` (within the upstream contract; changing upstream precision is `N2`, see
  `train-opt-numerics-validation`).
- **Silent compiler auto-remat above ~80-85% HBM**: pushing batch size until HBM exceeds ~85% causes XLA's memory
  scheduler to silently rematerialize activations or spill scoped VMEM (`MSA`), dropping step throughput without OOM.
  Check XProf recomputation counts when throughput regresses at larger batches.
