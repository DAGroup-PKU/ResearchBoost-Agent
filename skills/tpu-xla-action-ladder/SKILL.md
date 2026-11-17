---
name: tpu-xla-action-ladder
description: Optimization action ladder and numerical pitfalls for Cloud TPU (PyTorch/XLA and JAX/Flax/Pallas) - symptom -> first suspect, L0 static shapes and dispatch -> L1 Splash Attention and GQA -> L2 ICI mesh and sharding audit -> L3 static gather/scatter, scan and token compaction -> L4 Pallas kernels, mapped to N0-N5 tiers.
---

# TPU and XLA action ladder

Use when: choosing what to try next on Cloud TPU after profiling (`tpu-xprof-and-roofline`), or debugging XLA
recompilations, collective stalls, HBM fragmentation and silent SPMD numerical bugs. Every action lists its typical
numerics tier (`N0`-`N5`) as defined in `train-opt-numerics-validation`; anything above `N2` requires owner sign-off or
is rejected (`N5`), and **the task's frozen `bench.py check` is the sole verdict**.

## Symptom -> first suspect on XLA

Fix missed fast paths and graph breaks (`L0`/`L1`) before rewriting modules (`L3`) or authoring Pallas kernels (`L4`):

| Symptom in XProf or `metrics_report` | First suspect |
|---|---|
| `UncachedCompile > 0` or sawtooth step times in steady state | variable sequence lengths, dynamic boolean indexing `x[mask]`, or Python float hyperparameters (such as warmup LR) baked as HLO constants |
| `aten::_local_scalar_dense`, `aten::nonzero`, `TransferFromDeviceData` | mid-step `.item()`, `if tensor:` control flow, or an op lacking an XLA lowering falling back to CPU |
| Large `Host / Idle Gaps`, `ExecuteComputation` high, slow optimizer | per-parameter Python optimizer/EMA loop, CPU scalar tensors in `param_groups`, or uncoordinated Python GC pauses across hosts |
| `Exposed Collectives` > 15% of step time | `tp > 4` crossing single-host ICI Tray bounds, high `fsdp` when weights already fit per-chip HBM, or `jax.lax.map` without `shard_map` issuing cross-chip `all-gather`s inside the loop |
| `SplashAttention` slower at `128 * odd` sequence lengths (e.g. `384`, `640`, `1152`) | block size clamps to `gcd(seq_len, 512) = 128`, under-filling `v6e` `256x256` MXU tiles |
| Huge `LM Head & Cross-Entropy` time and temporary HBM spike | materializing full `[B, S, V]` logits over padding/unsupervised tokens instead of gathering `loss_mask == 1` positions and chunking vocabulary |
| Step throughput *drops* after increasing microbatch past ~80% HBM | XLA silently enables compiler auto-rematerialization or spills VMEM (`MSA`) to fit peak memory |
| `torch.compile(backend="openxla")` slower than `torch_xla` LazyTensor | Dynamo guard overhead and graph splits break cross-op XLA fusion; stick to single-graph LazyTensor + `torch_xla.sync()` |

## L0 switches, static shapes and host dispatch

| Action | Tier | Prior |
|---|---|---|
| Pad/bucket sequences to static multiples (`512` on `v6e`) and pad the final incomplete batch with `loss_mask = 0` | N0 | eliminates shape-driven 100-500 s XLA recompilations; verify masked positions contribute zero loss and gradient |
| Remove mid-step `.item()`, `print(tensor)` and `jax.debug.callback`; read metrics on host every `K` steps | N0 | restores host-ahead tracing overlap with device execution |
| Store scheduled LR / weight decay in 0-D device tensors (or JAX traced arrays) instead of Python `float`s | N0 | updating a Python `float` in `param_groups` mutates the HLO constant hash and recompiles every warmup step |
| Disable automatic Python GC (`gc.disable()`) and run `gc.collect()` at fixed step intervals across all workers | N0 | prevents single-host GC stalls from blocking every chip at synchronous ICI `all-reduce` barriers |
| `torch._foreach_*` optimizers and EMA (keep `step` counter on host or 0-D XLA tensor, never CPU `torch.tensor` with `capturable=True`) | N1 | collapses hundreds of per-parameter HLO nodes into multi-tensor ops; rounding order differs slightly from loop |
| Async collective overlap flags in `LIBTPU_INIT_ARGS` (such as async `all-gather` / `reduce-scatter`) | N0/N1 | verify flag names with `tpu_preflight_gate.py --check-flags` and confirm overlap in XProf before keeping |
| Explicit `bfloat16` trunk matmuls vs `float32` (`Precision.HIGH`/`HIGHEST` -> `DEFAULT`) | **N2** | `3x`-`6x` MXU gain when upstream accidentally ran multi-pass FP32 emulation, **only** if precision change is signed off; keep residuals, RoPE, norms, loss reductions and optimizer states in `float32` |

## L1 backends and collective-free layouts

- **Pallas `SplashAttention` (`N1`)**: replaces $O(S^2)$ HBM attention matrix materialization with VMEM-pipelined
  flash tiles. Pad sequence lengths and set `block_q` / `block_kv` to multiples of `512` on `v6e` (`256` minimum) so
  tiles stay `>= 256` on the `256x256` MXU; `SplashAttention` requires `Precision.DEFAULT`. Keep the upstream attention
  as a bitwise stand-in switch (`train-opt-numerics-validation`) while verifying other `N0` changes.
- **5D grouped GQA einsum (`N1`)**: replace explicit `repeat(k, G)` / `repeat(v, G)` (which copies KV tensors in HBM/VMEM)
  with a native 5D `[B, T, H_kv, G, D]` contraction or native GQA Splash kernel.
- **`shard_map`-wrapped microbatch `lax.map` for frozen towers (`N0`/`N1`)**: chunking a high-resolution frozen encoder
  with bare `jax.lax.map` under GSPMD causes the compiler to insert cross-chip `all-gather` collectives inside the
  generated `while_loop`. Wrap the per-shard microbatch loop inside `jax.shard_map` over the data/FSDP axis so each
  chip loops strictly over its local shard.

## L2 mesh topology, sharding verification and remat

- **Minimal viable parallelism**: keep `tp <= 4` (within one 4-chip host or 8-chip Tray; never cross DCN). When model
  weights + optimizer states fit per-chip HBM (or under ICI ZeRO-1 optimizer sharding), set `fsdp = 1` to avoid
  per-layer `all-gather` / `reduce-scatter` exposure. On `v6e-64` (`4x4` hosts), cut sub-slices along physical `2x4`
  8-chip Tray boundaries (`TPU_HOST_BOUNDS=2,4,1`) and `unset MEGASCALE_*` on single-slice jobs.
- **Selective `jax.checkpoint` (`remat`) (`N0`/`N1`)**: full-layer remat adds ~33% compute; saving matmul outputs
  (`jax.checkpoint_policies.dots_with_no_batch_dims`) while recomputing cheap VPU elementwise/norm chains keeps HBM in
  the `45%-75%` sweet spot without triggering XLA's silent auto-remat. Pass only differentiable activations as
  `checkpoint` arguments and capture static `cos`/`sin`/`mask` via closure.
- **Audit PyTree sharding on virtual CPU devices before running on TPU**: in JAX/Flax, building leaf paths with
  `"/".join(str(k) for k in path)` turns `DictKey('kernel')` into `"['kernel']"`, so `name.endswith("kernel")` returns
  `False` and silently falls back to `PartitionSpec()` (100% replicated across all chips, inflating persistent HBM by
  `fsdp`x with no warning). Audit any `(dp, fsdp, tp)` mesh in under a second without a TPU:

  ```bash
  python <this card's directory>/scripts/jax_cpu_sharding_audit.py \
      --dp 4 --fsdp 16 --tensor 1 --factory pkg.mod:build_params_and_specs \
      --fail-on-replicated-mib 4.0 [--allow-replicated-regex "norm|bias|emb"]
  ```

## L3 XLA-friendly graph rewrites

- **Static gather/scatter instead of boolean masks (`N0`/`N1`)**: replace `x[mask]` and `torch.nonzero(mask)` with
  fixed-shape `where(mask, x, 0)` for dense blends, or `argsort(~mask)` + static-capacity `take` / `scatter_add` for
  sparse routing (MoE / Mixture-of-Transformers).
- **Supervised-token compaction + vocab-chunked cross-entropy (`N1`)**: when `loss_mask == 1` covers a fraction of the
  sequence (for example SFT response tokens or interleaved multimodal prefixes), gather the supervised hidden states
  (padded to a static multiple of `256` or `512`) *before* the `[H, V]` LM Head projection, and accumulate log-sum-exp
  across vocabulary chunks (`V_chunk = 8192` or `16384`) instead of materializing full `[B, S, V]` FP32 logits in HBM.
- **Structural `jax.lax.scan` (`N0`/`N1`)**: fold homogeneous Transformer blocks and gradient-accumulation microbatches
  into `jax.lax.scan` to shrink HLO graph size and compile time by `5x-15x` and reduce DMA descriptor pressure.

## L4 Pallas / Mosaic TPU kernels

Only move to Pallas after `tpu_xplane_decompose.py` shows a genuine compute or VMEM fusion opportunity not covered by
XLA's native fusion or `SplashAttention`:

- **TPU kernel invariants**: accumulate reductions (`softmax`, `RMSNorm`, cross-entropy, attention rescaling) in `float32`
  VPU registers; tile minor dimensions to multiples of `(8, 128)` (`(256, 256)` for `v6e` MXU matmuls); pipeline HBM <->
  VMEM copies with `pltpu.PrefetchScalarGridSpec`.
- **Custom backward and HLO verification**: register explicit `jax.custom_vjp` backward kernels tested at real call-site
  shapes, keep the reference op behind a stand-in switch (`train-opt-numerics-validation`), and inspect lowered HLO at
  `T0` to prove `tpu_custom_call` is emitted (silent fallback to decomposed HLO ops looks identical functionally).

## Silent numerical and determinism traps on TPU

1. **2D SPMD `torch.randn_like` backward sharding corruption (`N5` bug if unguarded)**: on a multi-axis mesh (such as
   `data x fsdp`), feeding an unannotated `noise = torch.randn_like(x)` into a consumer sharded on `("data", "fsdp", ...)`
   lets GSPMD propagate consumer sharding backward into `rng-bit-generator`. The partitioned RNG feeds the forward/backward
   graph a *different* random tensor from the one returned to the host, stalling diffusion / flow-matching training
   while host inspection of `noise` looks clean. Pin drawn noise to replicated `(None, ...)` immediately before its
   consumer, or fold JAX PRNG keys deterministically per `(step, global_sample_idx)`.
2. **False-positive RNG state advance on `torch_xla.sync()`**: `torch_xla` advances its root seed via the LCG
   `s' = (1012031 + 7012063 * s) mod 2^64` on every `sync()` call even when the compiled graph contains zero random ops.
   At `T0` (`train-opt-numerics-validation`), verify determinism by scanning lowered HLO for `rng-bit-generator` /
   `philox` / `three_fry` rather than comparing host `xm.get_rng_state()` across different sync counts:

   ```bash
   python <this card's directory>/scripts/xla_spmd_rng_hlo_audit.py --self-test
   python <this card's directory>/scripts/xla_spmd_rng_hlo_audit.py --verify-lcg 4396 --steps 5
   ```
3. **Mask before `exp` / `log` / `sqrt` under static padding**: in static-shape XLA graphs, padded slots still execute
   algebraically; `where(mask, exp(x), 0.0)` produces `0.0 * inf = NaN` in backward when padded `x` overflows. Always
   clamp or mask inputs *inside* the nonlinear op (`exp(where(mask, x, 0.0))`, `sqrt(sum_sq + 1e-8)`).
4. **BF16 `1 + dt` step-size underflow**: `bfloat16` has 7 mantissa bits (`ulp(1.0) = 2^-7 = 0.0078125`). Computing
   ODE/flow-matching updates or EMA decays `1.0 - alpha` in `bf16` rounds small steps directly to `1.0`; keep step
   interpolation and residual updates in `float32`.
