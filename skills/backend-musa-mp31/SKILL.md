---
name: backend-musa-mp31
description: Hardware backend priors for Moore Threads MUSA GPUs (MTT S5000 / MP31) — CUDA-to-MUSA tool/library mapping, L0–L4 action-ladder overrides, MP31 architecture (64-bank SMEM, SQMMA/TME), msys/mcu profiling, and cross-platform numerics attribution. Read ONLY when running on or porting to Moore Threads MUSA GPUs; skip on NVIDIA/CUDA runs.
---

# Backend priors: Moore Threads MUSA (MP31)

Use when: optimizing, profiling, or porting training workloads on **Moore Threads MUSA GPUs (MP31 / MTT S5000)**. Read alongside [`train-opt-tiered-profiling`](../train-opt-tiered-profiling/SKILL.md), [`train-opt-action-ladder`](../train-opt-action-ladder/SKILL.md), [`train-opt-measurement`](../train-opt-measurement/SKILL.md), and [`train-opt-numerics-validation`](../train-opt-numerics-validation/SKILL.md). **Skip on NVIDIA/CUDA runs.**

These are hardware-backend priors to re-verify on the target node and SDK; the verdict is always the task's own `bench.py check`.

## CUDA → MUSA stack mapping

When applying the other `train-opt-*` cards on MUSA, substitute the corresponding platform tools and libraries:

| CUDA / NVIDIA concept | Moore Threads MUSA / MP31 equivalent |
|---|---|
| `nvidia-smi` | `mthreads-gmi` (`mthreads-gmi -L`) |
| `CUDA_VISIBLE_DEVICES` / `/dev/nvidiaN` | `MUSA_VISIBLE_DEVICES`, `MTHREADS_VISIBLE_DEVICES` / `--device mthreads.com/gpu=gpuN` |
| `CUDA_LAUNCH_BLOCKING=1` | `MUSA_LAUNCH_BLOCKING=1` |
| `PYTORCH_CUDA_ALLOC_CONF` | `PYTORCH_MUSA_ALLOC_CONF` (e.g., `expandable_segments:True`) |
| `torch.profiler` (`ProfilerActivity.CUDA`) | `ProfilerActivity.MUSA` with fallback to `ProfilerActivity.PrivateUse1` |
| `nsys` / `ncu` / `compute-sanitizer` | `msys` (Moore Perf System) / `mcu` (Moore Perf Compute) / `mt-compute-sanitizer` |
| `cuBLAS` / `cuDNN` / FlashAttention / `NCCL` | `muBLAS` / `muDNN` / `MATE` (`mate.flash_attn_varlen_func`) / `MCCL` |
| `nvcc` / CUTLASS (CuTe) / Triton | `mcc --offload-arch=mp_31` / `MUTLASS` (`MuTe`) / `TileLang-MUSA` (`target="musa"`) |
| 32-bank SMEM (128 B/row), WGMMA + TMA | **64-bank SMEM (256 B/row)**, **128-thread SQMMA Squad + TME** |

## Overrides to `train-opt-action-ladder`

- **L0 switches & runtime**:
  - Verify `torch.compile` and MUSA graph capture actually produce compiled/graph kernels in traces before relying on them; silent fallback or regressions must be caught by counters just as on CUDA.
  - Set `PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True` when allocator fragmentation limits batch size or forces activation checkpointing.
  - **Communication / overlap**: Check whether the installed `MCCL` / FSDP2 build runs collectives on dedicated Copy Engines (CE) or on compute MPs. If collectives compete for MPs, aggressive compute–communication overlap or fine-grained sharding can regress step time — A/B test coarse bucketing and overlap off.
  - **Parallelism (DP vs. CP/TP)**: Measure single-pair P2P bandwidth against aggregate all-to-all interconnect bandwidth before introducing sequence/context parallelism; when single-pair P2P is much narrower than full interconnect, keeping sequences on one GPU via pure DP + memory optimization often wins.
  - When using gradient accumulation, defer `reduce_scatter` / gradient synchronization to the final micro-batch.
- **L1 libraries & fast paths**:
  - Benchmark `muBLAS`, `muDNN`, and `MATE` (MP31 FlashAttention) before writing custom dense kernels; avoid rewriting dense FlashAttention if `MATE` already approaches the measured hardware ceiling.
  - Benchmark fused library calls (such as `addmm(beta=1.0)`) per shape rather than substituting globally — some square/near-square shapes can regress relative to separate ops.
  - Record actual kernel names and guard conditions in the task's fast-path inventory.
- **L4 custom kernels (`mcc` / MUTLASS / TileLang-MUSA)**:
  - Keep `mcc`, headers, device libraries, and `libmusart` from the same SDK release; always pass `--offload-arch=mp_31 -fmusa-flush-denormals-to-zero`.
  - Do not carry over NVIDIA 32-bank SMEM padding constants — MP31 has **64 banks (256 B/line)** and 192 KiB SMEM per MP.
  - Enforce 64-bit offsets (`int64_t` in MUSA C++, 64-bit index promotion in TileLang) for any tensor that can exceed `2 GiB` (`INT32_MAX`).
  - For TileLang-MUSA, inspect `kernel.get_kernel_source()` to confirm TME/SQMMA/async-barrier lowering, and never modify `.py` files on a shared filesystem while multi-GPU JIT jobs are compiling (`inspect.getsource()` race).
  - Details and hardware specs: [mp31-architecture.md](mp31-architecture.md).

## Overrides to `train-opt-tiered-profiling` and `train-opt-measurement`

- **Calibrate locally measured peaks**: Before computing Roofline or MFU, measure `muBLAS` GEMM (per target dtype), `MATE` Dense Attention, and 128-bit streaming HBM bandwidth on the target node rather than using datasheet peak specs.
- **Profiler activity fallback**: Depending on `torch_musa` version, `torch.profiler` may register device events under `ProfilerActivity.PrivateUse1` instead of `ProfilerActivity.MUSA`; aggregate GPU time from `self_device_time_total`.
- **`msys` / `mcu` hygiene**: Wrap `msys` with `timeout --foreground ... </dev/null` in non-interactive/SSH shells to avoid `SIGTTIN` stalls; filter `mcu` to 1–2 representative kernels and never report replayed timings as throughput numbers.
- **Async fault isolation vs. timing**: `exit 0` does not rule out asynchronous OOB writes. Reproduce tests and failures under `MUSA_LAUNCH_BLOCKING=1` (and call explicit `musaDeviceSynchronize()` + `musaGetLastError()` around raw `ctypes` kernel launches), then unset `MUSA_LAUNCH_BLOCKING` for all timing windows.
- Tool commands and container setup: [profiling-and-runtime.md](profiling-and-runtime.md).

## Overrides to `train-opt-numerics-validation`

- **Precision switches are N2**: Enabling reduced-precision math modes (such as TF32) changes accumulation precision; verify both forward outputs and backward gradient norms under the task's numerics gate before keeping them.
- **Cross-platform porting attribution (CUDA → MUSA)**: When establishing a MUSA baseline from a CUDA reference, separate vendor-library rounding from custom-kernel errors using three stages:
  1. **Stage 0 (CUDA golden)**: Frozen inputs, weights, RNG state, forward outputs, loss, grad norms, and discrete selections.
  2. **Stage 1 (stock MUSA libraries only)**: Standard PyTorch + `muBLAS` / `muDNN` with no custom kernels — establishes the natural cuBLAS-vs-muBLAS rounding envelope (including rare tie/boundary flips in low-precision argmax or top-k).
  3. **Stage N (custom MUSA kernels & distributed stack)**: Validate against both Stage 1 (isolating kernel bugs) and Stage 0 (end-to-end cross-platform parity).

## References

- [mp31-architecture.md](mp31-architecture.md) — MP31/MTT S5000 hardware properties, SQMMA/WMMA/TME primitives, `mcc` & TileLang-MUSA kernel rules, CUDA→MUSA migration checklist, official docs.
- [profiling-and-runtime.md](profiling-and-runtime.md) — container GPU passthrough, MUSA environment variables, environment snapshot, Torch-MUSA profiler snippet, `msys`/`mcu` usage, and async OOB triage.
- Minimal smoke tests: [examples/musa_vector_add.cu](examples/musa_vector_add.cu) and [examples/tilelang_vector_add.py](examples/tilelang_vector_add.py).
