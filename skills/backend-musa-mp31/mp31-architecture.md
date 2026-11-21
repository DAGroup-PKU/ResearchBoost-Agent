# MP31 Architecture & Kernel Development (MUSA C++ / TileLang-MUSA)

---

## 1. MP31 / MTT S5000 Hardware Architecture

Always confirm exact device properties on the target SKU via `musaGetDeviceProperties()` (see [examples/musa_vector_add.cu](examples/musa_vector_add.cu)). Typical MTT S5000 (MP31) properties:

| Property | Typical MP31 / MTT S5000 Value | Kernel & Occupancy Implications |
|---|---|---|
| Compute Capability / Target | `3.1` (`--offload-arch=mp_31`) | Unified offload target for MP31 kernels |
| Multiprocessors (MPs) | `60` MPs (S5000); max `3,072` threads (`96` warps) & `256` blocks per MP | Align grid sizes to multiples of MP count to reduce tail-wave idle time |
| Execution Hierarchy | Warp = `32` threads; SQMMA Squad = `4` warps (`128` threads); max `1,024` threads/block | Structure warp-specialized producer/consumer pipelines around 128-thread squads |
| Register File | `131,072` × 32-bit registers per MP (and per block) | ~`42` regs/thread at 3,072 threads/MP; up to `256` regs/thread at 512 threads/MP |
| Shared Memory (SMEM) | `192 KiB` (`196,608 B`) per MP / block | Large enough to keep multi-stage matrix tiles resident on-chip |
| **SMEM Bank Geometry** | **64 banks (256 B per row)** | **Differs from NVIDIA 32-bank / 128 B**; pad manual SMEM layouts for 64 banks (e.g., stride `65` for 32-bit) |
| L2 Cache | `60 MiB` (`62,914,560 B`) | Coalesce wide reductions and streaming stores to full 128 B cache lines |
| Global Memory (HBM) | `80 GiB` (`640-bit` bus), `asyncEngineCount = 2` | Vectorize aligned loads/stores to 128-bit (`uint4`) |

### Local Roofline Calibration

Before classifying a kernel as compute- or bandwidth-bound, run quick microbenchmarks on the target node and SDK to measure practical ceilings:
- **Compute**: `muBLAS` GEMM (BF16/FP16/FP32) and `MATE` Dense FlashAttention.
- **Memory**: Streaming 128-bit (`uint4`) copy/triad for sustained HBM bandwidth.

---

## 2. Native Tensor Core (TCE) & Async Copy (TME) Primitives

Check the installed MUTLASS / MuTe headers (`mute/arch/`) for exact template signatures on the active SDK:

- **SQMMA** (128-thread Squad = 4 warps, `shared × shared` and `reg × shared`, `mute/arch/mma_mp31_sqmma.hpp`):
  - Native `M × N` shapes: `128×{128,64,32}`, `64×{128,64,32,16}`, `32×{128,64,32}`, `16×64`.
  - Native `K` steps & dtypes: `TF32` (`K=8/16/32`), `BF16/FP16` (`K=16/32/64`), `FP8 (E4M3/E5M2)` / `INT8/UINT8` (`K=32/64/128`), plus mixed-precision `BF16/FP16 × INT8/INT4` (FP32 / INT32 accumulators).
- **WMMA** (single 32-thread warp, `frag × frag`):
  - Native tiles: `16×16×{16,32,64}`, `16×8×{4,8,16}` across `BF16/FP16/TF32/FP8/INT8/UINT8` and mixed precision.
- **TME (Tensor Memory Engine)** (`mute/arch/copy_mp31_tme.hpp`, `copy_mp31_desc.hpp`):
  - 1D–5D asynchronous tile loads/stores (`__musa_tme_ld_tile_{1d..5d}`, `__musa_tme_st_{1d..5d}`) and prefetches (`__musa_prefetch_only`).
  - 64-bank Swizzle parameters: `Granularity ∈ {NONE, B16, B32, B64}`, `Stride ∈ {B32, B64, B128, B256}`, `Line ∈ {B128, B256}`.
  - Cache hints: `CacheHint ∈ {CACHE_NONE, CACHE_ONCE, CACHE_NORMAL, CACHE_PERSIST}`.

---

## 3. MUSA C/C++ (`mcc`) Kernel Priors

### Toolchain Consistency & Smoke Build

Pair `mcc`, headers, device libraries, and `libmusart` from the same SDK release (mixing host compiler and container headers causes device intrinsic mismatches). Smoke test: [examples/musa_vector_add.cu](examples/musa_vector_add.cu).

```bash
export MUSA_VISIBLE_DEVICES="$gpu_index"
export MTHREADS_VISIBLE_DEVICES="$gpu_index"
"$sdk/bin/mcc" --musa-path="$sdk" -x musa --offload-arch=mp_31 -O3 \
  -fmusa-flush-denormals-to-zero \
  examples/musa_vector_add.cu -L"$sdk/lib" -lmusart -o /tmp/vector_add
LD_LIBRARY_PATH="$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" /tmp/vector_add
```

### Architecture-Rooted Coding Rules

- **Always pass `-fmusa-flush-denormals-to-zero`**: Without FTZ, transcendental math (such as `tanhf` and GELU) can take slow IEEE denormal-safe paths.
- **Design for 64-bank SMEM**: Pad or swizzle shared memory for 64 banks / 256 B rows instead of copying NVIDIA 32-bank constants.
- **Enforce `int64_t` offsets on `> 2 GiB` tensors**: Long-sequence activation and QKV buffers routinely exceed `INT32_MAX` bytes; cast linear offset arithmetic to `int64_t` before multiplication.
- **Mark read-only pointers `const T* __restrict__`**: Preserves read-only cache paths and avoids unnecessary load/store aliasing stalls (especially on in-place vs. out-of-place strided kernels).
- **Cover >= 128 B/CTA in wide reductions**: Map at least one full cache line per CTA (e.g., >= 64 BF16 columns = `8 × uint4`) to avoid L2 thrashing.
- **Publish dynamic `.so` artifacts atomically**: Key JIT caches by SHA256 of source + flags and write via temp file + `os.replace()` to avoid multi-rank races.

---

## 4. TileLang-MUSA Priors

The `tilelang_musa` package imports as `tilelang` with `target="musa"`. Smoke test: [examples/tilelang_vector_add.py](examples/tilelang_vector_add.py).

- **Inspect lowered MUSA C**: Always check `kernel.get_kernel_source()` on hot kernels to verify that TME async copies, SQMMA, async barriers, and burst loads/stores actually lowered as intended.
- **Keep 64-bit index promotion for `> 2 GiB` tensors**: Include a `> 2 GiB` shape under `MUSA_LAUNCH_BLOCKING=1` in kernel self-tests.
- **Freeze `.py` sources during JIT**: TileLang uses `inspect.getsource()`; editing Python files on shared storage while multi-GPU or background jobs compile causes spurious frontend failures.
- **Stage complexity**: Validate a synchronous baseline first, then A/B test swizzling, multi-stage pipelining, and warp specialization on real shapes.

---

## 5. CUDA → MUSA Migration Checklist

- **Classify CUDA kernels before porting**:
  - *Portable CUDA C++* (standard SIMT kernels without architecture-specific PTX): generally ports cleanly to MUSA C++ or TileLang-MUSA.
  - *SM90/SM100-specific kernels* (`tcgen05`, TMEM, `CUtensorMap`, CuTe/WGMMA PTX): cannot be translated mechanically; replace with `muBLAS` / `MATE` or rewrite around MP31 MUTLASS/MuTe TME + SQMMA.
- **Audit PyTorch integration points**: Check for hardcoded `x.is_cuda`, `get_device_capability`, eager `nvcc` extension imports, stream handles (`torch.musa.current_stream(dev).musa_stream`), device guards, and gradient accumulation semantics.

### Official Documentation

- [MUSA Thread Hierarchy](https://docs.mthreads.com/musa-sdk/version-5.1.0/programming_guide/programming_model/thread_hierarchy/)
- [MUSA C++ Extensions](https://docs.mthreads.com/musa-sdk/version-5.2.0/programming_guide/musa_cpp_syntax/intro_to_musa_cpp/)
- [MUSA Tensor Programming](https://docs.mthreads.com/musa-sdk/version-5.2.0/programming_guide/musa_cpp_syntax/tensor_programming/)
- [MUTLASS (MooreThreads)](https://github.com/MooreThreads/mutlass)
- [TileLang-MUSA MP31 Examples](https://github.com/tile-ai/tilelang-musa/tree/main/testing/musa/mp31)
