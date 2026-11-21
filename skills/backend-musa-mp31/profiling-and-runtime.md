# Profiling, Runtime & Debugging on MUSA

---

## 1. Container GPU Passthrough & Environment Variables

Pass specific GPUs via CDI (`--device mthreads.com/gpu=gpuN`) and place JIT/compiler caches on node-local `/tmp` rather than shared network filesystems:

```bash
docker run --rm \
  --device mthreads.com/gpu=gpu0 \
  --device mthreads.com/gpu=gpu1 \
  --ipc=host --shm-size=64g \
  -v /tmp/musa_jit_cache:/root/.tilelang/cache \
  -e PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True \
  "$IMAGE" bash -lc "..."
```

| Variable | Recommended Usage | Notes |
|---|---|---|
| `MUSA_VISIBLE_DEVICES` / `MTHREADS_VISIBLE_DEVICES` | Set explicitly per rank/process | `torch_musa` typically reads `MUSA_VISIBLE_DEVICES`; container/platform runtimes may read `MTHREADS_VISIBLE_DEVICES`; compatibility shims may inspect `CUDA_VISIBLE_DEVICES`. |
| `PYTORCH_MUSA_ALLOC_CONF` | `expandable_segments:True` | Reduces allocator fragmentation and false OOMs on long-sequence / high-VRAM workloads. |
| `MUSA_LAUNCH_BLOCKING` | `1` for tests/debug; unset for timing | Synchronizes launches to surface async OOB faults at the offending op. Never enable during steady-state timing windows. |

### Baseline Environment Snapshot

```bash
hostname && mthreads-gmi -L && mthreads-gmi
mcc --version
python3 -c "import torch, torch_musa; print(torch.__version__, torch_musa.__version__, torch.musa.is_available(), torch.musa.device_count())"
```

Distinguish four verification states when bringing up an environment: *binary present*, *shared library loadable*, *minimal GPU smoke pass*, and *target workload pass*.

### Layered Bring-Up Checks

1. **Container & host runtime**: Verify container execution before GPU passthrough. If rootless Docker fails with `unable to join session keyring: disk quota exceeded` despite free disk space, check `/proc/key-users` and `/proc/sys/kernel/keys/maxkeys`.
2. **Dynamic libs & compiler toolchain**: Use `ldd`/`readelf` for ABI/symbol mismatches. If `mcc -v` picks a host GCC version whose C++ headers are absent, align `--gcc-toolchain` / `-isystem` / `-L` or build inside a self-contained SDK container.
3. **Single-GPU smoke**: Check VRAM allocation, BF16/FP32 GEMM, backward, sync, and actual Attention backend selection (confirm no silent fallback to eager math).
4. **Multi-GPU MCCL**: Run a minimal 2-GPU `all_reduce` + `barrier` smoke test before full-scale training. On hangs, check per-rank OOM, shape mismatch, or collective ordering divergence first.

---

## 2. Profiling Tools & Minimal Commands

| Tool | Use |
|---|---|
| `mthreads-gmi` | Device model, driver, VRAM usage, utilization, concurrent processes |
| Torch-MUSA Profiler | End-to-end op breakdown, CPU/Python dispatch gaps, implicit host syncs |
| `msys` (Moore Perf System) | System-wide CPU/GPU timeline, MUSA APIs, kernels, memcpy, stream syncs, rank skew |
| `mcu` (Moore Perf Compute) | Single-kernel hardware counters: occupancy, compute/memory throughput, cache hit rates, Roofline |
| `muPTI` | Programmatic activity/event tracing API |
| `mt-compute-sanitizer` | Global memory bounds checking, API errors, leaks (`--leak-check=full`); requires matching Driver/SDK |
| MUSA-GDB / Snapshot | Single-GPU minimal-repro debugging (verify availability in the target SDK) |

### Torch-MUSA Profiler Snippet

```python
import torch

dev_act = getattr(
    torch.profiler.ProfilerActivity,
    "MUSA",
    torch.profiler.ProfilerActivity.PrivateUse1,
)
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, dev_act],
    record_shapes=False,
) as prof:
    run_one_step()
    torch.musa.synchronize()

evts = prof.key_averages()
gpu_ms = sum(getattr(e, "self_device_time_total", 0) for e in evts) / 1e3
```

Compare wall-clock step time against summed device kernel time: a large gap usually points to Python/dispatch bubbles, implicit CPU syncs (`.item()`, `.cpu()`, dynamic host control flow), or collective stragglers.

### `msys` and `mcu` Usage

Confirm tool paths and versions with `--version`, then capture minimal traces:

```bash
# System timeline + GPU kernel summary
"$msys" profile --device="$device" --trace=musa -o /tmp/system "$app"
"$msys" stats --report musa_gpu_kern_sum /tmp/system.msys-rep

# Targeted kernel counters (restrict to 1-2 representative kernels)
"$mcu" --devices "$device" --kernel-name 'regex:.*vector_add.*' -o /tmp/kernel "$app"
```

- **Prevent SSH PTY stalls in `msys`**: Wrap non-interactive/SSH invocations with `timeout --foreground ... </dev/null` if job control triggers `SIGTTIN`.
- **Never use `mcu` for timing comparisons**: Kernel replay distorts wall-clock execution; always validate speedups with uninstrumented steady-state runs.
- **Counter conflicts**: If `mcu` reports counter allocation failures, check whether another profiler or monitoring daemon (e.g., MTDCGM) holds hardware counters on the same GPU.

---

## 3. Asynchronous OOB & Silent Error Triage

Under asynchronous launch, illegal addresses or OOB writes may surface late—or intermittently on specific ranks—while a smoke test still exits `0`:

1. Reproduce with fixed seeds and full target shapes under `MUSA_LAUNCH_BLOCKING=1`.
2. Check all `> 2 GiB` tensors for `int32` index wraparound in custom MUSA C++ (`int64_t`) and code-generated kernels.
3. Explicitly check `musaGetLastError()` and `musaDeviceSynchronize()` in C++ wrappers (raw `ctypes` kernel launches remain asynchronous even when PyTorch blocking flags are set).
4. Ensure no concurrent process edits shared-filesystem `.py` files while TileLang or Python JIT is compiling.

### Official Documentation

- [MUSA SDK Release Notes](https://docs.mthreads.com/musa-sdk/version-5.2.0/releasenote/)
- [MUSA Docker Guide](https://docs.mthreads.com/musa-sdk/version-5.2.0/docker_guide/)
- [MSYS User Guide](https://docs.mthreads.com/mooreperf/mooreperf-doc-online/moore_perf_system/user_guide/)
- [MCU User Guide](https://docs.mthreads.com/mooreperf/mooreperf-doc-online/moore_perf_compute/user_guide/)
- [MUSA Compute Sanitizer User Guide](https://docs.mthreads.com/mooreperf/mooreperf-doc-online/musa_compute_sanitizer/user_guide/)
