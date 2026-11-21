---
name: ascend-accuracy-runtime-debug
description: Diagnose accuracy regressions, Ascend operator failures, NPU runtime errors, OOM or graph failures after an optimization.
---

# Ascend accuracy and runtime debugging

Use the original task's reference and frozen numerical contract; the RBA numerics card defines the accepted comparison. Keep the same inputs, seed, shape, dtype, model state, launch path and parallelism when comparing a baseline and a candidate. Gather the earliest failure rather than treating a later asynchronous device error as the root cause.

## Accuracy mismatch

1. Identify the first differing phase, layer, operator and output index. Compare intermediate tensors on a minimal reproducible input; distinguish NaN/Inf from finite mismatch and record absolute/relative error under the task's rules. Do not silently change precision, tolerances or the reference.
2. If errors cluster at the end of a tile, compare aligned and unaligned shapes. Check per-core ranges, valid element counts, byte alignment, GM bounds, actual DataCopy/DataCopyPad overload and UB buffer capacity. A tile tail and 32-byte DMA alignment are separate conditions.
3. If errors repeat at block boundaries, inspect strides, layout, broadcasting and reductions. For nondeterministic results inspect uninitialized data, races, ordering, random-state handling and distributed synchronization. Bitwise equality can be required when the task contract says so.
4. For training, locate the first nonfinite activation/gradient or mismatched step before inspecting later loss divergence. Check checkpoint, optimizer state and data partition. For inference, check prefill/decode state, KV cache and input/output contracts when relevant.

## Runtime, memory and graph faults

- Record the first error, rank/device, timestamps and versions. Correlate application logs with CANN/device logs and the failing operator. Use synchronization only as a diagnostic to localize asynchronous failures; remove diagnostic perturbations before final measurement.
- For Ascend C memory faults, use the installed version's supported sanitizer/dump tools on a minimal case. Check bounds, initialization, lifetime, address space and synchronization; use exception-dump and AI Core diagnostics when their inputs are present.
- For OOM, distinguish model/optimizer state, activations, temporary workspace, fragmentation and peak concurrency using the available allocator evidence. Preserve the task's required shape and semantics when evaluating a repair.
- For graph capture/replay issues, identify the actual backend and whether failure occurs at compile, capture or replay. Compare eager and graph paths with the same inputs; inspect dynamic shapes, side effects and unsupported operations. A graph-specific failure does not establish a general compiler failure.

Validate the smallest fix with the task's frozen checks, then profile separately. References: [operator precision](https://gitcode.com/cann/cannbot-skills/tree/2af66d732e595297cf34159cf8e1d49fa7354c2c/ops/ascendc-precision-debug), [runtime errors](https://gitcode.com/cann/cannbot-skills/tree/2af66d732e595297cf34159cf8e1d49fa7354c2c/ops/ascendc-runtime-debug), [training precision](https://gitcode.com/cann/cannbot-skills/tree/2af66d732e595297cf34159cf8e1d49fa7354c2c/model/model-train-precision-numerical-mismatch), [graph triage](https://gitcode.com/cann/cannbot-skills/tree/2af66d732e595297cf34159cf8e1d49fa7354c2c/graph/torch-npugraph-ex-dfx-triage).
