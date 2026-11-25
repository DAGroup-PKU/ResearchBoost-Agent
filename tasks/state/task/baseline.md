# Baseline: the reference as measured during formulation

Machine: 1 x A800 80GB PCIe, shared host (other users' jobs run on it). GPU 4 for the numbers below.

**Slice:** 400k cells sampled with a per-shard quota from 199 shards of Tahoe-100M, spanning **13 plates,
50 cell lines, 380 drugs, 1243 samples** (5.6M cells downloaded = 5.9% of the full 95.6M; the benchmark itself
uses 400k). 47,942 genes remain after dropping the ~24% with zero expression.

## Step time

| Measurement | Value |
|---|---|
| Steady-state iteration (30 warmup + 3 x 40 iterations) | **0.380 s** frozen; 0.371 s on the acceptance run |
| Process start to first iteration | ~22 s (imports, dataset scan, model build, DDP-less fitting) |
| Peak GPU memory | **9.8 GiB allocated** |
| Run-to-run spread over three reference runs | **17–20%** on this shared host (this sets the timing bands) |

One iteration = one batch of 8 cell sentences x 512 cells (4096 cell rows), one forward, one backward,
one Adam step, gradient clipping at 10.

## Where the time goes

Lightning's own FLOP counter on the reference batch: **5.18e12 FLOPs per batch** (forward + backward).

| | Value |
|---|---|
| Achieved | ~8.8 TFLOPS at 0.444 s/iteration |
| A800 fp32 tensor-core peak used for the ratio | 312 TFLOPS |
| **Model FLOPs utilization** | **~2.8%** |
| Data loading (measured alone, 8 workers, median) | 0.116 s per batch — about 20-25% of the step |

So the loop is GPU-bound, not loader-bound, and it leaves roughly 97% of the GPU idle. That is the headroom
this task is about.

## Why MFU is this low (hypotheses to test in round 1, not conclusions)

- The model is small for the batch: hidden 1440, 4 layers, 8 sequences of 512 tokens. Per-kernel work is
  small enough that launch overhead and memory-bound elementwise ops dominate.
- The distribution loss (`geomloss` energy distance) is O(S^2) per sample set and is not counted as matmul
  FLOPs; it may be a large share of wall time while contributing nothing to the measured MFU.
- No `torch.compile` and no fused optimizer in the reference path.
- Several separate small GEMMs per cell set (encoder MLPs, decoder MLP to 47,320 genes) rather than batched
  work.

## What the profile says

`runs/profile/timing_profile.profile.txt` (torch profiler, 5 iterations after 30 warmup) is the authoritative
breakdown. Read it before choosing a hotspot; the hand-written guesses above are priors, not findings.

## Acceptance numbers recorded at formulation

| | |
|---|---|
| Reference, unmodified tree | `UNAVAILABLE` / `no_benefit`, speedup 0.95 (bar 1.15) |
| Control arm, lr x1.02 | `INCORRECT`, gate 2, exact failure on `invariants` |
| Tampered frozen file | `ERROR`, gate 0 |
| Reference records, two runs | 794/794 tensors bitwise identical |

## Measured breakdown of one iteration (direct, not profiler-attributed)

| Part | Time |
|---|---|
| forward + backward | **366 ms** |
| `optimizer.step` (Adam, 209.1M params) | **7.7 ms** |

So the step is compute-bound in forward/backward, and the optimizer is not the problem. At 5.18e12 FLOPs per
batch and 366 ms of forward+backward, the achieved rate is ~14 TFLOPS, about **4.5% of the A800's peak**.

Read the torch-profiler summary with care: it attributes 274 ms/step to `Optimizer.step#Adam.step` and reports
"117% of CUDA time", which is a profiler scope artifact — the direct measurement above (7.7 ms) is the truth.
The profiler's *relative* kernel shares are still useful, and one entry in it is a real finding:

- `Memcpy DtoH (Device -> Pageable)`: 2.4 s over 355 calls in the profiled span (~6.9 ms each).
- `src/state/tx/callbacks/__init__.py: gradient_norm()` loops over parameters and calls `param_norm.item()`
  **per tensor** — one host sync per trainable parameter tensor, every step (79 syncs/step here). That serialises
  the GPU with the host and shows up as the DtoH traffic above.
