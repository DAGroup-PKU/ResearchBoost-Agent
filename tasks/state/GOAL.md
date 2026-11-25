# Goal

Bring the **time per training iteration** of the State transition model (Arc Institute, `arcinstitute/state`,
preprint configuration `tahoe_best`) on the frozen Tahoe-100M slice **down to 1/2 of the reference (a 2.0x
speedup)**, with a minimum of 1.3x, while passing the numerics checks below.

## The workload

One training step of the State transition model on one A800: a batch of 8 cell sentences of 512 cells each
(4096 cell rows), hidden dimension 1440, 4 transformer layers, an HVG input of 2000 genes and a gene decoder to
the full 47,320-gene space, with the energy-distance distribution loss.

- Frozen slice: `bench/dataset.toml` points at one directory of preprocessed Tahoe-100M h5ad files. The file,
  the fixture batches and the gene metadata are frozen by hash in `task/formulation.json`.
- Recipe: `bench/recipe.py`. It is not editable, and the bench passes every one of its values as a Hydra override,
  so the tree's config defaults cannot move them.
- Reference: `targets/reference`, Arc Institute's repository at commit `9bbfe78`, code unchanged.
  - The reference takes **0.40 s per iteration** in the timing windows, with about 25 s from process start to the
    first iteration.
  - One iteration is **366 ms of forward+backward and 7.7 ms of optimizer step** (209.1M parameters), i.e. it is
    compute-bound in forward/backward, not in the optimizer.
  - Its achieved rate is ~14 TFLOPS against the A800's fp32 tensor-core peak: **about 4.5% MFU**. That is the
    headroom this task is about.
  - `src/state/tx/callbacks/__init__.py: gradient_norm()` calls `.item()` once per parameter tensor (79 syncs per
    step), which stalls the GPU against the host every iteration.

## What counts as reaching the goal (the single criterion)

```
python bench/bench.py check --tree state --config <switch file> --out <output dir>
```

It writes `<output dir>/check.json`.

- **Done**: `status` is `PASS` and `performance.speedup` >= 2.0.
- If profile evidence shows that 2.0 cannot be reached within the rules, `status` `PASS` with
  `performance.speedup` >= 1.3 is also acceptable for wrapping up; write down in `WORK_LOG.md` why each remaining
  hotspot cannot be addressed.

`check` applies the gates below in order and stops at the first that fails:

0. **Structure (CPU, seconds)**: the launch path is byte-identical to the reference: `src/state/__main__.py`, the
   `_cli/` package and the four Hydra configs in `bench/frozen_files.txt`. The frozen batch fixture must still hash
   the same. Everything else in the tree is yours: the model, the loss, the callbacks, the optimizer, the kernels.

1. **Benefit (about 4 minutes)**: reference -> candidate -> reference, three independent runs of the real
   `state tx train` entry point with cold compile caches.
   - Each run has 10 warmup iterations + 2 windows of 20 iterations.
   - `performance.speedup` = the mean total window time of the two reference runs / the candidate's.
   - A fingerprint (command, math mode, GPU UUIDs, library versions, timing plan) that differs from the first
     reference run -> ERROR.
   - The random-number stream must match the frozen reference iteration by iteration, otherwise ERROR: the order
     in which random numbers are drawn changes what the model does.
   - Interference makes a run INVALID, to be rerun a little later: a reference deviating from the frozen timing,
     the two reference runs disagreeing, or a window ratio drifting.
   - A speedup that does not exceed `max(1.02, 1 + 2 x the reference spread)` -> UNAVAILABLE (no_benefit),
     rejected without a precision evaluation.

2. **Bitwise (about 1 minute)**: in deterministic mode, record a 3-step case through the numerics adapter
   (`bench/state_adapter.py`).
   - Every exact item must hold, otherwise INCORRECT: phase, RNG fingerprint before and after each step, number of
     optimizer steps, learning rate, weight decay, step counter, tensor names and shapes, all values finite.
   - All recorded tensors (`loss/`, `out/`, `grad/`, `delta/`, `optim/`, `buffer/`) bitwise identical -> **PASS
     (N0)**, accepted directly.
   - Tensors larger than 4096 elements are stored as a linear CountSketch with exact norms and hashes, so
     "bitwise identical" stays a bitwise judgment without storing gigabytes per step.
   - **The record fixes the SDPA backends to the math implementation.** Flash attention's backward uses
     nondeterministic atomics: with it on, the reference cannot reproduce its own gradients (measured: 79 of 79
     gradients differ between two runs; with it off, 0 of 79). The reference and every candidate are recorded the
     same way, and `math_mode` records the choice per step, so a candidate that switches it back is rejected.
     Gate 1 still measures speed with the recipe untouched.

3. **Statistical**: only when gate 2 is not bitwise. Paired short trajectories on the frozen seeds; per endpoint
   the bias against the reference must stay within 3 sigma, sigma the reference's own seed-to-seed spread.

- **Tier**: N0 = everything bitwise identical with the same dtypes and math modes; N2 = the dtype or math mode of
  any tensor changed; N1 otherwise. Beyond N2 is INCORRECT.

Two other modes: `--quick` (gate 0 and 2 only, no timing) and `--skip-timing`.

## What every round must do

1. **Profile first, then change.**
   ```
   python bench/bench.py profile --tree state --config <switch file> --out runs/<round>/profile [--iters N]
   ```
   It writes `profile.json` and a Chrome trace (rank 0): GPU busy share, kernel time by category, the top kernels,
   and each iteration's host time. Read `skills/train-opt-tiered-profiling` and
   `skills/kernel-utilization-analysis` before choosing a hotspot: with ~3% MFU the question is which kernels are
   launch-bound, which are memory-bound below peak bandwidth, and which chains of small kernels could be fused.
2. **Time only with bench.py**: `python bench/bench.py time --tree state --config <switch file> --out <dir>`
   reports the speedup against the frozen reference. Do not write your own timing scripts.
3. Every round appends four items to `WORK_LOG.md`: profile findings, the change, the `time` result, the `check`
   result. Failed and no-gain rounds are logged too.

## Pace: at most 30 minutes per round

Each round: profile, change one hotspot, re-time, log. The wall clock starts when you take the GPU lock. If the
time is up, wrap up the round anyway and record its conclusion.

## Constraints

- **Training semantics must not change** (gate 0 compares structure, gates 2 and 3 compare numerics): data and its
  order, per-GPU batch of 8 cell sentences of 512 cells, the model architecture as configured, learning rate and
  optimizer hyperparameters, the loss composition, the update schedule (one Adam step per iteration, gradient clip
  at 10), and the order and use of random draws.
- **Precision**: the baseline is the recipe's precision (fp32 weights, fp32 optimizer state, TF32 per the frozen
  math mode). Whole-step lower-precision switches (bf16, FP8, INT8) are not allowed. Local precision changes are
  allowed up to tier N2 and must pass the statistical gate.
- **Deterministic mode**: the numerics record runs with `torch.use_deterministic_algorithms`. If a kernel you add
  has no deterministic implementation, it must take a deterministic path of the same precision when
  `torch.are_deterministic_algorithms_enabled()` is true. Do not read `STATE_BENCH_*` variables, and do not behave
  differently depending on whether you are being timed.
- **Frozen files** (`bench/frozen_files.txt`): `src/state/__main__.py`, the `_cli/` package and four Hydra configs.
  Do not edit them. Everything else is your work surface.
- **Switch file** format: `{"environment": {"X": "1"}}`. Variables the bench manages may not be set (see
  `reserved_env` in `bench/bench.json`).
- **Read-only paths and network**: `bench/`, `numerics/`, `task/`, `targets/` and `$ASSETS` are read-only; there is
  no network and no new packages.

## Environment

- 1 A800 80GB PCIe (PCIe, no NVLink). GPU lock: `/tmp/state_gpu.lock`; `bench/bench.py` takes it itself.
- Python: `$ASSETS/env/bin/python`. torch 2.15.0+cu132, triton (bundled), lightning 2.6.6, cell_load, anndata,
  scanpy, geomloss.
- HF weights and data are local, offline mode.
- The frozen artifacts under `task/` are readable: `task/timing_ref.json` (the reference timing and its random
  fingerprints), `task/formulation.json` (frozen input hashes).

## References (skills)

Read the relevant cards before choosing what to try: `train-opt-tiered-profiling`, `train-opt-measurement`,
`train-opt-action-ladder`, `train-opt-numerics-validation`, `train-opt-fusion-patterns`. Re-verify these priors on
this task before treating them as conclusions.

## Wrap-up

- Commit each change separately in `state/` (branch `opt`), stating what changed, the tier and the `time` result.
- Once the goal is reached, write a summary at the end of `WORK_LOG.md`: the final switch file, the final check
  path and result, the breakdown of gains by change, and the directions that did not work.

---

# Round 2 — from 1.885x to 2.5x

You are starting from a tree that already worked. `state` is on branch `opt`, two commits on top of `9bbfe78`,
and the last full check measured **1.885x** (candidate 0.1883 s/iteration, frozen reference 0.3713,
PASS with 794/794 tensors bit-identical, tier N0). Read `WORK_LOG.round1.md` before changing anything: it has
the ablation table, the four rounds that got there, and — more useful — the directions that were measured and
rejected.

## The hard constraint

Every one of the **794 recorded tensors must stay bit-identical** (tier N0), and the step must stay on the
same code path while being timed. Anything that changes a produced value, a reduction order, or the number of
random draws is out of scope for this round, however fast it is. That includes:

- bf16 / fp16 for the large GEMMs (roughly doubles throughput, changes every output)
- a different energy-distance estimator, or chunking its reduction along cells or features
- fused Adam (changes the `optim/` tensors)
- compiling a block that contains dropout without preserving its draw order

Gate 2 catches these. Do not spend the round on them.

## Where the 188 ms actually is

| part | cost | note |
|---|---|---|
| cutlass GEMMs (transformer + decoder projection) | ~81 ms | TF32; `max-autotune` measured slower |
| elementwise | ~33 ms | dominated by the 4096 x 47942 dense target |
| attention | ~9.5 ms | |
| energy loss (3 pairwise GEMMs + backward) | ~27 ms | ~67 TFLOPS, already reasonable |
| **Python/autograd dispatch in the eager blocks** | **~50 ms** | **the largest addressable block** |
| Adam step | 8.7 ms | 209.8 M parameters |

GPU busy is ~130 ms of the 188 ms step.

## Ranked levers, in the order worth trying

1. **Make the eager blocks compile without moving a single random draw.** `pert_encoder`, `basal_encoder`,
   `project_out` and `gene_decoder` stay eager because their p=0.1 dropout draws differently once fused.
   Hoisting the dropout out of each compiled region — generate the mask first, pass it in — should preserve
   the draw order and let the compiler take the block. Expected 20–40 ms. This is the main prize and also the
   one that can silently break the RNG; verify with `check --quick` after every step, not at the end.
2. **Overlap the data path with compute.** The loader is now a ~48 MiB CSR bundle, but it still sits on the
   critical path. Prefetch the next batch on a worker thread with pinned staging so the host never idles
   between steps. Expected 10–15 ms.
3. **Anything else you can prove is bitwise-safe.** Reordering host work, caching per-file handles, avoiding
   temporary allocations — but each one needs the same 794/794 proof.

## What "done" looks like

`python bench/bench.py check --tree <tree> --out <dir>` reporting PASS at >= 2.5, bitwise, tier N0, with the
candidate under ~0.15 s/iteration. Keep `FINAL.json` pointing at the best tree you have evidence for at all
times. If you stop between 2.0 and 2.5, say exactly what is left in WORK_LOG.md — the numbers, not a guess.
