# Case report — State (Arc Institute), Tahoe-100M perturbation-response training

A ResearchBoost-Agent case: a frozen judge around one training step of the State transition model on
Tahoe-100M, judged on a single A800. This document is the case's own record: what was built, what was
measured, which defects the formulation had, and what an unsupervised optimization run found.

---

## 1. Why this case

State is a public model (Arc Institute, *Cell*), trained on public data (Tahoe-100M), at a size that fits one
GPU. That combination makes it a clean target for an autonomous optimizer: no private weights, no
multi-node topology, and — importantly — no single obvious knob. The interesting question is not "can it be
made faster" but "is the frozen judge strong enough to tell a real speedup from a broken one".

The case is therefore judged on three things at once: it must get measurably faster (gate 1), produce
bit-identical training tensors (gate 2), and not change the numerics tier (gate 3) — while the reference
itself is re-timed in the same session so that the speedup cannot come from a lucky machine.

## 2. The workload

| | |
|---|---|
| Upstream | `github.com/arcinstitute/state` @ `9bbfe78` |
| Config | `tahoe_best` (`model.kwargs.cell_set_len=512`) |
| Step | batch 8 cell sentences x 512 cells = 4096 cell rows |
| Parameters | 209.1 M |
| Hardware | 1 x A800 80 GB PCIe, `world_size=1` |
| Data | `X_hvg`, `pert_col=perturbation`, controls `DMSO_TF` |

The batch shape is fixed by the recipe, not by the candidate: `DEVICES=1`, `BATCH_SIZE=8`, `CELL_SET_LEN=512`.
An optimizer may not fill the machine to win — with one card the measured quantity stays one step of one
GPU, and gate 1 cannot be satisfied by adding parallelism.

### The data slice

The full Tahoe-100M is 3388 parquet shards (337.6 GB, 95.6 M cells) ordered by plate, so a naive head-of-file
slice contains a single plate and a single cell line — an unrealistically easy batch distribution. The slice
is therefore built with a per-shard quota that keeps plate diversity:

- **13 plates, 50 cell lines, 380 drugs, 1243 samples**
- **399,990 cells x 47,942 genes**, 7.4 GB on disk

All-zero genes (24.5 % of the original panel) are dropped before the slice is written, because gene filtering
in `scanpy` cannot run with a nullable `_index`. `cell_load` additionally needs a plain-string `_index` and a
`gem_group` batch column; both are written *after* `preprocess_train`, since that step round-trips the object
and drops them.

## 3. The judge

```
python bench/bench.py check --tree <candidate> --out <dir>
```

| Gate | Question | Failure |
|---|---|---|
| 0 structure | Did the tree change in a way that makes the measurement meaningless (devices, batch, dtype, precision mode, reserved env)? | `ERROR` |
| 1 benefit | Is the candidate faster than **two live reference runs** in the same session? | `UNAVAILABLE` / `no_benefit` |
| 2 bitwise | Are all 794 recorded tensors bit-identical to the frozen reference, on the same code path? | `INCORRECT` |
| 3 statistical | Does the numerics tier stay at N0? | `INCORRECT` |

Frozen against the candidate tree: `bench/records/ref` (2 steps x 397 tensors), `bench/thresholds.json`
(794 gates), `task/timing_ref.json` (0.3713 s/iteration, 120 iterations), `task/formulation.json`
(three input fingerprints). `bench/`, `numerics/`, `task/` and `targets/` are read-only inside the run's
sandbox, and the harness refuses a verdict if any file under them changed since launch.

Timing is 30 warm-up iterations, then 3 windows of 40, judged on **window means**, with bands calibrated from
the measured 17–20 % run-to-run noise of this shared host. A single-window comparison would have accepted a
noise artefact.

### The judge was tested against known-wrong candidates

| Candidate | Expected | Observed |
|---|---|---|
| Unmodified reference | no benefit | `UNAVAILABLE` / `no_benefit`, 0.9384x |
| Control arm: `lr x 1.02` | gate 2 | `INCORRECT`, `invariants`: lr 0.0001 vs 0.000102 |
| Tampered frozen file | gate 0 | `ERROR` at structure |
| Pristine reference | pass | `PASS`, bitwise, tier N0, 794/794 |

## 4. Formulation defects found while hardening the case

These are the reason this section exists: each one was discovered by running the judge on something that
should not have passed — or on something that should have and did not.

1. **Two-GPU DDP deadlocked.** NCCL spun at a constant 715 MiB. The workload was fixed at one GPU, which is
   also what makes the measurement comparable across runs.
2. **The reference was not bitwise reproducible** (479 differing tensors). Cause: flash attention's backward
   uses atomics. A deterministic SDPA stand-in is installed whenever `are_deterministic_algorithms_enabled()`
   is true; 79/79 flash runs diverged, 0/79 did not.
3. **The adapter measured the pip-installed `state`, not the candidate tree.** A control arm falsely passed.
   Fixed by rebinding `sys.path` and purging `sys.modules` before import, then asserting the imported package
   lives inside the tree.
4. **The adapter built its own optimizer with `weight_decay=5e-4`.** Upstream uses `Adam(lr)` with no weight
   decay, so every candidate was being judged against a different optimizer. Fixed by calling
   `configure_optimizers()`.
5. **The slice collapsed to one plate**, making the batch distribution trivial. Fixed by per-shard quota
   sampling.
6. **Timing noise made every run invalid.** Bands were re-derived from measured noise and gate 1 compares
   window means rather than single windows.
7. **The reference was recorded on the host but judged in the sandbox.** After the sandbox environment was
   fixed, the reference showed a *deterministic* RNG offset (identical across two runs, tensors still
   bit-identical). Re-recorded inside the sandbox; the two records then matched bit for bit and the
   recalibrated thresholds kept the same digest.
8. **The sandbox hid `$HOME` while the interpreter lived there.** `--tmpfs "$HOME"` removed
   `/home/hitsz/miniconda3`, which was the venv's base interpreter and the location of `torch`. The first
   optimization run could not find Python and spent its budget looking for one. The environment now lives
   under `$ASSETS` and does not reference `$HOME` at all.

## 5. Results — two runs on the same frozen judge

Both runs used the same frozen judge, the same slice, the same card and the same harness. They are reported
separately because they answer different questions.

| run | start | harness verdict | candidate s/iter |
|---|---|---|---|
| 1 — cold, unmodified tree | `9bbfe78` | PASS, **1.8848x**, N0 | 0.1883 |
| 2 — warm, from round 1's tree | round 1 (`opt`) | PASS, **2.0611x**, N0 | 0.1797 |

A third harness run measured 2.0797x (candidate 0.1776, reference 0.3693). Between 2.06x and 2.08x is one
candidate at 0.178-0.180 s/iteration and a denominator that moves by 11 %.

The published artifacts were verified end to end: `Faster-State` was cloned from GitHub into a fresh
directory and judged by the frozen bench — `check --quick` passed bitwise, and its `source_digest`
(`4cf4e2bd…`) is identical to the tree the 2.08x verdict was measured on.

Round 1 took 81 minutes and 1.23 M tokens; round 2 took 2 hours 34 minutes and 2.91 M tokens. Round 2's
workspace contained round 1's work log and a hotspot map written by the formulator, so its number measures
how much headroom is left once the obvious work is done — not what an unassisted agent achieves from the
same tree. The cold number is the comparable one.

The round-2 agent's own full `check` runs of the same code measured 2.1451x and 2.2187x; the harness run
measured 2.0611x. All three are the same candidate at 0.174–0.180 s/iteration. The spread is the denominator:
gate 1 divides by two live reference runs, and those land on either side of the frozen 0.3713 (0.3599,
0.3704, 0.3811 across the three runs; `reference_spread` 0.113–0.114).

### Where the time went

| step | s/iter | vs frozen |
|---|---|---|
| reference tree (`9bbfe78`) | 0.396 | 0.94 |
| + vectorised gradient norm | 0.368 | 1.08 |
| + CSR densify in `fetch_gene_expression` | 0.343 | 1.16 |
| + CSR batch shipped, densified on device | 0.212 | 1.87 |
| + RNG-safe compile of the backbone | 0.186 | 1.885 |
| + one CUDA graph for forward, both losses, backward | 0.179 | 2.06 |

### The finding that mattered, in round 1

The task brief assumed a compute-bound step. Ablation said otherwise:

| variant | s/iter |
|---|---|
| full step | 0.368 |
| forward only | 0.368 |
| forward only, 750 MiB transfer skipped | 0.377 |
| **no model at all** (batch + trivial loss) | **0.384** |

The step was bound by the *production* of one dense `[4096, 47942]` float32 batch per iteration — 749 MiB of
which 97.6 % is zero. `cell_load`'s `_fetch_gene_expression_raw` built each cell as a one-row
`sparse_csr_tensor(...).to_dense()`, measured at 2.81 ms/cell (512 cells per sentence), and the collate ran
two full-tensor scans whose result was discarded. A NumPy scatter over the CSR arrays does the same work in
0.037 ms/cell, verified bitwise on 400 random cells.

### The finding that mattered, in round 2

With the host out of the way, the kernel profile said something the first round could not see: **TF32 is off
in the frozen math mode** (`float32_matmul_precision = "highest"`), so every GEMM runs on the fp32 FFMA
units. A 4096x2048x47942 GEMM measures 44.6 ms = **17.7 TFLOPS**, against the A800's 19.5 TFLOPS fp32 peak —
**90 % of peak**. The arithmetic was never underutilised; it only looked that way when compared with the
tensor-core number, which this math mode does not allow.

That is why round 2 went after the host instead: one CUDA graph captures forward, both losses and backward.
Four things had to be solved to get there, and each is a way the judge could have been fooled:

- **Dropout outside the graph.** A capture freezes its random draws, which would freeze the masks and change
  the RNG fingerprint. Every `nn.Dropout` becomes a `MaskDropout` that draws with `torch.native_dropout` on a
  scratch tensor of the same shape, in the recorded execution order; the mask depends only on the generator
  state and the tensor shape, so the reference's draw sequence is consumed exactly.
- **geomloss's weights moved to the device** — it rebuilt them on the CPU and copied them on every call, which
  a capture rejects.
- **Gradients via `torch.autograd.grad`** — the persistent `AccumulateGrad` node pins the default stream and
  aborts the capture.
- **Clip and Adam stay eager**, after the replay: the harness counts steps through a Python `optimizer.step()`
  hook, and capture-safe Adam is slower anyway (10.97 vs 9.15 ms).

Deterministic mode (the numerics record) never builds a graph, so the recorded path is the plain eager one.

## 6. Why it stops at ~2.2x

Round 2 measured its own floor rather than guessing it:

| part | cost |
|---|---|
| GPU kernels | 131 ms — 82 ms of it cutlass GEMMs at ~90 % of the fp32 FFMA peak |
| eager clip + Adam | 11 ms — the harness's step counter requires a real `optimizer.step()` |
| batch transfer | 7 ms — 48 MB of CSR non-zeros over PCIe, issued at the start of the step |
| host between replays | ~20–25 ms, now spread over items of a few hundred microseconds each |

2.5x needs 0.1485 s/iteration, i.e. all of that plus the host gap to disappear. The only large item left is
arithmetic, and moving it means changing precision — exactly what tier N0 forbids, and what would push the
case into N1 and a statistical gate.

Directions measured and rejected: Adam inside the graph (slower, and it breaks the step counter), the whole
step in one graph (~195 ms), compiling the whole forward (1.99x in round 1), `max-autotune`, and a
data-only step (the round-1 ablation above).

## 7. Protocol conformance — closed

The harness's final evaluation (`goal/clawo/judge.sh`) always runs `bench.py check --final`. This bench
predated that protocol: `check` did not accept the flag, and `check.json` carried no `final`/`n1` fields, so
the harness's own check could not run even though the bench's own `check` passed. Found by round 1.

`bench.py` now accepts `--final` and `--n1`, records `final`/`n1`/`quick`/`skip_timing` in `check.json`, and
refuses `--final` unless `task/final-evaluation` exists — a marker the harness creates outside the sandbox, so
a candidate cannot request a final verdict of its own. The judge digest was re-frozen and the three negative
controls re-run afterwards; they still fail on the gates they target.

## 8. Reproducing

```bash
# assets: the slice, gene_metadata.parquet and bench/fixture.pt, pinned by sha256 in task/formulation.json
# published at huggingface.co/datasets/YHDGD5/state-tahoe-case
export ASSETS=<a directory holding data/dataset/tahoe_slice.h5ad, data/gene_metadata.parquet, env/>

python bench/bench.py check --tree state --out runs/verify     # the judge (reference -> candidate -> reference)
python bench/bench.py time  --tree state --out runs/t          # timing only
python bench/bench.py profile --tree state --out runs/profile

# negative controls: the three candidates that must fail
python bench/bench.py check --quick --tree targets/neg_structure --out runs/neg/structure
CONTROL_ARM_LR=1 python bench/bench.py check --quick --tree targets/neg_lr --out runs/neg/lr5
```

`bench/records/ref` and `bench/fixture.pt` are machine-specific and are not committed: record the reference
twice inside the judging environment, compare them byte for byte, then calibrate `bench/thresholds.json`.
See `formulation/prep/README.md`.
