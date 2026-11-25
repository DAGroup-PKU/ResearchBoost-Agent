# Supervisor notes — State / Tahoe-100M slice

State of the task at the end of formulation. **Read this before launching or intervening.**

## Status: formulated, verified on the final slice, not yet launched

The benchmark slice is the cross-plate one (13 plates / 50 cell lines / 380 drugs / 1243 samples), built from
199 Tahoe-100M shards by `prep/rebuild_slice.sh`. Frozen reference timing: 0.380 s/iteration.

| Check | Result |
|---|---|
| Reference records, two runs | bitwise identical, 794/794 tensors, all exact items equal |
| Unmodified reference through `check` | `UNAVAILABLE` / `no_benefit` (0.94x, bar 1.15, no interference) — as required |
| Negative: tampered frozen file | `ERROR`, gate 0, names the file |
| Negative: control arm lr x1.02 | `INCORRECT`, gate 2, exact failure on `invariants` (lr) |
| Timing noise floor (3 reference runs) | 17–20% run-to-run on this shared host |

## Things that were wrong before, and must stay fixed

1. **The tree under test must be the tree the judge imports.** The environment has an editable install of
   upstream `state` on `sys.path`. Before `StateAdapter._bind_tree()` existed, `state` was imported from that
   install, so gate 2 was judging the installed copy and ignored every candidate change. The control arm passed
   for exactly this reason. `_bind_tree()` now puts `<tree>/src` first, drops already-imported `state` modules,
   and raises if `state.__file__` is outside the tree under test.
2. **Flash attention cannot be recorded bitwise.** With flash SDPA on, the reference's gradients differ from
   run to run (79/79 parameters); with it off, 0/79. The adapter therefore fixes the SDPA backends to the math
   implementation whenever deterministic mode is on, for the reference and every candidate alike, and `math_mode`
   records that choice per step. Gate 1 still measures speed with the recipe untouched. If someone "fixes" this
   by switching flash back on, gate 2 stops being a bitwise gate.
3. **The adapter must use the tree's own `configure_optimizers()`.** It used to build `Adam(lr, weight_decay=5e-4)`
   itself, but upstream builds `Adam(lr)` with **no weight decay**. A candidate changing the optimizer would have
   been invisible, and the reference was being judged against an optimizer the real run never uses.
4. **`state tx train` needs `gem_group` and plain string `_index`.** The benchmark h5ad was written by recent
   anndata (nullable-string encoding), which cell_load cannot read. `prep/normalize_h5ad.py` rewrites those.

## Environment and machines

- GPUs 3 and 4 were assigned. **GPU 3 is shared with other users** (a `STGAT_train.py` job appeared during
  formulation). The bench takes `/tmp/state_gpu.lock` and waits for its own GPUs to be idle, but it cannot tell
  whose job is running: on a busy box a timing window can be disturbed. When a run comes back INVALID with
  drift, rerun it later rather than widening the bands.
- Band settings come from the measured noise floor (`runs/base2/baseline.json`), not from taste: 17-20% run-to-run
  even with 30 warmup + 3x40 iterations. The reference-vs-reference check is judged on WINDOW MEANS; per-window
  pairs are reported as a diagnostic only. Judging on per-window pairs made every run INVALID.

## Formulation material

- `bench/records/ref`, `ref2` — the two reference recordings that must stay identical.
- `task/timing_ref.json` — the frozen reference timing gate 1 compares against.
- `task/formulation.json` — hashes of the dataset, fixture and gene metadata.
- `prep/` — how the slice was built (`build_h5ad.py`, `filter_genes.py`, `normalize_h5ad.py`,
  `build_fixture.py`) and how the environment was put together.
