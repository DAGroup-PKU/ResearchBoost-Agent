# prep — how the frozen inputs were built

Everything here ran once, on the machine that formulated the task, against the raw Tahoe-100M shards.
None of it runs inside the optimizing agent's sandbox.

| script | does |
|---|---|
| `fetch_parquet.sh`, `fetch_cross_plate.sh` | download the Tahoe-100M parquet shards used by the slice |
| `build_h5ad.py` | per-shard quota sampling → one `.h5ad`; the quota is what keeps 13 plates / 50 cell lines / 380 drugs in the slice instead of a single plate |
| `filter_genes.py` | drop all-zero genes (24.5 % of the panel; `scanpy`'s HVG selection aborts on a nullable `_index`) |
| `normalize_h5ad.py` | nullable `_index` → plain strings, and `obs/batch` → `obs/gem_group`; must run **after** `preprocess_train`, which round-trips the object and drops both |
| `build_fixture.py` (in `bench/`) | freeze 6 batches the numerics judge replays |
| `time_loader.py`, `time_loader2.py` | the loader probe that showed the step was data-bound |
| `rebuild_slice.sh` | the whole chain: preprocess → normalize → fixture → record twice and compare bit for bit → calibrate thresholds → freeze timing → freeze inputs |

## The reference tree

`bench.json` points `reference` at `targets/reference`, and `check_reference()` refuses to run unless that
directory is a **clean git checkout of `9bbfe78a434a55205e4de834e1ea99f85f7a3add`** with no modified tracked
files. It is not committed, like the other tasks. Create it before anything else:

```bash
git clone https://github.com/arcinstitute/state.git targets/reference
git -C targets/reference checkout 9bbfe78a434a55205e4de834e1ea99f85f7a3add
```

Gate 1 times this tree twice per check, and gate 0 compares the candidate's launch path against it.

## Two things that are not obvious

1. **Reference records are machine-specific and are not committed.** `bench/records/ref` (2 steps x 397
   tensors) must be re-recorded on the machine that will judge, twice, and the two recordings compared byte
   for byte before anything is trusted. `bench/thresholds.json` is then calibrated from that reference; its
   digest is recorded in `task/formulation.json`.
2. **The reference must be recorded in the same environment the judge runs in.** The first attempt recorded on
   the host and judged inside the sandbox; the two environments differed in the interpreter prefix, and the
   global RNG stream shifted deterministically (tensors stayed bit-identical, the fingerprint did not). Record
   inside the sandbox.

## Assets

The slice (7.4 GB), `gene_metadata.parquet` and `bench/fixture.pt` (1.8 GB) are published at
`YHDGD5/state-tahoe-case` on Hugging Face and are pinned by sha256 in `task/formulation.json`. Rebuilding
them from the raw shards reproduces the slice; the fixture is a deterministic draw from the tree's own
dataloader with a fixed seed.
