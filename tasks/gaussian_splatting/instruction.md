# Task: faster 3D Gaussian Splatting training

Make the training implementation in `/submission/gaussian-splatting` faster over a full training run without
changing what it computes. The score is the speedup of your submitted tree over the unchanged reference, measured by
the judge described below; a faster tree counts only if it passes every gate: the exact items identical to the
reference, and the trained models statistically equivalent to the reference's (tier N1).

## Setup

- **Hardware:** 1 Hopper GPU (141 GB) for the whole trial. Your work and the final evaluation use the same GPU.
- **Before you started**, the container ran the judge on the unchanged reference on this GPU: its quick record made
  twice (identical exact items; frozen as `bench/records/ref.json`), eight full training runs of the reference with
  seeds 1–8, each scored by the judge's evaluation, with their exact items and the frozen spread of every endpoint
  (`bench/traj/`), its own quick and full checks, and three controls (a tree that changes the camera order, rejected
  by the exact items; a tree with half the D-SSIM weight, rejected by the statistical gate; a tree that computes SSIM
  with upstream's PyTorch code instead of fused-ssim, accepted at N1). These files are read-only; the log is
  `/var/log/researchboost/prepare.log`.
- **Final evaluation:** when your working time ends, every process you started is stopped and the judge's full check runs
  in this same container on `/submission/gaussian-splatting` and `/submission/switches.json`, as the unprivileged
  `agent` user and without network. Its result is the score: when `status` is `PASS`, the reward is `accepted` 1 and
  `speedup` = `performance.speedup`; otherwise both are 0.
- **Network:** you have internet access. The container holds no credentials.
- **Python environments:**
  - `/opt/gs-env` is the image's training environment: torch 2.8.0+cu128, torchvision 0.23.0+cu128 and upstream's
    dependencies (plyfile, tqdm, opencv-python-headless, joblib). The full list is
    `/opt/researchboost/installed-packages.txt`. It is read-only. The judge runs with it, and so does every reference
    run. Upstream's three CUDA extensions are not installed into it (see "CUDA extensions" below).
  - `/opt/candidate-env` is yours and comes first on `PATH` (`python`, `pip`). It sees every package of the training
    environment, and anything you install there (`pip install ...`, torch included) takes precedence. The judge runs
    your tree and builds its CUDA extensions with `/opt/candidate-env/bin/python`. User site-packages are ignored.
  - The judge records the library versions of the reference and the candidate runs (`performance.library_versions`);
    a difference is not an error.

## Working time

- Your working time is limited, and you are not told how much of it is left. When it ends, every process you started
  is stopped without warning and the final evaluation runs on what is in `/submission` at that moment: keep it in a
  state you want judged.
- The durations of the judge commands on this GPU are in `/var/log/researchboost/prepare.log` (the reference's own
  runs and checks ran there). One full training run takes several minutes; your full `check` runs one timed
  training, a quick record and one scored training, and holds the GPU for all of that time. Time one `time` run
  early.

## The workload

- Code: `/submission/gaussian-splatting` is upstream
  [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) at commit `54c035f`
  with upstream's accelerated rasterizer (the `3dgs_accel` branch of diff-gaussian-rasterization, commit `26ce026`,
  the Taming-3DGS drop-in of upstream's README "Training speed acceleration"), simple-knn `86710c2` and fused-ssim
  `1272e21`, as one git repository on a branch `opt` (the three CUDA submodules are plain directories of the tree;
  SIBR_viewers is left out). Make your changes there. `/opt/researchboost/ws/targets/reference` is the same tree,
  unchanged.
- The judge runs every tree through its own entry, `bench/gs_entry.py`, from a fresh copy of the tree, in one
  process: it imports the tree's `train.py` and calls its `training()` with the frozen recipe of `bench/recipe.py`
  (every hyper-parameter given explicitly on the command line, so the defaults in `arguments/__init__.py` do not
  matter). It installs a global optimizer step post-hook (`torch.optim.optimizer.register_optimizer_step_post_hook`);
  after the last step of the Gaussian optimizer the run ends (upstream's last iteration takes no step, so the
  parameters after step 29,999 are the trained ones).
- The recipe (`bench/recipe.py`) is upstream's `train.py` defaults on the Tanks&Temples **truck** scene of upstream's
  T&T+DB data set with `--eval` (every 8th image is a test view): 30,000 iterations, SH degree 3 (one degree more every
  1,000 iterations), densification and pruning every 100 iterations from 600 to 14,900 (gradient threshold 0.0002),
  opacity reset every 3,000 iterations, loss 0.8 × L1 + 0.2 × (1 − SSIM), Adam (`--optimizer_type default`) with
  upstream's learning rates and the exponential position schedule, images at their stored size (979 × 546; resolution
  -1 would only scale images wider than 1,600 pixels) and kept on the GPU, black background, no depth regularization, no antialiasing, no viewer, no test or save iterations inside
  the run. Random, numpy and torch are seeded by the entry (seed 0 for timing and the quick record, as upstream's
  `safe_state`).
- Data: the judge's frozen copy of the truck scene (COLMAP images and sparse model) in
  `/opt/researchboost/assets/tandt/truck`, read-only; the VGG-16 and LPIPS weights of the judge's evaluation in
  `/opt/researchboost/assets/torch`.
- What one iteration does (`train.py`, the `for iteration in ...` loop): the learning-rate schedule, the SH degree
  schedule, a camera drawn without replacement (`random.randint` over the remaining training cameras, refilled when
  empty), one `render()` of it, the loss and its backward, densification statistics and, at the scheduled iterations,
  densification and pruning or the opacity reset, then one step of the exposure optimizer and one step of the
  Gaussian optimizer.

## CUDA extensions

The rasterizer (`submodules/diff-gaussian-rasterization`), `submodules/simple-knn` and `submodules/fused-ssim` are
built FROM YOUR TREE by every judge command that runs it: the three directories are copied out of the tree and built
with `/opt/candidate-env/bin/python -m pip install --no-deps --no-build-isolation --target <cache>` (TORCH_CUDA_ARCH_LIST
9.0, the image's CUDA 12.8 in `/usr/local/cuda`) into `/tmp/gs-ext/<digest of their sources>`, and the run puts that
directory first on `sys.path`. Editing the CUDA sources is therefore a change of your tree like any other; a build
failure is an ERROR of the check. `bench/bench.py build --tree <dir>` builds them the way the judge does and prints
the directory (the cache is reused while the sources do not change). Extensions you install into
`/opt/candidate-env` yourself are shadowed by the judge's build of the tree; other packages you install there are
used. Each directory must keep a `setup.py`.

## The judge

Run it from `/opt/researchboost/ws` with the training environment's Python:

```
cd /opt/researchboost/ws
/opt/gs-env/bin/python bench/bench.py check --tree /submission/gaussian-splatting --config /submission/switches.json --out /tmp/check-1
```

It writes `<output directory>/check.json`. It runs every gate the final evaluation runs, with the same rules; only
gate 1 differs: your `check` times your tree once against the frozen reference timing made at startup, the final
evaluation times reference → candidate → reference. A result counts when the final evaluation's `status` is `PASS`;
the score is its `performance.speedup`.

Other commands (see `/opt/gs-env/bin/python bench/bench.py --help`); none of them is a score:

- `check --quick`: gates 0 and 2 only (no timing, no full runs): `PASS` (verdict `exact_items_pass`) when the exact
  items of the first 3,000 steps match the reference, `INCORRECT` when one differs. `check --skip-timing`: everything
  but gate 1.
- `check --n1`: gates 0 and 3 only (no timing, no quick record): the statistical gate again, for example after a
  change that only affects numerics, without repeating the timing.
- The final evaluation's own mode (`check --final`) is not available to you.
- `time --tree <dir> --config <file> --out <dir> [--paired]`: one timed full training (seconds in each window and in
  total, peak memory, and `speedup_vs_frozen_reference`, the speed relative to the reference timing frozen on this
  machine before you started, which gate 1 of your `check` also uses); `--paired` also times the reference right
  before it (a screening number, not a verdict).
- `profile --tree <dir> --config <file> --out <dir> [--from K] [--iters M]`: torch.profiler over Gaussian-optimizer
  steps K..K+M-1 (default 7,000 and 20). Nsight Systems is not installed.
- `build --tree <dir>`: build the tree's CUDA extensions as the judge does.

`bench/bench.py` takes the GPU lock `/tmp/gs_gpu.lock` itself; do not wrap it in `flock`. Wrap any other command that
uses the GPU as `flock /tmp/gs_gpu.lock <command>`. Every judge run needs 20 GiB free under `/tmp`; put output
directories under `/tmp` and delete them when you no longer need them. The startup measurements are in
`bench/records/` and `bench/traj/` (the frozen reference record, the reference runs' exact items and scores), `task/`
(frozen inputs, the reference commit, the reference timing of this machine) and `runs/bootstrap/` (logs of the
reference runs).

`check` applies the gates in order and stops at the first that fails:

0. **Structure**: `train.py` with a top-level `training()`, `gaussian_renderer/__init__.py`, `scene/__init__.py`,
   `scene/gaussian_model.py`, `arguments/__init__.py` and the three extension directories with their `setup.py` must
   exist, and the extensions must build. Otherwise ERROR.
1. **Benefit**: full trainings, production settings, on this machine.
   - Time of a run = from the end of Gaussian-optimizer step 1 to the end of step 29,999 (CUDA synchronized at steps
     1, 10,000, 20,000 and 29,999 only; the three windows between them are reported). Scene loading and the start of
     the process are not timed.
   - Your `check`: one candidate run; `performance.speedup` = the frozen reference time (`task/timing_ref.json`, the
     faster of the reference's two runs at startup) / candidate, and the spread is the frozen one (the relative gap
     between those two runs).
   - The final evaluation: reference → candidate → reference, three independent full trainings measured in that
     check; `performance.speedup` = mean(reference A, reference B) / candidate, the spread = the relative gap between
     A and B. A fingerprint (command, GPU, timing plan) that differs from the first reference run → ERROR; library
     versions are recorded for both and may differ. The two reference runs differing by more than 5% → INVALID
     (measured again later).
   - A speedup that does not exceed max(1.02, 1 + 2 × the spread) → UNAVAILABLE (no_benefit), rejected without a
     precision evaluation.
2. **Exact items**: the first 3,000 steps of the recipe with seed 0, observed (no timing). Otherwise INCORRECT:
   exactly one `scene.gaussian_model.GaussianModel` created and one Gaussian optimizer stepped; the image name of the
   camera of every call of `gaussian_renderer.render` made by the training (one per iteration), in order; the learning
   rate of every parameter group of the Gaussian optimizer and the active SH degree at steps 1, 500, 1,000, ...,
   3,000; the steps at which the number of Gaussians changes (densification and pruning); the dtypes and per-point
   shapes of the six Gaussian parameters and the dtypes of the optimizer state (float32, as the reference). The full
   runs of gate 3 check the same items over the whole training, per seed, against the reference's run with that
   seed.
3. **Statistical**: one full training of the recipe on the first frozen seed, seed 1 (the seed seeds random, numpy and torch: the camera
   order, the scene's shuffle and the densification's random draws follow from it; the rasterizer's backward, which
   accumulates gradients with atomics, adds its own run-to-run variation, which is why even the reference is not
   bitwise reproducible and this task has no bitwise tier).
   - Each trained model is scored by the judge's own evaluation (`bench/gs_eval.py`, never your code): the six raw
     parameters and the active SH degree saved at the end of the run are loaded into the REFERENCE `GaussianModel`,
     the 32 test views are rendered by the REFERENCE renderer and rasterizer, render and ground truth are quantized to
     8 bits as upstream's `render.py` saves them, and PSNR, SSIM and LPIPS (VGG) are computed as upstream's
     `metrics.py` does, averaged over the test views.
   - 4 endpoints: PSNR, SSIM, LPIPS and log(number of Gaussians).
   - σ = the reference's standard deviation across its own 8 seeds (frozen in `bench/traj/frozen.json`).
   - The bias d = candidate − reference of each endpoint on seed 1: every |d| ≤ 3σ → **PASS (N1)**; any |d| > 3σ →
     INCORRECT. One seed, no further seeds; the same rule in your `check`, `check --n1` and the final evaluation.
- **Tier**: N1 = every exact item identical and statistically equivalent; the only accepted tier. A changed dtype of
  a Gaussian parameter or of the optimizer state is an exact-item failure.

## Rules

- **Training semantics must not change** (gate 2 compares the exact items, gate 3 the trained models): the scene and
  its train/test split, the camera of every iteration and their order, the number of iterations, the
  representation (3D Gaussians with SH degree 3, upstream's parameterization and activations: log-scales,
  logit-opacities, unnormalized quaternions, SH coefficients), the image formation model of the rasterizer (EWA
  splatting of the projected Gaussians, front-to-back alpha compositing in depth order, upstream's limits and
  thresholds), the loss (0.8 × L1 + 0.2 × (1 − SSIM) over the full image), the densification, pruning and opacity-reset
  rules and their schedule, the optimizer — Adam with upstream's update rule applied to every Gaussian at every step
  (a visibility-masked or sparse update such as upstream's `sparse_adam` is a different optimizer) — and its
  hyper-parameters and learning-rate schedules, one optimizer step per iteration.
- **Precision**: the six Gaussian parameters and the optimizer state stay float32. Changes of the arithmetic itself
  (fusion, reduction order, kernel design, launch configuration, tiling, culling that does not change the image,
  memory layout, mixed precision inside kernels) are allowed and are judged by gate 3.
- The code may not detect whether it is being timed or judged, read `GS_BENCH_*` variables, or behave differently
  during evaluation.
- **Hook points the judge needs** (`bench/gs_entry.py`): `train.py` with a top-level `training(dataset, opt, pipe,
  testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from)` (called once, with empty
  test, save and checkpoint lists); the argument groups `ModelParams`, `OptimizationParams` and `PipelineParams` of
  `arguments` (they parse the recipe's command line); exactly one `scene.gaussian_model.GaussianModel` whose
  `_xyz`, `_features_dc`, `_features_rest`, `_opacity`, `_scaling`, `_rotation` and `active_sh_degree` are the
  trained model in upstream's parameterization; every training iteration renders its camera once through
  `gaussian_renderer.render` (looked up when `train.py` is imported; you may change how it is implemented); the
  Gaussian optimizer is a `torch.optim.Optimizer` whose parameter groups are named `xyz`, `f_dc`, `f_rest`,
  `opacity`, `scaling`, `rotation`, stepped once per iteration through its `step()`.
- **Switches**: put environment overrides for your code in `/submission/switches.json` as
  `{"environment": {"NAME": "value"}}` (empty mapping if none). Variables the judge manages (`reserved_env` in
  `bench/bench.json`) and `GS_BENCH_*` are refused; an invalid file is a rejection.
- **Read-only**: `/opt/researchboost` (judge, reference, the reference's built extensions, startup measurements), the
  assets (`/opt/researchboost/assets`) and the training environment `/opt/gs-env`. Code you need must be part of
  `/submission/gaussian-splatting` or installed in `/opt/candidate-env`; the image's CUDA toolkit is in
  `/usr/local/cuda`.
- Commit your changes in `/submission/gaussian-splatting`. Terminate your own processes before you finish.
