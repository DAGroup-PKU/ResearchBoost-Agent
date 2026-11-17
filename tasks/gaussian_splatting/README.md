# gaussian_splatting: 3D Gaussian Splatting training (1 GPU, tier N1)

Reference: [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) @ `54c035f`
with upstream's accelerated rasterizer (diff-gaussian-rasterization, branch `3dgs_accel` @ `26ce026`), simple-knn
`86710c2` and fused-ssim `1272e21`, trained with upstream's `train.py` defaults (`bench/recipe.py`: 30,000
iterations, `--eval`) on the Tanks&Temples truck scene of upstream's T&T+DB data set, on one Hopper GPU with at least
130 GiB (H200). Accepted: tier N1 only: every exact item identical to the reference (the camera sequence, the learning
rates and active SH degree at the marks, the steps where the number of Gaussians changes, the dtypes of the parameters
and optimizer states), and a full run on the first frozen seed whose test-view PSNR, SSIM, LPIPS and log number of
Gaussians, scored by the judge's own evaluation with the reference rasterizer, each differ from the reference's run on
that seed by at most 3 × the reference's seed-to-seed standard deviation (one seed, the same rule in every check). The
agent's `check` times the candidate once against the reference timing frozen at preparation; only the final
evaluation (`check --final`, the verifier) times reference → candidate → reference. The judge is `environment/workspace/bench/` (`bench.py`, `gs_entry.py`, `gs_eval.py`,
`recipe.py`, `traj.py`, `stats.py`).

## Assets

`RESEARCHBOOST_ASSETS_ROOT/gaussian_splatting` holds the following (about 2 GB); provision it before the first trial
with `python scripts/prepare_assets.py --task gaussian_splatting` (or
`python3 -I environment/asset-prep/scripts/prepare_gs_assets.py --assets-dir <root>/gaussian_splatting`):

- `download/tandt_db.zip`: upstream's T&T+DB COLMAP data set (the link in upstream's README), SHA-256 `816e62f2...`;
- `tandt/truck/`: the truck scene extracted from it (`images/`, `sparse/0/`), plus `sparse/0/points3D.ply` written
  from `points3D.bin` exactly as upstream writes it on first use (the scene is mounted read-only in the trial);
- `torch/hub/checkpoints/`: torchvision's VGG-16 weights (`vgg16-397923af.pth`) and the LPIPS v0.1 VGG weights
  (`vgg.pth`) of the judge's evaluation, both SHA-256-checked;
- optionally `pack/`: a preparation pack (below).

## Preparation

Full preparation (`environment/bootstrap.py`) takes about 100 minutes on an H200: two quick records of the reference
(their exact items must be identical), eight full reference runs (seeds 1–8) scored by the judge, the frozen inputs,
the unchanged tree's quick and full checks, and three controls (a tree that always draws the first remaining camera
and a tree with half the D-SSIM weight must be INCORRECT, a tree that computes SSIM with upstream's PyTorch code
instead of fused-ssim must PASS at N1).

A preparation pack skips the reference runs, the full check and the controls (about 14 minutes on an H200). Make it from a
container whose full preparation with controls finished (`task/ready.json`: mode `full`, controls true): copy the
workspace's `bench/records/` and `bench/traj/` into `<assets>/pack/bench/` and write `<assets>/pack/pack.json` with at
least `reference_commit` (the content of the workspace's `task/reference_commit`). A trial restores the pack only when
its image has the same reference commit and its own quick record of the reference has the pack's exact items;
otherwise it runs the full preparation.

## Scratch

`RESEARCHBOOST_SCRATCH_ROOT/gaussian_splatting` is the trial's `/tmp` and is emptied when a trial starts. Trials of
this task that share a scratch root empty each other's: give each concurrent trial its own scratch root (for example
one per GPU) in its env file.
