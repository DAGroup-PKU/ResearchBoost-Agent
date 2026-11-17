# Task: faster nanochat D24 pretraining

Make the training implementation in `/submission/nanochat` faster per optimizer update without changing what it
computes. The score is the speedup of your submitted tree over the unchanged upstream reference, measured by the judge
described below; a faster tree counts only if it passes every gate: bitwise identical to the reference (tier N0), or
not bitwise identical but statistically equivalent on short training trajectories (tier N1).

## Setup

- **Hardware:** 2 Hopper GPUs (141 GB each) for the whole trial. Your work and the final evaluation use the same two
  GPUs.
- **Before you started**, the container ran the judge on the unchanged reference on these GPUs: the reference
  recorded twice in deterministic mode (bitwise identical; frozen as `bench/records/ref.json`), its trajectories for
  six seeds and their frozen spread (`bench/traj/`), its own quick and full checks, and two controls (a tree with a
  changed loss soft-cap, rejected; a tree that sums the same update in another order, accepted at N1). These files are
  read-only; the log is `/var/log/researchboost/prepare.log`.
- **Final evaluation:** when your time is up, every process you started is stopped and the judge's full check runs
  in this same container on `/submission/nanochat` and `/submission/switches.json`, as the unprivileged `agent` user
  and without network. It runs the gates of `check` below, with gate 1 timed as reference → candidate → reference.
  Its result is the score: when `status` is `PASS`, the reward is `accepted` 1 and `speedup` =
  `performance.speedup`; otherwise both are 0.
- **Network:** you have internet access. The container holds no credentials.
- **Python environments:**
  - `/opt/nanochat-env` is the image's training environment: upstream's `uv.lock` for the `gpu` extra (torch
    2.9.1+cu128, triton 3.5.1, kernels 0.11.7, numpy 1.26.4, pyarrow 21.0.0, rustbpe 0.1.0, tiktoken 0.11.0, ...).
    The full list is `/opt/researchboost/installed-packages.txt`. It is read-only. The judge runs with it, and so
    does every reference run.
  - FlashAttention-3 is loaded as upstream loads it, `kernels.get_kernel("varunneal/flash-attention-3")`, from the
    image's Hugging Face cache `/opt/researchboost/hf` (pinned to commit `de87b9b5`, offline).
  - `/opt/candidate-env` is yours and comes first on `PATH` (`python`, `pip`). It sees every package of the training
    environment, and anything you install there (`pip install ...`, torch included) takes precedence. The judge runs
    every process of your tree with `/opt/candidate-env/bin/python`. User site-packages are ignored.
  - The judge records the library versions of the reference and the candidate runs (`performance.library_versions`);
    a difference is not an error.

## Working time

- Your working time is limited, and you are not told how much of it is left. When it ends, every process you started
  is stopped without warning and the final evaluation runs on what is in `/submission` at that moment: keep it in a
  state you want judged.
- The approximate durations of the judge's runs on these GPUs are in `/var/log/researchboost/prepare.log` (the
  reference's own checks ran there; its full check timed three runs, reference → candidate → reference, where your
  `check` times one). Startup and compilation take a few minutes of every run; time one `time` run early. Your full
  `check` is about one `time` run when it stops at gate 1 (for example no_benefit), plus one record run for a bitwise
  identical tree, plus one gate-3 trajectory (150 updates) for one that is not; `check --n1` is the record-free part:
  one trajectory. A check holds the GPUs for all of that time.

## The workload

- Code: `/submission/nanochat` is upstream [karpathy/nanochat](https://github.com/karpathy/nanochat) at commit
  `92d63d4` on a branch `opt`; make your changes there. `/opt/researchboost/ws/targets/reference` is the same tree,
  unchanged.
- The judge runs every tree through its own entry, `bench/nanochat_entry.py`, from a fresh copy of the tree with
  `torchrun --nnodes=1 --nproc-per-node=2`: it runs the tree's `scripts/base_train.py` as `__main__` (as
  `python -m scripts.base_train` does) with the frozen recipe of `bench/recipe.py`. It installs a global optimizer
  step post-hook (`torch.optim.optimizer.register_optimizer_step_post_hook`); after each update of the training
  optimizer the hook synchronizes CUDA and stamps the time, and after the last update the gate measures, every process
  exits.
- The recipe (`bench/recipe.py`) is upstream's D24 run of `runs/speedrun.sh`: depth 24 (model dim 1536, 12 heads of
  128, about 1.38B parameters), sliding-window pattern SSSL, sequence length 2048, 16 sequences per GPU, 262,144
  tokens per update (4 accumulation steps on 2 GPUs: upstream's 1,048,576 on 8 GPUs scaled by 2/8), tensorwise FP8 linear layers (`--fp8`), data:param ratio 8
  (the training horizon, which sets the learning-rate warmdown, the Muon momentum and the weight-decay schedules),
  every evaluation, sampling and checkpoint disabled. Upstream's optimizer is unchanged: `MuonAdamW` (Muon for the
  matrices of the transformer blocks, AdamW for embeddings, the head and the scalars), sharded across the ranks.
- Data: the judge's frozen ClimbMix shards (`karpathy/climbmix-400b-shuffle`; the last shard is the validation
  split) and the tokenizer trained on them with upstream's `scripts.tok_train`, in `/opt/researchboost/assets/data`,
  read-only. Every run gets a fresh `NANOCHAT_BASE_DIR` whose `base_data_climbmix` and `tokenizer` link to them; set
  your own when you run the training outside the judge.
- What one update does (`scripts/base_train.py`, the `while True:` loop): 4 micro-batches per rank, each a forward
  and backward of 16 × 2048 tokens of the BOS-aligned best-fit loader through the compiled model, the next batch
  fetched while the GPU works; then the learning-rate, momentum and weight-decay schedules, one `optimizer.step()`
  (gradients averaged across the ranks by reduce-scatter / all-reduce inside the optimizer, the update of each rank's
  shard, the parameters all-gathered) and `zero_grad`.

## The judge

Run it from `/opt/researchboost/ws` with the training environment's Python:

```
cd /opt/researchboost/ws
/opt/nanochat-env/bin/python bench/bench.py check --tree /submission/nanochat --config /submission/switches.json --out /tmp/check-1
```

It writes `<output directory>/check.json`. It runs every gate the final evaluation runs, with the same rules; only
gate 1 differs: your `check` times your tree once against the frozen reference timing made at startup, the final
evaluation times reference → candidate → reference. A result counts when the final evaluation's `status` is `PASS`;
the score is its `performance.speedup`.

Other commands (see `/opt/nanochat-env/bin/python bench/bench.py --help`); none of them is a score:

- `check --quick`: gates 0 and 2 only (no timing, no trajectories): `PASS` when bitwise identical, `NOT_BITWISE`
  when the exact items pass but the parameters differ (gate 3 decides those), `INCORRECT` when an exact item fails.
  `check --skip-timing`: everything but gate 1.
- `check --n1`: gates 0 and 3 only. After a check whose gate 2 was not bitwise identical, this runs the statistical
  gate again (for example after a change that only affects numerics) without repeating the timing and the record.
- The final evaluation's own mode (`check --final`) is not available to you.
- `time --tree <dir> --config <file> --out <dir> [--paired]`: one timing run (seconds per update in each window,
  tokens/s, peak memory, and the speed relative to the frozen reference timing measured on this machine before you
  started, `speedup_vs_frozen_reference`, over the mean of the windows); `--paired` also times the reference right
  before it (a screening number, not a verdict).
- `profile --tree <dir> --config <file> --out <dir> [--from K] [--iters M]`: torch.profiler over updates K..K+M-1
  on rank 0 (default 12 and 2). Nsight Systems is not installed.

`bench/bench.py` takes the GPU lock `/tmp/nanochat_gpu.lock` itself; do not wrap it in `flock`. Wrap any other command
that uses the GPUs as `flock /tmp/nanochat_gpu.lock <command>`. Every judge run needs 50 GiB free under `/tmp`; put
output directories under `/tmp` and delete them when you no longer need them. The startup measurements are in
`bench/records/` and `bench/traj/` (the frozen reference record and trajectories), `bench/data/probe.pt` (the probe
batch), `task/` (frozen inputs, the reference timing of this machine) and `runs/bootstrap/` (logs of the reference
runs).

`check` applies the gates in order and stops at the first that fails:

0. **Structure**: `scripts/base_train.py`, `nanochat/gpt.py` and `nanochat/dataloader.py` must exist. Otherwise
   ERROR.
1. **Benefit**: independent runs with cold compile caches on this machine, production settings.
   - Each run has 34 updates: updates 0–9 warm up (compilation included), then three windows of eight updates (10–17,
     18–25, 26–33). Seconds per update of a window = (end of its last update − end of the update before it) / 8, the
     slowest rank deciding.
   - Your `check`: one run of your tree. Per window, ratio = the frozen reference timing (`task/timing_ref.json`:
     per window the faster of the 2 reference runs made at startup) / your run; `performance.speedup` = the mean of the
     window ratios.
   - The final evaluation: reference → candidate → reference, all measured in that check. Per window, ratio =
     min(reference A, reference B) / candidate; `performance.speedup` = the mean of the window ratios.
   - In the final evaluation, a fingerprint (command, GPUs, timing plan) that differs from the first reference run →
     ERROR. Library versions are recorded for both and may differ.
   - In the final evaluation, the two reference runs differing by more than 5% in a window → INVALID (measured again
     later). Your own `check` reports a frozen reference spread beyond 5% (`notes`) but does not reject the run.
   - A speedup that does not exceed max(1.02, 1 + 2 × the reference spread) → UNAVAILABLE (no_benefit), rejected
     without a precision evaluation. The reference spread is the largest relative gap between the two reference runs
     over the windows: those of the final evaluation there, the frozen ones (`spread` in `task/timing_ref.json`) in
     your `check`.
   - A run whose log shows torch.compile hitting its recompile limit, or CUDA graphs requested but skipped, is not a
     valid timing run.
2. **Record**: in deterministic mode (`torch.use_deterministic_algorithms` (warn only), Inductor kernels never
   timed against each other: the judge takes the first configuration Inductor lists for each kernel, cuDNN
   deterministic, `CUBLAS_WORKSPACE_CONFIG`, and FlashAttention-3's deterministic backward forced on the module
   `kernels.get_kernel` returns), 10 updates (0–9).
   - Exact items, otherwise INCORRECT: the number of updates and no steps of any other optimizer; exactly one model
     initialized through `nanochat.gpt.GPT.init_weights` and exactly one training loader created through
     `nanochat.dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit`; the hyper-parameters every
     parameter was stepped with at every update (kind, learning rate, momentum, weight decay, betas, eps, Newton–Schulz
     steps, beta2); the token rows every rank consumed, in order (an exact digest of the inputs and targets of each
     row; rows may be grouped into micro-batches differently); the names, shapes and dtypes of the model's
     parameters; no optimizer-state tensor of a dtype the reference does not use.
   - Then the SHA-256 of every model parameter after every update. All identical to the frozen reference record →
     **PASS (N0)**. Otherwise gate 3 decides.
3. **Statistical**: for every candidate, also when gate 2 is bitwise identical (a bitwise candidate keeps N0 when
   gate 3 passes).
   - With the production settings, as timed (not deterministic mode; Inductor autotunes as it does in your timing
     runs), a paired short trajectory from scratch on the first frozen seed (seed 1): 150 updates of
     the recipe. The seed multiplies every initial parameter by (1 + 2⁻⁶ z), z ~ N(0, 1) from a generator seeded with
     it; the data stream is the recipe's, identical for every seed.
   - After every update each rank evaluates the model (the instance `GPT.init_weights` initialized, called as
     `model(inputs, targets)` without gradients) on its rows of the frozen probe batch (4 rows of 2048 tokens of the
     validation shard per rank); the update's value is the mean over the ranks.
   - 6 endpoints, each the mean of these values over a window of updates: 5–15, 15–30, 30–50, 50–80, 80–115, 115–150.
   - σ = the reference's standard deviation across its own 6 seeds (`sigma_seed` in `bench/traj/frozen.json`).
   - The bias d = candidate − reference of each endpoint on that seed: every |d| ≤ 3σ → **PASS (N1)** (verdict
     `equivalent`); any |d| > 3σ → INCORRECT (verdict `different`). One seed, no further seeds; the same rule in your
     `check`, `check --n1` and the final evaluation.
- **Tier**: N0 = everything bitwise identical; N1 = not bitwise identical, every exact item identical, statistically
  equivalent. A changed dtype of a parameter or of the optimizer state is an exact-item failure (INCORRECT).

## Rules

- **Training semantics must not change** (gate 2 compares the exact items, gates 2 and 3 the numerics): the data and
  its order (the rows each rank trains on), the tokens per update, the model, its architecture and its parameters,
  the loss, the optimizer (Muon for the block matrices, AdamW for the rest) and its hyper-parameters, the learning-rate,
  momentum and weight-decay schedules, one optimizer step per update on every parameter.
- **Precision**: the reference is upstream's recipe: tensorwise FP8 for the linear layers it converts, bf16 activations,
  fp32 parameters (bf16 embeddings) and fp32 optimizer state. Do not move more of the model to FP8, to another FP8
  scaling recipe, to INT8 or to FP16, and do not lower the precision of parameters or optimizer state. Changes of the
  arithmetic itself (fusion, reduction order, kernel choice, communication layout) are allowed and are judged by
  gates 2 and 3.
- **Deterministic mode**: gate 2 enables torch's deterministic algorithms; gate 3 and the timing run the
  production settings, so gate 3 judges the path that is timed, for every candidate. Compile-time autotuning
  (Inductor's max-autotune, coordinate descent and the like) is allowed: gate 3 judges the kernels it picks. Code you add that has no
  deterministic implementation must take a deterministic path of the same precision when
  `torch.are_deterministic_algorithms_enabled()` is true. Otherwise the code may not detect whether it is being timed
  or judged, read `NANOCHAT_BENCH_*` variables, or behave differently during evaluation.
- **Hook points the judge needs** (`bench/nanochat_entry.py`): `scripts/base_train.py` run as `__main__`; the model
  is a `nanochat.gpt.GPT` initialized once through `GPT.init_weights` (its parameters are the trained weights, its
  `forward(inputs, targets)` returns the mean loss); the training batches come from one
  `nanochat.dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit` (you may change how it is
  implemented, not what it yields); the training optimizer is a `torch.optim.Optimizer` stepped once per update
  through its `step()`; FlashAttention-3 comes from `kernels.get_kernel` (or your own kernel with a deterministic path).
- **No existing optimized versions.** Do not look at, download or use code from published optimized versions of
  this training: forks or repositories whose purpose is to speed up nanochat's pretraining (including any published by
  the authors of this benchmark). Upstream karpathy/nanochat itself, PyTorch, kernel libraries and their
  documentation are fine. The agent's logs are checked after the trial; a trial that accessed such code is void.
- **Switches**: put environment overrides for your code in `/submission/switches.json` as
  `{"environment": {"NAME": "value"}}` (empty mapping if none). Variables the judge manages (`reserved_env` in
  `bench/bench.json`) and `NANOCHAT_BENCH_*` are refused; an invalid file is a rejection.
- **Read-only**: `/opt/researchboost` (judge, reference, startup measurements, the kernel cache), the assets
  (`/opt/researchboost/assets`) and the training environment `/opt/nanochat-env`. Code you need must be part of
  `/submission/nanochat` or installed in `/opt/candidate-env`; the image's CUDA toolkit is in `/usr/local/cuda`.
- Commit your changes in `/submission/nanochat`. Terminate your own processes before you finish.
