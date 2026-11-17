# Task: faster YOCO-3B pretraining

Make the training implementation in `/submission/yoco` faster per training update without changing what it
computes. The score is the speedup of your submitted tree over the unchanged upstream reference, measured by the judge
described below; a faster tree counts only if it passes every gate.

## Setup

- **Hardware:** 2 × H200 for the whole trial. Your work and the final evaluation use the same two GPUs.
- **Before you started**, the container ran the judge on the unchanged reference on these GPUs: the reference's
  bitwise records, the loader digest, its trajectories and their frozen spread, the per-update random-number stream,
  and the judge's own checks. These files are read-only; the log is `/var/log/researchboost/prepare.log`.
- **Final evaluation:** when your time is up, every process you started is stopped and the judge's full check runs
  in this same container on `/submission/yoco` and `/submission/switches.json`, as the unprivileged `agent` user and
  without network. It runs the gates of `check` below, with gate 1 timed as reference → candidate → reference.
  Its result is the score: when `status` is `PASS`, the reward is `accepted` 1 and `speedup` =
  `performance.speedup`; otherwise both are 0.
- **Network:** you have internet access. The container holds no credentials.
- **Python environments:**
  - `/opt/yoco-env` is the image's training environment: torch 2.3.1+cu121, torchvision 0.18.1 and numpy 1.23.0
    (the only pinned packages), then upstream's requirements at the commits the judge was formulated with: the
    fairseq fork upstream names (sunyt32/fairseq, branch `zero_fs`, at `e872403f`), infinibatch, torchscale,
    fairscale 0.4.0, tiktoken; NVIDIA apex (`70018365`, C++ and CUDA extensions for sm_90) and flash-attn 2.6.3. The
    full list is `/opt/researchboost/installed-packages.txt`. It is read-only. The judge runs with it, and so does
    every reference run.
  - `/opt/candidate-env` is yours and comes first on `PATH` (`python`, `pip`). It sees every package of the training
    environment, and anything you install there (`pip install ...`, torch or a fairseq of your own included) takes
    precedence. The judge runs every process of your tree with `/opt/candidate-env/bin/python`, the gate-0 probes
    included. User site-packages (`pip install --user`) are ignored.
  - The judge records the library versions of the reference and the candidate runs (`performance.library_versions`);
    a difference is not an error. Everything else in the gates below applies unchanged.

## Working time

- Your working time is limited, and you are not told how much of it is left. When it ends, every process you started
  is stopped without warning and the final evaluation runs on what is in `/submission` at that moment: keep it in a
  state you want judged.
- Approximate durations of the judge commands for this task, measured on 2 × H200 (2026-10-27), each including
  about a minute of startup: `time` about 1.5 min; `check --quick` about 5 min; one timing run about 1.5 min and one
  gate-3 trajectory about 2.5 min, so the full `check` takes about 2.5 min when it stops at gate 1 (for example
  no_benefit), about 7 min for a bitwise identical candidate and about 10 min for one that is not; `check --n1`
  (gates 0 and 3) about 4 min; `check --skip-timing` is the full check without its timing run. The reference's own
  checks in `/var/log/researchboost/prepare.log` ran in the final evaluation's mode (three timing runs). A check
  holds the GPUs throughout.

## The workload

- Code: `/submission/yoco` is upstream [microsoft/unilm](https://github.com/microsoft/unilm) at commit `f4f7b27a`,
  its `YOCO/` directory, as a repository of its own on a branch `opt`; make your changes there.
  `/opt/researchboost/ws/targets/reference` is the same tree, unchanged.
  - The entry point is `yoco/train.py`: `import models, tasks, criterions; from fairseq_cli.train import cli_main`,
    then `cli_main()`. The judge runs it from `<tree>/yoco` (as upstream's `scripts/train.sh` does: `cd yoco/`), so
    `models`, `tasks` and `criterions` are imported from your tree.
  - The judge launches it from a copy of the tree with `torchrun --nnodes=1 --nproc-per-node=2`.
- The recipe is frozen in `bench/recipe.py`: upstream `scripts/train.sh` with these changes only:
  - `--arch yoco_3b` (the script's `yoco_base` is not registered): dim 3072, FFN 8192, 26 layers (13 self-decoder
    layers with gated retention, 13 cross-decoder layers sharing one KV cache), 24 heads, 8 KV heads in the
    cross-attention, the cl100k_base vocabulary.
  - **The per-GPU load is upstream's**: 2048 tokens per sample, batch 1 per GPU, `--update-freq 1`, bf16
    (`--bf16`: fp32 master weights in fairseq's FP16Optimizer, bf16 model weights), Adam (betas 0.9/0.95, eps 1e-6,
    weight decay 0.05), gradient clipping at 2.0, learning rate 1.5e-4 with polynomial decay over 300000 updates
    after 50 warmup updates.
  - 2 data-parallel processes on one machine. The global batch is 2 × 2048 tokens per update.
  - `--zero-sharding os` (the optimizer state is sharded across the ranks), `--seed 1`, no validation set,
    `--save-dir` in a temporary directory, `--max-update` only to end a run (the schedule uses
    `--total-num-update 300000`).
- Data and the tokenizer: the corpus in `/opt/researchboost/assets/data/Data` (fineweb-edu and codeparrot, mixed
  0.8/0.2 by upstream's infinibatch loader, in upstream's README layout) and tiktoken's cl100k_base in
  `/opt/researchboost/assets/tiktoken_cache`, read-only. The judge sets `TIKTOKEN_CACHE_DIR` to that directory for
  every run it launches; set it yourself when you run the training outside the judge.
- What one update does (fairseq's `train()` loop: `for i, samples in enumerate(progress):` →
  `trainer.train_step(samples)`): on every GPU one forward and backward of the 2048-token sample through the model
  and the cross-entropy criterion, the gradients reduced across the GPUs, normalized by the sample size and clipped,
  one fp32 Adam step on the master weights (each rank stepping its shard of the optimizer state, then the updated
  weights broadcast), the bf16 model weights refreshed from the master weights, and the scheduler step.

## The judge

Run it from `/opt/researchboost/ws` with the training environment's Python:

```
cd /opt/researchboost/ws
/opt/yoco-env/bin/python bench/bench.py check --tree /submission/yoco --config /submission/switches.json --out /tmp/check-1
```

It writes `<output directory>/check.json`. It runs every gate the final evaluation runs, with the same rules; only
gate 1 differs: your `check` times your tree once against the frozen reference timing made at startup, the final
evaluation times reference → candidate → reference. A result counts when the final evaluation's `status` is `PASS`;
the score is its `performance.speedup`.

Other commands (see `/opt/yoco-env/bin/python bench/bench.py --help`); none of them is a score:

- `check --quick`: gates 0 and 2 only (no timing, no trajectories); `check --skip-timing`: everything but gate 1.
- `check --n1`: gates 0 and 3 only. After a check whose gate 2 was not bitwise identical, this runs the statistical
  gate again (for example after a change that only affects numerics) without repeating the timing and the records.
- The final evaluation's own mode (`check --final`) is not available to you.
- `time --tree <dir> --config <file> --out <dir> [--paired]`: one timing run (s/update, tokens/s, peak memory) with
  its speedup over the frozen reference timing (`speedup_vs_frozen_reference` = the frozen reference's mean update
  time ÷ this run's; larger is faster). The frozen reference is the mean of 3 reference runs made at startup on this
  machine; `reference_noise` gives their spread (max − min) / mean, and `beyond_reference_noise` says whether the gain
  over the frozen reference is larger than twice that spread, the margin the bar of gate 1 also uses. If the
  frozen reference timing is missing, `time` first times the reference once and reports the speedup over that run
  (kept for later `time` runs on the same GPUs); `reference_noise` then says its noise is not measured. `--paired`
  also times the reference right before it and reports that ratio (screening numbers, not a verdict).
- `profile --tree <dir> --config <file> --out <dir> [--iters N]`: torch.profiler over N updates after the warmup
  (rank 0). Nsight Systems is not installed.

`bench/bench.py` takes the GPU lock `/tmp/yoco_gpu.lock` itself; do not wrap it in `flock`. Wrap any other command
that uses the GPUs as `flock /tmp/yoco_gpu.lock <command>`. Every judge run needs 50 GiB free under `/tmp`; put
output directories under `/tmp` and delete them when you no longer need them. The startup measurements are in
`bench/records/` and `bench/traj/` (reference records and trajectories), `task/` (loader digest, random-number stream,
frozen inputs) and `runs/bootstrap/` (logs of the reference runs).

`check` applies the gates in order and stops at the first that fails:

0. **Structure (CPU)**
   - The entry `yoco/train.py`, the live `cli_main()` and `main()` of `fairseq_cli.train`, and `train()` except the
     body of its `for i, samples in enumerate(progress):` loop must have the same AST as the reference, taken after
     importing your tree's modules the way the entry does (`bench/yoco_setup.py`). Different → ERROR. Reason: the
     numerics adapter (`bench/yoco_adapter.py`) re-creates that setup and runs your loop body unchanged at every
     update. Changes go where the loop body calls them (`Trainer.train_step`, the task, criterion, model, optimizer)
     or into the loop body.
   - Your tree's data loader (`task.get_batch_iterator`, as the trainer calls it) must yield bitwise the same first
     batches as the reference on every rank's shard, otherwise INCORRECT.
1. **Benefit**: independent runs of the real `torchrun` entry point with cold compile caches on this machine. Each
   run has 5 warmup updates + 2 windows of 10 updates; a run's update time is its windows ÷ its updates.
   - Your `check`: one run of your tree. `performance.speedup` = the frozen reference timing (the mean of the 3
     reference runs made at startup, `task/timing_ref.json`) ÷ your run's update time.
   - The final evaluation: reference → candidate → reference, measured then; `performance.speedup` = the mean update
     time of the two reference runs ÷ the candidate's. The median update times are reported as diagnostics.
   - Every rank of every judge run (timing, records, trajectories) is pinned to its share of the container's CPUs:
     the physical cores on its GPU's NUMA node, split between the ranks there, with PyTorch's intra-op threads set
     to that share's physical cores (`bench/cpu_affinity.py`). Reference and candidate are pinned the same way. Each
     run reports how many CPUs the container kept busy (`cpu.busy_cpus`, and `busy_cpus_in_windows` for timing).
   - In the final evaluation, a fingerprint (command, math mode, GPUs, timing plan) that differs from the first
     reference run → ERROR. Library versions are recorded for both and may differ.
   - The random-number stream at the end of each update must match the frozen reference update by update and rank
     by rank, otherwise ERROR.
   - In the final evaluation interference makes the result INVALID (measured again later): a run's window range
     ((max − min) / mean) above 6%, or a reference run more than 15% off the frozen reference timing. Your own
     `check` and `time` report these (`notes`) but do not reject the run.
   - A speedup that does not exceed max(1.02, 1 + 2 × the reference spread) → UNAVAILABLE (no_benefit), rejected
     without a precision evaluation. The reference spread is the relative gap between the two reference runs in the
     final evaluation, and the frozen reference runs' (max − min) / mean in your `check`.
   - A run that still compiles inside a timing window, logs `gradient overflow detected` (a skipped update) or matches
     another `invalid_run_signatures` entry of `bench/bench.json` is not a valid timing run.
2. **Bitwise**: in deterministic mode (`torch.use_deterministic_algorithms` (warn only), cuDNN deterministic,
   `CUBLAS_WORKSPACE_CONFIG`, flash-attn's deterministic backward), one case, one process: updates 0 and 1 (warmup,
   learning rate 0 and 3e-6) and, after your loop body has run the updates in between, updates 60 and 61 (learning
   rate about 1.5e-4).
   - Exact items, otherwise INCORRECT: RNG fingerprints before and after each update, forward call counts,
     invariants (update counter, scheduled learning rate and the rate after the update, Adam hyperparameters and step,
     dtypes of the master weights, master gradients and Adam states, each rank's batch digest and a cumulative digest
     of every batch consumed, sample size and token count, criterion calls and fp32 optimizer steps per update, the
     parsed configuration, trainable parameter names and shapes), tensor names and shapes, all values finite,
     cross-rank pattern.
   - All recorded tensors (`loss/`, `grad/`, `delta/`, `optim/`, `buffer/`; gradients and weights of the fp32 master
     weights, Adam's states, the bf16 weights after the update) bitwise identical → **PASS (N0)**. Tensors larger than
     4096 elements are stored as linear CountSketches with exact norms and hashes, so this is still a bitwise
     judgment.
   - Exact items are judged first. Recording fewer tensors than the reference (for example, a parameter that is
     never updated) is an exact-item failure.
3. **Statistical**: for every candidate, also when gate 2 is bitwise identical (a bitwise candidate keeps N0 when
   gate 3 passes).
   - With the production settings, as timed (not deterministic mode), a paired short trajectory from scratch on the first frozen seed: 200 updates of the
     recipe (50 warmup updates, then about 1.5e-4), one process. The seed multiplies the initial fp32 weights by
     (1 + 2⁻⁶ z), z ~ N(0, 1) from a generator seeded with it; the data stream is the recipe's, identical for every
     seed.
   - 6 endpoints, each the mean loss per token (nats, averaged over the ranks) over a window of updates: 5–15,
     15–25, 25–35, 35–50, 50–100, 100–200. An endpoint is one such number per run: the judge compares these six
     numbers, not the whole loss curve.
   - σ = the reference's standard deviation across its own 6 seeds (frozen in `bench/traj/frozen.json`).
   - The bias d = candidate − reference of each endpoint on that seed: every |d| ≤ 3σ → PASS (N1); any
     |d| > 3σ → INCORRECT. One seed, no further seeds; the same rule in your `check`, `check --n1` and the final
     evaluation.
- **Tier**: N0 = everything bitwise identical with the same dtypes and math modes; N1 = every other candidate that
  passes gates 2 and 3. A changed dtype or math mode of any tensor is never N0; the check lists it
  (`correctness.numerics.precision_changed`) and the statistical gate decides as for any N1 candidate.
- Per-tensor tolerances (`bench/thresholds.json`) are diagnostic output only.
  Beyond them, the comparison's status (`correctness.numerics.status`) reads `NOT_BITWISE` and gate 3
  decides; `INCORRECT` there means an exact item failed.

## Rules

- **Training semantics must not change** (gate 0 compares structure, gates 2 and 3 numerics):
  - data and order, per-GPU batch and sequence length, the model and its architecture, the loss, the optimizer and
    its hyperparameters (Adam betas and eps, weight decay, gradient clipping at 2.0), the learning-rate schedule;
  - one optimizer step per update, on every trainable parameter, with gradients normalized by the sample size as the
    reference does;
  - the order and use of random draws (checked update by update and rank by rank in gate 1, and before and after
    each update in gate 2).
- **Precision**: the reference is upstream's recipe: bf16 model weights and activations with fp32 master weights and
  fp32 Adam state. FP8, INT8 and FP16 are not allowed, nor is removing the fp32 master weights or optimizer state.
  Local precision changes are allowed; such a candidate is N1 and must pass the statistical gate.
- **Deterministic mode**: gate 2 enables torch's deterministic algorithms; gate 3 and the timing run the
  production settings, so gate 3 judges the path that is timed, for every candidate. Code you add that has no
  deterministic implementation must take a deterministic path of the same precision when
  `torch.are_deterministic_algorithms_enabled()` is true. Otherwise the code may not read `YOCO_BENCH_*`
  variables, detect whether it is being timed or judged, or behave differently during evaluation.
- **Hook points the judge needs** (the adapter's contract, `bench/yoco_adapter.py`):
  - the loop `for i, samples in enumerate(progress):` in `train()` of `fairseq_cli.train` (its body runs unchanged at
    every update);
  - `trainer.get_model()`, `trainer.get_criterion()` (called as a module once per micro-batch),
    `trainer.optimizer.fp16_params` / `fp32_params` and `trainer.optimizer.fp32_optimizer.step()`; Adam state as torch
    optimizer state (`exp_avg`, `exp_avg_sq`) keyed by the fp32 master parameter, in
    `trainer.optimizer.fp32_optimizer.optimizer` or its `.optim`, with `step` in the param group or the state. If you
    change this layout, keep these attributes as views.
- **Switches**: put environment overrides for your code in `/submission/switches.json` as
  `{"environment": {"NAME": "value"}}` (empty mapping if none). Variables the judge manages (`reserved_env` in
  `bench/bench.json`) and `YOCO_BENCH_*` are refused; an invalid file is a rejection.
- **Read-only**: `/opt/researchboost` (judge, reference, startup measurements), the assets
  (`/opt/researchboost/assets`) and the training environment `/opt/yoco-env`. Code you need must be part of
  `/submission/yoco` or installed in `/opt/candidate-env`; the image's CUDA toolkit is in `/usr/local/cuda`.
- Commit your changes in `/submission/yoco`. Terminate your own processes before you finish.
