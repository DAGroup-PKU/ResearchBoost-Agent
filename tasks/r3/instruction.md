# Task: faster R3 Tree-RL steps for BAGEL-7B-MoT

Make the training implementation in `/submission/r3` faster per RL step without changing what it computes, to the
bit. The score is the speedup of your submitted tree over the unchanged upstream reference, measured by the judge
described below; a faster tree counts only if it is **bitwise identical** to the reference (tier N0). There is no
statistical tolerance in this task.

## Setup

- **Hardware:** 4 Hopper GPUs (141 GB each) for the whole trial. Your work and the final evaluation use the same four
  GPUs. Every run also starts the judge's GenEval reward service on these GPUs (four workers, one per GPU, about
  10 GB each).
- **Before you started**, the container ran the judge on the unchanged reference on these GPUs: the reference
  recorded twice in deterministic mode (bitwise identical; frozen as `bench/records/ref.json`), its own quick and full
  checks, and one control (a tree with shifted rollout seeds, rejected). These files are read-only; the log is
  `/var/log/researchboost/prepare.log`.
- **Final evaluation:** when your time is up, every process you started is stopped and the judge's full check runs
  in this same container on `/submission/r3` and `/submission/switches.json`, as the unprivileged `agent` user and
  without network. It runs the gates of `check` below, with gate 1 timed as reference → candidate → reference. Its
  result is the score: when `status` is `PASS`, the reward is `accepted` 1 and `speedup` = `performance.speedup`;
  otherwise both are 0.
- **Network:** you have internet access. The container holds no credentials.
- **Python environments:**
  - `/opt/r3-env` is the image's training environment: upstream's `requirements.txt` pins (torch 2.9.1+cu128,
    torchvision 0.24.1, flash-attn 2.5.8 built for sm_90, transformers 4.46.2, accelerate 0.34.2, numpy 1.26.4, ...;
    opencv-python replaced by the headless build of the same version). The full list is
    `/opt/researchboost/installed-packages.txt`. It is read-only. The judge runs with it, and so does every reference
    run.
  - `/opt/candidate-env` is yours and comes first on `PATH` (`python`, `pip`). It sees every package of the training
    environment, and anything you install there (`pip install ...`, torch included) takes precedence. The judge runs
    every process of your tree with `/opt/candidate-env/bin/python`. User site-packages are ignored.
  - `/opt/reward-env` runs the reward service (upstream yifan123/reward-server, mmdetection 2.28.2, open_clip). It is
    part of the judge, read-only, and the same for the reference and for you.
  - The judge records the library versions of the reference and the candidate runs (`performance.library_versions`);
    a difference is not an error.

## Working time

- Your working time is limited, and you are not told how much of it is left. When it ends, every process you started
  is stopped without warning and the final evaluation runs on what is in `/submission` at that moment: keep it in a
  state you want judged.
- The approximate durations of the judge's runs on these GPUs are in `/var/log/researchboost/prepare.log` (the
  reference's own checks ran there; its full check timed three runs, reference → candidate → reference, where your
  `check` times one). Loading the model takes several minutes of every run; time one `time` run early. Your full
  `check` is about one `time` run when it stops at gate 1 (for example no_benefit), and one record run more
  otherwise. A check holds the GPUs for all of that time.

## The workload

- Code: `/submission/r3` is upstream [sen-ye/R3](https://github.com/sen-ye/R3) at commit `761b0b0` on a branch
  `opt`; make your changes there. `/opt/researchboost/ws/targets/reference` is the same tree, unchanged.
- The judge runs every tree through its own entry, `bench/r3_entry.py`, from a fresh copy of the tree with
  `torchrun --nnodes=1 --nproc-per-node=4`: it imports `train.online_rl` from the tree and calls its `main()` with
  the frozen recipe of `bench/recipe.py`. It installs a global optimizer step post-hook
  (`torch.optim.optimizer.register_optimizer_step_post_hook`); after each step of the training optimizer the hook
  synchronizes CUDA and stamps the time, and after the last step the gate measures, every process exits.
- The recipe (`bench/recipe.py`) is upstream `scripts/train_rl.sh` with the GenEval data and reward of the upstream
  tutorial: BAGEL-7B-MoT (14.6B parameters, `--finetune_from_ema True`), group size 16 with policy groups of 4, two
  rounds (even steps generate an image from the prompt, odd steps reflect on and refine an image of the previous
  step), learning rate 5e-6 after 10 warm-up steps, AdamW beta2 0.95 and eps 1e-8, gradient clipping 1.0, CE/MSE
  weights 1/2, text/image KL weights 0.001, frozen ViT, 196/384 reasoning tokens, 16/21 denoising steps, CFG text/image
  4.0/1.5, SDE sampling, max latent size 64, seeds 4396 / 42, `--total_steps 1000` (the per-step seeds depend on it),
  evaluation at step 0 on the first eight GenEval test prompts, two per rank (`--eval_freq 50`), wandb offline.
  Upstream runs on 8 GPUs; here the 4 ranks form one policy group, so each step does half of upstream's global work
  with the same work per GPU, and FSDP shards over the 4 ranks (`--num_shard 4`).
- Data and weights: the judge's frozen copies of upstream's `data/rl_train/train_geneval.jsonl` and of the first
  eight lines of `data/rl_test/test_geneval.jsonl` (in `bench/data/`), and the public BAGEL-7B-MoT checkpoint
  (revision `5019f57d`) in `/opt/researchboost/assets/BAGEL-7B-MoT`, read-only.
- What one RL step does (`train/online_rl.py`, the loop `for curr_step in range(...)`): every rank rolls out its
  samples (text reasoning with sampling, then image denoising with the SDE sampler; the refine step conditions on an
  image from the rollout buffer), the images go to the GenEval reward service, advantages are computed per policy
  group, the samples are packed, the frozen reference model and the policy run forward, the GRPO and KL losses
  backward, then gradient clipping, one AdamW step and the scheduler step. On these 4 GPUs the reference peaks at
  about 131 GiB allocated per GPU, with the reward service's worker on the same GPU: about 6–10 GB is left.

## The judge

Run it from `/opt/researchboost/ws` with the training environment's Python:

```
cd /opt/researchboost/ws
/opt/r3-env/bin/python bench/bench.py check --tree /submission/r3 --config /submission/switches.json --out /tmp/check-1
```

It writes `<output directory>/check.json`. It runs every gate the final evaluation runs, with the same rules; only
gate 1 differs: your `check` times your tree once against the frozen reference timing made at startup, the final
evaluation times reference → candidate → reference. A result counts when the final evaluation's `status` is `PASS`;
the score is its `performance.speedup`. There is no statistical gate in this task: gate 2 accepts only a bitwise
identical tree.

Other commands (see `/opt/r3-env/bin/python bench/bench.py --help`); none of them is a score:

- `check --quick`: gates 0 and 2 only (no timing); `check --skip-timing`: everything but gate 1.
- The final evaluation's own mode (`check --final`) is not available to you.
- `time --tree <dir> --config <file> --out <dir> [--paired]`: one timing run (seconds per step in each window, peak
  memory, and the speed relative to the frozen reference timing measured on this machine before you started,
  `speedup_vs_frozen_reference`, over the mean of the windows); `--paired` also times the reference right before it
  (a screening number, not a verdict).
- `profile --tree <dir> --config <file> --out <dir> [--from K] [--iters M]`: torch.profiler over optimizer steps
  K..K+M-1 on rank 0 (default 11 and 2). Nsight Systems is not installed.

`bench/bench.py` takes the GPU lock `/tmp/r3_gpu.lock` itself; do not wrap it in `flock`. Wrap any other command that
uses the GPUs as `flock /tmp/r3_gpu.lock <command>`. Every judge run needs 100 GiB free under `/tmp`; put output
directories under `/tmp` and delete them when you no longer need them. The startup measurements are in
`bench/records/` (the frozen reference record), `task/` (frozen inputs, the reference timing of this machine) and
`runs/bootstrap/` (logs of the reference runs).

`check` applies the gates in order and stops at the first that fails:

0. **Structure**: `train/online_rl.py` must exist and its `main()` must run the frozen recipe through the optimizer
   steps a gate measures. Otherwise ERROR.
1. **Benefit**: independent runs with cold compile caches and a fresh reward service each, on this machine,
   production settings.
   - Each run has 16 optimizer steps: steps 0–9 warm up (step 0 includes the evaluation), then three windows of two
     steps (10–11, 12–13, 14–15), each with one generate and one refine step. Seconds per step of a window = (end of
     its last step − end of the step before it) / 2, the slowest rank deciding.
   - Your `check`: one run of your tree. Per window, ratio = the frozen reference timing (`task/timing_ref.json`:
     per window the faster of the 2 reference runs made at startup) / your run; `performance.speedup` = the mean of the
     window ratios.
   - The final evaluation: reference → candidate → reference, all measured in that check. Per window, ratio =
     min(reference A, reference B) / candidate; `performance.speedup` = the mean of the window ratios.
   - In the final evaluation, a fingerprint (command, GPUs, timing plan, reward service) that differs from the first
     reference run → ERROR. Library versions are recorded for both and may differ.
   - In the final evaluation, the two reference runs differing by more than 15% in a window → INVALID (measured again
     later). Your own `check` reports a frozen reference spread beyond 15% (`notes`) but does not reject the run.
   - A speedup that does not exceed max(1.02, 1 + 2 × the reference spread) → UNAVAILABLE (no_benefit), rejected
     without a precision evaluation. The reference spread is the largest relative gap between the two reference runs
     over the windows: those of the final evaluation there, the frozen ones (`spread` in `task/timing_ref.json`) in
     your `check`.
2. **Bitwise**: in deterministic mode (`torch.use_deterministic_algorithms` (warn only), cuDNN deterministic,
   `CUBLAS_WORKSPACE_CONFIG`, flash-attn's deterministic backward forced through the `flash_attn` module functions the
   training code calls), 14 optimizer steps (0–13, with the step-0 evaluation). Everything must equal the frozen
   reference record exactly, otherwise INCORRECT:
   - the number of optimizer steps, the learning rates of every step and the optimizer's hyper-parameters, and no
     steps of any other optimizer;
   - after step 13, the SHA-256 of the full tensor of every parameter the optimizer holds and of every tensor of its
     state (`exp_avg`, `exp_avg_sq`, `step`), compared as a set (the sharding may change; the values may not);
   - every request the reward service received during the 14 steps: the SHA-256 of each JPEG image, of the prompts'
     metadata, and of the service's response.
- **Tier**: only N0 (everything bitwise identical) is accepted in this task.

## Rules

- **Training semantics must not change** (gate 2 checks it bitwise): the model and its parameters, prompts and their
  order, rollout sampling and the number of samples, rewards and their use, advantages, losses and their weights, the
  frozen reference model, the optimizer and its hyper-parameters, the learning-rate schedule, gradient clipping, and
  the amount and order of random draws. No FP8, INT8 or FP16.
- **Bitwise identity is the bar.** Any change of arithmetic, reduction order, kernel choice with different rounding,
  or RNG consumption shows up in gate 2. Faster code that computes the same bits (less host overhead, overlap,
  avoided recomputation and copies, fused kernels that reproduce the reference's reduction order and rounding, better
  scheduling of rollouts and reward requests) is what can pass.
- **Deterministic mode**: gate 2 enables torch's deterministic algorithms. Code you add that has no deterministic
  implementation must take a deterministic path of the same precision when
  `torch.are_deterministic_algorithms_enabled()` is true. Otherwise the code may not detect whether it is being timed
  or judged, or behave differently during evaluation.
- **The reward service is the judge's**: its code, configuration and weights are fixed; your tree talks to it only
  through the URL the recipe passes, as upstream does.
- **No existing optimized versions.** Do not look at, download or use code from any published optimized fork of R3
  or BAGEL training. The agent's logs are checked after the trial; a trial that accessed them is
  void.
- **Switches**: put environment overrides for your code in `/submission/switches.json` as
  `{"environment": {"NAME": "value"}}` (empty mapping if none). Variables the judge manages (`reserved_env` in
  `bench/bench.json`) and `R3_*` are refused; an invalid file is a rejection.
- **Read-only**: `/opt/researchboost` (judge, reference, startup measurements), the assets
  (`/opt/researchboost/assets`), the training environment `/opt/r3-env`, the reward environment `/opt/reward-env` and
  service `/opt/reward-server`. Code you need must be part of `/submission/r3` or installed in `/opt/candidate-env`;
  the image's CUDA toolkit is in `/usr/local/cuda`.
- Commit your changes in `/submission/r3`. Terminate your own processes before you finish.
