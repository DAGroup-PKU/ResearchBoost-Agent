# Task: faster BAGEL-7B-MoT fine-tuning

Make the training implementation in `/submission/bagel_sft` faster per optimizer step without changing what it
computes. The score is the speedup of your submitted tree over the unchanged upstream reference, measured by the judge
described below; a faster tree counts only if it passes every gate: bitwise identical to the reference (tier N0), or
not bitwise identical but statistically equivalent on short training trajectories (tier N1).

## Setup

- **Hardware:** 4 × H200 (141 GB each) for the whole trial. Your work and the final evaluation use the same four GPUs.
- **Before you started**, the container ran the judge on the unchanged reference on these GPUs: the reference
  recorded twice in deterministic mode (bitwise identical; frozen as `bench/records/ref.json`), its trajectories for
  six seeds and their frozen spread (`bench/traj/`), its run on the held-out interleaved data of gate 2b, twice
  (identical; frozen as `bench/conform/ref.json`), three timing runs of the reference (their mean frozen as
  `task/timing/reference.json`), its own quick and full checks, and controls (a changed AdamW beta2 and a tree that
  skips the EMA update, both rejected; a tree that normalizes the CE loss in another order and one whose RMSNorm
  divides by the square root, both accepted; two trees whose attention masks are wrong only on interleaved layouts,
  rejected at gate 2b). These files are read-only; the log is `/var/log/researchboost/prepare.log`.
- **Final evaluation:** when your time is up, every process you started is stopped and the judge's full check runs
  in this same container on `/submission/bagel_sft` and `/submission/switches.json`, as the unprivileged `agent` user
  and without network. It runs the gates of `check` below, with gate 1 timed as reference → candidate → reference.
  Its result is the score: when `status` is `PASS`, the reward is
  `accepted` 1 and `speedup` = `performance.speedup`; otherwise both are 0.
- **Network:** you have internet access. The container holds no credentials.
- **Python environments:**
  - `/opt/bagel-env` is the image's training environment, installed as upstream's README says: Python 3.10,
    upstream's `requirements.txt` (torch 2.5.1+cu124, torchvision 0.20.1, transformers 4.49.0, numpy 1.24.4,
    safetensors 0.4.5, decord 0.6.0, ...) and flash-attn 2.5.8 (built from source for Hopper). The full list is
    `/opt/researchboost/installed-packages.txt`. It is read-only. The judge runs with it, and so does every reference
    run.
  - `/opt/candidate-env` is yours and comes first on `PATH` (`python`, `pip`). It sees every package of the training
    environment, and anything you install there (`pip install ...`, torch included) takes precedence. The judge runs
    every process of your tree with `/opt/candidate-env/bin/python`. User site-packages are ignored.
  - The judge records the library versions of the reference and the candidate runs (`performance.library_versions`);
    a difference is not an error.

## Working time

- Your working time is limited, and you are not told how much of it is left. When it ends, every process you started
  is stopped without warning and the final evaluation runs on what is in `/submission` at that moment: keep it in a
  state you want judged.
- The durations of the judge's runs on these GPUs are in `/var/log/researchboost/prepare.log` (the reference's own
  checks ran there, in the final evaluation's mode: three timing runs in gate 1). Approximately: your `check` makes
  one timing run in gate 1 instead of three, gate 2b one short run and gate 3 one trajectory (two when the first seed is beyond the bound); `check --n1` (gates
  0, 2b and 3) is one short run and one or two trajectory runs. Every run first builds the model and its EMA on the host and loads the released checkpoint into
  both, which takes a few minutes; time one `time` run early. A check holds the GPUs throughout.

## The workload

- Code: `/submission/bagel_sft` is upstream [ByteDance-Seed/Bagel](https://github.com/ByteDance-Seed/Bagel) at
  commit `a2fa77d` on a branch `opt`; make your changes there. `/opt/researchboost/ws/targets/reference` is the same
  tree, unchanged.
- The judge runs every tree through its own entry, `bench/bagel_entry.py`, from a fresh copy of the tree with
  `torchrun --nnodes=1 --nproc-per-node=4`: it runs the tree's `train/pretrain_unified_navit.py` as `__main__` with
  the frozen recipe of `bench/recipe.py`. It installs a global optimizer step post-hook
  (`torch.optim.optimizer.register_optimizer_step_post_hook`); after each step of the training optimizer the hook
  synchronizes CUDA and stamps the time, and after the last step the gate measures, every process exits.
- The recipe (`bench/recipe.py`) is upstream's fine-tuning command (TRAIN.md) on 4 GPUs instead of 8, with
  upstream's per-GPU load: the released BAGEL-7B-MoT checkpoint (about 14.6B parameters: a Qwen2 mixture-of-
  transformers language model with an understanding and a generation expert, a SigLIP vision encoder, the FLUX VAE
  frozen) initializes both the model and its EMA; packed sequences of 10240 expected tokens per GPU (at most 11520,
  samples up to 10240), so a step trains on 4 × 10240 tokens (upstream's 8 GPUs: 8 × 10240); FSDP `HYBRID_SHARD`
  over the 4 GPUs (`--num_shard 4`) with bf16 parameters, reductions and buffers in compute and fp32 master weights;
  activation checkpointing of the decoder, ViT and connector layers; learning rate 2e-5 with upstream's default
  2000-step linear warmup, constant afterwards; AdamW betas 0.9/0.95, eps 1e-15, no weight decay; gradient clipping
  at 1.0; EMA decay 0.9999; global seed 4396, data seed 42; one data worker; wandb offline.
- Data: upstream's `bagel_example` (TRAIN.md): text-to-image (10 parquet chunks), image editing (`seedxedit_multi`,
  10 chunks) and VLM conversations (`llava_ov_si.jsonl`, 1000 rows with their images), read-only in
  `/opt/researchboost/assets/data/bagel_example`, with upstream's data configuration `data/configs/example.yaml`
  (frozen as `bench/data_config.yaml`). Upstream's `data/dataset_info.py` names placeholder paths; the judge points
  its `t2i`, `seedxedit_multi` and `llava_ov` entries at that directory when the module is imported (the editing
  index is rewritten for that location), and changes nothing else. Set your own paths when you run the training
  outside the judge.
- What one step does (`train/pretrain_unified_navit.py`, the `for micro_step, data in enumerate(train_loader):`
  loop): the next packed batch, moved to the GPU; token counts all-reduced; the VAE encodes the images to be
  generated (no gradient); one forward of the FSDP-wrapped model under bf16 autocast returns the per-token CE of the
  text targets and the per-token flow-matching MSE of the image latents; both normalized by the global token counts
  and summed; backward; `clip_grad_norm_(1.0)`; `optimizer.step()`; the scheduler; the EMA update; logging (with its
  all-reduces) every step.

## The judge

Run it from `/opt/researchboost/ws` with the training environment's Python:

```
cd /opt/researchboost/ws
/opt/bagel-env/bin/python bench/bench.py check --tree /submission/bagel_sft --config /submission/switches.json --out /tmp/check-1
```

It writes `<output directory>/check.json`. It runs every gate the final evaluation runs, with the same rules; only
gate 1 differs: your `check` times your tree once against the frozen reference timing made at startup, the final
evaluation times reference → candidate → reference. A result counts when the final evaluation's `status` is `PASS`;
the score is its `performance.speedup`.

Other commands (see `/opt/bagel-env/bin/python bench/bench.py --help`); none of them is a score:

- `check --quick`: gates 0 and 2 only (no timing, no gate 2b, no trajectories): `PASS` when bitwise identical,
  `NOT_BITWISE` when the exact items pass but the numbers differ (gates 2b and 3 decide those), `INCORRECT` when an
  exact item fails. A quick `PASS` is not a full check's: gate 2b still runs in `check` and in the final evaluation.
  `check --skip-timing`: everything but gate 1.
- `check --n1`: gates 0, 2b and 3. After a check whose gate 2 was not bitwise identical, this runs the conformance
  and statistical gates again (for example after a change that only affects numerics) without repeating the timing
  and the record.
- The final evaluation's own mode (`check --final`) is not available to you.
- `time --tree <dir> --config <file> --out <dir> [--paired]`: one timing run (seconds per step in each window, peak
  memory) with its speedup over the frozen reference timing (`speedup_vs_frozen_reference` = the frozen reference's
  seconds per step ÷ this run's; larger is faster). The frozen reference is the mean of 3 reference runs made at
  startup on this machine; `reference_noise` gives their spread (max − min) / mean, and `beyond_reference_noise` says
  whether the gain over the frozen reference is larger than twice that spread, the margin the bar of gate 1 also
  uses. If the frozen reference timing is missing, `time` first times the reference once and reports the speedup over
  that run (kept for later `time` runs on the same GPUs); `reference_noise` then says its noise is not measured.
  `--paired` also times the reference right before it and reports that ratio (screening numbers, not a verdict).
- `profile --tree <dir> --config <file> --out <dir> [--from K] [--iters M]`: torch.profiler over steps K..K+M-1 on
  rank 0 (default 8 and 2). Nsight Systems is not installed.

`bench/bench.py` takes the GPU lock `/tmp/bagel_gpu.lock` itself; do not wrap it in `flock`. Wrap any other command
that uses the GPUs as `flock /tmp/bagel_gpu.lock <command>`. Every judge run needs 50 GiB free under `/tmp`; put
output directories under `/tmp` and delete them when you no longer need them. The startup measurements are in
`bench/records/` and `bench/traj/` (the frozen reference record and trajectories), `task/` (frozen inputs, the
reference timing of this machine) and `runs/bootstrap/` (logs of the reference runs).

Every rank of every judge run (timing, record, trajectories) is pinned to its share of the container's CPUs: the
physical cores on its GPU's NUMA node, split between the ranks there, with PyTorch's intra-op threads set to that
share's physical cores (`bench/cpu_affinity.py`). Reference and candidate are pinned the same way. Each run reports
how many CPUs the container kept busy (`cpu.busy_cpus`, and `busy_cpus_in_windows` for timing).

`check` applies the gates in order and stops at the first that fails:

0. **Structure**: `train/pretrain_unified_navit.py`, `modeling/bagel/bagel.py`, `data/dataset_base.py`,
   `data/dataset_info.py` and `train/fsdp_utils.py` must exist. Otherwise ERROR.
1. **Benefit**: independent runs on this machine, production settings.
   - Each run trains on `bench/data_config_interleave.yaml` (the data of gate 2b, not the recipe's; the final
     evaluation uses another data seed) and has 65 steps: steps 0–4 warm up, then two windows of 30 steps (5–34,
     35–64). Seconds per step of a window = (end of its last step − end of the step before it) / 30, the slowest
     rank deciding; a run's seconds
     per step = the mean of its two windows.
   - Your `check`: one run of your tree. `performance.speedup` = the frozen reference timing (the mean of the 3
     reference runs made at startup, `task/timing/reference.json`) ÷ your run's seconds per step.
   - The final evaluation: reference → candidate → reference, measured then; `performance.speedup` = the mean
     seconds per step of the two reference runs ÷ the candidate's.
   - In the final evaluation, a fingerprint (command, GPUs, timing plan) that differs from the first reference run
     → ERROR. Library versions are recorded for both and may differ.
   - In the final evaluation interference makes the result INVALID (measured again later): a run's window range
     ((max − min) / mean) above 6%, or a reference run more than 15% off the frozen reference timing. Your own
     `check` and `time` report these (`notes`) but do not reject the run.
   - A speedup that does not exceed max(1.02, 1 + 2 × the reference spread) → UNAVAILABLE (no_benefit), rejected
     without a precision evaluation. The reference spread is the relative gap between the two reference runs in the
     final evaluation, and the frozen reference runs' (max − min) / mean in your `check`.
   - A run whose log shows torch.compile hitting its recompile limit, or CUDA graphs requested but skipped, is not a
     valid timing run.
2. **Record**: in deterministic mode (`torch.use_deterministic_algorithms(True)`, strict: an operation without a
   deterministic implementation raises an error; cuDNN deterministic,
   `CUBLAS_WORKSPACE_CONFIG`, and FlashAttention's deterministic backward forced on every function of
   `flash_attn.flash_attn_interface` that has the option), 3 steps (0–2) from the released checkpoint.
   - Exact items, otherwise INCORRECT: the number of steps and no steps of any other optimizer; exactly two FSDP
     roots, the trained model and its EMA; the hyper-parameters of every parameter group at every step (learning
     rate, betas, eps, weight decay, amsgrad, maximize, the number of elements); the batches every rank moved to the
     GPU, in order (a SHA-256 of every field of each `SimpleCustomBatch`); the names, shapes and dtypes of the
     parameters of the model and of the EMA and which of them are trained; no optimizer-state tensor of a dtype the
     reference does not use.
   - Then the SHA-256 of every rank's shard of the model's flat parameters, of their gradients, of both AdamW
     moments and of the EMA's flat parameters, after every step. All identical to the frozen reference record →
     **N0** once gate 2b passes. Otherwise gates 2b and 3 decide (a different FSDP layout is never bitwise identical,
     only that).
2b. **Conformance on interleaved data**: in every check but `--quick`, also for a tree that is bitwise identical in
   gate 2. Upstream's attention mask of a packed sample depends on its layout (causal text, bidirectional image
   spans, noise spans that no later split of the sample may see). Under the recipe's data configuration the editing
   samples never fit the token limits, so every batch of gates 2 and 3 holds only layouts where a noise span is the
   last split of its sample and no two image spans are adjacent: an attention implementation can be wrong on other
   layouts and still be bitwise identical there. Gate 2b runs your tree on a held-out data configuration of the same
   data, `bench/data_config_interleave.yaml` (the editing and VLM groups mandatory, the images at most 512 px, so
   that two- and three-image editing chains fit the recipe's limits; everything else as the recipe), in
   deterministic mode and again in production settings (the code path timing runs), 16 steps each from the released
   checkpoint, and compares each with the frozen reference run of that mode. The final evaluation does this on
   another data seed.
   - The noise is the judge's: while `AutoEncoder.encode` (the VAE posterior sample) or `Bagel.forward` (the latent
     noise) runs, the judge answers their `torch.randn_like` / `torch.randn` / `Tensor.normal_` requests for the
     noise with values it draws itself, keyed by the call's ordinal on the rank (the k-th encode, the k-th forward),
     not by the global random state: reordering your RNG use (a VAE prefetch on a side stream, ...) does not change
     the noise you get. The timesteps are part of the batch.
   - Exact items, otherwise INCORRECT: the batches every rank moves to the GPU identical to the reference's on this
     data, one `Bagel.forward` per step, the same number of MSE tokens. If the judge could not give a call its noise
     (the noise not drawn inside those calls, or drawn as a different number of tensors or elements than upstream):
     ERROR (a rejection).
   - The per-token flow-matching MSE that `Bagel.forward` returns is split into two pools by the reference's own
     masks: N, the tokens of samples with a noise span followed by another split; J, of samples with two adjacent
     bidirectional spans; B, all others. R_N, R_J and R_B are the relative L2 distances of your per-token MSE from
     the reference's over each pool (all ranks and steps). Kernel, fusion and reduction-order changes move the pools
     alike; a mask that is wrong only on interleaved layouts moves N or J alone. R_N and R_J both ≤ 3 × max(R_B,
     0.005) → the gate passes (bench.json `conformance`); otherwise INCORRECT (`nonconforming`). Measured on the
     reference's controls and on accepted trees: legitimate changes give at most about 2.3 (numerics that only
     editing samples see, such as the VAE encoding of their conditioning images, count too); masks wrong on
     interleaved layouts give 15 or more. The gate takes 4–6 minutes of the check.
3. **Statistical**: for every candidate that passed gate 2b, also when gate 2 is bitwise identical (a bitwise
   candidate keeps N0 when gate 3 passes).
   - With the production settings, as timed (not deterministic mode), a paired short trajectory on the first frozen seed (seed 1): 150 steps of the recipe,
     one run. Seed S sets the global seed to 4396 + S and multiplies every trained parameter of the loaded model, and
     its EMA counterpart, by (1 + 2⁻⁷ z), z ~ N(0, 1) from a generator seeded with (S, the parameter's name); the data
     stream is the recipe's, identical for every seed.
   - Series: per step the global mean CE and MSE (from the per-token `ce` and `mse` that `Bagel.forward` returns,
     summed over all ranks and divided by the global counts, as the training script normalizes them) and the
     gradient norm before clipping (what `clip_grad_norm_` returns); every 10 steps the L2 norm of the change over
     the last 10 steps of the model's ViT parameters, its language-model parameters, its other parameters and of the
     EMA.
   - 14 endpoints, each the mean of one series over a window: CE, MSE and the log gradient norm over steps 50–100
     and 100–150; the four displacements over the 10-step blocks of steps 50–100 and 100–140. An endpoint is one
     such number per run: the judge compares these 14 numbers, not the whole curves.
   - σ = the reference's standard deviation across its own 6 seeds (frozen in `bench/traj/frozen.json`).
   - The bias d = candidate − reference of each endpoint on that seed: every |d| ≤ 3σ → **PASS (N1)**. If any
     |d| > 3σ, the candidate is measured once more on the next frozen seed (seed 2, a fresh run, the same rule):
     every |d| ≤ 3σ there → **PASS (N1)**; any |d| > 3σ on seed 2 as well → INCORRECT. At most two seeds; the same
     rule in your `check`, `check --n1` and the final evaluation.
- **Tier**: N0 = everything bitwise identical in gate 2 and gate 2b passed; N1 = not bitwise identical, every exact
  item identical, gate 2b passed, statistically equivalent. A changed dtype of a parameter or of the optimizer state is an exact-item failure (INCORRECT).

## Rules

- **Training semantics must not change** (gate 2 compares the exact items, gates 2, 2b and 3 the numerics): the data, its
  packing and its order (the batches each rank trains on), the tokens per step, the model, its architecture and its
  parameters, the losses and their normalization and weights, the optimizer and its hyper-parameters, the
  learning-rate schedule, gradient clipping, the EMA and its decay, one optimizer step per step on every trained
  parameter.
- **Precision**: the reference is upstream's: fp32 master weights, fp32 AdamW state and fp32 EMA; bf16 compute
  (FSDP mixed precision with bf16 parameters, gradient reductions and buffers, and bf16 autocast). Do not move any of
  it to FP8, INT8 or FP16, and do not lower the precision of parameters, optimizer state or EMA. Changes of the
  arithmetic itself (fusion, reduction order, kernel choice, recomputation, communication layout) are allowed and are
  judged by gates 2, 2b and 3.
- **Deterministic mode**: gate 2 and the first run of gate 2b enable torch's deterministic algorithms, strictly (an operation without a
  deterministic implementation raises). Code you add that has no deterministic implementation must take a
  deterministic path of the same precision when `torch.are_deterministic_algorithms_enabled()` is true. Gate 3 and
  the timing run the production settings, so gate 3 judges the path that is timed, for every candidate. Otherwise the code may not detect whether it is being timed
  or judged, read `BAGEL_BENCH_*` variables, or behave differently during evaluation.
- **Hook points the judge needs** (`bench/bagel_entry.py`): `train/pretrain_unified_navit.py` run as `__main__`;
  the model and its EMA are the only two `torch.distributed.fsdp.FullyShardedDataParallel` roots (the model's
  parameters are the training optimizer's, the EMA's are not), with FSDP's flat parameters; the training optimizer
  is a `torch.optim.Optimizer`, created once and stepped once per step through its `step()`; every training batch
  is moved to the GPU through `data.dataset_base.SimpleCustomBatch.cuda`; `modeling.bagel.bagel.Bagel.forward`
  returns the per-token `ce` and `mse` (a dict) and is called once per step, and draws the latent noise inside
  that call; the VAE's posterior sample is drawn inside `modeling.autoencoder.AutoEncoder.encode` (both with
  `torch.randn_like`, `torch.randn` or `Tensor.normal_`, as one tensor of upstream's size each); the gradient norm comes from one
  `FullyShardedDataParallel.clip_grad_norm_` call per step; `data/dataset_info.py` keeps upstream's `t2i`,
  `seedxedit_multi` and `llava_ov` entries.
- **No existing optimized versions.** Do not look at, download or use code from published optimized versions of
  this training: forks or repositories whose purpose is to speed up BAGEL's training (including any published by the
  authors of this benchmark). Upstream ByteDance-Seed/Bagel itself, PyTorch, kernel libraries and their
  documentation are fine. The agent's logs are checked after the trial; a trial that accessed such code is void.
- **Switches**: put environment overrides for your code in `/submission/switches.json` as
  `{"environment": {"NAME": "value"}}` (empty mapping if none). Variables the judge manages (`reserved_env` in
  `bench/bench.json`) and `BAGEL_BENCH_*` are refused; an invalid file is a rejection.
- **Read-only**: `/opt/researchboost` (judge, reference, startup measurements), the assets
  (`/opt/researchboost/assets`) and the training environment `/opt/bagel-env`. Code you need must be part of
  `/submission/bagel_sft` or installed in `/opt/candidate-env`; the image's CUDA toolkit is in `/usr/local/cuda`.
- Commit your changes in `/submission/bagel_sft`. Terminate your own processes before you finish.
