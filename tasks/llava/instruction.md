# Task: faster LLaVA-1.5 training

Make the training implementation in `/submission/llava` faster per optimizer step without changing what it
computes. The score is the speedup of your submitted tree over the unchanged upstream reference, measured by the
judge described below; a faster tree counts only if it passes every gate.

## Setup

- **Hardware:** 2 × H200 for the whole trial. Your work and the final evaluation use the same two GPUs.
- **Before you started**, the container ran the judge on the unchanged reference on these GPUs: the reference's data
  probe, bitwise records, trajectories and their frozen spread, the frozen reference timing, and the judge's own
  checks. These files are read-only; the log is `/var/log/researchboost/prepare.log`.
- **Final evaluation:** when your time is up, every process you started is stopped and the judge's full check runs
  in this same container on `/submission/llava` and `/submission/switches.json`, as the unprivileged `agent` user and
  without network. It runs the gates of `check` below, with gate 1 timed as reference → candidate → reference. Its result is the score: when `status` is `PASS`, the reward is `accepted` 1 and `speedup` =
  `performance.speedup`; otherwise both are 0.
- **Network:** you have internet access. The container holds no credentials.
- **Python environments:**
  - `/opt/llava-env` is the image's training environment: torch 2.1.2+cu121 and torchvision 0.16.2 (pinned, as are
    flash-attn 2.5.8 and peft 0.10.0), then upstream's own `pip install -e ".[train]"` dependencies (transformers
    4.37.2, DeepSpeed 0.12.6, accelerate 0.21.0, ...); the full list is `/opt/researchboost/installed-packages.txt`.
    It is read-only. The judge runs with it, and so does every reference run.
  - `/opt/candidate-env` is yours and comes first on `PATH` (`python`, `pip`). It sees every package of the training
    environment, and anything you install there (`pip install ...`, torch included) takes precedence. The judge runs
    every process of your tree with `/opt/candidate-env/bin/python`. User site-packages (`pip install --user`) are
    ignored.
  - The judge records the library versions of the reference and the candidate runs (`performance.library_versions`);
    a difference is not an error. Everything else in the gates below applies unchanged.

## Working time

- Your working time is limited, and you are not told how much of it is left. When it ends, every process you started
  is stopped without warning and the final evaluation runs on what is in `/submission` at that moment: keep it in a
  state you want judged.
- Approximate durations of the judge commands for this task, measured on 2 × H200 at this configuration
  (2026-10-27), each including about a minute of startup: `time` about 2 min; `check --quick` about 5 min; one
  timing run about 2 min and one gate-3 trajectory about 6 min, so the full `check` takes about 3 min when it stops
  at gate 1 (for example no_benefit), about 7 min for a bitwise identical candidate and about 13 min for one that
  is not; `check --n1` (gates 0 and 3) about 7 min; `check --skip-timing` is the full check without its timing run.
  A check holds the GPUs for all of that time.

## The workload

- Code: `/submission/llava` is upstream [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) at commit
  `c121f04` on a branch `opt`; make your changes there. `/opt/researchboost/ws/targets/reference` is the same
  commit, unchanged.
  - The judge launches `torchrun --nnodes=1 --nproc-per-node=2 llava/train/train_mem.py <frozen arguments>` from a
    copy of the tree, with the tree root first on `PYTHONPATH`. Scripts of your own that import `llava` must put the
    tree first on `PYTHONPATH` too.
- The recipe is frozen in `train_args` of `bench/bench.json`: upstream `scripts/v1_5/finetune.sh`, changed only in
  these points:
  - the model is 7B: `vicuna-7b-v1.5` + `clip-vit-large-patch14-336` (frozen) + the pretrained projector
    `llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5` (`mlp2x_gelu`);
  - the number of GPUs: upstream runs 8 GPUs × 16 per GPU × accumulation 1 = global 128; here 2 GPUs × 16 per
    GPU × accumulation 1 = global 32 (the per-GPU batch kept, the global batch scaled by 2/8).
    - **The load of each micro-batch on each GPU is upstream's**: 16 samples, `model_max_length` 2048, bf16 +
      tf32, DeepSpeed ZeRO-3 (`scripts/zero3.json`), flash-attention 2 (`train_mem.py`), gradient checkpointing.
    - LLaVA's sampler builds megabatches of (GPUs × accumulation steps) = 2 micro-batches of 16 = 32 samples per
      optimizer step.
  - `--report_to none` (no wandb); local paths.
  - Optimizer and schedule as upstream: AdamW, learning rate 2e-5, cosine, warmup ratio 0.03, weight decay 0, no
    gradient clipping.
- Data: timing uses a fixed fixture (`$ASSETS/data/fixture/timing.json`, images in `$ASSETS/data/images`, with
  `ASSETS=/opt/researchboost/assets`). It was drawn at random with a fixed seed from `llava_v1_5_mix665k.json`:
  image-text samples uniformly from all image samples, text-only samples (ShareGPT) drawn separately. With the
  sampler's grouping by modality, the 25 steps of a timing run are mostly image-text steps. The sampling
  method and the hashes of every frozen input are in `task/formulation.json`.

## The judge

Run it from `/opt/researchboost/ws` with the training environment's Python:

```
cd /opt/researchboost/ws
/opt/llava-env/bin/python bench/bench.py check --tree /submission/llava --config /submission/switches.json --out /tmp/check-1
```

It writes `<output directory>/check.json`. It runs every gate the final evaluation runs, with the same rules; only
gate 1 differs: your `check` times your tree once against the frozen reference timing made at startup, the final
evaluation times reference → candidate → reference. A result counts when the final evaluation's `status` is `PASS`;
the score is its `performance.speedup`.

Other commands (see `/opt/llava-env/bin/python bench/bench.py --help`); none of them is a score:

- `check --quick`: gate 0 and the exact items of gate 2 (no timing, no trajectories); `check --skip-timing`:
  everything but gate 1.
- `check --n1`: gates 0 and 3 only. After a check whose gate 2 was not bitwise identical, this runs the statistical
  gate again (for example after a change that only affects numerics) without repeating the timing and the records.
- The final evaluation's own mode (`check --final`) is not available to you.
- `time --tree <dir> --config <file> --out <dir>`: one timing run (the windows of gate 1), with its speedup over
  the frozen reference timing (`speedup_vs_frozen_reference` = the frozen reference's mean time per step ÷ this
  run's; larger is faster). The frozen reference is the mean of 3 reference runs made at startup; `reference_noise`
  gives their spread (max − min) / mean, and `beyond_reference_noise` says whether the gain over
  the frozen reference is larger than twice that spread, the margin the bar of gate 1 also uses. A speedup inside that margin is not distinguishable
  from run-to-run noise. If the frozen reference timing is missing, `time` first times the reference once and reports
  the speedup over that run (kept for later `time` runs on the same GPUs); `reference_noise` then says its noise is not
  measured.
- `sweep --tree <dir> --config a.json --config b.json ... --out <dir>`: reference, one timing run per switch file,
  reference again; a screen, not a verdict.
- `profile --tree <dir> --config <file> --out <dir> --detail [--steps K]`: the PyTorch profiler over K steps after
  the warmup, one trace per rank and a summary. Nsight Systems is not installed, so use `--detail`.

`bench/bench.py` takes the GPU lock `/tmp/llava_gpu.lock` itself; do not wrap it in `flock`. Wrap any other
command that uses the GPUs as `flock /tmp/llava_gpu.lock <command>`. Every judge run needs 200 GiB free under `/tmp`;
put output directories under `/tmp` and delete them when you no longer need them. The startup measurements are in
`task/timing/` (the frozen reference timing `reference.json` and the three reference runs it averages), `bench/records/ref` and
`bench/traj/` (reference records and trajectories) and `runs/bootstrap/` (logs of the reference runs).

`check` applies the gates in order and stops at the first that fails:

0. **Structure (CPU)**
   - `train()` and the classes `ModelArguments`, `DataArguments` and `TrainingArguments` in
     `llava/train/train.py` must have the same AST as the reference. Different → ERROR. Reason: the judge's
     adapter and data probe re-create this setup the reference's way. To change the setup, put the change into
     the modules it calls (model, trainer, data, the entry point `train_mem.py`).
   - Data probe (`bench/loader_probe.py`): for the first 16 micro-batches (the first 8 optimizer steps) that your
     data pipeline (dataset, collator, sampler, `get_train_dataloader`) produces from the timing fixture, the
     **global sample multiset of each optimizer step** must be bitwise the reference's (sorted per-row digests:
     input_ids and labels without padding, images as bf16), and so must the step's total token count, label count
     and the multiset of micro-batch sizes. Different → INCORRECT. Not compared: which samples share a
     micro-batch, which GPU and micro-step they go to, and the row order within a micro-batch.
1. **Benefit**: independent `torchrun` runs of the real entry point with cold compile caches. Each run has 25
   steps: 5 warmup steps, then 2 timed windows of 10 steps.
   - Your `check`: one run of your tree. `performance.speedup` = the frozen reference timing (the mean of the 3
     reference runs made at startup, `task/timing/reference.json`) ÷ your run's time per step.
   - The final evaluation: reference → candidate → reference; `performance.speedup` = the mean time per step of
     the two reference runs ÷ the candidate's.
   - Every rank of every judge run (timing, records, trajectories) is pinned to its share of the container's CPUs:
     the physical cores on its GPU's NUMA node, split between the ranks there, with PyTorch's intra-op threads set
     to that share's physical cores (`bench/cpu_affinity.py`). Reference and candidate are pinned the same way. Each
     run reports how many CPUs the container kept busy (`cpu.busy_cpus`, and `busy_cpus_in_windows` for timing).
   - In the final evaluation, a fingerprint (math mode, GPU UUIDs, compile cache, instrumentation, timing plan) that
     differs from the first reference run → ERROR. Library versions are recorded for both and may differ.
   - In the final evaluation interference makes the result INVALID (measured again later): a run's two windows
     differ by more than 6% relative to the frozen reference's windows; a reference run more than 15% off the frozen
     reference timing. Your own `check`, `time` and `sweep` report these (`notes`) but do not reject the run.
   - Before taking the GPU, `time`, `sweep` and `check` run the data probe and compare each optimizer step's
     samples with the frozen reference probe (`task/loader_reference.json`): a mismatch is ERROR without timing.
   - Logged-loss sanity check (precision is judged by gates 2 and 3): the first logged loss comes before any
     optimizer step and must be within 0.5% of the frozen reference's; later steps must not be 0, NaN or inf where
     the reference is not, and must stay within 25% of the reference. When the data probe shows that the candidate
     reassigned the samples within a step, both bands are 48.4%: the logged loss is a mean over micro-batches of
     each one's token mean, so it moves with the assignment even for the same samples
     (`task/partition_sensitivity.json`). Out of band → ERROR.
   - A speedup that does not exceed max(1.02, 1 + 2 × the reference spread) → UNAVAILABLE (no_benefit), rejected
     without a precision evaluation. The reference spread is the relative gap between the two reference runs in the
     final evaluation, and the frozen reference runs' (max − min) / mean in your `check`.
   - A run that still compiles inside a timing window, or whose log matches an `invalid_run_signatures` entry of
     `bench/bench.json`, is not a valid timing run.
2. **Bitwise**: in deterministic mode (see "Same code path everywhere" below), two short cases of one optimizer
   step each, one on an image-text batch and one on a text-only batch, with the run's 2 GPUs, accumulation steps
   and 16 samples per micro-batch.
   - The judge's adapter (`bench/llava_adapter.py`) runs your entry point and your trainer: it catches the trainer
     when `train()` calls `trainer.train()`, then lets the trainer's own training loop advance one optimizer step
     at a time.
   - Exact items, otherwise INCORRECT: the RNG fingerprint of each step (torch CPU and CUDA, numpy, Python `random`;
     the judge keeps Triton's own cache bookkeeping out of it); the number of optimizer steps, global_step
     and micro-steps; the global sample multiset each optimizer step hands to `training_step` (gathered across
     GPUs, as in gate 0) with the step's token count, label count and micro-batch sizes; learning rate, Adam
     hyperparameters, DeepSpeed's batch, accumulation, clipping and precision settings, and the set of trainable
     parameters; the dtypes of the fp32 master weights and Adam state; the cross-rank pattern; and every trainable
     parameter whose reference gradient is nonzero must have a finite, nonzero gradient (a module that silently
     stops training changes a short trajectory too little for gate 3 to see).
   - All recorded tensors bitwise identical → **PASS (N0)**. The tensors are each GPU's loss, the full fp32
     gradients of all trainable parameters, the fp32 master-weight updates and both Adam states, in a canonical
     view that does not depend on the ZeRO stage.
   - Per-tensor tolerances (`bench/thresholds.json`) are diagnostic output only.
     Beyond them, the comparison's status (`correctness.numerics.status`) reads `NOT_BITWISE` and gate 3
     decides; `INCORRECT` there means an exact item failed.
3. **Statistical**: for every candidate, also when gate 2 is bitwise identical (a bitwise candidate keeps N0 when
   gate 3 passes).
   - With the production settings, as timed (not deterministic mode), a paired short training trajectory on the first
     frozen seed: 60 optimizer steps of global
     batch 32 (per-device 16 × accumulation 1 × 2 GPUs, the recipe's own), cosine with warmup, the seed deciding the
     data order. Only losses are recorded.
   - Two endpoints, both computed by the judge itself through the model's forward (the mean cross entropy over the
     batch's supervised tokens), not taken from what your `training_step` or `compute_loss` returns:
     `train_loss_last20_canonical`, the mean over steps 40–60 of the model-level loss of the step's micro-batches,
     measured with the weights the step trains from; and `probe_loss_last20`, the mean over steps 40–60 of the loss
     on 32 held-out fixture samples. Both are averaged across GPUs. An endpoint is one such number per run: the
     judge compares these two numbers, not the whole loss curve.
   - σ = the reference's standard deviation across its own 6 seeds (frozen in `bench/traj/frozen.json`).
   - The bias d = candidate − reference of each endpoint on that seed: every |d| ≤ 3σ → PASS (N1); any |d| > 3σ →
     INCORRECT. One seed, no further seeds; the same rule in your `check`, `check --n1` and the final evaluation.
- **Tier**: N0 = everything bitwise identical with the same dtypes and math modes; N1 = every other candidate that
  passes gates 2 and 3. A changed dtype or math mode of any tensor (TF32, low-precision reductions, SDPA backend) is
  never N0; the check lists it (`correctness.numerics.precision_changed`) and gate 3 decides as for any N1 candidate.

## Rules

- **Training semantics must not change** (gate 0 compares setup and data, gates 2 and 3 numerics):
  - Data and order: the dataset, preprocessing (conversation template, tokenization, images padded to square and
    then through the CLIP processor), the sampler's grouping by modality, which 32 samples form each optimizer
    step, 16 samples per micro-batch, the recipe's accumulation steps and GPU count. Samples may not move to
    another optimizer step, be dropped or be duplicated.
  - Within an optimizer step the assignment is your choice: how the step's samples are spread over its
    micro-batches (which samples share a micro-batch, which GPU and micro-step, the row order within a
    micro-batch); the length key used for grouping (upstream's `modality_lengths` counts words split on spaces; it
    may be replaced by the token length after tokenization, keeping the sign convention, positive for image-text
    and negative for text-only, and the grouping by modality); and the balancing target of
    `split_to_even_chunks`. Such changes cannot be bitwise identical; they are judged N1 and decided by gate 3.
  - Loss normalization: either (1) upstream's, where each micro-batch averages over its own valid label tokens and
    the step's micro-batches are averaged with equal weight, or (2) the global token mean over the whole optimizer
    step: Σ(cross entropy of valid label tokens) ÷ Σ(number of valid label tokens). Both count only valid label
    tokens (padding labels are `IGNORE_INDEX` = -100). Option 2 needs the step's total number of valid tokens
    before the first backward (an all-reduce across GPUs, or the data pipeline providing it). Not allowed:
    weighting by sample count instead of tokens, skipping or repeating micro-batches, counting a valid token twice
    or missing one, soft labels, changing the meaning of `IGNORE_INDEX`. Normalize only at the Trainer level
    (`compute_loss` / `training_step`): `model.forward(labels=...)` must still return the mean cross entropy of that
    one batch over its own valid label tokens, because gate 3 measures through it.
  - Learning rate 2e-5, cosine, warmup 0.03, AdamW (betas, epsilon, weight decay 0), no gradient clipping, one
    optimizer step per step.
  - Model: CLIP frozen; the LLM and the projector train with all parameters; trainable parameter names and shapes
    must match the reference (the judge looks parameters up by name, so do not merge q/k/v or similar into new
    parameters).
- **Precision**: the reference is upstream's recipe: bf16 compute with tf32, fp32 master weights and fp32 Adam
  state. FP8, INT8 and FP16 training are not allowed, nor is removing the fp32 master weights or optimizer state
  (gate 2 compares their dtypes). Local precision changes (TF32, low-precision reductions, another SDPA backend) are
  allowed; such a candidate is N1 and must pass gate 3.
- **Same code path everywhere**: the code may not read the judge's internal state (`LLAVA_BENCH_*` variables),
  detect whether it is being timed or judged, or behave differently during evaluation; timing, bitwise and
  statistical runs must exercise the same code path. Gate 2 runs it with `torch.use_deterministic_algorithms(True, warn_only=True)`,
  cuDNN deterministic and flash-attn's deterministic backward (set by the judge), so that the same code reproduces the
  reference's records exactly; gate 3 and the timing run the production settings, so gate 3 judges what is timed.
- **Hook points the judge needs**:
  - the entry point `llava/train/train_mem.py` calls `train()`, which calls `trainer.train()`;
  - training goes through the HF Trainer loop: one `trainer.training_step(model, inputs)` call per micro-batch,
    with inputs in the reference's format (`input_ids`, `labels`, `attention_mask`, `images`), and the Trainer's
    `on_step_end` callback fires at the end of every optimizer step;
  - the optimizer step goes through the DeepSpeed engine: `engine.optimizer.step()` once per step, and
    `deepspeed.utils.safe_get_full_grad` / `safe_get_full_fp32_param` / `safe_get_full_optimizer_state` must be
    able to fetch the full gradients, master weights and Adam state before and after it. The ZeRO stage and other
    DeepSpeed settings (`scripts/zero3.json` in your tree, which the judge uses) may change; batch, accumulation,
    clipping and precision are exact items.
- **Switches**: put environment overrides for your code in `/submission/switches.json` as
  `{"environment": {"NAME": "value"}}` (empty mapping if none); the judge sets them for your code's processes only,
  never for the reference. Variables the judge manages (`TMPDIR`, `TMP`, `TEMP`, `TORCHINDUCTOR_CACHE_DIR`,
  `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`, `CUDA_VISIBLE_DEVICES`, `PYTHONPATH`, `NVIDIA_TF32_OVERRIDE`,
  `CUBLAS_WORKSPACE_CONFIG`, `HF_HOME`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `MASTER_PORT`, `MASTER_ADDR`) and
  `LLAVA_BENCH_*` are refused; an invalid file is a rejection.
- **Read-only**: `/opt/researchboost` (judge, reference, startup measurements), the assets
  (`/opt/researchboost/assets`) and the training environment `/opt/llava-env`. Code you need must be part of
  `/submission/llava` or installed in `/opt/candidate-env`; the image's CUDA toolkit is in `/usr/local/cuda`.
- Commit your changes in `/submission/llava`. Terminate your own processes before you finish.
