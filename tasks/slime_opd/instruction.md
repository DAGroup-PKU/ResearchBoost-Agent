# Task: faster on-policy distillation steps in slime

Make one step of the on-policy distillation (OPD) training in `/submission/slime_opd` faster without changing what it
computes. One step is a rollout (the student's sglang engines generate the samples, and a teacher model served by
sglang scores every sample: the log-probability of each response token under the teacher), and a training step
(Megatron: the student's log-probabilities, advantages from the reverse KL to the teacher, one optimizer update), with
the student's weights synchronized from training to its rollout engines in between. The score is the speedup of your
submitted trees over the unchanged reference on the full loop, measured by the judge described below; a faster
submission counts only if it passes every gate: the student's rollout statistically equivalent to the reference's
(token forcing through the reference engine), the teacher's log-probabilities equivalent to the reference teacher's on
the same tokens, and the training bitwise identical to the reference (tier N0) or statistically equivalent (tier N1).

## Setup

- **Hardware:** 4 Hopper GPUs (141 GB each) for the whole trial. Your work and the final evaluation use the same four
  GPUs.
- **Before you started**, the container ran the judge on the unchanged reference on these GPUs: a reference rollout
  with natural lengths (the frozen rollout fixture in `/opt/researchboost/prep/fixture`, which holds the teacher's
  answers), the reference rollout with the fixture's lengths scored through the reference engines, three replays of
  the reference teacher on the fixture, ten reference training records on the fixture, its own quick and full checks,
  the reference timings, and controls (a changed entry point, a logit bias in the engine, shifted and misaligned
  teacher log-probabilities, other engine and teacher layouts, a flipped loss sign and the reference with a 2% larger
  learning rate, all rejected; the teacher with another prefill chunking and a loss reduction summed in another
  order, accepted at N1). These files are read-only; the log is `/var/log/researchboost/prepare.log`.
- **Final evaluation:** when your time is up, every process you started is stopped and the judge's full check runs in
  this same container on `/submission/slime_opd` and `/submission/switches.json`, as the unprivileged `agent` user and
  without network. It times the full loop reference → candidate → reference and judges all three sides with the gates
  of `check` below. Its result is the score: when `status` is `PASS`, the reward is `accepted` 1 and `speedup` =
  `performance.speedup`; otherwise both are 0.
- **Network:** you have internet access. The container holds no credentials.
- **Python environments:**
  - `/opt/sglang` is the image's training environment (slime's official image for this commit: torch 2.13 with
    CUDA 13, sglang 0.5.21 with slime's patches, megatron-core, Transformer Engine 2.18, flash-linear-attention 0.5.2,
    Ray). The full list is `/opt/researchboost/installed-packages.txt`. It is read-only. The judge runs with it, and
    so does every reference run and the reference engines that score rollouts and teacher answers.
  - `/opt/candidate-env` is yours and comes first on `PATH` (`python`, `pip`). It sees every package of the training
    environment, and anything you install there takes precedence. The judge runs every process of your trees (the
    teacher server included) with `/opt/candidate-env/bin/python`. User site-packages are ignored.
  - The judge puts the three trees under test first on `PYTHONPATH` (`Megatron-LM`, `sglang/python`, `slime`), so
    the installed copies of slime, sglang and Megatron-LM in the training environment are not used for your runs.

## Working time

- Your working time is limited, and you are not told how much of it is left. When it ends, every process you started
  is stopped without warning and the final evaluation runs on what is in `/submission` at that moment: keep it in a
  state you want judged.
- The durations of the judge's runs on these GPUs are in `/var/log/researchboost/prepare.log` (the reference's own
  checks ran there). Every run starts the teacher server, Ray, the engines and Megatron from scratch; time one `time`
  run early.

## The workload

- Code: `/submission/slime_opd` holds three git repositories on a branch `opt`; make your changes there:
  - `slime/`: upstream [THUDM/slime](https://github.com/THUDM/slime) at commit `540ea919`;
  - `sglang/`: the sglang 0.5.21 source of the image, with slime's patches for it applied (it runs both the
    student's rollout engines and the teacher's server);
  - `Megatron-LM/`: the Megatron-LM source of the image, with slime's patches applied.
  `/opt/researchboost/ws/targets/reference`, `reference-sglang` and `reference-Megatron-LM` are the same trees,
  unchanged.
- The judge runs every tree from a fresh copy through `bench/launch.sh`: the teacher's sglang server, then one
  `slime/train.py` process with the frozen recipe below. Read `bench/launch.sh` for the exact command lines.
- The recipe is upstream's `examples/on_policy_distillation/run-qwen3-4B-opd.sh` (sglang teacher mode) with the
  models replaced: the student is Qwen3.5-9B (trained), the teacher is Qwen3.5-27B (scores only; the two share their
  tokenizer). Colocated on the 4 GPUs: the student's actor is Megatron with tensor parallel 2 × context parallel 2
  (data parallel 1, sequence parallel), its rollout is 2 sglang engines of tensor parallel 2 on the same GPUs, and the
  teacher is an sglang server of tensor parallel 4, resident on the same GPUs, started with upstream's arguments
  (chunked prefill 4096). dapo-math-17k prompts; 16 prompts × 4 samples per step (global batch 64, one optimizer step
  per rollout); responses up to 8192 tokens (upstream 131072), temperature 1. One deviation from upstream: the
  student's router policy is fixed to `round_robin` (slime's default is `cache_aware`, whose split of a rollout's 64
  requests over the two engines depends on arrival timing; an uneven split made a rollout ~7% slower at random), so
  that the reference timing is reproducible. Every finished sample is sent to the
  teacher (slime's custom reward function `slime.rollout.on_policy_distillation.reward_func`: the teacher's
  log-probability of every token); slime's reward post-processing (`post_process_rewards`) turns the teacher's answers
  into per-token teacher log-probabilities; the advantages are GRPO's (task rewards are 0) minus 1.0 × the per-token
  reverse KL between the student and the teacher; KL loss coefficient 0, entropy 0; Adam, learning rate 1e-6,
  constant, weight decay 0.1, with fp32 master weights and moments offloaded to the CPU; bf16 compute; full
  activation recompute; no checkpoint saving.
- Models and data: the public checkpoints `Qwen/Qwen3.5-9B` (revision `c2022362`) and `Qwen/Qwen3.5-27B` (revision
  `fc05daec`), the student's Megatron `torch_dist` conversion made with the reference slime, and
  `zhuzilin/dapo-math-17k`, in `/opt/researchboost/assets`, read-only.
- Rollout lengths: in every rollout and full-loop run of the judge, every sample decodes exactly the number of
  tokens the reference fixture recorded for it (`bench/force_len.py`: `max_new_tokens` = that length, `ignore_eos`),
  so that every tree does the same decode and scoring work.

## The judge

Run it from `/opt/researchboost/ws` with the training environment's Python:

```
cd /opt/researchboost/ws
/opt/sglang/bin/python3 bench/bench.py check --tree /submission/slime_opd --config /submission/switches.json --out /tmp/check-1
```

It writes `<output directory>/check.json`. It runs the correctness gates the final evaluation runs, with the same
rules; gate 1 differs: your `check` times each judged side once against the frozen reference timing made at startup
and projects the step speedup from them, the final evaluation times the full loop reference → candidate → reference.
A result counts when the final evaluation's `status` is `PASS`; the score is its `performance.speedup`.

**Sides.** The judge reads your diff against the reference (`sides` and `sides_exclusive` in `bench/bench.json`):
- **rollout**: files under `sglang/`, `slime/slime/rollout/`, `slime/slime/ray/rollout.py`, `slime/slime/ray/serving.py`
  and `slime/slime/router/`;
- **teacher**: files under `sglang/` (it runs the teacher too), and `slime/slime/rollout/on_policy_distillation.py`
  and `slime/slime/rollout/rm_hub/` (these two are the teacher side only); a `--config` that sets
  `OPD_TEACHER_SERVER_ARGS`;
- **train**: files under `Megatron-LM/`, `slime/slime/backends/megatron_utils/` and `slime/slime_plugins/models/`.
A check judges the sides your diff touches; a changed file outside all three (or no change at all) judges all three.
`--side rollout|teacher|train` adds a side.

Other commands (see `/opt/sglang/bin/python3 bench/bench.py --help`); none of them is a score:

- `check --quick` (or `--skip-timing`): the correctness gates only, no timing.
- `check --n1`: no timing and no bitwise attempt: the statistical gates only, for example after a change that only
  affects numerics. A training record that happens to be bitwise identical is still judged statistically (tier N1).
- The final evaluation's own mode (`check --final`) is not available to you.
- `time --tree <dir> --config <file> --out <dir> [--side rollout|teacher|train]`: one timing run per side (default:
  the sides your diff touches): seconds per window, the ratio to the frozen reference timing of this machine, and the
  projected step speedup.
- `profile --tree <dir> --config <file> --out <dir> --side rollout|teacher|train`: rollout, the torch profiler on
  every student sglang worker for 200 decode steps of rollout 1; teacher, the teacher server's profiler over 64
  forward steps while it scores replayed rollout 1; train, `torch.profiler` over one train step on every rank (fixture dump 2, run once before, as the timed steps). It
  writes the traces under `<out>/trace` and `<out>/kernel_report.md`: kernel classes, SM coverage, launch-bound
  classes and repeated kernel chains of the largest trace (`bench/kernel_util.py` makes the same report from any
  trace). Nsight Systems is not installed.

`bench/bench.py` takes the GPU lock `/tmp/slime_opd_gpu.lock` itself; do not wrap it in `flock`. Wrap any other
command that uses the GPUs as `flock /tmp/slime_opd_gpu.lock <command>`. Every judge run needs 60 GiB free under
`/tmp`; put output directories under `/tmp` and delete them when you no longer need them. The startup measurements are
in `task/` (the frozen references, the reference timing of this machine) and `runs/bootstrap/` (logs of the reference
runs). Host memory is shared by everything in the container: the CPU-offloaded optimizer needs a large part of it, so
do not run a second training process next to a judge run.

**Clocks.** The judge measures time itself (`bench/rollout_hook.py`, `bench/train_hook.py`,
`bench/teacher_replay.py`):
- a rollout lasts from the end of the previous rollout (after the judge's own dump of it) to the end of this one, as
  seen by slime's rollout manager when the samples are complete, teacher answers included; 4 rollouts, windows =
  rollouts 1–3;
- a teacher replay round lasts from the first request to the end of the post-processing, for the samples of one
  fixture rollout, all sent at once through your tree's reward path against your tree's teacher server alone;
  rollouts 0–3 of the fixture, windows = rounds 1–3;
- a train step lasts from the start of one train step to the start of the next (the slowest rank); 8 train steps on
  the frozen rollout dumps of the fixture in the order 4, 1, 2, 3, 1, 2, 3, 4, windows = steps 4–6 (dumps 1–3; the
  warm-up already ran dumps 1–4 once, dump 4 because step 7's forward passes on it start inside the last window);
- a full-loop step lasts from the end of one rollout to the end of the next; 5 steps, windows = steps 2–4; before it,
  an untimed train run of the same tree and switches on the fixture's dumps 0–4 fills the compile caches, and the
  timed run starts from a copy of them.
- So first uses that depend on the data's shapes (Triton autotuning and compilation; `TRITON_CACHE_AUTOTUNING=1` keeps
  autotune results in the cache) happen before the clock. Train and full-loop timings report `first_use`: the seconds
  of Triton autotuning and compilation inside each timed window (slowest process), normally about zero.
- Per window, ratio = reference / candidate; a side's speedup = the mean of its windows' ratios.

`check` applies the gates in order and stops at the first that fails:

0. **Structure**: the files the judge needs exist (`required_files` in `bench/bench.json`) and `slime/train.py` is the
   reference's (the same AST). Otherwise ERROR.
1. **Benefit**: fresh runs with cold compile caches on this machine.
   - Your `check`: one run of each judged side; per window, ratio = the frozen reference timing (`task/timing_ref.json`:
     per window the faster of 2 reference runs made at startup) / your run; side speedup = the mean of the ratios; the
     projected step speedup = 1 / (1 − Σ f × (1 − 1 / side speedup)) over the judged sides, with the frozen step
     fractions f (`fractions` in `task/timing_ref.json`). The teacher's fraction is its share of a step on the critical
     path: the end of a rollout minus the last generation end of its samples. When the rollout side is timed too, its
     windows already contain the teacher's share, and the teacher's term is left out. `performance.speedup` is the
     projected step speedup.
   - The final evaluation: the full loop, reference A → candidate → reference B; per window, ratio = min(A, B) /
     candidate; `performance.speedup` = the mean of the window ratios. The two reference runs differing by more than 5% in a
     window → INVALID (measured again later).
   - A speedup that does not exceed max(1.02, 1 + 2 × the reference spread) → UNAVAILABLE (no_benefit), rejected
     without a precision evaluation. The reference spread is the largest relative gap between the two reference runs
     over the windows (of the judged sides, frozen, in your `check`; of the full loop in the final evaluation).
2. **Layout**: every run reports the parallel layout it actually ran: Megatron's parallel state on every rank (tensor
   2, context 2, pipeline 1, data 1, expert 1, 4 ranks), every student sglang engine's server arguments (2 engines),
   and the teacher server's (`tp_size` 4); for the engines and the teacher, `tp_size`, `dp_size`, `pp_size`, `ep_size`,
   `nnodes` and `enable_dp_attention` as the reference's report them. Any difference → INCORRECT.
3. **Rollout side** (one rollout run of your trees with the fixture's lengths, your teacher online; the judge saves
   every rollout's samples itself, after the rollout's clock has stopped):
   - Exact items, otherwise INCORRECT: the same samples, prompts, prompt tokens and group indices as the reference
     rollout; every response exactly its forced length; finite log-probabilities of every response token; the
     teacher's answer kept in `sample.reward` (sglang's `/generate` response with `meta_info.input_token_logprobs`)
     with a finite log-probability for every response token.
   - Student token forcing: the samples are teacher-forced through the reference student engine (the unchanged
     sglang tree, 2 engines of tensor parallel 2); per response token d = the log-probability your rollout reported −
     the reference engine's. INCORRECT if the mean |d| exceeds 2× the reference rollout's own; if tokens with |d| > τ
     (2× the reference's 99.99th percentile of |d|) are more frequent than 2× the reference's rate with p < 0.001
     (binomial test); if the reference engine's log-probability of your sampled tokens shifts (paired t over the
     samples beyond 4: a sampler change).
   - The rollout's teacher log-probabilities: the reference teacher (the unchanged trees) replays your rollout's
     tokens; per response token d = the teacher log-probability your rollout carries − the reference teacher's. The
     rate, event and bias rules of the teacher side below apply, relative to the reference rollout's own online
     answers against the same replay (`teacher` in `task/rollout_ref.json`).
   - Train–inference mismatch: slime's `train_rollout_logprob_abs_diff` of the reference Megatron on your rollout
     (learning rate 0, one step) must not exceed 2× the reference's value on the reference rollout.
4. **Teacher side** (one teacher replay of your trees: the samples of fixture rollouts 0–3, scored through your tree's
   reward path — `slime.rollout.rm_hub.async_rm` with the custom reward function, then the custom post-processing —
   against your tree's teacher server):
   - Exact items, otherwise INCORRECT: the same samples, and one finite teacher log-probability per response token.
   - Per response token d = your teacher log-probability − the frozen reference replay's (`task/teacher_ref.npz`,
     the same tokens). The reference's own spread is the unchanged reference teacher re-batched four ways (requests
     one at a time, chunked prefill 8192, no overlap scheduling, no radix cache), each against the all-at-once replay,
     pooled (`baseline` in `task/teacher_ref.json`; `teacher_rules` in `bench/bench.json`). Why: Qwen3.5 is a hybrid
     model whose linear-attention state crosses the prefill chunks of a request, so any other batching or chunking
     moves the log-probabilities after a request's chunk boundaries (about 5% of tokens by more than 0.05), while
     repeats of one batching are bitwise identical; the tolerance is the reference's own re-batching noise. INCORRECT if the mean |d| exceeds 2× the reference's spread; if tokens with |d| > τ are more
     frequent than 2× the reference's rate with p < 0.001 (binomial test); if the per-sample mean of d is shifted
     (t over the samples beyond 4 and a mean beyond the floor).
5. **Training side** (a deterministic record run: 3 train steps on the fixture's rollout dumps 0–2, NCCL ring
   reductions and a fixed cuBLAS workspace; the record of the state before steps 0, 1 and 2 on every rank; the dumps
   carry the teacher's answers, which your tree's post-processing turns into teacher log-probabilities):
   - Exact items, otherwise INCORRECT: the steps and ranks recorded, the learning rate of every step, and the names,
     shapes and dtypes of every bf16 parameter, fp32 master weight and Adam moment.
   - Then the SHA-256 of every bf16 parameter, fp32 master shard and Adam moment after every update. All identical to
     the frozen reference record → **PASS (N0)**. The reference itself is not always bitwise repeatable here
     (`repeatable` in `task/train_ref.json`); otherwise the statistical gate decides.
   - Statistical gate (N1): the endpoints after two updates — the fp32 update norm, the norm of Adam's second moment,
     the mean loss and the first step's gradient norm (before any update) — must each differ from the reference's by at most 5× the reference's
     own run-to-run spread on the same data (5 reference repeats) and at most 3× the standard deviation over 6
     fixture data sets; and the cosine between your update and the reference's (a CountSketch of the fp32 update) must
     be at least the frozen minimum (`direction.cos_min` in `task/train_ref.json`). Otherwise INCORRECT.
   - Train–inference mismatch: your Megatron's `train_rollout_logprob_abs_diff` on the fixture rollout (first step)
     must not exceed 2× the reference's.
6. **Final evaluation only**: the full loop's train–inference mismatch, your run against both reference runs, must not
   exceed 2×.
- **Tier**: N0 = the training record bitwise identical and no rollout or teacher side judged; N1 = everything else
  that passes (the rollout and teacher sides are always judged statistically). The final evaluation judges all three
  sides.

## Rules

- **Training semantics must not change**: the data and its order, the prompts and samples of every rollout, the batch
  (16 prompts × 4 samples, one optimizer step per rollout), sampling (temperature 1, no truncation of the
  distribution), the teacher model and the log-probabilities it gives the student's tokens, the advantages (GRPO with
  zero task rewards, minus 1.0 × the student–teacher reverse KL), the loss, the optimizer and its hyper-parameters, the
  learning rate, and the synchronous order of the loop (each rollout uses the student weights of all updates before
  it; every sample of a rollout is scored by the teacher before the training step that uses it).
- **Parallel layout is fixed**: tensor, context, pipeline, data and expert parallel sizes of the actor, the number of
  student engines and the GPUs per engine, the teacher's tensor parallel size, and colocation. Everything else may
  change, on the student and on the teacher: kernels, fusions, communication, scheduling, batching and prefill
  chunking, memory and offload strategy, CUDA graphs, sglang, Megatron and slime internals; the teacher server's other arguments through `OPD_TEACHER_SERVER_ARGS`.
- **No speculative decoding**: no drafting of any kind on the student engines and the teacher: not the checkpoint's MTP / NEXTN head, not
  EAGLE or another draft model, not n-gram or lookahead drafts, in sglang or in your own code. Every sampled token
  comes from a decode step of the target model itself, never from a draft that the target then verifies. The judge
  reads every engine's server arguments and the teacher's: a speculative algorithm or a
  draft model there is INCORRECT (layout gate); drafting hidden elsewhere breaks this rule as well.
- **No work skipped by data value**: do not skip, shorten or reorder computation depending on the values of the
  rewards, the advantages, the losses or the samples of a step (for example, leaving out the backward of micro-batches
  whose advantages are all zero, or of samples whose reward equals their group's). This benchmark caps the response
  length, so far more samples of a group get the same reward, and a zero advantage, than in a real run; a shortcut that
  feeds on that does not carry over and counts as a change of the training semantics. Work that the recipe's
  configuration makes dead regardless of the data (a term whose coefficient is 0) may still be removed.
- **Precision**: bf16 compute with fp32 master weights and fp32 optimizer state, as the reference, for the student;
  the teacher in bf16, as the reference. No FP8, INT8 or FP16 and no quantized teacher, and no lower precision of the
  master weights or the optimizer state. Changes of the arithmetic itself (fusion, reduction order, kernel choice,
  communication layout) are allowed and are judged by the gates.
- **The judge's instruments are not part of the recipe**: its rollout dumps, the loading of the fixture's dumps in
  training runs, the teacher replay, its clocks and records. Do not change the code paths that only they use
  (`--debug-rollout-only`, `--debug-train-only`, `--load-debug-rollout-data`, the custom-function hooks) to gain time.
- **Hook points the judge needs**: `slime/train.py` unchanged and run as the entry; slime's custom rollout log
  function (`--custom-rollout-log-function-path`) called once per rollout with its complete samples; slime's custom
  Megatron before-train-step hook (`--custom-megatron-before-train-step-hook-path`) called before every train step
  with the model and the optimizer (Megatron's optimizer with the CPU-offloaded `HybridDeviceOptimizer`); the
  custom generate function (`--custom-generate-function-path`) used for every sample; `--load-debug-rollout-data` as
  the training input of train runs; the custom reward function and post-processing paths of the recipe, called as
  slime calls them (`slime.rollout.rm_hub.async_rm(args, sample)` per sample, then the post-processing function on the
  rollout's samples, with an `args` carrying `rm_url`, `custom_rm_path`, `custom_reward_post_process_path`,
  `rollout_temperature`, `reward_key` and `rm_type`); the teacher's answer kept in `sample.reward` as sglang's
  `/generate` response with `meta_info.input_token_logprobs`; the teacher log-probabilities in
  `sample.teacher_log_probs` after the post-processing; the samples' `rollout_log_probs`; the teacher server answering
  `/health_generate`, `/generate` and `/get_server_info`; slime's `step N: {...}` training metrics in the log.
- **No detection**: the code may not detect whether it is being timed or judged, read `SLIME_OPD_BENCH_*`
  variables, inspect the judge's files, or behave differently during evaluation (a teacher replay is scored like a
  rollout's samples). In the judge's deterministic record runs (`NCCL_ALGO=Ring`, `CUBLAS_WORKSPACE_CONFIG` set) your
  code must run the same computation as otherwise.
- **No existing optimized versions.** Do not look at, download or use code from published optimized versions of this
  training: forks or repositories whose purpose is to speed up slime, its OPD or GRPO recipes or this workload
  (including any published by the authors of this benchmark). Upstream slime, sglang, Megatron-LM, PyTorch, kernel
  libraries and their documentation are fine. The agent's logs are checked after the trial; a trial that accessed such
  code is void.
- **Switches**: put environment overrides for your code in `/submission/switches.json` as
  `{"environment": {"NAME": "value"}}` (empty mapping if none). Variables the judge manages (`reserved_env` in
  `bench/bench.json`) and `SLIME_OPD_BENCH_*` are refused; an invalid file is a rejection.
- **Read-only**: `/opt/researchboost` (judge, reference trees, startup measurements, the fixture), the assets
  (`/opt/researchboost/assets`) and the training environment. Code you need must be part of the three repositories in
  `/submission/slime_opd` or installed in `/opt/candidate-env`; the image's CUDA toolkit is in `/usr/local/cuda`.
- Commit your changes in the repositories of `/submission/slime_opd`. Terminate your own processes before you finish.
