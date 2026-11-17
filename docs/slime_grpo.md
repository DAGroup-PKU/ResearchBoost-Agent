# slime_grpo: Qwen3.5-9B GRPO in slime (4 GPUs)

Reference: [THUDM/slime](https://github.com/THUDM/slime) @ `540ea919`, in slime's official image of the same day
(`slimerl/slime@sha256:227723fe…`: sglang v0.5.21 and Megatron-LM `4b4acac9`, both with slime's patches; torch 2.13,
CUDA 13 with the image's forward-compatibility libraries on older drivers). The rollout stack (sglang) and the
training stack (Megatron-LM) are part of the candidate: `/submission/slime_grpo/{slime,sglang,Megatron-LM}`.

Port of the judge of the Faster-Slime-GRPO campaign (4 × H100 80 GB) to the bench's conventions and to 4 Hopper GPUs
of 143 GB. The workload is that campaign's recipe unchanged.

## Workload

Upstream `examples/fully_async/run-qwen3.5-9B-fully_async.sh` (the only Qwen3.5-9B GRPO recipe at this commit), made
synchronous and colocated on 4 GPUs:

| | upstream | here |
|---|---|---|
| GPUs | 8: actor 4 + rollout 4, fully asynchronous | 4, colocated, synchronous |
| actor | TP2 × CP2 (DP1) | the same |
| rollout | 2 sglang engines × TP2 | the same (on the actor's GPUs) |
| batch | 8 prompts × 4 samples, global batch 32 | 16 prompts × 4 samples, global batch 64, one optimizer step per rollout |
| response limit | 131072 | 8192 |
| checkpoint | saved every 9999 rollouts | not saved |

Everything else is upstream's: score centering (top-128 log-probs), REINFORCE with TIS, KL coefficient 0, Adam lr
1e-6 with CPU-offloaded fp32 master weights and moments, full recompute, dynamic batches of up to 4096 tokens per
GPU. The judge's command line is `bench/launch.sh`.

## Assets

`RESEARCHBOOST_ASSETS_ROOT/slime_grpo`, provisioned by `environment/asset-prep/prepare_slime_grpo_assets.sh <task
image> <assets root> [gpu]` (pinned revisions, `asset-prep/SHA256SUMS`):

- `Qwen3.5-9B/`: Hugging Face `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (19 GB);
- `data/dapo-math-17k.jsonl`: `zhuzilin/dapo-math-17k` @ `2e65612930298bde4c5d58fd97b3f23a483aaff9`;
- `Qwen3.5-9B_torch_dist/`: the reference slime's `tools/convert_hf_to_torch_dist.py` of the checkpoint with the
  reference Megatron-LM, on one GPU of the host, in the task image. It is an asset (made once per asset root, shared
  by every trial) rather than a bootstrap step, because it depends only on the checkpoint and the reference trees.

Every file is a frozen input of the judge (`task/formulation.json`): a change after preparation is an ERROR.

## Judge

`environment/workspace/bench/bench.py` (see its docstring). Changes against the campaign's judge:

- **Commands and conventions of the bench.** `check [--quick | --skip-timing] [--side S]`, `time [--side S]`, `profile
  --side S`, `check --final` behind the root marker; every run in a fresh copy of the three trees, in its own process
  group, with fresh TMPDIR and cold compile caches, under the GPU lock, after killing leftovers and waiting for idle
  GPUs; frozen-input fingerprints before and after; the source digest before and after; reserved variables refused
  as switches.
- **The judge's own clocks** (the campaign parsed slime's own `perf/*_time` log lines, which the candidate's code
  writes). `bench/rollout_hook.py` (slime's custom rollout log function) stamps the end of every rollout;
  `bench/train_hook.py` (slime's custom before-train-step hook) stamps the start of every train step on every rank.
- **The rollout dump outside the timed window** (known gap 2 of the campaign: the dump, about 11 s per rollout, was
  inside `perf/rollout_time`). The judge dumps the samples itself in the rollout log function, after the clock of
  that rollout stopped, and the next rollout's clock starts when the dump ends. Full-loop runs do not dump.
- **Top-k log-probabilities checked** (known gap 1: score centering consumes the top-128 ids and log-probs of every
  response token, which the campaign's token forcing did not compare). The scorer requests the reference engine's
  top-128 for every response token; the overlap deficit and the mean |difference| over the shared ids must stay
  within 2× the reference rollout's own.
- **Parallel layout checked from what ran** (known gap 3: an environment override of the engine's server arguments
  could change it unseen). The train hook records Megatron's parallel state on every rank; the rollout hook asks every
  engine behind the router for its server arguments. Any difference from the recipe (Megatron) or from the reference's
  engines (`task/layout_ref.json`) is INCORRECT, in every run of the candidate.
- **`profile --side train` fixed** (known gap 4: two steps against a profiler window of steps 1–2 captured nothing,
  and the profiler's temporary files filled the root disk). Four steps, step 2 traced; TMPDIR is the run's scratch.
- **`check --n1`**: no timing and no bitwise attempt; the statistical gates of both judged sides (layout, token
  forcing, top-k, mismatch; the train statistical gate, direction and mismatch), the harness's "repeat only the
  statistical gate".
- **Invalid-run signatures** (`bench.json invalid_run_signatures`), each a message string of this image's stack:
  `torch._dynamo hit config.recompile_limit` (torch/_dynamo/convert_frame.py), `skipping cudagraphs due to`
  (torch/_inductor/output_code.py), `Capture cuda graph failed` (sglang decode_cuda_graph_runner.py; it raises, so
  the run also fails) and `Watchdog caught collective operation timeout` (libtorch_cuda). None of them could be
  triggered on the reference, and none appears in any preparation log (no false positive). A Ray worker that dies is
  restarted by nothing in this recipe (the run fails, ERROR); no Ray message string is matched.
- **No frozen warm JIT cache.** Every run starts with empty compile caches (the bench's rule); the flashinfer JIT
  cache shipped in the image covers sglang's kernels.
- **Thresholds measured on the trial machine** during preparation (token forcing, top-k, mismatch, train noise and
  data-set sigma, timings), as the campaign froze them on its machine.

### Gates

See the task statement (`instruction.md`) and `bench.py`. In short: structure (ERROR); benefit (UNAVAILABLE); layout;
rollout side: exact items, token forcing (rate, events, sampler), top-k, mismatch; training side: exact items, bitwise
(N0) or the statistical gate (N1), mismatch; the final evaluation also checks the full loop's mismatch.

### Lite check

The task follows check-profile-v2: the agent's `check` times one candidate run per judged side against the frozen
reference timing (no reference run in the agent's check, no A–B–A); the train statistical gate is one candidate record
run on the frozen fixture, judged against 3 σ of the 6 fixture data sets; `check --final` (harness only) times the
full loop reference A → candidate → reference B. The train gate keeps two bounds of the original judge on top of the
3 σ rule, the paired run-to-run noise (5 × the largest difference among 5 reference repeats on the same data) and the
update-direction cosine: the data-set σ of this recipe is too wide to resolve a wrong update (the lr × 1.02 control
moves the update norm by 1.9%, the data-set σ is 43% of it, set 5 has two zero-advantage rollouts), while the paired
noise is 0.54%; a candidate runs on exactly the reference's data, so the paired bound is the meaningful one.

### Thresholds (measured on 4 × 143 GB Hopper GPUs, 2026-11-08)

| quantity | reference | frozen rule | unchanged tree | wrong tree |
|---|---|---|---|---|
| token forcing mean \|d\| | 0.00769 (2.07 M tokens) | ≤ 2× | 1.007× | logit bias 15.6× |
| events \|d\| > τ (τ = 0.493) | rate 8.2e-6 | binomial p < 1e-3 and > 2× rate | p = 0.999 | rate 8.2e-2 |
| sampler paired t | — | \|t\| ≤ 4 | −0.38 | logit bias −13.0 |
| top-k overlap deficit | 1.19% | ≤ 2× | 1.006× | logit bias 14.6× |
| top-k mean \|d\| | 0.0725 | ≤ 2× | 1.002× | top-k +0.05: 1.68× (passes this rule), logit bias 7.8× |
| top-k bias paired t | mean signed −0.00014 | \|t\| ≤ 4 | −0.04, −1.02 (`--n1` run) | top-k +0.05: 305–374; logit bias 16 |
| mismatch (rollout side) | 0.00812 | ≤ 2× | 1.00× | — |
| train update norm | 0.12167 | \|d\| ≤ 0.00066 (5 × noise) and ≤ 3 σ = 0.157 | d = −3.5e-6 | lr × 1.02: d = 0.0023 |
| train direction cosine | repeats 0.991–0.994 | ≥ 0.9554 | 0.9933 | loss sign: −0.994 |
| train loss mean | −0.002367 | \|d\| ≤ 2.5e-5 and ≤ 3 σ | d = −2e-9 | loss sign: beyond |
| train mismatch | 0.00816 | ≤ 2× | 1.00× | — |
| reduction order reversed | — | — | — | PASS N1 (cos 0.9936, every endpoint within) |

- The rollout's top-k log-probs come from the sampler on bf16 logits and sit on a 0.25 grid (probe of 4 samples:
  rank 0 \|d\| ≈ 0.007, ranks 1–127 ≈ 0.065, signed ≈ 0): the reference's top-k \|d\| is about 10× its
  sampled-token \|d\|. A +0.05 shift of every top-k log-prob but the first raises it only 1.68×, so the 2× ratio
  alone missed the top-k control in the first run; the paired bias rule (per-sample signed mean against the
  reference rollout's) was added and catches it at t ≈ 300, against \|t\| ≤ 1.02 for unchanged trees.
- The reference is not bitwise repeatable (6388 tensors differ between two deterministic records: Transformer
  Engine's attention backward), so every training change is judged at N1, as in the original task.
- Grad-norm and second-moment noise among the repeats is large (tolerances 70% and 230% of the value); these two
  endpoints do not constrain much, the update norm, the loss and the direction do.

### Timing (reference, 4 × 143 GB Hopper GPUs)

| | windows (s) | spread | notes |
|---|---|---|---|
| rollout (4 rollouts, forced lengths) | 113.7, 113.2, 110.7 | 2.45% | host otherwise idle; 4.28% with a busy neighbour lane |
| train (5 steps on the fixture) | 99.2, 86.1, 98.8 | 1.85% | host otherwise idle; 11.6% with a busy neighbour lane |
| full loop (5 RL steps, forced lengths) | 236.3, 179.9, 233.9 | 2.58% (A–B of the reference's own `--final`) | step fractions rollout 0.519, train 0.437 |

**Re-measuring noisy pairs** (`bootstrap.py timing_references`, full and pack preparation): each side's reference
timing is a pair of runs; when the two differ by more than 3% in a window, the pair is measured again with fresh runs,
up to 3 pairs; the first pair within 3% is frozen. If none is, the last pair is frozen and `timing_spread_flag`
({side: spread}) is written to `task/ready.json` and `prepare.log`; `ready.json timing_spread` always records the
frozen spreads. The agent's bar (1 + 2 × spread) therefore rests on a pair within 3% unless the flag says otherwise.

The agent's bar is 1 + 2 × 2.45% = 1.049 (projected step speedup).

**Two trials per 8-GPU host** (2026-11-09): both lanes pinned to their GPUs' NUMA node (GPUs 0-3
with CPUs 0-47,96-143, GPUs 4-7 with CPUs 48-95,144-191: each lane on its own NUMA node), both timing the reference
(a fresh rollout pair and a train pair) at the same time:

| judge | repetition | lane 0 train / rollout | lane 1 train / rollout |
|---|---|---|---|
| current | 1 | 1.53% / 2.88% | 1.51% / 2.02% |
| current | 2 | 1.36% / 2.57% | 1.50% / 0.72% |
| threads capped (OMP/MKL = CPUs / 4) | 1 | 1.12% / 2.42% | 1.43% / 0.54% |
| threads capped | 2 | 3.74% / 4.74% | 2.70% / 1.65% |
| threads capped | 3 | 2.86% / 2.84% | 1.42% / 0.39% |

Inside the cpuset torch sizes its pools from the affinity (96 threads per process, 4 ranks), so capping them was the
first suspect, but the cap neither sped up the train step (98-99 s either way) nor reduced the spread; the judge is
unchanged. The noise is run-to-run (one pair in 8 beyond 3%), not interference between the lanes: the 6.8% of the
one pack-validation trial was such an outlier. Two slime_grpo trials per host are run since 2026-11-09, and the
re-measuring rule below covers the outliers. The rollout spread stays above 2% even on an idle
host. Two trials on one 8-GPU host (two 4-GPU lanes) disturb each other's timing: the CPU-offloaded Adam of one
lane competes for the host's CPUs and memory bandwidth (train windows 11.6% apart). Run one slime_grpo trial per
host, or accept wider reference spreads (INVALID in the final evaluation beyond 5%).

The unchanged tree's own agent `check` measured 0.984 (rollout) and 0.961 (train, window 3) against the frozen
timing: projected 0.975, UNAVAILABLE as required; its `--final` measured 0.941 (window 2 of the candidate run
191 s against 180/182 s), also UNAVAILABLE.


## Scoring (since 2026-11-16)

- Per timed window, ratio = reference / candidate, and **the speedup is the mean of the window ratios** (it used to be
  the smallest; the reference windows are already averages over several runs while the candidate is timed once, so
  the smallest ratio systematically understated the candidate).
- The frozen full-loop reference of a lane is the mean of the three runs of the reference tree its preparation timed.
  `RBB_SHARED_TIMING_REF` (with `RBB_FROZEN_ANY_GPU=1`) lets every lane use one reference recorded on a quiet machine
  instead of its own; a reference recorded with another judge is an error, never a silent re-measurement.
- Shape-dependent first-use costs (Triton autotuning and compilation) are paid in an untimed priming run; the judge
  reports any first-use time left inside a timed window.
- The task statement forbids skipping, shortening or reordering computation depending on the values of a step's
  rewards, advantages, losses or samples: this recipe caps the response length, so far more samples of a group share
  a reward (and get a zero advantage) than in a real run, and a shortcut that feeds on that does not carry over. Work
  the recipe's configuration makes dead regardless of the data (the KL term, whose coefficient is 0) may be removed.

## Preparation

Durations of each phase on one 4-GPU lane (2026-11-08); a single trial runs them in this order, about 7 h 15 min:

| phase | minutes | depends on |
|---|---:|---|
| fixture (18 rollouts, natural lengths) | 39 | — |
| rollout reference (forced lengths, scored, mismatch) | 23 | fixture |
| train records (10 × 11.3) | 113 | fixture; independent of each other |
| frozen inputs | < 1 | the above |
| quick check of the unchanged tree | 34 | the above |
| side timings (one more rollout run, two train runs) | 34 | fixture, rollout reference |
| the reference's own `--final` | 71 | fixture |
| controls (entry 0, logit bias 17, top-k 17, layout 11, loss sign 11, reduction order 11, lr × 1.02 11, train profile 41) | 119 | rollout and train references; independent of each other |

Smoke preparation (in a real trial with `RESEARCHBOOST_SMOKE=1`): 89 min (fixture 13, rollout reference
22, two records 21, quick check 33). For validation the phases ran on 8 lanes in parallel (about 3 h of wall time);
everything after the fixture and the two references is independent, which is also the content of a future
preparation pack (fixture, `task/rollout_ref.json`, `task/train_ref.json`, `task/train_ref_sketch.npy`,
`task/layout_ref.json`; the timings stay per machine).

### Preparation pack

`RESEARCHBOOST_ASSETS_ROOT/slime_grpo/pack/` (16 GB, in the asset root because of the fixture's dumps; the asset root is
mounted read-only, so no Compose change). `tasks/slime_grpo/prep-pack/` holds a copy of its `pack.json` and
`MANIFEST.sha256` for reference only. Contents: the fixture (`rollout_ref/` 18 dumps, `rollout_ref_forced/`,
`lengths.json`), `task/layout_ref.json`, `rollout_ref.json` (token forcing, top-k with the per-sample signed baseline,
mismatch), `train_ref.json` (records, tolerances, data-set sigma, cosine floor), `train_ref_sketch.npy`, and the
controls' evidence (`controls/*/check.json`, the quick and full checks of the full preparation). Made with
`bootstrap.py --make-pack DEST` (root, in the task image, with that preparation's workspace mounted).

- Used when `pack.json` matches: task, world size, GPU model, judge workspace hash (every file of `bench/`),
  `PREP_VERSION` and the hash of `installed-packages.txt`, and every file matches the manifest. It is restored
  root-owned into `/opt/researchboost/prep/fixture` and `ws/task/`.
- Then on every machine: the frozen inputs, the unchanged tree's quick check against the pack's references (must
  PASS, else the pack is discarded and the full preparation runs), two fresh rollout and two train timing runs, and the
  reference's own `--final` (full-loop windows and step fractions). No compile cache is shipped: every judge run starts
  with empty caches, on every machine.
- A changed judge (any file of `bench/`) changes the hash: the pack is ignored and the full preparation runs.
- The pack was assembled from the validation of 2026-11-08 (phases on 8 lanes). Its references were produced with
  revisions of `bench.py` before the final one; the later changes (`--n1`, the invalid-run signatures, the top-k bias
  rule, whose per-sample baseline was rescored with the final scorer, the scorer's PYTHONPATH, `freeze-timing
  --fresh`) do not change these artifacts, and the pack's quick check verifies them on each machine.

Measured in a real trial (unchanged candidate, 2026-11-08): 150 min instead of about 7 h 15 min
(pack restore 8 s, frozen inputs 16 s, quick check 34 min: PASS N1 with token forcing 0.999×, top-k bias t = −1.69,
sampler t = −0.21, direction cosine 0.9922; timing 45 min; the reference's `--final` 69 min: UNAVAILABLE,
0.981, reference spread 0.65%). This host's frozen spreads were rollout 2.9%, train 6.8% (agent's bar 1.137): the
train spread varies between hosts and runs (1.85% on another host), and each trial uses its own.

`environment/bootstrap.py`: fixture (18 rollouts with natural lengths) → reference rollout with forced lengths,
scored (token forcing and top-k baselines, mismatch) → ten reference train records → frozen inputs → the unchanged
tree's quick check → side timings → the reference's own full check (`--final`: UNAVAILABLE, no benefit; its full-loop
windows give the step fractions) → controls. `RESEARCHBOOST_SMOKE=1`: a 5-rollout fixture, two records, the quick
check.

| control | change | expected |
|---|---|---|
| entry | `slime/train.py` gains a statement | ERROR (structure) |
| logit bias | sglang adds +2 to every 7th vocabulary logit | INCORRECT (token forcing) |
| top-k | slime's sampler top-k extraction shifts every top-k log-prob but the first by +0.05 | INCORRECT (top-k) |
| layout | slime's arguments set 1 GPU per engine (4 engines × TP1) | INCORRECT (layout) |
| loss sign | the REINFORCE loss sign flipped | INCORRECT (statistical) |
| reduction order | the per-token loss sum in reversed order | PASS (N1) |
| lr × 1.02 | the reference with `LR=1.02e-6` (the statistical gate alone; the lr is an exact item for any tree) | INCORRECT |
| profile | `profile --side train` on the reference | a trace |

## Hosts

- The base image is a build argument (`BASE_IMAGE`); on hosts that cannot reach Docker Hub it is a local copy loaded
  from the registry blobs (`rbb/slime-base:540ea919`, the same image id). The Dockerfile checks that it is the pinned
  build (sglang build commit, Megatron-LM commit, the hash of `/opt/slime/constraints.txt`).
- Harbor must build it with the docker builder (a docker-container BuildKit builder does not see local images and
  asks the registry mirror for it).
- CUDA 13 on a 570 driver: the entrypoint puts the image's forward-compatibility `libcuda` first in the loader cache
  for every process of the container (the base image's own rule: drivers 535, 550, 565, 570, 575).
- Host memory: two concurrent record runs on one host peaked at about 140 GB of the host's 2 TB.

## Audit

Agents have internet access. The task statement forbids published optimized versions; after a trial, search its
agent logs (`grep -r -i -E 'AI4AI-for-Anything|Faster-Slime' <trial>/agent`) and void a trial that accessed them.

## Unusable measurements are measured again

The reference's own `--final` in preparation and the verifier's `--final` run up to 3 times when the result is `INVALID` (the two reference runs differ by more than `spread_band` in a window) or an `ERROR` the candidate did not cause (a reference run crashed). The 5% band is unchanged. Failed attempts are kept: `full_attempt_<k>` in preparation, one `slime_grpo-final-*` directory per attempt in the verifier. Added 2026-11-09 after two preparations were lost to a 7.6% and a 6.0% window and one to a crash of reference A.
