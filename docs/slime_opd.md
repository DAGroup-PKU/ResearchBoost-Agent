# slime_opd: on-policy distillation in slime (student Qwen3.5-9B, teacher Qwen3.5-27B, 4 GPUs)

Status: run with nine agents (November 2026). The judge is model-agnostic: another student/teacher pair changes only
the recipe (`bench/launch.sh`), the layout (`bench.json`) and the assets.

Reference: [THUDM/slime](https://github.com/THUDM/slime) @ `540ea919`, in the image of `slime_grpo` (slime's official
image of the same day: sglang v0.5.21 and Megatron-LM `4b4acac9`, both with slime's patches; torch 2.13, CUDA 13). The
student's rollout stack and the teacher's server (sglang) and the training stack (Megatron-LM) are part of the
candidate: `/submission/slime_opd/{slime,sglang,Megatron-LM}`.

Built on `slime_grpo`'s judge (same conventions, clocks, layout check, token forcing, train records and statistical
gate); what is new is the teacher.

## Workload

Upstream `examples/on_policy_distillation/run-qwen3-4B-opd.sh` (sglang teacher mode), with the models replaced by
owner decision and made colocated on 4 GPUs:

| | upstream | here |
|---|---|---|
| student | Qwen3-4B-Instruct-2507 | Qwen3.5-9B |
| teacher | Qwen3-30B-A3B-Instruct-2507 (external sglang, TP2) | Qwen3.5-27B (sglang, TP4, resident on the student's GPUs) |
| GPUs | 8: actor 4 (TP1 × CP4) + rollout 2 (2 engines × TP1) + teacher 2 | 4, colocated: actor TP2 × CP2 (DP1), 2 engines × TP2, teacher TP4 |
| batch | 16 prompts × 4 samples, global batch 64 | the same |
| response limit | 131072 | 8192 |
| checkpoint | saved every 20 rollouts | not saved |

The student's parallel layout is the Qwen3.5-9B recipe's (`slime_grpo`), not the 4B OPD recipe's: the 9B student's
own layout at the same per-GPU load. Everything else is upstream's OPD recipe: GRPO advantage estimator with zero task
rewards (`reward_func` returns the teacher's answer, `post_process_rewards` returns 0 rewards and sets
`sample.teacher_log_probs`), `--use-opd --opd-type sglang --opd-kl-coef 1.0` (advantages minus the per-token reverse
KL), KL loss coefficient 0, entropy 0, Adam lr 1e-6 with CPU-offloaded fp32 master weights and moments, full
recompute, dynamic batches of up to 4096 tokens per GPU (the 9B recipe's), the teacher server with upstream's
arguments (chunked prefill 4096; static memory 0.25 here instead of 0.8, because it shares the GPUs). The student and
the teacher share their tokenizer byte for byte (`asset-prep` checks it): the teacher scores the student's token ids.

Memory budget (to be confirmed on the first run): teacher 0.25 × 143 GB (weights ~13.5 GB per GPU at TP4, the rest KV
for prefill-only scoring), student engines 0.45 × 143 GB while they run (released during training by slime's
offload), Megatron with CPU-offloaded Adam in what remains.

## Assets

`RESEARCHBOOST_ASSETS_ROOT/slime_opd`, provisioned by `environment/asset-prep/prepare_slime_opd_assets.sh <task image>
<assets root> [gpu]` (pinned revisions, `asset-prep/SHA256SUMS`):

- `Qwen3.5-9B/`: `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (the student, as `slime_grpo`);
- `Qwen3.5-27B/`: `Qwen/Qwen3.5-27B` @ `fc05daec18b0a78c049392ed2e771dde82bdf654` (the teacher, 52 GB; the shards'
  checksums are the LFS sha256 Hugging Face publishes for that revision, the small files' were computed from a copy
  whose git blob ids match the revision; README.md is not needed and not listed);
- `data/dapo-math-17k.jsonl`: `zhuzilin/dapo-math-17k` @ `2e65612930298bde4c5d58fd97b3f23a483aaff9`;
- `Qwen3.5-9B_torch_dist/`: the reference slime's `tools/convert_hf_to_torch_dist.py` of the student. The teacher
  needs no conversion (sglang only).

## Judge: three sides

`environment/workspace/bench/bench.py` (see its docstring). The agent declares nothing: the judge reads the diff.

| side | what changes it | timing | correctness |
|---|---|---|---|
| rollout | `sglang/`, `slime/slime/rollout/`, rollout manager, router | rollout-only, forced lengths, the candidate's teacher online | student token forcing (stock student engine), the rollout's teacher log-probabilities against the stock teacher replaying the same tokens, mismatch |
| teacher | `sglang/` (it also runs the teacher); `slime/slime/rollout/on_policy_distillation.py` and `rm_hub/` (teacher only); switch `OPD_TEACHER_SERVER_ARGS` | teacher replay: fixture rollouts 0–3 through the tree's reward path, the teacher alone | identical tokens against the frozen stock replay: exact items, rate, events, bias |
| train | `Megatron-LM/`, `slime/slime/backends/megatron_utils/`, `slime_plugins/models/` | train-only on the frozen fixture (whose dumps carry the teacher's answers) | `slime_grpo`'s records: bitwise N0 or the statistical gate N1, mismatch |

Why a teacher side: the teacher does prefill-only scoring (different hotspots from the student's decode), and its
input can be fixed exactly, so it gets a tight identical-token comparison instead of the rollout's statistical one.

**The teacher's reward path.** In slime, `async_rm` calls the custom reward function per sample during the rollout
(the teacher's raw `/generate` answer lands in `sample.reward`), and the batch builder calls the post-processing when
the rollout becomes training data (`sample.teacher_log_probs`). The rollout dump therefore holds raw answers, and a
train-only run applies the tree's own post-processing to the fixture. `bench/teacher_replay.py` calls the tree's
`async_rm` and post-processing exactly so, which covers every teacher code path of both runs: a change of the
post-processing that would alter the training data is caught by the teacher side on identical tokens.

**Teacher rules** (`teacher_rules` in `bench.json`; `teacher_verdict`): per token d = candidate − stock A.
The reference spread pools four re-batchings of the unchanged teacher (`teacher_rules.rebatchings`: requests one at
a time, chunked prefill 8192, `--disable-overlap-schedule`, `--disable-radix-cache`), each − stock A (all at once):
the closest stand-ins for a legitimate kernel or scheduling change. Rate: mean |d| ≤ 2 × max(spread, floor 0.001).
Events: τ = max(2 × the pooled 99.99th percentile, 0.05), binomial test against the pooled rate. Bias: the per-sample
mean of d with |t| > 4 and |mean| > 0.001 is INCORRECT.

Why re-batchings (2026-11-10, measured on one host against another host's reference replay): the teacher, Qwen3.5-27B, is a
hybrid model whose linear-attention (gated delta rule) layers carry a recurrent state across the prefill chunks of a
request. Repeats of one batching are bitwise identical (25 of 25 replays, also on the candidate code path and under
CPU load), and 8 requests at a time (the first design's "another batching") is no other batching at all: chunk
boundaries follow the first-come-first-served token stream. That froze the floors (τ 0.05, rate 1/n, bound 0.001),
which every legitimate re-batching fails: each of the four moves every token after a request's first chunk boundary,
mean |d| 6.7–6.9e-3, 5% of tokens > 0.05, 99.99th percentile 0.249, max 0.76–1.25 (each itself bitwise reproducible),
and the unchanged teacher once did so on its own (24 samples of one replay on a shared host, rate 3.3e-3: a
scheduling race). Pooled baseline: τ ≈ 0.50, 34 events in 8.2M tokens (rate ≈ 4e-6); against it the four re-batchings
and that race pass, and the +0.05 shift (rate 7.3×, bias t ≫ 4) and the off-by-one read (rate 88×, 6e5 events) stay
INCORRECT.

**The rollout's teacher gate** compares the teacher log-probabilities a candidate rollout carries (its online
answers) with the stock teacher replaying the same tokens; its baseline is the reference rollout's own online answers
against the same replay (`teacher` in `task/rollout_ref.json`), with the same rules.

**Projection.** The teacher's share of a step on the critical path is the end of a rollout minus the last generation
end of its samples (`bench/force_len.py` stamps generation ends): scoring overlaps the other samples' generation,
except for the tail. `fractions.teacher` = that tail / the full-loop step. When the rollout side is timed too, its
windows contain the teacher's share and the teacher's term is dropped. The final evaluation times the full loop.

Unchanged from `slime_grpo`: the commands and conventions, the judge's own clocks, the dump outside the timed window,
the layout from what ran (now also the teacher server's `get_server_info`), `check --n1`, invalid-run signatures,
cold compile caches, the train statistical gate with the paired-noise and direction bounds. Dropped: the top-k gate
(the OPD recipe has no score centering).

**Router policy (deviation from upstream).** The recipe fixes the student router to `--router-policy round_robin`
(slime's default is `cache_aware` with balance threshold 10). With `cache_aware` the split of a rollout's 64 requests
over the two engines depended on arrival timing (32/32, 31/33 or 28/36 in the development runs), and the 28/36 rollout took
~5 s longer (reference rollout pairs 7.0% and 7.3% apart, one slow window each); round robin splits 32/32 every time.
A candidate may still change the routing (it is on the rollout side).

**Train gate: the first step's gradient norm (2026-11-11).** The statistical gate's gradient-norm endpoint was the mean
over the 3 recorded steps. On one host the unchanged reference's quick check failed it (trial preparation, pack mode):
step gradient norms 0.997, 2.091, 0.881 against the frozen reference's 0.997, 1.001, 0.886 (mean 1.32 vs 0.96; d 0.36,
paired-noise tolerance 0.055, 3σ over data sets 0.28). Step 0 is computed on the frozen state before any update and is
tight (0.99641–0.99679 over 9 reference records on data set 0 from three hosts); steps 1–2 depend on the run's own
non-bitwise first update and are heavy-tailed (data set 3 also gave 1.67 at step 1). A rule the unchanged reference
cannot meet, so the endpoint is now the first step's gradient norm (tighter where it is defined; later steps stay
covered by the update norm, Adam's second moment, the mean loss and the update direction). No control depended on the
old endpoint: the flipped PPO loss is caught by the mean loss and the direction (cos −0.994), lr × 1.02 by the update
norm.

**Reference timing runs that crash.** A reference timing run died with an NCCL error (DistBackendError) in
train timing pair 1 and stopped the preparation; such a pair is now measured again (up to 2 times, not counted as a
pair).

## Controls (bootstrap)

| control | change | expected |
|---|---|---|
| entry | `slime/train.py` gains a statement | ERROR (structure) |
| logit bias | sglang adds +2 to every 7th vocabulary logit (student and teacher) | INCORRECT (student token forcing) |
| teacher shift | `post_process_rewards` adds +0.05 to every teacher log-prob | INCORRECT (teacher bias rule) |
| teacher off by one | `post_process_rewards` reads the teacher log-probabilities one token late (`[2:]` + the last again; `[:-1]` reads the first token's None and crashes) | INCORRECT (teacher rate/event rules) |
| teacher chunking | `OPD_TEACHER_SERVER_ARGS=--chunked-prefill-size 8192` (teacher-only diff; one of the four re-batchings of the baseline) | PASS (N1) |
| teacher layout | `OPD_TEACHER_SERVER_ARGS=--tp 2` | INCORRECT (layout) |
| student layout | slime's arguments set 1 GPU per engine | INCORRECT (layout) |
| loss sign | the PPO policy loss sign flipped (`compute_policy_loss`, the recipe's loss; the REINFORCE line slime_grpo flips is not on this path: flipping it passed as N1 on 11-10) | INCORRECT (statistical) |
| reduction order | the per-token loss sum in reversed order | PASS (N1) |
| lr × 1.02 | the reference with `LR=1.02e-6` (statistical gate alone) | INCORRECT |
| profiles | `profile --side train` and `--side teacher` on the reference | traces |

Note on the OPD coefficient: `opd_kl_coef × 1.02` is not a control. With zero task rewards the advantages are
proportional to the coefficient, and Adam is invariant to that scale (apart from epsilon), so it barely changes the
update; the teacher-side controls cover wrong teacher signals instead.

## Thresholds and timing

The reference step takes about 155 s on 4 Hopper GPUs: rollout about 76 s (including the teacher's scoring tail of about
16.5 s) and train about 74 s; the frozen step fractions are about 49% rollout, 48% train and 11% teacher replay (the
teacher overlaps the rollout). Each trial's own thresholds, spreads and control outcomes are in its `prepare.log` and
`task/ready.json`.


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

Order (`environment/bootstrap.py`): fixture (18 rollouts, natural lengths, teacher online) → reference rollout with
forced lengths (student token forcing, the rollout's teacher baseline, mismatch) → stock teacher replays A, B, C →
ten reference train records → frozen inputs → quick check (all three sides) → side timings → the reference's own
`--final` → controls. Durations are in `prepare.log`. `RESEARCHBOOST_SMOKE=1`: a 5-rollout fixture, the rollout and teacher
references, two records, the quick check.


## Audit

Agents have internet access. The task statement forbids published optimized versions; after a trial, search its
agent logs (`grep -r -i -E 'AI4AI-for-Anything|Faster-Slime' <trial>/agent`) and void a trial that accessed them.
