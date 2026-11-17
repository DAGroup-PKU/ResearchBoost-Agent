# yoco

Make YOCO-3B pretraining faster per update. The task statement is [`tasks/yoco/instruction.md`](../tasks/yoco/instruction.md).

## Workload

Upstream [microsoft/unilm](https://github.com/microsoft/unilm) at `f4f7b27a`, its `YOCO/` directory, entry point
`yoco/train.py` (fairseq's `cli_main()`), recipe `scripts/train.sh`, frozen in `bench/recipe.py`. Changes from
upstream's script:

- `--arch yoco_3b`: the script's `yoco_base` is not registered.
- The task's corpus (below) instead of upstream's internal data path; `--seed 1` written out; no validation set.
- `--zero-sharding os`: the optimizer state sharded across the ranks (without it the reference does not fit on
  80 GB GPUs; kept for the same optimizer layout).
- **2 data-parallel processes on one machine** (the judge was formulated with 8). The per-GPU load is upstream's:
  2048 tokens per sample, batch 1 per GPU, `--update-freq 1`, bf16 with fp32 master weights. The global batch is
  2 × 2048 tokens per update instead of 8 × 2048.

The judge times 5 warmup updates and 2 windows of 10 updates. The task runs on 2 × H200 (`cpus` 48, 12 physical
cores per GPU).

### From 8 to 2 GPUs

The judge (`bench/`, `numerics/`) was formulated on one machine with 8 GPUs. Everything in it that depends on the
machine or the rank count is recorded again at every startup on the trial's two GPUs, with the judge's own procedure
(`environment/bootstrap.py`): the bitwise reference records, the diagnostic thresholds, the loader digest (one shard per
rank: 2 instead of 8), the six reference trajectories and their frozen δ, the per-update random-number stream. The
gates, their thresholds and the decision rules are unchanged. Consequences:

- Each update averages the gradient over 2 sequences instead of 8, so the trajectories are noisier and δ (twice the
  reference's seed spread) is a property of the 2-GPU recipe; the acceptance controls below are what shows that the
  statistical gate still separates a wrong candidate (learning rate × 1.02, half the loss gradient dropped) from an
  equivalent one (an RMSNorm rewrite) at this size.
- ZeRO shards the optimizer state over 2 ranks, so each GPU holds half of the Adam state instead of an eighth (about
  62 GiB peak allocated per GPU on H200, measured, instead of 52 with 8 ranks; Hopper GPUs with 80 GB or more).
- Timing: the same per-GPU work with less communication per update; speedups are not comparable with an 8-GPU
  measurement.

## Data and tokenizer

`<assets root>/yoco`, filled by the Compose `assets` service or `python scripts/prepare_assets.py --task yoco`
(manifest: `tasks/yoco/environment/asset-prep/assets/yoco.json`):

| Path | Source | Size |
| --- | --- | --- |
| `data/Data/shard/fineweb_edu/` (73 shards) | HF dataset `HuggingFaceFW/fineweb-edu` @ `87f0914`, `sample/10BT/000_00000.parquet` (726,000 documents) | 2.9 GB |
| `data/Data/shard/codeparrot/` (10 shards) | HF dataset `codeparrot/codeparrot-clean` @ `35a59fb`, `file-000000000001.json.gz` (100,000 files) | 0.3 GB |
| `data/Data/json/` | `train.json` (fineweb_edu 0.8, codeparrot 0.2), `valid.json` (empty), the shard lists | |
| `tiktoken_cache/` | tiktoken's `cl100k_base` encoding | 1.1 MB |
| `downloads/` | the two source files (kept for a rebuild) | 1.1 GB |

The corpus is upstream's README layout (`{"text": ...}` lines, 10,000 per shard, in source-file row order, empty
documents skipped), built from the two pinned files. Every shard is checked against the sha256 of the corpus the
judge was formulated on; the corpus is published only when all 83 match.

## Image

Built on site from `tasks/yoco/environment/Dockerfile`: `nvidia/cuda:12.1.1-devel-ubuntu22.04` (pinned by digest),
Python 3.10 from Ubuntu.

- `/opt/yoco-env`, the training environment, root-owned. Pinned (`environment/constraints.txt`): torch 2.3.1+cu121,
  torchvision 0.18.1, numpy 1.23.0, the stack the judge was formulated on. Then upstream's `requirements.txt` at the
  commits of the formulated environment: the fairseq fork upstream names (`sunyt32/fairseq`, branch `zero_fs`, at
  `e872403f`, with its Megatron submodule), infinibatch (`cf9a5cb9`), torchscale (`4d1e0e82`), fairscale 0.4.0;
  tiktoken, sentencepiece, ninja, boto3 and iopath unpinned (resolved at build time). pip is pinned to 24.0 first:
  the fork pins omegaconf < 2.1, whose metadata newer pip rejects. NVIDIA apex at `70018365` with its C++ and CUDA
  extensions for sm_90, built against the base image's CUDA 12.1 toolkit (`setup.py bdist_wheel --cpp_ext
  --cuda_ext`: apex reads these flags from its own command line). flash-attn 2.6.3 from its official prebuilt wheel
  (torch 2.3, CUDA 12, C++11 ABI off, cp310). The build fails if `pip check` fails, a pinned version moved, or the
  entry's modules, apex's extensions or flash-attn do not import. Building apex takes most of the build time.
- `/opt/candidate-env`, owned by `agent` and first on `PATH`, as in the other tasks.
- `/submission/yoco`: the reference tree on branch `opt`, owned by `agent`. `/opt/researchboost/ws/targets/reference`
  is the reference: unilm's `YOCO/` directory at `f4f7b27a` as a repository of its own, one commit with fixed author
  and date (`d1ed4741`, reproducible; the image build checks it).

## Startup preparation

`environment/bootstrap.py` runs as root before the container reports ready (log:
`/var/log/researchboost/prepare.log`), following the judge's own formulation and acceptance procedure:

1. Exactly 2 GPUs of compute capability 9.x; GPU lock `/tmp/yoco_gpu.lock` with mode 0666; CPU affinity written into
   `bench/bench.json`. `RESEARCHBOOST_TEST_WORLD_SIZE` may be unset or 2 (this task's own size).
2. Reference records twice (`ref`, `ref_b`) and without deterministic algorithms (`var_nondet`); diagnostic
   thresholds from `ref_b` and `var_nondet`; `ref` and `ref_b` must be bitwise identical.
3. The loader digest (first 3 batches of each rank's shard) → `task/loader_digest.json`.
4. Reference trajectories on seeds 1–6, seed 1 again (it must reproduce exactly), the frozen δ.
5. Frozen input fingerprints (judge, loader digest, corpus, tokenizer, the training environment's fairseq trainer,
   optimizers and data iterator, apex's fused Adam, flash-attn's interface, infinibatch's iterators).
6. The reference timed three times (`bench.py baseline`): the per-update random-number stream must agree across the
   runs and is frozen in `task/timing_ref.json`, with the runs' mean update time, per-window times and spread
   ((max − min) / mean) in its `frozen` section (`prep/mkref.py`), what `time` and the final check's noise rules
   compare against.
7. The unchanged tree's quick check (PASS, bitwise), the numerics self-test, the acceptance controls, the unchanged
   tree's full check (UNAVAILABLE, no_benefit):

| Control | Edit | Mode | Expected |
| --- | --- | --- | --- |
| `setup_entry` | `yoco/train.py` turns on TF32 before `cli_main()` | `--quick` | ERROR (setup guard) |
| `wd0` | weight decay set to 0 in the optimizer's parameter groups | `--quick` | INCORRECT (exact items) |
| `freeze_norm` | the final RMSNorm weight never updated | `--quick` | INCORRECT (exact items) |
| `lr102` | the optimizer applies 1.02 × the scheduled learning rate | `--skip-timing` | INCORRECT (statistical gate) |
| `rms_reorder` | RMSNorm `x * rsqrt(m + eps)` → `x / sqrt(m + eps)` | `--skip-timing` | PASS (statistical gate, not bitwise) |
| `detach_half` | no gradient from the second half of every sequence's loss | `--skip-timing` | INCORRECT (statistical gate) |

8. Control trees and preparation code are deleted; `task/ready.json` is written. Any failure stops the container
   and the trial never starts.

`RESEARCHBOOST_SMOKE=1` keeps steps 1–3, one training run and the quick check.

**Preparation pack** (README): with a matching pack in `tasks/yoco/prep-pack/ws2/`, startup restores it, freezes the
inputs, times the reference three times on this machine (the `frozen` section above) and runs the unchanged tree's
quick check; the pack stands (`pack-match`) only when that quick check is bitwise identical to the pack's records,
otherwise the full preparation runs. `PREP_VERSION` in `bootstrap.py` must match the pack's.

## Judge notes

The judge comes from the earlier optimization work on this workload (the machine-b calibration of the published
judge, which also fixes a loader-probe hang). Gates, thresholds and decision rules are unchanged. Changes for this
bench:

- 2 ranks instead of 8 (`bench.json` `world_size`, `loader.shards`, `bench/recipe.py`); all machine-specific records
  re-recorded at startup (above).
- Interpreters: `bench.json` names `python` (`/opt/yoco-env/bin/python`) and `candidate_python`
  (`/opt/candidate-env/bin/python`). The tree under test runs with the candidate interpreter (torchrun as
  `python -m torch.distributed.run`, the setup-fingerprint and loader probes); reference runs and the formulator
  commands use the training environment.
- Library versions are read with each run's own interpreter and recorded in its fingerprint
  (`performance.library_versions` lists reference and candidate), but no longer compared.
- Paths: the corpus, the tokenizer cache (`tiktoken_cache`) and the CUDA toolkit (`cuda_home`) come from
  `bench.json` and `$ASSETS`; `cpu_affinity` (set at startup) keeps every run within the container's CPUs.
- Timing plan, `time` output, noise rules, CPU binding and the record module as in [llava](llava.md#judge-notes):
  `performance.speedup` = the mean update time of the two reference runs ÷ the candidate's (the ratio of median
  update times is reported as a diagnostic); in `check --final` a run whose window range ((max − min) / mean) exceeds
  6%, or a reference run more than 15% off the frozen reference, makes the result INVALID. Each rank is pinned to its
  NUMA-local physical cores inside the container's CPUs (`bench/cpu_affinity.py`, loaded by path so that the bench
  directory stays off the training process's `sys.path`). `check` records one process per case.
- Host-specific remarks in docstrings were removed; nothing else in the code paths changed.

## Validation (2026-10-24, 2 × H200, world size 2, agent `nop`)

At the earlier timing plan (50 warmup updates, 3 windows of 100, speedup the smaller of the total-time and median
ratios) and the earlier record module; the unified rules of 2026-10-27 are to be validated again.

On one machine whose other GPUs ran two other trials at the same time (against the README's one trial per host):

- Image: builds; apex's CUDA extensions take about 10 minutes of the build. `pip check` passes; the installed versions
  are those of the formulated environment (torch 2.3.1+cu121, triton 2.3.1, flash-attn 2.6.3, numpy 1.23.0,
  fairscale 0.4.0, tiktoken 0.14.0, the fairseq, infinibatch and torchscale commits above).
- Assets: the corpus rebuilt from the two pinned files matches the formulated corpus shard for shard (83 sha256).
- Smoke preparation: 44 min; the verifier's quick check of the unchanged tree: PASS, bitwise (10.9 min).
- Full preparation: 154 min. The reference records twice: 9616/9616 tensors bitwise identical; trajectory seed 1
  reproduces exactly; 0.236 s per update, 61.75 GiB peak allocated per GPU. δ per endpoint (updates 5–15, 15–25,
  25–35, 35–50, 50–100, 100–200): 0.0030, 0.0042, 0.0030, 0.0023, 0.0053, 0.0031. Controls: `setup_entry` ERROR
  (setup guard); `wd0`, `freeze_norm` INCORRECT (exact items); `lr102` INCORRECT after 3 seeds (the first four
  endpoints 0.007–0.022 below the reference, 3–7 δ); `rms_reorder` PASS (equivalent, every endpoint within δ after 3
  seeds); `detach_half` INCORRECT (every endpoint 0.03–0.24 off); the unchanged tree's full check UNAVAILABLE
  (no_benefit: measured 1.031, bar 1.058).
- **The verifier's full check of the unchanged tree was a false PASS** (`accepted` 1, `speedup` 1.032): in that check
  the two reference runs agreed within 1% (bar 1.02) and the candidate run happened to be 3–4% faster. Timing the
  reference and the unchanged tree alternately right after (same container, same GPUs) gave 0.2442, 0.2526, 0.2528,
  0.2389 s per update (reference, unchanged tree, reference, unchanged tree): swings of 3–5% within minutes,
  independent of the interpreter. The formulated judge measured 0.3% between three reference runs on a dedicated
  machine. The host was shared with two other trials whose processes used about 25 cores; YOCO's loader tokenizes on
  the fly in CPU processes. **Run yoco with no other trial on the host**; on a shared host its gate 1 is not reliable.

## Verifier

As for the other tasks (`tests/verify.py`): an invalid `switches.json` is a rejection, every `agent` process is
stopped, `bench.py check` runs as `agent` with the training environment's Python, the judge's output is kept under
`/logs/verifier`, and `check.json` becomes the reward as in the [README](../README.md#scoring).
