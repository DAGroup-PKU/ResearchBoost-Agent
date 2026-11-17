# llava

Make LLaVA-1.5 training faster per optimizer step. The task statement is
[`tasks/llava/instruction.md`](../tasks/llava/instruction.md).

## Workload

Upstream [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) at `c121f04`, stage-2 instruction tuning
(`scripts/v1_5/finetune.sh`, entry point `llava/train/train_mem.py`) with the 7B model: `vicuna-7b-v1.5`,
`clip-vit-large-patch14-336` (frozen) and the pretrained `mlp2x_gelu` projector. Upstream runs 8 GPUs × 16 per GPU
(global batch 128); here 2 × H200 with upstream's per-GPU micro-batch of 16 and gradient accumulation 1, so the global
batch is 2 × 16 = 32, upstream's scaled by 2/8 (`model_max_length` 2048, bf16 + tf32, DeepSpeed ZeRO-3,
flash-attention 2, gradient checkpointing). The frozen arguments are `train_args` in `bench/bench.json`. Timing: 5
warmup steps and 2 windows of 10 optimizer steps on a fixed fixture.

## Data and weights

`<assets root>/llava` (about 17 GB), filled by the Compose `assets` service or
`python scripts/prepare_assets.py --task llava` (manifest: `tasks/llava/environment/asset-prep/assets/llava.json`):

| Path | Source | Check |
| --- | --- | --- |
| `models/vicuna-7b-v1.5/` | HF `lmsys/vicuna-7b-v1.5` @ `3321f76` | size and sha256 per file |
| `models/clip-vit-large-patch14-336/` | HF `openai/clip-vit-large-patch14-336` @ `ce19dc9` | per file, and the directory digest the judge was formulated with |
| `models/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/` | HF `liuhaotian/…` @ `5414da8` | as above |
| `data/llava_v1_5_mix665k.json` | HF dataset `liuhaotian/LLaVA-Instruct-150K` @ `9d451dc` | sha256 |
| `data/fixture/{timing,case_mm,case_text,probe}.json` | rebuilt from the frozen source indices in `task/fixture_manifest.json` | sha256 per subset |
| `data/images/` (2,319 images) | members of public image zips on the Hub (COCO, GQA, TextVQA, VG, OCR-VQA mirrors at pinned revisions), read by HTTP range requests | zip size, member CRC32, directory digest |

Partial downloads and cached zip directories live outside the frozen directories; large files are fetched with
parallel, resumable range requests (`RESEARCHBOOST_DOWNLOAD_WORKERS`, default 8).

## Image

There is no prebuilt image: Harbor builds it on site from `tasks/llava/environment/Dockerfile` at the start of every
trial (fast when the builder still has the layer cache). The build needs network access to the base image
`nvidia/cuda:12.1.1-devel-ubuntu22.04` (pinned by digest), the Ubuntu archive, pypi.org (or a mirror: build argument
`PIP_INDEX_URL`) and download.pytorch.org (`TORCH_INDEX_URL`), and GitHub (the upstream code at the pinned commit and
the flash-attn release wheel). Data and weights are not in the image (see above). Python 3.10 from Ubuntu, as
upstream's README; nvcc 12.1 matches torch's CUDA for DeepSpeed and extension builds.

- `/opt/llava-env`, the training environment, root-owned (see provenance below).
- `/opt/candidate-env`, owned by `agent` and first on `PATH`: a venv made with the training environment's Python whose
  `.pth` file adds the training environment's site-packages after its own.
- `/submission/llava`: the upstream checkout on branch `opt`, owned by `agent`.

Environment provenance. **Pinned: only CUDA (the base image digest) and torch 2.1.2+cu121 / torchvision
0.16.2+cu121**, the stack the judge was formulated on (`environment/constraints.txt`, the constraints file of every
install, so nothing can replace them). The rest is upstream's own install, as its README says: `pip install --upgrade
pip`, `pip install -e ".[train]"` (pyproject.toml's pins: transformers 4.37.2, tokenizers 0.15.1, accelerate 0.21.0,
DeepSpeed 0.12.6, sentencepiece 0.1.99, scikit-learn 1.2.2, gradio 4.16.0, einops 0.6.1, timm 0.6.13, ...) and
`pip install flash-attn --no-build-isolation`. Unpinned packages resolve at build time, so **their versions may drift
between builds**. Differences from the README:

- the editable install runs on a scratch copy and the `llava` package itself is uninstalled afterwards: the judge
  imports `llava` from each tree under test, never from site-packages;
- DeepSpeed (upstream's 0.12.6) is installed first, without build isolation, because its `setup.py` imports torch;
- **extra pin peft 0.10.0**: pyproject leaves peft unpinned, and the current release (0.21.2 on 2026-10-23) imports
  `accelerate.utils.memory.clear_device_cache`, which upstream's accelerate 0.21.0 lacks, so `from transformers
  import Trainer` fails and the reference cannot start. 0.10.0 is the version the judge was formulated with;
- **extra pin flash-attn 2.5.8**: the README names no version, and current flash-attn has no prebuilt wheel for
  torch 2.1 (it would compile from source, if at all). 2.5.8 is the version the judge was formulated with. Its
  `setup.py` would fetch the matching release wheel (cu122, torch 2.1, pre-C++11 ABI, sm_80/sm_90) from GitHub in a
  single attempt; the build fetches the same wheel with retries and checks its sha256.

The build fails if `llava.train.train` does not import, and records the installed set in
`/opt/researchboost/installed-packages.txt`; the judge records each run's library versions. No optional performance
libraries are installed (no Apex, no Liger, no Nsight Systems): the agent can install what it wants into its
environment.

A build on 2026-10-23 keeps every version the judge's adapters depend on (torch, transformers, tokenizers,
accelerate, DeepSpeed, flash-attn, numpy 1.26.4, peft) and differs elsewhere from the environment the judge was
validated with, mostly through upstream's serving dependencies (gradio 4.16.0 and its stack, bitsandbytes 0.45.5,
wandb 0.31.0, now installed). Differences in packages the judge touches: **Pillow 10.4.0 (was 12.3.0**, gradio 4.16
requires Pillow < 11; image decoding feeds the loader digests), huggingface-hub 0.36.2 (0.20.3), safetensors 0.8.0
(0.4.2), protobuf 7.36.2 (4.25.3), scipy 1.15.3 (1.13.1), pydantic 2.14.1 (2.6.4). With these, the CPU loader probe of
the built image equals the formulation's probe exactly, sample digests included, at world size 4 and 2.

## Startup preparation

`environment/bootstrap.py` runs as root before the container reports ready:

1. Exactly 2 GPUs of compute capability 9.x (`RESEARCHBOOST_TEST_WORLD_SIZE` may be unset or 2, the task's own
   size); GPU lock `/tmp/llava_gpu.lock` with mode 0666; reference HEAD `c121f04`; absolute asset paths written into
   `bench/bench.json`; `task/formulation.json` (workload, fixture provenance).
2. Loader reference (`bench/loader_probe.py`, CPU), compared with the validated probe
   (`prep/loader_reference.validated.json`, made by the first preparation at 2 GPUs × accumulation 1 on 2026-10-27):
   a difference beyond the sample digests is fatal. A validated probe of another number of micro-batches per
   optimizer step (another recipe configuration) is not compared: the probe is only recorded in
   `task/loader_reference.json`.
3. Reference records twice and with the `sdpa` variant; the repeat must be bitwise identical; the recorded
   gradients must be exactly the trainable parameters, all nonzero in the image-text case.
4. Six reference trajectory seeds (0–5) and the frozen δ.
5. Frozen inputs; the reference timed three times (`task/timing/`, `ref_a` frozen provisionally, then `ref_b`,
   `ref_c`), and `bench.py freeze-timing` freezes their mean, per-window times and spread ((max − min) / mean) as
   `task/timing/reference.json`, what `time` and the final check's noise rules compare against. The first logged loss
   is compared with `VALIDATED_FIRST_LOSS` in `bootstrap.py`, 1.3839 at this configuration (more than 5% off fails,
   more than 1% warns).
6. The reference's quick check (PASS, bitwise) and full check (UNAVAILABLE, no_benefit); the numerics self-test.
7. Acceptance controls (`prep/make_accept.py`):

| Control | Change | Mode | Expected |
| --- | --- | --- | --- |
| `setup_train` | a line inside `train()` | `--quick` | ERROR (setup guard) |
| `sampler_no_accum` | the sampler forgets the accumulation factor | `--quick` | INCORRECT (loader check) |
| `projector0_detached` | the first projector layer gets no gradient | `--quick` | INCORRECT (dead gradient on exactly that layer) |
| `rmsnorm_div_sqrt` | RMSNorm `x / sqrt(v + eps)`, equivalent | `--skip-timing` | PASS (statistical gate) |
| `soft_labels` | label smoothing 0.1 | `--skip-timing` | INCORRECT (statistical gate) |

8. Control trees, scratch records and preparation code are deleted; `task/ready.json` records the cross-checks.

LLaVA's `bench.py check` exits 0 for every verdict it writes, so preparation and the verifier read the verdict from
`check.json`. `RESEARCHBOOST_SMOKE=1` keeps steps 1–3, one timing run and the quick check.

**Preparation pack** (README): with a matching pack in `tasks/llava/prep-pack/ws<world size>/` (mounted read-only),
startup restores it, checks it against its manifest and measures this machine's reference values: the reference
records twice, the timing baseline (three runs, their mean frozen), the reference's quick and full checks. With records
bitwise identical to the pack's, the pack's thresholds, trajectories, delta, self-test and control evidence stand
(`pack-match`); otherwise the records, the `sdpa` variant and the thresholds, and the reference trajectories of the
six seeds, are recorded on this machine, delta stays the pack's, and the self-test and every control run
(`pack-local-trajectories`). `PREP_VERSION` in `bootstrap.py` must match the pack's.

## Judge notes

The judge comes from the earlier optimization campaign for this workload. Gates, thresholds and decision rules are
unchanged. Changes for this bench:

- Interpreters: `bench.json` names `python` (`/opt/llava-env/bin/python`, used instead of `sys.executable`) and
  `candidate_python`. The tree under test (timing runs, records, trajectories, data probes) runs with the candidate
  interpreter; the reference, the formulation steps and a `time`/`profile` run of the reference checkout itself use
  the training environment, with its `bin` first on `PATH` and `PYTHONNOUSERSITE=1`.
- Library versions are read with each run's own interpreter and recorded in its fingerprint
  (`performance.library_versions`), but no longer compared, in `check` and in `sweep`. Every other fingerprint field
  is still compared.
- `bench/llava_adapter.py` makes flash-attn's backward deterministic for gates 2 and 3, also for flash-attn ≥ 2.7
  (the backward then goes through `torch.ops`); with 2.5.8 it behaves as before.
- Removed from the workspace: the judge's change log and a per-rank load report script; comments that pointed at the
  old task narrative now point at the task statement. `nsys` is `null`, so `bench.py profile` needs `--detail`.
- `reference_commit` is upstream `c121f04`; the image fetches upstream at that commit.
- Timing plan (the same on llava, rae and yoco): 5 warmup steps and 2 windows of 10 optimizer steps per run;
  `performance.speedup` = the mean step time of the check's two reference runs ÷ the candidate's. `time` reports
  `speedup_vs_frozen_reference` (the frozen reference's step time ÷ this run's), `reference_noise` (the frozen runs'
  spread) and `beyond_reference_noise` (whether the gain exceeds twice that spread). The speedup bar is
  max(1.02, 1 + 2 × the relative gap between the check's two reference runs).
- Noise makes a result INVALID only in `check --final` (the verifier): a run whose windows, each relative to the frozen
  reference's same window, differ by more than 6% (`max_window_spread`; the windows hold different mixes of
  image-text and text-only samples, so they are compared with the reference's own windows), or a reference run more
  than 15% off the frozen reference (`max_reference_drift`). Elsewhere these are reported in `notes`.
- CPU binding (`bench/cpu_affinity.py`, the same file on llava, rae and yoco): every rank of every judge run is pinned
  to the physical cores of its GPU's NUMA node that the container has, split evenly between the ranks there, and
  PyTorch's intra-op threads are set to that many physical cores. The task's `cpus` is 48 hardware threads, the
  machine's share of 2 GPUs (2 sockets × 48 physical cores for 8 GPUs = 12 physical cores per GPU). Each GPU run
  reports the CPUs the container kept busy (`cpu.busy_cpus`, `busy_cpus_in_windows` for timing runs).
- The record module (`numerics/record.py`, the same on llava, rae and yoco) reports the time of each recording phase
  (`record_phases`), and `check` reports the time of the comparison.

## Time budget and agent harness

- The statement's time budget gives approximate judge durations measured at the earlier configuration (world size 2
  with gradient accumulation 4 on 2 Hopper GPUs): `profile --detail` 4–5 min, `time` about
  4 min, `check --quick` 5–7 min, `check --skip-timing` about 11 min (bitwise) or 25–30 min (gate 3), the full check
  about 10 min when it stops at gate 1 and 35–40 min through gate 3. They are to be measured again at 2 × H200 with
  accumulation 1.
- With the [agent harness](../README.md#agent-harness) (`harness=rba`), the addendum's commands for llava are
  `profile --detail`, `time`, `check --quick` and `check` with `/opt/llava-env/bin/python bench/bench.py` on
  `/submission/llava`.

## Verifier

`tests/verify.py` checks `switches.json` with the judge's own `candidate_environment` (an invalid file scores 0),
kills every `agent` process, runs the full check as `agent` with the training environment's Python and
`PYTHONNOUSERSITE=1`, keeps the judge's output under `/logs/verifier` and converts `check.json` as in the
[README](../README.md#scoring). Only a full check (not `--quick` or `--skip-timing`) can be rewarded.

Before judging, it keeps the agent's `/submission/WORK_LOG.md` (if any) and the tree's `git log --stat` since the
reference commit plus `git status` (run as `agent`) as `WORK_LOG.md` and `git-log-stat.txt` in `/logs/verifier`;
they are trial records, not part of the score.
