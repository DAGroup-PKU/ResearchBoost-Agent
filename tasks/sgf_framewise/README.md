# SGF framewise training — Harbor task

This N1-only task freezes the four-H200 workload and judge contract. The GPU
judge is implemented; Harbor validation, reference repeatability, and pilot runs
are still required before using it to score submissions.

## Reproduce the upstream workload on four GPUs

Use a disposable checkout of `zhuang2002/Self_Gradient_Forcing` at
`79be9acbd874c9be050944a6b1de6693e988a171`. Provision a CUDA/PyTorch environment
compatible with your H200 driver; install the pinned checkout's requirements
and the upstream FlashAttention package as described upstream.
**`diffusers==0.31.0` is required** for `CausalWanModel` gradient checkpointing.
The Harbor environment provisions its training dependencies offline.

From the upstream checkout, with the Hugging Face `hf` CLI available:

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --revision 37ec512624d61f7aa208f7ea8140a131f93afc9a \
  --include config.json diffusion_pytorch_model.safetensors Wan2.1_VAE.pth \
    models_t5_umt5-xxl-enc-bf16.pth 'google/umt5-xxl/*' \
  --local-dir wan_models/Wan2.1-T2V-1.3B
hf download Wan-AI/Wan2.1-T2V-14B --revision a064a6c71f5be440641209c07bf2a5ce7a2ff5e4 \
  --include config.json 'diffusion_pytorch_model-*-of-*.safetensors' \
    diffusion_pytorch_model.safetensors.index.json \
  --local-dir wan_models/Wan2.1-T2V-14B
hf download JunhaoZhuang/Self_Gradient_Forcing init/framewise/ar_diffusion.pt \
  --revision ffaf49017e4558fe02319b931f458ea2379bf606 --local-dir checkpoints
hf download JunhaoZhuang/Self_Gradient_Forcing vidprom_filtered_extended.txt \
  --revision ffaf49017e4558fe02319b931f458ea2379bf606 --local-dir prompts
sha256sum -c <<'SUMS'
96b6b242ca1c2f24e9d02cd6596066fab6d310e2d7538f33ae267cb18d957e8f  wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
721ea4d8f6d23099ccbefc6f5741b170402f509738f5a7b2db10e91b76857825  checkpoints/init/framewise/ar_diffusion.pt
7896742f468bc8aef9e4547424d1ce0a951acdb2a82233790155401a99bf5aa5  prompts/vidprom_filtered_extended.txt
SUMS
```

These are spot checks against publisher hashes, not the production asset
manifest. Before packaging, verify every file against its pinned source and
freeze a complete `SHA256SUMS` manifest, including the teacher shards and tokenizer.

Use the upstream framewise YAML with global batch size four and the frozen
generator context: six recent latent frames including the current frame,
with the first three retained when available. Copy the exact config overrides
from `gate0.exact_items.config` in `environment/workspace/bench/bench.json`.
Keep batch one per rank, no accumulation, and the upstream dual-pass update.
For a separate upstream smoke run (not the judge), set an explicit nonzero
`seed` and a smoke-run `max_steps` of at least 6 to include a generator update.
Save that copy as `configs/sgf_framewise_4gpu_smoke.yaml` and run:

```bash
: "${SGF_ENV:?set the absolute path to the training venv}"
: "${SGF_LOGDIR:?set a fresh absolute log directory}"
"$SGF_ENV/bin/python" -c 'import diffusers; assert diffusers.__version__ == "0.31.0"'
CUDA_VISIBLE_DEVICES="${SGF_GPU_IDS:-0,1,2,3}" NUM_GPUS=4 NNODES=1 NODE_RANK=0 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT="${SGF_PORT:-29500}" ENABLE_WANDB=0 \
  TORCHRUN="$SGF_ENV/bin/torchrun" \
  bash scripts/train_self_gradient_forcing_framewise.sh \
    configs/sgf_framewise_4gpu_smoke.yaml "$SGF_LOGDIR" --no_save --no_visualize
```

## Harbor and host use

- Harbor builds the pinned reference under `/opt/researchboost/ws/targets/reference`,
  copies it to `/submission/sgf_framewise`, and creates an agent-writable
  `/opt/candidate-env`. The image does not download weights. Provision the
  frozen assets and an Ubuntu 22.04 / Python 3.10 CUDA-compatible wheelhouse
  with hash-locked `runtime/requirements.lock` on the asset mount before trials.
- Compose expects `sgf_framewise` subdirectories under `RESEARCHBOOST_ASSETS_ROOT`
  (read-only) and `RESEARCHBOOST_SCRATCH_ROOT` (isolated per trial).
- `bench.py` exposes `record`, `compare`, `time`, and `check`; `check --final`
  is reserved for the trusted verifier. `sgf_entry.py` wraps the pinned trainer
  without requiring modifications to the upstream checkout. On a host, use
  `bench.py <command> --help` to supply the paths to your checkout, frozen assets,
  and training Python. Keep output outside the training checkout.
- `check --n1` checks both held-out trajectories against independent reference
  calibration without running the performance gate. It cannot award reward.
- Missing prerequisites report `UNAVAILABLE` (exit 2), rejected submissions
  report `INCORRECT` (exit 1), and invalid runs report `INVALID` (exit 3).
  Only a complete final check can award nonzero reward.

## Frozen judge contract

The reference uses upstream mixed precision, gradient checkpointing, the trainer's
TF32 settings, and FlashAttention-2 through the pinned upstream dispatch. Other
upstream attention paths stay intact. The GPU judge must confirm the reference
path actually executes; a fallback is not a substitute. Use the four-rank overrides
and baseline in `environment/workspace/bench/bench.json` to reproduce the judge.

- **Gate 0:** Rebuild the effective recipe independently of the candidate. Match
  configuration and actual traces in `gate0.exact_items`, including all Pass 1
  stages, per-rank sampling, dtype, and optimizer order. Count a completed
  `Trainer.train` iteration at the `self.step` increment after the critic step.
- **Gate 1:** In final, measure reference → candidate → reference on the same
  held-out windows. The benefit threshold uses the larger of the configured
  floor and one plus twice the live reference spread. `timing.spread_band` is a
  provisional machine-consistency band, not the candidate's benefit threshold.
  Repeat judge A/A before adopting the band. Reject incomplete or invalid runs;
  use the declared log signatures even if a process reports success.
- **Gate 3:** Same production configuration as timing. Compare seed-paired
  endpoints at both critic-only and generator+critic updates. Bitwise matches
  still need the statistical gate; there is no N0 scoring path.
- **Switch manifest:** `{}` is valid. Only explicitly allowed, non-reference
  environment settings may be recorded under `environment`. The current
  allowlist is empty, and enabling reference defaults does not count toward
  the switch ladder or qualify as an implementation change.

## Maintainer Leak Audit

Do not include the private deny file in the task or image. Run `leak_guard.py`
with that file against the source and, when available, the built image. The
following grep-style checks are a source backstop. Run from the task directory
with `SGF_DENY_FILE` set to the absolute path; exit 1 from `rg` means no matches.
Inspect each hit and read
the instruction and copied sources for less literal hints. The pristine pinned
upstream checkout is checked for revision and cleanliness separately.

```bash
: "${SGF_DENY_FILE:?set the path to the private task deny file}"
paths=(instruction.md environment tests)
rg -ni --pcre2 -f <(awk 'NF && $1 !~ /^#/' "$SGF_DENY_FILE") "${paths[@]}"
rg -ni '[0-9]+([.][0-9]+)?[[:space:]]*[x×]([^[:alnum:]]|$)|[h]ot[- ]?spot|[b]ottleneck' "${paths[@]}"
```
