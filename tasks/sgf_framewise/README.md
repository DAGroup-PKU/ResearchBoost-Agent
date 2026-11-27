# SGF framewise training — Harbor task skeleton

Stage A (desk review and recipe definition) and the Stage B reference timing
and A/A runs are complete on a single node with four H200 GPUs. Stage B is not
fully qualified. Stage C (the real GPU judge and Harbor `nop`), Stage D (pilot),
and Stage E (clean rejudging) are pending. This is an N1-only task, not a runnable
benchmark judge yet.

## Reproduce the upstream workload on four GPUs

Use a disposable checkout of `zhuang2002/Self_Gradient_Forcing` at
`79be9acbd874c9be050944a6b1de6693e988a171`. Provision a CUDA/PyTorch environment
compatible with your H200 driver and build of FlashAttention; install the pinned
checkout's requirements and FlashAttention as described upstream.
**`diffusers==0.31.0` is required** for `CausalWanModel` gradient checkpointing.
The Harbor skeleton does not install this training environment.

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

Keep the upstream framewise YAML, changing only `total_batch_size` from 8 to 4
for the four-rank workload. Keep `batch_size: 1`, no accumulation, and all
generator/critic, denoising, cache and context-gradient settings unchanged.
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

## Harbor skeleton and judge status

- Harbor builds the pinned reference under `/opt/researchboost/ws/targets/reference`,
  copies it to `/submission/sgf_framewise`, and creates an agent-writable
  `/opt/candidate-env`. The structural image does not download or load weights.
- Compose expects `sgf_framewise` subdirectories under `RESEARCHBOOST_ASSETS_ROOT`
  (read-only) and `RESEARCHBOOST_SCRATCH_ROOT` (isolated per trial).
- `bench.py` exposes `record`, `compare`, `time`, and `check`; the commands return
  `UNAVAILABLE` (exit 2). `sgf_entry.py` prints a launch plan only. The shared
  verifier fails closed; its zero reward is not evidence of a real `nop` run.
- `timing.spread_band` stays `null` until Stage C repeats A/A through the actual
  judge. Do not use this skeleton to score candidates or build a reference pack.
