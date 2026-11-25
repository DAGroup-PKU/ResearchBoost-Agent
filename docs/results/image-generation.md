# Image Generation

[← All results](README.md) · [ResearchBoost-Agent](../../README.md#results)

4 projects · [G-STAR](#g-star) · [RAE](#faster-rae) · [DMD2](#faster-dmd) · [JiT](#faster-jit)

## G-STAR

**Text-to-image inference with the public code and weights: G-STAR-Pro-8B (32× compression) and G-STAR-Flash-1.7B (effective 64×), refiner included** · merged upstream in [G-STAR#2](https://github.com/DAGroup-PKU/G-STAR/pull/2)

| Item | Result |
|---|---|
| Speedup | **6.18×** Pro-8B + one-step SR2x refiner (the default) · **2.73×** Pro-8B + 52-step SR2x refiner · **2.72×** Flash-1.7B + g8 refiner |
| Hardware | 1 × H200 |
| Measurement scope | Seconds per image with the model loaded (language model, refiner and VAE decode; loading and the first warm-up image excluded), batch 1, 1024 × 1024 codes, two timed images after one warm-up; reference → optimized → reference in fresh processes: 65.07 → 10.53 s, 138.56 → 50.76 s, 33.25 → 12.21 s |
| Validation | Image codes bit-identical to the reference; refiner pixels within PSNR ≥ 48 dB of the reference (measured 57–59 dB) |

The language model and the DSpark image head run CUDA graphs, fused kernels and Triton replicas of cuBLAS's decode GEMMs that reproduce the reference arithmetic bit for bit, so the image codes are unchanged. The refiner gets a single-sample DiT path, FlashInfer's sm90 attention, large-tile cuBLASLt GEMMs and fused norm kernels; only floating-point rounding changes there. One code path serves both models and is on by default, with automatic fallbacks and a `GSTAR_FAST=0` opt-out. Running `python -m gstar.infer` once for a single image, including model loading, is 2.50× / 2.12× / 1.36× faster with a warm compile cache.

[Code ↗](https://github.com/DAGroup-PKU/G-STAR) · [Pull request ↗](https://github.com/DAGroup-PKU/G-STAR/pull/2)

## Faster-RAE

**Stage-1 decoder + GAN training**

| Item | Result |
|---|---|
| Speedup | **5.32×** |
| Hardware | 2 × H200 |
| Measurement scope | Decoder + GAN step time |
| Validation | Statistical equivalence |



### Further results

A later H200 benchmark reports **5.41×**. On **8 × H20** at global batch 512, step timing improves **3.22×**; a completed 16-epoch run achieves rFID **0.541**, compared with **0.569** for the public checkpoint.

[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-RAE) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-RAE/blob/main/docs/h20-results.md)

## Faster-DMD

**DMD2 one-step SD1.5 distillation + GAN**

| Item | Result |
|---|---|
| Speedup | **1.72×** |
| Hardware | 4 × H20 |
| Measurement scope | Distillation + GAN |
| Validation | Statistical equivalence |



### Further results

A 2,000-iteration continuation on **8 × H20**, global batch 2,048, achieves **1.67×** speedup: 16.39 → 9.79 h of logged iteration time, excluding checkpoint writes. FID-30k is **8.7030**, compared with baseline **8.6536** (seed 10).

[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-DMD) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-DMD/blob/main/docs/h20-results.md)

## Faster-JiT

**JiT-B/16 · ImageNet 256 training**

| Item | Result |
|---|---|
| Speedup | **1.56×** |
| Hardware | 2 × H200-class |
| Measurement scope | 100.02 → 64.04 ms/step |
| Validation | Statistical equivalence |

A separate 40-epoch run on 8 × H100 achieved FID-50K 39.24, compared with baseline 39.07 (one seed per arm).

[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-JiT) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-JiT/blob/main/docs/training.md#validation)
