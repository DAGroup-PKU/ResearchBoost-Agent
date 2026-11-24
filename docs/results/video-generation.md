# Video Generation

[← All results](README.md) · [ResearchBoost-Agent](../../README.md#results)

4 projects · [Wan2.2-VSA](#faster-wan22-vsa-musa), [SANA-WM](#faster-sana-wm-musa), [Self-Forcing](#faster-selfforcing), [MiniMax-H3](#faster-minimax-h3)

## Faster-Wan22-VSA-MUSA

**Wan2.2-TI2V-5B + Video Sparse Attention (VSA) full-parameter fine-tuning** · contributed by [@JoeyXuquant11](https://github.com/JoeyXuquant11)

| Item | Result |
|---|---|
| Speedup | **5.28×** (`25.266 → 4.784 s/iter`, exact `5.2816×`, `ref_a 25.264 s`, `ref_b 25.268 s`, spread `0.02%`) |
| Hardware | 8 × Moore Threads MTT S5000 (80 GiB HBM/GPU, MUSA 5.2.0) |
| Measurement scope | Official `Wan2.2-TI2V-5B` (30 layers, 5.28B trainable params) + 80% VSA (`topk=96/480`), `704×1280×121f` (`S=27,280`), `batch=1`/GPU (`global_batch=8`), FSDP2 BF16 + FP32 master/AdamW; paired A-B-A `check --final` (2 warmup + 2×4 timed steps) |
| Validation | **PASS (Tier N1, statistical equivalence)** — Gate 2: `71,008` tensors checked (`0` exact failures, `0` dtype changes); Gate 3 (6-step paired deterministic trajectory, `±3σ` vs 6-seed frozen ref): `loss_tail = -0.034σ`, `gate_top10_step1 = -0.350σ`. Extended **24-step (6-epoch)** paired training validation (`global_batch=8` on 4× MTT S5000 × `grad_accum=2`): loss curve `Pearson r = 0.999995`, `mean \|Δloss\| = 6.36e-5`, held-out step-24 eval MSE `0.014753` vs `0.014785` (`sp=0.80`, `0.22%` rel diff) and `0.015232` vs `0.015236` (`sp=0.00`, `0.027%` rel diff) |

All custom TileLang-MUSA block-sparse VSA kernels and pure-scalar MUSA C++ fused norms, RoPE, GELU+bias, VSA pre/post, and 1-pass FP32 AdamW kernels were written from scratch in a strict read-only 10-hour sandbox (Antigravity + Gemini 4 Neon (XHigh)), lifting effective MFU (`@500T` BF16) from `7.65%` to `40.39%` and cutting kernel launches per step from `34,054` to `4,470` (`-86.9%`). Gate 3 also caught a real `lse < -127` `NaN` in an unmasked `R12a` `dQ` candidate that had passed Gate 2 bitwise, leading to the bitwise-exact clamp-safe `R12` kernel (`3258fd7`).

[Code & reproduction README ↗](https://github.com/JoeyXuquant11/Faster-Wan22-VSA-MUSA) · [Optimization trace ↗](https://github.com/JoeyXuquant11/Faster-Wan22-VSA-MUSA/blob/main/TRACE.md) · [24-step E2E validation ↗](https://github.com/JoeyXuquant11/Faster-Wan22-VSA-MUSA/blob/main/docs/e2e_validation.md) · [Final check JSON ↗](https://github.com/JoeyXuquant11/Faster-Wan22-VSA-MUSA/blob/main/task/final_check.json)

## Faster-SANA-WM-MUSA

**SANA-WM Stage-1 bidirectional 720p FSDP2 training** · contributed by [@JoeyXuquant11](https://github.com/JoeyXuquant11)

| Item | Result |
|---|---|
| Speedup | **4.85×** (`11.502 → 2.370 s/iter`, exact `4.8543×`, `ref_a 11.462 s`, `ref_b 11.542 s`, spread `0.69%`) |
| Hardware | 4 × Moore Threads MTT S5000 (80 GiB HBM/GPU, MUSA 5.2.0) |
| Measurement scope | Upstream `NVlabs/Sana @ 27303b8` (`SanaMSVideoCamCtrl_1600M_P1_D20`, 2.66B params: 15 bidirectional GDN+UCPE + 5 Softmax+UCPE blocks, `GLUMBConvTemp` FFN, real published Stage-1 DiT weights + frozen online `gemma-2-2b-it`), 193 frames (`704×1280`, `N=22,000`), `batch=1`/GPU (`DP=4` FSDP2), BF16 + `fp32_attention` + FP32 master/AdamW; paired A-B-A `check --final` (3 warmup + 2×5 timed steps) |
| Validation | **PASS (Tier N1, statistical equivalence)** — Gate 2: `33,176` tensors checked (`0` exact failures, `0` dtype changes); Gate 3 (6-step paired deterministic trajectory, `±3σ` vs 6-seed frozen ref): `train_loss = -0.018σ`, `param_disp_l2 = +0.413σ`, `step_disp_top1pct = -0.445σ`. Extended **48-step (3-epoch)** paired training validation on native 4× MTT S5000: loss curve `Pearson r = 0.999800`, `mean \|Δloss\| = 1.36e-3` (`0.5377 → 0.4357` vs `0.5376 → 0.4338`), full 2.66B parameter checkpoint parity across all 850 tensors (`rel L2 = 2.48e-4` / `0.025%`, global update cosine similarity `0.9130`) |

All ~12,500 lines of custom MUSA C++ kernels (`Sana/diffusion/model/musa_ops/`: fused bidirectional GDN+UCPE, `GLUMBConvTemp` with backward `y_recomp`, block fusions, fast `Gemma-2-2B-IT`, and 1-pass `float4` FP32 AdamW+clip) plus a 16-byte-aligned FSDP2 collective fix (eliminating a `10.7×` unaligned MCCL `AllGather` slowdown) were written from scratch in a strict read-only 10-hour sandbox (Antigravity + Gemini 4 Neon (XHigh)), lifting effective MFU (`@500T` BF16) from `7.79%` to `37.82%` and cutting kernel launches per step from `74,279` to `6,195` (`-91.7%`).

[Code & reproduction README ↗](https://github.com/JoeyXuquant11/Faster-SANA-WM-MUSA) · [Optimization trace ↗](https://github.com/JoeyXuquant11/Faster-SANA-WM-MUSA/blob/main/TRACE.md) · [48-step E2E validation ↗](https://github.com/JoeyXuquant11/Faster-SANA-WM-MUSA/blob/main/docs/e2e_validation.md) · [Final check JSON ↗](https://github.com/JoeyXuquant11/Faster-SANA-WM-MUSA/blob/main/task/final_check.json)

## Faster-SelfForcing

**Self-Forcing video DMD training**

| Item | Result |
|---|---|
| Speedup | **2.08×** |
| Hardware | 2 × H200 |
| Measurement scope | Video DMD training |
| Validation | Statistical equivalence |



[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-SelfForcing) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-SelfForcing/blob/main/docs/training.md)

## Faster-MiniMax-H3

**MiniMax-H3 FL2VA LoRA fine-tuning** · contributed by [@2018cx](https://github.com/2018cx)

| Item | Result |
|---|---|
| Speedup | **1.8570×** (117.318886 → 63.176236 s / optimizer step) |
| Hardware | 8 × NVIDIA H20 |
| Measurement scope | BF16 FL2VA LoRA / ZeRO-3, batch 1 per rank, approximately 33,750 packed tokens per sample; two-update frozen reference/candidate/reference check |
| Validation | **PASS** — repaired Round 50; reference and optimized arms each ran 50 optimizer steps at seeds 424242, 424243 and 424244 (six arms total); all eight loss/gradient comparisons passed `|standardized mean difference| ≤ 2.0` |

The original Round 19 candidate measured **1.869055×** in the short speed check but was rejected by the separate trajectory gate: finite optimized-gradient spikes drove the step-25 full-LoRA-gradient comparison to **142,950.33** standardized units. Targeted replay diagnostics found that checkpoint inference and replay could select different compiled Q/K RMSNorm variants for identical inputs on five of eight ranks. The repair shares one detached compiled Q/K forward across both modes while preserving the compiled first-order derivative. Round 50 then passed the authoritative speed check at **1.8570097438×** and the fresh three-seed validation. This establishes short-trajectory agreement only, not full convergence or generated-output quality.

[Code & project README ↗](https://github.com/2018cx/Faster-MiniMax-H3) · [Related DiffSynth implementation ↗](https://github.com/2018cx/Faster-DiffSynth-H3) · [Performance and repair audit ↗](https://github.com/2018cx/Faster-MiniMax-H3/blob/main/docs/repair-results.md) · [Six-arm validation ↗](https://github.com/2018cx/Faster-MiniMax-H3/blob/main/docs/repaired-validation-results.md) · [Machine-readable evidence ↗](https://github.com/2018cx/Faster-MiniMax-H3/blob/main/docs/performance-data/repaired-validation.json)
