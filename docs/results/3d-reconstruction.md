# 3D Reconstruction

[← All results](README.md) · [ResearchBoost-Agent](../../README.md#results)

3 projects · [Original 3DGS](#original-3dgs) · [PIDG](#pidg) · [Faster-VGGT](#faster-vggt)

Original 3DGS and PIDG are contributed by [@HaoqinHong](https://github.com/HaoqinHong) on one NVIDIA A800-SXM4-80GB (shared GPU, original precision settings, task-metric validation). Faster-VGGT is contributed by [@SuperFCR](https://github.com/SuperFCR) on one NVIDIA A100-SXM4-80GB with a frozen-judge four-gate `PASS` and end-to-end training/validation verification. Judge statuses are listed below.

## Original 3DGS

**Mip-NeRF360 · bonsai, bicycle and garden · complete 30,000-step native pipelines**

| Item | Result |
|---|---|
| Speedup | **1.13×** across the three complete pipelines; **1.07–1.29×** per scene |
| Hardware | 1 × NVIDIA A800-SXM4-80GB |
| Measurement scope | 4269.73 → 3773.86 s total, including initialization, training, native evaluation and saving |
| Validation | Held-out-view PSNR: ΔPSNR −0.014 to +0.038 dB per scene |
| Judge status | `INCORRECT / exit 2`: 11 continuous-value violations among 1,287 checks; `formal_ResearchBoost_acceptance: false` |

| Scene | Baseline (s) | Experimental (s) | Speedup | PSNR baseline → experimental (dB) | ΔPSNR (dB) |
|---|---:|---:|---:|---:|---:|
| bonsai | 1050.32 | 816.46 | **1.29×** | 32.1943 → 32.2318 | +0.0375 |
| bicycle | 1587.13 | 1436.74 | **1.10×** | 25.2363 → 25.2222 | −0.0140 |
| garden | 1632.28 | 1520.65 | **1.07×** | 27.3985 → 27.3963 | −0.0023 |

Timing uses the eligible paired seed-1 runs. PSNR averages seeds 0 and 1 over all held-out views at step 30,000: 37 for bonsai, 25 for bicycle and 24 for garden. Official image scales are `images_2` for bonsai and `images_4` for bicycle/garden. PSNR follows the pinned historical implementation's per-channel averaging convention.

### Timing eligibility, noise and Gaussian counts

The eligibility policy was declared before optimization. Both paired runs must have `full_train.json` status `complete` and return code 0, a successful fresh CPU artifact-check subprocess with empty `CUDA_VISIBLE_DEVICES`, and exact frozen source/data/config/seed/schedule bindings. Native duration is the artifact-check subprocess start minus the full-training process start; the extra CPU check is outside that duration.

All three seed-0 baselines are ineligible for timing: bonsai used the legacy checker and lacks the same fresh-CPU boundary; bicycle/garden's original post-training artifact checks exited SIGBUS (−7). Separate strict recovery qualifies the saved models for image-quality evaluation only and preserves the failed original exits. Seed-1 runs provide comparable boundaries for all scenes.

There is **one eligible baseline/candidate timing pair per scene**. Baselines were collected before optimization and candidate full runs followed; there was no complete-pipeline baseline → candidate → baseline repeat. **Run-to-run noise and confidence intervals for complete-pipeline speedup were not estimated.** Noise from the separate fixed-N probes does not estimate full-pipeline noise.

| Scene | Baseline final Gaussians | Experimental final Gaussians | Change |
|---|---:|---:|---:|
| bonsai | 1,275,632 | 1,271,323 | −0.34% |
| bicycle | 6,071,245 | 5,971,655 | −1.64% |
| garden | 5,927,094 | 5,951,459 | +0.41% |

FP32 accumulation changes affect the training trajectory and densification. The complete-pipeline ratios include these Gaussian-count and work differences; they do not isolate equal-work kernel acceleration.

The patch reduces FP32 gradients for full warps with a common Gaussian ID in rasterization backward and retains the original atomic path otherwise. Separate fixed-Gaussian-count probes at checkpoints 100, 7,000 and 29,900 measured **1.04–2.17×** using baseline → candidate → baseline arms; they exclude densification/reset/evaluation/save events.

### Validation

The 11 numerical violations remain unresolved; exact-state, runtime-error, dtype and math-mode failures were zero. Quality comparison recorded 52 endpoints outside the frozen two-seed baseline descriptive band: 28 worsened and 24 improved. Two quality seeds do not establish statistical equivalence. There is no frozen-judge PASS for this case.

[Code & project README ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost) · [Native timing ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/results/gs3d/native-timing-results.csv) · [Timing policy, exclusions & noise ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/results/gs3d/native-timing-notes.json) · [Image quality ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/results/gs3d/quality-results.csv) · [Numerical checks ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/results/gs3d/numerics.json)

## PIDG

**Physics-Informed Deformable Gaussian Splatting · dry_ice · optical-flow module-group benchmark**

| Item | Result |
|---|---|
| Speedup | **4.17×**, final R06: retained flow optimization + restored original raster CUDA source |
| Hardware | 1 × NVIDIA A800-SXM4-80GB |
| Measurement scope | 9369.79 → 2244.31 ms for three native renders, the original velocity network, optical-flow/reconstruction losses, backward and five PNG writes |
| Module numerical status | R06 `checks[4]`: 2,435 checked, 1,368 bitwise-identical, 0 unresolved; generic `PASS` |
| Formal judge status | `formal_ResearchBoost_acceptance: false`; `formal_benefit_verdict: UNAVAILABLE_SHARED_WORKLOAD` |

### Historical labels and the failing/passing checks

`both` in the harness selects the flow file **and** raster extension from the candidate tree. It describes module selection, not which selected modules contain optimization code. The raw labels and failed checks remain unchanged in the published JSON.

| Summary index | Run / recorded variant | Effective implementation | Generic status | Bitwise identical | Continuous violations |
|---|---|---|---|---:|---:|
| `checks[0]` | R01 / `flow-only` | Dense embedding gather + original raster | `INCORRECT` | 1,168 | 82 |
| `checks[1]` | R03 / `flow-only` | Exact-zero-eliding gather + original raster | `PASS` | 1,360 | 0 |
| `checks[2]` | R04 / `both` | R03 gather + grouped raster reduction | `INCORRECT` | 1,367 | 84 |
| `checks[3]` | R04 / `raster-only` | Original flow + grouped raster reduction | `INCORRECT` | 1,372 | 84 |
| `checks[4]` | R06 / `both` | R03 gather + original raster source rebuilt in candidate tree | `PASS` | 1,368 | 0 |

Each row has 2,435 checks. R01's embedding gather was rejected; R03 replaced it with the exact-zero-eliding gather implementation. R04's raster-only and accelerated combined implementations were rejected. Before R06, the rejected raster CUDA source was restored byte-for-byte to baseline and its candidate-local extension rebuilt; the R03 flow source stayed unchanged. Thus **1,360** belongs to the R03 flow-only PASS, while the **4.17×** confirmation and **1,368** bitwise checks belong to R06, still recorded as `both`.

The R06 flow source hash is `3566f99a…26a37b`; its restored raster backward is `ea9d57d6…81ac6b`, versus rejected R04 `9edf9e28…273f1a`. The [run-provenance record](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/results/pidg/run-provenance.json) includes full source/binary hashes and archived run paths. All five checks retain the same thresholds-file SHA256, `ea0ad940…8d8232`; no threshold relaxation was used between these candidates. R02 timing was configuration-invalid; the common GLM include-path build fix precedes subsequent valid measurements.

R06's longer dry_ice confirmation uses baseline → candidate → baseline independent processes, one warmup and three windows of three iterations per arm. Active clocks were comparable and its observed ratio exceeded the 3.27% diagnostic resolution threshold. That threshold and the generic module PASS do not constitute formal benefit acceptance on the shared GPU.

The gather optimization omits exactly-zero gradient vectors, preserving forward computation, nonzero/nonfinite gradients and higher-order differentiation. Five-scene ablations use the original 100,000-point initialization and training frames `0002/0003`. The headline covers the module group above, excluding the deformation network, physical regularizers, densification and complete training. A 601-step native dry_ice run completed with finite saved artifacts.

[Code & project README ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost) · [Full run-history explanation ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/docs/pidg-run-history.md) · [Original summary & checks ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/results/pidg/summary.json) · [Five-scene ablations ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/results/pidg/measurements.csv)

## Reproduction

The public repository includes pinned upstream versions, retained and rejected patches, public-data SHA256 manifests, benchmark drivers and result tables. These runs used ResearchBoost methodology, skill cards and its unmodified generic numerical component; they did not launch `clawo`.

[Reproduction commands ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/docs/reproduce.md) · [Environment ↗](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost/blob/89241769ce2dccf2b80323e9a7e4f57ee09e006b/docs/environment.md)

## Faster-VGGT

Contributed by [@SuperFCR](https://github.com/SuperFCR).

**VGGT-1B fine-tuning (`facebookresearch/vggt @ a288dd0f`) · Virtual KITTI 2.0.3 · frozen 908M aggregator + trainable 202M `camera_head` & 12M `depth_head`**

| Item | Result |
|---|---|
| Speedup | **2.19×** (`2.1857×`, Tier N2 `configs/final.json`) · **1.86×** (`1.8630×`, pure lossless Tier N1 `configs/final_n1.json`) |
| Peak memory | **19.15 GiB** vs `28.04 GiB` (**−31.7%** allocated) on N2; `30.48 GiB` on N1 |
| Hardware | 1 × NVIDIA A100-SXM4-80GB, 12 vCPU, PyTorch `2.14.1+cu130`, Triton `3.8.0`, CUDA 13.0 |
| Measurement scope | Steady-state iteration time (`1.9250 → 0.8807 s/iter` N2; `1.9283 → 1.0351 s/iter` N1): `bench.py check --final` A-B-A (`ref_a → cand → ref_b`) in fresh cold-cache `torchrun` processes, 6 warm-up + 2 × 15 timed iterations each (`ref_spread = 0.13%` / `0.32%`) |
| Judge status | **`PASS` (`final: true`)** across all four gates; frozen `evaluator_sha256 = c5b59bf5…c966d` unchanged throughout; default-off tree is `3,776/3,776` tensors bitwise identical (`N0`) |
| Validation | Gate 3 (20-step deterministic trajectory vs 6 frozen seeds at `3σ`): `equivalent`, max `\|d\|/σ = 0.0642σ` on both N2 and N1 (`lr × 1.02` control arm rejected at `+12.18σ` / `+9.24σ`); real end-to-end training + checkpoint save/reload + held-out validation match baseline |

Six optimization rounds behind default-off `VGGT_OPT_*` switches (executed with Gemini 4 Neon `xhigh`): (1) pinned memory, 1-frame RoPE cos/sin cache across 48 alternating-attention blocks, branchless `nan_to_num` with single end-of-step scalar flush, and GPU camera/point normalization; (2) `_version`-keyed BF16 weight cache for the frozen aggregator plus fused Triton `LayerNorm→BF16`, `LayerScale+Add+LayerNorm→BF16` (exact FP32 residual), and fused QKV unpack + FP32 QK-Norm + 2D RoPE with zero-copy FlashAttention layout; (3) a 4-pass 8-bit multi-SM Triton radix-select quantile kernel (**160.2×** faster than `kthvalue`, bitwise N0) and dense branchless masked loss (**10.6×** faster `multitask_loss`); (4) cached DPT UV positional embeddings, NHWC `DPTHead`, unchunked frames, and fused `AdamW` with deferred clip sync; (5) cross-block `ls2+norm1` fusion across 70 block boundaries, zero-sync `rotation.py`, `do_not_specialize` on all 8 Triton kernels to eliminate dynamic-resolution JIT recompiles, and a deterministic atomic-free Triton NHWC bilinear upsample kernel (reaching **1.86×** pure lossless N1); and (6) local BF16 `DPTHead` convolutions with FP32 `F.linear` on the final 32→2 NHWC projection and FP32 `activate_head`, loss, optimizer and `CameraHead` (reaching **2.19×** N2 and **−31.7%** peak VRAM).

### End-to-end training & held-out validation

Executed via the unmodified upstream entrypoint `training/launch.py` (`seed=42`, 41 training steps / 82 microbatches, real `6.17 GiB` `checkpoint_0.pt` save and `strict=True` reload into a fresh `VGGT`, and held-out validation on Virtual KITTI 2.0.3 `split=test`, 10 batches):

| Arm | `val_loss_camera` | `val_loss_T` | `val_loss_R` | `val_loss_FL` | `val_loss_conf_depth` | `val_loss_reg_depth` | `val_loss_grad_depth` | Train `objective` | Steady-state Train / Val `Batch Time` | Peak Mem (Train / Val) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original (`targets/reference`) | 0.0039 | 0.0026 | 0.0002 | 0.0021 | −0.0542 | 0.0591 | 0.0464 | 0.0344 | 2.8256 s / 1.9073 s | 29.0 / 32.0 GB |
| Optimized (`configs/final.json`) | **0.0039** | **0.0027** | **0.0002** | **0.0021** | **−0.0560** | **0.0590** | **0.0466** | **0.0344** | **1.2702 s (2.22×) / 1.0178 s (1.87×)** | **20.0 / 19.0 GB** |

Full end-to-end wall clock (including 5 GiB checkpoint load, cold Triton JIT, 41 steps, 6.17 GiB checkpoint write and held-out validation): `192.74 → 145.01 s` (**1.33×**). Both checkpoints (`1,341` keys) have `0` non-finite tensors and reload cleanly with `strict=True`.

[Code & project README ↗](https://github.com/SuperFCR/Faster-VGGT) · [End-to-end validation ↗](https://github.com/SuperFCR/Faster-VGGT/blob/main/docs/validation.md) · [Frozen judge & measurement ↗](https://github.com/SuperFCR/Faster-VGGT/blob/main/docs/measurement.md) · [Final check JSON ↗](https://github.com/SuperFCR/Faster-VGGT/blob/main/results/final_check.json) · [Work log ↗](https://github.com/SuperFCR/Faster-VGGT/blob/main/WORK_LOG.md)
