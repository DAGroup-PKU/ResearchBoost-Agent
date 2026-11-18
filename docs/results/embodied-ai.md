# Embodied AI

[← All results](README.md) · [ResearchBoost-Agent](../../README.md#results)

5 projects · [RoboTwin](#faster-robotwin) · [Motus](#faster-motus) · [FastWAM](#faster-fastwam) · [OpenWAM](#openwam) · [SuperMimic](#faster-supermimic)

## Faster-RoboTwin

**π0.5 policy evaluation · RoboTwin 2.0**

| Item | Result |
|---|---|
| Speedup | **4.38×** (est.) |
| Hardware | 8 × H20 |
| Measurement scope | Estimated 73.5 → 16.8 h benchmark wall-clock |
| Validation | Compared episodes match bitwise |

Full-benchmark time is extrapolated from sampled jobs. Compared episodes match bitwise.

[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-RoboTwin) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-RoboTwin/blob/main/docs/measurement.md)

## Faster-Motus

**RoboTwin 2.0 fine-tuning · 27,244 episodes**

| Item | Result |
|---|---|
| Speedup | **9.59×** |
| Hardware | 8 × B200 |
| Measurement scope | 7.468 → 0.779 s/iteration |
| Validation | Bitwise identical · 1,100 iterations |

Validation covers all eight ranks across two epoch boundaries, including losses, batches, validation metrics, final parameters and optimizer state.

[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-Motus) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-Motus/blob/main/docs/full-data-results.md)

## Faster-FastWAM

**FastWAM-LIBERO training recipe**

| Item | Result |
|---|---|
| Speedup | **3.46×** |
| Hardware | 8 × B200 |
| Measurement scope | 1,541.4 → 445.7 ms/iteration |
| Validation | LIBERO evaluation reproduced |

LIBERO overall success: 96.90% for Faster-FastWAM and 97.15% for the released checkpoint.

[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-FastWAM) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-FastWAM/blob/main/docs/training.md)

## OpenWAM

**OpenWAM-α fine-tuning · seven SFT benchmarks** · by [@GONGSHUKAI](https://github.com/GONGSHUKAI), merged upstream in [OpenWAM#58](https://github.com/OpenWAM-Official/OpenWAM/pull/58)

| Item | Result |
|---|---|
| Speedup | **2.54–2.77×** |
| Hardware | 4 × H200 |
| Measurement scope | Warmed-up training throughput on seven SFT benchmarks, against the original `main` |
| Validation | LIBERO evaluation reproduced |

The original `main` recomputed every layer (all-layer gradient checkpointing); the measured configuration used no checkpointing and no 80 GiB memory cap, so the speedups include the recomputation it avoids. Each change was kept only if training stayed bitwise identical to `main` or within its seed-to-seed variation. LIBERO success after a full Sim-LIBERO fine-tuning run with the speed preset: 99.05%, against 99.2% for the released OpenWAM-Alpha-Sim-LIBERO checkpoint under the same evaluation. OpenWAM now loads the speed preset by default and keeps all-layer checkpointing on by default.

[Code & project README ↗](https://github.com/OpenWAM-Official/OpenWAM#training-acceleration) · [Pull request & review ↗](https://github.com/OpenWAM-Official/OpenWAM/pull/58)

## Faster-SuperMimic

**SuperMimic world-model + BPTT training · 40 LAFAN1 clips** · by [@hwbhwbhwb](https://github.com/hwbhwbhwb)

| Item | Result |
|---|---|
| Speedup | **1.38×** |
| Hardware | 1 × GPU |
| Measurement scope | 0.0852 → 0.0617 s/iteration, steady state (reference → candidate → reference) |
| Validation | Statistical judge PASS · 20k-iteration training reproduced |

The official [SuperMimic](https://github.com/HybridRobotics/supermimic) with [mjlab_research](https://github.com/HybridRobotics/mjlab_research), task `SuperMimic-GenCoord-G1` on 4096 MuJoCo Warp envs, code defaults. Nine commits remove host syncs in the env step, replay sampling and normalizers, compile and horizontally fuse small kernels, skip unread observation groups and align a TF32 GEMM; precision, simulator and random-draw order are unchanged. The reference itself is not bitwise reproducible across processes, so the judge compares 20 training endpoints of an 800-iteration run against six reference seeds (largest deviation 1.53σ). Trained for 20,000 iterations with two seeds per arm and evaluated with the original code on all 40 clips, the benchmark return is 0.860 for the original and 0.862 for the optimized code.

[Code & project README ↗](https://github.com/hwbhwbhwb/Faster-SuperMimic) · [Detailed evidence ↗](https://github.com/hwbhwbhwb/Faster-SuperMimic/blob/main/docs/measurement.md)
