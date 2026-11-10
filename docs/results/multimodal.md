# MLLM

[← All results](README.md) · [ResearchBoost-Agent](../../README.md#results)

4 projects · [LLaVA](#faster-llava) · [BAGEL](#faster-bagel) · [R3](#faster-r3) · [LLaVA-OneVision-2](#llava-onevision-2)

## Faster-LLaVA

**LLaVA-1.5-7B stage-2 fine-tuning**

| Item | Result |
|---|---|
| Speedup | **2.39×** |
| Hardware | 4 × H20 |
| Measurement scope | Stage-2 fine-tuning |
| Validation | Statistical equivalence |



### Further results

After one epoch of fine-tuning, scores on four benchmarks differ from the released official weights by **−0.50 to +0.44 points**.

[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-LLaVA) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-LLaVA/blob/main/docs/h20-results.md)

## Faster-BAGEL

**7B-MoT SFT · BF16 compute, no FP8**

| Item | Result |
|---|---|
| Speedup | **1.82×** |
| Hardware | 8 × Hopper 80GB |
| Measurement scope | 1.4367 → 0.7873 s/step |
| Validation | Statistical equivalence |

Validated over 2,000 steps × 4 seeds, with 2σ trajectory checks and held-out VLM cross-entropy evaluation.

[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-BAGEL) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-BAGEL/blob/main/docs/judge.md)

## Faster-R3

**BAGEL 7B-MoT Tree-RL**

| Item | Result |
|---|---|
| Speedup | **2.08×** |
| Hardware | 8 × H100 80GB |
| Measurement scope | 24.72 → 11.87 s/RL step · rollout, GenEval reward service and optimizer |
| Validation | Bitwise identical · 14 deterministic steps on 8 ranks |

Measured over 16 steps, with 10 warmup steps and three paired timing windows. Recorded training items match bitwise over 14 deterministic steps on all eight ranks.

### Further results

The **2.08×** result includes GenEval reward-service improvements. With the optimized reward service on both sides, the training code alone achieves **1.88×**.

[Code & project README ↗](https://github.com/AI4AI-for-Anything/Faster-R3) · [Detailed evidence ↗](https://github.com/AI4AI-for-Anything/Faster-R3/blob/main/docs/judge.md)

## LLaVA-OneVision-2

**LLaVA-OneVision-2-4B quick-start SFT · one epoch** · by [@GONGSHUKAI](https://github.com/GONGSHUKAI), merged upstream in [LLaVA-OneVision-2#205](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/pull/205)

| Item | Result |
|---|---|
| Speedup | **3.0×** on Blackwell · **2.2×** on Hopper |
| Hardware | 8 Blackwell GPUs (`blackwell` preset) · 4 Hopper GPUs (`hopper_80g` preset) |
| Measurement scope | One epoch of the quick-start recipe against the original `main`; MFU 11.6% → 32.5% on Blackwell and 22.1% → 45.5% on Hopper |
| Validation | Statistical equivalence |

The speedups are opt-in switches covering activation recomputation, FlashAttention-4 attention, Triton-fused kernels, host-sync removal and cross-rank load balancing; `OV2_PRESET` loads the preset for each GPU architecture. Each change was kept only if training stayed bitwise identical to `main` or within its seed-to-seed variation. With no switch set, training is bitwise identical to `main` (28,260 tensors over the first three iterations). After one epoch, loss on a fixed evaluation batch was 0.5782 (Blackwell) and 0.5781 (Hopper) with the presets, against 0.5788 and 0.5783 for `main` on the same GPUs. Multi-node training with tensor and pipeline parallelism also runs with the speedups.

The `hopper_80g` preset keeps peak memory at 74.8 GiB per GPU; it was timed on Hopper GPUs with more than 80 GB and has not yet run on 80 GB cards. The `blackwell` preset peaks at 157.6 GiB per GPU.

[Code & project README ↗](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2#training-acceleration) · [Switches & presets ↗](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/blob/main/docs/faster_training.md) · [Pull request & review ↗](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/pull/205)
